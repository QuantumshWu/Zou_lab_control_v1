"""One provisional occupancy-curve branch for an exact autonomous scan."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

from zlc_data import (
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    IndexSelection,
    Selection,
    SITE,
    dataset_revision_ref_to_tree,
    materialize_transformed_snapshot,
    transformed_snapshot_peak_nbytes,
)
from zlc_frontend.matplotlib_render import (
    SingleCurveAggRenderer,
    estimate_single_curve_raster_peak_nbytes,
)
from zlc_frontend.figure import (
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureEvaluator,
    FigureLayer,
    AxisViewRole,
    EvaluatedInput,
    ResolvedDataset,
    ResolvedDatasetMap,
    RepeatViewMode,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    estimate_view_evaluation_peak_nbytes,
    suggest_view,
)
from zlc_frontend.render import (
    BoardFrame,
    BoardPresenter,
    CoherenceStamp,
    PanelFrame,
    PanelPresentationIdentity,
    RasterBuffer,
    RenderSurface,
    SourceIdentity,
    detached_render_fault,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetPreviewSnapshot,
    ExactDatasetPreviewReader,
    dataset_storage_nbytes,
)
from zlc_neutral_atom.runtime.pipeline import ExactDatasetPreviewSpec
from zlc_neutral_atom.scan import ScanOutputContract
from zlc_storage import canonical_digest, canonical_text

from .workspace import (
    BoardController,
    BoardModel,
    BoardPublishPort,
    PanelSlot,
    PanelSourceBinding,
)


_PANEL_ID = "scan-curve"
_COHERENCE_GROUP = "scan-output"
_RASTER_WIDTH = 800
_RASTER_HEIGHT = 520


@dataclass(frozen=True, slots=True)
class ScanDisplayIntent:
    """Visible, non-authoritative site presentation choice for a scan panel."""

    site_mode: str = "auto"
    site_index: int = 0

    def __post_init__(self) -> None:
        mode = canonical_text(self.site_mode, "site_mode")
        if mode not in {"auto", "batch", "select"}:
            raise ValueError("site_mode must be 'auto', 'batch', or 'select'")
        if (
            isinstance(self.site_index, bool)
            or not isinstance(self.site_index, int)
            or self.site_index < 0
        ):
            raise ValueError("site_index must be a nonnegative integer")
        if mode != "select" and self.site_index != 0:
            raise ValueError("site_index is meaningful only when site_mode='select'")


@dataclass(frozen=True, slots=True)
class ProgressiveScanSpec:
    """Frozen display-only plan paired with one authoritative output contract."""

    output_contract: ScanOutputContract
    output_block_id: BlockId
    document: FigureDocument
    projection_summary: str
    transform_peak_bytes: int
    evaluation_peak_bytes: int
    preview_spec: ExactDatasetPreviewSpec
    display_selection: Selection | None
    display_preferences: ViewPreferences

    def __post_init__(self) -> None:
        if not isinstance(self.output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(self.preview_spec, ExactDatasetPreviewSpec):
            raise TypeError("preview_spec must be ExactDatasetPreviewSpec")
        if (
            self.output_contract.committed_transform.input_schema_fingerprint
            != self.preview_spec.source_schema_fingerprint
        ):
            raise ValueError("progressive transform belongs to another source schema")
        if not isinstance(self.output_block_id, BlockId):
            raise TypeError("output_block_id must be BlockId")
        if not isinstance(self.document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if self.display_selection is not None and not isinstance(
            self.display_selection,
            Selection,
        ):
            raise TypeError("display_selection must be Selection or None")
        if not isinstance(self.display_preferences, ViewPreferences):
            raise TypeError("display_preferences must be ViewPreferences")
        dataset_id = self.document.datasets[0].dataset_id if self.document.datasets else None
        if (
            len(self.document.datasets) != 1
            or self.document.datasets[0].schema_fingerprint
            != self.output_contract.output_schema_fingerprint
            or len(self.document.layers) != 1
            or self.document.layers[0].dataset_id != dataset_id
            or self.document.layers[0].view.intent is not ViewIntent.CURVE
            or any(
                binding.role is AxisViewRole.FACET
                for binding in self.document.layers[0].view.axis_bindings
            )
        ):
            raise ValueError(
                "progressive document must be one non-faceted CURVE over scan output"
            )
        object.__setattr__(
            self,
            "projection_summary",
            canonical_text(self.projection_summary, "projection_summary"),
        )
        for name in (
            "transform_peak_bytes",
            "evaluation_peak_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def dataset_id(self) -> DatasetId:
        return self.document.datasets[0].dataset_id


def build_occupancy_progressive_spec(
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    *,
    identity: str,
    display_intent: ScanDisplayIntent = ScanDisplayIntent(),
) -> ProgressiveScanSpec:
    """Derive one visible, non-authoritative curve view from declared axes."""

    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    if not isinstance(display_intent, ScanDisplayIntent):
        raise TypeError("display_intent must be ScanDisplayIntent")
    identity = canonical_text(identity, "identity")
    output_schema = output_contract.output_dataset_schema
    if not output_schema.point_axes:
        raise ValueError("progressive scan curve requires a declared point axis")
    x_axis = output_schema.point_axes[0]
    first_point = output_schema.point_layout.multi_index(0)
    terms = [
        IndexSelection(axis.axis_id, first_point[index])
        for index, axis in enumerate(output_schema.point_axes)
        if axis.axis_id != x_axis.axis_id
    ]
    # Information-bearing trailing axes are selected, never averaged.  The
    # exact coordinate is kept in ViewSpec and in the visible summary.
    data_axes = output_schema.cell_schema.data_axes
    site_axes = tuple(axis for axis in data_axes if axis.role == SITE)
    if len(site_axes) != 1:
        raise ValueError("occupancy output must declare exactly one SITE axis")
    site_axis = site_axes[0]
    if display_intent.site_mode == "batch":
        if not 1 < site_axis.size <= 32:
            raise ValueError("site batch display requires between 2 and 32 sites")
        batch_axis = site_axis
    elif display_intent.site_mode == "select":
        if display_intent.site_index >= site_axis.size:
            raise ValueError("selected site index exceeds the declared SITE axis")
        batch_axis = None
    else:
        batch_axis = site_axis if 1 < site_axis.size <= 32 else None
    terms.extend(
        IndexSelection(
            axis.axis_id,
            display_intent.site_index
            if axis is site_axis and display_intent.site_mode == "select"
            else 0,
        )
        for axis in data_axes
        if axis is not batch_axis
    )
    selection = None if not terms else Selection(tuple(terms))
    preferences = ViewPreferences(
        repeat_mode=RepeatViewMode.MEAN,
        x_axis_id=x_axis.axis_id,
        batch_axis_ids=(
            () if batch_axis is None else (batch_axis.axis_id,)
        ),
    )
    suggestion = suggest_view(
        output_schema,
        ViewIntent.CURVE,
        selection,
        preferences,
    )
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        detail = " · ".join(reason.message for reason in suggestion.reasons)
        raise ValueError(
            "occupancy progressive curve needs an explicit display choice"
            + ("" if not detail else f": {detail}")
        )
    view = suggestion.spec
    dataset_id = DatasetId(f"scan-preview-{identity}")
    document = FigureDocument(
        f"scan-preview-{identity}",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "Occupancy counts · PROVISIONAL",
                output_schema.fingerprint,
            ),
        ),
        (FigureLayer(_PANEL_ID, dataset_id, view),),
    )
    selections = []
    axes_by_id = {
        axis.axis_id: axis
        for axis in (
            output_schema.repeat_axis,
            *output_schema.point_axes,
            *output_schema.cell_schema.data_axes,
        )
    }
    for term in terms:
        axis = axes_by_id[term.axis_id]
        selections.append(f"{axis.name}={axis.coordinate_at(term.index)}")
    summary = f"x={x_axis.name} · repeat=mean/{output_schema.repeat_axis.size}"
    if batch_axis is not None:
        summary += f" · {batch_axis.name}=batch/{batch_axis.size}"
    if selections:
        summary += " · " + " · ".join(selections)
    transform_peak = transformed_snapshot_peak_nbytes(
        source_schema,
        output_contract.committed_transform,
    )
    evaluation_peak = estimate_view_evaluation_peak_nbytes(output_schema, view)
    raster_peak = estimate_single_curve_raster_peak_nbytes(
        _RASTER_WIDTH,
        _RASTER_HEIGHT,
        evaluated_data_upper_bound_bytes=evaluation_peak,
    )
    # source_terminal may freeze while the render worker still owns its source
    # snapshot.  One additional immutable source copy is cheaper and safer than
    # introducing another cross-thread handoff state machine.
    downstream_peak = (
        transform_peak
        + evaluation_peak
        + raster_peak
        + dataset_storage_nbytes(source_schema)
    )
    return ProgressiveScanSpec(
        output_contract,
        BlockId(f"scan-preview-output-{identity}"),
        document,
        summary,
        transform_peak,
        evaluation_peak,
        ExactDatasetPreviewSpec(source_schema.fingerprint, downstream_peak),
        selection,
        preferences,
    )


class ExactDatasetLiveSlot:
    """Workbench-owned reader slot; no builder/reservation authority escapes."""

    def __init__(self, spec: ExactDatasetPreviewSpec) -> None:
        if not isinstance(spec, ExactDatasetPreviewSpec):
            raise TypeError("spec must be ExactDatasetPreviewSpec")
        self._spec = spec
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._reader: ExactDatasetPreviewReader | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._terminal_snapshot: DatasetPreviewSnapshot | None = None
        self._listener: Callable[[], None] | None = None
        self._pending_change = False
        self._failure: str | None = None
        self._terminal = False
        self._closed = False

    @property
    def spec(self) -> ExactDatasetPreviewSpec:
        return self._spec

    @property
    def terminal(self) -> bool:
        with self._lock:
            return self._terminal

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._lock:
            if self._listener is not None:
                raise RuntimeError("progressive slot already has a listener")
            if self._closed:
                raise RuntimeError("progressive slot is closed")
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            listener()

    def bind(
        self,
        reader: ExactDatasetPreviewReader,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None:
        if not isinstance(reader, ExactDatasetPreviewReader):
            raise TypeError("reader must be ExactDatasetPreviewReader")
        if reader.schema.fingerprint != self.spec.source_schema_fingerprint:
            raise ValueError("progressive reader schema differs from preview spec")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        if reader.stream_generation.value != causation_domain_id:
            raise ValueError(
                "progressive reader generation differs from its causation domain"
            )
        if reader.terminal:
            raise RuntimeError("progressive reader is already terminal")
        listener = None
        with self._lock:
            if self._closed or self._terminal:
                raise RuntimeError("progressive slot is terminal")
            if self._reader is not None:
                raise RuntimeError("progressive slot is already bound")
            self._reader = reader
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id
            self._condition.notify_all()
            listener = self._notify_locked()
        if listener is not None:
            listener()

    def wait_and_freeze(
        self,
        after: DatasetRevision,
        *,
        timeout: float,
    ) -> tuple[str, str, DatasetPreviewSnapshot] | None:
        if not isinstance(after, DatasetRevision):
            raise TypeError("after must be DatasetRevision")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        deadline = time.monotonic() + float(timeout)
        while True:
            with self._condition:
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                terminal_snapshot = self._terminal_snapshot
                if terminal_snapshot is not None:
                    self._terminal_snapshot = None
                    if terminal_snapshot.ref.revision > after:
                        assert self._run_id is not None
                        assert self._causation_domain_id is not None
                        return (
                            self._run_id,
                            self._causation_domain_id,
                            terminal_snapshot,
                        )
                reader = self._reader
                run_id = self._run_id
                causation = self._causation_domain_id
                if reader is None:
                    return None
            remaining = max(0.0, deadline - time.monotonic())
            revision = reader.wait_for_change(after, remaining)
            if revision is not None:
                snapshot = reader.freeze_current()
                assert run_id is not None and causation is not None
                return run_id, causation, snapshot
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            # A sealed builder is not yet a safety-validated terminal.  Wait
            # for cleanup/post-safety to decide the slot instead of spinning.
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._failure is not None
                        or self._terminal
                        or self._closed
                        or self._reader is not reader
                    ),
                    remaining,
                )

    def fail(self, message: str) -> None:
        message = canonical_text(message, "message")
        listener = None
        with self._lock:
            if self._closed:
                return
            if self._failure is None:
                self._failure = message
            self._reader = None
            self._terminal_snapshot = None
            self._terminal = True
            self._condition.notify_all()
            listener = self._notify_locked()
        if listener is not None:
            listener()

    def source_terminal(self) -> None:
        with self._lock:
            if self._closed or self._terminal:
                return
            reader = self._reader
        try:
            if reader is None:
                raise RuntimeError("exact preview reached terminal before reader bind")
            if not reader.terminal:
                raise RuntimeError("exact preview source is not terminal")
            if reader.failed:
                raise RuntimeError("exact preview source aborted")
            final_snapshot = reader.freeze_current()
            if not final_snapshot.coverage.complete:
                raise RuntimeError("exact preview source terminal coverage is incomplete")
        except BaseException as error:
            self.fail(f"{type(error).__name__}: {error}")
            return
        listener = None
        with self._lock:
            if self._closed or self._terminal:
                return
            self._reader = None
            self._terminal_snapshot = final_snapshot
            self._terminal = True
            self._condition.notify_all()
            listener = self._notify_locked()
        if listener is not None:
            listener()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._terminal = True
            self._reader = None
            self._terminal_snapshot = None
            self._listener = None
            self._condition.notify_all()

    def _notify_locked(self) -> Callable[[], None] | None:
        listener = self._listener
        if listener is None:
            self._pending_change = True
        return listener


@dataclass(frozen=True, slots=True)
class _RenderedCandidate:
    run_id: str
    causation_domain_id: str
    output_ref: DatasetRevisionRef
    raster: RasterBuffer
    written_cells: int
    total_cells: int


class ProgressiveScanPreview:
    """One worker-owned raster loop and one owner-thread coherent front."""

    def __init__(
        self,
        slot: ExactDatasetLiveSlot,
        spec: ProgressiveScanSpec,
        presenter: BoardPresenter,
        *,
        submit_worker: Callable[[Callable[[], None]], object],
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(slot, ExactDatasetLiveSlot):
            raise TypeError("slot must be ExactDatasetLiveSlot")
        if not isinstance(spec, ProgressiveScanSpec):
            raise TypeError("spec must be ProgressiveScanSpec")
        if not isinstance(presenter, BoardPresenter):
            raise TypeError("presenter must implement BoardPresenter")
        if not callable(submit_worker) or not callable(request_owner_wake):
            raise TypeError("worker submission and owner wake must be callable")
        if slot.spec != spec.preview_spec:
            raise ValueError("slot and progressive display budgets differ")
        self._owner_thread = threading.get_ident()
        self._slot = slot
        self._spec = spec
        self._request_owner_wake = request_owner_wake
        self._submit_worker = submit_worker
        self._lock = threading.Lock()
        self._candidate: _RenderedCandidate | None = None
        self._fault: BaseException | None = None
        self._watch_started = False
        self._worker_done = False
        self._closed = False
        self._close_complete = False
        self._coverage = "0/0"
        self._presented = False
        self._sequence = 0
        self._port: BoardPublishPort | None = None
        self._presentation = PanelPresentationIdentity(
            _PANEL_ID,
            spec.document.document_id,
            spec.document.revision,
            0,
            0,
        )
        self._board = BoardController(
            BoardModel(
                f"scan-preview-board-{spec.dataset_id.value}",
                0,
                RenderSurface.WORKER_RASTER_LIVE,
                (PanelSlot(_PANEL_ID, "occupancy-curve", _COHERENCE_GROUP),),
            ),
            presenter,
            request_owner_wake,
        )
        slot.set_change_listener(self._source_changed)

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    @property
    def coverage(self) -> str:
        with self._lock:
            return self._coverage

    @property
    def presented(self) -> bool:
        with self._lock:
            return self._presented

    @property
    def worker_done(self) -> bool:
        with self._lock:
            return self._worker_done

    @property
    def terminal(self) -> bool:
        return self._slot.terminal

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def retired(self) -> bool:
        with self._lock:
            return self._close_complete

    def owner_cycle(self) -> None:
        self._require_owner()
        failure = self._slot.failure
        if failure is not None:
            self._set_fault(RuntimeError(failure), invalidate=True)
            return
        with self._lock:
            if self._closed:
                return
            candidate, self._candidate = self._candidate, None
        if candidate is not None:
            self._publish_candidate(candidate)
        self._board.present_pending()

    def close(self) -> None:
        self._require_owner()
        first_close = False
        with self._lock:
            if not self._closed:
                self._closed = True
                self._candidate = None
                if not self._watch_started:
                    self._worker_done = True
                first_close = True
        if first_close:
            self._slot.close()
        self._board.close()
        with self._lock:
            self._close_complete = True

    def _source_changed(self) -> None:
        failure = self._slot.failure
        if failure is not None:
            with self._lock:
                if not self._watch_started:
                    self._worker_done = True
            self._set_fault(RuntimeError(failure))
            return
        submit = False
        with self._lock:
            if not self._closed and not self._watch_started:
                self._watch_started = True
                submit = True
        if submit:
            try:
                submitted = self._submit_worker(self._watch)
                if isinstance(submitted, Future) and submitted.done():
                    error = submitted.exception()
                    if error is not None:
                        raise error
            except BaseException as error:
                with self._lock:
                    self._worker_done = True
                self._set_fault(error)
                try:
                    self._slot.fail(f"{type(error).__name__}: {error}")
                except BaseException:
                    pass
        self._request_owner_wake()

    def _watch(self) -> None:
        last_revision = DatasetRevision(0)
        candidate = None
        source = None
        output = None
        evaluated = None
        rendered = None
        renderer = None
        try:
            with self._lock:
                if self._closed:
                    return
            failure = self._slot.failure
            if failure is not None:
                raise RuntimeError(failure)
            evaluator = FigureEvaluator(
                FigureEvaluationPolicy(
                    max_live_nbytes=self._spec.evaluation_peak_bytes,
                )
            )
            renderer = SingleCurveAggRenderer(
                self._spec.document,
                width=_RASTER_WIDTH,
                height=_RASTER_HEIGHT,
            )
            while True:
                with self._lock:
                    if self._closed:
                        break
                candidate = self._slot.wait_and_freeze(
                    last_revision,
                    timeout=0.1,
                )
                if candidate is None:
                    if self._slot.terminal:
                        break
                    continue
                run_id, causation, source = candidate
                candidate = None
                if source.ref.revision <= last_revision:
                    source = None
                    continue
                last_revision = source.ref.revision
                output_ref = DatasetRevisionRef(
                    self._spec.output_block_id,
                    source.ref.stream_generation,
                    self._spec.output_contract.output_schema_fingerprint,
                    source.ref.revision,
                )
                output = materialize_transformed_snapshot(
                    source.snapshot,
                    self._spec.output_contract.committed_transform,
                    output_ref=output_ref,
                    output_schema=(
                        self._spec.output_contract.output_dataset_schema
                    ),
                    memory_limit_bytes=self._spec.transform_peak_bytes,
                )
                evaluated = evaluator.evaluate(
                    self._spec.document,
                    ResolvedDatasetMap(
                        (ResolvedDataset(self._spec.dataset_id, output),)
                    ),
                    cancel_requested=lambda: self.closed,
                )
                rendered = _RenderedCandidate(
                    run_id,
                    causation,
                    output_ref,
                    renderer.render(evaluated),
                    source.coverage.written_cells,
                    source.coverage.total_cells,
                )
                with self._lock:
                    if self._closed:
                        break
                    self._candidate = rendered
                    self._coverage = (
                        f"{rendered.written_cells}/{rendered.total_cells}"
                    )
                source = output = evaluated = rendered = None
                self._request_owner_wake()
        except BaseException as error:
            self._set_fault(error)
            try:
                self._slot.fail(f"{type(error).__name__}: {error}")
            except BaseException:
                pass
        finally:
            candidate = source = output = evaluated = rendered = None
            if renderer is not None:
                try:
                    renderer.close()
                except BaseException as error:
                    self._set_fault(error)
                    try:
                        self._slot.fail(f"{type(error).__name__}: {error}")
                    except BaseException:
                        pass
                renderer = None
            with self._lock:
                self._worker_done = True
            self._request_owner_wake()

    def _publish_candidate(self, candidate: _RenderedCandidate) -> None:
        source_ref = candidate.output_ref
        source = SourceIdentity(
            self._spec.dataset_id,
            source_ref.block_id,
            source_ref.stream_generation,
            source_ref.schema_fingerprint,
        )
        stamp = CoherenceStamp(
            candidate.run_id,
            candidate.causation_domain_id,
            "exact-dataset-revision",
            source_ref.schema_fingerprint,
            canonical_digest(
                {
                    "owner": "zlc_workbench.progressive-scan",
                    "run_id": candidate.run_id,
                    "causation_domain_id": candidate.causation_domain_id,
                    "source_ref": dataset_revision_ref_to_tree(source_ref),
                }
            ),
            (EvaluatedInput(self._spec.dataset_id, source_ref),),
            (self._presentation,),
        )
        if self._port is None:
            self._port = self._board.open_publish_port(
                (PanelSourceBinding(source, self._presentation),)
            )
        sequence = self._sequence
        self._sequence += 1
        token = self._port.admit(
            sequence,
            ((_COHERENCE_GROUP, stamp),),
        )
        frame = BoardFrame(
            self._board.model.board_id,
            self._board.model.layout_generation,
            sequence,
            (
                PanelFrame(
                    _PANEL_ID,
                    _COHERENCE_GROUP,
                    source,
                    stamp,
                    candidate.raster,
                ),
            ),
        )
        if self._port.publish(token, frame):
            with self._lock:
                self._presented = True

    def _set_fault(self, error: BaseException, *, invalidate: bool = False) -> None:
        first_fault = False
        should_invalidate = False
        with self._lock:
            if self._fault is None:
                self._fault = detached_render_fault(error)
                first_fault = True
            self._candidate = None
            if invalidate:
                self._presented = False
                should_invalidate = True
        if should_invalidate:
            try:
                self._board.invalidate()
            except BaseException:
                pass
        if first_fault:
            self._request_owner_wake()

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("ProgressiveScanPreview owner method used off owner thread")


__all__ = [
    "build_occupancy_progressive_spec",
    "ExactDatasetLiveSlot",
    "ProgressiveScanPreview",
    "ProgressiveScanSpec",
    "ScanDisplayIntent",
]
