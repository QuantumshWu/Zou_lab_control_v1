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
    SinglePanelAggRenderer,
    estimate_live_panel_raster_peak_nbytes,
)
from zlc_frontend.curve_display import (
    CurveDisplayState,
    numeric_curve_coordinates,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import (
    CURVE_CONTRACT,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureEvaluator,
    FigureLayer,
    AxisViewRole,
    EvaluatedAxis,
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
    CurvePanelPayload,
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
    interactive_curve: bool
    interaction_unavailable_reason: str | None

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
        if not isinstance(self.interactive_curve, bool):
            raise TypeError("interactive_curve must be bool")
        if self.interactive_curve:
            if self.interaction_unavailable_reason is not None:
                raise ValueError(
                    "interactive curve cannot have an unavailable reason"
                )
        else:
            object.__setattr__(
                self,
                "interaction_unavailable_reason",
                canonical_text(
                    self.interaction_unavailable_reason,
                    "interaction_unavailable_reason",
                ),
            )
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
    try:
        numeric_curve_coordinates(
            EvaluatedAxis(
                x_axis.axis_id,
                x_axis.name,
                x_axis.role,
                x_axis.unit,
                tuple(range(x_axis.size)),
                tuple(x_axis.coordinates),
            )
        )
    except (TypeError, ValueError) as error:
        interactive_curve = False
        interaction_unavailable_reason = f"{type(error).__name__}: {error}"
    else:
        interactive_curve = True
        interaction_unavailable_reason = None
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
    batch_limit = CURVE_CONTRACT.maximum_batch_series
    if display_intent.site_mode == "batch":
        if not 1 < site_axis.size <= batch_limit:
            raise ValueError(
                "site batch display requires between 2 and "
                f"{batch_limit} sites"
            )
        batch_axis = site_axis
    elif display_intent.site_mode == "select":
        if display_intent.site_index >= site_axis.size:
            raise ValueError("selected site index exceeds the declared SITE axis")
        batch_axis = None
    else:
        batch_axis = site_axis if 1 < site_axis.size <= batch_limit else None
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
    if not interactive_curve:
        assert interaction_unavailable_reason is not None
        summary += (
            " · static curve (interactive selector unavailable: "
            f"{interaction_unavailable_reason})"
        )
    transform_peak = transformed_snapshot_peak_nbytes(
        source_schema,
        output_contract.committed_transform,
    )
    evaluation_peak = estimate_view_evaluation_peak_nbytes(output_schema, view)
    raster_peak = estimate_live_panel_raster_peak_nbytes(
        _RASTER_WIDTH,
        _RASTER_HEIGHT,
        evaluated_data_upper_bound_bytes=evaluation_peak,
        extra_retained_fronts=(1 if interactive_curve else 0),
        extra_retained_evaluated_data_bytes=(
            evaluation_peak if interactive_curve else 0
        ),
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
        interactive_curve,
        interaction_unavailable_reason,
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
    presentation: PanelPresentationIdentity
    display_state: CurveDisplayState
    raster: RasterBuffer
    display_payload: CurvePanelPayload | None
    curve_has_valid_samples: bool
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
        curve_display: CurveDisplayState,
        submit_worker: Callable[[Callable[[], None]], object],
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(slot, ExactDatasetLiveSlot):
            raise TypeError("slot must be ExactDatasetLiveSlot")
        if not isinstance(spec, ProgressiveScanSpec):
            raise TypeError("spec must be ProgressiveScanSpec")
        if not isinstance(presenter, BoardPresenter):
            raise TypeError("presenter must implement BoardPresenter")
        if not isinstance(curve_display, CurveDisplayState):
            raise TypeError("curve_display must be CurveDisplayState")
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
        self._candidate_condition = threading.Condition(self._lock)
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
        self._port_source: SourceIdentity | None = None
        self._port_presentation: PanelPresentationIdentity | None = None
        self._curve_display = curve_display
        self._configuration_epoch = 0
        self._curve_y_limits: tuple[float, float] | None = None
        self._curve_relim_mode: RelimMode | None = None
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
    def curve_display(self) -> CurveDisplayState:
        with self._lock:
            return self._curve_display

    @property
    def interactive_curve(self) -> bool:
        return self._spec.interactive_curve

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
        with self._candidate_condition:
            if self._closed:
                return
            candidate = self._candidate
            current_state = self._curve_display
        try:
            if candidate is not None and candidate.display_state == current_state:
                self._publish_candidate(candidate)
            self._board.present_pending()
        finally:
            # Capacity one extends through the real owner-thread present
            # boundary.  Releasing the worker before this point makes the
            # next relim/deadband baseline depend on thread scheduling rather
            # than the previously accepted front.
            with self._candidate_condition:
                if self._candidate is candidate:
                    self._candidate = None
                self._candidate_condition.notify_all()

    def reconfigure_curve_display(self, state: CurveDisplayState) -> None:
        """Replace display-only curve intent without touching the scan source."""

        self._require_owner()
        if not isinstance(state, CurveDisplayState):
            raise TypeError("state must be CurveDisplayState")
        if not self._spec.interactive_curve:
            raise RuntimeError("this progressive curve uses a static fallback")
        with self._candidate_condition:
            if self._closed:
                raise RuntimeError("progressive preview is closed")
            if self._fault is not None:
                raise RuntimeError("progressive preview is faulted")
            current = self._curve_display
            if state == current:
                return
            if state.revision != current.revision + 1:
                raise ValueError("curve display revision must advance exactly once")
            self._curve_display = state
            self._configuration_epoch += 1
            self._candidate = None
            self._port = None
            self._port_source = None
            self._port_presentation = None
            self._candidate_condition.notify_all()
        # Revoke an admitted old-revision raster but deliberately retain the
        # visible front until the worker repaints this same exact dataset.
        self._board.revoke_pending_publication()

    def close(self) -> None:
        self._require_owner()
        first_close = False
        with self._candidate_condition:
            if not self._closed:
                self._closed = True
                self._candidate = None
                if not self._watch_started:
                    self._worker_done = True
                first_close = True
                self._candidate_condition.notify_all()
        if first_close:
            self._slot.close()
        self._board.close()
        with self._lock:
            self._close_complete = True

    def accept_worker_completion(self, error: BaseException | None) -> None:
        """Settle a watcher Future, including cancellation before it started."""

        self._require_owner()
        if error is not None and not isinstance(error, BaseException):
            raise TypeError("worker completion error must be BaseException or None")
        if error is not None:
            self._set_fault(error)
        with self._candidate_condition:
            # A normal watcher sets this in its own finally.  A Future that was
            # cancelled while queued never enters _watch(), so the owner must
            # close that otherwise permanent retirement gap explicitly.
            self._worker_done = True
            self._candidate_condition.notify_all()

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
        frozen_candidate = None
        source = None
        last_evaluated = None
        last_output_ref = None
        last_run_id = None
        last_causation = None
        last_written_cells = 0
        last_total_cells = 0
        rendered_output_ref = None
        rendered_epoch = -1
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
            renderer = SinglePanelAggRenderer(
                self._spec.document,
                width=_RASTER_WIDTH,
                height=_RASTER_HEIGHT,
            )
            while True:
                with self._lock:
                    if self._closed:
                        break
                    target_epoch = self._configuration_epoch
                frozen_candidate = self._slot.wait_and_freeze(
                    last_revision,
                    # Display-only commits are observed within one UI cycle
                    # without introducing another worker or touching the Run.
                    timeout=0.04,
                )
                if frozen_candidate is not None:
                    run_id, causation, source = frozen_candidate
                    frozen_candidate = None
                    if source.ref.revision > last_revision:
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
                        last_revision = source.ref.revision
                        next_evaluated = evaluator.evaluate(
                            self._spec.document,
                            ResolvedDatasetMap(
                                (ResolvedDataset(self._spec.dataset_id, output),)
                            ),
                            cancel_requested=lambda: self.closed,
                        )
                        last_evaluated = next_evaluated
                        last_output_ref = output.ref
                        last_run_id = run_id
                        last_causation = causation
                        last_written_cells = source.coverage.written_cells
                        last_total_cells = source.coverage.total_cells
                        output = None
                    source = None

                with self._lock:
                    if self._closed:
                        break
                    target_epoch = self._configuration_epoch
                    display_state = self._curve_display
                    accepted_y_limits = self._curve_y_limits
                    accepted_relim_mode = self._curve_relim_mode
                needs_render = (
                    last_evaluated is not None
                    and (
                        rendered_output_ref != last_output_ref
                        or rendered_epoch != target_epoch
                    )
                )
                if not needs_render:
                    if self._slot.terminal:
                        # The exact source may finish before the owner swaps to
                        # the canonical FINAL view.  Keep this same renderer
                        # available for display-only gestures on the frozen
                        # terminal revision; close() is the sole retirement
                        # signal and wakes this condition without polling.
                        with self._candidate_condition:
                            self._candidate_condition.wait_for(
                                lambda: (
                                    self._closed
                                    or self._configuration_epoch != target_epoch
                                    or self._curve_display != display_state
                                )
                            )
                            if self._closed:
                                break
                    continue

                assert last_evaluated is not None
                assert last_output_ref is not None
                assert last_run_id is not None
                assert last_causation is not None
                evaluated = last_evaluated
                display_payload = None
                curve_has_valid_samples = False
                if self._spec.interactive_curve:
                    raster, display_payload = renderer.render_interactive_curve(
                        evaluated,
                        display_state,
                        current_y_limits=accepted_y_limits,
                        previous_relim_mode=accepted_relim_mode,
                    )
                    curve_has_valid_samples = any(
                        bool(series.data.validity.any())
                        for series in display_payload.series
                    )
                else:
                    raster = renderer.render(evaluated)
                presentation = PanelPresentationIdentity(
                    _PANEL_ID,
                    self._spec.document.document_id,
                    self._spec.document.revision,
                    0,
                    display_state.revision,
                )
                rendered = _RenderedCandidate(
                    last_run_id,
                    last_causation,
                    last_output_ref,
                    presentation,
                    display_state,
                    raster,
                    display_payload,
                    curve_has_valid_samples,
                    last_written_cells,
                    last_total_cells,
                )
                installed = False
                with self._lock:
                    if self._closed:
                        break
                    if (
                        self._configuration_epoch == target_epoch
                        and self._curve_display == display_state
                    ):
                        self._candidate = rendered
                        self._coverage = (
                            f"{rendered.written_cells}/{rendered.total_cells}"
                        )
                        rendered_output_ref = last_output_ref
                        rendered_epoch = target_epoch
                        installed = True
                installed_candidate = rendered if installed else None
                evaluated = rendered = None
                if installed:
                    assert installed_candidate is not None
                    self._request_owner_wake()
                    with self._candidate_condition:
                        self._candidate_condition.wait_for(
                            lambda: (
                                self._closed
                                or self._candidate is not installed_candidate
                                or self._configuration_epoch != target_epoch
                                or self._curve_display != display_state
                            )
                        )
                    installed_candidate = None
        except BaseException as error:
            self._set_fault(error)
            try:
                self._slot.fail(f"{type(error).__name__}: {error}")
            except BaseException:
                pass
        finally:
            frozen_candidate = source = last_evaluated = evaluated = rendered = None
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
        with self._lock:
            if self._closed or candidate.display_state != self._curve_display:
                return
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
            (candidate.presentation,),
        )
        if (
            self._port is None
            or self._port_source != source
            or self._port_presentation != candidate.presentation
        ):
            self._port = self._board.open_publish_port(
                (PanelSourceBinding(source, candidate.presentation),)
            )
            self._port_source = source
            self._port_presentation = candidate.presentation
        port = self._port
        sequence = self._sequence
        self._sequence += 1
        token = port.admit(
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
                    candidate.display_payload,
                ),
            ),
        )
        if port.publish(token, frame):
            with self._lock:
                if (
                    not self._closed
                    and self._curve_display == candidate.display_state
                    and self._port is port
                ):
                    self._presented = True
                    payload = candidate.display_payload
                    if payload is not None:
                        if (
                            candidate.curve_has_valid_samples
                            or candidate.display_state.relim_mode is RelimMode.FIXED
                        ):
                            self._curve_y_limits = payload.viewport.y_limits
                        self._curve_relim_mode = candidate.display_state.relim_mode

    def _set_fault(self, error: BaseException, *, invalidate: bool = False) -> None:
        first_fault = False
        should_invalidate = False
        with self._candidate_condition:
            if self._fault is None:
                self._fault = detached_render_fault(error)
                first_fault = True
            self._candidate = None
            if invalidate:
                self._presented = False
                should_invalidate = True
            self._candidate_condition.notify_all()
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
