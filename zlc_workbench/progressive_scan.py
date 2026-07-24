"""One provisional occupancy-curve branch for an exact autonomous scan."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import threading
from typing import Callable

from zlc_data import (
    DatasetRevision,
    DatasetRevisionRef,
    dataset_revision_ref_to_tree,
)
from zlc_frontend.matplotlib_render import (
    SinglePanelAggRenderer,
)
from zlc_frontend.curve_display import (
    CurveDisplayState,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import (
    FigureEvaluator,
    EvaluatedInput,
    ResolvedDataset,
    ResolvedDatasetMap,
)
from zlc_frontend.scan_preview import (
    SCAN_CURVE_PANEL_ID,
    ScanCurvePresentation,
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
from zlc_neutral_atom.scan.application import PreparedExactScan
from zlc_storage import canonical_digest

from .exact_live_slot import ExactDatasetLiveSlot
from .workspace import (
    BoardController,
    BoardModel,
    BoardPublishPort,
    PanelSlot,
    PanelSourceBinding,
)


_COHERENCE_GROUP = "scan-output"
_RASTER_WIDTH = 800
_RASTER_HEIGHT = 520


@dataclass(frozen=True, slots=True)
class ProgressiveScanSpec:
    """Composition pairing of one authoritative output and frontend view."""

    output_owner: PreparedExactScan
    presentation: ScanCurvePresentation

    def __post_init__(self) -> None:
        if not isinstance(self.output_owner, PreparedExactScan):
            raise TypeError("output_owner must be PreparedExactScan")
        if not isinstance(self.presentation, ScanCurvePresentation):
            raise TypeError("presentation must be ScanCurvePresentation")
        if (
            self.presentation.document.datasets[0].schema_fingerprint
            != self.output_owner.output_contract.output_schema_fingerprint
        ):
            raise ValueError("progressive presentation belongs to another output schema")


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
        if slot.spec != spec.output_owner.preview_spec:
            raise ValueError("slot and progressive display contracts differ")
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
                f"scan-preview-board-{spec.presentation.dataset_id.value}",
                0,
                RenderSurface.WORKER_RASTER_LIVE,
                (
                    PanelSlot(
                        SCAN_CURVE_PANEL_ID,
                        "occupancy-curve",
                        _COHERENCE_GROUP,
                    ),
                ),
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
        return self._spec.presentation.interactive_curve

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
        if not self._spec.presentation.interactive_curve:
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
            evaluator = FigureEvaluator()
            renderer = SinglePanelAggRenderer(
                self._spec.presentation.document,
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
                        output = self._spec.output_owner.materialize_provisional_output(
                            source
                        )
                        last_revision = source.ref.revision
                        next_evaluated = evaluator.evaluate(
                            self._spec.presentation.document,
                            ResolvedDatasetMap(
                                (
                                    ResolvedDataset(
                                        self._spec.presentation.dataset_id,
                                        output,
                                    ),
                                )
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
                if self._spec.presentation.interactive_curve:
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
                    SCAN_CURVE_PANEL_ID,
                    self._spec.presentation.document.document_id,
                    self._spec.presentation.document.revision,
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
            self._spec.presentation.dataset_id,
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
            (EvaluatedInput(self._spec.presentation.dataset_id, source_ref),),
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
                    SCAN_CURVE_PANEL_ID,
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
    "ProgressiveScanPreview",
    "ProgressiveScanSpec",
]
