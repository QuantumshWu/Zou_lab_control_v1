"""Raw live IMAGE composition for Capture and Camera workbenches.

Acquisition owns immutable dataset revisions; this owner evaluates and renders
one accepted revision at a time.  Area/Cross/Fit remain Figure concerns and no
ROI, reduction, or derived scalar state enters the Measurement lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import (
    EvaluatedImage,
    EvaluatedInput,
    FigureDocument,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    validate_view_spec,
)
from zlc_frontend.image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
    resolve_image_color_limits,
)
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.render import (
    BoardFrame,
    CoherenceStamp,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
    detached_render_fault,
)
from zlc_neutral_atom.runtime.dataset import MonitorDatasetSnapshot

from .live_slot import LiveDatasetSlot
from .workspace import (
    BoardController,
    BoardPublishPort,
    PanelSourceBinding,
)


@dataclass(frozen=True, slots=True)
class LiveFrontStatus:
    """Diagnostics attached to one exact board sequence."""

    sequence: int
    raw_coverage: object
    image_display_revision: int
    image_color_limits: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _Configuration:
    epoch: int
    board_id: str
    layout_generation: int
    panel_id: str
    coherence_group: str
    presentation: PanelPresentationIdentity
    document: FigureDocument
    display: ImageDisplayState
    viewport: ImageViewportTransform


@dataclass(frozen=True, slots=True)
class _Candidate:
    run_id: str
    causation_domain_id: str
    snapshot: MonitorDatasetSnapshot
    configuration: _Configuration


@dataclass(frozen=True, slots=True)
class _RenderJob:
    candidate: _Candidate
    source: SourceIdentity
    stamp: CoherenceStamp
    sequence: int
    token: object
    port: BoardPublishPort


def _configuration(
    *,
    epoch: int,
    board: BoardController,
    document: FigureDocument,
    display: ImageDisplayState,
    viewport: ImageViewportTransform,
) -> _Configuration:
    model = board.model
    if len(model.panels) != 1:
        raise ValueError("raw live IMAGE board requires exactly one panel")
    panel = model.panels[0]
    if len(document.layers) != 1 or document.layers[0].layer_id != panel.panel_id:
        raise ValueError("live IMAGE document layer must match the board panel")
    if viewport.viewport_revision != display.revision:
        raise ValueError("image viewport and display revisions differ")
    image_viewport_for_display_state(display, viewport)
    return _Configuration(
        epoch,
        model.board_id,
        model.layout_generation,
        panel.panel_id,
        panel.coherence_group,
        PanelPresentationIdentity(
            panel.panel_id,
            document.document_id,
            document.revision,
            0,
            display.revision,
        ),
        document,
        display,
        viewport,
    )


class LiveBoardController:
    """Latest-only raw IMAGE renderer with one immutable atomic present."""

    def __init__(
        self,
        slot: LiveDatasetSlot,
        document: FigureDocument,
        board: BoardController,
        *,
        submit_worker: Callable[[Callable[[], None]], object],
        request_owner_wake: Callable[[], None],
        image_display: ImageDisplayState,
        image_viewport: ImageViewportTransform,
        raster_size: tuple[int, int] = (800, 520),
        worker_thread_affine: bool = False,
    ) -> None:
        if not isinstance(slot, LiveDatasetSlot):
            raise TypeError("slot must be LiveDatasetSlot")
        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if not isinstance(board, BoardController):
            raise TypeError("board must be BoardController")
        if not callable(submit_worker) or not callable(request_owner_wake):
            raise TypeError("worker submission and owner wake must be callable")
        if not isinstance(image_display, ImageDisplayState):
            raise TypeError("image_display must be ImageDisplayState")
        if not isinstance(image_viewport, ImageViewportTransform):
            raise TypeError("image_viewport must be ImageViewportTransform")
        if not worker_thread_affine:
            raise ValueError("live Agg panels require a thread-affine worker lane")
        if (
            not isinstance(raster_size, tuple)
            or len(raster_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in raster_size
            )
        ):
            raise ValueError("raster_size must contain two positive integers")
        self._validate_document(slot, document)
        self._slot = slot
        self._board = board
        self._configuration = _configuration(
            epoch=0,
            board=board,
            document=document,
            display=image_display,
            viewport=image_viewport,
        )
        self._evaluator = FigureEvaluator()
        self._raster_size = raster_size
        self._renderer = None
        self._submit_worker = submit_worker
        self._request_owner_wake = request_owner_wake
        self._owner_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._worker_gate = threading.Lock()
        self._dirty = False
        self._active = False
        self._candidate: _Candidate | None = None
        self._display_rerender: _Candidate | None = None
        self._presented_source: tuple[str, str, MonitorDatasetSnapshot] | None = None
        self._published_source: tuple[
            int, str, str, MonitorDatasetSnapshot
        ] | None = None
        self._port: BoardPublishPort | None = None
        self._source: SourceIdentity | None = None
        self._sequence = 0
        self._fault: BaseException | None = None
        self._front_status: LiveFrontStatus | None = None
        self._image_color_limits: tuple[float, float] | None = None
        self._image_relim_mode: RelimMode | None = None
        self._front_invalidated = False
        self._presentation_frozen = False
        self._closed = False
        self._close_complete = False
        slot.set_change_listener(self._source_changed)

    @staticmethod
    def _validate_document(slot: LiveDatasetSlot, document: FigureDocument) -> None:
        schema = slot.spec.dataset_edge.schema
        if (
            len(document.datasets) != 1
            or document.datasets[0].dataset_id != slot.dataset_id
            or document.datasets[0].schema_fingerprint != schema.fingerprint
            or len(document.layers) != 1
            or document.layers[0].dataset_id != slot.dataset_id
        ):
            raise ValueError("live document must contain the slot's one dataset and layer")
        view = document.layers[0].view
        validate_view_spec(schema, view)
        if view.intent is not ViewIntent.IMAGE:
            raise ValueError("live board requires one IMAGE document")

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    @property
    def front_status(self) -> LiveFrontStatus | None:
        with self._lock:
            return self._front_status

    def reconfigure_image_display(
        self,
        state: ImageDisplayState,
        viewport: ImageViewportTransform,
    ) -> None:
        """Render a display draft from the exact front the operator touched."""

        self._require_owner()
        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        with self._lock:
            if self._closed or self._fault is not None:
                raise RuntimeError("live board controller is unavailable")
            previous = self._configuration
            presented = self._presented_source
        if state == previous.display and viewport == previous.viewport:
            return
        if state.revision <= previous.display.revision:
            raise ValueError("image display revision must increase")
        if viewport.axes != previous.viewport.axes:
            raise ValueError("display reconfiguration cannot change source axes")
        configuration = _configuration(
            epoch=previous.epoch + 1,
            board=self._board,
            document=previous.document,
            display=state,
            viewport=viewport,
        )
        with self._lock:
            if self._configuration is not previous:
                raise RuntimeError("image display changed concurrently")
            self._configuration = configuration
            self._port = None
            if presented is not None:
                self._display_rerender = _Candidate(*presented, configuration)
            self._dirty = True
            if self._active or self._presentation_frozen:
                return
            self._active = True
            self._dirty = False
            candidate, self._display_rerender = self._display_rerender, None
            if candidate is not None:
                self._candidate = candidate
        if candidate is None:
            self._submit(self._freeze_latest)
        else:
            self._request_owner_wake()

    def accept_presented_front(self, sequence: int) -> bool:
        self._require_owner()
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("presented sequence must be a nonnegative integer")
        with self._lock:
            published = self._published_source
            if published is None or published[0] != sequence:
                return False
            self._presented_source = published[1:]
            return True

    def freeze_presentation(self) -> None:
        self._require_owner()
        with self._lock:
            if self._closed:
                return
            self._presentation_frozen = True
            self._dirty = False
            self._active = False
            self._candidate = None
            self._display_rerender = None
            self._published_source = None
            self._port = None
        self._board.freeze_front()

    def _source_changed(self) -> None:
        if self._slot.failure is not None or self._slot.withdrawn:
            self._revoke_source(self._slot.failure)
            self._request_owner_wake()
            return
        if self._slot.notification_failure is not None:
            self._set_fault(RuntimeError(self._slot.notification_failure))
            return
        with self._lock:
            if self._closed or self._presentation_frozen or self._fault is not None:
                return
            self._display_rerender = None
            self._dirty = True
            if self._active:
                return
            self._active = True
            self._dirty = False
        self._submit(self._freeze_latest)

    def _freeze_latest(self) -> None:
        try:
            frozen = self._slot.freeze_current()
            with self._lock:
                if self._closed or self._presentation_frozen or self._fault is not None:
                    self._active = False
                    return
                self._candidate = _Candidate(*frozen, self._configuration)
            self._request_owner_wake()
        except BaseException as error:
            if self._slot.withdrawn and self._slot.failure is None:
                with self._lock:
                    self._active = False
                self._request_owner_wake()
            else:
                self._set_fault(error)

    def reconcile_faults(self) -> bool:
        self._require_owner()
        failure = self._slot.failure or self._slot.notification_failure
        if failure is not None or self._slot.withdrawn:
            self._revoke_source(failure)
            with self._lock:
                invalidate = not self._front_invalidated
                self._front_invalidated = True
                self._front_status = None
            if invalidate:
                self._board.invalidate()
            return True
        return False

    def admit_pending(self) -> bool:
        """Owner-thread admission; expensive evaluation and raster stay worker-side."""

        self._require_owner()
        if self.reconcile_faults():
            return False
        with self._lock:
            if self._closed or self._presentation_frozen or self._fault is not None:
                return False
            candidate, self._candidate = self._candidate, None
            configuration = self._configuration
        if candidate is None:
            return False
        if candidate.configuration is not configuration:
            self._finish_cycle()
            return False
        try:
            owned = candidate.snapshot.snapshot
            ref = owned.ref
            head = candidate.snapshot.head
            if head is None:
                raise RuntimeError("live snapshot has no head event")
            source = SourceIdentity(
                self._slot.dataset_id,
                ref.block_id,
                ref.stream_generation,
                ref.schema_fingerprint,
            )
            evaluated_input = EvaluatedInput(self._slot.dataset_id, ref)
            stamp = CoherenceStamp(
                candidate.run_id,
                candidate.causation_domain_id,
                "single-source-event-payload",
                ref.schema_fingerprint,
                head.payload_digest,
                (evaluated_input,),
                (configuration.presentation,),
            )
            if self._port is None or source != self._source:
                self._port = self._board.open_publish_port(
                    (PanelSourceBinding(source, configuration.presentation),)
                )
                self._source = source
            port = self._port
            sequence = self._sequence
            self._sequence += 1
            token = port.admit(
                sequence,
                ((configuration.coherence_group, stamp),),
            )
            self._submit(
                lambda: self._render(
                    _RenderJob(candidate, source, stamp, sequence, token, port)
                ),
                ends_cycle=True,
            )
            return True
        except BaseException as error:
            self._set_fault(error)
            return False

    def _render(self, job: _RenderJob) -> None:
        try:
            if not self._job_is_current(job):
                return
            configuration = job.candidate.configuration
            owned = job.candidate.snapshot.snapshot
            evaluated_input = EvaluatedInput(self._slot.dataset_id, owned.ref)
            evaluated = self._evaluator.evaluate(
                configuration.document,
                ResolvedDatasetMap(
                    (ResolvedDataset(self._slot.dataset_id, owned),)
                ),
                cancel_requested=lambda: self._closed,
            )
            if evaluated.inputs != (evaluated_input,):
                raise RuntimeError("figure evaluation changed the admitted revision")
            if (
                len(evaluated.layers) != 1
                or len(evaluated.layers[0].cells) != 1
                or len(evaluated.layers[0].cells[0].series) != 1
            ):
                raise RuntimeError("live IMAGE document must evaluate one image")
            data = evaluated.layers[0].cells[0].series[0].data
            if not isinstance(data, EvaluatedImage):
                raise RuntimeError("live IMAGE document evaluated another kind")
            with self._lock:
                previous_limits = self._image_color_limits
                previous_relim = self._image_relim_mode
            data_range, color_limits = resolve_image_color_limits(
                data,
                configuration.display,
                current_color_limits=previous_limits,
                previous_relim_mode=previous_relim,
            )
            from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

            if self._renderer is None:
                self._renderer = ImagePanelAggRenderer(
                    width=self._raster_size[0],
                    height=self._raster_size[1],
                )
            raster, geometry = self._renderer.render(
                data,
                configuration.viewport,
                configuration.display,
                color_limits=color_limits,
                data_range=data_range,
                title=configuration.document.layers[0].layer_id,
                value_label=configuration.document.datasets[0].label,
                distribution_identity=evaluated_input.ref,
            )
            payload = ImagePanelPayload(
                data,
                evaluated_input,
                configuration.viewport,
                data_range,
                configuration.display.colormap,
                color_limits,
                geometry,
            )
            frame = BoardFrame(
                configuration.board_id,
                configuration.layout_generation,
                job.sequence,
                (
                    PanelFrame(
                        configuration.panel_id,
                        configuration.coherence_group,
                        job.source,
                        job.stamp,
                        raster,
                        payload,
                    ),
                ),
            )
            if not self._job_is_current(job) or not job.port.publish(job.token, frame):
                return
            status = LiveFrontStatus(
                job.sequence,
                job.candidate.snapshot.coverage,
                configuration.display.revision,
                color_limits,
            )
            with self._lock:
                if self._configuration is not configuration or self._port is not job.port:
                    return
                self._front_status = status
                self._published_source = (
                    job.sequence,
                    job.candidate.run_id,
                    job.candidate.causation_domain_id,
                    job.candidate.snapshot,
                )
                self._image_color_limits = color_limits
                if data_range is not None or configuration.display.relim_mode is RelimMode.FIXED:
                    self._image_relim_mode = configuration.display.relim_mode
            self._request_owner_wake()
        except BaseException as error:
            self._set_fault(error, expected=job)

    def _finish_cycle(self) -> None:
        with self._lock:
            if self._closed or self._presentation_frozen or self._fault is not None:
                self._active = False
                return
            if not self._dirty:
                self._active = False
                return
            self._dirty = False
            candidate, self._display_rerender = self._display_rerender, None
            if candidate is not None:
                self._candidate = candidate
        if candidate is None:
            self._submit(self._freeze_latest)
        else:
            self._request_owner_wake()

    def _submit(self, work: Callable[[], None], *, ends_cycle: bool = False) -> None:
        def serialized() -> None:
            with self._worker_gate:
                try:
                    work()
                finally:
                    if ends_cycle:
                        self._finish_cycle()

        try:
            self._submit_worker(serialized)
        except BaseException as error:
            self._set_fault(error)

    def _job_is_current(self, job: _RenderJob) -> bool:
        with self._lock:
            return (
                not self._closed
                and self._fault is None
                and self._configuration is job.candidate.configuration
                and self._port is job.port
            )

    def _revoke_source(self, failure: str | None) -> None:
        def install() -> bool:
            with self._lock:
                if self._closed:
                    return False
                if failure is not None and self._fault is None:
                    self._fault = RuntimeError(failure)
                self._candidate = None
                self._display_rerender = None
                self._published_source = None
                self._active = False
                self._dirty = False
                self._port = None
                return True

        self._board.revoke_pending_publication(install)

    def _set_fault(
        self,
        error: BaseException,
        *,
        expected: _RenderJob | None = None,
    ) -> None:
        def install() -> bool:
            with self._lock:
                if self._closed:
                    return False
                if expected is not None and (
                    self._configuration is not expected.candidate.configuration
                    or self._port is not expected.port
                ):
                    return False
                if self._fault is None:
                    self._fault = detached_render_fault(error)
                self._candidate = None
                self._display_rerender = None
                self._published_source = None
                self._active = False
                self._dirty = False
                self._port = None
                return True

        if self._board.revoke_pending_publication(install):
            self._request_owner_wake()

    def close(self) -> None:
        self._require_owner()
        with self._lock:
            if self._close_complete:
                return
            self._closed = True
            self._active = False
            self._dirty = False
            self._candidate = None
            self._display_rerender = None
            self._front_status = None
        try:
            self._slot.close()
        finally:
            self._board.close()

        def close_renderer() -> None:
            with self._worker_gate:
                renderer, self._renderer = self._renderer, None
                if renderer is not None:
                    renderer.close()

        self._submit_worker(close_renderer)
        with self._lock:
            self._fault = None
            self._close_complete = True

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("live board presentation is owner-thread affine")


__all__ = ["LiveBoardController", "LiveFrontStatus"]
