"""Finite exact-capture live preview composition for the target Workbench."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

from zlc_frontend.figure import (
    DatasetId,
    EvaluatedCurve,
    EvaluatedImage,
    EvaluatedInput,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    validate_view_spec,
)
from zlc_frontend.image_raster import rasterize_image_gray8
from zlc_frontend.render import (
    BoardFrame,
    CoherenceStamp,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
    detached_render_fault,
)
from zlc_neutral_atom.monitor_application import (
    CameraMonitorLiveDataset,
    CameraMonitorSnapshot,
    CameraMonitorViewSpec,
)
from zlc_neutral_atom.runtime.dataset import (
    MonitorCoverage,
    MonitorDataset,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.pipeline import CapturePreviewSpec
from zlc_storage import canonical_digest, canonical_text
from zlc_neutral_atom.runtime.streams import event_ref_to_tree

from .workspace import BoardController, BoardPublishPort, PanelSourceBinding


if TYPE_CHECKING:
    from zlc_frontend.matplotlib_render import SingleCurveAggRenderer


_ROI_JOIN_KEY_SCHEMA_FINGERPRINT = canonical_digest(
    {
        "type": "camera-roi-source-event-control",
        "fields": ("source_event_ref", "control_revision", "control_fingerprint"),
    }
)


class LiveDatasetSlot:
    """One capacity-one materializer handle plus coalesced revision notices."""

    def __init__(
        self,
        spec: CapturePreviewSpec | CameraMonitorViewSpec,
        *,
        dataset_id: DatasetId,
        scalar_dataset_id: DatasetId | None = None,
        evaluation_policy: FigureEvaluationPolicy,
        retain_on_terminal: bool = True,
    ) -> None:
        if not isinstance(spec, (CapturePreviewSpec, CameraMonitorViewSpec)):
            raise TypeError("spec must be a supported live dataset spec")
        if not isinstance(dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if scalar_dataset_id is not None and not isinstance(
            scalar_dataset_id,
            DatasetId,
        ):
            raise TypeError("scalar_dataset_id must be DatasetId or None")
        expects_scalar = (
            isinstance(spec, CameraMonitorViewSpec)
            and spec.scalar_dataset_edge is not None
        )
        if expects_scalar != (scalar_dataset_id is not None):
            raise ValueError("scalar DatasetId must match the admitted camera view spec")
        if not isinstance(evaluation_policy, FigureEvaluationPolicy):
            raise TypeError("evaluation_policy must be FigureEvaluationPolicy")
        if not isinstance(retain_on_terminal, bool):
            raise TypeError("retain_on_terminal must be bool")
        self.spec = spec
        self.dataset_id = dataset_id
        self.scalar_dataset_id = scalar_dataset_id
        self.evaluation_policy = evaluation_policy
        self._retain_on_terminal = retain_on_terminal
        self._lock = threading.Lock()
        self._dataset: MonitorDataset | CameraMonitorLiveDataset | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._listener: Callable[[], None] | None = None
        self._listener_claimed = False
        self._pending_change = False
        self._failure: str | None = None
        self._terminal = False
        self._withdrawn = False
        self._closed = False

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    @property
    def terminal(self) -> bool:
        with self._lock:
            return self._terminal

    @property
    def withdrawn(self) -> bool:
        """Whether the source revoked its last displayable snapshot."""

        with self._lock:
            return self._withdrawn

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._lock:
            if self._listener_claimed:
                raise RuntimeError("live slot already has a change listener")
            if self._closed:
                raise RuntimeError("live slot is closed")
            self._listener_claimed = True
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            listener()

    def bind(
        self,
        dataset: MonitorDataset | CameraMonitorLiveDataset,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None:
        expected = (
            CameraMonitorLiveDataset
            if isinstance(self.spec, CameraMonitorViewSpec)
            else MonitorDataset
        )
        if not isinstance(dataset, expected):
            raise TypeError(f"dataset must be {expected.__name__}")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("live slot is closed")
            if self._terminal:
                raise RuntimeError("live slot is terminal")
            if self._dataset is not None:
                raise RuntimeError("live slot already owns a materializer")
            self._dataset = dataset
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id

    def updated(self) -> None:
        with self._lock:
            if self._closed or self._dataset is None:
                raise RuntimeError("live slot has no active materializer")
            listener = self._listener
            if listener is None:
                self._pending_change = True
                return
        if listener is not None:
            listener()

    def freeze_current(
        self,
    ) -> tuple[str, str, MonitorDatasetSnapshot]:
        """Atomically materialize current state; never resolve a prior notice ref."""

        with self._lock:
            if self._closed or self._dataset is None:
                raise RuntimeError("live slot has no active materializer")
            dataset = self._dataset
            run_id = self._run_id
            causation = self._causation_domain_id
        snapshot = (
            dataset.materialize().raw
            if isinstance(dataset, CameraMonitorLiveDataset)
            else dataset.materialize(None)
        )
        with self._lock:
            if self._closed or self._dataset is not dataset:
                raise RuntimeError("live slot lifetime ended while freezing a snapshot")
        assert run_id is not None and causation is not None
        return run_id, causation, snapshot

    def freeze_camera_current(
        self,
    ) -> tuple[str, str, CameraMonitorSnapshot]:
        """Freeze one application-owned raw/scalar transaction atomically."""

        with self._lock:
            if self._closed or not isinstance(
                self._dataset,
                CameraMonitorLiveDataset,
            ):
                raise RuntimeError("live slot has no active camera monitor dataset")
            dataset = self._dataset
            run_id = self._run_id
            causation = self._causation_domain_id
        snapshot = dataset.materialize()
        with self._lock:
            if self._closed or self._dataset is not dataset:
                raise RuntimeError("live slot lifetime ended while freezing camera data")
        assert run_id is not None and causation is not None
        return run_id, causation, snapshot

    def fail(self, message: str) -> None:
        message = canonical_text(message, "preview failure")
        dataset, listener = self._detach(message)
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if listener is not None:
                try:
                    listener()
                except BaseException:
                    pass

    def source_terminal(self) -> None:
        if self._retain_on_terminal:
            with self._lock:
                self._terminal = True
            return
        dataset, listener = self._detach(None, withdrawn=True)
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if listener is not None:
                try:
                    listener()
                except BaseException:
                    pass

    def close(self) -> None:
        dataset, _listener = self._detach(None, closed=True)
        if dataset is not None:
            dataset.close()

    def _detach(
        self,
        failure: str | None,
        *,
        closed: bool = False,
        withdrawn: bool = False,
    ) -> tuple[
        MonitorDataset | CameraMonitorLiveDataset | None,
        Callable[[], None] | None,
    ]:
        with self._lock:
            if failure is not None and self._failure is not None:
                return None, None
            dataset, self._dataset = self._dataset, None
            if failure is not None and self._failure is None:
                self._failure = failure
            self._terminal = True
            self._withdrawn = self._withdrawn or withdrawn
            self._closed = self._closed or closed
            listener, self._listener = self._listener, None
            if (
                (failure is not None or withdrawn)
                and listener is None
                and not self._listener_claimed
            ):
                self._pending_change = True
            return dataset, listener


class LiveImageBoardController:
    """One-snapshot coalescer for an image and optional coherent ROI curve."""

    def __init__(
        self,
        slot: LiveDatasetSlot,
        document: FigureDocument,
        board: BoardController,
        *,
        submit_worker: Callable[[Callable[[], None]], object],
        request_owner_wake: Callable[[], None],
        companion_curve_document: FigureDocument | None = None,
        companion_curve_size: tuple[int, int] = (800, 520),
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
        if companion_curve_document is not None and not isinstance(
            companion_curve_document, FigureDocument
        ):
            raise TypeError("companion_curve_document must be FigureDocument or None")
        if not isinstance(worker_thread_affine, bool):
            raise TypeError("worker_thread_affine must be bool")
        if companion_curve_document is not None and not worker_thread_affine:
            raise ValueError("live Agg curve requires a thread-affine worker lane")
        if (
            not isinstance(companion_curve_size, tuple)
            or len(companion_curve_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in companion_curve_size
            )
        ):
            raise ValueError("companion_curve_size must contain two positive integers")
        _validate_live_documents(slot, document, companion_curve_document)
        model = board.model
        expected_panels = 1 if companion_curve_document is None else 2
        if len(model.panels) != expected_panels:
            raise ValueError("live board panel count does not match its frozen documents")
        coherence_groups = {panel.coherence_group for panel in model.panels}
        if len(coherence_groups) != 1:
            raise ValueError("one live snapshot board requires one coherence group")
        self._slot = slot
        self._document = document
        self._curve_document = companion_curve_document
        self._board = board
        self._panels = model.panels
        self._coherence_group = next(iter(coherence_groups))
        documents = (
            (document,)
            if companion_curve_document is None
            else (document, companion_curve_document)
        )
        self._presentations = tuple(
            PanelPresentationIdentity(
                panel.panel_id,
                panel_document.document_id,
                panel_document.revision,
                0,
                0,
            )
            for panel, panel_document in zip(model.panels, documents, strict=True)
        )
        self._evaluator = FigureEvaluator(slot.evaluation_policy)
        self._curve_size = companion_curve_size
        self._curve_renderer: SingleCurveAggRenderer | None = None
        self._submit_worker = submit_worker
        self._request_owner_wake = request_owner_wake
        self._lock = threading.Lock()
        self._worker_gate = threading.Lock()
        self._dirty = False
        self._active = False
        self._candidate: tuple[
            str,
            str,
            MonitorDatasetSnapshot | CameraMonitorSnapshot,
        ] | None = None
        self._port: BoardPublishPort | None = None
        self._sources: tuple[SourceIdentity, ...] | None = None
        self._sequence = 0
        self._fault: BaseException | None = None
        self._coverage: MonitorCoverage | None = None
        self._scalar_coverage: MonitorCoverage | None = None
        self._front_invalidated = False
        self._closed = False
        self._close_complete = False
        slot.set_change_listener(self._source_changed)

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    @property
    def coverage(self) -> MonitorCoverage | None:
        """Coverage paired with the latest accepted immutable board front."""

        with self._lock:
            return self._coverage

    @property
    def scalar_coverage(self) -> MonitorCoverage | None:
        """Scalar history coverage paired with the accepted board front."""

        with self._lock:
            return self._scalar_coverage

    def _source_changed(self) -> None:
        if self._slot.failure is not None or self._slot.withdrawn:
            self._request_owner_wake()
        else:
            self._request_snapshot()

    def _request_snapshot(self) -> None:
        with self._lock:
            if self._closed or self._fault is not None:
                return
            self._dirty = True
            if self._active:
                return
            self._active = True
            self._dirty = False
        self._submit(self._freeze_latest)

    def _freeze_latest(self) -> None:
        try:
            candidate = (
                self._slot.freeze_camera_current()
                if isinstance(self._slot.spec, CameraMonitorViewSpec)
                else self._slot.freeze_current()
            )
            with self._lock:
                if self._closed:
                    self._active = False
                    return
                self._candidate = candidate
            self._request_owner_wake()
        except BaseException as error:
            # A clean source withdrawal can race a worker that was just about
            # to freeze the former live dataset.  It is a stale render request,
            # not a source or renderer failure.
            if self._slot.withdrawn and self._slot.failure is None:
                with self._lock:
                    self._candidate = None
                    self._active = False
                self._request_owner_wake()
                return
            self._set_fault(error)

    def admit_pending(self) -> bool:
        failure = self._slot.failure
        withdrawn = self._slot.withdrawn
        if failure is not None or withdrawn:
            with self._lock:
                if self._closed:
                    return False
                invalidate = not self._front_invalidated
                self._front_invalidated = True
                if failure is not None and self._fault is None:
                    self._fault = RuntimeError(failure)
                self._candidate = None
                self._active = False
                self._port = None
                self._sources = None
                self._coverage = None
                self._scalar_coverage = None
            if invalidate:
                try:
                    self._board.invalidate()
                except BaseException:
                    with self._lock:
                        self._front_invalidated = False
                    raise
            return False
        with self._lock:
            if self._closed or self._fault is not None:
                return False
            candidate, self._candidate = self._candidate, None
        if candidate is None:
            return False
        try:
            run_id, causation, frozen = candidate
            if isinstance(frozen, CameraMonitorSnapshot):
                raw_snapshot = frozen.raw
                scalar_snapshot = frozen.scalar
                scalar_metadata = frozen.scalar_metadata
            else:
                raw_snapshot = frozen
                scalar_snapshot = None
                scalar_metadata = None
            ref = raw_snapshot.snapshot.ref
            head = raw_snapshot.head
            if head is None:
                raise RuntimeError("live snapshot has no head event")
            raw_source = SourceIdentity(
                self._slot.dataset_id,
                ref.block_id,
                ref.stream_generation,
                ref.schema_fingerprint,
            )
            if scalar_snapshot is None:
                inputs = (EvaluatedInput(self._slot.dataset_id, ref),)
                panel_sources = tuple(raw_source for _panel in self._panels)
                join_key_type = "single-source-event-payload"
                join_schema = ref.schema_fingerprint
                join_digest = head.payload_digest
            else:
                scalar_dataset_id = self._slot.scalar_dataset_id
                if scalar_dataset_id is None or scalar_metadata is None:
                    raise RuntimeError("ROI scalar snapshot lacks its admitted identity")
                scalar_ref = scalar_snapshot.snapshot.ref
                scalar_source = SourceIdentity(
                    scalar_dataset_id,
                    scalar_ref.block_id,
                    scalar_ref.stream_generation,
                    scalar_ref.schema_fingerprint,
                )
                inputs = (
                    EvaluatedInput(self._slot.dataset_id, ref),
                    EvaluatedInput(scalar_dataset_id, scalar_ref),
                )
                panel_sources = (raw_source, scalar_source)
                join_key_type = "camera-roi-source-event-control"
                join_schema = _ROI_JOIN_KEY_SCHEMA_FINGERPRINT
                join_digest = canonical_digest(
                    {
                        "source_event_ref": event_ref_to_tree(
                            scalar_metadata.source_event_ref
                        ),
                        "control_revision": scalar_metadata.control_revision,
                        "control_fingerprint": scalar_metadata.control_fingerprint,
                    }
                )
            stamp = CoherenceStamp(
                run_id,
                causation,
                join_key_type,
                join_schema,
                join_digest,
                inputs,
                self._presentations,
            )
            if self._port is None or panel_sources != self._sources:
                self._port = self._board.open_publish_port(
                    tuple(
                        PanelSourceBinding(source, presentation)
                        for source, presentation in zip(
                            panel_sources,
                            self._presentations,
                            strict=True,
                        )
                    )
                )
                self._sources = panel_sources
            sequence = self._sequence
            self._sequence += 1
            token = self._port.admit(
                sequence,
                ((self._coherence_group, stamp),),
            )
            self._submit(
                lambda: self._render(
                    candidate,
                    panel_sources,
                    stamp,
                    sequence,
                    token,
                ),
                ends_cycle=True,
            )
            return True
        except BaseException as error:
            self._set_fault(error)
            return False

    def _render(
        self,
        candidate: tuple[
            str,
            str,
            MonitorDatasetSnapshot | CameraMonitorSnapshot,
        ],
        sources: tuple[SourceIdentity, ...],
        stamp: CoherenceStamp,
        sequence: int,
        token: object,
    ) -> None:
        try:
            frozen = candidate[2]
            if isinstance(frozen, CameraMonitorSnapshot):
                raw_snapshot = frozen.raw
                scalar_snapshot = frozen.scalar
            else:
                raw_snapshot = frozen
                scalar_snapshot = None
            snapshot = raw_snapshot.snapshot
            raw_input = (EvaluatedInput(self._slot.dataset_id, snapshot.ref),)
            image_evaluated = self._evaluator.evaluate(
                self._document,
                ResolvedDatasetMap(
                    (ResolvedDataset(self._slot.dataset_id, snapshot),)
                ),
                cancel_requested=lambda: self._closed,
            )
            if image_evaluated.inputs != raw_input:
                raise RuntimeError("figure evaluation changed the admitted input revision")
            if (
                len(image_evaluated.layers) != 1
                or len(image_evaluated.layers[0].cells) != 1
                or len(image_evaluated.layers[0].cells[0].series) != 1
            ):
                raise RuntimeError("live IMAGE document must evaluate to one image")
            layer = image_evaluated.layers[0]
            data = layer.cells[0].series[0].data
            if not isinstance(data, EvaluatedImage):
                raise RuntimeError("live IMAGE document did not evaluate to an image")
            rasters = [rasterize_image_gray8(data)]
            curve_document = self._curve_document
            if curve_document is not None:
                curve_dataset_id = (
                    self._slot.dataset_id
                    if scalar_snapshot is None
                    else self._slot.scalar_dataset_id
                )
                if curve_dataset_id is None:
                    raise RuntimeError("ROI curve has no scalar DatasetId")
                curve_snapshot = (
                    snapshot
                    if scalar_snapshot is None
                    else scalar_snapshot.snapshot
                )
                curve_input = (EvaluatedInput(curve_dataset_id, curve_snapshot.ref),)
                curve_evaluated = self._evaluator.evaluate(
                    curve_document,
                    ResolvedDatasetMap(
                        (ResolvedDataset(curve_dataset_id, curve_snapshot),)
                    ),
                    cancel_requested=lambda: self._closed,
                )
                if curve_evaluated.inputs != curve_input:
                    raise RuntimeError("ROI curve evaluation changed the admitted input revision")
                curve_layer = curve_evaluated.layers[0]
                curve_data = curve_layer.cells[0].series[0].data
                if not isinstance(curve_data, EvaluatedCurve):
                    raise RuntimeError("live companion document did not evaluate to a curve")
                renderer = self._curve_renderer
                if renderer is None:
                    from zlc_frontend.matplotlib_render import SingleCurveAggRenderer

                    renderer = SingleCurveAggRenderer(
                        curve_document,
                        width=self._curve_size[0],
                        height=self._curve_size[1],
                    )
                    self._curve_renderer = renderer
                rasters.append(renderer.render(curve_evaluated))
            frame = BoardFrame(
                self._board.model.board_id,
                self._board.model.layout_generation,
                sequence,
                tuple(
                    PanelFrame(
                        panel.panel_id,
                        panel.coherence_group,
                        source,
                        stamp,
                        raster,
                    )
                    for panel, source, raster in zip(
                        self._panels,
                        sources,
                        rasters,
                        strict=True,
                    )
                ),
            )
            port = self._port
            if port is not None:
                published = port.publish(token, frame)
                if published:
                    with self._lock:
                        if not self._closed and self._port is port:
                            self._coverage = raw_snapshot.coverage
                            self._scalar_coverage = (
                                None
                                if scalar_snapshot is None
                                else scalar_snapshot.coverage
                            )
        except BaseException as error:
            self._set_fault(error)
            return

    def _submit(
        self,
        work: Callable[[], None],
        *,
        ends_cycle: bool = False,
    ) -> None:
        pending_work: Callable[[], None] | None = work

        def serialized() -> None:
            nonlocal pending_work
            restart = False
            with self._worker_gate:
                callback = pending_work
                if callback is None:
                    return
                try:
                    callback()
                except BaseException as error:
                    self._set_fault(error)
                finally:
                    # A render callback may own the only full snapshot.  Drop
                    # it before this gate admits another worker callback.
                    callback = None
                    pending_work = None
                if ends_cycle:
                    with self._lock:
                        restart = (
                            self._dirty
                            and not self._closed
                            and self._fault is None
                        )
                        if restart:
                            self._dirty = False
                        else:
                            self._active = False
            if restart:
                self._submit(self._freeze_latest)

        try:
            self._submit_worker(serialized)
        except BaseException as error:
            self._set_fault(error)

    def _set_fault(self, error: BaseException) -> None:
        with self._lock:
            if self._closed:
                self._candidate = None
                self._active = False
                return
            if self._fault is None:
                self._fault = detached_render_fault(error)
            self._candidate = None
            self._active = False
            self._coverage = None
            self._scalar_coverage = None
        try:
            self._slot.fail(f"{type(error).__name__}: {error}")
        except BaseException:
            pass

    def close(self) -> None:
        schedule_renderer_close = False
        with self._lock:
            if self._close_complete:
                return
            self._closed = True
            self._active = False
            self._dirty = False
            self._candidate = None
            self._coverage = None
            self._scalar_coverage = None
            self._front_invalidated = True
            schedule_renderer_close = self._curve_document is not None
        try:
            self._slot.close()
        finally:
            self._board.close()
        if schedule_renderer_close:
            # Use the same serialization gate and the construction-time proven
            # affine lane; close can neither race nor change OS thread.
            def close_renderer() -> None:
                with self._worker_gate:
                    self._close_curve_renderer()

            self._submit_worker(close_renderer)
        with self._lock:
            self._fault = None
            self._close_complete = True

    def _close_curve_renderer(self) -> None:
        renderer = self._curve_renderer
        self._curve_renderer = None
        if renderer is not None:
            renderer.close()


def _validate_live_documents(
    slot: LiveDatasetSlot,
    document: FigureDocument,
    curve_document: FigureDocument | None,
) -> None:
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
        raise ValueError("live image controller requires IMAGE intent")
    if curve_document is None:
        if isinstance(slot.spec, CameraMonitorViewSpec) and (
            slot.spec.scalar_dataset_edge is not None
        ):
            raise ValueError("admitted ROI scalar source requires a curve document")
        return
    scalar_edge = (
        slot.spec.scalar_dataset_edge
        if isinstance(slot.spec, CameraMonitorViewSpec)
        else None
    )
    curve_schema = schema if scalar_edge is None else scalar_edge.schema
    curve_dataset_id = (
        slot.dataset_id if scalar_edge is None else slot.scalar_dataset_id
    )
    if curve_dataset_id is None:
        raise ValueError("live curve has no admitted DatasetId")
    if (
        len(curve_document.datasets) != 1
        or curve_document.datasets[0].dataset_id != curve_dataset_id
        or curve_document.datasets[0].schema_fingerprint != curve_schema.fingerprint
        or len(curve_document.layers) != 1
        or curve_document.layers[0].dataset_id != curve_dataset_id
    ):
        raise ValueError("live curve document must contain its admitted dataset and layer")
    curve_view = curve_document.layers[0].view
    validate_view_spec(curve_schema, curve_view)
    if curve_view.intent is not ViewIntent.CURVE:
        raise ValueError("live companion document requires CURVE intent")


__all__ = ["LiveDatasetSlot", "LiveImageBoardController"]
