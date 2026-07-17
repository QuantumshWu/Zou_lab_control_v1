"""Bounded live-dataset and coherent-board composition for the Workbench."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Callable

from zlc_frontend.figure import (
    DatasetId,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedMeter,
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
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer


_ROI_JOIN_KEY_SCHEMA_FINGERPRINT = canonical_digest(
    {
        "type": "camera-roi-source-event-control",
        "fields": ("source_event_ref", "control_revision", "control_fingerprint"),
    }
)


@dataclass(frozen=True, slots=True)
class LiveFrontStatus:
    """One board sequence and the diagnostics derived from that exact front."""

    sequence: int
    raw_coverage: MonitorCoverage
    scalar_coverage: MonitorCoverage | None = None
    histogram_valid_samples: int | None = None
    histogram_dropped_samples: int | None = None
    latest_scalar_valid: bool | None = None
    scalar_source_missed: int | None = None


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
    """One-snapshot coalescer for IMAGE plus a closed scalar companion set."""

    def __init__(
        self,
        slot: LiveDatasetSlot,
        document: FigureDocument,
        board: BoardController,
        *,
        submit_worker: Callable[[Callable[[], None]], object],
        request_owner_wake: Callable[[], None],
        scalar_documents: tuple[FigureDocument, ...] = (),
        scalar_raster_size: tuple[int, int] = (800, 520),
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
        scalar_documents = tuple(scalar_documents)
        if any(not isinstance(item, FigureDocument) for item in scalar_documents):
            raise TypeError("scalar_documents must contain FigureDocument values")
        if not isinstance(worker_thread_affine, bool):
            raise TypeError("worker_thread_affine must be bool")
        if scalar_documents and not worker_thread_affine:
            raise ValueError("live scalar Agg panels require a thread-affine worker lane")
        if (
            not isinstance(scalar_raster_size, tuple)
            or len(scalar_raster_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in scalar_raster_size
            )
        ):
            raise ValueError("scalar_raster_size must contain two positive integers")
        _validate_live_documents(slot, document, scalar_documents)
        model = board.model
        expected_panels = 1 + len(scalar_documents)
        if len(model.panels) != expected_panels:
            raise ValueError("live board panel count does not match its frozen documents")
        coherence_groups = {panel.coherence_group for panel in model.panels}
        if len(coherence_groups) != 1:
            raise ValueError("one live snapshot board requires one coherence group")
        self._slot = slot
        self._document = document
        self._scalar_documents = scalar_documents
        self._board = board
        self._panels = model.panels
        self._coherence_group = next(iter(coherence_groups))
        documents = (document, *scalar_documents)
        if tuple(panel.panel_id for panel in model.panels) != tuple(
            panel_document.layers[0].layer_id for panel_document in documents
        ):
            raise ValueError(
                "live board panels must match frozen document layers in order"
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
        self._scalar_size = scalar_raster_size
        self._scalar_renderers: list[SinglePanelAggRenderer | None] = [
            None for _document in scalar_documents
        ]
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
        self._front_status: LiveFrontStatus | None = None
        self._front_invalidated = False
        self._closed = False
        self._close_complete = False
        slot.set_change_listener(self._source_changed)

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    @property
    def front_status(self) -> LiveFrontStatus | None:
        """Atomically return diagnostics for one successfully published sequence."""

        with self._lock:
            return self._front_status

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
                self._front_status = None
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
                panel_sources = (
                    raw_source,
                    *(scalar_source for _document in self._scalar_documents),
                )
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
                scalar_metadata = frozen.scalar_metadata
            else:
                raw_snapshot = frozen
                scalar_snapshot = None
                scalar_metadata = None
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
            histogram_valid_samples = None
            histogram_dropped_samples = None
            latest_scalar_valid = None
            if self._scalar_documents:
                if scalar_snapshot is None or self._slot.scalar_dataset_id is None:
                    raise RuntimeError("live scalar panels have no scalar snapshot identity")
                scalar_dataset_id = self._slot.scalar_dataset_id
                scalar_owned = scalar_snapshot.snapshot
                scalar_input = (EvaluatedInput(scalar_dataset_id, scalar_owned.ref),)
                resolved_scalar = ResolvedDatasetMap(
                    (ResolvedDataset(scalar_dataset_id, scalar_owned),)
                )
            for index, scalar_document in enumerate(self._scalar_documents):
                scalar_evaluated = self._evaluator.evaluate(
                    scalar_document,
                    resolved_scalar,
                    cancel_requested=lambda: self._closed,
                )
                if scalar_evaluated.inputs != scalar_input:
                    raise RuntimeError(
                        "scalar panel evaluation changed the admitted input revision"
                    )
                if (
                    len(scalar_evaluated.layers) != 1
                    or len(scalar_evaluated.layers[0].cells) != 1
                    or len(scalar_evaluated.layers[0].cells[0].series) != 1
                ):
                    raise RuntimeError("live scalar document must evaluate to one series")
                scalar_data = scalar_evaluated.layers[0].cells[0].series[0].data
                intent = scalar_document.layers[0].view.intent
                expected_type = {
                    ViewIntent.CURVE: EvaluatedCurve,
                    ViewIntent.HISTOGRAM: EvaluatedHistogram,
                    ViewIntent.METER: EvaluatedMeter,
                }[intent]
                if not isinstance(scalar_data, expected_type):
                    raise RuntimeError(
                        f"live {intent.value} document evaluated to another data kind"
                    )
                if isinstance(scalar_data, EvaluatedHistogram):
                    histogram_valid_samples = len(scalar_data.samples)
                    histogram_dropped_samples = scalar_data.dropped_count
                    scalar_coverage = scalar_snapshot.coverage
                    if (
                        histogram_valid_samples > scalar_coverage.written_cells
                        or histogram_valid_samples + histogram_dropped_samples
                        != scalar_coverage.total_cells
                    ):
                        raise RuntimeError(
                            "scalar histogram validity counts disagree with its snapshot"
                        )
                elif isinstance(scalar_data, EvaluatedMeter):
                    latest_scalar_valid = scalar_data.valid
                renderer = self._scalar_renderers[index]
                if renderer is None:
                    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

                    renderer = SinglePanelAggRenderer(
                        scalar_document,
                        width=self._scalar_size[0],
                        height=self._scalar_size[1],
                    )
                    self._scalar_renderers[index] = renderer
                rasters.append(renderer.render(scalar_evaluated))
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
                front_status = LiveFrontStatus(
                    sequence,
                    raw_snapshot.coverage,
                    None if scalar_snapshot is None else scalar_snapshot.coverage,
                    histogram_valid_samples,
                    histogram_dropped_samples,
                    latest_scalar_valid,
                    (
                        None
                        if scalar_metadata is None
                        else scalar_metadata.source_missed
                    ),
                )
                published = port.publish(token, frame)
                if published:
                    accepted_status = False
                    with self._lock:
                        if not self._closed and self._port is port:
                            self._front_status = front_status
                            accepted_status = True
                    if accepted_status:
                        # BoardController may have queued its wake before this
                        # status record was installed.  A second coalesced wake
                        # makes the matching sequence observable without ever
                        # attaching newer diagnostics to an older visible front.
                        self._request_owner_wake()
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
            self._front_status = None
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
            self._front_status = None
            self._front_invalidated = True
            schedule_renderer_close = bool(self._scalar_documents)
        try:
            self._slot.close()
        finally:
            self._board.close()
        if schedule_renderer_close:
            # Use the same serialization gate and the construction-time proven
            # affine lane; close can neither race nor change OS thread.
            def close_renderer() -> None:
                with self._worker_gate:
                    self._close_scalar_renderers()

            self._submit_worker(close_renderer)
        with self._lock:
            self._fault = None
            self._close_complete = True

    def _close_scalar_renderers(self) -> None:
        errors: list[BaseException] = []
        renderers = tuple(self._scalar_renderers)
        self._scalar_renderers = [None for _renderer in renderers]
        for renderer in renderers:
            if renderer is None:
                continue
            try:
                renderer.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]


def _validate_live_documents(
    slot: LiveDatasetSlot,
    document: FigureDocument,
    scalar_documents: tuple[FigureDocument, ...],
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
    scalar_edge = (
        slot.spec.scalar_dataset_edge
        if isinstance(slot.spec, CameraMonitorViewSpec)
        else None
    )
    if scalar_edge is None:
        if scalar_documents:
            raise ValueError("a source without ROI scalar data cannot have scalar panels")
        return
    if slot.scalar_dataset_id is None:
        raise ValueError("admitted ROI scalar source has no DatasetId")
    expected_intents = (
        ViewIntent.CURVE,
        ViewIntent.HISTOGRAM,
        ViewIntent.METER,
    )
    if len(scalar_documents) != len(expected_intents):
        raise ValueError(
            "admitted ROI scalar source requires CURVE, HISTOGRAM, and METER documents"
        )
    for scalar_document, expected_intent in zip(
        scalar_documents,
        expected_intents,
        strict=True,
    ):
        if (
            len(scalar_document.datasets) != 1
            or scalar_document.datasets[0].dataset_id != slot.scalar_dataset_id
            or scalar_document.datasets[0].schema_fingerprint
            != scalar_edge.schema.fingerprint
            or len(scalar_document.layers) != 1
            or scalar_document.layers[0].dataset_id != slot.scalar_dataset_id
        ):
            raise ValueError(
                "live scalar document must contain its admitted dataset and layer"
            )
        scalar_view = scalar_document.layers[0].view
        validate_view_spec(scalar_edge.schema, scalar_view)
        if scalar_view.intent is not expected_intent:
            raise ValueError(
                "live scalar documents must be ordered CURVE, HISTOGRAM, METER"
            )


__all__ = ["LiveDatasetSlot", "LiveFrontStatus", "LiveImageBoardController"]
