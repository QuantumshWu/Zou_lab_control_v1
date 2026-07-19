"""Bounded live-dataset and coherent-board composition for the Workbench."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Callable

from zlc_data import (
    BlockId,
    ReductionMethod,
    Selection,
    StreamGenerationId,
    ValidityPolicy,
)

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
from zlc_frontend.curve_display import CurveDisplayState
from zlc_frontend.display_range import DisplayRange, RelimMode
from zlc_frontend.histogram_display import (
    HistogramCountScale,
    HistogramDisplayState,
)
from zlc_frontend.image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
)
from zlc_frontend.image_raster import rasterize_image_indexed8
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.render import (
    BoardFrame,
    CoherenceStamp,
    CurvePanelPayload,
    DisplayPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
    detached_render_fault,
)
from zlc_neutral_atom.monitor_application import (
    CameraMonitorLiveDataset,
    CameraMonitorRoiState,
    CameraMonitorSnapshot,
    CameraMonitorViewSpec,
)
from zlc_neutral_atom.processing.roi_monitor import RoiScalarBinding, RoiScalarSample
from zlc_neutral_atom.runtime.control import ControlReceipt
from zlc_neutral_atom.runtime.dataset import (
    FrozenDatasetEdge,
    MonitorCoverage,
    MonitorDataset,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.pipeline import CapturePreviewSpec
from zlc_storage import canonical_digest, canonical_text
from zlc_neutral_atom.runtime.streams import event_ref_to_tree

from .workspace import (
    BoardController,
    BoardModel,
    BoardPublishPort,
    PanelSlot,
    PanelSourceBinding,
)


if TYPE_CHECKING:
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer


_ROI_JOIN_KEY_SCHEMA_FINGERPRINT = canonical_digest(
    {
        "type": "camera-roi-source-event-binding",
        "fields": (
            "source_event_ref",
            "binding_fingerprint",
            "control_revision",
        ),
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
    scalar_binding_fingerprint: str | None = None
    scalar_control_revision: int | None = None
    image_display_revision: int | None = None
    image_color_limits: tuple[float, float] | None = None
    curve_display_revision: int | None = None
    curve_y_limits: DisplayRange | None = None
    histogram_display_revision: int | None = None
    histogram_count_limits: DisplayRange | None = None


@dataclass(frozen=True, slots=True)
class _LiveRenderConfiguration:
    """One immutable identity/document set plus worker-lane-owned renderers."""

    epoch: int
    board_id: str
    layout_generation: int
    panels: tuple[PanelSlot, ...]
    coherence_group: str
    presentations: tuple[PanelPresentationIdentity, ...]
    image_document: FigureDocument
    image_display: ImageDisplayState | None
    image_viewport: ImageViewportTransform | None
    curve_display: CurveDisplayState | None
    histogram_display: HistogramDisplayState | None
    scalar_dataset_id: DatasetId | None
    scalar_documents: tuple[FigureDocument, ...]
    scalar_block_id: BlockId | None
    scalar_dataset_edge: FrozenDatasetEdge[RoiScalarSample] | None
    scalar_generation: StreamGenerationId | None
    scalar_binding_fingerprint: str | None
    scalar_control_revision: int | None
    strict_scalar_identity: bool
    scalar_renderers: list[SinglePanelAggRenderer | None]


@dataclass(frozen=True, slots=True)
class _LiveCandidate:
    run_id: str
    causation_domain_id: str
    frozen: MonitorDatasetSnapshot | CameraMonitorSnapshot
    configuration: _LiveRenderConfiguration


@dataclass(frozen=True, slots=True)
class _LiveRenderJob:
    candidate: _LiveCandidate
    sources: tuple[SourceIdentity, ...]
    stamp: CoherenceStamp
    sequence: int
    token: object
    port: BoardPublishPort


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
        self._camera_roi_state: CameraMonitorRoiState | None = None
        self._listener: Callable[[], None] | None = None
        self._listener_claimed = False
        self._pending_change = False
        self._failure: str | None = None
        self._notification_failure: str | None = None
        self._terminal = False
        self._withdrawn = False
        self._closed = False

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    @property
    def notification_failure(self) -> str | None:
        """Detached view-notification failure; the source dataset stays bound."""

        with self._lock:
            return getattr(self, "_notification_failure", None)

    @property
    def terminal(self) -> bool:
        with self._lock:
            return self._terminal

    @property
    def dataset_bound(self) -> bool:
        """Whether the source has installed the live materializer for this slot."""

        with self._lock:
            return not self._closed and self._dataset is not None

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
        camera_state = (
            dataset.current_roi_state()
            if isinstance(dataset, CameraMonitorLiveDataset)
            else None
        )
        if camera_state is not None and not isinstance(
            camera_state,
            CameraMonitorRoiState,
        ):
            raise TypeError("camera monitor returned an invalid ROI state")
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
            if camera_state is not None:
                self._cache_camera_roi_state_locked(camera_state)

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

    def notification_failed(self, message: str) -> None:
        """Expose a broken view notification without detaching the source data."""

        message = canonical_text(message, "live notification failure")
        listener = None
        with self._lock:
            if self._closed or self._dataset is None:
                return
            if self._failure is not None or self._notification_failure is not None:
                return
            self._notification_failure = message
            listener = self._listener
            if listener is None:
                self._pending_change = True
        if listener is not None:
            try:
                listener()
            except BaseException:
                pass

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

    def prepare_camera_roi_control(
        self,
        selection: Selection | None,
        reduction: ReductionMethod = ReductionMethod.MEAN,
        validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_ALL,
    ) -> RoiScalarBinding | None:
        """Prepare one ROI command without exposing the live dataset owner."""

        dataset = self._camera_dataset_for_control()
        candidate = dataset.prepare_roi_control(
            selection,
            reduction,
            validity_policy,
        )
        self._ensure_camera_dataset_still_bound(dataset)
        if candidate is not None and not isinstance(candidate, RoiScalarBinding):
            raise TypeError("camera ROI preparation returned an invalid candidate")
        return candidate

    def submit_camera_roi_control(
        self,
        candidate: RoiScalarBinding | None,
    ) -> ControlReceipt:
        """Submit a prepared command through the slot's narrow control seam."""

        if candidate is not None and not isinstance(candidate, RoiScalarBinding):
            raise TypeError("candidate must be RoiScalarBinding or None")
        dataset = self._camera_dataset_for_control()
        receipt = dataset.submit_roi_control(candidate)
        if not isinstance(receipt, ControlReceipt):
            raise TypeError("camera ROI submission returned an invalid receipt")
        return receipt

    def current_camera_roi_state(self) -> CameraMonitorRoiState:
        """Return the currently applied branch identity, never a pending command."""

        with self._lock:
            dataset = self._dataset
            cached = getattr(self, "_camera_roi_state", None)
        if not isinstance(dataset, CameraMonitorLiveDataset):
            if cached is None:
                raise RuntimeError("live slot has no camera monitor ROI state")
            return cached
        state = dataset.current_roi_state()
        if not isinstance(state, CameraMonitorRoiState):
            raise TypeError("camera monitor returned an invalid ROI state")
        with self._lock:
            self._cache_camera_roi_state_locked(state)
            cached = self._camera_roi_state
        assert cached is not None
        return cached

    @property
    def initial_camera_roi_receipt(self) -> ControlReceipt | None:
        """Initial ROI activation receipt, if the admitted view requested one."""

        dataset = self._camera_dataset_for_control()
        receipt = dataset.initial_roi_receipt
        self._ensure_camera_dataset_still_bound(dataset)
        if receipt is not None and not isinstance(receipt, ControlReceipt):
            raise TypeError("camera monitor returned an invalid initial ROI receipt")
        return receipt

    def _camera_dataset_for_control(self) -> CameraMonitorLiveDataset:
        with self._lock:
            if self._closed or not isinstance(
                self._dataset,
                CameraMonitorLiveDataset,
            ):
                raise RuntimeError("live slot has no active camera monitor dataset")
            return self._dataset

    def _ensure_camera_dataset_still_bound(
        self,
        dataset: CameraMonitorLiveDataset,
    ) -> None:
        with self._lock:
            if self._closed or self._dataset is not dataset:
                raise RuntimeError("live slot lifetime ended during camera ROI control")

    def _cache_camera_roi_state_locked(self, state: CameraMonitorRoiState) -> None:
        cached = getattr(self, "_camera_roi_state", None)
        if cached is None or state.state_revision > cached.state_revision:
            self._camera_roi_state = state
            return
        if state.state_revision == cached.state_revision and state != cached:
            raise RuntimeError(
                "camera ROI state changed without advancing state_revision"
            )

    def _refresh_camera_roi_state_cache(self) -> None:
        with self._lock:
            dataset = self._dataset
        if not isinstance(dataset, CameraMonitorLiveDataset):
            return
        try:
            state = dataset.current_roi_state()
        except Exception:
            # Detachment/close must still complete if the source is already
            # failing.  The most recent successfully observed state remains.
            return
        if not isinstance(state, CameraMonitorRoiState):
            return
        with self._lock:
            self._cache_camera_roi_state_locked(state)

    def fail(self, message: str) -> None:
        message = canonical_text(message, "preview failure")
        self._refresh_camera_roi_state_cache()
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
        self._refresh_camera_roi_state_cache()
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
        self._refresh_camera_roi_state_cache()
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


class LiveBoardController:
    """Latest-only live board with IMAGE and typed scalar presentations."""

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
        curve_display: CurveDisplayState | None = None,
        histogram_display: HistogramDisplayState | None = None,
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
        if not isinstance(image_display, ImageDisplayState):
            raise TypeError("image_display must be ImageDisplayState")
        if not isinstance(image_viewport, ImageViewportTransform):
            raise TypeError("image_viewport must be ImageViewportTransform")
        if image_viewport.viewport_revision != image_display.revision:
            raise ValueError("image viewport revision must match image display revision")
        if curve_display is not None and not isinstance(
            curve_display,
            CurveDisplayState,
        ):
            raise TypeError("curve_display must be CurveDisplayState or None")
        if histogram_display is not None and not isinstance(
            histogram_display,
            HistogramDisplayState,
        ):
            raise TypeError(
                "histogram_display must be HistogramDisplayState or None"
            )
        if bool(scalar_documents) != (
            curve_display is not None and histogram_display is not None
        ):
            raise ValueError(
                "curve and histogram display states must match the scalar panel set"
            )
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
        scalar_block_id = None
        scalar_edge = None
        scalar_binding_fingerprint = None
        if scalar_documents:
            assert isinstance(slot.spec, CameraMonitorViewSpec)
            assert slot.spec.scalar_block_id is not None
            assert slot.spec.scalar_dataset_edge is not None
            assert slot.spec.roi_binding is not None
            scalar_block_id = slot.spec.scalar_block_id
            scalar_edge = slot.spec.scalar_dataset_edge
            scalar_binding_fingerprint = slot.spec.roi_binding.fingerprint
        configuration = _build_live_configuration(
            epoch=0,
            model=model,
            image_document=document,
            image_display=image_display,
            image_viewport=image_viewport,
            curve_display=curve_display,
            histogram_display=histogram_display,
            scalar_dataset_id=slot.scalar_dataset_id,
            scalar_documents=scalar_documents,
            scalar_block_id=scalar_block_id,
            scalar_dataset_edge=scalar_edge,
            scalar_generation=None,
            scalar_binding_fingerprint=scalar_binding_fingerprint,
            scalar_control_revision=None,
            strict_scalar_identity=False,
        )
        self._slot = slot
        self._board = board
        self._configuration = configuration
        self._configuration_epoch = 0
        self._evaluator = FigureEvaluator(slot.evaluation_policy)
        self._scalar_size = scalar_raster_size
        self._worker_thread_affine = worker_thread_affine
        self._submit_worker = submit_worker
        self._request_owner_wake = request_owner_wake
        self._owner_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._worker_gate = threading.Lock()
        self._dirty = False
        self._active = False
        self._candidate: _LiveCandidate | None = None
        self._port: BoardPublishPort | None = None
        self._sources: tuple[SourceIdentity, ...] | None = None
        self._raw_source: SourceIdentity | None = None
        self._scalar_dataset_generations: dict[DatasetId, StreamGenerationId] = {}
        self._scalar_generation_datasets: dict[StreamGenerationId, DatasetId] = {}
        self._sequence = 0
        self._fault: BaseException | None = None
        self._front_status: LiveFrontStatus | None = None
        self._image_color_limits: tuple[float, float] | None = None
        # This records the mode of the last successfully published *valid*
        # image, not merely the configured intent.  An all-invalid first front
        # must not consume the forced relim owed by the first valid frame.
        self._image_relim_mode: RelimMode | None = None
        self._curve_y_limits: DisplayRange | None = None
        self._curve_relim_mode: RelimMode | None = None
        self._histogram_count_limits: DisplayRange | None = None
        self._histogram_relim_mode: RelimMode | None = None
        self._histogram_count_scale: HistogramCountScale | None = None
        self._front_invalidated = False
        self._presentation_frozen = False
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

    def reconfigure_scalar(
        self,
        state: CameraMonitorRoiState,
        scalar_dataset_id: DatasetId | None,
        scalar_documents: tuple[FigureDocument, ...],
        curve_display: CurveDisplayState | None,
        histogram_display: HistogramDisplayState | None,
    ) -> None:
        """Owner-thread switch to one applied scalar branch without clearing front."""

        self._require_owner()
        if not isinstance(self._slot.spec, CameraMonitorViewSpec):
            raise RuntimeError("only a camera monitor controller has an ROI branch")
        if not isinstance(state, CameraMonitorRoiState):
            raise TypeError("state must be CameraMonitorRoiState")
        scalar_documents = tuple(scalar_documents)
        if any(not isinstance(item, FigureDocument) for item in scalar_documents):
            raise TypeError("scalar_documents must contain FigureDocument values")
        active = state.binding is not None
        if active:
            if not isinstance(curve_display, CurveDisplayState):
                raise TypeError("an applied ROI branch requires curve_display")
            if not isinstance(histogram_display, HistogramDisplayState):
                raise TypeError(
                    "an applied ROI branch requires histogram_display"
                )
            if not isinstance(scalar_dataset_id, DatasetId):
                raise TypeError("an applied ROI branch requires scalar_dataset_id")
            if not self._worker_thread_affine:
                raise ValueError(
                    "live scalar Agg panels require a thread-affine worker lane"
                )
            assert state.scalar_dataset_edge is not None
            assert state.scalar_block_id is not None
            assert state.scalar_generation is not None
            _validate_scalar_documents(
                state.scalar_dataset_edge,
                scalar_dataset_id,
                scalar_documents,
            )
            known_generation = self._scalar_dataset_generations.get(
                scalar_dataset_id
            )
            known_dataset = self._scalar_generation_datasets.get(
                state.scalar_generation
            )
            if (
                known_generation is not None
                and known_generation != state.scalar_generation
            ):
                raise ValueError(
                    "scalar DatasetId cannot be reused for another stream generation"
                )
            if known_dataset is not None and known_dataset != scalar_dataset_id:
                raise ValueError(
                    "one scalar stream generation must retain its frontend DatasetId"
                )
        else:
            if curve_display is not None:
                raise ValueError("raw-only ROI state cannot have curve_display")
            if histogram_display is not None:
                raise ValueError(
                    "raw-only ROI state cannot have histogram_display"
                )
            if scalar_dataset_id is not None:
                raise ValueError("raw-only ROI state cannot have scalar_dataset_id")
            if scalar_documents:
                raise ValueError("raw-only ROI state cannot have scalar documents")
        with self._lock:
            if self._closed:
                raise RuntimeError("live board controller is closed")
            if self._fault is not None:
                raise RuntimeError("live board controller is faulted")
            if self._slot.failure is not None or self._slot.withdrawn:
                raise RuntimeError("live source is no longer available")
            epoch = self._configuration_epoch + 1
            previous = self._configuration
            previous_port = self._port
            previous_sources = self._sources
        target_model = self._board.model
        previous_panel_ids = tuple(panel.panel_id for panel in previous.panels)
        target_panel_ids = tuple(panel.panel_id for panel in target_model.panels)
        topology_changed = target_panel_ids != previous_panel_ids
        if target_model.board_id != previous.board_id:
            raise ValueError("scalar reconfiguration cannot change board identity")
        if target_model.layout_generation < previous.layout_generation:
            raise ValueError("scalar reconfiguration cannot use an older board layout")
        if (
            topology_changed
            and target_model.layout_generation <= previous.layout_generation
        ):
            raise ValueError(
                "scalar topology change requires a newer staged board layout"
            )
        configuration = _build_live_configuration(
            epoch=epoch,
            model=target_model,
            image_document=previous.image_document,
            image_display=previous.image_display,
            image_viewport=previous.image_viewport,
            curve_display=curve_display,
            histogram_display=histogram_display,
            scalar_dataset_id=scalar_dataset_id,
            scalar_documents=scalar_documents,
            scalar_block_id=state.scalar_block_id,
            scalar_dataset_edge=state.scalar_dataset_edge,
            scalar_generation=state.scalar_generation,
            scalar_binding_fingerprint=(
                None if state.binding is None else state.binding.fingerprint
            ),
            scalar_control_revision=state.control_revision,
            strict_scalar_identity=True,
        )
        scalar_semantics_changed = _scalar_semantic_identity(
            previous
        ) != _scalar_semantic_identity(configuration)
        with self._lock:
            if self._closed:
                raise RuntimeError("live board controller is closed")
            if self._fault is not None:
                raise RuntimeError("live board controller is faulted")
            if self._slot.failure is not None or self._slot.withdrawn:
                raise RuntimeError("live source is no longer available")
            if epoch != self._configuration_epoch + 1:
                raise RuntimeError("live scalar reconfiguration lost owner serialization")
            if self._configuration is not previous:
                raise RuntimeError("live scalar configuration changed unexpectedly")
            replacement_port: BoardPublishPort | None = None
            if not topology_changed and previous_port is not None:
                if previous_sources is None or len(previous_sources) != len(
                    configuration.presentations
                ):
                    raise RuntimeError("active live publish identity is incomplete")
                # Port minting stays under the live lock.  A fault/source fact
                # is installed under that same lock before Board revocation,
                # so no replacement capability can escape the revoke window.
                replacement_port = self._board.open_publish_port(
                    tuple(
                        PanelSourceBinding(source, presentation)
                        for source, presentation in zip(
                            previous_sources,
                            configuration.presentations,
                            strict=True,
                        )
                    )
                )
            candidate_waiting = self._candidate is not None
            self._configuration = configuration
            self._configuration_epoch = epoch
            self._candidate = None
            self._port = replacement_port
            self._sources = (
                previous_sources if replacement_port is not None else None
            )
            self._dirty = True
            if scalar_semantics_changed:
                self._curve_y_limits = None
                self._curve_relim_mode = None
                self._histogram_count_limits = None
                self._histogram_relim_mode = None
                self._histogram_count_scale = None
            if candidate_waiting:
                self._active = False
            request_now = not self._presentation_frozen and not self._active
            if active:
                assert scalar_dataset_id is not None
                assert state.scalar_generation is not None
                self._scalar_dataset_generations[
                    scalar_dataset_id
                ] = state.scalar_generation
                self._scalar_generation_datasets[
                    state.scalar_generation
                ] = scalar_dataset_id
        if previous.scalar_documents:
            self._submit(lambda: self._close_scalar_renderers(previous))
        if request_now:
            self._request_snapshot()

    def reconfigure_image_display(
        self,
        state: ImageDisplayState,
        viewport: ImageViewportTransform,
    ) -> None:
        """Replace one IMAGE presentation revision without touching its source Run."""

        self._require_owner()
        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if viewport.viewport_revision != state.revision:
            raise ValueError("image display and viewport revisions differ")
        with self._lock:
            if self._closed:
                raise RuntimeError("live board controller is closed")
            if self._fault is not None:
                raise RuntimeError("live board controller is faulted")
            if self._slot.failure is not None or self._slot.withdrawn:
                raise RuntimeError("live source is no longer available")
            previous = self._configuration
            previous_state = previous.image_display
            previous_viewport = previous.image_viewport
            previous_port = self._port
            previous_sources = self._sources
            epoch = self._configuration_epoch + 1
        if previous_state is None or previous_viewport is None:
            raise RuntimeError("live controller has no interactive IMAGE presentation")
        if state == previous_state and viewport == previous_viewport:
            return
        if state.revision <= previous_state.revision:
            raise ValueError("image display revision must increase")
        if viewport.axes != previous_viewport.axes:
            raise ValueError("image display reconfiguration cannot change source axes")
        target_model = self._board.model
        if (
            target_model.board_id != previous.board_id
            or target_model.layout_generation != previous.layout_generation
            or target_model.panels != previous.panels
        ):
            raise RuntimeError(
                "image display reconfiguration cannot cross a board topology change"
            )
        configuration = _build_live_configuration(
            epoch=epoch,
            model=target_model,
            image_document=previous.image_document,
            image_display=state,
            image_viewport=viewport,
            curve_display=previous.curve_display,
            histogram_display=previous.histogram_display,
            scalar_dataset_id=previous.scalar_dataset_id,
            scalar_documents=previous.scalar_documents,
            scalar_block_id=previous.scalar_block_id,
            scalar_dataset_edge=previous.scalar_dataset_edge,
            scalar_generation=previous.scalar_generation,
            scalar_binding_fingerprint=previous.scalar_binding_fingerprint,
            scalar_control_revision=previous.scalar_control_revision,
            strict_scalar_identity=previous.strict_scalar_identity,
            scalar_renderers=previous.scalar_renderers,
        )
        # Display-only replacement shares the worker-lane-owned renderer
        # holder itself, not a racy snapshot of its current entries.
        request_now = self._install_display_configuration(
            previous=previous,
            configuration=configuration,
            epoch=epoch,
            previous_port=previous_port,
            previous_sources=previous_sources,
            operation="image display",
        )
        if request_now:
            self._request_snapshot()

    def reconfigure_curve_display(self, state: CurveDisplayState) -> None:
        """Replace one CURVE presentation revision without touching its source."""

        self._require_owner()
        if not isinstance(state, CurveDisplayState):
            raise TypeError("state must be CurveDisplayState")
        with self._lock:
            if self._closed:
                raise RuntimeError("live board controller is closed")
            if self._fault is not None:
                raise RuntimeError("live board controller is faulted")
            if self._slot.failure is not None or self._slot.withdrawn:
                raise RuntimeError("live source is no longer available")
            previous = self._configuration
            previous_state = previous.curve_display
            previous_port = self._port
            previous_sources = self._sources
            epoch = self._configuration_epoch + 1
        if previous_state is None:
            raise RuntimeError("live board has no interactive CURVE presentation")
        if state == previous_state:
            return
        if state.revision != previous_state.revision + 1:
            raise ValueError("curve display revision must advance exactly once")
        target_model = self._board.model
        if (
            target_model.board_id != previous.board_id
            or target_model.layout_generation != previous.layout_generation
            or target_model.panels != previous.panels
        ):
            raise RuntimeError(
                "curve display reconfiguration cannot cross a board topology change"
            )
        configuration = _build_live_configuration(
            epoch=epoch,
            model=target_model,
            image_document=previous.image_document,
            image_display=previous.image_display,
            image_viewport=previous.image_viewport,
            curve_display=state,
            histogram_display=previous.histogram_display,
            scalar_dataset_id=previous.scalar_dataset_id,
            scalar_documents=previous.scalar_documents,
            scalar_block_id=previous.scalar_block_id,
            scalar_dataset_edge=previous.scalar_dataset_edge,
            scalar_generation=previous.scalar_generation,
            scalar_binding_fingerprint=previous.scalar_binding_fingerprint,
            scalar_control_revision=previous.scalar_control_revision,
            strict_scalar_identity=previous.strict_scalar_identity,
            scalar_renderers=previous.scalar_renderers,
        )
        request_now = self._install_display_configuration(
            previous=previous,
            configuration=configuration,
            epoch=epoch,
            previous_port=previous_port,
            previous_sources=previous_sources,
            operation="curve display",
        )
        if request_now:
            self._request_snapshot()

    def reconfigure_histogram_display(
        self,
        state: HistogramDisplayState,
    ) -> None:
        """Replace one HISTOGRAM presentation without touching its source."""

        self._require_owner()
        if not isinstance(state, HistogramDisplayState):
            raise TypeError("state must be HistogramDisplayState")
        with self._lock:
            if self._closed:
                raise RuntimeError("live board controller is closed")
            if self._fault is not None:
                raise RuntimeError("live board controller is faulted")
            if self._slot.failure is not None or self._slot.withdrawn:
                raise RuntimeError("live source is no longer available")
            previous = self._configuration
            previous_state = previous.histogram_display
            previous_port = self._port
            previous_sources = self._sources
            epoch = self._configuration_epoch + 1
        if previous_state is None:
            raise RuntimeError(
                "live board has no interactive HISTOGRAM presentation"
            )
        if state == previous_state:
            return
        if state.revision != previous_state.revision + 1:
            raise ValueError(
                "histogram display revision must advance exactly once"
            )
        target_model = self._board.model
        if (
            target_model.board_id != previous.board_id
            or target_model.layout_generation != previous.layout_generation
            or target_model.panels != previous.panels
        ):
            raise RuntimeError(
                "histogram display reconfiguration cannot cross a board topology change"
            )
        configuration = _build_live_configuration(
            epoch=epoch,
            model=target_model,
            image_document=previous.image_document,
            image_display=previous.image_display,
            image_viewport=previous.image_viewport,
            curve_display=previous.curve_display,
            histogram_display=state,
            scalar_dataset_id=previous.scalar_dataset_id,
            scalar_documents=previous.scalar_documents,
            scalar_block_id=previous.scalar_block_id,
            scalar_dataset_edge=previous.scalar_dataset_edge,
            scalar_generation=previous.scalar_generation,
            scalar_binding_fingerprint=previous.scalar_binding_fingerprint,
            scalar_control_revision=previous.scalar_control_revision,
            strict_scalar_identity=previous.strict_scalar_identity,
            scalar_renderers=previous.scalar_renderers,
        )
        request_now = self._install_display_configuration(
            previous=previous,
            configuration=configuration,
            epoch=epoch,
            previous_port=previous_port,
            previous_sources=previous_sources,
            operation="histogram display",
        )
        if request_now:
            self._request_snapshot()

    def _install_display_configuration(
        self,
        *,
        previous: _LiveRenderConfiguration,
        configuration: _LiveRenderConfiguration,
        epoch: int,
        previous_port: BoardPublishPort | None,
        previous_sources: tuple[SourceIdentity, ...] | None,
        operation: str,
    ) -> bool:
        """Atomically replace one same-topology display-only configuration."""

        with self._lock:
            if self._closed:
                raise RuntimeError("live board controller is closed")
            if self._fault is not None:
                raise RuntimeError("live board controller is faulted")
            if self._slot.failure is not None or self._slot.withdrawn:
                raise RuntimeError("live source is no longer available")
            if epoch != self._configuration_epoch + 1:
                raise RuntimeError(f"{operation} reconfiguration lost owner serialization")
            if self._configuration is not previous:
                raise RuntimeError(f"live {operation} configuration changed unexpectedly")
            replacement_port: BoardPublishPort | None = None
            if previous_port is not None:
                if previous_sources is None or len(previous_sources) != len(
                    configuration.presentations
                ):
                    raise RuntimeError("active live publish identity is incomplete")
                replacement_port = self._board.open_publish_port(
                    tuple(
                        PanelSourceBinding(source, presentation)
                        for source, presentation in zip(
                            previous_sources,
                            configuration.presentations,
                            strict=True,
                        )
                    )
                )
            candidate_waiting = self._candidate is not None
            self._configuration = configuration
            self._configuration_epoch = epoch
            self._candidate = None
            self._port = replacement_port
            self._sources = previous_sources if replacement_port is not None else None
            self._dirty = True
            if candidate_waiting:
                self._active = False
            return not self._presentation_frozen and not self._active

    def freeze_presentation(self) -> None:
        """Revoke all render work while preserving the last coherent front."""

        self._require_owner()
        with self._lock:
            if self._closed:
                return
            self._presentation_frozen = True
            self._dirty = False
            self._active = False
            self._candidate = None
            self._port = None
            self._sources = None
        # Clearing the live-side port makes all future jobs stale.  Revoking the
        # board capability afterwards closes the check-then-publish race for a
        # worker that already passed _job_is_current(), and drops any frame that
        # was pending but had not yet become the coherent front.
        self._board.freeze_front()

    def _source_changed(self) -> None:
        notification_failure = self._slot.notification_failure
        if notification_failure is not None:
            if self.fault is None:
                # notification_failed() invokes this listener synchronously.
                # Board's presentation gate installs the Live fault and revokes
                # the pending capability at one boundary before this synchronous
                # terminal notice returns.
                self._set_view_fault(RuntimeError(notification_failure))
            else:
                self._request_owner_wake()
            return
        failure = self._slot.failure
        if failure is not None or self._slot.withdrawn:
            self._revoke_source_capability(failure)
            self._request_owner_wake()
        else:
            self._request_snapshot()

    def _revoke_source_capability(self, failure: str | None) -> None:
        """Synchronously fence queued source work before owner-thread clearing."""

        def install_source_fact() -> bool:
            with self._lock:
                if self._closed:
                    return False
                if failure is not None and self._fault is None:
                    self._fault = RuntimeError(failure)
                self._candidate = None
                self._active = False
                self._dirty = False
                self._port = None
                self._sources = None
                return True

        self._board.revoke_pending_publication(install_source_fact)

    def _request_snapshot(self) -> None:
        with self._lock:
            if self._closed or self._presentation_frozen or self._fault is not None:
                return
            self._dirty = True
            if self._active:
                return
            self._active = True
            self._dirty = False
        self._submit(self._freeze_latest)

    def _freeze_latest(self) -> None:
        try:
            with self._lock:
                configuration = self._configuration
            frozen = (
                self._slot.freeze_camera_current()
                if isinstance(self._slot.spec, CameraMonitorViewSpec)
                else self._slot.freeze_current()
            )
            candidate = _LiveCandidate(*frozen, configuration)
            with self._lock:
                if (
                    self._closed
                    or self._presentation_frozen
                    or self._fault is not None
                    or self._slot.withdrawn
                ):
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
            if self._slot.failure is not None:
                with self._lock:
                    self._candidate = None
                    self._active = False
                self._request_owner_wake()
                return
            self._set_view_fault(error)

    def reconcile_faults(self) -> bool:
        """Revoke presentation before the owner may present a queued frame."""

        notification_failure = self._slot.notification_failure
        if notification_failure is not None:
            if self.fault is None:
                self._set_view_fault(RuntimeError(notification_failure))
            return True
        failure = self._slot.failure
        withdrawn = self._slot.withdrawn
        if failure is not None or withdrawn:
            self._revoke_source_capability(failure)
            with self._lock:
                if self._closed:
                    return False
                invalidate = not self._front_invalidated
                self._front_invalidated = True
                self._front_status = None
            if invalidate:
                try:
                    self._board.invalidate()
                except BaseException:
                    with self._lock:
                        self._front_invalidated = False
                    raise
            return True
        return False

    def admit_pending(self) -> bool:
        if self.reconcile_faults():
            return False
        with self._lock:
            if self._closed or self._presentation_frozen or self._fault is not None:
                return False
            candidate, self._candidate = self._candidate, None
            configuration = self._configuration
        if candidate is None:
            return False
        if (
            candidate.configuration is not configuration
            or not self._candidate_matches_configuration(candidate)
        ):
            self._finish_stale_cycle()
            return False
        try:
            frozen = candidate.frozen
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
            if self._raw_source is None:
                self._raw_source = raw_source
            elif raw_source != self._raw_source:
                raise RuntimeError(
                    "one live controller cannot change its raw source identity"
                )
            if scalar_snapshot is None:
                inputs = (EvaluatedInput(self._slot.dataset_id, ref),)
                panel_sources = tuple(raw_source for _panel in configuration.panels)
                join_key_type = "single-source-event-payload"
                join_schema = ref.schema_fingerprint
                join_digest = head.payload_digest
            else:
                scalar_dataset_id = configuration.scalar_dataset_id
                if scalar_dataset_id is None or scalar_metadata is None:
                    raise RuntimeError("ROI scalar snapshot lacks its admitted identity")
                scalar_ref = scalar_snapshot.snapshot.ref
                self._claim_scalar_dataset_identity(
                    scalar_dataset_id,
                    scalar_ref.stream_generation,
                )
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
                    *(
                        scalar_source
                        for _document in configuration.scalar_documents
                    ),
                )
                join_key_type = "camera-roi-source-event-binding"
                join_schema = _ROI_JOIN_KEY_SCHEMA_FINGERPRINT
                join_digest = canonical_digest(
                    {
                        "source_event_ref": event_ref_to_tree(
                            scalar_metadata.source_event_ref
                        ),
                        "binding_fingerprint": scalar_metadata.binding_fingerprint,
                        "control_revision": scalar_metadata.control_revision,
                    }
                )
            stamp = CoherenceStamp(
                candidate.run_id,
                candidate.causation_domain_id,
                join_key_type,
                join_schema,
                join_digest,
                inputs,
                configuration.presentations,
            )
            stale_after_evaluation = False
            job: _LiveRenderJob | None = None
            with self._lock:
                stale_after_evaluation = (
                    self._closed
                    or self._presentation_frozen
                    or self._fault is not None
                    or self._slot.failure is not None
                    or self._slot.notification_failure is not None
                    or self._slot.withdrawn
                    or self._configuration is not configuration
                )
                if not stale_after_evaluation:
                    if self._port is None or panel_sources != self._sources:
                        self._port = self._board.open_publish_port(
                            tuple(
                                PanelSourceBinding(source, presentation)
                                for source, presentation in zip(
                                    panel_sources,
                                    configuration.presentations,
                                    strict=True,
                                )
                            )
                        )
                        self._sources = panel_sources
                    port = self._port
                    assert port is not None
                    sequence = self._sequence
                    self._sequence += 1
                    token = port.admit(
                        sequence,
                        ((configuration.coherence_group, stamp),),
                    )
                    job = _LiveRenderJob(
                        candidate,
                        panel_sources,
                        stamp,
                        sequence,
                        token,
                        port,
                    )
            if stale_after_evaluation:
                self._finish_stale_cycle()
                return False
            assert job is not None
            self._submit(
                lambda: self._render(job),
                ends_cycle=True,
            )
            return True
        except BaseException as error:
            self._set_view_fault(error)
            return False

    def _render(
        self,
        job: _LiveRenderJob,
    ) -> None:
        try:
            if not self._job_is_current(job):
                return
            candidate = job.candidate
            configuration = candidate.configuration
            frozen = candidate.frozen
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
                configuration.image_document,
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
            if not self._job_is_current(job):
                return
            image_payload: ImagePanelPayload | None = None
            image_color_limits: tuple[float, float] | None = None
            image_display = configuration.image_display
            image_viewport = configuration.image_viewport
            if image_display is not None:
                assert image_viewport is not None
                with self._lock:
                    if self._configuration is not configuration:
                        return
                    previous_color_limits = self._image_color_limits
                    previous_relim_mode = self._image_relim_mode
                (
                    image_raster,
                    image_data_range,
                    image_histogram,
                    image_color_limits,
                ) = rasterize_image_indexed8(
                    data,
                    image_display,
                    current_color_limits=previous_color_limits,
                    previous_relim_mode=previous_relim_mode,
                )
                if not self._job_is_current(job):
                    return
                # Keep Matplotlib and its palette owner on the render worker;
                # Qt receives only the immutable sampled QRgb table.
                from zlc_frontend.render_style import indexed_colormap

                image_payload = ImagePanelPayload(
                    image=data,
                    evaluated_input=raw_input[0],
                    viewport=image_viewport,
                    data_range=image_data_range,
                    histogram_counts=image_histogram,
                    base_palette=indexed_colormap(image_display.colormap.value),
                    color_limits=image_color_limits,
                )
            else:
                raise RuntimeError("live IMAGE configuration has no display state")
            rasters = [image_raster]
            payloads: list[DisplayPayload | None] = [image_payload]
            histogram_valid_samples = None
            histogram_dropped_samples = None
            latest_scalar_valid = None
            curve_y_limits: DisplayRange | None = None
            curve_has_valid_samples = False
            histogram_count_limits: DisplayRange | None = None
            histogram_has_valid_samples = False
            if configuration.scalar_documents:
                if (
                    scalar_snapshot is None
                    or configuration.scalar_dataset_id is None
                ):
                    raise RuntimeError("live scalar panels have no scalar snapshot identity")
                scalar_dataset_id = configuration.scalar_dataset_id
                scalar_owned = scalar_snapshot.snapshot
                scalar_input = (EvaluatedInput(scalar_dataset_id, scalar_owned.ref),)
                resolved_scalar = ResolvedDatasetMap(
                    (ResolvedDataset(scalar_dataset_id, scalar_owned),)
                )
            for index, scalar_document in enumerate(
                configuration.scalar_documents
            ):
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
                ):
                    raise RuntimeError("live scalar document must evaluate to one cell")
                scalar_series = scalar_evaluated.layers[0].cells[0].series
                if not scalar_series:
                    raise RuntimeError("live scalar document evaluated no series")
                intent = scalar_document.layers[0].view.intent
                if intent is ViewIntent.CURVE:
                    if any(
                        not isinstance(series.data, EvaluatedCurve)
                        for series in scalar_series
                    ):
                        raise RuntimeError(
                            "live CURVE document evaluated to another data kind"
                        )
                else:
                    if len(scalar_series) != 1:
                        raise RuntimeError(
                            f"live {intent.value} document must evaluate one series"
                        )
                    expected_type = {
                        ViewIntent.HISTOGRAM: EvaluatedHistogram,
                        ViewIntent.METER: EvaluatedMeter,
                    }[intent]
                    if not isinstance(scalar_series[0].data, expected_type):
                        raise RuntimeError(
                            f"live {intent.value} document evaluated to another data kind"
                        )
                scalar_data = scalar_series[0].data
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
                renderer = configuration.scalar_renderers[index]
                if renderer is None:
                    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

                    renderer = SinglePanelAggRenderer(
                        scalar_document,
                        width=self._scalar_size[0],
                        height=self._scalar_size[1],
                    )
                    configuration.scalar_renderers[index] = renderer
                if intent is ViewIntent.CURVE:
                    curve_display = configuration.curve_display
                    if curve_display is None:
                        raise RuntimeError(
                            "live CURVE document has no display configuration"
                        )
                    with self._lock:
                        if self._configuration is not configuration:
                            return
                        accepted_y_limits = self._curve_y_limits
                        accepted_relim_mode = self._curve_relim_mode
                    curve_raster, curve_payload = renderer.render_interactive_curve(
                        scalar_evaluated,
                        curve_display,
                        current_y_limits=accepted_y_limits,
                        previous_relim_mode=accepted_relim_mode,
                    )
                    curve_y_limits = curve_payload.viewport.y_limits
                    curve_has_valid_samples = any(
                        bool(series.data.validity.any())
                        for series in scalar_series
                    )
                    rasters.append(curve_raster)
                    payloads.append(curve_payload)
                elif intent is ViewIntent.HISTOGRAM:
                    histogram_display = configuration.histogram_display
                    if histogram_display is None:
                        raise RuntimeError(
                            "live HISTOGRAM document has no display configuration"
                        )
                    with self._lock:
                        if self._configuration is not configuration:
                            return
                        accepted_count_limits = self._histogram_count_limits
                        accepted_relim_mode = self._histogram_relim_mode
                        accepted_count_scale = self._histogram_count_scale
                    histogram_raster, histogram_payload = (
                        renderer.render_interactive_histogram(
                            scalar_evaluated,
                            histogram_display,
                            current_count_limits=accepted_count_limits,
                            previous_relim_mode=accepted_relim_mode,
                            previous_count_scale=accepted_count_scale,
                        )
                    )
                    histogram_count_limits = (
                        histogram_payload.viewport.count_limits
                    )
                    histogram_has_valid_samples = any(
                        len(series.data.samples) > 0
                        for series in scalar_series
                        if isinstance(series.data, EvaluatedHistogram)
                    )
                    rasters.append(histogram_raster)
                    payloads.append(histogram_payload)
                else:
                    meter_raster, meter_payload = renderer.render_meter(
                        scalar_evaluated,
                        display_revision=(
                            configuration.presentations[index + 1].panel_revision
                        ),
                    )
                    rasters.append(meter_raster)
                    payloads.append(meter_payload)
            frame = BoardFrame(
                configuration.board_id,
                configuration.layout_generation,
                job.sequence,
                tuple(
                    PanelFrame(
                        panel.panel_id,
                        panel.coherence_group,
                        source,
                        job.stamp,
                        raster,
                        payload,
                    )
                    for panel, source, raster, payload in zip(
                        configuration.panels,
                        job.sources,
                        rasters,
                        payloads,
                        strict=True,
                    )
                ),
            )
            if self._job_is_current(job):
                front_status = LiveFrontStatus(
                    sequence=job.sequence,
                    raw_coverage=raw_snapshot.coverage,
                    scalar_coverage=(
                        None if scalar_snapshot is None else scalar_snapshot.coverage
                    ),
                    histogram_valid_samples=histogram_valid_samples,
                    histogram_dropped_samples=histogram_dropped_samples,
                    latest_scalar_valid=latest_scalar_valid,
                    scalar_source_missed=(
                        None
                        if scalar_metadata is None
                        else scalar_metadata.source_missed
                    ),
                    scalar_binding_fingerprint=(
                        None
                        if scalar_metadata is None
                        else scalar_metadata.binding_fingerprint
                    ),
                    scalar_control_revision=(
                        None
                        if scalar_metadata is None
                        else scalar_metadata.control_revision
                    ),
                    image_display_revision=(
                        None if image_display is None else image_display.revision
                    ),
                    image_color_limits=image_color_limits,
                    curve_display_revision=(
                        None
                        if configuration.curve_display is None
                        else configuration.curve_display.revision
                    ),
                    curve_y_limits=curve_y_limits,
                    histogram_display_revision=(
                        None
                        if configuration.histogram_display is None
                        else configuration.histogram_display.revision
                    ),
                    histogram_count_limits=histogram_count_limits,
                )
                published = job.port.publish(job.token, frame)
                if published:
                    accepted_status = False
                    with self._lock:
                        if (
                            not self._closed
                            and self._fault is None
                            and self._configuration is configuration
                            and self._port is job.port
                        ):
                            self._front_status = front_status
                            if image_display is not None:
                                assert image_color_limits is not None
                                self._image_color_limits = image_color_limits
                                self._image_relim_mode = _accepted_relim_mode(
                                    self._image_relim_mode,
                                    image_display.relim_mode,
                                    image_data_range,
                                )
                            curve_display = configuration.curve_display
                            if curve_display is not None:
                                assert curve_y_limits is not None
                                if (
                                    curve_has_valid_samples
                                    or curve_display.relim_mode is RelimMode.FIXED
                                ):
                                    self._curve_y_limits = curve_y_limits
                                self._curve_relim_mode = _accepted_relim_mode(
                                    self._curve_relim_mode,
                                    curve_display.relim_mode,
                                    (
                                        curve_y_limits
                                        if curve_has_valid_samples
                                        else None
                                    ),
                                )
                            histogram_display = configuration.histogram_display
                            if histogram_display is not None:
                                assert histogram_count_limits is not None
                                if (
                                    histogram_has_valid_samples
                                    or histogram_display.relim_mode
                                    is RelimMode.FIXED
                                ):
                                    self._histogram_count_limits = (
                                        histogram_count_limits
                                    )
                                self._histogram_relim_mode = (
                                    histogram_display.relim_mode
                                )
                                self._histogram_count_scale = (
                                    histogram_display.count_scale
                                )
                            accepted_status = True
                    if accepted_status:
                        # BoardController may have queued its wake before this
                        # status record was installed.  A second coalesced wake
                        # makes the matching sequence observable without ever
                        # attaching newer diagnostics to an older visible front.
                        self._request_owner_wake()
        except BaseException as error:
            self._set_view_fault(error, expected_job=job)
            return

    def _candidate_matches_configuration(self, candidate: _LiveCandidate) -> bool:
        configuration = candidate.configuration
        frozen = candidate.frozen
        if not isinstance(frozen, CameraMonitorSnapshot):
            return not configuration.scalar_documents
        scalar_snapshot = frozen.scalar
        scalar_metadata = frozen.scalar_metadata
        if not configuration.scalar_documents:
            return scalar_snapshot is None and scalar_metadata is None
        if scalar_snapshot is None or scalar_metadata is None:
            return False
        scalar_ref = scalar_snapshot.snapshot.ref
        scalar_edge = configuration.scalar_dataset_edge
        if scalar_edge is None:
            return False
        if (
            configuration.scalar_block_id is not None
            and scalar_ref.block_id != configuration.scalar_block_id
        ):
            return False
        if scalar_ref.schema_fingerprint != scalar_edge.schema.fingerprint:
            return False
        if (
            configuration.scalar_generation is not None
            and scalar_ref.stream_generation != configuration.scalar_generation
        ):
            return False
        if (
            scalar_metadata.binding_fingerprint
            != configuration.scalar_binding_fingerprint
        ):
            return False
        if (
            configuration.strict_scalar_identity
            and scalar_metadata.control_revision
            != configuration.scalar_control_revision
        ):
            return False
        return True

    def _finish_stale_cycle(self) -> None:
        restart = False
        with self._lock:
            if self._closed or self._presentation_frozen or self._fault is not None:
                self._active = False
                return
            if self._dirty:
                self._dirty = False
                restart = True
            else:
                self._active = False
        if restart:
            self._submit(self._freeze_latest)

    def _claim_scalar_dataset_identity(
        self,
        dataset_id: DatasetId,
        generation: StreamGenerationId,
    ) -> None:
        known_generation = self._scalar_dataset_generations.get(dataset_id)
        known_dataset = self._scalar_generation_datasets.get(generation)
        if known_generation is not None and known_generation != generation:
            raise RuntimeError(
                "scalar DatasetId changed stream generation without reconfiguration"
            )
        if known_dataset is not None and known_dataset != dataset_id:
            raise RuntimeError(
                "scalar stream generation changed its frontend DatasetId"
            )
        self._scalar_dataset_generations[dataset_id] = generation
        self._scalar_generation_datasets[generation] = dataset_id

    def _job_is_current(self, job: _LiveRenderJob) -> bool:
        with self._lock:
            return (
                not self._closed
                and self._fault is None
                and self._configuration is job.candidate.configuration
                and self._port is job.port
            )

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("live scalar reconfiguration is owner-thread-only")

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
                    self._set_view_fault(error)
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
                            and not self._presentation_frozen
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
            self._set_view_fault(error)

    def _set_view_fault(
        self,
        error: BaseException,
        *,
        expected_job: _LiveRenderJob | None = None,
    ) -> bool:
        def install_view_fault() -> bool:
            with self._lock:
                if self._closed:
                    self._candidate = None
                    self._active = False
                    return False
                if expected_job is not None and (
                    self._configuration is not expected_job.candidate.configuration
                    or self._port is not expected_job.port
                ):
                    return False
                if self._fault is None:
                    self._fault = detached_render_fault(error)
                self._candidate = None
                self._active = False
                self._dirty = False
                self._port = None
                self._sources = None
                return True

        # Board owns the presentation gate and invokes the state callback while
        # that gate is held.  The only nested order is therefore Board -> Live,
        # matching presenter callbacks; no path waits for Board while holding
        # the Live lock, and no visible fault can precede its Board revocation.
        revoked = self._board.revoke_pending_publication(install_view_fault)
        if revoked:
            self._request_owner_wake()
        return revoked

    def close(self) -> None:
        schedule_renderer_close = False
        with self._lock:
            if self._close_complete:
                return
            self._closed = True
            self._active = False
            self._dirty = False
            self._candidate = None
            self._presentation_frozen = False
            self._front_status = None
            self._front_invalidated = True
            configuration = self._configuration
            schedule_renderer_close = bool(configuration.scalar_documents)
        try:
            self._slot.close()
        finally:
            self._board.close()
        if schedule_renderer_close:
            # Use the same serialization gate and the construction-time proven
            # affine lane; close can neither race nor change OS thread.
            def close_renderer() -> None:
                with self._worker_gate:
                    self._close_scalar_renderers(configuration)

            self._submit_worker(close_renderer)
        with self._lock:
            self._fault = None
            self._close_complete = True

    def _close_scalar_renderers(
        self,
        configuration: _LiveRenderConfiguration,
    ) -> None:
        errors: list[BaseException] = []
        renderers = tuple(configuration.scalar_renderers)
        configuration.scalar_renderers[:] = [None for _renderer in renderers]
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
        raise ValueError("live board requires an IMAGE primary document")
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
    _validate_scalar_documents(
        scalar_edge,
        slot.scalar_dataset_id,
        scalar_documents,
    )


def _accepted_relim_mode(
    previous: RelimMode | None,
    mode: RelimMode,
    value_range: tuple[float, float] | None,
) -> RelimMode | None:
    """Advance only the range policy fully initialized by an accepted front."""

    if not isinstance(mode, RelimMode):
        raise TypeError("mode must be RelimMode")
    if value_range is not None or mode is RelimMode.FIXED:
        return mode
    return previous


def _validate_scalar_documents(
    scalar_edge: FrozenDatasetEdge[RoiScalarSample],
    scalar_dataset_id: DatasetId,
    scalar_documents: tuple[FigureDocument, ...],
) -> None:
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
            or scalar_document.datasets[0].dataset_id != scalar_dataset_id
            or scalar_document.datasets[0].schema_fingerprint
            != scalar_edge.schema.fingerprint
            or len(scalar_document.layers) != 1
            or scalar_document.layers[0].dataset_id != scalar_dataset_id
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


def _scalar_semantic_identity(
    configuration: _LiveRenderConfiguration,
) -> tuple[object, ...] | None:
    """Identity whose change invalidates accepted scalar display history."""

    if not configuration.scalar_documents:
        return None
    return (
        configuration.scalar_dataset_id,
        tuple(
            (document.document_id, document.revision)
            for document in configuration.scalar_documents
        ),
        configuration.scalar_binding_fingerprint,
        configuration.scalar_control_revision,
    )


def _build_live_configuration(
    *,
    epoch: int,
    model: BoardModel,
    image_document: FigureDocument,
    image_display: ImageDisplayState | None,
    image_viewport: ImageViewportTransform | None,
    curve_display: CurveDisplayState | None,
    histogram_display: HistogramDisplayState | None,
    scalar_dataset_id: DatasetId | None,
    scalar_documents: tuple[FigureDocument, ...],
    scalar_block_id: BlockId | None,
    scalar_dataset_edge: FrozenDatasetEdge[RoiScalarSample] | None,
    scalar_generation: StreamGenerationId | None,
    scalar_binding_fingerprint: str | None,
    scalar_control_revision: int | None,
    strict_scalar_identity: bool,
    scalar_renderers: list[SinglePanelAggRenderer | None] | None = None,
) -> _LiveRenderConfiguration:
    if not isinstance(model, BoardModel):
        raise TypeError("model must be BoardModel")
    if (image_display is None) != (image_viewport is None):
        raise ValueError("image display state and viewport must be paired")
    if image_display is not None:
        if not isinstance(image_display, ImageDisplayState):
            raise TypeError("image_display must be ImageDisplayState")
        if not isinstance(image_viewport, ImageViewportTransform):
            raise TypeError("image_viewport must be ImageViewportTransform")
        if image_viewport.viewport_revision != image_display.revision:
            raise ValueError("image display and viewport revisions differ")
        image_viewport_for_display_state(image_display, image_viewport)
    if curve_display is not None and not isinstance(
        curve_display,
        CurveDisplayState,
    ):
        raise TypeError("curve_display must be CurveDisplayState or None")
    if histogram_display is not None and not isinstance(
        histogram_display,
        HistogramDisplayState,
    ):
        raise TypeError(
            "histogram_display must be HistogramDisplayState or None"
        )
    panels = model.panels
    documents = (image_document, *scalar_documents)
    if bool(scalar_documents) != (
        curve_display is not None and histogram_display is not None
    ):
        raise ValueError(
            "curve and histogram display states must match scalar documents"
        )
    if len(panels) != len(documents):
        raise ValueError("live board panel count does not match its frozen documents")
    coherence_groups = {panel.coherence_group for panel in panels}
    if len(coherence_groups) != 1:
        raise ValueError("one live snapshot board requires one coherence group")
    if tuple(panel.panel_id for panel in panels) != tuple(
        panel_document.layers[0].layer_id for panel_document in documents
    ):
        raise ValueError("live board panels must match frozen document layers in order")
    presentations = tuple(
        PanelPresentationIdentity(
            panel.panel_id,
            panel_document.document_id,
            panel_document.revision,
            0,
            (
                image_display.revision
                if index == 0 and image_display is not None
                else (
                    curve_display.revision
                    if index == 1 and curve_display is not None
                    else (
                        histogram_display.revision
                        if index == 2 and histogram_display is not None
                        else 0
                    )
                )
            ),
        )
        for index, (panel, panel_document) in enumerate(
            zip(panels, documents, strict=True)
        )
    )
    renderers = (
        [None for _document in scalar_documents]
        if scalar_renderers is None
        else scalar_renderers
    )
    if len(renderers) != len(scalar_documents):
        raise ValueError("scalar renderer holder does not match scalar documents")
    return _LiveRenderConfiguration(
        epoch,
        model.board_id,
        model.layout_generation,
        panels,
        next(iter(coherence_groups)),
        presentations,
        image_document,
        image_display,
        image_viewport,
        curve_display,
        histogram_display,
        scalar_dataset_id,
        scalar_documents,
        scalar_block_id,
        scalar_dataset_edge,
        scalar_generation,
        scalar_binding_fingerprint,
        scalar_control_revision,
        strict_scalar_identity,
        renderers,
    )


__all__ = ["LiveBoardController", "LiveDatasetSlot", "LiveFrontStatus"]
