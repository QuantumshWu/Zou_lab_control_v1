"""Qt composition for one free-running camera and typed ROI monitor."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    MONITOR_HISTORY,
    ReductionMethod,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
)
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureLayer,
    FixedIndex,
    SuggestionStatus,
    ViewIntent,
    ViewSpec,
    estimate_view_evaluation_peak_nbytes,
    suggest_view,
    validate_view_spec,
)
from zlc_frontend.image_raster import estimate_gray8_raster_peak_nbytes
from zlc_frontend.matplotlib_render import (
    DEFAULT_HISTOGRAM_BINS,
    estimate_live_panel_raster_peak_nbytes,
)
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentSwitch,
    GREEN,
    ImageViewportTransform,
    ORANGE,
    QtOwnerWake,
    QtRasterBoard,
    RectangleGesture,
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    release_window,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
)
from zlc_frontend.render import RenderSurface
from zlc_neutral_atom.monitor_application import (
    CameraMonitorRequest,
    CameraMonitorViewSpec,
    PreparedCameraMonitor,
)
from zlc_neutral_atom.runtime.run import CancelOutcome, RunHandle, RunSnapshot, RunState
from zlc_workbench.live import LiveDatasetSlot, LiveImageBoardController
from zlc_workbench.run_owner import QtRunOwnerMailbox
from zlc_workbench.workspace import BoardController, BoardModel, PanelSlot


_IMAGE_PANEL_ID = "camera-monitor-image"
_CURVE_PANEL_ID = "camera-monitor-roi-curve"
_HISTOGRAM_PANEL_ID = "camera-monitor-roi-histogram"
_METER_PANEL_ID = "camera-monitor-roi-meter"
_SCALAR_PANEL_IDS = (
    _CURVE_PANEL_ID,
    _HISTOGRAM_PANEL_ID,
    _METER_PANEL_ID,
)
_SCALAR_RASTER_SIZE = (800, 520)
_RAW_PROJECTION_TEXT = "latest raw frame · history slot 0 · DISPLAY ONLY"


def _roi_scalar_views(schema, binding):
    axes = (schema.repeat_axis, *schema.point_axes, *schema.cell_schema.data_axes)
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    if len(history) != 1 or schema.cell_schema.data_axes:
        raise ValueError("ROI scalar views require one scalar MONITOR_HISTORY axis")
    repeat_binding = AxisViewBinding(
        schema.repeat_axis.axis_id,
        AxisViewRole.SELECTED,
        selector=FixedIndex(0),
    )
    curve = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(history[0].axis_id, AxisViewRole.X),
            repeat_binding,
        ),
    )
    histogram = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(history[0].axis_id, AxisViewRole.SAMPLE),
            repeat_binding,
        ),
    )
    meter = ViewSpec(
        schema.fingerprint,
        ViewIntent.METER,
        (
            AxisViewBinding(
                history[0].axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            repeat_binding,
        ),
    )
    views = (curve, histogram, meter)
    if any(len(view.axis_bindings) != len(axes) for view in views):
        raise ValueError("ROI scalar views refuse undeclared extra axes")
    for view in views:
        validate_view_spec(schema, view)
    input_axes = {axis.axis_id: axis for axis in binding.input_contract.value_schema.data_axes}
    terms = {term.axis_id: term for term in binding.selection.terms}
    description = ", ".join(
        f"{input_axes[axis_id].name}={term.lower}..{term.upper}"
        for axis_id, term in sorted(terms.items(), key=lambda item: item[0].value)
    )
    summary = (
        f"latest raw frame + ROI {binding.reduction.value.lower()} scalar "
        f"[{description}] · binding {binding.fingerprint[:12]} · "
        f"validity {binding.validity_policy.value.lower()} · "
        f"scalar history 0..{history[0].size - 1} (0 newest) · "
        f"curve + {DEFAULT_HISTOGRAM_BINS}-bin histogram + latest meter · "
        "MONITOR DERIVED / DISPLAY ONLY"
    )
    return views, summary


@dataclass(frozen=True, slots=True)
class _PreparedMonitorView:
    """One fully checked headless view product awaiting a single Run start."""

    command: PreparedCameraMonitor
    generation: int
    dataset_id: DatasetId
    scalar_dataset_id: DatasetId | None
    image_document: FigureDocument
    scalar_documents: tuple[FigureDocument, ...]
    evaluation_policy: FigureEvaluationPolicy
    downstream_peak_bytes: int
    viewport: ImageViewportTransform
    projection_text: str


def _prepare_monitor_view(
    command: PreparedCameraMonitor,
    generation: int,
) -> _PreparedMonitorView:
    """Finish every pure display check before an existing Run is replaced."""

    if not isinstance(command, PreparedCameraMonitor):
        raise TypeError("command must be PreparedCameraMonitor")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("monitor view generation must be a positive integer")
    schema = command.view_schema
    scalar_schema = command.scalar_view_schema
    roi_binding = command.roi_binding
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    y_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_Y
    )
    x_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_X
    )
    if (
        len(history) != 1
        or len(y_axes) != 1
        or len(x_axes) != 1
        or len(schema.cell_schema.data_axes) != 2
    ):
        raise ValueError(
            "camera monitor requires one history axis and declared "
            "SPATIAL_Y/SPATIAL_X image axes"
        )
    selection = Selection.index(history[0].axis_id, 0)
    suggestion = suggest_view(schema, ViewIntent.IMAGE, selection)
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        raise ValueError("camera monitor IMAGE view needs an explicit axis choice")
    image_view = suggestion.spec
    image_evaluation_peak = estimate_view_evaluation_peak_nbytes(schema, image_view)
    downstream_peak = image_evaluation_peak + estimate_gray8_raster_peak_nbytes(
        y_axes[0].size,
        x_axes[0].size,
    )

    scalar_views: tuple[ViewSpec, ...] = ()
    scalar_evaluation_peaks: tuple[int, ...] = ()
    projection_text = _RAW_PROJECTION_TEXT
    viewport = ImageViewportTransform(
        (y_axes[0], x_axes[0]),
        viewport_revision=generation,
    )
    if scalar_schema is None:
        if roi_binding is not None or command.request.roi is not None:
            raise RuntimeError("ROI request has no scalar view product")
    else:
        if roi_binding is None or command.request.roi is None:
            raise RuntimeError("scalar view product has no admitted ROI binding")
        if (
            roi_binding.selection != command.request.roi
            or roi_binding.reduction is not command.request.roi_reduction
        ):
            raise RuntimeError("prepared ROI binding differs from its immutable request")
        scalar_views, projection_text = _roi_scalar_views(scalar_schema, roi_binding)
        default_policy = FigureEvaluationPolicy()
        scalar_history = tuple(
            axis for axis in scalar_schema.point_axes if axis.role == MONITOR_HISTORY
        )
        if (
            len(scalar_history) != 1
            or scalar_history[0].size > default_policy.max_histogram_samples
        ):
            raise ValueError(
                "ROI scalar history exceeds the live histogram sample limit "
                f"{default_policy.max_histogram_samples}"
            )
        scalar_evaluation_peaks = tuple(
            estimate_view_evaluation_peak_nbytes(scalar_schema, view)
            for view in scalar_views
        )
        for view, evaluation_peak in zip(
            scalar_views,
            scalar_evaluation_peaks,
            strict=True,
        ):
            downstream_peak += evaluation_peak + estimate_live_panel_raster_peak_nbytes(
                *_SCALAR_RASTER_SIZE,
                evaluated_data_upper_bound_bytes=evaluation_peak,
                histogram_bins=(
                    DEFAULT_HISTOGRAM_BINS
                    if view.intent is ViewIntent.HISTOGRAM
                    else None
                ),
            )
        # Resolve once during preparation so binding the GUI overlay after the
        # old Run is gone cannot discover a coordinate/selection mismatch.
        viewport.normalized_bounds_for_selection(roi_binding.selection)

    policy = FigureEvaluationPolicy(
        max_live_nbytes=max((image_evaluation_peak, *scalar_evaluation_peaks))
    )
    required_peak = command.descriptor.base_peak_bytes + downstream_peak
    if required_peak > command.request.memory_limit_bytes:
        raise MemoryError(
            f"camera monitor peak budget {required_peak} exceeds limit "
            f"{command.request.memory_limit_bytes}"
        )

    dataset_id = DatasetId(f"camera-monitor-{generation}")
    scalar_dataset_id = (
        None
        if scalar_schema is None
        else DatasetId(f"camera-monitor-roi-scalar-{generation}")
    )
    image_document = FigureDocument(
        f"camera-monitor-image-{generation}",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "Raw monitor camera frame",
                schema.fingerprint,
            ),
        ),
        (FigureLayer(_IMAGE_PANEL_ID, dataset_id, image_view),),
    )
    scalar_documents: tuple[FigureDocument, ...] = ()
    if scalar_views:
        assert scalar_schema is not None and scalar_dataset_id is not None
        assert roi_binding is not None
        descriptor = DatasetDescriptor(
            scalar_dataset_id,
            f"ROI {roi_binding.reduction.value.lower()} scalar monitor",
            scalar_schema.fingerprint,
        )
        scalar_documents = tuple(
            FigureDocument(
                f"{panel_id}-{generation}",
                0,
                (descriptor,),
                (FigureLayer(panel_id, scalar_dataset_id, view),),
            )
            for panel_id, view in zip(
                _SCALAR_PANEL_IDS,
                scalar_views,
                strict=True,
            )
        )
    return _PreparedMonitorView(
        command,
        generation,
        dataset_id,
        scalar_dataset_id,
        image_document,
        scalar_documents,
        policy,
        downstream_peak,
        viewport,
        projection_text,
    )


class CameraMonitorWorkbenchWindow(QtWidgets.QWidget):
    """Nonblocking owner of one continuous monitor Run and immutable front."""

    def __init__(self, prepare, request: CameraMonitorRequest) -> None:
        super().__init__()
        if not callable(prepare):
            raise TypeError("camera monitor prepare must be callable")
        if not isinstance(request, CameraMonitorRequest):
            raise TypeError("request must be CameraMonitorRequest")
        self._prepare = prepare
        self._request = request
        self._projection_text = (
            _RAW_PROJECTION_TEXT
            if request.roi is None
            else (
                f"typed ROI {request.roi_reduction.value.lower()} scalar · "
                f"history {request.scalar_history_capacity} · PREPARING"
            )
        )
        self._prepared: _PreparedMonitorView | None = None
        self._prepared_apply: _PreparedMonitorView | None = None
        self._pending_request: CameraMonitorRequest | None = None
        self._applied_request = request
        self._running_binding = None
        self._visible_binding_fingerprint: str | None = None
        self._draft_selection: Selection | None = None
        self._viewport_transform = None
        self._selector_interacting = False
        self._apply_phase: str | None = None
        self._slot: LiveDatasetSlot | None = None
        self._live: LiveImageBoardController | None = None
        self._board: BoardController | None = None
        self._last_snapshot: RunSnapshot | None = None
        self._prepare_inflight = False
        self._local_diagnostic = ""
        self._manual_stop_requested = False
        self._closing = False
        self._allow_close = False

        self.setWindowTitle("Camera Monitor")
        self._run_status = FluentLabel("Monitor: PREPARING")
        self._run_status.setObjectName("monitorStatus")
        self._view_status = FluentLabel(f"View: WAITING · {self._projection_text}")
        self._view_status.setObjectName("monitorViewStatus")
        self._view_status.setWordWrap(True)
        self._projection_status = FluentLabel(f"Display: {self._projection_text}")
        self._projection_status.setObjectName("projectionStatus")
        self._projection_status.setWordWrap(True)
        self._diagnostics = FluentLabel("")
        self._diagnostics.setObjectName("diagnostics")
        self._diagnostics.setWordWrap(True)
        self._board_widget: QtRasterBoard | None = None
        self._board_panel_ids: tuple[str, ...] | None = None
        self._board_container = QtWidgets.QWidget(self)
        self._board_layout = QtWidgets.QVBoxLayout(self._board_container)
        self._board_layout.setContentsMargins(0, 0, 0, 0)
        self._configure_board_widget(scalar=request.roi is not None)
        self._selector_switch = FluentSwitch("ROI selector", self)
        self._selector_switch.setObjectName("roiSelectorSwitch")
        self._selector_switch.setChecked(False)
        self._selector_switch.setEnabled(False)
        self._reducer_combo = FluentComboBox(self)
        self._reducer_combo.setObjectName("roiReducerCombo")
        for label, reduction in (
            ("Mean", ReductionMethod.MEAN),
            ("Sum", ReductionMethod.SUM),
            ("Max", ReductionMethod.MAX),
        ):
            self._reducer_combo.addItem(label, reduction)
        initial_reducer = self._reducer_combo.findData(request.roi_reduction)
        if initial_reducer >= 0:
            self._reducer_combo.setCurrentIndex(initial_reducer)
        self._reducer_combo.setEnabled(False)
        self._apply_roi_button = FluentButton("Apply ROI", self, color=ORANGE)
        self._apply_roi_button.setObjectName("applyRoiButton")
        self._apply_roi_button.setEnabled(False)
        self._roi_status = FluentLabel(
            (
                "ROI: start the raw monitor, then draw a rectangle"
                if request.roi is None
                else "ROI: fixed request"
            )
        )
        self._roi_status.setObjectName("roiStatus")
        self._roi_status.setWordWrap(True)
        self._start_button = FluentButton("Start", self, color=GREEN)
        self._start_button.setObjectName("startButton")
        self._start_button.setEnabled(False)
        self._stop_button = FluentButton("Stop", self, color=ORANGE)
        self._stop_button.setObjectName("stopButton")
        self._stop_button.setEnabled(False)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._selector_switch)
        controls.addWidget(self._reducer_combo)
        controls.addWidget(self._apply_roi_button)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._run_status)
        layout.addWidget(self._view_status)
        layout.addWidget(self._projection_status)
        layout.addWidget(self._roi_status)
        layout.addWidget(self._board_container, 1)
        layout.addWidget(self._diagnostics)
        layout.addLayout(controls)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._run_owner = QtRunOwnerMailbox(
            self._wake.request_owner_wake,
            thread_name_prefix="zlc-camera-monitor-workbench",
            max_workers=1,
        )
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll_run_snapshot)
        self._timer.start()
        self._start_button.clicked.connect(self._start_monitor)
        self._stop_button.clicked.connect(self._cancel_monitor)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
        self._reducer_combo.currentIndexChanged.connect(self._update_roi_controls)
        self._apply_roi_button.clicked.connect(self._apply_roi_draft)
        self._submit_prepare()

    @property
    def _handle(self) -> RunHandle | None:
        """Read-only product view of the shared Run owner."""

        return self._run_owner.handle

    @property
    def worker_idle(self) -> bool:
        return self._run_owner.worker_idle

    def _submit_worker(self, work):
        return self._run_owner.submit_render(work)

    def _sync_live_pause(self) -> None:
        live = self._live
        if live is None:
            return
        if self._selector_interacting or self._apply_phase in {
            "PREPARING",
            "REPLACING",
        }:
            live.pause()
        else:
            live.resume()

    def _set_selector_enabled(self, enabled: bool) -> None:
        board = self._board_widget
        setter = getattr(board, "set_rectangle_selector_enabled", None)
        if callable(setter):
            setter(bool(enabled))

    def _set_selector_interacting(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise TypeError("selector interaction state must be bool")
        self._selector_interacting = active
        self._sync_live_pause()
        self._wake.request_owner_wake()

    def _accept_rectangle_gesture(self, gesture: RectangleGesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("selector callback requires RectangleGesture")
        board = self._board_widget
        viewport = self._viewport_transform
        if not isinstance(board, QtRasterBoard) or viewport is None:
            raise RuntimeError("ROI selector has no active raster viewport")
        front = board.front_frame
        if (
            front is None
            or gesture.panel_id != _IMAGE_PANEL_ID
            or gesture.board_id != front.board_id
            or gesture.layout_generation != front.layout_generation
            or gesture.sequence != front.sequence
            or gesture.viewport_revision != viewport.viewport_revision
        ):
            raise RuntimeError("ROI gesture is stale relative to the visible front")
        image_panel = next(
            (panel for panel in front.panels if panel.panel_id == _IMAGE_PANEL_ID),
            None,
        )
        if image_panel is None or image_panel.source_identity != gesture.source_identity:
            raise RuntimeError("ROI gesture source differs from the visible image")
        selection = viewport.selection_for_normalized_bounds(
            gesture.normalized_bounds
        )
        if selection == self._applied_request.roi:
            selection = None
        self._draft_selection = selection
        board.set_selector_draft_selection(selection)
        if selection is None:
            self._roi_status.setText("ROI: no unapplied rectangle change")
        else:
            terms = sorted(selection.terms, key=lambda term: term.axis_id.value)
            coordinates = ", ".join(
                f"{term.axis_id.value}={term.lower}..{term.upper}"
                for term in terms
            )
            self._roi_status.setText(
                f"ROI: DRAFT · {coordinates} · click Apply ROI"
            )
        self._update_roi_controls()

    def _selected_reducer(self) -> ReductionMethod:
        value = self._reducer_combo.currentData()
        if not isinstance(value, ReductionMethod):
            raise TypeError("ROI reducer control contains an invalid value")
        return value

    def _update_roi_controls(self, *_args) -> None:
        busy = (
            self._closing
            or self._manual_stop_requested
            or self._apply_phase is not None
        )
        active = (
            self._handle is not None
            and not self._handle.snapshot().state.terminal
            and self._board_widget is not None
            and self._board_widget.has_front
        )
        changed = (
            (
                self._draft_selection is not None
                and self._draft_selection != self._applied_request.roi
            )
            or (
                self._applied_request.roi is not None
                and self._selected_reducer() is not self._applied_request.roi_reduction
            )
        )
        self._selector_switch.setEnabled(not busy and active)
        self._reducer_combo.setEnabled(not busy and active)
        self._apply_roi_button.setEnabled(
            not busy and active and changed
        )
        self._set_selector_enabled(
            self._selector_switch.isChecked() and not busy and active
        )

    def _apply_roi_draft(self) -> None:
        if (
            self._closing
            or self._manual_stop_requested
            or self._apply_phase is not None
        ):
            return
        handle = self._handle
        if handle is None or handle.snapshot().state.terminal:
            return
        selection = self._draft_selection or self._applied_request.roi
        if selection is None:
            return
        candidate = replace(
            self._applied_request,
            roi=selection,
            roi_reduction=self._selected_reducer(),
        )
        if candidate == self._applied_request:
            return
        generation = self._run_owner.generation
        view_generation = generation + 1
        self._pending_request = candidate
        self._apply_phase = "PREPARING"
        self._roi_status.setText("ROI: validating replacement request")
        self._stop_button.setEnabled(False)
        self._sync_live_pause()
        self._update_roi_controls()

        def prepare() -> _PreparedMonitorView:
            command = self._prepare(candidate)
            if not isinstance(command, PreparedCameraMonitor):
                raise TypeError("ROI apply prepare returned an unexpected command")
            if command.request != candidate:
                raise ValueError("ROI apply prepare changed the immutable request")
            if command.roi_binding is None:
                raise ValueError("ROI apply prepare produced no scalar binding")
            return _prepare_monitor_view(command, view_generation)

        self._run_owner.submit("apply_prepare", prepare, generation=generation)

    def _configure_board_widget(self, *, scalar: bool) -> None:
        panel_ids = (
            (_IMAGE_PANEL_ID, *_SCALAR_PANEL_IDS)
            if scalar
            else (_IMAGE_PANEL_ID,)
        )
        current = self._board_widget
        if isinstance(current, QtRasterBoard) and self._board_panel_ids == panel_ids:
            return
        if current is not None:
            self._board_layout.removeWidget(current)
            current.setParent(None)
            current.deleteLater()
        widget = QtRasterBoard(
            panel_ids,
            self._board_container,
            columns=2 if scalar else 1,
            empty_text="Monitor stopped",
        )
        widget.setObjectName("cameraMonitorImageBoard")
        self._board_layout.addWidget(widget)
        self._board_widget = widget
        self._board_panel_ids = panel_ids

    def _submit_prepare(self) -> None:
        if self._closing or self._prepare_inflight or self._prepared is not None:
            return
        self._prepare_inflight = True
        generation = self._run_owner.generation
        view_generation = generation + 1
        request = self._request

        def prepare() -> _PreparedMonitorView:
            command = self._prepare(request)
            if not isinstance(command, PreparedCameraMonitor):
                raise TypeError("camera monitor prepare returned an unexpected command")
            if command.request != request:
                raise ValueError("prepared camera monitor differs from the frozen UI request")
            return _prepare_monitor_view(command, view_generation)

        self._run_owner.submit("prepare", prepare, generation=generation)

    def _start_monitor(self) -> None:
        if self._closing or self._prepared is None:
            return
        if self._handle is not None and not self._handle.snapshot().state.terminal:
            return
        if not self._run_owner.owner_reaped:
            return
        prepared_view = self._prepared
        command = prepared_view.command
        if prepared_view.generation != self._run_owner.generation + 1:
            self._prepared = None
            if self._apply_phase == "WAITING_FRONT":
                self._apply_phase = None
                self._roi_status.setText(
                    "ROI: replacement view became stale · preparing retry"
                )
            self._record_local_failure(
                "Prepared monitor view is stale relative to the next Run generation"
            )
            self._submit_prepare()
            return
        roi_binding = command.roi_binding
        self._running_binding = roi_binding
        self._visible_binding_fingerprint = None
        if self._apply_phase != "WAITING_FRONT":
            self._draft_selection = None
        self._close_live()
        generation = self._run_owner.begin_generation()
        try:
            selector_board = self._board_widget
            selector_viewport = prepared_view.viewport
            if not isinstance(selector_board, QtRasterBoard):
                raise RuntimeError("ROI selector host changed after preparation")
            if selector_viewport.viewport_revision != generation:
                raise RuntimeError("selector viewport generation prediction changed")
            selector_board.bind_rectangle_selector(
                _IMAGE_PANEL_ID,
                selector_viewport,
                self._accept_rectangle_gesture,
                interaction_callback=self._set_selector_interacting,
                enabled=False,
            )
            selector_board.set_selector_applied_selection(
                None if roi_binding is None else roi_binding.selection
            )
            selector_board.set_selector_draft_selection(None)
            self._viewport_transform = selector_viewport
        except BaseException as error:
            self._run_owner.mark_owner_reaped()
            self._running_binding = None
            self._prepared = None
            if self._apply_phase == "WAITING_FRONT":
                self._apply_phase = None
                self._roi_status.setText(
                    "ROI: replacement selector setup failed · preparing retry"
                )
            self._record_local_failure(
                f"ROI selector setup failed: {type(error).__name__}: {error}"
            )
            self._submit_prepare()
            return
        self._prepared = None
        self._last_snapshot = None
        self._manual_stop_requested = False
        self._local_diagnostic = ""
        self._projection_text = prepared_view.projection_text
        self._refresh_diagnostics(None)
        self._run_status.setText("Monitor: STARTING")
        self._view_status.setText(f"View: WAITING · {self._projection_text}")
        self._projection_status.setText(f"Display: {self._projection_text}")
        if roi_binding is not None:
            self._roi_status.setText(
                "ROI: waiting for first coherent front · binding "
                f"{roi_binding.fingerprint[:12]}"
            )
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._update_roi_controls()

        dataset_id = prepared_view.dataset_id
        scalar_dataset_id = prepared_view.scalar_dataset_id
        image_document = prepared_view.image_document
        scalar_documents = prepared_view.scalar_documents

        def factory(spec: CameraMonitorViewSpec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=dataset_id,
                scalar_dataset_id=scalar_dataset_id,
                evaluation_policy=prepared_view.evaluation_policy,
                retain_on_terminal=False,
            )
            attached = threading.Event()
            response: dict[str, object] = {}
            self._run_owner.post_attachment(
                (slot, image_document, scalar_documents, attached, response),
                generation=generation,
            )
            if not attached.wait(5.0):
                slot.fail("GUI monitor attachment timed out")
                raise TimeoutError("GUI monitor attachment timed out")
            if response.get("accepted") is not True:
                reason = str(response.get("error", "GUI monitor attachment was rejected"))
                slot.fail(reason)
                raise RuntimeError(reason)
            return slot

        self._run_owner.submit(
            "start",
            lambda: command.start_with_view(
                downstream_peak_bytes=prepared_view.downstream_peak_bytes,
                factory=factory,
            ),
            generation=generation,
        )

    def _cancel_monitor(self) -> None:
        handle = self._handle
        if handle is None or handle.snapshot().state.terminal:
            return
        outcome = handle.cancel("Workbench user requested monitor stop")
        self._manual_stop_requested = True
        self._stop_button.setEnabled(False)
        self._local_diagnostic = f"Stop: {outcome.value}"
        self._refresh_diagnostics(self._last_snapshot)
        self._update_roi_controls()

    def _poll_run_snapshot(self) -> None:
        handle = self._handle
        if handle is None:
            if self._closing:
                self._wake.request_owner_wake()
            return
        if self._run_owner.poll_snapshot(self._last_snapshot) or self._closing:
            self._wake.request_owner_wake()

    def _owner_cycle(self) -> None:
        try:
            self._drain_results()
            self._drain_attachments()
            live = self._live
            board = self._board
            view_frozen = self._selector_interacting or self._apply_phase in {
                "PREPARING",
                "REPLACING",
            }
            if board is not None and board.fault is None and not view_frozen:
                board.present_pending()
            if live is not None and not view_frozen:
                live.admit_pending()
            pending = self._run_owner.take_pending_snapshot()
            if pending is not None:
                generation, snapshot = pending
                handle = self._handle
                if (
                    generation == self._run_owner.generation
                    and handle is not None
                    and snapshot.run_id == handle.run_id
                ):
                    self._reconcile_snapshot(snapshot)
            self._refresh_view_status()
            self._maybe_finish_close()
        except BaseException as error:
            message = f"Workbench owner failed: {type(error).__name__}: {error}"
            slot = self._slot
            if slot is not None:
                try:
                    slot.fail(message)
                except BaseException:
                    pass
            self._record_local_failure(message)
            self._maybe_finish_close()

    def _drain_results(self) -> None:
        for completion in self._run_owner.drain_completions():
            kind = completion.kind
            generation = completion.generation
            future = completion.future
            if kind == "render":
                try:
                    future.result()
                except BaseException as error:
                    self._record_local_failure(
                        f"Render worker failed: {type(error).__name__}: {error}"
                    )
                continue
            if kind == "apply_prepare":
                try:
                    prepared = future.result()
                    if not isinstance(prepared, _PreparedMonitorView):
                        raise TypeError("ROI apply returned an invalid view product")
                except BaseException as error:
                    if (
                        not self._closing
                        and generation == self._run_owner.generation
                        and self._apply_phase == "PREPARING"
                    ):
                        self._prepared_apply = None
                        self._pending_request = None
                        self._apply_phase = None
                        self._roi_status.setText(
                            "ROI: rejected before restart · "
                            f"{type(error).__name__}: {error}"
                        )
                        self._stop_button.setEnabled(True)
                        self._sync_live_pause()
                        self._update_roi_controls()
                    continue
                if (
                    self._closing
                    or generation != self._run_owner.generation
                    or self._apply_phase != "PREPARING"
                ):
                    continue
                candidate = self._pending_request
                if (
                    candidate is None
                    or prepared.command.request != candidate
                    or prepared.generation != generation + 1
                ):
                    self._prepared_apply = None
                    self._pending_request = None
                    self._apply_phase = None
                    self._roi_status.setText("ROI: rejected · staged request identity changed")
                    self._stop_button.setEnabled(True)
                    self._sync_live_pause()
                    self._update_roi_controls()
                    continue
                handle = self._handle
                if handle is None or handle.snapshot().state.terminal:
                    self._prepared_apply = None
                    self._pending_request = None
                    self._apply_phase = None
                    self._roi_status.setText("ROI: rejected · monitor is no longer running")
                    self._sync_live_pause()
                    self._update_roi_controls()
                    continue
                self._prepared_apply = prepared
                outcome = handle.cancel(
                    "Workbench is applying a new immutable ROI request"
                )
                if outcome is not CancelOutcome.REQUESTED:
                    self._prepared_apply = None
                    self._pending_request = None
                    self._apply_phase = None
                    self._roi_status.setText(
                        "ROI: not applied · monitor was already stopping "
                        f"({outcome.value})"
                    )
                    self._sync_live_pause()
                    self._update_roi_controls()
                    continue
                self._apply_phase = "REPLACING"
                self._roi_status.setText(
                    "ROI: request valid · stopping old monitor generation"
                )
                self._stop_button.setEnabled(False)
                self._update_roi_controls()
                continue
            if kind == "prepare":
                if generation == self._run_owner.generation:
                    self._prepare_inflight = False
                try:
                    prepared = future.result()
                    if not isinstance(prepared, _PreparedMonitorView):
                        raise TypeError("camera prepare returned an invalid view product")
                    if prepared.generation != generation + 1:
                        raise RuntimeError("prepared camera view generation changed")
                except BaseException as error:
                    if not self._closing and generation == self._run_owner.generation:
                        self._run_status.setText("Monitor: NOT READY")
                        self._record_local_failure(
                            f"Preparation failed: {type(error).__name__}: {error}"
                        )
                    continue
                if not self._closing and generation == self._run_owner.generation:
                    self._prepared = prepared
                    scalar_schema = prepared.command.scalar_view_schema
                    binding = prepared.command.roi_binding
                    self._configure_board_widget(scalar=scalar_schema is not None)
                    if scalar_schema is not None and binding is None:
                        raise RuntimeError("prepared ROI scalar schema has no binding")
                    self._projection_text = prepared.projection_text
                    self._projection_status.setText(
                        f"Display: {self._projection_text}"
                    )
                    self._view_status.setText(
                        f"View: WAITING · {self._projection_text}"
                    )
                    if (
                        self._last_snapshot is not None
                        and self._last_snapshot.state is RunState.CANCELLED
                    ):
                        self._run_status.setText("Monitor: READY · previous STOPPED")
                    else:
                        self._run_status.setText("Monitor: READY")
                    self._update_start_button()
                    self._update_roi_controls()
                continue
            if kind == "start":
                try:
                    handle = future.result()
                except BaseException as error:
                    if generation == self._run_owner.generation:
                        self._run_owner.mark_owner_reaped()
                        self._run_status.setText("Monitor: FAILED")
                        self._record_local_failure(
                            f"Start failed: {type(error).__name__}: {error}"
                        )
                        self._close_live()
                        if self._apply_phase == "WAITING_FRONT":
                            self._apply_phase = None
                            self._roi_status.setText(
                                "ROI: replacement start failed · use Start to retry"
                            )
                        self._submit_prepare()
                        self._update_roi_controls()
                    continue
                if self._closing or generation != self._run_owner.generation:
                    if self._closing:
                        self._run_owner.set_handle(handle)
                    handle.cancel("Workbench closed before monitor admission returned")
                else:
                    self._run_owner.set_handle(handle)
                    self._run_status.setText("Monitor: RUNNING")
                    self._stop_button.setEnabled(True)
                    self._run_owner.enqueue_snapshot(
                        handle.snapshot(),
                        generation=generation,
                    )
                    self._update_roi_controls()
                continue
            if kind == "reap":
                try:
                    future.result()
                except BaseException as error:
                    self._run_owner.finish_terminal_job(
                        generation,
                        owner_reaped=False,
                    )
                    self._record_local_failure(
                        f"Run owner reap failed: {type(error).__name__}: {error}"
                    )
                else:
                    self._run_owner.finish_terminal_job(
                        generation,
                        owner_reaped=True,
                    )
                    self._close_live()
                    if not self._closing:
                        if (
                            self._apply_phase == "REPLACING"
                            and self._prepared_apply is not None
                            and self._pending_request is not None
                        ):
                            if self._manual_stop_requested:
                                self._prepared_apply = None
                                self._pending_request = None
                                self._apply_phase = None
                                self._request = self._applied_request
                                self._roi_status.setText(
                                    "ROI: not applied · monitor stopped by user"
                                )
                                self._submit_prepare()
                            elif (
                                self._last_snapshot is None
                                or self._last_snapshot.state is not RunState.CANCELLED
                            ):
                                self._prepared_apply = None
                                self._pending_request = None
                                self._apply_phase = None
                                self._request = self._applied_request
                                self._roi_status.setText(
                                    "ROI: replacement aborted · old monitor did not "
                                    "terminate by the requested cancellation"
                                )
                                self._submit_prepare()
                            else:
                                self._request = self._pending_request
                                self._pending_request = None
                                self._prepared = self._prepared_apply
                                self._prepared_apply = None
                                self._configure_board_widget(
                                    scalar=(
                                        self._prepared.command.scalar_view_schema
                                        is not None
                                    )
                                )
                                self._apply_phase = "WAITING_FRONT"
                                self._roi_status.setText(
                                    "ROI: old generation terminal · starting replacement"
                                )
                                self._start_monitor()
                        else:
                            self._submit_prepare()
                continue

    def _drain_attachments(self) -> None:
        for generation, payload in self._run_owner.drain_attachments():
            slot, image_document, scalar_documents, attached, response = payload
            try:
                if self._closing or generation != self._run_owner.generation:
                    raise RuntimeError("Workbench no longer accepts this monitor")
                if slot.terminal:
                    raise RuntimeError("monitor slot became terminal before attachment")
                if scalar_documents and len(scalar_documents) != len(_SCALAR_PANEL_IDS):
                    raise RuntimeError(
                        "ROI monitor attachment requires its closed three-panel scalar set"
                    )
                self._close_live()
                panels = [
                    PanelSlot(
                        _IMAGE_PANEL_ID,
                        "camera-monitor",
                        "monitor-camera",
                    )
                ]
                panels.extend(
                    PanelSlot(
                        panel_id,
                        "camera-monitor",
                        "monitor-camera",
                    )
                    for panel_id in _SCALAR_PANEL_IDS[: len(scalar_documents)]
                )
                model = BoardModel(
                    "camera-monitor-board",
                    generation,
                    RenderSurface.WORKER_RASTER_LIVE,
                    tuple(panels),
                )
                board_widget = self._board_widget
                if board_widget is None:
                    raise RuntimeError("camera monitor board was not configured")
                board = BoardController(
                    model,
                    board_widget,
                    self._wake.request_owner_wake,
                )
                live = LiveImageBoardController(
                    slot,
                    image_document,
                    board,
                    submit_worker=self._submit_worker,
                    request_owner_wake=self._wake.request_owner_wake,
                    scalar_documents=scalar_documents,
                    scalar_raster_size=_SCALAR_RASTER_SIZE,
                    worker_thread_affine=self._run_owner.worker_thread_affine,
                )
                self._slot = slot
                self._board = board
                self._live = live
                response["accepted"] = True
            except BaseException as error:
                response["accepted"] = False
                response["error"] = f"{type(error).__name__}: {error}"
                try:
                    slot.fail(str(response["error"]))
                except BaseException:
                    pass
            finally:
                attached.set()

    def _reconcile_snapshot(self, snapshot: RunSnapshot) -> None:
        self._last_snapshot = snapshot
        state = snapshot.state
        if not state.terminal:
            self._run_status.setText(f"Monitor: {state.value} / {snapshot.phase}")
            self._stop_button.setEnabled(
                not self._manual_stop_requested
                and self._apply_phase not in {"PREPARING", "REPLACING"}
            )
        else:
            self._stop_button.setEnabled(False)
            if state is RunState.CANCELLED:
                self._run_status.setText("Monitor: STOPPED")
            else:
                self._run_status.setText(f"Monitor: {state.value}")
            handle = self._handle
            assert handle is not None
            self._run_owner.begin_terminal_job("reap", handle.wait)
        self._refresh_diagnostics(snapshot)

    def _refresh_view_status(self) -> None:
        slot = self._slot
        live = self._live
        board = self._board
        failure = None if slot is None else slot.failure
        fault = None if live is None else live.fault
        if fault is None and board is not None:
            fault = board.fault
        if failure is not None or fault is not None:
            detail = failure if failure is not None else str(fault)
            self._view_status.setText(f"View: FAILED · {detail}")
            return
        board_widget = self._board_widget
        if isinstance(board_widget, QtRasterBoard):
            selector_fault = board_widget.selector_fault
            if selector_fault is not None:
                self._selector_switch.setChecked(False)
                self._roi_status.setText(
                    f"ROI selector disabled: {selector_fault}"
                )
        if live is not None and board_widget is not None and board_widget.has_front:
            front = board_widget.front_frame
            status = live.front_status
            if (
                front is None
                or status is None
                or status.sequence != front.sequence
            ):
                self._view_status.setText(
                    f"View: LIVE · {self._projection_text} · diagnostics pending"
                )
                return
            self._reconcile_visible_roi(status)
            coverage = status.raw_coverage
            scalar_coverage = status.scalar_coverage
            histogram_valid = status.histogram_valid_samples
            histogram_dropped = status.histogram_dropped_samples
            latest_valid = status.latest_scalar_valid
            source_missed = status.scalar_source_missed
            scalar_suffix = ""
            if scalar_coverage is not None:
                if histogram_valid is None or histogram_dropped is None:
                    scalar_counts = "scalar counts pending"
                else:
                    invalid_retained = (
                        scalar_coverage.written_cells - histogram_valid
                    )
                    unfilled = (
                        scalar_coverage.total_cells
                        - scalar_coverage.written_cells
                    )
                    scalar_counts = (
                        f"scalar valid/retained/capacity={histogram_valid}/"
                        f"{scalar_coverage.written_cells}/"
                        f"{scalar_coverage.total_cells} · "
                        f"hist dropped={histogram_dropped} "
                        f"(invalid={invalid_retained}, unfilled={unfilled})"
                    )
                latest_text = (
                    "pending"
                    if latest_valid is None
                    else ("valid" if latest_valid else "invalid")
                )
                source_text = "pending" if source_missed is None else str(source_missed)
                scalar_suffix = (
                    f" · {scalar_counts} · latest={latest_text} · "
                    f"scalar stream missed={scalar_coverage.missed_events} · "
                    "scalar current_gap="
                    f"{'yes' if scalar_coverage.current_gap else 'no'} · "
                    f"upstream before latest={source_text}"
                )
            suffix = (
                f"raw missed={coverage.missed_events} · "
                f"raw current_gap={'yes' if coverage.current_gap else 'no'}"
                + scalar_suffix
            )
            self._view_status.setText(
                f"View: LIVE · {self._projection_text} · {suffix}"
            )
        elif self._handle is not None and not self._handle.snapshot().state.terminal:
            self._view_status.setText(f"View: WAITING · {self._projection_text}")
        else:
            self._view_status.setText(f"View: STOPPED · {self._projection_text}")
        self._update_roi_controls()

    def _reconcile_visible_roi(self, status) -> None:
        binding = self._running_binding
        if binding is None:
            return
        if status.scalar_binding_fingerprint != binding.fingerprint:
            raise RuntimeError(
                "visible scalar front differs from the running ROI binding"
            )
        board_widget = self._board_widget
        if (
            isinstance(board_widget, QtRasterBoard)
            and self._visible_binding_fingerprint != binding.fingerprint
        ):
            board_widget.set_selector_applied_selection(binding.selection)
            self._visible_binding_fingerprint = binding.fingerprint
        if self._request != self._applied_request:
            self._applied_request = self._request
            self._draft_selection = None
            if isinstance(board_widget, QtRasterBoard):
                board_widget.set_selector_draft_selection(None)
            self._apply_phase = None
            self._roi_status.setText(
                "ROI: APPLIED · monitor generation "
                f"{self._run_owner.generation} · binding {binding.fingerprint[:12]}"
            )
            slot = self._slot
            if slot is None or slot.spec.scalar_dataset_edge is None:
                raise RuntimeError("visible ROI front has no scalar dataset edge")
            self._projection_text = _roi_scalar_views(
                slot.spec.scalar_dataset_edge.schema,
                binding,
            )[1]
            self._projection_status.setText(f"Display: {self._projection_text}")
            self._sync_live_pause()
        elif self._roi_status.text() == "ROI: fixed request" or self._roi_status.text().startswith(
            "ROI: waiting for first coherent front"
        ):
            self._roi_status.setText(
                "ROI: visible · monitor generation "
                f"{self._run_owner.generation} · binding {binding.fingerprint[:12]}"
            )

    def _refresh_diagnostics(self, snapshot: RunSnapshot | None) -> None:
        parts = [self._local_diagnostic] if self._local_diagnostic else []
        if snapshot is not None:
            if snapshot.primary_error and snapshot.state is not RunState.CANCELLED:
                parts.append(f"Error: {snapshot.primary_error}")
            parts.extend(f"Cleanup: {value}" for value in snapshot.cleanup_errors)
            if snapshot.recovery_instruction:
                parts.append(f"Recovery: {snapshot.recovery_instruction}")
        self._diagnostics.setText("\n".join(parts))

    def _record_local_failure(self, message: str) -> None:
        self._local_diagnostic = str(message)
        self._refresh_diagnostics(self._last_snapshot)

    def _update_start_button(self) -> None:
        self._start_button.setEnabled(
            not self._closing
            and self._prepared is not None
            and self._run_owner.owner_reaped
        )

    def _close_live(self) -> None:
        live = self._live
        if live is None:
            return
        live.close()
        self._live = None
        self._slot = None
        self._board = None

    def closeEvent(self, event) -> None:
        if self._allow_close:
            release_window(self)
            event.accept()
            return
        event.ignore()
        if not self._closing:
            self._closing = True
            self._prepared_apply = None
            self._pending_request = None
            self._apply_phase = None
            self._start_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            self._run_status.setText("Monitor: CLOSING")
            handle = self._handle
            if handle is not None and not handle.snapshot().state.terminal:
                handle.cancel("Workbench camera monitor window is closing")
        self._wake.request_owner_wake()

    def _maybe_finish_close(self) -> None:
        if not self._closing:
            return
        handle = self._handle
        if handle is not None:
            snapshot = handle.snapshot()
            if not snapshot.state.terminal:
                return
            if self._run_owner.begin_terminal_job("reap", handle.wait):
                return
        if self._run_owner.has_pending_owner_work:
            return
        try:
            self._close_live()
        except BaseException as error:
            self._record_local_failure(
                f"Monitor view close failed: {type(error).__name__}: {error}"
            )
            return
        if self._run_owner.has_pending_owner_work:
            return
        self._run_owner.shutdown()
        self._timer.stop()
        self._wake.detach()
        self._allow_close = True
        QtCore.QTimer.singleShot(0, self.close)


def open_camera_monitor_workbench(
    prepare,
    request: CameraMonitorRequest,
) -> CameraMonitorWorkbenchWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("camera monitor Workbench must open on the Qt GUI thread")
    set_fluent_scale(None)
    window = CameraMonitorWorkbenchWindow(prepare, request)
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = ["CameraMonitorWorkbenchWindow", "open_camera_monitor_workbench"]
