"""Qt composition for one free-running camera and typed ROI monitor."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time

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
from zlc_frontend.curve_display import (
    CurveDisplayState,
    curve_display_form_spec,
    curve_display_form_values,
    curve_display_from_form,
    curve_display_with_x_view,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.image_display import (
    ImageDisplayState,
    image_display_for_viewport,
    image_display_form_spec,
    image_display_form_values,
    image_display_from_form,
    image_viewport_for_display_state,
)
from zlc_frontend.image_raster import estimate_indexed8_raster_peak_nbytes
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.matplotlib_render import (
    DEFAULT_HISTOGRAM_BINS,
    estimate_live_panel_raster_peak_nbytes,
)
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentComboBox,
    FluentRevisionedFormEditor,
    FluentLabel,
    FluentPopup,
    FluentSwitch,
    FluentTabWidget,
    GREEN,
    GREY,
    ORANGE,
    QtOwnerWake,
    QtRasterBoard,
    RectangleGesture,
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    popup_gap,
    release_window,
    retain_window,
    screen_fit_window_size,
    scaled_px,
    set_fluent_scale,
)
from zlc_frontend.render import RenderSurface
from zlc_frontend.selector import (
    CurveInteractionIntent,
    CurveRangeGesture,
    CurveViewportCommit,
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportCommit,
    PanelInteractionOrigin,
)
from zlc_neutral_atom.monitor_application import (
    CameraMonitorRoiState,
    CameraMonitorRequest,
    CameraMonitorViewSpec,
    PreparedCameraMonitor,
)
from zlc_neutral_atom.processing.roi_monitor import RoiScalarBinding
from zlc_neutral_atom.runtime.control import (
    ControlAckStatus,
    ControlReceipt,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot, RunState
from zlc_workbench.live import LiveBoardController, LiveDatasetSlot
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


def _roi_scalar_view_specs(schema) -> tuple[ViewSpec, ViewSpec, ViewSpec]:
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
    return views


def _roi_scalar_views(schema, binding):
    views = _roi_scalar_view_specs(schema)
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    assert len(history) == 1
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


def _scalar_presentation_peak(
    schema,
) -> tuple[int, tuple[int, ...], int]:
    """Return the full three-panel live reserve for one possible scalar schema."""

    views = _roi_scalar_view_specs(schema)
    default_policy = FigureEvaluationPolicy()
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    if len(history) != 1 or history[0].size > default_policy.max_histogram_samples:
        raise ValueError(
            "ROI scalar history exceeds the live histogram sample limit "
            f"{default_policy.max_histogram_samples}"
        )
    evaluation_peaks = tuple(
        estimate_view_evaluation_peak_nbytes(schema, view) for view in views
    )
    total = 0
    curve_hold_extra = 0
    for view, evaluation_peak in zip(views, evaluation_peaks, strict=True):
        raster_peak = estimate_live_panel_raster_peak_nbytes(
            *_SCALAR_RASTER_SIZE,
            evaluated_data_upper_bound_bytes=evaluation_peak,
            histogram_bins=(
                DEFAULT_HISTOGRAM_BINS
                if view.intent is ViewIntent.HISTOGRAM
                else None
            ),
        )
        total += evaluation_peak + raster_peak
        if view.intent is ViewIntent.CURVE:
            held_peak = estimate_live_panel_raster_peak_nbytes(
                *_SCALAR_RASTER_SIZE,
                evaluated_data_upper_bound_bytes=evaluation_peak,
                extra_retained_fronts=1,
                extra_retained_evaluated_data_bytes=evaluation_peak,
            )
            curve_hold_extra = held_peak - raster_peak
    if curve_hold_extra <= 0:
        raise RuntimeError("scalar presentation has no curve hold reserve")
    return total, evaluation_peaks, curve_hold_extra


def _scalar_documents(
    schema,
    binding,
    dataset_id: DatasetId,
    *,
    identity: str,
) -> tuple[tuple[FigureDocument, ...], str]:
    """Build the three admitted documents for one applied scalar generation."""

    views, projection_text = _roi_scalar_views(schema, binding)
    descriptor = DatasetDescriptor(
        dataset_id,
        f"ROI {binding.reduction.value.lower()} scalar monitor",
        schema.fingerprint,
    )
    documents = tuple(
        FigureDocument(
            f"{panel_id}-{identity}",
            0,
            (descriptor,),
            (FigureLayer(panel_id, dataset_id, view),),
        )
        for panel_id, view in zip(_SCALAR_PANEL_IDS, views, strict=True)
    )
    return documents, projection_text


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


@dataclass(frozen=True, slots=True)
class _PendingRoiControl:
    receipt: ControlReceipt
    request: CameraMonitorRequest
    binding: RoiScalarBinding | None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, ControlReceipt):
            raise TypeError("ROI control receipt must be ControlReceipt")
        if not isinstance(self.request, CameraMonitorRequest):
            raise TypeError("ROI control request must be CameraMonitorRequest")
        if self.binding is not None and not isinstance(self.binding, RoiScalarBinding):
            raise TypeError("ROI control binding must be RoiScalarBinding or None")
        if (self.request.roi is None) != (self.binding is None):
            raise ValueError("ROI control request and binding activity differ")


def _prepare_monitor_view(
    command: PreparedCameraMonitor,
    generation: int,
) -> _PreparedMonitorView:
    """Finish source-lifetime display admission before the camera is armed."""

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
    image_base_raster_peak = estimate_indexed8_raster_peak_nbytes(
        y_axes[0].size,
        x_axes[0].size,
        value_itemsize=schema.cell_schema.dtype.itemsize,
        retained_fronts=1,
        retained_sample_fronts=1,
    )
    image_held_raster_peak = estimate_indexed8_raster_peak_nbytes(
        y_axes[0].size,
        x_axes[0].size,
        value_itemsize=schema.cell_schema.dtype.itemsize,
        retained_fronts=2,
        retained_sample_fronts=2,
    )
    image_hold_extra = image_held_raster_peak - image_base_raster_peak
    downstream_peak = image_evaluation_peak + image_base_raster_peak

    scalar_views: tuple[ViewSpec, ...] = ()
    projection_text = _RAW_PROJECTION_TEXT
    viewport = ImageViewportTransform(
        (y_axes[0], x_axes[0]),
        viewport_revision=0,
    )
    possible_scalar_reserves = tuple(
        _scalar_presentation_peak(possible_schema)
        for possible_schema in command.roi_control_schemas
    )
    if not possible_scalar_reserves:
        raise RuntimeError("camera monitor exposes no admitted ROI control schema")
    # Runtime ROI create/retarget must not acquire memory after the source was
    # armed.  Reserve the largest possible scalar board even for a raw-only
    # initial request; worker-affine renderer replacement is serialized, so
    # mutually exclusive scalar schemas contribute a maximum rather than a sum.
    downstream_peak += max(
        total for total, _peaks, _hold_extra in possible_scalar_reserves
    )
    # QtRasterBoard owns at most one pointer hold across the whole board.  The
    # reserve therefore admits the larger exact IMAGE/CURVE hold once, rather
    # than pretending both can coexist or undercounting retained curve arrays.
    downstream_peak += max(
        image_hold_extra,
        *(hold_extra for _total, _peaks, hold_extra in possible_scalar_reserves),
    )
    possible_scalar_evaluation_peaks = tuple(
        peak
        for _total, peaks, _hold_extra in possible_scalar_reserves
        for peak in peaks
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
        # Resolve once during preparation so binding the GUI overlay after the
        # source starts cannot discover a coordinate/selection mismatch.
        viewport.normalized_bounds_for_selection(roi_binding.selection)

    policy = FigureEvaluationPolicy(
        max_live_nbytes=max(
            (image_evaluation_peak, *possible_scalar_evaluation_peaks)
        )
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
        scalar_documents, checked_projection = _scalar_documents(
            scalar_schema,
            roi_binding,
            scalar_dataset_id,
            identity=str(generation),
        )
        if checked_projection != projection_text:
            raise RuntimeError("ROI scalar projection summary changed during preparation")
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
        self._applied_request = request
        self._running_binding = None
        self._applied_control_revision: int | None = None
        self._initial_roi_receipt: ControlReceipt | None = None
        self._initial_roi_receipt_loaded = False
        self._roi_controls: dict[int, _PendingRoiControl] = {}
        self._latest_roi_revision: int | None = None
        self._roi_terminal_notice: str | None = None
        self._roi_presentation_failure: str | None = None
        self._observed_roi_state_revision = 0
        self._visible_binding_fingerprint: str | None = None
        self._visible_projection_text: str | None = None
        self._draft_selection: Selection | None = None
        self._image_display = ImageDisplayState()
        self._curve_display = CurveDisplayState()
        self._image_viewport: ImageViewportTransform | None = None
        self._pending_image_interaction_origin: PanelInteractionOrigin | None = None
        self._pending_curve_interaction_origin: PanelInteractionOrigin | None = None
        self._curve_range_candidate: tuple[PanelInteractionOrigin, tuple[float, float]] | None = None
        self._slot: LiveDatasetSlot | None = None
        self._live: LiveBoardController | None = None
        self._board: BoardController | None = None
        self._last_snapshot: RunSnapshot | None = None
        self._prepare_inflight = False
        self._view_attach_inflight = False
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
        self._projection_status = FluentLabel(
            f"Display: no coherent front · target {self._projection_text}"
        )
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
        self._selector_switch = FluentSwitch("Selector", self)
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
        self._clear_roi_button = FluentButton("Clear ROI", self, color=ORANGE)
        self._clear_roi_button.setObjectName("clearRoiButton")
        self._clear_roi_button.setEnabled(False)
        self._setting_button = FluentButton("Setting…", self, color=GREY)
        self._setting_button.setObjectName("displaySettingButton")
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
        controls.addWidget(self._clear_roi_button)
        controls.addWidget(self._setting_button)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)

        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName("cameraMonitorTabs")
        self._live_page = QtWidgets.QWidget(self._tabs)
        self._live_page.setObjectName("cameraMonitorLiveTab")
        live_layout = QtWidgets.QVBoxLayout(self._live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.addWidget(self._board_container, 1)
        self._edit_display_tabs = FluentTabWidget(self._tabs)
        self._edit_display_tabs.setObjectName("displayEditTabs")
        self._edit_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._edit_display_tabs,
        )
        self._edit_image_display.setObjectName("imageDisplayEditEditor")
        self._edit_curve_display = FluentRevisionedFormEditor(
            curve_display_form_spec(),
            "curve display",
            runtime_placeholder_fields=("y_min", "y_max"),
            parent=self._edit_display_tabs,
        )
        self._edit_curve_display.setObjectName("curveDisplayEditEditor")
        self._edit_display_tabs.addTab(self._edit_image_display, "Image")
        self._edit_display_tabs.addTab(self._edit_curve_display, "Curve")
        self._tabs.addTab(self._live_page, "Live")
        self._tabs.addTab(self._edit_display_tabs, "Edit")

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("displaySettingsPopup")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        settings_layout.setContentsMargins(*(scaled_px(10, minimum=8),) * 4)
        self._setting_display_tabs = FluentTabWidget(self._settings_popup)
        self._setting_display_tabs.setObjectName("displaySettingTabs")
        self._setting_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._setting_display_tabs,
        )
        self._setting_image_display.setObjectName("imageDisplaySettingEditor")
        self._setting_curve_display = FluentRevisionedFormEditor(
            curve_display_form_spec(),
            "curve display",
            runtime_placeholder_fields=("y_min", "y_max"),
            parent=self._setting_display_tabs,
        )
        self._setting_curve_display.setObjectName("curveDisplaySettingEditor")
        self._setting_display_tabs.addTab(self._setting_image_display, "Image")
        self._setting_display_tabs.addTab(self._setting_curve_display, "Curve")
        settings_layout.addWidget(self._setting_display_tabs)
        self._settings_dismissed_at = float("-inf")
        self._settings_popup._on_hidden = self._record_settings_dismissed
        self._sync_image_display_editors()
        self._sync_curve_display_editors()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._run_status)
        layout.addWidget(self._view_status)
        layout.addWidget(self._projection_status)
        layout.addWidget(self._roi_status)
        layout.addWidget(self._tabs, 1)
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
        self._reducer_combo.currentIndexChanged.connect(self._reducer_changed)
        self._clear_roi_button.clicked.connect(self._clear_roi)
        self._setting_button.clicked.connect(self._open_display_settings)
        self._edit_image_display.applyRequested.connect(
            lambda revision, values: self._apply_image_display_form(
                self._edit_image_display,
                revision,
                values,
            )
        )
        self._setting_image_display.applyRequested.connect(
            lambda revision, values: self._apply_image_display_form(
                self._setting_image_display,
                revision,
                values,
            )
        )
        self._edit_curve_display.applyRequested.connect(
            lambda revision, values: self._apply_curve_display_form(
                self._edit_curve_display,
                revision,
                values,
            )
        )
        self._setting_curve_display.applyRequested.connect(
            lambda revision, values: self._apply_curve_display_form(
                self._setting_curve_display,
                revision,
                values,
            )
        )
        self._edit_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._edit_image_display)
        )
        self._setting_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._setting_image_display)
        )
        self._edit_curve_display.cancelRequested.connect(
            lambda: self._reload_curve_display_editor(self._edit_curve_display)
        )
        self._setting_curve_display.cancelRequested.connect(
            lambda: self._reload_curve_display_editor(self._setting_curve_display)
        )
        self._update_display_controls()
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

    def _set_selector_enabled(self, enabled: bool) -> None:
        board = self._board_widget
        setter = getattr(board, "set_selectors_enabled", None)
        if callable(setter):
            setter(bool(enabled))

    def _open_display_settings(self) -> None:
        if not self._setting_button.isEnabled():
            return
        popup = self._settings_popup
        if popup.isVisible():
            popup.hide()
            return
        if time.monotonic() - self._settings_dismissed_at < 0.25:
            return
        self._reload_image_display_editor(self._setting_image_display)
        self._reload_curve_display_editor(self._setting_curve_display)
        popup.adjustSize()
        form_hint = self._setting_display_tabs.sizeHint()
        editor_hint = self._setting_display_tabs.sizeHint()
        button_top_left = self._setting_button.mapToGlobal(QtCore.QPoint(0, 0))
        button_bottom_right = self._setting_button.mapToGlobal(
            QtCore.QPoint(
                max(0, self._setting_button.width() - 1),
                max(0, self._setting_button.height() - 1),
            )
        )
        button_center = QtCore.QPoint(
            (button_top_left.x() + button_bottom_right.x()) // 2,
            (button_top_left.y() + button_bottom_right.y()) // 2,
        )
        screen = QtWidgets.QApplication.screenAt(button_center)
        if screen is None and hasattr(self, "screen"):
            screen = self.screen()
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        available = None if screen is None else screen.availableGeometry()
        desired_width = max(
            scaled_px(360, minimum=320),
            form_hint.width() + scaled_px(28, minimum=20),
            editor_hint.width(),
        )
        desired_height = max(
            scaled_px(300, minimum=260),
            form_hint.height() + scaled_px(120, minimum=96),
            editor_hint.height(),
        )
        if available is not None:
            desired_width = min(
                desired_width,
                max(1, int(available.width() * 0.55)),
            )
            desired_height = min(desired_height, max(1, int(available.height() * 0.90)))
            gap = popup_gap()
            below_top = button_bottom_right.y() + 1 + gap
            above_bottom = button_top_left.y() - gap - 1
            below_space = max(0, available.bottom() - below_top + 1)
            above_space = max(0, above_bottom - available.top() + 1)
            if desired_height <= below_space:
                top = below_top
            elif desired_height <= above_space:
                top = above_bottom - desired_height + 1
            elif above_space >= below_space:
                desired_height = max(1, min(desired_height, above_space))
                top = above_bottom - desired_height + 1
            else:
                desired_height = max(1, min(desired_height, below_space))
                top = below_top
        else:
            top = button_bottom_right.y() + 1 + popup_gap()
        popup.resize(desired_width, desired_height)
        left = button_bottom_right.x() - popup.width() + 1
        if available is not None:
            left = max(
                available.left(),
                min(left, available.right() - popup.width() + 1),
            )
            top = max(
                available.top(),
                min(top, available.bottom() - popup.height() + 1),
            )
        popup.move(left, top)
        popup.show()
        popup.raise_()

    def _record_settings_dismissed(self) -> None:
        self._settings_dismissed_at = time.monotonic()

    def _update_display_controls(self) -> None:
        live_healthy = self._live is None or self._live.fault is None
        image_enabled = (
            not self._closing
            and self._image_viewport is not None
            and not self._view_attach_inflight
            and live_healthy
            and (
                self._live is None
                or self._visible_image_matches_display_revision()
            )
        )
        curve_enabled = (
            not self._closing
            and not self._view_attach_inflight
            and live_healthy
            and self._visible_curve_matches_current_state()
        )
        self._setting_button.setEnabled(image_enabled or curve_enabled)
        self._edit_image_display.setEnabled(image_enabled)
        self._setting_image_display.setEnabled(image_enabled)
        self._edit_curve_display.setEnabled(curve_enabled)
        self._setting_curve_display.setEnabled(curve_enabled)

    def _visible_image_color_limits(self) -> tuple[float, float] | None:
        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            return None
        payload = board.visible_image_payload()
        return None if payload is None else payload.color_limits

    def _visible_image_matches_display_revision(self) -> bool:
        """Return whether interaction would target the authored display state.

        A display commit is accepted by the live controller before its worker
        can publish the replacement front.  During that bounded interval the
        old pixels remain visible, but must not be allowed to author another
        gesture against the newer Workbench revision.
        """

        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            return False
        origin = board.visible_image_origin()
        return (
            origin is not None
            and origin.panel_id == _IMAGE_PANEL_ID
            and origin.presentation.panel_revision == self._image_display.revision
        )

    def _visible_curve_y_limits(self) -> tuple[float, float] | None:
        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            return None
        if self._live is not None and not self._visible_curve_matches_current_state():
            return None
        payload = board.visible_curve_payload()
        return None if payload is None else payload.viewport.y_limits

    def _visible_curve_matches_current_state(self) -> bool:
        """Require painted curve presentation and ROI provenance to be current."""

        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            return False
        origin = board.visible_curve_origin()
        front = board.front_frame
        live = self._live
        status = None if live is None else live.front_status
        binding = self._running_binding
        return (
            origin is not None
            and origin.panel_id == _CURVE_PANEL_ID
            and origin.presentation.panel_revision == self._curve_display.revision
            and front is not None
            and status is not None
            and status.sequence == front.sequence
            and binding is not None
            and status.scalar_binding_fingerprint == binding.fingerprint
            and self._applied_control_revision is not None
            and status.scalar_control_revision == self._applied_control_revision
        )

    @staticmethod
    def _runtime_range_placeholders(
        value: tuple[float, float] | None,
        low_key: str,
        high_key: str,
    ) -> dict[str, str] | None:
        if value is None:
            return None
        return {
            low_key: f"{value[0]:.12g}",
            high_key: f"{value[1]:.12g}",
        }

    def _reload_image_display_editor(
        self,
        editor: FluentRevisionedFormEditor,
    ) -> None:
        editors = (
            self._edit_image_display,
            self._setting_image_display,
        )
        self._require_display_editor(editor, editors, "image")
        editor.load(
            revision=self._image_display.revision,
            semantic_identity=self._image_display,
            values=image_display_form_values(self._image_display),
            runtime_placeholders=self._runtime_range_placeholders(
                self._visible_image_color_limits(),
                "color_min",
                "color_max",
            ),
        )

    def _sync_image_display_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        limits = self._visible_image_color_limits()
        values = image_display_form_values(self._image_display)
        placeholders = self._runtime_range_placeholders(
            limits,
            "color_min",
            "color_max",
        )
        self._sync_display_editors(
            (self._edit_image_display, self._setting_image_display),
            revision=self._image_display.revision,
            semantic_identity=self._image_display,
            values=values,
            runtime_placeholders=placeholders,
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _reload_curve_display_editor(
        self,
        editor: FluentRevisionedFormEditor,
    ) -> None:
        editors = (
            self._edit_curve_display,
            self._setting_curve_display,
        )
        self._require_display_editor(editor, editors, "curve")
        editor.load(
            revision=self._curve_display.revision,
            semantic_identity=self._curve_display,
            values=curve_display_form_values(self._curve_display),
            runtime_placeholders=self._runtime_range_placeholders(
                self._visible_curve_y_limits(),
                "y_min",
                "y_max",
            ),
        )

    def _sync_curve_display_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        values = curve_display_form_values(self._curve_display)
        placeholders = self._runtime_range_placeholders(
            self._visible_curve_y_limits(),
            "y_min",
            "y_max",
        )
        self._sync_display_editors(
            (self._edit_curve_display, self._setting_curve_display),
            revision=self._curve_display.revision,
            semantic_identity=self._curve_display,
            values=values,
            runtime_placeholders=placeholders,
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    @staticmethod
    def _require_display_editor(
        editor: FluentRevisionedFormEditor,
        editors: tuple[FluentRevisionedFormEditor, ...],
        kind: str,
    ) -> None:
        if editor not in editors:
            raise ValueError(f"{kind} display editor does not belong to this window")

    @staticmethod
    def _sync_display_editors(
        editors: tuple[FluentRevisionedFormEditor, ...],
        *,
        revision: int,
        semantic_identity: object,
        values: dict[str, object],
        runtime_placeholders: dict[str, str] | None,
        accepted_editor: FluentRevisionedFormEditor | None,
        accepted_base_revision: int | None,
    ) -> None:
        if accepted_editor is not None and accepted_editor not in editors:
            raise ValueError("accepted display editor is not one of the synchronized surfaces")
        for editor in editors:
            if editor is accepted_editor:
                if accepted_base_revision is None:
                    raise RuntimeError("accepted display editor has no base revision")
                editor.accept_commit(
                    base_revision=accepted_base_revision,
                    revision=revision,
                    semantic_identity=semantic_identity,
                    values=values,
                    runtime_placeholders=runtime_placeholders,
                )
            else:
                editor.load(
                    revision=revision,
                    semantic_identity=semantic_identity,
                    values=values,
                    runtime_placeholders=runtime_placeholders,
                )

    def _apply_image_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        if editor not in (
            self._edit_image_display,
            self._setting_image_display,
        ):
            raise ValueError("image display editor does not belong to this window")
        try:
            if (
                self._closing
                or self._view_attach_inflight
            ):
                raise RuntimeError(
                    "image display cannot change while the live view is attaching"
                )
            if base_revision != self._image_display.revision:
                raise RuntimeError(
                    f"image display edit base r{base_revision} is stale; "
                    f"current revision is r{self._image_display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("image display form must emit one exact mapping")
            candidate = image_display_from_form(
                self._image_display,
                values,
                current_color_limits=self._visible_image_color_limits(),
            )
            viewport = self._viewport_for_image_display(candidate)
            self._commit_image_display(
                candidate,
                viewport,
                accepted_editor=editor,
                accepted_base_revision=base_revision,
            )
        except Exception as error:
            self._record_local_failure(
                f"Image display edit rejected: {type(error).__name__}: {error}"
            )

    def _apply_curve_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        if editor not in (
            self._edit_curve_display,
            self._setting_curve_display,
        ):
            raise ValueError("curve display editor does not belong to this window")
        try:
            if self._closing or self._view_attach_inflight:
                raise RuntimeError(
                    "curve display cannot change while the live view is attaching"
                )
            if self._live is not None and not self._visible_curve_matches_current_state():
                raise RuntimeError(
                    "curve display cannot change until the current ROI front is visible"
                )
            if base_revision != self._curve_display.revision:
                raise RuntimeError(
                    f"curve display edit base r{base_revision} is stale; "
                    f"current revision is r{self._curve_display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("curve display form must emit one exact mapping")
            candidate = curve_display_from_form(
                self._curve_display,
                values,
                current_y_limits=self._visible_curve_y_limits(),
            )
            self._commit_curve_display(
                candidate,
                accepted_editor=editor,
                accepted_base_revision=base_revision,
            )
        except Exception as error:
            self._record_local_failure(
                f"Curve display edit rejected: {type(error).__name__}: {error}"
            )

    def _viewport_for_image_display(
        self,
        state: ImageDisplayState,
    ) -> ImageViewportTransform:
        reference = self._image_viewport
        if reference is None and self._prepared is not None:
            reference = self._prepared.viewport
        if reference is None:
            raise RuntimeError("image display axes are not prepared")
        return image_viewport_for_display_state(state, reference)

    def _accept_image_interaction(
        self,
        command: ImageInteractionCommit,
    ) -> None:
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("image interaction callback received an unknown command")
        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            raise RuntimeError("image interaction has no raster board")
        origin = command.origin
        if origin.panel_id != _IMAGE_PANEL_ID:
            raise RuntimeError("image interaction belongs to another panel")
        if board.visible_image_origin() != origin:
            raise RuntimeError("image interaction origin is no longer painted")
        if origin.presentation.panel_revision != self._image_display.revision:
            raise RuntimeError("image interaction origin is stale")
        current_viewport = self._image_viewport
        if current_viewport is None:
            raise RuntimeError("image interaction has no committed viewport")

        if isinstance(command, ImageViewportCommit):
            if command.viewport.viewport_revision != self._image_display.revision + 1:
                raise RuntimeError("image viewport interaction must advance once")
            candidate = image_display_for_viewport(
                self._image_display,
                command.viewport,
            )
            viewport = command.viewport
        else:
            candidate = replace(
                self._image_display,
                revision=self._image_display.revision + 1,
                relim_mode=RelimMode.FIXED,
                fixed_color_limits=command.color_limits,
            )
            viewport = image_viewport_for_display_state(
                candidate,
                current_viewport,
            )
        self._commit_image_display(
            candidate,
            viewport,
            interaction_origin=origin,
        )

    def _commit_image_display(
        self,
        state: ImageDisplayState,
        viewport: ImageViewportTransform,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
        interaction_origin: PanelInteractionOrigin | None = None,
    ) -> None:
        current = self._image_display
        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if viewport.viewport_revision != state.revision:
            raise ValueError("image display and viewport revisions differ")
        changed = state != current
        if changed and state.revision != current.revision + 1:
            raise ValueError("image display commit must advance exactly once")
        if not changed and state.revision != current.revision:
            raise ValueError("image display no-op changed revision")
        if accepted_editor is not None:
            if accepted_base_revision != current.revision:
                raise ValueError("accepted editor base differs from current revision")
            if accepted_editor.base_revision != accepted_base_revision:
                raise ValueError("accepted editor no longer owns its emitted base")
        if interaction_origin is not None and not changed:
            raise ValueError("image interaction cannot commit a semantic no-op")

        live = self._live
        if changed and live is not None:
            # This is the only runtime mutation.  It replaces presentation
            # identity while preserving the source Run, stream, slot and board
            # topology.  Install it before publishing the authored state.
            live.reconfigure_image_display(state, viewport)

        if changed:
            self._image_display = state
            self._image_viewport = viewport
            if live is None and self._prepared is not None:
                self._prepared = replace(self._prepared, viewport=viewport)
            self._pending_image_interaction_origin = interaction_origin
        self._sync_image_display_editors(
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )
        # Keep the user's selector switch intent checked, but disarm the board
        # until a front carrying this exact revision is actually painted.
        self._update_roi_controls()
        self._update_display_controls()
        if (
            changed
            and live is None
            and self._prepared is None
            and not self._prepare_inflight
        ):
            # A newly discovered camera schema can reject retained authored
            # coordinate pins while stopped.  Clearing/changing those pins is
            # the recovery action: retry the existing one-shot preparation
            # only after the new display state is authoritative, so the fresh
            # axes are mapped against that state rather than the rejected one.
            self._run_status.setText("Monitor: PREPARING")
            self._local_diagnostic = ""
            self._refresh_diagnostics(self._last_snapshot)
            self._submit_prepare()

    def _accept_curve_interaction(
        self,
        command: CurveInteractionIntent,
    ) -> None:
        if not isinstance(command, (CurveViewportCommit, CurveRangeGesture)):
            raise TypeError("curve interaction callback received an unknown command")
        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            raise RuntimeError("curve interaction has no raster board")
        origin = command.origin
        if origin.panel_id != _CURVE_PANEL_ID:
            raise RuntimeError("curve interaction belongs to another panel")
        if not self._visible_curve_matches_current_state():
            raise RuntimeError("curve interaction ROI provenance is stale")
        if board.visible_curve_origin() != origin:
            raise RuntimeError("curve interaction origin is no longer painted")
        if origin.presentation.panel_revision != self._curve_display.revision:
            raise RuntimeError("curve interaction origin is stale")
        if isinstance(command, CurveRangeGesture):
            self._set_curve_range_candidate(origin, command.x_span)
            self._roi_status.setText(
                "Curve: DISPLAY ONLY x span "
                f"{command.x_span[0]:.6g}..{command.x_span[1]:.6g}"
            )
            return
        if command.viewport.display_revision != self._curve_display.revision + 1:
            raise RuntimeError("curve viewport interaction must advance once")
        candidate = curve_display_with_x_view(
            self._curve_display,
            command.viewport.x_limits,
        )
        self._commit_curve_display(
            candidate,
            interaction_origin=origin,
        )

    def _commit_curve_display(
        self,
        state: CurveDisplayState,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
        interaction_origin: PanelInteractionOrigin | None = None,
    ) -> None:
        current = self._curve_display
        if not isinstance(state, CurveDisplayState):
            raise TypeError("state must be CurveDisplayState")
        changed = state != current
        if changed and state.revision != current.revision + 1:
            raise ValueError("curve display commit must advance exactly once")
        if not changed and state.revision != current.revision:
            raise ValueError("curve display no-op changed revision")
        if accepted_editor is not None:
            if accepted_base_revision != current.revision:
                raise ValueError("accepted curve editor base differs from current revision")
            if accepted_editor.base_revision != accepted_base_revision:
                raise ValueError("accepted curve editor no longer owns its emitted base")
        if interaction_origin is not None and not changed:
            raise ValueError("curve interaction cannot commit a semantic no-op")

        live = self._live
        if changed and live is not None:
            live.reconfigure_curve_display(state)
        if changed:
            self._curve_display = state
            self._pending_curve_interaction_origin = interaction_origin
        self._sync_curve_display_editors(
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )
        self._update_roi_controls()
        self._update_display_controls()

    def _discard_pending_image_interaction(self) -> None:
        origin = self._pending_image_interaction_origin
        if origin is None:
            return
        board = self._board_widget
        if isinstance(board, QtRasterBoard):
            board.discard_pending_image_interaction(origin)
        self._pending_image_interaction_origin = None

    def _discard_pending_curve_interaction(self) -> None:
        origin = self._pending_curve_interaction_origin
        if origin is None:
            return
        board = self._board_widget
        if isinstance(board, QtRasterBoard):
            board.discard_pending_curve_interaction(origin)
        self._pending_curve_interaction_origin = None

    def _set_curve_range_candidate(
        self,
        origin: PanelInteractionOrigin,
        x_span: tuple[float, float],
    ) -> None:
        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            raise RuntimeError("curve range candidate has no raster board")
        if board.visible_curve_origin() != origin:
            raise RuntimeError("curve range candidate origin is no longer painted")
        self._curve_range_candidate = (origin, x_span)
        board.set_curve_range_candidate(x_span)

    def _clear_curve_range_candidate(self) -> None:
        self._curve_range_candidate = None
        board = self._board_widget
        if isinstance(board, QtRasterBoard):
            board.set_curve_range_candidate(None)

    def _accept_rectangle_gesture(self, gesture: RectangleGesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("selector callback requires RectangleGesture")
        board = self._board_widget
        if not isinstance(board, QtRasterBoard):
            raise RuntimeError("ROI selector has no active raster viewport")
        if gesture.panel_id != _IMAGE_PANEL_ID:
            raise RuntimeError("ROI gesture belongs to another panel")
        selection = board.selection_for_rectangle_gesture(gesture)
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
                f"ROI: DRAFT · {coordinates} · submitting"
            )
        self._update_roi_controls()
        if selection is not None:
            self._submit_roi_control(selection)

    def _selected_reducer(self) -> ReductionMethod:
        value = self._reducer_combo.currentData()
        if not isinstance(value, ReductionMethod):
            raise TypeError("ROI reducer control contains an invalid value")
        return value

    def _update_roi_controls(self, *_args) -> None:
        busy = self._closing or self._manual_stop_requested
        slot = self._slot
        live = self._live
        board = self._board
        view_healthy = (
            slot is not None
            and slot.failure is None
            and live is not None
            and live.fault is None
            and board is not None
            and board.fault is None
            and self._roi_presentation_failure is None
        )
        image_interaction_ready = False
        curve_interaction_ready = False
        if isinstance(self._board_widget, QtRasterBoard):
            image_interaction_ready = (
                self._board_widget.selector_fault is None
                and self._visible_image_matches_display_revision()
            )
            curve_interaction_ready = (
                self._board_widget.curve_selector_fault is None
                and self._visible_curve_matches_current_state()
            )
            self._board_widget.set_interaction_readiness(
                image=image_interaction_ready,
                curve=curve_interaction_ready,
            )
        selector_healthy = image_interaction_ready or curve_interaction_ready
        active = (
            self._handle is not None
            and not self._handle.snapshot().state.terminal
            and self._board_widget is not None
            and self._board_widget.has_front
            and view_healthy
        )
        self._selector_switch.setEnabled(not busy and active and selector_healthy)
        self._reducer_combo.setEnabled(not busy and active)
        self._clear_roi_button.setEnabled(
            not busy
            and active
            and (
                self._applied_request.roi is not None
                or any(
                    not pending.receipt.snapshot().terminal
                    and pending.binding is not None
                    for pending in self._roi_controls.values()
                )
            )
        )
        self._set_selector_enabled(
            self._selector_switch.isChecked()
            and not busy
            and active
            and selector_healthy
        )

    def _reducer_changed(self, *_args) -> None:
        self._update_roi_controls()
        if self._closing or self._manual_stop_requested:
            return
        selection = self._draft_selection or self._applied_request.roi
        if selection is not None:
            self._submit_roi_control(selection)

    def _clear_roi(self) -> None:
        if self._closing or self._manual_stop_requested:
            return
        self._submit_roi_control(None)

    def _submit_roi_control(self, selection: Selection | None) -> None:
        if self._closing or self._manual_stop_requested:
            return
        handle = self._handle
        if handle is None or handle.snapshot().state.terminal:
            return
        slot = self._slot
        if slot is None:
            return
        reduction = self._selected_reducer()
        candidate_request = replace(
            self._applied_request,
            roi=selection,
            roi_reduction=reduction,
        )
        if candidate_request == self._applied_request and not self._roi_controls:
            return
        try:
            binding = slot.prepare_camera_roi_control(
                selection,
                reduction,
                self._applied_request.roi_validity_policy,
            )
            receipt = slot.submit_camera_roi_control(binding)
        except BaseException as error:
            self._roi_terminal_notice = (
                f"ROI: REJECTED before acceptance · {type(error).__name__}: {error}"
            )
            self._roi_status.setText(self._roi_terminal_notice)
            self._record_local_failure(
                f"ROI control preflight failed: {type(error).__name__}: {error}"
            )
            self._update_roi_controls()
            return
        revision = receipt.snapshot().revision
        self._roi_controls[revision] = _PendingRoiControl(
            receipt,
            candidate_request,
            binding,
        )
        self._latest_roi_revision = revision
        self._roi_terminal_notice = None
        action = "delete" if binding is None else "retarget"
        self._roi_status.setText(f"ROI: ACCEPTED r{revision} · {action} pending")
        self._update_roi_controls()

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
        self._pending_image_interaction_origin = None
        self._pending_curve_interaction_origin = None
        self._clear_curve_range_candidate()
        self._visible_binding_fingerprint = None
        self._visible_projection_text = None

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
            self._record_local_failure(
                "Prepared monitor view is stale relative to the next Run generation"
            )
            self._submit_prepare()
            return
        roi_binding = command.roi_binding
        self._close_live()
        self._running_binding = roi_binding
        self._applied_control_revision = None
        self._roi_terminal_notice = None
        self._roi_presentation_failure = None
        self._observed_roi_state_revision = 0
        self._draft_selection = (
            None if roi_binding is None else roi_binding.selection
        )
        generation = self._run_owner.begin_generation()
        try:
            selector_board = self._board_widget
            selector_viewport = prepared_view.viewport
            image_display = self._image_display
            if not isinstance(selector_board, QtRasterBoard):
                raise RuntimeError("ROI selector host changed after preparation")
            if selector_viewport.viewport_revision != image_display.revision:
                raise RuntimeError("selector viewport and image display revisions differ")
            self._selector_switch.setChecked(False)
            selector_board.bind_rectangle_selector(
                _IMAGE_PANEL_ID,
                selector_viewport,
                self._accept_rectangle_gesture,
                enabled=False,
                interaction_callback=self._accept_image_interaction,
            )
            if prepared_view.scalar_documents:
                selector_board.bind_curve_interaction(
                    _CURVE_PANEL_ID,
                    self._accept_curve_interaction,
                    enabled=False,
                )
            else:
                selector_board.unbind_curve_interaction()
            if not selector_board.has_front:
                selector_board.set_selector_applied_selection(None)
            selector_board.set_selector_draft_selection(self._draft_selection)
            self._image_viewport = selector_viewport
        except BaseException as error:
            self._run_owner.mark_owner_reaped()
            self._running_binding = None
            self._prepared = None
            self._record_local_failure(
                f"ROI selector setup failed: {type(error).__name__}: {error}"
            )
            self._submit_prepare()
            return
        self._prepared = None
        self._view_attach_inflight = True
        self._last_snapshot = None
        self._manual_stop_requested = False
        self._local_diagnostic = ""
        self._projection_text = prepared_view.projection_text
        self._refresh_diagnostics(None)
        self._run_status.setText("Monitor: STARTING")
        self._view_status.setText(f"View: WAITING · {self._projection_text}")
        self._show_projection_target()
        if roi_binding is not None:
            self._roi_status.setText(
                "ROI: waiting for first coherent front · binding "
                f"{roi_binding.fingerprint[:12]}"
            )
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._update_roi_controls()
        self._update_display_controls()

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
                (
                    slot,
                    image_document,
                    scalar_documents,
                    image_display,
                    selector_viewport,
                    attached,
                    response,
                ),
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
            self._drain_roi_controls()
            live = self._live
            board = self._board
            view_faulted = False
            if live is not None:
                view_faulted = live.reconcile_faults()
            if (
                board is not None
                and board.fault is None
                and not view_faulted
            ):
                board.present_pending()
            if live is not None and not view_faulted:
                # Present the previous completed raster before admitting the
                # next candidate, so front_status remains sequence-gated.
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
            self._update_display_controls()
            self._maybe_finish_close()
        except BaseException as error:
            message = f"Workbench owner failed: {type(error).__name__}: {error}"
            # Presentation/control faults are downstream of the camera source.
            # They remain visible and fail-closed locally, but may not revoke the
            # source Run or its raw stream as an error-propagation shortcut.
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
            if kind == "prepare":
                if generation == self._run_owner.generation:
                    self._prepare_inflight = False
                try:
                    prepared = future.result()
                    if not isinstance(prepared, _PreparedMonitorView):
                        raise TypeError("camera prepare returned an invalid view product")
                    if prepared.generation != generation + 1:
                        raise RuntimeError("prepared camera view generation changed")
                    prepared = replace(
                        prepared,
                        viewport=image_viewport_for_display_state(
                            self._image_display,
                            prepared.viewport,
                        ),
                    )
                except BaseException as error:
                    if not self._closing and generation == self._run_owner.generation:
                        self._run_status.setText("Monitor: NOT READY")
                        self._record_local_failure(
                            f"Preparation failed: {type(error).__name__}: {error}"
                        )
                    continue
                if not self._closing and generation == self._run_owner.generation:
                    self._prepared = prepared
                    self._image_viewport = prepared.viewport
                    scalar_schema = prepared.command.scalar_view_schema
                    binding = prepared.command.roi_binding
                    self._configure_board_widget(scalar=scalar_schema is not None)
                    if scalar_schema is not None and binding is None:
                        raise RuntimeError("prepared ROI scalar schema has no binding")
                    self._projection_text = prepared.projection_text
                    self._show_projection_target()
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
                    self._sync_image_display_editors()
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
                        self._submit_prepare()
                continue

    def _drain_roi_controls(self) -> None:
        handle = self._handle
        if (
            self._closing
            or self._manual_stop_requested
            or (handle is not None and handle.snapshot().state.terminal)
        ):
            # Terminal source cleanup intentionally publishes a clean raw state.
            # It cannot negate an APPLIED create/retarget receipt that the GUI
            # owner has not folded yet.  Detach folds those receipts first.
            return
        self._drain_initial_roi_control()
        changed = False
        for revision in tuple(sorted(self._roi_controls)):
            pending = self._roi_controls[revision]
            ack = pending.receipt.snapshot()
            if not ack.terminal:
                continue
            changed = True
            if self._closing:
                del self._roi_controls[revision]
                continue
            state = None
            try:
                if ack.status is ControlAckStatus.APPLIED:
                    slot = self._slot
                    if slot is None:
                        raise RuntimeError(
                            "applied ROI control has no live source slot"
                        )
                    state = slot.current_camera_roi_state()
                    if state.control_revision < revision:
                        raise RuntimeError(
                            "ROI state revision trails its APPLIED receipt"
                        )
                    if state.control_revision == revision:
                        self._promote_roi_state(state, pending)
                elif ack.status is ControlAckStatus.REJECTED:
                    if revision == self._latest_roi_revision:
                        self._roi_terminal_notice = (
                            f"ROI: REJECTED r{revision} · {ack.reason}"
                        )
                        self._roi_status.setText(self._roi_terminal_notice)
                elif ack.status is ControlAckStatus.SUPERSEDED:
                    if revision == self._latest_roi_revision:
                        self._roi_terminal_notice = (
                            f"ROI: SUPERSEDED r{revision} by r{ack.superseded_by}"
                        )
                        self._roi_status.setText(self._roi_terminal_notice)
                elif ack.status is ControlAckStatus.TERMINATED:
                    if revision == self._latest_roi_revision:
                        self._roi_terminal_notice = (
                            f"ROI: TERMINATED r{revision} · {ack.reason}"
                        )
                        self._roi_status.setText(self._roi_terminal_notice)
                else:  # pragma: no cover - enum exhaustiveness guard
                    raise RuntimeError(
                        f"unexpected terminal ROI ack {ack.status.value}"
                    )
            except BaseException as error:
                request = self._applied_request
                draft = self._draft_selection
                if ack.status is ControlAckStatus.APPLIED:
                    if state is None:
                        request = pending.request
                    elif state.binding is None:
                        request = replace(pending.request, roi=None)
                        draft = pending.request.roi
                    else:
                        request = self._request_for_roi_state(state, pending.request)
                        if draft == request.roi:
                            draft = None
                self._record_roi_presentation_failure(
                    f"ROI APPLIED r{revision} could not be presented",
                    error,
                    request=request,
                    state=state,
                    draft_selection=draft,
                )
            finally:
                # A terminal receipt is folded exactly once.  Presentation
                # failure freezes the old front; it must never retry this
                # receipt and inflate layout/configuration generations.
                del self._roi_controls[revision]
        if changed:
            self._latest_roi_revision = (
                max(self._roi_controls) if self._roi_controls else None
            )
            self._update_roi_controls()
        self._drain_roi_state_notice()

    def _drain_roi_state_notice(self) -> None:
        """Converge a spontaneous downstream failure without touching the source."""

        if self._closing or self._manual_stop_requested or self._roi_presentation_failure:
            return
        slot = self._slot
        handle = self._handle
        if slot is None or handle is None or handle.snapshot().state.terminal:
            return
        try:
            state = slot.current_camera_roi_state()
        except RuntimeError:
            return
        if state.state_revision <= self._observed_roi_state_revision:
            return
        self._observed_roi_state_revision = state.state_revision
        if state.failure_reason is None:
            return
        previous_selection = self._applied_request.roi
        request = self._request_for_roi_state(state, self._applied_request)
        try:
            board_widget = self._adopt_roi_presentation(state, stage=True)
        except BaseException as error:
            self._record_roi_presentation_failure(
                "ROI failure presentation could not converge",
                error,
                request=request,
                state=state,
                draft_selection=previous_selection,
                downstream_reason=state.failure_reason,
            )
            return
        self._applied_request = request
        self._request = request
        self._draft_selection = previous_selection
        board_widget.set_selector_draft_selection(previous_selection)
        self._roi_terminal_notice = (
            f"ROI: downstream failed at state {state.state_revision} · "
            f"{state.failure_reason} · raw source continues"
        )
        self._roi_status.setText(self._roi_terminal_notice)

    def _drain_initial_roi_control(self) -> None:
        if not self._initial_roi_receipt_loaded:
            slot = self._slot
            handle = self._handle
            if (
                slot is None
                or handle is None
                or handle.snapshot().state.terminal
                or not slot.dataset_bound
            ):
                return
            self._initial_roi_receipt = slot.initial_camera_roi_receipt
            self._initial_roi_receipt_loaded = True
        receipt = self._initial_roi_receipt
        if receipt is None:
            return
        ack = receipt.snapshot()
        if not ack.terminal:
            return
        self._initial_roi_receipt = None
        if self._closing:
            return
        state = None
        try:
            slot = self._slot
            if slot is None:
                raise RuntimeError("initial ROI receipt lost its live source slot")
            state = slot.current_camera_roi_state()
            if ack.status is ControlAckStatus.APPLIED:
                if state.control_revision != ack.revision:
                    raise RuntimeError(
                        "initial ROI APPLIED receipt differs from source state"
                    )
                if state.binding is None:
                    if state.failure_reason is None:
                        raise RuntimeError(
                            "initial ROI became raw without a downstream failure"
                        )
                    rejected_selection = self._applied_request.roi
                    raw_request = replace(self._applied_request, roi=None)
                    board_widget = self._adopt_roi_presentation(state, stage=True)
                    self._applied_request = raw_request
                    self._request = raw_request
                    self._draft_selection = rejected_selection
                    board_widget.set_selector_draft_selection(rejected_selection)
                    self._roi_terminal_notice = (
                        f"ROI: downstream failed after APPLIED r{ack.revision} · "
                        f"{state.failure_reason} · raw source continues"
                    )
                    self._roi_status.setText(self._roi_terminal_notice)
                    return
                if state.binding != self._running_binding:
                    raise RuntimeError(
                        "initial ROI APPLIED receipt differs from source state"
                    )
                self._adopt_roi_presentation(state, stage=False)
                self._roi_terminal_notice = None
                self._roi_status.setText(
                    f"ROI: APPLIED r{ack.revision} · waiting for coherent front"
                )
                return
            if ack.status is ControlAckStatus.REJECTED:
                if state.binding is not None or state.control_revision is not None:
                    raise RuntimeError(
                        "rejected initial ROI left an applied scalar branch"
                    )
                rejected_selection = self._applied_request.roi
                raw_request = replace(self._applied_request, roi=None)
                board_widget = self._adopt_roi_presentation(state, stage=True)
                self._applied_request = raw_request
                self._request = raw_request
                self._draft_selection = rejected_selection
                board_widget.set_selector_draft_selection(rejected_selection)
                self._roi_terminal_notice = (
                    f"ROI: REJECTED r{ack.revision} · {ack.reason}"
                )
                self._roi_status.setText(self._roi_terminal_notice)
                return
            if ack.status is ControlAckStatus.TERMINATED:
                self._roi_terminal_notice = (
                    f"ROI: TERMINATED r{ack.revision} · {ack.reason}"
                )
                self._roi_status.setText(self._roi_terminal_notice)
                return
            if ack.status is ControlAckStatus.SUPERSEDED:
                raise RuntimeError(
                    "initial ROI control cannot be superseded before first front"
                )
            raise RuntimeError(f"unexpected initial ROI ack {ack.status.value}")
        except BaseException as error:
            request = self._applied_request
            draft = self._draft_selection
            if ack.status is ControlAckStatus.REJECTED:
                draft = request.roi
                request = replace(request, roi=None)
            elif ack.status is ControlAckStatus.APPLIED and state is not None:
                if state.binding is None:
                    draft = request.roi
                    request = replace(request, roi=None)
                else:
                    request = self._request_for_roi_state(state, request)
                    if draft == request.roi:
                        draft = None
            self._record_roi_presentation_failure(
                f"initial ROI r{ack.revision} could not be presented",
                error,
                request=request,
                state=state,
                draft_selection=draft,
                downstream_reason=(
                    ack.reason
                    if ack.status is ControlAckStatus.REJECTED
                    else None
                ),
            )

    def _promote_roi_state(
        self,
        state: CameraMonitorRoiState,
        pending: _PendingRoiControl,
    ) -> None:
        if not isinstance(state, CameraMonitorRoiState):
            raise TypeError("camera ROI state has an unexpected type")
        revision = pending.receipt.snapshot().revision
        if state.control_revision != revision:
            raise RuntimeError("APPLIED ROI receipt differs from downstream state")
        if state.binding is None:
            raw_request = replace(pending.request, roi=None)
            board_widget = self._adopt_roi_presentation(state, stage=True)
            self._applied_request = raw_request
            self._request = raw_request
            if pending.binding is None:
                self._draft_selection = None
                board_widget.set_selector_draft_selection(None)
                self._roi_terminal_notice = None
                self._roi_status.setText(
                    f"ROI: APPLIED r{revision} · waiting for coherent raw front"
                )
                return
            if state.failure_reason is None:
                raise RuntimeError(
                    "APPLIED ROI create/retarget became raw without a failure reason"
                )
            self._draft_selection = pending.request.roi
            board_widget.set_selector_draft_selection(self._draft_selection)
            self._roi_terminal_notice = (
                f"ROI: downstream failed after APPLIED r{revision} · "
                f"{state.failure_reason} · raw source continues"
            )
            self._roi_status.setText(self._roi_terminal_notice)
            return
        if state.binding != pending.binding:
            raise RuntimeError("APPLIED ROI receipt differs from downstream state")
        board_widget = self._adopt_roi_presentation(state, stage=True)
        self._roi_terminal_notice = None
        self._applied_request = pending.request
        self._request = pending.request
        self._roi_status.setText(
            f"ROI: APPLIED r{revision} · waiting for coherent front"
        )

    def _adopt_roi_presentation(
        self,
        state: CameraMonitorRoiState,
        *,
        stage: bool,
    ) -> QtRasterBoard:
        if not isinstance(state, CameraMonitorRoiState):
            raise TypeError("camera ROI state has an unexpected type")
        previous_curve_semantics = (
            (
                None
                if self._running_binding is None
                else self._running_binding.fingerprint
            ),
            self._applied_control_revision,
        )
        live = self._live
        board = self._board
        board_widget = self._board_widget
        if live is None or board is None or not isinstance(board_widget, QtRasterBoard):
            raise RuntimeError("APPLIED ROI state has no live presentation owner")

        if state.binding is None:
            scalar_dataset_id = None
            scalar_documents: tuple[FigureDocument, ...] = ()
            projection_text = _RAW_PROJECTION_TEXT
            panel_ids = (_IMAGE_PANEL_ID,)
            columns = 1
        else:
            if (
                state.scalar_dataset_edge is None
                or state.scalar_generation is None
                or state.scalar_block_id is None
            ):
                raise RuntimeError("active ROI state lacks scalar generation identity")
            visible_sources = {
                panel.source_identity
                for panel in (
                    ()
                    if board_widget.front_frame is None
                    else board_widget.front_frame.panels
                )
                if panel.source_identity.stream_generation == state.scalar_generation
            }
            if len({source.dataset_id for source in visible_sources}) > 1:
                raise RuntimeError("one scalar generation has multiple visible DatasetIds")
            if visible_sources:
                source = next(iter(visible_sources))
                if (
                    source.block_id != state.scalar_block_id
                    or source.schema_fingerprint
                    != state.scalar_dataset_edge.schema.fingerprint
                ):
                    raise RuntimeError("visible scalar identity differs from applied ROI state")
                scalar_dataset_id = source.dataset_id
            else:
                scalar_dataset_id = DatasetId(
                    f"camera-monitor-roi-scalar-{state.scalar_generation.value}"
                )
            scalar_documents, projection_text = _scalar_documents(
                state.scalar_dataset_edge.schema,
                state.binding,
                scalar_dataset_id,
                identity=(
                    f"{self._run_owner.generation}-r{state.control_revision}-"
                    f"{state.scalar_generation.value}"
                ),
            )
            panel_ids = (_IMAGE_PANEL_ID, *_SCALAR_PANEL_IDS)
            columns = 2

        self._projection_text = projection_text
        self._show_projection_target()
        if stage:
            panels = tuple(
                PanelSlot(panel_id, "camera-monitor", "monitor-camera")
                for panel_id in panel_ids
            )
            current_model = board.model
            topology_changed = current_model.panel_ids != panel_ids
            staged_model = None
            board_staged = False
            widget_staged = False
            try:
                if topology_changed:
                    staged_model = current_model.replace_panels(panels)
                    board_widget.stage_layout(
                        panel_ids,
                        board_id=staged_model.board_id,
                        layout_generation=staged_model.layout_generation,
                        columns=columns,
                    )
                    widget_staged = True
                    board.reconfigure(staged_model)
                    board_staged = True
                live.reconfigure_scalar(
                    state,
                    scalar_dataset_id=scalar_dataset_id,
                    scalar_documents=scalar_documents,
                    curve_display=(
                        self._curve_display if scalar_documents else None
                    ),
                )
            except BaseException as error:
                rollback_errors = []
                if staged_model is not None:
                    if board_staged:
                        try:
                            if not board.discard_staged_model(staged_model):
                                rollback_errors.append(
                                    RuntimeError(
                                        "staged BoardModel was not available for rollback"
                                    )
                                )
                        except BaseException as rollback_error:
                            rollback_errors.append(rollback_error)
                    if widget_staged:
                        try:
                            if not board_widget.discard_staged_layout(
                                board_id=staged_model.board_id,
                                layout_generation=staged_model.layout_generation,
                            ):
                                rollback_errors.append(
                                    RuntimeError(
                                        "staged Qt layout was not available for rollback"
                                    )
                                )
                        except BaseException as rollback_error:
                            rollback_errors.append(rollback_error)
                if rollback_errors:
                    detail = "; ".join(
                        f"{type(item).__name__}: {item}" for item in rollback_errors
                    )
                    raise RuntimeError(
                        "ROI presentation reconfiguration failed and rollback "
                        f"did not converge: {detail}"
                    ) from error
                raise

        next_curve_semantics = (
            None if state.binding is None else state.binding.fingerprint,
            state.control_revision,
        )
        if next_curve_semantics != previous_curve_semantics:
            self._clear_curve_range_candidate()
        self._running_binding = state.binding
        self._applied_control_revision = state.control_revision
        self._observed_roi_state_revision = state.state_revision
        return board_widget

    def _drain_attachments(self) -> None:
        for generation, payload in self._run_owner.drain_attachments():
            (
                slot,
                image_document,
                scalar_documents,
                image_display,
                image_viewport,
                attached,
                response,
            ) = payload
            try:
                if self._closing or generation != self._run_owner.generation:
                    raise RuntimeError("Workbench no longer accepts this monitor")
                if slot.terminal:
                    raise RuntimeError("monitor slot became terminal before attachment")
                if scalar_documents and len(scalar_documents) != len(_SCALAR_PANEL_IDS):
                    raise RuntimeError(
                        "ROI monitor attachment requires its closed three-panel scalar set"
                    )
                if self._live is not None:
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
                live = LiveBoardController(
                    slot,
                    image_document,
                    board,
                    submit_worker=self._submit_worker,
                    request_owner_wake=self._wake.request_owner_wake,
                    scalar_documents=scalar_documents,
                    scalar_raster_size=_SCALAR_RASTER_SIZE,
                    worker_thread_affine=self._run_owner.worker_thread_affine,
                    image_display=image_display,
                    image_viewport=image_viewport,
                    curve_display=(
                        self._curve_display if scalar_documents else None
                    ),
                )
                self._slot = slot
                self._board = board
                self._live = live
                self._view_attach_inflight = False
                self._initial_roi_receipt = None
                self._initial_roi_receipt_loaded = False
                self._update_display_controls()
                response["accepted"] = True
            except BaseException as error:
                self._view_attach_inflight = False
                self._update_display_controls()
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
        if fault is None:
            fault = self._roi_presentation_failure
        if failure is not None or fault is not None:
            self._discard_pending_image_interaction()
            self._discard_pending_curve_interaction()
            detail = failure if failure is not None else str(fault)
            self._view_status.setText(f"View: FAILED · {detail}")
            self._update_roi_controls()
            return
        board_widget = self._board_widget
        if isinstance(board_widget, QtRasterBoard):
            image_selector_fault = board_widget.selector_fault
            curve_selector_fault = board_widget.curve_selector_fault
            image_selector_healthy = (
                image_selector_fault is None
                and board_widget.visible_image_origin() is not None
            )
            curve_selector_healthy = (
                curve_selector_fault is None
                and board_widget.visible_curve_origin() is not None
            )
            if (
                not image_selector_healthy
                and not curve_selector_healthy
                and (
                    image_selector_fault is not None
                    or curve_selector_fault is not None
                )
            ):
                self._selector_switch.setChecked(False)
            if image_selector_fault is not None:
                self._roi_status.setText(
                    f"ROI selector disabled: {image_selector_fault}"
                )
            elif curve_selector_fault is not None:
                self._roi_status.setText(
                    f"Curve interaction disabled: {curve_selector_fault}"
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
                    "View: WAITING · previous front retained · diagnostics pending"
                )
                self._update_roi_controls()
                return
            if not self._reconcile_visible_roi(status):
                previous = (
                    "raw"
                    if status.scalar_control_revision is None
                    else f"control r{status.scalar_control_revision}"
                )
                self._view_status.setText(
                    "View: WAITING · previous front retained "
                    f"({previous}) · target {self._projection_text}"
                )
                self._update_roi_controls()
                return
            if status.image_display_revision != self._image_display.revision:
                shown = (
                    "unknown"
                    if status.image_display_revision is None
                    else f"r{status.image_display_revision}"
                )
                self._view_status.setText(
                    "View: WAITING · previous image display retained "
                    f"({shown}) · target r{self._image_display.revision}"
                )
                self._sync_image_display_editors()
                self._update_roi_controls()
                return
            if (
                self._running_binding is not None
                and status.curve_display_revision != self._curve_display.revision
            ):
                shown = (
                    "unknown"
                    if status.curve_display_revision is None
                    else f"r{status.curve_display_revision}"
                )
                self._view_status.setText(
                    "View: WAITING · previous curve display retained "
                    f"({shown}) · target r{self._curve_display.revision}"
                )
                self._sync_image_display_editors()
                self._sync_curve_display_editors()
                self._update_roi_controls()
                return
            pending_origin = self._pending_image_interaction_origin
            if (
                pending_origin is not None
                and self._image_display.revision
                > pending_origin.presentation.panel_revision
            ):
                self._pending_image_interaction_origin = None
            pending_curve_origin = self._pending_curve_interaction_origin
            if (
                pending_curve_origin is not None
                and self._curve_display.revision
                > pending_curve_origin.presentation.panel_revision
            ):
                self._pending_curve_interaction_origin = None
            self._sync_image_display_editors()
            self._sync_curve_display_editors()
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
                    f"control=r{status.scalar_control_revision} · "
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
                f"View: LIVE · {self._projection_text} · "
                f"display r{self._image_display.revision} "
                f"{self._image_display.relim_mode.value}/"
                f"{self._image_display.colormap.value}"
                + (
                    ""
                    if status.curve_display_revision is None
                    else (
                        f" · curve r{status.curve_display_revision} "
                        f"{self._curve_display.relim_mode.value} "
                        f"y={status.curve_y_limits}"
                    )
                )
                + f" · {suffix}"
            )
        elif self._handle is not None and not self._handle.snapshot().state.terminal:
            self._view_status.setText(f"View: WAITING · {self._projection_text}")
        else:
            self._view_status.setText(f"View: STOPPED · {self._projection_text}")
        self._update_roi_controls()

    def _show_projection_target(self) -> None:
        """Label a target without claiming that the retained front already shows it."""

        visible = self._visible_projection_text
        if visible is None:
            text = f"Display: no coherent front · target {self._projection_text}"
        elif visible == self._projection_text:
            text = f"Display: {visible} · previous coherent front retained"
        else:
            text = f"Display: {visible} · target {self._projection_text} pending"
        self._projection_status.setText(text)

    def _mark_projection_visible(self, projection_text: str) -> None:
        self._visible_projection_text = projection_text
        self._projection_status.setText(f"Display: {projection_text}")

    def _reconcile_visible_roi(self, status) -> bool:
        binding = self._running_binding
        board_widget = self._board_widget
        if not isinstance(board_widget, QtRasterBoard):
            return False
        front = board_widget.front_frame
        if front is None:
            return False
        if binding is None:
            if (
                status.scalar_binding_fingerprint is not None
                or status.scalar_control_revision is not None
            ):
                return False
            board_widget.set_selector_applied_selection(None)
            self._visible_binding_fingerprint = None
            self._mark_projection_visible(_RAW_PROJECTION_TEXT)
            self._board_panel_ids = tuple(panel.panel_id for panel in front.panels)
            if self._applied_control_revision is not None:
                text = self._roi_terminal_notice or (
                    f"ROI: VISIBLE r{self._applied_control_revision} · raw only"
                )
                self._roi_status.setText(text)
            return True
        if self._applied_control_revision is None:
            slot = self._slot
            if slot is None:
                return False
            state = slot.current_camera_roi_state()
            if state.binding != binding:
                return False
            self._applied_control_revision = state.control_revision
        if (
            status.scalar_binding_fingerprint != binding.fingerprint
            or status.scalar_control_revision != self._applied_control_revision
        ):
            # During staged promotion the old complete front remains visible.
            # It is stale presentation, not a data/source failure.
            return False
        if board_widget.visible_curve_origin() is None:
            board_widget.bind_curve_interaction(
                _CURVE_PANEL_ID,
                self._accept_curve_interaction,
                enabled=board_widget.selectors_enabled,
            )
        if self._visible_binding_fingerprint != binding.fingerprint:
            board_widget.set_selector_applied_selection(binding.selection)
            self._visible_binding_fingerprint = binding.fingerprint
        if self._draft_selection == binding.selection:
            self._draft_selection = None
            board_widget.set_selector_draft_selection(None)
        self._mark_projection_visible(self._projection_text)
        self._board_panel_ids = tuple(panel.panel_id for panel in front.panels)
        text = self._roi_terminal_notice or (
            f"ROI: VISIBLE r{self._applied_control_revision} · "
            f"binding {binding.fingerprint[:12]}"
        )
        self._roi_status.setText(text)
        return True

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

    def _record_roi_presentation_failure(
        self,
        context: str,
        error: BaseException,
        *,
        request: CameraMonitorRequest,
        state: CameraMonitorRoiState | None,
        draft_selection: Selection | None,
        downstream_reason: str | None = None,
    ) -> None:
        """Fold data-plane truth once while freezing the last coherent front."""

        freeze_failure = None
        live = self._live
        if live is not None:
            try:
                live.freeze_presentation()
            except BaseException as freeze_error:
                freeze_failure = (
                    f"; freeze: {type(freeze_error).__name__}: {freeze_error}"
                )
        self._applied_request = request
        self._request = request
        self._draft_selection = draft_selection
        if state is not None:
            self._running_binding = state.binding
            self._applied_control_revision = state.control_revision
            self._observed_roi_state_revision = state.state_revision
        downstream_causes = tuple(
            dict.fromkeys(
                reason
                for reason in (
                    downstream_reason,
                    None if state is None else state.failure_reason,
                )
                if reason
            )
        )
        downstream_diagnostic = (
            "" if not downstream_causes else "; downstream: " + " | ".join(downstream_causes)
        )
        downstream_status = (
            "" if not downstream_causes else " · downstream: " + " | ".join(downstream_causes)
        )
        self._roi_presentation_failure = (
            f"{context}: {type(error).__name__}: {error}"
            + downstream_diagnostic
            + ("" if freeze_failure is None else freeze_failure)
        )
        self._roi_terminal_notice = (
            f"ROI: PRESENTATION FAILED · {context}"
            + downstream_status
            + " · raw source continues"
        )
        self._roi_status.setText(self._roi_terminal_notice)
        self._record_local_failure(self._roi_presentation_failure)
        self._update_roi_controls()

    def _update_start_button(self) -> None:
        self._start_button.setEnabled(
            not self._closing
            and self._prepared is not None
            and self._run_owner.owner_reaped
        )

    def _close_live(self) -> None:
        self._view_attach_inflight = False
        self._selector_switch.setChecked(False)
        board_widget = self._board_widget
        if isinstance(board_widget, QtRasterBoard):
            board_widget.unbind_rectangle_selector()
            board_widget.unbind_curve_interaction()
        self._pending_image_interaction_origin = None
        self._pending_curve_interaction_origin = None
        self._clear_curve_range_candidate()
        live = self._live
        if live is None:
            self._sync_image_display_editors()
            self._sync_curve_display_editors()
            self._update_display_controls()
            return
        self._fold_roi_state_for_detach()
        try:
            live.close()
        finally:
            if board_widget is not None and not board_widget.has_front:
                # Board close may have cleared the real front before a later
                # slot/renderer cleanup error escapes.  Visible markers follow
                # that physical fact even when the overall close must retry.
                self._visible_binding_fingerprint = None
                self._visible_projection_text = None
                self._show_projection_target()
        self._initial_roi_receipt = None
        self._initial_roi_receipt_loaded = False
        self._roi_controls.clear()
        self._latest_roi_revision = None
        self._roi_terminal_notice = None
        self._roi_presentation_failure = None
        self._live = None
        self._slot = None
        self._board = None
        self._sync_image_display_editors()
        self._sync_curve_display_editors()
        self._update_display_controls()

    def _fold_roi_state_for_detach(self) -> None:
        """Persist the latest applied downstream intent before slot teardown."""

        slot = self._slot
        if slot is None:
            return
        request = self._applied_request
        initial = self._initial_roi_receipt
        if initial is not None:
            initial_ack = initial.snapshot()
            if (
                initial_ack.terminal
                and initial_ack.status is not ControlAckStatus.APPLIED
                and request.roi is not None
            ):
                self._draft_selection = request.roi
                request = replace(request, roi=None)
        for revision in sorted(self._roi_controls):
            pending = self._roi_controls[revision]
            if pending.receipt.snapshot().status is ControlAckStatus.APPLIED:
                request = pending.request
        try:
            state = slot.current_camera_roi_state()
        except RuntimeError:
            state = None
        if state is not None and state.binding is not None:
            request = self._request_for_roi_state(state, request)
        elif state is not None and state.failure_reason is not None:
            if request.roi is not None:
                self._draft_selection = request.roi
            request = replace(request, roi=None)
        self._applied_request = request
        self._request = request
        if state is not None:
            self._running_binding = state.binding
            self._applied_control_revision = state.control_revision
            self._observed_roi_state_revision = state.state_revision

    @staticmethod
    def _request_for_roi_state(
        state: CameraMonitorRoiState,
        base: CameraMonitorRequest,
    ) -> CameraMonitorRequest:
        binding = state.binding
        if binding is None:
            return replace(base, roi=None)
        return replace(
            base,
            roi=binding.selection,
            roi_reduction=binding.reduction,
            roi_validity_policy=binding.validity_policy,
        )

    def closeEvent(self, event) -> None:
        if self._allow_close:
            release_window(self)
            event.accept()
            return
        event.ignore()
        if not self._closing:
            self._closing = True
            self._start_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            self._settings_popup.hide()
            self._update_display_controls()
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
