"""Qt composition for one free-running raw camera monitor."""

from __future__ import annotations

import threading

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    MONITOR_HISTORY,
    SPATIAL_X,
    SPATIAL_Y,
    CoordinateRangeSelection,
    ReductionMethod,
    Selection,
)
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    DisplayReduction,
    DisplayReductionMethod,
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
from zlc_frontend.matplotlib_render import estimate_single_curve_raster_peak_nbytes
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    GREEN,
    ORANGE,
    QtImageBoard,
    QtOwnerWake,
    QtRasterBoard,
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
    CameraMonitorViewSpec,
    PreparedCameraMonitor,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot, RunState
from zlc_workbench.live import LiveDatasetSlot, LiveImageBoardController
from zlc_workbench.run_owner import QtRunOwnerMailbox
from zlc_workbench.workspace import BoardController, BoardModel, PanelSlot


_IMAGE_PANEL_ID = "camera-monitor-image"
_CURVE_PANEL_ID = "camera-monitor-roi-curve"
_CURVE_RASTER_SIZE = (800, 520)
_RAW_PROJECTION_TEXT = "latest raw frame · history slot 0 · DISPLAY ONLY"


def _roi_curve_view(schema, roi: Selection, reduction: ReductionMethod):
    if not isinstance(reduction, ReductionMethod):
        raise TypeError("camera monitor ROI reduction must be ReductionMethod")
    axes = (
        schema.repeat_axis,
        *schema.point_axes,
        *schema.cell_schema.data_axes,
    )
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    x_axes = tuple(axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_X)
    y_axes = tuple(axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_Y)
    if len(history) != 1 or len(x_axes) != 1 or len(y_axes) != 1:
        raise ValueError("ROI curve requires one history and one spatial x/y axis")
    if (
        len(roi.terms) != 2
        or any(not isinstance(term, CoordinateRangeSelection) for term in roi.terms)
        or {term.axis_id for term in roi.terms}
        != {x_axes[0].axis_id, y_axes[0].axis_id}
    ):
        raise ValueError("camera monitor ROI must be one typed spatial rectangle")
    if reduction is ReductionMethod.MEAN:
        display_method = DisplayReductionMethod.MEAN
    elif reduction is ReductionMethod.SUM:
        display_method = DisplayReductionMethod.SUM
    else:
        raise ValueError("camera monitor ROI reduction must be MEAN or SUM")
    bindings = [
        AxisViewBinding(history[0].axis_id, AxisViewRole.X),
        AxisViewBinding(
            schema.repeat_axis.axis_id,
            AxisViewRole.SELECTED,
            selector=FixedIndex(0),
        ),
        AxisViewBinding(
            x_axes[0].axis_id,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(display_method),
        ),
        AxisViewBinding(
            y_axes[0].axis_id,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(display_method),
        ),
    ]
    if len(bindings) != len(axes):
        raise ValueError("camera monitor ROI curve refuses undeclared extra axes")
    view = ViewSpec(schema.fingerprint, ViewIntent.CURVE, tuple(bindings), (roi,))
    validate_view_spec(schema, view)
    terms = {term.axis_id: term for term in roi.terms}
    x_term = terms[x_axes[0].axis_id]
    y_term = terms[y_axes[0].axis_id]
    summary = (
        f"latest raw frame + ROI {reduction.value.lower()} "
        f"[{x_axes[0].name}={x_term.lower}..{x_term.upper}, "
        f"{y_axes[0].name}={y_term.lower}..{y_term.upper}] · "
        f"history 0..{history[0].size - 1} (0 newest) · DISPLAY ONLY"
    )
    return view, summary


class CameraMonitorWorkbenchWindow(QtWidgets.QWidget):
    """Nonblocking owner of one continuous monitor Run and immutable front."""

    def __init__(
        self,
        prepare,
        *,
        roi: Selection | None = None,
        roi_reduction: ReductionMethod = ReductionMethod.MEAN,
    ) -> None:
        super().__init__()
        if not callable(prepare):
            raise TypeError("camera monitor prepare must be callable")
        if roi is not None and not isinstance(roi, Selection):
            raise TypeError("roi must be zlc_data.Selection or None")
        if not isinstance(roi_reduction, ReductionMethod):
            raise TypeError("roi_reduction must be zlc_data.ReductionMethod")
        if roi_reduction not in (ReductionMethod.MEAN, ReductionMethod.SUM):
            raise ValueError("camera monitor ROI reduction must be MEAN or SUM")
        self._prepare = prepare
        self._roi = roi
        self._roi_reduction = roi_reduction
        self._projection_text = (
            _RAW_PROJECTION_TEXT
            if roi is None
            else f"latest raw frame + typed ROI {roi_reduction.value.lower()} history · DISPLAY ONLY"
        )
        self._prepared: PreparedCameraMonitor | None = None
        self._slot: LiveDatasetSlot | None = None
        self._live: LiveImageBoardController | None = None
        self._board: BoardController | None = None
        self._last_snapshot: RunSnapshot | None = None
        self._prepare_inflight = False
        self._local_diagnostic = ""
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
        self._board_widget = (
            QtImageBoard(_IMAGE_PANEL_ID, self, empty_text="Monitor stopped")
            if roi is None
            else QtRasterBoard(
                (_IMAGE_PANEL_ID, _CURVE_PANEL_ID),
                self,
                columns=2,
                empty_text="Monitor stopped",
            )
        )
        self._board_widget.setObjectName("cameraMonitorImageBoard")
        self._start_button = FluentButton("Start", self, color=GREEN)
        self._start_button.setObjectName("startButton")
        self._start_button.setEnabled(False)
        self._stop_button = FluentButton("Stop", self, color=ORANGE)
        self._stop_button.setObjectName("stopButton")
        self._stop_button.setEnabled(False)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._run_status)
        layout.addWidget(self._view_status)
        layout.addWidget(self._projection_status)
        layout.addWidget(self._board_widget, 1)
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

    def _submit_prepare(self) -> None:
        if self._closing or self._prepare_inflight or self._prepared is not None:
            return
        self._prepare_inflight = True
        generation = self._run_owner.generation

        def prepare() -> PreparedCameraMonitor:
            command = self._prepare()
            if not isinstance(command, PreparedCameraMonitor):
                raise TypeError("camera monitor prepare returned an unexpected command")
            command.view_schema
            return command

        self._run_owner.submit("prepare", prepare, generation=generation)

    def _start_monitor(self) -> None:
        if self._closing or self._prepared is None:
            return
        if self._handle is not None and not self._handle.snapshot().state.terminal:
            return
        if not self._run_owner.owner_reaped:
            return
        command = self._prepared
        try:
            schema = command.view_schema
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
            if self._roi is not None and history[0].size < 2:
                raise ValueError("ROI curve requires history_capacity >= 2")
            selection = Selection.index(history[0].axis_id, 0)
            suggestion = suggest_view(schema, ViewIntent.IMAGE, selection)
            if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
                raise ValueError("camera monitor IMAGE view needs an explicit axis choice")
            image_view = suggestion.spec
            image_evaluation_peak = estimate_view_evaluation_peak_nbytes(
                schema,
                image_view,
            )
            downstream_peak = image_evaluation_peak + estimate_gray8_raster_peak_nbytes(
                y_axes[0].size,
                x_axes[0].size,
            )
            curve_view = None
            curve_evaluation_peak = 0
            if self._roi is not None:
                curve_view, self._projection_text = _roi_curve_view(
                    schema,
                    self._roi,
                    self._roi_reduction,
                )
                curve_evaluation_peak = estimate_view_evaluation_peak_nbytes(
                    schema,
                    curve_view,
                )
                downstream_peak += (
                    curve_evaluation_peak
                    + estimate_single_curve_raster_peak_nbytes(
                        *_CURVE_RASTER_SIZE,
                        evaluated_data_upper_bound_bytes=curve_evaluation_peak,
                    )
                )
            policy = FigureEvaluationPolicy(
                max_live_nbytes=max(image_evaluation_peak, curve_evaluation_peak)
            )
        except BaseException as error:
            self._record_local_failure(
                f"View preparation failed: {type(error).__name__}: {error}"
            )
            return

        self._close_live()
        generation = self._run_owner.begin_generation()
        self._prepared = None
        self._last_snapshot = None
        self._local_diagnostic = ""
        self._refresh_diagnostics(None)
        self._run_status.setText("Monitor: STARTING")
        self._view_status.setText(f"View: WAITING · {self._projection_text}")
        self._projection_status.setText(f"Display: {self._projection_text}")
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)

        dataset_id = DatasetId(f"camera-monitor-{generation}")
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
        curve_document = None
        if curve_view is not None:
            curve_document = FigureDocument(
                f"camera-monitor-roi-curve-{generation}",
                0,
                (
                    DatasetDescriptor(
                        dataset_id,
                        f"ROI {self._roi_reduction.value.lower()} (DISPLAY ONLY)",
                        schema.fingerprint,
                    ),
                ),
                (FigureLayer(_CURVE_PANEL_ID, dataset_id, curve_view),),
            )

        def factory(spec: CameraMonitorViewSpec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=dataset_id,
                evaluation_policy=policy,
                retain_on_terminal=False,
            )
            attached = threading.Event()
            response: dict[str, object] = {}
            self._run_owner.post_attachment(
                (slot, image_document, curve_document, attached, response),
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
                downstream_peak_bytes=downstream_peak,
                factory=factory,
            ),
            generation=generation,
        )

    def _cancel_monitor(self) -> None:
        handle = self._handle
        if handle is None or handle.snapshot().state.terminal:
            return
        outcome = handle.cancel("Workbench user requested monitor stop")
        self._stop_button.setEnabled(False)
        self._local_diagnostic = f"Stop: {outcome.value}"
        self._refresh_diagnostics(self._last_snapshot)

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
            if live is not None:
                live.admit_pending()
            if board is not None and board.fault is None:
                board.present_pending()
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
            if kind == "prepare":
                if generation == self._run_owner.generation:
                    self._prepare_inflight = False
                try:
                    prepared = future.result()
                except BaseException as error:
                    if not self._closing and generation == self._run_owner.generation:
                        self._run_status.setText("Monitor: NOT READY")
                        self._record_local_failure(
                            f"Preparation failed: {type(error).__name__}: {error}"
                        )
                    continue
                if not self._closing and generation == self._run_owner.generation:
                    self._prepared = prepared
                    if (
                        self._last_snapshot is not None
                        and self._last_snapshot.state is RunState.CANCELLED
                    ):
                        self._run_status.setText("Monitor: READY · previous STOPPED")
                    else:
                        self._run_status.setText("Monitor: READY")
                    self._update_start_button()
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

    def _drain_attachments(self) -> None:
        for generation, payload in self._run_owner.drain_attachments():
            slot, image_document, curve_document, attached, response = payload
            try:
                if self._closing or generation != self._run_owner.generation:
                    raise RuntimeError("Workbench no longer accepts this monitor")
                if slot.terminal:
                    raise RuntimeError("monitor slot became terminal before attachment")
                self._close_live()
                panels = [
                    PanelSlot(
                        _IMAGE_PANEL_ID,
                        "camera-monitor",
                        "monitor-camera",
                    )
                ]
                if curve_document is not None:
                    panels.append(
                        PanelSlot(
                            _CURVE_PANEL_ID,
                            "camera-monitor",
                            "monitor-camera",
                        )
                    )
                model = BoardModel(
                    "camera-monitor-board",
                    generation,
                    RenderSurface.WORKER_RASTER_LIVE,
                    tuple(panels),
                )
                board = BoardController(
                    model,
                    self._board_widget,
                    self._wake.request_owner_wake,
                )
                live = LiveImageBoardController(
                    slot,
                    image_document,
                    board,
                    submit_worker=self._submit_worker,
                    request_owner_wake=self._wake.request_owner_wake,
                    companion_curve_document=curve_document,
                    companion_curve_size=_CURVE_RASTER_SIZE,
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
            self._stop_button.setEnabled(True)
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
        if live is not None and self._board_widget.has_front:
            coverage = live.coverage
            suffix = (
                "coverage pending"
                if coverage is None
                else (
                    f"missed={coverage.missed_events} · "
                    f"current_gap={'yes' if coverage.current_gap else 'no'}"
                )
            )
            self._view_status.setText(
                f"View: LIVE · {self._projection_text} · {suffix}"
            )
        elif self._handle is not None and not self._handle.snapshot().state.terminal:
            self._view_status.setText(f"View: WAITING · {self._projection_text}")
        else:
            self._view_status.setText(f"View: STOPPED · {self._projection_text}")

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
    *,
    roi: Selection | None = None,
    roi_reduction: ReductionMethod = ReductionMethod.MEAN,
) -> CameraMonitorWorkbenchWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("camera monitor Workbench must open on the Qt GUI thread")
    set_fluent_scale(None)
    window = CameraMonitorWorkbenchWindow(
        prepare,
        roi=roi,
        roi_reduction=roi_reduction,
    )
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = ["CameraMonitorWorkbenchWindow", "open_camera_monitor_workbench"]
