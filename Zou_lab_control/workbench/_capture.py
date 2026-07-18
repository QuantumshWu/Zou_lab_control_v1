"""Qt composition for one finite exact camera capture and live image."""

from __future__ import annotations

import threading
import uuid

from PyQt5 import QtCore, QtWidgets

from Zou_lab_control.notebook.facade import (
    CaptureRequest,
    Experiment,
    _prepare_capture_for_workbench,
)
from zlc_data import BlockId, SPATIAL_X, SPATIAL_Y
from zlc_frontend.figure import (
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureLayer,
    SuggestionStatus,
    ViewIntent,
    estimate_view_evaluation_peak_nbytes,
    suggest_view,
)
from zlc_frontend.image_display import ImageDisplayState
from zlc_frontend.image_raster import estimate_indexed8_raster_peak_nbytes
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    GREEN,
    ORANGE,
    QtImageBoard,
    QtOwnerWake,
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    release_window,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
)
from zlc_frontend.render import RenderSurface
from zlc_neutral_atom.capture_application import PreparedFiniteCapture
from zlc_neutral_atom.runtime.pipeline import CapturePreviewSpec
from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot, RunState
from zlc_workbench.live import LiveBoardController, LiveDatasetSlot
from zlc_workbench.run_owner import QtRunOwnerMailbox
from zlc_workbench.workspace import BoardController, BoardModel, PanelSlot


_PROJECTION_TEXT = "latest rendered raw frame · per-frame auto contrast"


class CaptureWorkbenchWindow(QtWidgets.QWidget):
    """A nonblocking product surface for one-event finite exact captures."""

    def __init__(
        self,
        experiment: Experiment,
        request: CaptureRequest,
    ) -> None:
        super().__init__()
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be Experiment")
        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        self._experiment = experiment
        self._request = request
        self._prepared: PreparedFiniteCapture | None = None
        self._slot: LiveDatasetSlot | None = None
        self._live: LiveBoardController | None = None
        self._board: BoardController | None = None
        self._last_snapshot: RunSnapshot | None = None
        self._final_reference = None
        self._attempt_failed = False
        self._prepare_inflight = False
        self._local_diagnostic = ""
        self._closing = False
        self._allow_close = False

        self.setWindowTitle("Finite Camera Capture")
        self._capture_status = FluentLabel("Capture: PREPARING")
        self._capture_status.setObjectName("captureStatus")
        self._preview_status = FluentLabel(
            f"Preview: PROVISIONAL · {_PROJECTION_TEXT}"
        )
        self._preview_status.setObjectName("previewStatus")
        self._preview_status.setWordWrap(True)
        self._projection_status = FluentLabel(
            f"Display: {_PROJECTION_TEXT}"
        )
        self._projection_status.setObjectName("projectionStatus")
        self._projection_status.setWordWrap(True)
        self._diagnostics = FluentLabel("")
        self._diagnostics.setObjectName("diagnostics")
        self._diagnostics.setWordWrap(True)
        self._board_widget = QtImageBoard("capture-image", self)
        self._board_widget.setObjectName("captureImageBoard")
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
        layout.addWidget(self._capture_status)
        layout.addWidget(self._preview_status)
        layout.addWidget(self._projection_status)
        layout.addWidget(self._board_widget, 1)
        layout.addWidget(self._diagnostics)
        layout.addLayout(controls)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._run_owner = QtRunOwnerMailbox(
            self._wake.request_owner_wake,
            thread_name_prefix="zlc-capture-workbench",
        )
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll_run_snapshot)
        self._timer.start()
        self._start_button.clicked.connect(self._start_capture)
        self._stop_button.clicked.connect(self._cancel_capture)
        self._submit_prepare()

    @property
    def final_reference(self):
        return self._final_reference

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

        def prepare() -> PreparedFiniteCapture:
            command = _prepare_capture_for_workbench(
                self._experiment,
                self._request,
            )
            command.preview_schema
            return command

        self._run_owner.submit("prepare", prepare, generation=generation)

    def _start_capture(self) -> None:
        if self._closing or self._prepared is None:
            return
        if self._handle is not None and not self._handle.snapshot().state.terminal:
            return
        try:
            self._close_live()
        except BaseException as error:
            self._record_local_failure(
                f"Previous preview close failed: {type(error).__name__}: {error}"
            )
            return
        generation = self._run_owner.begin_generation()
        command, self._prepared = self._prepared, None
        self._last_snapshot = None
        self._final_reference = None
        self._local_diagnostic = ""
        self._attempt_failed = False
        self._refresh_diagnostics(None)
        self._prepare_inflight = False
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._capture_status.setText("Capture: STARTING · NOT FINAL")
        self._preview_status.setText(
            f"Preview: PROVISIONAL · {_PROJECTION_TEXT}"
        )

        schema = command.preview_schema
        y_axes = tuple(axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_Y)
        x_axes = tuple(axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_X)
        if len(y_axes) != 1 or len(x_axes) != 1 or len(schema.cell_schema.data_axes) != 2:
            self._record_local_failure(
                "finite capture image requires exactly one declared SPATIAL_Y and SPATIAL_X axis"
            )
            self._submit_prepare()
            return
        suggestion = suggest_view(schema, ViewIntent.IMAGE)
        if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
            self._record_local_failure("IMAGE view needs an explicit axis choice")
            self._submit_prepare()
            return
        view = suggestion.spec
        evaluation_peak = estimate_view_evaluation_peak_nbytes(schema, view)
        downstream_peak = evaluation_peak + estimate_indexed8_raster_peak_nbytes(
            y_axes[0].size,
            x_axes[0].size,
            value_itemsize=schema.cell_schema.dtype.itemsize,
            retained_sample_fronts=1,
        )
        policy = FigureEvaluationPolicy(max_live_nbytes=evaluation_peak)
        dataset_id = DatasetId(f"capture-preview-{generation}")
        document = FigureDocument(
            f"capture-preview-{generation}",
            0,
            (DatasetDescriptor(dataset_id, "Raw camera frame", schema.fingerprint),),
            (FigureLayer("capture-image", dataset_id, view),),
        )
        block_id = BlockId(f"capture-preview-{uuid.uuid4().hex}")
        image_display = ImageDisplayState()
        image_viewport = ImageViewportTransform((y_axes[0], x_axes[0]))

        def factory(spec: CapturePreviewSpec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=dataset_id,
                evaluation_policy=policy,
            )
            attached = threading.Event()
            response: dict[str, object] = {}
            self._run_owner.post_attachment(
                (
                    slot,
                    document,
                    image_display,
                    image_viewport,
                    attached,
                    response,
                ),
                generation=generation,
            )
            if not attached.wait(5.0):
                slot.fail("GUI preview attachment timed out")
                raise TimeoutError("GUI preview attachment timed out")
            if response.get("accepted") is not True:
                reason = str(response.get("error", "GUI preview attachment was rejected"))
                slot.fail(reason)
                raise RuntimeError(reason)
            return slot

        self._run_owner.submit(
            "start",
            lambda: command.start_with_preview(
                block_id=block_id,
                downstream_peak_bytes=downstream_peak,
                factory=factory,
            ),
            generation=generation,
        )

    def _cancel_capture(self) -> None:
        handle = self._handle
        if handle is None:
            return
        outcome = handle.cancel("Workbench user requested stop")
        self._local_diagnostic = f"Stop: {outcome.value}"
        self._refresh_diagnostics(self._last_snapshot)

    def _poll_run_snapshot(self) -> None:
        handle = self._handle
        if handle is None:
            if self._closing:
                self._wake.request_owner_wake()
            return
        queued = self._run_owner.poll_snapshot(self._last_snapshot)
        if not queued:
            last = self._last_snapshot
            retry_preview_close = (
                self._live is not None
                and last is not None
                and last.state.terminal
                and not (last.state is RunState.SUCCEEDED and last.final_committed)
            )
            if self._closing or retry_preview_close:
                self._wake.request_owner_wake()
            return
        self._wake.request_owner_wake()

    def _owner_cycle(self) -> None:
        try:
            self._drain_worker_results()
            self._drain_attachments()
            try:
                board = self._board
                if board is not None and board.fault is None:
                    board.present_pending()
                live = self._live
                if live is not None:
                    live.admit_pending()
            except BaseException as error:
                message = f"Preview failed: {type(error).__name__}: {error}"
                slot = self._slot
                if slot is not None:
                    try:
                        slot.fail(message)
                    except BaseException:
                        pass
                self._record_local_failure(message)
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
            self._retire_nonfinal_preview()
            self._refresh_preview_status()
            self._maybe_finish_close()
        except BaseException as error:
            self._record_local_failure(f"{type(error).__name__}: {error}")
            self._maybe_finish_close()

    def _drain_worker_results(self) -> None:
        for completion in self._run_owner.drain_completions():
            kind = completion.kind
            generation = completion.generation
            future = completion.future
            if kind == "render":
                try:
                    future.result()
                except BaseException as error:
                    self._record_local_failure(
                        f"Preview worker failed: {type(error).__name__}: {error}"
                    )
                continue
            if kind == "prepare":
                if generation != self._run_owner.generation:
                    continue
                self._prepare_inflight = False
                try:
                    prepared = future.result()
                except BaseException as error:
                    if not self._closing:
                        snapshot = self._last_snapshot
                        if (
                            snapshot is not None
                            and snapshot.state is RunState.SUCCEEDED
                            and snapshot.final_committed
                        ):
                            self._capture_status.setText(
                                "Capture: FINAL · NEXT NOT READY"
                            )
                        elif snapshot is not None and snapshot.state.terminal:
                            self._capture_status.setText(
                                f"Capture: {snapshot.state.value} · NOT FINAL · NEXT NOT READY"
                            )
                        else:
                            self._capture_status.setText(
                                "Capture: NOT READY · NOT FINAL"
                            )
                        prefix = (
                            "Preparation failed"
                            if snapshot is None
                            else "Next preparation failed"
                        )
                        self._record_local_failure(
                            f"{prefix}: {type(error).__name__}: {error}"
                        )
                    continue
                if self._closing or generation != self._run_owner.generation:
                    continue
                self._prepared = prepared
                snapshot = self._last_snapshot
                if snapshot is None:
                    if self._attempt_failed:
                        self._capture_status.setText("Capture: FAILED · NOT FINAL")
                    else:
                        self._capture_status.setText("Capture: READY · NOT FINAL")
                elif snapshot.state is RunState.SUCCEEDED and snapshot.final_committed:
                    self._capture_status.setText("Capture: FINAL")
                else:
                    self._capture_status.setText(
                        f"Capture: {snapshot.state.value} · NOT FINAL"
                    )
                self._update_start_button()
                continue
            if kind == "start":
                try:
                    handle = future.result()
                except BaseException as error:
                    if generation == self._run_owner.generation:
                        self._run_owner.mark_owner_reaped()
                        self._attempt_failed = True
                        self._capture_status.setText("Capture: FAILED · NOT FINAL")
                        self._record_local_failure(
                            f"Start failed: {type(error).__name__}: {error}"
                        )
                        self._submit_prepare()
                    continue
                if self._closing or generation != self._run_owner.generation:
                    if self._closing:
                        self._run_owner.set_handle(handle)
                    handle.cancel("Workbench closed before Run admission returned")
                else:
                    self._run_owner.set_handle(handle)
                    self._stop_button.setEnabled(True)
                    self._run_owner.enqueue_snapshot(
                        handle.snapshot(),
                        generation=generation,
                    )
                continue
            if kind == "result":
                try:
                    reference = future.result()
                except BaseException as error:
                    self._run_owner.finish_terminal_job(
                        generation,
                        owner_reaped=False,
                    )
                    self._record_local_failure(
                        f"Final result retrieval failed: {type(error).__name__}: {error}"
                    )
                    continue
                self._run_owner.finish_terminal_job(
                    generation,
                    owner_reaped=True,
                )
                if generation == self._run_owner.generation:
                    self._final_reference = reference
                    self._update_start_button()
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
                    self._update_start_button()

    def _drain_attachments(self) -> None:
        for generation, payload in self._run_owner.drain_attachments():
            (
                slot,
                document,
                image_display,
                image_viewport,
                attached,
                response,
            ) = payload
            try:
                if self._closing or generation != self._run_owner.generation:
                    raise RuntimeError("Workbench no longer accepts this preview")
                if slot.terminal:
                    raise RuntimeError("preview became terminal before GUI attachment")
                self._close_live()
                model = BoardModel(
                    "capture-board",
                    generation,
                    RenderSurface.WORKER_RASTER_LIVE,
                    (PanelSlot("capture-image", "finite-capture", "capture"),),
                )
                board = BoardController(
                    model,
                    self._board_widget,
                    self._wake.request_owner_wake,
                )
                live = LiveBoardController(
                    slot,
                    document,
                    board,
                    submit_worker=self._submit_worker,
                    request_owner_wake=self._wake.request_owner_wake,
                    image_display=image_display,
                    image_viewport=image_viewport,
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
            self._capture_status.setText(
                f"Capture: {state.value} / {snapshot.phase} · NOT FINAL"
            )
            self._stop_button.setEnabled(True)
        elif state is RunState.SUCCEEDED and snapshot.final_committed:
            self._capture_status.setText("Capture: FINAL")
            self._stop_button.setEnabled(False)
            handle = self._handle
            assert handle is not None
            self._run_owner.begin_terminal_job("result", handle.result)
            self._submit_prepare()
        else:
            self._capture_status.setText(f"Capture: {state.value} · NOT FINAL")
            self._preview_status.setText(
                f"Preview: NOT FINAL · capture {state.value}"
            )
            self._stop_button.setEnabled(False)
            handle = self._handle
            assert handle is not None
            self._run_owner.begin_terminal_job("reap", handle.wait)
            self._submit_prepare()
        self._refresh_diagnostics(snapshot)

    def _retire_nonfinal_preview(self) -> None:
        snapshot = self._last_snapshot
        if (
            self._live is None
            or snapshot is None
            or not snapshot.state.terminal
            or (snapshot.state is RunState.SUCCEEDED and snapshot.final_committed)
        ):
            return
        try:
            self._close_live()
        except BaseException as error:
            self._record_local_failure(
                f"Preview close failed: {type(error).__name__}: {error}"
            )

    def _refresh_preview_status(self) -> None:
        slot = self._slot
        live = self._live
        board = self._board
        failure = None if slot is None else slot.failure
        fault = None if live is None else live.fault
        if fault is None and board is not None:
            fault = board.fault
        if failure is not None or fault is not None:
            detail = failure if failure is not None else str(fault)
            self._preview_status.setText(f"Preview: FAILED · {detail}")
            return
        snapshot = self._last_snapshot
        if (
            snapshot is not None
            and snapshot.state.terminal
            and not (snapshot.state is RunState.SUCCEEDED and snapshot.final_committed)
        ):
            self._preview_status.setText(
                f"Preview: NOT FINAL · capture {snapshot.state.value}"
            )
            return
        if (
            snapshot is not None
            and snapshot.state is RunState.SUCCEEDED
            and snapshot.final_committed
            and self._board_widget.has_front
        ):
            self._preview_status.setText(
                f"Preview: DISPLAY ONLY · capture FINAL · {_PROJECTION_TEXT}"
            )
        elif self._board_widget.has_front:
            self._preview_status.setText(f"Preview: PROVISIONAL · {_PROJECTION_TEXT}")

    def _refresh_diagnostics(self, snapshot: RunSnapshot | None) -> None:
        parts = [self._local_diagnostic] if self._local_diagnostic else []
        if snapshot is not None:
            if snapshot.primary_error:
                parts.append(f"Error: {snapshot.primary_error}")
            parts.extend(f"Cleanup: {value}" for value in snapshot.cleanup_errors)
            if snapshot.commit_recovery_warning:
                parts.append(f"Commit warning: {snapshot.commit_recovery_warning}")
            if snapshot.recovery_instruction:
                parts.append(f"Recovery: {snapshot.recovery_instruction}")
        self._diagnostics.setText("\n".join(parts))

    def _record_local_failure(self, message: str) -> None:
        self._local_diagnostic = str(message)
        self._refresh_diagnostics(self._last_snapshot)

    def _update_start_button(self) -> None:
        handle = self._handle
        previous_settled = handle is None or (
            handle.snapshot().state.terminal and self._run_owner.owner_reaped
        )
        self._start_button.setEnabled(
            not self._closing
            and self._prepared is not None
            and previous_settled
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
            self._capture_status.setText("Capture: CLOSING")
            try:
                self._close_live()
            except BaseException as error:
                self._record_local_failure(
                    f"Preview close failed: {type(error).__name__}: {error}"
                )
            handle = self._handle
            if handle is not None and not handle.snapshot().state.terminal:
                handle.cancel("Workbench window is closing")
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
                f"Preview close failed: {type(error).__name__}: {error}"
            )
            return
        self._run_owner.shutdown()
        self._timer.stop()
        self._wake.detach()
        self._allow_close = True
        QtCore.QTimer.singleShot(0, self.close)


def open_capture_workbench(
    experiment: Experiment,
    request: CaptureRequest,
) -> CaptureWorkbenchWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("capture Workbench must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = CaptureWorkbenchWindow(experiment, request)
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = ["CaptureWorkbenchWindow", "open_capture_workbench"]
