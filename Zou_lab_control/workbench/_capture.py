"""Qt composition for one finite exact camera capture and live image."""

from __future__ import annotations

from dataclasses import replace
import threading
import time
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
    FigureLayer,
    SuggestionStatus,
    ViewIntent,
    suggest_view,
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
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    FluentPopup,
    FluentRevisionedFormEditor,
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
    release_window,
    retain_window,
    runtime_range_placeholders,
    screen_fit_window_size,
    scaled_px,
    set_fluent_scale,
    FluentSettingsPopupAnchor,
    sync_revisioned_form_editors,
)
from zlc_frontend.render import RenderSurface
from zlc_frontend.selector import (
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportCommit,
    PanelInteractionOrigin,
)
from zlc_neutral_atom.capture_application import PreparedFiniteCapture
from zlc_neutral_atom.runtime.pipeline import CapturePreviewSpec
from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot, RunState
from zlc_workbench.live import LiveBoardController, LiveDatasetSlot
from zlc_workbench.run_owner import QtRunOwnerMailbox
from zlc_workbench.workspace import BoardController, BoardModel, PanelSlot


_IMAGE_PANEL_ID = "capture-image"
_PROJECTION_TEXT = "latest rendered raw frame · DISPLAY ONLY"


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
        self._image_display = ImageDisplayState()
        self._image_viewport: ImageViewportTransform | None = None
        self._pending_image_interaction_origin: PanelInteractionOrigin | None = None
        self._rectangle_candidate = None
        self._view_attach_inflight = False
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
        self._interaction_status = FluentLabel(
            "Area: DISPLAY ONLY · Selectors are parked"
        )
        self._interaction_status.setObjectName("captureInteractionStatus")
        self._interaction_status.setWordWrap(True)
        self._diagnostics = FluentLabel("")
        self._diagnostics.setObjectName("diagnostics")
        self._diagnostics.setWordWrap(True)
        self._board_widget = QtRasterBoard(
            (_IMAGE_PANEL_ID,),
            self,
            columns=1,
            empty_text="No capture preview",
        )
        self._board_widget.setObjectName("captureImageBoard")
        self._selector_switch = FluentSwitch("Selectors", self)
        self._selector_switch.setObjectName("captureSelectorSwitch")
        self._selector_switch.setChecked(False)
        self._selector_switch.setEnabled(False)
        self._setting_button = FluentButton("Setting…", self, color=GREY)
        self._setting_button.setObjectName("captureDisplaySettingButton")
        self._setting_button.setEnabled(False)
        self._start_button = FluentButton("Start", self, color=GREEN)
        self._start_button.setObjectName("startButton")
        self._start_button.setEnabled(False)
        self._stop_button = FluentButton("Stop", self, color=ORANGE)
        self._stop_button.setObjectName("stopButton")
        self._stop_button.setEnabled(False)

        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName("captureTabs")
        self._live_page = QtWidgets.QWidget(self._tabs)
        live_layout = QtWidgets.QVBoxLayout(self._live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.addWidget(self._board_widget, 1)
        self._edit_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._tabs,
        )
        self._edit_image_display.setObjectName("captureImageDisplayEditEditor")
        self._tabs.addTab(self._live_page, "Live")
        self._tabs.addTab(self._edit_image_display, "Edit")

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("captureDisplaySettingsPopup")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        settings_layout.setContentsMargins(*(scaled_px(10, minimum=8),) * 4)
        self._setting_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._settings_popup,
        )
        self._setting_image_display.setObjectName(
            "captureImageDisplaySettingEditor"
        )
        settings_layout.addWidget(self._setting_image_display)
        self._settings_anchor = FluentSettingsPopupAnchor(
            self._settings_popup,
            self._setting_button,
        )
        self._sync_image_display_editors()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._selector_switch)
        controls.addWidget(self._setting_button)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._capture_status)
        layout.addWidget(self._preview_status)
        layout.addWidget(self._projection_status)
        layout.addWidget(self._interaction_status)
        layout.addWidget(self._tabs, 1)
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
        self._start_button.clicked.connect(self._start_capture)
        self._stop_button.clicked.connect(self._cancel_capture)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
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
        self._edit_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._edit_image_display)
        )
        self._setting_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._setting_image_display)
        )
        self._update_interaction_controls()
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

    def _set_selector_enabled(self, enabled: bool) -> None:
        self._board_widget.set_selectors_enabled(bool(enabled))
        if enabled:
            self._interaction_status.setText(
                "Area: DISPLAY ONLY · A/C/Z/H and exact hover armed"
            )
        elif self._board_widget.selector_fault is not None:
            self._interaction_status.setText(
                f"Image interaction disabled: {self._board_widget.selector_fault}"
            )
        elif self._rectangle_candidate is None:
            self._interaction_status.setText(
                "Area: DISPLAY ONLY · Selectors are parked"
            )

    def _open_display_settings(self) -> None:
        self._settings_anchor.toggle(
            self._setting_image_display,
            prepare=lambda: self._reload_image_display_editor(
                self._setting_image_display
            ),
        )

    def _visible_image_matches_display_revision(self) -> bool:
        origin = self._board_widget.visible_image_origin()
        front = self._board_widget.front_frame
        live = self._live
        status = None if live is None else live.front_status
        return (
            origin is not None
            and origin.panel_id == _IMAGE_PANEL_ID
            and origin.presentation.panel_revision == self._image_display.revision
            and front is not None
            and status is not None
            and status.sequence == front.sequence
            and status.image_display_revision == self._image_display.revision
        )

    def _visible_image_color_limits(self):
        if not self._visible_image_matches_display_revision():
            return None
        payload = self._board_widget.visible_image_payload()
        return None if payload is None else payload.color_limits

    def _reload_image_display_editor(
        self,
        editor: FluentRevisionedFormEditor,
    ) -> None:
        if editor not in (
            self._edit_image_display,
            self._setting_image_display,
        ):
            raise ValueError("image display editor does not belong to this window")
        editor.load(
            revision=self._image_display.revision,
            semantic_identity=self._image_display,
            values=image_display_form_values(self._image_display),
            runtime_placeholders=runtime_range_placeholders(
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
        sync_revisioned_form_editors(
            (self._edit_image_display, self._setting_image_display),
            revision=self._image_display.revision,
            semantic_identity=self._image_display,
            values=image_display_form_values(self._image_display),
            runtime_placeholders=runtime_range_placeholders(
                self._visible_image_color_limits(),
                "color_min",
                "color_max",
            ),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
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
            if self._closing or self._view_attach_inflight:
                raise RuntimeError(
                    "image display cannot change while the preview is attaching"
                )
            if self._live is not None and not self._visible_image_matches_display_revision():
                raise RuntimeError(
                    "image display cannot change until its current front is visible"
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
            reference = self._image_viewport
            if reference is None:
                raise RuntimeError("finite capture image axes are not prepared")
            self._commit_image_display(
                candidate,
                image_viewport_for_display_state(candidate, reference),
                accepted_editor=editor,
                accepted_base_revision=base_revision,
            )
        except Exception as error:
            self._record_local_failure(
                f"Image display edit rejected: {type(error).__name__}: {error}"
            )

    def _accept_rectangle_gesture(self, gesture: RectangleGesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("finite image area callback requires RectangleGesture")
        origin = self._board_widget.visible_image_origin()
        if origin is None or not self._visible_image_matches_display_revision():
            raise RuntimeError("finite image area origin is stale")
        if gesture.panel_id != _IMAGE_PANEL_ID or (
            gesture.board_id,
            gesture.layout_generation,
            gesture.sequence,
            gesture.source_identity,
            gesture.viewport_revision,
        ) != (
            origin.board_id,
            origin.layout_generation,
            origin.sequence,
            origin.source_identity,
            self._image_display.revision,
        ):
            raise RuntimeError("finite image area differs from its painted origin")
        self._rectangle_candidate = gesture.normalized_bounds
        self._board_widget.set_image_rectangle_candidate(gesture.normalized_bounds)
        left, top, right, bottom = gesture.normalized_bounds
        self._interaction_status.setText(
            "Area: DISPLAY ONLY · full-raster "
            f"({left:.4f}, {top:.4f})..({right:.4f}, {bottom:.4f})"
        )

    def _accept_image_interaction(
        self,
        command: ImageInteractionCommit,
    ) -> None:
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("image interaction callback received an unknown command")
        origin = command.origin
        if origin.panel_id != _IMAGE_PANEL_ID:
            raise RuntimeError("image interaction belongs to another panel")
        if (
            not self._visible_image_matches_display_revision()
            or self._board_widget.visible_image_origin() != origin
            or origin.presentation.panel_revision != self._image_display.revision
        ):
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
            live.reconfigure_image_display(state, viewport)
        if changed:
            self._image_display = state
            self._image_viewport = viewport
            self._pending_image_interaction_origin = interaction_origin
        self._sync_image_display_editors(
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )
        self._update_interaction_controls()

    def _discard_pending_image_interaction(self) -> None:
        origin = self._pending_image_interaction_origin
        if origin is None:
            return
        self._board_widget.discard_pending_image_interaction(origin)
        self._pending_image_interaction_origin = None

    def _update_interaction_controls(self) -> None:
        live = self._live
        slot = self._slot
        board = self._board
        ready = (
            not self._closing
            and not self._view_attach_inflight
            and live is not None
            and live.fault is None
            and slot is not None
            and slot.failure is None
            and board is not None
            and board.fault is None
            and self._board_widget.selector_fault is None
            and self._visible_image_matches_display_revision()
        )
        self._board_widget.set_interaction_readiness(
            image=ready,
            curve=False,
            histogram=False,
        )
        if self._board_widget.selector_fault is not None:
            self._selector_switch.setChecked(False)
        self._selector_switch.setEnabled(ready)
        self._board_widget.set_selectors_enabled(
            self._selector_switch.isChecked() and ready
        )
        display_enabled = (
            not self._closing
            and not self._view_attach_inflight
            and self._image_viewport is not None
            and (live is None or ready)
        )
        self._setting_button.setEnabled(display_enabled)
        self._edit_image_display.setEnabled(display_enabled)
        self._setting_image_display.setEnabled(display_enabled)

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
        if not self._run_owner.owner_reaped:
            return
        if not self._run_owner.worker_idle:
            return

        command = self._prepared
        next_generation = self._run_owner.generation + 1
        try:
            schema = command.preview_schema
            y_axes = tuple(
                axis
                for axis in schema.cell_schema.data_axes
                if axis.role == SPATIAL_Y
            )
            x_axes = tuple(
                axis
                for axis in schema.cell_schema.data_axes
                if axis.role == SPATIAL_X
            )
            if (
                len(y_axes) != 1
                or len(x_axes) != 1
                or len(schema.cell_schema.data_axes) != 2
            ):
                raise ValueError(
                    "finite capture image requires exactly one declared "
                    "SPATIAL_Y and SPATIAL_X axis"
                )
            suggestion = suggest_view(schema, ViewIntent.IMAGE)
            if (
                suggestion.status is SuggestionStatus.NEEDS_INPUT
                or suggestion.spec is None
            ):
                raise ValueError("IMAGE view needs an explicit axis choice")
            view = suggestion.spec
            dataset_id = DatasetId(f"capture-preview-{next_generation}")
            document = FigureDocument(
                f"capture-preview-{next_generation}",
                0,
                (
                    DatasetDescriptor(
                        dataset_id,
                        "Raw camera frame",
                        schema.fingerprint,
                    ),
                ),
                (FigureLayer(_IMAGE_PANEL_ID, dataset_id, view),),
            )
            image_display = self._image_display
            image_viewport = image_viewport_for_display_state(
                image_display,
                ImageViewportTransform((y_axes[0], x_axes[0])),
            )
        except BaseException as error:
            self._record_local_failure(
                f"Preview preflight failed: {type(error).__name__}: {error}"
            )
            return

        try:
            self._close_live()
            self._selector_switch.setChecked(False)
            self._board_widget.bind_rectangle_selector(
                _IMAGE_PANEL_ID,
                image_viewport,
                self._accept_rectangle_gesture,
                enabled=False,
                interaction_callback=self._accept_image_interaction,
            )
        except BaseException as error:
            self._record_local_failure(
                f"Preview interaction setup failed: {type(error).__name__}: {error}"
            )
            return

        generation = self._run_owner.begin_generation()
        if generation != next_generation:
            raise RuntimeError("capture generation changed during GUI preflight")
        self._prepared = None
        self._image_viewport = image_viewport
        self._rectangle_candidate = None
        self._pending_image_interaction_origin = None
        self._view_attach_inflight = True
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
        self._interaction_status.setText(
            "Area: DISPLAY ONLY · Selectors reset OFF for this capture"
        )
        self._sync_image_display_editors()
        self._update_interaction_controls()
        block_id = BlockId(f"capture-preview-{uuid.uuid4().hex}")

        def factory(spec: CapturePreviewSpec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=dataset_id,
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
                live = self._live
                if live is not None:
                    live.reconcile_faults()
                board = self._board
                if board is not None and board.fault is None:
                    board.present_pending()
                    if live is not None:
                        # The publication wake can reach Qt just before the
                        # live owner records its matching source transaction.
                        # A second coalesced wake therefore rechecks the actual
                        # visible front even when there is nothing new to swap.
                        frame = self._board_widget.front_frame
                        if frame is not None:
                            live.accept_presented_front(frame.sequence)
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
        finally:
            # A result/prepare completion may make the next Run eligible and
            # live.admit_pending() may then enqueue the old generation's final
            # raster in the same owner cycle.  Recompute after both actions so
            # a repeated finite Run cannot overlap that retained exact work.
            self._update_start_button()
            self._sync_poll_timer()

    def _sync_poll_timer(self) -> None:
        """Poll only a live RunHandle or an explicitly retried preview close."""

        handle = self._handle
        snapshot = self._last_snapshot
        active_run = (
            handle is not None
            and not self._run_owner.owner_reaped
            and (snapshot is None or not snapshot.state.terminal)
        )
        retry_preview_close = (
            self._live is not None
            and snapshot is not None
            and snapshot.state.terminal
            and not (
                snapshot.state is RunState.SUCCEEDED
                and snapshot.final_committed
            )
        )
        if active_run or retry_preview_close:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _drain_worker_results(self) -> None:
        for completion in self._run_owner.drain_completions():
            kind = completion.kind
            generation = completion.generation
            future = completion.future
            if kind == "render":
                try:
                    future.result()
                except BaseException as error:
                    self._discard_pending_image_interaction()
                    self._record_local_failure(
                        f"Preview worker failed: {type(error).__name__}: {error}"
                    )
                self._update_interaction_controls()
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
                        self._view_attach_inflight = False
                        self._run_owner.mark_owner_reaped()
                        self._attempt_failed = True
                        self._capture_status.setText("Capture: FAILED · NOT FINAL")
                        self._record_local_failure(
                            f"Start failed: {type(error).__name__}: {error}"
                        )
                        self._submit_prepare()
                        self._update_interaction_controls()
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
            board = None
            live = None
            try:
                if self._closing or generation != self._run_owner.generation:
                    raise RuntimeError("Workbench no longer accepts this preview")
                if slot.terminal:
                    raise RuntimeError("preview became terminal before GUI attachment")
                if self._live is not None or self._board is not None or self._slot is not None:
                    raise RuntimeError("finite preview already has a presentation owner")
                if image_display != self._image_display:
                    raise RuntimeError("preview display state changed before attachment")
                if image_viewport != self._image_viewport:
                    raise RuntimeError("preview viewport changed before attachment")
                model = BoardModel(
                    "capture-board",
                    generation,
                    RenderSurface.WORKER_RASTER_LIVE,
                    (PanelSlot(_IMAGE_PANEL_ID, "finite-capture", "capture"),),
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
                    submit_worker=self._run_owner.submit_render,
                    request_owner_wake=self._wake.request_owner_wake,
                    image_display=image_display,
                    image_viewport=image_viewport,
                )
                self._slot = slot
                self._board = board
                self._live = live
                self._view_attach_inflight = False
                self._sync_image_display_editors()
                self._update_interaction_controls()
                response["accepted"] = True
            except BaseException as error:
                self._view_attach_inflight = False
                response["accepted"] = False
                response["error"] = f"{type(error).__name__}: {error}"
                try:
                    if live is not None:
                        live.close()
                    elif board is not None:
                        board.close()
                except BaseException:
                    pass
                try:
                    slot.fail(str(response["error"]))
                except BaseException:
                    pass
                self._update_interaction_controls()
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
            self._discard_pending_image_interaction()
            detail = failure if failure is not None else str(fault)
            self._preview_status.setText(f"Preview: FAILED · {detail}")
            self._update_interaction_controls()
            return
        selector_fault = self._board_widget.selector_fault
        if selector_fault is not None:
            self._interaction_status.setText(
                f"Image interaction disabled: {selector_fault}"
            )
        snapshot = self._last_snapshot
        if (
            snapshot is not None
            and snapshot.state.terminal
            and not (snapshot.state is RunState.SUCCEEDED and snapshot.final_committed)
        ):
            self._preview_status.setText(
                f"Preview: NOT FINAL · capture {snapshot.state.value}"
            )
            self._update_interaction_controls()
            return
        if live is not None and self._board_widget.has_front:
            front = self._board_widget.front_frame
            status = live.front_status
            if front is None or status is None or status.sequence != front.sequence:
                self._preview_status.setText(
                    "Preview: WAITING · previous coherent front retained"
                )
                self._update_interaction_controls()
                return
            if status.image_display_revision != self._image_display.revision:
                shown = (
                    "unknown"
                    if status.image_display_revision is None
                    else f"r{status.image_display_revision}"
                )
                self._preview_status.setText(
                    "Preview: WAITING · previous display retained "
                    f"({shown}) · target r{self._image_display.revision}"
                )
                self._sync_image_display_editors()
                self._update_interaction_controls()
                return
            pending = self._pending_image_interaction_origin
            if (
                pending is not None
                and self._image_display.revision
                > pending.presentation.panel_revision
            ):
                self._pending_image_interaction_origin = None
            self._sync_image_display_editors()
            self._projection_status.setText(
                f"Display: r{self._image_display.revision} · "
                f"{self._image_display.relim_mode.value}/"
                f"{self._image_display.colormap.value} · {_PROJECTION_TEXT}"
            )
            if (
                snapshot is not None
                and snapshot.state is RunState.SUCCEEDED
                and snapshot.final_committed
            ):
                self._preview_status.setText(
                    f"Preview: DISPLAY ONLY · capture FINAL · {_PROJECTION_TEXT}"
                )
            else:
                self._preview_status.setText(
                    f"Preview: PROVISIONAL · {_PROJECTION_TEXT}"
                )
        self._update_interaction_controls()

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
            and self._run_owner.worker_idle
        )

    def _close_live(self) -> None:
        self._view_attach_inflight = False
        self._selector_switch.setChecked(False)
        self._board_widget.unbind_rectangle_selector()
        self._pending_image_interaction_origin = None
        self._rectangle_candidate = None
        live = self._live
        if live is None:
            self._sync_image_display_editors()
            self._update_interaction_controls()
            return
        live.close()
        self._live = None
        self._slot = None
        self._board = None
        self._sync_image_display_editors()
        self._update_interaction_controls()

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
