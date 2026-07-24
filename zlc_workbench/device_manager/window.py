"""Formal current DeviceManager window.

The window is deliberately a thin Qt projection of
:class:`DeviceManagerController`.  Ordinary editor changes are keyed leaf
deltas: they never read the whole form, rebuild the form, poll a runtime, or
manufacture a whole-window snapshot.  Runtime control belongs to
DeviceViewer; this surface owns installation configuration and the two
lifecycle boundaries only (initialize and shutdown-for-restart).
"""

from __future__ import annotations

from pathlib import Path
from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    ElidedLabel,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentParameterForm,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusDot,
    FluentStatusStrip,
    FluentTabWidget,
    ensure_qt_app,
    launch_fluent_window,
    muted_note_label,
    release_window,
    scaled_px,
    screen_fit_window_size,
    setting_label_width,
    signals_blocked,
    window_pad,
)
from zlc_neutral_atom.installation_plan import (
    InstallationDevicePlan,
    installation_device_plan,
)

from .controller import DeviceAdminState, DeviceManagerController
from .editor_session import form_spec


_BACKEND_LABELS = {
    "virtual": "Virtual",
    "remote_pulse": "Remote pulse",
}


class _DeviceSummaryCard(FluentFrame):
    """One compact, stable device row shared by draft and loaded sections."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, bordered=True)
        layout = QtWidgets.QHBoxLayout(self)
        pad = window_pad(0.55)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(scaled_px(8, minimum=5))
        self.role_label = ElidedLabel("")
        role_font = self.role_label.font()
        role_font.setBold(True)
        self.role_label.setFont(role_font)
        self.role_label.setMinimumWidth(scaled_px(86, minimum=68))
        self.adapter_label = ElidedLabel("")
        self.detail_label = muted_note_label("")
        self.detail_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.domain_label = muted_note_label("")
        self.domain_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.role_label)
        layout.addWidget(self.adapter_label, 1)
        layout.addWidget(self.detail_label, 2)
        layout.addWidget(self.domain_label)

    def update_content(
        self,
        *,
        role: str,
        domain: str,
        adapter_kind: str,
        detail: str,
    ) -> None:
        self.role_label.setText(str(role))
        self.domain_label.setText(str(domain))
        self.domain_label.setToolTip(str(domain))
        self.adapter_label.setText(str(adapter_kind))
        self.detail_label.setText(str(detail))
        self.detail_label.setToolTip(str(detail))


class DeviceManagerWindowBody(QtWidgets.QWidget):
    """Event-driven installation config/admin surface for one controller."""

    _owner_close_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        controller: DeviceManagerController,
        *,
        shutdown_on_owner_close: bool,
        parent=None,
    ) -> None:
        if not isinstance(controller, DeviceManagerController):
            raise TypeError("controller must be DeviceManagerController")
        super().__init__(parent)
        self._controller = controller
        self._shutdown_on_owner_close = bool(shutdown_on_owner_close)
        self._window = None
        self._permanently_closed = False
        self._owner_close_pending = False
        self._close_after_shutdown = False
        self._last_status = ("ready", "info")
        self._configured_cards: dict[str, _DeviceSummaryCard] = {}
        self._loaded_cards: dict[str, _DeviceSummaryCard] = {}

        self._build_ui()
        self._connect_signals()
        self._replace_document_projection()
        self._sync_runtime(self._controller.state)
        self._refresh_chrome()
        self.status_strip.show_message("ready", severity="info")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        pad = window_pad(1.0)
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(window_pad(0.55))

        self.tabs = FluentTabWidget(self)
        self.config_page = QtWidgets.QWidget(self.tabs)
        self.tabs.add_permanent_tab(self.config_page, "Config")
        outer.addWidget(self.tabs, 1)

        page_layout = QtWidgets.QVBoxLayout(self.config_page)
        page_layout.setContentsMargins(pad, pad, pad, pad)
        page_layout.setSpacing(window_pad(0.55))

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel("Devices"))
        header.addStretch(1)
        self.state_dot = FluentStatusDot(color=GREY, size=12)
        self.document_name = ElidedLabel("untitled")
        self.document_name.setMinimumWidth(scaled_px(150, minimum=100))
        header.addWidget(self.state_dot)
        header.addWidget(self.document_name)
        page_layout.addLayout(header)

        columns = QtWidgets.QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(window_pad(0.65))
        self.left_scroll, self.left_content, left_layout = self._scroll_column()
        self.right_scroll, self.right_content, right_layout = self._scroll_column()
        columns.addWidget(self.left_scroll, 3)
        columns.addWidget(self.right_scroll, 2)
        page_layout.addLayout(columns, 1)

        self.installation_group = FluentGroupBox("Installation", self.left_content)
        installation_layout = QtWidgets.QVBoxLayout(self.installation_group)
        group_pad = window_pad(0.7)
        installation_layout.setContentsMargins(group_pad, group_pad, group_pad, group_pad)
        installation_layout.setSpacing(window_pad(0.45))

        self.backend_combo = FluentComboBox(self.installation_group)
        for backend, label in _BACKEND_LABELS.items():
            self.backend_combo.addItem(label, backend)
        backend_row = FluentSettingRow(
            "Backend",
            self.backend_combo,
            label_width=setting_label_width(("Backend", "Transport timeout")),
            parent=self.installation_group,
        )
        installation_layout.addWidget(backend_row)

        editor = self._controller.editor
        self.form = FluentParameterForm(
            form_spec(editor.backend),
            editor.values,
            self.installation_group,
        )
        installation_layout.addWidget(self.form)
        left_layout.addWidget(self.installation_group)

        self.configured_group = FluentGroupBox("Configured devices", self.left_content)
        self.configured_layout = QtWidgets.QVBoxLayout(self.configured_group)
        self.configured_layout.setContentsMargins(group_pad, group_pad, group_pad, group_pad)
        self.configured_layout.setSpacing(window_pad(0.4))
        left_layout.addWidget(self.configured_group)
        left_layout.addStretch(1)

        self.available_group = FluentGroupBox("Available", self.right_content)
        available_layout = QtWidgets.QVBoxLayout(self.available_group)
        available_layout.setContentsMargins(group_pad, group_pad, group_pad, group_pad)
        available_layout.setSpacing(window_pad(0.4))
        self.virtual_template_button = FluentButton("Virtual", color=ACCENT)
        self.remote_template_button = FluentButton("Remote pulse", color=ACCENT)
        for button in (self.virtual_template_button, self.remote_template_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            available_layout.addWidget(button)
        right_layout.addWidget(self.available_group)

        self.loaded_group = FluentGroupBox("Loaded (session)", self.right_content)
        self.loaded_layout = QtWidgets.QVBoxLayout(self.loaded_group)
        self.loaded_layout.setContentsMargins(group_pad, group_pad, group_pad, group_pad)
        self.loaded_layout.setSpacing(window_pad(0.4))
        self.loaded_empty = muted_note_label("No active installation")
        self.loaded_empty.setWordWrap(True)
        self.loaded_layout.addWidget(self.loaded_empty)
        right_layout.addWidget(self.loaded_group)
        runtime_note = muted_note_label(
            "Live device controls belong to DeviceViewer; this window edits "
            "installation configuration."
        )
        runtime_note.setWordWrap(True)
        right_layout.addWidget(runtime_note)
        right_layout.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(scaled_px(7, minimum=5))
        self.new_combo = FluentComboBox(self.config_page)
        self.new_combo.addItem("New…", None)
        self.new_combo.addItem("Virtual", "virtual")
        self.new_combo.addItem("Remote pulse", "remote_pulse")
        self.load_button = FluentButton("Load…", color=ORANGE)
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_as_button = FluentButton("Save as…", color=ACCENT)
        self.cancel_button = FluentButton("Cancel", color=GREY)
        self.lifecycle_button = FluentButton("Init devices", color=GREEN)
        actions.addWidget(self.new_combo)
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.save_as_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        actions.addWidget(self.lifecycle_button)
        page_layout.addLayout(actions)

        self.status_strip = FluentStatusStrip(self.config_page)
        page_layout.addWidget(self.status_strip)

        self._action_widgets = (
            self.backend_combo,
            self.new_combo,
            self.virtual_template_button,
            self.remote_template_button,
            self.load_button,
            self.save_button,
            self.save_as_button,
            self.cancel_button,
            self.lifecycle_button,
        )

    @staticmethod
    def _scroll_column():
        scroll = FluentScrollArea()
        content = FluentFrame(bordered=False)
        layout = QtWidgets.QVBoxLayout(content)
        pad = window_pad(0.35)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(window_pad(0.6))
        scroll.setWidget(content)
        return scroll, content, layout

    def _connect_signals(self) -> None:
        self.form.changed.connect(self._field_changed)
        self.backend_combo.activated.connect(self._backend_selected)
        self.new_combo.activated.connect(self._new_selected)
        self.virtual_template_button.clicked.connect(self._controller.new_virtual)
        self.remote_template_button.clicked.connect(self._controller.new_remote_pulse)
        self.load_button.clicked.connect(self._load)
        self.save_button.clicked.connect(self._save)
        self.save_as_button.clicked.connect(self._save_as)
        self.cancel_button.clicked.connect(self._controller.cancel)
        self.lifecycle_button.clicked.connect(self._run_lifecycle_action)

        self._controller.draft_changed.connect(self._draft_changed)
        self._controller.document_replaced.connect(self._replace_document_projection)
        self._controller.runtime_changed.connect(self._sync_runtime)
        self._controller.busy_changed.connect(self._busy_changed)
        self._controller.status_changed.connect(self._show_status)
        self._owner_close_requested.connect(
            self._close_from_owner,
            type=QtCore.Qt.QueuedConnection,
        )

    # ------------------------------------------------------------------
    # keyed editor projection
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(str)
    def _field_changed(self, key: str) -> None:
        """Read and commit exactly the leaf that emitted the edit signal."""

        try:
            value = self.form.read_value(key)
        except (TypeError, ValueError) as error:
            self._controller.set_field_error(key, error)
            return
        self._controller.set_field(key, value)

    @QtCore.pyqtSlot(int)
    def _backend_selected(self, index: int) -> None:
        backend = self.backend_combo.itemData(index)
        if backend is not None and backend != self._controller.editor.backend:
            self._controller.switch_backend(str(backend))

    @QtCore.pyqtSlot(int)
    def _new_selected(self, index: int) -> None:
        backend = self.new_combo.itemData(index)
        with signals_blocked(self.new_combo):
            self.new_combo.setCurrentIndex(0)
        if backend == "virtual":
            self._controller.new_virtual()
        elif backend == "remote_pulse":
            self._controller.new_remote_pulse()

    @QtCore.pyqtSlot(str)
    def _draft_changed(self, key: str) -> None:
        # No form.populate/reconcile and no document construction here.  The
        # current leaf is already in the editor; only dependent text and chrome
        # are projected onto stable widgets.
        if key:
            self._update_configured_details_in_place()
        self._refresh_chrome()
        errors = self._controller.field_errors
        if errors:
            first_key = sorted(errors)[0]
            self.status_strip.show_message(
                f"{first_key}: {errors[first_key]}",
                severity="error",
            )
        elif key:
            self.status_strip.show_message(
                "unsaved configuration edit",
                severity="info",
            )

    @QtCore.pyqtSlot()
    def _replace_document_projection(self) -> None:
        editor = self._controller.editor
        backend_index = self.backend_combo.findData(editor.backend)
        if backend_index >= 0:
            with signals_blocked(self.backend_combo):
                self.backend_combo.setCurrentIndex(backend_index)
        self.form.reconcile(form_spec(editor.backend), editor.values)
        self._sync_configured_rows(self._configured_rows_for_current_draft())
        self._refresh_chrome()

    def _configured_rows_for_current_draft(
        self,
    ) -> tuple[InstallationDevicePlan, ...]:
        return installation_device_plan(self._controller.editor.backend)

    def _sync_configured_rows(
        self,
        rows: tuple[InstallationDevicePlan, ...],
    ) -> None:
        desired = {row.role: row for row in rows}
        for role in tuple(self._configured_cards):
            if role not in desired:
                card = self._configured_cards.pop(role)
                self.configured_layout.removeWidget(card)
                card.deleteLater()
        for index, row in enumerate(rows):
            card = self._configured_cards.get(row.role)
            if card is None:
                card = _DeviceSummaryCard(self.configured_group)
                self._configured_cards[row.role] = card
            card.update_content(
                role=row.role,
                domain=row.domain,
                adapter_kind=row.adapter_kind.rsplit(".", 1)[-1],
                detail=self._configured_detail(row.role),
            )
            self.configured_layout.insertWidget(index, card)

    def _update_configured_details_in_place(self) -> None:
        for role, card in self._configured_cards.items():
            detail = self._configured_detail(role)
            card.detail_label.setText(detail)
            card.detail_label.setToolTip(detail)

    def _configured_detail(self, role: str) -> str:
        editor = self._controller.editor
        values = editor.values
        if editor.backend == "remote_pulse":
            host = str(values.get("host", "")).strip() or "<host>"
            port = values.get("port", 18861)
            timeout = values.get("transport_timeout_seconds", 120.0)
            return f"{host}:{port}; timeout={timeout} s"
        seed = values.get("seed")
        seed_text = "random" if seed is None else str(seed)
        details = {
            "sequencer": f"In-process pulse target execution; seed={seed_text}",
            "rf": "In-process RF-table source driven by the virtual sequencer",
            "camera": "Externally triggered readout camera",
            "mot_camera": (
                "MOT camera; free-running live and externally triggered "
                "finite acquisition"
            ),
        }
        try:
            return details[role]
        except KeyError as error:
            raise RuntimeError(
                f"installation plan contains an unknown virtual role {role!r}"
            ) from error

    # ------------------------------------------------------------------
    # runtime observation (event driven; no timer and no control plane)
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(object)
    def _sync_runtime(self, state: DeviceAdminState) -> None:
        if not isinstance(state, DeviceAdminState):
            return
        catalog = state.catalog
        rows = () if catalog is None else tuple(catalog.items())
        desired_roles = {role for role, _info in rows}
        for role in tuple(self._loaded_cards):
            if role not in desired_roles:
                card = self._loaded_cards.pop(role)
                self.loaded_layout.removeWidget(card)
                card.deleteLater()
        for index, (role, info) in enumerate(rows):
            card = self._loaded_cards.get(role)
            if card is None:
                card = _DeviceSummaryCard(self.loaded_group)
                self._loaded_cards[role] = card
            card.update_content(
                role=role,
                domain=info.domain,
                adapter_kind=info.adapter_kind,
                detail=f"{info.availability} · {info.health}",
            )
            self.loaded_layout.insertWidget(index, card)
        self.loaded_empty.setVisible(not rows)
        self._refresh_chrome()
        if self._close_after_shutdown and state.closed:
            self._finalize_owner_close()

    @QtCore.pyqtSlot(bool, str)
    def _busy_changed(self, busy: bool, label: str) -> None:
        self._refresh_chrome()
        if busy and label:
            self.status_strip.show_message(label, severity="task")
        elif not busy and self._owner_close_pending:
            # DeviceManagerController emits busy=False before it installs the
            # completed lifecycle state.  Resume on the next owner turn so a
            # just-finished initialize is observed before deciding whether a
            # shutdown is still required.
            QtCore.QTimer.singleShot(0, self._resume_owner_close)

    @QtCore.pyqtSlot(str, str)
    def _show_status(self, text: str, severity: str) -> None:
        mapped = severity if severity in FluentStatusStrip.SEVERITIES else "info"
        self._last_status = (str(text), mapped)
        self.status_strip.show_message(str(text), severity=mapped)
        if mapped == "error" and self._close_after_shutdown:
            self._close_after_shutdown = False
            if self._window is not None:
                self._window.show()

    def _refresh_chrome(self) -> None:
        editor = self._controller.editor
        state = self._controller.state
        busy = self._controller.busy
        path = editor.path
        name = (
            "session config"
            if path is None and state.active_config is not None
            else "untitled"
            if path is None
            else path.name
        )
        shown_name = f"{name}{'*' if editor.dirty else ''}"
        self.document_name.setText(shown_name)
        self.document_name.setToolTip(shown_name if path is None else str(path))

        if state.active_config is None:
            self.state_dot.set_color(GREY)
            self.state_dot.setToolTip("No active installation")
        elif editor.restart_required:
            self.state_dot.set_color(ORANGE)
            self.state_dot.setToolTip("Saved/draft configuration differs from the active installation")
        else:
            self.state_dot.set_color(GREEN)
            self.state_dot.setToolTip("Configuration matches the active installation")

        self.save_button.set_dirty(editor.dirty, dirty_color=YELLOW)
        for widget in self._action_widgets:
            widget.setEnabled(not busy)
        self.form.setEnabled(not busy)
        has_errors = bool(self._controller.field_errors)
        draft_ready = not has_errors and self._draft_has_required_identity()
        self.save_button.setEnabled(not busy and path is not None and draft_ready)
        self.save_as_button.setEnabled(not busy and draft_ready)
        self.cancel_button.setEnabled(not busy and editor.dirty)

        if state.active_config is not None:
            if editor.restart_required:
                self.lifecycle_button.setText("Shutdown for restart")
            else:
                self.lifecycle_button.setText("Shutdown devices")
            self.lifecycle_button.setEnabled(not busy)
        elif state.closed or not state.can_initialize:
            self.lifecycle_button.setText("Restart process")
            self.lifecycle_button.setEnabled(False)
        else:
            self.lifecycle_button.setText("Init devices")
            self.lifecycle_button.setEnabled(not busy and draft_ready)

    def _draft_has_required_identity(self) -> bool:
        editor = self._controller.editor
        if editor.backend != "remote_pulse":
            return True
        return bool(str(editor.values.get("host", "")).strip())

    # ------------------------------------------------------------------
    # explicit file/lifecycle boundaries
    # ------------------------------------------------------------------

    def _load(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load installation config",
            str(self._dialog_start_directory()),
            "ZLC installation config (*.json);;All files (*)",
        )
        if path:
            self._call(self._controller.load_file, path)

    def _save(self) -> None:
        if self._controller.editor.path is None:
            self._save_as()
            return
        self._call(self._controller.save_file)

    def _save_as(self) -> None:
        current = self._controller.editor.path
        initial = (
            self._dialog_start_directory() / "installation.json"
            if current is None
            else current
        )
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save installation config",
            str(initial),
            "ZLC installation config (*.json);;All files (*)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix == "":
            target = target.with_suffix(".json")
        self._call(self._controller.save_file, target)

    def _dialog_start_directory(self) -> Path:
        path = self._controller.editor.path
        return Path.home() if path is None else path.parent

    def _run_lifecycle_action(self) -> None:
        state = self._controller.state
        if state.active_config is not None:
            self._call(self._controller.shutdown_for_restart)
        elif state.active_config is None and state.can_initialize:
            self._call(self._controller.initialize)

    def _call(self, operation, *args) -> None:
        try:
            operation(*args)
        except BaseException as error:
            self.status_strip.show_message(
                f"{type(error).__name__}: {error}",
                severity="error",
            )

    # ------------------------------------------------------------------
    # notebook / launcher lifecycle
    # ------------------------------------------------------------------

    @property
    def permanently_closed(self) -> bool:
        return self._permanently_closed

    def restore_window(self) -> None:
        if self._permanently_closed:
            raise RuntimeError("Device manager is closed")
        if self._window is None:
            raise RuntimeError("Device manager window is unavailable")
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    def request_owner_close(self) -> None:
        """Retire a notebook-owned window on the Qt owner thread."""

        if self._permanently_closed:
            return
        if QtCore.QThread.currentThread() is self.thread():
            self._close_from_owner()
        else:
            self._owner_close_requested.emit()

    @QtCore.pyqtSlot()
    def _close_from_owner(self) -> None:
        if self._permanently_closed:
            return
        self._owner_close_pending = True
        if self._shutdown_on_owner_close:
            self._begin_owned_close()
            return
        if self._controller.busy:
            self.status_strip.show_message(
                "waiting for the device operation before closing",
                severity="task",
            )
            return
        self._finalize_owner_close()

    @QtCore.pyqtSlot()
    def _resume_owner_close(self) -> None:
        if self._permanently_closed or not self._owner_close_pending:
            return
        self._close_from_owner()

    def _begin_owned_close(self) -> bool:
        """Start the one async shutdown path used by every owned window."""

        if self._permanently_closed:
            return True
        if self._controller.busy:
            self.status_strip.show_message(
                "wait for the device operation to finish before closing",
                severity="warning",
            )
            return False
        if self._controller.state.active_config is not None:
            self._close_after_shutdown = True
            self.status_strip.show_message(
                "shutting down devices before closing",
                severity="task",
            )
            self._call(self._controller.shutdown_for_restart)
            return False
        self._finalize_owner_close()
        return True

    def _finalize_owner_close(self) -> None:
        if self._permanently_closed:
            return
        self._permanently_closed = True
        self._owner_close_pending = False
        self._close_after_shutdown = False
        self._controller.close()
        window = self._window
        self._window = None
        if window is not None:
            release_window(window)
            window.hide()
            window.deleteLater()

    def _request_standalone_close(self) -> bool:
        return self._begin_owned_close()

    def _attach_window(self, window) -> None:
        self._window = window


def launch_device_manager_window(
    controller: DeviceManagerController,
    *,
    hide_on_close: bool = False,
    shutdown_on_owner_close: bool = False,
) -> DeviceManagerWindowBody:
    """Launch the DeviceManager through the one shared Fluent sequence."""

    ensure_qt_app()
    body = DeviceManagerWindowBody(
        controller,
        shutdown_on_owner_close=shutdown_on_owner_close,
    )
    initial = screen_fit_window_size(WINDOW_SCREEN_FRACTION)

    def wire(window) -> None:
        body._attach_window(window)
        if not hide_on_close:
            window.set_close_guard(body._request_standalone_close)

    launch_fluent_window(
        body,
        title="Devices@Zou lab",
        hide_on_close=hide_on_close,
        fixed_size=False,
        size=(initial.width(), initial.height()),
        wire=wire,
    )
    return body


__all__ = ["DeviceManagerWindowBody", "launch_device_manager_window"]
