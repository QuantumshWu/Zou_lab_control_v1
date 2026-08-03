"""Fluent projection of the ordered DeviceInstance graph editor."""

from __future__ import annotations

from pathlib import Path
import re
import threading

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
    FluentLineEdit,
    FluentParameterForm,
    FluentScrollArea,
    FluentSectionLabel,
    FluentStatusDot,
    FluentStatusStrip,
    FluentTabWidget,
    ensure_qt_app,
    launch_fluent_window,
    muted_note_label,
    release_window,
    scaled_px,
    screen_fit_window_size,
    signals_blocked,
    wait_for_owner_retirement,
    window_pad,
)
from zlc_neutral_atom.device_types import (
    device_type,
    discover_device_types,
)
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    discover_installation_templates,
)
from zlc_neutral_atom.logic_node import SelectionParameterPatch

from .controller import DeviceAdminState, DeviceManagerController


_OBJECT_TOKEN = re.compile(r"[^A-Za-z0-9_]+")


def _object_token(value: str) -> str:
    return _OBJECT_TOKEN.sub("_", value).strip("_") or "item"


class _DeviceSummaryCard(FluentFrame):
    """Compact read-only row for discovered or currently loaded devices."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, bordered=True)
        layout = QtWidgets.QHBoxLayout(self)
        pad = window_pad(0.5)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(scaled_px(7, minimum=5))
        self.title_label = ElidedLabel("")
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setMinimumWidth(scaled_px(92, minimum=72))
        self.detail_label = muted_note_label("")
        self.detail_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label, 1)

    def update_content(self, title: str, detail: str) -> None:
        self.title_label.setText(title)
        self.title_label.setToolTip(title)
        self.detail_label.setText(detail)
        self.detail_label.setToolTip(detail)


class _DiscoveredDeviceCard(_DeviceSummaryCard):
    add_requested = QtCore.pyqtSignal(object)

    def __init__(self, value: DeviceInstanceConfig, parent=None) -> None:
        super().__init__(parent)
        self.value = value
        self.add_button = FluentButton("Add", color=ACCENT)
        self.add_button.setObjectName(
            f"device_discovered_add_{_object_token(value.instance_id)}"
        )
        self.layout().addWidget(self.add_button)
        self.add_button.clicked.connect(
            lambda _checked=False: self.add_requested.emit(self.value)
        )
        self.reconcile(value)

    def reconcile(self, value: DeviceInstanceConfig) -> None:
        self.value = value
        descriptor = device_type(value.type_id)
        self.update_content(
            value.role,
            f"{descriptor.label} · {value.instance_id}",
        )


class _DeviceEditorCard(FluentFrame):
    """One stable editor card; only its type-owned form body may reconcile."""

    def __init__(self, controller: DeviceManagerController, instance_id: str, parent=None):
        super().__init__(parent, bordered=True)
        self._controller = controller
        self._instance_id = instance_id
        row = controller.editor.row(instance_id)
        self._domain = device_type(row.type_id).domain
        self._collapsed = False

        layout = QtWidgets.QVBoxLayout(self)
        pad = window_pad(0.6)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(window_pad(0.4))

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(scaled_px(7, minimum=5))
        self.collapse_button = FluentButton("−", color=GREY)
        self.collapse_button.setFixedWidth(scaled_px(32, minimum=26))
        self.identity_label = muted_note_label(instance_id)
        self.identity_label.setMinimumWidth(scaled_px(100, minimum=80))
        self.role_edit = FluentLineEdit()
        self.role_edit.setPlaceholderText("role")
        self.role_edit.setObjectName(f"device_role_{_object_token(instance_id)}")
        self.type_combo = FluentComboBox()
        self.type_combo.setObjectName(f"device_type_{_object_token(instance_id)}")
        self.remove_button = FluentButton("Remove", color=ORANGE)
        self.remove_button.setObjectName(
            f"device_remove_{_object_token(instance_id)}"
        )
        header.addWidget(self.collapse_button)
        header.addWidget(self.identity_label)
        header.addWidget(self.role_edit, 1)
        header.addWidget(self.type_combo, 2)
        header.addWidget(self.remove_button)
        layout.addLayout(header)

        self.form = FluentParameterForm(
            controller.editor.form_spec(instance_id),
            row.parameters,
            self,
        )
        self.form.setObjectName(
            f"device_parameters_{_object_token(instance_id)}"
        )
        layout.addWidget(self.form)

        self.collapse_button.clicked.connect(self._toggle_collapsed)
        self.role_edit.textChanged.connect(self._role_changed)
        self.type_combo.activated.connect(self._type_changed)
        self.remove_button.clicked.connect(
            lambda _checked=False: self._controller.remove(self._instance_id)
        )
        self.form.changed.connect(self._parameter_changed)
        self.setObjectName(f"device_card_{_object_token(instance_id)}")
        self.reconcile(row)

    @property
    def domain(self) -> str:
        return self._domain

    def reconcile(self, row) -> None:
        descriptor = device_type(row.type_id)
        if descriptor.domain != self._domain:
            self._domain = descriptor.domain
            self._fill_type_combo()
        elif self.type_combo.count() == 0:
            self._fill_type_combo()
        with signals_blocked(self.role_edit, self.type_combo):
            self.role_edit.setText(row.role)
            index = self.type_combo.findData(row.type_id)
            if index < 0:
                raise RuntimeError(
                    f"device type {row.type_id!r} is absent from domain {self._domain!r}"
                )
            self.type_combo.setCurrentIndex(index)
        self.form.reconcile(
            self._controller.editor.form_spec(self._instance_id),
            row.parameters,
        )
        self.identity_label.setText(row.instance_id)
        self.identity_label.setToolTip(
            f"Stable instance id: {row.instance_id}"
        )

    def set_busy(self, busy: bool) -> None:
        enabled = not busy
        self.collapse_button.setEnabled(True)
        self.role_edit.setEnabled(enabled)
        self.type_combo.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.form.setEnabled(enabled)

    def _fill_type_combo(self) -> None:
        with signals_blocked(self.type_combo):
            self.type_combo.clear()
            for descriptor in discover_device_types():
                if descriptor.domain == self._domain:
                    self.type_combo.addItem(descriptor.label, descriptor.type_id)

    @QtCore.pyqtSlot()
    def _toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self.form.setVisible(not self._collapsed)
        self.collapse_button.setText("+" if self._collapsed else "−")

    @QtCore.pyqtSlot(str)
    def _role_changed(self, value: str) -> None:
        self._controller.set_role(self._instance_id, value)

    @QtCore.pyqtSlot(int)
    def _type_changed(self, index: int) -> None:
        type_id = self.type_combo.itemData(index)
        if isinstance(type_id, str):
            self._controller.retype(self._instance_id, type_id)

    @QtCore.pyqtSlot(str)
    def _parameter_changed(self, key: str) -> None:
        try:
            value = self.form.read_value(key)
        except (TypeError, ValueError) as error:
            self._controller.set_field_error(
                self._instance_id,
                key,
                error,
            )
            return
        self._controller.set_parameter(self._instance_id, key, value)


class DeviceManagerWindowBody(QtWidgets.QWidget):
    """Event-driven graph authoring surface for one application authority."""

    _owner_close_requested = QtCore.pyqtSignal()

    def __init__(self, controller: DeviceManagerController, parent=None) -> None:
        if not isinstance(controller, DeviceManagerController):
            raise TypeError("controller must be DeviceManagerController")
        super().__init__(parent)
        self.setObjectName("device_manager_body")
        self._controller = controller
        self._window = None
        self._permanently_closed = False
        self._owner_closed = threading.Event()
        self._owner_close_pending = False
        self._device_cards: dict[str, _DeviceEditorCard] = {}
        self._discovered_cards: dict[str, _DiscoveredDeviceCard] = {}
        self._loaded_cards: dict[str, _DeviceSummaryCard] = {}
        self._domain_layouts: dict[str, QtWidgets.QVBoxLayout] = {}
        self._domain_add_buttons: dict[str, FluentButton] = {}

        self._build_ui()
        self._connect_signals()
        self._reconcile_document()
        self._sync_discovered(self._controller.discovered)
        self._sync_runtime(self._controller.state)
        self._refresh_chrome()
        self.status_strip.show_message("ready", severity="info")

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        pad = window_pad(1.0)
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(window_pad(0.55))

        self.tabs = FluentTabWidget(self)
        self.config_page = QtWidgets.QWidget(self.tabs)
        self.tabs.add_permanent_tab(self.config_page, "Config")
        outer.addWidget(self.tabs, 1)

        page = QtWidgets.QVBoxLayout(self.config_page)
        page.setContentsMargins(pad, pad, pad, pad)
        page.setSpacing(window_pad(0.55))

        header = QtWidgets.QHBoxLayout()
        header.addWidget(FluentSectionLabel("Devices"))
        header.addStretch(1)
        self.state_dot = FluentStatusDot(color=GREY, size=12)
        self.document_name = ElidedLabel("untitled")
        self.document_name.setMinimumWidth(scaled_px(150, minimum=100))
        header.addWidget(self.state_dot)
        header.addWidget(self.document_name)
        page.addLayout(header)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(window_pad(0.65))
        self.left_scroll, _left_content, left = self._scroll_column()
        self.right_scroll, _right_content, right = self._scroll_column()
        columns.addWidget(self.left_scroll, 3)
        columns.addWidget(self.right_scroll, 2)
        page.addLayout(columns, 1)

        domains = tuple(dict.fromkeys(
            descriptor.domain for descriptor in discover_device_types()
        ))
        for domain in domains:
            group = FluentGroupBox(domain.replace("_", " ").title())
            group.setObjectName(f"device_domain_{_object_token(domain)}")
            group_layout = QtWidgets.QVBoxLayout(group)
            group_pad = window_pad(0.7)
            group_layout.setContentsMargins(
                group_pad, group_pad, group_pad, group_pad
            )
            group_layout.setSpacing(window_pad(0.4))
            cards = QtWidgets.QVBoxLayout()
            cards.setContentsMargins(0, 0, 0, 0)
            cards.setSpacing(window_pad(0.4))
            group_layout.addLayout(cards)
            button = FluentButton("Add device", color=ACCENT)
            button.setObjectName(f"device_add_{_object_token(domain)}")
            button.clicked.connect(
                lambda _checked=False, value=domain: self._call(
                    self._controller.add, value
                )
            )
            group_layout.addWidget(button)
            self._domain_layouts[domain] = cards
            self._domain_add_buttons[domain] = button
            left.addWidget(group)
        left.addStretch(1)

        group_pad = window_pad(0.7)
        self.discovered_group = FluentGroupBox("Discovered hardware")
        discovered_layout = QtWidgets.QVBoxLayout(self.discovered_group)
        discovered_layout.setContentsMargins(
            group_pad, group_pad, group_pad, group_pad
        )
        discovered_layout.setSpacing(window_pad(0.4))
        self.discover_button = FluentButton("Scan hardware", color=ACCENT)
        self.discover_button.setObjectName("device_manager_discover")
        discovered_layout.addWidget(self.discover_button)
        self.discovered_cards_layout = QtWidgets.QVBoxLayout()
        self.discovered_cards_layout.setSpacing(window_pad(0.4))
        discovered_layout.addLayout(self.discovered_cards_layout)
        self.discovered_empty = muted_note_label("No discovered hardware")
        self.discovered_empty.setWordWrap(True)
        discovered_layout.addWidget(self.discovered_empty)
        right.addWidget(self.discovered_group)

        self.loaded_group = FluentGroupBox("Loaded session")
        self.loaded_layout = QtWidgets.QVBoxLayout(self.loaded_group)
        self.loaded_layout.setContentsMargins(
            group_pad, group_pad, group_pad, group_pad
        )
        self.loaded_layout.setSpacing(window_pad(0.4))
        self.loaded_empty = muted_note_label("No active installation")
        self.loaded_empty.setWordWrap(True)
        self.loaded_layout.addWidget(self.loaded_empty)
        right.addWidget(self.loaded_group)
        note = muted_note_label(
            "Live controls belong to Device Viewer; this window authors the "
            "installation graph."
        )
        note.setWordWrap(True)
        right.addWidget(note)
        right.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(scaled_px(7, minimum=5))
        self.new_combo = FluentComboBox()
        self.new_combo.setObjectName("device_manager_new_template")
        self.new_combo.addItem("New…", None)
        for name in discover_installation_templates():
            self.new_combo.addItem(name.replace("_", " ").title(), name)
        self.load_button = FluentButton("Load…", color=ORANGE)
        self.load_button.setObjectName("device_manager_load")
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_button.setObjectName("device_manager_save")
        self.save_as_button = FluentButton("Save as…", color=ACCENT)
        self.save_as_button.setObjectName("device_manager_save_as")
        self.apply_button = FluentButton("Apply", color=GREEN)
        self.apply_button.setObjectName("device_manager_apply")
        actions.addWidget(self.new_combo)
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.save_as_button)
        actions.addStretch(1)
        actions.addWidget(self.apply_button)
        page.addLayout(actions)

        self.status_strip = FluentStatusStrip()
        page.addWidget(self.status_strip)

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
        self.new_combo.activated.connect(self._new_selected)
        self.load_button.clicked.connect(self._load)
        self.save_button.clicked.connect(self._save)
        self.save_as_button.clicked.connect(self._save_as)
        self.apply_button.clicked.connect(
            lambda _checked=False: self._call(self._controller.apply)
        )
        self.discover_button.clicked.connect(
            lambda _checked=False: self._call(self._controller.discover)
        )
        self._controller.draft_changed.connect(self._draft_changed)
        self._controller.device_added.connect(self._device_added)
        self._controller.device_removed.connect(self._device_removed)
        self._controller.device_retyped.connect(self._device_retyped)
        self._controller.document_replaced.connect(self._reconcile_document)
        self._controller.discovery_changed.connect(self._sync_discovered)
        self._controller.runtime_changed.connect(self._sync_runtime)
        self._controller.busy_changed.connect(self._busy_changed)
        self._controller.operation_finished.connect(self._operation_finished)
        self._controller.status_changed.connect(self._show_status)
        self._controller.parameter_patch_staged.connect(
            self._insert_or_reconcile_card
        )
        self._owner_close_requested.connect(
            self._close_from_owner,
            type=QtCore.Qt.QueuedConnection,
        )

    @QtCore.pyqtSlot(int)
    def _new_selected(self, index: int) -> None:
        template = self.new_combo.itemData(index)
        with signals_blocked(self.new_combo):
            self.new_combo.setCurrentIndex(0)
        if isinstance(template, str):
            self._call(self._controller.replace_new, template)

    @QtCore.pyqtSlot(str, str)
    def _draft_changed(self, instance_id: str, key: str) -> None:
        del instance_id, key
        self._refresh_chrome()
        errors = self._controller.field_errors
        if errors:
            (device, field), message = next(iter(sorted(errors.items())))
            self.status_strip.show_message(
                f"{device}.{field}: {message}",
                severity="error",
            )

    @QtCore.pyqtSlot(str)
    def _device_added(self, instance_id: str) -> None:
        self._insert_or_reconcile_card(instance_id)
        self._reconcile_requirement_forms()
        self._refresh_discovered_buttons()
        self._refresh_chrome()

    @QtCore.pyqtSlot(str)
    def _device_removed(self, instance_id: str) -> None:
        card = self._device_cards.pop(instance_id)
        self._domain_layouts[card.domain].removeWidget(card)
        card.hide()
        card.deleteLater()
        self._reconcile_requirement_forms()
        self._refresh_discovered_buttons()
        self._refresh_chrome()

    @QtCore.pyqtSlot(str)
    def _device_retyped(self, instance_id: str) -> None:
        self._insert_or_reconcile_card(instance_id)
        self._reconcile_requirement_forms()
        self._refresh_chrome()

    @QtCore.pyqtSlot()
    def _reconcile_document(self) -> None:
        desired = self._controller.editor.instance_ids
        desired_set = set(desired)
        for instance_id in tuple(self._device_cards):
            if instance_id not in desired_set:
                card = self._device_cards.pop(instance_id)
                self._domain_layouts[card.domain].removeWidget(card)
                card.hide()
                card.deleteLater()
        for instance_id in desired:
            self._insert_or_reconcile_card(instance_id)
        by_domain: dict[str, list[str]] = {domain: [] for domain in self._domain_layouts}
        for instance_id in desired:
            by_domain[self._device_cards[instance_id].domain].append(instance_id)
        for domain, instance_ids in by_domain.items():
            layout = self._domain_layouts[domain]
            for index, instance_id in enumerate(instance_ids):
                layout.insertWidget(index, self._device_cards[instance_id])
        self._refresh_discovered_buttons()
        self._refresh_chrome()

    def _insert_or_reconcile_card(self, instance_id: str) -> None:
        row = self._controller.editor.row(instance_id)
        domain = device_type(row.type_id).domain
        card = self._device_cards.get(instance_id)
        if card is None:
            card = _DeviceEditorCard(self._controller, instance_id)
            self._device_cards[instance_id] = card
            self._domain_layouts[domain].addWidget(card)
        else:
            previous_domain = card.domain
            card.reconcile(row)
            if card.domain != previous_domain:
                self._domain_layouts[previous_domain].removeWidget(card)
                self._domain_layouts[card.domain].addWidget(card)
        card.set_busy(self._controller.busy)

    def _reconcile_requirement_forms(self) -> None:
        """Refresh only stable card forms after graph topology changes."""

        for instance_id, card in self._device_cards.items():
            card.reconcile(self._controller.editor.row(instance_id))

    @QtCore.pyqtSlot(object)
    def _sync_discovered(self, values) -> None:
        devices = tuple(values)
        desired = {value.instance_id: value for value in devices}
        for instance_id in tuple(self._discovered_cards):
            if instance_id not in desired:
                card = self._discovered_cards.pop(instance_id)
                self.discovered_cards_layout.removeWidget(card)
                card.hide()
                card.deleteLater()
        for index, value in enumerate(devices):
            if not isinstance(value, DeviceInstanceConfig):
                raise TypeError("discovery projection requires DeviceInstanceConfig")
            card = self._discovered_cards.get(value.instance_id)
            if card is None:
                card = _DiscoveredDeviceCard(value)
                card.add_requested.connect(self._add_discovered)
                self._discovered_cards[value.instance_id] = card
            else:
                card.reconcile(value)
            self.discovered_cards_layout.insertWidget(index, card)
        self.discovered_empty.setVisible(not devices)
        self._refresh_discovered_buttons()

    @QtCore.pyqtSlot(object)
    def _add_discovered(self, value: DeviceInstanceConfig) -> None:
        self._call(self._controller.add_discovered, value)

    def _refresh_discovered_buttons(self) -> None:
        configured = set(self._controller.editor.instance_ids)
        for instance_id, card in self._discovered_cards.items():
            card.add_button.setEnabled(
                not self._controller.busy and instance_id not in configured
            )

    @QtCore.pyqtSlot(object)
    def _sync_runtime(self, state: DeviceAdminState) -> None:
        catalog = state.catalog
        rows = () if catalog is None else tuple(catalog.items())
        desired = {instance_id for instance_id, _info in rows}
        for instance_id in tuple(self._loaded_cards):
            if instance_id not in desired:
                card = self._loaded_cards.pop(instance_id)
                self.loaded_layout.removeWidget(card)
                card.hide()
                card.deleteLater()
        for index, (instance_id, info) in enumerate(rows):
            card = self._loaded_cards.get(instance_id)
            if card is None:
                card = _DeviceSummaryCard()
                card.setObjectName(
                    f"device_loaded_{_object_token(instance_id)}"
                )
                self._loaded_cards[instance_id] = card
            descriptor = device_type(info.ref.type_id)
            capabilities = ", ".join(info.capabilities) or "no capabilities"
            card.update_content(
                info.role,
                f"{descriptor.label} · {instance_id} · {capabilities}",
            )
            self.loaded_layout.insertWidget(index, card)
        self.loaded_empty.setVisible(not rows)
        self._refresh_chrome()

    @QtCore.pyqtSlot(bool, str)
    def _busy_changed(self, busy: bool, label: str) -> None:
        self._refresh_chrome()
        if busy and label:
            self.status_strip.show_message(label, severity="task")

    @QtCore.pyqtSlot(str, bool)
    def _operation_finished(self, kind: str, success: bool) -> None:
        if not self._owner_close_pending:
            return
        if kind == "dispose":
            if success:
                self._finalize_owner_close()
            else:
                self._owner_close_pending = False
                if self._window is not None:
                    self._window.show()
            return
        QtCore.QTimer.singleShot(0, self._begin_owner_close)

    @QtCore.pyqtSlot(str, str)
    def _show_status(self, text: str, severity: str) -> None:
        mapped = severity if severity in FluentStatusStrip.SEVERITIES else "info"
        self.status_strip.show_message(str(text), severity=mapped)

    def _refresh_chrome(self) -> None:
        editor = self._controller.editor
        state = self._controller.state
        busy = self._controller.busy
        path = editor.path
        name = (
            "session config"
            if (
                path is None
                and state.active_config is not None
                and not editor.differs_from_active
            )
            else "untitled"
            if path is None
            else path.name
        )
        shown = f"{name}{'*' if editor.dirty else ''}"
        self.document_name.setText(shown)
        self.document_name.setToolTip(shown if path is None else str(path))

        if state.active_config is None:
            self.state_dot.set_color(GREY)
            self.state_dot.setToolTip("No active installation")
        elif editor.differs_from_active:
            self.state_dot.set_color(ORANGE)
            self.state_dot.setToolTip("Draft differs from the loaded session")
        else:
            self.state_dot.set_color(GREEN)
            self.state_dot.setToolTip("Draft matches the loaded session")

        errors = bool(self._controller.field_errors)
        self.save_button.set_dirty(editor.dirty, dirty_color=YELLOW)
        self.new_combo.setEnabled(not busy)
        self.load_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy and not errors and path is not None)
        self.save_as_button.setEnabled(not busy and not errors)
        needs_apply = (
            state.active_config is None or editor.differs_from_active
        )
        self.apply_button.setEnabled(
            not busy
            and not errors
            and needs_apply
            and state.can_apply
            and not state.closed
        )
        has_discovery = any(
            descriptor.discover is not None
            for descriptor in discover_device_types()
        )
        self.discover_button.setEnabled(not busy and has_discovery)
        self.discover_button.setToolTip(
            "" if has_discovery else "No installed device type declares discovery"
        )
        for button in self._domain_add_buttons.values():
            button.setEnabled(not busy)
        for card in self._device_cards.values():
            card.set_busy(busy)
        self._refresh_discovered_buttons()

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
        if not target.suffix:
            target = target.with_suffix(".json")
        self._call(self._controller.save_file, target)

    def _dialog_start_directory(self) -> Path:
        path = self._controller.editor.path
        return Path.home() if path is None else path.parent

    def _call(self, operation, *args) -> None:
        try:
            operation(*args)
        except BaseException as error:
            self.status_strip.show_message(
                f"{type(error).__name__}: {error}",
                severity="error",
            )

    def stage_parameter_patch(self, patch: SelectionParameterPatch) -> bool:
        """Stage a generic leaf patch in the visible Device Manager draft.

        This is intentionally not an Apply shortcut: the controller validates
        and reconciles the affected device card, while the operator still
        presses the ordinary Apply button at the device-owner boundary.
        """

        if not isinstance(patch, SelectionParameterPatch):
            raise TypeError("patch must be SelectionParameterPatch")
        return self._controller.stage_parameter_patch(patch)

    @property
    def permanently_closed(self) -> bool:
        return self._permanently_closed

    def restore_window(self) -> None:
        if self._permanently_closed or self._window is None:
            raise RuntimeError("Device manager is closed")
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    def wait_owner_closed(self, timeout: float) -> bool:
        return wait_for_owner_retirement(
            self,
            self._owner_closed,
            timeout=timeout,
        )

    def request_owner_close(self) -> None:
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
        if self._controller.busy:
            self.status_strip.show_message(
                "waiting for the device operation before closing",
                severity="task",
            )
            return
        self._begin_owner_close()

    def _begin_owner_close(self) -> None:
        try:
            self._controller.close()
        except BaseException as error:
            self._owner_close_pending = False
            self.status_strip.show_message(
                f"{type(error).__name__}: {error}",
                severity="error",
            )
            return

    def _finalize_owner_close(self) -> None:
        self._permanently_closed = True
        self._owner_close_pending = False
        window = self._window
        self._window = None
        if window is not None:
            release_window(window)
            window.hide()
            window.deleteLater()
        self._owner_closed.set()

    def _request_standalone_close(self) -> bool:
        if self._permanently_closed:
            return True
        self._owner_close_pending = True
        if self._controller.busy:
            self.status_strip.show_message(
                "wait for the device operation to finish before closing",
                severity="warning",
            )
            return False
        self._begin_owner_close()
        return False

    def _attach_window(self, window) -> None:
        self._window = window


def launch_device_manager_window(
    controller: DeviceManagerController,
    *,
    hide_on_close: bool = False,
) -> DeviceManagerWindowBody:
    """Launch DeviceManager through the shared Fluent window sequence."""

    ensure_qt_app()
    body = DeviceManagerWindowBody(controller)
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
