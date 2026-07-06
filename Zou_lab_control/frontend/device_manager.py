"""Device manager: the experiment's DEVICE-INIT entry -- a real config EDITOR + hardware panel.

Two columns, confocal-GUI's DeviceManagerGUI shape on the house Fluent primitives:

* LEFT -- the CONFIG EDITOR.  One :class:`FluentGroupBox` per registered device DOMAIN
  (``device_domains()`` -- never a camera-special list) holding one collapsible card per
  config entry: role NAME (line edit), TYPE (a combo over ``device_class_registry()``
  filtered by the domain's base type; a class that fails to import is greyed with the
  reason, never a crash) and the class's OWN declared param form
  (``BaseDevice.config_params()`` -> the shared ``PARAM_WIDGETS`` registry -- typed
  controls, no free ``key: value`` text and NOTHING ever ``eval``'d).  ``$device:``
  cross-references render as a choice filled from the working config's other entries.
* RIGHT -- HARDWARE.  "Discovered": the bus scan (``discover_devices``), each row that
  carries a ready config gaining an "Add to config" button that inserts it on the left.
  "Loaded": the bound session's live instances with per-device Snapshot popups and the
  Open-devices action (present only when a session is bound).

The data model is the ONE round-trippable config dict ``{role: {"type", "params"}}``
(``read_config`` / ``load_devices`` / ``DeviceSet.to_config``): the editor holds a WORKING
copy; Save/Save-as writes it to JSON (independent of the live set); Load reads a JSON into
the editor; Apply hands it to the session (``session.load_config`` -- built first, so a bad
config leaves the session untouched).  Opened WITHOUT a session (``na.device_manager()``)
the Apply button reads "Init devices" and connects a NEW session from the working config,
upgrading this same window in place.  Dirty state is EVENT-DRIVEN (recomputed on each edit,
no polling): the header shows ``<config name>[*]`` (unsaved) plus a status dot -- grey =
no session, green = working config == the applied one, orange = unsynced edits.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace as _replace
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from ..neutral_atom.core.params import DEVICE_REF_PREFIX
from .param_widgets import PARAM_WIDGETS, ParamWidgetContext
from .qt_fluent import (
    GREEN,
    GREY,
    ORANGE,
    RED,
    YELLOW,
    ElidedLabel,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPopup,
    FluentReadoutMultiline,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusDot,
    batched_updates,
    ensure_qt_app,
    fluent_font_size,
    launch_fluent_window,
    scaled_px,
    set_fluent_scale,
    setting_label_width,
    signals_blocked,
    window_pad,
)

#: Window size in DESIGN px (scaled by the shared fluent scale at launch) -- the single
#: source the launcher and its contract test read.
WINDOW_SIZE = (960, 660)

_OTHER_SECTION = "Other devices"


def _registry():
    """The device registry module, reached lazily (it imports the virtual backend)."""
    from ..neutral_atom.devices import registry

    return registry


def _canon(cfg) -> str:
    """Canonical text of a config dict for EQUALITY (dirty/synced) checks only."""
    return json.dumps(cfg, sort_keys=True, default=str)


def _domain_of(cls, domains):
    """The registered domain a device class belongs to (first base-type match), or None."""
    if not isinstance(cls, type):
        return None
    for domain in domains:
        if issubclass(cls, domain.base_type):
            return domain
    return None


class _DeviceEntryCard(FluentFrame):
    """ONE config entry: ``[collapse | role name | type | Remove]`` header + the class's
    own declared param form.  The card never interprets a param itself -- every control is
    built/read by the entry class's ``config_params()`` through the ONE ``PARAM_WIDGETS``
    registry, and values round-trip through the class's ``config_to_form``/``form_to_config``."""

    def __init__(self, panel: "DeviceManagerPanel", name: str, entry: dict,
                 type_names, catalog: dict, expanded: bool, parent=None):
        super().__init__(parent, bordered=True)
        self._panel = panel
        self.entry_name = str(name)
        self.cls = None
        self.values: dict = {}
        self.decls: dict = {}
        self.handlers: dict = {}
        self.widgets: dict = {}

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(*(scaled_px(8),) * 4)
        v.setSpacing(scaled_px(4, minimum=2))

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(scaled_px(6, minimum=4))
        self._toggle = FluentButton("▾" if expanded else "▸", color=GREY)
        self._toggle.setFixedWidth(scaled_px(26))
        self._toggle.setToolTip("Show / hide this device's parameters.")
        self._toggle.clicked.connect(self._on_toggle)
        header.addWidget(self._toggle)
        self._name_edit = FluentLineEdit(self.entry_name)
        self._name_edit.setMinimumWidth(scaled_px(104, minimum=84))
        self._name_edit.setToolTip("Config ROLE name (the key measurements bind to; "
                                   "\"camera\" / \"sequencer\" are the conventional defaults).")
        self._name_edit.editingFinished.connect(
            lambda: panel._rename_entry(self, self._name_edit.text()))
        header.addWidget(self._name_edit, 1)
        self._type_combo = FluentComboBox()
        for type_name in type_names:
            self._type_combo.addItem(str(type_name))
            cls, err = catalog.get(str(type_name), (None, ""))
            if cls is None:
                item = self._type_combo.model().item(self._type_combo.count() - 1)
                if item is not None:                     # greyed, with the import failure as reason
                    item.setEnabled(False)
                    item.setToolTip(f"import failed: {err}" if err else "class not importable")
        current_type = str(entry.get("type", ""))
        if current_type and self._type_combo.findText(current_type) < 0:
            self._type_combo.addItem(current_type)       # keep an out-of-registry type visible
        with signals_blocked(self._type_combo):
            self._type_combo.setCurrentText(current_type)
        self._type_combo.setToolTip("Device class; picking one prefills its declared param form.")
        self._type_combo.activated.connect(
            lambda *_: panel._retype_entry(self, self._type_combo.currentText()))
        header.addWidget(self._type_combo, 1)
        remove_btn = FluentButton("Remove", color=RED)
        remove_btn.clicked.connect(lambda: panel._remove_entry(self))
        header.addWidget(remove_btn)
        v.addLayout(header)

        self._body = QtWidgets.QWidget()
        self._body.setStyleSheet("background: transparent;")
        body_v = QtWidgets.QVBoxLayout(self._body)
        body_v.setContentsMargins(0, 0, 0, 0)
        body_v.setSpacing(scaled_px(4, minimum=2))
        v.addWidget(self._body)
        self._body.setVisible(bool(expanded))
        self._build_form(entry, catalog)

    # ------------------------------------------------------------------ form
    def _build_form(self, entry: dict, catalog: dict) -> None:
        body = self._body.layout()
        cls, err = catalog.get(str(entry.get("type", "")), (None, ""))
        self.cls = cls
        if cls is None:
            self.values = dict(entry.get("params") or {})   # raw params kept verbatim
            note = FluentLabel(f"class not importable: {err}" if err else "unknown device class")
            note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            body.addWidget(note)
            return
        self.values = dict(cls.config_to_form(dict(entry.get("params") or {})))
        decls = [self._panel._with_device_choices(d, self.entry_name) for d in cls.config_params()]
        width = setting_label_width([self._row_label(d) for d in decls] or [""])
        ctx = ParamWidgetContext(
            on_change=self._panel._refresh_state,
            instant_apply=lambda key, value, card=self: self._panel._apply_param(card, key, value),
        )
        for decl in decls:
            handler = PARAM_WIDGETS[decl.kind]
            widget = handler.build(decl, self.values.get(decl.key), ctx)
            body.addWidget(FluentSettingRow(self._row_label(decl), widget, label_width=width))
            self.decls[decl.key] = decl
            self.handlers[decl.key] = handler
            self.widgets[decl.key] = widget

    @staticmethod
    def _row_label(decl) -> str:
        text = decl.label or decl.key
        if decl.unit:
            text += f" ({decl.unit})"
        if decl.required:
            text += " *"
        return text

    def _on_toggle(self) -> None:
        expanded = not self._body.isVisible()
        self._body.setVisible(expanded)
        self._toggle.setText("▾" if expanded else "▸")
        self._panel._set_expanded(self.entry_name, expanded)


class DeviceManagerPanel(QtWidgets.QWidget):
    """The device-manager body (see the module docstring for the full design).

    ``device_set`` seeds the working config from a live set's ``to_config()`` when neither
    ``initial_config`` nor a ``session_binding`` is given (a bare view still edits/saves).
    ``session_binding`` is the duck-typed callback bundle a session wires (``na._gui``):
    ``{"devices": () -> DeviceSet, "apply_config": (dict) -> DeviceSet, "open_devices":
    () -> DeviceSet}`` -- the panel stays session-agnostic.  ``on_init_session`` (the
    no-session entry) takes the working config dict and returns a fresh session's binding;
    on success the window upgrades in place.  ``discover`` overrides the bus scanner."""

    def __init__(self, device_set=None, *, initial_config=None, session_binding=None,
                 on_init_session=None, discover=None, config_dir=None, parent=None):
        super().__init__(parent)
        self._discover = discover
        self._binding = dict(session_binding) if session_binding else None
        self._on_init_session = on_init_session
        self._config_dir = str(config_dir) if config_dir else ""
        self._expanded: set[str] = set()
        self._config_path: str | None = None

        if initial_config is not None:
            self._working = deepcopy(dict(initial_config))
            self._config_name = "untitled"
        elif self._binding is not None:
            self._working = deepcopy(dict(self._binding["devices"]().to_config()))
            self._config_name = "session config"
        elif device_set is not None and callable(getattr(device_set, "to_config", None)):
            self._working = deepcopy(dict(device_set.to_config()))
            self._config_name = "session config"
        else:
            self._working = {}
            self._config_name = "untitled"
        # the dirty-star reference: what the editor last opened/saved (see _refresh_state)
        self._saved_ref = deepcopy(self._working)
        # the synced-dot reference: what the bound session is actually running
        self._applied = deepcopy(self._working) if self._binding is not None else None

        outer = QtWidgets.QVBoxLayout(self)
        # SAME window-edge inset as every other GUI: window_pad(1) all four sides; the
        # header<->body and card<->card gaps are HALF that unit (window_pad(0.5)).
        outer.setContentsMargins(*(window_pad(1),) * 4)
        outer.setSpacing(window_pad(0.5))

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(scaled_px(8, minimum=4))
        header.addWidget(FluentSectionLabel("Devices"))
        header.addStretch(1)
        self._dot = FluentStatusDot(color=GREY, size=12)
        self._dot.setToolTip("grey = no session · green = applied · orange = unsynced edits")
        header.addWidget(self._dot)
        self._name_label = FluentLabel("")
        header.addWidget(self._name_label)
        outer.addLayout(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(window_pad(0.5))

        editor_scroll = FluentScrollArea()
        editor_host = QtWidgets.QWidget()
        self._editor_body = QtWidgets.QVBoxLayout(editor_host)
        self._editor_body.setContentsMargins(0, 0, 0, 0)
        self._editor_body.setSpacing(scaled_px(8, minimum=4))
        editor_scroll.setWidget(editor_host)
        editor_scroll.setWidgetResizable(True)
        body.addWidget(editor_scroll, 3)

        right_scroll = FluentScrollArea()
        right_host = QtWidgets.QWidget()
        self._right_body = QtWidgets.QVBoxLayout(right_host)
        self._right_body.setContentsMargins(0, 0, 0, 0)
        self._right_body.setSpacing(scaled_px(8, minimum=4))
        right_scroll.setWidget(right_host)
        right_scroll.setWidgetResizable(True)
        body.addWidget(right_scroll, 2)
        outer.addLayout(body, 1)

        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(scaled_px(8, minimum=4))
        self._new_combo = FluentComboBox()
        self._new_combo.setToolTip("Start a fresh working config from a template "
                                   "(the bundled configs + virtual).")
        self._reset_new_combo()
        self._new_combo.activated.connect(self._on_new_template)
        tools.addWidget(self._new_combo)
        load_btn = FluentButton("Load…", color=ORANGE)
        load_btn.clicked.connect(self._load)
        tools.addWidget(load_btn)
        tools.addStretch(1)
        self._save_btn = FluentButton("Save")
        self._save_btn.clicked.connect(lambda: self._save(save_as=False))
        tools.addWidget(self._save_btn)
        save_as_btn = FluentButton("Save as…")
        save_as_btn.clicked.connect(lambda: self._save(save_as=True))
        tools.addWidget(save_as_btn)
        self._apply_btn = FluentButton(
            "Apply" if self._binding is not None else
            ("Init devices" if self._on_init_session is not None else "Apply"), color=GREEN)
        self._apply_btn.clicked.connect(self._apply)
        tools.addWidget(self._apply_btn)
        outer.addLayout(tools)

        self._status = self._muted("")           # one-line result of the last action
        outer.addWidget(self._status)

        self._discovered_card = None
        self._discovered_body = None
        self._loaded_card = None
        self._loaded_body = None
        self._build_right_column()
        self._rebuild_editor()
        self._refresh_state()

    # ------------------------------------------------------------------ model access
    def working_config(self) -> dict:
        """A deep copy of the WORKING config dict (the round-trippable
        ``{role: {"type", "params"}}`` that ``read_config``/``load_devices`` consume)."""
        return deepcopy(self._working)

    def _set_status(self, text: str) -> None:
        self._status.setText(str(text))

    def _set_expanded(self, name: str, expanded: bool) -> None:
        (self._expanded.add if expanded else self._expanded.discard)(name)

    def _muted(self, text: str) -> QtWidgets.QLabel:
        lbl = FluentLabel(text)
        lbl.setStyleSheet(f"color: {GREY}; background: transparent; border: none; "
                          f'font: {fluent_font_size()}pt "Segoe UI";')
        return lbl

    # ------------------------------------------------------------------ registry views
    def _catalog(self) -> dict:
        """``{class name: (cls | None, import error)}`` over the device registry -- an
        import failure is CARRIED (a greyed combo row with its reason), never raised."""
        reg = _registry()
        out: dict = {}
        for name in reg.device_class_registry():
            try:
                out[name] = (reg.resolve_class(name), "")
            except Exception as exc:
                out[name] = (None, str(exc))
        return out

    def _type_names_for(self, domain, catalog: dict) -> list:
        """The type-combo choices for a section: the domain's importable classes, plus
        every class that FAILED to import (greyed; its domain is unknowable without the
        class object)."""
        names = []
        for name, (cls, _err) in sorted(catalog.items()):
            if cls is None or domain is None or issubclass(cls, domain.base_type):
                names.append(name)
        return names

    def _with_device_choices(self, decl, entry_name: str):
        """Fill a ``device``-kind decl's choices from the working config: a blank (= unset)
        plus ``$device:<other entry>`` for every OTHER entry -- the editor owns this (only
        it knows the live config), exactly as the decl contract states."""
        if decl.kind != "device":
            return decl
        refs = [f"{DEVICE_REF_PREFIX}{name}" for name in self._working if name != entry_name]
        return _replace(decl, choices=tuple([""] + refs))

    # ------------------------------------------------------------------ left column (editor)
    def _rebuild_editor(self) -> None:
        reg = _registry()
        domains = reg.device_domains()
        catalog = self._catalog()
        by_section: dict = {d.key: [] for d in domains}
        other: list = []
        for name, entry in self._working.items():
            domain = _domain_of(catalog.get(str(entry.get("type", "")), (None, ""))[0], domains)
            (by_section[domain.key] if domain is not None else other).append(name)

        with batched_updates(self):
            while self._editor_body.count():
                item = self._editor_body.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self._entry_cards: list[_DeviceEntryCard] = []
            for domain in domains:
                self._editor_body.addWidget(
                    self._section_card(domain.label, by_section[domain.key], catalog,
                                       domain=domain))
            if other:
                self._editor_body.addWidget(
                    self._section_card(_OTHER_SECTION, other, catalog, domain=None))
            self._editor_body.addStretch(1)

    def _section_card(self, label: str, names, catalog: dict, *, domain) -> FluentGroupBox:
        card = FluentGroupBox(label)
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(*(scaled_px(10),) * 2, scaled_px(10), scaled_px(10))
        v.setSpacing(scaled_px(6, minimum=3))
        type_names = self._type_names_for(domain, catalog)
        for name in names:
            entry_card = _DeviceEntryCard(self, name, self._working[name], type_names,
                                          catalog, expanded=name in self._expanded)
            self._entry_cards.append(entry_card)
            v.addWidget(entry_card)
        if not names:
            v.addWidget(self._muted("no device of this role in the config"))
        if domain is not None:
            add_btn = FluentButton(f"+ Add {domain.label.lower()}")
            add_btn.clicked.connect(lambda _c=False, d=domain: self._add_entry(d))
            v.addWidget(add_btn, 0, QtCore.Qt.AlignLeft)
        return card

    # ------------------------------------------------------------------ entry mutations
    def _unique_name(self, base: str) -> str:
        base = str(base).strip() or "device"
        if base not in self._working:
            return base
        index = 2
        while f"{base}_{index}" in self._working:
            index += 1
        return f"{base}_{index}"

    def _add_entry(self, domain) -> None:
        catalog = self._catalog()
        importable = [n for n, (cls, _e) in sorted(catalog.items())
                      if cls is not None and issubclass(cls, domain.base_type)]
        # prefer a virtual class (constructs with no hardware attached) for a fresh row
        virtual = [n for n in importable if n.startswith("Virtual")]
        type_name = (virtual or importable or [""])[0]
        name = self._unique_name(domain.key)
        self._working[name] = {"type": type_name, "params": {}}
        self._expanded.add(name)
        self._rebuild_editor()
        self._refresh_state()
        self._set_status(f"added {name} ({type_name})")

    def _remove_entry(self, card: _DeviceEntryCard) -> None:
        self._working.pop(card.entry_name, None)
        self._expanded.discard(card.entry_name)
        self._rebuild_editor()
        self._refresh_state()
        self._set_status(f"removed {card.entry_name}")

    def _rename_entry(self, card: _DeviceEntryCard, new_name: str) -> None:
        old, new = card.entry_name, str(new_name).strip()
        if new == old:
            return
        if not new or new in self._working:
            self._set_status(f"name {new!r} is empty or already used")
            with signals_blocked(card._name_edit):
                card._name_edit.setText(old)
            return
        self._working = {new if key == old else key: value
                         for key, value in self._working.items()}
        if old in self._expanded:
            self._expanded.discard(old)
            self._expanded.add(new)
        # $device: references FOLLOW the rename so the config graph stays wired
        for entry in self._working.values():
            params = entry.get("params")
            if isinstance(params, dict):
                entry["params"] = self._renamed_refs(params, old, new)
        self._rebuild_editor()
        self._refresh_state()

    def _renamed_refs(self, value, old: str, new: str):
        target = f"{DEVICE_REF_PREFIX}{old}"
        if isinstance(value, str):
            return f"{DEVICE_REF_PREFIX}{new}" if value == target else value
        if isinstance(value, list):
            return [self._renamed_refs(v, old, new) for v in value]
        if isinstance(value, dict):
            return {k: self._renamed_refs(v, old, new) for k, v in value.items()}
        return value

    def _retype_entry(self, card: _DeviceEntryCard, type_name: str) -> None:
        entry = self._working.get(card.entry_name)
        if entry is None:
            return
        entry["type"] = str(type_name)
        new_cls, _err = self._catalog().get(str(type_name), (None, ""))
        if new_cls is not None:
            # prefill = the NEW class's declared form; values whose key survives carry over
            new_keys = {d.key for d in new_cls.config_params()}
            kept = {k: v for k, v in card.values.items() if k in new_keys and v is not None}
            entry["params"] = new_cls.form_to_config(kept)
        self._expanded.add(card.entry_name)
        self._rebuild_editor()
        self._refresh_state()

    def _apply_param(self, card: _DeviceEntryCard, key: str, value) -> None:
        """A form edit writes THROUGH to the working config (event-driven dirty: no
        polling): store the flat form value, then regroup via the class's own mapping."""
        card.values[key] = value
        entry = self._working.get(card.entry_name)
        if entry is None or card.cls is None:
            return
        entry["params"] = card.cls.form_to_config(self._clean_values(card))
        self._refresh_state()

    @staticmethod
    def _clean_values(card: _DeviceEntryCard) -> dict:
        """Form values worth WRITING into the entry: ``None`` (blank optional) is omitted
        -- the constructor default applies -- and an unset device reference likewise."""
        out = {}
        for key, value in card.values.items():
            decl = card.decls.get(key)
            if value is None:
                continue
            if decl is not None and decl.kind == "device" and not str(value).strip():
                continue
            out[key] = value
        return out

    # ------------------------------------------------------------------ right column
    def _build_right_column(self) -> None:
        self._discovered_card = FluentGroupBox("Discovered")
        v = QtWidgets.QVBoxLayout(self._discovered_card)
        v.setContentsMargins(*(scaled_px(10),) * 2, scaled_px(10), scaled_px(10))
        v.setSpacing(scaled_px(4, minimum=2))
        scan_btn = FluentButton("Scan hardware", color=GREEN)
        scan_btn.clicked.connect(self._scan)
        v.addWidget(scan_btn, 0, QtCore.Qt.AlignLeft)
        self._discovered_body = QtWidgets.QVBoxLayout()
        self._discovered_body.setSpacing(scaled_px(4, minimum=2))
        v.addLayout(self._discovered_body)
        self._right_body.addWidget(self._discovered_card)

        self._loaded_card = FluentGroupBox("Loaded (session)")
        lv = QtWidgets.QVBoxLayout(self._loaded_card)
        lv.setContentsMargins(*(scaled_px(10),) * 2, scaled_px(10), scaled_px(10))
        lv.setSpacing(scaled_px(4, minimum=2))
        self._loaded_body = QtWidgets.QVBoxLayout()
        self._loaded_body.setSpacing(scaled_px(4, minimum=2))
        lv.addLayout(self._loaded_body)
        self._right_body.addWidget(self._loaded_card)
        self._right_body.addStretch(1)
        self._rebuild_loaded()

    def _scan(self) -> None:
        discover = self._discover
        if discover is None:
            from ..neutral_atom.devices.discovery import discover_devices
            discover = discover_devices
        try:
            rows = list(discover(display=False))
        except Exception as exc:                 # a scan must never crash the panel
            rows, note = [], f"scan failed: {exc}"
        else:
            note = f"{len(rows)} found" if rows else "nothing on the buses"
        while self._discovered_body.count():
            item = self._discovered_body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for row in rows:
            line = QtWidgets.QHBoxLayout()
            line.setSpacing(scaled_px(6, minimum=4))
            ident = str(getattr(row, "ident", "?"))
            label = str(getattr(row, "label", ""))
            line.addWidget(FluentLabel(ident), 0, QtCore.Qt.AlignTop)
            note_label = self._muted(label)
            note_label.setWordWrap(True)     # a long driver note WRAPS -- it must never widen
            line.addWidget(note_label, 1)    # the column and push controls past the viewport
            if getattr(row, "config", None) is not None:
                add_btn = FluentButton("Add to config")
                add_btn.clicked.connect(lambda _c=False, r=row: self._add_discovered(r))
                line.addWidget(add_btn)
            host = QtWidgets.QWidget()
            host.setStyleSheet("background: transparent;")
            host.setLayout(line)
            self._discovered_body.addWidget(host)
        if not rows:
            self._discovered_body.addWidget(self._muted(note))
        self._set_status(f"scan: {note}")

    def _add_discovered(self, row) -> None:
        """Insert a discovered row's READY config entry into the working config, named by
        its device domain (``camera`` / ``camera_2`` / ...)."""
        config = deepcopy(getattr(row, "config", None) or {})
        if not config.get("type"):
            self._set_status("row carries no ready config")
            return
        reg = _registry()
        domain = _domain_of(self._catalog().get(str(config["type"]), (None, ""))[0],
                            reg.device_domains())
        name = self._unique_name(domain.key if domain else str(getattr(row, "kind", "device")))
        self._working[name] = {"type": str(config["type"]),
                               "params": dict(config.get("params") or {})}
        self._expanded.add(name)
        self._rebuild_editor()
        self._refresh_state()
        self._set_status(f"added {name} from {getattr(row, 'ident', '?')}")

    def _rebuild_loaded(self) -> None:
        while self._loaded_body.count():
            item = self._loaded_body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if self._binding is None:
            self._loaded_card.setVisible(False)
            return
        self._loaded_card.setVisible(True)
        open_btn = FluentButton("Open devices", color=GREEN)
        open_btn.setToolTip("Initialise / connect the loaded hardware (cameras last).")
        open_btn.clicked.connect(self._open_devices)
        self._loaded_body.addWidget(open_btn, 0, QtCore.Qt.AlignLeft)
        try:
            device_set = self._binding["devices"]()
            devices = dict(getattr(device_set, "devices", {}) or {})
        except Exception as exc:
            self._loaded_body.addWidget(self._muted(f"session devices unavailable: {exc}"))
            return
        width = setting_label_width(list(devices) or ["(none)"])
        for name in sorted(devices):
            device = devices[name]
            line = QtWidgets.QHBoxLayout()
            line.setSpacing(scaled_px(6, minimum=4))
            control = QtWidgets.QWidget()
            control.setStyleSheet("background: transparent;")
            control.setLayout(line)
            # elides (full name in the tooltip) so the row can never widen the column into
            # a horizontal scroll of the whole right panel
            line.addWidget(ElidedLabel(type(device).__name__), 1)
            snap_btn = FluentButton("Snapshot")
            snap_btn.clicked.connect(
                lambda _c=False, n=name, d=device, b=snap_btn: self._show_snapshot(n, d, b))
            line.addWidget(snap_btn)
            self._loaded_body.addWidget(FluentSettingRow(name, control, label_width=width))
        if not devices:
            self._loaded_body.addWidget(self._muted("no device loaded"))

    def _open_devices(self) -> None:
        if self._binding is None:
            return
        try:
            self._binding["open_devices"]()
        except Exception as exc:                 # a hardware open must never crash the panel
            self._set_status(f"open failed: {exc}")
            return
        self._rebuild_loaded()
        self._set_status("devices opened")

    def _show_snapshot(self, name: str, device, anchor: QtWidgets.QWidget) -> None:
        """A read-only ``device.snapshot()`` tree in a FluentPopup (indentation = nesting;
        selectable for copy, never editable)."""
        try:
            snapshot = device.snapshot()
        except Exception as exc:
            self._set_status(f"snapshot failed: {exc}")
            return
        popup = FluentPopup(self)
        v = QtWidgets.QVBoxLayout(popup)
        v.setContentsMargins(*(scaled_px(10),) * 4)
        v.setSpacing(scaled_px(6, minimum=3))
        v.addWidget(FluentSectionLabel(f"{name} snapshot"))
        scroll = FluentScrollArea()
        text = FluentReadoutMultiline(json.dumps(snapshot, indent=2, default=str))
        scroll.setWidget(text)
        scroll.setWidgetResizable(True)
        v.addWidget(scroll, 1)
        popup.setFixedSize(scaled_px(340), scaled_px(360))
        popup.move(anchor.mapToGlobal(QtCore.QPoint(0, anchor.height() + scaled_px(4))))
        popup.show()

    # ------------------------------------------------------------------ config file actions
    def _reset_new_combo(self) -> None:
        reg = _registry()
        with signals_blocked(self._new_combo):
            self._new_combo.clear()
            self._new_combo.addItem("New…")
            self._new_combo.addItem("(empty)")
            try:
                for name in reg.available_device_configs():
                    self._new_combo.addItem(str(name))
            except Exception:
                pass
            self._new_combo.setCurrentIndex(0)

    def _on_new_template(self, index: int) -> None:
        if index <= 0:
            return
        name = self._new_combo.itemText(index)
        try:
            cfg = {} if name == "(empty)" else dict(_registry().read_config(name))
        except Exception as exc:
            self._set_status(f"template failed: {exc}")
            self._reset_new_combo()
            return
        self._adopt_config(cfg, name="untitled" if name == "(empty)" else name, path=None)
        self._reset_new_combo()
        self._set_status(f"new config from {name}")

    def _adopt_config(self, cfg: dict, *, name: str, path: str | None) -> None:
        self._working = deepcopy(dict(cfg))
        self._saved_ref = deepcopy(self._working)
        self._config_name = str(name)
        self._config_path = path
        self._expanded.clear()
        self._rebuild_editor()
        self._refresh_state()

    def _default_dir(self) -> str:
        return self._config_dir or ""

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load device config", self._default_dir(), "Config (*.json);;All files (*)")
        if not path:
            return
        try:
            cfg = dict(_registry().read_config(path))
        except Exception as exc:                 # a bad file must never crash the panel
            self._set_status(f"load failed: {exc}")
            return
        self._adopt_config(cfg, name=Path(path).stem, path=path)
        self._set_status(f"loaded {Path(path).name} (Apply to run it)")

    def _save(self, *, save_as: bool) -> None:
        path = self._config_path
        if save_as or not path:
            start = (str(Path(self._default_dir()) / "my_experiment.json")
                     if self._default_dir() else "my_experiment.json")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save device config", start, "Config (*.json);;All files (*)")
            if not path:
                return
        path = str(Path(path).with_suffix(".json"))
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(self._working, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        except Exception as exc:
            self._set_status(f"save failed: {exc}")
            return
        self._config_path = path
        self._config_name = Path(path).stem
        self._saved_ref = deepcopy(self._working)
        self._refresh_state()
        self._set_status(f"saved {Path(path).name}")

    # ------------------------------------------------------------------ apply / init
    def _validation_problems(self) -> list[str]:
        problems = []
        for card in getattr(self, "_entry_cards", []):
            for key, widget in card.widgets.items():
                decl, handler = card.decls[key], card.handlers[key]
                if decl.required and handler.is_empty(widget):
                    problems.append(f"{card.entry_name}.{key} is required")
                    continue
                try:
                    handler.read(widget)
                except ValueError as exc:        # e.g. a json field mid-garbage
                    problems.append(f"{card.entry_name}.{key}: {exc}")
        return problems

    def _apply(self) -> None:
        problems = self._validation_problems()
        if problems:
            self._set_status("; ".join(problems))
            return
        cfg = deepcopy(self._working)
        if self._binding is None and self._on_init_session is not None:
            try:
                binding = self._on_init_session(cfg)
            except Exception as exc:             # a bad config never crashes the editor
                self._set_status(f"init failed: {exc}")
                return
            if binding:
                self._binding = dict(binding)
                self._apply_btn.setText("Apply")
            self._applied = deepcopy(cfg)
            self._rebuild_loaded()
            self._refresh_state()
            self._set_status(f"initialised {len(cfg)} device(s)")
            return
        if self._binding is None:
            self._set_status("no session bound -- Save this config and na.connect(<path>) it")
            return
        try:
            self._binding["apply_config"](cfg)   # session builds the NEW set FIRST (bad config = no-op)
        except Exception as exc:
            self._set_status(f"apply failed: {exc}")
            return
        self._applied = deepcopy(cfg)
        self._rebuild_loaded()
        self._refresh_state()
        self._set_status(f"applied {len(cfg)} device(s)")

    # ------------------------------------------------------------------ event-driven state
    def _refresh_state(self) -> None:
        dirty = _canon(self._working) != _canon(self._saved_ref)
        self._name_label.setText(f"{self._config_name}{'*' if dirty else ''}")
        self._save_btn.set_dirty(dirty, dirty_color=YELLOW)
        if self._binding is None and self._on_init_session is None:
            self._dot.set_color(GREY)
        elif self._applied is not None and _canon(self._working) == _canon(self._applied):
            self._dot.set_color(GREEN)
        else:
            self._dot.set_color(ORANGE)


def show_device_manager(device_set=None, *, initial_config=None, session_binding=None,
                        on_init_session=None, discover=None, config_dir=None):
    """Open the device manager in a standalone Fluent window (parity with show_pulse_gui).

    ``session_binding`` (wired by ``exp.device_manager()`` through ``na._gui``) binds a live
    session: Apply routes to ``session.load_config`` and the Loaded card lists its devices.
    ``on_init_session`` (wired by ``na.device_manager()``) is the NO-session entry: the
    Apply button reads "Init devices" and connects a fresh session from the working config.
    A bare ``device_set`` opens a view/edit-only editor seeded from ``to_config()``."""
    # The app + the shared fluent scale must exist BEFORE the panel ctor (a QWidget needs the
    # QApplication; scaled_px reads the process-global scale at construction).
    ensure_qt_app()
    set_fluent_scale(None)   # the ONE shared GUI-scale rule (auto from the primary screen)
    panel = DeviceManagerPanel(device_set, initial_config=initial_config,
                               session_binding=session_binding,
                               on_init_session=on_init_session,
                               discover=discover, config_dir=config_dir)
    # ONE launcher sequence (launch_fluent_window: wrap -> size -> centre -> show -> retain).
    # destroy-on-close (hide_on_close=False) is deliberate: exp.device_manager() rebuilds a
    # fresh panel per open (CC round).  The explicit size replaces fixed_size -- the editor
    # is a scroll area whose content hint would collapse the window.
    return launch_fluent_window(panel, title="Devices@Zou lab", fixed_size=False,
                                size=(scaled_px(WINDOW_SIZE[0]), scaled_px(WINDOW_SIZE[1])))
