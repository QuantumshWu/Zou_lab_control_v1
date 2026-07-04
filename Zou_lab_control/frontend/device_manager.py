"""Device-manager panel: see every device the current config loaded, grouped by ROLE-TYPE, and
scan the hardware buses -- the GUI face of the device-DOMAIN registry.

It reads the ONE device source (a ``DeviceSet``) and the ONE role-type source
(``neutral_atom.devices.registry.device_domains``): a section per domain (Camera / Sequencer /
Trap array / a future RF source), each listing the loaded devices of that type.  Nothing here
names a concrete type -- registering a new device domain adds a section automatically.  Built
entirely from the shared Fluent primitives (``FluentGroupBox`` cards, ``FluentSettingRow`` rows,
``FluentButton`` scan) so it matches pulse_gui / task_console exactly.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .qt_fluent import (
    BG,
    GREEN,
    GREY,
    TEXT,
    FluentButton,
    FluentGroupBox,
    FluentLabel,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentWindow,
    fluent_font_size,
    scaled_px,
    setting_label_width,
    window_pad,
)


class DeviceManagerPanel(QtWidgets.QWidget):
    """List the loaded devices by role-type and scan the buses.  ``device_set`` is the ``DeviceSet``
    the session loaded; ``discover`` is the scan callable (defaults to ``discover_devices``)."""

    def __init__(self, device_set, *, discover=None, on_load_config=None, on_save_config=None,
                 on_open_devices=None, config_dir=None, parent=None):
        super().__init__(parent)
        self._device_set = device_set
        self._discover = discover
        # Session-wired actions (None when opened bare, e.g. show_device_manager(device_set) with no
        # session): load/save the experiment device config, open (initialise) the hardware.  Each is a
        # plain callback -- the panel stays session-agnostic, exactly like ``discover``.
        self._on_load_config = on_load_config
        self._on_save_config = on_save_config
        self._on_open_devices = on_open_devices
        self._config_dir = str(config_dir) if config_dir else ""
        outer = QtWidgets.QVBoxLayout(self)
        # SAME window-edge inset as every other GUI: ``window_pad(1)`` all four sides (title-aligned),
        # header<->body and card<->card gaps are HALF that unit (``window_pad(0.5)``).
        outer.setContentsMargins(*(window_pad(1),) * 4)
        outer.setSpacing(window_pad(0.5))

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(scaled_px(8, minimum=4))
        header.addWidget(FluentSectionLabel("Devices"))
        header.addStretch(1)
        self._scan_btn = FluentButton("Scan hardware", color=GREEN)
        self._scan_btn.clicked.connect(self._scan)
        header.addWidget(self._scan_btn)
        outer.addLayout(header)

        # Config toolbar: load / save the experiment device config + open (initialise) the hardware.
        # Only shown when the session wired the action -- a bare panel has nothing to load/save/open.
        if any((on_load_config, on_save_config, on_open_devices)):
            tools = QtWidgets.QHBoxLayout()
            tools.setSpacing(scaled_px(8, minimum=4))
            if on_open_devices is not None:
                b = FluentButton("Open devices", color=GREEN); b.clicked.connect(self._open_devices)
                tools.addWidget(b)
            tools.addStretch(1)
            if on_load_config is not None:
                b = FluentButton("Load config…"); b.clicked.connect(self._load_config); tools.addWidget(b)
            if on_save_config is not None:
                b = FluentButton("Save config…"); b.clicked.connect(self._save_config); tools.addWidget(b)
            outer.addLayout(tools)

        scroll = FluentScrollArea()
        body = QtWidgets.QWidget()
        self._body = QtWidgets.QVBoxLayout(body)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(scaled_px(8, minimum=4))
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        self._status = self._muted("")                  # one-line result of the last load/save/open action
        outer.addWidget(self._status)

        self._scan_card = None
        self._rebuild()

    def _set_status(self, text: str) -> None:
        self._status.setText(str(text))

    def _default_dir(self) -> str:
        return self._config_dir or ""

    # ---------------------------------------------------------------- experiment config load / save / open
    def _load_config(self) -> None:
        """Pick an experiment config JSON and rebuild the session's devices from it (via the wired
        ``on_load_config``), then re-list the -- now different -- devices.  A bad config leaves the
        session untouched (the callback builds the new set before swapping) and reports the error."""
        if self._on_load_config is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load experiment config", self._default_dir(), "Config (*.json);;All files (*)")
        if not path:
            return
        try:
            new_set = self._on_load_config(path)
        except Exception as exc:                         # a bad config must never crash the panel
            self._set_status(f"load failed: {exc}")
            return
        if new_set is not None:
            self._device_set = new_set
        self._rebuild()
        self._set_status(f"loaded {Path(path).name}")

    def _save_config(self) -> None:
        """Write the session's current device config to a JSON file (via ``on_save_config``) so
        ``na.connect(that_file)`` / Load config reproduces it."""
        if self._on_save_config is None:
            return
        start = str(Path(self._default_dir()) / "my_experiment.json") if self._default_dir() else "my_experiment.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save experiment config", start, "Config (*.json);;All files (*)")
        if not path:
            return
        try:
            written = self._on_save_config(path)
        except Exception as exc:
            self._set_status(f"save failed: {exc}")
            return
        self._set_status(f"saved {Path(str(written)).name}")

    def _open_devices(self) -> None:
        """Initialise / connect the loaded hardware (via ``on_open_devices``) and re-list the devices."""
        if self._on_open_devices is None:
            return
        try:
            new_set = self._on_open_devices()
        except Exception as exc:                         # a hardware open must never crash the panel
            self._set_status(f"open failed: {exc}")
            return
        if new_set is not None:
            self._device_set = new_set
        self._rebuild()
        self._set_status("devices opened")

    # ---------------------------------------------------------------- loaded devices, by domain
    def _rebuild(self) -> None:
        from ..neutral_atom.devices.registry import device_domains

        while self._body.count():
            item = self._body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._scan_card = None

        devices = getattr(self._device_set, "devices", {}) or {}
        for domain in device_domains():
            names = self._device_set.device_names(domain.base_type)
            card = FluentGroupBox(domain.label)
            v = QtWidgets.QVBoxLayout(card)
            v.setContentsMargins(*(scaled_px(10),) * 2, scaled_px(10), scaled_px(10))
            v.setSpacing(scaled_px(4, minimum=2))
            width = setting_label_width(list(names) or ["(none)"])
            if names:
                for name in names:
                    cls = type(devices.get(name)).__name__ if name in devices else "?"
                    v.addWidget(FluentSettingRow(name, FluentLabel(cls), label_width=width))
            else:
                v.addWidget(FluentSettingRow("(none)", self._muted("no device of this role loaded"),
                                             label_width=width))
            self._body.addWidget(card)
        self._body.addStretch(1)

    def _muted(self, text: str) -> QtWidgets.QLabel:
        lbl = FluentLabel(text)
        lbl.setStyleSheet(f"color: {GREY}; background: transparent; border: none; "
                          f'font: {fluent_font_size()}pt "Segoe UI";')
        return lbl

    # ---------------------------------------------------------------- hardware scan
    def _scan(self) -> None:
        discover = self._discover
        if discover is None:
            from ..neutral_atom.devices.discovery import discover_devices
            discover = discover_devices
        try:
            rows = list(discover(display=False))
        except Exception as exc:                       # a scan must never crash the panel (confocal contract)
            rows = []
            note = f"scan failed: {exc}"
        else:
            note = f"{len(rows)} found" if rows else "nothing on the buses"

        if self._scan_card is not None:
            self._scan_card.deleteLater()
        card = FluentGroupBox(f"Discovered ({note})")
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(*(scaled_px(10),) * 2, scaled_px(10), scaled_px(10))
        v.setSpacing(scaled_px(4, minimum=2))
        idents = [str(getattr(r, "ident", "")) for r in rows] or ["-"]
        width = setting_label_width(idents)
        for r in rows:
            label = str(getattr(r, "ident", "?"))
            ready = "ready" if getattr(r, "config", None) is not None else str(getattr(r, "label", ""))
            v.addWidget(FluentSettingRow(label, FluentLabel(ready), label_width=width))
        if not rows:
            v.addWidget(self._muted(note))
        # insert the scan results ABOVE the trailing stretch (last item)
        self._body.insertWidget(max(0, self._body.count() - 1), card)
        self._scan_card = card


def show_device_manager(device_set, *, discover=None, on_load_config=None, on_save_config=None,
                        on_open_devices=None, config_dir=None):
    """Open the device manager in a standalone Fluent window (parity with show_pulse_gui).

    ``on_load_config`` / ``on_save_config`` / ``on_open_devices`` (wired by the session through
    ``exp.device_manager()``) enable the Load config / Save config + Open-devices toolbar; omit them
    (a bare ``device_set``) and the window is view-only (device list + Scan hardware)."""
    panel = DeviceManagerPanel(device_set, discover=discover, on_load_config=on_load_config,
                               on_save_config=on_save_config, on_open_devices=on_open_devices,
                               config_dir=config_dir)
    window = FluentWindow(widget=panel, title="Devices@Zou lab")
    window.resize(scaled_px(460), scaled_px(560))
    window.show()
    return window
