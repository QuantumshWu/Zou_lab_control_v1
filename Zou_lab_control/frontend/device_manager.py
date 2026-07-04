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
)


class DeviceManagerPanel(QtWidgets.QWidget):
    """List the loaded devices by role-type and scan the buses.  ``device_set`` is the ``DeviceSet``
    the session loaded; ``discover`` is the scan callable (defaults to ``discover_devices``)."""

    def __init__(self, device_set, *, discover=None, parent=None):
        super().__init__(parent)
        self._device_set = device_set
        self._discover = discover
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(*(scaled_px(10),) * 4)
        outer.setSpacing(scaled_px(8, minimum=4))

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(scaled_px(8, minimum=4))
        header.addWidget(FluentSectionLabel("Devices"))
        header.addStretch(1)
        self._scan_btn = FluentButton("Scan hardware", color=GREEN)
        self._scan_btn.clicked.connect(self._scan)
        header.addWidget(self._scan_btn)
        outer.addLayout(header)

        scroll = FluentScrollArea()
        body = QtWidgets.QWidget()
        self._body = QtWidgets.QVBoxLayout(body)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(scaled_px(8, minimum=4))
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        self._scan_card = None
        self._rebuild()

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


def show_device_manager(device_set, *, discover=None):
    """Open the device manager in a standalone Fluent window (parity with show_pulse_gui)."""
    panel = DeviceManagerPanel(device_set, discover=discover)
    window = FluentWindow(widget=panel, title="Devices@Zou lab")
    window.resize(scaled_px(420), scaled_px(560))
    window.show()
    return window
