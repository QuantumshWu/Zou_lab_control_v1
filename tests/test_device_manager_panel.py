"""The device-manager panel is type-generic: the config EDITOR has a section per REGISTERED
device domain, driven by device_domains() -- never a camera-special list.  Registering a new
domain adds a section with zero panel edits.  These contracts pin that + the entry points."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets

import Zou_lab_control.neutral_atom as na
import Zou_lab_control.frontend as zf
from Zou_lab_control.frontend.device_manager import DeviceManagerPanel
from Zou_lab_control.frontend.qt_fluent import FluentGroupBox
from Zou_lab_control.neutral_atom.devices.registry import device_domains, load_devices


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _editor_cards(panel):
    return [panel._editor_body.itemAt(i).widget() for i in range(panel._editor_body.count())
            if isinstance(panel._editor_body.itemAt(i).widget(), FluentGroupBox)]


def test_editor_has_one_section_per_registered_domain():
    devices = load_devices("virtual")
    panel = DeviceManagerPanel(devices)
    titles = [c.title() for c in _editor_cards(panel)]
    # ONE editor card per domain -- type-generic, driven by the registry (add a domain -> a card)
    assert titles == [d.label for d in device_domains()]
    assert panel.grab().width() > 0                       # renders
    # every virtual entry landed in a domain section (no "Other devices" card needed)
    assert "Other devices" not in titles


def test_scan_populates_the_discovered_card_without_crashing():
    devices = load_devices("virtual")
    panel = DeviceManagerPanel(devices)
    panel._scan()                                          # scans real buses, must not raise
    assert panel._discovered_body.count() >= 1             # rows or the "nothing found" note


def test_device_manager_entry_points():
    """TWO public entries, one window per session: ``exp.device_manager()`` (bound) and the
    module-level ``na.device_manager()`` (device-INIT, no session yet).  The widget factory
    itself stays OFF the public ``zf`` surface -- it is inherently coupled to na's device
    registry, so na's two wrappers are the only doors."""
    assert "show_device_manager" not in zf.__all__
    for gone in ("show_device_manager", "DeviceManagerPanel"):
        assert not hasattr(zf, gone), f"{gone} must not be a public zf attribute"
    assert callable(na.device_manager)                     # the module-level init entry
    assert "device_manager" in na.__all__
    exp = na.connect("virtual")
    try:
        assert callable(exp.device_manager)
        window = exp.device_manager()
        assert window is not None and exp.device_manager() is window   # one-per-session singleton
    finally:
        exp.close()
    # the internal factory is still importable from the submodule (na._gui is its caller)
    from Zou_lab_control.frontend.device_manager import show_device_manager
    assert callable(show_device_manager)
