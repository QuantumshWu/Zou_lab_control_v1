"""The device-manager panel is type-generic: a section per REGISTERED device domain, driven by
device_domains() -- never a camera-special list.  Registering a new domain adds a section with
zero panel edits.  These contracts pin that + the launcher export."""
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
from Zou_lab_control.neutral_atom.devices.registry import device_domains


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _cards(panel):
    return [panel._body.itemAt(i).widget() for i in range(panel._body.count())
            if isinstance(panel._body.itemAt(i).widget(), FluentGroupBox)]


def test_panel_has_one_section_per_registered_domain():
    devices = na.load_devices("virtual")
    panel = DeviceManagerPanel(devices)
    titles = [c.title() for c in _cards(panel)]
    # ONE card per domain -- type-generic, driven by the registry (add a domain -> a card appears)
    assert titles == [d.label for d in device_domains()]
    assert panel.grab().width() > 0                       # renders


def test_scan_adds_a_discovered_card_without_crashing():
    devices = na.load_devices("virtual")
    panel = DeviceManagerPanel(devices)
    before = len(_cards(panel))
    panel._scan()                                          # virtual set: scans real buses, must not raise
    after = _cards(panel)
    assert len(after) == before + 1 and after[-1].title().startswith("Discovered")


def test_launcher_is_exported():
    assert "show_device_manager" in zf.__all__ and callable(zf.show_device_manager)
