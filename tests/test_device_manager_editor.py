"""Device-manager CONFIG-EDITOR contracts: the editor's working dict IS the one
round-trippable config model.

* what the editor holds/produces is consumable by ``read_config`` / ``load_devices``
  (and, untouched, EQUALS the session's ``to_config()`` -- opening the editor never
  rewrites a config);
* a form edit writes THROUGH to the working config and survives a real ``load_devices``;
* a discovered row's ready config inserts as a properly-named entry;
* Apply routes to the bound session (``load_config``) end-to-end on the virtual backend;
* the no-session entry initialises a session from the working config and upgrades the
  panel in place;
* an unparseable typed field (a json literal mid-garbage) BLOCKS Apply by name instead of
  silently applying a stale value.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
from Zou_lab_control.frontend.device_manager import DeviceManagerPanel
from Zou_lab_control.neutral_atom._gui import _session_device_binding
from Zou_lab_control.neutral_atom.devices.registry import load_devices, read_config


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def exp():
    session = na.connect("virtual")
    yield session
    session.close()


def _bound_panel(session) -> DeviceManagerPanel:
    return DeviceManagerPanel(session.devices, session_binding=_session_device_binding(session))


def _card(panel, name):
    return next(c for c in panel._entry_cards if c.entry_name == name)


def test_untouched_editor_round_trips_the_session_config(exp):
    panel = _bound_panel(exp)
    cfg = panel.working_config()
    assert cfg == exp.devices.to_config()          # opening the editor rewrites NOTHING
    rebuilt = load_devices(read_config(cfg))       # and the dict is the real config model
    try:
        assert set(rebuilt.devices) == set(exp.devices.devices)
    finally:
        rebuilt.close()


def test_form_edit_writes_through_and_load_devices_consumes_it(exp):
    panel = _bound_panel(exp)
    _card(panel, "camera").widgets["exposure"].setValue(0.005)   # a real widget edit
    cfg = panel.working_config()
    assert cfg["camera"]["params"]["exposure"] == 0.005
    assert cfg["camera"]["params"]["trap_array"] == "$device:trap_array"   # refs intact
    rebuilt = load_devices(cfg)
    try:
        assert rebuilt["camera"].exposure == 0.005
    finally:
        rebuilt.close()


def test_add_discovered_row_inserts_a_ready_entry(exp):
    panel = _bound_panel(exp)
    row = SimpleNamespace(kind="basler", ident="12345", label="acA1920-155um",
                          config={"type": "PylonCamera", "params": {"serial": "12345"}})
    panel._add_discovered(row)
    cfg = panel.working_config()
    # named by its DOMAIN ("camera" is taken by the virtual entry -> camera_2)
    assert cfg["camera_2"] == {"type": "PylonCamera", "params": {"serial": "12345"}}


def test_apply_routes_to_the_session(exp):
    panel = _bound_panel(exp)
    _card(panel, "camera").widgets["exposure"].setValue(0.007)
    panel._apply()
    assert exp.devices["camera"].exposure == 0.007     # session.load_config really ran
    assert panel._applied == panel.working_config()    # dot state: synced


def test_invalid_json_field_blocks_apply_by_name(exp):
    panel = _bound_panel(exp)
    before = exp.devices
    _card(panel, "camera").widgets["capture_trigger_channels"].setText("not json")
    problems = panel._validation_problems()
    assert any("camera.capture_trigger_channels" in p for p in problems)
    panel._apply()
    assert exp.devices is before                       # blocked: session untouched


def test_no_session_entry_inits_and_upgrades_in_place():
    inited: list = []

    def _init(cfg):
        session = na.connect(cfg)
        inited.append(session)
        return _session_device_binding(session)

    panel = DeviceManagerPanel(initial_config=read_config("virtual"), on_init_session=_init)
    try:
        assert panel._apply_btn.text() == "Init devices"
        assert not panel._loaded_card.isVisible()      # no session yet
        panel._apply()
        assert inited, "on_init_session must be called with the working config"
        assert panel._apply_btn.text() == "Apply"      # upgraded in place
        assert inited[0].devices["camera"] is not None
    finally:
        for session in inited:
            session.close()
