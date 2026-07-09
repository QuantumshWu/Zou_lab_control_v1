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


def test_na_device_manager_hands_the_created_session_back_via_window():
    """#1: ``na.device_manager()`` returns the manager window, whose ``.session`` is None until
    "Init devices" and the live session afterwards -- so the notebook can do
    ``mgr = na.device_manager("virtual"); ...Init...; exp = mgr.session`` (the created session is
    no longer stuck inside the launcher's closure)."""
    window = na.device_manager("virtual")
    try:
        assert window.session is None                  # editor open, nothing connected yet
        panel = window.loaded                          # the DeviceManagerPanel inside the window
        panel._apply()                                 # press "Init devices"
        exp = window.session
        assert exp is not None                         # the session is handed back on the window
        assert exp.devices["camera"] is not None and hasattr(exp, "readout")
        exp.close()
    finally:
        window.close()


def test_exp_device_manager_window_exposes_the_bound_session(exp):
    """The session-bound entry (``exp.device_manager()``) exposes the SAME session on the
    window from the start -- symmetric with the no-session entry."""
    window = exp.device_manager()
    try:
        assert window.session is exp
    finally:
        window.close()


def test_added_entry_does_not_bloat_the_config_with_constructor_defaults():
    """Adding a device persists ONLY what the constructor cannot supply -- a param the constructor
    already DEFAULTS (PylonCamera.trigger_source) stays OUT of the config, so opening the editor
    rewrites nothing and the saved config stays minimal (the other half of the single-source fix)."""
    panel = DeviceManagerPanel(initial_config={"cam": {"type": "PylonCamera", "params": {}}},
                               on_init_session=lambda cfg: None)
    params = panel.working_config()["cam"]["params"]
    assert "trigger_source" not in params      # constructor supplies it -> not persisted (no bloat)


def test_constructor_required_params_drive_which_prefills_persist():
    """The core of the fix: a config param with NO constructor default (RemoteSequencer host / port /
    channels) is a required kwarg whose prefill MUST reach the config; one the constructor DEFAULTS
    (clock_hz / ssl) must not.  This is what makes ``RemoteSequencer(**params)`` succeed at Init."""
    from Zou_lab_control.frontend.device_manager import _constructor_required_params
    from Zou_lab_control.neutral_atom.devices.sequencer import RemoteSequencer
    required = _constructor_required_params(RemoteSequencer)
    assert {"host", "port", "channels"} <= required        # no signature default -> persist the prefill
    assert "clock_hz" not in required and "ssl" not in required   # defaulted -> constructor supplies them


def test_remote_sequencer_entry_carries_port_and_channels_into_the_config():
    """The exact real-hardware failure: a remote sequencer with ONLY ``host`` typed still Init's,
    because the prefilled ``port`` + ``channels`` (required-with-default kwargs) reach the config and
    ``RemoteSequencer(**params)`` constructs with no missing-arg TypeError."""
    from Zou_lab_control.neutral_atom.devices.sequencer import RemoteSequencer
    try:
        RemoteSequencer.config_params()                     # needs the server module importable
    except Exception as exc:                                # pragma: no cover - depends on lab install
        pytest.skip(f"RemoteSequencer config unavailable: {exc}")
    panel = DeviceManagerPanel(
        initial_config={"seq": {"type": "RemoteSequencer", "params": {"host": "10.0.0.5"}}},
        on_init_session=lambda cfg: None)
    params = panel.working_config()["seq"]["params"]
    assert params["host"] == "10.0.0.5"
    assert isinstance(params.get("port"), int) and params.get("channels")
    RemoteSequencer(**params)                                # constructs (no 'missing port'); does not connect
