"""Contract: camera frame-signal naming has ONE source -- ``DeviceSet.camera_signal_prefix``.

The DEFAULT camera publishes the bare conventional ``frame_i`` (what occupancy judging,
pulse-scan and the default 2-D panel bind); every OTHER camera publishes DEVICE-PREFIXED
frames (``monitor_camera_frame_0``), so the name states its sensor and two cameras can never
impersonate each other on the hub.  (Regression: the console's camera row built with NO
prefix, so the monitor camera published the bare ``frame_0`` too -- a panel bound to the
monitor row showed the MAIN camera's lingering block while the legend attributed it to the
monitor node.)  Declared == published on the console path (the H4b rule): the row legend,
the picker and the running node all read the same rule.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.devices.registry import load_devices


def test_device_set_owns_the_single_naming_rule():
    """``camera_signal_prefix`` -- default camera (by name, blank or instance) -> ``""``; any
    other camera -> ``"<name>_"``; an instance outside the set (a hand-built notebook camera)
    keeps the bare names.  The rule keys on the DEFAULT-camera convention, never on a literal
    device name."""
    exp = na.connect("virtual")
    other = na.connect("virtual")
    try:
        devs = exp.devices
        assert devs.default_camera_name() == "camera"
        # the default camera keeps the bare conventional names, however it is referred to
        assert devs.camera_signal_prefix("camera") == ""
        assert devs.camera_signal_prefix(None) == ""
        assert devs.camera_signal_prefix("") == ""
        assert devs.camera_signal_prefix(devs["camera"]) == ""          # instance reverse lookup
        # any OTHER camera is device-prefixed -- name and resolved instance agree (one rule)
        assert devs.camera_signal_prefix("monitor_camera") == "monitor_camera_"
        assert devs.camera_signal_prefix(devs["monitor_camera"]) == "monitor_camera_"
        # an instance NOT in this set has no set name to prefix with -> bare (tolerated)
        assert devs.camera_signal_prefix(other.devices["camera"]) == ""
    finally:
        other.close()
        exp.close()

    # bare names mean "the default camera", NOT "the device literally named 'camera'": in a
    # config whose ONLY camera is monitor_camera, that camera IS the default -> no prefix.
    solo = load_devices({"monitor_camera": {"type": "VirtualMotCamera", "params": {}}},
                        open_devices=False)
    assert solo.default_camera_name() == "monitor_camera"
    assert solo.camera_signal_prefix("monitor_camera") == ""


def test_camera_measurement_derives_the_device_prefix():
    """``readout.camera_measurement`` (the ONE build path notebook + console share) derives its
    frame prefix from the device rule: monitor camera -> ``monitor_camera_frame_i``, default
    camera -> bare ``frame_i``, the two sets DISJOINT; an explicit ``prefix`` (even ``""``)
    still wins (notebook naming authority)."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        mon = exp.readout.camera_measurement(hub, camera="monitor_camera", frames_per_cycle=2)
        assert set(mon.published_signals()) == {"monitor_camera_frame_0", "monitor_camera_frame_1"}
        default = exp.readout.camera_measurement(hub)
        assert set(default.published_signals()) == {"frame_0"}
        assert set(mon.published_signals()).isdisjoint(default.published_signals())
        # the spec path (device-role injection passes a RESOLVED instance) derives the same name
        spec_node = exp.readout.camera_spec().build(SignalHub(), camera="monitor_camera")
        assert spec_node.camera is exp.devices["monitor_camera"]
        assert set(spec_node.published_signals()) == {"monitor_camera_frame_0"}
        # explicit prefix overrides the derivation -- "" included
        pinned = exp.readout.camera_measurement(hub, camera="monitor_camera", prefix="")
        assert set(pinned.published_signals()) == {"frame_0"}
    finally:
        exp.close()


def test_console_camera_row_declared_equals_published_for_monitor_camera():
    """The console TRUE call side: a camera row whose Edit form selects ``monitor_camera``
    starts a node publishing ``monitor_camera_frame_0``, and the row's DECLARED prefix + full
    signal names equal the running node's ``published_signals()`` (H4b: declared == published).
    Re-starting the row onto the monitor camera also unlinks the previous build's bare
    ``frame_0`` from the hub -- the exact lingering block that used to impersonate the
    monitor camera on a bound panel."""
    from conftest import add_logic_row, fire_live_imaging, make_console

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("camera", "live"))         # real Add-Panel path (editor opens)
        # a fresh row (no camera picked yet) declares the DEFAULT camera's bare names
        assert con._declared_node_prefix(row) == ""
        assert con._declared_signal_keys(row) == ["frame_0"]

        # start on the default camera and land one real frame so the hub holds a bare frame_0
        con._start_logic_node(row)
        fire_live_imaging(exp)                               # On Pulse: the passive camera streams
        con._logic_nodes[id(row)].step()                     # one synchronous frame into the hub
        assert con.hub.latest("frame_0") is not None

        # pick the monitor camera in the row's REAL Edit form (Start reads collect_values())
        editor = con._logic_editors[id(row)]
        editor.form.seed_values({"camera": "monitor_camera"})
        con._start_logic_node(row)
        node = con._logic_nodes[id(row)]
        assert node is not None and node.camera is exp.devices["monitor_camera"]
        assert set(node.published_signals()) == {"monitor_camera_frame_0"}
        # declared == published: same prefix, same full names (legend / picker / hub agree)
        assert con._declared_node_prefix(row) == "monitor_camera_"
        assert set(con._declared_signal_keys(row)) == set(node.published_signals())
        # the previous build's bare frame_0 was unlinked (#5) -- no main-camera lingering block
        # left on the hub for a monitor-bound panel to mistake for monitor data.
        assert "frame_0" not in set(con.hub.signal_versions())
    finally:
        con.shutdown()
        exp.close()
