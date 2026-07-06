"""Contract: a signal's namespace is the LOGIC-NODE INSTANCE that produces it -- one rule
(``_logic_node_prefix``) for every kind (camera / measurement / processor).

* A lone instance publishes its kind-semantic base names: camera/processor the SHORT natural
  names (``frame_0`` / ``occupied``), a measurement its ``<spec.key>_<quantity>``.
* A SECOND instance whose names would collide is auto-disambiguated by its ROW TITLE slug
  (``camera_live_frames_2_frame_0``) -- instances of any kind can never overwrite each other.
* No device / backend identity ever appears in a signal name: WHICH sensor a camera row images
  is a build parameter; a consumer binds the instance's output, not a piece of hardware.  The
  same row can therefore switch its camera device and restart -- its signal names stay, the new
  device's frames flow under them, and every bound panel follows seamlessly.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import Zou_lab_control.neutral_atom as na
from conftest import add_logic_row, make_console


def _start_with_camera(con, row, camera_name=None):
    """Start a camera row through the REAL form path (Start collects the editor's values)."""
    if camera_name is not None:
        con._logic_editors[id(row)].form.seed_values({"camera": camera_name})
    con._start_logic_node(row)
    return con._logic_nodes[id(row)]


def test_lone_camera_row_publishes_the_short_natural_names():
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("camera", "live"))
        node = _start_with_camera(con, row)
        assert set(node.published_signals()) == {"frame_0"}
        assert con._declared_signal_keys(row) == ["frame_0"]    # declared == published (#prebind)
    finally:
        con.shutdown()
        exp.close()


def test_second_camera_row_is_disambiguated_by_its_instance_title():
    """Two rows of the SAME kind: the second gets its row-title slug -- never a clash, and never
    a device name (the second row images a different sensor, which must stay invisible)."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        first = _start_with_camera(con, add_logic_row(con, ("camera", "live")))
        row2 = add_logic_row(con, ("camera", "live"))
        second = _start_with_camera(con, row2, camera_name="monitor_camera")
        assert set(first.published_signals()) == {"frame_0"}
        (name,) = second.published_signals()
        assert name != "frame_0" and name.endswith("frame_0")  # instances never overwrite each other
        assert "monitor" not in name                            # the sensor is a build param, not a name
        assert name == f"{second.prefix}frame_0" and second.prefix  # prefix = the instance slug
        assert con._declared_signal_keys(row2) == [name]        # declared == published for it too
    finally:
        con.shutdown()
        exp.close()


def test_switching_the_rows_device_keeps_its_signal_names():
    """The user's flow: one camera row, watched by a panel on frame_0, is SWITCHED to the
    monitor camera and restarted.  The instance keeps its names (a restart reclaims its own
    lingering signals), the monitor's frames flow under frame_0, the panel follows seamlessly
    -- because the signal belongs to the measurement instance, not to a sensor."""
    from conftest import fire_live_imaging

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("camera", "live"))
        node = _start_with_camera(con, row)
        fire_live_imaging(exp)
        node.step()
        main_frame = np.asarray(con.hub.latest("frame_0"))
        assert main_frame.shape[-2:] == exp.devices["camera"].sensor_shape

        exp.devices.sequencer.set_safe_state()                   # the user hits Stop Pulse
        node2 = _start_with_camera(con, row, camera_name="monitor_camera")
        assert set(node2.published_signals()) == {"frame_0"}     # SAME names: same instance
        node2.step()                                             # monitor free-runs (no pulse needed)
        mon_frame = np.asarray(con.hub.latest("frame_0"))
        assert mon_frame.shape[-2:] == exp.devices["monitor_camera"].sensor_shape
    finally:
        con.shutdown()
        exp.close()


def test_second_measurement_row_of_the_same_spec_is_disambiguated():
    """The same instance rule covers measurements: a lone row publishes ``<spec.key>_<q>``; a
    second row of the SAME spec whose names are already on the hub (an earlier run's scan
    deliberately lingers past Stop) gets its row-title slug instead of silently overwriting it."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        kc = con.kind_combo
        data = next(kc.itemData(j) for j in range(kc.count())
                    if isinstance(kc.itemData(j), tuple) and kc.itemData(j)[0] == "measurement")
        row1 = add_logic_row(con, data)
        row2 = add_logic_row(con, data)
        spec = con._spec_for_logic(row1.node)
        p1 = con._declared_node_prefix(row1)
        assert p1 == f"{spec.key}_"                              # lone instance: the spec slug
        # an earlier run of this measurement left its scan on the hub (STOP keeps signals)
        con.hub.publish({f"{spec.key}_{k}": 0.0 for k in con._node_bare_keys(row1.node)})
        p2 = con._logic_node_prefix(row2.node)
        assert p2 and p2 != p1                                   # second instance: title-slug, no clash
    finally:
        con.shutdown()
        exp.close()
