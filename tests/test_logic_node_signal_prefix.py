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

from conftest import raw_device_set

import os
from pathlib import Path
import sys
import time

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
    pending = con._starting_nodes.get(id(row))
    if pending is not None:
        con._legacy_handles[id(pending)].wait_started(2.0)
        con._poll_logic_nodes()
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
        assert main_frame.shape[-2:] == raw_device_set(exp)["camera"].sensor_shape

        raw_device_set(exp).sequencer.set_safe_state()                   # the user hits Stop Pulse
        node2 = _start_with_camera(con, row, camera_name="monitor_camera")
        assert node2 is not node and node2.camera is raw_device_set(exp)["monitor_camera"]
        assert set(node2.published_signals()) == {"frame_0"}     # SAME names: same instance
        # The started node owns camera.acquire; never race it with a test-thread step().  Wait for the
        # selected free-running monitor's first publish through the real worker path.  Its HxW differs
        # from the old camera, so this also proves the row transferred schema ownership and replaced
        # the old frame_0 version/history instead of silently retaining the stale frame.
        expected = raw_device_set(exp)["monitor_camera"].sensor_shape
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                if np.asarray(con.hub.latest("frame_0")).shape[-2:] == expected:
                    break
            except KeyError:
                pass
            time.sleep(0.01)
        mon_frame = np.asarray(con.hub.latest("frame_0"))
        assert mon_frame.shape[-2:] == expected
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


def test_prefix_join_is_owned_by_the_base_class():
    """DRY guard: the prefix join lives ONCE, on LogicNode -- ``step()`` joins at publish time,
    ``published_signals``/``output_specs`` join the declaration, all three from the same
    ``self.prefix``.  A subclass declares BARE names via ``_bare_published_signals`` /
    ``_bare_output_specs`` and never re-spells the join; overriding the final joiners is the
    five-copies drift this pins.  The ONE declared exception is ``Task``: it is off the hub, so
    its output_specs stays keyed by the bare provides/mid_run names (no hub prefix exists)."""
    import inspect

    from Zou_lab_control.frontend.figure_viewer import LoadedFigureNode
    from Zou_lab_control.neutral_atom.operations import logic as L
    from Zou_lab_control.neutral_atom.operations.tasks.mot_field import OptimizeMotFieldTask

    subclasses = [obj for obj in vars(L).values()
                  if inspect.isclass(obj) and issubclass(obj, L.LogicNode) and obj is not L.LogicNode]
    subclasses += [LoadedFigureNode, OptimizeMotFieldTask]
    assert len(subclasses) >= 8                      # the family is really being swept
    for cls in subclasses:
        assert cls.published_signals is L.LogicNode.published_signals, \
            f"{cls.__name__} re-spells the published_signals join"
        expected = L.Task.output_specs if issubclass(cls, L.Task) else L.LogicNode.output_specs
        assert cls.output_specs is expected, f"{cls.__name__} re-spells the output_specs join"
