"""Contract: mutual exclusion is DEVICE OCCUPANCY, and access is DECLARATION-AS-CAPABILITY.

Each logic node declares its devices with an access mode (``LogicNode._devices``:
attribute name -> EXCLUSIVE | OBSERVE).  Starting a node stops exactly the running nodes
whose EXCLUSIVE devices intersect the new node's -- nodes on disjoint hardware coexist.
The declaration is machine-enforced, not documentation:

* EXCLUSIVE -- the node DRIVES the hardware; the attribute is the raw device and is what
  the console's mutual exclusion intersects (``occupied_devices``).
* OBSERVE -- the node only READS state; the base narrows the attribute to the device's
  ``ReadOnlyDevice`` view (its class's ``OBSERVE_API`` whitelist), so a later edit that
  tries to drive an observed device raises ``PermissionError`` instead of racing the owner.
* A declared name that is not a real attribute fails loud at ``start()`` (a silent rename
  once dropped a device from the exclusion).

User-visible law: the monitor camera's live view keeps running while the MAIN camera's
measurement or calibration task starts; two rows driving the SAME sensor still exclude
each other (a camera cannot be armed twice).
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
from conftest import add_logic_row, make_console


def _start_camera_row(con, camera_name=None):
    row = add_logic_row(con, ("camera", "live"))
    if camera_name is not None:
        con._logic_editors[id(row)].form.seed_values({"camera": camera_name})
    con._start_logic_node(row)
    return row, con._logic_nodes[id(row)]


def test_nodes_on_disjoint_cameras_coexist():
    """THE user case: the monitor camera's live view must survive the main camera's
    measurement starting -- their occupied devices do not overlap."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, mon = _start_camera_row(con, camera_name="monitor_camera")
        _, main = _start_camera_row(con)                     # main camera row starts AFTER
        assert mon.running and mon in con.running_nodes      # ...and the monitor keeps running
        assert main.running and main in con.running_nodes
        assert set(mon.occupied_devices()) == {exp.devices["monitor_camera"]}
        assert set(main.occupied_devices()) == {exp.devices["camera"]}
    finally:
        con.shutdown()
        exp.close()


def test_same_camera_rows_still_exclude_each_other():
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, first = _start_camera_row(con)
        _, second = _start_camera_row(con)                   # same default sensor -> conflict
        assert not first.running and first not in con.running_nodes
        assert second.running
    finally:
        con.shutdown()
        exp.close()


def test_calibration_task_stops_only_the_conflicting_camera():
    """Starting the Calibrate-readout task (occupies main camera + sequencer) stops the main
    camera's row but leaves the monitor camera's live view running."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, mon = _start_camera_row(con, camera_name="monitor_camera")
        _, main = _start_camera_row(con)
        taskrow = add_logic_row(con, ("task", "Calibrate readout"))
        con._start_logic_node(taskrow)
        task = con._logic_nodes[id(taskrow)]
        assert task is not None
        assert not main.running and main not in con.running_nodes   # shares the main camera
        assert mon.running and mon in con.running_nodes             # disjoint hardware -> survives
        devs = set(task.occupied_devices())
        assert exp.devices["camera"] in devs and exp.devices.sequencer in devs
        con._stop_logic_node(taskrow)                               # tidy: don't leave the cali running
    finally:
        con.shutdown()
        exp.close()


def test_observe_declaration_narrows_to_a_read_only_view():
    """Declaration-as-capability: CameraMeasurement declares ``sequencer`` OBSERVE, so the
    attribute IS the read-only view -- whitelisted reads pass through, driving or mutating
    raises PermissionError -- while the EXCLUSIVE ``camera`` stays the raw device (the node
    must drive it) and is the node's sole occupancy claim."""
    import pytest

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.devices.base import ReadOnlyDevice
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual")
    try:
        node = CameraMeasurement(SignalHub(), exp.devices.camera, sequencer=exp.devices.sequencer)
        assert isinstance(node.sequencer, ReadOnlyDevice)             # narrowed, not the raw device
        assert node.sequencer.firing == exp.devices.sequencer.firing  # whitelisted read passes through
        with pytest.raises(PermissionError):
            node.sequencer.fire()                                     # driving an observed device
        with pytest.raises(PermissionError):
            node.sequencer.sleep_scale = 0.0                          # mutating an observed device
        assert node.camera is exp.devices.camera                      # EXCLUSIVE = the raw device...
        assert set(node.occupied_devices()) == {exp.devices.camera}   # ...and the only claim
    finally:
        exp.close()


def test_devices_declaration_must_name_real_attributes():
    """Fail loud at start(): a declared name with no matching attribute would silently drop
    the device from the exclusion / the observe narrowing (the drift a private-name rename
    once caused)."""
    import pytest

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.devices.base import EXCLUSIVE
    from Zou_lab_control.neutral_atom.operations.logic import LogicNode

    class GhostNode(LogicNode):
        _devices = {"ghost": EXCLUSIVE}

        def shot(self):
            return {}

    node = GhostNode(SignalHub())
    with pytest.raises(AttributeError, match="ghost"):
        node.start()


def test_saved_dir_one_shot_neither_borrows_devices_nor_stops_live_nodes():
    """A one-shot processor gets hardware ONLY for the ctx roles its SPEC declares
    (``ProcessorSpec.devices``).  'Readout fidelity' reads a saved folder and declares none,
    so the console hands it None for camera AND sequencer: it occupies nothing, and starting
    it leaves an unrelated live camera view running (a pure folder characterization must
    never stop the live stream)."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, cam = _start_camera_row(con)
        row = add_logic_row(con, ("processor", "Readout fidelity"))
        built = con._build_logic_node(row.node, dict(row.node.values))
        assert built.camera is None and built.sequencer is None   # spec declares no ctx roles
        assert set(built.occupied_devices()) == set()             # -> claims no hardware
        con._start_logic_node(row)                                # the REAL Start path
        assert cam.running and cam in con.running_nodes           # the live view survived
        con._stop_logic_node(row)
    finally:
        con.shutdown()
        exp.close()


def test_hardware_declaring_one_shot_borrows_only_drivable_devices():
    """A one-shot spec that DOES declare ctx roles receives the running nodes' device
    instances -- but only DRIVABLE ones: a running CameraMeasurement records its sequencer
    as an OBSERVE read-only proxy, which a ProcessorRun (EXCLUSIVE driver) must never be
    handed (the first prepare/fire would PermissionError).  The read-only record is skipped,
    never unwrapped."""
    from Zou_lab_control.frontend.task_console import LogicNodeConfig
    from Zou_lab_control.neutral_atom.devices.base import ReadOnlyDevice
    from Zou_lab_control.neutral_atom.operations.processor import ProcessorSpec

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, cam = _start_camera_row(con)
        assert isinstance(cam.sequencer, ReadOnlyDevice)          # the OBSERVE record (precondition)
        live_spec = ProcessorSpec(name="Live grab probe", params=(), run=lambda ctx: {"ok": 1.0},
                                  result_keys=("ok",), devices=("camera", "sequencer"))
        con.processors.append(live_spec)
        row = con._add_logic_node(LogicNodeConfig(kind="processor", name="Live grab probe"),
                                  focus=False)
        built = con._build_logic_node(row.node, {})
        assert built.camera is exp.devices["camera"]              # declared role -> the drivable instance
        assert built.sequencer is None                            # the read-only proxy was skipped,
        assert not isinstance(built.sequencer, ReadOnlyDevice)    # never passed through
        assert set(built.occupied_devices()) == {exp.devices["camera"]}   # honest claim = what it holds
    finally:
        con.shutdown()
        exp.close()
