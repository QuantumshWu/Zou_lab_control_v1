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

from conftest import raw_device_set

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
        assert set(mon.occupied_devices()) == {raw_device_set(exp)["monitor_camera"]}
        assert set(main.occupied_devices()) == {raw_device_set(exp)["camera"]}
    finally:
        con.shutdown()
        exp.close()


def test_same_camera_rows_report_conflict_without_preempting_the_owner():
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, first = _start_camera_row(con)
        second_row, _second = _start_camera_row(con)         # same default sensor -> conflict
        con._poll_logic_nodes()
        assert first.running and first in con.running_nodes
        assert con._logic_nodes[id(second_row)] is None
        assert "start failed" in second_row.status_label.text().lower()
    finally:
        con.shutdown()
        exp.close()


def test_camera_trigger_wire_is_lifecycle_dependency_not_host_observation():
    """The camera node never calls the sequencer API; it only rides its trigger wire."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual")
    try:
        node = CameraMeasurement(SignalHub(), raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer)
        assert node.sequencer is raw_device_set(exp).sequencer
        assert set(node.occupied_devices()) == {raw_device_set(exp).camera}
        assert set(node.referenced_devices()) == {raw_device_set(exp).camera}
        assert set(node.lifecycle_devices()) == {raw_device_set(exp).sequencer}
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
    instances -- but only DRIVABLE ones.  CameraMeasurement's sequencer is trigger wiring,
    not an EXCLUSIVE capability, so ProcessorRun must not borrow it."""
    from Zou_lab_control.frontend.task_console import LogicNodeConfig
    from Zou_lab_control.neutral_atom.operations.processor import ProcessorSpec

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, cam = _start_camera_row(con)
        assert cam.sequencer is raw_device_set(exp).sequencer
        live_spec = ProcessorSpec(name="Live grab probe", params=(), run=lambda ctx: {"ok": 1.0},
                                  result_keys=("ok",), devices=("camera", "sequencer"))
        con.processors.append(live_spec)
        row = con._add_logic_node(LogicNodeConfig(kind="processor", name="Live grab probe"),
                                  focus=False)
        built = con._build_logic_node(row.node, {})
        assert built.camera is raw_device_set(exp)["camera"]              # declared role -> the drivable instance
        assert built.sequencer is None                            # wiring record is not borrowed
        assert set(built.occupied_devices()) == {raw_device_set(exp)["camera"]}   # honest claim = what it holds
    finally:
        con.shutdown()
        exp.close()


def test_runtime_and_lifecycle_device_sets_are_distinct():
    """Trigger wiring participates in generation replacement, not resource arbitration."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual")
    try:
        node = CameraMeasurement(SignalHub(), raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer)
        assert set(node.occupied_devices()) == {raw_device_set(exp).camera}
        assert set(node.referenced_devices()) == {raw_device_set(exp).camera}
        assert set(node.lifecycle_devices()) == {raw_device_set(exp).sequencer}
    finally:
        exp.close()


def test_swapping_one_device_stops_only_its_nodes_exclusive_and_observe():
    """THE user requirement: reinitialising a device (``load_config`` swap) stops EVERY running
    node that references it -- EXCLUSIVE drivers AND OBSERVE observers -- while nodes on the
    UNTOUCHED devices keep running.  Here a monitor-camera view and the main camera's view both
    OBSERVE the shared sequencer; swapping only the main camera stops the main view (drives it)
    but leaves the monitor view running (disjoint camera), even though both observe the sequencer
    that did NOT change."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, mon = _start_camera_row(con, camera_name="monitor_camera")
        _, main = _start_camera_row(con)
        assert mon.running and main.running
        seq_before = raw_device_set(exp)["sequencer"]

        # the change-hook path the session drives when ONLY the main camera is swapped
        con.stop_nodes_using({id(raw_device_set(exp)["camera"])})

        assert not main.running and main not in con.running_nodes   # references the swapped camera
        assert mon.running and mon in con.running_nodes             # disjoint camera -> survives
        assert raw_device_set(exp)["sequencer"] is seq_before               # (nothing actually swapped here)
    finally:
        con.shutdown()
        exp.close()


def test_load_config_swap_stops_referencing_console_nodes_end_to_end():
    """The runtime stops the old owner; the GUI only reconciles its terminal handle."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _, cam = _start_camera_row(con)
        assert cam.running
        exp.load_config("virtual")                         # full rebuild -> every device swapped
        con._poll_logic_nodes()                             # presentation-only Qt reconciliation
        assert not cam.running and cam not in con.running_nodes
        assert con._current_runtime_fence() is exp._require_runtime_services()
    finally:
        con.shutdown()
        exp.close()
