"""Contract: mutual exclusion is DEVICE OCCUPANCY, not a global "one driver at a time".

Each logic node declares the hardware instances it DRIVES (``LogicNode.occupied_devices``,
from its ``_occupies`` attribute names); starting a node stops exactly the running nodes
whose occupied devices intersect the new node's -- nodes on disjoint hardware coexist.

* The monitor camera's live view keeps running while the MAIN camera's measurement or
  calibration task starts (the user's case: nothing they use overlaps).
* Two rows driving the SAME sensor still exclude each other (a camera cannot be armed twice).
* Holding a reference is not occupying: the passive CameraMeasurement records its
  ``sequencer`` but declares only ``("camera",)``, so it coexists with a sequencer driver
  on a different sensor.
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


def test_passive_camera_records_but_does_not_occupy_the_sequencer():
    """Holding is not occupying: the passive CameraMeasurement carries ``sequencer`` as a
    record (it never prepares/fires), so its claim is the sensor alone."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual")
    try:
        node = CameraMeasurement(SignalHub(), exp.devices.camera, sequencer=exp.devices.sequencer)
        assert node.sequencer is exp.devices.sequencer           # the record is there...
        assert set(node.occupied_devices()) == {exp.devices.camera}   # ...but not claimed
    finally:
        exp.close()
