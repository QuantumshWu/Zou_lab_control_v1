"""User flow 2026-07-05: a 2-D panel on ``frame_0`` must survive a calibration task.

camera measurement + 2d panel OK  ->  Calibrate readout runs to COMPLETION (mutual
exclusion stopped the camera)  ->  the operator restarts the camera while NO pulse is
firing yet (the cali parked the sequencer on a finished finite program)  ->  the panel
must keep rendering the hub's lingering ``(ring, 1, H, W)`` block.

It used to error "2D panel needs an image ... got value shape (1, 96, 128),
data_shape=(), grid_shape=()" because THREE independent defects lined up:

* the restarted ``CameraMeasurement`` declared ``data_shape=()`` until its first frame
  (which never came without a trigger) -- now seeded at BUILD time from the camera's
  own ``frame_shape`` contract;
* ``coerce_panel_value`` treated an UNDECLARED data shape as a declared non-image and
  raised instead of falling back to the value's own 2-D core (pinned in
  test_panel_coerce.py);
* the self-finished task never left ``running_nodes`` (only the Stop button did) --
  the lifecycle endpoint is now the ONE ``_stop_logic_node`` path.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.neutral_atom as na
from conftest import add_logic_row, fire_live_imaging, make_console


def _wait_terminated(node, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if getattr(node, "finished", False) or not node.running:
            return
        time.sleep(0.05)
    raise AssertionError("calibration task did not finish in time")


def test_2d_panel_survives_calibration_task_and_camera_restart():
    from Zou_lab_control.frontend.task_console import PanelConfig

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        # 1) camera measurement + a 2d panel bound to frame_0 -- renders fine.
        camrow = add_logic_row(con, ("camera", "live"))
        con._start_logic_node(camrow)
        fire_live_imaging(exp)
        con._logic_nodes[id(camrow)].step()          # one synchronous frame into the hub
        card = con._new_panel_card(PanelConfig(kind="2d", title="IMG", size="4x4",
                                               source="value = frame_0", params={}))
        con._attach_card(card)
        con._tick()
        assert card._status_error is False, card._status_text

        # 2) the Calibrate readout task runs to completion through the REAL console path
        #    (its start stops the camera via the device-driver mutual exclusion).
        taskrow = add_logic_row(con, ("task", "Calibrate readout"))
        con._start_logic_node(taskrow)
        task_node = con._logic_nodes[id(taskrow)]
        assert task_node is not None
        _wait_terminated(task_node)
        con._tick()                                   # the console observes the finish
        # the self-finished task reached the SAME endpoint the Stop button produces:
        # out of running_nodes (shot-clock + provider resolution see only LIVE nodes).
        assert task_node not in con.running_nodes
        assert con._logic_nodes.get(id(taskrow)) is None

        # 3) restart the camera while NOTHING is firing: its structure is declared at
        #    build time (camera.frame_shape), so the panel keeps rendering the hub's
        #    lingering frame instead of erroring on data_shape=().
        con._start_logic_node(camrow)
        cam2 = con._logic_nodes[id(camrow)]
        assert cam2.data_shape == exp.devices.camera.frame_shape
        card._render_version = -1                     # force a recompose this tick
        con._tick()
        assert card._status_error is False, card._status_text

        # 4) the pulse comes back -- live streaming resumes seamlessly.
        fire_live_imaging(exp)
        cam2.step()
        card._render_version = -1
        con._tick()
        assert card._status_error is False, card._status_text
    finally:
        con.shutdown()
        exp.close()
