"""Contract: the window-close path (``stop_all_nodes``) goes through THE lifecycle endpoint.

Closing the session-bound console (the X -> ``close_requested`` -> ``stop_all_nodes``) must
leave NO zombie in ``running_nodes``: a bare ``node.stop()`` once left the dead node behind,
so its row stayed painted "running" and -- worse -- its never-advancing provenance froze the
WHOLE board's shot clock (``_display_shot`` = min over bound-and-live signals).  The user
symptom was exactly that: reopen the console, rows look live, images never update; and
stopping one camera row appeared to stop the OTHER camera's image too (the zombie min).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")

import Zou_lab_control.neutral_atom as na
from conftest import add_logic_row, make_console, tick


def _monitor_row(con):
    row = add_logic_row(con, ("camera", "live"))
    con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
    con._start_logic_node(row)
    return row, con._logic_nodes[id(row)]


def test_close_stops_through_the_lifecycle_endpoint_no_zombies():
    """After stop_all_nodes: running_nodes empty, row repainted stopped, live node ref nulled."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row, node = _monitor_row(con)
        node.step()
        con.stop_all_nodes()                                  # the window-close path
        assert con.running_nodes == []                        # no zombie holds the shot clock
        assert con._logic_nodes.get(id(row)) is None          # the live ref is released
        assert not node.running
        assert row.start_button.isEnabled() and not row.stop_button.isEnabled()  # row = stopped
    finally:
        con.shutdown()
        exp.close()


def test_reopen_and_restart_updates_the_image_again():
    """The user flow: run + bound 2d panel -> close window -> reopen -> Start again ->
    the panel must LIVE again (display shot advances, image pixels change)."""
    from Zou_lab_control.frontend.task_console import PanelConfig

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row, node = _monitor_row(con)
        sig = sorted(node.published_signals())[0]
        node.step()
        card = con._new_panel_card(PanelConfig(kind="2d", title="MON", size="4x4",
                                               source=f"value = {sig}", params={}))
        con._attach_card(card)
        tick(con)
        assert card._status_error is False

        con.stop_all_nodes()                                  # close the window...
        con._start_logic_node(row)                            # ...reopen + press Start
        node2 = con._logic_nodes[id(row)]
        assert node2 is not None and node2.running

        def image():
            imgs = card.plotter.fig.axes[0].images
            return np.asarray(imgs[0].get_array(), dtype=float).copy() if imgs else None

        node2.step()
        tick(con)
        before = image()
        shot_before = con._display_shot()
        for _ in range(3):
            node2.step()
        tick(con)
        after = image()
        shot_after = con._display_shot()
        assert shot_after is not None and (shot_before is None or shot_after > shot_before)
        assert before is not None and after is not None and not np.array_equal(before, after)
        assert card._status_error is False
    finally:
        con.shutdown()
        exp.close()


def test_stopping_one_camera_row_never_freezes_the_other():
    """The reported symptom head-on: two disjoint camera rows, each with its own 2d panel;
    stopping ONE row leaves the other row running AND its panel still updating."""
    from Zou_lab_control.frontend.task_console import PanelConfig
    from conftest import fire_live_imaging

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        camrow = add_logic_row(con, ("camera", "live"))
        con._start_logic_node(camrow)
        cam = con._logic_nodes[id(camrow)]
        fire_live_imaging(exp)
        monrow, mon = _monitor_row(con)
        cam.step(); mon.step()
        mon_sig = sorted(mon.published_signals())[0]
        card = con._new_panel_card(PanelConfig(kind="2d", title="MON", size="4x4",
                                               source=f"value = {mon_sig}", params={}))
        con._attach_card(card)
        tick(con)

        con._stop_logic_node(camrow)                          # stop the OTHER row
        assert mon.running and mon in con.running_nodes
        imgs = card.plotter.fig.axes[0].images
        before = np.asarray(imgs[0].get_array(), dtype=float).copy()
        for _ in range(3):
            mon.step()
        tick(con)
        after = np.asarray(card.plotter.fig.axes[0].images[0].get_array(), dtype=float)
        assert not np.array_equal(before, after)              # the monitor image stays LIVE
    finally:
        con.shutdown()
        exp.close()
