# -*- coding: utf-8 -*-
"""Contract: the console's background render pipeline (frontend/render_loop.py).

* ``busy`` holds from ``submit`` until the GUI CONSUMED the batch -- a finished-but-
  undelivered batch still owns its figures, so a tick in that window is refused (the
  premature-idle double-submit race stays dead).
* ``barrier()`` waits out the compose AND delivers the batch itself; the queued
  ``job_done`` delivery afterwards is a no-op (single consumption, never a double
  present).
* Console tick path: a STRUCTURAL panel comes back to the GUI pass (plotter built
  there), and the presented canvas carries a front buffer + the mouse ownership
  barrier.
* Fairness: a beat falling on a busy tick is OWED (``_beat_owed``) and served on the
  next idle tick regardless of the modulo -- a slow-beat panel can never phase-lock
  onto busy ticks and starve behind a heavy fast-beat sibling.
"""
import threading
import time

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
from Zou_lab_control.frontend.render_loop import RenderLoop

from conftest import add_logic_row, make_console, tick


def test_barrier_on_the_render_worker_short_circuits():
    """A compose-phase re-entry into ``barrier()`` (a stock ``canvas.draw()`` fired mid-compose --
    e.g. an mpl selector ``set_active`` doing an ``update_background`` draw) must NOT wait on its own
    in-flight job: it already OWNS the figure it is composing.  Before the fix this self-deadlocked
    for the full 5 s timeout -- the drag freeze (#4-A: worker compose -> canvas.draw() -> barrier)."""
    ensure_qt_app()
    loop = RenderLoop(lambda _r: None)
    try:
        seen = {}

        def job():
            t0 = time.monotonic()
            seen["result"] = loop.barrier()          # re-enter the barrier from the worker thread
            seen["elapsed"] = time.monotonic() - t0
            return "batch"

        assert loop.submit(job)
        assert loop._published.wait(5.0)             # the job returned (did not self-deadlock)
        assert seen["result"] is True                # short-circuited on the worker, not a timeout
        assert seen["elapsed"] < 1.0                 # immediate -- never the 5 s self-wait
    finally:
        loop.stop()


def test_busy_holds_until_the_gui_consumes_and_barrier_delivers():
    ensure_qt_app()
    got = []
    loop = RenderLoop(got.append)
    try:
        release = threading.Event()

        def job():
            release.wait(5.0)
            return "batch"

        assert loop.submit(job)
        assert loop.busy                          # composing
        release.set()
        assert loop._published.wait(5.0)          # worker finished computing...
        assert loop.busy                          # ...but unconsumed -> STILL busy
        assert not loop.submit(lambda: "again")   # a tick in this window is refused
        assert got == []                          # nothing delivered (no event loop spun)
        assert loop.barrier()                     # waits AND delivers the batch itself
        assert got == ["batch"] and not loop.busy
        assert loop.deliver_pending() is False    # the queued delivery afterwards: no-op
        assert got == ["batch"]
    finally:
        loop.stop()


def _live_2d_setup(con):
    from Zou_lab_control.frontend.task_console import PanelConfig
    row = add_logic_row(con, ("camera", "live"))
    con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
    con._start_logic_node(row)
    node = con._logic_nodes[id(row)]
    node.step()
    sig = sorted(node.published_signals())[0]

    def panel(title, update_ms):
        card = con._new_panel_card(PanelConfig(
            kind="2d", title=title, size="2x2", source=f"value = {sig}",
            params={"update_ms": update_ms}))
        con._attach_card(card)
        return card

    return node, panel


def test_tick_hands_structural_builds_back_to_the_gui_and_presents():
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _node, panel = _live_2d_setup(con)
        card = panel("RC", 100)
        assert card.plotter is None
        con._tick()                              # SUBMIT only: no Qt event loop runs here
        con._render_loop.barrier()               # deliver -> structural GUI build + present
        assert card.plotter is not None                      # built on the GUI pass
        assert card._render_version != -1                    # version advanced on delivery
        assert getattr(card.canvas, "_zlc_front", None) is not None   # presented front buffer
        assert callable(getattr(card.canvas, "_zlc_render_barrier", None))  # mouse barrier hooked
    finally:
        con.shutdown()
        exp.close()


def test_a_beat_on_a_busy_tick_is_owed_and_served_off_beat():
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _node, panel = _live_2d_setup(con)
        fast, slow = panel("F", 100), panel("S", 800)
        con._recompute_tick_interval()
        assert con._base_interval_ms == 100
        release = threading.Event()
        assert con._render_loop.submit(lambda: release.wait(5.0))   # occupy the worker...
        con._tick_count = 7
        con._tick()                              # ...exactly over the slow panel's 800 ms beat
        assert slow._beat_owed and slow.plotter is None
        release.set()
        con._render_loop.barrier()               # drain the blocker (non-dict result: dropped)
        con._tick()                              # elapsed = 900: NOT a slow beat -- but it is OWED
        con._render_loop.barrier()
        assert slow.plotter is not None and not slow._beat_owed
        assert fast.plotter is not None
    finally:
        con.shutdown()
        exp.close()


def test_tick_helper_gives_one_deterministic_frame():
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        node, panel = _live_2d_setup(con)
        card = panel("T", 100)
        tick(con)
        assert card.plotter is not None
        v1 = card._render_version
        node.step()                              # a new shot -> the next frame must advance
        tick(con)
        assert card._render_version != v1
    finally:
        con.shutdown()
        exp.close()
