"""#shot-clock: the console's _ShotClock picks ONE coherent display shot per render beat.

Pure non-blocking state machine: given the hub's {name: latest source-shot id} map and the signals the
panels are bound to, it returns the NEWEST shot every bound lineage signal has reached (so frame and its
derived frame_judged are shown together), holding the faster producer back -- with a stall grace so a
stopped producer never freezes the display, and freezing entirely while paused.
"""

from __future__ import annotations

import os
import pytest

pytest.importorskip("PyQt5")          # _ShotClock lives in task_console (a Qt module); logic is Qt-free
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Zou_lab_control.frontend.task_console import _ShotClock
from Zou_lab_control.neutral_atom.core.signals import NO_LINEAGE


def test_all_bound_at_same_shot_advances_to_it():
    c = _ShotClock(cache_time_s=0.1)
    assert c.advance({"frame": 5, "occupied": 5}, ["frame", "occupied"], now=1.0) == 5


def test_faster_producer_held_back_to_the_slower():
    c = _ShotClock(cache_time_s=10.0)        # long hold so the stall grace never fires
    # frame at 5, occupied lags at 3 -> show shot 3 (the newest BOTH have reached)
    assert c.advance({"frame": 5, "occupied": 3}, ["frame", "occupied"], now=1.0) == 3
    # occupied catches up to 5 -> advance to 5
    assert c.advance({"frame": 5, "occupied": 5}, ["frame", "occupied"], now=1.1) == 5


def test_stalled_producer_released_after_cache_time():
    c = _ShotClock(cache_time_s=0.5)
    # occupied stuck at 3 while frame climbs; before the grace elapses, hold at 3
    assert c.advance({"frame": 5, "occupied": 3}, ["frame", "occupied"], now=10.0) == 3
    assert c.advance({"frame": 6, "occupied": 3}, ["frame", "occupied"], now=10.3) == 3
    # past the grace (occupied unchanged > 0.5 s) -> drop it, advance to the live frame
    assert c.advance({"frame": 7, "occupied": 3}, ["frame", "occupied"], now=10.7) == 7


def test_never_runs_backwards():
    c = _ShotClock(cache_time_s=0.1)
    assert c.advance({"frame": 5, "occupied": 5}, ["frame", "occupied"], now=1.0) == 5
    # a transient where a signal reports an older latest must never rewind the display
    assert c.advance({"frame": 5, "occupied": 4}, ["frame", "occupied"], now=1.05) == 5


def test_no_lineage_scalar_never_blocks():
    c = _ShotClock(cache_time_s=0.1)
    # 'rate' is NO_LINEAGE -> it must not gate the advance; frame's shot wins
    assert c.advance({"frame": 4, "rate": NO_LINEAGE}, ["frame", "rate"], now=1.0) == 4


def test_paused_freezes_then_immediate_reflects_current():
    c = _ShotClock(cache_time_s=0.1)
    c.advance({"frame": 2, "occupied": 2}, ["frame", "occupied"], now=1.0)
    c.paused = True
    assert c.advance({"frame": 9, "occupied": 9}, ["frame", "occupied"], now=2.0) == 2   # frozen
    # an immediate (synchronous) read bypasses the pause and reflects the newest coherent shot
    assert c.advance({"frame": 9, "occupied": 9}, ["frame", "occupied"], now=2.0, immediate=True) == 9


def test_scalar_only_dashboard_follows_newest_lineage():
    c = _ShotClock(cache_time_s=0.1)
    # no BOUND lineage signal, but a lineage signal exists in the hub -> follow it (so a scalar card still
    # advances its display shot rather than sticking at 0)
    assert c.advance({"frame": 3}, ["rate"], now=1.0) == 3
