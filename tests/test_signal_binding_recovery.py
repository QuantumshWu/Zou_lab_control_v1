"""#H3w-2: a panel that errors on a bound signal must RECOVER (re-pick / next tick) -- never latch into
a state where you "must remove the panel and start over".

Two invariants:
1. ``refresh`` updates ``_last_namespace`` only on a SUCCESSFUL render, so an error (e.g. the bound
   signal is still WAITING / not in this namespace) never overwrites the last-good namespace with one
   that re-errors on replay; a later display change / re-pick replays VALID data.
2. ``_apply_source`` (a signal re-pick routes through it) resets ``_render_version`` to -1, so the
   console's next tick re-renders the panel against a FRESH hub namespace -- recovering a panel that
   errored on a now-available signal without removing it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_task_console_measurement_gui import _camera_console  # noqa: E402


def test_error_does_not_overwrite_last_good_namespace():
    console, node, cam, card, editor = _camera_console()
    try:
        good = {"frame": np.full((64, 80), 50.0), "shot": 1}
        card.refresh(good)                                  # a successful render
        assert card._last_namespace is good                 # last-good remembered

        # a namespace MISSING the bound 'frame' -> the source eval raises -> status error, but the
        # last-good namespace is NOT clobbered (so a re-pick / display change still replays valid data)
        card.refresh({"shot": 2})
        assert card._last_namespace is good                 # still the last GOOD one, not the error one
        assert card._status_error                           # the panel shows the error (not silent)
    finally:
        console.shutdown()


def test_repick_forces_a_fresh_rerender_so_a_stuck_panel_recovers():
    console, node, cam, card, editor = _camera_console()
    try:
        card.refresh({"frame": np.full((64, 80), 50.0), "shot": 1})
        card._render_version = 999                           # pretend the tick gate is "up to date"
        # a signal re-pick / source re-apply must reset the gate so the NEXT tick re-renders fresh
        card.source_edit.setText("value = frame")
        card._apply_source()
        assert card._render_version == -1                   # forces the next console tick to refresh
    finally:
        console.shutdown()
