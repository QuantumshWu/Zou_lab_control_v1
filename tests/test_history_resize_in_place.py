"""#5: changing the rolling-monitor history length is an IN-PLACE resize, never a figure rebuild.

A fast scroll on the history spin used to force a full plotter teardown + build-then-swap on every tick;
faster than the Qt holder could reflow, the figure flickered / "grew" / vanished (a race).  The root fix
makes ``length`` a display-only knob applied via ``LiveLive.resize_history`` -- it resizes the data
buffers + x-axis on the EXISTING axes (the same figure/axes objects survive), so there is no teardown to
race and no geometry move.  ``apply_param('length')`` returns True so the console takes the in-place path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _monitor(show_dist):
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend import panel_plot
    ensure_qt_app()
    p = panel_plot(np.arange(300, dtype=float), np.full(300, np.nan), kind="monitor",
                   size="2x2", interactions=False, show_dist=show_dist,
                   labels=("Shots ago", "rate", "Z"))
    return p


def test_history_resize_is_in_place_no_figure_rebuild():
    """A rapid sequence of length changes (a fast scroll) keeps the SAME figure/axes objects and resizes
    the buffer to the requested length each time -- never tears down + rebuilds (the #5 race)."""
    for show_dist in (False, True):
        p = _monitor(show_dist)
        for v in (3.0, 4.0, 5.0, 6.0):
            p.roll(float(v), draw=False)
        fid, axid = id(p.fig), id(p.ax)
        axdis_id = id(getattr(p, "axdis", None))
        for L in (280, 260, 240, 520, 100, 90, 300):
            assert p.apply_param("length", L) is True, "length must be handled in place (return True)"
            assert p.data_y.shape == (L, 1), f"buffer must resize to {L}, got {p.data_y.shape}"
            assert p.points_total == L
            assert id(p.fig) == fid and id(p.ax) == axid, "resize must NOT rebuild the figure/axes (#5)"
            if show_dist:
                assert id(p.axdis) == axdis_id, "the side-dist axes must survive a resize too"


def test_history_resize_keeps_newest_samples_and_repins_xaxis():
    """Shrinking then growing keeps the newest sample at row 0 (newest-first buffer) and re-pins the
    x-axis to 0..length-1 -- the trace is continuous, not blanked."""
    p = _monitor(show_dist=True)
    for v in (10.0, 20.0, 30.0):           # newest (30) ends at row 0
        p.roll(float(v), draw=False)
    p.apply_param("length", 50)            # shrink-ish then grow
    p.apply_param("length", 400)
    assert p.data_y[0, 0] == 30.0, "the newest sample must survive a resize (newest-first)"
    assert p.ax.get_xlim() == (0.0, 399.0), "x-axis must re-pin to 0..length-1"
    assert p.n_bins == int(max(3, min(p.points_total // 4, 50))), "dist bins re-derive from the new length"
