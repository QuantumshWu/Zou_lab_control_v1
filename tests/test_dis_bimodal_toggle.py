"""The `dis` histogram FIT is an explicit chooser (none / single / double), driven by the toggle.

The fit drives the display DIRECTLY -- it is never auto-decided from the data's shape (the old bug: the
fitted-separation gate decided whether one or two curves drew, so the same panel looked single for some
signals and double for others, and selecting "double" on near-unimodal data silently reverted to single
-- a "broken button").  Now: "double" always draws the dark/bright two-Gaussian decomposition, "single"
one Gaussian, "none" no fit at all.  The fidelity STAT stays honest (reported only when the two peaks
cleanly separate, else 'fit F=N/A'), but that no longer changes which curves draw.
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

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app  # noqa: E402
from Zou_lab_control.frontend import plot  # noqa: E402


def _two_peaks_drawn(fig) -> bool:
    return len(fig.fit_line_left.get_xdata()) > 0 and len(fig.fit_line_right.get_xdata()) > 0


def _only_single_drawn(fig) -> bool:
    return (len(fig.fit_line_total.get_xdata()) > 0
            and len(fig.fit_line_left.get_xdata()) == 0
            and len(fig.fit_line_right.get_xdata()) == 0)


def _no_fit_drawn(fig) -> bool:
    return (len(fig.fit_line_total.get_xdata()) == 0
            and len(fig.fit_line_left.get_xdata()) == 0
            and len(fig.fit_line_right.get_xdata()) == 0)


def test_double_draws_two_peaks_even_on_unimodal_data():
    """fit="double" (the default) draws the two-Gaussian decomposition for ANY data -- including a single
    broad blob (raw frame pixels) -- so selecting double always DOES something, never auto-collapsing to
    one peak."""
    ensure_qt_app()
    unimodal = np.random.default_rng(0).normal(300.0, 20.0, 400)
    fig = plot(unimodal, kind="hist")                       # fit defaults to "double"
    assert _two_peaks_drawn(fig), "double must draw the bimodal decomposition even on unimodal data"
    # but the fidelity is honest: an unseparated blob reports N/A, NOT a fake number
    assert not fig._fit_separated
    assert fig._fit_fidelity(fig.fit_threshold or float(np.median(unimodal))) is None


def test_single_draws_one_gaussian():
    ensure_qt_app()
    unimodal = np.random.default_rng(1).normal(300.0, 20.0, 400)
    fig = plot(unimodal, kind="hist", fit="single")
    assert _only_single_drawn(fig), 'fit="single" must draw ONE Gaussian, not the two-peak split'


def test_none_draws_no_fit():
    ensure_qt_app()
    vals = np.concatenate([np.random.default_rng(9).normal(300.0, 15.0, 300),
                           np.random.default_rng(8).normal(460.0, 15.0, 200)])
    fig = plot(vals, kind="hist", fit="none")
    assert _no_fit_drawn(fig), 'fit="none" must draw NO fit curve at all'


def test_separated_readout_reports_a_fidelity():
    """A genuinely bimodal readout (dark + bright) draws two peaks AND reports a finite fidelity (the
    separation gate is met)."""
    ensure_qt_app()
    rng = np.random.default_rng(2)
    readout = np.concatenate([rng.normal(300.0, 15.0, 300), rng.normal(460.0, 15.0, 200)])
    fig = plot(readout, kind="hist")
    assert _two_peaks_drawn(fig) and fig._fit_separated
    fidelity = fig._fit_fidelity(fig.fit_threshold or 380.0)
    assert fidelity is not None and 0.5 <= fidelity <= 1.0


def test_log_and_fit_toggle_in_place_without_rebuilding_the_figure():
    """#dis-resize: toggling the log y-scale or the fit chooser is DISPLAY-ONLY -- apply_param updates
    the EXISTING axes (update_core + draw), it must NEVER rebuild the figure/axes (a rebuild reflows the
    Qt holder so the plot visibly resizes/flashes).  Assert the SAME fig/ax objects survive each toggle
    and only the scale / fit curves change; a STRUCTURE knob (bins) is NOT handled in place."""
    ensure_qt_app()
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(20.0, 3.0, 800), rng.normal(80.0, 4.0, 800)])
    fig = plot(vals, kind="hist", fit="double")
    fig_id, ax_id = id(fig.fig), id(fig.ax)
    assert fig.ax.get_yscale() == "linear"
    assert fig.apply_param("ylog", True) is True
    assert fig.ax.get_yscale() == "log"
    assert id(fig.fig) == fig_id and id(fig.ax) == ax_id, "log toggle must NOT rebuild the figure/axes"
    assert _two_peaks_drawn(fig)
    assert fig.apply_param("fit", "single") is True
    assert _only_single_drawn(fig)
    assert id(fig.fig) == fig_id and id(fig.ax) == ax_id, "fit toggle must NOT rebuild the figure/axes"
    assert fig.apply_param("bins", 80) is False           # a STRUCTURE knob falls back to the rebuild path
