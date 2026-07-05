"""MECHANICAL guard for the live-plot BLIT render path (``BaseLivePlot._compose_blit``).

A live data tick on the embedded Qt canvas must NOT re-rasterise the whole 300-dpi figure; once the
chrome is stable it restores a cached chrome background and redraws only the data artists (~20x). The
guarantees pinned here:

1.  **Blit engages** for the live plot kinds once their chrome settles (the dead-banded limits stop
    moving), i.e. the cached background is captured and reused.
2.  **A blit renders the SAME pixels as a full draw** at the same state -- the speed-up is invisible,
    agreeing to within a hairline anti-alias fringe at the axes frame (intrinsic to mpl blitting).
3.  **No ghosting.** Streaming different frames through the blit path shows only the latest -- the bg
    is restored every tick, so an old trace never lingers.
4.  **Non-embedded canvases fall back** to the full draw with no blit and no error.

Failure history: the frontend reimplemented Confocal-GUIv2 but never ported its blit machinery, so
every live tick did a full ``draw_idle`` (~12 ms of tick/label text); 5-6 panels at a 100 ms cadence
lagged.  The blit path fixes it; this test keeps it correct (engages, blit == full, no ghost).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from Zou_lab_control.frontend.qt_canvas import EmbeddedFigureCanvas

if EmbeddedFigureCanvas is None:  # pragma: no cover - matplotlib-qt missing
    pytest.skip("matplotlib Qt canvas unavailable", allow_module_level=True)

from Zou_lab_control.frontend.live import panel_plot
from Zou_lab_control.frontend.qt_canvas import panel_canvas
from Zou_lab_control.frontend.qt_fluent import ensure_qt_app


def _read(canvas) -> np.ndarray:
    w, h = int(canvas.renderer.width), int(canvas.renderer.height)
    return np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()


def _full(canvas) -> np.ndarray:
    canvas.draw()
    return _read(canvas)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


# in-range random frames: the value range stays put, so the dead-banded limits settle and the chrome
# stabilises -> the blit engages after a couple of ticks (worst case is wide-range noise, tested by the
# ghosting case which flips between two in-range frames).
def _frame(kind, n, rng, lo=0.2, hi=0.8):
    span = hi - lo
    if kind == "hist":
        # A readout histogram has a STABLE value range (dark/bright peaks at ~fixed counts); pin the two
        # endpoints so the bin span (the value x-axis) settles and the chrome can stabilise, exactly as a
        # real streaming distribution does -- uniform noise with a wandering min/max is the worst case.
        s = rng.rand(n) * span + lo
        s[0], s[1] = lo, hi
        return s
    if kind == "2d":
        return rng.rand(n) * span + lo
    return (rng.rand(n) * span + lo).reshape(-1, 1)


def _make(kind):
    ensure_qt_app()
    rng = np.random.RandomState(0)
    if kind in ("1d", "monitor"):
        x = np.arange(120, dtype=float)
        p = panel_plot(x, _frame(kind, 120, rng), kind=kind, size="2x2", title=f"{kind} panel")
        n = 120
    elif kind == "hist":
        p = panel_plot(_frame("hist", 200, rng), kind="hist", size="2x2", title="hist panel")
        n = 200
    elif kind == "2d":
        h, w = 8, 10
        ys, xs = np.mgrid[0:h, 0:w]
        x2 = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
        p = panel_plot(x2, _frame("2d", h * w, rng), kind="2d", size="2x2", title="2d panel")
        n = h * w
    else:
        raise ValueError(kind)
    panel_canvas(p.fig)                       # attach the EmbeddedFigureCanvas the GUI uses
    return p, n


def _engage(p, kind, n, rng, tries=12):
    """Stream in-range frames until the blit background is captured (chrome settled)."""
    for _ in range(tries):
        p.update(_frame(kind, n, rng))
        if p._blit_bg is not None:
            return True
    return False


ALL_KINDS = ["1d", "monitor", "hist", "2d"]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_blit_engages_and_matches_full_draw(kind):
    """Once the chrome settles the blit engages, and its frame equals a monolithic draw of the state.

    Because ``_engage`` STREAMS several different in-range frames and the last render is a blit, this
    also guards against ghosting: a lingering old frame would make the blit buffer diverge from the
    monolithic draw (the bg is restored every tick, so nothing should linger)."""
    p, n = _make(kind)
    rng = np.random.RandomState(7)
    assert _engage(p, kind, n, rng), f"{kind}: blit never engaged (chrome never stabilised)"
    blit = _read(p.fig.canvas)                # last render was a blit of the latest streamed frame
    full = _full(p.fig.canvas)                # monolithic render of the SAME state
    assert full.std() > 5, "nothing rendered -- test built an empty panel"
    assert _mean_abs(full, blit) < 1.5, f"{kind}: blit diverged from full draw (mismatch or ghost)"


def test_blit_recaptures_background_on_limit_change():
    """A genuine y-limit shift invalidates the cached bg (else the ticks/labels go stale)."""
    p, n = _make("1d")
    rng = np.random.RandomState(1)
    assert _engage(p, "1d", n, rng)
    sig0 = p._blit_sig
    p.update(np.linspace(0.0, 500.0, n).reshape(-1, 1))   # 100x range jump -> ylim + ticks move
    assert p._blit_sig != sig0, "chrome signature did not react to a y-limit change"


def test_non_embedded_canvas_falls_back_without_error():
    """A plotter whose canvas is NOT the embedded one keeps the full-draw path (no blit, no crash)."""
    ensure_qt_app()
    x = np.arange(64, dtype=float)
    p = panel_plot(x, np.random.RandomState(1).rand(64), kind="1d", size="2x2", title="plain")
    assert type(p.fig.canvas).__name__ != "EmbeddedFigureCanvas"
    p.draw()
    p.draw()
    assert p._blit_bg is None                 # blit never ran
