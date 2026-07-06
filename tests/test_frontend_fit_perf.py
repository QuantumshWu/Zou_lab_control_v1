"""The live general-fit path must not repaint the whole figure every tick.

Two single-source perf contracts on the fit-annotation path (the residual "7-8x redraw" the
main data tick already fixed everywhere else):

* :meth:`DataFigure._place_text` places the formula box by measuring its 6 candidate corners off
  the ALREADY-fetched renderer -- never a full ``canvas.draw()`` per candidate.  That draw was 6x
  a 300dpi raster just to size a text box, on the every-tick live-fit path.
* :meth:`Live1D._reapply_general_fit` is guarded by a data+command fingerprint (the flat-panel
  counterpart of :class:`HistogramFigure`'s ``_fit_cache_key``): a live tick that did not move the
  data re-runs NO ``curve_fit`` and repaints NO annotation.  A "sticky" live fit then costs one
  fingerprint per tick instead of a fresh fit + 6-candidate bbox search every frame.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

import Zou_lab_control.frontend.live as live_mod
from Zou_lab_control.frontend.live import Live1D


def _gaussian_plot() -> Live1D:
    """A flat 1D panel over a clean gaussian bump (so a ``gaussian`` fit converges and places its
    formula annotation), built undisplayed exactly as a console panel host builds it."""
    x = np.linspace(-5.0, 5.0, 60)
    y = 3.0 * np.exp(-(x ** 2) / (2 * 1.2 ** 2)) + 0.4
    plot = Live1D(x.reshape(-1, 1), y.reshape(-1, 1))
    plot.show(display=False)
    plot._fit_model = "gaussian"
    plot._fit_cmd = ""
    return plot


def test_place_text_does_no_draw_per_candidate():
    """``_place_text`` picks the least-overlapping corner by measuring each of its 6 candidates off
    the ALREADY-fetched renderer -- so it triggers ZERO full ``canvas.draw()`` (was 6, one per
    candidate: a 300dpi raster just to size a text box).  Tested in isolation so the assertion is
    backend-independent (no terminal ``draw_idle`` in the way)."""
    from Zou_lab_control.frontend.data_figure import DataFigure

    x = np.linspace(0.0, 1.0, 20)
    y = np.linspace(0.0, 1.0, 20)
    fig, ax = plt.subplots()
    df = DataFigure(fig=fig, data_x=x.reshape(-1, 1), data_y=y.reshape(-1, 1))
    df.plot_type = "1D"
    fig.canvas.draw()                                     # one legit initial render establishes the transforms
    text = ax.text(0.5, 0.5, "f(x)=a\na=1.0", transform=ax.transAxes)
    draws = {"n": 0}
    orig = fig.canvas.draw
    try:
        fig.canvas.draw = lambda *a, **k: draws.__setitem__("n", draws["n"] + 1)
        df._place_text(ax, text)                          # the 6-candidate corner search
        assert draws["n"] == 0, f"_place_text forced {draws['n']} full draw() (want 0; was 6)"
    finally:
        fig.canvas.draw = orig
        plt.close(fig)


def test_a_full_fit_apply_repaints_the_figure_once():
    """A whole fit apply issues exactly ONE terminal repaint (annotation + curve painted together),
    never the old 6 per-candidate draws + a premature ``_display_popt`` repaint before the curve."""
    plot = _gaussian_plot()
    canvas = plot.fig.canvas
    draws = {"n": 0}
    orig = canvas.draw                                    # Agg routes draw_idle -> draw synchronously
    try:
        canvas.draw = lambda *a, **k: draws.__setitem__("n", draws["n"] + 1)
        plot._reapply_general_fit()
        assert any(isinstance(a, Text) for a in plot._general_fit_artists), \
            "the gaussian fit must place its formula annotation (so _place_text is exercised)"
        assert draws["n"] == 1, f"a fit apply repainted the figure {draws['n']}x (want 1)"
    finally:
        canvas.draw = orig
        plt.close(plot.fig)


def test_general_fit_is_cached_by_data_fingerprint(monkeypatch):
    """A tick with unchanged data does not re-run the fit; a data change re-fits; clearing the
    model drops the cache key so a later re-enable always re-fits."""
    plot = _gaussian_plot()
    calls = {"n": 0}
    real = live_mod.apply_fit_to_figure

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)                              # call through: real artists populate the cache

    monkeypatch.setattr(live_mod, "apply_fit_to_figure", spy)
    try:
        plot._reapply_general_fit()                       # first fit on this data
        assert calls["n"] == 1

        plot._reapply_general_fit()                       # identical data + command -> cache hit
        assert calls["n"] == 1, "unchanged data must not re-run curve_fit"

        plot.data_y = plot.data_y + 1.0                   # the data moves -> cache miss -> re-fit
        plot._reapply_general_fit()
        assert calls["n"] == 2, "changed data must re-run the fit"

        plot._fit_model = "none"                          # fit cleared -> drop the cache key + curve
        plot._reapply_general_fit()
        assert plot._general_fit_key is None
        assert plot._general_fit_artists == []
    finally:
        plt.close(plot.fig)


def test_a_wiped_curve_refits_even_on_identical_data():
    """The cache trusts a hit ONLY while the drawn curve is still attached: a focus swap / axes
    rebuild that removed the artists must re-fit even though the data did not change (else the
    curve would silently vanish)."""
    plot = _gaussian_plot()
    try:
        plot._reapply_general_fit()
        assert plot._general_fit_artists, "a converged fit leaves at least the curve attached"
        for art in plot._general_fit_artists:            # simulate an axes rebuild wiping the artists
            art.remove()
        plot._reapply_general_fit()                       # identical data, but artists gone -> must re-fit
        assert plot._general_fit_artists, "a wiped fit curve must be redrawn on the next tick"
    finally:
        plt.close(plot.fig)
