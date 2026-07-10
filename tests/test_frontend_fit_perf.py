"""The live general-fit path must not repaint the whole figure every tick.

Two single-source perf contracts on the fit-annotation path (the residual "7-8x redraw" the
main data tick already fixed everywhere else):

* :meth:`DataFigure._place_text` places the formula box by measuring its 6 candidate corners off
  the ALREADY-fetched renderer -- never a full ``canvas.draw()`` per candidate.  That draw was 6x
  a 300dpi raster just to size a text box, on the every-tick live-fit path.
* :meth:`Live1D._reapply_general_fit` is guarded by a data+command fingerprint (the flat-panel
  counterpart of :class:`HistogramFigure`'s ``_fit_cache_key``): a live tick that did not move the
  data re-runs NO solver and repaints NO annotation.  A "sticky" live fit then costs one
  fingerprint per tick instead of a fresh fit + 6-candidate bbox search every frame.
"""

from __future__ import annotations

import os
import tracemalloc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

import Zou_lab_control.frontend.live as live_mod
from Zou_lab_control.frontend.live import Live1D
from Zou_lab_control.neutral_atom.core.fitting import FitRequest


def _gaussian_plot() -> Live1D:
    """A flat 1D panel over a clean gaussian bump (so a ``gaussian`` fit converges and places its
    formula annotation), built undisplayed exactly as a console panel host builds it."""
    x = np.linspace(-5.0, 5.0, 60)
    y = 3.0 * np.exp(-(x ** 2) / (2 * 1.2 ** 2)) + 0.4
    plot = Live1D(x.reshape(-1, 1), y.reshape(-1, 1))
    plot.show(display=False)
    plot._fit_request = FitRequest("gaussian").to_dict()
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
        assert calls["n"] == 1, "unchanged data must not re-run the solver"

        plot.data_y = plot.data_y + 1.0                   # the data moves -> cache miss -> re-fit
        plot._reapply_general_fit()
        assert calls["n"] == 2, "changed data must re-run the fit"

        plot._fit_request = None                          # fit cleared -> drop the cache key + curve
        plot._reapply_general_fit()
        assert plot._general_fit_key is None
        assert plot._general_fit_artists == []
    finally:
        plt.close(plot.fig)


def test_monitor_gauss_fit_is_cached_across_unchanged_ticks(monkeypatch):
    """The rolling ``monitor`` panel (console default) does not re-run the core fit every tick when
    its distribution counts are unchanged -- the counterpart of the histogram's ``_fit_cache_key``.
    A moved distribution re-fits."""
    from Zou_lab_control.frontend.live import LiveLiveDis

    plot = LiveLiveDis(np.arange(30.0).reshape(-1, 1))
    plot.show(display=False)
    plot.ylim_min, plot.ylim_max = 0.0, 10.0
    plot.n = np.array([1.0, 3.0, 6.0, 3.0, 1.0])          # a fittable side distribution
    plot.bins = np.linspace(0.0, 10.0, 6)

    calls = {"n": 0}
    real = live_mod.fit_histogram

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(live_mod, "fit_histogram", spy)
    try:
        plot._update_gauss_fit()
        assert calls["n"] == 1
        plot._update_gauss_fit()                          # identical counts -> cache hit
        assert calls["n"] == 1, "unchanged distribution must not re-run the solver every tick"
        plot.n = np.array([1.0, 2.0, 8.0, 2.0, 1.0])      # the distribution moved -> re-fit
        plot._update_gauss_fit()
        assert calls["n"] == 2
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


def test_fit_fingerprint_streams_a_constant_size_digest_without_frame_copy():
    plot = Live1D(np.arange(2.0).reshape(-1, 1), np.arange(2.0).reshape(-1, 1))
    # The 7.2 MiB payload is allocated before tracing.  The fingerprint may scan
    # it but must neither return nor transiently allocate a payload-sized bytes.
    plot.data_x = np.zeros((300_000, 2), dtype=np.float64)
    plot.data_y = np.ones((300_000, 1), dtype=np.float64)

    tracemalloc.start()
    key = plot._fit_data_fingerprint()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert [len(part[2]) for part in key] == [32, 32]
    assert peak < 64 * 1024, f"fingerprint allocated {peak:,} B for constant-size digests"


def test_unchanged_fit_request_runs_zero_solver_and_zero_draw(monkeypatch):
    plot = _gaussian_plot()
    solver_calls = {"n": 0}
    draw_calls = {"n": 0}
    real_apply = live_mod.apply_fit_to_figure
    try:
        def spy_apply(*args, **kwargs):
            solver_calls["n"] += 1
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(live_mod, "apply_fit_to_figure", spy_apply)
        plot._reapply_general_fit()
        assert solver_calls["n"] == 1
        original_draw = plot.fig.canvas.draw
        plot.fig.canvas.draw = lambda *args, **kwargs: draw_calls.__setitem__("n", draw_calls["n"] + 1)
        for _ in range(100):
            plot._reapply_general_fit()
        assert solver_calls["n"] == 1
        assert draw_calls["n"] == 0
    finally:
        if "original_draw" in locals():
            plot.fig.canvas.draw = original_draw
        plt.close(plot.fig)
