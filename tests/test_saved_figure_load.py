"""Contract for the SAVE -> LOAD round-trip of a front-end figure (the data layer, no GUI window).

``DataFigure.save`` writes ``<name>_<time>.png`` + a matching ``.npz`` (``data_x`` / ``data_y`` /
``info``).  ``frontend.load_figure`` (and its ``na.load_figure`` facade) reopen that npz -- with no
hardware or session -- as a lightweight :class:`SavedFigure` that answers "what was saved"
(``info_summary``), lists how the data can be viewed (``compatible_kinds``) and re-renders it
(``plot`` / ``plot(kind=...)``) through the SAME ``plot()`` factory the live figure used.

These pin:
1. save -> load round-trips ``data_x`` / ``data_y`` exactly + the info keys (labels/name/unit/kind/
   source + the saved view state);
2. ``.plot(kind=<saved kind>)`` reproduces a DataFigure (does not raise);
3. ``.plot(kind=<another compatible kind>)`` re-interprets the SAME arrays (does not raise);
4. ``.compatible_kinds()`` contains the saved kind;
5. an OLD payload (only ``data_x`` / ``data_y`` + a minimal ``info``) loads without crashing and the
   view state defaults.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import Zou_lab_control.frontend as zf  # noqa: E402
from Zou_lab_control.frontend import SavedFigure, load_figure, plot  # noqa: E402
from Zou_lab_control.frontend.data_figure import DataFigure  # noqa: E402


def _hist_datafigure():
    """A histogram DataFigure with a bimodal sample (a realistic dark/bright readout)."""
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(300.0, 20.0, 400), rng.normal(460.0, 20.0, 300)])
    p = plot(vals, kind="hist", display=False, update="once", data_figure=True)
    return p, p.to_data_figure()


def _saved_hist(tmp_path, *, extra_info=None):
    """Save a hist figure to ``tmp_path`` and return (out_dict, original DataFigure)."""
    p, df = _hist_datafigure()
    info = {"source": "counts", "kind": "hist",
            "view": {"relim": "fixed", "fixed_lo": 0.0, "fixed_hi": 200.0,
                     "unit_index": 0, "cmap": "", "repeat_mode": "pool"}}
    if extra_info:
        info.update(extra_info)
    out = df.save(str(tmp_path / "readout"), extra_info=info)
    plt.close(p.fig)
    return out, df


def test_save_load_round_trips_arrays_and_info(tmp_path):
    out, df = _saved_hist(tmp_path)
    saved = load_figure(out["data"])
    try:
        assert isinstance(saved, SavedFigure)
        # data_x / data_y are byte-for-byte the saved originals
        np.testing.assert_array_equal(saved.data_x, np.asarray(df.data_x_original))
        np.testing.assert_array_equal(saved.data_y, np.asarray(df.data_y))
        # the key provenance survived
        assert saved.kind == "hist"
        assert saved.info.get("source") == "counts"
        assert saved.name == df.name
        assert saved.labels == list(df.labels)
        assert saved.unit is not None                          # a display unit was recorded
        # the view state round-tripped (the whole point of the storage enhancement)
        assert saved.view.get("relim") == "fixed"
        assert float(saved.view.get("fixed_lo")) == 0.0
        assert float(saved.view.get("fixed_hi")) == 200.0
        assert saved.view.get("repeat_mode") == "pool"
        # info_summary is a non-empty human string mentioning the essentials
        summary = saved.info_summary()
        assert isinstance(summary, str) and "hist" in summary and "counts" in summary
    finally:
        plt.close("all")


def test_plot_saved_kind_reproduces_a_datafigure(tmp_path):
    out, _ = _saved_hist(tmp_path)
    saved = load_figure(out["data"])
    df = saved.plot(kind=saved.kind)                           # saved kind = original figure
    try:
        assert isinstance(df, DataFigure)
        assert df.fig is not None
    finally:
        plt.close("all")


def test_plot_another_compatible_kind_reinterprets_same_data(tmp_path):
    out, _ = _saved_hist(tmp_path)
    saved = load_figure(out["data"])
    others = [k for k in saved.compatible_kinds() if k != saved.kind]
    assert others, "a 1-D save must offer more than one compatible kind (line / rolling / hist)"
    df = saved.plot(kind=others[0])                            # swap plotter, SAME data
    try:
        assert isinstance(df, DataFigure)
        assert df.fig is not None
    finally:
        plt.close("all")


def test_compatible_kinds_contains_the_saved_kind(tmp_path):
    out, _ = _saved_hist(tmp_path)
    saved = load_figure(out["data"])
    kinds = saved.compatible_kinds()
    assert saved.kind in kinds
    assert kinds[0] == saved.kind, "the saved kind must be listed FIRST (reproduces the original)"
    # a 1-D save (hist) must not offer the 2-D families
    assert "2d" not in kinds and "sites" not in kinds


def test_old_format_npz_loads_and_view_defaults(tmp_path):
    """An OLD payload carries only ``data_x`` / ``data_y`` + a minimal ``info`` (no ``view`` sub-dict,
    no ``kind``).  ``load_figure`` must read it without crashing, default the view state to empty, and
    still re-render through the shape-inferred kind."""
    x = np.linspace(0, 1, 40).reshape(-1, 1)
    y = np.sin(np.linspace(0, 6, 40)).reshape(-1, 1)
    path = tmp_path / "legacy.npz"
    np.savez(path, data_x=x, data_y=y, info={"labels": ["X", "Y", "Z"], "name": "legacy"})
    saved = load_figure(path)
    try:
        assert isinstance(saved, SavedFigure)
        assert saved.view == {}                                # no stored view -> empty (defaults)
        assert saved.kind is None                              # old npz did not record a kind
        assert "1d" in saved.compatible_kinds()                # 1-column data_x -> 1-D families
        # info_summary and plot both work with the minimal payload
        assert "legacy" in saved.info_summary()
        df = saved.plot()                                      # kind falls back to "auto" (shape-inferred)
        assert isinstance(df, DataFigure)
    finally:
        plt.close("all")


def test_fit_is_stored_in_saved_info(tmp_path):
    """When a fit has been applied to the live figure, the saved ``info['fit']`` carries the function
    name, parameter names and coefficients so a reader can see / re-apply it -- the view-state fold."""
    p, df = _hist_datafigure()
    # A hist DataFigure has render_family 1D, so a gaussian fit applies; run it non-displayed.
    df.gaussian(is_display=False)
    out = df.save(str(tmp_path / "fitted"), extra_info={"kind": "hist"})
    plt.close(p.fig)
    saved = load_figure(out["data"])
    try:
        fit = saved.info.get("fit")
        assert isinstance(fit, dict) and fit.get("func") == "gaussian"
        assert fit.get("names") and len(fit["popt"]) == len(fit["names"])
        assert "fit" in saved.info_summary()
    finally:
        plt.close("all")


def test_na_facade_exposes_load_figure(tmp_path):
    """``na.load_figure`` reaches the frontend LAZILY (the notebook one-liner
    ``na.load_figure('x.npz').info_summary()``) and returns the same SavedFigure."""
    out, _ = _saved_hist(tmp_path)
    import Zou_lab_control.neutral_atom as na

    saved = na.load_figure(out["data"])
    assert isinstance(saved, SavedFigure)
    assert saved.kind == "hist"
