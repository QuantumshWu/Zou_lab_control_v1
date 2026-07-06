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


def test_saved_npz_has_one_loader_shared_by_both_public_paths():
    """The saved-figure npz TRUST BOUNDARY is ONE function: ``data_figure._load_saved_npz`` holds
    the only ``np.load(allow_pickle=True)`` in the frontend, and both public reopen paths
    (``load_figure`` and the static ``frontend.load``) parse the ``data_x``/``data_y``/``info``
    triple through it.  A static AST scan (mirroring ``test_save_capture_single_source``'s
    guard), so even a lazily imported second copy is caught."""
    import ast
    import Zou_lab_control.frontend as frontend_pkg

    pkg_dir = Path(frontend_pkg.__file__).resolve().parent
    np_load_calls: dict[str, list[int]] = {}
    for py in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("np", "numpy")):
                np_load_calls.setdefault(py.name, []).append(node.lineno)
    assert set(np_load_calls) == {"data_figure.py"}, \
        f"np.load outside the ONE saved-npz loader: {np_load_calls}"
    # ... and every occurrence sits INSIDE _load_saved_npz (the single trust boundary)
    df_tree = ast.parse((pkg_dir / "data_figure.py").read_text(encoding="utf-8"))
    loader = next(n for n in ast.walk(df_tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_load_saved_npz")
    assert all(loader.lineno <= line <= loader.end_lineno for line in np_load_calls["data_figure.py"])
    # live.load delegates to the SAME loader (the import is the delegation seam)
    live_tree = ast.parse((pkg_dir / "live.py").read_text(encoding="utf-8"))
    imported = [alias.name for node in ast.walk(live_tree) if isinstance(node, ast.ImportFrom)
                and (node.module or "").endswith("data_figure") for alias in node.names]
    assert "_load_saved_npz" in imported, "frontend.load must go through data_figure._load_saved_npz"


def test_na_facade_exposes_load_figure(tmp_path):
    """``na.load_figure`` reaches the frontend LAZILY (the notebook one-liner
    ``na.load_figure('x.npz').info_summary()``) and returns the same SavedFigure."""
    out, _ = _saved_hist(tmp_path)
    import Zou_lab_control.neutral_atom as na

    saved = na.load_figure(out["data"])
    assert isinstance(saved, SavedFigure)
    assert saved.kind == "hist"


# ---- bare-save kind round-trip: plot.save() stamps the plotter's OWN kind (single source) ----------
def _build_plotter_of_kind(key: str):
    """Build ONE plotter of ``key`` with kind-appropriate data, through the SAME factories a notebook
    uses -- so ``plotter.save()`` (no explicit ``kind``) is exercised for every savable kind."""
    from Zou_lab_control.frontend.live import site_histogram_grid, site_psf_grid
    rng = np.random.default_rng(1)
    if key == "2d":
        xy = np.column_stack([np.tile(np.arange(7), 7), np.repeat(np.arange(7), 7)])
        return plot(xy, rng.normal(size=49), kind="2d", display=False, update="once", data_figure=True)
    if key == "sites":
        centers = np.column_stack([(np.arange(9) % 3) * 5.0, (np.arange(9) // 3) * 5.0])
        frame = rng.normal(600.0, 30.0, (30, 30))
        return plot(centers, rng.random(9), kind="sites", display=False, update="once",
                    data_figure=True, image=frame)
    if key == "hist":
        vals = np.concatenate([rng.normal(3, 1, 150), rng.normal(40, 4, 160)])
        return plot(vals, kind="hist", display=False, update="once", data_figure=True)
    if key == "grid":
        return site_histogram_grid([rng.normal(3, 1, 60) for _ in range(6)],
                                   thresholds=[3.0] * 6, display=False)
    if key == "grid_image":
        return site_psf_grid([np.abs(rng.normal(size=(7, 7))) for _ in range(6)], display=False)
    # 1d / monitor
    x = np.linspace(0, 1, 40).reshape(-1, 1)
    y = np.sin(np.linspace(0, 6, 40)).reshape(-1, 1)
    return plot(x, y, kind=key, display=False, update="once", data_figure=True)


import pytest as _pytest  # noqa: E402


@_pytest.mark.parametrize("key", ["2d", "sites", "1d", "monitor", "hist", "grid", "grid_image"])
def test_bare_save_round_trips_the_plotters_own_kind(tmp_path, key):
    """THE single-source kind round-trip: a bare ``plotter.save()`` (no explicit ``kind`` argument) stamps
    the plotter's OWN registered kind into the saved ``info``, so ``load_figure().kind`` round-trips to the
    SAME kind that drew it -- a HistogramFigure -> ``hist``, a LiveSiteMap -> ``sites``, a grid -> ``grid``
    (not a shape-inferred fallback that flips a hist to a line or a site map to a 2-D image).  The saved
    kind is also FIRST in ``compatible_kinds`` (reproduces the original)."""
    plotter = _build_plotter_of_kind(key)
    expected = "grid" if key == "grid_image" else key
    try:
        out = plotter.save(str(tmp_path / f"bare_{key}"))
        saved = load_figure(out["data"])
        assert saved.kind == expected, f"{key}: bare save did not round-trip kind (got {saved.kind!r})"
        assert saved.compatible_kinds()[0] == expected, f"{key}: saved kind must be first in compatible_kinds"
    finally:
        plt.close("all")


def test_kind_for_plotter_is_single_source_reverse_lookup():
    """``kind_for_plotter`` reverse-looks-up a plotter's kind from the ONE PLOT_KINDS table (never a
    hand-typed name), resolving a SUBCLASS to its base kind (SiteHistogramGrid / a psf grid -> ``grid``;
    the rolling-trace LiveLiveDis -> ``monitor``) and ``None`` for a non-plotter."""
    from Zou_lab_control.frontend.live import (kind_for_plotter, HistogramFigure, LiveSiteMap,
                                               Live2DDis, Live1D, LiveLiveDis, SiteHistogramGrid,
                                               PulseSequenceFigure)
    rng = np.random.default_rng(2)
    cases = {
        "hist": _build_plotter_of_kind("hist"),
        "sites": _build_plotter_of_kind("sites"),
        "2d": _build_plotter_of_kind("2d"),
        "1d": _build_plotter_of_kind("1d"),
        "monitor": _build_plotter_of_kind("monitor"),
        "grid": _build_plotter_of_kind("grid"),
    }
    try:
        for expected, plotter in cases.items():
            assert kind_for_plotter(plotter) == expected, (expected, type(plotter).__name__)
        assert kind_for_plotter(object()) is None, "a non-plotter has no kind to stamp"
    finally:
        plt.close("all")


# ---- grid -> figure_recipe save -> faithful reproduction (structured-figure path, mirrors pulse) ----
def test_grid_save_stores_a_faithful_figure_recipe(tmp_path):
    """A per-site grid save stores a REPLAY RECIPE (the grid counterpart of the pulse recipe), not just a
    flat array: the npz ``info`` carries ``kind == 'grid'`` and ``figure_recipe['kind'] == 'grid'`` whose
    ``per_cell`` / ``thresholds`` rebuild the exact grid.  ``grid`` is the only compatible kind (the arrays
    are a lossy fallback)."""
    from Zou_lab_control.frontend.live import site_histogram_grid
    rng = np.random.default_rng(3)
    per_site = [np.concatenate([rng.normal(200, 20, 40), rng.normal(1200, 60, 40)]) for _ in range(6)]
    grid = site_histogram_grid(per_site, thresholds=[700.0] * 6, site_fidelities=[0.99] * 6, display=False)
    try:
        out = grid.save(str(tmp_path / "site_grid"))
        assert Path(out["figure"]).exists() and Path(out["data"]).exists()
        saved = load_figure(out["data"])
        assert saved.kind == "grid"
        recipe = saved.figure_recipe
        assert isinstance(recipe, dict) and recipe.get("kind") == "grid" and recipe.get("sub_plot_kind") == "hist"
        assert len(recipe["per_cell"]) == 6
        assert recipe["thresholds"] == [700.0] * 6
        assert saved.compatible_kinds() == ["grid"], "structured grid figure: only its recipe kind"
    finally:
        plt.close("all")


def test_grid_recipe_reopens_as_faithful_grid(tmp_path):
    """``na.load_figure(npz).plot()`` reproduces the ORIGINAL grid faithfully (a rebuilt SiteHistogramGrid
    with the same site count + thresholds), not a flattened line -- the grid mirror of the pulse recipe
    reopen.  A PSF-kernel grid (cell='image') reopens as an image-cell grid the same way."""
    from Zou_lab_control.frontend.live import site_histogram_grid, site_psf_grid, SiteHistogramGrid, GridPlot
    import Zou_lab_control.neutral_atom as na
    rng = np.random.default_rng(4)

    hist_grid = site_histogram_grid([rng.normal(3, 1, 50) for _ in range(6)],
                                    thresholds=[3.0] * 6, display=False)
    out = hist_grid.save(str(tmp_path / "hist_grid"))
    plt.close(hist_grid.fig)
    saved = na.load_figure(out["data"])
    try:
        reopened = saved.plot()                             # the recipe replay path
        grid_plotter = reopened.grid
        assert isinstance(grid_plotter, SiteHistogramGrid), "a hist grid reopens as a SiteHistogramGrid"
        assert grid_plotter.n_cells == 6, "the reproduced grid has the same site count"
        assert list(grid_plotter.thresholds) == [3.0] * 6, "the reproduced thresholds match"
    finally:
        plt.close("all")

    psf_grid = site_psf_grid([np.abs(rng.normal(size=(7, 7))) for _ in range(6)], display=False)
    out2 = psf_grid.save(str(tmp_path / "psf_grid"))
    plt.close(psf_grid.fig)
    saved2 = na.load_figure(out2["data"])
    try:
        assert saved2.kind == "grid" and saved2.figure_recipe["sub_plot_kind"] == "2d"
        reopened2 = saved2.plot()
        assert isinstance(reopened2.grid, GridPlot) and reopened2.grid.n_cells == 6
    finally:
        plt.close("all")


def test_focused_dis_grid_save_reopens_as_a_faithful_hist_grid(tmp_path):
    """Part C: a distribution grid that was ENLARGED (focused) into one cell and had its DISPLAY knobs
    changed (bins / ylog) saves a recipe that records the ``focused_cell_index`` + ``display_params`` +
    ``panel_size`` -- and ``na.load_figure(npz).plot()`` reopens it as a FAITHFUL hist grid (cell='hist'),
    re-applies the display knobs, and re-focuses the SAME cell as a full HistogramFigure.  The reported bug
    was a focused dis grid that would not draw on reopen / fell back to the wrong (site) kind; this pins the
    hist cell family driving the reopen (never sites) and the view-state round-trip."""
    from Zou_lab_control.frontend.live import site_histogram_grid, SiteHistogramGrid, HistogramFigure
    import Zou_lab_control.neutral_atom as na
    rng = np.random.default_rng(7)
    per = [np.concatenate([rng.normal(200, 25, 60), rng.normal(1200, 70, 40)]) for _ in range(6)]
    grid = site_histogram_grid(per, thresholds=[700.0] * 6, site_fidelities=[0.99] * 6, display=False)
    grid.apply_param("bins", 48)             # a display knob that re-bins the cells
    grid.apply_param("ylog", True)           # a display knob that takes effect on a focused cell
    grid.focus(3)                            # enlarge cell 3 into a full HistogramFigure
    out = grid.save(str(tmp_path / "focused_dis_grid"))
    plt.close(grid.fig)
    saved = na.load_figure(out["data"])
    try:
        assert saved.kind == "grid", "a dis grid saves as kind='grid'"
        recipe = saved.figure_recipe
        assert recipe["sub_plot_kind"] == "hist", "the recipe's sub_plot_kind is hist (drives the reopen -> NOT sites)"
        assert recipe["focused_cell_index"] == 3, "the focused cell index is recorded"
        assert recipe["display_params"].get("bins") == 48 and recipe["display_params"].get("ylog") is True
        assert saved.compatible_kinds() == ["grid"], "a structured grid figure offers only its recipe kind"
        reopened = saved.plot().grid                       # the recipe replay path -- must draw, not crash
        assert isinstance(reopened, SiteHistogramGrid), "reopens as a hist grid (NOT a site map)"
        assert reopened._display_params.get("bins") == 48, "the saved display knobs are restored"
        assert reopened._focused == 3, "the reopen re-focuses the saved cell"
        assert isinstance(reopened._focus_plotter, HistogramFigure), "the enlarged cell is a full hist figure"
    finally:
        plt.close("all")


def test_recipeless_grid_save_falls_back_to_arrays_not_a_crash(tmp_path, monkeypatch):
    """A grid save whose recipe capture FAILS must NOT stamp ``kind='grid'`` (which would send the reopen
    down ``plot(kind='grid')`` -- rejected, like pulse -- and crash).  Without a recipe the save has NO
    ``kind`` and the ``data_x`` / ``data_y`` fallback reopens through shape inference, exactly as the save's
    docstring promises ('degrades to the array-only fallback').  This pins the kind/recipe pairing so a
    future GridCell family with no recipe cannot produce an unloadable npz."""
    from Zou_lab_control.frontend import live as live_mod
    from Zou_lab_control.frontend.live import site_histogram_grid
    # Force the recipe capture to fail (a future/custom cell family that has no recipe).
    monkeypatch.setattr(live_mod, "grid_recipe_from_cells",
                        lambda grid: (_ for _ in ()).throw(TypeError("no recipe")))
    grid = site_histogram_grid([np.random.normal(0, 1, 30) for _ in range(4)], display=False)
    try:
        out = grid.save(str(tmp_path / "recipeless"))
        saved = load_figure(out["data"])                    # must NOT raise (the reported latent crash)
        assert saved.kind is None, "a recipe-less grid save carries NO grid kind (avoids plot(kind='grid'))"
        assert saved.figure_recipe is None
        df = saved.plot()                                   # shape-inferred array fallback -- must not raise
        assert df.fig is not None
    finally:
        plt.close("all")


# ---- pulse-preview -> figure_recipe save -> faithful reproduction (structured-figure path) --------
def _pulse_editor():
    """A headless PulseSequenceEditor whose preview draws real MULTI-channel 0/1 transitions AND a real
    analog DAC bus (``da0..da3`` grouped into one signed analog trace), so a faithful reproduction has
    genuine channels + an analog bus to compare against (not a single 1-D line)."""
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod

    periods = [
        PulsePeriod(duration=10, unit="us", name="p0", states=(1, 0, 0, 1, 0, 1)),
        PulsePeriod(duration=20, unit="us", name="p1", states=(0, 1, 1, 0, 1, 0)),
    ]
    st = PulseTableState(
        channels=["probe", "trig", "da0", "da1", "da2", "da3"], periods=periods, name="demo_pulse",
        analog_buses={"da": ["da0", "da1", "da2", "da3"]},
    )
    return PulseSequenceEditor(st), st


def _preview_plotter(state, *, include_always_off=True):
    """The ORIGINAL pulse plotter (the reference the reproduction must match), built through the SAME
    module-level renderer the editor preview uses -- ``build_pulse_preview_plot``."""
    from Zou_lab_control.frontend.pulse_gui import build_pulse_preview_plot
    plotter, _channels, _repeat = build_pulse_preview_plot(state, include_always_off=include_always_off)
    return plotter


def test_pulse_preview_save_stores_a_faithful_figure_recipe(tmp_path):
    """The pulse-preview Save stores a REPLAY RECIPE, not a flattened 1-D trace.  The npz ``info`` carries
    ``figure_recipe['kind'] == 'pulse'`` whose ``pulse_state`` round-trips through ``PulseTableState.
    from_dict`` -- so the ORIGINAL structured figure (multi-channel + analog bus) can be rebuilt exactly.
    The kind is ``pulse`` and the recipe is the only compatible kind (the arrays are a lossy fallback)."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState

    ed, state = _pulse_editor()
    df = ed._preview_data_figure(state, include_always_off=True)
    try:
        assert df.info.get("kind") == "pulse"
        recipe = df.info.get("figure_recipe")
        assert isinstance(recipe, dict) and recipe.get("kind") == "pulse"
        # the stored pulse_state faithfully round-trips (from_dict rebuilds the exact editor state)
        rebuilt = PulseTableState.from_dict(recipe["pulse_state"])
        assert list(rebuilt.channels) == list(state.channels)
        assert len(rebuilt.periods) == len(state.periods)
        assert dict(rebuilt.analog_buses) == dict(state.analog_buses)
        out = df.save(str(tmp_path / "demo_pulse.png"))
        assert Path(out["figure"]).exists() and Path(out["data"]).exists()
        data = np.load(out["data"], allow_pickle=True)
        info = data["info"].item()
        assert info["figure_recipe"]["kind"] == "pulse"
        assert "periods" in info["figure_recipe"]["pulse_state"]
    finally:
        plt.close("all")


def test_pulse_recipe_reopens_as_faithful_pulse_figure(tmp_path):
    """``na.load_figure(npz).plot()`` reproduces the ORIGINAL pulse figure, not a 1-D line: the reopened
    plotter is a ``PulseSequenceFigure`` whose channels / analog traces / pulse rows match the original
    editor preview exactly (many channels + an analog bus + many pulse rows -- not a single line)."""
    ed, state = _pulse_editor()
    original = _preview_plotter(state)
    orig_channels = list(original.channels)
    orig_analog = len(original.analog_traces)
    orig_pulses = len(original.pulses)
    plt.close(original.fig)

    df = ed._preview_data_figure(state, include_always_off=True)
    out = df.save(str(tmp_path / "demo_pulse.png"))
    plt.close(df.fig)

    import Zou_lab_control.neutral_atom as na

    saved = na.load_figure(out["data"])
    try:
        assert saved.kind == "pulse"
        assert saved.figure_recipe is not None and saved.figure_recipe["kind"] == "pulse"
        assert saved.compatible_kinds() == ["pulse"]        # structured figure: only its recipe kind
        # THE FAITHFUL REPRODUCTION: rebuild through the recipe -> a PulseSequenceFigure matching the original
        reopened = saved.plot()
        plotter = reopened.live_plot
        assert type(plotter).__name__ == "PulseSequenceFigure"
        assert list(plotter.channels) == orig_channels, "the reproduced channels match the original"
        assert len(plotter.analog_traces) == orig_analog, "the reproduced analog bus traces match"
        assert len(plotter.pulses) == orig_pulses, "the reproduced pulse rows match"
        assert orig_analog >= 1 and orig_pulses > 1, "the original really was multi-row (not a 1-D line)"
        # more than one data-carrying line on the axes -- a faithful multi-channel figure, not one trace
        lines = [ln for ln in plotter.ax.get_lines() if len(ln.get_xdata()) > 0]
        assert len(lines) > 1
        plt.close(reopened.fig)
    finally:
        plt.close("all")


def test_pulse_recipe_prefers_provenance_sequencer_payload(tmp_path):
    """When the save ALSO carries provenance (a FIRED pulse), the reproduction is sourced from the
    sequencer snapshot's ``last_payload_json`` (the SAME ``PulseTableState.to_dict()`` format the recipe
    uses) -- one source with the device state, no duplicate copy.  The recipe's own ``pulse_state`` is the
    fallback for a pure preview that was never fired.  This pins that provenance WINS when present."""
    import json
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod

    recipe_state = PulseTableState(channels=["a", "b"],
                                   periods=[PulsePeriod(duration=5, unit="us", states=(1, 0))], name="recipe")
    fired_state = PulseTableState(channels=["a", "b", "c"],
                                  periods=[PulsePeriod(duration=7, unit="us", states=(1, 0, 1))], name="fired")
    info = {
        "kind": "pulse", "name": "x",
        "figure_recipe": {"kind": "pulse", "pulse_state": recipe_state.to_dict(), "include_always_off": True},
        "provenance": {"devices": {"sequencer": {"last_payload_json": json.dumps(fired_state.to_dict())}}},
    }
    npz = tmp_path / "fired_pulse.npz"
    np.savez(npz, data_x=np.zeros((1, 1)), data_y=np.zeros((1, 1)), info=info)

    saved = load_figure(npz)
    try:
        df = saved.plot()                                   # provenance-sourced reproduction
        assert list(df.live_plot.channels) == ["a", "b", "c"], "provenance sequencer payload wins over the recipe copy"
    finally:
        plt.close("all")


def test_pulse_reopens_as_a_normal_panelcard_on_the_same_board(tmp_path):
    """A pulse figure reopens through the EXACT SAME load path as every other kind: ``LoadedFigureNode``
    publishes ``fig_value`` and ``_seed_state`` seeds a ``kind="pulse"`` panel bound to it -- so the
    loaded pulse card is a NORMAL :class:`PanelCard` (not a bespoke class), rendered by PanelCard's
    ``pulse`` branch.  Pins: (1) the console is a real ``TaskConsole`` (never replaced); (2) the Monitor
    tab survives; (3) the seeded card is a plain ``PanelCard`` with ``kind == "pulse"`` wired to
    ``fig_value`` -- SAME seed path as a hist; (4) it renders a faithful ``PulseSequenceFigure`` (channels
    + analog bus, not a 1-D line) through the console tick; (5) Add Panel still works; (6) ``pulse`` is
    NOT offered in the live Add-Panel dropdown (seedable, not live-addable)."""
    from Zou_lab_control.frontend.figure_viewer import show_figure_viewer, FIG_PREFIX, FIG_VALUE_KEY
    from Zou_lab_control.frontend.task_console import PanelCard, TaskConsole

    ed, state = _pulse_editor()
    df = ed._preview_data_figure(state, include_always_off=True)
    out = df.save(str(tmp_path / "demo_pulse.png"))
    plt.close(df.fig)

    viewer = show_figure_viewer(out["data"])
    try:
        assert viewer.saved is not None and viewer.saved.kind == "pulse"
        con = viewer.console
        # (1) real console, never replaced; (2) Monitor tab survives
        assert isinstance(con, TaskConsole), "a pulse figure opens on the SAME console (not a replacement)"
        tab_titles = {con.tabs.tabText(i) for i in range(con.tabs.count())}
        assert "Monitor" in tab_titles, "the Monitor tab must survive a pulse load"
        # (3) the seeded card is a NORMAL PanelCard of kind=pulse wired to fig_value -- SAME seed path
        assert len(con.cards) == 1
        card = con.cards[0]
        assert type(card) is PanelCard, "the loaded pulse card is a normal PanelCard (not a bespoke class)"
        assert card.config.kind == "pulse"
        assert card.config.inputs[0] == FIG_PREFIX + FIG_VALUE_KEY
        assert card.config.source == f"value = {FIG_PREFIX}{FIG_VALUE_KEY}"
        assert (FIG_PREFIX + FIG_VALUE_KEY) in set(viewer.hub.names()), "fig_value published like every kind"
        # (4) it BUILDS a faithful PulseSequenceFigure through the console tick (the pulse render branch)
        con._tick()
        plotter = card.plotter
        assert type(plotter).__name__ == "PulseSequenceFigure", "PanelCard renders pulse faithfully"
        assert len(plotter.channels) > 1 and len(plotter.analog_traces) >= 1 and len(plotter.pulses) > 1
        assert "figure_recipe" in viewer.raw_info.toPlainText()
        # (5) Add Panel still works beside the pulse panel
        con.kind_combo.setCurrentIndex(0)          # a live-addable "Plot: ..." kind
        con._add_panel()
        assert len(con.cards) == 2, "Add Panel still works with a pulse panel on the board"
        # (6) pulse is a SEEDABLE kind but NOT live-addable -> absent from the Add-Panel dropdown
        combo_kinds = {con.kind_combo.itemData(i) for i in range(con.kind_combo.count())}
        assert "pulse" not in {k for k in combo_kinds if isinstance(k, str)}, \
            "pulse is not offered in the live Add-Panel dropdown (seedable, not live-addable)"
    finally:
        win = viewer.window()
        if win is not None:
            win.close(); win.deleteLater()
        viewer.teardown()
        plt.close("all")
