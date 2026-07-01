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


def test_pulse_recipe_reopens_in_figure_viewer(tmp_path):
    """A pulse recipe npz reopens in ``show_figure_viewer``: the window does NOT crash, the right pane
    shows the REPRODUCED pulse figure on an embedded canvas (a structured figure is ``panel=False`` so it
    is NOT a console board), and the Info column carries the recipe."""
    ed, state = _pulse_editor()
    df = ed._preview_data_figure(state, include_always_off=True)
    out = df.save(str(tmp_path / "demo_pulse.png"))
    plt.close(df.fig)

    from Zou_lab_control.frontend.figure_viewer import show_figure_viewer

    viewer = show_figure_viewer(out["data"])
    try:
        assert viewer.saved is not None and viewer.saved.kind == "pulse"
        assert viewer.console is None, "a structured (recipe) figure is NOT seeded onto a console board"
        assert viewer._recipe_card is not None, "the right pane shows the reproduced pulse figure"
        assert viewer._recipe_figure is not None
        assert "figure_recipe" in viewer.raw_info.toPlainText()
    finally:
        win = viewer.window()
        if win is not None:
            win.close(); win.deleteLater()
        viewer.teardown()
        plt.close("all")
