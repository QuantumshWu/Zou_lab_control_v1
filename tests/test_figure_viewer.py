"""Contract for the ``FigureViewer`` window (``exp.figure_viewer()`` / ``show_figure_viewer``).

The viewer reopens a saved figure ``.npz`` INTO the Task console: the loaded figure becomes ONE hub
SIGNAL (published by a :class:`~Zou_lab_control.frontend.figure_viewer.LoadedFigureNode`) and a panel of
the SAVED kind is seeded on a real :class:`~Zou_lab_control.frontend.task_console.TaskConsole`, so the
whole board / Add-Panel / signal-picker / re-wire / processing reuse comes for free.  These pins are
about that WIRING (the data layer round-trip is ``test_saved_figure_load.py``):

1. loading a save publishes its data as ``fig_value`` on the console's hub, and the seeded panel is
   wired to it with kind == the SAVED kind (a faithful reproduction);
2. the Info column exposes EVERY key the npz stored (name / kind / labels / view / the raw info dict);
3. the console is a real board: a second panel can be added reading the SAME ``fig_value`` signal;
4. the seeded panel's plotter builds (through the console tick) as the saved kind's plot class.

Runs headless (``QT_QPA_PLATFORM=offscreen``); the window is built + torn down in-process.
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

from Zou_lab_control.frontend import plot, show_figure_viewer  # noqa: E402
from Zou_lab_control.frontend.figure_viewer import (  # noqa: E402
    FIG_PREFIX,
    FIG_VALUE_KEY,
    FigureViewer,
    LoadedFigureNode,
)
from Zou_lab_control.frontend.task_console import PanelConfig, TaskConsole  # noqa: E402


FIG_SIGNAL = FIG_PREFIX + FIG_VALUE_KEY   # the hub name the loaded figure's primary data lands on


def _saved_hist_npz(tmp_path, *, name="readout") -> Path:
    """Save a bimodal-readout hist figure with a stored view state, and return its ``.npz`` path."""
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(300.0, 20.0, 400), rng.normal(460.0, 20.0, 300)])
    p = plot(vals, kind="hist", display=False, update="once", data_figure=True)
    df = p.to_data_figure()
    info = {"source": "counts", "kind": "hist",
            "view": {"relim": "fixed", "fixed_lo": 0.0, "fixed_hi": 200.0,
                     "unit_index": 0, "cmap": "", "repeat_mode": "pool"}}
    out = df.save(str(tmp_path / name), extra_info=info)
    plt.close(p.fig)
    return Path(out["data"])


def _saved_1d_npz(tmp_path) -> Path:
    """Save a 1-D vector figure with a real x-axis label, and return its ``.npz`` path."""
    x = np.linspace(0.0, 10.0, 40)
    y = np.sin(x)
    p = plot(x.reshape(-1, 1), y.reshape(-1, 1), kind="1d", display=False, update="once",
             data_figure=True, labels=("Trap-off time (s)", "survival", "Z"))
    df = p.to_data_figure()
    out = df.save(str(tmp_path / "scan1d"), extra_info={"source": "temperature", "kind": "1d",
                                                        "view": {"relim": "tight"}})
    plt.close(p.fig)
    return Path(out["data"])


@pytest.fixture
def viewer(tmp_path):
    """A FigureViewer window opened on a saved hist npz, torn down after the test."""
    npz = _saved_hist_npz(tmp_path)
    v = show_figure_viewer(npz)
    try:
        yield v, npz
    finally:
        win = v.window()
        if win is not None:
            win.close()
            win.deleteLater()
        v.teardown()
        plt.close("all")


def test_load_publishes_signal_and_seeds_reproduction_panel(viewer):
    v, _ = viewer
    con = v.console
    assert isinstance(con, TaskConsole), "the loaded figure opens on a real Task console board"
    # (1) the seeded panel reproduces: kind == the SAVED kind, wired to the fig_value signal
    assert len(con.cards) == 1, "one panel is seeded per loaded figure"
    card = con.cards[0]
    assert card.config.kind == v.saved.kind == "hist", "the seeded panel kind == the saved kind"
    assert card.config.inputs[0] == FIG_SIGNAL, "the seeded panel is wired to the fig_value signal"
    assert card.config.source == f"value = {FIG_SIGNAL}"
    # (2) that signal is actually PUBLISHED on the console's hub (the node ran a shot)
    assert FIG_SIGNAL in set(v.hub.names()), "the loaded figure's data is published as fig_value"
    assert isinstance(v.node, LoadedFigureNode)
    assert FIG_SIGNAL in set(v.node.published_signals())


def test_seeded_panel_restores_the_saved_view(viewer):
    v, _ = viewer
    card = v.console.cards[0]
    # the saved info['view'] was passed verbatim as the panel params -> the panel restores relim=fixed
    assert card.config.params.get("relim") == "fixed"
    assert float(card.config.params.get("fixed_lo")) == 0.0
    assert float(card.config.params.get("fixed_hi")) == 200.0
    assert card._relim() == "fixed", "the panel resolves the saved relim mode from its params"


def test_info_panel_exposes_every_stored_key(viewer):
    v, npz = viewer
    # (2) the Info column lists the facts (one row per fact) ...
    assert v.info_layout.count() > 0, "the Info panel lists what the file holds"
    # ... and the raw-info field exposes EVERY key the npz stored, verbatim
    raw = v.raw_info.text()
    assert raw, "the raw-info field shows the full stored info dict"
    for key in ("source", "kind", "labels", "view"):
        assert key in raw, f"raw info exposes the stored '{key}' key"


def test_seeded_panel_plotter_builds_as_saved_kind(viewer):
    v, _ = viewer
    card = v.console.cards[0]
    v.console._tick()                      # drive one refresh, exactly like the live timer does
    assert card.plotter is not None, "the seeded panel builds its plotter (reproduces the figure)"
    assert type(card.plotter).__name__ == "HistogramFigure", "a hist save reproduces as HistogramFigure"


def test_board_reuse_add_second_panel_reads_same_signal(viewer):
    v, _ = viewer
    con = v.console
    # (4) the board is real: add a SECOND panel reading the SAME fig_value signal, another kind
    con.state.panels.append(PanelConfig(kind="monitor", title="again",
                                        source=f"value = {FIG_SIGNAL}", inputs=[FIG_SIGNAL]))
    con.load_state(con.state)
    con._tick()
    assert len(con.cards) == 2, "a second panel was added to the same board"
    assert all(c.config.inputs[0] == FIG_SIGNAL for c in con.cards), \
        "both panels read the one loaded-figure signal (board reuse)"
    kinds = {c.config.kind for c in con.cards}
    assert kinds == {"hist", "monitor"}, "the same signal is viewed as two different kinds"


def test_window_opens_at_screen_fit_not_content_width(tmp_path):
    """The viewer opens at the shared screen-fraction size (the task console / pulse editor both return
    ``screen_fit_window_size`` verbatim from ``sizeHint``), NEVER collapsed to the bare Info-column
    content width -- and at that SAME size whether or not a figure is loaded, because an empty console
    board always fills the right column.  This is the guard for 'the window opened crammed': a
    content-driven ``sizeHint`` (or building no console when nothing is loaded) would shrink the empty
    window to the narrow Info strip and this test would fail."""
    from Zou_lab_control.frontend.qt_fluent import (
        WINDOW_SCREEN_FRACTION, ensure_qt_app, screen_fit_window_size)
    ensure_qt_app()
    fit = screen_fit_window_size(WINDOW_SCREEN_FRACTION)

    empty = FigureViewer(path=None)                       # the double-click default: nothing loaded
    try:
        assert empty.console is not None, \
            "an empty viewer still builds a real console board (never a bare Info strip)"
        assert empty.sizeHint().width() == fit.width(), \
            "the window opens at the screen-fit WIDTH, not the content/Info-column width"
        assert empty.sizeHint().height() == fit.height(), "and at the screen-fit height"
        assert empty.sizeHint().width() > empty._info_col_w * 2, \
            "far wider than the Info column alone -- not crammed"
        empty_hint = empty.sizeHint()
    finally:
        empty.teardown()

    npz = _saved_1d_npz(tmp_path)
    loaded = show_figure_viewer(npz)                      # loaded opens at the SAME size (content-independent)
    try:
        assert loaded.sizeHint().width() == empty_hint.width() == fit.width(), \
            "a loaded viewer opens at the same screen-fit width as the empty one"
        assert loaded.sizeHint().height() == empty_hint.height()
    finally:
        win = loaded.window()
        if win is not None:
            win.close(); win.deleteLater()
        loaded.teardown()
        plt.close("all")


def test_one_d_save_reproduces_with_saved_x_axis(tmp_path):
    """A 1-D save publishes a companion fig_x so the seeded 1d panel draws vs the saved x with the
    saved x-axis label (faithful reproduction of the x axis, not a bare index)."""
    npz = _saved_1d_npz(tmp_path)
    v = show_figure_viewer(npz)
    try:
        con = v.console
        card = con.cards[0]
        assert card.config.kind == "1d"
        assert (FIG_PREFIX + "x") in set(v.hub.names()), "a 1-D save also publishes its companion x"
        con._tick()
        assert card.plotter is not None
        assert card.plotter.ax.get_xlabel() == "Trap-off time (s)", \
            "the 1d panel draws vs the saved x with the saved x-axis label"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")
