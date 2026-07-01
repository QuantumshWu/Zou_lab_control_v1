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


def _saved_hist_pair(tmp_path, *, name="readout") -> tuple[Path, Path]:
    """Save a bimodal-readout hist figure and return ``(image_png, data_npz)`` -- the SAME same-base
    pair ``DataFigure.save`` writes, so picking the image must load the sibling npz."""
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(300.0, 20.0, 400), rng.normal(460.0, 20.0, 300)])
    p = plot(vals, kind="hist", display=False, update="once", data_figure=True)
    df = p.to_data_figure()
    info = {"source": "counts", "kind": "hist", "view": {"relim": "tight"}}
    out = df.save(str(tmp_path / name), extra_info=info, image_ext="png")
    plt.close(p.fig)
    return Path(out["figure"]), Path(out["data"])


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
    # ... and the raw-info field (the Raw tab's multi-line code editor) exposes EVERY key the npz
    # stored, verbatim
    raw = v.raw_info.toPlainText()
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


def _saved_sites_with_signals_npz(tmp_path) -> Path:
    """Write a SITE-MAP npz that stored ``info['signals']`` (the faithful save side): a value occupancy
    block ``(r, 1, N)``, its ``(N, 2)`` centres and -- crucially -- the judged camera FRAME block
    ``(r, 1, H, W)``.  Return the ``.npz`` path (no live plot -- a hand-built payload, no hardware)."""
    R, N, H, W = 3, 12, 40, 50
    rng = np.random.default_rng(1)
    centers = np.stack([5.0 * (np.arange(N) % 4) + 6, 5.0 * (np.arange(N) // 4) + 6], axis=1)
    occ = (rng.random((R, 1, N)) > 0.4).astype(float)
    frame = rng.normal(600.0, 40.0, (R, 1, H, W))

    def sig(block, ps, ds, label, unit, role):
        return {"block": np.asarray(block), "points_shape": ps, "data_shape": ds,
                "label": label, "unit": unit, "role": role}

    info = {"name": "occupancy", "kind": "sites", "source": "value = occupied",
            "labels": ["site", "occupancy", "Z"], "unit": "", "view": {"relim": "tight"},
            "signals": {
                "occupied": sig(occ, [1], [N], "occupancy", "", "value"),
                "centers": sig(centers, None, None, "site centre", "px", "centers"),
                "frame_judged": sig(frame, [1], [H, W], "camera image", "counts", "frame"),
            }}
    path = tmp_path / "occ_sites.npz"
    np.savez(path, data_x=centers, data_y=occ.mean(0).reshape(N, 1), info=info)
    return path


def test_sitemap_signals_save_reproduces_rings_and_background_frame(tmp_path):
    """A site-map save that stored ``info['signals']`` reopens FAITHFULLY: the ``LoadedFigureNode``
    re-publishes the occupancy value + the ``(N, 2)`` centres + the judged FRAME block, and wires
    ``sitemap_centers_key`` / ``sitemap_image_key`` so the seeded sites panel builds WITH its background
    camera frame (an imshow underlay) -- the bug where a sitemap save built an empty board is gone."""
    npz = _saved_sites_with_signals_npz(tmp_path)
    v = show_figure_viewer(npz)
    try:
        node = v.node
        assert isinstance(node, LoadedFigureNode)
        # the frame block was re-published and wired as the sitemap underlay (bare "frame")
        assert node.sitemap_centers_key == "centers"
        assert node.sitemap_image_key == "frame", "the stored frame block is wired as the sitemap underlay"
        assert (FIG_PREFIX + "frame") in set(v.hub.names())
        # the seeded sites panel BUILDS (the reported bug: it did not) and carries a background image
        card = v.console.cards[0]
        assert card.config.kind == "sites"
        v.console._tick()
        assert card.plotter is not None, "the seeded sites panel builds its plotter"
        assert len(card.plotter.ax.get_images()) >= 1, \
            "the sitemap shows its stored background camera frame (an imshow underlay), not bare rings"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_info_tabs_group_facts_and_browse_auto_loads(tmp_path):
    """The Info column is TABBED (Plot / Measurement / Device / Raw) and Browse (a valid .npz path on the
    path field) AUTO-LOADS with no separate Load button.  Pins: the four tabs exist; the Raw tab's editor
    holds the whole info dict; setting a real .npz on the path field loads it onto the board."""
    v = FigureViewer(path=None)                       # open empty, then drive Browse == a path change
    try:
        titles = {v.info_tabs.tabText(i) for i in range(v.info_tabs.count())}
        assert {"Plot", "Measurement", "Device", "Raw"} <= titles, "the Info column groups facts into tabs"
        assert not hasattr(v, "load_button"), "Browse auto-loads -- there is no separate Load button"

        npz = _saved_1d_npz(tmp_path)
        v.path_edit.setText(str(npz))                 # Browse sets the field -> changed(str) -> auto-load
        assert v.saved is not None and v.saved.kind == "1d", "picking a valid .npz loads it automatically"
        assert v.console is not None and len(v.console.cards) == 1
        # the Raw tab shows the whole stored info dict verbatim (multi-line, not one crushed line)
        assert "labels" in v.raw_info.toPlainText() and "kind" in v.raw_info.toPlainText()
    finally:
        v.teardown()
        plt.close("all")


def _saved_1d_with_repeat_npz(tmp_path) -> Path:
    """A 1-D save whose stored ``info['signals']`` value block carries a REAL repeat axis ``(R, P, 1)``
    (R distinct traces) -- so a panel's ``repeat_mode`` = ``create`` draws one line per repeat and
    ``average`` draws a single mean line.  Written as a hand-built payload (no hardware)."""
    R, P = 3, 25
    x = np.linspace(0.0, 6.0, P)
    block = np.stack([np.sin(x) + 0.15 * r for r in range(R)], axis=0).reshape(R, P, 1)

    def sig(b, ps, ds, label, unit, role):
        return {"block": np.asarray(b), "points_shape": ps, "data_shape": ds,
                "label": label, "unit": unit, "role": role}

    info = {"name": "survival", "kind": "1d", "source": "value = survival",
            "labels": ["t (s)", "survival", "Z"], "unit": "",
            "view": {"relim": "tight", "repeat_mode": "create"},
            "signals": {"survival": sig(block, None, None, "survival", "", "value"),
                        "t": sig(x, None, None, "t (s)", "", "x")}}
    path = tmp_path / "survival_repeat.npz"
    np.savez(path, data_x=x.reshape(P, 1), data_y=block.mean(0).reshape(P, 1), info=info)
    return path


def _line_count(plotter) -> int:
    """Data-carrying Line2D artists on the plotter's main axes (a create trace draws one per repeat)."""
    return len([ln for ln in plotter.ax.get_lines() if len(ln.get_xdata()) > 0])


def test_edit_tab_repeat_mode_create_redraws_snapshot_per_repeat(tmp_path):
    """Regression (#3): changing ``repeat_mode`` in a panel's Edit tab must REDRAW the Edit-tab
    snapshot, not just the live card.  The snapshot previously rebuilt from only column 0 of the live
    plotter's data, so a 1d ``create`` (one line per repeat) always showed a single line -- the reported
    'selecting create has no effect'.  A save with a real ``(R, P, 1)`` repeat axis, opened in the
    viewer: the Edit-tab snapshot must show MORE lines under ``create`` (one per repeat) than under
    ``average`` (a single mean line)."""
    from Zou_lab_control.frontend.task_console import PanelEditor
    npz = _saved_1d_with_repeat_npz(tmp_path)
    v = show_figure_viewer(npz)
    try:
        con = v.console
        card = con.cards[0]
        con._tick()
        assert card.config.kind == "1d"
        # open the panel's real Edit tab (the closable PanelEditor) exactly as the Edit… button does
        con._edit_card(card)
        editor = next(con.tabs.widget(i) for i in range(con.tabs.count())
                      if isinstance(con.tabs.widget(i), PanelEditor) and con.tabs.widget(i).card is card)
        assert editor._plotter is not None, "the Edit tab builds its snapshot"

        # create: the live card draws one line per repeat, and the snapshot must MIRROR that
        card._set_param("repeat_mode", "create"); card._run_pending_rebuild(); con._tick()
        editor.rebuild()
        create_live = _line_count(card.plotter)
        create_snap = _line_count(editor._plotter)
        assert create_live == 3, "the live 1d panel draws one line per repeat in create mode"

        # average: a single mean line, on BOTH the live card AND the snapshot
        card._set_param("repeat_mode", "average"); card._run_pending_rebuild(); con._tick()
        editor.rebuild()
        average_live = _line_count(card.plotter)
        average_snap = _line_count(editor._plotter)
        assert average_live == 1, "average reduces the repeats to one line on the live card"

        # THE FIX: the Edit-tab snapshot RESPONDS to repeat_mode -- create shows the extra per-repeat
        # lines that average collapses.  The delta equals the live delta (the extra repeats), so the
        # snapshot is no longer frozen to a single column.
        assert create_snap > average_snap, \
            "the Edit-tab snapshot redraws per repeat_mode (create shows more lines than average)"
        assert create_snap - average_snap == create_live - average_live, \
            "the snapshot's create/average line delta matches the live panel's (all repeats drawn)"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_browse_filter_lists_saved_figure_images(tmp_path):
    """The File field's Browse dialog filter lists the saved-figure IMAGES (png / jpg / jpeg) next to the
    .npz, so the operator can eye-ball a thumbnail to find the right run (confocal lists ``*.npz *.jpg``
    in its 'Select Data File' dialog).  Pins the filter string carries the image suffixes."""
    v = FigureViewer(path=None)
    try:
        filt = v.path_edit._filter
        for suffix in ("png", "jpg", "jpeg", "npz"):
            assert f"*.{suffix}" in filt, f"the Browse filter offers *.{suffix}"
    finally:
        v.teardown()
        plt.close("all")


def test_picking_image_loads_sibling_npz(tmp_path):
    """Picking a saved-figure IMAGE (its .png) loads the SIBLING .npz data -- the save writes
    ``<name>_<time>.png`` + ``<name>_<time>.npz`` as a same-base pair, so ``image.with_suffix('.npz')``
    is the data.  open_path(png) and typing the png into the path field must BOTH load it, exactly as
    picking the .npz does (confocal's on_load strips the suffix and resolves the sibling .npz)."""
    png, npz = _saved_hist_pair(tmp_path)
    assert png.suffix == ".png" and npz.suffix == ".npz" and png.stem == npz.stem

    # open_path(png) loads the sibling npz and seeds the reproduction panel
    v = show_figure_viewer(png)
    try:
        assert v.saved is not None and v.saved.kind == "hist", "picking the .png loaded its sibling .npz"
        assert v.console is not None and len(v.console.cards) == 1, "the seeded panel built from the npz"
        assert v._current_path == npz, "the loaded path is the sibling npz, not the image"
        v.console._tick()
        assert v.console.cards[0].plotter is not None, "the seeded panel builds (equivalent to loading npz)"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")

    # typing the image path into the field (== Browse picking it) auto-loads the sibling npz too
    v2 = FigureViewer(path=None)
    try:
        v2.path_edit.setText(str(png))                # fires changed -> _on_path_changed
        assert v2.saved is not None and v2.saved.kind == "hist", \
            "setting the path field to the image auto-loads its sibling npz"
        assert v2._current_path == npz
    finally:
        v2.teardown()
        plt.close("all")


def test_picking_image_without_sibling_npz_warns(tmp_path):
    """A picked IMAGE with no sibling .npz beside it reports it in the status line and loads NOTHING --
    never a crash (the reported-figure counterpart of confocal's fuzzy-search miss)."""
    lonely = tmp_path / "bar.png"
    lonely.write_bytes(b"\x89PNG\r\n\x1a\n")             # a bare file with no sibling bar.npz
    v = FigureViewer(path=None)
    try:
        v.open_path(lonely)
        assert v.saved is None, "no sibling .npz -> nothing loaded"
        assert "no matching .npz" in v.status.text(), \
            f"the status warns there is no matching data, got: {v.status.text()!r}"
        assert v.console is not None and len(v.console.cards) == 0, "the board stays empty (no crash)"
    finally:
        v.teardown()
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
