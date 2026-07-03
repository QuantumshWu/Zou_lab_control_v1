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


def test_every_panel_kind_saves_its_declared_display_knobs_in_the_view():
    """MECHANICAL round-trip guard for the 'figure_viewer 载入的 fig 不统一' root cause: every display
    knob a panel CARRIES -- its ``PANEL_PARAMS[kind]`` catalog -- must appear in the saved
    ``info['view']``, so re-opening a saved figure restores EXACTLY what was on screen (bins / fit /
    ylog / cmap / length / show_dist), not just the handful of cross-kind knobs.

    ``PanelEditor._save_view_state`` DERIVES its key set from the ONE ``PANEL_PARAMS`` source (never a
    hand-typed dict), so a newly declared display param is round-tripped the instant it is added, with
    no second edit site to forget.  This test re-derives the expected keys from that same source, so it
    can never drift into re-typing the literals it guards."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        PANEL_PARAMS, PanelConfig, TaskConsole, default_console_state)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    exp = na.connect("virtual")
    con = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                      measurements=exp.readout.measurement_specs(),
                      processors=exp.readout.processor_specs(),
                      tasks=exp.readout.task_specs(), window_px=(900, 600))

    def _saved_view(kind, params):
        card = con._new_panel_card(PanelConfig(kind=kind, title="P", size="2x2",
                                               source="value = __probe__", params=dict(params)))
        con._attach_card(card)
        con._edit_card(card)
        return card, con._panel_editors[id(card)]._save_view_state()

    try:
        # (1) EVERY declared knob of EVERY kind is present in the saved view (key completeness).
        for kind in ("hist", "monitor", "2d", "sites", "1d"):
            card, view = _saved_view(kind, {})
            pk = card._param_kind()
            missing = [d.key for d in PANEL_PARAMS.get(pk, ()) if d.key not in view]
            assert not missing, (
                f"{kind}: declared display knobs missing from saved view: {missing} "
                f"-- _save_view_state must derive its keys from PANEL_PARAMS[{pk!r}]")
        # (2) the SET value (not the catalog default) is what round-trips -- change one knob per kind.
        _, hv = _saved_view("hist", {"bins": 33, "fit": "single", "ylog": True})
        assert (hv["bins"], hv["fit"], hv["ylog"]) == (33, "single", True), \
            "a hist panel's SET bins/fit/ylog (not the defaults) round-trip through the view"
        _, mv = _saved_view("monitor", {"length": 123, "show_dist": False})
        assert (mv["length"], mv["show_dist"]) == (123, False), \
            "a monitor panel's SET length/show_dist round-trip through the view"
        _, iv = _saved_view("2d", {"cmap": "viridis"})
        assert iv["cmap"] == "viridis", "a 2d panel's SET cmap round-trips through the view"
    finally:
        con.shutdown()
        exp.close()


def test_grid_panel_exposes_editable_cell_title_template_and_size():
    """#5: a GRID panel's Edit tab exposes the per-cell title TEMPLATE + fixed font SIZE as editable
    controls (grid-only -- a flat panel has no per-cell title), an edit re-titles EVERY cell through the
    ONE ``card._set_param`` -> ``consume_param`` path, and both round-trip through the saved ``info['view']``
    -- the ``_panel_display_decls`` single source folds the grid title knobs into the same enumeration
    bins/cmap use, so they are shown, applied, saved and reopened identically."""
    import numpy as np

    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        GRID_TITLE_PARAMS, PanelConfig, TaskConsole, _panel_display_decls, default_console_state)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    # grid folds the title knobs into its enumeration; a flat panel does NOT.
    grid_keys = [d.key for d in _panel_display_decls("grid", "hist")]
    assert grid_keys[-2:] == [d.key for d in GRID_TITLE_PARAMS] == ["title_template", "title_size"]
    assert "title_template" not in [d.key for d in _panel_display_decls("hist", "hist")]

    ensure_qt_app()
    exp = na.connect("virtual")
    con = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                      measurements=exp.readout.measurement_specs(),
                      processors=exp.readout.processor_specs(),
                      tasks=exp.readout.task_specs(), window_px=(1000, 700))
    try:
        block = np.stack([np.concatenate([np.random.default_rng(k).normal(3, 1, 50),
                                          np.random.default_rng(k).normal(40, 3, 60)]) for k in range(4)],
                         axis=0).reshape(4, 1, 110)
        card = con._new_panel_card(PanelConfig(kind="grid", title="G", size="4x4",
                                               source="value = __probe__",
                                               params={"sub_plot_kind": "hist", "facet": "repeat",
                                                       "points_shape": (1,)}))
        con._attach_card(card)
        card._render(block, {})
        con._tick()
        con._edit_card(card)
        ed = con._panel_editors[id(card)]
        # editing the template (the ONE set_param path) re-titles EVERY cell.
        card._set_param("title_template", "{id}  n={k}"); card._run_pending_rebuild(); con._tick()
        assert [card.plotter.cell_renderer.cell_title(k) for k in range(4)] == \
            ["rep 0  n=0", "rep 1  n=1", "rep 2  n=2", "rep 3  n=3"]
        card._set_param("title_size", 7); card._run_pending_rebuild()
        assert card.plotter.cell_renderer.title_size_pt() == 7.0
        # both round-trip through the saved view.
        view = ed._save_view_state()
        assert view.get("title_template") == "{id}  n={k}" and view.get("title_size") == 7
    finally:
        con.shutdown()
        exp.close()


def test_edit_x_range_pin_is_global_live_and_persisted():
    """#3: the Edit tab's x-range pin has NO separate y input (the VALUE axis is relim-owned), Apply
    reaches the LIVE panel and EVERY grid cell (not just the frozen snapshot), it is persisted in
    ``config.params['view_xlim']`` so a rebuild keeps it, and it round-trips through the saved
    ``info['view']`` -- one x-range mechanism that is live, global-per-grid, and reopens as saved."""
    import numpy as np

    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        PanelConfig, TaskConsole, default_console_state)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    exp = na.connect("virtual")
    con = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                      measurements=exp.readout.measurement_specs(),
                      processors=exp.readout.processor_specs(),
                      tasks=exp.readout.task_specs(), window_px=(900, 600))
    try:
        # a GRID panel: Apply must fan the x-range out to every cell.
        block = np.random.default_rng(0).normal(20, 5, (6, 1, 80))
        card = con._new_panel_card(PanelConfig(kind="grid", title="G", size="4x4",
                                               source="value = __probe__",
                                               params={"sub_plot_kind": "hist", "facet": "repeat",
                                                       "points_shape": (1,)}))
        con._attach_card(card)
        card._render(block, {})
        con._tick()
        con._edit_card(card)
        ed = con._panel_editors[id(card)]
        assert ed.ymin is None and ed.ymax is None, "the Edit Limits section has NO y input (#3)"

        ed.xmin.setText("10"); ed.xmax.setText("30"); ed.apply_limits()
        # LIVE + GLOBAL: every cell's x-axis is pinned (not just the snapshot).
        assert all(tuple(ax.get_xlim()) == (10.0, 30.0) for ax in card.plotter.site_axes)
        # persisted -> survives a rebuild (re-applied in _apply_display_params).
        assert card.config.params.get("view_xlim") == (10.0, 30.0)
        card._apply_display_params()
        assert all(tuple(ax.get_xlim()) == (10.0, 30.0) for ax in card.plotter.site_axes)
        # round-trips through the saved view (so a reopened figure keeps the x-window).
        assert ed._save_view_state().get("view_xlim") == [10.0, 30.0]
    finally:
        con.shutdown()
        exp.close()


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
    """Data-carrying Line2D artists the plotter itself tracks (a create trace draws one per repeat).

    Count the plotter's OWN ``self.lines`` list, NOT every ``ax.get_lines()`` -- the axes also holds
    the AreaSelector's invisible ``RectangleSelector`` drag handles (corner/edge/centre ToolHandles =
    4+4+1-point Line2D markers that ``attach_interaction`` adds for box-select -> scan-param), which
    are NOT data and would inflate the count by three."""
    lines = getattr(plotter, "lines", None)
    if lines is None:
        lines = plotter.ax.get_lines()
    return len([ln for ln in lines if len(ln.get_xdata()) > 0])


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


def _saved_pulse_recipe_npz(tmp_path) -> Path:
    """A pulse-recipe npz built inline (a real ``PulseTableState.to_dict`` under ``figure_recipe``), no
    pulse editor -- so the viewer's structured-figure path can be exercised without the GUI editor."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    st = PulseTableState(
        channels=["probe", "trig", "da0", "da1"],
        periods=[PulsePeriod(duration=10, unit="us", name="p0", states=(1, 0, 0, 1)),
                 PulsePeriod(duration=20, unit="us", name="p1", states=(0, 1, 1, 0))],
        name="demo_pulse", analog_buses={"da": ["da0", "da1"]})
    info = {"kind": "pulse", "name": "demo_pulse", "source": "pulse preview",
            "figure_recipe": {"kind": "pulse", "pulse_state": st.to_dict(), "include_always_off": True}}
    path = tmp_path / "demo_pulse.npz"
    np.savez(path, data_x=np.zeros((1, 1)), data_y=np.zeros((1, 1)), info=info)
    return path


def _saved_grid_npz(tmp_path, *, cell="hist") -> Path:
    """Save a per-site GRID figure (hist or psf-image cell) through its OWN save, and return the ``.npz``
    path -- a real grid save (kind='grid' + figure_recipe), so the viewer's grid load path is exercised."""
    from Zou_lab_control.frontend.live import site_histogram_grid, site_psf_grid
    rng = np.random.default_rng(7)
    if cell == "image":
        g = site_psf_grid([np.abs(rng.normal(size=(7, 7))) for _ in range(6)], display=False)
    else:
        g = site_histogram_grid([np.concatenate([rng.normal(200, 20, 40), rng.normal(1200, 60, 40)])
                                 for _ in range(6)], thresholds=[700.0] * 6,
                                site_fidelities=[0.99] * 6, display=False)
    out = g.save(str(tmp_path / f"grid_{cell}"))
    plt.close(g.fig)
    return Path(out["data"])


def test_grid_reopens_as_a_normal_panelcard_on_the_same_board(tmp_path):
    """A per-site GRID figure reopens through the EXACT SAME load path as pulse (and every other kind):
    ``LoadedFigureNode`` publishes ``fig_value`` (a numeric placeholder; the grid recipe is carried on the
    node), ``_seed_state`` seeds a ``kind="grid"`` panel bound to it, and the SAME :class:`PanelCard`
    renders it via its ``grid`` branch (``build_grid_figure``).  Pins: real console (never replaced),
    Monitor tab survives, the seeded card is a NORMAL PanelCard of kind=grid wired to ``fig_value``, it
    renders a faithful GridPlot through the tick, and ``grid`` is NOT in the live Add-Panel dropdown."""
    from Zou_lab_control.frontend.task_console import PanelCard
    npz = _saved_grid_npz(tmp_path)
    v = show_figure_viewer(npz)
    try:
        con = v.console
        assert isinstance(con, TaskConsole), "a grid figure opens on the SAME console (never replaced)"
        assert v.saved.kind == "grid"
        tab_titles = {con.tabs.tabText(i) for i in range(con.tabs.count())}
        assert "Monitor" in tab_titles, "the Monitor tab survives a grid load"
        assert len(con.cards) == 1
        card = con.cards[0]
        assert type(card) is PanelCard, "the loaded grid card is a NORMAL PanelCard (same seed path)"
        assert card.config.kind == "grid" and card.config.inputs[0] == FIG_SIGNAL
        assert card.config.size == "2x2", "a grid panel opens at the default 2x2 preset like every kind"
        assert FIG_SIGNAL in set(v.hub.names()), "fig_value is published like every other kind"
        # the producing node carries the reproduction recipe (the hub is float-only)
        assert v.node is not None and v.node.grid_recipe is not None
        con._tick()
        plotter = card.plotter
        assert type(plotter).__name__ in ("SiteHistogramGrid", "GridPlot"), "PanelCard renders grid faithfully"
        assert plotter.n_cells == 6, "the reproduced grid has the saved site count"
        # grid is BOTH seedable (a loaded figure, this test) and live-addable (the facet
        # axis-expander, round R) -- so it appears in the Add-Panel dropdown like every plot kind.
        combo_kinds = {con.kind_combo.itemData(i) for i in range(con.kind_combo.count())}
        assert "grid" in {k for k in combo_kinds if isinstance(k, str)}
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_grid_image_cell_reopens_faithfully(tmp_path):
    """A PSF-kernel GRID (sub_plot_kind='2d') reopens the same way and renders imshow cells (a GridPlot of
    ImageCell), not a hist grid -- the recipe's ``sub_plot_kind`` selects the family faithfully."""
    npz = _saved_grid_npz(tmp_path, cell="image")
    v = show_figure_viewer(npz)
    try:
        assert v.saved.grid_recipe()["sub_plot_kind"] == "2d"
        card = v.console.cards[0]
        assert card.config.kind == "grid"
        v.console._tick()
        assert card.plotter is not None and card.plotter.n_cells == 6
        # image cells carry an imshow per cell (no threshold lines)
        assert any(len(ax.get_images()) >= 1 for ax in card.plotter.site_axes)
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_pulse_panel_opens_at_optimal_size_and_edit_tab_is_not_blank(tmp_path):
    """Regression pins: (1) a loaded pulse panel with no recorded ``panel_size`` opens at the SAME default
    ``optimal_pulse_size`` the preview uses (a small pulse -> a small preset), NOT a forced 2x2; (2) the
    pulse panel's Edit tab is NOT blank -- ``PanelEditor.rebuild`` has a ``pulse`` branch that rebuilds a
    real ``PulseSequenceFigure`` through the SAME renderer the live card uses (the reported blank-Edit bug)."""
    from Zou_lab_control.frontend.task_console import PanelEditor
    from Zou_lab_control.frontend.data_figure import load_figure
    from Zou_lab_control.frontend.live import default_pulse_size
    npz = _saved_pulse_recipe_npz(tmp_path)
    # The expected default is the SINGLE default source applied to this saved state's drawn content.
    saved = load_figure(npz)
    state, include_off = saved.pulse_state()
    expected = default_pulse_size(state, include_always_off=include_off)
    v = show_figure_viewer(npz)
    try:
        con = v.console
        card = con.cards[0]
        assert card.config.kind == "pulse"
        assert card.config.size == expected, \
            f"a recipe with no panel_size opens at the optimal default ({expected}), not a forced 2x2"
        con._tick()
        # open the real Edit tab and rebuild its snapshot -- it must build a PulseSequenceFigure, not blank
        con._edit_card(card)
        editor = next(con.tabs.widget(i) for i in range(con.tabs.count())
                      if isinstance(con.tabs.widget(i), PanelEditor) and con.tabs.widget(i).card is card)
        editor.rebuild()
        assert editor._plotter is not None, "the pulse Edit tab builds a snapshot (not blank)"
        assert type(editor._plotter).__name__ == "PulseSequenceFigure", \
            "the Edit-tab snapshot is a faithful pulse figure (rebuild has a pulse branch)"
        assert len(editor._plotter.channels) >= 1
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_pulse_panel_size_round_trips_from_recipe(tmp_path):
    """A pulse figure that RECORDS its ``panel_size`` in the recipe reopens at EXACTLY that size (the
    operator's saved size wins over the optimal default) -- the size flows with the figure."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    st = PulseTableState(
        channels=["probe", "trig"],
        periods=[PulsePeriod(duration=10, unit="us", name="p0", states=(1, 0)),
                 PulsePeriod(duration=20, unit="us", name="p1", states=(0, 1))],
        name="sized_pulse")
    info = {"kind": "pulse", "name": "sized_pulse", "source": "pulse preview",
            "figure_recipe": {"kind": "pulse", "pulse_state": st.to_dict(),
                              "include_always_off": True, "panel_size": "4x4"}}
    npz = tmp_path / "sized_pulse.npz"
    np.savez(npz, data_x=np.zeros((1, 1)), data_y=np.zeros((1, 1)), info=info)
    v = show_figure_viewer(npz)
    try:
        card = v.console.cards[0]
        assert card.config.kind == "pulse"
        assert card.config.size == "4x4", "the recorded panel_size (4x4) wins over the optimal default"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_calibration_report_every_npz_loads_as_correct_kind(tmp_path):
    """END-TO-END: ``save_calibration_report`` (6 sites, template, psf_weights, by_method) writes its
    report npz, and EVERY one reopens via ``load_figure`` at the CORRECT kind -- pooled -> ``hist``,
    site_map -> ``sites`` (WITH its template underlay frame, not an empty board), site_distribution_* /
    psf_grid -> ``grid``.  This is the reported bug fixed end-to-end (grid unloadable, sites frame missing,
    single figures falling back to the wrong type)."""
    from Zou_lab_control.frontend.calibration_report import save_calibration_report
    from Zou_lab_control.frontend.data_figure import load_figure

    rng = np.random.default_rng(0)
    n_sites, n_shots = 6, 120
    dark = rng.normal(200, 40, size=(n_shots // 2, n_sites))
    bright = rng.normal(1200, 120, size=(n_shots - n_shots // 2, n_sites))
    counts = np.vstack([dark, bright])
    thresholds = np.full(n_sites, 700.0)
    fidelity = np.full(n_sites, 0.99)
    centers = np.array([[10 + 9 * i, 20.0] for i in range(n_sites)], dtype=float)
    template = rng.normal(600, 30, size=(48, 64)).astype(float)
    psf_weights = np.abs(rng.normal(size=(n_sites, 7, 7)))
    by_method = {
        "box": {"counts": counts, "thresholds": thresholds, "fidelity": fidelity},
        "psf": {"counts": counts + rng.normal(0, 10, counts.shape), "thresholds": thresholds, "fidelity": fidelity},
    }
    paths = save_calibration_report(
        tmp_path, counts=counts, thresholds=thresholds, fidelity=fidelity, centers=centers,
        template=template, threshold_method="otsu", timestamp="t", by_method=by_method,
        psf_weights=psf_weights)

    expected_kind = {
        "global_distribution": "hist",
        "site_map": "sites",
        "site_distribution_box": "grid",
        "site_distribution_psf": "grid",
        "psf_grid": "grid",
    }
    try:
        for name, kind in expected_kind.items():
            assert name in paths, f"the report is missing {name}"
            npz = Path(paths[name]).with_suffix(".npz")
            assert npz.exists(), f"{name} wrote no .npz"
            saved = load_figure(npz)
            assert saved.kind == kind, f"{name} loaded as {saved.kind!r}, expected {kind!r}"
        # the site map is self-contained: its stored signals carry the template FRAME (a non-empty
        # underlay), so a reopen shows the background image, not bare rings on an empty board
        site_map = load_figure(Path(paths["site_map"]).with_suffix(".npz"))
        signals = site_map.info.get("signals")
        assert isinstance(signals, dict) and any(e.get("role") == "frame" for e in signals.values()), \
            "the site_map save is self-contained -- it stored its template underlay frame"
    finally:
        plt.close("all")


def test_pulse_seeds_a_normal_panelcard_via_the_same_load_path(tmp_path):
    """The reported bug: loading a PULSE figure wiped the console / Monitor tab (an old special-case
    replaced the whole board).  A pulse figure now takes the EXACT SAME load path as every other kind:
    ``LoadedFigureNode`` publishes ``fig_value``, ``_seed_state`` seeds a ``kind="pulse"`` panel bound to
    it, and the SAME :class:`PanelCard` renders it via its ``pulse`` branch.  Pins: a real console (never
    replaced), the Monitor tab present, the seeded card is a NORMAL PanelCard of kind=pulse wired to
    ``fig_value`` (fig_value IS on the hub -- as a numeric placeholder; the PulseTableState is resolved
    off the producing node), and it renders a faithful ``PulseSequenceFigure`` through the tick."""
    from Zou_lab_control.frontend.task_console import PanelCard
    npz = _saved_pulse_recipe_npz(tmp_path)
    v = show_figure_viewer(npz)
    try:
        con = v.console
        assert isinstance(con, TaskConsole), "a pulse figure opens on the SAME console (never replaced)"
        assert v.saved.kind == "pulse"
        tab_titles = {con.tabs.tabText(i) for i in range(con.tabs.count())}
        assert "Monitor" in tab_titles, "the Monitor tab survives a pulse load"
        assert len(con.cards) == 1
        card = con.cards[0]
        assert type(card) is PanelCard, "the loaded pulse card is a normal PanelCard (same seed path)"
        assert card.config.kind == "pulse" and card.config.inputs[0] == FIG_SIGNAL
        assert FIG_SIGNAL in set(v.hub.names()), "fig_value is published like every other kind"
        # the producing node carries the reproduction state (the hub is float-only)
        assert v.node is not None and v.node.pulse_state is not None
        con._tick()
        plotter = card.plotter
        assert type(plotter).__name__ == "PulseSequenceFigure", "PanelCard renders pulse faithfully"
        assert len(plotter.channels) > 1 and len(plotter.analog_traces) >= 1
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


# --------------------------------------------------------------------------- #
# Visual-consistency contracts (#1 tab card parity + #6 multi-line Info values).
# These guard the mechanical facts: the Info value control WRAPS a long value
# (never truncates), the Info tab card body uses the SAME card padding as the
# console's own tabs, and the tab overflow popup is rounded the shared way (not a
# bare square QMenu).
# --------------------------------------------------------------------------- #

def test_readout_multiline_wraps_long_value_taller_and_untruncated():
    """#6: a long Info value goes into a :class:`FluentReadoutMultiline` that SOFT-WRAPS it over
    several lines (taller than a one-line field) and keeps the WHOLE text (``toPlainText`` complete),
    where the old single-line ``FluentReadoutEdit`` would clip it to one row."""
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.qt_fluent import FluentReadoutMultiline, ensure_qt_app

    ensure_qt_app()
    # A realistic long value (a resolved data path) that cannot fit one narrow column line.
    long_value = ("C:/Users/lab/Dropbox/experiments/rb87/2026-07-01/"
                  "overnight_survival_scan_run17_readout_calibration_final.npz")
    host = QtWidgets.QWidget()
    host.resize(240, 400)                                  # a narrow Info-column-like width -> forces wrap
    lay = QtWidgets.QVBoxLayout(host)

    short = FluentReadoutMultiline("42")
    long = FluentReadoutMultiline(long_value)
    lay.addWidget(short)
    lay.addWidget(long)
    host.show()                                           # realise geometry so wrapping + _adjust_height run
    try:
        QtWidgets.QApplication.processEvents()
        # the whole value is preserved (no truncation, unlike a clipped single-line edit)
        assert long.toPlainText() == long_value, "the multi-line readout keeps the full value verbatim"
        assert long.text() == long_value, "its line-edit-style .text() also returns the full value"
        # and it grew TALLER than the one-line field (it wrapped onto multiple visual lines)
        assert long.height() > short.height(), \
            "a long value wraps onto several lines -> taller than a single-line value"
    finally:
        host.close()
        host.deleteLater()


def test_info_rows_tab_body_uses_card_padding(viewer):
    """#1(b): the Info tab card body is inset by the component-library card padding (``CARD_PAD``),
    so it matches the console's own tab bodies (Logic / Edit use ``scaled_px(CARD_PAD, minimum=6)``)
    instead of the tighter ``scaled_px(6)`` that read as crammed next to the right column."""
    from Zou_lab_control.frontend.qt_fluent import CARD_PAD, scaled_px

    v, _ = viewer
    expected = scaled_px(CARD_PAD, minimum=6)
    margins = v.plot_layout.contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == \
        (expected, expected, expected, expected), \
        "the Info rows-tab body uses CARD_PAD on every edge (matches the console tab bodies)"


def test_tab_overflow_popup_uses_rounded_card_single_source():
    """#1(a): the tab overflow ``...`` popup is rounded the SAME way every other Fluent popup is --
    a translucent, frameless :class:`_FluentRoundedMenu` that PAINTS the antialiased rounded card --
    NOT a bare opaque square ``QMenu`` relying on a stylesheet border-radius (which shows a corner nub
    and reads inconsistent beside the combo drop-down / Setting card)."""
    from PyQt5 import QtCore, QtWidgets
    from Zou_lab_control.frontend.qt_fluent import (
        FluentTabWidget, _FluentRoundedMenu, ensure_qt_app)

    ensure_qt_app()
    tabs = FluentTabWidget()
    for name in ("Monitor", "Logic", "Edit alpha", "Edit beta", "Edit gamma"):
        tabs.add_permanent_tab(QtWidgets.QWidget(), name)
    try:
        menu = tabs._overflow_menu()
        # the shared rounded-card menu subclass, not a plain QMenu
        assert isinstance(menu, _FluentRoundedMenu), \
            "the overflow popup is a _FluentRoundedMenu (the shared rounded-card single source)"
        assert type(menu) is not QtWidgets.QMenu, "it is NOT a bare square QMenu"
        # translucent + frameless so nothing opaque/square sits behind the painted arc
        assert menu.testAttribute(QtCore.Qt.WA_TranslucentBackground), \
            "the popup window is translucent (no opaque square behind the rounded arc)"
        assert bool(menu.windowFlags() & QtCore.Qt.FramelessWindowHint), "and frameless"
        # it paints the card itself (overrides paintEvent) -- the same drawRoundedRect recipe
        assert type(menu).paintEvent is not QtWidgets.QMenu.paintEvent, \
            "the menu paints its own rounded card (not the default square QMenu paint)"
        # one action per tab, and it selects the picked tab (behaviour unchanged by the chrome)
        assert len(menu.actions()) == tabs.count()
        menu.actions()[3].trigger()
        assert tabs.currentIndex() == 3, "picking a tab in the overflow menu selects it"
        menu.deleteLater()
    finally:
        tabs.deleteLater()


def test_readout_multiline_single_row_matches_line_edit_height():
    """#3(a): a ONE-line :class:`FluentReadoutMultiline` is EXACTLY as tall as the single-line
    :class:`FluentReadoutEdit`, so an Info row with a short value reads identically whether it is a
    line edit or the multi-line field (the reported 'the multiline sits a few px taller than the
    single-line one').  The expected height is DERIVED from a real FluentReadoutEdit (not a re-typed
    literal), so the two can only ever agree by construction."""
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.qt_fluent import (
        FluentReadoutEdit, FluentReadoutMultiline, ensure_qt_app)

    ensure_qt_app()
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    single = FluentReadoutEdit("42")
    multi = FluentReadoutMultiline("42")
    lay.addWidget(single)
    lay.addWidget(multi)
    host.show()
    try:
        QtWidgets.QApplication.processEvents()
        assert multi.height() == single.height(), (
            "a one-line multiline field is exactly the single-line readout-edit height "
            f"(multi={multi.height()} single={single.height()})")
    finally:
        host.close()
        host.deleteLater()


def test_readout_multiline_shows_all_lines_without_inner_scroll():
    """#3(b): a long value grows to show EVERY wrapped line with NO inner scrollbar -- the whole form
    scrolls in the Info column's outer scroll area instead.  The reported confusion was an inner
    scrollbar on a field that already displays everything ('it's all shown, why scroll?').  Pins the
    real invariants (not a brittle pixel-exact height): the inner vertical scrollbar is hard-disabled
    by policy (so it can never be drawn no matter the layout), the field height reserves room for every
    wrapped GLYPH line (>= line count x the font's glyph height, so no line's text is clipped), and it
    is taller than a one-line value."""
    from PyQt5 import QtCore, QtWidgets
    from Zou_lab_control.frontend.qt_fluent import FluentReadoutMultiline, ensure_qt_app

    ensure_qt_app()
    long_value = ("C:/Users/lab/Dropbox/experiments/rb87/2026-07-01/"
                  "overnight_survival_scan_run17_readout_calibration_final.npz")
    host = QtWidgets.QWidget()
    host.resize(240, 600)                                  # narrow -> the long path wraps to many lines
    lay = QtWidgets.QVBoxLayout(host)
    short = FluentReadoutMultiline("42")
    long = FluentReadoutMultiline(long_value)
    lay.addWidget(short)
    lay.addWidget(long)
    lay.addStretch(1)
    host.show()
    try:
        QtWidgets.QApplication.processEvents()
        long._adjust_height()
        QtWidgets.QApplication.processEvents()
        # the policy HARD-disables the inner scrollbar (the whole form scrolls, not the field) -- so no
        # inner scrollbar is ever drawn regardless of the exact pixel height
        assert long.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff, \
            "the multiline readout never shows its own vertical scrollbar"
        assert not long.verticalScrollBar().isVisible(), "so no inner scrollbar is drawn"
        # the height fully contains the wrapped text -- the residual absorb leaves nothing scrolled off
        assert long.verticalScrollBar().maximum() == 0, \
            f"the field shows every wrapped line (nothing hidden), got max={long.verticalScrollBar().maximum()}"
        # and (belt-and-braces) it reserves a full GLYPH row per wrapped line
        lines = long._visual_line_count()
        glyph_h = long.fontMetrics().height()
        assert lines > 1, "the long value wrapped onto several lines"
        assert long.height() >= lines * glyph_h, \
            f"the field is tall enough for every wrapped line's glyphs (h={long.height()} " \
            f"lines={lines} glyph_h={glyph_h})"
        assert long.height() > short.height(), "and it is taller than a one-line value"
    finally:
        host.close()
        host.deleteLater()


def test_readout_multiline_takes_no_max_lines_arg():
    """#3(b): the row cap is gone -- :class:`FluentReadoutMultiline` no longer accepts a ``max_lines``
    (it always shows every line), so the old capped/scrolling behaviour cannot be reintroduced by a
    caller."""
    import inspect
    from Zou_lab_control.frontend.qt_fluent import FluentReadoutMultiline

    params = inspect.signature(FluentReadoutMultiline.__init__).parameters
    assert "max_lines" not in params, "the multiline readout has no max_lines cap parameter anymore"


def test_raw_tab_body_uses_card_padding(viewer):
    """#2: the Raw tab's code editor lives in a body inset by the SAME component-library card padding
    (``CARD_PAD``) as the rows tabs -- previously the FluentCodeEdit was added bare (no container, no
    margin) so its text sat flush against the card edge, unlike every other Info / console tab."""
    from Zou_lab_control.frontend.qt_fluent import CARD_PAD, scaled_px

    v, _ = viewer
    expected = scaled_px(CARD_PAD, minimum=6)
    # the raw editor's parent is the padded body; read that layout's margins
    body = v.raw_info.parentWidget()
    margins = body.layout().contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == \
        (expected, expected, expected, expected), \
        "the Raw tab body uses CARD_PAD on every edge (matches the rows tabs / console tab bodies)"


def test_tab_overflow_menu_actions_are_not_checkable():
    """#4: the overflow ``...`` menu actions are NOT checkable -- a checkable action makes Qt reserve
    an indicator gutter and paint its own (un-Fluent) tick, the reported 'ugly check mark'.  The
    active tab is already the visibly-selected pivot, so the list needs no tick."""
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.qt_fluent import FluentTabWidget, ensure_qt_app

    ensure_qt_app()
    tabs = FluentTabWidget()
    for name in ("Monitor", "Logic", "Edit alpha", "Edit beta", "Edit gamma"):
        tabs.add_permanent_tab(QtWidgets.QWidget(), name)
    try:
        menu = tabs._overflow_menu()
        assert menu.actions(), "the overflow menu lists the tabs"
        assert not any(a.isCheckable() for a in menu.actions()), \
            "no overflow action is checkable (Qt would draw an un-Fluent tick)"
        menu.deleteLater()
    finally:
        tabs.deleteLater()


def test_flow_tab_always_shows_at_least_raw_to_plot(tmp_path):
    """#1: the Flow tab NEVER shows a "no data-flow" message.  A BARE figure (a plain ``plot(arr).save()``
    with no producing node -- so its npz carries no ``flow_graph``) still lays out a ``raw data -> plot``
    tree when reopened, because the viewer synthesizes the fallback from the ONE source
    (figure_capture.raw_data_flow_graph) at load time."""
    npz = tmp_path / "bare.npz"
    p = plot(np.linspace(0, 1, 20), np.random.default_rng(0).normal(size=20), display=False)
    out = p.save(str(npz))
    v = show_figure_viewer(out["data"] if isinstance(out, dict) else npz)
    try:
        boxes = set(v.flow_view._boxes.keys())
        assert boxes, "the Flow tab lays out a tree even for a figure with no recorded flow_graph"
        roles = {n.get("role") for n in v.flow_view._graph["nodes"]}
        assert {"plot", "raw data"} <= roles, f"a bare figure shows raw data -> plot, got roles {roles}"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


def test_embedded_console_reserves_tab_shadow_bleed(tmp_path):
    """#2: the embedded TaskConsole reserves its own tab-card drop-shadow bleed at the BOTTOM of its root
    layout (a host margin sits OUTSIDE the console and cannot un-clip a shadow clipped inside its tree --
    the real 'bottom shadow cut off' root cause).  The bottom content margin must be >= the shadow bleed."""
    from Zou_lab_control.frontend.qt_fluent import fluent_tab_shadow_margin
    npz = tmp_path / "bare.npz"
    p = plot(np.linspace(0, 1, 20), np.random.default_rng(0).normal(size=20), display=False)
    out = p.save(str(npz))
    v = show_figure_viewer(out["data"] if isinstance(out, dict) else npz)
    try:
        con = v.console                                  # the real EMBEDDED TaskConsole the viewer hosts
        assert con.embedded
        bottom = con.layout().contentsMargins().bottom()
        assert bottom >= fluent_tab_shadow_margin(), \
            f"embedded console bottom margin {bottom} must reserve the tab shadow bleed {fluent_tab_shadow_margin()}"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")


@pytest.mark.parametrize("cell", ["hist", "image"])
def test_edit_tab_param_edits_keep_the_snapshots_enlarged_cell(tmp_path, cell):
    """ROOT-CAUSE contract for the Edit tab's grid snapshot: the user enlarges a cell ON THE SNAPSHOT
    (GridPlot.focus, the notebook path), then edits relim / repeat_mode / fixed lo-hi in the Edit tab.
    EVERY edit must land through the snapshot's own in-place ``apply_param`` -- the SAME plotter-owned
    logic every surface uses -- so the snapshot object survives and the enlarged cell NEVER bounces
    back to the thumbnail grid (the reported bug: ``_edit_param`` used to hard-skip relim and nuke the
    whole snapshot through rebuild()).  Flipping INTO fixed must FREEZE the enlarged cell's current
    view (BaseLivePlot.apply_param seeds fixed_lo/hi from current_lims), never pin it to 0..1."""
    npz = _saved_grid_npz(tmp_path, cell=cell)
    v = show_figure_viewer(npz)
    try:
        v.console.refresh_once()
        card = v.console.cards[0]
        v.console._edit_card(card)
        editor = v.console._panel_editors[id(card)]
        snap = editor._plotter
        snap.focus(1)                                       # the user's double-click on the snapshot
        assert getattr(snap, "_focused", None) == 1
        focus_view = snap._focus_plotter
        before = focus_view.current_lims()
        editor._edit_param("relim", "fixed")                # the Edit tab's relim combo
        assert editor._plotter is snap and snap._focused == 1, \
            "a relim edit must stay on the enlarged snapshot cell (no re-snapshot)"
        assert snap._focus_plotter.current_lims() == before, \
            "fixed FREEZES the enlarged cell's current view (never the 0..1 default)"
        editor._edit_param("repeat_mode", "add")            # a key the enlarged view has no use for
        assert editor._plotter is snap and snap._focused == 1
        assert snap._focus_plotter is focus_view, \
            "an inapplicable key is STORED only -- never an unfocus/refocus rebuild (the flash/bounce)"
        editor.ed_fixed_lo.setText("0")
        editor.ed_fixed_hi.setText("100")
        editor._edit_fixed_lim()                            # the Edit tab's lo/hi inputs
        assert editor._plotter is snap and snap._focused == 1, \
            "typing fixed lo/hi must stay on the enlarged snapshot cell"
        assert snap._focus_plotter.current_lims() == (0.0, 100.0)
        # ... and the pin reaches the THUMBNAILS too (the one store -> cell-state -> redraw
        # mechanism cmap/bins use): back on the grid, every cell shows the operator's lo/hi.
        snap.unfocus()
        if cell == "hist":
            assert snap.site_axes[0].get_ylim() == (0.0, 100.0), \
                "a hist thumbnail's count axis honours the pinned lims"
        else:
            assert snap.site_axes[0].get_images()[0].get_clim() == (0.0, 100.0), \
                "an image thumbnail's colour scale honours the pinned lims"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")
