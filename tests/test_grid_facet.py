"""The grid as an AXIS-EXPANDER (facet) contracts.

A grid expands ONE axis of a measurement's canonical ``(R,P,*data_shape)`` block into N aligned
cells, each a standard ``sub_plot_kind`` panel of the remaining axes.  These contracts pin the
single slicing rule (``facet_cells``), the auto per-cell kind, the live in-place update path
(artists move, never rebuild), the focused-cell feed, and the console panel's end-to-end flow.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from Zou_lab_control.frontend.live import (
    GRID_CELL_BY_KIND,
    HistogramFigure,
    ImageCell,
    Live1D,
    Live2DDis,
    _IMAGE_ORIGIN,
    build_grid_figure,
    default_sub_plot_kind,
    facet_axis_labels,
    facet_cells,
    grid,
    grid_recipe_from_cells,
    normalize_facet,
    plot,
)

RNG = np.random.default_rng(11)
#: one 3-level-scan block (R=4, P=3*5*6 flat, data_shape=(1,)) reused across the contracts
BLOCK = RNG.normal(100.0, 8.0, size=(4, 3 * 5 * 6, 1))
PTS = (3, 5, 6)


# --------------------------------------------------------------------------- the ONE slicing rule
def test_facet_cells_slices_every_axis_group_per_the_shape_iron_law():
    """repeat / points:i / data:i each slice their OWN axis of (R,P,*data_shape); every
    cell keeps the remaining axes, shaped for its kind (hist pools, 2d frames, 1d curves)."""
    # points:0 -> one 2-D frame per OUTER scan step, repeats averaged
    cells = facet_cells(BLOCK, "points:0", sub_plot_kind="2d", points_shape=PTS)
    assert len(cells) == 3 and cells[0].shape == (5, 6)
    np.testing.assert_allclose(cells[1], BLOCK[:, 30:60, 0].mean(axis=0).reshape(5, 6))
    # repeat -> one distribution per repeat, everything left pooled as samples
    hist = facet_cells(BLOCK, "repeat", sub_plot_kind="hist", points_shape=PTS)
    assert len(hist) == 4 and hist[0].shape == (90,)
    np.testing.assert_allclose(hist[2], BLOCK[2].reshape(-1))
    # points:1 -> one curve per MIDDLE scan step (repeat averaged, rest flattened)
    curves = facet_cells(BLOCK, "points:1", sub_plot_kind="1d", points_shape=PTS)
    assert len(curves) == 5 and curves[0].shape == (18,)
    # data:0 -> one cell per site of a per-site vector
    per_site = RNG.normal(size=(3, 7, 6))
    sites = facet_cells(per_site, "data:0", sub_plot_kind="hist")
    assert len(sites) == 6 and sites[0].shape == (21,)
    np.testing.assert_allclose(sites[4], per_site[:, :, 4].reshape(-1))
    # repeat_mode verbs collapse the repeat axis for image/line cells (hist always pools)
    added = facet_cells(BLOCK, "points:0", sub_plot_kind="2d", points_shape=PTS, repeat_mode="add")
    np.testing.assert_allclose(added[0], BLOCK[:, :30, 0].sum(axis=0).reshape(5, 6))
    latest = facet_cells(BLOCK, "points:0", sub_plot_kind="2d", points_shape=PTS, repeat_mode="replace")
    np.testing.assert_allclose(latest[0], BLOCK[-1, :30, 0].reshape(5, 6))


def test_all_nan_facet_collapse_is_a_silent_gap_not_a_warning():
    """The up-front EMPTY grid a scanning task publishes (an all-NaN (repeat, points, 1) block before
    its first point) collapses to all-NaN cells BY INTENT -- a blank grid -- and must NOT spam numpy's
    benign 'Mean of empty slice' RuntimeWarning once per empty cell.  Pins the gap-safe average."""
    import warnings
    empty = np.full((4, 3 * 5 * 6, 1), np.nan)          # every repeat/point NaN -> every cell averages to NaN
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # ANY RuntimeWarning here fails the test
        cells = facet_cells(empty, "points:0", sub_plot_kind="2d", points_shape=PTS)
    assert len(cells) == 3 and all(c.shape == (5, 6) and np.isnan(c).all() for c in cells)


def test_normalize_facet_accepts_only_the_canonical_string_grammar():
    assert normalize_facet("repeat") == ("repeat", 0)
    assert normalize_facet("points:0") == ("points", 0)
    assert normalize_facet("points:2") == ("points", 2)
    assert normalize_facet("data:1") == ("data", 1)
    for bad in (
        None, "", "points", "data", "repeat:0", "repeat:3", "dim:0", "banana",
        "points:-1", "points:+1", "points:01", " points:0", "points:0 ",
        ("data", 1), ["points", 0], ("repeat",), ("unknown-axis", 1),
    ):
        with pytest.raises(ValueError):
            normalize_facet(bad)


def test_default_sub_plot_kind_follows_what_each_cell_has_left():
    """The ONE auto rule (#H3o's by-dimensionality): 2 remaining points axes -> an image cell, any
    other multi-valued remainder -> a curve, nothing left but repeats -> a distribution."""
    assert default_sub_plot_kind("points:0", points_shape=PTS) == "2d"
    assert default_sub_plot_kind("repeat", points_shape=(90,)) == "1d"
    assert default_sub_plot_kind("points:0", points_shape=(4, 7)) == "1d"
    assert default_sub_plot_kind("data:0", points_shape=(1,), data_shape=(35,)) == "hist"


def test_data_facet_preserves_remaining_data_shape_and_never_merges_it_with_p():
    """Faceting removes exactly one declared data_shape axis.  A remaining 2-D data tuple stays a
    2-D frame; a 1-D cell refuses to merge P with the remaining data axis into one vector."""
    block = np.arange(2 * 1 * 3 * 4 * 5, dtype=float).reshape(2, 1, 3, 4, 5)
    frames = facet_cells(block, "data:0", sub_plot_kind="2d", points_shape=(1,))
    assert len(frames) == 3 and all(frame.shape == (4, 5) for frame in frames)
    np.testing.assert_allclose(frames[2], block[:, 0, 2].mean(axis=0))

    matrix = np.arange(2 * 7 * 3, dtype=float).reshape(2, 7, 3)
    with pytest.raises(ValueError, match="cannot merge P with data_shape"):
        facet_cells(matrix, "repeat", sub_plot_kind="1d", points_shape=(7,))


@pytest.mark.parametrize("data_shape", [(1, 4, 5), (4, 1, 5), (4, 5, 1), (1, 1, 4, 5)])
def test_2d_facet_never_deletes_unselected_singleton_data_axes(data_shape):
    """Singleton length does not make a declared data axis disposable: a 2-D image is legal only
    when exactly two data axes remain, so every higher-rank form requires an explicit data facet or
    reduction instead of ``squeeze`` silently changing the schema."""
    block = np.zeros((2, 1, *data_shape), dtype=float)
    with pytest.raises(ValueError, match="exact 2-D frame"):
        facet_cells(block, "repeat", sub_plot_kind="2d", points_shape=(1,))
    with pytest.raises(ValueError, match="no unambiguous automatic renderer"):
        default_sub_plot_kind("repeat", points_shape=(1,), data_shape=data_shape)


def test_facet_auto_classifier_never_selects_an_impossible_renderer():
    """Auto and slicing share one retained-axis classifier: valid auto choices execute, while a
    2-D points grid carrying non-scalar data fails before a renderer can be selected."""
    cases = [
        ("points:0", (3, 4, 5), (1,), np.zeros((2, 60, 1))),
        ("repeat", (7,), (1,), np.zeros((2, 7, 1))),
        ("data:0", (1,), (3, 4, 5), np.zeros((2, 1, 3, 4, 5))),
        ("data:0", (1,), (5,), np.zeros((2, 1, 5))),
        ("repeat", (1,), (4, 5), np.zeros((2, 1, 4, 5))),
    ]
    for facet, points_shape, data_shape, block in cases:
        kind = default_sub_plot_kind(facet, points_shape=points_shape, data_shape=data_shape)
        assert facet_cells(block, facet, sub_plot_kind=kind, points_shape=points_shape)

    with pytest.raises(ValueError, match="no unambiguous automatic renderer"):
        default_sub_plot_kind("repeat", points_shape=(3, 4), data_shape=(2,))


def test_every_auto_classified_facet_is_executable_by_the_canonical_slicer():
    """For every axis group and representative retained rank, auto may either reject ambiguity or
    return a renderer that ``facet_cells`` can actually execute on the same canonical tensor."""
    executable = 0
    for points_shape in ((1,), (4,), (2, 3), (2, 3, 4)):
        for data_shape in ((), (1,), (5,), (3, 4), (2, 3, 4)):
            block = np.zeros((2, int(np.prod(points_shape)), *data_shape), dtype=float)
            facets = (["repeat"]
                      + [f"points:{i}" for i in range(len(points_shape))]
                      + [f"data:{i}" for i in range(len(data_shape))])
            for facet in facets:
                try:
                    kind = default_sub_plot_kind(
                        facet, points_shape=points_shape, data_shape=data_shape)
                except ValueError:
                    continue
                cells = facet_cells(
                    block, facet, sub_plot_kind=kind, points_shape=points_shape)
                group, index = normalize_facet(facet)
                expected = (2 if group == "repeat" else
                            points_shape[index] if group == "points" else data_shape[index])
                assert len(cells) == expected
                executable += 1
    assert executable > 0


# --------------------------------------------------- figure-level axis labels follow the facet (#6)
def test_facet_axis_labels_map_the_remaining_axes_to_the_cell_x_y():
    """The ONE remaining-axis -> (xlabel, ylabel) rule, mirroring facet_cells' slicing and the cell
    renderers' orientation: a 2d cell's frame is core.reshape(pts_rest) drawn through the shared
    _IMAGE_ORIGIN, so rows (pts_rest[0]) run along y and columns (pts_rest[1]) along x; a 1d cell's single remaining
    points axis is its x; a hist cell's samples ARE the value; a camera frame keeps pixel axes."""
    names = ["Bx", "By", "Bz"]
    # 3-D scan, 2d cells: each facet pick leaves a DIFFERENT pair of scan axes as the frame
    assert facet_axis_labels("points:0", "2d", points_shape=PTS, data_shape=(1,),
                             param_names=names) == ("Bz", "By")
    assert facet_axis_labels("points:1", "2d", points_shape=PTS, data_shape=(1,),
                             param_names=names) == ("Bz", "Bx")
    assert facet_axis_labels("points:2", "2d", points_shape=PTS, data_shape=(1,),
                             param_names=names) == ("By", "Bx")
    # repeat facet of a 1-D scan -> a 1d cell: x IS the swept axis, y the produced quantity
    assert facet_axis_labels("repeat", "1d", points_shape=(41,), data_shape=(1,),
                             param_names=["delay"], value_label="atom rate") == ("delay", "atom rate")
    # a remaining 2-D data_shape (a camera frame) keeps its pixel axes -- never a scan name
    assert facet_axis_labels("repeat", "2d", points_shape=(1,), data_shape=(20, 24),
                             param_names=names) == ("x (px)", "y (px)")
    # hist pools samples of the VALUE: x = the value's axis name, y = shot count
    assert facet_axis_labels("repeat", "hist", points_shape=PTS, data_shape=(1,),
                             value_label="signal (counts)") == ("signal (counts)", "Shots")
    # missing / too-short metadata degrades to the kind's stock defaults -- never raises
    assert facet_axis_labels("points:0", "2d", points_shape=PTS, data_shape=(1,)) == ("x (px)", "y (px)")
    assert facet_axis_labels("repeat", "1d", points_shape=(41,), data_shape=(1,)) == ("X", "Y")
    assert facet_axis_labels("repeat", "hist", points_shape=PTS, data_shape=(1,)) == ("Signal", "Shots")


def test_2d_facet_thumbnail_and_focus_share_one_imshow_origin():
    """Root-cause guard for the double-click FLIP: the grid THUMBNAIL (``ImageCell.draw``) and the
    enlarged FOCUS view (the standalone ``Live2DDis`` a double-click opens) render the SAME frame
    through ONE origin (``_IMAGE_ORIGIN``), so a 2d facet cell and its focus can never disagree on
    which way is up.  Before, the thumbnail hard-coded ``origin='lower'`` while the focus used
    ``'upper'`` -- so every 2d cell rendered VERTICALLY FLIPPED vs its enlarged view (their live
    update orders diverged, top-down vs bottom-up)."""
    from Zou_lab_control.frontend.style import style_context

    frame = np.zeros((4, 5))
    frame[0, :] = 1.0                                       # array ROW 0 (top of the ndarray) is the marker
    cell = ImageCell([frame])
    cell.prepare()
    with style_context():                                  # the grid renders under the frontend style
        fig, ax = plt.subplots()
        cell.draw(ax, 0)
        thumb_origin = cell._image_artists[0].origin
        plt.close(fig)                                     # release the bare fig before the frontend builds its own
        # the FOCUS is the standalone 2d fed the cell's OWN pixel coords -- the exact double-click path
        data_x, data_y = cell._pixel_coords(0)
        focus = plot(data_x, data_y, kind="2d", display=False)
        focus_origin = focus.image.origin
    plt.close("all")
    assert thumb_origin == focus_origin == _IMAGE_ORIGIN == "upper", (
        f"thumbnail origin {thumb_origin!r} != focus origin {focus_origin!r}: a 2d facet cell and "
        "its double-click focus would render flipped")


def test_grid_factory_derives_axis_labels_from_the_facet():
    """grid(value, facet=..., param_names=..., value_label=...) with no explicit labels derives the
    figure-level x/y names through the ONE facet_axis_labels rule, threads them to the fig.text
    slots AND the cell renderer (whose data_figure/focus_data read self.labels -- so the enlarged
    view and the saved npz carry the same names).  An explicit ``labels=`` still wins."""
    names = ["Bx", "By", "Bz"]
    g = grid(BLOCK, sub_plot_kind="2d", facet="points:0", points_shape=PTS,
             param_names=names, display=False)
    try:
        assert (g.labels[0], g.labels[1]) == ("Bz", "By")
        assert g._xlabel_text.get_text() == "Bz" and g._ylabel_text.get_text() == "By"
        assert tuple(g.cell_renderer.labels[:2]) == ("Bz", "By")   # focus/npz read the same labels
    finally:
        plt.close(g.fig)
    explicit = grid(BLOCK, sub_plot_kind="2d", facet="points:0", points_shape=PTS,
                    labels=("custom x", "custom y"), param_names=names, display=False)
    try:
        assert (explicit.labels[0], explicit.labels[1]) == ("custom x", "custom y")
    finally:
        plt.close(explicit.fig)
    plt.close("all")


# ------------------------------------------------------------------- live update: artists move
def test_update_cells_moves_existing_artists_in_place_for_every_family():
    """The live facet feed NEVER rebuilds a thumbnail's artists: the image's AxesImage, the hist's
    PolyCollection and the line's Line2D survive an update (identity) and show the new data."""
    g2 = grid(BLOCK, sub_plot_kind="2d", facet="points:0", points_shape=PTS, display=False)
    im0 = g2.cell_renderer._image_artists[0]
    g2.update_cells(facet_cells(BLOCK + 50, "points:0", sub_plot_kind="2d", points_shape=PTS))
    assert g2.cell_renderer._image_artists[0] is im0
    np.testing.assert_allclose(np.asarray(im0.get_array()),
                               (BLOCK + 50)[:, :30, 0].mean(axis=0).reshape(5, 6))
    gh = grid(BLOCK, sub_plot_kind="hist", facet="repeat", points_shape=PTS, display=False)
    c0 = gh.cell_renderer._bar_colls[0][0]
    gh.update_cells(facet_cells(BLOCK + 50, "repeat", sub_plot_kind="hist", points_shape=PTS))
    assert gh.cell_renderer._bar_colls[0][0] is c0
    gl = grid(BLOCK, sub_plot_kind="1d", facet="points:1", points_shape=PTS, display=False)
    l0 = gl.cell_renderer._line_artists[0]
    gl.update_cells(facet_cells(BLOCK + 50, "points:1", sub_plot_kind="1d", points_shape=PTS))
    assert gl.cell_renderer._line_artists[0] is l0
    plt.close("all")


def test_update_cells_feeds_only_the_enlarged_view_while_focused():
    """While a cell is enlarged (the grid's own focus OR the host's focus channel) the tick feeds
    ONLY the standalone focus view; the thumbnails wait for unfocus -- the same defer rule every
    display knob uses."""
    g = grid(BLOCK, sub_plot_kind="2d", facet="points:0", points_shape=PTS, display=False)
    g.focus(1)
    focus = g._focus_plotter
    thumb = g.cell_renderer._image_artists[0]
    before = np.asarray(thumb.get_array()).copy()
    g.update_cells(facet_cells(BLOCK + 9, "points:0", sub_plot_kind="2d", points_shape=PTS))
    assert g._focus_plotter is focus and g._focused == 1, "stays enlarged across live ticks"
    np.testing.assert_allclose(np.asarray(thumb.get_array()), before)   # thumbnails deferred
    np.testing.assert_allclose(np.asarray(focus.data_y[:, 0]),
                               (BLOCK + 9)[:, 30:60, 0].mean(axis=0).reshape(-1))
    g.unfocus()                                                          # thumbnails catch up
    np.testing.assert_allclose(np.asarray(g.cell_renderer.images[0]),
                               (BLOCK + 9)[:, :30, 0].mean(axis=0).reshape(5, 6))
    plt.close("all")


def test_update_cells_rejects_a_cell_count_change():
    g = grid(BLOCK, sub_plot_kind="2d", facet="points:0", points_shape=PTS, display=False)
    with pytest.raises(ValueError):
        g.update_cells([np.zeros((5, 6))] * 4)          # 4 cells for a 3-cell grid = a rebuild
    plt.close("all")


# --------------------------------------------------------------- the "1d" family is a real panel
def test_every_cell_family_enlarges_to_its_standard_panel():
    """GRID_CELL_BY_KIND's three families all focus into the REAL standalone panel of their kind --
    the whole point of the facet design: zero bespoke enlarged-view code."""
    expected = {"hist": HistogramFigure, "2d": Live2DDis, "1d": Live1D}
    assert set(GRID_CELL_BY_KIND) == set(expected)
    for kind, feed in (("hist", "repeat"), ("2d", "points:0"), ("1d", "points:1")):
        g = grid(BLOCK, sub_plot_kind=kind, facet=feed, points_shape=PTS, display=False)
        focus = g.build_focus_plotter(0, size="2x2", interactions=False)
        assert isinstance(focus, expected[kind]), f"{kind} cell must enlarge to {expected[kind].__name__}"
        plt.close(focus.fig)
        plt.close(g.fig)


def test_line_grid_recipe_roundtrip():
    """A 1d facet grid saves + reopens through the SAME recipe seam as hist/2d (append-only keys)."""
    g = grid(BLOCK, sub_plot_kind="1d", facet="points:0", points_shape=PTS, display=False)
    recipe = grid_recipe_from_cells(g)
    assert recipe["sub_plot_kind"] == "1d" and len(recipe["per_cell"]) == 3
    g2 = build_grid_figure(recipe, interactions=False, display=False)
    assert g2.sub_plot_kind == "1d" and g2.n_cells == 3
    np.testing.assert_allclose(g2.cell_renderer.ys[2], g.cell_renderer.ys[2])
    plt.close("all")


# ------------------------------------------------------------------------- console panel end-to-end
STRUCT = {"points_shape": (90,), "data_shape": (1,), "grid_shape": PTS}


def _facet_card(params, struct=STRUCT):
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["scan_y"],
                      source="value = signal", params=dict(params))
    return PanelCard(cfg, structure_provider=lambda _name: dict(struct))


def test_console_facet_panel_builds_updates_and_stays_enlarged():
    """The console grid panel bound to a live block: facet choices derive from the node's declared
    structure; the first tick builds, later ticks move cells in place (same plotter, same artists);
    an enlarged cell keeps updating and NEVER bounces; unfocus shows the latest data."""
    card = _facet_card({"facet": "points:0"})
    try:
        options = [card.facet_combo.itemData(i) for i in range(card.facet_combo.count())]
        assert {"repeat", "points:0", "points:1", "points:2"} <= set(options)
        assert "" not in options      # "(saved figure)" exists only when a recipe node is bound
        card._render(card._coerce(BLOCK), {})
        grid_plot = card.plotter
        assert card._param_kind() == "2d" and grid_plot.n_cells == 3
        artist = grid_plot.cell_renderer._image_artists[0]
        card._render(card._coerce(BLOCK + 30), {})
        assert card.plotter is grid_plot and grid_plot.cell_renderer._image_artists[0] is artist, \
            "a live tick moves the cells in place -- never a rebuild"

        class _Dbl:
            dblclick, button = True, 1

        ev = _Dbl()
        ev.inaxes = grid_plot.site_axes[1]
        card._on_grid_canvas_click(ev)
        focus = card.plotter
        card._render(BLOCK + 60, {})                     # focused branch coerces the raw block itself
        assert card._grid_focus is not None and card.plotter is focus, \
            "live ticks feed the ENLARGED cell and never bounce back to the grid"
        np.testing.assert_allclose(np.asarray(focus.data_y[:, 0]),
                                   (BLOCK + 60)[:, 30:60, 0].mean(axis=0).reshape(-1))
        card._unfocus_grid_cell()
        np.testing.assert_allclose(np.asarray(grid_plot.cell_renderer.images[2]),
                                   (BLOCK + 60)[:, 60:, 0].mean(axis=0).reshape(5, 6))
    finally:
        card.shutdown()
        plt.close("all")


def test_console_facet_choices_publish_only_data_tokens():
    card = _facet_card(
        {"facet": "data:0"},
        struct={"points_shape": (1,), "data_shape": (2, 3), "grid_shape": ()},
    )
    try:
        options = [card.facet_combo.itemData(i) for i in range(card.facet_combo.count())]
        assert set(options) == {"repeat", "data:0", "data:1"}
        assert all(isinstance(token, str) and normalize_facet(token) for token in options)
    finally:
        card.shutdown()


def test_console_facet_config_uses_none_for_absence_and_rejects_noncanonical_values():
    card = _facet_card({})
    try:
        assert card._facet() is None
        assert "facet" not in card.config.params
    finally:
        card.shutdown()
    with pytest.raises(ValueError, match="facet must be exactly"):
        _facet_card({"facet": ""})


def test_console_facet_change_rebuilds_to_the_new_structure():
    """Picking another facet axis rebuilds the panel to the new cell count AND per-cell kind (the
    auto rule follows the slice), and the pick round-trips through config.params (save/load)."""
    card = _facet_card({"facet": "points:0"})
    try:
        card._render(card._coerce(BLOCK), {})
        assert (card.plotter.n_cells, card.plotter.sub_plot_kind) == (3, "2d")
        index = next(i for i in range(card.facet_combo.count())
                     if card.facet_combo.itemData(i) == "repeat")
        card.facet_combo.setCurrentIndex(index)
        card._on_facet_changed(index)
        assert card.config.params["facet"] == "repeat"
        assert card.config.to_dict()["params"]["facet"] == "repeat"     # persists with the layout
        card._render(card._coerce(BLOCK), {})
        assert (card.plotter.n_cells, card.plotter.sub_plot_kind) == (4, "1d")
    finally:
        card.shutdown()
        plt.close("all")


def test_console_facet_switch_relabels_the_figure_axes():
    """End-to-end #6: a mot_field-style structure_provider (3-D scan with self-described
    param_names) bound to a console grid panel -- the figure-level x/y axis labels FOLLOW the
    facet pick through the SAME Setting path the operator uses (_on_facet_changed), because
    _build_facet_plotter hands facet_axis_labels' derivation to the ONE grid factory."""
    mot_struct = dict(STRUCT, param_names=["Bx", "By", "Bz"],
                      points_coords=[list(np.linspace(-12, 12, n)) for n in PTS])
    card = _facet_card({"facet": "points:2"}, struct=mot_struct)
    try:
        card._render(card._coerce(BLOCK), {})
        p = card.plotter
        assert (p._xlabel_text.get_text(), p._ylabel_text.get_text()) == ("By", "Bx")
        index = next(i for i in range(card.facet_combo.count())
                     if card.facet_combo.itemData(i) == "points:0")
        card.facet_combo.setCurrentIndex(index)
        card._on_facet_changed(index)          # the Setting path: persist + reset + rebuild
        card._render(card._coerce(BLOCK), {})
        p = card.plotter
        assert (p._xlabel_text.get_text(), p._ylabel_text.get_text()) == ("Bz", "By")
        # the Edit-tab snapshot shares the ONE builder -> same labels, no drift
        snap = card._build_facet_plotter(card._coerce(BLOCK), interactions=True)
        assert (snap._xlabel_text.get_text(), snap._ylabel_text.get_text()) == ("Bz", "By")
        plt.close(snap.fig)
    finally:
        card.shutdown()
        plt.close("all")


# --------------------------------------------------------- the camera-frame rules (user round R)
def test_camera_frame_facet_resolves_to_an_image_cell_per_repeat():
    """A camera block (repeat, 1, H, W) faceted on repeat leaves ONE 2-D frame per cell -> the auto
    rule picks '2d' (it used to fall through to a flattened '1d' curve); a per-scan-point frame
    resolves the same way; a curve stays a curve; and the 2d shaper NEVER silently picks one
    component of a non-trivial leftover data_shape -- it raises."""
    assert default_sub_plot_kind("repeat", points_shape=(1,), data_shape=(20, 24)) == "2d"
    assert default_sub_plot_kind("points:0", points_shape=(5,), data_shape=(20, 24)) == "2d"
    assert default_sub_plot_kind("data:0", points_shape=(11,), data_shape=(35,)) == "1d"
    frames = facet_cells(RNG.normal(size=(4, 1, 20, 24)), "repeat",
                         sub_plot_kind="2d", points_shape=(1,))
    assert len(frames) == 4 and frames[0].shape == (20, 24)
    with pytest.raises(ValueError, match="2-D frame"):
        facet_cells(np.zeros((2, 12, 3)), "repeat", sub_plot_kind="2d", points_shape=(3, 4))


def _camera_console(repeat=4):
    """A REAL console + a real CameraMeasurement (virtual sensor) -- the user's actual flow."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement
    ensure_qt_app()

    class _Cam:
        _exposure, _roi = 3.0, (0, 24, 0, 20)
        exposure = property(lambda s: s._exposure)
        roi = property(lambda s: s._roi)
        # the device contract's DERIVED (height, width) -- CameraMeasurement seeds its
        # data_shape from it at BUILD time (ROI order: (x, width, y, height))
        frame_shape = property(lambda s: (int(s._roi[3]), int(s._roi[1])))

        def configure(self, **kw):
            pass

        def acquire(self, frames=1, **kw):
            return [RNG.normal(200, 20, size=(20, 24)) for _ in range(int(frames))]

    hub = SignalHub()
    node = CameraMeasurement(hub, _Cam(), repeat=repeat)
    console = TaskConsole(hub=hub, state=default_console_state(), running_nodes=[node])
    console._timer.stop()
    return console, node


def test_add_panel_grid_shows_every_repeat_as_a_cell_and_rows_follow_the_kind(monkeypatch):
    """The WHOLE Add-Panel flow on a real console (round R): a new grid panel defaults to
    facet=repeat; binding the camera signal shows repeat COUNT cells (the plot pipeline must NOT
    reduce the repeat axis first -- that collapsed every facet grid to one cell); the resolved
    per-cell kind is 2d (a frame per repeat) and the Setting param rows are the 2d family; an
    explicit sub-plot pick switches the cells AND the rows; the Edit tab snapshots the facet grid
    (no 'no producing node' error)."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    console, node = _camera_console(repeat=4)
    try:
        node.step()                # ONE shot: the block carries 1 repeat, the declared ring is 4
        kc = console.kind_combo
        kc.setCurrentIndex(next(i for i in range(kc.count()) if kc.itemText(i) == "Plot: Site grid"))
        console._add_panel()
        card = console.cards[-1]
        assert card.config.params.get("facet") == "repeat"      # the Add-Panel grid IS the expander
        card.config.inputs = ["frame_0"]
        card.source_edit.setText("value = signal")
        card._apply_source()
        console.refresh_once()
        # The declared ring (repeat=4) fixes the cell count from the FIRST shot: not-yet-taken
        # repeats show as blank NaN placeholder cells, so the grid never rebuilds as shots fill.
        assert card.plotter is not None and card.plotter.n_cells == 4      # one cell PER REPEAT
        assert card.plotter.sub_plot_kind == "2d"                          # a frame per cell
        assert "cmap" in card.param_widgets and "bins" not in card.param_widgets
        first_plotter = card.plotter
        for _ in range(3):
            node.step()                                   # fill the remaining repeats
            console.refresh_once()
        assert card.plotter is first_plotter              # filled IN PLACE (update_cells fast path),
        assert card.plotter.n_cells == 4                  # never a per-shot full rebuild
        # explicit sub-plot pick -> hist cells + the hist rows replace the 2d rows
        index = next(i for i in range(card.sub_kind_combo.count())
                     if card.sub_kind_combo.itemData(i) == "hist")
        card.sub_kind_combo.setCurrentIndex(index)
        card._on_sub_kind_changed(index)
        console.refresh_once()
        assert card.plotter.sub_plot_kind == "hist" and card.plotter.n_cells == 4
        assert "bins" in card.param_widgets and "cmap" not in card.param_widgets
        # the Edit tab snapshots the facet grid through the SAME shared builder
        console._edit_card(card)
        editor = console._panel_editors[id(card)]
        editor.rebuild()
        from Zou_lab_control.frontend.live import GridPlot
        assert isinstance(editor._plotter, GridPlot) and editor._plotter.n_cells == 4
        assert "could not snapshot" not in editor.status.text()
    finally:
        console.shutdown()
        plt.close("all")


def test_every_display_param_of_the_sub_kind_is_consumed_by_the_grid():
    """MECHANICAL anti-drift (round S): every display ParamDecl of a cell family's standalone kind
    must be CONSUMED on the grid -- the cell family renders it (consume_param -> thumbnails redraw)
    through the SAME primitives the standalone plot uses (core fit_histogram, set_yscale, cmap).
    A new PANEL_PARAMS knob that silently does nothing on a grid fails here (the 'grid params are
    baked / the fit chooser is dead' class of bug the user hit twice)."""
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS

    per_kind_data = {
        "hist": [RNG.normal(100, 8, size=200) for _ in range(4)],
        "2d": [RNG.normal(100, 8, size=(6, 7)) for _ in range(4)],
        "1d": [RNG.normal(100, 8, size=30) for _ in range(4)],
    }
    for kind, data in per_kind_data.items():
        g = grid(data, sub_plot_kind=kind, display=False)
        try:
            cell = g.cell_renderer
            for spec in PANEL_PARAMS.get(kind, ()):
                if not spec.display:
                    continue
                current = getattr(cell, spec.key, spec.default)
                if spec.kind == "bool":
                    new = not bool(current)
                elif spec.kind == "choice":
                    new = next(c for c in spec.choices if c != current)
                else:
                    new = int(current if current is not None else spec.default) + 1
                assert g.store_display_param(spec.key, new) is True, \
                    f"{kind} grid must consume its standalone kind's {spec.key!r} (got a dead knob)"
        finally:
            plt.close(g.fig)
    plt.close("all")


def test_grid_hist_fit_draws_the_standard_curves_on_the_thumbnails():
    """apply_param('fit', 'double') on a hist grid fits + draws the SAME three curves the standalone
    dis draws (ONE primitive: core fit_histogram), keeps them tracking on a live in-place update,
    and seeds the enlarged view's fit -- the 'dis fit is missing on the grid' fix."""
    vals = [np.concatenate([RNG.normal(100, 8, 300), RNG.normal(200, 12, 200)]) for _ in range(4)]
    g = grid(vals, sub_plot_kind="hist", display=False)
    try:
        cell = g.cell_renderer
        # the grid fits by DEFAULT ("double" -- DEFAULT_HIST_FIT, the ONE source the Setting UI's
        # default reads too): what the UI says is what the thumbnails draw, from the first build
        assert cell.fit == "double"
        lines = cell._fit_lines[0]
        assert lines is not None and len(lines) == 3
        assert len(lines[0].get_xdata()) == 400 and len(lines[2].get_xdata()) == 400
        cell.set_cell_data(0, np.concatenate([RNG.normal(90, 8, 300), RNG.normal(210, 12, 200)]))
        g.update_cells([cell.values[k] for k in range(4)])
        assert len(cell._fit_lines[0][2].get_xdata()) == 400        # the fit tracks the moved samples
        g.apply_param("ylog", True)
        assert g.site_axes[0].get_yscale() == "log"                 # the dis's log-count rule, per cell
        focus = g.build_focus_plotter(0, size="2x2")
        assert focus._fit == "double"                               # the enlarged view inherits the pick
        plt.close(focus.fig)
    finally:
        plt.close(g.fig)
        plt.close("all")


def test_grid_general_fit_survives_notebook_focus_unfocus():
    """A general curve fit on the grid MUST survive a focus->unfocus round trip.  The notebook /
    Edit-tab enlarge path is ``GridPlot.focus`` / ``unfocus``; unfocus rebuilds the cells via
    ``_rebuild_grid`` -> ``init_core``, which re-asserts the stored VIEW knobs (the general fit) BY
    CONSTRUCTION.  Guards the ONE invariant 'a from-scratch cell draw re-applies the view knobs' so no
    rebuild path can silently wipe the grid's fit (the 'double-click a cell and return kills the fit'
    bug, reproducible in the Edit tab)."""
    cells = facet_cells(BLOCK, "points:0", sub_plot_kind="2d", points_shape=PTS)   # 3 image cells
    g = grid(cells, sub_plot_kind="2d", display=False)
    try:
        from Zou_lab_control.neutral_atom.core.fitting import FitRequest
        g.apply_param("fit_request", FitRequest("center").to_dict())

        def live_fit_artists() -> int:
            drawn = set()
            for ax in g.site_axes:
                drawn |= {id(c) for c in ax.get_children()}
            return sum(1 for arts in g._general_fit_artists.values()
                       for a in arts if id(a) in drawn)      # count ARTISTS still on a CURRENT axes

        before = live_fit_artists()
        assert before > 0, "the center fit must draw artists on the grid cells"
        g.focus(0)                                          # notebook focus (the Edit-tab enlarge path)
        g.unfocus()                                         # -> _rebuild_grid -> init_core -> _apply_view_knobs
        assert live_fit_artists() == before, \
            "the general fit was LOST across notebook focus/unfocus -- a rebuild path dropped the view knobs"
    finally:
        plt.close(g.fig)
        plt.close("all")


def test_grid_edge_label_policy_survives_every_display_toggle():
    """The shared-axes tick policy (round U): y tick labels ONLY on each row's leftmost cell, x tick
    labels ONLY on each column's lowest cell (including a ragged column's mid-grid bottom), nothing
    on inner cells -- and the rule must survive ylog / fit / bins toggles (set_yscale rebuilds the
    locators; the redraw paths must re-assert the ONE policy)."""
    vals = [RNG.normal(100, 8, size=200) for _ in range(8)]      # 3 cols -> ragged last row
    g = grid(vals, sub_plot_kind="hist", display=False)
    try:
        def check():
            ncols = g.ncols
            for k, ax in enumerate(g.site_axes):
                left = (k % ncols) == 0
                bottom = (k + ncols) >= g.n_cells
                assert (len(ax.get_yticks()) > 0) == left, f"cell {k} y-ticks vs leftmost rule"
                assert (len(ax.get_xticks()) > 0) == bottom, f"cell {k} x-ticks vs bottom rule"
        check()
        for key, value in (("ylog", True), ("fit", "none"), ("bins", 24), ("ylog", False)):
            g.apply_param(key, value)
            check()
    finally:
        plt.close(g.fig)
        plt.close("all")


def test_grid_focus_is_a_stock_2x2_figure_and_unfocus_restores():
    """Double-click zoom (round U): the enlarged cell is a STOCK 2x2 standalone panel -- the grid's
    own FIGURE resizes to the 2x2 spec (an embedded Qt canvas follows via refresh_design_size), and
    unfocus restores the grid-sized figure exactly."""
    from Zou_lab_control.frontend.live import panel_plot
    vals = [RNG.normal(100, 8, size=200) for _ in range(8)]
    ref = panel_plot(vals[0], kind="hist", size="2x2")
    ref_size = tuple(ref.fig.get_size_inches())
    plt.close(ref.fig)
    g = grid(vals, sub_plot_kind="hist", display=False, size="4x2", interactions=True)
    try:
        grid_size = tuple(g.fig.get_size_inches())
        g.focus(1)
        assert tuple(g.fig.get_size_inches()) == ref_size, "the zoom IS a stock 2x2 figure"
        g.unfocus()
        assert tuple(g.fig.get_size_inches()) == grid_size, "unfocus restores the grid size"
    finally:
        plt.close(g.fig)
        plt.close("all")
