"""Contracts for the frontend DRY / latent-bug single-source fixes (review batch).

These pin five "a rule that CAN be a test MUST be one" guards so the de-duplicated
sources cannot silently drift back into copies:

  * the side distribution-band count(x)-axis upper limit is ONE rule
    (``live._dist_count_xlim``) -- every band's ``axdis`` upper limit equals it, so
    the five call sites cannot re-split into two headroom formulas;
  * ``Live1D.update_core`` grows a Line2D for every column added AFTER init, so a
    notebook ``update_point(mode='create')`` repeat is actually drawn;
  * the scalar validators ``_positive_float`` / ``_strict_bool`` live in exactly one
    place (``frontend/_validate.py``);
  * the three retired pulse_gui layout orphans are gone.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

_FRONTEND = Path(__file__).resolve().parent.parent / "Zou_lab_control" / "frontend"


# --------------------------------------------------------------- dist count xlim
def test_dist_count_xlim_is_monotonic_and_floored():
    from Zou_lab_control.frontend.live import _dist_count_xlim

    # never below the 10 floor, even for an empty / all-zero band
    assert _dist_count_xlim(np.array([])) == 10
    assert _dist_count_xlim(np.zeros(8)) == 10
    # headroom: the tallest bar never sits flush against the axis edge (peak < limit)
    for peak in (3, 10, 40, 123):
        n = np.array([peak, 1, 0], dtype=float)
        lim = _dist_count_xlim(n)
        assert lim > peak, f"peak {peak} -> xlim {lim} leaves no headroom"
    # the rule is a pure function of the peak: equal peaks -> equal limit regardless of shape
    assert _dist_count_xlim(np.array([40, 0])) == _dist_count_xlim(np.array([0, 40, 5, 40]))


def test_all_distribution_bands_share_one_count_xlim(monkeypatch):
    """Every side-band plotter's ``axdis`` count-axis upper limit equals ``_dist_count_xlim(plotter.n)``
    after a real update -- so Live2DDis / LiveSiteMap / LiveLiveDis (INIT and UPDATE) cannot drift into
    two competing headroom formulas the way they had (no-headroom vs 1.5x)."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.live import _dist_count_xlim

    console = dt.demo_console(shots=25)
    try:
        band_plotters = [
            c.plotter for c in console.cards
            if getattr(c.plotter, "axdis", None) is not None and getattr(c.plotter, "n", None) is not None
        ]
        assert band_plotters, "demo board should carry at least one side-distribution panel"
        for p in band_plotters:
            kind = type(p).__name__
            expected = _dist_count_xlim(p.n)
            assert int(p.axdis.get_xlim()[1]) == expected, (
                f"{kind} axdis count-xlim {p.axdis.get_xlim()[1]} != single-source {expected}")
    finally:
        console.shutdown()


# ------------------------------------------------------- Live1D 'create' columns
def test_live1d_grows_lines_for_created_repeat_columns(monkeypatch):
    """A notebook ``update_point(mode='create')`` pads ``data_y`` with a new column per repeat; the
    plotter must grow a Line2D per column (frozen-at-show ``self.lines`` was the latent bug) and each
    line's ydata must match its column."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import Zou_lab_control.frontend as zf

    n = 5
    plot = zf.Live1D(np.arange(n), np.full(n, np.nan), labels=("Index", "Signal", "Signal")).show(display=False)
    try:
        assert plot.data_y.shape[1] == 1 and len(plot.lines) == 1
        # repeat 1: fill column 0
        for i in range(n):
            plot.update_point(i, float(i), mode="create", repeat_cur=1, draw=False)
        # repeats 2 and 3: each create-pass adds a NEW column
        for rep in (2, 3):
            for i in range(n):
                plot.update_point(i, float(i + 10 * (rep - 1)), mode="create", repeat_cur=rep, draw=False)
            assert len(plot.lines) == plot.data_y.shape[1] == rep, (
                f"repeat {rep}: {len(plot.lines)} lines for {plot.data_y.shape[1]} columns")
        # every line draws its OWN column (not the frozen first column)
        for col, line in enumerate(plot.lines):
            ydata = np.asarray(line.get_ydata(), dtype=float)
            assert np.allclose(ydata, plot.data_y[:, col], equal_nan=True), f"line {col} ydata != column {col}"
    finally:
        import matplotlib.pyplot as plt
        plt.close(plot.fig)


# ----------------------------------------------------------- validators single-source
def _defs_named(path: Path, names: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in names]


def test_scalar_validators_defined_once_in_validate_module():
    targets = {"_positive_float", "_strict_bool"}
    hits = {}
    for py in _FRONTEND.glob("*.py"):
        for name in _defs_named(py, targets):
            hits.setdefault(name, []).append(py.name)
    for name in targets:
        owners = hits.get(name, [])
        assert owners == ["_validate.py"], f"{name} defined in {owners}, must be ONLY _validate.py"


# ---------------------------------------------------------------- pulse_gui orphans
def test_pulse_gui_layout_orphans_removed():
    src = (_FRONTEND / "pulse_gui.py").read_text(encoding="utf-8")
    for dead in ("_bar_title", "_delay_edit_width", "_period_top_label_width"):
        assert f"def {dead}" not in src, f"{dead} is a retired orphan and must be deleted"


# ------------------------------------------------------- suspend_draws silences present
def test_suspend_draws_silences_present_and_compose():
    """suspend_draws == the WHOLE draw is silent, present() included.  present() escaping
    through flush_events()->processEvents() ran a slice of the Qt event loop inside the
    "atomic" panel-resize transaction; on a SHRINK Qt had already invalidated the old card
    area on the unfrozen parent board, so that pump painted board background over the panel
    -- the "panel vanishes then rebuilds" flash (a GROW leaves no board dirt, hence only
    shrinking flashed).  Pin: inside the block draw() must not touch the canvas at all;
    after the block the same draw() really presents."""
    from Zou_lab_control.frontend.live import Live1D

    plot = Live1D(np.arange(8.0).reshape(-1, 1))
    plot.show(display=False)                 # build the figure (undisplayed), as panel hosts do
    canvas = plot.fig.canvas
    calls: list[str] = []
    orig = {n: getattr(canvas, n) for n in ("flush_events", "draw_idle")}
    try:
        for name in orig:
            setattr(canvas, name, lambda *a, _n=name, **k: calls.append(_n))
        with plot.suspend_draws():
            plot.draw()
        assert calls == [], f"draw() escaped suspend_draws via {calls}"
        plot.draw()
        assert calls, "after the block the same draw() must actually present"
    finally:
        for name, fn in orig.items():
            setattr(canvas, name, fn)
        import matplotlib.pyplot as plt
        plt.close(plot.fig)
# ------------------------------------------------- image family band update single-source (#25)
def _class_method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(f"class {class_name} not found in {path.name}")


def test_image_family_band_and_display_update_single_source():
    """The image family's per-tick behaviour (side-band update, clim drag, decimated display
    refresh + its imshow/callback wiring) is ONE implementation on BaseLivePlot -- Live2DDis and
    LiveSiteMap may only supply the two hooks (``_band_image`` / ``_display_source``) and thin
    ``apply_relim_now`` delegators, never a second method body (the site map's hand copies had
    already drifted: no dead-band, dragged clim lines cleared every tick, 'off' re-autoscaled)."""
    live_py = _FRONTEND / "live.py"
    shared = {"update_clim", "_refresh_display_image", "_init_display_image",
              "_update_distribution_band", "_band_apply_relim_now"}
    base = _class_method_names(live_py, "BaseLivePlot")
    assert shared <= base, f"BaseLivePlot must own the shared image-family layer, missing {shared - base}"
    for cls in ("Live2DDis", "LiveSiteMap"):
        dup = _class_method_names(live_py, cls) & (shared | {"_update_distribution"})
        assert not dup, f"{cls} re-defines shared image-family methods {dup} -- ONE implementation only"
    # and the shared method bodies exist exactly ONCE in the file (no third copy anywhere)
    tree = ast.parse(live_py.read_text(encoding="utf-8"))
    counts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in shared:
            counts[node.name] = counts.get(node.name, 0) + 1
    assert all(v == 1 for v in counts.values()), f"duplicated shared methods: {counts}"


def test_sitemap_band_inherits_deadband_and_drag_preservation(monkeypatch):
    """Behavioural pin of the single-source band update on the SITE MAP (the drifted copy):
    (1) DEAD-BAND -- an in-band frame wobble commits NO new colour limit (no per-tick clim/tick
    flicker, the blit floor survives); (2) drag PRESERVATION -- clim lines dragged OUTSIDE the
    data range survive an update (the old copy reset them unconditionally every tick);
    (3) a line left INSIDE the data range (it would clip data) is still reset, like Live2DDis."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import matplotlib.pyplot as plt
    from Zou_lab_control.frontend.live import LiveSiteMap

    rng = np.random.RandomState(0)
    frame = rng.normal(600.0, 30.0, size=(24, 24))
    centers = np.array([[6.0, 6.0], [18.0, 18.0]])
    p = LiveSiteMap(centers, np.array([1.0, 0.0]), image=frame,
                    labels=("x", "y", "counts")).show(display=False)
    try:
        p.update(draw=False)                              # settle the committed limits once
        committed = (p.ylim_min, p.ylim_max)
        # (1) dead-band: a small in-band wobble keeps the committed colour limit
        p.set_background(frame + rng.normal(0.0, 1.0, frame.shape))
        p.update(draw=False)
        assert (p.ylim_min, p.ylim_max) == committed, "in-band frame noise must NOT recompute the clim"
        assert p._band_image().get_clim() == committed, "the frame clim must stay the committed range"
        # (2) drag preservation: lines dragged OUTSIDE the data range survive the next update
        lo = float(np.nanmin(p._frame_values())) - 50.0
        hi = float(np.nanmax(p._frame_values())) + 50.0
        p.line_l.set_ydata([lo, lo]); p.line_h.set_ydata([hi, hi]); p.update_clim()
        p.update(draw=False)
        assert float(p.line_l.get_ydata()[0]) == lo and float(p.line_h.get_ydata()[0]) == hi,             "a dragged-out clim line must survive a steady update (was: cleared every tick)"
        # (3) a line INSIDE the data range would clip data -> reset to the committed limits
        mid = float(np.nanmean(p._frame_values()))
        p.line_l.set_ydata([mid, mid]); p.update_clim()
        p.update(draw=False)
        assert float(p.line_l.get_ydata()[0]) == p.ylim_min, "a clipping clim line is reset (2D parity)"
    finally:
        plt.close(p.fig)


# --------------------------------------------- embedded-canvas identity predicate (#29)
def test_embedded_canvas_identity_predicate_single_source():
    """The "is this the embedded Qt canvas" fact is ONE predicate (``live._is_embedded_canvas``
    reading the ``_zlc_embedded`` marker) -- live.py must not re-spell a class-name string compare
    (the old three copies had already drifted), and the predicate must hold for SUBCLASSES (the
    string test silently dropped them back to the notebook draw_idle path)."""
    src = (_FRONTEND / "live.py").read_text(encoding="utf-8")
    assert src.count('"EmbeddedFigureCanvas"') == 0,         "live.py must test canvas identity ONLY via _is_embedded_canvas (marker), never the class name"
    from Zou_lab_control.frontend.live import _is_embedded_canvas

    assert not _is_embedded_canvas(None)
    assert not _is_embedded_canvas(object())
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.qt_canvas import EmbeddedFigureCanvas
    if EmbeddedFigureCanvas is None:
        pytest.skip("matplotlib-qt missing")
    assert EmbeddedFigureCanvas._zlc_embedded is True     # the marker rides the class
    class _Sub(EmbeddedFigureCanvas):                      # a subclass keeps the blit path
        pass
    assert _is_embedded_canvas.__doc__ and "_zlc_embedded" in _is_embedded_canvas.__doc__
    assert bool(getattr(_Sub, "_zlc_embedded", False)), "subclasses must inherit the marker"


# ------------------------------------------------------- param_widgets stays a leaf (#30)
def test_param_widgets_is_a_leaf_no_task_console_back_edge():
    """param_widgets is the LEAF the console depends on -- it must never import task_console,
    not even lazily inside a function body (the old lazy back-imports made a cycle that
    contradicted its own docstring).  AST-scanned so a docstring MENTION stays legal."""
    tree = ast.parse((_FRONTEND / "param_widgets.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "task_console" not in str(node.module or ""), \
                f"param_widgets imports task_console at line {node.lineno} -- the leaf grew a back-edge"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "task_console" not in alias.name, \
                    f"param_widgets imports task_console at line {node.lineno} -- the leaf grew a back-edge"


def test_grouped_picker_helpers_single_source_in_param_widgets():
    """The grouped-signal-picker cluster is defined ONCE, in param_widgets (task_console
    forward-imports it) -- so the console cannot regrow a private copy."""
    targets = {"coerce_short_labels", "fill_grouped_signal_combo", "read_editable_combo",
               "grouped_signal_items", "signal_tree_groups", "signal_state"}
    hits = {}
    for py in _FRONTEND.glob("*.py"):
        for name in _defs_named(py, targets):
            hits.setdefault(name, []).append(py.name)
    for name in sorted(targets):
        assert hits.get(name) == ["param_widgets.py"],             f"{name} defined in {hits.get(name)}, must be ONLY param_widgets.py"
