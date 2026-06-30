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
