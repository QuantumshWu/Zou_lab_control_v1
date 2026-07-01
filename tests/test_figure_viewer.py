"""Contract for the ``FigureViewer`` window (``exp.figure_viewer()`` / ``show_figure_viewer``).

The viewer is a PURE reopen-and-re-view GUI over a saved figure ``.npz``: no hardware, no session.
It loads through the SAME :class:`~Zou_lab_control.frontend.data_figure.SavedFigure` / ``DataFigure``
stack the notebook ``na.load_figure(...)`` uses, so these pins are all about the WINDOW wiring, not
the data layer (that is ``test_saved_figure_load.py``):

1. opening a ``.npz`` fills the Info panel (a non-empty set of rows + the raw-info dict) and builds
   the centre canvas + a live ``DataFigure``;
2. the plot-kind combo lists ``compatible_kinds`` (saved kind first) and switching kind rebuilds the
   canvas WITHOUT raising;
3. the fixed lo/hi row is ALWAYS in the layout (constant footprint -> no jitter) and only its inputs
   enable in ``relim == "fixed"``;
4. Save writes a fresh png + npz.

Runs headless (``QT_QPA_PLATFORM=offscreen``); the window is built + torn down in-process.
"""

from __future__ import annotations

import glob
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
from Zou_lab_control.frontend.data_figure import DataFigure  # noqa: E402
from Zou_lab_control.frontend.figure_viewer import FigureViewer  # noqa: E402


def _saved_hist_npz(tmp_path) -> Path:
    """Save a bimodal-readout hist figure with a stored view state, and return its ``.npz`` path."""
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(300.0, 20.0, 400), rng.normal(460.0, 20.0, 300)])
    p = plot(vals, kind="hist", display=False, update="once", data_figure=True)
    df = p.to_data_figure()
    info = {"source": "counts", "kind": "hist",
            "view": {"relim": "fixed", "fixed_lo": 0.0, "fixed_hi": 200.0,
                     "unit_index": 0, "cmap": "", "repeat_mode": "pool"}}
    out = df.save(str(tmp_path / "readout"), extra_info=info)
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


def test_open_fills_info_and_builds_canvas(viewer):
    v, _ = viewer
    # (1) the Info panel has content (one row per fact) + the raw-info dict is populated
    assert v.info_layout.count() > 0, "the Info panel must list what the file holds"
    assert v.raw_info.text(), "the raw-info field must show the full stored info dict"
    assert "source" in v.raw_info.text(), "raw info exposes EVERY stored key (e.g. 'source')"
    # (2) the centre canvas + its DataFigure were built
    assert v._canvas is not None, "the centre canvas must be built from the reopened figure"
    assert isinstance(v._df, DataFigure) and v._df.fig is not None


def test_kind_combo_lists_compatible_kinds_and_switch_rebuilds(viewer):
    v, _ = viewer
    # the combo carries the saved kind FIRST, then the other compatible kinds (data by key)
    keys = [v.kind_combo.itemData(i) for i in range(v.kind_combo.count())]
    assert keys[0] == v._saved.kind == "hist"
    assert set(keys) == set(v._saved.compatible_kinds())
    others = [i for i in range(v.kind_combo.count()) if v.kind_combo.itemData(i) != v._saved.kind]
    assert others, "a 1-D save must offer more than one compatible kind"
    before = v._canvas
    v.kind_combo.setCurrentIndex(others[0])       # switch plotter -> rebuild, must not raise
    assert v._canvas is not None and v._canvas is not before, "switching kind rebuilds the canvas"


def test_fixed_lim_row_is_permanent_and_only_enables_in_fixed(viewer):
    v, _ = viewer
    # the saved view was relim=fixed, so the row is visible + inputs enabled
    assert v.relim_combo.currentText() == "fixed"
    assert v.fixed_row.isVisible() and v.fixed_lo.isEnabled() and v.fixed_hi.isEnabled()
    # switching to tight must NOT hide the row (no setVisible jitter) -- only DISABLE the inputs
    v.relim_combo.setCurrentText("tight")
    assert v.fixed_row.isVisible(), "the lo/hi row stays in the layout (constant footprint, no jump)"
    assert not v.fixed_lo.isEnabled() and not v.fixed_hi.isEnabled()
    # back to fixed re-enables them, still same permanent row
    v.relim_combo.setCurrentText("fixed")
    assert v.fixed_row.isVisible() and v.fixed_lo.isEnabled()


def test_save_writes_png_and_npz(viewer, tmp_path):
    v, _ = viewer
    out_dir = tmp_path / "resaved"
    out_dir.mkdir()
    v.save_dir_edit.setText(str(out_dir))
    v.save()
    pngs = glob.glob(str(out_dir / "*.png"))
    npzs = glob.glob(str(out_dir / "*.npz"))
    assert len(pngs) == 1 and len(npzs) == 1, "Save writes exactly one png + one matching npz"
    assert "saved" in v.status.text()


def test_folder_open_fills_gallery(tmp_path):
    """Opening a FOLDER lists every .npz as a gallery row (a calibration run drops one per site)."""
    for i in range(3):
        _saved_hist_npz(tmp_path)               # three npz in one folder (names differ by timestamp/seq)
    npz_count = len(glob.glob(str(tmp_path / "*.npz")))
    v = show_figure_viewer(str(tmp_path))
    try:
        assert len(v._gallery_buttons) == npz_count, "every .npz in the folder is a gallery row"
        assert v._canvas is not None, "opening a folder selects + renders its first figure"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")
