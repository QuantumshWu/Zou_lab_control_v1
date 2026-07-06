"""The signal-value -> panel-input adapter lives WITH the plots (live.coerce_panel_value),
NOT in task_console.  The console only gathers inputs + dispatches; it holds ZERO per-kind
reshape logic.  These contracts pin both the mapping and the decoupling."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

from Zou_lab_control.frontend.live import coerce_panel_value


def test_coerce_maps_each_kind_to_its_plot_input():
    # 2d: a data-shape (H, W) frame IS the image, at native size (no cap)
    frame = np.arange(1200 * 1920, dtype=float).reshape(1200, 1920)
    img = coerce_panel_value("2d", frame, structure={"data_shape": (1200, 1920), "grid_shape": ()})
    assert img.shape == (1200, 1920) and np.array_equal(img, frame)
    # 2d scan: flat points un-flatten to grid_shape
    pts = np.arange(35, dtype=float)
    m = coerce_panel_value("2d", pts, structure={"data_shape": (), "grid_shape": (5, 7)})
    assert m.shape == (5, 7)
    # 1d: a 2-D data core flattens to one trace unless repeat_mode == "create"
    assert coerce_panel_value("1d", frame[:4, :4], structure={"data_shape": (4, 4), "grid_shape": ()}).ndim == 1
    assert coerce_panel_value("1d", frame[:4, :4], structure={"data_shape": (4, 4), "grid_shape": ()},
                              repeat_mode="create").ndim == 2
    # monitor: a scalar
    assert coerce_panel_value("monitor", np.array([[3.5]])) == 3.5
    # sites: one value per site (a 'create' concat collapses back to n_sites)
    got = coerce_panel_value("sites", np.arange(70, dtype=float), structure={"data_shape": (35,), "grid_shape": ()})
    assert got.shape == (35,)
    # pulse / grid pass their structured / raw block through
    class _Seq: ...
    seq = _Seq()
    assert coerce_panel_value("pulse", seq) is seq


def test_coerce_rejects_a_non_image_2d_value():
    with pytest.raises(ValueError):
        coerce_panel_value("1d", np.zeros(0), structure=None)   # empty
    with pytest.raises(ValueError):
        coerce_panel_value("2d", np.arange(10.0), structure={"data_shape": (10,), "grid_shape": ()})  # 1-D, no map


def test_coerce_2d_undeclared_data_shape_falls_back_to_the_value():
    """UNDECLARED is not DECLARED-non-image: a producer that honestly does not know its frame
    size yet (a camera backend without ``frame_shape``, before its first frame) carries
    ``data_shape=()`` -- the panel then reads the (repeat-reduced) block's own 2-D core, the
    same fallback a structure-less custom expression gets.  The reject above stays reserved
    for a REAL contradiction (a declared non-2-D data core on an image panel)."""
    block = np.arange(96 * 128, dtype=float).reshape(1, 96, 128)
    img = coerce_panel_value("2d", block, structure={"data_shape": (), "grid_shape": ()})
    assert img.shape == (96, 128) and np.array_equal(img, block[0])


def test_console_coerce_is_a_thin_dispatcher_no_per_kind_logic():
    """PanelCard._coerce holds ZERO per-kind reshape logic -- it dispatches to the plot layer.
    Re-adding a per-kind branch (an ``if kind == ...`` reshape) to the console would resurrect the
    coupling this decoupling removed (a 192-stride image cap once lived here)."""
    from Zou_lab_control.frontend.task_console import PanelCard
    src = inspect.getsource(PanelCard._coerce)
    assert "coerce_panel_value" in src                     # delegates to the plot-layer adapter
    # no per-kind branching, no reshape CALL in the console (comments may mention the word)
    assert "if kind ==" not in src and ".reshape(" not in src
