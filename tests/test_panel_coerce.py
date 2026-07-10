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
    # 2d: a reduced canonical P=1, data_shape=(H,W) frame is the native image.
    frame = np.arange(1200 * 1920, dtype=float).reshape(1200, 1920)
    img = coerce_panel_value(
        "2d", frame[None],
        structure={"points_shape": (1,), "data_shape": (1200, 1920), "grid_shape": ()})
    assert img.shape == (1200, 1920) and np.array_equal(img, frame)
    # 2d scan: the scalar data cell is retained while flat P restores its logical point grid.
    pts = np.arange(35, dtype=float)[:, None]
    m = coerce_panel_value(
        "2d", pts,
        structure={"points_shape": (5, 7), "data_shape": (1,), "grid_shape": (5, 7)})
    assert m.shape == (5, 7)
    # 1d: P and the one-dimensional data_shape remain distinct; create adds columns, not axes.
    trace_structure = {"points_shape": (4,), "data_shape": (3,), "grid_shape": ()}
    assert coerce_panel_value("1d", np.zeros((4, 3)), structure=trace_structure).shape == (4, 3)
    assert coerce_panel_value(
        "1d", np.zeros((4, 6)), structure=trace_structure, repeat_mode="create").shape == (4, 6)
    # monitor: a scalar
    assert coerce_panel_value("monitor", np.array([[3.5]])) == 3.5
    # sites: one point carrying the complete per-site data axis.
    got = coerce_panel_value(
        "sites", np.arange(35, dtype=float)[None],
        structure={"points_shape": (1,), "data_shape": (35,), "grid_shape": ()})
    assert got.shape == (35,)
    # pulse / grid pass their structured object / canonical raw tensor through.
    class _Seq: ...
    seq = _Seq()
    assert coerce_panel_value("pulse", seq) is seq
    grid = np.zeros((2, 4, 3))
    assert coerce_panel_value(
        "grid", grid,
        structure={"points_shape": (4,), "data_shape": (3,), "grid_shape": ()}) is grid


def test_coerce_rejects_a_non_image_2d_value():
    with pytest.raises(ValueError):
        coerce_panel_value("1d", np.zeros(0), structure=None)   # empty
    with pytest.raises(ValueError):
        coerce_panel_value(
            "2d", np.arange(10.0)[None, :],
            structure={"points_shape": (1,), "data_shape": (10,), "grid_shape": ()})


def test_coerce_2d_external_expression_treats_the_complete_array_as_one_datum():
    """Only the explicit external-data boundary lacks a SignalSchema.  A transformed/raw
    expression may therefore hand over a complete 2-D image, while registered producers must
    always use the canonical P/data axes tested above."""
    image = np.arange(96 * 128, dtype=float).reshape(96, 128)
    got = coerce_panel_value("2d", image, structure=None)
    assert got.shape == (96, 128) and np.array_equal(got, image)


def test_console_coerce_is_a_thin_dispatcher_no_per_kind_logic():
    """PanelCard._coerce holds ZERO per-kind reshape logic -- it dispatches to the plot layer.
    Re-adding a per-kind branch (an ``if kind == ...`` reshape) to the console would resurrect the
    coupling this decoupling removed (a 192-stride image cap once lived here)."""
    from Zou_lab_control.frontend.task_console import PanelCard
    src = inspect.getsource(PanelCard._coerce)
    assert "coerce_panel_value" in src                     # delegates to the plot-layer adapter
    # no per-kind branching, no reshape CALL in the console (comments may mention the word)
    assert "if kind ==" not in src and ".reshape(" not in src
