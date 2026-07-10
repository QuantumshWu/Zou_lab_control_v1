"""Contract (#H3o): the three measurement axes -- repeat / points / data -- are ORTHOGONAL, and the
plot AUTO-reshapes by the DATA dimensionality (declared ``data_shape``), NOT a size threshold:

* ``data_shape`` is 1-D (a scan's series) -> each data series is its OWN line; the data does NOT
  reshape into an image.
* ``data_shape`` is 2-D (a camera frame H*W) -> only a 2-D image view accepts it.  A 1-D view must
  not flatten away those axes; the user first chooses a component or explicit reduction.
* ``repeat_mode='create'`` adds lines only for one-dimensional per-point data.
* a flattened 2-D scan's points reshape to an image via ``grid_shape``.

This is the machine guard that replaces the deleted ``_DIM_MULTILINE_MAX`` heuristic; it encodes the
corner-case matrix verified during the redesign.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets  # noqa: E402
from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig  # noqa: E402
from Zou_lab_control.frontend.live import reduce_repeat  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

R, H, W, P, N0, N1 = 5, 12, 16, 8, 4, 5
# (block, structure) per node kind, mirroring the real node contracts.
NODES = {
    "camera":  (np.random.default_rng(0).random((R, 1, H, W)), {"points_shape": (1,), "data_shape": (H, W), "grid_shape": ()}),
    "scan_d1": (np.random.default_rng(1).random((R, P, 1)),    {"points_shape": (P,), "data_shape": (1,), "grid_shape": ()}),
    "scan_d3": (np.random.default_rng(2).random((R, P, 3)),    {"points_shape": (P,), "data_shape": (3,), "grid_shape": ()}),
    "scan_2d": (np.random.default_rng(3).random((R, N0 * N1, 1)), {"points_shape": (N0 * N1,), "data_shape": (1,), "grid_shape": (N0, N1)}),
}


def _card(kind, mode, struct):
    return PanelCard(PanelConfig(kind=kind, role="plot", source="value = signal", inputs=["sig"],
                                 params={"repeat_mode": mode, "cmap": "viridis"}),
                     structure_provider=lambda name, s=struct: s)


@pytest.mark.parametrize("node,mode,expected_lines", [
    ("scan_d1", "average", 1),    # one series -> one line
    ("scan_d3", "average", 3),    # data_shape=(3,) -> three lines (not repeat)
    ("scan_2d", "average", 1),    # a flattened grid as one trace on a 1-D panel
    ("scan_d1", "create",  R),    # one line per repeat
    ("scan_d3", "create",  R * 3),  # repeat x one-dimensional data_shape, orthogonal columns
    ("scan_2d", "create",  R),    # one flattened-grid trace per repeat
])
def test_1d_panel_line_count(node, mode, expected_lines):
    blk, st = NODES[node]
    c = _card("1d", mode, st)
    try:
        c.refresh({"sig": blk, "shot": 1})
        assert len(c.plotter.lines) == expected_lines
    finally:
        c.shutdown()


@pytest.mark.parametrize("mode", ["average", "create"])
def test_1d_panel_rejects_multidimensional_image_data(mode):
    block, structure = NODES["camera"]
    card = _card("1d", mode, structure)
    try:
        card.refresh({"sig": block, "shot": 1})
        assert card.plotter is None
        assert "multidimensional data_shape" in card.status.text()
    finally:
        card.shutdown()


@pytest.mark.parametrize("node,expected_shape", [
    ("camera", (H, W)),        # the data IS the image
    ("scan_2d", (N0, N1)),     # the flattened points reshape to the grid (grid_shape)
])
def test_2d_panel_imshow_shape(node, expected_shape):
    blk, st = NODES[node]
    c = _card("2d", "average", st)
    try:
        out = np.asarray(c._coerce(reduce_repeat(blk, "average")))
        assert out.shape == expected_shape
    finally:
        c.shutdown()


@pytest.mark.parametrize("node", ["scan_d1", "scan_d3"])
def test_2d_panel_rejects_a_curve(node):
    """A 1-D-data / 1-D-points value has no map -> a 2D panel RAISES (never a silently-wrong image)."""
    blk, st = NODES[node]
    c = _card("2d", "average", st)
    try:
        with pytest.raises(ValueError):
            c._coerce(reduce_repeat(blk, "average"))
    finally:
        c.shutdown()


def test_custom_expression_falls_back_to_shape_inference():
    """structure is authoritative ONLY for the identity source; a custom ``value=`` expression makes
    _bound_structure return None so reshape degrades to shape inference (no crash, no false image)."""
    blk, st = NODES["camera"]
    c = PanelCard(PanelConfig(kind="1d", role="plot", source="value = signal[0]", inputs=["sig"],
                              params={"repeat_mode": "average"}),
                  structure_provider=lambda name: st)
    try:
        assert c._bound_structure() is None      # custom expr -> structure NOT trusted
    finally:
        c.shutdown()


def test_average_reduce_of_partial_scan_emits_no_runtime_warning():
    """A LIVE scan publishes a (R,P,*data_shape) block that is only partially filled -- the
    not-yet-measured point columns are all-NaN.  Averaging the repeat axis of an all-NaN column is
    NaN BY INTENT (a gap in the curve), so reduce_repeat('average') must NOT leak numpy's spurious
    'Mean of empty slice' RuntimeWarning (the temperature live-scan noise).  Result still correct."""
    import warnings
    block = np.full((3, 4, 1), np.nan)            # 3 repeats x 4 points x 1 series, nothing measured...
    block[:, 0, 0] = [1.0, 2.0, 3.0]              # ...except point 0 -> points 1..3 are all-NaN columns
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # any leaked RuntimeWarning -> test failure
        out = np.asarray(reduce_repeat(block, "average"))
    assert out.shape == (4, 1)
    assert abs(float(out[0, 0]) - 2.0) < 1e-9          # measured point = mean over repeats
    assert np.isnan(out[1:, 0]).all()                  # unmeasured points stay NaN gaps
