"""Contract (#H3n): EVERY acquiring measurement publishes its primary data block with the UNIFORM
shape ``(repeat, *points_shape, *data_shape)`` -- ``repeat`` is the repeat-axis depth, ``points_shape``
is the swept parameter space (a camera = ``(1,)`` since one frame sweeps no input parameter; a 1-D
scan = ``(n_points,)``), and ``data_shape`` is the per-point data (a camera frame's ``(H, W)`` image;
a scan's ``(dim,)`` scalar/vector).  The plot relies on this to auto-reshape ANY measurement (camera
frame OR 2-D scan) into the right view -- so the contract is enforced here MECHANICALLY, not by prose.

This is the user's "measurement 底层约定": the camera frame is ``repeat x (data points) x (H*W)`` and
a 2-D scan is ``repeat x (param1*param2) x (data)`` -- the SAME 3-conceptual-axis shape.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import (
    CameraMeasurement, ScannedMeasurementNode, LogicNode)
import Zou_lab_control.neutral_atom as na
from conftest import fire_live_imaging


def _block_matches_contract(node, block) -> bool:
    """The published block's shape is EXACTLY ``(repeat, *points_shape, *data_shape)``."""
    block = np.asarray(block)
    expected = (int(node.repeat), *tuple(node.points_shape), *tuple(node.data_shape))
    return block.shape == expected


def test_base_class_declares_the_contract_attributes():
    """The contract lives on the base ``LogicNode`` (defaults empty), so every node carries the
    declaration and the plot can read ``points_shape`` / ``data_shape`` uniformly."""
    assert LogicNode.points_shape == () and LogicNode.data_shape == ()


def test_camera_frame_is_repeat_x_one_point_x_image():
    """Camera ``frame`` = ``(repeat, 1, H, W)``: repeat x ONE data point (a frame does not sweep an
    input parameter) x the H*W image DATA."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                repeat=4, free_run=True)
        fire_live_imaging(exp)
        for _ in range(6):
            cam.step()
        block = hub.latest("frame")
        assert cam.points_shape == (1,)                         # one frame = one point
        assert len(cam.data_shape) == 2                         # the H*W image is the DATA
        assert cam.primary_signal == "frame"                    # declares its contract block (#H3r-F4)
        assert _block_matches_contract(cam, block)              # (repeat, 1, H, W)
        # repeat is the user's number -- ``free_run`` does NOT discard it (depth = 4, not a constant)
        assert np.asarray(block).shape[0] == 4
    finally:
        exp.close()


def test_camera_repeat_is_finite_when_not_free_running():
    """``repeat=N`` + ``free_run=False`` takes EXACTLY N photos then finishes (not forever)."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                repeat=3, free_run=False)
        fire_live_imaging(exp)
        for _ in range(8):
            cam.step()
        assert cam.points_done == 3 and cam.finished            # N photos then stop
        assert np.asarray(hub.latest("frame")).shape[0] == 3
    finally:
        exp.close()


class _Reducer:
    n_series = 2
    labels = ("x", "y", "z")


class _Axis:
    label = "x"; unit = ""; values = (10.0, 20.0, 30.0, 40.0)


class _StubMeasurement:
    def __init__(self):
        self.axis = _Axis(); self.reducer = _Reducer()

    def measure(self, value, index):
        return [float(index), float(index) * 2.0]              # n_series = 2 -> data_shape (2,)


def test_scan_block_is_repeat_x_points_x_data():
    """A 1-D scan ``(repeat, n_points, dim)`` obeys the SAME contract: points_shape = the swept axis,
    data_shape = the reducer's series count."""
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _StubMeasurement(), x_key="x", y_key="y", prefix="m_", repeat=2)
    node.run_to_completion()
    block = hub.latest("m_y")
    assert node.points_shape == (4,) and node.data_shape == (2,)
    assert node.primary_signal == "y"                           # declares its contract block (#H3r-F4)
    assert _block_matches_contract(node, block)                 # (2, 4, 2)


def test_base_assertion_rejects_a_wrong_shaped_primary_block():
    """The base ``Measurement._assert_primary_shape`` is the enforcement: ANY acquiring measurement
    (incl. a future one) that publishes its primary block with the wrong shape -- e.g. forgets the
    repeat axis -- fails LOUD at publish time, not as a silently-wrong plot (#H3r-F4)."""
    from Zou_lab_control.neutral_atom.operations.logic import Measurement

    class _BadShape(Measurement):
        def __init__(self, hub):
            super().__init__(hub)
            self.repeat = 2
            self.points_shape = (3,)
            self.data_shape = (1,)
            self.primary_signal = "y"

        def shot(self):
            return self._assert_primary_shape({"y": np.zeros((3, 1))})   # (3,1) != (2,3,1): no repeat axis

    node = _BadShape(SignalHub())
    with pytest.raises(ValueError, match="output contract"):
        node.shot()


def test_base_assertion_passes_a_correct_block():
    from Zou_lab_control.neutral_atom.operations.logic import Measurement

    class _GoodShape(Measurement):
        def __init__(self, hub):
            super().__init__(hub)
            self.repeat = 2
            self.points_shape = (3,)
            self.data_shape = (1,)
            self.primary_signal = "y"

        def shot(self):
            return self._assert_primary_shape({"y": np.zeros((2, 3, 1))})

    node = _GoodShape(SignalHub())
    assert np.asarray(node.shot()["y"]).shape == (2, 3, 1)
