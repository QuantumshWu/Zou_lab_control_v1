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
    """The published block's shape is EXACTLY ``(ring, *points_shape, *data_shape)`` where the ring
    depth = ``max(1, repeat)`` (finite K keeps K slices; ``repeat=0`` = ∞ keeps a 1-deep rolling ring)."""
    block = np.asarray(block)
    ring = int(getattr(node, "_ring", max(1, int(node.repeat))))
    expected = (ring, *tuple(node.points_shape), *tuple(node.data_shape))
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
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=4)
        fire_live_imaging(exp)
        for _ in range(6):
            cam.step()
        block = hub.latest("frame_0")
        assert cam.points_shape == (1,)                         # one frame = one point
        assert len(cam.data_shape) == 2                         # the H*W image is the DATA
        assert cam.primary_signal == "frame_0"                  # declares its contract block (#H3r-F4)
        assert _block_matches_contract(cam, block)              # (repeat, 1, H, W)
        # repeat is the user's number -- the kept block depth IS repeat (4, not a constant)
        assert np.asarray(block).shape[0] == 4
    finally:
        exp.close()


def test_camera_repeat_K_takes_exactly_K_photos_then_finishes():
    """``repeat=N`` (N>0) takes EXACTLY N photos then finishes (not forever)."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=3)
        fire_live_imaging(exp)
        for _ in range(8):
            cam.step()
        assert cam.points_done == 3 and cam.finished            # N photos then stop
        assert np.asarray(hub.latest("frame_0")).shape[0] == 3
    finally:
        exp.close()


def test_camera_repeat_zero_is_infinite_rolling_1_deep():
    """``repeat=0`` = ∞ (the ONLY infinite knob, no free-run toggle): a 1-deep rolling ring that
    never finishes -- the live monitor showing the latest frame."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=0)
        fire_live_imaging(exp)
        for _ in range(6):
            cam.step()
        block = np.asarray(hub.latest("frame_0"))
        assert not cam.finished and cam.total_points == 0       # rolls forever
        assert _block_matches_contract(cam, block)              # (1, 1, H, W)
        assert block.shape[0] == 1                              # a 1-deep rolling ring (the latest frame)
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


def test_scan_done_spec_is_single_sourced_across_both_swept_twins():
    """#single-source: the finite-scan ``scan_done`` sentinel (key + label + meaning) AND the swept-block
    publish dict are declared ONCE on ``_SweptBlockMeasurement`` -- NEITHER scan driver
    (``ScannedMeasurementNode``, ``PulseScanNode``) may redefine them.  This is the MECHANICAL guard
    against the drift that HAD occurred: the two twins published ``scan_done`` with different SignalSpec
    labels ("scan complete" vs "scan done") because each twin spelled the spec/publish dict itself."""
    from Zou_lab_control.neutral_atom.operations.logic import (
        _SweptBlockMeasurement, PulseScanNode)
    for twin in (ScannedMeasurementNode, PulseScanNode):
        # inherited unchanged (same function object) -> the spec label + publish dict cannot fork per twin
        assert twin._scan_done_spec is _SweptBlockMeasurement._scan_done_spec
        assert twin._swept_publish is _SweptBlockMeasurement._swept_publish
    # the ONE spec carries the sentinel key both twins publish under ``prefix``; it is set on completion
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _StubMeasurement(), x_key="x", y_key="y", prefix="m_", repeat=1)
    assert _SweptBlockMeasurement.SCAN_DONE in node._bare_published_signals()
    spec = {s.name: s for s in node.output_specs()}["m_" + _SweptBlockMeasurement.SCAN_DONE]
    assert spec.label == "scan complete"                    # the single-sourced label (not "scan done")
    node.run_to_completion()
    assert float(hub.latest("m_scan_done")) == 1.0          # published under the shared key on completion


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


def test_iron_law_signalspec_rejects_a_dropped_data_points_axis():
    """#iron-law: a repeat-carrying signal is ALWAYS ``(repeat, data_points, *data_dim)`` -- the
    data_points axis is mandatory.  ``SignalSpec`` therefore forbids the empty ``()`` (which would drop
    it): points_shape/data_shape are EITHER None (no repeat contract: static/scalar) OR a non-empty
    tuple.  Mechanically enforced so the 'drop the vestigial 1' drift can never come back."""
    from Zou_lab_control.neutral_atom.operations.logic import SignalSpec
    # legal: no contract (static), or a full (data_points>=1, data) slot
    assert SignalSpec("centers", "c").core_ndim is None                       # None/None -> no repeat contract
    assert SignalSpec("occupied", "o", points_shape=(1,), data_shape=(35,)).core_ndim == 2   # no-scan = 1 point
    assert SignalSpec("y", "y", points_shape=(8,), data_shape=(3,)).core_ndim == 2           # a scan
    assert SignalSpec("frame_0", "image", history=8).history == 8             # producer-declared hub storage
    # ILLEGAL: the empty () drops the mandatory data_points / data axis
    for bad in ({"points_shape": ()}, {"data_shape": ()}, {"points_shape": (), "data_shape": (5,)}):
        with pytest.raises(ValueError):
            SignalSpec("bad", "b", **bad)
    with pytest.raises(ValueError):
        SignalSpec("bad_history", "b", history=0)


def test_iron_law_primary_block_must_carry_data_points():
    """#iron-law: a measurement that publishes a primary block must declare a NON-EMPTY points_shape --
    ``_assert_primary_shape`` raises LOUD if a node forgets it (publishes a (repeat, *data) block that
    drops data_points).  A no-scan acquisition sets points_shape=(1,)."""
    from Zou_lab_control.neutral_atom.operations.logic import Measurement
    class _NoPoints(Measurement):
        primary_signal = "frame_0"
        points_shape = ()                  # the forbidden empty -> drops data_points
        data_shape = (4, 5)
        _ring = 2
    with pytest.raises(ValueError, match="data_points"):
        _NoPoints.__new__(_NoPoints)._assert_primary_shape({"frame_0": np.zeros((2, 4, 5))})


def test_iron_law_occupancy_outputs_carry_data_points():
    """Every repeat-carrying occupancy output declares the data_points axis (points_shape non-empty),
    so a frame keeps ONE shape raw or judged and occupancy is (repeat, 1, n_sites) -- not the dropped
    (repeat, n_sites) the historical patch produced."""
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.readout.sitemap(frames=4, display=False); exp.readout.thresholds(frames=20, display=False)
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame_0"], "source": "value = signal"}, method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)
    cam.step(); det.step()
    specs = {s.name: s for s in det.output_specs()}
    for repeat_signal in ("occupied", "counts", "frame_judged"):
        ps = specs[repeat_signal].points_shape
        assert ps and tuple(ps) == (1,), f"{repeat_signal} dropped its data_points axis: points_shape={ps}"
    assert not specs["rate"].has_structure and not specs["centers"].has_structure   # derived/static: exempt
    exp.close()
