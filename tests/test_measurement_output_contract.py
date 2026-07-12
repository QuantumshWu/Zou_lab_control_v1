"""Producer contract: every hub signal is ``(R,P,*data_shape)``.

SignalSpec is the only producer declaration and converts directly to the
SignalSchema registered by LogicNode's common publish boundary.  No node-level
shape, primary-signal exception, control sentinel, or ndim inference exists.
"""

import numpy as np
import pytest
from conftest import raw_device_set

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import (
    CameraMeasurement,
    Measurement,
    OccupancyProcessor,
    ScannedMeasurementNode,
    SignalSpec,
)
from Zou_lab_control.neutral_atom.operations.measurement import ScanAxis, ScannedMeasurement
from conftest import fire_live_imaging


def test_signal_spec_is_always_a_complete_schema():
    scalar = SignalSpec("value", "value")
    assert scalar.points_shape == (1,)
    assert scalar.data_shape == (1,)
    schema = scalar.to_schema(dtype=np.float64)
    assert schema.point_shape == (1,) and schema.data_shape == (1,)

    for bad in (
        {"points_shape": ()},
        {"data_shape": ()},
        {"points_shape": None},
        {"data_shape": None},
    ):
        with pytest.raises((TypeError, ValueError)):
            SignalSpec("bad", "bad", **bad)
    with pytest.raises(ValueError):
        SignalSpec("bad", "bad", history=0)


def test_camera_schema_is_repeat_one_point_image():
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        camera = CameraMeasurement(
            hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=4)
        fire_live_imaging(exp)
        for _ in range(6):
            camera.step()

        schema = hub.schema("frame_0")
        frame = hub.latest("frame_0")
        assert schema.point_shape == (1,)
        assert schema.data_shape == raw_device_set(exp).camera.frame_shape
        assert schema.repeat_capacity == 4
        assert frame.shape == (4, 1, *schema.data_shape)
        assert camera.finished and camera.points_done == 4
    finally:
        exp.close()


def test_camera_repeat_zero_is_bounded_live_ring():
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        camera = CameraMeasurement(
            hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=0)
        fire_live_imaging(exp)
        for _ in range(6):
            camera.step()
        assert hub.latest("frame_0").shape == (
            1, 1, *raw_device_set(exp).camera.frame_shape)
        assert hub.schema("frame_0").repeat_capacity == 1
        assert not camera.finished and camera.total_points == 0
    finally:
        exp.close()


class _Reducer:
    data_shape = (2,)
    labels = ("x", "y", "z")


class _Axis:
    label = "x"
    unit = ""
    values = (10.0, 20.0, 30.0, 40.0)


class _StubMeasurement:
    def __init__(self):
        self.axis = _Axis()
        self.reducer = _Reducer()

    def measure(self, value, index):
        return np.asarray([float(index), float(index) * 2.0])


def test_scan_schema_is_repeat_flat_points_data_and_uses_patches():
    hub = SignalHub()
    node = ScannedMeasurementNode(
        hub, _StubMeasurement(), x_key="delay", y_key="survival",
        prefix="m_", repeat=2)
    node.run_to_completion()

    schema = hub.schema("m_survival")
    assert schema.point_shape == (4,)
    assert schema.data_shape == (2,)
    assert schema.repeat_capacity == 2
    assert hub.latest("m_survival").shape == (2, 4, 2)
    assert hub.latest("m_delay").shape == (1, 4, 1)
    assert hub.storage_stats("m_survival")["patch_updates"] == 8
    assert set(hub.registered_names()) == {"m_delay", "m_survival"}


class _TensorReducer:
    data_shape = (2, 3)
    labels = ("x", "tensor", "tensor")

    def __init__(self, rows):
        self._rows = [np.asarray(row, dtype=float) for row in rows]
        self._index = 0

    def reduce(self, frames, calibration):
        del frames, calibration
        row = self._rows[self._index % len(self._rows)]
        self._index += 1
        return row.copy()


class _TensorPulse:
    sequencer = None

    def frame_sequence(self, *args, **kwargs):
        del args, kwargs
        return object()

    def set_time(self, value):
        self.value = float(value)
        return self

    def set_slot(self, name, value):
        self.slot = str(name)
        self.value = float(value)
        return self


class _TensorPlan:
    n_frames = 1

    def sequence_for(self, pulse, axis, value):
        del pulse, axis, value
        return object()


class _TensorCamera:
    def acquire(self, frames=1, *, stop=None, **kwargs):
        del stop, kwargs
        return [np.zeros((2, 2), dtype=float) for _ in range(int(frames))]


def _tensor_measurement(rows, *, shots=1, values=(1.0,)):
    return ScannedMeasurement(
        pulse=_TensorPulse(),
        camera=_TensorCamera(),
        sequencer=None,
        calibration=None,
        axis=ScanAxis(slot="duration", values=values, kind="duration"),
        plan=_TensorPlan(),
        reducer=_TensorReducer(rows),
        shots_per_point=shots,
    )


def test_scanned_measurement_preserves_full_data_shape_across_shots_and_run():
    first = np.arange(6, dtype=float).reshape(2, 3)
    second = first + 12.0
    measurement = _tensor_measurement(
        [first, second], shots=2, values=(1.0, 2.0))

    result = measurement.run(live=False, display=False)

    expected = (first + second) / 2.0
    assert measurement.data_shape == (2, 3)
    assert result.data_y.shape == (2, 2, 3)
    assert result.points_done == 2 and result.finished
    assert result.summary()["data_shape"] == (2, 3)
    np.testing.assert_allclose(result.data_y[0], expected)
    np.testing.assert_allclose(result.data_y[1], expected)


def test_scanned_measurement_rejects_a_reducer_output_that_breaks_its_data_shape():
    measurement = _tensor_measurement([np.arange(6, dtype=float)], shots=1)

    with pytest.raises(ValueError, match=r"returned shape \(6,\).*declared data_shape is \(2, 3\)"):
        measurement.measure(1.0)


def test_scanned_measurement_node_publishes_every_reducer_data_axis():
    first = np.arange(6, dtype=float).reshape(2, 3)
    second = first + 10.0
    measurement = _tensor_measurement(
        [first, second], shots=1, values=(1.0, 2.0))
    hub = SignalHub()
    node = ScannedMeasurementNode(
        hub, measurement, x_key="x", y_key="tensor", repeat=2)

    node.run_to_completion()

    schema = hub.schema("tensor")
    assert schema.data_shape == (2, 3)
    assert hub.latest("tensor").shape == (2, 2, 2, 3)
    np.testing.assert_allclose(hub.latest("tensor")[0, 0], first)
    np.testing.assert_allclose(hub.latest("tensor")[0, 1], second)


def test_common_publish_boundary_rejects_wrong_shape_for_any_output():
    class BadMeasurement(Measurement):
        def __init__(self, hub):
            super().__init__(hub)
            self.done = False

        def _bare_published_signals(self):
            return frozenset({"y"})

        def _bare_output_specs(self):
            return (SignalSpec(
                "y", "y", points_shape=(3,), data_shape=(1,),
                dtype=np.float64, repeat_capacity=2),)

        def shot(self):
            self.done = True
            return {"y": np.zeros((2, 3))}  # missing trailing data_shape=(1,)

    with pytest.raises(ValueError, match="schema requires physical"):
        BadMeasurement(SignalHub()).step()


def test_occupancy_declares_every_output_without_static_exceptions():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        exp.readout.sitemap(frames=4, display=False)
        exp.readout.thresholds(frames=20, display=False)
        hub = SignalHub()
        camera = CameraMeasurement(
            hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer)
        occupancy = OccupancyProcessor(
            hub,
            calibration=exp.readout.current,
            source_expr={"inputs": ["frame_0"], "source": "value = signal"},
            method="box",
            grid_shape=(3, 4),
        )
        fire_live_imaging(exp)
        camera.step()
        occupancy.step()

        schemas = {name: hub.schema(name) for name in occupancy.published_signals()}
        for schema in schemas.values():
            assert schema.point_shape and schema.data_shape
        assert schemas["occupied"].point_shape == (1,)
        assert schemas["occupied"].data_shape == (12,)
        assert schemas["rate"].data_shape == (1,)
        assert schemas["centers"].point_shape == (1,)
        assert schemas["centers"].data_shape == (12, 2)
        assert hub.latest("centers").shape == (1, 1, 12, 2)
    finally:
        exp.close()
