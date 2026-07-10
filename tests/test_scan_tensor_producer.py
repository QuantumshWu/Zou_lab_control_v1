from types import SimpleNamespace
import threading

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub, SignalSchema, SignalTensor
from Zou_lab_control.neutral_atom.operations.logic import (
    PulseScanNode,
    SignalExpr,
    SignalSpec,
    _SweptBlockMeasurement,
)
from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT


class _MatrixScan(_SweptBlockMeasurement):
    def __init__(self, hub):
        super().__init__(hub)
        self.x_key = "delay"
        self.y_key = "image"
        self._rows = [
            np.arange(6, dtype=float).reshape(2, 3),
            np.arange(6, 12, dtype=float).reshape(2, 3),
        ]
        self._init_swept_block(values=[0.1, 0.2], data_shape=(2, 3), repeat=1)

    def shot(self):
        index = self._index
        patch = self._fill_point(index, self._rows[index])
        self._current_source_shot = self.hub.next_source_shot()
        return self._swept_publish(patch)

    def _bare_output_specs(self):
        return (
            SignalSpec(
                "delay", "delay", "s", "scan coordinate",
                points_shape=self._point_shape, data_shape=(1,),
                dtype=np.float64, repeat_capacity=1,
            ),
            SignalSpec(
                "image", "image", "counts", "matrix at each point",
                points_shape=self._point_shape, data_shape=self._data_shape,
                dtype=np.float64, repeat_capacity=1,
            ),
        )


def test_scan_patch_preserves_multidimensional_data_shape():
    hub = SignalHub(history=16)
    node = _MatrixScan(hub)
    node.step()
    node.step()

    assert hub.latest("image").shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(hub.latest("image")[0, 0], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(hub.latest("image")[0, 1], np.arange(6, 12).reshape(2, 3))
    assert hub.schema("image").point_shape == (2,)
    assert hub.schema("image").data_shape == (2, 3)
    assert hub.signal_versions()["delay"] == 1  # immutable coordinate is not republished per point
    stats = hub.storage_stats("image")
    assert stats["patch_updates"] == 2 and stats["full_updates"] == 0


def test_scan_registers_immutable_schema_once_not_per_patch():
    class CountingHub(SignalHub):
        def __init__(self):
            super().__init__()
            self.register_calls = {}

        def register_signal(self, name, schema, **kwargs):
            self.register_calls[name] = self.register_calls.get(name, 0) + 1
            return super().register_signal(name, schema, **kwargs)

    hub = CountingHub()
    node = _MatrixScan(hub)
    node.step()
    node.step()
    assert hub.register_calls == {"delay": 1, "image": 1}


def test_scan_point_rejects_flattened_data_for_matrix_schema():
    node = _MatrixScan(SignalHub())
    with pytest.raises(ValueError, match="declared data_shape"):
        node._fill_point(0, np.arange(6, dtype=float))


def _pulse_plan(*, points=6, y_input="rate", y_source="value = signal", scan_shape=(2, 3),
                sequencer=None):
    return SimpleNamespace(
        pulse_state=object(),
        sweep_kind=SWEEP_SCAN_SLOT,
        api_handles=[],
        scan_names=["da_x", "da_y"],
        scan_arrays=[np.arange(points, dtype=float), np.arange(10, 10 + points, dtype=float)],
        scan_shape=scan_shape,
        sequencer=sequencer if sequencer is not None else object(),
        y_expr=SignalExpr([y_input], y_source),
        y_key="fluorescence",
        axis_label="x detuning",
        axis_unit="MHz",
    )


def _publish_rate_after_start(hub, started):
    """Run the y source as an independent Hub producer after the scan has subscribed."""

    errors = []

    def produce():
        try:
            if not started.wait(timeout=2.0):
                raise TimeoutError("PulseScan did not start")
            hub.publish({"rate": np.asarray([[[1.0]]])}, provenance=1)
        except BaseException as exc:  # surfaced on the test thread after join
            errors.append(exc)

    thread = threading.Thread(target=produce, name="test-rate-producer", daemon=True)
    thread.start()
    return thread, errors


def test_pulse_scan_publishes_semantic_coordinates_and_one_y_schema():
    node = PulseScanNode(SignalHub(), _pulse_plan(), x_key="param", prefix="scan_")
    assert node.x_key == "da_x"
    assert node.published_signals() == {
        "scan_da_x", "scan_da_y", "scan_fluorescence"
    }
    assert "scan_param" not in node.published_signals()
    assert "scan_fluorescence_grid" not in node.published_signals()

    specs = {spec.name: spec for spec in node.output_specs()}
    assert specs["scan_fluorescence"].points_shape == (2, 3)
    assert specs["scan_fluorescence"].data_shape == (1,)
    assert specs["scan_fluorescence"].metadata["coordinate_signals"] == (
        "scan_da_x", "scan_da_y")


def test_pulse_scan_y_requires_exactly_one_valid_scalar_cell():
    node = PulseScanNode(SignalHub(), _pulse_plan())
    schema = SignalSchema(dtype=np.float64, repeat_capacity=2)
    tensor = SignalTensor(
        np.asarray([[[1.0]], [[2.0]]]), schema,
        valid=np.asarray([[True], [True]]),
    )
    node._selected_y_tensors = {"rate": tensor}
    with pytest.raises(ValueError, match="exactly one"):
        node._read_y({"rate": tensor.data})

    one = SignalTensor(
        tensor.data, schema,
        valid=np.asarray([[False], [True]]),
    )
    node._selected_y_tensors = {"rate": one}
    assert node._read_y({"rate": one.data}) == 2.0

    node.y_expr = SignalExpr(["rate"], "value = np.mean(signal)")
    assert node._read_y({"rate": tensor.data}) == 1.5  # explicit scalar reduction is allowed


def test_pulse_scan_reserves_all_finite_updates_before_fire(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    hub = SignalHub(history=8)
    hub.register_signal("rate", SignalSchema(dtype=np.float64, repeat_capacity=1), history=8)
    node = PulseScanNode(
        hub,
        _pulse_plan(points=20, scan_shape=(4, 5)),
        repeat=2,
    )
    fired = []

    def fake_prepare(sequencer, state, *, scan_repeats):
        assert hub.history_limit("rate") == 40  # reservation exists before hardware fire
        fired.append(scan_repeats)
        return state

    monkeypatch.setattr(logic, "prepare_hardware_scan", fake_prepare)
    node._start_execution(0)
    assert fired == [2]
    assert node._y_history_reservation.capacity == 40
    node.stop()
    assert hub.history_limit("rate") == 8


def test_pulse_scan_consumes_twenty_burst_updates_with_default_history_eight(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    hub = SignalHub(history=8)
    hub.register_signal("rate", SignalSchema(dtype=np.float64, repeat_capacity=1), history=8)
    node = PulseScanNode(
        hub,
        _pulse_plan(points=20, scan_shape=(4, 5)),
        repeat=1,
    )
    monkeypatch.setattr(
        logic, "prepare_hardware_scan",
        lambda sequencer, state, *, scan_repeats: state,
    )
    node._start_execution(0)
    for value in range(20):
        hub.publish(
            {"rate": np.asarray(value, dtype=np.float64).reshape(1, 1, 1)},
            provenance=value,
        )

    seen = []
    for _ in range(20):
        fresh, _lineage, snapshot = node._await_y_sample()
        assert fresh
        seen.append(node._read_y(snapshot))
    assert seen == [float(value) for value in range(20)]
    node.stop()


def test_pulse_scan_preflight_rejects_direct_frame_and_reports_reduction_cost(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    hub = SignalHub(history=8)
    image_schema = SignalSchema(
        point_shape=(1,), data_shape=(4, 5), dtype=np.float64, repeat_capacity=1)
    hub.register_signal("frame_0", image_schema, history=8)
    direct = PulseScanNode(
        hub,
        _pulse_plan(points=20, y_input="frame_0", scan_shape=(4, 5)),
    )
    fired = []
    monkeypatch.setattr(
        logic, "prepare_hardware_scan",
        lambda *args, **kwargs: fired.append(True),
    )
    with pytest.raises(ValueError, match="reduce explicitly"):
        direct._start_execution(0)
    assert fired == []

    direct_named = PulseScanNode(
        hub,
        _pulse_plan(
            points=20,
            y_input="frame_0",
            y_source="value = frame_0",
            scan_shape=(4, 5),
        ),
    )
    with pytest.raises(ValueError, match="reduce explicitly"):
        direct_named._start_execution(0)
    assert fired == []

    reduced = PulseScanNode(
        hub,
        _pulse_plan(
            points=20,
            y_input="frame_0",
            y_source="value = np.mean(signal)",
            scan_shape=(4, 5),
        ),
    )
    reduced._start_execution(0)
    assert reduced.y_history_reservation_bytes == 20 * 4 * 5 * 8
    reduced.stop()


@pytest.mark.parametrize(
    "expr, message",
    [
        (SignalExpr([], "value = 1.0"), "at least one explicit y input"),
        (SignalExpr([], 'value = np.mean(latest("frame_0"))'), "cannot use live/history"),
        (SignalExpr(["frame_0"], 'value = np.mean(history("frame_0", 2))'),
         "cannot use live/history"),
    ],
)
def test_pulse_scan_expression_cannot_bypass_cursor_reservation(monkeypatch, expr, message):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    hub = SignalHub(history=8)
    hub.register_signal(
        "frame_0",
        SignalSchema(data_shape=(2, 3), dtype=np.float64, repeat_capacity=1),
        history=8,
    )
    plan = _pulse_plan(points=2, y_input="frame_0", scan_shape=None)
    plan.y_expr = expr
    node = PulseScanNode(hub, plan)
    fired = []
    monkeypatch.setattr(
        logic, "prepare_hardware_scan",
        lambda *args, **kwargs: fired.append(True),
    )

    with pytest.raises(ValueError, match=message):
        node._start_execution(0)
    assert fired == []


def test_pulse_scan_direct_signal_dependency_is_cursor_reserved(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    hub = SignalHub(history=3)
    hub.register_signal("a", SignalSchema(dtype=np.float64, repeat_capacity=1), history=3)
    hub.register_signal("b", SignalSchema(dtype=np.float64, repeat_capacity=1), history=3)
    plan = _pulse_plan(points=7, y_input="a", scan_shape=None)
    plan.y_expr = SignalExpr(["a"], "value = signal + b")
    node = PulseScanNode(hub, plan)
    monkeypatch.setattr(logic, "prepare_hardware_scan", lambda *args, **kwargs: object())

    node._start_execution(0)
    assert set(node._y_cursors) == {"a", "b"}
    assert hub.history_limit("a") == 7
    assert hub.history_limit("b") == 7
    node.stop()


def test_pulse_scan_publish_boundary_failure_is_terminal_and_safes_hardware(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    class Sequencer:
        def __init__(self):
            self.safe_calls = 0

        def set_safe_state(self):
            self.safe_calls += 1

    sequencer = Sequencer()
    hub = SignalHub(history=2)
    hub.register_signal("rate", SignalSchema(dtype=np.float64, repeat_capacity=1), history=2)
    node = PulseScanNode(
        hub,
        _pulse_plan(points=4, scan_shape=None, sequencer=sequencer),
        repeat=1,
    )
    started = threading.Event()

    def prepare(*args, **kwargs):
        started.set()
        return object()

    monkeypatch.setattr(logic, "prepare_hardware_scan", prepare)
    producer, producer_errors = _publish_rate_after_start(hub, started)
    monkeypatch.setattr(
        node, "_register_output_schemas",
        lambda values: (_ for _ in ()).throw(RuntimeError("schema boundary failed")),
    )

    try:
        with pytest.raises(RuntimeError, match="schema boundary failed"):
            node.step()
    finally:
        producer.join(timeout=2.0)
    assert not producer.is_alive()
    assert producer_errors == []

    assert sequencer.safe_calls == 1
    assert node._stop.is_set()
    assert node.consecutive_errors == 1
    assert node.last_error == "RuntimeError: schema boundary failed"
    assert node._y_history_reservation is None
    assert hub.history_limit("rate") == 2


def test_pulse_scan_hub_publish_failure_is_terminal_and_safes_hardware(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    class Sequencer:
        def __init__(self):
            self.safe_calls = 0

        def set_safe_state(self):
            self.safe_calls += 1

    sequencer = Sequencer()
    hub = SignalHub(history=2)
    hub.register_signal("rate", SignalSchema(dtype=np.float64, repeat_capacity=1), history=2)
    node = PulseScanNode(
        hub,
        _pulse_plan(points=4, scan_shape=None, sequencer=sequencer),
        repeat=1,
    )
    started = threading.Event()

    def prepare(*args, **kwargs):
        started.set()
        return object()

    monkeypatch.setattr(logic, "prepare_hardware_scan", prepare)
    original_publish = hub.publish

    def fail_scan_output(values, **kwargs):
        if "fluorescence" in values:
            raise RuntimeError("hub publish failed")
        return original_publish(values, **kwargs)

    monkeypatch.setattr(hub, "publish", fail_scan_output)
    producer, producer_errors = _publish_rate_after_start(hub, started)

    try:
        with pytest.raises(RuntimeError, match="hub publish failed"):
            node.step()
    finally:
        producer.join(timeout=2.0)
    assert not producer.is_alive()
    assert producer_errors == []

    assert sequencer.safe_calls == 1
    assert node._stop.is_set()
    assert node.consecutive_errors == 1
    assert node.last_error == "RuntimeError: hub publish failed"
    assert node._y_history_reservation is None
    assert hub.history_limit("rate") == 2


def test_pulse_scan_rejects_duplicate_or_out_of_order_real_provenance(monkeypatch):
    import Zou_lab_control.neutral_atom.operations.logic as logic

    hub = SignalHub(history=8)
    hub.register_signal("rate", SignalSchema(dtype=np.float64, repeat_capacity=1), history=8)
    node = PulseScanNode(hub, _pulse_plan(points=2, scan_shape=None), repeat=1)
    monkeypatch.setattr(logic, "prepare_hardware_scan", lambda *args, **kwargs: object())
    node._start_execution(0)
    hub.publish({"rate": np.asarray([[[2.0]]])}, provenance=2)
    hub.publish({"rate": np.asarray([[[1.0]]])}, provenance=1)

    first, lineage, _snapshot = node._await_y_sample()
    assert first and lineage == 2
    with pytest.raises(ValueError, match="strictly increasing"):
        node._await_y_sample()
    node.stop()


def test_pulse_scan_scalar_contract_failure_immediately_safes_hardware(monkeypatch):
    class Sequencer:
        def __init__(self):
            self.safe_calls = 0

        def set_safe_state(self):
            self.safe_calls += 1

    sequencer = Sequencer()
    node = PulseScanNode(
        SignalHub(), _pulse_plan(sequencer=sequencer), repeat=1)
    node._run_started = True
    tensor = SignalTensor(
        np.asarray([[[1.0]], [[2.0]]]),
        SignalSchema(dtype=np.float64, repeat_capacity=2),
        valid=np.asarray([[True], [True]]),
    )
    node._selected_y_tensors = {"rate": tensor}
    monkeypatch.setattr(
        node, "_await_y_sample",
        lambda: (True, 1, {"rate": tensor.data}),
    )
    with pytest.raises(ValueError, match="exactly one"):
        node.shot()
    assert sequencer.safe_calls == 1
    assert node._stop.is_set()
