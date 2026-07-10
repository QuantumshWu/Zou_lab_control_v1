"""PulseScan has two explicit execution strategies and one signal contract."""

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.ports import PortCatalog


def _experiment():
    import Zou_lab_control.neutral_atom as na

    return na.connect("virtual", sitemap={"grid_shape": (2, 2), "image_shape": (24, 24)})


def _template(path: Path, *, scan_slot: bool = False, api_slot: bool = False,
              persisted_program: bool = False) -> str:
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod

    state = PulseTableState(
        port_catalog=PortCatalog.from_channels(["trap", "probe", "emCCD"]),
        periods=[
            PulsePeriod(0.002, (1, 0, 0), unit="s", name="load"),
            PulsePeriod(0.005, (1, 1, 1), unit="s", name="image"),
        ],
    )
    if scan_slot:
        state.bind_field("duration", "1", label="probe_time", unit="us")
        if persisted_program:
            state.scan_code = "scan_table = [[10000.0], [20000.0], [30000.0]]"
            state.set_scan_table([[10000.0], [20000.0], [30000.0]])
    if api_slot:
        state.bind_api_field("duration", "1", name="a1", unit="us")
    state.save(path)
    return str(path)


def _spec(exp):
    return {spec.name: spec for spec in exp.readout.measurement_specs()}["Pulse scan"]


def test_scan_slot_plan_attaches_snapped_table_and_semantic_coordinates(tmp_path):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    exp = _experiment()
    try:
        plan = _spec(exp).build(
            template=_template(tmp_path / "scan.json", scan_slot=True),
            pulse_slots={
                "api": {},
                "sweep_kind": SWEEP_SCAN_SLOT,
                "program": "scan_table = [[10001.2], [20003.7], [30005.1]]",
            },
            y={"inputs": ["rate"], "source": "value = signal"},
        )
        assert plan.sweep_kind == SWEEP_SCAN_SLOT
        assert plan.scan_names == [plan.pulse_state.scan_slots[0].name]
        assert plan.scan_names[0] not in {"s0", "a1"}
        assert plan.api_handles == []
        assert len(plan.pulse_state.scan_table) == 3
        np.testing.assert_allclose(
            plan.scan_arrays[0], np.asarray(plan.pulse_state.scan_table)[:, 0])
        assert not hasattr(plan, "camera")
    finally:
        exp.close()


def test_api_slot_plan_uses_semantic_coordinate_but_private_api_handle(tmp_path):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_API_SLOT

    exp = _experiment()
    try:
        plan = _spec(exp).build(
            template=_template(tmp_path / "api.json", api_slot=True),
            pulse_slots={
                "api": {"a1": 7.5},
                "sweep_kind": SWEEP_API_SLOT,
                "program": "scan_table = [[5.0], [15.0], [25.0]]",
            },
        )
        assert plan.sweep_kind == SWEEP_API_SLOT
        assert plan.api_handles == ["a1"]
        assert plan.scan_names == ["image_duration"]
        np.testing.assert_allclose(plan.scan_arrays[0], [5.0, 15.0, 25.0])
        assert plan.pulse_state.scan_table == []
    finally:
        exp.close()


def test_template_selects_its_available_strategy_and_persisted_hardware_program(tmp_path):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_API_SLOT, SWEEP_SCAN_SLOT

    exp = _experiment()
    try:
        hardware = _spec(exp).build(
            template=_template(tmp_path / "persisted.json", scan_slot=True,
                               persisted_program=True),
            pulse_slots={"api": {}},
        )
        assert hardware.sweep_kind == SWEEP_SCAN_SLOT
        np.testing.assert_allclose(hardware.scan_arrays[0], [10000.0, 20000.0, 30000.0])

        api = _spec(exp).build(
            template=_template(tmp_path / "api_default.json", api_slot=True),
            pulse_slots={"api": {}},
        )
        assert api.sweep_kind == SWEEP_API_SLOT
        assert api.scan_arrays[0].size > 1
    finally:
        exp.close()


def test_invalid_or_unavailable_sweep_contract_is_rejected(tmp_path):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    exp = _experiment()
    try:
        with pytest.raises(ValueError, match="unsupported pulse-scan field"):
            _spec(exp).build(
                template=_template(tmp_path / "unknown.json", scan_slot=True),
                pulse_slots={"api": {}, "unexpected": 1},
            )
        with pytest.raises(ValueError, match="no scan slot"):
            _spec(exp).build(
                template=_template(tmp_path / "api_only.json", api_slot=True),
                pulse_slots={
                    "api": {},
                    "sweep_kind": SWEEP_SCAN_SLOT,
                    "program": "scan_table = [[1.0], [2.0]]",
                },
            )
    finally:
        exp.close()


class _Sequencer:
    def __init__(self):
        self.prepared = []
        self.fires = 0
        self.stops = 0
        self.waits = 0
        self._fired = threading.Condition()

    def set_safe_state(self):
        self.stops += 1

    def prepare(self, state):
        self.prepared.append(state)
        return SimpleNamespace(duration=0.001)

    def fire(self):
        with self._fired:
            self.fires += 1
            self._fired.notify_all()

    def wait_for_fires(self, count, timeout=2.0):
        with self._fired:
            return self._fired.wait_for(lambda: self.fires >= count, timeout=timeout)

    def wait_done(self, timeout=None, stop=None):
        self.waits += 1
        return True


def _run_node(exp, plan, values):
    from Zou_lab_control.neutral_atom.core.signals import SignalHub, SignalSchema
    from Zou_lab_control.neutral_atom.operations.logic import PulseScanNode
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    sequencer = _Sequencer()
    plan.sequencer = sequencer
    hub = SignalHub()
    hub.register_signal("rate", SignalSchema(dtype=np.float64, repeat_capacity=1), history=2)
    lineages = []
    producer_errors = []

    def publish_y():
        try:
            if plan.sweep_kind == SWEEP_SCAN_SLOT \
                    and not sequencer.wait_for_fires(1):
                raise TimeoutError("hardware scan never fired")
            for fire_count, value in enumerate(values, start=1):
                if plan.sweep_kind != SWEEP_SCAN_SLOT \
                        and not sequencer.wait_for_fires(fire_count):
                    raise TimeoutError(f"API point {fire_count} never fired")
                lineage = hub.next_source_shot()
                lineages.append(lineage)
                hub.publish({"rate": value}, provenance=lineage)
        except BaseException as exc:  # surfaced on the test thread after join
            producer_errors.append(exc)

    spec = _spec(exp)
    node = PulseScanNode(
        hub, plan, x_key=spec.x_key, y_key=spec.y_key, prefix="pulse_scan_", repeat=1)
    producer = threading.Thread(target=publish_y, name="test-y-producer", daemon=True)
    producer.start()
    try:
        node.run_to_completion()
    finally:
        producer.join(timeout=2.0)
    assert not producer.is_alive(), "independent y producer did not stop"
    assert producer_errors == []
    np.testing.assert_allclose(
        np.asarray(hub.latest(node.y_signal)).reshape(-1), np.asarray(values, dtype=float))
    assert hub.latest_provenance(node.y_signal) == lineages[-1]
    assert not any(name.startswith("frame") for name in node.published_signals())
    assert set(node._devices) == {"sequencer"}
    return sequencer


def test_scan_slot_node_uploads_and_fires_the_complete_table_once(tmp_path):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    exp = _experiment()
    try:
        plan = _spec(exp).build(
            template=_template(tmp_path / "hardware_node.json", scan_slot=True),
            pulse_slots={
                "api": {},
                "sweep_kind": SWEEP_SCAN_SLOT,
                "program": "scan_table = [[10000.0], [20000.0], [30000.0]]",
            },
            y={"inputs": ["rate"], "source": "value = signal"},
        )
        sequencer = _run_node(exp, plan, [0.1, 0.2, 0.3])
        assert sequencer.fires == 1 and len(sequencer.prepared) == 1
        assert sequencer.waits == 1
        submitted = sequencer.prepared[0]
        assert submitted.scan_slots and submitted.scan_table == plan.pulse_state.scan_table
        assert submitted.scan_repeats == 1
    finally:
        exp.close()


def test_finite_single_row_scan_slot_is_rejected_before_hardware_fire(tmp_path):
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import PulseScanNode
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    exp = _experiment()
    try:
        plan = _spec(exp).build(
            template=_template(tmp_path / "single_row.json", scan_slot=True),
            pulse_slots={
                "api": {},
                "sweep_kind": SWEEP_SCAN_SLOT,
                "program": "scan_table = [[10000.0]]",
            },
            y={"inputs": ["rate"], "source": "value = signal"},
        )
        with pytest.raises(ValueError, match="at least two table rows"):
            PulseScanNode(SignalHub(), plan, prefix="pulse_scan_", repeat=1)
    finally:
        exp.close()


def test_api_slot_node_submits_one_finite_resolved_pulse_per_point(tmp_path):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_API_SLOT

    exp = _experiment()
    try:
        plan = _spec(exp).build(
            template=_template(tmp_path / "api_node.json", api_slot=True),
            pulse_slots={
                "api": {"a1": 9.0},
                "sweep_kind": SWEEP_API_SLOT,
                "program": "scan_table = [[5.0], [15.0], [25.0]]",
            },
            y={"inputs": ["rate"], "source": "value = signal"},
        )
        sequencer = _run_node(exp, plan, [0.4, 0.5, 0.6])
        assert sequencer.fires == 3 and len(sequencer.prepared) == 3
        assert sequencer.waits == 3
        resolved = [state._read_api_field(state.api_slots[0])
                    for state in sequencer.prepared]
        np.testing.assert_allclose(resolved, [5.0, 15.0, 25.0])
        assert all(not state.scan_slots and not state.scan_table
                   for state in sequencer.prepared)
        assert all(not state.repeat_forever for state in sequencer.prepared)
    finally:
        exp.close()
