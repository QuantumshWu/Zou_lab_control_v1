"""Current flat autonomous pulse-to-camera capture contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_data.axis import REPEAT, SCAN_POINT, AxisId, AxisSpec
from zlc_data.schema import PointColumn, PointTable
from zlc_data.value import BlockId
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_CAPTURE,
    CAPABILITY_PULSE_EXECUTE,
)
from zlc_neutral_atom.installation_config import installation_template
from zlc_neutral_atom.installation_runtime import create_installation
from zlc_neutral_atom.runtime.cleanup import CleanupReport, run_cleanup_steps
from zlc_neutral_atom.capture.coordination import (
    execute_autonomous_single_fire,
    validate_single_trigger_capture_binding,
)
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_pulse import (
    PulseExecutionForm,
    PulseFieldRef,
    ScanParameter,
    freeze_scan_table,
    load_pulse_document,
)


ROOT = Path(__file__).parents[1]


def _contract_with_required_trigger_interval(contract, seconds: float):
    """Re-mint the current evidence chain with one changed timing fact."""

    evidence = contract.capability.camera_capability_evidence
    evidence = replace(
        evidence,
        physical_facts=replace(
            evidence.physical_facts,
            required_external_trigger_interval_seconds=seconds,
        ),
    )
    capability = replace(
        contract.capability,
        camera_capability_evidence=evidence,
    )
    return replace(
        contract,
        capability=capability,
    )


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def test_ordered_cleanup_runs_later_physical_steps_after_earlier_exception():
    calls = []
    first = RuntimeError("sequencer cleanup failed")
    second = RuntimeError("camera cleanup reported an error")

    def pulse_cleanup():
        calls.append("pulse")
        raise first

    def camera_cleanup():
        calls.append("camera")
        return CleanupReport(errors=(second,))

    report = run_cleanup_steps(pulse_cleanup, camera_cleanup)
    assert calls == ["pulse", "camera"]
    assert report.errors == (first, second)


def test_single_fire_failure_poisons_both_finite_owners():
    calls = []
    primary = RuntimeError("pulse prepare failed")

    class Pulse:
        def prepare(self, _context):
            calls.append("prepare")
            raise primary

        def fail(self):
            calls.append("pulse-fail")

    class Capture:
        def fail(self, error):
            assert error is primary
            calls.append("capture-fail")
            raise RuntimeError("capture poison also failed")

    with pytest.raises(RuntimeError) as failure:
        execute_autonomous_single_fire(
            object(),
            pulse=Pulse(),
            capture=Capture(),
        )
    assert failure.value is primary
    assert calls == ["prepare", "capture-fail", "pulse-fail"]
    assert any("capture poison also failed" in note for note in primary.__notes__)


def test_trigger_interval_gate_is_exact_for_single_and_cross_point_edges():
    installation = create_installation(
        installation_template("virtual", seed=7)
    )
    runtime = installation.runtime
    try:
        catalog = runtime.device_catalog
        camera_port = runtime.require_capability(
            catalog.require("camera").ref,
            CAPABILITY_CAMERA_CAPTURE,
        )
        pulse_port = runtime.require_capability(
            catalog.require("sequencer").ref,
            CAPABILITY_PULSE_EXECUTE,
        )
        document = load_pulse_document(
            ROOT / "pulses" / "imaging_template.json"
        )
        repeat_axis = _axis("capture.repeat", REPEAT, 1)

        static = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=document,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                repeat_axis,
                AxisId("capture.event"),
                AxisId("capture.ordinal"),
                readout_events_per_repeat=3,
            ),
            camera_instance_id="camera",
        )
        schedule = static.compiled_artifact.trigger_schedules[0]
        assert schedule.minimum_interval_ticks is not None
        actual_interval = (
            schedule.minimum_interval_ticks
            / static.compiled_artifact.target_ir.clock_hz
        )
        pulse_binding = PulseCaptureBinding(
            static.compiled_artifact,
            static.trigger_channel,
            static.cell_plan,
        )
        accepted = validate_single_trigger_capture_binding(
            capture_spec=static.capture.capture_spec,
            contract=_contract_with_required_trigger_interval(
                static.capture.capture_contract,
                actual_interval,
            ),
            pulse_binding=pulse_binding,
        )
        assert accepted is schedule
        with pytest.raises(
            ValueError,
            match="shorter than the broker-attested required external trigger interval",
        ):
            validate_single_trigger_capture_binding(
                capture_spec=static.capture.capture_spec,
                contract=_contract_with_required_trigger_interval(
                    static.capture.capture_contract,
                    float(np.nextafter(actual_interval, np.inf)),
                ),
                pulse_binding=pulse_binding,
            )

        trigger_index = document.target.raw_lanes.index("ch11")
        periods = []
        for index, period in enumerate(document.periods):
            states = list(period.states)
            if index > 1:
                states[trigger_index] = 0
            periods.append(replace(period, states=tuple(states)))
        single_edge = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=replace(document, periods=tuple(periods)),
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                repeat_axis,
                AxisId("single.event"),
                AxisId("single.ordinal"),
                readout_events_per_repeat=1,
            ),
            camera_instance_id="camera",
        )
        single_schedule = single_edge.compiled_artifact.trigger_schedules[0]
        assert single_schedule.total == 1
        assert single_schedule.minimum_interval_ticks is None
        assert validate_single_trigger_capture_binding(
            capture_spec=single_edge.capture.capture_spec,
            contract=_contract_with_required_trigger_interval(
                single_edge.capture.capture_contract,
                1_000.0,
            ),
            pulse_binding=PulseCaptureBinding(
                single_edge.compiled_artifact,
                single_edge.trigger_channel,
                single_edge.cell_plan,
            ),
        ) is single_schedule

        scan_document = replace(
            document,
            periods=document.periods[:3],
            api_parameters=(),
        )
        first_period = scan_document.periods[0]
        parameter = ScanParameter(
            "capture_scan",
            PulseFieldRef("duration", first_period.period_id),
            "capture scan",
            first_period.unit,
        )
        scan_document = replace(scan_document, scan_parameters=(parameter,))
        table, _report = freeze_scan_table(
            scan_document,
            (parameter.parameter_id,),
            ((0.030,), (0.021,), (0.040,)),
        )
        scan_document = replace(scan_document, scan_table=table)
        scan_axis = _axis("capture.scan", SCAN_POINT, 3)
        scan_layout = TriggeredCameraLayout(
            repeat_axis,
            AxisId("scan.event"),
            scan_point_table=PointTable(
                scan_axis.size,
                (
                    PointColumn(
                        scan_axis.axis_id,
                        scan_axis.name,
                        scan_axis.role,
                        PointColumn.NUMERIC,
                        scan_axis.coordinates,
                    ),
                ),
            ),
            readout_events_per_repeat=1,
        )
        with pytest.raises(ValueError, match="requires a finite pulse form"):
            bind_triggered_camera_acquisition(
                pulse_port,
                camera_port,
                pulse_document=scan_document,
                execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
                trigger_channel="ch11",
                layout=scan_layout,
                camera_instance_id="camera",
            )
        scanned = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=scan_document,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            trigger_channel="ch11",
            layout=scan_layout,
            camera_instance_id="camera",
        )
        scan_schedule = scanned.compiled_artifact.trigger_schedules[0]
        edges = tuple(scan_schedule.iter_edges())
        assert tuple(edge.point_index for edge in edges) == (0, 1, 2)
        global_intervals = tuple(
            right.tick_from_run_start - left.tick_from_run_start
            for left, right in zip(edges, edges[1:])
        )
        assert scan_schedule.minimum_interval_ticks == min(global_intervals)
        required = float(
            np.nextafter(
                min(global_intervals)
                / scanned.compiled_artifact.target_ir.clock_hz,
                np.inf,
            )
        )
        with pytest.raises(
            ValueError,
            match="shorter than the broker-attested required external trigger interval",
        ):
            validate_single_trigger_capture_binding(
                capture_spec=scanned.capture.capture_spec,
                contract=_contract_with_required_trigger_interval(
                    scanned.capture.capture_contract,
                    required,
                ),
                pulse_binding=PulseCaptureBinding(
                    scanned.compiled_artifact,
                    scanned.trigger_channel,
                    scanned.cell_plan,
                ),
            )
    finally:
        assert runtime.shutdown(timeout=2.0)
