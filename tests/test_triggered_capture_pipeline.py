"""Current flat autonomous pulse-to-camera capture contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from zlc_data import AxisId, AxisSpec, BlockId, PointLayout, REPEAT, SCAN_POINT
from zlc_neutral_atom.bootstrap._installation import create_virtual_installation
from zlc_neutral_atom.bootstrap._triggered_capture import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.timing._coordination import (
    execute_autonomous_single_fire,
    run_cleanup_steps,
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
    provenance = replace(
        contract.camera_provenance,
        capability_fingerprint=capability.capability_fingerprint,
    )
    return replace(
        contract,
        capability=capability,
        camera_provenance=provenance,
    )


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _run_isolated(script: str, workspace: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(workspace)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "isolated current-composition probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    marker = "RESULT_JSON="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    pytest.fail(f"isolated probe returned no result marker: {completed.stdout}")


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
    runtime = create_virtual_installation(seed=7)
    try:
        catalog = runtime.device_catalog
        camera_port = runtime.camera_port(catalog.require("camera").ref)
        pulse_port = runtime.pulse_port(catalog.require("sequencer").ref)
        document = load_pulse_document(
            ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"
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
            capture_spec=static.measurement.capture_spec,
            contract=_contract_with_required_trigger_interval(
                static.measurement.capture_contract,
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
                capture_spec=static.measurement.capture_spec,
                contract=_contract_with_required_trigger_interval(
                    static.measurement.capture_contract,
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
        )
        single_schedule = single_edge.compiled_artifact.trigger_schedules[0]
        assert single_schedule.total == 1
        assert single_schedule.minimum_interval_ticks is None
        assert validate_single_trigger_capture_binding(
            capture_spec=single_edge.measurement.capture_spec,
            contract=_contract_with_required_trigger_interval(
                single_edge.measurement.capture_contract,
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
            scan_axes=(scan_axis,),
            scan_point_layout=PointLayout.rect_c((3,)),
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
            )
        scanned = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=scan_document,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            trigger_channel="ch11",
            layout=scan_layout,
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
                capture_spec=scanned.measurement.capture_spec,
                contract=_contract_with_required_trigger_interval(
                    scanned.measurement.capture_contract,
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


def test_public_current_capture_is_one_autonomous_fire_with_exact_reconciliation(
    tmp_path,
):
    result = _run_isolated(
        """
        import json
        from pathlib import Path
        import sys

        from Zou_lab_control.notebook import connect
        from zlc_pulse import load_pulse_document

        workspace = Path(sys.argv[1])
        document = load_pulse_document(
            Path("zlc_neutral_atom/assets/imaging_template.json")
        )
        experiment = connect("virtual", repository=workspace, seed=7)
        try:
            request = experiment.readout.capture_request(
                document,
                repeat_count=1,
                readout_events_per_repeat=3,
            )
            descriptor = experiment.inspect(request)
            handle = experiment.start(request)
            reference = handle.result(10.0)
            artifact = experiment.readout.load_capture(reference)
            pulse = artifact.pulse_evidence
            assert pulse is not None
            schedule = tuple(artifact.frame_source.iter_cell_schedule())
            result = {
                "expected_frames": descriptor.expected_frames,
                "descriptor_shape": list(descriptor.output_shape),
                "physical_shape": list(artifact.frame_source.schema.physical_shape),
                "data_shape": list(
                    artifact.frame_source.schema.cell_schema.data_shape
                ),
                "cells": [
                    [cell.repeat_index, cell.point_storage_index]
                    for cell in schedule
                ],
                "produced": artifact.terminal.produced_count,
                "drained": artifact.terminal.drained_count,
                "pulse_trigger_count": dict(
                    pulse.terminal.receipt.expected_trigger_counts_from_completed_schedule
                )["ch11"],
                "final_committed": handle.snapshot().final_committed,
                "execution_form": pulse.compiled_artifact.execution_form.value,
            }
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
        finally:
            experiment.close()
        """,
        tmp_path / "current-capture",
    )
    assert result == {
        "cells": [[0, 0], [0, 1], [0, 2]],
        "data_shape": [96, 128],
        "descriptor_shape": [1, 3, 96, 128],
        "drained": 3,
        "execution_form": "STATIC_ONCE",
        "expected_frames": 3,
        "final_committed": True,
        "physical_shape": [1, 3, 96, 128],
        "produced": 3,
        "pulse_trigger_count": 3,
    }


def test_exact_preview_filters_frozen_source_ordinals_before_capacity_one_ingest(
    tmp_path,
):
    from Zou_lab_control.notebook import connect

    class RecordingPreview:
        def __init__(self, spec) -> None:
            self.spec = spec
            self.terminal = False
            self.failure = None
            self.source_ordinals = []
            self.head_sequences = []
            self.missed_events = []
            self.dataset = None

        def bind(self, dataset, *, run_id: str, causation_domain_id: str) -> None:
            assert run_id and causation_domain_id
            self.dataset = dataset

        def updated(self) -> None:
            snapshot = self.dataset.materialize(None)
            self.source_ordinals.append(
                snapshot.cell_metadata[0].source_ordinal
            )
            self.head_sequences.append(snapshot.head.sequence)
            self.missed_events.append(snapshot.coverage.missed_events)

        def fail(self, message: str) -> None:
            self.failure = message
            self.terminal = True
            self.close()

        def source_terminal(self) -> None:
            self.terminal = True

        def close(self) -> None:
            dataset, self.dataset = self.dataset, None
            if dataset is not None:
                dataset.close()

    experiment = connect(
        "virtual",
        repository=tmp_path / "preview-selection",
        seed=7,
    )
    ports = []
    try:
        sequence = experiment.readout.sitemap_request(frames=2)
        grouping = sequence.capture_request.within_point_grouping
        assert grouping is not None
        reference_event = sequence.analysis.layout.reference_event_indices[0]
        selected = tuple(
            source_ordinal
            for source_ordinal, (_repeat, event) in enumerate(grouping)
            if event == reference_event
        )
        assert selected == (0, 3)

        def run_preview(source_ordinals, suffix):
            prepared = experiment.readout.prepare_capture(sequence.capture_request)

            def factory(spec):
                port = RecordingPreview(spec)
                ports.append(port)
                return port

            handle = prepared.start_with_preview(
                factory=factory,
                source_ordinals=source_ordinals,
            )
            return ports[-1], handle.result(10.0)

        selected_port, selected_ref = run_preview(selected, "selected")
        assert selected_port.source_ordinals == [0, 3], selected_port.failure
        assert selected_port.head_sequences == [0, 3]
        assert selected_port.missed_events == [0, 0]
        assert selected_port.failure is None and selected_port.terminal
        assert tuple(
            sample.metadata.source_ordinal
            for _cell, sample in experiment.readout.load_capture(
                selected_ref
            ).frame_source.iter_event_order()
        ) == tuple(range(6))

        all_port, _all_ref = run_preview(None, "all")
        assert all_port.source_ordinals == list(range(6))
        assert all_port.head_sequences == list(range(6))
        assert all_port.missed_events == [0] * 6
        assert all_port.failure is None and all_port.terminal

        rejected = experiment.readout.prepare_capture(sequence.capture_request)

        def rejected_factory(spec):
            port = RecordingPreview(spec)
            ports.append(port)
            return port

        with pytest.raises(ValueError, match="frozen cell schedule"):
            rejected.start_with_preview(
                factory=rejected_factory,
                source_ordinals=(6,),
            )
        assert ports[-1].terminal
        assert "frozen cell schedule" in ports[-1].failure
    finally:
        for port in ports:
            port.close()
        experiment.close()


def test_host_stepped_scan_is_not_reintroduced_as_a_capture_mode():
    import zlc_neutral_atom.timing.capture as capture_module

    assert not hasattr(capture_module, "HOST_STEPPED_GROUP")
    assert not hasattr(capture_module, "HostSteppedCaptureSpec")
