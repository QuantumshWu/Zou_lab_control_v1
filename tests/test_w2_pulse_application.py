from __future__ import annotations

from dataclasses import replace
import time

import pytest

from Zou_lab_control.notebook import PulseRunResult, connect
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.pulse_application import PulseRunRequest, PulseTargetDescriptor
from zlc_neutral_atom.runtime.run import CancelOutcome, RunState
from zlc_pulse import (
    DAC_OFFSET_BINARY,
    FIELD_DURATION,
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    AnalogStep,
    ApiParameter,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseTarget,
    RepeatRegion,
    ScanParameter,
    build_pulse_timeline,
    compile_pulse_artifact,
    resolve_api_parameters,
    save_pulse_document,
)
from zlc_workbench.pulse import PulseEditorSession


def _target() -> PulseTarget:
    return PulseTarget(
        ("ttl_lane", "dac_bit_0", "dac_bit_1", "dac_clock_lane"),
        (
            PulsePortSpec(
                "ttl",
                PORT_DIGITAL,
                ("ttl_lane",),
                "TTL",
                None,
                1,
                "binary",
                0,
                None,
            ),
            PulsePortSpec(
                "dac",
                PORT_DAC,
                ("dac_bit_0", "dac_bit_1"),
                "DAC",
                0,
                2,
                DAC_OFFSET_BINARY,
                2,
                "dac_clock",
            ),
            PulsePortSpec(
                "dac_clock",
                PORT_CLOCK,
                ("dac_clock_lane",),
                "DAC clock",
                None,
                1,
                "binary",
                0,
                None,
            ),
        ),
    )


def _document() -> PulseDocument:
    target = _target()
    return PulseDocument(
        "timeline proof",
        target,
        10.0,
        (
            PulsePeriod(
                "on",
                100,
                "ns",
                "on",
                (1, 0, 0, 0),
                (AnalogStep("dac", "edge", 1),),
            ),
            PulsePeriod(
                "ramp",
                100,
                "ns",
                "ramp",
                (0, 0, 0, 0),
                (AnalogStep("dac", "ramp", -1),),
            ),
        ),
        visible_ports=("ttl", "dac"),
        delays=(OutputDelay("ttl", 20, "ns"),),
        repeat=RepeatRegion("on", "ramp", 2),
    )


def _descriptor(target: PulseTarget) -> PulseTargetDescriptor:
    return PulseTargetDescriptor(
        DeviceRef("installation", "runtime", "sequencer"),
        target,
        100e6,
        0,
        16,
    )


def test_preview_is_exact_for_delayed_digital_repeat_and_dac_ramp():
    document = _document()
    _revision, timeline = PulseEditorSession(
        _descriptor(document.target),
        document,
    ).preview()

    assert timeline.logical_duration_ticks == 40
    assert timeline.duration_ticks == 40
    assert timeline.reference_label == "compiled static pulse"
    digital, dac = timeline.rows
    assert [
        (item.start_tick, item.stop_tick, item.start_value, item.stop_value)
        for item in digital.segments
    ] == [
        (0, 2, 0, 0),
        (2, 12, 1, 1),
        (12, 22, 0, 0),
        (22, 32, 1, 1),
        (32, 40, 0, 0),
    ]
    assert [
        (item.start_tick, item.stop_tick, item.start_value, item.stop_value)
        for item in dac.segments
    ] == [
        (0, 10, 1, 1),
        (10, 20, 1, -1),
        (20, 30, 1, 1),
        (30, 40, 1, -1),
    ]
    assert [item.label for item in timeline.annotations if item.kind == "period"] == [
        "on",
        "ramp",
        "on",
        "ramp",
    ]
    assert [item.label for item in timeline.annotations if item.kind == "repeat"] == [
        "repeat ×2"
    ]


def test_scan_preview_uses_visible_nominal_values_not_the_first_scan_row():
    base = _document()
    parameter = ScanParameter(
        "on_duration",
        PulseFieldRef(FIELD_DURATION, "on"),
        "on duration",
        "ns",
    )
    document = replace(
        base,
        scan_parameters=(parameter,),
        scan_table=FrozenScanTable(("on_duration",), ((200,),)),
    )

    _revision, timeline = PulseEditorSession(
        _descriptor(document.target),
        document,
    ).preview()

    assert timeline.reference_label == "nominal scan/API reference"
    assert timeline.logical_duration_ticks == 40


def test_timeline_rejects_labels_from_another_source_document():
    document = _document()
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        live_target=document.target,
    )

    with pytest.raises(ValueError, match="another PulseDocument"):
        build_pulse_timeline(
            replace(document, name="wrong labels"),
            artifact,
            reference_label="compiled static pulse",
        )


def test_hardware_run_requires_api_values_to_be_explicitly_resolved():
    document = _document()
    duration = PulseFieldRef(FIELD_DURATION, "on")
    unresolved = replace(
        document,
        api_parameters=(ApiParameter("on_duration", duration, "ns"),),
    )
    reference = DeviceRef("installation", "runtime", "sequencer")

    with pytest.raises(ValueError, match="unresolved API parameters"):
        PulseRunRequest(
            unresolved,
            PulseExecutionForm.STATIC_ONCE,
            reference,
            1.0,
        )

    resolved = resolve_api_parameters(unresolved, {"on_duration": 150})
    assert resolved.api_parameters == ()
    assert resolved.field_value(duration) == (150, "ns")
    PulseRunRequest(
        resolved,
        PulseExecutionForm.STATIC_ONCE,
        reference,
        1.0,
    )


def test_editor_save_detects_an_external_current_document_change(tmp_path):
    document = _document()
    session = PulseEditorSession(_descriptor(document.target), document)
    path = session.save(tmp_path / "pulse.json")
    session.replace_document(replace(document, name="local edit"))
    save_pulse_document(replace(document, name="external edit"), path)

    with pytest.raises(RuntimeError, match="changed on disk"):
        session.save()
    path.unlink()
    with pytest.raises(RuntimeError, match="changed on disk"):
        session.save()


def test_notebook_pulse_run_and_cancelled_hold_share_the_runtime_safe_path(tmp_path):
    with connect(repository=tmp_path / "experiment") as experiment:
        target = experiment.pulse.target
        document = PulseDocument(
            "runtime pulse",
            target.target,
            target.time_step_ns,
            (
                PulsePeriod(
                    "idle",
                    100,
                    "ns",
                    "idle",
                    tuple(0 for _ in target.target.raw_lanes),
                ),
            ),
            visible_ports=tuple(
                port.key
                for port in target.target.ports
                if port.kind in (PORT_DIGITAL, PORT_DAC)
            ),
        )

        finite = experiment.pulse.request(document)
        descriptor = experiment.pulse.inspect(finite)
        result = experiment.pulse.run(finite)
        assert isinstance(result, PulseRunResult)
        assert result.artifact_digest == descriptor.artifact_digest
        assert result.execution_form is PulseExecutionForm.STATIC_ONCE

        scan_parameter = ScanParameter(
            "idle_duration",
            PulseFieldRef(FIELD_DURATION, "idle"),
            "idle duration",
            "ns",
        )
        scan_document = replace(
            document,
            scan_parameters=(scan_parameter,),
            scan_table=FrozenScanTable(
                (scan_parameter.parameter_id,),
                ((100,), (200,)),
            ),
        )
        scan_result = experiment.pulse.run(
            experiment.pulse.request(
                scan_document,
                PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            )
        )
        assert scan_result.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE

        hold = experiment.pulse.start(
            experiment.pulse.request(
                document,
                PulseExecutionForm.CONTINUOUS_MONITOR,
            )
        )
        deadline = time.monotonic() + 2.0
        while hold.snapshot().phase != "holding-pulse":
            if time.monotonic() >= deadline:
                raise AssertionError("continuous pulse never reached its HOLD phase")
            time.sleep(0.005)
        assert hold.cancel("test Stop Pulse") is CancelOutcome.REQUESTED
        terminal = hold.wait(5.0)

    assert terminal.state is RunState.CANCELLED
    assert terminal.safety_bundle_id is not None
    assert not terminal.final_committed
