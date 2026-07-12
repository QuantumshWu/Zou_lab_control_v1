"""CompiledPulseArtifact is one self-consistent current compilation result."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    PORT_CLOCK,
    PORT_DAC,
    CompiledPulseArtifact,
    PulseExecutionForm,
    compiled_pulse_artifact_from_tree,
    compiled_pulse_artifact_to_tree,
    load_pulse_document,
)
from zlc_workbench.pulse_compile_bridge import compile_pulse_artifact


ROOT = Path(__file__).parents[1]


def test_static_artifact_binds_source_ir_wire_and_trigger_schedule():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("emCCD",),
        live_target=document.target,
    )
    assert artifact.source_document_digest == document.fingerprint
    assert artifact.wire_image.source_ir_digest
    assert artifact.trigger_schedules[0].channel == "emCCD"
    assert artifact.trigger_schedules[0].total > 0
    assert artifact.target_abi_fingerprint == document.target.abi_fingerprint
    assert compiled_pulse_artifact_from_tree(
        compiled_pulse_artifact_to_tree(artifact)
    ) == artifact


def test_scan_artifact_preserves_each_physical_point_in_trigger_provenance():
    document = replace(
        load_pulse_document(ROOT / "pulses" / "mot_field_template.json"),
        scan_table=((0.0, 0.0, 0.0), (20.0, -10.0, 5.0), (-30.0, 15.0, 10.0)),
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("mot_trigger",),
    )
    schedule = artifact.trigger_schedules[0]
    assert schedule.point_count == 3
    assert [edge.point_index for edge in schedule.edges] == [0, 1, 2]
    assert [edge.point_trigger_ordinal for edge in schedule.edges] == [0, 0, 0]


def test_artifact_rejects_wire_or_schedule_from_another_compilation():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("emCCD",),
    )
    with pytest.raises(ValueError, match="wire image"):
        replace(
            artifact,
            wire_image=replace(artifact.wire_image, source_ir_digest="0" * 64),
        )
    schedule = artifact.trigger_schedules[0]
    changed_edge = replace(schedule.edges[0], tick_from_run_start=schedule.edges[0].tick_from_run_start + 1)
    with pytest.raises(ValueError, match="deterministic"):
        replace(
            artifact,
            trigger_schedules=(replace(schedule, edges=(changed_edge, *schedule.edges[1:])),),
        )


def test_continuous_artifact_has_no_finite_trigger_schedule():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    assert artifact.target_ir.repeat_forever
    assert artifact.trigger_schedules == ()
    with pytest.raises(ValueError, match="continuous monitor"):
        compile_pulse_artifact(
            document,
            clock_hz=100e6,
            execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
            trigger_channels=("emCCD",),
        )


@pytest.mark.parametrize("kind", [PORT_DAC, PORT_CLOCK])
def test_only_physical_digital_lanes_can_be_declared_as_triggers(kind):
    document = load_pulse_document(ROOT / "pulses" / "pulse_test.json")
    port = next(port for port in document.target.ports if port.kind == kind)
    with pytest.raises(ValueError, match="not a digital port"):
        compile_pulse_artifact(
            document,
            clock_hz=100e6,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=(port.lanes[0],),
        )


def test_compiled_artifact_requires_exact_current_schema():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    tree = compiled_pulse_artifact_to_tree(artifact)
    tree["legacy_alias"] = True
    with pytest.raises(ValueError, match="unknown field"):
        compiled_pulse_artifact_from_tree(tree)
