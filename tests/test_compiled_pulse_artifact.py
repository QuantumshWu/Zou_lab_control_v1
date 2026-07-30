"""CompiledPulseArtifact is one self-consistent current compilation result."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zlc_pulse.artifact as pulse_artifact_module
import zlc_pulse.fpga as pulse_fpga_module
import zlc_pulse.ir as pulse_ir_module

from zlc_pulse import (
    PORT_CLOCK,
    PORT_DAC,
    CompiledPulseArtifact,
    PulseExecutionForm,
    compile_pulse_artifact,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
    freeze_scan_table,
    load_pulse_document,
)
from zlc_pulse.artifact import (
    compiled_pulse_artifact_from_tree,
    compiled_pulse_artifact_to_tree,
)


ROOT = Path(__file__).parents[1]
IMAGING_PULSE = ROOT / "pulses" / "imaging_template.json"


def test_static_artifact_binds_source_ir_wire_and_trigger_schedule():
    document = load_pulse_document(IMAGING_PULSE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    assert artifact.source_document_digest == document.fingerprint
    assert artifact.wire_image.source_ir_digest
    assert artifact.trigger_schedules[0].channel == "ch11"
    assert artifact.trigger_schedules[0].total > 0
    assert artifact.target_abi_fingerprint == document.target.abi_fingerprint
    assert compiled_pulse_artifact_from_tree(
        compiled_pulse_artifact_to_tree(artifact)
    ) == artifact
    assert decode_compiled_pulse_artifact(
        encode_compiled_pulse_artifact(artifact)
    ) == artifact


def test_compiled_identity_getters_do_not_rehash_after_construction(monkeypatch):
    document = load_pulse_document(IMAGING_PULSE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    expected = (
        artifact.target_ir.fingerprint,
        artifact.wire_image.digest,
        artifact.fingerprint,
    )

    def unexpected_digest(*_args, **_kwargs):
        raise AssertionError("immutable identity getter recomputed its canonical digest")

    for module, name in (
        (pulse_ir_module, "canonical_digest"),
        (pulse_fpga_module, "canonical_digest"),
        (pulse_artifact_module, "sha256_digest"),
    ):
        monkeypatch.setattr(module, name, unexpected_digest)
    for _ in range(2):
        assert artifact.target_ir.fingerprint == expected[0]
        assert artifact.wire_image.digest == expected[1]
        assert artifact.fingerprint == expected[2]


def test_scan_artifact_preserves_each_physical_point_in_trigger_provenance():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    table, _report = freeze_scan_table(
        document,
        ("da_x", "da_y", "da_z"),
        ((0.0, 0.0, 0.0), (20.0, -10.0, 5.0), (-30.0, 15.0, 10.0)),
    )
    document = replace(document, scan_table=table)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch06",),
    )
    schedule = artifact.trigger_schedules[0]
    assert schedule.point_count == 3
    edges = tuple(schedule.iter_edges())
    assert [edge.point_index for edge in edges] == [0, 1, 2]
    assert [edge.point_trigger_ordinal for edge in edges] == [0, 0, 0]


def test_artifact_rejects_wire_or_schedule_from_another_compilation():
    document = load_pulse_document(IMAGING_PULSE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    with pytest.raises(ValueError, match="wire image"):
        replace(
            artifact,
            wire_image=replace(artifact.wire_image, source_ir_digest="0" * 64),
        )
    schedule = artifact.trigger_schedules[0]
    changed_ticks = np.array(schedule.ticks_from_run_start, copy=True)
    changed_ticks[0] += 1
    with pytest.raises(ValueError, match="deterministic"):
        replace(
            artifact,
            trigger_schedules=(replace(schedule, ticks_from_run_start=changed_ticks),),
        )


def test_continuous_artifact_has_no_finite_trigger_schedule():
    document = load_pulse_document(IMAGING_PULSE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    assert artifact.target_ir.repeat_forever
    assert artifact.trigger_schedules == ()
    with pytest.raises(ValueError, match="continuous monitor"):
        compile_pulse_artifact(
            document,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
            trigger_channels=("ch11",),
        )


@pytest.mark.parametrize("kind", [PORT_DAC, PORT_CLOCK])
def test_only_physical_digital_lanes_can_be_declared_as_triggers(kind):
    document = load_pulse_document(
        ROOT / "pulses" / "camera_imaging_address_switch.json"
    )
    port = next(port for port in document.target.ports if port.kind == kind)
    with pytest.raises(ValueError, match="not a digital port"):
        compile_pulse_artifact(
            document,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.STATIC_REFERENCE_POINT,
            trigger_channels=(port.lanes[0],),
        )


def test_compiled_artifact_requires_exact_current_schema():
    document = load_pulse_document(IMAGING_PULSE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    tree = compiled_pulse_artifact_to_tree(artifact)
    tree["legacy_alias"] = True
    with pytest.raises(ValueError, match="unknown field"):
        compiled_pulse_artifact_from_tree(tree)
