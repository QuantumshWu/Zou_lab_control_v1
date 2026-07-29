"""Focused current contracts for pure frozen-scan execution semantics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import private_pulse_backend_snapshot

from fpga.pulse_streamer.host.image import CtrlWords, STATUS_DONE, StreamerParams
from zlc_pulse.artifact import (
    PulseExecutionForm,
    compiled_pulse_artifact_from_tree,
    compiled_pulse_artifact_to_tree,
)
from zlc_pulse.compiler import compile_pulse_artifact, compile_pulse_document
from zlc_pulse.document import (
    FIELD_DURATION,
    ApiParameter,
    FrozenScanTable,
    PulseFieldRef,
    RepeatRegion,
    load_pulse_document,
)
from zlc_pulse.evidence import (
    STATIC_STATUS_READ_RECIPE,
    StaticOnceTerminalEvidence,
    validate_terminal_for_artifact,
)
from zlc_pulse.scan_execution import (
    materialize_scan_sweeps,
    resolve_scan_point,
)
from zlc_pulse.server import PulseExecutionService
from zlc_pulse.manifest import pulse_target_manifest_from_lanes


ROOT = Path(__file__).parents[1]
MOT_SCAN = ROOT / "pulses" / "mot_field_template.json"


def _execution_document():
    source = load_pulse_document(MOT_SCAN)
    return replace(
        source,
        scan_table=FrozenScanTable(
            ("da_z", "da_x", "da_y"),
            ((30, 10, 20), (31, 11, 21)),
        ),
        api_parameters=(
            ApiParameter(
                "p2_duration",
                PulseFieldRef(FIELD_DURATION, "p2"),
                "ns",
            ),
        ),
        repeat=RepeatRegion("p1", "p3", 2),
    )


def _continuous_artifact():
    document = _execution_document()
    return document, compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    )


def test_materialize_scan_sweeps_repeats_only_frozen_rows_sweep_major():
    document = _execution_document()
    original_rows = document.scan_table.rows

    expanded = materialize_scan_sweeps(document, 3)

    assert expanded.scan_table.rows == original_rows * 3
    assert expanded.scan_table.columns == document.scan_table.columns
    assert expanded.scan_parameters == document.scan_parameters
    assert expanded.api_parameters == document.api_parameters
    assert expanded.repeat == document.repeat
    assert expanded.scan_recipe is None
    assert document.scan_table.rows == original_rows
    assert materialize_scan_sweeps(document, 1) is document


def test_materialize_scan_sweeps_validates_the_requested_repeat_count():
    document = _execution_document()

    with pytest.raises(TypeError, match="integer"):
        materialize_scan_sweeps(document, True)
    with pytest.raises(ValueError, match="positive"):
        materialize_scan_sweeps(document, 0)
    with pytest.raises(ValueError, match="frozen scan table"):
        materialize_scan_sweeps(replace(document, scan_table=None), 1)


def test_resolve_scan_point_joins_columns_by_parameter_id_and_preserves_api():
    document = _execution_document()

    resolved = resolve_scan_point(document, 1)

    expected = {"da_z": 31, "da_x": 11, "da_y": 21}
    for parameter in document.scan_parameters:
        assert resolved.field_value(parameter.field) == (
            expected[parameter.parameter_id],
            parameter.unit,
        )
    assert resolved.scan_parameters == ()
    assert resolved.scan_table is None
    assert resolved.scan_recipe is None
    assert resolved.api_parameters == document.api_parameters
    assert resolved.repeat == document.repeat
    assert document.scan_table is not None

    with pytest.raises(TypeError, match="integer"):
        resolve_scan_point(document, False)
    with pytest.raises(IndexError, match="outside"):
        resolve_scan_point(document, len(document.scan_table.rows))


def test_continuous_scan_compile_changes_only_outer_execution_cycle():
    document = _execution_document()
    finite = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    continuous = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    )

    assert continuous == replace(finite, repeat_forever=True)
    assert continuous.scan_enabled
    assert continuous.scan_points == finite.scan_points
    assert continuous.loop_count == document.repeat.count


def test_continuous_scan_artifact_wire_codec_and_invariants_are_closed():
    document, continuous = _continuous_artifact()
    words = continuous.wire_image.as_dict()

    assert continuous.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
    assert continuous.target_ir.repeat_forever
    assert continuous.trigger_schedules == ()
    assert words[CtrlWords.REPEAT_FOREVER] == 1
    assert words[CtrlWords.SCAN_COUNT] == len(document.scan_table.rows)
    assert compiled_pulse_artifact_from_tree(
        compiled_pulse_artifact_to_tree(continuous)
    ) == continuous

    with pytest.raises(ValueError, match="finite scan TargetIR"):
        replace(continuous, execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE)
    finite = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    with pytest.raises(ValueError, match="cyclic scan TargetIR"):
        replace(
            finite,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        )
    with pytest.raises(ValueError, match="continuous scan"):
        compile_pulse_artifact(
            document,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
            trigger_channels=("ch11",),
        )
    with pytest.raises(ValueError, match="frozen scan table"):
        compile_pulse_document(
            replace(document, scan_table=None),
            clock_hz=50e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        )


def test_continuous_scan_rejects_terminal_evidence_and_service_completion():
    document, artifact = _continuous_artifact()
    evidence = StaticOnceTerminalEvidence(
        STATIC_STATUS_READ_RECIPE,
        "focused-test",
        STATUS_DONE,
        STATUS_DONE,
        False,
        2,
    )
    with pytest.raises(ValueError, match="no finite terminal evidence"):
        validate_terminal_for_artifact(evidence, artifact)

    backend = _RecordingBackend()
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
        connection_generation="focused-generation",
    )
    reference = service.prepare(artifact)
    service.fire(reference)

    with pytest.raises(RuntimeError, match="no logical completion"):
        service.complete(reference, timeout=1.0)
    assert "await_completion" not in backend.actions


class _RecordingBackend:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.artifact = None
        self.state = "IDLE"

    def prepare(self, _artifact) -> None:
        self.actions.append("prepare")
        self.artifact = _artifact
        self.state = "PREPARED"

    def fire(self, _artifact) -> None:
        self.actions.append("fire")
        self.state = "RUNNING"

    def await_completion(self, _artifact, _timeout):
        self.actions.append("await_completion")
        raise AssertionError("continuous execution must not await finite completion")

    def safe_state(self) -> None:
        self.actions.append("safe_state")
        self.artifact = None
        self.state = "SAFE"

    def request_interrupt(self) -> None:
        self.actions.append("request_interrupt")

    def snapshot(self) -> dict[str, object]:
        return private_pulse_backend_snapshot(
            state=self.state,
            raw_lane_count=len(_execution_document().target.raw_lanes),
            artifact=self.artifact,
        )
