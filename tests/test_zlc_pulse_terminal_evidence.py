"""Raw hardware facts remain separate from compiled terminal expectations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import zlc_pulse.evidence as pulse_evidence_module
import zlc_pulse.transport as pulse_transport

from conftest import pulse_backend_completion_for
from fpga.pulse_streamer.host.image import STATUS_DONE, STATUS_UNDERFLOW
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    SimulatedPulseReceipt,
    pulse_terminal_ack_from_tree,
    pulse_terminal_ack_to_tree,
    validate_pulse_terminal_for_artifact,
)
from zlc_pulse import (
    AUTONOMOUS_TABLE_READ_RECIPE,
    POST_TERMINAL_TAIL_WAIT_RECIPE,
    STATIC_STATUS_READ_RECIPE,
    AutonomousTableTerminalEvidence,
    PostTerminalTailEvidence,
    PulseBackendCompletion,
    PulseCompletion,
    PulseExecutionForm,
    PreparedPulseRef,
    StaticOnceTerminalEvidence,
    build_pulse_playback,
    compile_pulse_artifact,
    freeze_scan_table,
    hardware_terminal_evidence_from_tree,
    hardware_terminal_evidence_to_tree,
    load_pulse_document,
    post_terminal_tail_evidence_from_tree,
    post_terminal_tail_evidence_to_tree,
    validate_backend_completion_for_artifact,
    validate_terminal_for_artifact,
)


ROOT = Path(__file__).parents[1]


def _static_artifact():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    return compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )


def _scan_artifact():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    table, _report = freeze_scan_table(
        document,
        ("da_x", "da_y", "da_z"),
        ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 4.0, 6.0)),
    )
    document = replace(document, scan_table=table)
    return compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch06",),
    )


def test_owner_codecs_round_trip_static_table_and_host_tail_evidence():
    for artifact in (_static_artifact(), _scan_artifact()):
        completion = pulse_backend_completion_for(artifact)
        terminal = completion.hardware_terminal
        tail = completion.post_terminal_tail
        assert hardware_terminal_evidence_from_tree(
            hardware_terminal_evidence_to_tree(terminal)
        ) == terminal
        assert post_terminal_tail_evidence_from_tree(
            post_terminal_tail_evidence_to_tree(tail)
        ) == tail
        validate_backend_completion_for_artifact(completion, artifact)


def test_validated_terminal_evidence_identity_is_bound_once(monkeypatch):
    completion = pulse_backend_completion_for(_static_artifact())
    terminal = completion.hardware_terminal
    tail = completion.post_terminal_tail
    expected = (terminal.fingerprint, tail.fingerprint)

    def unexpected_digest(*_args, **_kwargs):
        raise AssertionError("terminal evidence getter recomputed its canonical digest")

    monkeypatch.setattr(pulse_evidence_module, "canonical_digest", unexpected_digest)
    for _ in range(2):
        assert terminal.fingerprint == expected[0]
        assert tail.fingerprint == expected[1]


def test_neutral_terminal_ack_codec_preserves_hardware_and_simulated_receipts():
    artifact = _static_artifact()
    backend = pulse_backend_completion_for(artifact, transport_id="codec-test")
    counts = tuple(
        (schedule.channel, schedule.total)
        for schedule in artifact.trigger_schedules
    )
    reference = PreparedPulseRef(
        "generation-1",
        artifact.fingerprint,
    )
    hardware = PulseTerminalAck(
        "session-1",
        "binding-1",
        PulseCompletion(
            reference,
            backend.hardware_terminal,
            backend.post_terminal_tail,
            counts,
        ),
    )
    simulated = PulseTerminalAck(
        "session-2",
        "binding-2",
        SimulatedPulseReceipt(
            artifact.fingerprint,
            "test-simulator",
            counts,
            0.25,
            0.01,
        ),
    )
    for acknowledgement in (hardware, simulated):
        assert pulse_terminal_ack_from_tree(
            pulse_terminal_ack_to_tree(acknowledgement)
        ) == acknowledgement

    cross_tagged = pulse_terminal_ack_to_tree(simulated)
    cross_tagged["receipt_kind"] = "HARDWARE"
    with pytest.raises(ValueError):
        pulse_terminal_ack_from_tree(cross_tagged)


def test_success_completion_and_neutral_boundary_reject_fabricated_evidence():
    artifact = _static_artifact()
    reference = PreparedPulseRef(
        "generation-1",
        artifact.fingerprint,
    )
    failed_terminal = StaticOnceTerminalEvidence(
        STATIC_STATUS_READ_RECIPE,
        "fabricated",
        0,
        0,
        False,
        2,
    )
    failed_tail = PostTerminalTailEvidence(
        failed_terminal.fingerprint,
        POST_TERMINAL_TAIL_WAIT_RECIPE,
        0,
        artifact.target_ir.clock_hz,
        0,
    )
    with pytest.raises(ValueError, match="does not report DONE"):
        PulseCompletion(reference, failed_terminal, failed_tail, (("ch11", 3),))

    valid = pulse_backend_completion_for(artifact)
    acknowledgement = PulseTerminalAck(
        "session",
        "binding",
        PulseCompletion(
            reference,
            valid.hardware_terminal,
            valid.post_terminal_tail,
            (),
        ),
    )
    with pytest.raises(ValueError, match="expected counts differ"):
        validate_pulse_terminal_for_artifact(acknowledgement, artifact)


def test_neutral_boundary_rejects_simulated_duration_or_tail_not_derived_from_artifact():
    artifact = _static_artifact()
    logical_duration = build_pulse_playback(artifact).logical_duration
    counts = tuple(
        (schedule.channel, schedule.total)
        for schedule in artifact.trigger_schedules
    )
    for receipt, message in (
        (
            SimulatedPulseReceipt(
                artifact.fingerprint,
                "test-simulator",
                counts,
                999.0,
                artifact.max_configured_output_delay_ticks
                / artifact.target_ir.clock_hz,
            ),
            "duration differs",
        ),
        (
            SimulatedPulseReceipt(
                artifact.fingerprint,
                "test-simulator",
                counts,
                logical_duration,
                999.0,
            ),
            "tail differs",
        ),
    ):
        acknowledgement = PulseTerminalAck(
            "session",
            "binding",
            receipt,
        )
        with pytest.raises(ValueError, match=message):
            validate_pulse_terminal_for_artifact(acknowledgement, artifact)


def test_current_transport_surface_has_no_boolean_terminal_or_mixed_legacy_type():
    assert not hasattr(pulse_transport.DeployedStreamerSession, "wait_done")
    assert not hasattr(pulse_transport, "PulseHardwareTerminal")


def test_static_and_table_variants_are_not_interchangeable():
    static = _static_artifact()
    scan = _scan_artifact()
    static_terminal = StaticOnceTerminalEvidence(
        STATIC_STATUS_READ_RECIPE,
        "test",
        STATUS_DONE,
        STATUS_DONE,
        False,
        2,
    )
    table_terminal = AutonomousTableTerminalEvidence(
        AUTONOMOUS_TABLE_READ_RECIPE,
        "test",
        STATUS_DONE,
        2,
        STATUS_DONE,
        2,
        False,
        2,
    )
    validate_terminal_for_artifact(static_terminal, static)
    validate_terminal_for_artifact(table_terminal, scan)
    with pytest.raises(ValueError, match="forbids cursor"):
        validate_terminal_for_artifact(table_terminal, static)
    with pytest.raises(ValueError, match="requires cursor"):
        validate_terminal_for_artifact(static_terminal, scan)


@pytest.mark.parametrize(
    "terminal, message",
    [
        (
            StaticOnceTerminalEvidence(
                STATIC_STATUS_READ_RECIPE,
                "test",
                STATUS_DONE,
                STATUS_DONE | STATUS_UNDERFLOW,
                False,
                2,
            ),
            "not stable",
        ),
        (
            StaticOnceTerminalEvidence(
                STATIC_STATUS_READ_RECIPE,
                "test",
                STATUS_DONE,
                STATUS_DONE,
                True,
                2,
            ),
            "underflow",
        ),
    ],
)
def test_unstable_status_and_observed_underflow_are_rejected(terminal, message):
    with pytest.raises(ValueError, match=message):
        validate_terminal_for_artifact(terminal, _static_artifact())


def test_wrong_cursor_and_incomplete_tail_are_rejected():
    artifact = _scan_artifact()
    valid = pulse_backend_completion_for(artifact)
    terminal = replace(valid.hardware_terminal, cursor_second=1)
    with pytest.raises(ValueError, match="not stable"):
        validate_terminal_for_artifact(terminal, artifact)

    tail = PostTerminalTailEvidence(
        valid.hardware_terminal.fingerprint,
        POST_TERMINAL_TAIL_WAIT_RECIPE,
        artifact.max_configured_output_delay_ticks,
        artifact.target_ir.clock_hz,
        0,
    )
    completion = PulseBackendCompletion(valid.hardware_terminal, tail)
    if artifact.max_configured_output_delay_ticks:
        with pytest.raises(ValueError, match="tail is incomplete"):
            validate_backend_completion_for_artifact(completion, artifact)
