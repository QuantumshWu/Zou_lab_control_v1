"""Current target-owned virtual sequencer endpoint contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_neutral_atom.devices.simulation.sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.devices.simulation.apparatus import VirtualSequencer
from zlc_neutral_atom.runtime.ports import (
    DeviceBroker,
    SafetyOperation,
    SessionCloseCommand,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)
from zlc_neutral_atom.devices.sequencer.port import (
    CompletePulseCommand,
    ContinuousPulseExecutionRequest,
    FinitePulseExecutionRequest,
    FirePulseCommand,
    PreparePulseCommand,
    PulseTerminalEvidenceKind,
    SequencerCapabilitySnapshot,
    SimulatedPulseReceipt,
)
from zlc_pulse import (
    PulseExecutionForm,
    compile_pulse_artifact,
    freeze_scan_table,
    load_deployed_geometry_facts,
    load_pulse_document,
    pulse_target_manifest_from_lanes,
)


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


def _bound_virtual_sequencer(document):
    sequencer = VirtualSequencer(
        document.target,
        clock_hz=50e6,
        sleep_scale=0,
    )
    endpoint = VirtualSequencerExecutionEndpoint(
        sequencer,
        pulse_target_manifest_from_lanes(document.target),
        geometry_fingerprint=(
            load_deployed_geometry_facts().geometry_fingerprint
        ),
    )
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: PhysicalDeviceIdentity(
            stable_device_identity="virtual-sequencer:main",
            evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            asset_map_revision="test-assets-v1",
        )
    )
    binding = None

    def current_binding():
        assert binding is not None
        return binding

    binding = broker.bind(
        key=ResourceKey.parse("device/sequencer/main"),
        identity=identity,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(), command
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(), command
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    capability = broker.verify_capability(binding).snapshot
    assert isinstance(capability, SequencerCapabilitySnapshot)
    return sequencer, endpoint, broker, binding, capability


def _commands(request, capability, *, session_id="test-session", run_id="test-run"):
    artifact = request.artifact
    return (
        PreparePulseCommand(
            session_id,
            run_id,
            request,
            capability.capability_fingerprint,
            2.0,
        ),
        FirePulseCommand(session_id, artifact.fingerprint),
        CompletePulseCommand(session_id, artifact.fingerprint, 2.0),
    )


def _execute_finite(endpoint, binding, request, capability, *, session_id="test-session"):
    prepare, fire, complete = _commands(
        request,
        capability,
        session_id=session_id,
    )
    prepared = endpoint.execute_command(binding, prepare)
    fired = endpoint.execute_command(binding, fire)
    terminal = endpoint.execute_command(binding, complete)
    closed = endpoint.close_session(
        binding,
        SessionCloseCommand(session_id, 2.0),
    )
    assert prepared.session_id == fired.session_id == terminal.session_id == session_id
    assert closed.is_terminal
    return terminal


def _shutdown(broker, sequencer) -> None:
    broker.shutdown()
    sequencer.close()


def test_finite_pulse_runs_prepare_fire_terminal_then_verified_safe() -> None:
    document = load_pulse_document(IMAGING_TEMPLATE)
    sequencer, endpoint, broker, binding, capability = _bound_virtual_sequencer(
        document
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    terminal = _execute_finite(
        endpoint,
        binding,
        FinitePulseExecutionRequest(document, artifact),
        capability,
    )

    assert isinstance(terminal.receipt, SimulatedPulseReceipt)
    assert terminal.evidence_kind is PulseTerminalEvidenceKind.SIMULATED
    assert terminal.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert terminal.artifact_digest == artifact.fingerprint
    assert sequencer.snapshot()["state"] == "safe"
    _shutdown(broker, sequencer)


def test_live_wire_geometry_mismatch_is_rejected_before_prepare() -> None:
    document = load_pulse_document(IMAGING_TEMPLATE)
    sequencer, endpoint, broker, binding, capability = _bound_virtual_sequencer(
        document
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        params=replace(StreamerParams(), bank_size=1024),
    )
    request = FinitePulseExecutionRequest(document, artifact)
    prepare, _fire, _complete = _commands(request, capability)

    with pytest.raises(ValueError, match="wire geometry"):
        endpoint.execute_command(binding, prepare)
    assert sequencer.snapshot()["state"] == "safe"
    _shutdown(broker, sequencer)


@pytest.mark.parametrize(
    ("filename", "form", "trigger_channel"),
    [
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "ch11"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "ch06"),
    ],
)
def test_reference_and_scan_forms_execute_the_exact_compiled_ir(
    filename,
    form,
    trigger_channel,
) -> None:
    document = load_pulse_document(ROOT / "pulses" / filename)
    sequencer, endpoint, broker, binding, capability = _bound_virtual_sequencer(
        document
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=form,
        trigger_channels=(trigger_channel,),
        live_target=document.target,
    )
    terminal = _execute_finite(
        endpoint,
        binding,
        FinitePulseExecutionRequest(document, artifact),
        capability,
    )

    assert terminal.expected_trigger_counts_from_completed_schedule == (
        (trigger_channel, artifact.trigger_schedules[0].total),
    )
    assert sequencer.snapshot()["state"] == "safe"
    _shutdown(broker, sequencer)


def test_zero_time_virtual_continuous_scan_cursor_is_explicitly_unavailable() -> None:
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    table, _report = freeze_scan_table(
        document,
        ("da_x", "da_y", "da_z"),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    document = replace(document, scan_table=table)
    sequencer, endpoint, broker, binding, capability = _bound_virtual_sequencer(
        document
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        live_target=document.target,
    )
    request = ContinuousPulseExecutionRequest(document, artifact)
    prepare, fire, _complete = _commands(
        request,
        capability,
        session_id="continuous-scan",
        run_id="continuous-run",
    )
    endpoint.execute_command(binding, prepare)
    endpoint.execute_command(binding, fire)

    progress = endpoint.observe_scan_progress(
        binding,
        "continuous-scan",
        "continuous-run",
        artifact.fingerprint,
        2,
    )

    assert not progress.available
    assert "zero-time" in progress.unavailable_reason
    endpoint.close_session(
        binding,
        SessionCloseCommand("continuous-scan", 2.0),
    )
    _shutdown(broker, sequencer)


@pytest.mark.parametrize(
    ("phase", "fail_at", "expected_actions"),
    [
        ("prepare", 2, ["prepare", "safe"]),
        ("fire", 4, ["prepare", "fire", "safe"]),
        ("complete", 6, ["prepare", "fire", "complete", "safe"]),
    ],
)
def test_post_physical_transition_failure_seals_safe_and_preserves_primary(
    monkeypatch,
    phase,
    fail_at,
    expected_actions,
) -> None:
    document = load_pulse_document(IMAGING_TEMPLATE)
    sequencer, endpoint, broker, binding, capability = _bound_virtual_sequencer(
        document
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    request = FinitePulseExecutionRequest(document, artifact)
    prepare, fire, complete = _commands(
        request,
        capability,
        session_id="fault-session",
        run_id="fault-run",
    )

    actions = []
    real_prepare = endpoint._backend_prepare
    real_fire = endpoint._backend_fire
    real_complete = endpoint._backend_complete
    real_safe = endpoint._backend_set_safe_state
    real_validate = endpoint._owner._validate_binding
    validation_calls = 0

    def record_prepare(session):
        actions.append("prepare")
        return real_prepare(session)

    def record_fire(session):
        actions.append("fire")
        return real_fire(session)

    def record_complete(session, timeout_seconds):
        actions.append("complete")
        return real_complete(session, timeout_seconds)

    def fail_post_physical(value):
        nonlocal validation_calls
        validation_calls += 1
        real_validate(value)
        if validation_calls == fail_at:
            raise RuntimeError(f"injected {phase} post-physical binding failure")

    def safe_then_report_failure(timeout_seconds):
        actions.append("safe")
        real_safe(timeout_seconds)
        raise RuntimeError("injected safe-state reporting failure")

    monkeypatch.setattr(endpoint, "_backend_prepare", record_prepare)
    monkeypatch.setattr(endpoint, "_backend_fire", record_fire)
    monkeypatch.setattr(endpoint, "_backend_complete", record_complete)
    monkeypatch.setattr(endpoint._owner, "_validate_binding", fail_post_physical)
    monkeypatch.setattr(endpoint, "_backend_set_safe_state", safe_then_report_failure)

    if phase != "prepare":
        endpoint.execute_command(binding, prepare)
    if phase == "complete":
        endpoint.execute_command(binding, fire)
    failing_command = {"prepare": prepare, "fire": fire, "complete": complete}[phase]
    with pytest.raises(
        RuntimeError,
        match=rf"injected {phase} post-physical binding failure",
    ) as caught:
        endpoint.execute_command(binding, failing_command)

    assert actions == expected_actions
    assert sequencer.snapshot()["state"] == "safe"
    assert any(
        "sequencer fail-safe transition also failed" in note
        and "injected safe-state reporting failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )

    monkeypatch.setattr(endpoint._owner, "_validate_binding", real_validate)
    monkeypatch.setattr(endpoint, "_backend_set_safe_state", real_safe)
    failed_closed = endpoint.close_session(
        binding,
        SessionCloseCommand("fault-session", 2.0),
    )
    assert failed_closed.is_terminal

    retry = PreparePulseCommand(
        "retry-session",
        "retry-run",
        request,
        capability.capability_fingerprint,
        2.0,
    )
    acknowledgement = endpoint.execute_command(binding, retry)
    assert acknowledgement.session_id == "retry-session"
    closed = endpoint.close_session(
        binding,
        SessionCloseCommand("retry-session", 2.0),
    )
    assert closed.is_terminal
    _shutdown(broker, sequencer)
