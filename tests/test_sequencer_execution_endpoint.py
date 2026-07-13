"""Typed finite pulse sessions own prepare/FIRE/terminal/safe as one Run."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import StreamerParams
from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from Zou_lab_control.neutral_atom.ports import PortCatalog, PortSpec
from zlc_neutral_atom.runtime import (
    CleanupReport,
    CleanupStepAck,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    MemoryQuarantineJournal,
    ResourceArbiter,
    ResourceKey,
    RunController,
    RunMode,
    RunPlan,
    SafeStateAck,
    SafetyOperation,
)
from zlc_neutral_atom.timing import (
    FinitePulseExecutionRequest,
    PulseSessionState,
    PulseTerminalEvidenceKind,
    SimulatedPulseReceipt,
)
from zlc_pulse import (
    PulseExecutionForm,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_workbench.legacy_runtime import LegacyDeviceRegistration, LegacyDeviceRegistry
from zlc_workbench.sequencer_execution import (
    SequencerBindingRequest,
    VirtualSequencerExecutionEndpoint,
    bind_sequencer_port,
)


ROOT = Path(__file__).parents[1]


def _bound_virtual_sequencer(document):
    catalog = PortCatalog(
        document.target.raw_lanes,
        tuple(
            PortSpec(
                port.key,
                port.kind,
                port.lanes,
                port.label,
                port.bus_index,
                port.width,
                port.encoding,
                port.safe_value,
                port.latch_clock,
            )
            for port in document.target.ports
        ),
    )
    sequencer = VirtualSequencer(sleep_scale=0, port_catalog=catalog)
    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    endpoint = VirtualSequencerExecutionEndpoint(sequencer)
    key = ResourceKey.parse("device/sequencer/main")

    def cleanup():
        sequencer.set_safe_state()
        return CleanupStepAck(SafetyOperation.SAFE_STATE, "virtual-sequencer-safe-command")

    def verify():
        snapshot = dict(sequencer.snapshot())
        if snapshot.get("state") != "safe" or sequencer.firing is not None:
            raise RuntimeError("virtual sequencer is not physically safe")
        return SafeStateAck("virtual-sequencer-safe-readback")

    registry.register(
        LegacyDeviceRegistration(
            sequencer,
            key,
            lambda: DeviceIdentityAck(
                "virtual-sequencer:main",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
                "virtual-sequencer-connection",
                "test-assets-v1",
            ),
            {SafetyOperation.SAFE_STATE: cleanup},
            (SafetyOperation.SAFE_STATE,),
            verify,
            target_endpoint=endpoint.target_endpoint,
        )
    )
    port = bind_sequencer_port(
        type("DeviceSet", (), {"devices": {"sequencer": sequencer}})(),
        registry,
        SequencerBindingRequest(),
    )
    return sequencer, port, document


def _plan(port, request):
    holder = {}

    def preflight(_context):
        session = port.open_session(request)
        holder["session"] = session
        return session

    def execute(context, session):
        session.prepare(context)
        session.fire(context)
        return session.complete(context)

    def cleanup(context, session, _primary):
        return port.verify_idle(context) if session is None else session.cleanup(context)

    return RunPlan(
        "typed finite pulse",
        RunMode.FINITE_EXACT,
        (port.resource_claim,),
        (port.hazard_claim,),
        (port.device,),
        preflight,
        execute,
        cleanup,
        lambda _context, result: result,
        port.interrupt_operations,
        3.0,
    ), holder


def test_finite_pulse_runs_prepare_fire_terminal_then_verified_safe():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    sequencer, port, document = _bound_virtual_sequencer(document)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    plan, holder = _plan(port, FinitePulseExecutionRequest(document, artifact))
    terminal = RunController(ResourceArbiter(MemoryQuarantineJournal())).run(plan)

    assert isinstance(terminal.receipt, SimulatedPulseReceipt)
    assert terminal.evidence_kind is PulseTerminalEvidenceKind.SIMULATED
    assert terminal.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert terminal.artifact_digest == artifact.fingerprint
    assert holder["session"].state is PulseSessionState.COMPLETED
    assert dict(sequencer.snapshot())["state"] == "safe"
    assert [item["action"] for item in sequencer.history] == [
        "prepare",
        "fire",
        "wait_done",
        "safe",
    ]


def test_live_target_mismatch_is_rejected_before_prepare_or_fire():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    sequencer, port, document = _bound_virtual_sequencer(document)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        params=replace(StreamerParams(), bank_size=1024),
    )
    with pytest.raises(ValueError, match="wire geometry"):
        port.open_session(FinitePulseExecutionRequest(document, artifact))
    assert sequencer.history == []


@pytest.mark.parametrize(
    ("filename", "form", "trigger_channel"),
    [
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "ch11"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "ch11"),
    ],
)
def test_reference_and_scan_forms_execute_the_exact_compiled_ir(
    filename,
    form,
    trigger_channel,
):
    document = load_pulse_document(ROOT / "pulses" / filename)
    sequencer, port, document = _bound_virtual_sequencer(document)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=form,
        trigger_channels=(trigger_channel,),
        live_target=document.target,
    )
    plan, _holder = _plan(port, FinitePulseExecutionRequest(document, artifact))
    terminal = RunController(ResourceArbiter(MemoryQuarantineJournal())).run(plan)
    assert terminal.expected_trigger_counts_from_completed_schedule == (
        (trigger_channel, artifact.trigger_schedules[0].total),
    )
    assert dict(sequencer.snapshot())["state"] == "safe"
