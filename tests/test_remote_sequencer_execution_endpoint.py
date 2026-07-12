from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.runtime import (
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
from zlc_neutral_atom.timing import FinitePulseExecutionRequest
from zlc_pulse import (
    PulseExecutionForm,
    PulseExecutionService,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_pulse.server import (
    decode_artifact_message,
    decode_prepared_ref_message,
    encode_completion_message,
    encode_prepared_ref_message,
)
from zlc_storage import encode
from zlc_workbench.legacy_runtime import LegacyDeviceRegistration, LegacyDeviceRegistry
from zlc_workbench.sequencer_execution import (
    RemotePulseExecutionEndpoint,
    SequencerBindingRequest,
    bind_sequencer_port,
)


ROOT = Path(__file__).parents[1]


class Backend:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.prepared = None
        self.safe = True

    def prepare(self, artifact):
        self.actions.append("prepare")
        self.prepared = artifact
        self.safe = False

    def fire(self, artifact):
        assert artifact is self.prepared
        self.actions.append("fire")

    def wait_done(self, artifact, timeout):
        assert artifact is self.prepared
        self.actions.append("wait")
        return True

    def safe_state(self):
        self.actions.append("safe")
        self.prepared = None
        self.safe = True

    def snapshot(self):
        return {"safe": self.safe}


class Root:
    def __init__(self, service):
        self.service = service

    def current_snapshot(self):
        return encode(self.service.snapshot())

    def current_prepare(self, payload):
        return encode_prepared_ref_message(
            self.service.prepare(decode_artifact_message(bytes(payload)))
        )

    def current_fire(self, payload):
        self.service.fire(decode_prepared_ref_message(bytes(payload)))
        return True

    def current_complete(self, payload, timeout):
        return encode_completion_message(
            self.service.complete(
                decode_prepared_ref_message(bytes(payload)), timeout=timeout
            )
        )

    def current_safe_state(self):
        self.service.safe_state()
        return True


class Connection:
    def __init__(self, service):
        self.root = Root(service)

    def close(self):
        pass


def _plan(port, request):
    def preflight(_context):
        return port.open_session(request)

    def execute(context, session):
        session.prepare(context)
        session.fire(context)
        return session.complete(context)

    def cleanup(context, session, _primary):
        return port.verify_idle(context) if session is None else session.cleanup(context)

    return RunPlan(
        "remote current pulse",
        RunMode.FINITE_EXACT,
        (port.resource_claim,),
        (port.hazard_claim,),
        (port.device,),
        preflight,
        execute,
        cleanup,
        lambda _context, result: result,
        port.interrupt_operations,
        5.0,
    )


def test_remote_current_endpoint_runs_exact_artifact_and_closes_safe():
    document = load_pulse_document(ROOT / "pulses" / "T.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    backend = Backend()
    service = PulseExecutionService(document.target, clock_hz=50e6, backend=backend)
    client = RemotePulseExecutionClient(Connection(service), transport_timeout_seconds=10.0)
    endpoint = RemotePulseExecutionEndpoint(client, endpoint_label="test-fpga", max_blocking_call_seconds=5.0)
    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    key = ResourceKey.parse("device/sequencer/remote")

    def cleanup():
        client.safe_state()
        return CleanupStepAck(SafetyOperation.SAFE_STATE, "remote-safe-command")

    def verify():
        if client.snapshot().state != "SAFE":
            raise RuntimeError("remote server is not safe")
        return SafeStateAck("test-qualified-remote-safe")

    registry.register(
        LegacyDeviceRegistration(
            client,
            key,
            lambda: DeviceIdentityAck(
                "installation-endpoint:test-fpga",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
                "remote-current-connection",
                "test-assets-v1",
            ),
            {SafetyOperation.SAFE_STATE: cleanup},
            (SafetyOperation.SAFE_STATE,),
            verify,
            target_endpoint=endpoint.target_endpoint,
        )
    )
    port = bind_sequencer_port(
        type("DeviceSet", (), {"devices": {"sequencer": client}})(),
        registry,
        SequencerBindingRequest(),
    )

    terminal = RunController(ResourceArbiter(MemoryQuarantineJournal())).run(
        _plan(port, FinitePulseExecutionRequest(document, artifact))
    )

    assert terminal.logical_done
    assert terminal.completed_schedule_trigger_counts == (("ch11", 2),)
    assert terminal.artifact_digest == artifact.fingerprint
    assert backend.actions == ["prepare", "fire", "wait", "safe"]
