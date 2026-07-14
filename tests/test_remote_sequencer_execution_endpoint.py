from __future__ import annotations

from pathlib import Path
import threading
from conftest import pulse_backend_completion_for

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
from zlc_neutral_atom.timing.pulse import (
    FinitePulseExecutionRequest,
    PreparePulseCommand,
    PulseTerminalEvidenceKind,
)
from zlc_pulse import (
    PulseCompletion,
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
        self.completion = None

    def prepare(self, artifact):
        self.actions.append("prepare")
        self.prepared = artifact
        self.safe = False

    def fire(self, artifact):
        assert artifact is self.prepared
        self.actions.append("fire")

    def await_completion(self, artifact, timeout):
        assert artifact is self.prepared
        self.actions.append("wait")
        self.completion = pulse_backend_completion_for(
            artifact,
            transport_id="remote-test",
        )
        return self.completion

    def safe_state(self):
        self.actions.append("safe")
        self.prepared = None
        self.safe = True

    def request_interrupt(self):
        pass

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

    def current_interrupt_safe_state(self, generation):
        return encode(self.service.safe_state_for_generation(generation))


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
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    backend = Backend()
    service = PulseExecutionService(document.target, clock_hz=50e6, backend=backend)
    client = RemotePulseExecutionClient(
        Connection(service),
        Connection(service),
        transport_timeout_seconds=10.0,
    )
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

    assert isinstance(terminal.receipt, PulseCompletion)
    assert terminal.evidence_kind is PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS
    assert terminal.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert terminal.receipt.hardware_terminal.transport_id == "remote-test"
    assert terminal.receipt.hardware_terminal == backend.completion.hardware_terminal
    assert terminal.receipt.post_terminal_tail == backend.completion.post_terminal_tail
    assert terminal.artifact_digest == artifact.fingerprint
    assert backend.actions == ["prepare", "fire", "wait", "safe"]


def test_interrupt_fences_a_provisional_remote_prepare_before_it_can_fire():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )

    class BlockingBeforeServiceRoot(Root):
        def __init__(self, service):
            super().__init__(service)
            self.entered = threading.Event()
            self.release = threading.Event()

        def current_prepare(self, payload):
            self.entered.set()
            assert self.release.wait(1.0)
            return super().current_prepare(payload)

    backend = Backend()
    service = PulseExecutionService(document.target, clock_hz=50e6, backend=backend)
    control_connection = Connection(service)
    blocking_root = BlockingBeforeServiceRoot(service)
    control_connection.root = blocking_root
    client = RemotePulseExecutionClient(
        control_connection,
        Connection(service),
        transport_timeout_seconds=10.0,
    )
    endpoint = RemotePulseExecutionEndpoint(
        client,
        endpoint_label="test-fpga",
        max_blocking_call_seconds=5.0,
    )
    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    key = ResourceKey.parse("device/sequencer/remote-race")
    registry.register(
        LegacyDeviceRegistration(
            client,
            key,
            lambda: DeviceIdentityAck(
                "installation-endpoint:test-fpga-race",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
                "remote-current-connection",
                "test-assets-v1",
            ),
            {
                SafetyOperation.SAFE_STATE: lambda: (
                    client.safe_state(),
                    CleanupStepAck(SafetyOperation.SAFE_STATE, "remote-safe-command"),
                )[1]
            },
            (SafetyOperation.SAFE_STATE,),
            lambda: SafeStateAck("test-qualified-remote-safe"),
            target_endpoint=endpoint.target_endpoint,
        )
    )
    port = bind_sequencer_port(
        type("DeviceSet", (), {"devices": {"sequencer": client}})(),
        registry,
        SequencerBindingRequest(),
    )
    request = FinitePulseExecutionRequest(document, artifact)
    command = PreparePulseCommand(
        "race-session",
        "race-run",
        request,
        port.capability.capability_fingerprint,
        5.0,
    )
    errors = []

    def prepare():
        try:
            endpoint.execute_command(port.device, command)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=prepare)
    worker.start()
    assert blocking_root.entered.wait(1.0), errors
    endpoint.interrupt()
    blocking_root.release.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert errors
    assert "superseded by interrupt" in str(errors[0])
    assert backend.actions == ["safe", "prepare", "safe"]
    assert service.snapshot()["state"] == "SAFE"
    assert endpoint._session is not None and endpoint._session.closed
