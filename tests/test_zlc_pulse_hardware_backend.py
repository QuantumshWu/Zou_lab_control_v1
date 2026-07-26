from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    STATUS_DONE,
    STATUS_LOADED,
    STATUS_RUNNING,
    CtrlWords,
    StreamerParams,
    build_fingerprint,
)
from fpga.pulse_streamer.host import uart_frame
from fpga.pulse_streamer.host.uart_bridge_model import UartBridgeModel
from zlc_pulse import (
    PulseExecutionForm,
    PulseExecutionService,
    pack_target_ir,
    compile_pulse_artifact,
    load_pulse_document,
    pulse_target_manifest_from_lanes,
)
from zlc_pulse.transport import (
    DeployedStreamerSession,
    UartRegisterTransport,
    VivadoAxiRegisterTransport,
)


ROOT = Path(__file__).parents[1]


class MemoryDeviceLease:
    def __init__(self):
        self.acquired = False

    def acquire(self):
        if self.acquired:
            raise RuntimeError("test device lease is already held")
        self.acquired = True

    def release(self):
        self.acquired = False


def _decode_axi_writes(text: str) -> list[tuple[int, int]]:
    writes: list[tuple[int, int]] = []
    for address_hex, data_hex, length_text in re.findall(
        r"-address ([0-9A-Fa-f]+) -data ([0-9A-Fa-f]+) -len (\d+) -type write",
        text,
    ):
        base = int(address_hex, 16) // 4
        length = int(length_text)
        words = [
            int(data_hex[index * 8 : (index + 1) * 8], 16)
            for index in range(length)
        ]
        for offset, value in enumerate(reversed(words)):
            writes.append((base + offset, value))
    return writes


class CurrentStreamerHardware:
    def __init__(self, params: StreamerParams) -> None:
        self.params = params
        self.words: dict[int, int] = {}
        self.status = 0
        self.fired = False

    def __call__(self, lines, action, timeout):
        text = "\n".join(lines)
        for address, value in _decode_axi_writes(text):
            self.words[address] = value
            if address != CtrlWords.COMMAND or value == 0:
                continue
            if value & CMD_SAFE:
                self.status = 0
                self.fired = False
            if value & CMD_LOAD:
                self.status = STATUS_LOADED
            if value & CMD_FIRE:
                self.status = STATUS_RUNNING
                self.fired = True
        match = re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
        if match:
            address = int(match.group(1), 16) // 4
            if address == CtrlWords.LAYOUT_ID:
                return f"ZLCDATA {build_fingerprint(self.params):08X}\n"
            if address == CtrlWords.STATUS:
                if self.fired:
                    self.status |= STATUS_DONE
                return f"ZLCDATA {self.status:08X}\n"
            return f"ZLCDATA {self.words.get(address, 0):08X}\n"
        return "ok\n"


class ModelUartLink:
    """Test-only UART link backed by the cycle-faithful register model."""

    def __init__(self, layout_id):
        self.model = UartBridgeModel(layout_id=layout_id)
        self.writes = self.model.regfile

    def open(self):
        pass

    def close(self):
        pass

    def exchange(self, request, *, deadline, stop=None):
        events = [
            event
            for event in self.model.feed(request)
            if event.op in {"write", "read"}
        ]
        if not events:
            raise RuntimeError("UART model received a corrupt frame")
        for event in events:
            if (
                event.op == "write"
                and event.base == CtrlWords.COMMAND
                and event.values
            ):
                command = event.values[-1] & 0xF
                if command == CMD_LOAD:
                    self.model.regfile[CtrlWords.STATUS] = STATUS_LOADED
                elif command == CMD_FIRE:
                    self.model.regfile[CtrlWords.STATUS] = STATUS_RUNNING
                elif command == CMD_SAFE:
                    self.model.regfile[CtrlWords.STATUS] = 0
        return events[-1].reply

    def write_batch(self, requests, *, deadline, stop=None):
        return [
            self.exchange(request, deadline=deadline, stop=stop)
            for request in requests
        ]


def _axi_session(tmp_path, target, params, hardware, *, clock_hz=50e6):
    return DeployedStreamerSession(
        VivadoAxiRegisterTransport(
            state_dir=tmp_path,
            tcl_executor=hardware,
        ),
        device_lease=MemoryDeviceLease(),
        deployed_target=target,
        params=params,
        clock_hz=clock_hz,
    ).start()


def _uart_session(tmp_path, target, params, link):
    return DeployedStreamerSession(
        UartRegisterTransport(state_dir=tmp_path, link=link),
        device_lease=MemoryDeviceLease(),
        deployed_target=target,
        params=params,
        clock_hz=50e6,
    ).start()


def _artifact(params: StreamerParams):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    return document, compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        params=params,
    )


def test_current_artifact_bytes_drive_the_existing_axi_transport_exactly(tmp_path):
    params = StreamerParams()
    document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = _axi_session(tmp_path, document.target, params, hardware)
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=session,
        params=params,
    )

    reference = service.prepare(artifact)
    for address, value in artifact.wire_image.words:
        assert hardware.words[address] == value
    service.fire(reference)
    completion = service.complete(reference, timeout=1.0)

    assert completion.expected_trigger_counts_from_completed_schedule == (("ch11", 3),)
    assert service.snapshot()["backend"]["prepared_artifact_digest"] == artifact.fingerprint
    service.safe_state()
    assert service.snapshot()["backend"]["prepared_artifact_digest"] is None


def test_current_session_rejects_clock_mismatch_before_hardware_access(tmp_path):
    params = StreamerParams()
    document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = _axi_session(
        tmp_path,
        document.target,
        params,
        hardware,
        clock_hz=40e6,
    )

    with pytest.raises(ValueError, match="clock"):
        session.prepare(artifact)
    assert hardware.words == {}


def test_prepare_trusts_the_topology_bound_by_the_session_constructor(
    monkeypatch,
    tmp_path,
):
    params = StreamerParams()
    document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = _axi_session(tmp_path, document.target, params, hardware)

    def forbidden_static_revalidation(*args, **kwargs):
        raise AssertionError("prepare must trust the constructor-bound deployment")

    monkeypatch.setattr(
        "zlc_pulse.deployment.validate_deployed_target",
        forbidden_static_revalidation,
    )
    session.prepare(artifact)
    assert session.snapshot()["prepared_artifact_digest"] == artifact.fingerprint


def test_current_artifact_uses_the_same_contract_over_uart(tmp_path):
    params = StreamerParams()
    document, artifact = _artifact(params)
    transport = ModelUartLink(build_fingerprint(params))
    session = _uart_session(tmp_path, document.target, params, transport)
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=session,
        params=params,
    )

    reference = service.prepare(artifact)
    for address, value in artifact.wire_image.words:
        assert transport.writes[address] == value
    service.fire(reference)
    transport.model.regfile[CtrlWords.STATUS] |= STATUS_DONE
    assert service.complete(reference, timeout=1.0).prepared_ref == reference
    assert service.snapshot()["backend"]["transport"] == "uart"


def _self_attested_wrong_topology_artifact(params):
    document, artifact = _artifact(params)
    dac_lane = next(
        port.lanes[0] for port in document.target.ports if port.kind == "dac"
    )
    bit = 1 << document.target.raw_lanes.index(dac_lane)
    masks = list(artifact.target_ir.masks)
    masks[0] |= bit
    wrong_ir = replace(artifact.target_ir, masks=tuple(masks))
    return document, replace(
        artifact,
        target_ir=wrong_ir,
        wire_image=pack_target_ir(wrong_ir, params),
        trigger_schedules=(),
    )


def test_axi_rejects_self_attested_wrong_topology_before_any_write(tmp_path):
    params = StreamerParams()
    document, artifact = _self_attested_wrong_topology_artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = _axi_session(tmp_path, document.target, params, hardware)

    with pytest.raises(ValueError, match="non-digital"):
        session.prepare(artifact)
    assert hardware.words == {}


def test_uart_rejects_self_attested_wrong_topology_before_any_write(tmp_path):
    params = StreamerParams()
    document, artifact = _self_attested_wrong_topology_artifact(params)
    transport = ModelUartLink(build_fingerprint(params))
    session = _uart_session(tmp_path, document.target, params, transport)

    with pytest.raises(ValueError, match="non-digital"):
        session.prepare(artifact)
    assert transport.writes == {}


def test_session_has_only_the_current_service_backend_vocabulary(tmp_path):
    params = StreamerParams()
    document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = _axi_session(tmp_path, document.target, params, hardware)

    assert all(callable(getattr(session, name)) for name in (
        "prepare",
        "fire",
        "await_completion",
        "safe_state",
        "request_interrupt",
        "snapshot",
    ))
    assert not hasattr(session, "prepare_compiled_artifact")
    assert not hasattr(session, "scan_progress")
    assert hardware.words == {}


def test_fire_and_await_are_constant_time_identity_checks_after_prepare(
    monkeypatch,
    tmp_path,
):
    params = StreamerParams()
    document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = _axi_session(tmp_path, document.target, params, hardware)
    session.prepare(artifact)

    def forbidden_revalidation(*args, **kwargs):
        raise AssertionError("FIRE/await must not repack after prepare")

    monkeypatch.setattr(
        "zlc_pulse.transport.session._validate_artifact_against_bound_deployment",
        forbidden_revalidation,
    )
    session.fire(artifact)
    assert session.await_completion(artifact, timeout=1.0) is not None
