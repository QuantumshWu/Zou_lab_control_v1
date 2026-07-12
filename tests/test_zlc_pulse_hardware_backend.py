from __future__ import annotations

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
from zlc_pulse import (
    PulseExecutionForm,
    PulseExecutionService,
    PulseStreamerSessionBackend,
    compile_pulse_artifact,
    load_pulse_document,
)
from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
from Zou_lab_control.neutral_atom.devices.uart_session import FakeUartTransport, UartStreamerSession


ROOT = Path(__file__).parents[1]


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


def _artifact(params: StreamerParams):
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    return document, compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("emCCD",),
        params=params,
    )


def test_current_artifact_bytes_drive_the_existing_axi_transport_exactly(tmp_path):
    params = StreamerParams()
    document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = VivadoAxiStreamerSession(
        state_dir=tmp_path,
        params=params,
        tcl_executor=hardware,
    )
    service = PulseExecutionService(
        document.target,
        clock_hz=50e6,
        backend=PulseStreamerSessionBackend(session),
        params=params,
    )

    reference = service.prepare(artifact)
    for address, value in artifact.wire_image.words:
        assert hardware.words[address] == value
    service.fire(reference)
    completion = service.complete(reference, timeout=1.0)

    assert completion.logical_done
    assert completion.completed_schedule_trigger_counts == (("emCCD", 3),)
    assert service.snapshot()["backend"]["prepared_artifact_digest"] == artifact.fingerprint
    service.safe_state()
    assert service.snapshot()["backend"]["prepared_artifact_digest"] is None


def test_current_session_rejects_clock_mismatch_before_hardware_access(tmp_path):
    params = StreamerParams()
    _document, artifact = _artifact(params)
    hardware = CurrentStreamerHardware(params)
    session = VivadoAxiStreamerSession(
        state_dir=tmp_path,
        params=params,
        clock_hz=40e6,
        tcl_executor=hardware,
    )

    with pytest.raises(ValueError, match="clock"):
        session.prepare_compiled_artifact(artifact)
    assert hardware.words == {}


def test_current_artifact_uses_the_same_contract_over_uart(tmp_path):
    params = StreamerParams()
    document, artifact = _artifact(params)
    transport = FakeUartTransport(layout_id=build_fingerprint(params))
    session = UartStreamerSession(
        state_dir=tmp_path,
        params=params,
        transport=transport,
    )
    service = PulseExecutionService(
        document.target,
        clock_hz=50e6,
        backend=PulseStreamerSessionBackend(session),
        params=params,
    )

    reference = service.prepare(artifact)
    for address, value in artifact.wire_image.words:
        assert transport.writes[address] == value
    service.fire(reference)
    transport.model.regfile[CtrlWords.STATUS] |= STATUS_DONE
    assert service.complete(reference, timeout=1.0).logical_done
    assert service.snapshot()["backend"]["transport"] == "uart"
