"""Sequencer devices and the runtime pulse-table service boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .base import SequencerDevice
from ..timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    PulseSequence,
    channel_names,
    count_trigger_pulses,
    positive_float,
)
from ..timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle


@dataclass(frozen=True)
class RuntimeSequenceProgram:
    """Runtime edge-table program uploaded to a pulse-streamer-like FPGA."""

    sequence_id: str
    sequence_name: str
    clock_hz: float
    channels: list[str]
    ticks: list[int]
    masks: list[int]
    duration: float
    trigger_count: int
    source_sequence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema": "Zou_lab_control.neutral_atom.RuntimeSequenceProgram",
            "version": 1,
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "clock_hz": self.clock_hz,
            "channels": list(self.channels),
            "ticks": list(self.ticks),
            "masks": list(self.masks),
            "duration": self.duration,
            "trigger_count": self.trigger_count,
        }
        if self.source_sequence is not None:
            payload["source_sequence"] = self.source_sequence
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeSequenceProgram":
        if payload.get("schema") != "Zou_lab_control.neutral_atom.RuntimeSequenceProgram":
            raise ValueError("unsupported runtime sequence program schema.")
        return cls(
            sequence_id=str(payload["sequence_id"]),
            sequence_name=str(payload["sequence_name"]),
            clock_hz=positive_float(payload["clock_hz"], "clock_hz"),
            channels=list(channel_names(payload["channels"], "channels")),
            ticks=[int(v) for v in payload["ticks"]],
            masks=[int(v) for v in payload["masks"]],
            duration=float(payload["duration"]),
            trigger_count=int(payload["trigger_count"]),
            source_sequence=None if payload.get("source_sequence") is None else dict(payload["source_sequence"]),
        )


def compile_runtime_program(
    sequence: PulseSequence,
    *,
    channels: Sequence[str],
    clock_hz: float = 250e6,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> RuntimeSequenceProgram:
    """Compile a ``PulseSequence`` into an uploadable edge table."""

    channels = list(channel_names(channels, "channels"))
    clock_hz = positive_float(clock_hz, "clock_hz")
    ticks, masks, channels = sequence.edges(clock_hz=clock_hz, channels=channels)
    payload = {
        "sequence": sequence.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
    }
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=sequence.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=list(ticks),
        masks=list(masks),
        duration=sequence.duration,
        trigger_count=count_trigger_pulses(sequence, trigger_channels=trigger_channels),
        source_sequence=sequence.to_dict(),
    )


class SequencerService:
    """Stateful service that mirrors the final FPGA runtime protocol.

    The same object can run in-process for tests, or be exposed over RPyC on
    the FPGA/Vivado computer.  Hardware-specific callbacks can be attached
    later without changing the client-side ``SequencerDevice`` contract.
    """

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = 250e6,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        prepare_callback: Callable[[RuntimeSequenceProgram], None] | None = None,
        fire_callback: Callable[[RuntimeSequenceProgram], None] | None = None,
        wait_done_callback: Callable[[RuntimeSequenceProgram, float | None], bool] | None = None,
        safe_state_callback: Callable[[], None] | None = None,
        sleep_scale: float = 0.0,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.prepare_callback = prepare_callback
        self.fire_callback = fire_callback
        self.wait_done_callback = wait_done_callback
        self.safe_state_callback = safe_state_callback
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        self._lock = threading.RLock()
        self.prepared_program: RuntimeSequenceProgram | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []

    def prepare(self, sequence_payload) -> dict[str, object]:
        sequence = sequence_from_payload(sequence_payload)
        program = compile_runtime_program(
            sequence,
            channels=self.channels,
            clock_hz=self.clock_hz,
            trigger_channels=self.trigger_channels,
        )
        with self._lock:
            if self.prepare_callback is not None:
                self.prepare_callback(program)
            self.prepared_program = program
            self.state = "prepared"
            self.history.append({"action": "prepare", "sequence_id": program.sequence_id, "triggers": program.trigger_count})
        return program.to_dict()

    def fire(self, sequence_payload=None) -> dict[str, object]:
        with self._lock:
            program = self._require_prepared()
            if sequence_payload is not None:
                requested = compile_runtime_program(
                    sequence_from_payload(sequence_payload),
                    channels=self.channels,
                    clock_hz=self.clock_hz,
                    trigger_channels=self.trigger_channels,
                )
                if requested.sequence_id != program.sequence_id:
                    raise RuntimeError("fire(sequence) does not match the prepared runtime program.")
            if self.fire_callback is not None:
                self.fire_callback(program)
            self.state = "running"
            self.history.append({"action": "fire", "sequence_id": program.sequence_id})
            return program.to_dict()

    def wait_done(self, timeout: float | None = None) -> bool:
        with self._lock:
            program = self._require_prepared()
        if self.wait_done_callback is not None:
            ok = bool(self.wait_done_callback(program, timeout))
        else:
            delay = program.duration * self.sleep_scale
            if timeout is not None and delay > float(timeout):
                ok = False
            else:
                if delay > 0:
                    time.sleep(delay)
                ok = True
        with self._lock:
            self.state = "done" if ok else "timeout"
            self.history.append({"action": "wait_done", "sequence_id": program.sequence_id, "ok": ok})
        return ok

    def abort(self) -> None:
        with self._lock:
            self.state = "aborted"
            self.history.append({"action": "abort"})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def set_safe_state(self) -> None:
        with self._lock:
            self.state = "safe"
            self.history.append({"action": "safe"})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "type": type(self).__name__,
                "channels": list(self.channels),
                "clock_hz": self.clock_hz,
                "trigger_channels": list(self.trigger_channels),
                "state": self.state,
                "prepared_program": None if self.prepared_program is None else self.prepared_program.to_dict(),
                "history_length": len(self.history),
            }

    def _require_prepared(self) -> RuntimeSequenceProgram:
        if self.prepared_program is None:
            raise RuntimeError("sequencer service has no prepared sequence.")
        return self.prepared_program


class RuntimeSequencer(SequencerDevice):
    """Local device adapter for the runtime edge-table protocol."""

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = 250e6,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        sleep_scale: float = 0.0,
    ):
        self.service = SequencerService(
            channels=channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            sleep_scale=sleep_scale,
        )
        self.channels = self.service.channels
        self.clock_hz = self.service.clock_hz
        self.last_program: RuntimeSequenceProgram | None = None

    def prepare(self, sequence: PulseSequence) -> RuntimeSequenceProgram:
        self.last_program = RuntimeSequenceProgram.from_dict(self.service.prepare(sequence.to_dict()))
        return self.last_program

    def fire(self, sequence: PulseSequence | None = None) -> None:
        self.service.fire(None if sequence is None else sequence.to_dict())

    def wait_done(self, timeout: float | None = None) -> bool:
        return self.service.wait_done(timeout)

    def abort(self) -> None:
        self.service.abort()

    def set_safe_state(self) -> None:
        self.service.set_safe_state()

    def snapshot(self) -> dict[str, object]:
        out = self.service.snapshot()
        out["type"] = type(self).__name__
        return out


class ManualSequencer(SequencerDevice):
    """Sequencer adapter for first-light tests with a manually started FPGA.

    ``fire`` intentionally does not drive hardware.  It records that the camera
    is armed and that the operator or an external free-running FPGA must provide
    the trigger pulses before the qCMOS timeout expires.
    """

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = 250e6,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        message: str | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.message = message or "Camera is armed. Start the FPGA/manual trigger sequence now."
        self.prepared_sequence: PulseSequence | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []

    def prepare(self, sequence: PulseSequence) -> RuntimeSequenceProgram:
        sequence.validate(clock_hz=self.clock_hz, channels=self.channels).raise_if_failed()
        program = compile_runtime_program(
            sequence,
            channels=self.channels,
            clock_hz=self.clock_hz,
            trigger_channels=self.trigger_channels,
        )
        self.prepared_sequence = sequence
        self.state = "prepared"
        self.history.append({"action": "prepare", "sequence_id": program.sequence_id, "triggers": program.trigger_count})
        return program

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self.prepared_sequence is None:
            raise RuntimeError("ManualSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self.prepared_sequence:
            raise RuntimeError("ManualSequencer.fire() received a sequence that was not prepared.")
        self.state = "manual_trigger_wait"
        self.history.append({"action": "fire_manual", "message": self.message})
        print(self.message)

    def wait_done(self, timeout: float | None = None) -> bool:
        self.state = "unknown_done"
        self.history.append({"action": "wait_done_manual", "timeout": timeout})
        return True

    def abort(self) -> None:
        self.state = "aborted"
        self.history.append({"action": "abort"})

    def set_safe_state(self) -> None:
        self.state = "safe_requested"
        self.history.append({"action": "safe"})

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "channels": list(self.channels),
            "clock_hz": self.clock_hz,
            "trigger_channels": list(self.trigger_channels),
            "state": self.state,
            "prepared": self.prepared_sequence is not None,
            "history_length": len(self.history),
        }


class RemoteSequencer(SequencerDevice):
    """RPyC client-side sequencer proxy for the FPGA/Vivado computer."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        channels: Sequence[str],
        clock_hz: float = 250e6,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        ssl: bool = False,
        ca_certs: str | None = None,
        connect_on_init: bool = False,
    ):
        self.host = str(host).strip()
        if self.host in {"", "0.0.0.0", "::"}:
            raise ValueError("RemoteSequencer host must be the server address reachable from the control computer.")
        self.port = int(port)
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.ssl = bool(ssl)
        self.ca_certs = ca_certs
        self._conn = None
        self._last_program: RuntimeSequenceProgram | None = None
        if connect_on_init:
            self.open()

    def open(self) -> "RemoteSequencer":
        if self._conn is not None:
            return self
        try:
            import rpyc
        except ImportError as exc:  # pragma: no cover - depends on lab install
            raise RuntimeError("RemoteSequencer requires `rpyc`. Install it on the control computer.") from exc
        if self.ssl:
            self._conn = rpyc.utils.classic.ssl_connect(host=self.host, port=self.port, ca_certs=self.ca_certs)
        else:
            self._conn = rpyc.connect(self.host, self.port, config={"allow_pickle": True, "sync_request_timeout": None})
        snap = self._conn.root.snapshot()
        self.channels = list(snap.get("channels", self.channels))
        self.clock_hz = float(snap.get("clock_hz", self.clock_hz))
        return self

    def prepare(self, sequence: PulseSequence) -> RuntimeSequenceProgram:
        self.open()
        program = self._conn.root.prepare(json.dumps(sequence.to_dict()))
        payload = json.loads(program) if isinstance(program, (str, bytes)) else dict(program)
        self._last_program = RuntimeSequenceProgram.from_dict(payload)
        return self._last_program

    def fire(self, sequence: PulseSequence | None = None) -> None:
        self.open()
        self._conn.root.fire(None if sequence is None else json.dumps(sequence.to_dict()))

    def wait_done(self, timeout: float | None = None) -> bool:
        self.open()
        return bool(self._conn.root.wait_done(timeout))

    def abort(self) -> None:
        if self._conn is not None:
            self._conn.root.abort()

    def set_safe_state(self) -> None:
        self.open()
        self._conn.root.set_safe_state()

    def snapshot(self) -> dict[str, object]:
        out = {
            "type": type(self).__name__,
            "host": self.host,
            "port": self.port,
            "channels": list(self.channels),
            "clock_hz": self.clock_hz,
            "trigger_channels": list(self.trigger_channels),
            "connected": self._conn is not None,
            "last_program": None if self._last_program is None else self._last_program.to_dict(),
        }
        if self._conn is not None:
            try:
                out["remote"] = dict(self._conn.root.snapshot())
            except Exception as exc:
                out["remote_error"] = str(exc)
        return out

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


class VerilogSequencer(SequencerDevice):
    """Prepare writes generated Verilog; fire calls an optional hardware hook."""

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = 250e6,
        output_dir: str | Path = "generated_sequences",
        module_name: str = "zlc_sequence",
        pin_map: Mapping[str, str] | None = None,
        fire_callback: Callable[[VerilogBuild], None] | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.output_dir = Path(output_dir)
        self.module_name = str(module_name)
        self.pin_map = None if pin_map is None else dict(pin_map)
        self.fire_callback = fire_callback
        self.last_build: VerilogBuild | None = None
        self.last_files: VerilogFiles | None = None
        self.prepared_sequence: PulseSequence | None = None

    def prepare(self, sequence: PulseSequence) -> VerilogBuild:
        build = generate_verilog(sequence, channels=self.channels, clock_hz=self.clock_hz, module_name=self.module_name)
        self.last_build = build
        self.last_files = write_verilog_bundle(build, self.output_dir, pin_map=self.pin_map)
        self.prepared_sequence = sequence
        return build

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self.last_build is None or self.prepared_sequence is None:
            raise RuntimeError("VerilogSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self.prepared_sequence:
            raise RuntimeError("VerilogSequencer.fire() received a sequence that was not prepared.")
        if self.fire_callback is None:
            raise RuntimeError(
                "Verilog files were generated, but no fire_callback is configured. "
                "For real hardware, pass a callback that starts the FPGA after the camera is armed."
            )
        self.fire_callback(self.last_build)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "channels": self.channels,
            "clock_hz": self.clock_hz,
            "output_dir": str(self.output_dir),
            "module_name": self.module_name,
            "last_verilog": None if self.last_files is None else str(self.last_files.verilog_path),
            "last_manifest": None if self.last_files is None else str(self.last_files.manifest_path),
            "prepared": self.prepared_sequence is not None,
        }

    def close(self) -> None:
        pass


def sequence_from_payload(payload) -> PulseSequence:
    """Accept a local ``PulseSequence`` or its JSON/RPyC-safe dict payload."""

    if isinstance(payload, PulseSequence):
        return payload
    if isinstance(payload, (str, bytes)):
        return PulseSequence.from_dict(json.loads(payload))
    if isinstance(payload, Mapping):
        return PulseSequence.from_dict(dict(payload))
    if hasattr(payload, "items"):
        return PulseSequence.from_dict(_plain_rpc_payload(payload))
    raise TypeError("sequence payload must be a PulseSequence or PulseSequence.to_dict() mapping.")


def _plain_rpc_payload(value):
    """Recursively convert RPyC netrefs/proxies into local JSON-like objects."""

    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain_rpc_payload(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [_plain_rpc_payload(item) for item in value]
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return [_plain_rpc_payload(item) for item in iterator]


def nonnegative_float(value, name: str) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0.")
    return out


def serve_runtime_sequencer(
    service: SequencerService,
    *,
    host: str = "0.0.0.0",
    port: int = 18861,
    start: bool = True,
):
    """Expose ``SequencerService`` over RPyC on the FPGA/Vivado computer."""

    try:
        import rpyc
        from rpyc.utils.server import ThreadedServer
    except ImportError as exc:  # pragma: no cover - depends on lab install
        raise RuntimeError("serve_runtime_sequencer requires `rpyc` on the FPGA computer.") from exc

    class RPyCSequencerService(rpyc.Service):
        def exposed_prepare(self, sequence_payload):
            return json.dumps(service.prepare(sequence_payload))

        def exposed_fire(self, sequence_payload=None):
            return json.dumps(service.fire(sequence_payload))

        def exposed_wait_done(self, timeout=None):
            return service.wait_done(timeout)

        def exposed_abort(self):
            return service.abort()

        def exposed_set_safe_state(self):
            return service.set_safe_state()

        def exposed_snapshot(self):
            return service.snapshot()

    server = ThreadedServer(
        RPyCSequencerService,
        hostname=host,
        port=int(port),
        protocol_config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": None},
    )
    if start:
        server.start()
    return server


__all__ = [
    "ManualSequencer",
    "RemoteSequencer",
    "RuntimeSequenceProgram",
    "RuntimeSequencer",
    "SequencerService",
    "VerilogSequencer",
    "compile_runtime_program",
    "serve_runtime_sequencer",
]
