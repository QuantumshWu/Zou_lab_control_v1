"""Sequencer devices and the runtime pulse-table service boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ..core.analysis import positive_int
from .base import SequencerDevice
from ..timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    PulseSequence,
    PulseTableState,
    channel_names,
    count_trigger_pulses,
    positive_float,
    sequence_for_frame_count,
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
    source_table: dict[str, Any] | None = None
    repeat_forever: bool = False
    loop_start_index: int = 0
    loop_end_tick: int = 0
    loop_count: int = 1

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema": "Zou_lab_control.neutral_atom.RuntimeSequenceProgram",
            "version": 2,
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "clock_hz": self.clock_hz,
            "channels": list(self.channels),
            "ticks": list(self.ticks),
            "masks": list(self.masks),
            "duration": self.duration,
            "trigger_count": self.trigger_count,
            "repeat_forever": bool(self.repeat_forever),
            "loop_start_index": int(self.loop_start_index),
            "loop_end_tick": int(self.loop_end_tick),
            "loop_count": int(self.loop_count),
        }
        if self.source_sequence is not None:
            payload["source_sequence"] = self.source_sequence
        if self.source_table is not None:
            payload["source_table"] = self.source_table
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
            source_table=None if payload.get("source_table") is None else dict(payload["source_table"]),
            repeat_forever=bool(payload.get("repeat_forever", False)),
            loop_start_index=int(payload.get("loop_start_index", 0)),
            loop_end_tick=int(payload.get("loop_end_tick", 0)),
            loop_count=int(payload.get("loop_count", 1)),
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
    base_sequence = sequence.without_repeat()
    ticks, masks, channels = base_sequence.edges(clock_hz=clock_hz, channels=channels)
    repeat_period = sequence.repeat_period or base_sequence.duration
    loop_end_tick = _time_to_ticks(repeat_period, clock_hz, "repeat_period") if repeat_period > 0 else (int(ticks[-1]) if ticks else 0)
    ticks, masks = _ensure_final_off_edge(ticks, masks, loop_end_tick)
    payload = {
        "sequence": sequence.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
        "repeat_count": sequence.repeat_count,
        "repeat_forever": sequence.repeat_forever,
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
        repeat_forever=bool(sequence.repeat_forever),
        loop_start_index=0,
        loop_end_tick=loop_end_tick,
        loop_count=int(sequence.repeat_count),
    )


def compile_pulse_table_runtime_program(
    state: PulseTableState,
    *,
    channels: Sequence[str] | None = None,
    clock_hz: float = 250e6,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
    x_ns: float | None = None,
    repeat_forever: bool = True,
) -> RuntimeSequenceProgram:
    """Compile GUI period-card state into an unexpanded FPGA loop program.

    ``PulseTableState`` carries the frontend repeat-bracket semantics.  The
    runtime FPGA should receive one copy of the period table plus loop metadata,
    not a fully expanded edge table.  A bracket becomes one finite inner loop;
    the whole table may still be repeated forever by the FPGA.
    """

    channels = list(channel_names(state.channels if channels is None else channels, "channels"))
    unknown_channels = [channel for channel in state.channels if channel not in channels]
    if unknown_channels:
        raise ValueError(f"pulse table channels are not in hardware channels: {unknown_channels}.")
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    x_value = state.x_ns if x_ns is None else x_ns
    state.validate(x_ns=x_value, time_step_ns=clock_step_ns)

    sequence = state.to_sequence(x_ns=x_value, time_step_ns=clock_step_ns, expand_repeat=False)
    period_starts = _pulse_table_period_starts_ticks(state, x_ns=x_value, time_step_ns=clock_step_ns)
    has_delays = _pulse_table_has_delays(state, x_ns=x_value, time_step_ns=clock_step_ns)
    if has_delays:
        _validate_pulse_table_delays_for_hardware_loop(
            state,
            period_starts=period_starts,
            x_ns=x_value,
            time_step_ns=clock_step_ns,
        )
        ticks, masks, channels = sequence.edges(clock_hz=clock_hz, channels=channels)
        base_duration_tick = _time_to_ticks(max(sequence.duration, period_starts[-1] / clock_hz), clock_hz, "pulse table duration")
        ticks, masks = _ensure_final_off_edge(ticks, masks, base_duration_tick)
    else:
        ticks, masks, channels = _pulse_table_edge_table(state, channels=channels, x_ns=x_value, time_step_ns=clock_step_ns)
    repeat_count = int(state.repeat_count)
    if state.repeat_start is None or state.repeat_end is None:
        loop_start_index = 0
        loop_end_tick = int(ticks[-1]) if has_delays else int(period_starts[-1])
        loop_count = 1
    else:
        loop_start_tick = int(period_starts[int(state.repeat_start)])
        loop_end_tick = int(period_starts[int(state.repeat_end) + 1])
        if has_delays:
            ticks, masks, loop_start_index = _insert_mask_edge_at_tick(ticks, masks, loop_start_tick)
        else:
            loop_start_index = _edge_index_at_or_after(ticks, loop_start_tick)
        loop_count = repeat_count

    effective_duration_ticks = _pulse_table_effective_duration_ticks(state, x_ns=x_value, time_step_ns=clock_step_ns)
    if has_delays and state.repeat_start is None and state.repeat_end is None:
        effective_duration_ticks = int(ticks[-1])
    payload = {
        "table": state.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
        "repeat_forever": bool(repeat_forever),
        "loop_start_index": loop_start_index,
        "loop_end_tick": loop_end_tick,
        "loop_count": loop_count,
    }
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=state.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=list(ticks),
        masks=list(masks),
        duration=effective_duration_ticks / clock_hz,
        trigger_count=_pulse_table_trigger_count(state, trigger_channels=trigger_channels),
        source_sequence=sequence.to_dict(),
        source_table=state.to_dict(),
        repeat_forever=bool(repeat_forever),
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
    )


def compile_runtime_program_for_payload(
    payload: PulseSequence | PulseTableState,
    *,
    channels: Sequence[str],
    clock_hz: float = 250e6,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> RuntimeSequenceProgram:
    """Compile either finite sequence data or GUI pulse-table data."""

    if isinstance(payload, PulseTableState):
        return compile_pulse_table_runtime_program(
            payload,
            channels=channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            repeat_forever=payload.repeat_forever,
        )
    return compile_runtime_program(payload, channels=channels, clock_hz=clock_hz, trigger_channels=trigger_channels)


def finite_frame_sequence(
    payload: PulseSequence | PulseTableState,
    frames: int,
    *,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> PulseSequence:
    """Return a finite ``PulseSequence`` with exactly ``frames`` trigger rises."""

    frames = positive_int(frames, "frames")
    trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
    if isinstance(payload, PulseTableState):
        sequence = payload.to_sequence(x_ns=payload.x_ns, time_step_ns=payload.time_step_ns, expand_repeat=False)
        base_period_s = sum(
            period.duration_steps(x_ns=payload.x_ns, time_step_ns=payload.time_step_ns) for period in payload.periods
        ) * payload.time_step_ns * 1e-9
        triggers = count_trigger_pulses(sequence, trigger_channels=trigger_channels)
        if triggers == frames:
            return sequence
        if triggers == 1 and frames > 1:
            return sequence.repeated(frames, period=base_period_s)
        raise ValueError(
            f"sequence {sequence.name!r} has {triggers} camera trigger pulses, "
            f"but acquisition requested {frames} frame(s)."
        )
    if isinstance(payload, PulseSequence):
        return sequence_for_frame_count(payload, frames, trigger_channels=trigger_channels)
    raise TypeError("frame acquisition sequence must be a PulseSequence or PulseTableState.")


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
        cache_prepared: bool = True,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.prepare_callback = prepare_callback
        self.fire_callback = fire_callback
        self.wait_done_callback = wait_done_callback
        self.safe_state_callback = safe_state_callback
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        self.cache_prepared = bool(cache_prepared)
        self._lock = threading.RLock()
        self.prepared_program: RuntimeSequenceProgram | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []

    def prepare(self, sequence_payload) -> dict[str, object]:
        timing_payload = timing_from_payload(sequence_payload)
        program = compile_runtime_program_for_payload(
            timing_payload,
            channels=self.channels,
            clock_hz=self.clock_hz,
            trigger_channels=self.trigger_channels,
        )
        with self._lock:
            cached = (
                self.cache_prepared
                and self.prepared_program is not None
                and self.prepared_program.sequence_id == program.sequence_id
            )
            if self.prepare_callback is not None and not cached:
                self.prepare_callback(program)
            self.prepared_program = program
            self.state = "prepared"
            self.history.append(
                {
                    "action": "prepare",
                    "sequence_id": program.sequence_id,
                    "triggers": program.trigger_count,
                    "cached": cached,
                }
            )
        return program.to_dict()

    def fire(self, sequence_payload=None) -> dict[str, object]:
        with self._lock:
            program = self._require_prepared()
            if sequence_payload is not None:
                requested = compile_runtime_program_for_payload(
                    timing_from_payload(sequence_payload),
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
        if program.repeat_forever and timeout is None:
            raise RuntimeError("sequencer wait_done cannot wait forever for a repeat_forever program; pass a timeout or stop the pulse.")
        if self.wait_done_callback is not None:
            ok = bool(self.wait_done_callback(program, timeout))
        elif program.repeat_forever:
            ok = False
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
            self.prepared_program = None
            self.state = "aborted"
            self.history.append({"action": "abort", "invalidated": True})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def set_safe_state(self) -> None:
        with self._lock:
            self.prepared_program = None
            self.state = "safe"
            self.history.append({"action": "safe", "invalidated": True})
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
        self.trigger_channels = self.service.trigger_channels
        self.last_program: RuntimeSequenceProgram | None = None

    def prepare(self, sequence: PulseSequence | PulseTableState) -> RuntimeSequenceProgram:
        self.last_program = RuntimeSequenceProgram.from_dict(self.service.prepare(timing_payload_to_dict(sequence)))
        return self.last_program

    def fire(self, sequence: PulseSequence | PulseTableState | None = None) -> None:
        self.service.fire(None if sequence is None else timing_payload_to_dict(sequence))

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
        self.trigger_channels = tuple(channel_names(snap.get("trigger_channels", self.trigger_channels), "trigger_channels"))
        return self

    def prepare(self, sequence: PulseSequence | PulseTableState) -> RuntimeSequenceProgram:
        self.open()
        program = self._conn.root.prepare(json.dumps(timing_payload_to_dict(sequence)))
        payload = json.loads(program) if isinstance(program, (str, bytes)) else dict(program)
        self._last_program = RuntimeSequenceProgram.from_dict(payload)
        return self._last_program

    def fire(self, sequence: PulseSequence | PulseTableState | None = None) -> None:
        self.open()
        self._conn.root.fire(None if sequence is None else json.dumps(timing_payload_to_dict(sequence)))

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


class PulseController:
    """Notebook helper that binds a pulse payload to a sequencer.

    It keeps readout scans terse: ``pulse.x = 200; pulse.on_pulse()``.  The
    controller owns no hardware; it delegates to the supplied local or remote
    ``SequencerDevice``.
    """

    def __init__(self, sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState):
        self.sequencer = sequencer
        self.pulse = pulse
        self.x = float(getattr(pulse, "x_ns", 0.0))
        self.last_program: RuntimeSequenceProgram | None = None

    @property
    def x_ns(self) -> float:
        return self.x

    @x_ns.setter
    def x_ns(self, value: float) -> None:
        self.x = float(value)

    def payload(self, *, x_ns: float | None = None, repeat_forever: bool | None = None) -> PulseSequence | PulseTableState:
        if isinstance(self.pulse, PulseTableState):
            payload = self.pulse.with_x(self.x if x_ns is None else x_ns)
            if repeat_forever is not None:
                data = payload.to_dict()
                data["repeat_forever"] = bool(repeat_forever)
                payload = PulseTableState.from_dict(data)
            return payload
        if repeat_forever is not None:
            data = self.pulse.to_dict()
            data["repeat_forever"] = bool(repeat_forever)
            return PulseSequence.from_dict(data)
        return self.pulse

    def frame_sequence(
        self,
        frames: int,
        *,
        x_ns: float | None = None,
        trigger_channels: Sequence[str] | None = None,
    ) -> PulseSequence:
        """Return a finite ``PulseSequence`` with exactly ``frames`` triggers."""

        frames = positive_int(frames, "frames")
        trigger_channels = tuple(channel_names(
            getattr(self.sequencer, "trigger_channels", DEFAULT_CAMERA_TRIGGER_CHANNELS) if trigger_channels is None else trigger_channels,
            "trigger_channels",
        ))
        payload = self.payload(x_ns=x_ns, repeat_forever=False)
        return finite_frame_sequence(payload, frames, trigger_channels=trigger_channels)

    def prepare(self, *, repeat_forever: bool | None = None) -> RuntimeSequenceProgram:
        self.last_program = self.sequencer.prepare(self.payload(repeat_forever=repeat_forever))
        return self.last_program

    def on_pulse(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        repeat_forever: bool | None = None,
    ) -> RuntimeSequenceProgram:
        payload = self.payload(repeat_forever=repeat_forever)
        if wait and timeout is None and bool(getattr(payload, "repeat_forever", False)):
            raise RuntimeError(
                "pulse.on_pulse(wait=True) cannot wait for a repeat_forever pulse without a timeout. "
                "Use pulse.on_pulse(wait=False, repeat_forever=True) for continuous scope output, "
                "or pulse.on_pulse(wait=True, repeat_forever=False) for a finite shot."
            )
        self.last_program = self.sequencer.prepare(payload)
        program = self.last_program
        self.sequencer.fire()
        if wait:
            if not self.sequencer.wait_done(timeout=timeout):
                raise TimeoutError(f"sequencer did not report done for pulse {program.sequence_name!r}.")
        return program

    def wait_done(self, timeout: float | None = None) -> bool:
        return bool(self.sequencer.wait_done(timeout=timeout))

    def stop(self) -> None:
        if hasattr(self.sequencer, "set_safe_state"):
            self.sequencer.set_safe_state()
        elif hasattr(self.sequencer, "abort"):
            self.sequencer.abort()

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe summary for notebook/debug display."""

        last = None
        if self.last_program is not None:
            last = {
                "sequence_name": self.last_program.sequence_name,
                "channels": list(self.last_program.channels),
                "edge_count": len(self.last_program.ticks),
                "trigger_count": int(self.last_program.trigger_count),
                "duration": float(self.last_program.duration),
                "repeat_forever": bool(self.last_program.repeat_forever),
                "loop_count": int(self.last_program.loop_count),
            }
        return {
            "type": type(self).__name__,
            "pulse_type": type(self.pulse).__name__,
            "x_ns": float(self.x),
            "sequencer_type": type(self.sequencer).__name__,
            "sequencer_channels": list(getattr(self.sequencer, "channels", [])),
            "clock_hz": float(getattr(self.sequencer, "clock_hz", 0.0)),
            "trigger_channels": list(getattr(self.sequencer, "trigger_channels", [])),
            "last_program": last,
        }


def bind_pulse(sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState) -> PulseController:
    """Return a ``PulseController`` for concise notebook pulse scans."""

    return PulseController(sequencer, pulse)


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


def timing_payload_to_dict(payload: PulseSequence | PulseTableState) -> dict[str, object]:
    """Return the JSON-safe timing payload for a sequence or pulse table."""

    if isinstance(payload, (PulseSequence, PulseTableState)):
        return payload.to_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("timing payload must be a PulseSequence, PulseTableState, or mapping.")


def timing_from_payload(payload) -> PulseSequence | PulseTableState:
    """Accept local timing objects or their JSON/RPyC-safe dict payload."""

    if isinstance(payload, PulseSequence):
        return payload
    if isinstance(payload, PulseTableState):
        return payload
    if isinstance(payload, (str, bytes)):
        return timing_from_payload(json.loads(payload))
    if isinstance(payload, Mapping):
        data = dict(payload)
        schema = data.get("schema", "Zou_lab_control.neutral_atom.PulseSequence")
        if schema == "Zou_lab_control.neutral_atom.PulseTableState":
            return PulseTableState.from_dict(data)
        if schema == "Zou_lab_control.neutral_atom.PulseSequence":
            return PulseSequence.from_dict(data)
        raise ValueError(f"unsupported timing payload schema {schema!r}.")
    if hasattr(payload, "items"):
        return timing_from_payload(_plain_rpc_payload(payload))
    raise TypeError("timing payload must be a PulseSequence/PulseTableState or a to_dict() mapping.")


def sequence_from_payload(payload) -> PulseSequence:
    """Accept a local ``PulseSequence`` or its JSON/RPyC-safe dict payload."""

    timing = timing_from_payload(payload)
    if not isinstance(timing, PulseSequence):
        raise TypeError("sequence payload must be a PulseSequence or PulseSequence.to_dict() mapping.")
    return timing


def _time_to_ticks(value_s: float, clock_hz: float, name: str) -> int:
    raw = float(value_s) * float(clock_hz)
    ticks = int(round(raw))
    if not math.isclose(raw, ticks, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{name}={value_s:g} s is not on the {clock_hz:g} Hz clock grid.")
    if ticks <= 0:
        raise ValueError(f"{name} must be at least one clock tick.")
    return ticks


def _ensure_final_off_edge(ticks: Sequence[int], masks: Sequence[int], final_tick: int) -> tuple[list[int], list[int]]:
    ticks = [int(tick) for tick in ticks]
    masks = [int(mask) for mask in masks]
    final_tick = int(final_tick)
    if not ticks:
        return [final_tick], [0]
    if final_tick < ticks[-1]:
        raise ValueError("repeat period is shorter than the base sequence edge table.")
    if final_tick == ticks[-1]:
        masks[-1] = 0
        return ticks, masks
    ticks.append(final_tick)
    masks.append(0)
    return ticks, masks


def _insert_mask_edge_at_tick(ticks: Sequence[int], masks: Sequence[int], tick: int) -> tuple[list[int], list[int], int]:
    """Insert a snapshot edge at ``tick`` and return its index.

    Hardware loops restart by loading ``mask_mem[loop_start_index]``.  Delayed
    pulse sequences may not naturally have an edge at the GUI repeat-bracket
    boundary, so the compiler inserts a complete state snapshot there.
    """

    out_ticks = [int(item) for item in ticks]
    out_masks = [int(item) for item in masks]
    tick = int(tick)
    current_mask = 0
    for index, candidate in enumerate(out_ticks):
        candidate = int(candidate)
        if candidate == tick:
            return out_ticks, out_masks, index
        if candidate > tick:
            out_ticks.insert(index, tick)
            out_masks.insert(index, current_mask)
            return out_ticks, out_masks, index
        current_mask = out_masks[index]
    out_ticks.append(tick)
    out_masks.append(current_mask)
    return out_ticks, out_masks, len(out_ticks) - 1


def _pulse_table_period_starts_ticks(
    state: PulseTableState,
    *,
    x_ns: float,
    time_step_ns: float,
) -> list[int]:
    starts = [0]
    for period in state.periods:
        starts.append(starts[-1] + period.duration_steps(x_ns=x_ns, time_step_ns=time_step_ns))
    return starts


def _pulse_table_effective_duration_ticks(
    state: PulseTableState,
    *,
    x_ns: float,
    time_step_ns: float,
) -> int:
    starts = _pulse_table_period_starts_ticks(state, x_ns=x_ns, time_step_ns=time_step_ns)
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        return starts[-1]
    loop_ticks = starts[int(state.repeat_end) + 1] - starts[int(state.repeat_start)]
    return starts[-1] + (int(state.repeat_count) - 1) * loop_ticks


def _pulse_table_edge_table(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    x_ns: float,
    time_step_ns: float,
) -> tuple[list[int], list[int], list[str]]:
    hardware_channels = list(channel_names(channels, "channels"))
    state_index = {channel: index for index, channel in enumerate(state.channels)}
    starts = _pulse_table_period_starts_ticks(state, x_ns=x_ns, time_step_ns=time_step_ns)
    ticks: list[int] = []
    masks: list[int] = []
    for period_index, period in enumerate(state.periods):
        mask = 0
        for bit, channel in enumerate(hardware_channels):
            source_index = state_index.get(channel)
            if source_index is not None and int(period.states[source_index]):
                mask |= 1 << bit
        ticks.append(int(starts[period_index]))
        masks.append(mask)
    ticks.append(int(starts[-1]))
    masks.append(0)
    return ticks, masks, hardware_channels


def _validate_pulse_table_delays_for_hardware_loop(
    state: PulseTableState,
    *,
    period_starts: Sequence[int],
    x_ns: float,
    time_step_ns: float,
) -> None:
    """Reject delayed edges that a compact FPGA loop cannot replay correctly."""

    table_start = 0
    table_end = int(period_starts[-1])
    has_bracket = state.repeat_start is not None and state.repeat_end is not None
    loop_start = int(period_starts[int(state.repeat_start)]) if has_bracket else table_start
    loop_end = int(period_starts[int(state.repeat_end) + 1]) if has_bracket else table_end
    delayed_spans = _pulse_table_delayed_channel_spans(
        state,
        period_starts=period_starts,
        x_ns=x_ns,
        time_step_ns=time_step_ns,
    )
    for channel, raw_start, raw_stop, delay_steps in delayed_spans:
        shifted_start = raw_start + delay_steps
        shifted_stop = raw_stop + delay_steps
        if shifted_start < table_start or shifted_stop > table_end:
            raise ValueError(
                f"delay for {channel!r} moves a pulse outside the uploaded period table; "
                "add guard/idle periods or reduce the delay before using FPGA repeat."
            )
        if not has_bracket:
            continue
        raw_inside = raw_start >= loop_start and raw_stop <= loop_end
        shifted_inside = shifted_start >= loop_start and shifted_stop <= loop_end
        shifted_intersects = shifted_start < loop_end and shifted_stop > loop_start
        if raw_inside != shifted_inside or (not raw_inside and shifted_intersects):
            raise ValueError(
                f"delay for {channel!r} moves a pulse across the repeat bracket boundary; "
                "move the bracket, add guard periods, or keep delayed edges inside the bracket."
            )


def _pulse_table_delayed_channel_spans(
    state: PulseTableState,
    *,
    period_starts: Sequence[int],
    x_ns: float,
    time_step_ns: float,
) -> list[tuple[str, int, int, int]]:
    spans: list[tuple[str, int, int, int]] = []
    for channel_index, channel in enumerate(state.channels):
        delay_steps = state.delay_steps(channel, x_ns=x_ns, time_step_ns=time_step_ns)
        if delay_steps == 0:
            continue
        active_start: int | None = None
        for period_index, period in enumerate(state.periods):
            state_value = int(period.states[channel_index])
            if state_value and active_start is None:
                active_start = int(period_starts[period_index])
            elif not state_value and active_start is not None:
                spans.append((channel, active_start, int(period_starts[period_index]), delay_steps))
                active_start = None
        if active_start is not None:
            spans.append((channel, active_start, int(period_starts[-1]), delay_steps))
    return spans


def _edge_index_at_or_after(ticks: Sequence[int], tick: int) -> int:
    for index, candidate in enumerate(ticks):
        if int(candidate) >= int(tick):
            return index
    raise ValueError(f"repeat bracket starts at tick {tick}, but no edge exists at or after that tick.")


def _pulse_table_trigger_count(
    state: PulseTableState,
    *,
    trigger_channels: Sequence[str],
) -> int:
    trigger_channels = list(channel_names(trigger_channels, "trigger_channels", allow_empty=True))
    total = 0
    for channel in trigger_channels:
        if channel in state.channels:
            total += _pulse_table_channel_rises(state, channel)
    return total


def _pulse_table_channel_rises(state: PulseTableState, channel: str) -> int:
    index = state.channel_index(channel)
    states = [int(period.states[index]) for period in state.periods]
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        count, _last = _count_rises(states, initial=0)
        return count

    repeat_start = int(state.repeat_start)
    repeat_end = int(state.repeat_end)
    repeat_count = int(state.repeat_count)
    pre = states[:repeat_start]
    loop = states[repeat_start : repeat_end + 1]
    post = states[repeat_end + 1 :]

    count, last = _count_rises(pre, initial=0)
    first_count, last_after_loop = _count_rises(loop, initial=last)
    count += first_count
    if repeat_count > 1:
        loop_again_count, last_after_loop = _count_rises(loop, initial=last_after_loop)
        count += (repeat_count - 1) * loop_again_count
    post_count, _last = _count_rises(post, initial=last_after_loop)
    return count + post_count


def _count_rises(states: Sequence[int], *, initial: int) -> tuple[int, int]:
    last = 1 if int(initial) else 0
    count = 0
    for state in states:
        state = 1 if int(state) else 0
        if state and not last:
            count += 1
        last = state
    return count, last


def _pulse_table_has_delays(
    state: PulseTableState,
    *,
    x_ns: float,
    time_step_ns: float,
) -> bool:
    return any(state.delay_steps(channel, x_ns=x_ns, time_step_ns=time_step_ns) != 0 for channel in state.channels)


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
    "PulseController",
    "RemoteSequencer",
    "RuntimeSequenceProgram",
    "RuntimeSequencer",
    "SequencerService",
    "VerilogSequencer",
    "bind_pulse",
    "compile_pulse_table_runtime_program",
    "compile_runtime_program",
    "compile_runtime_program_for_payload",
    "serve_runtime_sequencer",
]
