"""Minimal pulse timing objects used by the first neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence
import json
import math

import numpy as np

from ..core.analysis import finite_float, positive_int


CLOCK_GRID_RTOL = 1e-12
CLOCK_GRID_ATOL_TICKS = 1e-9


@dataclass(frozen=True)
class Pulse:
    channel: str
    start: float
    duration: float
    value: int = 1
    name: str = ""

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def shifted(self, dt: float) -> "Pulse":
        return replace(self, start=self.start + finite_float(dt, "dt"))

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "start": self.start,
            "duration": self.duration,
            "value": self.value,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Pulse":
        return cls(
            channel=channel_name(payload["channel"]),
            start=finite_float(payload["start"], "start"),
            duration=finite_float(payload["duration"], "duration"),
            value=digital_value(payload.get("value", 1)),
            name=str(payload.get("name", "")),
        )


class PulseSequence:
    """Physical-time pulse table with simple per-channel delay support."""

    def __init__(
        self,
        pulses: Iterable[Pulse] | None = None,
        *,
        name: str = "sequence",
        delays: dict[str, float] | None = None,
        repeat_count: int = 1,
        repeat_period: float | None = None,
        repeat_forever: bool = False,
    ):
        self.name = str(name)
        self.pulses = tuple(pulses or ())
        self.delays = {channel_name(k): finite_float(v, f"delay for {k!r}") for k, v in dict(delays or {}).items()}
        self.repeat_count = positive_int(repeat_count, "repeat_count")
        self.repeat_period = None if repeat_period is None else positive_float(repeat_period, "repeat_period")
        self.repeat_forever = bool(repeat_forever)

    def pulse(self, channel: str, start: float, duration: float, *, value: int = 1, name: str = "") -> "PulseSequence":
        channel = channel_name(channel)
        start = finite_float(start, "start")
        duration = finite_float(duration, "duration")
        if start < 0:
            raise ValueError("start must be >= 0.")
        if duration <= 0:
            raise ValueError("duration must be > 0.")
        pulse = Pulse(channel, start, duration, digital_value(value), str(name))
        return PulseSequence(
            (*self.pulses, pulse),
            name=self.name,
            delays=self.delays,
            repeat_count=self.repeat_count,
            repeat_period=self.repeat_period,
            repeat_forever=self.repeat_forever,
        )

    def on(self, channel: str, start: float, stop: float, *, value: int = 1, name: str = "") -> "PulseSequence":
        start = finite_float(start, "start")
        stop = finite_float(stop, "stop")
        return self.pulse(channel, start, stop - start, value=value, name=name)

    def delay(self, channel: str, dt: float) -> "PulseSequence":
        channel = channel_name(channel)
        dt = finite_float(dt, "dt")
        delays = dict(self.delays)
        delays[channel] = delays.get(channel, 0.0) + dt
        return PulseSequence(
            self.pulses,
            name=self.name,
            delays=delays,
            repeat_count=self.repeat_count,
            repeat_period=self.repeat_period,
            repeat_forever=self.repeat_forever,
        )

    def repeated(self, repeats: int, *, period: float | None = None) -> "PulseSequence":
        repeats = positive_int(repeats, "repeats")
        period = self.base_duration if period is None else finite_float(period, "period")
        if period <= 0:
            raise ValueError("period must be > 0.")
        if period + 1e-15 < self.base_duration:
            raise ValueError("period must be at least the base sequence duration.")
        out = PulseSequence(self.pulses, name=self.name, delays=self.delays, repeat_count=repeats, repeat_period=period)
        out.validate().raise_if_failed()
        return out

    def forever(self, *, period: float | None = None) -> "PulseSequence":
        period = self.base_duration if period is None else finite_float(period, "period")
        if period <= 0:
            raise ValueError("period must be > 0.")
        if period + 1e-15 < self.base_duration:
            raise ValueError("period must be at least the base sequence duration.")
        out = PulseSequence(self.pulses, name=self.name, delays=self.delays, repeat_count=1, repeat_period=period, repeat_forever=True)
        out.validate().raise_if_failed()
        return out

    @property
    def channels(self) -> list[str]:
        return sorted({pulse.channel for pulse in self.pulses})

    @property
    def duration(self) -> float:
        base = self.base_duration
        if self.repeat_forever:
            return base
        if self.repeat_count > 1:
            return self.repeat_count * (self.repeat_period or base)
        return base

    @property
    def base_duration(self) -> float:
        if not self.pulses:
            return 0.0
        return max(pulse.stop + self.delays.get(pulse.channel, 0.0) for pulse in self.pulses)

    def base_pulses(self) -> tuple[Pulse, ...]:
        """Return one unexpanded copy of the sequence with delays applied."""

        return tuple(replace(pulse, start=pulse.start + self.delays.get(pulse.channel, 0.0)) for pulse in self.pulses)

    def effective_pulses(self) -> tuple[Pulse, ...]:
        base = self.base_pulses()
        if self.repeat_forever or self.repeat_count == 1:
            return base
        period = self.repeat_period or self.base_duration
        return tuple(pulse.shifted(repeat * period) for repeat in range(self.repeat_count) for pulse in base)

    def without_repeat(self) -> "PulseSequence":
        return PulseSequence(self.pulses, name=self.name, delays=self.delays)

    def validate(self, *, clock_hz: float | None = None, channels: Sequence[str] | None = None) -> "PulseReport":
        errors: list[str] = []
        warnings: list[str] = []
        clock = None
        if clock_hz is not None:
            try:
                clock = positive_float(clock_hz, "clock_hz")
            except ValueError as exc:
                errors.append(str(exc))

        allowed = None if channels is None else set(channel_names(channels, "channels", allow_empty=True))
        by_channel: dict[str, list[tuple[float, float, int]]] = {}
        raw_channels = {pulse.channel for pulse in self.pulses}
        for channel in self.delays:
            if channel not in raw_channels:
                warnings.append(f"delay for {channel!r} is unused.")
        base_duration = self.base_duration
        if self.repeat_forever or self.repeat_count > 1:
            period = self.repeat_period or base_duration
            if period <= 0:
                errors.append("repeat period must be > 0.")
            elif period + 1e-15 < base_duration:
                errors.append("repeat period must be at least the base sequence duration.")
            if clock is not None and period > 0 and _time_to_clock_tick(period, clock) is None:
                errors.append(f"repeat period={period:g} s is not on the {clock:g} Hz clock grid.")

        for index, pulse in enumerate(self.base_pulses()):
            if allowed is not None and pulse.channel not in allowed:
                errors.append(f"channel {pulse.channel!r} is not in the sequencer channel list.")
            if pulse.start < 0:
                errors.append(f"pulse {pulse.channel!r} starts before t=0 after delay.")
            if pulse.duration <= 0:
                errors.append(f"pulse {pulse.channel!r} has non-positive duration.")
            if clock is not None:
                start_tick = _time_to_clock_tick(pulse.start, clock)
                stop_tick = _time_to_clock_tick(pulse.stop, clock)
                if start_tick is None:
                    errors.append(f"pulse {pulse.channel!r} start={pulse.start:g} s is not on the {clock:g} Hz clock grid.")
                    start_tick = int(round(pulse.start * clock))
                if stop_tick is None:
                    errors.append(f"pulse {pulse.channel!r} stop={pulse.stop:g} s is not on the {clock:g} Hz clock grid.")
                    stop_tick = int(round(pulse.stop * clock))
                if stop_tick <= start_tick:
                    errors.append(f"pulse {pulse.channel!r} is shorter than one clock tick at {clock:g} Hz.")
            by_channel.setdefault(pulse.channel, []).append((pulse.start, pulse.stop, index))

        for channel, intervals in by_channel.items():
            intervals.sort()
            active_stop = -np.inf
            active_index = -1
            for start, stop, index in intervals:
                if start < active_stop - 1e-15:
                    errors.append(f"channel {channel!r} has overlapping pulses near events {active_index} and {index}.")
                if stop > active_stop:
                    active_stop = stop
                    active_index = index
        return PulseReport(not errors, self.name, len(self.pulses), clock, tuple(errors), tuple(warnings))

    def edges(self, *, clock_hz: float = 250e6, channels: Sequence[str] | None = None) -> tuple[list[int], list[int], list[str]]:
        channels = list(self.channels if channels is None else channel_names(channels, "channels", allow_empty=True))
        if not channels:
            raise ValueError("channels must contain at least one channel.")
        self.validate(clock_hz=clock_hz, channels=channels).raise_if_failed()
        edges: dict[int, dict[str, int]] = {}
        for pulse in self.effective_pulses():
            start = int(round(pulse.start * clock_hz))
            stop = int(round(pulse.stop * clock_hz))
            edges.setdefault(start, {})[pulse.channel] = pulse.value
            edges.setdefault(stop, {})[pulse.channel] = 0
        state = {channel: 0 for channel in channels}
        ticks: list[int] = []
        masks: list[int] = []
        for tick in sorted(edges):
            for channel, value in sorted(edges[tick].items()):
                if channel in state:
                    state[channel] = value
            ticks.append(int(tick))
            masks.append(state_to_mask(state, channels))
        return ticks, masks, channels

    def table(self) -> list[dict[str, object]]:
        return [pulse.to_dict() for pulse in sorted(self.effective_pulses(), key=lambda item: (item.start, item.channel))]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "Zou_lab_control.neutral_atom.PulseSequence",
            "version": 2,
            "name": self.name,
            "delays": dict(self.delays),
            "repeat_count": self.repeat_count,
            "repeat_period": self.repeat_period,
            "repeat_forever": self.repeat_forever,
            "pulses": [pulse.to_dict() for pulse in self.pulses],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PulseSequence":
        if payload.get("schema", "Zou_lab_control.neutral_atom.PulseSequence") != "Zou_lab_control.neutral_atom.PulseSequence":
            raise ValueError("unsupported PulseSequence schema.")
        return cls(
            [Pulse.from_dict(item) for item in payload.get("pulses", [])],
            name=str(payload.get("name", "sequence")),
            delays=dict(payload.get("delays", {})),
            repeat_count=int(payload.get("repeat_count", 1)),
            repeat_period=None if payload.get("repeat_period") is None else finite_float(payload.get("repeat_period"), "repeat_period"),
            repeat_forever=bool(payload.get("repeat_forever", False)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class PulseReport:
    ok: bool
    sequence_name: str
    pulse_count: int
    clock_hz: float | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Pulse sequence validation failed: " + "; ".join(self.errors))


def imaging_sequence(
    *,
    exposure: float = 20e-3,
    trigger_width: float = 20e-6,
    pre_trigger: float = 100e-6,
    load: bool = False,
    cooling: float = 2e-3,
    name: str = "imaging",
) -> PulseSequence:
    """Build the minimal load/probe/qCMOS-trigger sequence used in notebooks."""

    exposure = positive_float(exposure, "exposure")
    trigger_width = positive_float(trigger_width, "trigger_width")
    pre_trigger = finite_float(pre_trigger, "pre_trigger")
    if pre_trigger < 0:
        raise ValueError("pre_trigger must be >= 0.")
    cooling = finite_float(cooling, "cooling")
    if cooling < 0:
        raise ValueError("cooling must be >= 0.")

    offset = cooling + pre_trigger if load else pre_trigger
    total = offset + exposure + trigger_width
    seq = PulseSequence(name=name).pulse("trap", 0.0, total, name="trap_hold")
    if load and cooling > 0:
        seq = seq.pulse("cooling", 0.0, cooling, name="load")
    seq = seq.pulse("probe", offset, exposure, name="probe")
    seq = seq.pulse("qcm_trigger", offset, trigger_width, name="camera_trigger")
    return seq


DEFAULT_CAMERA_TRIGGER_CHANNELS = ("qcm_trigger", "camera_trigger", "trig")


def count_trigger_pulses(sequence: PulseSequence, *, trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS) -> int:
    """Count rising camera-trigger pulses in a sequence."""

    channels = set(channel_names(trigger_channels, "trigger_channels"))
    base_count = sum(1 for pulse in sequence.base_pulses() if pulse.value and pulse.channel in channels)
    if sequence.repeat_forever:
        return base_count
    return base_count * int(sequence.repeat_count)


def sequence_for_frame_count(
    sequence: PulseSequence,
    frames: int,
    *,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> PulseSequence:
    """Return a sequence whose camera-trigger count matches ``frames``.

    Single-shot imaging sequences are common in notebooks.  Real qCMOS
    multi-frame acquisitions still need one hardware trigger per frame, so a
    one-trigger sequence is repeated automatically.  Ambiguous sequences fail
    early instead of timing out after the camera is armed.
    """

    frames = positive_int(frames, "frames")
    triggers = count_trigger_pulses(sequence, trigger_channels=trigger_channels)
    if triggers == frames:
        return sequence
    if triggers == 1 and frames > 1:
        return sequence.repeated(frames)
    raise ValueError(
        f"sequence {sequence.name!r} has {triggers} camera trigger pulses, "
        f"but acquisition requested {frames} frame(s)."
    )


def exposure_from_sequence(sequence: PulseSequence | None, *, default: float, channel: str = "probe") -> float:
    """Infer camera exposure from uniform probe pulses in a sequence.

    Real camera adapters use this to keep DCAM exposure time synchronized with
    the pulse sequence used to illuminate atoms.  Repeated multi-frame
    sequences are allowed, but all probe pulses must have the same duration.
    """

    default = positive_float(default, "default exposure")
    if sequence is None:
        return default
    channel = channel_name(channel)
    durations = [int(round(pulse.duration * 1e15)) for pulse in sequence.base_pulses() if pulse.channel == channel and pulse.value]
    if not durations:
        return default
    unique = sorted(set(durations))
    if len(unique) != 1:
        raise ValueError(f"sequence {sequence.name!r} has non-uniform {channel!r} pulse durations.")
    return positive_float(unique[0] / 1e15, f"{channel} exposure")


def plot_sequence(sequence: PulseSequence, *, clock_hz: float = 250e6, display: bool = True):
    """Plot a pulse timeline with ``Zou_lab_control.frontend``."""

    from Zou_lab_control import frontend as zf

    channels = sequence.edges(clock_hz=clock_hz)[2]
    return zf.plot(sequence, kind="pulse", channels=channels, labels=("Time (s)", "Pulse", "State"), title=sequence.name, display=display)


def state_to_mask(state: dict[str, int], channels: Sequence[str]) -> int:
    mask = 0
    for index, channel in enumerate(channels):
        if int(state.get(channel, 0)):
            mask |= 1 << index
    return mask


def channel_name(channel) -> str:
    if isinstance(channel, (bool, np.bool_)):
        raise ValueError("channel must be a name, not a boolean.")
    out = str(channel)
    if not out:
        raise ValueError("channel must not be empty.")
    return out


def channel_names(channels, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(channels, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of channel names, not one string.")
    try:
        out = tuple(channel_name(channel) for channel in channels)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of channel names.") from exc
    if not out and not allow_empty:
        raise ValueError(f"{name} must contain at least one channel.")
    return out


def digital_value(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("digital value must be 0 or 1, not a boolean.")
    out = finite_float(value, "digital value")
    if int(out) != out or int(out) not in (0, 1):
        raise ValueError("digital value must be 0 or 1.")
    return int(out)


def positive_float(value, name: str) -> float:
    out = finite_float(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be > 0.")
    return out


def _time_to_clock_tick(time_s: float, clock_hz: float) -> int | None:
    raw_tick = time_s * clock_hz
    tick = int(round(raw_tick))
    if not math.isclose(raw_tick, tick, rel_tol=CLOCK_GRID_RTOL, abs_tol=CLOCK_GRID_ATOL_TICKS):
        return None
    return tick


__all__ = [
    "Pulse",
    "PulseReport",
    "PulseSequence",
    "DEFAULT_CAMERA_TRIGGER_CHANNELS",
    "count_trigger_pulses",
    "exposure_from_sequence",
    "imaging_sequence",
    "plot_sequence",
    "sequence_for_frame_count",
]
