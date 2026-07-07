"""Minimal pulse timing objects used by the first neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence
import json
import math

import numpy as np

from Zou_lab_control._clock import DEFAULT_CLOCK_HZ  # absolute: dependency-free clock seam (config single source)
from ..core.analysis import finite_float, nonnegative_float, positive_float, positive_int


CLOCK_GRID_RTOL = 1e-12
CLOCK_GRID_ATOL_TICKS = 1e-9

# Floating-point slack when comparing a repeat period against the base sequence
# duration: period >= base_duration is the physical requirement, but a period set
# exactly equal to base_duration can land a few ULP below it after arithmetic.
_PERIOD_DURATION_SLACK = 1e-15


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

    # The JSON payload identity, single-sourced as class attributes (the same shape the
    # sister ``PulseTableState`` carries): every writer (``to_dict``), reader
    # (``from_dict``) and payload dispatcher (subsystems / devices) references THESE --
    # the schema string is an on-disk persistence contract, never a retyped literal.
    schema = "Zou_lab_control.neutral_atom.PulseSequence"
    version = 2

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
        out = PulseSequence(self.pulses, name=self.name, delays=self.delays, repeat_count=repeats, repeat_period=period)
        out.validate().raise_if_failed()
        return out

    def forever(self, *, period: float | None = None) -> "PulseSequence":
        period = self.base_duration if period is None else finite_float(period, "period")
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
            elif period + _PERIOD_DURATION_SLACK < base_duration:
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

    def edges(self, *, clock_hz: float = DEFAULT_CLOCK_HZ, channels: Sequence[str] | None = None) -> tuple[list[int], list[int], list[str]]:
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
            "schema": self.schema,
            "version": self.version,
            "name": self.name,
            "delays": dict(self.delays),
            "repeat_count": self.repeat_count,
            "repeat_period": self.repeat_period,
            "repeat_forever": self.repeat_forever,
            "pulses": [pulse.to_dict() for pulse in self.pulses],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PulseSequence":
        if payload.get("schema", cls.schema) != cls.schema:
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


# The trap-held gap between consecutive emCCD frames of a long-short-long reference bracket.
# The camera trigger drops during this gap so the next exposure registers as a DISTINCT trigger.
# Single source for the gap: ``reference_bracket_sequence`` builds a bracket from scratch, and the
# shipped imaging template (``default_imaging_template`` / ``pulses/imaging_template.json``) puts a
# trap-held gap of this length between its image frames.
READOUT_GAP_SECONDS = 100e-6

#: The default readout exposure -- the qCMOS ~20 ms working point where single-shot bright/dark counts
#: fully separate.  THE single source for every "default exposure": the camera config defaults (qCMOS /
#: virtual), the session's fallback when no camera exposure is known, the imaging-sequence + reference-
#: bracket LONG exposure, and the virtual fake-data reference frame -- so moving the working point moves
#: them all together (the short readout / calibration exposures are distinct values, left as-is).
DEFAULT_EXPOSURE_S = 20e-3


def imaging_sequence(
    *,
    exposure: float = DEFAULT_EXPOSURE_S,
    trigger_width: float = 20e-6,
    pre_trigger: float = 100e-6,
    load: bool = False,
    cooling: float = 2e-3,
    name: str = "imaging",
    trap_channel: str = "trap",
    cooling_channel: str = "cooling",
    probe_channel: str = "probe",
    trigger_channel: str = "emCCD",
) -> PulseSequence:
    """Build the minimal load/probe/camera-trigger sequence used in notebooks."""

    exposure = positive_float(exposure, "exposure")
    trigger_width = positive_float(trigger_width, "trigger_width")
    pre_trigger = nonnegative_float(pre_trigger, "pre_trigger")
    cooling = nonnegative_float(cooling, "cooling")

    offset = cooling + pre_trigger if load else pre_trigger
    total = offset + exposure + trigger_width
    trap_channel = channel_name(trap_channel)
    cooling_channel = channel_name(cooling_channel)
    probe_channel = channel_name(probe_channel)
    trigger_channel = channel_name(trigger_channel)

    seq = PulseSequence(name=name).pulse(trap_channel, 0.0, total, name="trap_hold")
    if load and cooling > 0:
        seq = seq.pulse(cooling_channel, 0.0, cooling, name="load")
    seq = seq.pulse(probe_channel, offset, exposure, name="probe")
    seq = seq.pulse(trigger_channel, offset, trigger_width, name="camera_trigger")
    return seq


def reference_bracket_sequence(
    *,
    ref_exposure: float = DEFAULT_EXPOSURE_S,
    readout_exposure: float = 5e-3,
    n_ref: int = 2,
    trigger_width: float = 20e-6,
    pre_trigger: float = 100e-6,
    gap: float = READOUT_GAP_SECONDS,
    cooling: float = 2e-3,
    name: str = "reference_bracket",
    trap_channel: str = "trap",
    cooling_channel: str = "cooling",
    probe_channel: str = "probe",
    trigger_channel: str = "emCCD",
) -> PulseSequence:
    """The Rb87 readout-fidelity bracket: ONE cooling load, then ``n_ref`` LONG reference
    images interleaved with ONE short readout (e.g. long-short-long), all imaging the SAME
    atoms (the trap is held on and there is NO re-cooling between triggers, so a following
    frame sees the previous frame's survivors).  Comparing the long reference frames tells
    whether the atom survived; when they agree they vote the GROUND-TRUTH occupancy that the
    middle readout frame is scored against.  The readout is placed in the MIDDLE (index
    ``n_ref // 2``); the others are references.  ``n_ref + 1`` camera triggers total."""

    ref_exposure = positive_float(ref_exposure, "ref_exposure")
    readout_exposure = positive_float(readout_exposure, "readout_exposure")
    n_ref = positive_int(n_ref, "n_ref")
    trigger_width = positive_float(trigger_width, "trigger_width")
    # Coerce AND assign back (#C4): the old loop validated the values but threw the coerced
    # floats away, so a string-like input that its sibling imaging_sequence accepts reached
    # the cursor arithmetic below as a str and crashed with a bare TypeError.
    pre_trigger = nonnegative_float(pre_trigger, "pre_trigger")
    gap = nonnegative_float(gap, "gap")
    cooling = nonnegative_float(cooling, "cooling")
    trap_channel = channel_name(trap_channel)
    cooling_channel = channel_name(cooling_channel)
    probe_channel = channel_name(probe_channel)
    trigger_channel = channel_name(trigger_channel)

    readout_index = n_ref // 2
    exposures = [readout_exposure if i == readout_index else ref_exposure for i in range(n_ref + 1)]

    cursor = cooling + pre_trigger
    starts: list[float] = []
    for exposure in exposures:
        starts.append(cursor)
        cursor += exposure + gap
    total = cursor
    seq = PulseSequence(name=name).pulse(trap_channel, 0.0, total, name="trap_hold")
    if cooling > 0:
        seq = seq.pulse(cooling_channel, 0.0, cooling, name="load")
    for i, (start, exposure) in enumerate(zip(starts, exposures)):
        seq = seq.pulse(probe_channel, start, exposure, name=f"probe_{i}")
        seq = seq.pulse(trigger_channel, start, trigger_width, name=f"camera_trigger_{i}")
    return seq


def imaging_channel_kwargs(sequencer: object, *, trigger_channel: str | None = None) -> dict[str, str]:
    """Channel kwargs for :func:`imaging_sequence`, derived from a bound sequencer.

    Single source of truth shared by the session and the loading readout: a real
    streamer config names channels ``ch00..chNN`` (the imaging sequence must
    target THOSE, not the ``trap``/``cooling``/``probe``/``emCCD`` placeholder
    names, or every pulse references a channel the device does not have).  Maps
    the conventional roles onto whatever the sequencer actually exposes; returns
    ``{}`` when neither convention is present so the caller falls back to
    :func:`imaging_sequence`'s own placeholder defaults (virtual / notebook use).

    ``trigger_channel`` is the CAMERA's capture-trigger channel: the camera device owns
    "which line triggers me" (the sequencer does NOT), so a caller building an imaging
    pulse for a real chNN streamer passes its camera's ``capture_trigger_channels[0]`` --
    that selects which chNN the imaging pulse pulses to trigger the camera.  The placeholder
    (virtual) convention names the trigger ``emCCD`` directly, so the virtual path needs no
    explicit trigger channel."""

    channels = list(getattr(sequencer, "channels", ()) or ())
    if all(ch in channels for ch in ("ch00", "ch03", "ch09")) and trigger_channel and trigger_channel in channels:
        return {
            "trap_channel": "ch09",
            "cooling_channel": "ch00",
            "probe_channel": "ch03",
            "trigger_channel": trigger_channel,
        }
    if all(ch in channels for ch in ("trap", "cooling", "probe", "emCCD")):
        return {
            "trap_channel": "trap",
            "cooling_channel": "cooling",
            "probe_channel": "probe",
            "trigger_channel": "emCCD",
        }
    return {}


def probe_channel_set(channel: str) -> list[str]:
    """The set of pulse-channel names that count as the probe for ``channel`` -- the SINGLE
    source of the probe->channel alias.  When the caller hands the placeholder name ``probe``
    (no chNN config resolved it yet) we ALSO match the legacy ``ch02`` it historically mapped to,
    so an exposure inferred from a placeholder-named sequence still finds its probe pulses.  A
    resolved chNN (e.g. ``ch03``) matches only itself.  Every site that expands a probe channel
    (exposure inference, the qCMOS + virtual camera bracket exposure) routes through HERE so the
    alias is defined once and cannot drift (#5.15)."""
    channel = channel_name(channel)
    return [channel, "ch02"] if channel == "probe" else [channel]


def exposure_from_sequence(sequence: PulseSequence | None, *, default: float, channel: str = "probe") -> float:
    """Infer camera exposure from uniform probe pulses in a sequence.

    Real camera adapters use this to keep DCAM exposure time synchronized with
    the pulse sequence used to illuminate atoms.  Repeated multi-frame
    sequences are allowed, but all probe pulses must have the same duration.
    """

    default = positive_float(default, "default exposure")
    if sequence is None:
        return default
    if not hasattr(sequence, "base_pulses"):
        return default
    channel = channel_name(channel)
    probe_candidates = probe_channel_set(channel)
    durations = [
        int(round(pulse.duration * 1e15))
        for pulse in sequence.base_pulses()
        if pulse.channel in probe_candidates and pulse.value
    ]
    if not durations:
        return default
    unique = sorted(set(durations))
    if len(unique) != 1:
        raise ValueError(f"sequence {sequence.name!r} has non-uniform {channel!r} pulse durations.")
    return positive_float(unique[0] / 1e15, f"{channel} exposure")


def plot_sequence(sequence: PulseSequence, *, clock_hz: float = DEFAULT_CLOCK_HZ, display: bool = True):
    """Plot a pulse timeline through the registered viewer (``None`` if headless)."""

    from Zou_lab_control._viewer_registry import active_plotter

    plotter = active_plotter()
    if plotter is None:
        return None
    channels = sequence.edges(clock_hz=clock_hz)[2]
    return plotter.plot(sequence, kind="pulse", channels=channels, labels=("Time (s)", "", "State"), title=sequence.name, display=display)


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


def _time_to_clock_tick(time_s: float, clock_hz: float) -> int | None:
    raw_tick = time_s * clock_hz
    tick = int(round(raw_tick))
    if not math.isclose(raw_tick, tick, rel_tol=CLOCK_GRID_RTOL, abs_tol=CLOCK_GRID_ATOL_TICKS):
        return None
    return tick


def round_ticks(raw_ticks: float) -> int:
    """Round a tick count to the nearest integer, ties AWAY FROM ZERO -- the ONE tie
    rule for putting a continuous time onto the hardware clock grid (2.5 ticks -> 3,
    -2.5 -> -3; the confocal ``align_to_resolution`` semantics MAINTAINER_NOTES §4
    documents).  Both the seconds-domain snap (:func:`snap_seconds_to_clock`) and the
    ns-domain quantizer (``pulse_table.quantized_time_steps``) delegate here, so the
    SAME user duration lands on the SAME tick whether it enters through a notebook
    scan axis or the GUI / scan-table path.  (Python's built-in ``round`` is
    ties-to-even -- 2.5 -> 2 -- which is why this primitive exists: two independent
    spellings of the tie rule once sent 50 ns to 40 ns on one path and 60 ns on the
    other.)  Callers keep their own CLAMP policy (min-1-tick vs allow-zero) -- this
    helper only owns the tie rounding."""
    value = float(raw_ticks)
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def snap_seconds_to_clock(time_s: float, clock_hz: float, *, min_ticks: int = 1) -> float:
    """Snap a continuous time (seconds) to the nearest whole clock tick.

    The hardware can only land on integer ticks (``validate`` rejects off-grid times,
    ``edges`` rounds to them), so any CONTINUOUS user duration -- a linspace sweep point
    (readout-duration -> fidelity), a release-recapture hold, a typed exposure -- is
    quantized HERE, the single seconds-domain snap source.  Tie rounding is the shared
    :func:`round_ticks` rule (ties away from zero -- identical to the ns-domain
    ``quantized_time_steps`` path).  ``min_ticks`` (>=1) clamps a sub-tick request up
    to a realizable duration rather than to zero."""
    clock = positive_float(clock_hz, "clock_hz")
    tick = round_ticks(finite_float(time_s, "time_s") * clock)
    return max(int(min_ticks), tick) / clock


__all__ = [
    "Pulse",
    "PulseReport",
    "PulseSequence",
    "READOUT_GAP_SECONDS",
    "exposure_from_sequence",
    "imaging_channel_kwargs",
    "imaging_sequence",
    "plot_sequence",
    "probe_channel_set",
    "reference_bracket_sequence",
    "round_ticks",
    "snap_seconds_to_clock",
]


def decode_analog_bus(sequence: "PulseSequence", members, at_time: float) -> int:
    """Reconstruct a DAC bus's SIGNED level at ``at_time`` from the compiled bit-channel pulses --
    the INVERSE of the offset-binary encoding ``apply_analog_bus_modes_to_period_states`` bakes into
    the period states (signed value + 2^(n-1) -> per-bit digital pulses, ``members`` ordered
    LSB..MSB).  The bits are read off the DELAYED base-cycle timeline (``base_pulses()`` -- the same
    vocabulary the camera-trigger/exposure helpers use), because that is what the hardware actually
    PLAYS: a ``.delay()`` on the bit channels shifts the driven window, and ``edges()`` streams the
    shifted pulses.  Reading the raw ``sequence.pulses`` list here decoded the PRE-delay level -- a
    level the fired sequence never drives at that time.  A sense time falls inside the base cycle,
    so repeats need no expansion.  The virtual backend's analog-aware devices (e.g. the MOT monitor
    camera reading the coil DACs) use this to SENSE what the fired sequence actually drives -- never
    the task's own set-points -- so virtual sensing goes through the same compiled artefact the
    hardware plays.  Contract tests pin this against ``set_api`` (encoder and decoder cannot drift)
    and against the ``edges()`` timeline under a delayed bus (decoder and streamer cannot drift).

    An UNDRIVEN bus -- no pulse at all on ANY member channel (the encoder's "unused" marker:
    ``apply_analog_bus_modes_to_period_states`` never projects bits for an untouched bus) -- decodes
    to the SAFE level ``BUS_SAFE_SIGNED_LEVEL`` (signed 0 = the hardware's mid-scale ``BUS_SAFE_VALUE``
    idle, 0 V), NOT the all-bits-low word (offset-binary code 0 = negative full scale).  Every other
    layer already speaks this convention (RTL reset/idle, engine_model rest, the encoder's unused
    marker); before this branch a TTL-only fire made the virtual MOT camera sense every coil at
    -2^(B-1) and the spot vanished.

    A DRIVEN bus sampled BEFORE its delayed window decodes to the SAME safe level: the hardware's
    per-DA-bit delay scheduler idles each bit at its ``SAFE_BIT`` -- the bits of the mid-scale
    ``BUS_SAFE_VALUE`` code -- until the first scheduled change emerges at ``t == d``, i.e.
    ``out[t] = in[t-d]`` holding safe before the window (``engine_model.bus_delay_line_reference``
    and its RTL register mirror; ``zlc_edge_streamer.v`` ``g_busdly``).  The window start needs no
    ``bus_delays`` plumbing: a touched bus projects bits for EVERY period (leading holds carry the
    mid code, whose MSB is high), so the earliest member pulse on the delayed timeline IS the
    delay-window start.

    Known un-decodable corners (the projection erases full-negative-code periods -- no bits
    anywhere): a bus driven to the full-negative code EVERYWHERE, or over its LEADING periods, is
    bit-identical to "never driven" / "not yet driven" and unifies to the safe level; a
    full-negative period INSIDE the driven window still decodes exactly."""
    members = [str(m) for m in members]
    if not members:
        raise ValueError("decode_analog_bus needs the bus's member bit channels (LSB..MSB).")
    t = float(at_time)
    bit_of = {channel: bit for bit, channel in enumerate(members)}
    word = 0
    driven_start = None    # earliest member pulse on the delayed timeline = the delay-window start
    for p in sequence.base_pulses():
        bit = bit_of.get(p.channel)
        if bit is None:
            continue
        if driven_start is None or p.start < driven_start:
            driven_start = p.start
        if p.start <= t < p.start + p.duration and p.value:
            word |= 1 << bit
    if driven_start is None or t < driven_start:
        # Lazy import: pulse_table imports from this module at load time (the constant lives
        # beside its wire-layer siblings bus_zero_code/bus_signed_range -- the one safe-state source).
        from .pulse_table import BUS_SAFE_SIGNED_LEVEL
        return BUS_SAFE_SIGNED_LEVEL
    # code -> signed = code - zero_code (the ONE mid-scale helper, not the re-typed 1<<(B-1));
    # lazy import for the same load-time cycle reason as BUS_SAFE_SIGNED_LEVEL above.
    from .pulse_table import bus_zero_code
    return word - bus_zero_code(len(members))
