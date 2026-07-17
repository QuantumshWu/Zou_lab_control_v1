"""Pulse-owned digital playback projection for virtual hardware adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .artifact import CompiledPulseArtifact
from .ir import evaluate_affine_tick
from .physical import (
    iter_physical_digital_high_intervals,
    physical_digital_playback_terminal_tick,
)


MAX_MATERIALIZED_PLAYBACK_PULSES = 1_000_000


@dataclass(frozen=True)
class PlaybackPulse:
    channel: str
    start: float
    duration: float
    value: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("playback pulse channel must be non-empty text")
        for field in ("start", "duration"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"playback pulse {field} must be finite")
        if self.start < 0 or self.duration <= 0:
            raise ValueError("playback pulse must have non-negative start and positive duration")
        if self.value not in (0, 1):
            raise ValueError("playback pulse value must be binary")

    @property
    def stop(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class PlaybackTriggerGroup:
    """Compiled trigger cardinality for one pulse point/loop iteration."""

    point_index: int
    loop_iteration: int
    channel_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for field in ("point_index", "loop_iteration"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        counts = tuple(self.channel_counts)
        channels = tuple(channel for channel, _count in counts)
        if len(channels) != len(set(channels)):
            raise ValueError("playback trigger-group channels must be unique")
        for channel, count in counts:
            if not isinstance(channel, str) or not channel:
                raise ValueError("playback trigger-group channel must be non-empty text")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("playback trigger-group counts must be positive integers")
        object.__setattr__(self, "channel_counts", counts)


@dataclass(frozen=True)
class PulsePlayback:
    name: str
    pulses: tuple[PlaybackPulse, ...]
    logical_duration: float
    duration: float
    repeat_forever: bool
    repeat_count: int = 1
    trigger_channels: tuple[str, ...] = ()
    full_point_loop_channels: tuple[str, ...] = ()
    trigger_groups: tuple[PlaybackTriggerGroup, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("playback name must be non-empty text")
        pulses = tuple(self.pulses)
        if any(not isinstance(pulse, PlaybackPulse) for pulse in pulses):
            raise TypeError("playback pulses must contain PlaybackPulse values")
        object.__setattr__(self, "pulses", pulses)
        for field in ("logical_duration", "duration"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"playback {field} must be finite and positive")
        if self.duration < self.logical_duration:
            raise ValueError("physical playback duration cannot precede logical terminal")
        if type(self.repeat_forever) is not bool:
            raise TypeError("repeat_forever must be bool")
        if isinstance(self.repeat_count, bool) or not isinstance(self.repeat_count, int) or self.repeat_count < 1:
            raise ValueError("repeat_count must be a positive integer")
        if any(pulse.stop > self.duration + 1e-12 for pulse in pulses):
            raise ValueError("playback pulse exceeds playback duration")
        trigger_channels = tuple(self.trigger_channels)
        if any(not isinstance(channel, str) or not channel for channel in trigger_channels):
            raise ValueError("playback trigger channels must be non-empty text")
        if len(trigger_channels) != len(set(trigger_channels)):
            raise ValueError("playback trigger channels must be unique")
        object.__setattr__(self, "trigger_channels", trigger_channels)
        full_point = tuple(self.full_point_loop_channels)
        if len(full_point) != len(set(full_point)) or any(
            channel not in trigger_channels for channel in full_point
        ):
            raise ValueError(
                "full-point loop channels must be a unique trigger-channel subset"
            )
        object.__setattr__(self, "full_point_loop_channels", full_point)
        groups = tuple(self.trigger_groups)
        if any(not isinstance(group, PlaybackTriggerGroup) for group in groups):
            raise TypeError("trigger_groups must contain PlaybackTriggerGroup values")
        keys = tuple((group.point_index, group.loop_iteration) for group in groups)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("playback trigger groups must use unique point/loop order")
        if any(
            channel not in trigger_channels
            for group in groups
            for channel, _count in group.channel_counts
        ):
            raise ValueError("playback trigger group contains an undeclared channel")
        object.__setattr__(self, "trigger_groups", groups)

    def base_pulses(self) -> tuple[PlaybackPulse, ...]:
        return self.pulses

    def effective_pulses(self) -> tuple[PlaybackPulse, ...]:
        """Return the finite timeline or one exact cyclic base-cycle projection."""

        return self.pulses

    @property
    def repeat_period(self) -> float:
        """Logical hardware rewind period; meaningful when ``repeat_forever``."""

        return self.logical_duration

    def trigger_group_sizes(self, channels: tuple[str, ...]) -> tuple[int, ...]:
        """Frame counts per compiled point/loop group for selected trigger lines."""

        selected = tuple(channels)
        if len(selected) != len(set(selected)):
            raise ValueError("selected trigger channels must be unique")
        if len(selected) != 1:
            raise ValueError(
                "virtual shot grouping requires exactly one trigger channel; "
                "multiple delayed lines can interleave source groups"
            )
        unknown = tuple(channel for channel in selected if channel not in self.trigger_channels)
        if unknown:
            raise ValueError(
                f"playback has no compiled trigger grouping for channels {unknown!r}"
            )
        partial = tuple(
            channel
            for channel in selected
            if channel not in self.full_point_loop_channels
        )
        if partial:
            raise ValueError(
                "virtual shot grouping requires a full-point source loop; "
                f"partial-loop trigger channels are {partial!r}"
            )
        wanted = set(selected)
        return tuple(
            count
            for group in self.trigger_groups
            if (
                count := sum(
                    value
                    for channel, value in group.channel_counts
                    if channel in wanted
                )
            )
        )


def sample_compiled_bus_codes(
    artifact: CompiledPulseArtifact,
    *,
    point_index: int = 0,
    phase: float = 0.5,
) -> tuple[tuple[str, int], ...]:
    """Sample held DAC output codes from one compiled base-cycle/scan point.

    This is the pulse-owned virtual-hardware projection of the waveform the
    FPGA plays.  It deliberately consumes the compiled TargetIR rather than
    notebook parameters.  A sample that lands inside a live ramp is rejected:
    the compact IR does not expose a second approximate ramp evaluator, and a
    virtual sensor must not silently invent a different physical waveform.
    """

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if isinstance(point_index, bool) or not isinstance(point_index, int):
        raise TypeError("point_index must be an integer")
    if isinstance(phase, bool) or not isinstance(phase, (int, float)):
        raise TypeError("phase must be a number")
    fraction = float(phase)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("phase must be finite in [0, 1)")

    ir = artifact.target_ir
    points = ir.scan_points or ((),)
    if not 0 <= point_index < len(points):
        raise IndexError("point_index is outside the compiled scan table")
    point = points[point_index]
    terminal_tick = evaluate_affine_tick(
        ir.ticks[-1],
        ir.tick_slot_coeffs[-1],
        point,
        ir.scan_coeff_frac_bits,
    )
    sample_tick = fraction * terminal_tick
    delays = {item.bus_index: item.delay_ticks for item in ir.bus_delays}
    sampled: list[tuple[str, int]] = []
    for bus_index, bus_name in enumerate(ir.bus_names):
        current = int(ir.bus_safe_values[bus_index])
        delay = delays.get(bus_index, 0)
        segments = sorted(
            (
                item
                for item in ir.bus_segments
                if item.bus_index == bus_index
            ),
            key=lambda item: (
                evaluate_affine_tick(
                    item.start_tick,
                    item.start_tick_coeffs,
                    point,
                    ir.scan_coeff_frac_bits,
                ),
                item.stop_tick,
            ),
        )
        for segment in segments:
            source_start = evaluate_affine_tick(
                segment.start_tick,
                segment.start_tick_coeffs,
                point,
                ir.scan_coeff_frac_bits,
            )
            source_stop = evaluate_affine_tick(
                segment.stop_tick,
                segment.stop_tick_coeffs,
                point,
                ir.scan_coeff_frac_bits,
            )
            visible_start = source_start + delay + (1 if source_start else 0)
            if sample_tick < visible_start:
                continue
            stop_selector = segment.stop_value_select
            stop_value = (
                int(point[stop_selector - 1])
                if stop_selector
                else int(segment.stop_value)
            )
            if segment.mode == "edge" or source_stop <= source_start:
                current = stop_value
                continue
            visible_stop = source_stop + delay + (1 if source_stop else 0)
            if sample_tick < visible_stop:
                raise ValueError(
                    f"cannot sample DAC bus {bus_name!r} inside a live ramp"
                )
            current = stop_value
        sampled.append((bus_name, current))
    return tuple(sampled)


def build_pulse_playback(
    artifact: CompiledPulseArtifact,
    *,
    name: str = "compiled-pulse",
) -> PulsePlayback:
    """Project the exact finite/cyclic digital TargetIR waveform for simulation.

    This is a read-only adapter projection.  The compiled TargetIR remains the
    authority; the virtual camera receives the same delayed digital transitions
    that the frozen output engine would expose.
    """

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    ir = artifact.target_ir
    terminal_tick = physical_digital_playback_terminal_tick(ir)
    pulses: list[PlaybackPulse] = []
    for interval in iter_physical_digital_high_intervals(ir):
        if len(pulses) >= MAX_MATERIALIZED_PLAYBACK_PULSES:
            raise ValueError(
                "pulse playback exceeds the materialization limit of "
                f"{MAX_MATERIALIZED_PLAYBACK_PULSES} high intervals"
            )
        pulses.append(
            PlaybackPulse(
                interval.lane,
                interval.start_tick / ir.clock_hz,
                (interval.stop_tick - interval.start_tick) / ir.clock_hz,
                1,
                interval.output_key,
            )
        )
    terminal_with_delay = terminal_tick + max(ir.channel_delays, default=0)
    group_counts: dict[tuple[int, int], dict[str, int]] = {}
    for schedule in artifact.trigger_schedules:
        for edge in schedule.iter_edges():
            counts = group_counts.setdefault(
                (edge.point_index, edge.loop_iteration),
                {},
            )
            counts[schedule.channel] = counts.get(schedule.channel, 0) + 1
    trigger_channels = tuple(
        schedule.channel for schedule in artifact.trigger_schedules
    )
    full_point_loop_channels = tuple(
        schedule.channel
        for schedule in artifact.trigger_schedules
        if schedule.full_point_loop
    )
    trigger_groups = tuple(
        PlaybackTriggerGroup(
            point_index,
            loop_iteration,
            tuple(
                (channel, counts[channel])
                for channel in trigger_channels
                if channel in counts
            ),
        )
        for (point_index, loop_iteration), counts in sorted(group_counts.items())
    )
    return PulsePlayback(
        name=name,
        pulses=tuple(sorted(pulses, key=lambda pulse: (pulse.start, pulse.channel))),
        logical_duration=terminal_tick / ir.clock_hz,
        duration=terminal_with_delay / ir.clock_hz,
        repeat_forever=ir.repeat_forever,
        repeat_count=1,
        trigger_channels=trigger_channels,
        full_point_loop_channels=full_point_loop_channels,
        trigger_groups=trigger_groups,
    )


__all__ = [
    "MAX_MATERIALIZED_PLAYBACK_PULSES",
    "PlaybackPulse",
    "PlaybackTriggerGroup",
    "PulsePlayback",
    "build_pulse_playback",
    "sample_compiled_bus_codes",
]
