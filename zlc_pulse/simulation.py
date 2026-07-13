"""Pulse-owned digital playback projection for virtual hardware adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .artifact import CompiledPulseArtifact
from .ir import TargetIR, evaluate_affine_tick


MAX_MATERIALIZED_PLAYBACK_TRANSITIONS = 1_000_000


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
        """The playback is already the exact fully expanded finite timeline."""

        return self.pulses

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
    transitions, terminal_tick = _expanded_mask_transitions(ir)
    pulses: list[PlaybackPulse] = []
    for bit, channel in enumerate(ir.channels):
        if ir.clk_enable & (1 << bit):
            continue
        delay = ir.channel_delays[bit]
        active: int | None = None
        previous = 0
        for tick, mask in transitions:
            current = (mask >> bit) & 1
            shifted = tick + delay
            if current and not previous:
                active = shifted
            elif previous and not current:
                if active is None or shifted <= active:
                    raise ValueError("compiled digital waveform contains an invalid high interval")
                pulses.append(
                    PlaybackPulse(
                        channel,
                        active / ir.clock_hz,
                        (shifted - active) / ir.clock_hz,
                        1,
                        channel,
                    )
                )
                active = None
            previous = current
        if active is not None:
            stop = terminal_tick + delay
            if stop <= active:
                raise ValueError("compiled digital waveform remains high without a terminal interval")
            pulses.append(
                PlaybackPulse(
                    channel,
                    active / ir.clock_hz,
                    (stop - active) / ir.clock_hz,
                    1,
                    channel,
                )
            )
    terminal_with_delay = terminal_tick + max(ir.channel_delays, default=0)
    group_counts: dict[tuple[int, int], dict[str, int]] = {}
    for schedule in artifact.trigger_schedules:
        for edge in schedule.edges:
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


def _expanded_mask_transitions(ir: TargetIR) -> tuple[list[tuple[int, int]], int]:
    points = ir.scan_points or ((),)
    transitions: list[tuple[int, int]] = []
    run_offset = 0
    for point in points:
        effective = tuple(
            evaluate_affine_tick(base, coeffs, point, ir.scan_coeff_frac_bits)
            for base, coeffs in zip(ir.ticks, ir.tick_slot_coeffs)
        )
        final_tick = effective[-1]
        loop_start_tick = effective[ir.loop_start_index]
        loop_end_tick = evaluate_affine_tick(
            ir.loop_end_tick,
            ir.loop_end_slot_coeffs,
            point,
            ir.scan_coeff_frac_bits,
        )
        loop_span = loop_end_tick - loop_start_tick
        prefix_indices = tuple(
            index
            for index in range(ir.loop_start_index)
            if effective[index] < final_tick
        )
        body_indices = tuple(
            index
            for index in range(ir.loop_start_index, len(ir.ticks))
            if effective[index] < loop_end_tick and effective[index] < final_tick
        )
        tail_indices = tuple(
            index
            for index in range(ir.loop_start_index, len(ir.ticks))
            if loop_end_tick <= effective[index] < final_tick
        )
        projected = (
            len(transitions)
            + len(prefix_indices)
            + len(body_indices) * ir.loop_count
            + len(tail_indices)
        )
        if projected > MAX_MATERIALIZED_PLAYBACK_TRANSITIONS:
            raise ValueError(
                "pulse playback exceeds the materialization limit of "
                f"{MAX_MATERIALIZED_PLAYBACK_TRANSITIONS} transitions"
            )
        for index in prefix_indices:
            transitions.append((run_offset + effective[index], ir.masks[index]))
        for iteration in range(ir.loop_count):
            shift = iteration * loop_span
            for index in body_indices:
                tick = effective[index]
                transitions.append((run_offset + tick + shift, ir.masks[index]))
        tail_shift = (ir.loop_count - 1) * loop_span
        for index in tail_indices:
            tick = effective[index]
            transitions.append((run_offset + tick + tail_shift, ir.masks[index]))
        run_offset += final_tick + tail_shift
    transitions.append((run_offset, 0))
    transitions.sort(key=lambda item: item[0])
    merged: list[tuple[int, int]] = []
    for tick, mask in transitions:
        if merged and merged[-1][0] == tick:
            merged[-1] = (tick, mask)
        else:
            merged.append((tick, mask))
    return merged, run_offset


__all__ = [
    "MAX_MATERIALIZED_PLAYBACK_TRANSITIONS",
    "PlaybackPulse",
    "PlaybackTriggerGroup",
    "PulsePlayback",
    "build_pulse_playback",
]
