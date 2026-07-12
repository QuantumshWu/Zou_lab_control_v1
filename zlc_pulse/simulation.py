"""Pulse-owned digital playback projection for virtual hardware adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .artifact import CompiledPulseArtifact
from .ir import TargetIR


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
class PulsePlayback:
    name: str
    pulses: tuple[PlaybackPulse, ...]
    logical_duration: float
    duration: float
    repeat_forever: bool
    repeat_count: int = 1

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

    def base_pulses(self) -> tuple[PlaybackPulse, ...]:
        return self.pulses


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
    return PulsePlayback(
        name=name,
        pulses=tuple(sorted(pulses, key=lambda pulse: (pulse.start, pulse.channel))),
        logical_duration=terminal_tick / ir.clock_hz,
        duration=terminal_with_delay / ir.clock_hz,
        repeat_forever=ir.repeat_forever,
        repeat_count=1,
    )


def _expanded_mask_transitions(ir: TargetIR) -> tuple[list[tuple[int, int]], int]:
    points = ir.scan_points or ((),)
    transitions: list[tuple[int, int]] = []
    run_offset = 0
    for point in points:
        effective = tuple(
            _effective_tick(base, coeffs, point, ir.scan_coeff_frac_bits)
            for base, coeffs in zip(ir.ticks, ir.tick_slot_coeffs)
        )
        final_tick = effective[-1]
        loop_start_tick = effective[ir.loop_start_index]
        loop_end_tick = _effective_tick(
            ir.loop_end_tick,
            ir.loop_end_slot_coeffs,
            point,
            ir.scan_coeff_frac_bits,
        )
        loop_span = loop_end_tick - loop_start_tick
        for index in range(ir.loop_start_index):
            if effective[index] < final_tick:
                transitions.append((run_offset + effective[index], ir.masks[index]))
        for iteration in range(ir.loop_count):
            shift = iteration * loop_span
            for index in range(ir.loop_start_index, len(ir.ticks)):
                tick = effective[index]
                if tick >= loop_end_tick or tick >= final_tick:
                    break
                transitions.append((run_offset + tick + shift, ir.masks[index]))
        tail_shift = (ir.loop_count - 1) * loop_span
        for index in range(ir.loop_start_index, len(ir.ticks)):
            tick = effective[index]
            if loop_end_tick <= tick < final_tick:
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


def _effective_tick(
    base: int,
    coefficients: tuple[int, ...],
    point: tuple[int, ...],
    frac_bits: int,
) -> int:
    return int(base) + (
        sum(coefficient * value for coefficient, value in zip(coefficients, point))
        >> frac_bits
    )


__all__ = ["PlaybackPulse", "PulsePlayback", "build_pulse_playback"]
