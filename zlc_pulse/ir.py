"""Pulse-owned immutable TargetIR immediately above the FPGA wire image."""

from __future__ import annotations

import math
from dataclasses import dataclass

from zlc_storage import canonical_digest


TARGET_IR_SCHEMA = "zlc_pulse.TargetIR/v1"
BUS_MODES = frozenset(("hold", "edge", "ramp"))
SLOT_KINDS = frozenset(("duration", "dac"))


def _text(value: object, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value or (not empty and not value):
        raise ValueError(f"{field} must be canonical text")
    return value


def _integer(value: object, field: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return normalized


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class TargetBusDelay:
    bus_index: int
    delay_ticks: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bus_index", _integer(self.bus_index, "bus_index", nonnegative=True))
        object.__setattr__(
            self,
            "delay_ticks",
            _integer(self.delay_ticks, "delay_ticks", nonnegative=True),
        )


@dataclass(frozen=True)
class TargetBusSegment:
    bus_index: int
    bus_name: str
    start_tick: int
    stop_tick: int
    start_value: int
    stop_value: int
    mode: str
    value_select: int
    stop_value_select: int
    start_tick_coeffs: tuple[int, ...]
    stop_tick_coeffs: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bus_index", _integer(self.bus_index, "bus_index", nonnegative=True))
        object.__setattr__(self, "bus_name", _text(self.bus_name, "bus_name"))
        for field in ("start_tick", "stop_tick", "value_select", "stop_value_select"):
            object.__setattr__(self, field, _integer(getattr(self, field), field, nonnegative=True))
        for field in ("start_value", "stop_value"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        if self.stop_tick < self.start_tick:
            raise ValueError("bus segment stop_tick precedes start_tick")
        mode = _text(self.mode, "bus segment mode")
        if mode not in BUS_MODES:
            raise ValueError(f"unsupported bus segment mode {mode!r}")
        object.__setattr__(self, "mode", mode)
        for field in ("start_tick_coeffs", "stop_tick_coeffs"):
            values = tuple(_integer(value, f"{field} item") for value in getattr(self, field))
            object.__setattr__(self, field, values)


@dataclass(frozen=True)
class TargetIR:
    clock_hz: float
    target_abi_fingerprint: str
    channels: tuple[str, ...]
    ticks: tuple[int, ...]
    masks: tuple[int, ...]
    duration_seconds: float
    repeat_forever: bool
    loop_start_index: int
    loop_end_tick: int
    loop_count: int
    slot_kinds: tuple[str, ...] = ()
    loop_end_slot_coeffs: tuple[int, ...] = ()
    tick_slot_coeffs: tuple[tuple[int, ...], ...] = ()
    scan_points: tuple[tuple[int, ...], ...] = ()
    scan_point_durations: tuple[float, ...] = ()
    scan_coeff_frac_bits: int = 0
    scan_repeats: int = 0
    bus_names: tuple[str, ...] = ()
    bus_segments: tuple[TargetBusSegment, ...] = ()
    bus_delays: tuple[TargetBusDelay, ...] = ()
    channel_delays: tuple[int, ...] = ()
    clk_enable: int = 0

    @property
    def slot_count(self) -> int:
        return len(self.slot_kinds)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_hz", _positive_float(self.clock_hz, "clock_hz"))
        _sha256(self.target_abi_fingerprint, "target_abi_fingerprint")
        channels = tuple(_text(channel, "TargetIR channel") for channel in self.channels)
        if not channels or len(channels) != len(set(channels)):
            raise ValueError("TargetIR channels must be unique and non-empty")
        object.__setattr__(self, "channels", channels)
        ticks = tuple(_integer(value, "edge tick", nonnegative=True) for value in self.ticks)
        masks = tuple(_integer(value, "edge mask", nonnegative=True) for value in self.masks)
        if not ticks or len(ticks) != len(masks):
            raise ValueError("TargetIR requires equally-sized non-empty ticks and masks")
        if ticks[0] != 0:
            raise ValueError("TargetIR edge ticks must start at zero")
        mask_limit = 1 << len(channels)
        if any(mask >= mask_limit for mask in masks):
            raise ValueError("TargetIR edge mask exceeds channel width")
        object.__setattr__(self, "ticks", ticks)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(
            self,
            "duration_seconds",
            _positive_float(self.duration_seconds, "duration_seconds"),
        )
        if type(self.repeat_forever) is not bool:
            raise TypeError("repeat_forever must be bool")
        loop_start = _integer(self.loop_start_index, "loop_start_index", nonnegative=True)
        loop_end = _integer(self.loop_end_tick, "loop_end_tick", nonnegative=True)
        loop_count = _integer(self.loop_count, "loop_count", nonnegative=True)
        if loop_start >= len(ticks) or loop_end < ticks[loop_start] or loop_count < 1:
            raise ValueError("TargetIR loop metadata is outside its edge table")
        object.__setattr__(self, "loop_start_index", loop_start)
        object.__setattr__(self, "loop_end_tick", loop_end)
        object.__setattr__(self, "loop_count", loop_count)
        slot_kinds = tuple(_text(kind, "slot kind") for kind in self.slot_kinds)
        if any(kind not in SLOT_KINDS for kind in slot_kinds):
            raise ValueError("TargetIR contains an unsupported slot kind")
        object.__setattr__(self, "slot_kinds", slot_kinds)
        slot_count = len(slot_kinds)
        loop_coeffs = tuple(
            _integer(value, "loop-end coefficient") for value in self.loop_end_slot_coeffs
        )
        if len(loop_coeffs) != slot_count:
            raise ValueError("loop-end coefficient width differs from slot count")
        object.__setattr__(self, "loop_end_slot_coeffs", loop_coeffs)
        coeff_rows = tuple(
            tuple(_integer(value, "edge coefficient") for value in row)
            for row in self.tick_slot_coeffs
        )
        if len(coeff_rows) != len(ticks) or any(len(row) != slot_count for row in coeff_rows):
            raise ValueError("edge coefficient matrix shape differs from ticks/slots")
        object.__setattr__(self, "tick_slot_coeffs", coeff_rows)
        points = tuple(
            tuple(_integer(value, "scan point slot") for value in point)
            for point in self.scan_points
        )
        if any(len(point) != slot_count for point in points):
            raise ValueError("scan point width differs from slot count")
        if bool(points) != bool(slot_count):
            raise ValueError("scan points and slot schema must appear together")
        object.__setattr__(self, "scan_points", points)
        if not points and any(right <= left for left, right in zip(ticks, ticks[1:])):
            raise ValueError("static TargetIR edge ticks must strictly increase")
        durations = tuple(
            _positive_float(value, "scan point duration")
            for value in self.scan_point_durations
        )
        if len(durations) != len(points):
            raise ValueError("scan point duration count differs from scan points")
        object.__setattr__(self, "scan_point_durations", durations)
        frac = _integer(self.scan_coeff_frac_bits, "scan_coeff_frac_bits", nonnegative=True)
        repeats = _integer(self.scan_repeats, "scan_repeats", nonnegative=True)
        if not points and (frac != 0 or repeats != 0):
            raise ValueError("scan-only metadata cannot appear on a static TargetIR")
        object.__setattr__(self, "scan_coeff_frac_bits", frac)
        object.__setattr__(self, "scan_repeats", repeats)
        bus_names = tuple(_text(value, "bus name") for value in self.bus_names)
        if len(bus_names) != len(set(bus_names)):
            raise ValueError("bus names must be unique")
        object.__setattr__(self, "bus_names", bus_names)
        segments = tuple(self.bus_segments)
        if any(not isinstance(value, TargetBusSegment) for value in segments):
            raise TypeError("bus_segments must contain TargetBusSegment values")
        for segment in segments:
            if segment.bus_index >= len(bus_names) or bus_names[segment.bus_index] != segment.bus_name:
                raise ValueError("bus segment index/name differs from bus_names")
            if (
                len(segment.start_tick_coeffs) != slot_count
                or len(segment.stop_tick_coeffs) != slot_count
            ):
                raise ValueError("bus segment coefficient width differs from slot count")
            if segment.value_select > slot_count or segment.stop_value_select > slot_count:
                raise ValueError("bus segment value selector exceeds slot count")
        object.__setattr__(self, "bus_segments", segments)
        delays = tuple(self.bus_delays)
        if any(not isinstance(value, TargetBusDelay) for value in delays):
            raise TypeError("bus_delays must contain TargetBusDelay values")
        if len({value.bus_index for value in delays}) != len(delays) or any(
            value.bus_index >= len(bus_names) for value in delays
        ):
            raise ValueError("bus delays must uniquely reference bus_names")
        object.__setattr__(self, "bus_delays", delays)
        channel_delays = tuple(
            _integer(value, "channel delay", nonnegative=True)
            for value in self.channel_delays
        )
        if len(channel_delays) != len(channels):
            raise ValueError("channel delay vector must match channel count")
        object.__setattr__(self, "channel_delays", channel_delays)
        clk_enable = _integer(self.clk_enable, "clk_enable", nonnegative=True)
        if clk_enable >= mask_limit or any(mask & clk_enable for mask in masks):
            raise ValueError("clk_enable exceeds channel width or conflicts with edge masks")
        object.__setattr__(self, "clk_enable", clk_enable)
        for point_index, point in enumerate(points):
            effective = tuple(
                _effective_tick(base, coeff, point, frac)
                for base, coeff in zip(ticks, coeff_rows)
            )
            if effective[0] != 0 or any(
                right <= left for left, right in zip(effective, effective[1:])
            ):
                raise ValueError(
                    f"scan point {point_index} produces non-increasing effective ticks"
                )
            effective_loop_end = _effective_tick(loop_end, loop_coeffs, point, frac)
            if effective_loop_end < effective[loop_start]:
                raise ValueError(f"scan point {point_index} produces an invalid loop end")

    @property
    def scan_enabled(self) -> bool:
        return bool(self.scan_points)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(target_ir_to_tree(self))


def _effective_tick(base: int, coeffs: tuple[int, ...], point: tuple[int, ...], frac: int) -> int:
    return int(base) + (sum(coefficient * value for coefficient, value in zip(coeffs, point)) >> frac)


def target_ir_to_tree(value: TargetIR) -> dict[str, object]:
    if not isinstance(value, TargetIR):
        raise TypeError("value must be TargetIR")
    return {
        "schema": TARGET_IR_SCHEMA,
        "clock_hz": value.clock_hz,
        "target_abi_fingerprint": value.target_abi_fingerprint,
        "channels": list(value.channels),
        "ticks": list(value.ticks),
        "masks": list(value.masks),
        "duration_seconds": value.duration_seconds,
        "repeat_forever": value.repeat_forever,
        "loop_start_index": value.loop_start_index,
        "loop_end_tick": value.loop_end_tick,
        "loop_count": value.loop_count,
        "slot_kinds": list(value.slot_kinds),
        "loop_end_slot_coeffs": list(value.loop_end_slot_coeffs),
        "tick_slot_coeffs": [list(row) for row in value.tick_slot_coeffs],
        "scan_points": [list(row) for row in value.scan_points],
        "scan_point_durations": list(value.scan_point_durations),
        "scan_coeff_frac_bits": value.scan_coeff_frac_bits,
        "scan_repeats": value.scan_repeats,
        "bus_names": list(value.bus_names),
        "bus_segments": [_bus_segment_to_tree(segment) for segment in value.bus_segments],
        "bus_delays": [
            {"bus_index": delay.bus_index, "delay_ticks": delay.delay_ticks}
            for delay in value.bus_delays
        ],
        "channel_delays": list(value.channel_delays),
        "clk_enable": value.clk_enable,
    }


def target_ir_from_tree(tree: object) -> TargetIR:
    fields = {
        "schema",
        "clock_hz",
        "target_abi_fingerprint",
        "channels",
        "ticks",
        "masks",
        "duration_seconds",
        "repeat_forever",
        "loop_start_index",
        "loop_end_tick",
        "loop_count",
        "slot_kinds",
        "loop_end_slot_coeffs",
        "tick_slot_coeffs",
        "scan_points",
        "scan_point_durations",
        "scan_coeff_frac_bits",
        "scan_repeats",
        "bus_names",
        "bus_segments",
        "bus_delays",
        "channel_delays",
        "clk_enable",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("TargetIR has an unknown field set")
    if tree["schema"] != TARGET_IR_SCHEMA:
        raise ValueError("TargetIR schema differs")
    for field in (
        "channels",
        "ticks",
        "masks",
        "slot_kinds",
        "loop_end_slot_coeffs",
        "tick_slot_coeffs",
        "scan_points",
        "scan_point_durations",
        "bus_names",
        "bus_segments",
        "bus_delays",
        "channel_delays",
    ):
        if not isinstance(tree[field], list):
            raise TypeError(f"TargetIR {field} must be a list")
    return TargetIR(
        tree["clock_hz"],
        tree["target_abi_fingerprint"],
        tuple(tree["channels"]),
        tuple(tree["ticks"]),
        tuple(tree["masks"]),
        tree["duration_seconds"],
        tree["repeat_forever"],
        tree["loop_start_index"],
        tree["loop_end_tick"],
        tree["loop_count"],
        tuple(tree["slot_kinds"]),
        tuple(tree["loop_end_slot_coeffs"]),
        tuple(_row(row, "tick_slot_coeffs") for row in tree["tick_slot_coeffs"]),
        tuple(_row(row, "scan_points") for row in tree["scan_points"]),
        tuple(tree["scan_point_durations"]),
        tree["scan_coeff_frac_bits"],
        tree["scan_repeats"],
        tuple(tree["bus_names"]),
        tuple(_bus_segment_from_tree(item) for item in tree["bus_segments"]),
        tuple(_bus_delay_from_tree(item) for item in tree["bus_delays"]),
        tuple(tree["channel_delays"]),
        tree["clk_enable"],
    )


def _bus_segment_to_tree(value: TargetBusSegment) -> dict[str, object]:
    return {
        "bus_index": value.bus_index,
        "bus_name": value.bus_name,
        "start_tick": value.start_tick,
        "stop_tick": value.stop_tick,
        "start_value": value.start_value,
        "stop_value": value.stop_value,
        "mode": value.mode,
        "value_select": value.value_select,
        "stop_value_select": value.stop_value_select,
        "start_tick_coeffs": list(value.start_tick_coeffs),
        "stop_tick_coeffs": list(value.stop_tick_coeffs),
    }


def _bus_segment_from_tree(tree: object) -> TargetBusSegment:
    fields = {
        "bus_index",
        "bus_name",
        "start_tick",
        "stop_tick",
        "start_value",
        "stop_value",
        "mode",
        "value_select",
        "stop_value_select",
        "start_tick_coeffs",
        "stop_tick_coeffs",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("TargetBusSegment has an unknown field set")
    return TargetBusSegment(
        tree["bus_index"],
        tree["bus_name"],
        tree["start_tick"],
        tree["stop_tick"],
        tree["start_value"],
        tree["stop_value"],
        tree["mode"],
        tree["value_select"],
        tree["stop_value_select"],
        _row(tree["start_tick_coeffs"], "start_tick_coeffs"),
        _row(tree["stop_tick_coeffs"], "stop_tick_coeffs"),
    )


def _bus_delay_from_tree(tree: object) -> TargetBusDelay:
    if not isinstance(tree, dict) or set(tree) != {"bus_index", "delay_ticks"}:
        raise ValueError("TargetBusDelay has an unknown field set")
    return TargetBusDelay(tree["bus_index"], tree["delay_ticks"])


def _row(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} rows must be lists")
    return tuple(value)


__all__ = [
    "BUS_MODES",
    "SLOT_KINDS",
    "TARGET_IR_SCHEMA",
    "TargetBusDelay",
    "TargetBusSegment",
    "TargetIR",
    "target_ir_from_tree",
    "target_ir_to_tree",
]
