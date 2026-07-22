"""Pulse-owned immutable TargetIR immediately above the FPGA wire image."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    integer as _integer,
    positive_real as _positive_float,
    sha256_text as _sha256,
)


TARGET_IR_SCHEMA = "zlc_pulse.TargetIR"
BUS_MODES = frozenset(("edge", "ramp"))
SLOT_KINDS = frozenset(("duration", "dac"))
_SCAN_POINT_DTYPE = np.dtype("<i4")
_SCAN_DURATION_DTYPE = np.dtype("<f8")


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
    bus_names: tuple[str, ...] = ()
    bus_segments: tuple[TargetBusSegment, ...] = ()
    bus_delays: tuple[TargetBusDelay, ...] = ()
    channel_delays: tuple[int, ...] = ()
    clk_enable: int = 0
    logical_digital_outputs: tuple[tuple[str, str], ...] = ()
    bus_safe_values: tuple[int, ...] = ()
    _fingerprint: str = field(init=False, repr=False, compare=False)

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
        if masks[-1] != 0:
            raise ValueError("TargetIR final mask must be zero for a safe terminal state")
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
        if loop_start >= len(ticks) or loop_count < 1:
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
        _validate_time_coefficients(loop_coeffs, slot_kinds, "loop end")
        object.__setattr__(self, "loop_end_slot_coeffs", loop_coeffs)
        coeff_rows = tuple(
            tuple(_integer(value, "edge coefficient") for value in row)
            for row in self.tick_slot_coeffs
        )
        if len(coeff_rows) != len(ticks) or any(len(row) != slot_count for row in coeff_rows):
            raise ValueError("edge coefficient matrix shape differs from ticks/slots")
        for row in coeff_rows:
            _validate_time_coefficients(row, slot_kinds, "edge tick")
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
        if not points and frac != 0:
            raise ValueError("scan-only metadata cannot appear on a static TargetIR")
        object.__setattr__(self, "scan_coeff_frac_bits", frac)
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
            _validate_time_coefficients(
                segment.start_tick_coeffs,
                slot_kinds,
                "bus segment start",
            )
            _validate_time_coefficients(
                segment.stop_tick_coeffs,
                slot_kinds,
                "bus segment stop",
            )
            for selector in (segment.value_select, segment.stop_value_select):
                if selector and slot_kinds[selector - 1] != "dac":
                    raise ValueError(
                        "bus segment value selector must reference a DAC slot"
                    )
            if segment.value_select and segment.start_value != 0:
                raise ValueError(
                    "a selected bus start value requires a canonical zero literal"
                )
            if segment.stop_value_select and segment.stop_value != 0:
                raise ValueError(
                    "a selected bus stop value requires a canonical zero literal"
                )
            if segment.mode == "ramp" and segment.start_value != 0:
                raise ValueError(
                    "ramp start_value is physically live-state driven and must be zero"
                )
            if segment.mode == "edge" and (
                segment.start_tick != segment.stop_tick
                or segment.start_tick_coeffs != segment.stop_tick_coeffs
                or segment.start_value != segment.stop_value
                or segment.value_select != segment.stop_value_select
            ):
                raise ValueError("edge bus segment must describe one instantaneous value")
        object.__setattr__(self, "bus_segments", segments)
        if segments and loop_count > 1 and loop_start != 0:
            raise ValueError(
                "DAC bus segments do not support a compact finite inner repeat bracket"
            )
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
        logical_outputs_list = []
        for raw_pair in self.logical_digital_outputs:
            try:
                pair = tuple(raw_pair)
            except TypeError as exc:
                raise TypeError(
                    "logical_digital_outputs entries must be (key, lane) pairs"
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    "logical_digital_outputs entries must be (key, lane) pairs"
                )
            logical_outputs_list.append(
                (
                    _text(pair[0], "logical digital output key"),
                    _text(pair[1], "logical digital output lane"),
                )
            )
        logical_outputs = tuple(logical_outputs_list)
        keys = tuple(key for key, _lane in logical_outputs)
        lanes = tuple(lane for _key, lane in logical_outputs)
        if len(keys) != len(set(keys)) or len(lanes) != len(set(lanes)):
            raise ValueError("logical digital output keys and lanes must be unique")
        unknown_lanes = tuple(lane for lane in lanes if lane not in channels)
        if unknown_lanes:
            raise ValueError(
                f"logical digital outputs reference unknown lanes {unknown_lanes!r}"
            )
        if any(clk_enable & (1 << channels.index(lane)) for lane in lanes):
            raise ValueError("clock-mux lanes cannot be logical digital outputs")
        object.__setattr__(
            self,
            "logical_digital_outputs",
            tuple(sorted(logical_outputs)),
        )
        safe_values = tuple(
            _integer(value, "bus safe value", nonnegative=True)
            for value in self.bus_safe_values
        )
        if len(safe_values) != len(bus_names):
            raise ValueError("bus_safe_values must align exactly with bus_names")
        object.__setattr__(self, "bus_safe_values", safe_values)
        for point_index, point in enumerate(points):
            effective = tuple(
                evaluate_affine_tick(base, coeff, point, frac)
                for base, coeff in zip(ticks, coeff_rows)
            )
            if effective[0] != 0 or any(
                right <= left for left, right in zip(effective, effective[1:])
            ):
                raise ValueError(
                    f"scan point {point_index} produces non-increasing effective ticks"
                )
            effective_loop_end = evaluate_affine_tick(
                loop_end,
                loop_coeffs,
                point,
                frac,
            )
            if not effective[loop_start] < effective_loop_end <= effective[-1]:
                raise ValueError(f"scan point {point_index} produces an invalid loop end")
            expected_duration = (
                effective[-1]
                + (loop_count - 1)
                * (effective_loop_end - effective[loop_start])
            ) / self.clock_hz
            if not _same_duration(durations[point_index], expected_duration):
                raise ValueError(
                    f"scan point {point_index} duration disagrees with loop metadata"
                )
        if points:
            expected_total = sum(durations)
        else:
            if not ticks[loop_start] < loop_end <= ticks[-1]:
                raise ValueError("static TargetIR produces an invalid loop end")
            expected_total = (
                ticks[-1]
                + (loop_count - 1) * (loop_end - ticks[loop_start])
            ) / self.clock_hz
        if not _same_duration(self.duration_seconds, expected_total):
            raise ValueError("TargetIR duration disagrees with its executable schedule")
        for point_index, point in enumerate(points or ((),)):
            effective_final = evaluate_affine_tick(
                ticks[-1],
                coeff_rows[-1],
                point,
                frac,
            )
            previous_by_bus: dict[int, tuple[int, int]] = {}
            for segment in segments:
                start = evaluate_affine_tick(
                    segment.start_tick,
                    segment.start_tick_coeffs,
                    point,
                    frac,
                )
                stop = evaluate_affine_tick(
                    segment.stop_tick,
                    segment.stop_tick_coeffs,
                    point,
                    frac,
                )
                valid_shape = start == stop if segment.mode == "edge" else start < stop
                if (
                    not valid_shape
                    or not 0 <= start < effective_final
                    or stop > effective_final
                ):
                    raise ValueError(
                        f"bus segment is outside scan point {point_index} timing bounds"
                    )
                previous = previous_by_bus.get(segment.bus_index)
                if previous is not None and (
                    start <= previous[0] or start < previous[1]
                ):
                    raise ValueError(
                        f"bus segment order overlaps or regresses at scan point {point_index}"
                    )
                previous_by_bus[segment.bus_index] = (start, stop)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(target_ir_to_tree(self)),
        )

    @property
    def scan_enabled(self) -> bool:
        return bool(self.scan_points)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


def evaluate_affine_tick(
    base: int,
    coefficients: Sequence[int],
    point: Sequence[int],
    frac_bits: int,
) -> int:
    """Evaluate the one fixed-point affine timing formula used by pulse IR."""

    return int(base) + (
        sum(
            int(coefficient) * int(value)
            for coefficient, value in zip(coefficients, point)
        )
        >> int(frac_bits)
    )


def _same_duration(left: float, right: float) -> bool:
    tolerance = max(1e-15, 4 * math.ulp(float(right)))
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _validate_time_coefficients(
    coefficients: tuple[int, ...],
    slot_kinds: tuple[str, ...],
    field: str,
) -> None:
    if any(
        coefficient and kind != "duration"
        for coefficient, kind in zip(coefficients, slot_kinds)
    ):
        raise ValueError(f"{field} coefficients must reference duration slots")


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
        "scan_points": _scan_point_matrix(value),
        "scan_point_durations": np.asarray(
            value.scan_point_durations,
            dtype=_SCAN_DURATION_DTYPE,
        ),
        "scan_coeff_frac_bits": value.scan_coeff_frac_bits,
        "bus_names": list(value.bus_names),
        "bus_segments": [_bus_segment_to_tree(segment) for segment in value.bus_segments],
        "bus_delays": [
            {"bus_index": delay.bus_index, "delay_ticks": delay.delay_ticks}
            for delay in value.bus_delays
        ],
        "channel_delays": list(value.channel_delays),
        "clk_enable": value.clk_enable,
        "logical_digital_outputs": [
            {"key": key, "lane": lane}
            for key, lane in value.logical_digital_outputs
        ],
        "bus_safe_values": list(value.bus_safe_values),
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
        "bus_names",
        "bus_segments",
        "bus_delays",
        "channel_delays",
        "clk_enable",
        "logical_digital_outputs",
        "bus_safe_values",
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
        "bus_names",
        "bus_segments",
        "bus_delays",
        "channel_delays",
        "logical_digital_outputs",
        "bus_safe_values",
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
        _scan_point_rows_from_tree(
            tree["scan_points"],
            slot_count=len(tree["slot_kinds"]),
        ),
        _scan_durations_from_tree(tree["scan_point_durations"]),
        tree["scan_coeff_frac_bits"],
        tuple(tree["bus_names"]),
        tuple(_bus_segment_from_tree(item) for item in tree["bus_segments"]),
        tuple(_bus_delay_from_tree(item) for item in tree["bus_delays"]),
        tuple(tree["channel_delays"]),
        tree["clk_enable"],
        tuple(_logical_digital_output_from_tree(item) for item in tree["logical_digital_outputs"]),
        tuple(tree["bus_safe_values"]),
    )


def _scan_point_matrix(value: TargetIR) -> np.ndarray:
    try:
        return np.asarray(value.scan_points, dtype=_SCAN_POINT_DTYPE).reshape(
            len(value.scan_points),
            value.slot_count,
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("scan point value exceeds the signed 32-bit artifact domain") from exc


def _scan_point_rows_from_tree(
    value: object,
    *,
    slot_count: int,
) -> tuple[tuple[int, ...], ...]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != _SCAN_POINT_DTYPE
        or value.ndim != 2
        or value.shape[1] != slot_count
    ):
        raise TypeError(
            "TargetIR scan_points must be a two-dimensional <i4 ndarray "
            "matching slot_kinds"
        )
    return tuple(tuple(int(item) for item in row) for row in value)


def _scan_durations_from_tree(value: object) -> tuple[float, ...]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != _SCAN_DURATION_DTYPE
        or value.ndim != 1
    ):
        raise TypeError(
            "TargetIR scan_point_durations must be a one-dimensional <f8 ndarray"
        )
    return tuple(float(item) for item in value)


def _logical_digital_output_from_tree(tree: object) -> tuple[object, object]:
    if not isinstance(tree, dict) or set(tree) != {"key", "lane"}:
        raise ValueError("logical digital output has an unknown field set")
    return tree["key"], tree["lane"]


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
    "evaluate_affine_tick",
    "target_ir_from_tree",
    "target_ir_to_tree",
]
