"""Deterministic logical-point attribution for compiled digital trigger edges."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from zlc_storage import canonical_digest

from .ir import TargetIR, evaluate_affine_tick


MAX_MATERIALIZED_TRIGGER_EDGES = 1_000_000
MAX_TRIGGER_TICK = (1 << 64) - 1
_POINT_INDEX_DTYPE = np.dtype("<u4")
_LOOP_ITERATION_DTYPE = np.dtype("<u4")
_TRIGGER_TICK_DTYPE = np.dtype("<u8")


@dataclass(frozen=True)
class TriggerEdge:
    channel: str
    point_index: int
    loop_iteration: int
    trigger_ordinal: int
    point_trigger_ordinal: int
    tick_from_run_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("trigger channel must be non-empty text")
        for field in (
            "point_index",
            "loop_iteration",
            "trigger_ordinal",
            "point_trigger_ordinal",
            "tick_from_run_start",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.tick_from_run_start > MAX_TRIGGER_TICK:
            raise ValueError("tick_from_run_start exceeds the uint64 schedule domain")


@dataclass(frozen=True, eq=False)
class DigitalTriggerSchedule:
    channel: str
    point_indices: np.ndarray
    loop_iterations: np.ndarray
    ticks_from_run_start: np.ndarray
    point_count: int
    loop_count: int = 1
    full_point_loop: bool = True
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("trigger schedule channel must be non-empty text")
        if isinstance(self.point_count, bool) or not isinstance(self.point_count, int) or self.point_count < 1:
            raise ValueError("point_count must be a positive integer")
        if (
            isinstance(self.loop_count, bool)
            or not isinstance(self.loop_count, int)
            or self.loop_count < 1
        ):
            raise ValueError("loop_count must be a positive integer")
        if type(self.full_point_loop) is not bool:
            raise TypeError("full_point_loop must be bool")
        point_indices = _immutable_column(
            self.point_indices,
            dtype=_POINT_INDEX_DTYPE,
            field="point_indices",
        )
        loop_iterations = _immutable_column(
            self.loop_iterations,
            dtype=_LOOP_ITERATION_DTYPE,
            field="loop_iterations",
        )
        ticks = _immutable_column(
            self.ticks_from_run_start,
            dtype=_TRIGGER_TICK_DTYPE,
            field="ticks_from_run_start",
        )
        total = len(point_indices)
        if len(loop_iterations) != total or len(ticks) != total:
            raise ValueError("trigger schedule columns must have equal length")
        if total > MAX_MATERIALIZED_TRIGGER_EDGES:
            raise ValueError(
                "digital trigger schedule exceeds the materialization limit "
                f"of {MAX_MATERIALIZED_TRIGGER_EDGES} edges"
            )
        if total > 1 and bool(np.any(point_indices[1:] < point_indices[:-1])):
            raise ValueError("trigger point indices must be monotonic")
        if total > 1 and bool(np.any(ticks[1:] <= ticks[:-1])):
            raise ValueError("trigger edge ticks must strictly increase")
        if total and int(point_indices[-1]) >= self.point_count:
            raise ValueError("trigger edge point index exceeds point_count")
        if total and int(np.max(loop_iterations)) >= self.loop_count:
            raise ValueError("trigger edge loop iteration exceeds loop_count")
        previous_point = -1
        previous_loop = 0
        for raw_point, raw_loop in zip(point_indices, loop_iterations, strict=True):
            point = int(raw_point)
            loop = int(raw_loop)
            if point == previous_point and loop < previous_loop:
                raise ValueError("loop iterations must be monotonic within each point")
            if point != previous_point:
                previous_point = point
                previous_loop = 0
            previous_loop = loop
        if self.full_point_loop:
            per_group: dict[tuple[int, int], int] = {}
            for raw_point, raw_loop in zip(point_indices, loop_iterations, strict=True):
                key = (int(raw_point), int(raw_loop))
                per_group[key] = per_group.get(key, 0) + 1
            total_groups = self.point_count * self.loop_count
            if per_group and (
                len(per_group) != total_groups
                or len(set(per_group.values())) > 1
            ):
                raise ValueError(
                    "a full-point loop must emit the same trigger count in every group"
                )
        object.__setattr__(self, "point_indices", point_indices)
        object.__setattr__(self, "loop_iterations", loop_iterations)
        object.__setattr__(self, "ticks_from_run_start", ticks)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(digital_trigger_schedule_to_tree(self)),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DigitalTriggerSchedule)
            and self.channel == other.channel
            and self.point_count == other.point_count
            and self.loop_count == other.loop_count
            and self.full_point_loop == other.full_point_loop
            and np.array_equal(self.point_indices, other.point_indices)
            and np.array_equal(self.loop_iterations, other.loop_iterations)
            and np.array_equal(self.ticks_from_run_start, other.ticks_from_run_start)
        )

    def __hash__(self) -> int:
        return hash((DigitalTriggerSchedule, self.fingerprint))

    @property
    def total(self) -> int:
        return len(self.point_indices)

    def iter_edges(self) -> Iterator[TriggerEdge]:
        """Yield typed row views without retaining one Python object per edge."""

        previous_point = -1
        point_ordinal = 0
        for ordinal, (raw_point, raw_loop, raw_tick) in enumerate(
            zip(
                self.point_indices,
                self.loop_iterations,
                self.ticks_from_run_start,
                strict=True,
            )
        ):
            point = int(raw_point)
            if point != previous_point:
                previous_point = point
                point_ordinal = 0
            yield TriggerEdge(
                self.channel,
                point,
                int(raw_loop),
                ordinal,
                point_ordinal,
                int(raw_tick),
            )
            point_ordinal += 1

    @property
    def minimum_interval_ticks(self) -> int | None:
        if self.total < 2:
            return None
        return int(np.min(self.ticks_from_run_start[1:] - self.ticks_from_run_start[:-1]))


def _immutable_column(value: object, *, dtype: np.dtype, field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != dtype:
        raise TypeError(f"{field} must be a one-dimensional {dtype.str} ndarray")
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=dtype)


def _column(values: list[int], *, dtype: np.dtype, field: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=dtype)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} exceeds the fixed-width schedule domain") from exc


def build_digital_trigger_schedules(
    ir: TargetIR,
    channels: tuple[str, ...],
) -> tuple[DigitalTriggerSchedule, ...]:
    """Expand finite engine semantics without simulating every hardware tick.

    Logical point ownership is assigned before applying the channel delay, then
    the constant physical output delay is added to the edge tick.  Therefore an
    edge delayed past a point boundary keeps the compiled point ordinal instead
    of sampling a later runtime cursor.
    """

    requested, indices = _trigger_channel_indices(ir, channels)
    if not requested:
        return ()
    points = ir.scan_points or ((),)
    schedules = []
    remaining_budget = MAX_MATERIALIZED_TRIGGER_EDGES
    for channel, bit_index in zip(requested, indices):
        point_indices: list[int] = []
        loop_iterations: list[int] = []
        ticks_from_run_start: list[int] = []
        for point_index, loop_iteration, tick_from_run_start in _iter_digital_trigger_edge_rows(
            ir,
            bit_index,
        ):
            if len(point_indices) >= remaining_budget:
                raise ValueError(
                    "digital trigger schedule exceeds the materialization limit "
                    f"of {MAX_MATERIALIZED_TRIGGER_EDGES} edges"
                )
            if tick_from_run_start > MAX_TRIGGER_TICK:
                raise ValueError(
                    "digital trigger schedule exceeds the uint64 run-tick domain"
                )
            point_indices.append(point_index)
            loop_iterations.append(loop_iteration)
            ticks_from_run_start.append(tick_from_run_start)
        schedules.append(
            DigitalTriggerSchedule(
                channel,
                _column(
                    point_indices,
                    dtype=_POINT_INDEX_DTYPE,
                    field="point_indices",
                ),
                _column(
                    loop_iterations,
                    dtype=_LOOP_ITERATION_DTYPE,
                    field="loop_iterations",
                ),
                _column(
                    ticks_from_run_start,
                    dtype=_TRIGGER_TICK_DTYPE,
                    field="ticks_from_run_start",
                ),
                len(points),
                ir.loop_count,
                _full_point_loop(ir),
            )
        )
        remaining_budget -= len(point_indices)
    return tuple(schedules)


def _same_physical_digital_trigger_schedules(
    ir: TargetIR,
    schedules: tuple[DigitalTriggerSchedule, ...],
) -> bool:
    """Stream expected wire facts without constructing a second edge graph.

    Source loop grouping is a sidecar that may differ after output-delay lowering,
    so this compares the same physical fields as the previous materialized check.
    """

    actual = tuple(schedules)
    if any(not isinstance(item, DigitalTriggerSchedule) for item in actual):
        return False
    requested = tuple(item.channel for item in actual)
    requested, indices = _trigger_channel_indices(ir, requested)
    points = ir.scan_points or ((),)
    remaining_budget = MAX_MATERIALIZED_TRIGGER_EDGES
    for schedule, channel, bit_index in zip(actual, requested, indices, strict=True):
        if schedule.channel != channel or schedule.point_count != len(points):
            return False
        actual_index = 0
        for expected_point, _expected_loop, expected_tick in _iter_digital_trigger_edge_rows(
            ir,
            bit_index,
        ):
            if remaining_budget <= 0:
                raise ValueError(
                    "digital trigger schedule exceeds the materialization limit "
                    f"of {MAX_MATERIALIZED_TRIGGER_EDGES} edges"
                )
            if actual_index >= schedule.total:
                return False
            if (
                int(schedule.point_indices[actual_index]) != expected_point
                or int(schedule.ticks_from_run_start[actual_index]) != expected_tick
            ):
                return False
            actual_index += 1
            remaining_budget -= 1
        if actual_index != schedule.total:
            return False
    return True


def _trigger_channel_indices(
    ir: TargetIR,
    channels: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    requested = tuple(channels)
    if len(set(requested)) != len(requested):
        raise ValueError("trigger channels must be unique")
    if ir.repeat_forever and requested:
        raise ValueError("a cyclic TargetIR has no finite trigger schedule")
    indices = []
    for channel in requested:
        if not isinstance(channel, str) or channel not in ir.channels:
            raise ValueError(f"unknown trigger channel {channel!r}")
        index = ir.channels.index(channel)
        if ir.clk_enable & (1 << index):
            raise ValueError(
                f"clock-mux channel {channel!r} has no finite pulse trigger schedule"
            )
        indices.append(index)
    return requested, tuple(indices)


def _full_point_loop(ir: TargetIR) -> bool:
    return (
        ir.loop_start_index == 0
        and ir.loop_end_tick == ir.ticks[-1]
        and ir.loop_end_slot_coeffs == ir.tick_slot_coeffs[-1]
    )


def _iter_digital_trigger_edge_rows(
    ir: TargetIR,
    bit_index: int,
):
    previous = 0
    run_offset = 0
    points = ir.scan_points or ((),)
    for point_index, point in enumerate(points):
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
        prefix = tuple(
            (effective[index], ir.masks[index])
            for index in range(ir.loop_start_index)
            if effective[index] < final_tick
        )
        body = tuple(
            (effective[index], ir.masks[index])
            for index in range(ir.loop_start_index, len(ir.ticks))
            if effective[index] < loop_end_tick and effective[index] < final_tick
        )
        tail = tuple(
            (effective[index], ir.masks[index])
            for index in range(ir.loop_start_index, len(ir.ticks))
            if loop_end_tick <= effective[index] < final_tick
        )
        def rows(
            rises: tuple[int, ...],
            shift: int = 0,
            loop_iteration: int = 0,
        ):
            for tick in rises:
                yield (
                    point_index,
                    loop_iteration,
                    run_offset
                    + tick
                    + shift
                    + ir.channel_delays[bit_index],
                )

        rises, previous = _rising_ticks(prefix, bit_index, previous)
        yield from rows(rises)
        rises, previous = _rising_ticks(body, bit_index, previous)
        yield from rows(rises)
        if ir.loop_count > 1:
            steady_rises, steady_end = _rising_ticks(body, bit_index, previous)
            if steady_rises:
                for iteration in range(1, ir.loop_count):
                    yield from rows(
                        steady_rises,
                        iteration * loop_span,
                        iteration,
                    )
            previous = steady_end
        tail_shift = (ir.loop_count - 1) * loop_span
        rises, previous = _rising_ticks(tail, bit_index, previous)
        yield from rows(rises, tail_shift, ir.loop_count - 1)
        run_offset += final_tick + tail_shift


def _rising_ticks(
    assignments: tuple[tuple[int, int], ...],
    bit_index: int,
    previous: int,
) -> tuple[tuple[int, ...], int]:
    rises = []
    state = int(previous)
    for tick, mask in assignments:
        current = (int(mask) >> bit_index) & 1
        if current and not state:
            rises.append(int(tick))
        state = current
    return tuple(rises), state


def digital_trigger_schedule_to_tree(value: DigitalTriggerSchedule) -> dict[str, object]:
    if not isinstance(value, DigitalTriggerSchedule):
        raise TypeError("value must be DigitalTriggerSchedule")
    return {
        "channel": value.channel,
        "point_count": value.point_count,
        "loop_count": value.loop_count,
        "full_point_loop": value.full_point_loop,
        "point_indices": value.point_indices,
        "loop_iterations": value.loop_iterations,
        "ticks_from_run_start": value.ticks_from_run_start,
    }


def digital_trigger_schedule_from_tree(tree: object) -> DigitalTriggerSchedule:
    if not isinstance(tree, dict) or set(tree) != {
        "channel",
        "point_count",
        "loop_count",
        "full_point_loop",
        "point_indices",
        "loop_iterations",
        "ticks_from_run_start",
    }:
        raise ValueError("DigitalTriggerSchedule has an unknown field set")
    return DigitalTriggerSchedule(
        tree["channel"],
        tree["point_indices"],
        tree["loop_iterations"],
        tree["ticks_from_run_start"],
        tree["point_count"],
        tree["loop_count"],
        tree["full_point_loop"],
    )


__all__ = [
    "DigitalTriggerSchedule",
    "MAX_MATERIALIZED_TRIGGER_EDGES",
    "MAX_TRIGGER_TICK",
    "TriggerEdge",
    "build_digital_trigger_schedules",
    "digital_trigger_schedule_from_tree",
    "digital_trigger_schedule_to_tree",
]
