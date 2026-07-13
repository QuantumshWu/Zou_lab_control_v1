"""Deterministic logical-point attribution for compiled digital trigger edges."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import TargetIR, evaluate_affine_tick


MAX_MATERIALIZED_TRIGGER_EDGES = 1_000_000


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


@dataclass(frozen=True)
class DigitalTriggerSchedule:
    channel: str
    edges: tuple[TriggerEdge, ...]
    point_count: int
    loop_count: int = 1
    full_point_loop: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("trigger schedule channel must be non-empty text")
        edges = tuple(self.edges)
        if any(not isinstance(edge, TriggerEdge) or edge.channel != self.channel for edge in edges):
            raise ValueError("trigger schedule edges belong to another channel")
        if tuple(edge.trigger_ordinal for edge in edges) != tuple(range(len(edges))):
            raise ValueError("trigger ordinals must be contiguous from zero")
        if any(right.tick_from_run_start <= left.tick_from_run_start for left, right in zip(edges, edges[1:])):
            raise ValueError("trigger edge ticks must strictly increase")
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
        if any(edge.point_index >= self.point_count for edge in edges):
            raise ValueError("trigger edge point index exceeds point_count")
        if any(edge.loop_iteration >= self.loop_count for edge in edges):
            raise ValueError("trigger edge loop iteration exceeds loop_count")
        per_point: dict[int, int] = {}
        previous_loop_by_point: dict[int, int] = {}
        for edge in edges:
            expected = per_point.get(edge.point_index, 0)
            if edge.point_trigger_ordinal != expected:
                raise ValueError("per-point trigger ordinals must be contiguous")
            per_point[edge.point_index] = expected + 1
            previous_loop = previous_loop_by_point.get(edge.point_index, 0)
            if edge.loop_iteration < previous_loop:
                raise ValueError("loop iterations must be monotonic within each point")
            previous_loop_by_point[edge.point_index] = edge.loop_iteration
        if self.full_point_loop:
            per_group: dict[tuple[int, int], int] = {}
            for edge in edges:
                key = (edge.point_index, edge.loop_iteration)
                per_group[key] = per_group.get(key, 0) + 1
            total_groups = self.point_count * self.loop_count
            if per_group and (
                len(per_group) != total_groups
                or len(set(per_group.values())) > 1
            ):
                raise ValueError(
                    "a full-point loop must emit the same trigger count in every group"
                )
        object.__setattr__(self, "edges", edges)

    @property
    def total(self) -> int:
        return len(self.edges)

    @property
    def minimum_interval_ticks(self) -> int | None:
        if len(self.edges) < 2:
            return None
        return min(
            right.tick_from_run_start - left.tick_from_run_start
            for left, right in zip(self.edges, self.edges[1:])
        )


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

    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    requested = tuple(channels)
    if len(set(requested)) != len(requested):
        raise ValueError("trigger channels must be unique")
    if not requested:
        return ()
    if ir.repeat_forever:
        raise ValueError("a cyclic TargetIR has no finite trigger schedule")
    indices = []
    for channel in requested:
        if not isinstance(channel, str) or channel not in ir.channels:
            raise ValueError(f"unknown trigger channel {channel!r}")
        index = ir.channels.index(channel)
        if ir.clk_enable & (1 << index):
            raise ValueError(f"clock-mux channel {channel!r} has no finite pulse trigger schedule")
        indices.append(index)
    points = ir.scan_points or ((),)
    schedules = []
    remaining_budget = MAX_MATERIALIZED_TRIGGER_EDGES
    for channel, bit_index in zip(requested, indices):
        previous = 0
        run_offset = 0
        edges: list[TriggerEdge] = []
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
                if effective[index] < loop_end_tick
                and effective[index] < final_tick
            )
            tail = tuple(
                (effective[index], ir.masks[index])
                for index in range(ir.loop_start_index, len(ir.ticks))
                if loop_end_tick <= effective[index] < final_tick
            )
            point_ordinal = 0

            def append_rises(
                rises: tuple[int, ...],
                shift: int = 0,
                loop_iteration: int = 0,
            ) -> None:
                nonlocal point_ordinal
                if len(edges) + len(rises) > remaining_budget:
                    raise ValueError(
                        "digital trigger schedule exceeds the materialization limit "
                        f"of {MAX_MATERIALIZED_TRIGGER_EDGES} edges"
                    )
                for tick in rises:
                    edges.append(
                        TriggerEdge(
                            channel,
                            point_index,
                            loop_iteration,
                            len(edges),
                            point_ordinal,
                            run_offset
                            + tick
                            + shift
                            + ir.channel_delays[bit_index],
                        )
                    )
                    point_ordinal += 1

            rises, previous = _rising_ticks(prefix, bit_index, previous)
            append_rises(rises)
            rises, previous = _rising_ticks(body, bit_index, previous)
            append_rises(rises)
            if ir.loop_count > 1:
                steady_rises, steady_end = _rising_ticks(body, bit_index, previous)
                repeat_edges = len(steady_rises) * (ir.loop_count - 1)
                if len(edges) + repeat_edges > remaining_budget:
                    raise ValueError(
                        "digital trigger schedule exceeds the materialization limit "
                        f"of {MAX_MATERIALIZED_TRIGGER_EDGES} edges"
                    )
                if steady_rises:
                    for iteration in range(1, ir.loop_count):
                        append_rises(
                            steady_rises,
                            iteration * loop_span,
                            iteration,
                        )
                previous = steady_end
            tail_shift = (ir.loop_count - 1) * loop_span
            rises, previous = _rising_ticks(tail, bit_index, previous)
            append_rises(rises, tail_shift, ir.loop_count - 1)
            run_offset += final_tick + tail_shift
        schedules.append(
            DigitalTriggerSchedule(
                channel,
                tuple(edges),
                len(points),
                ir.loop_count,
                (
                    ir.loop_start_index == 0
                    and ir.loop_end_tick == ir.ticks[-1]
                    and ir.loop_end_slot_coeffs == ir.tick_slot_coeffs[-1]
                ),
            )
        )
        remaining_budget -= len(edges)
    return tuple(schedules)


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
        "edges": [
            {
                "channel": edge.channel,
                "point_index": edge.point_index,
                "loop_iteration": edge.loop_iteration,
                "trigger_ordinal": edge.trigger_ordinal,
                "point_trigger_ordinal": edge.point_trigger_ordinal,
                "tick_from_run_start": edge.tick_from_run_start,
            }
            for edge in value.edges
        ],
    }


def digital_trigger_schedule_from_tree(tree: object) -> DigitalTriggerSchedule:
    if not isinstance(tree, dict) or set(tree) != {
        "channel",
        "point_count",
        "loop_count",
        "full_point_loop",
        "edges",
    }:
        raise ValueError("DigitalTriggerSchedule has an unknown field set")
    if not isinstance(tree["edges"], list):
        raise TypeError("DigitalTriggerSchedule edges must be a list")
    edges = []
    fields = {
        "channel",
        "point_index",
        "loop_iteration",
        "trigger_ordinal",
        "point_trigger_ordinal",
        "tick_from_run_start",
    }
    for item in tree["edges"]:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("TriggerEdge has an unknown field set")
        edges.append(TriggerEdge(**item))
    return DigitalTriggerSchedule(
        tree["channel"],
        tuple(edges),
        tree["point_count"],
        tree["loop_count"],
        tree["full_point_loop"],
    )


__all__ = [
    "DigitalTriggerSchedule",
    "MAX_MATERIALIZED_TRIGGER_EDGES",
    "TriggerEdge",
    "build_digital_trigger_schedules",
    "digital_trigger_schedule_from_tree",
    "digital_trigger_schedule_to_tree",
]
