"""Pulse-derived physical waveform inside one camera integration window."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from zlc_pulse.artifact import CompiledPulseArtifact
from zlc_pulse.document import PulseDocument
from zlc_pulse.physical import (
    PhysicalWindowProjection,
    build_physical_waveform_index,
)
from zlc_pulse.target import PORT_DIGITAL
from zlc_data import DatasetSchema, READOUT_EVENT
from zlc_storage import (
    canonical_text,
    nonnegative_integer,
    nonnegative_real,
    positive_real,
    sha256_text,
)

from zlc_neutral_atom.runtime.dataset import DatasetCellAddress
from zlc_neutral_atom.timing.lineage import (
    PulseCaptureBinding,
    PulseCaptureEvidence,
)


def _noop_checkpoint() -> None:
    return None


def digital_outputs_falling_after_period(
    document: PulseDocument,
    period_id: str,
) -> tuple[str, ...]:
    """Return stable digital output keys whose physical high ends with a period.

    Readout light is a waveform fact, not a presentation label.  The caller may
    further exclude a known trigger line or intersect the result with a recorded
    calibration context, but no layer may infer this role from ``PulsePortSpec.label``.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    try:
        period = document.period_by_id[period_id]
    except KeyError as error:
        raise ValueError("period_id is absent from PulseDocument") from error
    index = document.periods.index(period)
    next_states = (
        document.periods[index + 1].states
        if index + 1 < len(document.periods)
        else (0,) * len(document.target.raw_lanes)
    )
    lane_index = {
        lane: position for position, lane in enumerate(document.target.raw_lanes)
    }
    return tuple(
        port.key
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and period.states[lane_index[port.lanes[0]]] == 1
        and next_states[lane_index[port.lanes[0]]] == 0
    )


@dataclass(frozen=True)
class DigitalReadoutTrace:
    """One logical digital output normalized to the trigger edge."""

    output_key: str
    high_at_window_start: bool
    transitions: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        canonical_text(self.output_key, "output_key")
        if type(self.high_at_window_start) is not bool:
            raise TypeError("high_at_window_start must be bool")
        transitions = tuple(tuple(item) for item in self.transitions)
        normalized = []
        for item in transitions:
            if len(item) != 2:
                raise ValueError("digital transitions must be (relative_tick, high) pairs")
            tick = nonnegative_integer(item[0], "digital transition tick")
            if tick == 0:
                raise ValueError("tick-zero state belongs in high_at_window_start")
            if type(item[1]) is not bool:
                raise TypeError("digital transition state must be bool")
            normalized.append((tick, item[1]))
        if tuple(tick for tick, _state in normalized) != tuple(
            sorted({tick for tick, _state in normalized})
        ):
            raise ValueError("digital transition ticks must be unique and increasing")
        state = self.high_at_window_start
        for _tick, next_state in normalized:
            if next_state is state:
                raise ValueError("digital trace contains a redundant transition")
            state = next_state
        object.__setattr__(self, "transitions", tuple(normalized))


@dataclass(frozen=True)
class BusReadoutTrace:
    """One decoded DAC bus held/edge waveform normalized to the trigger edge."""

    bus_name: str
    value_at_window_start: int
    changes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        canonical_text(self.bus_name, "bus_name")
        start = nonnegative_integer(
            self.value_at_window_start,
            "value_at_window_start",
        )
        changes = tuple(tuple(item) for item in self.changes)
        normalized = []
        for item in changes:
            if len(item) != 2:
                raise ValueError("bus changes must be (relative_tick, value) pairs")
            tick = nonnegative_integer(item[0], "bus change tick")
            if tick == 0:
                raise ValueError("tick-zero bus value belongs in value_at_window_start")
            value = nonnegative_integer(item[1], "bus change value")
            normalized.append((tick, value))
        if tuple(tick for tick, _value in normalized) != tuple(
            sorted({tick for tick, _value in normalized})
        ):
            raise ValueError("bus change ticks must be unique and increasing")
        value = start
        for _tick, next_value in normalized:
            if next_value == value:
                raise ValueError("bus trace contains a redundant change")
            value = next_value
        object.__setattr__(self, "value_at_window_start", start)
        object.__setattr__(self, "changes", tuple(normalized))


@dataclass(frozen=True)
class ReadoutPhysicalContext:
    """Complete external pulse-output waveform seen during one integration."""

    clock_hz: float
    target_abi_fingerprint: str
    integration_start_offset_seconds: float
    integration_seconds: float
    digital: tuple[DigitalReadoutTrace, ...]
    buses: tuple[BusReadoutTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_hz", positive_real(self.clock_hz, "clock_hz"))
        object.__setattr__(
            self,
            "target_abi_fingerprint",
            sha256_text(self.target_abi_fingerprint, "target_abi_fingerprint"),
        )
        object.__setattr__(
            self,
            "integration_start_offset_seconds",
            nonnegative_real(
                self.integration_start_offset_seconds,
                "integration_start_offset_seconds",
            ),
        )
        object.__setattr__(
            self,
            "integration_seconds",
            positive_real(self.integration_seconds, "integration_seconds"),
        )
        digital = tuple(self.digital)
        buses = tuple(self.buses)
        if any(not isinstance(item, DigitalReadoutTrace) for item in digital):
            raise TypeError("digital must contain DigitalReadoutTrace values")
        if any(not isinstance(item, BusReadoutTrace) for item in buses):
            raise TypeError("buses must contain BusReadoutTrace values")
        if tuple(item.output_key for item in digital) != tuple(
            sorted({item.output_key for item in digital})
        ):
            raise ValueError("digital traces must have unique sorted output keys")
        if tuple(item.bus_name for item in buses) != tuple(
            sorted({item.bus_name for item in buses})
        ):
            raise ValueError("bus traces must have unique sorted names")
        object.__setattr__(self, "digital", digital)
        object.__setattr__(self, "buses", buses)


def derive_readout_physical_context(
    pulse_binding: PulseCaptureBinding,
    *,
    readout_event_index: int,
    integration_start_offset_seconds: float,
    integration_seconds: float,
) -> ReadoutPhysicalContext:
    """Derive one context and reject a single-model capture that contains many."""

    if not isinstance(pulse_binding, PulseCaptureBinding):
        raise TypeError("pulse_binding must be PulseCaptureBinding")
    artifact = pulse_binding.compiled_artifact
    cell_plan = pulse_binding.cell_plan
    event_index = nonnegative_integer(readout_event_index, "readout_event_index")
    if integration_start_offset_seconds is None:
        raise ValueError(
            "readout physical context requires a qualified integration start offset"
        )
    integration_offset = nonnegative_real(
        integration_start_offset_seconds,
        "integration_start_offset_seconds",
    )
    exposure = positive_real(integration_seconds, "integration_seconds")
    schedule = pulse_binding.trigger_schedule
    grouping = cell_plan.join_contract.within_point_grouping
    return _derive_readout_context(
        artifact,
        cell_plan.trigger_channel,
        (
            edge.tick_from_run_start
            for edge in schedule.iter_edges()
            if grouping[edge.point_trigger_ordinal][1] == event_index
        ),
        integration_start_offset_seconds=integration_offset,
        integration_seconds=exposure,
    )


def _derive_readout_physical_context_from_evidence(
    evidence: PulseCaptureEvidence,
    schema: DatasetSchema,
    cell_schedule: Iterable[DatasetCellAddress],
    *,
    readout_event_index: int,
    integration_start_offset_seconds: float,
    integration_seconds: float,
    checkpoint: Callable[[], None] | None = None,
) -> ReadoutPhysicalContext:
    """Derive a persisted context by ordinal-joining pulse edges and cells."""

    if not isinstance(evidence, PulseCaptureEvidence):
        raise TypeError("evidence must be PulseCaptureEvidence")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    edges = evidence.trigger_schedule.iter_edges()
    event_columns = tuple(
        column
        for column in schema.point_table.columns
        if column.role == READOUT_EVENT
    )
    if len(event_columns) != 1:
        raise ValueError(
            "persisted triggered capture requires one READOUT_EVENT column"
        )
    event_column = event_columns[0]
    event_index = nonnegative_integer(readout_event_index, "readout_event_index")
    if event_index not in event_column.values:
        raise ValueError("readout_event_index is outside DatasetSchema")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")

    def anchor_ticks():
        for ordinal, (edge, cell) in enumerate(
            zip(edges, cell_schedule, strict=True)
        ):
            if checkpoint is not None:
                checkpoint()
            if not isinstance(cell, DatasetCellAddress):
                raise TypeError(
                    "cell_schedule must contain DatasetCellAddress values"
                )
            if edge.trigger_ordinal != ordinal:
                raise ValueError("pulse trigger ordinals are not contiguous")
            if event_column.values[cell.point_ordinal] == event_index:
                yield edge.tick_from_run_start

    return _derive_readout_context(
        evidence.compiled_artifact,
        evidence.trigger_channel,
        anchor_ticks(),
        integration_start_offset_seconds=integration_start_offset_seconds,
        integration_seconds=integration_seconds,
        checkpoint=checkpoint,
    )


def _derive_readout_context(
    artifact: CompiledPulseArtifact,
    trigger_channel: str,
    anchor_ticks: Iterable[int],
    *,
    integration_start_offset_seconds: float,
    integration_seconds: float,
    checkpoint: Callable[[], None] | None = None,
) -> ReadoutPhysicalContext:
    integration_offset = nonnegative_real(
        integration_start_offset_seconds,
        "integration_start_offset_seconds",
    )
    exposure = positive_real(integration_seconds, "integration_seconds")
    ir = artifact.target_ir
    trigger_outputs = tuple(
        key for key, lane in ir.logical_digital_outputs if lane == trigger_channel
    )
    if len(trigger_outputs) != 1:
        raise ValueError("capture trigger lane is not one logical digital output")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    callback = _noop_checkpoint if checkpoint is None else checkpoint
    index = build_physical_waveform_index(
        ir,
        checkpoint=callback,
    )
    iterator = iter(anchor_ticks)
    try:
        first_tick = next(iterator)
    except StopIteration as exc:
        raise ValueError("readout event has no capture cells") from exc
    first = _context_from_projection(
        index.window(
            first_tick,
            integration_start_offset_seconds=integration_offset,
            integration_seconds=exposure,
            exclude_output_key=trigger_outputs[0],
            checkpoint=callback,
        )
    )
    for anchor_tick in iterator:
        item = _context_from_projection(
            index.window(
                anchor_tick,
                integration_start_offset_seconds=integration_offset,
                integration_seconds=exposure,
                exclude_output_key=trigger_outputs[0],
                checkpoint=callback,
            )
        )
        if item != first:
            raise ValueError(
                "selected readout physical context varies across repeat/scan cells"
            )
    return first


def _context_from_projection(
    projection: PhysicalWindowProjection,
) -> ReadoutPhysicalContext:
    if not isinstance(projection, PhysicalWindowProjection):
        raise TypeError("projection must be PhysicalWindowProjection")
    digital = tuple(
        DigitalReadoutTrace(
            item.output_key,
            item.high_at_window_start,
            item.transitions,
        )
        for item in projection.digital
    )
    buses = tuple(
        BusReadoutTrace(
            item.bus_name,
            item.value_at_window_start,
            item.changes,
        )
        for item in projection.buses
    )
    return ReadoutPhysicalContext(
        projection.clock_hz,
        projection.target_abi_fingerprint,
        projection.integration_start_offset_seconds,
        projection.integration_seconds,
        digital,
        buses,
    )


def readout_physical_context_to_tree(value: ReadoutPhysicalContext) -> dict[str, object]:
    if not isinstance(value, ReadoutPhysicalContext):
        raise TypeError("value must be ReadoutPhysicalContext")
    return {
        "clock_hz": value.clock_hz,
        "target_abi_fingerprint": value.target_abi_fingerprint,
        "integration_start_offset_seconds": value.integration_start_offset_seconds,
        "integration_seconds": value.integration_seconds,
        "digital": [
            {
                "output_key": item.output_key,
                "high_at_window_start": item.high_at_window_start,
                "transitions": [list(change) for change in item.transitions],
            }
            for item in value.digital
        ],
        "buses": [
            {
                "bus_name": item.bus_name,
                "value_at_window_start": item.value_at_window_start,
                "changes": [list(change) for change in item.changes],
            }
            for item in value.buses
        ],
    }


def readout_physical_context_from_tree(tree: object) -> ReadoutPhysicalContext:
    if not isinstance(tree, dict) or set(tree) != {
        "clock_hz",
        "target_abi_fingerprint",
        "integration_start_offset_seconds",
        "integration_seconds",
        "digital",
        "buses",
    }:
        raise ValueError("ReadoutPhysicalContext has an unknown field set")
    if not isinstance(tree["digital"], list) or not isinstance(tree["buses"], list):
        raise TypeError("ReadoutPhysicalContext traces must be lists")
    digital = []
    for item in tree["digital"]:
        if not isinstance(item, dict) or set(item) != {
            "output_key",
            "high_at_window_start",
            "transitions",
        }:
            raise ValueError("DigitalReadoutTrace has an unknown field set")
        if not isinstance(item["transitions"], list):
            raise TypeError("digital transitions must be a list")
        digital.append(
            DigitalReadoutTrace(
                item["output_key"],
                item["high_at_window_start"],
                tuple(_pair(change, "digital transition") for change in item["transitions"]),
            )
        )
    buses = []
    for item in tree["buses"]:
        if not isinstance(item, dict) or set(item) != {
            "bus_name",
            "value_at_window_start",
            "changes",
        }:
            raise ValueError("BusReadoutTrace has an unknown field set")
        if not isinstance(item["changes"], list):
            raise TypeError("bus changes must be a list")
        buses.append(
            BusReadoutTrace(
                item["bus_name"],
                item["value_at_window_start"],
                tuple(_pair(change, "bus change") for change in item["changes"]),
            )
        )
    return ReadoutPhysicalContext(
        tree["clock_hz"],
        tree["target_abi_fingerprint"],
        tree["integration_start_offset_seconds"],
        tree["integration_seconds"],
        tuple(digital),
        tuple(buses),
    )


def _pair(value: object, field: str) -> tuple[object, object]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item list")
    return value[0], value[1]


__all__ = [
    "BusReadoutTrace",
    "DigitalReadoutTrace",
    "ReadoutPhysicalContext",
    "derive_readout_physical_context",
    "readout_physical_context_from_tree",
    "readout_physical_context_to_tree",
]
