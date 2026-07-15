"""Deterministic compiled-trigger to exact-dataset cell association.

This plan binds scheduled trigger ordinals and physical pulse rows to dataset
storage.  It deliberately does not claim hardware-observed edge receipt or bind
scan-axis coordinates/units to Pulse parameters; Formal scan output requires a
separate ScanOutputContract for those physical x-axis semantics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import InitVar, dataclass

from zlc_data import (
    AxisId,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    SCAN_POINT,
    point_layout_from_tree,
    point_layout_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    DigitalTriggerSchedule,
    PulseExecutionForm,
    digital_trigger_schedule_to_tree,
)
from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    nonnegative_integer as _index,
)

from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    dataset_cell_permutation_digest,
)


@dataclass(frozen=True)
class CaptureCellJoinContract:
    """Compact authored facts that reconstruct trigger-ordinal cell identity."""

    scan_point_layout: PointLayout
    within_point_grouping: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scan_point_layout, PointLayout):
            raise TypeError("scan_point_layout must be PointLayout")
        grouping = tuple(
            tuple(_index(index, "within_point_grouping index") for index in pair)
            for pair in self.within_point_grouping
        )
        if any(len(pair) != 2 for pair in grouping):
            raise ValueError(
                "within_point_grouping entries must be (repeat, event) pairs"
            )
        if len(grouping) != len(set(grouping)):
            raise ValueError("within_point_grouping entries must be unique")
        object.__setattr__(self, "within_point_grouping", grouping)

    def expected_cell_schedule_digest(
        self,
        schedule: DigitalTriggerSchedule,
        schema: DatasetSchema,
    ) -> str:
        return dataset_cell_permutation_digest(
            schema,
            self.iter_cell_schedule(schedule, schema),
        )

    def iter_cell_schedule(
        self,
        schedule: DigitalTriggerSchedule,
        schema: DatasetSchema,
    ) -> Iterator[DatasetCellAddress]:
        """Yield the compact contract's exact trigger-ordinal cell mapping."""

        if not isinstance(schedule, DigitalTriggerSchedule):
            raise TypeError("schedule must be DigitalTriggerSchedule")
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        event_axes = tuple(
            (position, axis)
            for position, axis in enumerate(schema.point_axes)
            if axis.role == READOUT_EVENT
        )
        if len(event_axes) != 1:
            raise ValueError("capture join requires exactly one READOUT_EVENT axis")
        event_position, event_axis = event_axes[0]
        scan_axes = tuple(
            axis
            for position, axis in enumerate(schema.point_axes)
            if position != event_position
        )
        if any(axis.role != SCAN_POINT for axis in scan_axes):
            raise ValueError("capture join non-event axes must be scan-point axes")
        if self.scan_point_layout.logical_shape != tuple(
            axis.size for axis in scan_axes
        ):
            raise ValueError("capture join scan layout differs from DatasetSchema")
        if self.scan_point_layout.storage_size != schedule.point_count:
            raise ValueError("capture join scan layout differs from pulse points")
        repeat_count = schema.repeat_axis.size
        event_count = event_axis.size
        if (
            len(self.within_point_grouping) != repeat_count * event_count
            or any(
                repeat >= repeat_count or event >= event_count
                for repeat, event in self.within_point_grouping
            )
        ):
            raise ValueError("capture join grouping is not a complete R by E domain")
        if schema.point_layout.storage_size != schedule.point_count * event_count:
            raise ValueError("capture join DatasetSchema storage differs from P * E")
        expected_total = schedule.point_count * len(self.within_point_grouping)
        if schedule.total != expected_total:
            raise ValueError("capture join cardinality differs from pulse and dataset")

        scan_position = {
            axis.axis_id: position for position, axis in enumerate(scan_axes)
        }
        for edge in schedule.edges:
            try:
                repeat_index, event_index = self.within_point_grouping[
                    edge.point_trigger_ordinal
                ]
            except IndexError as exc:
                raise ValueError(
                    "capture join grouping differs from pulse point ordinals"
                ) from exc
            scan_multi = self.scan_point_layout.multi_index(edge.point_index)
            full_multi = tuple(
                event_index
                if position == event_position
                else scan_multi[scan_position[axis.axis_id]]
                for position, axis in enumerate(schema.point_axes)
            )
            yield DatasetCellAddress(
                repeat_index,
                schema.point_layout.storage_index(full_multi),
            )


def capture_cell_join_contract_to_tree(
    value: CaptureCellJoinContract,
) -> dict[str, object]:
    if not isinstance(value, CaptureCellJoinContract):
        raise TypeError("value must be CaptureCellJoinContract")
    return {
        "scan_point_layout": point_layout_to_tree(value.scan_point_layout),
        "within_point_grouping": [
            [repeat, event] for repeat, event in value.within_point_grouping
        ],
    }


def capture_cell_join_contract_from_tree(tree: object) -> CaptureCellJoinContract:
    if not isinstance(tree, dict) or set(tree) != {
        "scan_point_layout",
        "within_point_grouping",
    }:
        raise ValueError("CaptureCellJoinContract has an unknown field set")
    grouping = tree["within_point_grouping"]
    if not isinstance(grouping, list) or any(
        not isinstance(pair, list) or len(pair) != 2 for pair in grouping
    ):
        raise TypeError("within_point_grouping must contain two-item lists")
    return CaptureCellJoinContract(
        point_layout_from_tree(tree["scan_point_layout"]),
        tuple(tuple(pair) for pair in grouping),
    )


_COMPILED_CAPTURE_CELL_PLAN_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CompiledCaptureCellPlan:
    _factory_token: InitVar[object]
    compiled_pulse_artifact_digest: str
    execution_form: PulseExecutionForm
    trigger_channel: str
    trigger_schedule_digest: str
    dataset_schema: DatasetSchema
    join_contract: CaptureCellJoinContract
    cell_schedule: DatasetCellSchedule

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _COMPILED_CAPTURE_CELL_PLAN_TOKEN:
            raise TypeError("CompiledCaptureCellPlan must be built by its compiler")

    def validate_dataset_schema(self, schema: DatasetSchema) -> None:
        """Recompute every stored scan/event cell against its DatasetSchema."""

        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if self.dataset_schema.fingerprint != schema.fingerprint:
            raise ValueError("capture plan DatasetSchema fingerprint differs")

    @property
    def total_events(self) -> int:
        return len(self.cell_schedule)

def compile_capture_cell_plan(
    artifact: CompiledPulseArtifact,
    trigger_channel: str,
    dataset_schema: DatasetSchema,
    *,
    readout_event_axis_id: AxisId,
    scan_point_layout: PointLayout,
    within_point_grouping: tuple[tuple[int, int], ...] | None = None,
) -> CompiledCaptureCellPlan:
    """Compile a declared within-point trigger grouping into dataset cells.

    No trigger timing, shape, or cardinality heuristic assigns repeat/event
    meaning.  A singleton R or E has one unambiguous ordering; when both exceed
    one the caller must freeze the physical grouping explicitly.
    """

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
        raise ValueError("continuous pulse execution has no finite capture cell plan")
    _text(trigger_channel, "trigger_channel")
    if not isinstance(dataset_schema, DatasetSchema):
        raise TypeError("dataset_schema must be DatasetSchema")
    if not isinstance(readout_event_axis_id, AxisId):
        raise TypeError("readout_event_axis_id must be AxisId")
    if not isinstance(scan_point_layout, PointLayout):
        raise TypeError("scan_point_layout must be PointLayout")
    schedules = tuple(
        schedule
        for schedule in artifact.trigger_schedules
        if schedule.channel == trigger_channel
    )
    if len(schedules) != 1:
        raise ValueError("capture plan requires exactly one trigger-channel schedule")
    schedule = schedules[0]
    try:
        event_axis = next(
            axis
            for axis in dataset_schema.point_axes
            if axis.axis_id == readout_event_axis_id
        )
    except StopIteration as exc:
        raise ValueError("readout event axis is absent from DatasetSchema") from exc
    if event_axis.role != READOUT_EVENT:
        raise ValueError("readout event axis must have role 'readout-event'")
    repeat_count = dataset_schema.repeat_axis.size
    event_count = event_axis.size
    if within_point_grouping is None:
        if repeat_count > 1 and event_count > 1:
            raise ValueError(
                "within_point_grouping is required when repeat and readout-event "
                "axes both exceed one"
            )
        grouping = _repeat_major_capture_grouping(repeat_count, event_count)
    else:
        grouping = within_point_grouping

    join_contract = CaptureCellJoinContract(scan_point_layout, grouping)
    cell_schedule = DatasetCellSchedule.from_cells(
        dataset_schema,
        join_contract.iter_cell_schedule(schedule, dataset_schema),
    )
    plan = CompiledCaptureCellPlan(
        _COMPILED_CAPTURE_CELL_PLAN_TOKEN,
        compiled_pulse_artifact_digest=artifact.fingerprint,
        execution_form=artifact.execution_form,
        trigger_channel=trigger_channel,
        trigger_schedule_digest=canonical_digest(
            digital_trigger_schedule_to_tree(schedule)
        ),
        dataset_schema=dataset_schema,
        join_contract=join_contract,
        cell_schedule=cell_schedule,
    )
    return plan


def _repeat_major_capture_grouping(
    repeat_count: int,
    readout_event_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return the explicit repeat-major/event-minor grouping declaration."""

    repeats = _index(repeat_count, "repeat_count")
    events = _index(readout_event_count, "readout_event_count")
    if repeats < 1 or events < 1:
        raise ValueError("repeat and readout-event counts must be positive")
    return tuple(
        (repeat, event)
        for repeat in range(repeats)
        for event in range(events)
    )


__all__ = [
    "CaptureCellJoinContract",
    "CompiledCaptureCellPlan",
    "capture_cell_join_contract_from_tree",
    "capture_cell_join_contract_to_tree",
    "compile_capture_cell_plan",
]
