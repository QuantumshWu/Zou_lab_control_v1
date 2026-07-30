"""Deterministic compiled-trigger to exact Dataset cell association."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import InitVar, dataclass

from zlc_data import AxisId, DatasetSchema, READOUT_EVENT
from zlc_pulse import CompiledPulseArtifact, DigitalTriggerSchedule, PulseExecutionForm
from zlc_storage import (
    canonical_text as _text,
    nonnegative_integer as _index,
    positive_integer as _positive,
)

from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
)


@dataclass(frozen=True)
class CaptureCellJoinContract:
    """Minimal authored facts mapping trigger ordinals to R/P addresses."""

    base_point_count: int
    readout_event_axis_id: AxisId
    readout_event_count: int
    within_point_grouping: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_point_count",
            _positive(self.base_point_count, "base_point_count"),
        )
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        object.__setattr__(
            self,
            "readout_event_count",
            _positive(self.readout_event_count, "readout_event_count"),
        )
        grouping = tuple(
            tuple(_index(index, "within_point_grouping index") for index in pair)
            for pair in self.within_point_grouping
        )
        if any(len(pair) != 2 for pair in grouping):
            raise ValueError("grouping entries must be (repeat, event) pairs")
        if len(grouping) != len(set(grouping)):
            raise ValueError("within_point_grouping entries must be unique")
        object.__setattr__(self, "within_point_grouping", grouping)

    def expected_cell_schedule(
        self,
        schedule: DigitalTriggerSchedule,
        schema: DatasetSchema,
    ) -> DatasetCellSchedule:
        return DatasetCellSchedule.from_cells(
            schema,
            self.iter_cell_schedule(schedule, schema),
        )

    def iter_cell_schedule(
        self,
        schedule: DigitalTriggerSchedule,
        schema: DatasetSchema,
    ) -> Iterator[DatasetCellAddress]:
        if not isinstance(schedule, DigitalTriggerSchedule):
            raise TypeError("schedule must be DigitalTriggerSchedule")
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        event_columns = tuple(
            column
            for column in schema.point_table.columns
            if column.role == READOUT_EVENT
        )
        if len(event_columns) != 1:
            raise ValueError("capture join requires exactly one READOUT_EVENT column")
        event_column = event_columns[0]
        if event_column.coordinate_id != self.readout_event_axis_id:
            raise ValueError("capture join names another READOUT_EVENT column")
        expected_rows = self.base_point_count * self.readout_event_count
        if schema.point_table.row_count != expected_rows:
            raise ValueError("capture Dataset P differs from base points × events")
        expected_events = tuple(
            event
            for _point in range(self.base_point_count)
            for event in range(self.readout_event_count)
        )
        if event_column.values != expected_events:
            raise ValueError("READOUT_EVENT rows must be base-major/event-minor")

        repeat_count = schema.repeat_axis.size
        repeat_major_points = (
            schedule.loop_count == 1
            and schedule.full_point_loop
            and schedule.point_count == repeat_count * self.base_point_count
        )
        if repeat_major_points:
            if self.within_point_grouping != tuple(
                (0, event) for event in range(self.readout_event_count)
            ):
                raise ValueError("expanded scan grouping is not the event domain")
            expected_total = repeat_count * expected_rows
        else:
            if schedule.point_count != self.base_point_count:
                raise ValueError("capture base point count differs from pulse points")
            expected_grouping = repeat_count * self.readout_event_count
            if (
                len(self.within_point_grouping) != expected_grouping
                or any(
                    repeat >= repeat_count or event >= self.readout_event_count
                    for repeat, event in self.within_point_grouping
                )
            ):
                raise ValueError("capture grouping is not a complete R by E domain")
            expected_total = self.base_point_count * expected_grouping
        if schedule.total != expected_total:
            raise ValueError("capture cardinality differs from pulse and Dataset")

        for edge in schedule.iter_edges():
            try:
                grouped_repeat, event_index = self.within_point_grouping[
                    edge.point_trigger_ordinal
                ]
            except IndexError as exc:
                raise ValueError("capture grouping differs from pulse ordinals") from exc
            if repeat_major_points:
                repeat_index, base_point_ordinal = divmod(
                    edge.point_index,
                    self.base_point_count,
                )
            else:
                repeat_index = grouped_repeat
                base_point_ordinal = edge.point_index
            yield DatasetCellAddress(
                repeat_index,
                base_point_ordinal * self.readout_event_count + event_index,
            )


def capture_cell_join_contract_to_tree(
    value: CaptureCellJoinContract,
) -> dict[str, object]:
    if not isinstance(value, CaptureCellJoinContract):
        raise TypeError("value must be CaptureCellJoinContract")
    return {
        "base_point_count": value.base_point_count,
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "readout_event_count": value.readout_event_count,
        "within_point_grouping": [
            [repeat, event] for repeat, event in value.within_point_grouping
        ],
    }


def capture_cell_join_contract_from_tree(tree: object) -> CaptureCellJoinContract:
    fields = {
        "base_point_count",
        "readout_event_axis_id",
        "readout_event_count",
        "within_point_grouping",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CaptureCellJoinContract has an unknown field set")
    grouping = tree["within_point_grouping"]
    if not isinstance(grouping, list) or any(
        not isinstance(pair, list) or len(pair) != 2 for pair in grouping
    ):
        raise TypeError("within_point_grouping must contain two-item lists")
    return CaptureCellJoinContract(
        tree["base_point_count"],
        AxisId(tree["readout_event_axis_id"]),
        tree["readout_event_count"],
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
    base_point_count: int,
    within_point_grouping: tuple[tuple[int, int], ...] | None = None,
) -> CompiledCaptureCellPlan:
    """Compile declared trigger grouping directly into Dataset row ordinals."""

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.execution_form in (
        PulseExecutionForm.CONTINUOUS_MONITOR,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    ):
        raise ValueError("continuous pulse execution has no finite capture cell plan")
    _text(trigger_channel, "trigger_channel")
    if not isinstance(dataset_schema, DatasetSchema):
        raise TypeError("dataset_schema must be DatasetSchema")
    if not isinstance(readout_event_axis_id, AxisId):
        raise TypeError("readout_event_axis_id must be AxisId")
    base_count = _positive(base_point_count, "base_point_count")
    schedules = tuple(
        schedule
        for schedule in artifact.trigger_schedules
        if schedule.channel == trigger_channel
    )
    if len(schedules) != 1:
        raise ValueError("capture plan requires exactly one trigger schedule")
    schedule = schedules[0]
    event_column = dataset_schema.point_table.column(readout_event_axis_id)
    if event_column.role != READOUT_EVENT:
        raise ValueError("readout event column has the wrong role")
    event_values = tuple(dict.fromkeys(event_column.values))
    if event_values != tuple(range(len(event_values))):
        raise ValueError("READOUT_EVENT values must be canonical zero-based indices")
    event_count = len(event_values)
    repeat_count = dataset_schema.repeat_axis.size
    repeat_major_points = (
        schedule.loop_count == 1
        and schedule.full_point_loop
        and schedule.point_count == repeat_count * base_count
    )
    if repeat_major_points and within_point_grouping is None:
        grouping = _repeat_major_capture_grouping(1, event_count)
    elif within_point_grouping is None:
        if repeat_count > 1 and event_count > 1:
            raise ValueError(
                "within_point_grouping is required when R and event both exceed one"
            )
        grouping = _repeat_major_capture_grouping(repeat_count, event_count)
    else:
        grouping = within_point_grouping
    join_contract = CaptureCellJoinContract(
        base_count,
        readout_event_axis_id,
        event_count,
        grouping,
    )
    cell_schedule = DatasetCellSchedule.from_cells(
        dataset_schema,
        join_contract.iter_cell_schedule(schedule, dataset_schema),
    )
    return CompiledCaptureCellPlan(
        _COMPILED_CAPTURE_CELL_PLAN_TOKEN,
        compiled_pulse_artifact_digest=artifact.fingerprint,
        execution_form=artifact.execution_form,
        trigger_channel=trigger_channel,
        trigger_schedule_digest=schedule.fingerprint,
        dataset_schema=dataset_schema,
        join_contract=join_contract,
        cell_schedule=cell_schedule,
    )


def _repeat_major_capture_grouping(
    repeat_count: int,
    readout_event_count: int,
) -> tuple[tuple[int, int], ...]:
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
