"""Deterministic compiled-trigger to exact-dataset cell association.

This plan binds scheduled trigger ordinals and physical pulse rows to dataset
storage.  It deliberately does not claim hardware-observed edge receipt or bind
scan-axis coordinates/units to Pulse parameters; Formal scan output requires a
separate ScanOutputContract for those physical x-axis semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

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
    PulseExecutionForm,
    digital_trigger_schedule_to_tree,
)
from zlc_storage import canonical_digest, decode, encode

from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    dataset_cell_permutation_fingerprint,
)


COMPILED_CAPTURE_CELL_PLAN_SCHEMA = (
    "zlc_neutral_atom.CompiledCaptureCellPlan/v1"
)


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _index(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class CaptureCellAssignment:
    trigger_ordinal: int
    pulse_point_index: int
    point_trigger_ordinal: int
    repeat_index: int
    readout_event_index: int
    point_storage_index: int

    def __post_init__(self) -> None:
        for field in (
            "trigger_ordinal",
            "pulse_point_index",
            "point_trigger_ordinal",
            "repeat_index",
            "readout_event_index",
            "point_storage_index",
        ):
            object.__setattr__(self, field, _index(getattr(self, field), field))

    @property
    def cell(self) -> DatasetCellAddress:
        return DatasetCellAddress(self.repeat_index, self.point_storage_index)


@dataclass(frozen=True)
class CompiledCaptureCellPlan:
    compiled_pulse_artifact_digest: str
    execution_form: PulseExecutionForm
    trigger_channel: str
    trigger_schedule_digest: str
    dataset_schema_fingerprint: str
    repeat_axis_id: AxisId
    scan_axis_ids: tuple[AxisId, ...]
    scan_point_layout: PointLayout
    readout_event_axis_id: AxisId
    repeat_count: int
    scan_point_count: int
    readout_events_per_repeat: int
    within_point_grouping: tuple[tuple[int, int], ...]
    assignments: tuple[CaptureCellAssignment, ...]

    def __post_init__(self) -> None:
        _digest(
            self.compiled_pulse_artifact_digest,
            "compiled_pulse_artifact_digest",
        )
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            raise ValueError("continuous pulse execution has no finite capture cell plan")
        _text(self.trigger_channel, "trigger_channel")
        _digest(self.trigger_schedule_digest, "trigger_schedule_digest")
        _digest(self.dataset_schema_fingerprint, "dataset_schema_fingerprint")
        if not isinstance(self.repeat_axis_id, AxisId):
            raise TypeError("repeat_axis_id must be AxisId")
        scan_ids = tuple(self.scan_axis_ids)
        if any(not isinstance(axis_id, AxisId) for axis_id in scan_ids):
            raise TypeError("scan_axis_ids must contain AxisId values")
        if len(set(scan_ids)) != len(scan_ids):
            raise ValueError("scan_axis_ids must be unique")
        object.__setattr__(self, "scan_axis_ids", scan_ids)
        if not isinstance(self.scan_point_layout, PointLayout):
            raise TypeError("scan_point_layout must be PointLayout")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if self.readout_event_axis_id in scan_ids:
            raise ValueError("readout event axis cannot also be a scan axis")
        if self.repeat_axis_id in (*scan_ids, self.readout_event_axis_id):
            raise ValueError("repeat axis must be distinct from scan and readout axes")
        for field in (
            "repeat_count",
            "scan_point_count",
            "readout_events_per_repeat",
        ):
            value = _index(getattr(self, field), field)
            if value < 1:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        if self.scan_point_layout.storage_size != self.scan_point_count:
            raise ValueError("scan PointLayout storage differs from pulse point count")
        if len(self.scan_point_layout.logical_shape) != len(scan_ids):
            raise ValueError("scan PointLayout rank differs from scan_axis_ids")
        grouping = tuple(
            tuple(_index(index, "within_point_grouping index") for index in pair)
            for pair in self.within_point_grouping
        )
        if any(len(pair) != 2 for pair in grouping):
            raise ValueError("within_point_grouping entries must be (repeat, event) pairs")
        expected_grouping = {
            (repeat, event)
            for repeat in range(self.repeat_count)
            for event in range(self.readout_events_per_repeat)
        }
        if len(grouping) != len(expected_grouping) or set(grouping) != expected_grouping:
            raise ValueError(
                "within_point_grouping must be a complete unique R by E permutation"
            )
        object.__setattr__(self, "within_point_grouping", grouping)
        assignments = tuple(self.assignments)
        if any(not isinstance(item, CaptureCellAssignment) for item in assignments):
            raise TypeError("assignments must contain CaptureCellAssignment values")
        expected_total = (
            self.repeat_count
            * self.scan_point_count
            * self.readout_events_per_repeat
        )
        if len(assignments) != expected_total:
            raise ValueError("assignment count differs from R * P * E")
        if tuple(item.trigger_ordinal for item in assignments) != tuple(
            range(expected_total)
        ):
            raise ValueError("assignment trigger ordinals must be contiguous from zero")
        point_storage_size = (
            self.scan_point_count * self.readout_events_per_repeat
        )
        per_point: dict[int, int] = {}
        for item in assignments:
            if item.pulse_point_index >= self.scan_point_count:
                raise ValueError("assignment pulse point is outside the compiled schedule")
            expected_point_ordinal = per_point.get(item.pulse_point_index, 0)
            if item.point_trigger_ordinal != expected_point_ordinal:
                raise ValueError("assignment point trigger ordinals are not contiguous")
            per_point[item.pulse_point_index] = expected_point_ordinal + 1
            if item.repeat_index >= self.repeat_count:
                raise ValueError("assignment repeat index is outside DatasetSchema")
            if item.readout_event_index >= self.readout_events_per_repeat:
                raise ValueError("assignment readout event is outside DatasetSchema")
            if item.point_storage_index >= point_storage_size:
                raise ValueError("assignment point storage index is outside DatasetSchema")
            if (
                item.repeat_index,
                item.readout_event_index,
            ) != grouping[item.point_trigger_ordinal]:
                raise ValueError(
                    "assignment differs from the declared within-point grouping"
                )
        if set(per_point) != set(range(self.scan_point_count)) or any(
            count != self.repeat_count * self.readout_events_per_repeat
            for count in per_point.values()
        ):
            raise ValueError("every pulse point must have exactly R * E assignments")
        expected_domain = {
            DatasetCellAddress(repeat, point)
            for repeat in range(self.repeat_count)
            for point in range(point_storage_size)
        }
        if set(item.cell for item in assignments) != expected_domain:
            raise ValueError("assignments must cover every dataset cell exactly once")
        object.__setattr__(self, "assignments", assignments)

    def validate_dataset_schema(self, schema: DatasetSchema) -> None:
        """Recompute every stored scan/event cell against its DatasetSchema."""

        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if self.dataset_schema_fingerprint != schema.fingerprint:
            raise ValueError("capture plan DatasetSchema fingerprint differs")
        if self.repeat_axis_id != schema.repeat_axis.axis_id:
            raise ValueError("capture plan repeat axis differs from DatasetSchema")
        if self.repeat_count != schema.repeat_axis.size:
            raise ValueError("capture plan repeat count differs from DatasetSchema")
        try:
            event_axis = next(
                axis
                for axis in schema.point_axes
                if axis.axis_id == self.readout_event_axis_id
            )
        except StopIteration as exc:
            raise ValueError("capture plan readout axis is absent from DatasetSchema") from exc
        if (
            event_axis.role != READOUT_EVENT
            or event_axis.size != self.readout_events_per_repeat
        ):
            raise ValueError("capture plan readout axis differs from DatasetSchema")
        scan_axes = tuple(
            axis
            for axis in schema.point_axes
            if axis.axis_id != self.readout_event_axis_id
        )
        if tuple(axis.axis_id for axis in scan_axes) != self.scan_axis_ids:
            raise ValueError("capture plan scan axes differ from DatasetSchema")
        if any(axis.role != SCAN_POINT for axis in scan_axes):
            raise ValueError("capture plan scan axes must have scan-point role")
        if self.scan_point_layout.logical_shape != tuple(
            axis.size for axis in scan_axes
        ):
            raise ValueError("capture plan scan layout shape differs from DatasetSchema")
        if schema.point_layout.storage_size != (
            self.scan_point_count * self.readout_events_per_repeat
        ):
            raise ValueError("Dataset PointLayout storage differs from capture plan")
        for assignment in self.assignments:
            scan_multi = self.scan_point_layout.multi_index(
                assignment.pulse_point_index
            )
            logical_by_axis = dict(zip(self.scan_axis_ids, scan_multi, strict=True))
            logical_by_axis[self.readout_event_axis_id] = (
                assignment.readout_event_index
            )
            full_multi = tuple(
                logical_by_axis[axis.axis_id]
                for axis in schema.point_axes
            )
            if schema.point_layout.storage_index(full_multi) != (
                assignment.point_storage_index
            ):
                raise ValueError(
                    "capture plan assignment differs from persisted scan layout"
                )

    def validate_against(
        self,
        artifact: CompiledPulseArtifact,
        dataset_schema: DatasetSchema,
    ) -> None:
        """Recompile from both authorities and require byte-exact plan identity."""

        rebuilt = compile_capture_cell_plan(
            artifact,
            self.trigger_channel,
            dataset_schema,
            readout_event_axis_id=self.readout_event_axis_id,
            scan_point_layout=self.scan_point_layout,
            within_point_grouping=self.within_point_grouping,
        )
        if rebuilt != self:
            raise ValueError("capture cell plan differs from compiled pulse and DatasetSchema")

    @property
    def total_events(self) -> int:
        return len(self.assignments)

    @property
    def expected_cells(self) -> tuple[DatasetCellAddress, ...]:
        return tuple(item.cell for item in self.assignments)

    @property
    def cell_permutation_digest(self) -> str:
        return dataset_cell_permutation_fingerprint(
            self.dataset_schema_fingerprint,
            self.expected_cells,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_digest(compiled_capture_cell_plan_to_tree(self))


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
    scan_axes = tuple(
        axis
        for axis in dataset_schema.point_axes
        if axis.axis_id != readout_event_axis_id
    )
    if any(axis.role != SCAN_POINT for axis in scan_axes):
        raise ValueError("non-event point axes must have role 'scan-point'")
    if scan_point_layout.logical_shape != tuple(axis.size for axis in scan_axes):
        raise ValueError("scan PointLayout shape differs from declared scan axes")
    if scan_point_layout.storage_size != schedule.point_count:
        raise ValueError("scan PointLayout storage differs from pulse point count")
    repeat_count = dataset_schema.repeat_axis.size
    event_count = event_axis.size
    if within_point_grouping is None:
        if repeat_count > 1 and event_count > 1:
            raise ValueError(
                "within_point_grouping is required when repeat and readout-event "
                "axes both exceed one"
            )
        grouping = repeat_major_capture_grouping(repeat_count, event_count)
    else:
        grouping = tuple(tuple(pair) for pair in within_point_grouping)
    if dataset_schema.point_layout.storage_size != schedule.point_count * event_count:
        raise ValueError("Dataset PointLayout storage must equal P * E")
    expected_per_point = repeat_count * event_count
    counts = {point: 0 for point in range(schedule.point_count)}
    for edge in schedule.edges:
        counts[edge.point_index] += 1
    if any(count != expected_per_point for count in counts.values()):
        raise ValueError("every pulse point must emit exactly R * E trigger edges")

    scan_axis_ids = tuple(axis.axis_id for axis in scan_axes)
    assignments = []
    for edge in schedule.edges:
        try:
            repeat_index, event_index = grouping[edge.point_trigger_ordinal]
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "within_point_grouping differs from the trigger schedule"
            ) from exc
        scan_multi = scan_point_layout.multi_index(edge.point_index)
        logical_by_axis = dict(zip(scan_axis_ids, scan_multi, strict=True))
        logical_by_axis[readout_event_axis_id] = event_index
        full_multi = tuple(
            logical_by_axis[axis.axis_id]
            for axis in dataset_schema.point_axes
        )
        assignments.append(
            CaptureCellAssignment(
                edge.trigger_ordinal,
                edge.point_index,
                edge.point_trigger_ordinal,
                repeat_index,
                event_index,
                dataset_schema.point_layout.storage_index(full_multi),
            )
        )
    return CompiledCaptureCellPlan(
        artifact.fingerprint,
        artifact.execution_form,
        trigger_channel,
        canonical_digest(digital_trigger_schedule_to_tree(schedule)),
        dataset_schema.fingerprint,
        dataset_schema.repeat_axis.axis_id,
        scan_axis_ids,
        scan_point_layout,
        readout_event_axis_id,
        repeat_count,
        schedule.point_count,
        event_count,
        grouping,
        tuple(assignments),
    )


def repeat_major_capture_grouping(
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


def compiled_capture_cell_plan_to_tree(
    value: CompiledCaptureCellPlan,
) -> dict[str, object]:
    if not isinstance(value, CompiledCaptureCellPlan):
        raise TypeError("value must be CompiledCaptureCellPlan")
    return {
        "schema": COMPILED_CAPTURE_CELL_PLAN_SCHEMA,
        "compiled_pulse_artifact_digest": value.compiled_pulse_artifact_digest,
        "execution_form": value.execution_form.value,
        "trigger_channel": value.trigger_channel,
        "trigger_schedule_digest": value.trigger_schedule_digest,
        "dataset_schema_fingerprint": value.dataset_schema_fingerprint,
        "repeat_axis_id": value.repeat_axis_id.value,
        "scan_axis_ids": [axis_id.value for axis_id in value.scan_axis_ids],
        "scan_point_layout": point_layout_to_tree(value.scan_point_layout),
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "repeat_count": value.repeat_count,
        "scan_point_count": value.scan_point_count,
        "readout_events_per_repeat": value.readout_events_per_repeat,
        "within_point_grouping": [
            [repeat, event]
            for repeat, event in value.within_point_grouping
        ],
        "assignments": [
            {
                "trigger_ordinal": item.trigger_ordinal,
                "pulse_point_index": item.pulse_point_index,
                "point_trigger_ordinal": item.point_trigger_ordinal,
                "repeat_index": item.repeat_index,
                "readout_event_index": item.readout_event_index,
                "point_storage_index": item.point_storage_index,
            }
            for item in value.assignments
        ],
    }


def compiled_capture_cell_plan_from_tree(tree: object) -> CompiledCaptureCellPlan:
    fields = {
        "schema",
        "compiled_pulse_artifact_digest",
        "execution_form",
        "trigger_channel",
        "trigger_schedule_digest",
        "dataset_schema_fingerprint",
        "repeat_axis_id",
        "scan_axis_ids",
        "scan_point_layout",
        "readout_event_axis_id",
        "repeat_count",
        "scan_point_count",
        "readout_events_per_repeat",
        "within_point_grouping",
        "assignments",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CompiledCaptureCellPlan has an unknown field set")
    if tree["schema"] != COMPILED_CAPTURE_CELL_PLAN_SCHEMA:
        raise ValueError("CompiledCaptureCellPlan schema differs")
    scan_ids = tree["scan_axis_ids"]
    assignments = tree["assignments"]
    grouping = tree["within_point_grouping"]
    if not isinstance(scan_ids, list):
        raise TypeError("scan_axis_ids must be a list")
    if not isinstance(assignments, list):
        raise TypeError("assignments must be a list")
    if not isinstance(grouping, list) or any(
        not isinstance(pair, list) or len(pair) != 2 for pair in grouping
    ):
        raise TypeError("within_point_grouping must be a list of two-item lists")
    assignment_fields = {
        "trigger_ordinal",
        "pulse_point_index",
        "point_trigger_ordinal",
        "repeat_index",
        "readout_event_index",
        "point_storage_index",
    }
    parsed = []
    for item in assignments:
        if not isinstance(item, dict) or set(item) != assignment_fields:
            raise ValueError("CaptureCellAssignment has an unknown field set")
        parsed.append(
            CaptureCellAssignment(
                item["trigger_ordinal"],
                item["pulse_point_index"],
                item["point_trigger_ordinal"],
                item["repeat_index"],
                item["readout_event_index"],
                item["point_storage_index"],
            )
        )
    return CompiledCaptureCellPlan(
        tree["compiled_pulse_artifact_digest"],
        PulseExecutionForm(tree["execution_form"]),
        tree["trigger_channel"],
        tree["trigger_schedule_digest"],
        tree["dataset_schema_fingerprint"],
        AxisId(tree["repeat_axis_id"]),
        tuple(AxisId(item) for item in scan_ids),
        point_layout_from_tree(tree["scan_point_layout"]),
        AxisId(tree["readout_event_axis_id"]),
        tree["repeat_count"],
        tree["scan_point_count"],
        tree["readout_events_per_repeat"],
        tuple(tuple(pair) for pair in grouping),
        tuple(parsed),
    )


def encode_compiled_capture_cell_plan(value: CompiledCaptureCellPlan) -> bytes:
    return encode(compiled_capture_cell_plan_to_tree(value))


def decode_compiled_capture_cell_plan(payload: bytes) -> CompiledCaptureCellPlan:
    value = compiled_capture_cell_plan_from_tree(decode(payload))
    if encode_compiled_capture_cell_plan(value) != bytes(payload):
        raise ValueError("CompiledCaptureCellPlan payload is not canonical")
    return value


__all__ = [
    "COMPILED_CAPTURE_CELL_PLAN_SCHEMA",
    "CaptureCellAssignment",
    "CompiledCaptureCellPlan",
    "compile_capture_cell_plan",
    "compiled_capture_cell_plan_from_tree",
    "compiled_capture_cell_plan_to_tree",
    "decode_compiled_capture_cell_plan",
    "encode_compiled_capture_cell_plan",
    "repeat_major_capture_grouping",
]
