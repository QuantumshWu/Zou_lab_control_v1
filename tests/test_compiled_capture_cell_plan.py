"""Compiled camera edges bind to typed dataset cells without ordinal guessing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_data.axis import (
    AxisId,
    AxisSpec,
    COMPONENT,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from zlc_data.validity import ValidityContract
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress
from zlc_neutral_atom.timing.capture_plan import (
    CaptureCellJoinContract,
    capture_cell_join_contract_from_tree,
    capture_cell_join_contract_to_tree,
    compile_capture_cell_plan,
)
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_pulse import (
    PulseExecutionForm,
    PulseFieldRef,
    ScanParameter,
    compile_pulse_artifact,
    freeze_scan_table,
    load_pulse_document,
)
from zlc_storage import decode, encode


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


def axis(name, role, size):
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def image_schema(*, component_count: int | None = None):
    axes = [
        axis("camera.y", SPATIAL_Y, 2),
        axis("camera.x", SPATIAL_X, 3),
    ]
    if component_count is not None:
        axes.append(axis("camera.component", COMPONENT, component_count))
    return ValueSchema(
        tuple(axes),
        ValidityContract.value(),
        np.dtype("<u2"),
        "count",
    )


def scanned_artifact(
    *,
    points: int,
    four_edges: bool = False,
    execution_form: PulseExecutionForm = PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
):
    document = load_pulse_document(IMAGING_TEMPLATE)
    periods = document.periods[:5]
    if four_edges:
        high = document.periods[5]
        low = document.periods[4]
        periods = (
            *periods,
            high,
            replace(low, period_id="p7"),
            replace(high, period_id="p8"),
            replace(low, period_id="p9"),
        )
    document = replace(document, periods=periods, api_parameters=())
    first = min(document.periods, key=lambda period: float(period.duration))
    parameter = ScanParameter(
        "capture_scan",
        PulseFieldRef("duration", first.period_id),
        "capture scan",
        first.unit,
    )
    document = replace(
        document,
        scan_parameters=(parameter,),
    )
    table, _report = freeze_scan_table(
        document,
        (parameter.parameter_id,),
        tuple((first.duration,) for _ in range(points)),
    )
    document = replace(document, scan_table=table)
    trigger_channels = (
        ()
        if execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
        else ("ch11",)
    )
    return compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=execution_form,
        trigger_channels=trigger_channels,
    )


def test_continuous_scan_has_no_finite_capture_cell_plan():
    artifact = scanned_artifact(
        points=2,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    )
    scan = axis("capture.scan", SCAN_POINT, 2)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 1),
        PointTable(
            2,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates * 2,
                ),
            ),
        ),
        None,
        image_schema(),
    )

    with pytest.raises(ValueError, match="no finite capture cell plan"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            base_point_count=2,
        )


def test_static_three_frame_capture_uses_readout_event_not_fake_scan_axis():
    document = load_pulse_document(IMAGING_TEMPLATE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    event = axis("capture.event", READOUT_EVENT, 3)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 1),
        PointTable(
            3,
            (
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates,
                ),
            ),
        ),
        None,
        image_schema(component_count=4),
    )

    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        base_point_count=1,
    )

    assert tuple(plan.cell_schedule) == tuple(
        DatasetCellAddress(0, event_index) for event_index in range(3)
    )
    assert schema.physical_shape == (1, 3, 2, 3, 4)
    assert schema.cell_schema.data_shape == (2, 3, 4)


def test_r2_p3_e1_uses_physical_point_major_order_without_row_shift():
    artifact = scanned_artifact(points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            3,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates * 3,
                ),
            ),
        ),
        None,
        image_schema(),
    )

    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        base_point_count=3,
    )

    assert tuple(plan.cell_schedule) == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 1),
        DatasetCellAddress(0, 2),
        DatasetCellAddress(1, 2),
    )


def test_authored_point_rows_drive_physical_ordinals_with_grid_topology():
    artifact = scanned_artifact(points=3, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    scan_y = axis("capture.scan_y", SCAN_POINT, 1)
    scan_x = axis("capture.scan_x", SCAN_POINT, 3)
    row_to_cell = (
        (0, 2, 0),
        (0, 2, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    )
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            6,
            (
                PointColumn(
                    scan_y.axis_id,
                    scan_y.name,
                    scan_y.role,
                    PointColumn.NUMERIC,
                    (0,) * 6,
                ),
                PointColumn(
                    scan_x.axis_id,
                    scan_x.name,
                    scan_x.role,
                    PointColumn.NUMERIC,
                    (2, 2, 0, 0, 1, 1),
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    (0, 1) * 3,
                ),
            ),
        ),
        GridTopology(
            (scan_y.axis_id, scan_x.axis_id, event.axis_id),
            (scan_y.coordinates, scan_x.coordinates, event.coordinates),
            row_to_cell,
        ),
        image_schema(),
    )

    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        base_point_count=3,
        within_point_grouping=((0, 0), (0, 1), (1, 0), (1, 1)),
    )

    assert tuple(plan.cell_schedule)[:4] == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(1, 1),
    )
    assert schema.grid_topology is not None
    assert schema.grid_topology.row_to_cell == row_to_cell
    assert len(set(plan.cell_schedule)) == 12


def test_per_point_cardinality_mismatch_is_rejected_even_when_total_can_match():
    artifact = scanned_artifact(points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 3)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 1),
        PointTable(
            9,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    tuple(point for point in scan.coordinates for _ in range(3)),
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates * 3,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    with pytest.raises(ValueError, match="cardinality differs"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            base_point_count=3,
        )


def test_join_contract_codec_is_canonical():
    artifact = scanned_artifact(points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            3,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates * 3,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        base_point_count=3,
    )
    tree = capture_cell_join_contract_to_tree(plan.join_contract)
    payload = encode(tree)
    decoded = capture_cell_join_contract_from_tree(decode(payload))
    assert decoded == plan.join_contract
    assert encode(capture_cell_join_contract_to_tree(decoded)) == payload

    with pytest.raises(ValueError, match="unknown field set"):
        capture_cell_join_contract_from_tree({**tree, "assignments": []})


def test_r2_e2_requires_an_explicit_non_guessed_within_point_grouping():
    artifact = scanned_artifact(points=1, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            2,
            (
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    with pytest.raises(ValueError, match="within_point_grouping is required"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            base_point_count=1,
        )


def test_explicit_repeat_major_and_event_major_groupings_remain_distinct():
    artifact = scanned_artifact(points=1, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            2,
            (
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    common = {
        "readout_event_axis_id": event.axis_id,
        "base_point_count": 1,
    }
    repeat_major = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        within_point_grouping=((0, 0), (0, 1), (1, 0), (1, 1)),
        **common,
    )
    event_major = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        within_point_grouping=((0, 0), (1, 0), (0, 1), (1, 1)),
        **common,
    )
    assert tuple(repeat_major.cell_schedule) == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(1, 1),
    )
    assert tuple(event_major.cell_schedule) == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 1),
    )
    assert repeat_major != event_major
    PulseCaptureBinding(artifact, "ch11", repeat_major)
    PulseCaptureBinding(artifact, "ch11", event_major)
    repeat_major.validate_dataset_schema(schema)
    event_major.validate_dataset_schema(schema)


@pytest.mark.parametrize(
    "grouping",
    [
        ((0, 0), (0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
    ],
)
def test_grouping_must_be_a_complete_unique_r_by_e_permutation(grouping):
    artifact = scanned_artifact(points=1, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            2,
            (
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    with pytest.raises(ValueError, match="grouping"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            base_point_count=1,
            within_point_grouping=grouping,
        )


def test_join_contract_and_compiled_artifact_binding_are_revalidated():
    artifact = scanned_artifact(points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            3,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates * 3,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        base_point_count=3,
    )
    tampered = CaptureCellJoinContract(
        1,
        event.axis_id,
        1,
        plan.join_contract.within_point_grouping,
    )
    schedule = next(
        schedule
        for schedule in artifact.trigger_schedules
        if schedule.channel == "ch11"
    )
    with pytest.raises(ValueError, match="base points × events"):
        tuple(tampered.iter_cell_schedule(schedule, schema))

    other_artifact = scanned_artifact(points=2)
    other_scan = axis("capture.scan", SCAN_POINT, 2)
    other_schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        PointTable(
            2,
            (
                PointColumn(
                    other_scan.axis_id,
                    other_scan.name,
                    other_scan.role,
                    PointColumn.NUMERIC,
                    other_scan.coordinates,
                ),
                PointColumn(
                    event.axis_id,
                    event.name,
                    event.role,
                    PointColumn.NUMERIC,
                    event.coordinates * 2,
                ),
            ),
        ),
        None,
        image_schema(),
    )
    other_plan = compile_capture_cell_plan(
        other_artifact,
        "ch11",
        other_schema,
        readout_event_axis_id=event.axis_id,
        base_point_count=2,
    )
    with pytest.raises(ValueError, match="another compiled artifact"):
        PulseCaptureBinding(
            artifact,
            "ch11",
            other_plan,
        )
