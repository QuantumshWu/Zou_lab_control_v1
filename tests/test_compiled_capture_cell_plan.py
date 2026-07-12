"""Compiled camera edges bind to typed dataset cells without ordinal guessing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    COMPONENT,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.runtime import DatasetCellAddress
from zlc_neutral_atom.timing import (
    CaptureCellAssignment,
    CompiledCaptureCellPlan,
    compile_capture_cell_plan,
    decode_compiled_capture_cell_plan,
    encode_compiled_capture_cell_plan,
    repeat_major_capture_grouping,
)
from zlc_pulse import (
    PulseExecutionForm,
    PulseFieldRef,
    ScanParameter,
    compile_pulse_artifact,
    freeze_scan_table,
    load_pulse_document,
)


ROOT = Path(__file__).parents[1]


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


def scanned_artifact(template: str, *, points: int, four_edges: bool = False):
    document = load_pulse_document(
        ROOT / "pulses" / ("imaging_template.json" if four_edges else template)
    )
    if four_edges:
        low = replace(document.periods[2], period_id="p7")
        high = replace(document.periods[5], period_id="p8")
        document = replace(document, periods=(*document.periods, low, high))
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
    return compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch11",),
    )


def test_static_three_frame_capture_uses_readout_event_not_fake_scan_axis():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    event = axis("capture.event", READOUT_EVENT, 3)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 1),
        (event,),
        PointLayout.rect_c((3,)),
        image_schema(component_count=4),
    )

    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        scan_point_layout=PointLayout.rect_c(()),
    )

    assert plan.expected_cells == tuple(
        DatasetCellAddress(0, event_index) for event_index in range(3)
    )
    assert schema.physical_shape == (1, 3, 2, 3, 4)
    assert schema.cell_schema.data_shape == (2, 3, 4)


def test_r2_p3_e1_uses_physical_point_major_order_without_row_shift():
    artifact = scanned_artifact("T.json", points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (scan, event),
        PointLayout.rect_c((3, 1)),
        image_schema(),
    )

    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        scan_point_layout=PointLayout.rect_c((3,)),
    )

    assert plan.expected_cells == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 1),
        DatasetCellAddress(0, 2),
        DatasetCellAddress(1, 2),
    )


def test_event_axis_position_and_explicit_scan_layout_drive_storage_mapping():
    artifact = scanned_artifact("T.json", points=3, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    scan_y = axis("capture.scan_y", SCAN_POINT, 1)
    scan_x = axis("capture.scan_x", SCAN_POINT, 3)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (event, scan_y, scan_x),
        PointLayout.rect_c((2, 1, 3)),
        image_schema(),
    )
    scan_layout = PointLayout.explicit(
        (1, 3),
        ((0, 2), (0, 0), (0, 1)),
    )

    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        scan_point_layout=scan_layout,
        within_point_grouping=repeat_major_capture_grouping(2, 2),
    )

    assert plan.expected_cells[:4] == (
        DatasetCellAddress(0, 2),
        DatasetCellAddress(0, 5),
        DatasetCellAddress(1, 2),
        DatasetCellAddress(1, 5),
    )
    assert len(set(plan.expected_cells)) == 12


def test_per_point_cardinality_mismatch_is_rejected_even_when_total_can_match():
    artifact = scanned_artifact("T.json", points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 3)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 1),
        (scan, event),
        PointLayout.rect_c((3, 3)),
        image_schema(),
    )
    with pytest.raises(ValueError, match=r"R \* E"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            scan_point_layout=PointLayout.rect_c((3,)),
        )


def test_plan_codec_is_canonical_and_rejects_assignment_reordering():
    artifact = scanned_artifact("T.json", points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (scan, event),
        PointLayout.rect_c((3, 1)),
        image_schema(),
    )
    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        scan_point_layout=PointLayout.rect_c((3,)),
    )
    payload = encode_compiled_capture_cell_plan(plan)
    assert decode_compiled_capture_cell_plan(payload) == plan
    assert decode_compiled_capture_cell_plan(payload).fingerprint == plan.fingerprint

    swapped = list(plan.assignments)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="trigger ordinals"):
        CompiledCaptureCellPlan(
            plan.compiled_pulse_artifact_digest,
            plan.execution_form,
            plan.trigger_channel,
            plan.trigger_schedule_digest,
            plan.dataset_schema_fingerprint,
            plan.repeat_axis_id,
            plan.scan_axis_ids,
            plan.scan_point_layout,
            plan.readout_event_axis_id,
            plan.repeat_count,
            plan.scan_point_count,
            plan.readout_events_per_repeat,
            plan.within_point_grouping,
            tuple(swapped),
        )

    broken = replace(
        plan.assignments[0],
        point_storage_index=plan.assignments[2].point_storage_index,
    )
    with pytest.raises(ValueError, match="cover every dataset cell"):
        CompiledCaptureCellPlan(
            plan.compiled_pulse_artifact_digest,
            plan.execution_form,
            plan.trigger_channel,
            plan.trigger_schedule_digest,
            plan.dataset_schema_fingerprint,
            plan.repeat_axis_id,
            plan.scan_axis_ids,
            plan.scan_point_layout,
            plan.readout_event_axis_id,
            plan.repeat_count,
            plan.scan_point_count,
            plan.readout_events_per_repeat,
            plan.within_point_grouping,
            (broken, *plan.assignments[1:]),
        )


def test_r2_e2_requires_an_explicit_non_guessed_within_point_grouping():
    artifact = scanned_artifact("T.json", points=1, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (event,),
        PointLayout.rect_c((2,)),
        image_schema(),
    )
    with pytest.raises(ValueError, match="within_point_grouping is required"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            scan_point_layout=PointLayout.rect_c(()),
        )


def test_explicit_repeat_major_and_event_major_groupings_remain_distinct():
    artifact = scanned_artifact("T.json", points=1, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (event,),
        PointLayout.rect_c((2,)),
        image_schema(),
    )
    common = {
        "readout_event_axis_id": event.axis_id,
        "scan_point_layout": PointLayout.rect_c(()),
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
    assert repeat_major.expected_cells == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(1, 1),
    )
    assert event_major.expected_cells == (
        DatasetCellAddress(0, 0),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 1),
    )
    assert repeat_major != event_major
    repeat_major.validate_against(artifact, schema)
    event_major.validate_against(artifact, schema)


@pytest.mark.parametrize(
    "grouping",
    [
        ((0, 0), (0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
    ],
)
def test_grouping_must_be_a_complete_unique_r_by_e_permutation(grouping):
    artifact = scanned_artifact("T.json", points=1, four_edges=True)
    event = axis("capture.event", READOUT_EVENT, 2)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (event,),
        PointLayout.rect_c((2,)),
        image_schema(),
    )
    with pytest.raises(ValueError, match="grouping"):
        compile_capture_cell_plan(
            artifact,
            "ch11",
            schema,
            readout_event_axis_id=event.axis_id,
            scan_point_layout=PointLayout.rect_c(()),
            within_point_grouping=grouping,
        )


def test_persisted_scan_layout_is_revalidated_not_only_hashed():
    artifact = scanned_artifact("T.json", points=3)
    scan = axis("capture.scan", SCAN_POINT, 3)
    event = axis("capture.event", READOUT_EVENT, 1)
    schema = DatasetSchema(
        axis("capture.repeat", REPEAT, 2),
        (scan, event),
        PointLayout.rect_c((3, 1)),
        image_schema(),
    )
    plan = compile_capture_cell_plan(
        artifact,
        "ch11",
        schema,
        readout_event_axis_id=event.axis_id,
        scan_point_layout=PointLayout.rect_c((3,)),
    )
    tampered = replace(
        plan,
        scan_point_layout=PointLayout.explicit((3,), ((2,), (0,), (1,))),
    )
    with pytest.raises(ValueError, match="persisted scan layout"):
        tampered.validate_dataset_schema(schema)
    with pytest.raises(ValueError, match="compiled pulse and DatasetSchema"):
        tampered.validate_against(artifact, schema)
