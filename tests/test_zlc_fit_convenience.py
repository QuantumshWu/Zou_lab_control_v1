"""Independent contracts for explicit Fit sources and result binding."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from zlc_data.axis import (
    COMPONENT,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSourceRef,
    AxisSpec,
    CoordinateFrameId,
)
from zlc_data.fit import fit_spec_for
from zlc_data.fit_contract import FitBatchStatus
from zlc_data.fit_model import evaluate_fit_model
from zlc_data.fit_problem import (
    bind_fit,
    validate_fit_result_source_binding,
)
from zlc_data.layout import AxisLayout
from zlc_data.schema import DatasetSchema, PointColumn, PointTable, ValueSchema
from zlc_data.validity import DatasetComponentValidity, VALID, ValidityContract
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)


def _axis(
    identity: str,
    role,
    size: int,
    *,
    coordinates=None,
    unit: str | None = None,
    frame: str | None = None,
) -> AxisSpec:
    return AxisSpec(
        AxisId(identity),
        identity,
        role,
        size,
        None if coordinates is None else tuple(coordinates),
        unit,
        None if frame is None else CoordinateFrameId(frame),
    )


@pytest.mark.parametrize(
    "axis_order",
    tuple(permutations(("component", "site", "readout"))),
)
def test_arbitrary_multidimensional_data_axes_remain_named_batches(axis_order):
    scan = _axis(
        "detuning",
        SCAN_POINT,
        6,
        coordinates=np.linspace(-1.0, 1.0, 6),
    )
    by_name = {
        "component": _axis("component", COMPONENT, 2),
        "site": _axis("site", SITE, 3),
        "readout": _axis("readout", READOUT_EVENT, 4),
    }
    data_axes = tuple(by_name[name] for name in axis_order)
    repeat = _axis("repeat", REPEAT, 2)
    schema = DatasetSchema(
        repeat,
        PointTable(
            scan.size,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                ),
            ),
        ),
        None,
        ValueSchema(
            data_axes,
            ValidityContract.components(*(axis.axis_id for axis in data_axes)),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.zeros(schema.physical_shape, dtype=np.float64)
    validity = np.zeros(schema.physical_shape, dtype=bool)
    invalid_by_name = {"component": 1, "site": 2, "readout": 3}
    selected_data_index = tuple(invalid_by_name[name] for name in axis_order)
    selected_repeat = 1
    coordinates = np.asarray(scan.coordinates)
    expected = (2.5, 0.7, 0.8, 0.25)
    values[(selected_repeat, slice(None), *selected_data_index)] = evaluate_fit_model(
        "gaussian_offset",
        (coordinates,),
        expected,
    )
    validity[(selected_repeat, slice(None), *selected_data_index)] = True
    block = DataBlock(
        BlockId("fit-many-data-axes"),
        DatasetRevision(1),
        values,
        DatasetComponentValidity(
            tuple(axis.axis_id for axis in data_axes),
            validity,
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("fit-many-data-axes-generation")),
        block,
    )
    independent = (AxisSourceRef.point_coordinate(scan.axis_id),)
    batch = (
        AxisSourceRef.tensor(repeat.axis_id),
        *(AxisSourceRef.tensor(axis.axis_id) for axis in data_axes),
    )

    spec = fit_spec_for(
        schema,
        "gaussian_offset",
        independent_sources=independent,
        batch_sources=batch,
    )
    result = bind_fit(spec, schema).run(snapshot)

    assert spec.independent_sources == independent
    assert spec.batch_sources == batch
    assert result.batch_layout.logical_shape == (
        2,
        *(axis.size for axis in data_axes),
    )
    assert np.all(result.present_observation_counts == scan.size)
    assert np.count_nonzero(result.used_observation_counts == scan.size) == 1
    assert np.count_nonzero(result.used_observation_counts == 0) == (
        result.batch_layout.storage_size - 1
    )
    for storage_index, batch_index in enumerate(
        np.ndindex(result.batch_layout.logical_shape)
    ):
        assert result.batch_layout.multi_index(storage_index) == batch_index
    selected_batch = (selected_repeat, *selected_data_index)
    selected_storage = next(
        index
        for index in range(result.batch_layout.storage_size)
        if result.batch_layout.multi_index(index) == selected_batch
    )
    assert result.statuses[selected_storage] is FitBatchStatus.CONVERGED
    np.testing.assert_allclose(
        result.parameter_values[selected_storage],
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
    assert all(
        status is FitBatchStatus.NO_VALID_DATA
        for index, status in enumerate(result.statuses)
        if index != selected_storage
    )


def _gaussian_snapshot() -> OwnedSnapshot:
    coordinates = np.linspace(-3.0, 3.0, 41)
    scan = _axis(
        "detuning",
        SCAN_POINT,
        coordinates.size,
        coordinates=coordinates,
        unit="MHz",
    )
    repeat = _axis("repeat", REPEAT, 2)
    schema = DatasetSchema(
        repeat,
        PointTable(
            scan.size,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                    scan.unit,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    curve = evaluate_fit_model(
        "gaussian_offset",
        (coordinates,),
        (2.5, 0.7, 0.8, 0.25),
    )
    block = DataBlock(
        BlockId("fit-binding-source"),
        DatasetRevision(2),
        np.tile(curve, (2, 1))[..., np.newaxis],
        VALID,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId("fit-binding-generation")),
        block,
    )


def test_result_binding_uses_schema_facts_without_repacking_values():
    snapshot = _gaussian_snapshot()
    schema = snapshot.block.schema
    scan_id = schema.point_table.columns[0].coordinate_id
    bound = fit_spec_for(
        schema,
        "gaussian_offset",
        independent_sources=(AxisSourceRef.point_coordinate(scan_id),),
        batch_sources=(AxisSourceRef.tensor(schema.repeat_axis.axis_id),),
    )
    bound = bind_fit(bound, schema)
    result = bound.run(snapshot)
    validate_fit_result_source_binding(result, snapshot.ref, schema)

    changed_axis = replace(result.fit_axis_specs[0], name="forged detuning")
    axis_drift = replace(result, fit_axis_specs=(changed_axis,))
    with pytest.raises(ValueError, match="axis specifications"):
        validate_fit_result_source_binding(axis_drift, snapshot.ref, schema)

    reversed_layout = AxisLayout.explicit((2,), ((1,), (0,)))
    layout_drift = replace(result, batch_layout=reversed_layout)
    with pytest.raises(ValueError, match="batch layout"):
        validate_fit_result_source_binding(layout_drift, snapshot.ref, schema)

    present = np.asarray(result.present_observation_counts).copy()
    present[0] += 1
    count_drift = replace(result, present_observation_counts=present)
    with pytest.raises(ValueError, match="present_observation_counts"):
        validate_fit_result_source_binding(count_drift, snapshot.ref, schema)
