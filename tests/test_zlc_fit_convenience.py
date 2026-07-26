"""Independent contracts for fit references, role selection, and result binding."""

from __future__ import annotations

import ast
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from zlc_data import (
    COMPONENT,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisLayout,
    AxisSpec,
    BlockId,
    DatasetComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    bind_fit,
    fit_spec_for,
    validate_fit_result_source_binding,
)
from zlc_data.fit_problem import build_fit_problem
from zlc_data.fit_model import evaluate_fit_model


def test_production_fit_consumers_use_the_public_zlc_data_facade():
    repository = Path(__file__).resolve().parents[1]
    roots = (
        repository / "zlc_neutral_atom",
        repository / "zlc_frontend",
        repository / "zlc_workbench",
        repository / "Zou_lab_control" / "api",
    )
    implementation_modules = {
        "zlc_data.fit_codec",
        "zlc_data.fit_contract",
        "zlc_data.fit_model",
        "zlc_data.fit_problem",
        "zlc_data.fit_solver",
    }
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in implementation_modules:
                    violations.append(f"{path.relative_to(repository)}:{node.lineno}")
                elif isinstance(node, ast.Import) and any(
                    alias.name in implementation_modules for alias in node.names
                ):
                    violations.append(f"{path.relative_to(repository)}:{node.lineno}")
    assert violations == []


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


def _schema(
    *,
    repeat: int = 3,
    point_axes: tuple[AxisSpec, ...] = (),
    data_axes: tuple[AxisSpec, ...] = (),
) -> DatasetSchema:
    cell_schema = (
        ValueSchema(data_axes, ValidityContract.value(), np.dtype("<f8"), "count")
        if data_axes
        else ValueSchema.scalar(np.dtype("<f8"), "count")
    )
    return DatasetSchema(
        _axis("repeat", REPEAT, repeat),
        point_axes,
        PointLayout.rect_c(tuple(axis.size for axis in point_axes)),
        cell_schema,
    )


def test_authoritative_curve_auto_requires_one_complete_semantic_matching():
    scan = _axis(
        "detuning",
        SCAN_POINT,
        11,
        coordinates=np.linspace(-2.0, 2.0, 11),
        unit="MHz",
    )
    spectrum = _axis(
        "spectrum",
        SPECTRAL,
        7,
        coordinates=np.linspace(1.0, 7.0, 7),
        unit="MHz",
    )
    y = _axis("camera-y", SPATIAL_Y, 5, coordinates=range(5), unit="px")
    x = _axis("camera-x", SPATIAL_X, 4, coordinates=range(4), unit="px")
    schema = _schema(point_axes=(scan, spectrum), data_axes=(y, x))

    with pytest.raises(ValueError, match="ambiguous declared-role axis matchings"):
        fit_spec_for(schema, "gaussian_offset")

    explicit_scan = fit_spec_for(
        schema,
        "gaussian_offset",
        fit_axis_ids=(scan.axis_id,),
    )
    assert explicit_scan.batch_axis_ids == (
        schema.repeat_axis.axis_id,
        spectrum.axis_id,
        y.axis_id,
        x.axis_id,
    )
    assert explicit_scan.committed_transform is None

    explicit_spectrum = fit_spec_for(
        schema,
        "gaussian_offset",
        fit_axis_ids=(spectrum.axis_id,),
    )
    assert explicit_spectrum.batch_axis_ids == (
        schema.repeat_axis.axis_id,
        scan.axis_id,
        y.axis_id,
        x.axis_id,
    )


def test_authoritative_curve_auto_uses_the_only_semantically_valid_axis():
    scan = _axis(
        "detuning",
        SCAN_POINT,
        11,
        coordinates=np.linspace(-2.0, 2.0, 11),
        unit="MHz",
    )
    schema = _schema(point_axes=(scan,))

    spec = fit_spec_for(schema, "gaussian_offset")

    assert spec.fit_axis_ids == (scan.axis_id,)
    assert spec.batch_axis_ids == (schema.repeat_axis.axis_id,)


@pytest.mark.parametrize(
    "axis_order",
    tuple(permutations(("component", "site", "readout"))),
)
def test_arbitrary_multidimensional_data_axes_remain_named_batches(
    axis_order,
):
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
    schema = DatasetSchema(
        _axis("repeat", REPEAT, 2),
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            data_axes,
            ValidityContract.components(
                *(axis.axis_id for axis in data_axes)
            ),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.arange(
        np.prod(schema.physical_shape),
        dtype=np.float64,
    ).reshape(schema.physical_shape)
    validity = np.ones(schema.physical_shape, dtype=bool)
    invalid_by_name = {"component": 1, "site": 2, "readout": 3}
    invalid_data_index = tuple(invalid_by_name[name] for name in axis_order)
    invalid_repeat = 1
    invalid_scan = 2
    validity[(invalid_repeat, invalid_scan, *invalid_data_index)] = False
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

    spec = fit_spec_for(schema, "gaussian_offset")
    problem = build_fit_problem(bind_fit(spec, schema), snapshot)

    assert spec.fit_axis_ids == (scan.axis_id,)
    assert spec.batch_axis_ids == (
        schema.repeat_axis.axis_id,
        *(axis.axis_id for axis in data_axes),
    )
    assert problem.batch_layout.logical_shape == (
        2,
        *(axis.size for axis in data_axes),
    )
    assert np.all(problem.present_observation_counts == scan.size)
    assert np.count_nonzero(problem.used_observation_counts == scan.size - 1) == 1
    assert np.count_nonzero(problem.used_observation_counts == scan.size) == (
        problem.batch_layout.storage_size - 1
    )
    for storage_index, batch_index in enumerate(
        np.ndindex(problem.batch_layout.logical_shape)
    ):
        assert problem.batch_layout.multi_index(storage_index) == batch_index
        expected = values[(batch_index[0], slice(None), *batch_index[1:])]
        if batch_index == (invalid_repeat, *invalid_data_index):
            expected = np.delete(expected, invalid_scan)
        start = int(problem.batch_offsets[storage_index])
        stop = int(problem.batch_offsets[storage_index + 1])
        np.testing.assert_array_equal(
            problem.observations[start:stop],
            expected,
        )


def test_two_dimensional_model_selects_declared_spatial_roles_and_batches_scan():
    scan = _axis(
        "detuning",
        SCAN_POINT,
        3,
        coordinates=(-1.0, 0.0, 1.0),
    )
    x = _axis(
        "camera-x",
        SPATIAL_X,
        9,
        coordinates=range(9),
        unit="px",
        frame="camera",
    )
    y = _axis(
        "camera-y",
        SPATIAL_Y,
        7,
        coordinates=range(7),
        unit="px",
        frame="camera",
    )
    schema = _schema(point_axes=(scan,), data_axes=(y, x))

    spec = fit_spec_for(schema, "radial_gaussian_center")

    assert spec.fit_axis_ids == (x.axis_id, y.axis_id)
    assert spec.batch_axis_ids == (schema.repeat_axis.axis_id, scan.axis_id)


def test_role_driven_selection_rejects_semantic_ambiguity_and_incomplete_axes():
    first = _axis("scan-a", SCAN_POINT, 4, coordinates=range(4))
    second = _axis("scan-b", SCAN_POINT, 5, coordinates=range(5))
    ambiguous = _schema(point_axes=(first, second))
    with pytest.raises(ValueError, match="ambiguous"):
        fit_spec_for(ambiguous, "gaussian_offset")

    x = _axis(
        "x",
        SPATIAL_X,
        5,
        coordinates=range(5),
        unit="px",
        frame="camera",
    )
    y = _axis(
        "y",
        SPATIAL_Y,
        5,
        coordinates=range(5),
        unit="px",
        frame="camera",
    )
    image = _schema(data_axes=(x, y))
    with pytest.raises(ValueError, match="requires 2 fit axes"):
        fit_spec_for(
            image,
            "radial_gaussian_center",
            fit_axis_ids=(x.axis_id,),
        )
    with pytest.raises(ValueError, match="cover every information axis exactly"):
        fit_spec_for(
            image,
            "radial_gaussian_center",
            fit_axis_ids=(x.axis_id, AxisId("missing-y")),
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
    schema = _schema(repeat=2, point_axes=(scan,))
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
    bound = bind_fit(
        fit_spec_for(snapshot.block.schema, "gaussian_offset"),
        snapshot.block.schema,
    )
    result = bound.run(snapshot)
    validate_fit_result_source_binding(
        result,
        snapshot.ref,
        snapshot.block.schema,
    )

    changed_axis = replace(result.fit_axis_specs[0], name="forged detuning")
    axis_drift = replace(
        result,
        fit_axis_specs=(changed_axis,),
    )
    with pytest.raises(ValueError, match="axis specifications"):
        validate_fit_result_source_binding(
            axis_drift,
            snapshot.ref,
            snapshot.block.schema,
        )

    reversed_layout = AxisLayout.explicit((2,), ((1,), (0,)))
    layout_drift = replace(result, batch_layout=reversed_layout)
    with pytest.raises(ValueError, match="batch layout"):
        validate_fit_result_source_binding(
            layout_drift,
            snapshot.ref,
            snapshot.block.schema,
        )

    present = np.asarray(result.present_observation_counts).copy()
    present[0] += 1
    count_drift = replace(result, present_observation_counts=present)
    with pytest.raises(ValueError, match="present_observation_counts"):
        validate_fit_result_source_binding(
            count_drift,
            snapshot.ref,
            snapshot.block.schema,
        )
