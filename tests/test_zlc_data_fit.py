"""Named-axis fit contracts, packing, models, budgets, and artifact codecs."""

from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
import time

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisLayoutMode,
    AxisRoleId,
    AxisSpec,
    BlockId,
    BoundFit,
    CellValidity,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitNumericPolicy,
    FitParameterConstraint,
    FitSpec,
    OwnedSnapshot,
    PointLayout,
    Select,
    Selection,
    StreamGenerationId,
    TransformOrigin,
    TransformRevision,
    TypedCodecError,
    VALID,
    ValidityContract,
    ValueSchema,
    bind_fit,
    build_fit_problem,
    commit_transform,
    decode_fit_result_batch,
    decode_fit_spec,
    encode_fit_result_batch,
    encode_fit_spec,
    evaluate_fit_model,
    fit_analysis,
    fit_model_catalog,
    fit_model_definition,
)
from zlc_data.fit_codec import fit_result_batch_to_tree, fit_spec_to_tree
from zlc_storage.canonical import encode


def axis(
    name: str,
    role: AxisRoleId,
    size: int,
    *,
    coordinates=None,
    unit: str | None = None,
    frame: str | None = None,
) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        None if coordinates is None else tuple(coordinates),
        unit,
        None if frame is None else CoordinateFrameId(frame),
    )


def snapshot_for(
    *,
    repeat: int,
    point_axes: tuple[AxisSpec, ...],
    point_layout: PointLayout,
    data_axes: tuple[AxisSpec, ...] = (),
    values,
    validity=VALID,
    validity_contract: ValidityContract | None = None,
    value_unit: str | None = "count",
    dtype=np.dtype("<f8"),
    block_id: str = "fit-source",
) -> OwnedSnapshot:
    repeat_axis = axis("repeat", REPEAT, repeat)
    schema = DatasetSchema(
        repeat_axis,
        point_axes,
        point_layout,
        ValueSchema(
            data_axes,
            validity_contract or ValidityContract.value(),
            np.dtype(dtype),
            value_unit,
        ),
    )
    block = DataBlock(
        BlockId(block_id),
        DatasetRevision(3),
        np.asarray(values, dtype=dtype),
        validity,
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("fit-generation")), block)


def gaussian_snapshot(*, repeat: int = 2) -> tuple[OwnedSnapshot, AxisSpec]:
    coordinates = np.linspace(-4.0, 4.0, 81)
    scan = axis("detuning", SCAN_POINT, coordinates.size, coordinates=coordinates, unit="MHz")
    signal = evaluate_fit_model(
        "gaussian_offset",
        (coordinates,),
        (3.0, 1.2, 0.8, 0.7),
    )
    snapshot = snapshot_for(
        repeat=repeat,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=np.tile(signal, (repeat, 1)),
    )
    return snapshot, scan


def gaussian_spec(snapshot: OwnedSnapshot, scan: AxisSpec, **kwargs) -> FitSpec:
    return FitSpec(
        snapshot.block.schema.fingerprint,
        kwargs.pop("committed_transform", None),
        (scan.axis_id,),
        kwargs.pop("batch_axis_ids", (snapshot.block.schema.repeat_axis.axis_id,)),
        "gaussian_offset",
        constraints=kwargs.pop("constraints", ()),
        numeric_policy=kwargs.pop("numeric_policy", FitNumericPolicy()),
        **kwargs,
    )


def test_catalog_is_closed_canonical_and_has_no_legacy_aliases():
    assert tuple(model.model_id for model in fit_model_catalog()) == (
        "lorentzian",
        "gaussian_offset",
        "zeeman_double_lorentzian",
        "damped_sine",
        "exponential_decay",
        "radial_gaussian_center",
    )
    for legacy in ("lorent", "lorent_zeeman", "rabi", "decay", "center"):
        with pytest.raises(ValueError, match="unknown fit model"):
            fit_model_definition(legacy)


def test_importing_zlc_data_keeps_scipy_solver_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, zlc_data; assert 'scipy' not in sys.modules; "
            "assert callable(zlc_data.fit_analysis)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("model_id", "x", "parameters"),
    (
        ("lorentzian", np.linspace(-4, 4, 101), (0.4, 1.2, 3.0, 0.7)),
        ("gaussian_offset", np.linspace(-4, 4, 101), (3.0, 0.7, 0.9, 0.4)),
        (
            "zeeman_double_lorentzian",
            np.linspace(-4, 4, 121),
            (0.2, 0.7, 2.0, 0.4, 2.1),
        ),
        ("damped_sine", np.linspace(0, 4, 160), (2.0, 0.3, 0.7, 3.2, 0.5)),
        ("exponential_decay", np.linspace(0, 4, 100), (2.0, 0.3, 1.2)),
    ),
)
def test_every_curve_model_recovers_clean_synthetic_data(model_id, x, parameters):
    scan = axis("scan", SCAN_POINT, x.size, coordinates=x, unit="ms")
    values = evaluate_fit_model(model_id, (x,), parameters).reshape(1, -1)
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=values,
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        model_id,
    )
    result = fit_analysis(bind_fit(spec, snapshot.block.schema), snapshot)
    assert result.statuses == (FitBatchStatus.SUCCESS,)
    np.testing.assert_allclose(result.parameter_values[0], parameters, rtol=2e-5, atol=2e-5)


def test_binding_is_axis_total_role_checked_and_declared_coordinates_are_not_ignored():
    snapshot, scan = gaussian_snapshot()
    with pytest.raises(ValueError, match="do not cover the effective schema"):
        bind_fit(
            replace(gaussian_spec(snapshot, scan), batch_axis_ids=()),
            snapshot.block.schema,
        )

    site_like = axis("site_like", SITE, scan.size, coordinates=range(scan.size))
    wrong = snapshot_for(
        repeat=1,
        point_axes=(site_like,),
        point_layout=PointLayout.rect_c((site_like.size,)),
        values=np.ones((1, site_like.size)),
    )
    wrong_spec = FitSpec(
        wrong.block.schema.fingerprint,
        None,
        (site_like.axis_id,),
        (wrong.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    with pytest.raises(ValueError, match="does not satisfy model roles"):
        bind_fit(wrong_spec, wrong.block.schema)

    labels = axis("labels", SCAN_POINT, 6, coordinates=("a", "b", "c", "d", "e", "f"))
    labelled = snapshot_for(
        repeat=1,
        point_axes=(labels,),
        point_layout=PointLayout.rect_c((labels.size,)),
        values=np.ones((1, labels.size)),
    )
    labelled_spec = FitSpec(
        labelled.block.schema.fingerprint,
        None,
        (labels.axis_id,),
        (labelled.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    with pytest.raises(TypeError, match="declared coordinates.*entirely numeric"):
        bind_fit(labelled_spec, labelled.block.schema)


def test_radial_center_requires_x_then_y_with_compatible_units_and_frames():
    x = axis("x", SPATIAL_X, 9, coordinates=range(9), unit="px", frame="camera")
    y = axis("y", SPATIAL_Y, 7, coordinates=range(7), unit="px", frame="camera")
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=np.ones((1, 1, x.size, y.size)),
    )
    base = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "radial_gaussian_center",
    )
    bind_fit(base, snapshot.block.schema)
    with pytest.raises(ValueError, match="does not satisfy model roles"):
        bind_fit(replace(base, fit_axis_ids=(y.axis_id, x.axis_id)), snapshot.block.schema)

    bad_y = axis("y", SPATIAL_Y, 7, coordinates=range(7), unit="mm", frame="other")
    bad = snapshot_for(
        repeat=1,
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, bad_y),
        values=np.ones((1, 1, x.size, bad_y.size)),
    )
    bad_spec = replace(
        base,
        input_schema_fingerprint=bad.block.schema.fingerprint,
        fit_axis_ids=(x.axis_id, bad_y.axis_id),
    )
    with pytest.raises(ValueError, match="compatible coordinate units"):
        bind_fit(bad_spec, bad.block.schema)


def test_identity_and_committed_transform_keep_complete_source_lineage():
    snapshot, scan = gaussian_snapshot(repeat=2)
    identity_bound = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema)
    identity = identity_bound.run(snapshot)
    assert identity.source_ref == snapshot.ref
    assert identity.spec.committed_transform is None
    assert identity.effective_schema_fingerprint == snapshot.block.schema.fingerprint

    committed = commit_transform(
        snapshot.block.schema,
        DataTransformSpec(
            (Select(Selection.index(snapshot.block.schema.repeat_axis.axis_id, 0)),)
        ),
        revision=TransformRevision(1),
        origin=TransformOrigin.USER,
    )
    transformed_spec = gaussian_spec(
        snapshot,
        scan,
        committed_transform=committed,
        batch_axis_ids=(),
    )
    transformed = bind_fit(transformed_spec, snapshot.block.schema).run(snapshot)
    assert transformed.source_ref == snapshot.ref
    assert transformed.spec.committed_transform == committed
    assert transformed.effective_schema_fingerprint == committed.output_schema_fingerprint
    np.testing.assert_allclose(transformed.parameter_values[0], (3.0, 1.2, 0.8, 0.7))


def test_public_bound_fit_cannot_forge_effective_axis_metadata():
    snapshot, scan = gaussian_snapshot(repeat=1)
    bound = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema)
    shifted = replace(
        scan,
        coordinates=tuple(float(value) + 100.0 for value in scan.coordinates),
    )
    forged_schema = replace(
        bound.effective_schema,
        cell_axes=(snapshot.block.schema.repeat_axis, shifted),
    )
    with pytest.raises(ValueError, match="not derived from FitSpec authority"):
        BoundFit(bound.spec, bound.expected_schema, forged_schema, bound.model)


def test_grid_site_batch_uses_component_validity_without_collapsing_sites():
    x = np.linspace(-3.0, 3.0, 9)
    scan = axis("scan", SCAN_POINT, x.size, coordinates=x, unit="MHz")
    site = axis("site", SITE, 3)
    parameters = ((2.0, 0.2, 0.7, -0.4), (3.0, 0.5, 0.9, 0.1), (1.5, 0.7, 0.5, 0.8))
    values = np.stack(
        [evaluate_fit_model("gaussian_offset", (x,), item) for item in parameters],
        axis=-1,
    ).reshape(1, x.size, site.size)
    mask = np.ones((1, x.size, site.size), dtype=bool)
    mask[:, :, 1] = False
    mask[:, 3:, 2] = False
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        data_axes=(site,),
        values=values,
        validity=ComponentValidity((site.axis_id,), mask),
        validity_contract=ValidityContract.components(site.axis_id),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id, site.axis_id),
        "gaussian_offset",
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (
        FitBatchStatus.SUCCESS,
        FitBatchStatus.NO_VALID_DATA,
        FitBatchStatus.INSUFFICIENT_POINTS,
    )
    np.testing.assert_array_equal(result.present_observation_counts, (9, 9, 9))
    np.testing.assert_array_equal(result.valid_observation_counts, (9, 0, 3))
    np.testing.assert_array_equal(result.used_observation_counts, (9, 0, 3))
    np.testing.assert_allclose(result.parameter_values[0], parameters[0], rtol=1e-5, atol=1e-5)


def test_sparse_missing_batch_is_absent_while_present_invalid_batch_is_failure_and_product():
    group = axis("group", SITE, 3)
    scan = axis("scan", SCAN_POINT, 6, coordinates=np.linspace(-2, 2, 6))
    mapping = tuple((group_index, scan_index) for group_index in (0, 2) for scan_index in range(6))
    layout = PointLayout.explicit((group.size, scan.size), mapping)
    scan_indices = layout.axis_indices(1)
    values = evaluate_fit_model(
        "gaussian_offset",
        (np.asarray(scan.coordinates)[scan_indices],),
        (2.0, 0.4, 0.8, 0.2),
    )
    valid = np.ones((2, layout.storage_size), dtype=bool)
    group_indices = layout.axis_indices(0)
    valid[:, group_indices == 2] = False
    snapshot = snapshot_for(
        repeat=2,
        point_axes=(group, scan),
        point_layout=layout,
        values=np.tile(values, (2, 1)),
        validity=CellValidity(valid),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id, group.axis_id),
        "gaussian_offset",
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.batch_layout.mode is AxisLayoutMode.PRODUCT
    assert tuple(result.batch_layout.multi_index(i) for i in range(4)) == (
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 2),
    )
    assert (0, 1) not in tuple(result.batch_layout.multi_index(i) for i in range(4))
    assert result.statuses == (
        FitBatchStatus.SUCCESS,
        FitBatchStatus.NO_VALID_DATA,
        FitBatchStatus.SUCCESS,
        FitBatchStatus.NO_VALID_DATA,
    )


@pytest.mark.parametrize(
    ("point_layout", "expected_mode"),
    (
        (PointLayout.rect_c((2, 2, 6)), AxisLayoutMode.RECT_C),
        (PointLayout.rect_f((2, 2, 6)), AxisLayoutMode.RECT_F),
    ),
)
def test_batch_layout_preserves_rectangular_c_and_f(point_layout, expected_mode):
    first = axis("first", SITE, 2)
    second = axis("second", AxisRoleId("grid-y"), 2)
    scan = axis("scan", SCAN_POINT, 6, coordinates=np.linspace(-2, 2, 6))
    scan_indices = point_layout.axis_indices(2)
    signal = evaluate_fit_model(
        "gaussian_offset",
        (np.asarray(scan.coordinates)[scan_indices],),
        (2.0, 0.4, 0.8, 0.2),
    )
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(first, second, scan),
        point_layout=point_layout,
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (
            snapshot.block.schema.repeat_axis.axis_id,
            first.axis_id,
            second.axis_id,
        ),
        "gaussian_offset",
    )
    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert problem.batch_layout.mode is expected_mode


def test_nonfactor_sparse_mapping_stays_explicit_after_authoritative_repeat_selection():
    first = axis("first", SITE, 2)
    second = axis("second", AxisRoleId("grid-y"), 2)
    scan = axis("scan", SCAN_POINT, 6, coordinates=np.linspace(-2, 2, 6))
    pairs = ((0, 0), (1, 0), (1, 1))
    mapping = tuple((*pair, point) for pair in pairs for point in range(scan.size))
    layout = PointLayout.explicit((2, 2, scan.size), mapping)
    scan_indices = layout.axis_indices(2)
    values = evaluate_fit_model(
        "gaussian_offset",
        (np.asarray(scan.coordinates)[scan_indices],),
        (2.0, 0.4, 0.8, 0.2),
    ).reshape(1, -1)
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(first, second, scan),
        point_layout=layout,
        values=values,
    )
    committed = commit_transform(
        snapshot.block.schema,
        DataTransformSpec(
            (Select(Selection.index(snapshot.block.schema.repeat_axis.axis_id, 0)),)
        ),
        revision=TransformRevision(1),
        origin=TransformOrigin.USER,
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        committed,
        (scan.axis_id,),
        (first.axis_id, second.axis_id),
        "gaussian_offset",
    )
    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert problem.batch_layout.mode is AxisLayoutMode.EXPLICIT
    assert problem.batch_layout.storage_to_multi == pairs


def test_large_radial_image_is_sampled_before_coordinate_packing_and_fits_2d():
    x_values = np.linspace(-3.0, 3.0, 160)
    y_values = np.linspace(-2.0, 2.0, 120)
    x = axis("x", SPATIAL_X, x_values.size, coordinates=x_values, unit="mm", frame="camera")
    y = axis("y", SPATIAL_Y, y_values.size, coordinates=y_values, unit="mm", frame="camera")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    expected = (4.0, 0.5, 1.1, 0.4, -0.3)
    image = evaluate_fit_model("radial_gaussian_center", (xx, yy), expected)
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )
    policy = FitNumericPolicy(sample_budget_per_batch=1_000)
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "radial_gaussian_center",
        numeric_policy=policy,
    )
    bound = bind_fit(spec, snapshot.block.schema)
    problem = build_fit_problem(bound, snapshot)
    assert problem.present_observation_counts[0] == image.size
    assert problem.valid_observation_counts[0] == image.size
    assert problem.used_observation_counts[0] == 1_000
    assert problem.observations.size == 1_000
    assert tuple(values.size for values in problem.independent_values) == (1_000, 1_000)
    result = bound.run(snapshot)
    assert result.statuses == (FitBatchStatus.SUCCESS,)
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-3, atol=2e-3)


def test_2d_sampling_is_cartesian_not_a_rank_deficient_flattened_diagonal():
    x_values = np.linspace(-3.0, 3.0, 100)
    y_values = np.linspace(-2.0, 2.0, 100)
    x = axis("x", SPATIAL_X, 100, coordinates=x_values, unit="mm", frame="camera")
    y = axis("y", SPATIAL_Y, 100, coordinates=y_values, unit="mm", frame="camera")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    expected = (4.0, 0.5, 1.1, 0.4, -0.3)
    image = evaluate_fit_model("radial_gaussian_center", (xx, yy), expected)
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=image.reshape(1, 1, 100, 100),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "radial_gaussian_center",
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=100),
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.SUCCESS,)
    assert result.r_squared[0] > 0.999999
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-4, atol=2e-4)


def test_2d_point_axes_sampling_recovers_center_and_ignores_storage_permutation():
    x_values = np.linspace(-3.0, 3.0, 100)
    y_values = np.linspace(-2.0, 2.0, 100)
    x = axis("x", SPATIAL_X, 100, coordinates=x_values, unit="mm", frame="camera")
    y = axis("y", SPATIAL_Y, 100, coordinates=y_values, unit="mm", frame="camera")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    expected = (4.0, 0.5, 1.1, 0.4, -0.3)
    logical_image = evaluate_fit_model("radial_gaussian_center", (xx, yy), expected)

    def run(layout: PointLayout) -> object:
        physical_values = np.fromiter(
            (logical_image[layout.multi_index(row)] for row in range(layout.storage_size)),
            dtype=np.float64,
            count=layout.storage_size,
        )
        snapshot = snapshot_for(
            repeat=1,
            point_axes=(x, y),
            point_layout=layout,
            values=physical_values.reshape(1, -1),
        )
        spec = FitSpec(
            snapshot.block.schema.fingerprint,
            None,
            (x.axis_id, y.axis_id),
            (snapshot.block.schema.repeat_axis.axis_id,),
            "radial_gaussian_center",
            numeric_policy=FitNumericPolicy(sample_budget_per_batch=100),
        )
        return bind_fit(spec, snapshot.block.schema).run(snapshot)

    ordered = run(PointLayout.rect_c((100, 100)))
    mapping = list(PointLayout.rect_c((100, 100)).multi_index(i) for i in range(10_000))
    np.random.default_rng(11).shuffle(mapping)
    permuted = run(PointLayout.explicit((100, 100), tuple(mapping)))
    assert ordered.statuses == permuted.statuses == (FitBatchStatus.SUCCESS,)
    np.testing.assert_allclose(ordered.parameter_values[0], expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(ordered.parameter_values, permuted.parameter_values)


def test_sampling_is_invariant_to_explicit_physical_row_permutation():
    coordinate_values = np.linspace(-5.0, 5.0, 1_000)
    scan = axis("scan", SCAN_POINT, 1_000, coordinates=coordinate_values, unit="MHz")
    logical_signal = evaluate_fit_model(
        "gaussian_offset",
        (coordinate_values,),
        (3.0, 0.5, 1.1, 0.7),
    )

    def fit_permutation(permutation: np.ndarray):
        layout = PointLayout.explicit(
            (scan.size,),
            tuple((int(index),) for index in permutation),
        )
        snapshot = snapshot_for(
            repeat=1,
            point_axes=(scan,),
            point_layout=layout,
            values=logical_signal[permutation].reshape(1, -1),
        )
        spec = FitSpec(
            snapshot.block.schema.fingerprint,
            None,
            (scan.axis_id,),
            (snapshot.block.schema.repeat_axis.axis_id,),
            "gaussian_offset",
            numeric_policy=FitNumericPolicy(sample_budget_per_batch=50),
        )
        return bind_fit(spec, snapshot.block.schema).run(snapshot)

    ordered = fit_permutation(np.arange(scan.size))
    shuffled_indices = np.arange(scan.size)
    np.random.default_rng(7).shuffle(shuffled_indices)
    shuffled = fit_permutation(shuffled_indices)
    np.testing.assert_array_equal(ordered.parameter_values, shuffled.parameter_values)
    np.testing.assert_array_equal(ordered.residual_sum_squares, shuffled.residual_sum_squares)

    randomized_scan = axis(
        "scan",
        SCAN_POINT,
        scan.size,
        coordinates=coordinate_values[shuffled_indices],
        unit="MHz",
    )
    randomized_snapshot = snapshot_for(
        repeat=1,
        point_axes=(randomized_scan,),
        point_layout=PointLayout.rect_c((randomized_scan.size,)),
        values=logical_signal[shuffled_indices].reshape(1, -1),
    )
    randomized_spec = FitSpec(
        randomized_snapshot.block.schema.fingerprint,
        None,
        (randomized_scan.axis_id,),
        (randomized_snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=50),
    )
    randomized = bind_fit(
        randomized_spec,
        randomized_snapshot.block.schema,
    ).run(randomized_snapshot)
    np.testing.assert_array_equal(ordered.parameter_values, randomized.parameter_values)


@pytest.mark.parametrize(
    ("model_id", "parameters"),
    (
        ("exponential_decay", (2.0, 0.3, 1.2)),
        ("damped_sine", (2.0, 0.3, 0.7, 3.2, 0.5)),
    ),
)
def test_time_models_use_declared_axis_minimum_as_recorded_reference(model_id, parameters):
    x = np.linspace(100.0, 104.0, 160)
    scan = axis("time", SCAN_POINT, x.size, coordinates=x, unit="ms")
    signal = evaluate_fit_model(model_id, (x - x.min(),), parameters)
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        model_id,
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.SUCCESS,)
    assert result.r_squared[0] > 0.999999
    np.testing.assert_allclose(result.parameter_values[0], parameters, rtol=2e-5, atol=2e-5)


def test_fixed_bounded_initial_constraints_and_numeric_budgets_fail_closed():
    snapshot, scan = gaussian_snapshot(repeat=1)
    fixed_spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("offset", fixed=1.2),),
    )
    fixed = bind_fit(fixed_spec, snapshot.block.schema).run(snapshot)
    assert fixed.statuses == (FitBatchStatus.SUCCESS,)
    assert fixed.parameter_values[0, 1] == 1.2

    impossible = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=4),
    )
    with pytest.raises(ValueError, match="below.*minimum_observations"):
        bind_fit(impossible, snapshot.block.schema)

    bad_initial = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("center", initial=100.0),),
    )
    failed = bind_fit(bad_initial, snapshot.block.schema).run(snapshot)
    assert failed.statuses == (FitBatchStatus.INITIALIZATION_FAILED,)
    assert not np.any(failed.parameter_values)

    limited = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=FitNumericPolicy(max_evaluations=1),
    )
    limited_result = bind_fit(limited, snapshot.block.schema).run(snapshot)
    assert limited_result.statuses == (FitBatchStatus.EVALUATION_LIMIT,)
    assert limited_result.evaluation_counts[0] == 1

    timed = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=FitNumericPolicy(max_seconds_per_batch=1e-12),
    )
    timed_result = bind_fit(timed, snapshot.block.schema).run(snapshot)
    assert timed_result.statuses == (FitBatchStatus.TIMEOUT,)


def test_cancel_is_checked_inside_model_evaluations_and_host_deadline_aborts():
    snapshot, scan = gaussian_snapshot(repeat=1)
    bound = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema)
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 15

    with pytest.raises(FitCancelled):
        bound.run(snapshot, cancel_check=cancel)
    assert checks >= 15
    with pytest.raises(FitDeadlineExceeded):
        fit_analysis(bound, snapshot, deadline_monotonic=time.monotonic() - 1.0)

    large_x = np.linspace(-5.0, 5.0, 2_048)
    large_scan = axis("large_scan", SCAN_POINT, large_x.size, coordinates=large_x)
    large = snapshot_for(
        repeat=1,
        point_axes=(large_scan,),
        point_layout=PointLayout.rect_c((large_scan.size,)),
        values=np.ones((1, large_scan.size)),
    )
    large_spec = FitSpec(
        large.block.schema.fingerprint,
        None,
        (large_scan.axis_id,),
        (large.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    packing_checks = 0

    def cancel_packing() -> bool:
        nonlocal packing_checks
        packing_checks += 1
        return packing_checks >= 5

    with pytest.raises(FitCancelled):
        bind_fit(large_spec, large.block.schema).run(large, cancel_check=cancel_packing)


def test_complex_observations_are_rejected_before_float_conversion():
    scan = axis("scan", SCAN_POINT, 8, coordinates=range(8))
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=np.ones((1, scan.size), dtype=np.complex128) * (1 + 2j),
        dtype=np.dtype("<c16"),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    with pytest.raises(TypeError, match="real numeric dtype"):
        bind_fit(spec, snapshot.block.schema)

    integer_snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=np.full((1, scan.size), 2**53 + 1, dtype=np.int64),
        dtype=np.dtype("<i8"),
    )
    integer_spec = replace(
        spec,
        input_schema_fingerprint=integer_snapshot.block.schema.fingerprint,
    )
    with pytest.raises(ValueError, match="not exactly float64-representable"):
        build_fit_problem(
            bind_fit(integer_spec, integer_snapshot.block.schema),
            integer_snapshot,
        )


def test_fit_spec_and_result_strict_codecs_and_public_entrypoints_agree():
    snapshot, scan = gaussian_snapshot(repeat=1)
    spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(
            FitParameterConstraint("sigma", lower=0.1),
            FitParameterConstraint("amplitude", initial=2.5),
        ),
    )
    restored_spec = decode_fit_spec(encode_fit_spec(spec))
    assert restored_spec == spec
    bound = bind_fit(restored_spec, snapshot.block.schema)
    from_method = bound.run(snapshot)
    from_function = fit_analysis(bound, snapshot)
    assert from_method.digest == from_function.digest
    assert from_method.parameter_units == ("count", "count", "MHz", "MHz")

    payload = encode_fit_result_batch(from_method)
    restored = decode_fit_result_batch(payload)
    assert restored.digest == from_method.digest
    assert encode_fit_result_batch(restored) == payload

    spec_tree = fit_spec_to_tree(spec)
    spec_tree["unexpected"] = True
    with pytest.raises(ValueError, match="exactly"):
        decode_fit_spec(encode(spec_tree))

    result_tree = fit_result_batch_to_tree(from_method)
    result_tree["parameter_values"] = from_method.parameter_values.astype(np.float32)
    with pytest.raises(TypedCodecError, match="non-canonical typed representation"):
        decode_fit_result_batch(encode(result_tree))


def test_result_constructor_rejects_impossible_success_and_noncanonical_payloads():
    snapshot, scan = gaussian_snapshot(repeat=1)
    result = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema).run(snapshot)
    with pytest.raises(ValueError, match="negative counts"):
        replace(result, present_observation_counts=np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError, match="too few used"):
        replace(result, used_observation_counts=np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="valid covariance"):
        replace(
            result,
            covariance=np.full_like(result.covariance, np.nan),
            covariance_valid=np.array([True]),
        )
    with pytest.raises(ValueError, match="invalid covariance payload"):
        replace(
            result,
            covariance=np.ones_like(result.covariance),
            covariance_valid=np.array([False]),
        )
    with pytest.raises(ValueError, match="finite non-negative metrics"):
        replace(result, rmse=np.array([np.nan]))
    with pytest.raises(ValueError, match="NO_VALID_DATA status"):
        replace(
            result,
            statuses=(FitBatchStatus.NO_VALID_DATA,),
            errors=("forged",),
            parameter_values=np.zeros_like(result.parameter_values),
            covariance=np.zeros_like(result.covariance),
            covariance_valid=np.array([False]),
            evaluation_counts=np.array([0]),
            residual_sum_squares=np.array([0.0]),
            rmse=np.array([0.0]),
            r_squared=np.array([0.0]),
            r_squared_valid=np.array([False]),
        )
    with pytest.raises(ValueError, match="solver_contract_id disagrees"):
        replace(result, solver_contract_id="fake-solver-v99")
    with pytest.raises(ValueError, match="initializer_id disagrees"):
        replace(result, initializer_id="fake-initializer-v99")
