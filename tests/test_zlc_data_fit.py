"""Named-axis fit contracts, packing, models, and artifact codecs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import subprocess
import sys
import time

import numpy as np
import pytest

from zlc_data.axis import (
    REPEAT,
    SCAN_POINT,
    HISTOGRAM_BIN,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisRoleId,
    AxisSourceRef,
    AxisSpec,
    CoordinateFrameId,
)
from zlc_data.bimodal_distribution import (
    BimodalDistributionAnalysis,
    analyze_bimodal_distribution,
)
from zlc_data.fit_codec import (
    decode_fit_result_batch,
    decode_fit_spec,
    encode_fit_result_batch,
    encode_fit_spec,
    fit_spec_from_tree,
    fit_spec_to_tree,
)
from zlc_data.fit_contract import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitNumericPolicy,
    FitParameterConstraint,
    FitResultBatch,
    FitSpec,
)
from zlc_data.fit_model import (
    FitModelDefinition,
    FitParameterDefinition,
    FitParameterDomain,
    ParameterUnitRelation,
    evaluate_fit_model,
    fit_model_catalog,
    fit_model_definition,
    initialize_fit_model,
)
from zlc_data.fit_problem import bind_fit, build_fit_problem
from zlc_data.layout import AxisLayout
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from zlc_data.selection import Selection
from zlc_data.transform import DataTransformSpec, commit_transform
from zlc_data.validity import (
    VALID,
    CellValidity,
    DatasetComponentValidity,
    ValidityContract,
)
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)
from zlc_storage.canonical import decode, encode


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
    points: PointTable,
    grid_topology: GridTopology | None = None,
    data_axes: tuple[AxisSpec, ...] = (),
    values,
    validity=VALID,
    validity_contract: ValidityContract | None = None,
    value_unit: str | None = "count",
    dtype=np.dtype("<f8"),
    block_id: str = "fit-source",
) -> OwnedSnapshot:
    repeat_axis = axis("repeat", REPEAT, repeat)
    array = np.asarray(values, dtype=dtype)
    cell_schema = (
        ValueSchema(
            data_axes,
            validity_contract or ValidityContract.value(),
            np.dtype(dtype),
            value_unit,
        )
        if data_axes
        else ValueSchema.scalar(np.dtype(dtype), value_unit)
    )
    if not data_axes:
        array = array[..., np.newaxis]
    schema = DatasetSchema(
        repeat_axis,
        points,
        grid_topology,
        cell_schema,
    )
    block = DataBlock(
        BlockId(block_id),
        DatasetRevision(3),
        array,
        validity,
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("fit-generation")), block)


def point_column(axis_spec: AxisSpec, values=None) -> PointColumn:
    if values is None:
        values = (
            axis_spec.coordinates
            if axis_spec.coordinates is not None
            else tuple(range(axis_spec.index_origin, axis_spec.index_origin + axis_spec.size))
        )
    values = tuple(values)
    value_kind = (
        PointColumn.TEXT
        if any(isinstance(value, str) for value in values if value is not None)
        else PointColumn.NUMERIC
    )
    return PointColumn(
        axis_spec.axis_id,
        axis_spec.name,
        axis_spec.role,
        value_kind,
        values,
        None if value_kind == PointColumn.TEXT else axis_spec.unit,
        axis_spec.coordinate_frame,
    )


def point_table(axis_spec: AxisSpec, values=None) -> PointTable:
    column = point_column(axis_spec, values)
    return PointTable(len(column.values), (column,))


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
        points=point_table(scan),
        values=np.tile(signal, (repeat, 1)),
    )
    return snapshot, scan


def gaussian_spec(snapshot: OwnedSnapshot, scan: AxisSpec, **kwargs) -> FitSpec:
    transform = kwargs.pop("committed_transform", None)
    if transform is None:
        transform = commit_transform(snapshot.block.schema, DataTransformSpec())
    default_batch = (
        (AxisSourceRef.tensor(snapshot.block.schema.repeat_axis.axis_id),)
        if snapshot.block.schema.repeat_axis.size > 1
        else ()
    )
    return FitSpec(
        transform,
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        kwargs.pop("batch_sources", default_batch),
        "gaussian_offset",
        constraints=kwargs.pop("constraints", ()),
        numeric_policy=kwargs.pop("numeric_policy", FitNumericPolicy()),
        **kwargs,
    )


def test_catalog_is_closed_canonical_and_rejects_unknown_models():
    catalog = fit_model_catalog()
    assert all(isinstance(model, FitModelDefinition) for model in catalog)
    assert all(
        isinstance(parameter, FitParameterDefinition)
        and isinstance(parameter.domain, FitParameterDomain)
        and isinstance(parameter.unit_relation, ParameterUnitRelation)
        for model in catalog
        for parameter in model.parameters
    )
    assert tuple(model.model_id for model in catalog) == (
        "lorentzian",
        "gaussian_offset",
        "histogram_gaussian",
        "bimodal_gaussian",
        "symmetric_lorentzian_doublet",
        "damped_sine",
        "exponential_decay",
        "radial_gaussian_center",
    )
    with pytest.raises(ValueError, match="unknown fit model"):
        fit_model_definition("unknown-model")
    by_id = {model.model_id: model for model in catalog}
    assert {model_id: model.parameter_names for model_id, model in by_id.items()} == {
        "lorentzian": ("center", "fwhm", "amplitude", "offset"),
        "gaussian_offset": ("amplitude", "offset", "sigma", "center"),
        "histogram_gaussian": ("amplitude", "center", "sigma"),
        "bimodal_gaussian": (
            "center",
            "center_splitting",
            "left_amplitude",
            "left_sigma",
            "right_amplitude",
            "right_sigma",
        ),
        "symmetric_lorentzian_doublet": (
            "center",
            "common_fwhm",
            "component_amplitude",
            "offset",
            "center_splitting",
        ),
        "damped_sine": ("amplitude", "offset", "baseband_frequency", "decay_time", "phase"),
        "exponential_decay": ("amplitude", "offset", "decay_time"),
        "radial_gaussian_center": (
            "amplitude",
            "offset",
            "one_over_e_radius",
            "center_x",
            "center_y",
        ),
    }


def test_public_fit_spec_codec_and_bound_editor_metadata_have_one_owner():
    snapshot, scan = gaussian_snapshot()
    spec = gaussian_spec(snapshot, scan)
    payload = encode_fit_spec(spec)
    tree = fit_spec_to_tree(spec)

    assert encode(tree) == payload
    assert decode_fit_spec(payload) == spec
    assert fit_spec_from_tree(tree) == spec
    with pytest.raises(ValueError):
        fit_spec_from_tree({**tree, "legacy_version": 1})

    bound = bind_fit(spec, snapshot.block.schema)
    assert bound.parameter_definitions == bound.model.parameters
    assert bound.parameter_units == ("count", "count", "MHz", "MHz")


def test_model_characteristic_points_define_physical_parameter_semantics():
    np.testing.assert_allclose(
        evaluate_fit_model("lorentzian", (np.array([1.0, 2.0]),), (1.0, 2.0, 4.0, 0.5)),
        (4.5, 2.5),
    )
    np.testing.assert_allclose(
        evaluate_fit_model(
            "gaussian_offset",
            (np.array([1.0, 3.0]),),
            (4.0, 0.5, 2.0, 1.0),
        ),
        (4.5, 0.5 + 4.0 / np.sqrt(np.e)),
    )
    np.testing.assert_allclose(
        evaluate_fit_model(
            "symmetric_lorentzian_doublet",
            (np.array([0.0, 2.0]),),
            (0.0, 2.0, 3.0, 0.5, 4.0),
        ),
        (1.7, 3.676470588235294),
    )
    np.testing.assert_allclose(
        evaluate_fit_model(
            "damped_sine",
            (np.array([0.0, 2.0]),),
            (4.0, 0.5, 0.5, 2.0, np.pi / 2.0),
        ),
        (4.5, 0.5 + 4.0 / np.e),
    )
    np.testing.assert_allclose(
        evaluate_fit_model(
            "exponential_decay",
            (np.array([0.0, 2.0]),),
            (4.0, 0.5, 2.0),
        ),
        (4.5, 0.5 + 4.0 / np.e),
    )
    np.testing.assert_allclose(
        evaluate_fit_model(
            "radial_gaussian_center",
            (np.array([1.0, 3.0]), np.array([-2.0, -2.0])),
            (4.0, 0.5, 2.0, 1.0, -2.0),
        ),
        (4.5, 0.5 + 4.0 / np.e),
    )


def test_initializers_match_independent_seed_examples():
    x = np.arange(5.0)
    peak = np.array([1.0, 2.0, 5.0, 2.0, 1.0])
    lorentzian = initialize_fit_model(fit_model_definition("lorentzian"), (x,), peak)
    assert lorentzian == ((2.0, 1.0, 4.0, 1.0), (0.0, 1.0, -4.0, 5.0))
    gaussian = initialize_fit_model(fit_model_definition("gaussian_offset"), (x,), peak)
    assert gaussian == (
        (4.0, 1.0, 2.0 / 3.0, 2.0),
        (-4.0, 5.0, 2.0 / 3.0, 0.0),
    )

    oscillation = initialize_fit_model(
        fit_model_definition("damped_sine"),
        (np.arange(8.0),),
        np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]),
    )
    assert oscillation == (
        (1.0, 0.0, 0.25, 7.0, -np.pi / 2.0),
        (1.0, 0.0, 0.25, 7.0, 0.0),
        (1.0, 0.0, 0.25, 7.0, np.pi / 2.0),
    )
    decay = initialize_fit_model(
        fit_model_definition("exponential_decay"),
        (np.arange(4.0),),
        np.array([9.0, 5.0, 3.0, 2.0]),
    )
    assert decay == ((7.0, 4.75, 1.5), (-7.0, 4.75, 1.5))

    doublet_x = np.arange(9.0)
    bright_doublet = initialize_fit_model(
        fit_model_definition("symmetric_lorentzian_doublet"),
        (doublet_x,),
        np.array([0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0]),
    )
    assert any(
        np.allclose(seed[[0, 2, 3, 4]], (4.0, 5.0, 0.0, 4.0))
        for seed in map(np.asarray, bright_doublet)
    )

    xx, yy = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    signed_spots = np.zeros((3, 3))
    signed_spots[1, 1] = 5.0
    signed_spots[0, 0] = -4.0
    radial = initialize_fit_model(
        fit_model_definition("radial_gaussian_center"),
        (xx.reshape(-1), yy.reshape(-1)),
        signed_spots.reshape(-1),
    )
    assert radial == (
        (5.0, 0.0, 1.0, 1.0, 1.0),
        (-4.0, 0.0, 1.0, 0.0, 0.0),
    )


def test_radial_full_image_jacobian_matches_central_difference():
    import zlc_data.fit_model as model_module

    x, y = np.meshgrid(
        np.linspace(-3.0, 4.0, 13),
        np.linspace(-2.0, 5.0, 11),
        indexing="ij",
    )
    coordinates = (x.reshape(-1), y.reshape(-1))
    parameters = np.array((4.2, 0.7, 1.6, 0.3, -0.4))
    analytic = model_module._radial_gaussian_center_jacobian(
        coordinates,
        parameters,
    )
    numeric = np.empty_like(analytic)
    for index, value in enumerate(parameters):
        step = 1e-6 * max(abs(float(value)), 1.0)
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        numeric[:, index] = (
            evaluate_fit_model(
                "radial_gaussian_center",
                coordinates,
                upper,
            )
            - evaluate_fit_model(
                "radial_gaussian_center",
                coordinates,
                lower,
            )
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-9)


def test_histogram_bin_axes_use_the_histogram_model_family():
    x = np.linspace(-3.0, 3.0, 31)
    bins = axis("bins", HISTOGRAM_BIN, x.size, coordinates=x, unit="count")
    values = 4.0 * np.exp(-((x - 0.4) ** 2) / (2.0 * 0.8**2))
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(bins),
        values=values.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(bins.axis_id),),
        (),
        "gaussian_offset",
    )
    with pytest.raises(ValueError, match="does not satisfy model roles"):
        bind_fit(spec, snapshot.block.schema)
    histogram_spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(bins.axis_id),),
        (),
        "histogram_gaussian",
    )
    result = bind_fit(histogram_spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], (4.0, 0.4, 0.8), rtol=1e-5)
    np.testing.assert_allclose(
        evaluate_fit_model(
            "exponential_decay",
            (np.array([0.0, 2.0]),),
            (4.0, 0.5, 2.0),
        ),
        (4.5, 0.5 + 4.0 / np.e),
    )


def test_bimodal_distribution_analysis_uses_the_catalogue_and_is_immutable():
    centers = np.linspace(-6.0, 7.0, 80)
    expected = np.array((0.2, 4.4, 1100.0, 0.55, 720.0, 0.72))
    counts = evaluate_fit_model("bimodal_gaussian", (centers,), expected)

    analysis = analyze_bimodal_distribution(centers, counts)

    assert isinstance(analysis, BimodalDistributionAnalysis)
    assert analysis.status is FitBatchStatus.CONVERGED
    assert analysis.diagnostic == ""
    assert analysis.separated
    assert analysis.threshold is not None
    assert expected[0] - expected[1] / 2.0 < analysis.threshold
    assert analysis.threshold < expected[0] + expected[1] / 2.0
    left, right, total = analysis.component_predictions
    np.testing.assert_allclose(left + right, total, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(total, counts, rtol=2e-7, atol=2e-7)
    assert not analysis.coordinates.flags.writeable
    assert all(not values.flags.writeable for values in analysis.component_predictions)
    with pytest.raises(ValueError):
        analysis.component_predictions[0][0] = 0.0


def test_bimodal_distribution_does_not_publish_unresolved_threshold():
    centers = np.linspace(-4.0, 4.0, 100)
    # Ashman's D is below the resolved-bimodality boundary even though two
    # catalogue components mathematically exist and can be fitted exactly.
    expected = np.array((0.0, 1.0, 800.0, 0.8, 760.0, 0.9))
    counts = evaluate_fit_model("bimodal_gaussian", (centers,), expected)

    analysis = analyze_bimodal_distribution(centers, counts)

    assert analysis.status is FitBatchStatus.CONVERGED
    assert not analysis.separated
    assert analysis.threshold is None
    assert len(analysis.component_predictions) == 3


def test_bimodal_component_threshold_requires_main_separation_and_one_crossing():
    from zlc_data.bimodal_distribution import _resolved_bimodal_threshold

    # Main's boundary is inclusive: splitting / (sigma_L + sigma_R) == 1.5.
    boundary = np.array((10.0, 6.0, 4.0, 2.0, 4.0, 2.0))
    assert _resolved_bimodal_threshold(boundary, support=(0.0, 20.0)) == pytest.approx(
        10.0
    )
    below = boundary.copy()
    below[1] = np.nextafter(6.0, 0.0)
    assert _resolved_bimodal_threshold(below, support=(0.0, 20.0)) is None

    # Even a wide split does not define a between-means cut when the enormous
    # left component still dominates at the right component's own mean.
    one_dominant_component = np.array((10.0, 8.0, 1e16, 1.0, 1.0, 1.0))
    assert (
        _resolved_bimodal_threshold(
            one_dominant_component,
            support=(0.0, 20.0),
        )
        is None
    )


def test_bimodal_distribution_returns_typed_failure_without_invented_payload():
    centers = np.linspace(-1.0, 1.0, 40)
    analysis = analyze_bimodal_distribution(centers, np.zeros_like(centers))

    assert analysis.status is FitBatchStatus.INITIALIZATION_FAILED
    assert analysis.diagnostic
    assert analysis.component_predictions == ()
    assert analysis.threshold is None
    assert not analysis.separated


def test_bimodal_distribution_validates_carrier_and_honours_host_abort():
    with pytest.raises(ValueError, match="same shape"):
        analyze_bimodal_distribution(np.arange(8.0), np.ones(7))
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_bimodal_distribution(np.array((0.0, 2.0, 1.0)), np.ones(3))
    with pytest.raises(ValueError, match="cannot be negative"):
        analyze_bimodal_distribution(np.arange(8.0), -np.ones(8))
    with pytest.raises(FitCancelled):
        analyze_bimodal_distribution(
            np.arange(8.0),
            np.ones(8),
            cancel_check=lambda: True,
        )
    np.testing.assert_allclose(
        evaluate_fit_model(
            "radial_gaussian_center",
            (np.array([1.0, 3.0]), np.array([2.0, 2.0])),
            (4.0, 0.5, 2.0, 1.0, 2.0),
        ),
        (4.5, 0.5 + 4.0 / np.e),
    )


def test_importing_zlc_data_keeps_scipy_solver_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from zlc_data.fit_contract import BoundFit; "
            "assert 'scipy' not in sys.modules; assert callable(BoundFit.run)",
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
            "symmetric_lorentzian_doublet",
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
        points=point_table(scan),
        values=values,
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        model_id,
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], parameters, rtol=2e-5, atol=2e-5)


def test_binding_is_axis_total_role_checked_and_declared_coordinates_are_not_ignored():
    snapshot, scan = gaussian_snapshot()
    with pytest.raises(ValueError, match="cover every effective information axis"):
        bind_fit(
            replace(gaussian_spec(snapshot, scan), batch_sources=()),
            snapshot.block.schema,
        )

    site_like = axis("site_like", SITE, scan.size, coordinates=range(scan.size))
    wrong = snapshot_for(
        repeat=1,
        points=point_table(site_like),
        values=np.ones((1, site_like.size)),
    )
    wrong_spec = FitSpec(
        commit_transform(wrong.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(site_like.axis_id),),
        (),
        "gaussian_offset",
    )
    with pytest.raises(ValueError, match="does not satisfy model roles"):
        bind_fit(wrong_spec, wrong.block.schema)

    labels = axis("labels", SCAN_POINT, 6, coordinates=("a", "b", "c", "d", "e", "f"))
    labelled = snapshot_for(
        repeat=1,
        points=point_table(labels),
        values=np.ones((1, labels.size)),
    )
    labelled_spec = FitSpec(
        commit_transform(labelled.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(labels.axis_id),),
        (),
        "gaussian_offset",
    )
    with pytest.raises(TypeError, match="point-coordinate source must be numeric"):
        bind_fit(labelled_spec, labelled.block.schema)


def test_implicit_fit_coordinates_require_consecutive_float64_identity():
    scan = AxisSpec(
        AxisId("scan"),
        "scan",
        SCAN_POINT,
        3,
        index_origin=2**53 + 2,
    )
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(scan,),
        values=np.zeros((1, 1, 3), dtype=np.float64),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(scan.axis_id),),
        (),
        "gaussian_offset",
    )
    with pytest.raises(ValueError, match="consecutively float64-representable"):
        bind_fit(spec, snapshot.block.schema)


def test_sparse_implicit_fit_tracks_present_rows_and_preserves_axis_unit():
    logical_size = 5_000_000
    scan = AxisSpec(
        AxisId("scan"),
        "scan",
        SCAN_POINT,
        2,
        (0, logical_size - 1),
        unit="MHz",
    )
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=np.array([[1.0, 2.0]]),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "gaussian_offset",
    )
    bound = bind_fit(spec, snapshot.block.schema)
    problem = build_fit_problem(bound, snapshot)

    np.testing.assert_array_equal(
        problem.independent_values[0],
        (0.0, float(logical_size - 1)),
    )
    assert bound.parameter_units == ("count", "count", "MHz", "MHz")


def test_radial_center_requires_x_then_y_with_compatible_units_and_frames():
    x = axis("x", SPATIAL_X, 9, coordinates=range(9), unit="px", frame="camera")
    y = axis("y", SPATIAL_Y, 7, coordinates=range(7), unit="px", frame="camera")
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=np.ones((1, 1, x.size, y.size)),
    )
    base = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )
    bind_fit(base, snapshot.block.schema)
    with pytest.raises(ValueError, match="does not satisfy model roles"):
        bind_fit(
            replace(
                base,
                independent_sources=(
                    AxisSourceRef.tensor(y.axis_id),
                    AxisSourceRef.tensor(x.axis_id),
                ),
            ),
            snapshot.block.schema,
        )

    bad_y = axis("y", SPATIAL_Y, 7, coordinates=range(7), unit="mm", frame="other")
    bad = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, bad_y),
        values=np.ones((1, 1, x.size, bad_y.size)),
    )
    bad_spec = FitSpec(
        commit_transform(bad.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(bad_y.axis_id)),
        (),
        "radial_gaussian_center",
    )
    with pytest.raises(ValueError, match="compatible coordinate units"):
        bind_fit(bad_spec, bad.block.schema)


def test_identity_and_committed_transform_keep_complete_source_lineage():
    snapshot, scan = gaussian_snapshot(repeat=2)
    identity_bound = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema)
    identity = identity_bound.run(snapshot)
    assert identity.source_ref == snapshot.ref
    assert identity.effective_schema_fingerprint == (
        identity.spec.committed_transform.output_schema_fingerprint
    )

    committed = commit_transform(
        snapshot.block.schema,
        DataTransformSpec(
            (Selection.index(snapshot.block.schema.repeat_axis.axis_id, 0),)
        ),
    )
    transformed_spec = gaussian_spec(
        snapshot,
        scan,
        committed_transform=committed,
        batch_sources=(),
    )
    transformed = bind_fit(transformed_spec, snapshot.block.schema).run(snapshot)
    assert transformed.source_ref == snapshot.ref
    assert transformed.spec.committed_transform == committed
    assert transformed.effective_schema_fingerprint == committed.output_schema_fingerprint
    np.testing.assert_allclose(transformed.parameter_values[0], (3.0, 1.2, 0.8, 0.7))


def test_fit_packing_rejects_a_subclass_that_skips_proof_admission():
    snapshot, scan = gaussian_snapshot(repeat=1)

    class UncheckedBoundFit(BoundFit):
        def __post_init__(self):
            pass

    forged = UncheckedBoundFit(gaussian_spec(snapshot, scan), snapshot.block.schema)
    with pytest.raises(TypeError, match="bound must be BoundFit"):
        build_fit_problem(forged, snapshot)


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
        points=point_table(scan),
        data_axes=(site,),
        values=values,
        validity=DatasetComponentValidity((site.axis_id,), mask),
        validity_contract=ValidityContract.components(site.axis_id),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (AxisSourceRef.tensor(site.axis_id),),
        "gaussian_offset",
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (
        FitBatchStatus.CONVERGED,
        FitBatchStatus.NO_VALID_DATA,
        FitBatchStatus.INSUFFICIENT_POINTS,
    )
    np.testing.assert_array_equal(result.present_observation_counts, (9, 9, 9))
    np.testing.assert_array_equal(result.valid_observation_counts, (9, 0, 3))
    np.testing.assert_array_equal(result.used_observation_counts, (9, 0, 3))
    np.testing.assert_allclose(result.parameter_values[0], parameters[0], rtol=1e-5, atol=1e-5)


def test_sparse_missing_batch_is_absent_while_present_invalid_batch_is_failure():
    group = axis("group", SITE, 3)
    scan = axis("scan", SCAN_POINT, 6, coordinates=np.linspace(-2, 2, 6))
    mapping = tuple((group_index, scan_index) for group_index in (0, 2) for scan_index in range(6))
    group_indices = np.asarray(tuple(item[0] for item in mapping))
    scan_indices = np.asarray(tuple(item[1] for item in mapping))
    values = evaluate_fit_model(
        "gaussian_offset",
        (np.asarray(scan.coordinates)[scan_indices],),
        (2.0, 0.4, 0.8, 0.2),
    )
    valid = np.ones((2, len(mapping)), dtype=bool)
    valid[:, group_indices == 2] = False
    points = PointTable(
        len(mapping),
        (
            point_column(group, group_indices),
            point_column(scan, np.asarray(scan.coordinates)[scan_indices]),
        ),
    )
    snapshot = snapshot_for(
        repeat=2,
        points=points,
        grid_topology=GridTopology(
            (group.axis_id, scan.axis_id),
            (tuple(range(group.size)), tuple(scan.coordinates)),
            mapping,
        ),
        values=np.tile(values, (2, 1)),
        validity=CellValidity(valid),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.grid_dimension(scan.axis_id),),
        (
            AxisSourceRef.tensor(snapshot.block.schema.repeat_axis.axis_id),
            AxisSourceRef.grid_dimension(group.axis_id),
        ),
        "gaussian_offset",
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert tuple(result.batch_layout.multi_index(i) for i in range(4)) == (
        (0, 0),
        (1, 0),
        (0, 2),
        (1, 2),
    )
    assert (0, 1) not in tuple(result.batch_layout.multi_index(i) for i in range(4))
    assert result.statuses == (
        FitBatchStatus.CONVERGED,
        FitBatchStatus.CONVERGED,
        FitBatchStatus.NO_VALID_DATA,
        FitBatchStatus.NO_VALID_DATA,
    )


def test_mixed_cell_data_batch_plan_matches_source_values():
    group = axis("group", SITE, 3)
    scan = axis("scan", SCAN_POINT, 6, coordinates=np.linspace(-2, 2, 6))
    site = axis("site", AxisRoleId("grid-x"), 2)
    mapping = tuple((group_index, scan_index) for group_index in (0, 2) for scan_index in range(6))
    group_indices = np.asarray(tuple(item[0] for item in mapping))
    scan_indices = np.asarray(tuple(item[1] for item in mapping))
    values = np.empty((2, len(mapping), site.size), dtype=np.float64)
    for repeat_index in range(2):
        for point_ordinal in range(len(mapping)):
            for site_index in range(site.size):
                values[repeat_index, point_ordinal, site_index] = (
                    100 * repeat_index
                    + 10 * group_indices[point_ordinal]
                    + site_index
                    + scan_indices[point_ordinal] / 10
                )
    points = PointTable(
        len(mapping),
        (
            point_column(group, group_indices),
            point_column(scan, np.asarray(scan.coordinates)[scan_indices]),
        ),
    )
    snapshot = snapshot_for(
        repeat=2,
        points=points,
        grid_topology=GridTopology(
            (group.axis_id, scan.axis_id),
            (tuple(range(group.size)), tuple(scan.coordinates)),
            mapping,
        ),
        data_axes=(site,),
        values=values,
    )
    repeat_axis = snapshot.block.schema.repeat_axis
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.grid_dimension(scan.axis_id),),
        (
            AxisSourceRef.tensor(site.axis_id),
            AxisSourceRef.grid_dimension(group.axis_id),
            AxisSourceRef.tensor(repeat_axis.axis_id),
        ),
        "gaussian_offset",
    )
    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)

    present_groups = tuple(sorted(set(int(item) for item in group_indices)))
    expected_keys = {
        (site_index, group_index, repeat_index)
        for site_index in range(site.size)
        for group_index in present_groups
        for repeat_index in range(repeat_axis.size)
    }
    actual_keys = tuple(
        problem.batch_layout.multi_index(index)
        for index in range(problem.batch_layout.storage_size)
    )
    assert set(actual_keys) == expected_keys
    for batch_index, (site_index, group_index, repeat_index) in enumerate(actual_keys):
        rows = np.flatnonzero(group_indices == group_index)
        rows = rows[np.argsort(scan_indices[rows], kind="stable")]
        start, stop = problem.batch_offsets[batch_index : batch_index + 2]
        np.testing.assert_array_equal(
            problem.present_observation_counts[batch_index],
            rows.size,
        )
        np.testing.assert_allclose(
            problem.independent_values[0][start:stop],
            np.asarray(scan.coordinates)[scan_indices[rows]],
        )
        np.testing.assert_allclose(
            problem.observations[start:stop],
            values[repeat_index, rows, site_index],
        )


def test_large_radial_image_uses_every_valid_observation_and_fits_2d():
    x_values = np.linspace(-3.0, 3.0, 160)
    y_values = np.linspace(-2.0, 2.0, 120)
    x = axis("x", SPATIAL_X, x_values.size, coordinates=x_values, unit="mm", frame="camera")
    y = axis("y", SPATIAL_Y, y_values.size, coordinates=y_values, unit="mm", frame="camera")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    expected = (4.0, 0.5, 1.1, 0.4, -0.3)
    image = evaluate_fit_model("radial_gaussian_center", (xx, yy), expected)
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )
    bound = bind_fit(spec, snapshot.block.schema)
    problem = build_fit_problem(bound, snapshot)
    assert problem.present_observation_counts[0] == image.size
    assert problem.valid_observation_counts[0] == image.size
    assert problem.used_observation_counts[0] == image.size
    assert problem.observations.size == image.size
    assert tuple(values.size for values in problem.independent_values) == (
        image.size,
        image.size,
    )
    result = bound.run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-3, atol=2e-3)


def test_megapixel_radial_fit_scores_every_camera_pixel():
    width, height = 1920, 1200
    x = axis(
        "x",
        SPATIAL_X,
        width,
        coordinates=range(width),
        unit="px",
        frame="camera",
    )
    y = axis(
        "y",
        SPATIAL_Y,
        height,
        coordinates=range(height),
        unit="px",
        frame="camera",
    )
    expected_center = np.array((950.4, 610.2))
    expected_radius = 17.0
    x_grid = np.arange(width, dtype=np.float64)[:, None]
    y_grid = np.arange(height, dtype=np.float64)[None, :]
    rng = np.random.default_rng(421)
    image = np.clip(
        7.0
        + 8.0
        * np.exp(
            -(
                (x_grid - expected_center[0]) ** 2
                + (y_grid - expected_center[1]) ** 2
            )
            / expected_radius**2
        )
        + rng.normal(0.0, 1.5, size=(width, height)),
        0.0,
        255.0,
    ).astype(np.uint8)
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, width, height),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)

    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.used_observation_counts[0] == image.size
    np.testing.assert_allclose(
        result.parameter_values[0, 3:5],
        expected_center,
        atol=1.0,
    )
    assert result.parameter_values[0, 2] == pytest.approx(
        expected_radius,
        abs=0.75,
    )
    assert result.covariance_valid[0]

    prediction = evaluate_fit_model(
        "radial_gaussian_center",
        (
            np.broadcast_to(x_grid, image.shape),
            np.broadcast_to(y_grid, image.shape),
        ),
        result.parameter_values[0],
    )
    authoritative_residual = prediction - image
    expected_rss = float(np.sum(authoritative_residual**2))
    expected_total = float(np.sum((image - float(np.mean(image))) ** 2))
    assert result.residual_sum_squares[0] == pytest.approx(expected_rss, rel=1e-12)
    assert result.r_squared[0] == pytest.approx(
        1.0 - expected_rss / expected_total,
        rel=1e-12,
    )


def test_radial_roi_keeps_absolute_coordinate_less_centers_and_index_units():
    x = axis("x", SPATIAL_X, 61)
    y = axis("y", SPATIAL_Y, 41)
    xx, yy = np.meshgrid(np.arange(x.size), np.arange(y.size), indexing="ij")
    expected = (8.0, 0.4, 4.0, 30.0, 15.0)
    image = (
        expected[1]
        + expected[0]
        * np.exp(
            -(
                (xx - expected[3]) ** 2
                + (yy - expected[4]) ** 2
            )
            / expected[2] ** 2
        )
    )
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )

    def run(committed_transform):
        spec = FitSpec(
            committed_transform,
            (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
            (),
            "radial_gaussian_center",
        )
        return bind_fit(spec, snapshot.block.schema).run(snapshot)

    full = run(commit_transform(snapshot.block.schema, DataTransformSpec()))
    roi_transform = commit_transform(
        snapshot.block.schema,
        DataTransformSpec(
            (
                Selection.index_range(x.axis_id, 20, 41),
                Selection.index_range(y.axis_id, 5, 26),
            )
        ),
    )
    roi_spec = FitSpec(
        roi_transform,
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )
    roi_bound = bind_fit(roi_spec, snapshot.block.schema)
    roi_problem = build_fit_problem(roi_bound, snapshot)
    roi = roi_bound.run(snapshot)

    assert full.statuses == roi.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(full.parameter_values[0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(roi.parameter_values[0], expected, rtol=1e-5, atol=1e-5)
    assert roi.parameter_units == ("count", "count", "index", "index", "index")
    assert tuple(axis.unit for axis in roi.fit_axis_specs) == (None, None)
    assert tuple(axis.coordinates for axis in roi.fit_axis_specs) == (None, None)
    assert tuple(axis.index_origin for axis in roi.fit_axis_specs) == (20, 5)
    np.testing.assert_array_equal(
        np.unique(roi_problem.independent_values[0]),
        np.arange(20.0, 41.0),
    )
    np.testing.assert_array_equal(
        np.unique(roi_problem.independent_values[1]),
        np.arange(5.0, 26.0),
    )


def test_full_2d_fit_keeps_a_narrow_feature():
    x = axis("x", SPATIAL_X, 600)
    y = axis("y", SPATIAL_Y, 600)
    xx, yy = np.meshgrid(np.arange(x.size), np.arange(y.size), indexing="ij")
    expected = (5_000.0, 100.0, 2.0, 8.0, 13.0)
    image = (
        expected[1]
        + expected[0]
        * np.exp(
            -(
                (xx - expected[3]) ** 2
                + (yy - expected[4]) ** 2
            )
            / expected[2] ** 2
        )
    )
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )

    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert problem.used_observation_counts[0] == image.size
    assert np.max(problem.observations) == pytest.approx(5_100.0)
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-3, atol=2e-3)


def test_full_2d_fit_does_not_turn_one_outlier_into_the_only_feature():
    x = axis("x", SPATIAL_X, 240)
    y = axis("y", SPATIAL_Y, 240)
    xx, yy = np.meshgrid(np.arange(x.size), np.arange(y.size), indexing="ij")
    expected = (4.0, 0.5, 35.0, 125.0, 112.0)
    image = (
        expected[1]
        + expected[0]
        * np.exp(
            -(
                (xx - expected[3]) ** 2
                + (yy - expected[4]) ** 2
            )
            / expected[2] ** 2
        )
    )
    image[7, 13] = 1_000.0
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.parameter_values[0, 2] > 30.0
    np.testing.assert_allclose(
        result.parameter_values[0, 3:5],
        expected[3:5],
        atol=0.05,
    )


@pytest.mark.parametrize("amplitude", (9.0, -9.0))
def test_noisy_radial_seed_tracks_coherent_bright_and_dark_features(amplitude):
    width, height = 320, 240
    x = axis("x", SPATIAL_X, width, frame="camera")
    y = axis("y", SPATIAL_Y, height, frame="camera")
    x_grid = np.arange(width, dtype=np.float64)[:, None]
    y_grid = np.arange(height, dtype=np.float64)[None, :]
    expected = (amplitude, 20.0, 9.0, 173.2, 108.7)
    rng = np.random.default_rng(917 if amplitude > 0.0 else 918)
    image = evaluate_fit_model(
        "radial_gaussian_center",
        (
            np.broadcast_to(x_grid, (width, height)),
            np.broadcast_to(y_grid, (width, height)),
        ),
        expected,
    ) + rng.normal(0.0, 0.8, size=(width, height))
    validity = np.ones((1, 1, width, height), dtype=np.bool_)
    validity[:, :, :7, :11] = False
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, width, height),
        validity=DatasetComponentValidity((x.axis_id, y.axis_id), validity),
        validity_contract=ValidityContract.components(x.axis_id, y.axis_id),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)

    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert np.sign(result.parameter_values[0, 0]) == np.sign(amplitude)
    np.testing.assert_allclose(
        result.parameter_values[0, 2:5],
        expected[2:5],
        atol=0.25,
    )


def test_valid_nonfinite_is_included_fail_closed_while_invalid_nonfinite_is_absent():
    coordinate_values = np.linspace(-5.0, 5.0, 100)
    scan = axis(
        "nonfinite_scan",
        SCAN_POINT,
        coordinate_values.size,
        coordinates=coordinate_values,
    )
    signal = 0.5 + 3.0 * np.exp(-((coordinate_values - 0.7) ** 2) / (2.0 * 1.1**2))
    signal[90:92] = np.nan
    validity = np.ones((1, scan.size), dtype=bool)
    validity[0, 91] = False
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=signal.reshape(1, -1),
        validity=CellValidity(validity),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "gaussian_offset",
    )

    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert np.count_nonzero(np.isnan(problem.observations)) == 1
    assert coordinate_values[90] in problem.independent_values[0]
    assert coordinate_values[91] not in problem.independent_values[0]
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.NUMERIC_ERROR,)


def test_2d_observation_coordinates_are_cartesian_not_a_flattened_diagonal():
    x_values = np.linspace(-3.0, 3.0, 100)
    y_values = np.linspace(-2.0, 2.0, 100)
    x = axis("x", SPATIAL_X, 100, coordinates=x_values, unit="mm", frame="camera")
    y = axis("y", SPATIAL_Y, 100, coordinates=y_values, unit="mm", frame="camera")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    expected = (4.0, 0.5, 1.1, 0.4, -0.3)
    image = evaluate_fit_model("radial_gaussian_center", (xx, yy), expected)
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=image.reshape(1, 1, 100, 100),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.r_squared[0] > 0.999999
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-4, atol=2e-4)


def test_2d_grid_fit_recovers_center_and_ignores_authored_row_permutation():
    x_values = np.linspace(-3.0, 3.0, 100)
    y_values = np.linspace(-2.0, 2.0, 100)
    x = axis("x", SPATIAL_X, 100, coordinates=x_values, unit="mm", frame="camera")
    y = axis("y", SPATIAL_Y, 100, coordinates=y_values, unit="mm", frame="camera")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    expected = (4.0, 0.5, 1.1, 0.4, -0.3)
    logical_image = evaluate_fit_model("radial_gaussian_center", (xx, yy), expected)

    def run(mapping: tuple[tuple[int, int], ...]) -> object:
        physical_values = np.fromiter(
            (logical_image[cell] for cell in mapping),
            dtype=np.float64,
            count=len(mapping),
        )
        points = PointTable(
            len(mapping),
            (
                point_column(x, tuple(x_values[cell[0]] for cell in mapping)),
                point_column(y, tuple(y_values[cell[1]] for cell in mapping)),
            ),
        )
        snapshot = snapshot_for(
            repeat=1,
            points=points,
            grid_topology=GridTopology(
                (x.axis_id, y.axis_id),
                (tuple(x_values), tuple(y_values)),
                mapping,
            ),
            values=physical_values.reshape(1, -1),
        )
        spec = FitSpec(
            commit_transform(snapshot.block.schema, DataTransformSpec()),
            (
                AxisSourceRef.grid_dimension(x.axis_id),
                AxisSourceRef.grid_dimension(y.axis_id),
            ),
            (),
            "radial_gaussian_center",
        )
        return bind_fit(spec, snapshot.block.schema).run(snapshot)

    ordered_mapping = tuple(np.ndindex(100, 100))
    ordered = run(ordered_mapping)
    mapping = list(ordered_mapping)
    np.random.default_rng(11).shuffle(mapping)
    permuted = run(tuple(mapping))
    assert ordered.statuses == permuted.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(ordered.parameter_values[0], expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        ordered.parameter_values,
        permuted.parameter_values,
        rtol=1e-14,
        atol=1e-14,
    )


def test_fit_packing_is_invariant_to_explicit_physical_row_permutation():
    coordinate_values = np.linspace(-5.0, 5.0, 1_000)
    scan = axis("scan", SCAN_POINT, 1_000, coordinates=coordinate_values, unit="MHz")
    logical_signal = evaluate_fit_model(
        "gaussian_offset",
        (coordinate_values,),
        (3.0, 0.5, 1.1, 0.7),
    )

    def fit_permutation(permutation: np.ndarray):
        points = point_table(scan, coordinate_values[permutation])
        snapshot = snapshot_for(
            repeat=1,
            points=points,
            values=logical_signal[permutation].reshape(1, -1),
        )
        spec = FitSpec(
            commit_transform(snapshot.block.schema, DataTransformSpec()),
            (AxisSourceRef.point_coordinate(scan.axis_id),),
            (),
            "gaussian_offset",
        )
        return bind_fit(spec, snapshot.block.schema).run(snapshot)

    ordered = fit_permutation(np.arange(scan.size))
    shuffled_indices = np.arange(scan.size)
    np.random.default_rng(7).shuffle(shuffled_indices)
    shuffled = fit_permutation(shuffled_indices)
    np.testing.assert_allclose(
        ordered.parameter_values,
        shuffled.parameter_values,
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        ordered.residual_sum_squares,
        shuffled.residual_sum_squares,
        rtol=1e-12,
        atol=1e-24,
    )

    randomized_scan = axis(
        "scan",
        SCAN_POINT,
        scan.size,
        coordinates=coordinate_values[shuffled_indices],
        unit="MHz",
    )
    randomized_snapshot = snapshot_for(
        repeat=1,
        points=point_table(randomized_scan),
        values=logical_signal[shuffled_indices].reshape(1, -1),
    )
    randomized_spec = FitSpec(
        commit_transform(randomized_snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(randomized_scan.axis_id),),
        (),
        "gaussian_offset",
    )
    randomized = bind_fit(
        randomized_spec,
        randomized_snapshot.block.schema,
    ).run(randomized_snapshot)
    np.testing.assert_allclose(
        ordered.parameter_values,
        randomized.parameter_values,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("model_id", "parameters"),
    (
        ("exponential_decay", (2.0, 0.3, 1.2)),
        ("damped_sine", (2.0, 0.3, 0.7, 3.2, 0.5)),
    ),
)
def test_time_models_keep_absolute_declared_coordinates_for_replay(model_id, parameters):
    x = np.linspace(0.2, 4.2, 160)
    scan = axis("time", SCAN_POINT, x.size, coordinates=x, unit="ms")
    signal = evaluate_fit_model(model_id, (x,), parameters)
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        model_id,
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.r_squared[0] > 0.999999
    np.testing.assert_allclose(result.parameter_values[0], parameters, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        result.evaluate_batch(0, (x,)),
        signal,
        rtol=2e-5,
        atol=2e-5,
    )
    if model_id == "damped_sine":
        assert result.parameter_values[0, 0] >= 0.0
        assert -np.pi <= result.parameter_values[0, 4] <= np.pi


@pytest.mark.parametrize(
    ("model_id", "parameters"),
    (
        ("exponential_decay", (1000.0, 0.3, 2.0)),
        ("damped_sine", (100.0, 1.0, 0.5, 4.0, 0.3)),
    ),
)
def test_absolute_time_parameters_are_not_clipped_to_selected_window_contrast(
    model_id,
    parameters,
):
    x = np.linspace(10.0, 14.0, 241)
    if model_id == "exponential_decay":
        amplitude, offset, decay_time = parameters
        signal = amplitude * np.exp(-x / decay_time) + offset
    else:
        amplitude, offset, frequency, decay_time, phase = parameters
        signal = (
            amplitude
            * np.sin(2.0 * np.pi * frequency * x + phase)
            * np.exp(-x / decay_time)
            + offset
        )
    time_axis = axis("absolute_time", SCAN_POINT, x.size, coordinates=x)
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(time_axis),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(time_axis.axis_id),),
        (),
        model_id,
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], parameters, rtol=2e-5, atol=2e-5)


def test_short_exponential_window_recovers_absolute_decay_parameters():
    x_values = np.linspace(0.0, 0.5, 101)
    signal = 2.0 * np.exp(-x_values / 1.0) + 0.3
    scan = axis("short_decay_window", SCAN_POINT, x_values.size, coordinates=x_values)
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "exponential_decay",
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)

    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], (2.0, 0.3, 1.0), rtol=2e-5)


def test_fully_fixed_hypothesis_bypasses_data_derived_initializer():
    x = axis("fixed_x", SPATIAL_X, 3, coordinates=(-1.0, 0.0, 1.0), frame="camera")
    y = axis("fixed_y", SPATIAL_Y, 3, coordinates=(-1.0, 0.0, 1.0), frame="camera")
    snapshot = snapshot_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x, y),
        values=np.ones((1, 1, 3, 3)),
    )
    names = ("amplitude", "offset", "one_over_e_radius", "center_x", "center_y")
    values = (0.0, 1.0, 1.0, 0.0, 0.0)
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.tensor(x.axis_id), AxisSourceRef.tensor(y.axis_id)),
        (),
        "radial_gaussian_center",
        constraints=tuple(
            FitParameterConstraint(name, fixed=value)
            for name, value in zip(names, values)
        ),
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_array_equal(result.parameter_values[0], values)

    explicit_seed = replace(
        spec,
        constraints=tuple(
            FitParameterConstraint(name, initial=value)
            for name, value in zip(names, values)
        ),
    )
    seeded = bind_fit(explicit_seed, snapshot.block.schema).run(snapshot)
    assert seeded.statuses == (FitBatchStatus.CONVERGED,)


def test_fixed_gaussian_geometry_is_a_hypothesis_not_a_location_inference():
    x_values = np.array([-10.0, -9.0, 9.0, 10.0])
    scan = axis("fixed_tail_scan", SCAN_POINT, x_values.size, coordinates=x_values)
    amplitude = 1e20
    signal = amplitude * np.exp(-(x_values**2) / 2.0)
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "gaussian_offset",
        constraints=tuple(
            FitParameterConstraint(name, fixed=value)
            for name, value in zip(
                ("amplitude", "offset", "sigma", "center"),
                (amplitude, 0.0, 1.0, 0.0),
            )
        ),
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)

    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.r_squared[0] == 1.0


def test_minimum_observations_follow_the_free_parameters_not_a_catalog_constant():
    x_values = np.array([-1.0, 0.2])
    scan = axis("fixed_hypothesis_scan", SCAN_POINT, 2, coordinates=x_values)
    expected = np.exp(-(x_values**2) / 2.0)
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=expected.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "gaussian_offset",
        constraints=tuple(
            FitParameterConstraint(name, fixed=value)
            for name, value in zip(
                ("amplitude", "offset", "sigma", "center"),
                (1.0, 0.0, 1.0, 0.0),
            )
        ),
    )

    bound = bind_fit(spec, snapshot.block.schema)
    result = bound.run(snapshot)

    assert bound.minimum_observation_count == 2
    assert result.statuses == (FitBatchStatus.CONVERGED,)


def test_one_numeric_overflow_isolated_from_sibling_fit_cells():
    x_values = np.linspace(-2.0, 2.0, 9)
    scan = axis("overflow_scan", SCAN_POINT, x_values.size, coordinates=x_values)
    expected = np.exp(-(x_values**2) / 2.0)
    snapshot = snapshot_for(
        repeat=2,
        points=point_table(scan),
        values=np.stack((expected, np.full(scan.size, 1e308))),
    )
    fixed = (1.0, 0.0, 1.0, 0.0)
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (AxisSourceRef.tensor(snapshot.block.schema.repeat_axis.axis_id),),
        "gaussian_offset",
        constraints=tuple(
            FitParameterConstraint(name, fixed=value)
            for name, value in zip(
                ("amplitude", "offset", "sigma", "center"),
                fixed,
            )
        ),
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)

    assert result.statuses == (
        FitBatchStatus.CONVERGED,
        FitBatchStatus.NUMERIC_ERROR,
    )
    assert "sum of squares overflowed" in result.errors[1]


@pytest.mark.parametrize("entrypoint", ("initialize_fit_model", "evaluate_fit_model"))
def test_internal_fit_contract_defects_abort_the_whole_analysis(monkeypatch, entrypoint):
    import zlc_data.fit_solver as solver_module

    snapshot, scan = gaussian_snapshot(repeat=2)
    bound = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema)

    def broken(*_args, **_kwargs):
        raise AssertionError("simulated implementation invariant failure")

    monkeypatch.setattr(solver_module, entrypoint, broken)
    with pytest.raises(AssertionError, match="implementation invariant"):
        bound.run(snapshot)


def test_r_squared_centers_huge_finite_observations_without_sum_overflow():
    x_values = np.array([-1.0, -0.2, 0.3, 1.5])
    scan = axis("huge_offset_scan", SCAN_POINT, x_values.size, coordinates=x_values)
    amplitude = 4e307
    offset = 8e307
    signal = amplitude * np.exp(-(x_values**2) / 2.0) + offset
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "gaussian_offset",
        constraints=tuple(
            FitParameterConstraint(name, fixed=value)
            for name, value in zip(
                ("amplitude", "offset", "sigma", "center"),
                (amplitude, offset, 1.0, 0.0),
            )
        ),
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)

    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.r_squared[0] == 1.0


def test_active_solver_bound_invalidates_covariance_without_hiding_result():
    bounded_x = np.linspace(-2.0, 2.0, 81)
    bounded_signal = 3.0 * np.exp(-(bounded_x**2) / (2.0 * 0.1**2)) + 0.2
    scan = axis("bounded_scan", SCAN_POINT, bounded_x.size, coordinates=bounded_x)
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=bounded_signal.reshape(1, -1),
    )
    active_bound = bind_fit(
        gaussian_spec(
            snapshot,
            scan,
            constraints=(FitParameterConstraint("sigma", lower=0.5),),
        ),
        snapshot.block.schema,
    ).run(
        snapshot,
    )
    assert active_bound.statuses == (FitBatchStatus.CONVERGED,)
    assert active_bound.parameter_values[0, 2] == pytest.approx(0.5)
    assert not active_bound.covariance_valid[0]
    np.testing.assert_array_equal(active_bound.covariance[0], 0.0)


def test_covariance_matches_the_independent_normal_equation_oracle():
    from zlc_data.fit_solver import _covariance

    jacobian = np.array(
        (
            (1.0, -2.0),
            (1.0, -1.0),
            (1.0, 0.5),
            (1.0, 2.0),
            (1.0, 4.0),
        )
    )
    residual = np.array((0.2, -0.1, 0.3, -0.2, 0.1))
    free_indices = np.array((0, 2), dtype=np.int64)

    covariance, valid = _covariance(
        jacobian,
        residual,
        free_indices,
        parameter_count=4,
        rcond=1e-12,
    )
    residual_variance = float(residual @ residual) / (
        residual.size - free_indices.size
    )
    expected_free = np.linalg.inv(jacobian.T @ jacobian) * residual_variance
    expected = np.zeros((4, 4))
    expected[np.ix_(free_indices, free_indices)] = expected_free

    assert valid
    np.testing.assert_allclose(covariance, expected, rtol=2e-15, atol=1e-18)
    np.testing.assert_array_equal(covariance[1], 0.0)
    np.testing.assert_array_equal(covariance[3], 0.0)


def test_parameter_domains_guard_requests_evaluation_and_converged_artifacts():
    with pytest.raises(ValueError, match="POSITIVE domain"):
        evaluate_fit_model(
            "gaussian_offset",
            (np.array([0.0, 1.0]),),
            (1.0, 0.0, -1.0, 0.0),
        )

    snapshot, scan = gaussian_snapshot(repeat=1)
    with pytest.raises(ValueError, match="constraint fixed.*POSITIVE domain"):
        gaussian_spec(
            snapshot,
            scan,
            constraints=(FitParameterConstraint("sigma", fixed=-1.0),),
        )
    with pytest.raises(ValueError, match="bounds.*no free interval.*POSITIVE"):
        gaussian_spec(
            snapshot,
            scan,
            constraints=(FitParameterConstraint("sigma", upper=-1.0),),
        )
    with pytest.raises(ValueError, match="no free interval.*NONNEGATIVE"):
        FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
            "symmetric_lorentzian_doublet",
            constraints=(FitParameterConstraint("center_splitting", lower=-1.0, upper=0.0),),
        )
    with pytest.raises(ValueError, match="PHASE_RADIANS domain"):
        FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
            "damped_sine",
            constraints=(FitParameterConstraint("phase", fixed=np.pi),),
        )
    for constraint in (
        FitParameterConstraint("phase", lower=np.pi),
        FitParameterConstraint("phase", upper=-np.pi),
        FitParameterConstraint("phase", lower=np.nextafter(np.pi, -np.inf)),
    ):
        with pytest.raises(ValueError, match="no free interval.*PHASE_RADIANS"):
            FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
                "damped_sine",
                constraints=(constraint,),
            )
    with pytest.raises(ValueError, match="no free interval.*POSITIVE"):
        gaussian_spec(
            snapshot,
            scan,
            constraints=(
                FitParameterConstraint("sigma", upper=np.nextafter(0.0, np.inf)),
            ),
        )

    phase_boundary = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "damped_sine",
        constraints=(FitParameterConstraint("phase", fixed=-np.pi),),
    )
    assert phase_boundary.constraints[0].fixed == -np.pi

    result = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema).run(snapshot)
    forged = result.parameter_values.copy()
    forged[0, 2] = -1.0
    with pytest.raises(ValueError, match="converged batch parameter 'sigma'.*POSITIVE"):
        replace(result, parameter_values=forged)

    fixed_spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("offset", fixed=1.2),),
    )
    fixed_result = bind_fit(fixed_spec, snapshot.block.schema).run(snapshot)
    violates_fixed = fixed_result.parameter_values.copy()
    violates_fixed[0, 1] = 1.3
    with pytest.raises(ValueError, match="violates fixed 'offset'"):
        replace(fixed_result, parameter_values=violates_fixed)


def test_solver_uses_explicit_scipy_options(monkeypatch):
    import zlc_data.fit_solver as solver_module

    observed = []
    solve = solver_module.least_squares

    def record(*args, **kwargs):
        observed.append(dict(kwargs))
        return solve(*args, **kwargs)

    monkeypatch.setattr(solver_module, "least_squares", record)
    snapshot, scan = gaussian_snapshot(repeat=1)
    bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema).run(snapshot)
    assert observed
    for options in observed:
        assert options["jac"] == "2-point"
        assert options["method"] == "trf"
        assert options["ftol"] == options["xtol"] == options["gtol"] == 1e-8
        assert options["x_scale"] == 1.0
        assert options["loss"] == "linear"
        assert options["f_scale"] == 1.0
        assert options["diff_step"] is None
        assert options["tr_solver"] == "exact"
        assert options["tr_options"] is None


def test_fixed_bounded_initial_constraints_and_solver_limits_fail_closed():
    snapshot, scan = gaussian_snapshot(repeat=1)
    fixed_spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("offset", fixed=1.2),),
    )
    fixed = bind_fit(fixed_spec, snapshot.block.schema).run(snapshot)
    assert fixed.statuses == (FitBatchStatus.CONVERGED,)
    assert fixed.parameter_values[0, 1] == 1.2

    bad_initial = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("center", initial=100.0),),
    )
    failed = bind_fit(bad_initial, snapshot.block.schema).run(snapshot)
    assert failed.statuses == (FitBatchStatus.CONVERGED,)
    assert np.any(failed.parameter_values)

    limited = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=FitNumericPolicy(max_evaluations=1),
    )
    limited_result = bind_fit(limited, snapshot.block.schema).run(snapshot)
    assert limited_result.statuses == (FitBatchStatus.EVALUATION_LIMIT,)
    assert limited_result.evaluation_counts[0] == 1


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
        bound.run(snapshot, deadline_monotonic=time.monotonic() - 1.0)

    large_x = np.linspace(-5.0, 5.0, 2_048)
    large_scan = axis("large_scan", SCAN_POINT, large_x.size, coordinates=large_x)
    large = snapshot_for(
        repeat=1,
        points=point_table(large_scan),
        values=np.ones((1, large_scan.size)),
    )
    large_spec = FitSpec(
        commit_transform(large.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(large_scan.axis_id),),
        (),
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
        points=point_table(scan),
        values=np.ones((1, scan.size), dtype=np.complex128) * (1 + 2j),
        dtype=np.dtype("<c16"),
    )
    spec = FitSpec(
        commit_transform(snapshot.block.schema, DataTransformSpec()),
        (AxisSourceRef.point_coordinate(scan.axis_id),),
        (),
        "gaussian_offset",
    )
    with pytest.raises(TypeError, match="real numeric dtype"):
        bind_fit(spec, snapshot.block.schema)

    integer_snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=np.full((1, scan.size), 2**53 + 1, dtype=np.int64),
        dtype=np.dtype("<i8"),
    )
    integer_spec = replace(
        spec,
        committed_transform=commit_transform(
            integer_snapshot.block.schema,
            DataTransformSpec(),
        ),
    )
    with pytest.raises(ValueError, match="not exactly float64-representable"):
        build_fit_problem(
            bind_fit(integer_spec, integer_snapshot.block.schema),
            integer_snapshot,
        )

    ordered_integers = np.array(
        [0, 1, 2, 3, 4, 5, 2**53, 2**53 + 1],
        dtype=np.uint64,
    )
    ordered_snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=ordered_integers.reshape(1, -1),
        dtype=np.dtype("<u8"),
    )
    ordered_spec = replace(
        spec,
        committed_transform=commit_transform(
            ordered_snapshot.block.schema,
            DataTransformSpec(),
        ),
    )
    with pytest.raises(ValueError, match="not exactly float64-representable"):
        build_fit_problem(
            bind_fit(ordered_spec, ordered_snapshot.block.schema),
            ordered_snapshot,
        )


def test_fit_result_strict_codec_embeds_the_current_fit_spec():
    snapshot, scan = gaussian_snapshot(repeat=1)
    spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(
            FitParameterConstraint("sigma", lower=0.1),
            FitParameterConstraint("amplitude", initial=2.5),
        ),
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.parameter_units == ("count", "count", "MHz", "MHz")

    payload = encode_fit_result_batch(result)
    restored = decode_fit_result_batch(payload)
    assert encode_fit_result_batch(restored) == payload

    result_tree = decode(payload)
    assert result_tree["fit_spec"]["numeric_policy"] == {
        "covariance_rcond": 1e-12,
        "max_evaluations": 4_000,
    }
    result_tree["parameter_values"] = result.parameter_values.astype(np.float32)
    with pytest.raises(TypeError, match="dtype float32.*float64"):
        decode_fit_result_batch(encode(result_tree))

    unexpected = decode(payload)
    unexpected["unexpected_field"] = 1
    with pytest.raises(ValueError, match="exactly"):
        decode_fit_result_batch(encode(unexpected))

    unexpected_spec = decode(payload)
    unexpected_spec["fit_spec"]["unexpected_field"] = 1
    with pytest.raises(ValueError, match="exactly"):
        decode_fit_result_batch(encode(unexpected_spec))


def test_fit_result_wire_format_has_one_frozen_current_golden():
    scan = axis(
        "golden-scan",
        SCAN_POINT,
        5,
        coordinates=(-2.0, -1.0, 0.0, 1.0, 2.0),
        unit="MHz",
    )
    snapshot = snapshot_for(
        repeat=1,
        points=point_table(scan),
        values=np.zeros((1, scan.size)),
        block_id="golden-fit-source",
    )
    spec = gaussian_spec(snapshot, scan)
    bound = bind_fit(spec, snapshot.block.schema)
    result = FitResultBatch(
        source_ref=snapshot.ref,
        spec=spec,
        fit_axis_specs=bound.fit_axis_specs,
        batch_axis_specs=bound.batch_axis_specs,
        point_groups=bound.point_groups,
        batch_layout=AxisLayout.rect_c(()),
        value_unit="count",
        parameter_values=np.array(((2.0, 0.5, 1.0, 0.25),)),
        covariance=np.diag((0.1, 0.2, 0.3, 0.4)).reshape(1, 4, 4),
        covariance_valid=np.array((True,)),
        statuses=(FitBatchStatus.CONVERGED,),
        errors=(None,),
        present_observation_counts=np.array((5,), dtype=np.int64),
        valid_observation_counts=np.array((5,), dtype=np.int64),
        used_observation_counts=np.array((5,), dtype=np.int64),
        evaluation_counts=np.array((7,), dtype=np.int64),
        residual_sum_squares=np.array((0.25,)),
        r_squared=np.array((0.75,)),
        r_squared_valid=np.array((True,)),
        scipy_version="oracle-scipy",
    )

    payload = encode_fit_result_batch(result)
    assert len(payload) == 4430
    assert hashlib.sha256(payload).hexdigest() == (
        "1823c99c5eed80e5e73b79e7721c482036b4f710672ae8f23ac3efe89a6b1cfd"
    )

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
    indefinite = np.eye(result.parameter_values.shape[1])
    indefinite[0, 1] = indefinite[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(
            result,
            covariance=indefinite.reshape(1, *indefinite.shape),
            covariance_valid=np.array([True]),
        )
    with pytest.raises(ValueError, match="no greater than one"):
        replace(result, r_squared=np.array([1.01]), r_squared_valid=np.array([True]))
    fixed_spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("offset", fixed=1.2),),
    )
    fixed_result = bind_fit(fixed_spec, snapshot.block.schema).run(snapshot)
    forged_fixed_covariance = np.eye(fixed_result.parameter_values.shape[1]).reshape(
        fixed_result.covariance.shape
    )
    with pytest.raises(ValueError, match="fixed parameter covariance"):
        replace(
            fixed_result,
            covariance=forged_fixed_covariance,
            covariance_valid=np.array([True]),
        )
    np.testing.assert_allclose(
        result.rmse,
        np.sqrt(result.residual_sum_squares / result.used_observation_counts),
    )
    with pytest.raises(ValueError):
        result.rmse.flags.writeable = True
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
            r_squared=np.array([0.0]),
            r_squared_valid=np.array([False]),
        )
    positive_zero = replace(result, residual_sum_squares=np.array([0.0]))
    negative_zero = replace(result, residual_sum_squares=np.array([-0.0]))
    assert encode_fit_result_batch(positive_zero) == encode_fit_result_batch(negative_zero)
    assert not np.signbit(FitParameterConstraint("center", fixed=-0.0).fixed)
