"""Named-axis fit contracts, packing, models, budgets, and artifact codecs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    HISTOGRAM_BIN,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisLayout,
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
    FitModelDefinition,
    FitNumericPolicy,
    FitParameterConstraint,
    FitParameterDefinition,
    FitParameterDomain,
    FitResultBatch,
    FitSpec,
    OwnedSnapshot,
    ParameterUnitRelation,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    bind_fit,
    commit_transform,
    decode_fit_result_batch,
    decode_fit_spec,
    encode_fit_result_batch,
    encode_fit_spec,
    fit_model_catalog,
    fit_model_definition,
    fit_spec_from_tree,
    fit_spec_to_tree,
)
from zlc_data.fit_problem import build_fit_problem
from zlc_data.fit_model import (
    evaluate_fit_model,
    initialize_fit_model,
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


def test_generic_curve_models_accept_authoritative_histogram_bin_axes():
    x = np.linspace(-3.0, 3.0, 31)
    bins = axis("bins", HISTOGRAM_BIN, x.size, coordinates=x, unit="count")
    values = 0.5 + 4.0 * np.exp(-((x - 0.4) ** 2) / (2.0 * 0.8**2))
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(bins,),
        point_layout=PointLayout.rect_c((bins.size,)),
        values=values.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (bins.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], (4.0, 0.5, 0.8, 0.4), rtol=1e-5)
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
            "import sys, zlc_data; assert 'scipy' not in sys.modules; "
            "assert callable(zlc_data.BoundFit.run); "
            "assert not hasattr(zlc_data, 'fit_analysis')",
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
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((3,)),
        values=np.zeros((1, 3), dtype=np.float64),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    with pytest.raises(ValueError, match="consecutively float64-representable"):
        bind_fit(spec, snapshot.block.schema)


def test_sparse_implicit_fit_cost_tracks_present_rows_and_preserves_axis_unit():
    logical_size = 5_000_000
    scan = AxisSpec(
        AxisId("scan"),
        "scan",
        SCAN_POINT,
        logical_size,
        unit="MHz",
    )
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.explicit(
            (logical_size,),
            ((0,), (logical_size - 1,)),
        ),
        values=np.array([[1.0, 2.0]]),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
    )
    bound = bind_fit(spec, snapshot.block.schema)
    tracemalloc.start()
    tracemalloc.clear_traces()
    problem = build_fit_problem(bound, snapshot)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    np.testing.assert_array_equal(
        problem.independent_values[0],
        (0.0, float(logical_size - 1)),
    )
    assert bound.parameter_units == ("count", "count", "MHz", "MHz")
    assert peak < 2_000_000


def test_sparse_declared_fit_validates_coordinates_only_when_binding(monkeypatch):
    logical_size = 20_000
    group_count = 12
    group = axis("group", SITE, group_count)
    scan = axis(
        "scan",
        SCAN_POINT,
        logical_size,
        coordinates=range(logical_size),
        unit="MHz",
    )
    mapping = tuple(
        (group_index, scan_index)
        for group_index in range(group_count)
        for scan_index in (0, logical_size - 1)
    )
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(group, scan),
        point_layout=PointLayout.explicit((group_count, logical_size), mapping),
        values=np.ones((1, len(mapping))),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id, group.axis_id),
        "gaussian_offset",
    )
    bound = bind_fit(spec, snapshot.block.schema)

    import zlc_data.fit_contract as fit_contract_module

    def reject_revalidation(_axis):
        raise AssertionError("packing repeated fit-coordinate admission")

    monkeypatch.setattr(
        fit_contract_module,
        "_coordinate_source_for_axis",
        reject_revalidation,
    )
    problem = build_fit_problem(bound, snapshot)

    assert problem.batch_layout.storage_size == group_count
    assert bound.parameter_units == ("count", "count", "MHz", "MHz")


@pytest.mark.parametrize(
    "layout",
    (
        PointLayout.rect_f((2, 3)),
        PointLayout.explicit((2, 3), ((0, 0), (1, 2))),
    ),
    ids=("rect-f", "sparse-explicit"),
)
def test_identity_and_noop_transform_share_one_source_schema_owner(layout):
    group = axis("group", SITE, 2)
    scan = axis("scan", SCAN_POINT, 3, coordinates=(-1.0, 0.0, 1.0), unit="MHz")
    snapshot = snapshot_for(
        repeat=1,
        point_axes=(group, scan),
        point_layout=layout,
        values=np.zeros((1, layout.storage_size), dtype=np.float64),
    )
    identity_spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id, group.axis_id),
        "gaussian_offset",
    )
    no_op = commit_transform(
        snapshot.block.schema,
        DataTransformSpec((Selection.index_range(scan.axis_id, 0, scan.size),)),
    )
    transformed_spec = replace(identity_spec, committed_transform=no_op)

    identity_schema = bind_fit(identity_spec, snapshot.block.schema).effective_schema
    transformed_schema = bind_fit(
        transformed_spec, snapshot.block.schema
    ).effective_schema
    assert identity_schema == transformed_schema
    assert type(identity_schema.cell_layout) is AxisLayout


def test_identity_fit_defers_transformed_schema_digest_and_caches_first_read(monkeypatch):
    snapshot, scan = gaussian_snapshot(repeat=1)
    import zlc_data.transform as transform_module

    digest = transform_module.canonical_digest
    calls = 0

    def counted_digest(value):
        nonlocal calls
        calls += 1
        return digest(value)

    monkeypatch.setattr(transform_module, "canonical_digest", counted_digest)
    bound = bind_fit(gaussian_spec(snapshot, scan), snapshot.block.schema)
    assert calls == 0

    first = bound.effective_schema.fingerprint
    assert calls == 1
    assert bound.effective_schema.fingerprint == first
    assert calls == 1


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
            (Selection.index(snapshot.block.schema.repeat_axis.axis_id, 0),)
        ),
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


def test_bound_fit_derives_effective_axis_metadata_instead_of_accepting_it():
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
    with pytest.raises(TypeError):
        BoundFit(bound.spec, bound.expected_schema, forged_schema)


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
        FitBatchStatus.CONVERGED,
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
        FitBatchStatus.CONVERGED,
        FitBatchStatus.NO_VALID_DATA,
        FitBatchStatus.CONVERGED,
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


@pytest.mark.parametrize(
    "point_layout",
    (
        PointLayout.rect_f((3, 6)),
        PointLayout.explicit(
            (3, 6),
            tuple((group, scan) for group in (0, 2) for scan in range(6)),
        ),
    ),
    ids=("rect-f", "sparse-explicit"),
)
def test_mixed_cell_data_batch_plan_matches_source_values(point_layout):
    group = axis("group", SITE, 3)
    scan = axis("scan", SCAN_POINT, 6, coordinates=np.linspace(-2, 2, 6))
    site = axis("site", AxisRoleId("grid-x"), 2)
    group_indices = point_layout.axis_indices(0)
    scan_indices = point_layout.axis_indices(1)
    values = np.empty((2, point_layout.storage_size, site.size), dtype=np.float64)
    for repeat_index in range(2):
        for point_storage_index in range(point_layout.storage_size):
            for site_index in range(site.size):
                values[repeat_index, point_storage_index, site_index] = (
                    100 * repeat_index
                    + 10 * group_indices[point_storage_index]
                    + site_index
                    + scan_indices[point_storage_index] / 10
                )
    snapshot = snapshot_for(
        repeat=2,
        point_axes=(group, scan),
        point_layout=point_layout,
        data_axes=(site,),
        values=values,
    )
    repeat_axis = snapshot.block.schema.repeat_axis
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (site.axis_id, group.axis_id, repeat_axis.axis_id),
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
            (Selection.index(snapshot.block.schema.repeat_axis.axis_id, 0),)
        ),
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
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-3, atol=2e-3)


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
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )

    def run(committed_transform):
        spec = FitSpec(
            snapshot.block.schema.fingerprint,
            committed_transform,
            (x.axis_id, y.axis_id),
            (snapshot.block.schema.repeat_axis.axis_id,),
            "radial_gaussian_center",
        )
        return bind_fit(spec, snapshot.block.schema).run(snapshot)

    full = run(None)
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
        snapshot.block.schema.fingerprint,
        roi_transform,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
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


def test_value_aware_2d_sampling_keeps_a_narrow_feature_between_grid_points():
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
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "radial_gaussian_center",
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=12_000),
    )

    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert problem.used_observation_counts[0] == 12_000
    assert np.max(problem.observations) == pytest.approx(5_100.0)
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    np.testing.assert_allclose(result.parameter_values[0], expected, rtol=2e-3, atol=2e-3)


def test_value_aware_sampling_does_not_turn_one_outlier_into_the_only_feature():
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
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=image.reshape(1, 1, *image.shape),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "radial_gaussian_center",
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=12_000),
    )

    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.CONVERGED,)
    assert result.parameter_values[0, 2] > 30.0
    np.testing.assert_allclose(
        result.parameter_values[0, 3:5],
        expected[3:5],
        atol=0.05,
    )


def test_valid_nonfinite_is_sampled_fail_closed_while_invalid_nonfinite_is_absent():
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=signal.reshape(1, -1),
        validity=CellValidity(validity),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=5),
    )

    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert np.count_nonzero(np.isnan(problem.observations)) == 1
    assert coordinate_values[90] in problem.independent_values[0]
    assert coordinate_values[91] not in problem.independent_values[0]
    result = bind_fit(spec, snapshot.block.schema).run(snapshot)
    assert result.statuses == (FitBatchStatus.NUMERIC_ERROR,)


def test_total_packed_observation_budget_rejects_before_concatenation(monkeypatch):
    import zlc_data.fit_problem as problem_module

    coordinates = np.linspace(-2.0, 2.0, 5)
    scan = axis("scan", SCAN_POINT, coordinates.size, coordinates=coordinates)
    snapshot = snapshot_for(
        repeat=3,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=np.ones((3, scan.size)),
    )
    allowed_policy = FitNumericPolicy(
        sample_budget_per_batch=5,
        max_packed_observations=15,
    )
    allowed_spec = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=allowed_policy,
    )
    problem = build_fit_problem(
        bind_fit(allowed_spec, snapshot.block.schema),
        snapshot,
    )
    assert problem.observations.size == 15
    result = bind_fit(allowed_spec, snapshot.block.schema).run(snapshot)

    restricted_spec = replace(
        allowed_spec,
        numeric_policy=replace(allowed_policy, max_packed_observations=10),
    )
    with pytest.raises(ValueError, match="packed-observation budget"):
        replace(problem, spec=restricted_spec)
    with pytest.raises(ValueError, match="packed-observation budget"):
        replace(result, spec=restricted_spec)

    def unexpected_concatenation(_parts):
        raise AssertionError("packed arrays must not be materialized after the total cap")

    monkeypatch.setattr(
        problem_module,
        "_concatenate_float64",
        unexpected_concatenation,
    )
    with pytest.raises(ValueError, match="max_packed_observations=10"):
        build_fit_problem(
            bind_fit(restricted_spec, snapshot.block.schema),
            snapshot,
        )


def test_observation_order_is_derived_once_per_cell_batch_group(monkeypatch):
    import zlc_data.fit_problem as problem_module

    scan = axis("scan", SCAN_POINT, 5, coordinates=np.linspace(-2.0, 2.0, 5))
    site = axis("site", SITE, 4)
    snapshot = snapshot_for(
        repeat=3,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        data_axes=(site,),
        values=np.ones((3, scan.size, site.size)),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id, site.axis_id),
        "gaussian_offset",
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=5),
    )
    derive = problem_module._canonical_observation_order
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return derive(*args, **kwargs)

    monkeypatch.setattr(problem_module, "_canonical_observation_order", counted)
    problem = build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)
    assert problem.batch_layout.storage_size == 12
    assert calls == 3


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
    assert result.statuses == (FitBatchStatus.CONVERGED,)
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
    assert ordered.statuses == permuted.statuses == (FitBatchStatus.CONVERGED,)
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


def test_duplicate_coordinates_use_logical_index_not_storage_order_as_tie_break():
    coordinate_values = np.repeat(np.linspace(-4.0, 4.0, 20), 2)
    scan = axis(
        "duplicate_scan",
        SCAN_POINT,
        coordinate_values.size,
        coordinates=coordinate_values,
        unit="MHz",
    )
    logical_signal = (
        3.0 * np.exp(-((coordinate_values - 0.4) ** 2) / (2.0 * 1.1**2))
        + 0.5
        + np.tile((-0.2, 0.3), 20)
    )
    logical_validity = np.ones(scan.size, dtype=bool)
    logical_validity[7] = False

    def pack(permutation: np.ndarray):
        layout = PointLayout.explicit(
            (scan.size,),
            tuple((int(index),) for index in permutation),
        )
        snapshot = snapshot_for(
            repeat=1,
            point_axes=(scan,),
            point_layout=layout,
            values=logical_signal[permutation].reshape(1, -1),
            validity=CellValidity(logical_validity[permutation].reshape(1, -1)),
        )
        spec = FitSpec(
            snapshot.block.schema.fingerprint,
            None,
            (scan.axis_id,),
            (snapshot.block.schema.repeat_axis.axis_id,),
            "gaussian_offset",
            numeric_policy=FitNumericPolicy(sample_budget_per_batch=8),
        )
        return build_fit_problem(bind_fit(spec, snapshot.block.schema), snapshot)

    ordered = pack(np.arange(scan.size))
    pair_swapped = pack(
        np.arange(scan.size).reshape(-1, 2)[:, ::-1].reshape(-1)
    )
    np.testing.assert_array_equal(
        ordered.independent_values[0],
        pair_swapped.independent_values[0],
    )
    np.testing.assert_array_equal(ordered.observations, pair_swapped.observations)
    np.testing.assert_array_equal(
        ordered.used_observation_counts,
        pair_swapped.used_observation_counts,
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
        point_axes=(time_axis,),
        point_layout=PointLayout.rect_c((x.size,)),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (time_axis.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
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
        point_axes=(),
        point_layout=PointLayout.rect_c(()),
        data_axes=(x, y),
        values=np.ones((1, 1, 3, 3)),
    )
    names = ("amplitude", "offset", "one_over_e_radius", "center_x", "center_y")
    values = (0.0, 1.0, 1.0, 0.0, 0.0)
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (x.axis_id, y.axis_id),
        (snapshot.block.schema.repeat_axis.axis_id,),
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=expected.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
        "gaussian_offset",
        constraints=tuple(
            FitParameterConstraint(name, fixed=value)
            for name, value in zip(
                ("amplitude", "offset", "sigma", "center"),
                (1.0, 0.0, 1.0, 0.0),
            )
        ),
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=2),
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=np.stack((expected, np.full(scan.size, 1e308))),
    )
    fixed = (1.0, 0.0, 1.0, 0.0)
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=signal.reshape(1, -1),
    )
    spec = FitSpec(
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
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
            snapshot.block.schema.fingerprint,
            None,
            (scan.axis_id,),
            (snapshot.block.schema.repeat_axis.axis_id,),
            "symmetric_lorentzian_doublet",
            constraints=(FitParameterConstraint("center_splitting", lower=-1.0, upper=0.0),),
        )
    with pytest.raises(ValueError, match="PHASE_RADIANS domain"):
        FitSpec(
            snapshot.block.schema.fingerprint,
            None,
            (scan.axis_id,),
            (snapshot.block.schema.repeat_axis.axis_id,),
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
                snapshot.block.schema.fingerprint,
                None,
                (scan.axis_id,),
                (snapshot.block.schema.repeat_axis.axis_id,),
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
        snapshot.block.schema.fingerprint,
        None,
        (scan.axis_id,),
        (snapshot.block.schema.repeat_axis.axis_id,),
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


def test_fixed_bounded_initial_constraints_and_numeric_budgets_fail_closed():
    snapshot, scan = gaussian_snapshot(repeat=1)
    fixed_spec = gaussian_spec(
        snapshot,
        scan,
        constraints=(FitParameterConstraint("offset", fixed=1.2),),
    )
    fixed = bind_fit(fixed_spec, snapshot.block.schema).run(snapshot)
    assert fixed.statuses == (FitBatchStatus.CONVERGED,)
    assert fixed.parameter_values[0, 1] == 1.2

    impossible = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=4),
    )
    with pytest.raises(ValueError, match="below.*minimum observation count"):
        bind_fit(impossible, snapshot.block.schema)

    impossible_total = gaussian_spec(
        snapshot,
        scan,
        numeric_policy=FitNumericPolicy(max_packed_observations=4),
    )
    with pytest.raises(ValueError, match="max_packed_observations.*below.*minimum"):
        bind_fit(impossible_total, snapshot.block.schema)

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

    ordered_integers = np.array(
        [0, 1, 2, 3, 4, 5, 2**53, 2**53 + 1],
        dtype=np.uint64,
    )
    ordered_snapshot = snapshot_for(
        repeat=1,
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=ordered_integers.reshape(1, -1),
        dtype=np.dtype("<u8"),
    )
    ordered_spec = replace(
        spec,
        input_schema_fingerprint=ordered_snapshot.block.schema.fingerprint,
        numeric_policy=FitNumericPolicy(sample_budget_per_batch=5),
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
    assert result_tree["fit_spec"]["numeric_policy"]["max_packed_observations"] \
        == 2_000_000
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
        point_axes=(scan,),
        point_layout=PointLayout.rect_c((scan.size,)),
        values=np.zeros((1, scan.size)),
        block_id="golden-fit-source",
    )
    spec = gaussian_spec(snapshot, scan)
    result = FitResultBatch(
        source_ref=snapshot.ref,
        spec=spec,
        fit_axis_specs=(scan,),
        batch_axis_specs=(snapshot.block.schema.repeat_axis,),
        batch_layout=AxisLayout.rect_c((1,)),
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
    assert len(payload) == 2735
    assert hashlib.sha256(payload).hexdigest() == (
        "615906e94861e9e74fa84904f0bcfed183532b1cbd15febe472f25a6aa26e9a5"
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
