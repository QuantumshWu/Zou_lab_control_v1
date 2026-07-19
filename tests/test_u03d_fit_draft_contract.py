"""Headless authority-draft and lossless Fit replay contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_data import (
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
    CoordinateFrameId,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    FitNumericPolicy,
    FitResultBatch,
    IndexRangeSelection,
    MissingPolicy,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValidityPolicy,
    ValueSchema,
    ReductionMethod,
    ReductionSpec,
    bind_fit,
    bound_fit_execution_peak_upper_bound_nbytes,
    commit_transform,
    fit_spec_for,
    suggest_fit_draft,
)
from zlc_frontend.figure import (
    AxisViewRole,
    SuggestionStatus,
    selection_fit_view_projection,
    suggest_fit_view,
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


def _snapshot(schema: DatasetSchema, values: np.ndarray, identity: str) -> OwnedSnapshot:
    block = DataBlock(
        BlockId(identity),
        DatasetRevision(1),
        np.asarray(values, dtype=schema.cell_schema.dtype),
        VALID,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(identity + "-generation")),
        block,
    )


def _metadata_only_result(snapshot: OwnedSnapshot, bound) -> FitResultBatch:
    """Make a valid failed result so replay tests exercise metadata only."""

    fit_axes = tuple(
        bound.effective_schema.axis(axis_id)
        for axis_id in bound.spec.fit_axis_ids
    )
    batch_axes = tuple(
        bound.effective_schema.axis(axis_id)
        for axis_id in bound.spec.batch_axis_ids
    )
    batch_layout = AxisLayout.rect_c(tuple(axis.size for axis in batch_axes))
    batch_size = batch_layout.storage_size
    parameter_count = len(bound.parameter_definitions)
    zeros_i = np.zeros(batch_size, dtype=np.int64)
    zeros_f = np.zeros(batch_size, dtype=np.float64)
    return FitResultBatch(
        source_ref=snapshot.ref,
        spec=bound.spec,
        fit_axis_specs=fit_axes,
        batch_axis_specs=batch_axes,
        batch_layout=batch_layout,
        value_unit=bound.effective_schema.value_unit,
        parameter_values=np.zeros((batch_size, parameter_count), dtype=np.float64),
        covariance=np.zeros(
            (batch_size, parameter_count, parameter_count), dtype=np.float64
        ),
        covariance_valid=np.zeros(batch_size, dtype=bool),
        statuses=(FitBatchStatus.NO_VALID_DATA,) * batch_size,
        errors=("no valid observations",) * batch_size,
        present_observation_counts=zeros_i,
        valid_observation_counts=zeros_i,
        used_observation_counts=zeros_i,
        evaluation_counts=zeros_i,
        residual_sum_squares=zeros_f,
        r_squared=zeros_f,
        r_squared_valid=np.zeros(batch_size, dtype=bool),
        scipy_version="test",
    )


def _curve_product(repeats: int):
    repeat = _axis("repeat", REPEAT, repeats, coordinates=range(repeats))
    scan = _axis(
        "detuning",
        SCAN_POINT,
        9,
        coordinates=np.linspace(-2.0, 2.0, 9),
        unit="MHz",
    )
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema((), ValidityContract.value(), np.dtype("<f8"), "count"),
    )
    snapshot = _snapshot(schema, np.zeros(schema.physical_shape), f"curve-{repeats}")
    bound = suggest_fit_draft(
        schema,
        "gaussian_offset",
        fit_axis_ids=(scan.axis_id,),
    )
    return snapshot, bound, _metadata_only_result(snapshot, bound)


def _image_product(repeats: int):
    repeat = _axis("repeat", REPEAT, repeats, coordinates=range(repeats))
    frame = "camera"
    y_axis = _axis(
        "camera.y", SPATIAL_Y, 3, coordinates=range(3), unit="pixel", frame=frame
    )
    x_axis = _axis(
        "camera.x", SPATIAL_X, 4, coordinates=range(4), unit="pixel", frame=frame
    )
    schema = DatasetSchema(
        repeat,
        (),
        PointLayout.rect_c(()),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.value(),
            np.dtype("<f8"),
            "count",
        ),
    )
    snapshot = _snapshot(schema, np.zeros(schema.physical_shape), f"image-{repeats}")
    bound = suggest_fit_draft(
        schema,
        "radial_gaussian_center",
        fit_axis_ids=(x_axis.axis_id, y_axis.axis_id),
    )
    return snapshot, bound, _metadata_only_result(snapshot, bound)


def test_authority_draft_preserves_every_nonfit_axis_and_never_accepts_display_state():
    repeat = _axis("repeat", REPEAT, 2)
    scan = _axis("scan", SCAN_POINT, 9, coordinates=range(9))
    site = _axis("site", SITE, 3, coordinates=("a", "b", "c"))
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema((site,), ValidityContract.value(), np.dtype("<f8")),
    )
    policy = FitNumericPolicy(max_evaluations=123)

    bound = suggest_fit_draft(
        schema,
        "gaussian_offset",
        fit_axis_ids=(scan.axis_id,),
        numeric_policy=policy,
    )

    assert bound.spec.committed_transform is None
    assert bound.spec.fit_axis_ids == (scan.axis_id,)
    assert bound.spec.batch_axis_ids == (repeat.axis_id, site.axis_id)
    assert bound.spec.numeric_policy == policy
    with pytest.raises(ValueError, match="range-preserving"):
        suggest_fit_draft(
            schema,
            "gaussian_offset",
            fit_axis_ids=(scan.axis_id,),
            selection=Selection.index(scan.axis_id, 0),
        )
    with pytest.raises(ValueError, match="only explicit fit axes"):
        suggest_fit_draft(
            schema,
            "gaussian_offset",
            fit_axis_ids=(scan.axis_id,),
            selection=Selection.index_range(site.axis_id, 0, 2),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        suggest_fit_draft(
            schema,
            "gaussian_offset",
            fit_axis_ids=(scan.axis_id,),
            view_spec=object(),
        )


@pytest.mark.parametrize("fit_role", (SCAN_POINT, SPECTRAL))
@pytest.mark.parametrize("layout_kind", ("rect_c", "rect_f", "sparse"))
def test_point_fit_range_projection_reconstructs_resolved_layout_exactly(
    fit_role,
    layout_kind,
):
    repeat = _axis("repeat", REPEAT, 2)
    site = _axis("site", SITE, 2, coordinates=("left", "right"))
    fit_axis = _axis(
        "frequency",
        fit_role,
        7,
        coordinates=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        unit="MHz",
    )
    shape = (site.size, fit_axis.size)
    if layout_kind == "rect_c":
        point_layout = PointLayout.rect_c(shape)
    elif layout_kind == "rect_f":
        point_layout = PointLayout.rect_f(shape)
    else:
        point_layout = PointLayout.explicit(
            shape,
            tuple((0, index) for index in range(7))
            + tuple((1, index) for index in (0, 2, 3, 5, 6)),
        )
    schema = DatasetSchema(
        repeat,
        (site, fit_axis),
        point_layout,
        ValueSchema((), ValidityContract.value(), np.dtype("<f8")),
    )
    roi = Selection.index_range(fit_axis.axis_id, 1, 6)

    bound = suggest_fit_draft(
        schema,
        "gaussian_offset",
        fit_axis_ids=(fit_axis.axis_id,),
        selection=roi,
    )
    projected_schema, projected_roi = selection_fit_view_projection(bound)

    assert projected_roi == roi
    assert projected_schema.repeat_axis == bound.effective_schema.cell_axes[0]
    assert projected_schema.point_axes == bound.effective_schema.cell_axes[1:]
    assert projected_schema.cell_schema.data_axes == bound.effective_schema.data_axes
    assert projected_schema.cell_layout == bound.effective_schema.cell_layout
    assert projected_schema.point_axes[1].coordinates == (-2.0, -1.0, 0.0, 1.0, 2.0)
    assert projected_schema.point_axes[1].index_origin == 0
    assert bound.spec.batch_axis_ids == (repeat.axis_id, site.axis_id)


def test_spatial_box_draft_keeps_both_physical_fit_axes_and_all_batches():
    repeat = _axis("repeat", REPEAT, 3)
    scan = _axis("scan", SCAN_POINT, 2, coordinates=(10.0, 20.0))
    site = _axis("site", SITE, 2, coordinates=("left", "right"))
    frame = "camera"
    y_axis = _axis(
        "camera.y", SPATIAL_Y, 8, coordinates=range(8), unit="pixel", frame=frame
    )
    x_axis = _axis(
        "camera.x", SPATIAL_X, 10, coordinates=range(10), unit="pixel", frame=frame
    )
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (site, y_axis, x_axis),
            ValidityContract.value(),
            np.dtype("<f8"),
        ),
    )
    roi = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        2,
        7,
        1,
        5,
        coordinate_frame=CoordinateFrameId(frame),
    )

    bound = suggest_fit_draft(
        schema,
        "radial_gaussian_center",
        fit_axis_ids=(x_axis.axis_id, y_axis.axis_id),
        selection=roi,
    )
    projected_schema, projected_roi = selection_fit_view_projection(bound)

    assert projected_roi == roi
    assert bound.spec.fit_axis_ids == (x_axis.axis_id, y_axis.axis_id)
    assert bound.spec.batch_axis_ids == (
        repeat.axis_id,
        scan.axis_id,
        site.axis_id,
    )
    assert tuple(axis.size for axis in projected_schema.cell_schema.data_axes) == (2, 5, 6)
    assert projected_schema.cell_layout == bound.effective_schema.cell_layout


def test_projection_rejects_reduction_index_drop_and_batch_axis_range():
    repeat = _axis("repeat", REPEAT, 1)
    scan = _axis("scan", SCAN_POINT, 9, coordinates=range(9))
    site = _axis("site", SITE, 3, coordinates=("a", "b", "c"))
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema((site,), ValidityContract.value(), np.dtype("<f8")),
    )

    reduced = commit_transform(
        schema,
        DataTransformSpec(
            (
                Selection.index_range(scan.axis_id, 1, 8),
                ReductionSpec(
                    (site.axis_id,),
                    ReductionMethod.MEAN,
                    MissingPolicy.REQUIRE_ALL,
                    ValidityPolicy.REQUIRE_ALL,
                ),
            )
        ),
    )
    reduced_bound = bind_fit(
        fit_spec_for(
            schema,
            "gaussian_offset",
            committed_transform=reduced,
            fit_axis_ids=(scan.axis_id,),
        ),
        schema,
    )
    with pytest.raises(ValueError, match="exactly one range Selection"):
        selection_fit_view_projection(reduced_bound)

    for forbidden in (
        Selection.index(site.axis_id, 0),
        Selection.index_range(site.axis_id, 0, 2),
    ):
        committed = commit_transform(schema, DataTransformSpec((forbidden,)))
        bound = bind_fit(
            fit_spec_for(
                schema,
                "gaussian_offset",
                committed_transform=committed,
                fit_axis_ids=(scan.axis_id,),
            ),
            schema,
        )
        expected = "range-preserving" if forbidden.terms[0].__class__.__name__ == "IndexSelection" else "explicit fit axis"
        with pytest.raises(ValueError, match=expected):
            selection_fit_view_projection(bound)


@pytest.mark.parametrize(
    ("factory", "repeats", "expected_role", "expected_code"),
    (
        (_curve_product, 3, AxisViewRole.BATCH, None),
        (_curve_product, 33, None, "BATCH_LIMIT"),
        (_image_product, 2, AxisViewRole.FACET, None),
        (_image_product, 37, None, "FACET_LIMIT"),
    ),
)
def test_fit_replay_never_samples_repeat_zero_or_latest(
    factory,
    repeats,
    expected_role,
    expected_code,
):
    snapshot, _bound, result = factory(repeats)

    suggestion = suggest_fit_view(snapshot.block.schema, result)

    repeat_id = snapshot.block.schema.repeat_axis.axis_id
    if expected_code is not None:
        assert suggestion.status is SuggestionStatus.NEEDS_INPUT
        assert suggestion.spec is None
        assert suggestion.reasons[0].code == expected_code
        return
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec is not None
    binding = suggestion.spec.binding(repeat_id)
    assert binding.role is expected_role
    assert binding.selector is None
    assert all(
        term.axis_id != repeat_id
        for selection in suggestion.spec.display_selections
        for term in selection.terms
    )


def test_data_fit_owner_has_no_frontend_or_display_reducer_dependency():
    source = (Path(__file__).resolve().parents[1] / "zlc_data" / "fit.py").read_text(
        encoding="utf-8"
    )
    assert "zlc_frontend" not in source
    assert "ViewSpec" not in source
    assert "DisplayReduction" not in source


def test_execution_peak_bound_is_data_free_conservative_and_policy_monotone():
    _snapshot_value, small_bound, _result = _curve_product(3)
    _snapshot_value, larger_batch_bound, _result = _curve_product(33)
    scan_id = small_bound.spec.fit_axis_ids[0]
    wider_policy = replace(
        small_bound.spec.numeric_policy,
        max_evaluations=8_000,
        sample_budget_per_batch=24_000,
        max_packed_observations=4_000_000,
    )
    wider_bound = suggest_fit_draft(
        small_bound.expected_schema,
        small_bound.spec.model_id,
        fit_axis_ids=(scan_id,),
        numeric_policy=wider_policy,
    )

    small = bound_fit_execution_peak_upper_bound_nbytes(small_bound)
    larger_batch = bound_fit_execution_peak_upper_bound_nbytes(larger_batch_bound)
    wider = bound_fit_execution_peak_upper_bound_nbytes(wider_bound)

    assert small > 8 * 1024 * 1024
    assert larger_batch > small
    assert wider > small
    with pytest.raises(TypeError, match="bound must be BoundFit"):
        bound_fit_execution_peak_upper_bound_nbytes(object())
