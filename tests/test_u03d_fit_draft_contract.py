"""Headless authority-draft and lossless Fit replay contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisLayout,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    FitNumericPolicy,
    FitResultBatch,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    suggest_fit_draft,
)
from zlc_frontend.figure import (
    AxisViewRole,
    SuggestionStatus,
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
        ValueSchema.scalar(np.dtype("<f8"), "count"),
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


@pytest.mark.parametrize(
    ("factory", "repeats", "expected_role"),
    (
        (_curve_product, 3, AxisViewRole.BATCH),
        (_curve_product, 33, AxisViewRole.BATCH),
        (_image_product, 2, AxisViewRole.FACET),
        (_image_product, 37, AxisViewRole.FACET),
    ),
)
def test_fit_replay_preserves_every_repeat_without_sampling_or_limits(
    factory,
    repeats,
    expected_role,
):
    snapshot, _bound, result = factory(repeats)

    suggestion = suggest_fit_view(snapshot.block.schema, result)

    repeat_id = snapshot.block.schema.repeat_axis.axis_id
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
