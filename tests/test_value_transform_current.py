"""Focused current contracts for transforms over one multidimensional Value."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    CoordinateRangeSelection,
    DataTransformSpec,
    IndexRangeSelection,
    Invalid,
    ReductionMethod,
    ReductionSpec,
    Selection,
    Valid,
    ValidityContract,
    ValidityPolicy,
    Value,
    ValueSchema,
    apply_value_transform,
    resolve_value_transform_schema,
)


def _camera_value() -> tuple[Value, AxisSpec, AxisSpec, AxisSpec, np.ndarray]:
    camera_frame = CoordinateFrameId("camera-output-pixel")
    spectral_frame = CoordinateFrameId("spectrometer-wavelength")
    y_axis = AxisSpec(
        AxisId("camera-y"),
        "camera y",
        SPATIAL_Y,
        4,
        (10, 11, 12, 13),
        "pixel",
        camera_frame,
    )
    x_axis = AxisSpec(
        AxisId("camera-x"),
        "camera x",
        SPATIAL_X,
        5,
        None,
        "pixel",
        camera_frame,
        40,
    )
    spectral_axis = AxisSpec(
        AxisId("wavelength"),
        "wavelength",
        SPECTRAL,
        3,
        (780.0, 781.0, 782.0),
        "nm",
        spectral_frame,
    )
    values = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)
    valid = np.ones(values.shape, dtype=bool)
    valid[2, 2, 1] = False
    schema = ValueSchema(
        (y_axis, x_axis, spectral_axis),
        ValidityContract.components(
            y_axis.axis_id,
            x_axis.axis_id,
            spectral_axis.axis_id,
        ),
        values.dtype,
        "count",
    )
    return (
        Value(
            values,
            ComponentValidity(
                (y_axis.axis_id, x_axis.axis_id, spectral_axis.axis_id),
                valid,
            ),
            schema,
        ),
        y_axis,
        x_axis,
        spectral_axis,
        valid,
    )


def _rectangle(y_axis: AxisSpec, x_axis: AxisSpec) -> Selection:
    return Selection(
        (
            CoordinateRangeSelection(
                y_axis.axis_id,
                11,
                13,
                y_axis.coordinate_frame,
            ),
            IndexRangeSelection(x_axis.axis_id, 1, 4),
        )
    )


def test_value_rectangle_crop_preserves_unselected_axis_and_component_validity():
    value, y_axis, x_axis, spectral_axis, valid = _camera_value()
    spec = DataTransformSpec((_rectangle(y_axis, x_axis),))

    resolved = resolve_value_transform_schema(value.schema, spec)
    transformed = apply_value_transform(value, spec)

    assert transformed.schema == resolved
    assert tuple(axis.axis_id for axis in resolved.data_axes) == (
        y_axis.axis_id,
        x_axis.axis_id,
        spectral_axis.axis_id,
    )
    selected_y, selected_x, preserved_spectral = resolved.data_axes
    assert selected_y.coordinates == (11, 12, 13)
    assert selected_y.unit == y_axis.unit
    assert selected_y.coordinate_frame == y_axis.coordinate_frame
    assert selected_x.coordinates is None
    assert selected_x.index_origin == 41
    assert selected_x.unit == x_axis.unit
    assert selected_x.coordinate_frame == x_axis.coordinate_frame
    assert preserved_spectral is spectral_axis
    assert resolved.value_unit == "count"
    np.testing.assert_array_equal(
        transformed.values,
        value.values[1:4, 1:4, :],
    )
    assert isinstance(transformed.validity, ComponentValidity)
    assert transformed.validity.axis_ids == (
        y_axis.axis_id,
        x_axis.axis_id,
        spectral_axis.axis_id,
    )
    np.testing.assert_array_equal(
        transformed.validity.mask,
        valid[1:4, 1:4, :],
    )


def test_value_spatial_reduction_preserves_spectral_axis_and_validity():
    value, y_axis, x_axis, spectral_axis, valid = _camera_value()
    reduction = ReductionSpec(
        (y_axis.axis_id, x_axis.axis_id),
        ReductionMethod.SUM,
        validity_policy=ValidityPolicy.REQUIRE_ALL,
    )
    spec = DataTransformSpec((_rectangle(y_axis, x_axis), reduction))

    resolved = resolve_value_transform_schema(value.schema, spec)
    transformed = apply_value_transform(value, spec)

    assert transformed.schema == resolved
    assert resolved.data_axes == (spectral_axis,)
    assert resolved.validity_contract == ValidityContract.components(
        spectral_axis.axis_id
    )
    assert resolved.value_unit == "count"
    expected = np.sum(value.values[1:4, 1:4, :], axis=(0, 1))
    expected[1] = 0.0
    np.testing.assert_array_equal(transformed.values, expected)
    assert isinstance(transformed.validity, ComponentValidity)
    assert transformed.validity.axis_ids == (spectral_axis.axis_id,)
    np.testing.assert_array_equal(
        transformed.validity.mask,
        np.array([True, False, True]),
    )

    scalar = apply_value_transform(
        value,
        DataTransformSpec(
            (
                _rectangle(y_axis, x_axis),
                ReductionSpec(
                    (y_axis.axis_id, x_axis.axis_id, spectral_axis.axis_id),
                    ReductionMethod.SUM,
                    validity_policy=ValidityPolicy.REQUIRE_ALL,
                ),
            )
        ),
    )
    assert scalar.schema.data_axes == ()
    assert scalar.values.shape == ()
    assert isinstance(scalar.validity, Invalid)


def test_value_transform_has_no_cell_axis_namespace():
    value, _y_axis, _x_axis, _spectral_axis, _valid = _camera_value()
    cell_axis_spec = DataTransformSpec(
        (Selection.index(AxisId("repeat"), 0),)
    )

    with pytest.raises(KeyError, match="absent"):
        resolve_value_transform_schema(value.schema, cell_axis_spec)
    with pytest.raises(KeyError, match="absent"):
        apply_value_transform(value, cell_axis_spec)


@pytest.mark.parametrize("method", tuple(ReductionMethod))
@pytest.mark.parametrize("nonfinite", (np.nan, np.inf, -np.inf))
def test_value_reduction_rejects_valid_nonfinite_but_omits_invalid(
    method: ReductionMethod,
    nonfinite: float,
):
    spectral_axis = AxisSpec(
        AxisId("spectral"),
        "spectral",
        SPECTRAL,
        3,
        (1.0, 2.0, 3.0),
        "nm",
    )
    schema = ValueSchema(
        (spectral_axis,),
        ValidityContract.components(spectral_axis.axis_id),
        np.dtype(np.float64),
    )
    values = np.array([1.0, nonfinite, 3.0], dtype=np.float64)
    spec = DataTransformSpec(
        (
            ReductionSpec(
                (spectral_axis.axis_id,),
                method,
                validity_policy=ValidityPolicy.OMIT_INVALID,
            ),
        )
    )

    with pytest.raises(ValueError, match="valid non-finite"):
        apply_value_transform(
            Value(
                values,
                ComponentValidity(
                    (spectral_axis.axis_id,),
                    np.array([True, True, True]),
                ),
                schema,
            ),
            spec,
        )

    omitted = apply_value_transform(
        Value(
            values,
            ComponentValidity(
                (spectral_axis.axis_id,),
                np.array([True, False, True]),
            ),
            schema,
        ),
        spec,
    )
    assert isinstance(omitted.validity, Valid)
    expected = {
        ReductionMethod.SUM: 4.0,
        ReductionMethod.MEAN: 2.0,
        ReductionMethod.MIN: 1.0,
        ReductionMethod.MAX: 3.0,
    }[method]
    assert omitted.values.item() == expected
