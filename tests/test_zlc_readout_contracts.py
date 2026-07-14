from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import itertools
import random

import numpy as np
import pytest

from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    DatasetSchema,
    PointLayout,
    ValidityContract,
    ValueSchema,
)
from zlc_storage import encode
from zlc_neutral_atom.readout.contracts import (
    CalibrationCaptureLayout,
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    FrameContract,
    ReadoutBindingKey,
)
from zlc_neutral_atom.readout.codec import (
    calibration_capture_layout_from_tree,
    calibration_capture_layout_to_tree,
    camera_capture_descriptor_from_tree,
    camera_capture_descriptor_to_tree,
    camera_event_readout_setting_from_tree,
    camera_event_readout_setting_to_tree,
    decode_calibration_capture_layout,
    decode_camera_capture_descriptor,
    decode_camera_event_readout_setting,
    decode_frame_contract,
    decode_readout_binding_key,
    encode_calibration_capture_layout,
    encode_camera_capture_descriptor,
    encode_camera_event_readout_setting,
    encode_frame_contract,
    encode_readout_binding_key,
    frame_contract_from_tree,
    frame_contract_to_tree,
    readout_binding_key_from_tree,
    readout_binding_key_to_tree,
)
from zlc_neutral_atom.runtime.capture import CameraPhysicalFacts


Y = AxisId("camera-y")
X = AxisId("camera-x")
EVENT = AxisId("readout-event")
SCAN = AxisId("detuning")
PHASE = AxisId("phase")
FRAME = CoordinateFrameId("qcm-camera-pixels")
BINDING = ReadoutBindingKey("primary-readout")


def _event_settings() -> tuple[CameraEventReadoutSetting, ...]:
    return (
        CameraEventReadoutSetting(0, 0.001, 1.0, "fast", "0" * 64),
        CameraEventReadoutSetting(1, 0.002, 2.0, "low-noise", "1" * 64),
        CameraEventReadoutSetting(2, 0.003, 3.0, "low-noise", "2" * 64),
    )


def _descriptor(**changes: object) -> CameraCaptureDescriptor:
    value = CameraCaptureDescriptor(
        camera_identity="qcm-camera:serial-42",
        sensor_identity="qcm-sensor:serial-42",
        optical_path="science-imaging-v1",
        sensor_shape_yx=(100, 120),
        roi_origin_yx=(10, 20),
        roi_shape_yx=(40, 60),
        binning_yx=(2, 3),
        spatial_y_axis_id=Y,
        spatial_x_axis_id=X,
        coordinate_frame=FRAME,
        dtype=np.dtype("uint16"),
        count_unit="camera-count",
        readout_event_axis_id=EVENT,
        event_settings=_event_settings(),
        camera_arm_spec_fingerprint="a" * 64,
    )
    return replace(value, **changes)


def _default_point_axes() -> tuple[AxisSpec, ...]:
    return (
        AxisSpec(EVENT, "readout event", READOUT_EVENT, 3),
        AxisSpec(SCAN, "detuning", SCAN_POINT, 2, unit="MHz"),
    )


def _schema(
    *,
    descriptor: CameraCaptureDescriptor | None = None,
    point_axes: tuple[AxisSpec, ...] | None = None,
    point_layout: PointLayout | None = None,
    data_axes: tuple[AxisSpec, ...] | None = None,
    validity: ValidityContract | None = None,
    dtype: object | None = None,
    unit: str = "camera-count",
) -> DatasetSchema:
    descriptor = _descriptor() if descriptor is None else descriptor
    point_axes = _default_point_axes() if point_axes is None else point_axes
    if data_axes is None:
        height, width = descriptor.output_shape_yx
        data_axes = (
            AxisSpec(
                Y,
                "ROI-local y",
                SPATIAL_Y,
                height,
                coordinates=tuple(range(height)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        )
    if point_layout is None:
        point_layout = PointLayout.rect_c(tuple(axis.size for axis in point_axes))
    return DatasetSchema(
        repeat_axis=AxisSpec(AxisId("repeat"), "repeat", REPEAT, 4),
        point_axes=point_axes,
        point_layout=point_layout,
        cell_schema=ValueSchema(
            data_axes,
            ValidityContract.value() if validity is None else validity,
            descriptor.dtype if dtype is None else dtype,
            unit,
        ),
    )


def _calibration_layout(schema: DatasetSchema) -> CalibrationCaptureLayout:
    return CalibrationCaptureLayout.from_schema(
        schema,
        readout_event_axis_id=EVENT,
        reference_event_indices=(2, 0),
        readout_event_index=1,
    )


def _contract() -> FrameContract:
    descriptor = _descriptor()
    schema = _schema(descriptor=descriptor)
    return FrameContract.from_calibration_capture(
        BINDING,
        descriptor,
        schema,
        _calibration_layout(schema),
    )


def _physical_facts() -> CameraPhysicalFacts:
    descriptor = _descriptor()
    setting = descriptor.setting(0)
    return CameraPhysicalFacts(
        camera_identity=descriptor.camera_identity,
        sensor_identity=descriptor.sensor_identity,
        optical_path=descriptor.optical_path,
        capture_trigger_channels=("camera-trigger",),
        sensor_shape_yx=descriptor.sensor_shape_yx,
        roi_origin_yx=descriptor.roi_origin_yx,
        roi_shape_yx=descriptor.roi_shape_yx,
        binning_yx=descriptor.binning_yx,
        spatial_y_axis_id=descriptor.spatial_y_axis_id,
        spatial_x_axis_id=descriptor.spatial_x_axis_id,
        coordinate_frame=descriptor.coordinate_frame,
        dtype=descriptor.dtype,
        count_unit=descriptor.count_unit,
        exposure_seconds=setting.exposure_seconds,
        gain=setting.gain,
        readout_mode=setting.readout_mode,
        opaque_frame_settings_fingerprint="f" * 64,
    )


def test_camera_frame_values_share_geometry_and_count_dtype_normalization() -> None:
    changes = {
        "sensor_shape_yx": [np.int64(100), np.int64(120)],
        "roi_origin_yx": [np.int64(10), np.int64(20)],
        "roi_shape_yx": [np.int64(40), np.int64(60)],
        "binning_yx": [np.int64(2), np.int64(3)],
        "dtype": np.dtype(">u2"),
    }
    descriptor = replace(_descriptor(), **changes)
    contract = replace(_contract(), **changes)
    physical_facts = replace(_physical_facts(), **changes)

    expected_geometry = ((100, 120), (10, 20), (40, 60), (2, 3))
    for value in (descriptor, contract, physical_facts):
        assert (
            value.sensor_shape_yx,
            value.roi_origin_yx,
            value.roi_shape_yx,
            value.binning_yx,
        ) == expected_geometry
        assert value.dtype == np.dtype("<u2")
    assert descriptor.output_shape_yx == physical_facts.output_shape_yx == (20, 20)


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        (
            {"roi_origin_yx": (70, 20)},
            ValueError,
            "camera ROI lies outside the declared sensor geometry",
        ),
        (
            {"roi_shape_yx": (41, 60)},
            ValueError,
            "roi_shape_yx must be exactly divisible by binning_yx",
        ),
        (
            {"binning_yx": (0, 3)},
            ValueError,
            "binning_yx\\[0\\] must be at least 1",
        ),
        (
            {"dtype": np.dtype("complex64")},
            TypeError,
            "dtype must be a real integer or floating dtype",
        ),
    ],
)
def test_camera_frame_values_reject_the_same_invalid_primitives(
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    for value in (_descriptor(), _contract(), _physical_facts()):
        with pytest.raises(error_type, match=message):
            replace(value, **changes)


def test_frame_axes_are_explicit_roi_local_output_pixel_coordinates() -> None:
    descriptor = _descriptor()
    height, width = descriptor.output_shape_yx
    wrong_unit = _schema(
        descriptor=descriptor,
        data_axes=(
            AxisSpec(
                Y,
                "ROI-local y",
                SPATIAL_Y,
                height,
                coordinates=tuple(range(height)),
                unit="px",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        ),
    )
    with pytest.raises(ValueError, match="canonical 'pixel' unit"):
        descriptor.validate_schema(wrong_unit)
    shifted = _schema(
        descriptor=descriptor,
        data_axes=(
            AxisSpec(
                Y,
                "sensor-global y",
                SPATIAL_Y,
                height,
                coordinates=tuple(range(10, 10 + height)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        ),
    )
    with pytest.raises(ValueError, match="ROI-local"):
        descriptor.validate_schema(shifted)
    float_spelling = _schema(
        descriptor=descriptor,
        data_axes=(
            AxisSpec(
                Y,
                "ROI-local y",
                SPATIAL_Y,
                height,
                coordinates=tuple(float(index) for index in range(height)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        ),
    )
    with pytest.raises(ValueError, match="ROI-local"):
        descriptor.validate_schema(float_spelling)


def _single_event_descriptor(
    *,
    exposure: float = 0.002,
    gain: float = 2.0,
    mode: str = "low-noise",
    frame_fingerprint: str | None = "1" * 64,
    **changes: object,
) -> CameraCaptureDescriptor:
    return _descriptor(
        readout_event_axis_id=None,
        event_settings=(
            CameraEventReadoutSetting(
                0,
                exposure,
                gain,
                mode,
                frame_fingerprint,
            ),
        ),
        camera_arm_spec_fingerprint="b" * 64,
        **changes,
    )


def _single_event_schema(
    descriptor: CameraCaptureDescriptor,
    **changes: object,
) -> DatasetSchema:
    return _schema(
        descriptor=descriptor,
        point_axes=(AxisSpec(SCAN, "detuning", SCAN_POINT, 2),),
        **changes,
    )


def test_multievent_calibration_contract_matches_same_single_event_occupancy() -> None:
    calibrated = _contract()
    occupancy = _single_event_descriptor()
    occupancy_schema = _single_event_schema(occupancy)
    observed = FrameContract.from_schema(
        BINDING,
        occupancy,
        occupancy_schema,
        readout_event_index=0,
    )
    assert observed == calibrated
    calibrated.assert_compatible(
        BINDING,
        occupancy,
        occupancy_schema,
        readout_event_index=0,
    )
    assert observed.exposure_seconds == 0.002
    assert observed.gain == 2.0
    assert observed.readout_mode == "low-noise"


def test_reference_schedule_and_capture_fingerprint_do_not_leak_into_frame_applicability() -> None:
    descriptor = _descriptor(
        event_settings=(
            CameraEventReadoutSetting(
                0,
                0.009,
                8.0,
                "reference-only",
                "8" * 64,
            ),
            _event_settings()[1],
            CameraEventReadoutSetting(
                2,
                0.008,
                7.0,
                "reference-only",
                "7" * 64,
            ),
        ),
        camera_arm_spec_fingerprint="c" * 64,
    )
    schema = _schema(descriptor=descriptor)
    assert FrameContract.from_calibration_capture(
        BINDING, descriptor, schema, _calibration_layout(schema)
    ) == _contract()
    assert len(decode_camera_capture_descriptor(
        encode_camera_capture_descriptor(descriptor)
    ).event_settings) == 3


@pytest.mark.parametrize(
    ("exposure", "gain", "mode"),
    [
        (0.0021, 2.0, "low-noise"),
        (0.002, 2.1, "low-noise"),
        (0.002, 2.0, "alternate-mode"),
    ],
)
def test_selected_exposure_gain_and_mode_changes_are_rejected(
    exposure: float,
    gain: float,
    mode: str,
) -> None:
    descriptor = _single_event_descriptor(exposure=exposure, gain=gain, mode=mode)
    with pytest.raises(ValueError, match="frame contract mismatch"):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            _single_event_schema(descriptor),
            readout_event_index=0,
        )


@pytest.mark.parametrize("frame_fingerprint", ["9" * 64, None])
def test_selected_opaque_frame_settings_evidence_is_fail_closed(
    frame_fingerprint: str | None,
) -> None:
    descriptor = _single_event_descriptor(frame_fingerprint=frame_fingerprint)
    with pytest.raises(ValueError, match="opaque_frame_settings_fingerprint"):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            _single_event_schema(descriptor),
            readout_event_index=0,
        )


@pytest.mark.parametrize(
    "descriptor_change",
    [
        {"camera_identity": "qcm-camera:serial-99"},
        {"sensor_identity": "qcm-sensor:serial-99"},
        {"optical_path": "alternate-imaging-v1"},
        {"sensor_shape_yx": (101, 120)},
        {"roi_origin_yx": (11, 20)},
        {"roi_shape_yx": (42, 60)},
        {"binning_yx": (1, 3)},
        {"coordinate_frame": CoordinateFrameId("other-pixels")},
        {"dtype": np.dtype("uint32")},
        {"count_unit": "electron"},
    ],
)
def test_frame_contract_rejects_changed_physical_fact(
    descriptor_change: dict[str, object],
) -> None:
    descriptor = _single_event_descriptor(**descriptor_change)
    with pytest.raises(ValueError):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            _single_event_schema(descriptor),
            readout_event_index=0,
        )


def test_binding_validity_and_selected_event_metadata_mismatch_fail_closed() -> None:
    descriptor = _single_event_descriptor()
    schema = _single_event_schema(descriptor)
    with pytest.raises(ValueError, match="binding"):
        _contract().assert_compatible(
            ReadoutBindingKey("secondary-readout"),
            descriptor,
            schema,
            readout_event_index=0,
        )
    component_schema = _single_event_schema(
        descriptor,
        validity=ValidityContract.components(Y),
    )
    with pytest.raises(ValueError, match="frame_schema"):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            component_schema,
            readout_event_index=0,
        )
    wrong_event = _single_event_descriptor(mode="event-metadata-changed")
    with pytest.raises(ValueError, match="readout_mode"):
        _contract().assert_compatible(
            BINDING,
            wrong_event,
            _single_event_schema(wrong_event),
            readout_event_index=0,
        )


@pytest.mark.parametrize("dtype", [np.dtype("uint8"), np.dtype("int32"), np.dtype("float32")])
def test_real_integer_and_float_count_dtypes_are_supported(dtype: np.dtype) -> None:
    descriptor = _descriptor(dtype=dtype)
    FrameContract.from_calibration_capture(
        BINDING,
        descriptor,
        _schema(descriptor=descriptor),
        _calibration_layout(_schema(descriptor=descriptor)),
    )


@pytest.mark.parametrize("dtype", [np.dtype(bool), np.dtype("complex64")])
def test_bool_and_complex_count_dtypes_are_forbidden(dtype: np.dtype) -> None:
    with pytest.raises(TypeError, match="real integer or floating"):
        _descriptor(dtype=dtype)


def _axes_with_event_at(position: int) -> tuple[AxisSpec, ...]:
    axes = [
        AxisSpec(SCAN, "detuning", SCAN_POINT, 2),
        AxisSpec(PHASE, "phase", SCAN_POINT, 2),
    ]
    axes.insert(position, AxisSpec(EVENT, "readout event", READOUT_EVENT, 3))
    return tuple(axes)


def _assert_brackets_preserve_context(
    schema: DatasetSchema,
    capture_layout: CalibrationCaptureLayout,
) -> None:
    event_position = tuple(axis.axis_id for axis in schema.point_axes).index(EVENT)
    context_positions = tuple(
        position for position in range(len(schema.point_axes)) if position != event_position
    )
    expected_axis_ids = (
        schema.repeat_axis.axis_id,
        *(schema.point_axes[position].axis_id for position in context_positions),
    )
    brackets = capture_layout.brackets(schema)
    assert len(brackets) == schema.repeat_axis.size * 4
    assert len({bracket.context_key for bracket in brackets}) == len(brackets)
    for bracket in brackets:
        assert tuple(axis_id for axis_id, _ in bracket.context_key) == expected_axis_ids
        repeat_index = bracket.context_key[0][1]
        assert 0 <= repeat_index < schema.repeat_axis.size
        context = tuple(index for _, index in bracket.context_key[1:])
        for event_index, storage_row in bracket.reference_point_storage_rows:
            logical = schema.point_layout.multi_index(storage_row)
            assert logical[event_position] == event_index
            assert tuple(logical[position] for position in context_positions) == context
        logical = schema.point_layout.multi_index(bracket.readout_point_storage_row)
        assert logical[event_position] == capture_layout.readout_event_index
        assert tuple(logical[position] for position in context_positions) == context


@pytest.mark.parametrize("event_position", [0, 1, 2])
@pytest.mark.parametrize("order", ["C", "F"])
def test_context_join_handles_event_axis_anywhere_in_c_and_f_layouts(
    event_position: int,
    order: str,
) -> None:
    axes = _axes_with_event_at(event_position)
    shape = tuple(axis.size for axis in axes)
    point_layout = PointLayout.rect_c(shape) if order == "C" else PointLayout.rect_f(shape)
    schema = _schema(point_axes=axes, point_layout=point_layout)
    capture_layout = CalibrationCaptureLayout.from_schema(
        schema,
        readout_event_axis_id=EVENT,
        reference_event_indices=(0, 2),
        readout_event_index=1,
    )
    _assert_brackets_preserve_context(schema, capture_layout)
    assert FrameContract.from_calibration_capture(
        BINDING,
        _descriptor(),
        schema,
        capture_layout,
    ).exposure_seconds == 0.002


@pytest.mark.parametrize("event_position", [0, 1, 2])
def test_random_explicit_permutations_join_by_context_not_filtered_row_position(
    event_position: int,
) -> None:
    axes = _axes_with_event_at(event_position)
    shape = tuple(axis.size for axis in axes)
    logical_rows = list(itertools.product(*(range(size) for size in shape)))
    for seed in range(12):
        shuffled = logical_rows.copy()
        random.Random(seed).shuffle(shuffled)
        schema = _schema(
            point_axes=axes,
            point_layout=PointLayout.explicit(shape, tuple(shuffled)),
        )
        capture_layout = CalibrationCaptureLayout.from_schema(
            schema,
            readout_event_axis_id=EVENT,
            reference_event_indices=(2, 0),
            readout_event_index=1,
        )
        _assert_brackets_preserve_context(schema, capture_layout)


@pytest.mark.parametrize(
    "mapping",
    [
        # readout event 1 is entirely absent
        ((0, 0), (0, 1), (2, 0), (2, 1)),
        # event 1 lacks context scan=1 while both references have it
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
        # reference event 2 lacks context scan=0
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
    ],
)
def test_missing_or_unbalanced_context_sets_fail_closed(
    mapping: tuple[tuple[int, int], ...],
) -> None:
    schema = _schema(point_layout=PointLayout.explicit((3, 2), mapping))
    with pytest.raises(ValueError, match="context"):
        CalibrationCaptureLayout.from_schema(
            schema,
            readout_event_axis_id=EVENT,
            reference_event_indices=(0, 2),
            readout_event_index=1,
        )


def test_raw_event_rows_are_diagnostic_only_and_no_pairing_lists_are_exposed() -> None:
    schema = _schema()
    layout = _calibration_layout(schema)
    assert layout.diagnostic_storage_rows_for_event(schema, 1) == (2, 3)
    assert not hasattr(layout, "reference_storage_rows")
    assert not hasattr(layout, "readout_storage_rows")


def test_frame_contract_is_canonical_and_immutable() -> None:
    contract = _contract()
    assert len(contract.digest) == 64
    assert contract.fingerprint == contract.digest
    assert decode_frame_contract(encode_frame_contract(contract)) == contract
    assert frame_contract_from_tree(frame_contract_to_tree(contract)) == contract
    with pytest.raises(FrozenInstanceError):
        contract.binding = ReadoutBindingKey("other")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.roi_shape_yx = (1, 1)  # type: ignore[misc]


def test_event_schedule_coverage_and_event_axis_metadata_are_strict() -> None:
    descriptor = _descriptor(event_settings=_event_settings()[:2])
    with pytest.raises(ValueError, match="cover every"):
        descriptor.validate_schema(_schema(descriptor=descriptor))
    wrong_role = (
        AxisSpec(EVENT, "readout event", SCAN_POINT, 3),
        AxisSpec(SCAN, "detuning", SCAN_POINT, 2),
    )
    with pytest.raises(ValueError, match="wrong role"):
        _descriptor().validate_schema(_schema(point_axes=wrong_role))


def test_event_setting_permutation_has_one_canonical_capture_descriptor() -> None:
    forward = _descriptor()
    permuted = _descriptor(
        event_settings=(_event_settings()[2], _event_settings()[0], _event_settings()[1])
    )
    assert permuted == forward
    assert encode_camera_capture_descriptor(permuted) == encode_camera_capture_descriptor(
        forward
    )


@pytest.mark.parametrize(
    ("value", "encoder", "decoder", "projector", "parser"),
    [
        (
            BINDING,
            encode_readout_binding_key,
            decode_readout_binding_key,
            readout_binding_key_to_tree,
            readout_binding_key_from_tree,
        ),
        (
            _event_settings()[0],
            encode_camera_event_readout_setting,
            decode_camera_event_readout_setting,
            camera_event_readout_setting_to_tree,
            camera_event_readout_setting_from_tree,
        ),
        (
            _descriptor(),
            encode_camera_capture_descriptor,
            decode_camera_capture_descriptor,
            camera_capture_descriptor_to_tree,
            camera_capture_descriptor_from_tree,
        ),
        (
            _contract(),
            encode_frame_contract,
            decode_frame_contract,
            frame_contract_to_tree,
            frame_contract_from_tree,
        ),
        (
            CalibrationCaptureLayout(EVENT, (2, 0), 1),
            encode_calibration_capture_layout,
            decode_calibration_capture_layout,
            calibration_capture_layout_to_tree,
            calibration_capture_layout_from_tree,
        ),
    ],
)
def test_every_persistent_value_has_strict_current_canonical_codec(
    value: object,
    encoder: object,
    decoder: object,
    projector: object,
    parser: object,
) -> None:
    encoded = encoder(value)  # type: ignore[operator]
    assert decoder(encoded) == value  # type: ignore[operator]
    tree = projector(value)  # type: ignore[operator]
    tree["unknown_future_field"] = "forbidden"
    with pytest.raises(ValueError):
        parser(tree)  # type: ignore[operator]
    with pytest.raises(ValueError):
        decoder(encode(tree))  # type: ignore[operator]


def test_noncanonical_sequences_are_rejected_instead_of_silently_resorted() -> None:
    descriptor_tree = camera_capture_descriptor_to_tree(_descriptor())
    descriptor_tree["event_settings"] = list(reversed(descriptor_tree["event_settings"]))
    with pytest.raises(ValueError, match="non-canonical"):
        camera_capture_descriptor_from_tree(descriptor_tree)
    layout_tree = calibration_capture_layout_to_tree(
        CalibrationCaptureLayout(EVENT, (0, 2), 1)
    )
    layout_tree["reference_event_indices"] = [2, 0]
    with pytest.raises(ValueError, match="non-canonical"):
        calibration_capture_layout_from_tree(layout_tree)


def test_roi_binning_dtype_unit_frame_and_axis_order_are_cross_validated() -> None:
    descriptor = _descriptor()
    height, width = descriptor.output_shape_yx
    invalid_schemas = (
        _schema(descriptor=descriptor, dtype=np.dtype("uint32")),
        _schema(descriptor=descriptor, unit="electron"),
        _schema(
            descriptor=descriptor,
            data_axes=(
                AxisSpec(Y, "y", SPATIAL_Y, height, coordinate_frame=FRAME),
                AxisSpec(X, "x", SPATIAL_X, width - 1, coordinate_frame=FRAME),
            ),
        ),
        _schema(
            descriptor=descriptor,
            data_axes=(
                AxisSpec(
                    Y,
                    "y",
                    SPATIAL_Y,
                    height,
                    coordinate_frame=CoordinateFrameId("wrong-frame"),
                ),
                AxisSpec(X, "x", SPATIAL_X, width, coordinate_frame=FRAME),
            ),
        ),
        _schema(
            descriptor=descriptor,
            data_axes=(
                AxisSpec(X, "x", SPATIAL_X, width, coordinate_frame=FRAME),
                AxisSpec(Y, "y", SPATIAL_Y, height, coordinate_frame=FRAME),
            ),
        ),
    )
    for schema in invalid_schemas:
        with pytest.raises(ValueError):
            descriptor.validate_schema(schema)


def test_no_event_axis_requires_exactly_one_index_zero_setting() -> None:
    with pytest.raises(ValueError, match="event index 0"):
        _descriptor(
            readout_event_axis_id=None,
            event_settings=(CameraEventReadoutSetting(1, 0.001, 1.0, "fast"),),
        )
