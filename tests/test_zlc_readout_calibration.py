"""Closed calibration value domain: invariants, application, and codecs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import tracemalloc

import numpy as np
import pytest

from zlc_data import (
    INVALID,
    VALID,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_storage import decode, encode
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.readout import (
    BackgroundMode,
    BoxReadoutModel,
    BoxReducer,
    CalibrationArtifact,
    CalibrationArtifactRef,
    CalibrationCaptureLayout,
    CalibrationCodecError,
    CalibrationParameter,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    CalibrationSourceBinding,
    CalibrationStage,
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    DefaultModelPolicy,
    FrameContract,
    PerSitePsfReadoutModel,
    ReadoutBindingKey,
    ReadoutFeatureSpec,
    ReadoutModelHeader,
    ReadoutModelKind,
    ReadoutModelQuality,
    SiteMap,
    UniformPsfReadoutModel,
    apply_readout_model,
    bind_readout_feature_spec,
    decode_calibration_artifact,
    decode_calibration_artifact_ref,
    decode_calibration_source_binding,
    decode_readout_model,
    decode_site_map,
    derive_calibration_source_binding,
    encode_calibration_artifact,
    encode_calibration_artifact_ref,
    encode_calibration_source_binding,
    encode_readout_model,
    encode_site_map,
    extract_readout_features,
    extract_readout_signals,
    readout_application_scratch_nbytes,
    calibration_retained_array_nbytes,
    calibration_resource_summary,
    validate_calibration_artifact_resources,
    validate_calibration_artifact_source_compatibility,
    validate_calibration_resource_summary,
)
from zlc_neutral_atom.readout.calibration import (
    validate_readout_feature_spec_model,
)
from zlc_neutral_atom.readout.calibration_codec import (
    calibration_artifact_encoding_upper_bound,
    calibration_artifact_encoding_working_upper_bound,
    calibration_artifact_metadata_encoding_upper_bound,
)


def _contracts(*, axis_name_padding: int = 0):
    frame = CoordinateFrameId("qcam-roi-pixels")
    padding = "m" * axis_name_padding
    y_axis = AxisSpec(
        AxisId("camera-y"),
        f"ROI-local y{padding}",
        SPATIAL_Y,
        4,
        coordinates=tuple(range(4)),
        unit="pixel",
        coordinate_frame=frame,
    )
    x_axis = AxisSpec(
        AxisId("camera-x"),
        f"ROI-local x{padding}",
        SPATIAL_X,
        5,
        coordinates=tuple(range(5)),
        unit="pixel",
        coordinate_frame=frame,
    )
    schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<f8"),
        "count",
    )
    contract = FrameContract(
        ReadoutBindingKey("primary-readout"),
        "qcam-serial-1",
        "sensor-1",
        "objective-a",
        (4, 5),
        (0, 0),
        (4, 5),
        (1, 1),
        y_axis.axis_id,
        x_axis.axis_id,
        frame,
        np.dtype("<f8"),
        "count",
        0.002,
        1.0,
        "external-trigger",
        None,
        schema,
    )
    site_axis = AxisSpec(
        AxisId("trap-site"),
        "trap site",
        SITE,
        3,
        coordinates=("s0", "s1", "s2"),
    )
    site_map = SiteMap(
        site_axis,
        np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 3.0]], dtype="<f8"),
        frame,
        ComponentValidity((site_axis.axis_id,), np.array([True, False, True])),
        "1" * 64,
    )
    quality = ReadoutModelQuality(
        site_axis.axis_id,
        ComponentValidity((site_axis.axis_id,), np.array([True, False, True])),
        np.array([10, 0, 11], dtype="<u8"),
        np.array([10, 0, 9], dtype="<u8"),
        np.array([10, 0, 10], dtype="<u8"),
        np.array([10, 0, 10], dtype="<u8"),
        np.array([10, 0, 10], dtype="<u8"),
        np.array([9, 0, 9], dtype="<u8"),
        np.array([10, 0, 9], dtype="<u8"),
        np.array([10, 0, 9], dtype="<u8"),
        np.array([0.70, 0.0, 0.70], dtype="<f8"),
        np.array([0.60, 0.0, 0.70], dtype="<f8"),
        np.array([0.95, 0.0, 1.0], dtype="<f8"),
        ComponentValidity((site_axis.axis_id,), np.array([True, False, True])),
        "held-out-balanced-fidelity",
        "2",
        True,
    )
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [4, 3, 1, 1]], dtype="<i8")
    return contract, site_map, quality, boxes


def _header(
    model_id: str,
    contract,
    site_map,
    quality,
    *,
    thresholds=None,
    occupied_above=None,
):
    return ReadoutModelHeader(
        model_id,
        "1",
        contract.fingerprint,
        site_map.fingerprint,
        site_map.site_axis.axis_id,
        np.array([5.0, 0.0, 5.0], dtype="<f8") if thresholds is None else thresholds,
        (
            np.array([True, False, True], dtype=bool)
            if occupied_above is None
            else occupied_above
        ),
        quality,
        (CalibrationParameter("training-split", "even-odd"),),
    )


def _value(contract, frame, validity=VALID):
    return Value(frame, validity, contract.frame_schema)


def _capture_layout():
    return CalibrationCaptureLayout(
        AxisId("readout-event"),
        (0,),
        1,
    )


def _source_binding(contract):
    binding, derived_contract = derive_calibration_source_binding(
        _resolved_capture(contract=contract),
        _capture_layout(),
    )
    assert derived_contract.fingerprint == contract.fingerprint
    return binding


def _resolved_capture(*, point_layout=None, contract=None):
    if contract is None:
        contract, _, _, _ = _contracts()
    event_axis = AxisSpec(AxisId("readout-event"), "readout event", READOUT_EVENT, 3)
    layout = PointLayout.rect_c((3,)) if point_layout is None else point_layout
    schema = DatasetSchema(
        AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2),
        (event_axis,),
        layout,
        contract.frame_schema,
    )
    descriptor = CameraCaptureDescriptor(
        camera_identity=contract.camera_identity,
        sensor_identity=contract.sensor_identity,
        optical_path=contract.optical_path,
        sensor_shape_yx=contract.sensor_shape_yx,
        roi_origin_yx=contract.roi_origin_yx,
        roi_shape_yx=contract.roi_shape_yx,
        binning_yx=contract.binning_yx,
        spatial_y_axis_id=contract.spatial_y_axis_id,
        spatial_x_axis_id=contract.spatial_x_axis_id,
        coordinate_frame=contract.coordinate_frame,
        dtype=contract.dtype,
        count_unit=contract.count_unit,
        readout_event_axis_id=event_axis.axis_id,
        event_settings=(
            CameraEventReadoutSetting(0, 0.01, 1.0, "reference"),
            CameraEventReadoutSetting(
                1,
                contract.exposure_seconds,
                contract.gain,
                contract.readout_mode,
                contract.opaque_frame_settings_fingerprint,
            ),
            CameraEventReadoutSetting(2, 0.01, 1.0, "reference"),
        ),
        camera_arm_spec_fingerprint="6" * 64,
    )
    block = DataBlock(
        BlockId("calibration-source"),
        DatasetRevision(1),
        np.zeros(schema.physical_shape, dtype="<f8"),
        VALID,
        schema,
    )
    return SimpleNamespace(
        ref=CaptureArtifactRef("captures", "2" * 64),
        block=block,
        camera_provenance=SimpleNamespace(
            descriptor=descriptor,
            binding=contract.binding,
        ),
    )


def _models(contract, site_map, quality, boxes):
    box = BoxReadoutModel(
        _header("box-v1", contract, site_map, quality),
        boxes,
        BoxReducer.SUM,
    )
    per_site = PerSitePsfReadoutModel(
        _header("per-site-psf-v1", contract, site_map, quality),
        boxes,
        np.ones((3, 1, 1), dtype="<f8"),
        BackgroundMode.NONE,
        0,
    )
    uniform = UniformPsfReadoutModel(
        _header("uniform-psf-v1", contract, site_map, quality),
        boxes,
        np.ones((1, 1), dtype="<f8"),
        BackgroundMode.NONE,
        0,
    )
    return box, per_site, uniform


def test_readout_application_scratch_bound_matches_serial_operator_allocations():
    contract, site_map, quality, boxes = _contracts()
    box_model, psf_model, _ = _models(contract, site_map, quality, boxes)
    box_spec = bind_readout_feature_spec(box_model, site_map)
    psf_spec = bind_readout_feature_spec(psf_model, site_map)
    site_bytes = 96 * site_map.site_axis.size

    assert readout_application_scratch_nbytes(
        box_spec,
        contract.frame_schema,
    ) == site_bytes + 8
    assert readout_application_scratch_nbytes(
        psf_spec,
        contract.frame_schema,
    ) == site_bytes + 24

    annulus_spec = replace(
        psf_spec,
        background=BackgroundMode.ANNULUS_MEDIAN,
        background_padding=1,
    )
    index_bytes = np.dtype(np.intp).itemsize
    # Both usable 1x1 boxes touch a corner: the clipped outer window has four
    # pixels and its annulus has three, rather than the unbounded 3x3 window.
    float64_annulus = max(
        24,
        4 + (2 * np.dtype("<f8").itemsize + 2 + 2 * index_bytes) * 3,
    )
    assert readout_application_scratch_nbytes(
        annulus_spec,
        contract.frame_schema,
    ) == site_bytes + float64_annulus

    uint16_schema = replace(contract.frame_schema, dtype=np.dtype("<u2"))
    uint16_annulus = max(
        24,
        4 + (2 * np.dtype("<u2").itemsize + 2 + 2 * index_bytes) * 3,
    )
    assert readout_application_scratch_nbytes(
        annulus_spec,
        uint16_schema,
    ) == site_bytes + uint16_annulus
    assert uint16_annulus < float64_annulus

    # Geometry validation includes unusable sites even though their windows do
    # not contribute to the serial live-set maximum.
    outside = np.array(box_spec.boxes_xywh, copy=True)
    outside[1] = np.array([np.iinfo(np.int64).max, 0, 1, 1], dtype="<i8")
    with pytest.raises(ValueError, match="outside the FrameContract"):
        readout_application_scratch_nbytes(
            replace(box_spec, boxes_xywh=outside),
            contract.frame_schema,
        )

    complex_schema = replace(contract.frame_schema, dtype=np.dtype("<c16"))
    with pytest.raises(TypeError, match="real integer or floating"):
        readout_application_scratch_nbytes(box_spec, complex_schema)


def _artifact(*, model_order=(2, 0, 1), axis_name_padding: int = 0):
    contract, site_map, quality, boxes = _contracts(
        axis_name_padding=axis_name_padding
    )
    models = _models(contract, site_map, quality, boxes)
    artifact = CalibrationArtifact(
        _source_binding(contract),
        contract,
        site_map,
        tuple(models[index] for index in model_order),
        CalibrationStage.COMPLETE,
        (
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
        ),
        DefaultModelPolicy("primary-box", "1", default_kind=ReadoutModelKind.BOX),
        "rb87-readout-calibration",
        "1",
        (
            CalibrationParameter("psf-half-width", 0),
            CalibrationParameter("strict-validity", True),
        ),
    )
    return artifact


def test_site_map_and_models_are_intrinsically_immutable_and_uniform_stores_one_kernel():
    contract, site_map, quality, boxes = _contracts()
    box, per_site, uniform = _models(contract, site_map, quality, boxes)
    for array in (
        site_map.coordinates_xy,
        site_map.validity.mask,
        quality.dark_training_sample_counts,
        quality.bright_training_sample_counts,
        quality.held_out_dark_success_counts,
        quality.held_out_dark_total_counts,
        quality.held_out_dark_labeled_counts,
        quality.held_out_bright_success_counts,
        quality.held_out_bright_total_counts,
        quality.held_out_bright_labeled_counts,
        quality.held_out_dark_accuracy_lower_bounds,
        quality.held_out_bright_accuracy_lower_bounds,
        quality.held_out_fidelity,
        box.boxes_xywh,
        per_site.kernels,
        uniform.kernel,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]
    assert uniform.kernel.shape == (1, 1)
    assert not hasattr(uniform, "kernels")

    per_site_spec = bind_readout_feature_spec(per_site, site_map)
    uniform_spec = bind_readout_feature_spec(uniform, site_map)
    assert per_site_spec.boxes_xywh is per_site.boxes_xywh
    assert per_site_spec.per_site_kernels is per_site.kernels
    assert per_site_spec.site_validity is per_site.header.quality.usable_sites
    assert uniform_spec.boxes_xywh is uniform.boxes_xywh
    assert uniform_spec.uniform_kernel is uniform.kernel


def test_feature_spec_fingerprint_is_canonical_and_rejects_another_models_math():
    contract, site_map, quality, boxes = _contracts()
    selected_model = _models(contract, site_map, quality, boxes)[0]
    bound_spec = bind_readout_feature_spec(selected_model, site_map)
    constructed_spec = ReadoutFeatureSpec(
        selected_model.kind,
        selected_model.header.site_axis_id,
        selected_model.boxes_xywh,
        selected_model.header.quality.usable_sites,
        box_reducer=selected_model.reducer,
    )

    assert len(bound_spec.fingerprint) == 64
    assert bound_spec.fingerprint == constructed_spec.fingerprint
    validate_readout_feature_spec_model(bound_spec, selected_model)
    validate_readout_feature_spec_model(constructed_spec, selected_model)

    other_model = BoxReadoutModel(
        _header("mean-box-v1", contract, site_map, quality),
        boxes,
        BoxReducer.MEAN,
    )
    replacement_spec = bind_readout_feature_spec(other_model, site_map)
    assert replacement_spec.fingerprint != bound_spec.fingerprint
    with pytest.raises(ValueError, match="does not match the selected model"):
        validate_readout_feature_spec_model(replacement_spec, selected_model)

    object.__setattr__(constructed_spec, "box_reducer", BoxReducer.MEAN)
    with pytest.raises(ValueError, match="changed after construction"):
        _ = constructed_spec.fingerprint


def test_box_application_propagates_site_and_nonfinite_validity_without_dark_coercion():
    contract, site_map, quality, boxes = _contracts()
    box = _models(contract, site_map, quality, boxes)[0]
    frame = np.zeros((4, 5), dtype="<f8")
    frame[0, 0] = 9.0
    frame[1, 2] = 100.0  # SiteMap declares this site invalid.
    frame[3, 4] = np.nan

    decision = apply_readout_model(
        box,
        frame_contract=contract,
        site_map=site_map,
        frame=_value(contract, frame),
    )

    assert decision.signals.values.tolist() == [9.0, 0.0, 0.0]
    assert decision.validity.mask.tolist() == [True, False, False]
    assert decision.occupied.tolist() == [True, False, False]
    assert not decision.occupied[2]  # False is an explicitly invalid filler, not a dark result.


def test_per_site_occupied_direction_supports_inverted_signal_without_tuple_guessing():
    contract, site_map, quality, boxes = _contracts()
    model = BoxReadoutModel(
        _header(
            "inverted-site-zero",
            contract,
            site_map,
            quality,
            occupied_above=np.array([False, False, True], dtype=bool),
        ),
        boxes,
        BoxReducer.SUM,
    )
    frame = np.zeros((4, 5), dtype="<f8")
    frame[0, 0] = 4.0
    frame[3, 4] = 6.0
    decision = apply_readout_model(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=_value(contract, frame),
    )
    assert decision.occupied.tolist() == [True, False, True]


def test_component_invalid_finite_frame_filler_is_never_used_as_a_signal():
    contract, site_map, quality, boxes = _contracts()
    model = _models(contract, site_map, quality, boxes)[0]
    frame = np.zeros((4, 5), dtype="<f8")
    frame[0, 0] = 100.0
    frame[3, 4] = 8.0
    pixel_validity = np.ones((4, 5), dtype=bool)
    pixel_validity[0, 0] = False
    value = _value(
        contract,
        frame,
        ComponentValidity(
            tuple(axis.axis_id for axis in contract.frame_schema.data_axes),
            pixel_validity,
        ),
    )
    decision = apply_readout_model(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=value,
    )
    assert decision.signals.values.tolist() == [0.0, 0.0, 8.0]
    assert decision.validity.mask.tolist() == [False, False, True]
    assert decision.occupied.tolist() == [False, False, True]
    globally_invalid = apply_readout_model(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=_value(contract, frame, INVALID),
    )
    assert globally_invalid.signals.values.tolist() == [0.0, 0.0, 0.0]
    assert globally_invalid.validity.mask.tolist() == [False, False, False]
    assert globally_invalid.occupied.tolist() == [False, False, False]


@pytest.mark.parametrize("model_index", [1, 2])
def test_psf_models_share_application_semantics_but_uniform_has_no_copied_site_axis(model_index):
    contract, site_map, quality, boxes = _contracts()
    model = _models(contract, site_map, quality, boxes)[model_index]
    frame = np.zeros((4, 5), dtype="<f8")
    frame[0, 0], frame[3, 4] = 6.0, 4.0
    result = extract_readout_signals(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=_value(contract, frame),
    )
    assert result.values.tolist() == [6.0, 0.0, 4.0]
    assert result.validity.mask.tolist() == [True, False, True]


def test_annulus_nonfinite_pixel_invalidates_only_the_affected_site():
    contract, site_map, quality, _ = _contracts()
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [4, 3, 1, 1]], dtype="<i8")
    model = UniformPsfReadoutModel(
        _header("uniform-annulus", contract, site_map, quality),
        boxes,
        np.ones((1, 1), dtype="<f8"),
        BackgroundMode.ANNULUS_MEDIAN,
        1,
    )
    frame = np.zeros((4, 5), dtype="<f8")
    frame[0, 1] = np.nan  # inside site 0 ring, outside site 2 ring
    result = extract_readout_signals(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=_value(contract, frame),
    )
    assert result.validity.mask.tolist() == [False, False, True]


@pytest.mark.parametrize("dtype", ("<u2", "<f4", "<f8"))
@pytest.mark.parametrize("model_index", (0, 1, 2))
@pytest.mark.parametrize("validity_case", ("valid", "invalid", "component"))
def test_training_and_runtime_feature_paths_are_identical_for_all_models_dtypes_and_validity(
    dtype,
    model_index,
    validity_case,
):
    base_contract, site_map, quality, boxes = _contracts()
    dtype = np.dtype(dtype)
    frame_schema = ValueSchema(
        base_contract.frame_schema.data_axes,
        base_contract.frame_schema.validity_contract,
        dtype,
        base_contract.frame_schema.value_unit,
    )
    contract = replace(base_contract, dtype=dtype, frame_schema=frame_schema)
    model = _models(contract, site_map, quality, boxes)[model_index]
    image = np.arange(1, 21, dtype=dtype).reshape(4, 5)
    if validity_case == "valid":
        validity = VALID
    elif validity_case == "invalid":
        validity = INVALID
    else:
        mask = np.ones((4, 5), dtype=bool)
        mask[0, 0] = False
        validity = ComponentValidity(
            tuple(axis.axis_id for axis in frame_schema.data_axes),
            mask,
        )
    frame = Value(image, validity, frame_schema)

    training_path = extract_readout_features(
        bind_readout_feature_spec(model, site_map),
        frame,
    )
    runtime_path = extract_readout_signals(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=frame,
    )
    assert np.array_equal(training_path.values, runtime_path.values)
    assert np.array_equal(training_path.validity.mask, runtime_path.validity.mask)


@pytest.mark.parametrize(
    "model_kind",
    (ReadoutModelKind.PER_SITE_PSF, ReadoutModelKind.UNIFORM_PSF),
)
def test_training_and_runtime_feature_paths_share_annulus_background_and_invalid_ring(
    model_kind,
):
    contract, site_map, quality, boxes = _contracts()
    header = _header(f"{model_kind.value.lower()}-annulus", contract, site_map, quality)
    if model_kind is ReadoutModelKind.PER_SITE_PSF:
        model = PerSitePsfReadoutModel(
            header,
            boxes,
            np.ones((3, 1, 1), dtype="<f8"),
            BackgroundMode.ANNULUS_MEDIAN,
            1,
        )
    else:
        model = UniformPsfReadoutModel(
            header,
            boxes,
            np.ones((1, 1), dtype="<f8"),
            BackgroundMode.ANNULUS_MEDIAN,
            1,
        )
    image = np.full((4, 5), 2.0, dtype="<f8")
    image[0, 0] = 8.0
    image[3, 4] = 10.0
    pixel_validity = np.ones((4, 5), dtype=bool)
    pixel_validity[0, 1] = False
    frame = _value(
        contract,
        image,
        ComponentValidity(
            tuple(axis.axis_id for axis in contract.frame_schema.data_axes),
            pixel_validity,
        ),
    )

    training_path = extract_readout_features(
        bind_readout_feature_spec(model, site_map),
        frame,
    )
    runtime_path = extract_readout_signals(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=frame,
    )
    assert np.array_equal(training_path.values, runtime_path.values)
    assert np.array_equal(training_path.validity.mask, runtime_path.validity.mask)
    assert runtime_path.validity.mask.tolist() == [False, False, True]


@pytest.mark.parametrize("model_index", (0, 1, 2))
def test_training_and_runtime_feature_paths_share_nonfinite_rejection(model_index):
    contract, site_map, quality, boxes = _contracts()
    model = _models(contract, site_map, quality, boxes)[model_index]
    image = np.zeros((4, 5), dtype="<f8")
    image[0, 0] = np.nan
    image[3, 4] = 7.0
    frame = _value(contract, image)

    training_path = extract_readout_features(
        bind_readout_feature_spec(model, site_map),
        frame,
    )
    runtime_path = extract_readout_signals(
        model,
        frame_contract=contract,
        site_map=site_map,
        frame=frame,
    )
    assert np.array_equal(training_path.values, runtime_path.values)
    assert np.array_equal(training_path.validity.mask, runtime_path.validity.mask)
    assert runtime_path.validity.mask.tolist() == [False, False, True]


def test_shared_feature_core_matches_an_independent_nontrivial_psf_oracle():
    axis_id = AxisId("oracle-site")
    boxes = np.array([[2, 2, 3, 3], [5, 4, 3, 3]], dtype="<i8")
    kernels = np.array(
        (
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            ((1.0, 1.0, 1.0), (1.0, 8.0, 1.0), (1.0, 1.0, 1.0)),
        ),
        dtype="<f8",
    )
    kernels /= np.sum(kernels, axis=(1, 2), keepdims=True)
    spec = ReadoutFeatureSpec(
        ReadoutModelKind.PER_SITE_PSF,
        axis_id,
        boxes,
        ComponentValidity((axis_id,), np.array([True, True])),
        per_site_kernels=kernels,
        background=BackgroundMode.ANNULUS_MEDIAN,
        background_padding=1,
    )
    frame = CoordinateFrameId("oracle-frame")
    y_axis = AxisSpec(
        AxisId("oracle-y"),
        "oracle y",
        SPATIAL_Y,
        8,
        tuple(range(8)),
        "pixel",
        frame,
    )
    x_axis = AxisSpec(
        AxisId("oracle-x"),
        "oracle x",
        SPATIAL_X,
        9,
        tuple(range(9)),
        "pixel",
        frame,
    )
    schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<f8"),
        "count",
    )
    image = 5.0 + np.arange(72, dtype="<f8").reshape(8, 9) / 10.0
    image[2:5, 2:5] += np.array(
        [[1.0, 2.0, 1.0], [2.0, 8.0, 2.0], [1.0, 2.0, 1.0]]
    )
    pixel_validity = np.ones(image.shape, dtype=bool)
    pixel_validity[7, 8] = False  # site 1 annulus only
    value = Value(
        image,
        ComponentValidity((y_axis.axis_id, x_axis.axis_id), pixel_validity),
        schema,
    )

    observed = extract_readout_features(spec, value)
    outer = image[1:6, 1:6]
    ring_mask = np.ones((5, 5), dtype=bool)
    ring_mask[1:4, 1:4] = False
    independent_background = float(np.median(outer[ring_mask]))
    independent_site_zero = float(
        np.sum(kernels[0] * (image[2:5, 2:5] - independent_background))
    )
    assert observed.values[0] == pytest.approx(independent_site_zero)
    assert observed.validity.mask.tolist() == [True, False]
    assert observed.values[1] == 0.0


def test_application_requires_named_value_schema_and_rejects_fingerprint_drift():
    contract, site_map, quality, boxes = _contracts()
    model = _models(contract, site_map, quality, boxes)[0]
    with pytest.raises(TypeError, match="zlc_data.Value"):
        extract_readout_signals(
            model,
            frame_contract=contract,
            site_map=site_map,
            frame=np.zeros((4, 5), dtype="<f8"),
        )
    wrong_schema = ValueSchema(
        contract.frame_schema.data_axes,
        contract.frame_schema.validity_contract,
        np.dtype("<f4"),
        "count",
    )
    with pytest.raises(ValueError, match="ValueSchema"):
        extract_readout_signals(
            model,
            frame_contract=contract,
            site_map=site_map,
            frame=Value(np.zeros((4, 5), dtype="<f4"), VALID, wrong_schema),
        )
    moved = replace(site_map, detection_lineage_digest="3" * 64)
    with pytest.raises(ValueError, match="does not apply"):
        bind_readout_feature_spec(model, moved)
    with pytest.raises(ValueError, match="does not apply"):
        extract_readout_signals(
            model,
            frame_contract=contract,
            site_map=moved,
            frame=_value(contract, np.zeros((4, 5), dtype="<f8")),
        )


def test_artifact_order_is_canonical_and_default_selection_is_stable():
    first = _artifact(model_order=(2, 0, 1))
    second = _artifact(model_order=(1, 2, 0))
    assert [model.header.model_id for model in first.models] == [
        "box-v1",
        "per-site-psf-v1",
        "uniform-psf-v1",
    ]
    assert first.select_model().header.model_id == "box-v1"
    assert first.select_model(kind=ReadoutModelKind.UNIFORM_PSF).header.model_id == "uniform-psf-v1"
    assert encode_calibration_artifact(first) == encode_calibration_artifact(second)
    assert first.fingerprint == second.fingerprint


def test_capture_layout_is_bound_lineage_and_changes_artifact_identity():
    artifact = _artifact()
    changed_binding = replace(
        artifact.source_binding,
        layout=CalibrationCaptureLayout(
            artifact.source_binding.layout.readout_event_axis_id,
            (2,),
            1,
        ),
        bracket_witness_digest="5" * 64,
    )
    changed = replace(
        artifact,
        source_binding=changed_binding,
    )
    assert changed.source_binding.layout != artifact.source_binding.layout
    assert changed.fingerprint != artifact.fingerprint
    assert decode_calibration_artifact(
        encode_calibration_artifact(changed)
    ).source_binding == changed.source_binding


def test_source_binding_factory_rejects_axis_index_and_sparse_layout_attacks():
    capture = _resolved_capture()
    binding, frame_contract = derive_calibration_source_binding(
        capture,
        _capture_layout(),
    )
    assert binding.source_capture_ref == capture.ref
    assert binding.source_schema_fingerprint == capture.block.schema.fingerprint
    assert binding.frame_contract_fingerprint == frame_contract.fingerprint
    assert binding.bracket_count == capture.block.schema.repeat_axis.size

    with pytest.raises(ValueError, match="different READOUT_EVENT"):
        derive_calibration_source_binding(
            capture,
            CalibrationCaptureLayout(AxisId("wrong-event-axis"), (0,), 1),
        )
    with pytest.raises(ValueError, match="outside"):
        derive_calibration_source_binding(
            capture,
            CalibrationCaptureLayout(AxisId("readout-event"), (0,), 3),
        )
    sparse = _resolved_capture(
        point_layout=PointLayout.explicit((3,), ((1,), (2,))),
    )
    with pytest.raises(ValueError, match="no physical context|identical logical context"):
        derive_calibration_source_binding(sparse, _capture_layout())


def test_artifact_rejects_unbound_frame_and_source_validation_rederives_source():
    artifact = _artifact()
    with pytest.raises(ValueError, match="fingerprints differ"):
        replace(
            artifact,
            source_binding=replace(
                artifact.source_binding,
                frame_contract_fingerprint="0" * 64,
            ),
        )

    capture = _resolved_capture()
    resolved = validate_calibration_artifact_source_compatibility(
        artifact,
        capture_resolver=lambda requested: capture
        if requested == capture.ref
        else pytest.fail("repository requested another source capture"),
    )
    assert resolved is capture

    reordered = _resolved_capture(
        point_layout=PointLayout.explicit((3,), ((2,), (1,), (0,))),
    )
    with pytest.raises(ValueError, match="source binding differs"):
        validate_calibration_artifact_source_compatibility(
            artifact,
            capture_resolver=lambda _requested: reordered,
        )


def test_multi_model_artifact_without_default_is_valid_but_implicit_selection_is_not():
    original = _artifact()
    artifact = replace(
        original,
        default_model_policy=DefaultModelPolicy("explicit-selection", "1"),
    )
    assert len(artifact.models) == 3
    with pytest.raises(ValueError, match="resolved to 3 models"):
        artifact.select_model()
    assert artifact.select_model(model_id="uniform-psf-v1").kind is ReadoutModelKind.UNIFORM_PSF


def test_default_kind_ambiguity_and_missing_complete_kind_fail_closed():
    contract, site_map, quality, boxes = _contracts()
    box = _models(contract, site_map, quality, boxes)[0]
    second_box = BoxReadoutModel(
        _header("box-v2", contract, site_map, quality), boxes, BoxReducer.MEAN
    )
    with pytest.raises(ValueError, match="resolved to 2 models"):
        CalibrationArtifact(
            _source_binding(contract),
            contract,
            site_map,
            (box, second_box),
            CalibrationStage.COMPLETE,
            (ReadoutModelKind.BOX,),
            DefaultModelPolicy("ambiguous", "1", default_kind=ReadoutModelKind.BOX),
            "algorithm",
            "1",
        )
    with pytest.raises(ValueError, match="missing required"):
        CalibrationArtifact(
            _source_binding(contract),
            contract,
            site_map,
            (box,),
            CalibrationStage.COMPLETE,
            (ReadoutModelKind.BOX, ReadoutModelKind.PER_SITE_PSF),
            DefaultModelPolicy("box", "1", default_kind=ReadoutModelKind.BOX),
            "algorithm",
            "1",
        )


def test_invalid_sites_require_explicit_validity_and_canonical_zero_fillers():
    contract, site_map, quality, boxes = _contracts()
    bad_thresholds = np.array([5.0, 6.0, 5.0], dtype="<f8")
    with pytest.raises(ValueError, match="zero threshold"):
        _header("bad", contract, site_map, quality, thresholds=bad_thresholds)
    retained_diagnostic = replace(
        quality,
        dark_training_sample_counts=np.array([10, 1, 11], dtype="<u8"),
    )
    assert retained_diagnostic.dark_training_sample_counts[1] == 1
    with pytest.raises(ValueError, match="finite"):
        replace(
            site_map,
            coordinates_xy=np.array(
                [[0.0, 0.0], [np.nan, 0.0], [4.0, 3.0]],
                dtype="<f8",
            ),
        )
    with pytest.raises(ValueError, match="zero fillers"):
        replace(
            site_map,
            coordinates_xy=np.array(
                [[0.0, 0.0], [2.0, 1.0], [4.0, 3.0]], dtype="<f8"
            ),
        )
    with pytest.raises(ValueError, match="direction fillers"):
        _header(
            "bad-direction",
            contract,
            site_map,
            quality,
            occupied_above=np.array([True, True, True], dtype=bool),
        )
    bad_boxes = boxes.copy()
    bad_boxes[1] = [2, 1, 1, 1]
    with pytest.raises(ValueError, match="box fillers"):
        BoxReadoutModel(
            _header("bad-box-filler", contract, site_map, quality),
            bad_boxes,
            BoxReducer.SUM,
        )


def test_model_quality_gate_and_site_map_subset_are_artifact_invariants():
    contract, site_map, quality, boxes = _contracts()
    failed = replace(quality, gate_passed=False)
    failed_box = BoxReadoutModel(
        _header("failed", contract, site_map, failed), boxes, BoxReducer.SUM
    )
    with pytest.raises(ValueError, match="quality gate"):
        CalibrationArtifact(
            _source_binding(contract),
            contract,
            site_map,
            (failed_box,),
            CalibrationStage.COMPLETE,
            (ReadoutModelKind.BOX,),
            DefaultModelPolicy("box", "1", default_kind=ReadoutModelKind.BOX),
            "algorithm",
            "1",
        )


def test_quality_requires_both_classes_and_held_out_evidence_for_every_usable_site():
    _, _, quality, _ = _contracts()
    with pytest.raises(ValueError, match="both dark and bright"):
        replace(
            quality,
            dark_training_sample_counts=np.array([0, 0, 11], dtype="<u8"),
        )
    with pytest.raises(ValueError, match="held-out evidence"):
        replace(
            quality,
            held_out_fidelity=np.array([0.99, 0.0, 0.0], dtype="<f8"),
            held_out_validity=ComponentValidity(
                (quality.site_axis_id,),
                np.array([True, False, False]),
            ),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(
            quality,
            held_out_dark_success_counts=np.array([11, 0, 10], dtype="<u8"),
        )
    with pytest.raises(ValueError, match="differs from per-class evidence"):
        replace(
            quality,
            held_out_fidelity=np.array([0.90, 0.0, 1.0], dtype="<f8"),
        )
    with pytest.raises(ValueError, match="lower bound exceeds"):
        replace(
            quality,
            held_out_bright_accuracy_lower_bounds=np.array(
                [0.95, 0.0, 0.70],
                dtype="<f8",
            ),
        )


def test_checked_box_geometry_blocks_int64_wrap_empty_slice_and_center_mismatch():
    contract, site_map, quality, boxes = _contracts()
    overflow = boxes.copy()
    overflow[0] = [np.iinfo(np.int64).max, 0, 1, 1]
    overflow_model = BoxReadoutModel(
        _header("overflow", contract, site_map, quality),
        overflow,
        BoxReducer.SUM,
    )
    with pytest.raises(ValueError, match="outside"):
        extract_readout_signals(
            overflow_model,
            frame_contract=contract,
            site_map=site_map,
            frame=_value(contract, np.zeros((4, 5), dtype="<f8")),
        )

    misses_center = boxes.copy()
    misses_center[0] = [1, 0, 1, 1]
    center_model = BoxReadoutModel(
        _header("miss-center", contract, site_map, quality),
        misses_center,
        BoxReducer.SUM,
    )
    with pytest.raises(ValueError, match="does not lie inside"):
        extract_readout_signals(
            center_model,
            frame_contract=contract,
            site_map=site_map,
            frame=_value(contract, np.zeros((4, 5), dtype="<f8")),
        )


def test_standalone_model_cannot_bypass_site_count_applicability():
    contract, site_map, _, _ = _contracts()
    axis_id = site_map.site_axis.axis_id
    quality = ReadoutModelQuality(
        axis_id,
        ComponentValidity((axis_id,), np.array([True, True])),
        np.array([5, 5], dtype="<u8"),
        np.array([5, 5], dtype="<u8"),
        np.array([9, 9], dtype="<u8"),
        np.array([10, 10], dtype="<u8"),
        np.array([10, 10], dtype="<u8"),
        np.array([9, 9], dtype="<u8"),
        np.array([10, 10], dtype="<u8"),
        np.array([10, 10], dtype="<u8"),
        np.array([0.6, 0.6], dtype="<f8"),
        np.array([0.6, 0.6], dtype="<f8"),
        np.array([0.9, 0.9], dtype="<f8"),
        ComponentValidity((axis_id,), np.array([True, True])),
        "gate",
        "1",
        True,
    )
    header = ReadoutModelHeader(
        "wrong-count",
        "1",
        contract.fingerprint,
        site_map.fingerprint,
        axis_id,
        np.array([1.0, 1.0], dtype="<f8"),
        np.array([True, True]),
        quality,
    )
    model = BoxReadoutModel(
        header,
        np.array([[0, 0, 1, 1], [4, 3, 1, 1]], dtype="<i8"),
        BoxReducer.SUM,
    )
    with pytest.raises(ValueError, match="different site counts"):
        extract_readout_signals(
            model,
            frame_contract=contract,
            site_map=site_map,
            frame=_value(contract, np.zeros((4, 5), dtype="<f8")),
        )


def test_psf_kernels_are_nonnegative_and_unusable_sites_have_one_placeholder():
    contract, site_map, quality, _ = _contracts()
    boxes = np.array(
        [[0, 0, 2, 1], [0, 0, 2, 1], [3, 3, 2, 1]],
        dtype="<i8",
    )
    negative = np.array(
        [[[0.5, 0.5]], [[1.0, 0.0]], [[-0.1, 1.1]]],
        dtype="<f8",
    )
    with pytest.raises(ValueError, match="non-negative"):
        PerSitePsfReadoutModel(
            _header("negative", contract, site_map, quality),
            boxes,
            negative,
            BackgroundMode.NONE,
            0,
        )
    bad_filler = np.array(
        [[[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]],
        dtype="<f8",
    )
    with pytest.raises(ValueError, match="unit-impulse"):
        PerSitePsfReadoutModel(
            _header("bad-filler", contract, site_map, quality),
            boxes,
            bad_filler,
            BackgroundMode.NONE,
            0,
        )


def test_negative_zero_and_invalid_coordinate_payloads_have_one_identity():
    _, site_map, _, _ = _contracts()
    coordinates = site_map.coordinates_xy.copy()
    coordinates[1] = [-0.0, -0.0]
    normalized = replace(site_map, coordinates_xy=coordinates)
    assert normalized.fingerprint == site_map.fingerprint
    assert not np.any(np.signbit(normalized.coordinates_xy[1]))
    duplicate = site_map.coordinates_xy.copy()
    duplicate[2] = duplicate[0]
    with pytest.raises(ValueError, match="unique XY"):
        replace(
            site_map,
            coordinates_xy=duplicate,
            validity=ComponentValidity(
                (site_map.site_axis.axis_id,),
                np.array([True, False, True]),
            ),
        )


def test_resource_policy_rejects_before_apply_or_blob_decode():
    artifact = _artifact()
    frame = _value(
        artifact.frame_contract,
        np.zeros(artifact.frame_contract.frame_schema.data_shape, dtype="<f8"),
    )
    tiny_work = CalibrationResourcePolicy(max_sampled_pixels_per_model=1)
    with pytest.raises(CalibrationResourceExceeded, match="sampled pixels"):
        extract_readout_signals(
            artifact.select_model(),
            frame_contract=artifact.frame_contract,
            site_map=artifact.site_map,
            frame=frame,
            resource_policy=tiny_work,
        )
    with pytest.raises(CalibrationResourceExceeded, match="model count"):
        validate_calibration_artifact_resources(
            artifact,
            CalibrationResourcePolicy(max_models=2),
        )
    summary = calibration_resource_summary(artifact)
    with pytest.raises(CalibrationResourceExceeded, match="model count"):
        validate_calibration_resource_summary(
            summary,
            CalibrationResourcePolicy(max_models=2),
        )
    payload = encode_calibration_artifact(artifact)
    with pytest.raises(CalibrationResourceExceeded, match="payload"):
        decode_calibration_artifact(
            payload,
            resource_policy=CalibrationResourcePolicy(max_artifact_blob_bytes=1),
        )
    with pytest.raises(TypeError, match="bytes-like"):
        decode_calibration_artifact(10_000_000)  # type: ignore[arg-type]


def _artifact_encoding_bounds(artifact):
    summary = calibration_resource_summary(artifact)
    metadata_bound = calibration_artifact_metadata_encoding_upper_bound(
        source_binding=artifact.source_binding,
        frame_contract=artifact.frame_contract,
        artifact_parameters=artifact.parameters,
        model_parameters=tuple(
            model.header.parameters for model in artifact.models
        ),
        model_kinds=tuple(model.kind for model in artifact.models),
        default_model_policy=artifact.default_model_policy,
        algorithm_id=artifact.algorithm_id,
        algorithm_version=artifact.algorithm_version,
    )
    wire_bound = calibration_artifact_encoding_upper_bound(
        site_count=summary.site_count,
        model_count=summary.model_count,
        kernel_elements=summary.kernel_elements,
        metadata_encoding_upper_bound_bytes=metadata_bound,
    )
    return metadata_bound, wire_bound


def test_artifact_wire_and_profiled_canonical_working_memory_fit_owner_bounds():
    artifact = _artifact()
    payload = encode_calibration_artifact(artifact)
    metadata_bound, wire_bound = _artifact_encoding_bounds(artifact)
    assert len(payload) <= wire_bound
    assert wire_bound <= 4 * len(payload)

    kernel = np.zeros((1024, 1024), dtype="<f8")
    tracemalloc.start()
    try:
        encoded = encode({"kernel": kernel})
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(encoded) > kernel.nbytes
    assert peak <= calibration_artifact_encoding_working_upper_bound(
        kernel.nbytes,
        0,
    )


def test_artifact_bounds_include_adversarial_long_frame_schema_metadata():
    artifact = _artifact(axis_name_padding=30_000)
    metadata_bound, wire_bound = _artifact_encoding_bounds(artifact)
    retained = calibration_retained_array_nbytes(artifact)

    tracemalloc.start()
    try:
        payload = encode_calibration_artifact(artifact)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(payload) <= wire_bound
    assert peak <= calibration_artifact_encoding_working_upper_bound(
        retained,
        metadata_bound,
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "held_out_dark_success_counts",
        "held_out_dark_total_counts",
        "held_out_dark_labeled_counts",
        "held_out_bright_success_counts",
        "held_out_bright_total_counts",
        "held_out_bright_labeled_counts",
        "held_out_dark_accuracy_lower_bounds",
        "held_out_bright_accuracy_lower_bounds",
    ),
)
def test_direct_decode_site_budget_rejects_before_ndarray_materialization(
    monkeypatch,
    field_name,
):
    tree = decode(encode_calibration_artifact(_artifact()))
    quality = tree["models"][0]["header"]["quality"]
    source = quality[field_name]
    quality[field_name] = np.zeros(4, dtype=source.dtype)
    payload = encode(tree)
    import zlc_storage.canonical as canonical

    materializations = 0
    original = canonical._decode_array

    def counted(payload, *, path):
        nonlocal materializations
        materializations += 1
        return original(payload, path=path)

    monkeypatch.setattr(canonical, "_decode_array", counted)
    with pytest.raises(CalibrationResourceExceeded, match="site count"):
        decode_calibration_artifact(
            payload,
            resource_policy=CalibrationResourcePolicy(max_sites=3),
        )
    assert materializations == 0


def test_artifact_is_unhashable_and_fingerprint_is_cached(monkeypatch):
    artifact = _artifact()
    fingerprint = artifact.fingerprint
    with pytest.raises(TypeError):
        hash(artifact)
    import zlc_neutral_atom.readout.calibration_codec as codec

    monkeypatch.setattr(
        codec,
        "calibration_artifact_to_tree",
        lambda _artifact: pytest.fail("fingerprint was recomputed"),
    )
    assert artifact.fingerprint == fingerprint


def test_all_persistent_values_have_strict_current_canonical_codecs():
    artifact = _artifact()
    site_map = artifact.site_map
    model = artifact.models[0]
    round_trips = (
        (
            artifact.source_binding,
            encode_calibration_source_binding,
            decode_calibration_source_binding,
        ),
        (site_map, encode_site_map, decode_site_map),
        (model, encode_readout_model, decode_readout_model),
        (artifact, encode_calibration_artifact, decode_calibration_artifact),
    )
    for original, encoder, decoder in round_trips:
        payload = encoder(original)
        decoded = decoder(payload)
        assert encoder(decoded) == payload

    tree = decode(encode_calibration_artifact(artifact))
    tree["models"] = list(reversed(tree["models"]))
    with pytest.raises(CalibrationCodecError, match="non-canonical"):
        decode_calibration_artifact(encode(tree))
    tree = decode(encode_calibration_artifact(artifact))
    tree["unknown"] = 1
    with pytest.raises(ValueError, match="exactly"):
        decode_calibration_artifact(encode(tree))


def test_quality_codec_is_exact_and_carries_adverse_per_class_evidence():
    tree = decode(encode_calibration_artifact(_artifact()))
    quality = tree["models"][0]["header"]["quality"]
    assert quality["schema"] == "zlc_neutral_atom.readout-model-quality"
    assert set(quality) == {
        "schema",
        "site_axis_id",
        "usable_sites",
        "dark_training_sample_counts",
        "bright_training_sample_counts",
        "held_out_dark_success_counts",
        "held_out_dark_total_counts",
        "held_out_dark_labeled_counts",
        "held_out_bright_success_counts",
        "held_out_bright_total_counts",
        "held_out_bright_labeled_counts",
        "held_out_dark_accuracy_lower_bounds",
        "held_out_bright_accuracy_lower_bounds",
        "held_out_fidelity",
        "held_out_validity",
        "quality_gate_id",
        "quality_gate_version",
        "gate_passed",
    }

    missing = decode(encode_calibration_artifact(_artifact()))
    del missing["models"][0]["header"]["quality"]["held_out_dark_total_counts"]
    with pytest.raises(ValueError, match="exactly"):
        decode_calibration_artifact(encode(missing))
    unknown = decode(encode_calibration_artifact(_artifact()))
    unknown["models"][0]["header"]["quality"]["legacy_fidelity_gate"] = 0.9
    with pytest.raises(ValueError, match="exactly"):
        decode_calibration_artifact(encode(unknown))


def test_calibration_reference_codec_is_stable_and_namespace_typed():
    reference = CalibrationArtifactRef("lab-a", "a" * 64)
    assert decode_calibration_artifact_ref(
        encode_calibration_artifact_ref(reference)
    ) == reference
    assert CalibrationArtifactRef("lab-b", reference.manifest_digest) != reference


def test_out_of_frame_geometry_is_rejected_at_application_not_silently_clipped():
    contract, site_map, quality, boxes = _contracts()
    boxes = boxes.copy()
    boxes[2] = [5, 3, 1, 1]
    model = BoxReadoutModel(
        _header("bad-geometry", contract, site_map, quality),
        boxes,
        BoxReducer.SUM,
    )
    with pytest.raises(ValueError, match="outside"):
        extract_readout_signals(
            model,
            frame_contract=contract,
            site_map=site_map,
            frame=_value(contract, np.zeros((4, 5), dtype="<f8")),
        )
