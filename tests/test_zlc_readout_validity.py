"""Intentional fail-closed corrections to known ``main`` readout defects."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_data import (
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.readout.calibration.analysis import (
    CalibrationAnalysisResult,
    CalibrationAnalysisRequest,
    CalibrationComputation,
    _calibrate_readout_frames,
    reference_labels,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    BackgroundMode,
    BoxFeature,
    BoxReducer,
    CalibrationArtifact,
    CalibrationSourceBinding,
    GridOrder,
    PerSitePsfFeature,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    _annulus_background,
    apply_calibration,
    classify_occupancy,
    extract_readout_features,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import (
    CalibrationCaptureLayout,
    FrameContract,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.readout.physical_context import ReadoutPhysicalContext


def _physical_context(contract: FrameContract) -> ReadoutPhysicalContext:
    return ReadoutPhysicalContext(
        50e6,
        "a" * 64,
        0.0,
        contract.exposure_seconds,
        (),
        (),
    )


def _contracts(
    image_shape_yx: tuple[int, int],
) -> tuple[CalibrationSourceBinding, FrameContract]:
    height, width = image_shape_yx
    coordinate_frame = CoordinateFrameId("validity-camera-pixels")
    y_axis = AxisSpec(
        AxisId("validity-y"),
        "validity y",
        SPATIAL_Y,
        height,
        tuple(range(height)),
        "pixel",
        coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId("validity-x"),
        "validity x",
        SPATIAL_X,
        width,
        tuple(range(width)),
        "pixel",
        coordinate_frame,
    )
    schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<f8"),
        "count",
    )
    layout = CalibrationCaptureLayout(AxisId("validity-event"), (0, 2), 1)
    source = CalibrationSourceBinding(
        CaptureArtifactRef("validity-tests", "0" * 64),
        layout,
    )
    contract = FrameContract(
        ReadoutBindingKey("validity-readout"),
        "validity-camera",
        "validity-sensor",
        "validity-optics",
        image_shape_yx,
        (0, 0),
        image_shape_yx,
        (1, 1),
        y_axis.axis_id,
        x_axis.axis_id,
        coordinate_frame,
        np.dtype("<f8"),
        "count",
        0.005,
        1.0,
        "external-trigger",
        None,
        schema,
    )
    return source, contract


def _site_domain() -> tuple[AxisSpec, ComponentValidity]:
    axis = AxisSpec(AxisId("validity-site"), "validity site", SITE, 1)
    return axis, ComponentValidity((axis.axis_id,), np.array([True]))


def _frame(
    contract: FrameContract,
    values: np.ndarray,
    pixel_validity: np.ndarray,
) -> Value:
    axes = contract.frame_schema.data_axes
    return Value(
        np.asarray(values, dtype="<f8"),
        ComponentValidity(
            tuple(axis.axis_id for axis in axes),
            np.asarray(pixel_validity, dtype=bool),
        ),
        contract.frame_schema,
    )


def _constant_short_calibration(
    *,
    psf_half_width: int = 1,
) -> CalibrationComputation:
    source, contract = _contracts((9, 9))
    groups = 24
    occupied = np.arange(groups) % 2 == 0
    references = np.zeros((groups, 2, 9, 9), dtype="<f8")
    for group, has_atom in enumerate(occupied):
        references[group, :, 4, 4] = 100.0 if has_atom else 10.0
    short = np.full((groups, 9, 9), 20.0, dtype="<f8")
    request = CalibrationAnalysisRequest(
        source.layout,
        (1, 1),
        box_radius=0,
        box_reducer=BoxReducer.MEAN,
        psf_half_width=psf_half_width,
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        train_fraction=0.5,
        histogram_bins=20,
        max_drop=0,
    )
    return _calibrate_readout_frames(
        references,
        short,
        source_binding=source,
        frame_contract=contract,
        readout_physical_context=_physical_context(contract),
        request=request,
    )


def test_nonfinite_required_reference_is_invalid_not_dark() -> None:
    signals = np.empty((8, 2, 1), dtype="<f8")
    signals[:4, :, 0] = 1.0
    signals[4:, :, 0] = 10.0
    signals[0, 1, 0] = np.nan

    labels = reference_labels(signals)

    assert labels.fits[0].ok and labels.fits[0].bright_above
    assert not labels.valid[0, 0]
    assert not labels.dark[0, 0]
    assert not labels.occupied[0, 0]
    assert labels.valid[1, 0] and labels.dark[1, 0]
    assert labels.valid[4, 0] and labels.occupied[4, 0]


def test_box_empty_is_invalid_filler_but_partial_finite_pixels_reduce() -> None:
    _source, contract = _contracts((2, 2))
    site_axis, sites_valid = _site_domain()
    feature = BoxFeature(
        site_axis,
        np.array([[0, 0, 2, 2]]),
        BoxReducer.MEAN,
        sites_valid,
    )
    model = ReadoutModel(feature, np.array([2.0]), sites_valid)

    empty_signals = extract_readout_features(
        model.feature,
        _frame(contract, np.full((2, 2), 99.0), np.zeros((2, 2), dtype=bool)),
    )
    empty_occupied = classify_occupancy(model, empty_signals)
    np.testing.assert_array_equal(empty_signals.values, [0.0])
    np.testing.assert_array_equal(empty_occupied.values, [False])
    np.testing.assert_array_equal(empty_signals.validity.mask, [False])
    np.testing.assert_array_equal(empty_occupied.validity.mask, [False])

    partial_signals = extract_readout_features(
        model.feature,
        _frame(
            contract,
            np.array([[2.0, np.nan], [4.0, 999.0]]),
            np.array([[True, True], [True, False]]),
        ),
    )
    partial_occupied = classify_occupancy(model, partial_signals)
    np.testing.assert_allclose(partial_signals.values, [3.0])
    np.testing.assert_array_equal(partial_signals.validity.mask, [True])
    np.testing.assert_array_equal(partial_occupied.values, [True])


def test_annulus_background_edge_golden_and_full_frame_fallback() -> None:
    image = np.arange(1, 26, dtype=np.uint16).reshape(5, 5)
    valid = np.ones(image.shape, dtype=bool)

    # Main's clipped corner annulus is the 3x3 region minus the 2x2 site box:
    # [3, 8, 11, 12, 13], whose median is 11.
    assert _annulus_background(image, valid, (0, 0, 2, 2), 1) == 11.0

    corrected = image.astype(np.float64)
    corrected[0, 2] = np.nan
    valid[1, 2] = False
    # The shared training/runtime owner keeps the current validity and
    # non-finite correction; [11, 12, 13] remains and has median 12.
    assert _annulus_background(corrected, valid, (0, 0, 2, 2), 1) == 12.0

    fallback = np.array([[1.0, np.nan], [100.0, 5.0]])
    fallback_valid = np.array([[True, True], [False, True]])
    # A full-image box has no ring, so the established main behavior falls back
    # to the full valid finite frame.  No valid finite pixel yields zero.
    assert _annulus_background(fallback, fallback_valid, (0, 0, 2, 2), 3) == 3.0
    assert _annulus_background(
        fallback,
        np.zeros_like(fallback_valid),
        (0, 0, 2, 2),
        3,
    ) == 0.0


def test_psf_missing_required_pixel_invalidates_without_renormalizing() -> None:
    _source, contract = _contracts((2, 2))
    site_axis, sites_valid = _site_domain()
    feature = PerSitePsfFeature(
        site_axis,
        np.array([[0, 0, 2, 2]]),
        np.array([[[0.1, 0.2], [0.3, 0.4]]]),
        BackgroundMode.NONE,
        1,
        sites_valid,
    )
    model = ReadoutModel(feature, np.array([1.0]), sites_valid)
    frame = _frame(
        contract,
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        np.array([[True, True], [True, False]]),
    )

    signals = extract_readout_features(model.feature, frame)
    occupied = classify_occupancy(model, signals)

    np.testing.assert_array_equal(signals.values, [0.0])
    np.testing.assert_array_equal(occupied.values, [False])
    np.testing.assert_array_equal(signals.validity.mask, [False])
    np.testing.assert_array_equal(occupied.validity.mask, [False])


def test_reversed_short_polarity_is_diagnostic_only_and_bad_component_survives() -> None:
    import zlc_neutral_atom.logic_nodes.readout.calibration.analysis as analysis

    source, contract = _contracts((9, 9))
    groups = 24
    occupied = np.arange(groups) % 2 == 0
    references = np.zeros((groups, 2, 9, 9), dtype="<f8")
    short = np.zeros((groups, 9, 9), dtype="<f8")
    for group, has_atom in enumerate(occupied):
        references[group, :, 4, 4] = 100.0 if has_atom else 10.0
        # Deliberately reversed detector response: atoms produce fewer counts.
        short[group, 4, 4] = 5.0 if has_atom else 50.0
    short_validity = np.ones(short.shape, dtype=bool)
    short_validity[0, 4, 4] = False
    request = CalibrationAnalysisRequest(
        source.layout,
        (1, 1),
        box_radius=0,
        box_reducer=BoxReducer.MEAN,
        psf_half_width=1,
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        train_fraction=0.5,
        histogram_bins=20,
        max_drop=0,
    )

    result = _calibrate_readout_frames(
        references,
        short,
        source_binding=source,
        frame_contract=contract,
        readout_physical_context=_physical_context(contract),
        request=request,
        short_validity=short_validity,
    )
    report = result.report.model(ReadoutModelKind.BOX)
    model = result.artifact.select_model(ReadoutModelKind.BOX)

    assert not report.site_fidelity[0].bright_above
    assert report.predictions[2, 0]  # useful diagnostic of the reversed response
    np.testing.assert_array_equal(np.flatnonzero(~report.short_validity[:, 0]), [0])
    np.testing.assert_array_equal(model.usable_sites.mask, [False])

    signals = extract_readout_features(
        model.feature,
        _frame(contract, short[2], np.ones((9, 9), dtype=bool)),
    )
    runtime_occupancy = classify_occupancy(model, signals)
    np.testing.assert_allclose(signals.values, [5.0])
    np.testing.assert_array_equal(signals.validity.mask, [True])
    np.testing.assert_array_equal(runtime_occupancy.values, [False])
    np.testing.assert_array_equal(runtime_occupancy.validity.mask, [False])

    # If another requested method has no held-out model evidence, main's
    # all-method gate keeps every runtime threshold on its quick value.  That
    # threshold choice must not erase this method's explicit reversed-polarity
    # evidence and accidentally make the site usable again.
    missing_metric = replace(
        report.site_fidelity[0],
        threshold=float("nan"),
        model_fidelity=float("nan"),
    )
    missing_method = replace(
        report,
        kind=ReadoutModelKind.PER_SITE_PSF,
        site_fidelity=(missing_metric,),
    )
    use_reference_thresholds = analysis._main_reference_thresholds_available(
        (report, missing_method)
    )
    assert not use_reference_thresholds
    selected_thresholds, selected_usable = analysis._runtime_model_values(
        model.feature,
        report,
        request,
        use_reference_thresholds=use_reference_thresholds,
    )
    np.testing.assert_array_equal(selected_thresholds, report.quick_thresholds)
    np.testing.assert_array_equal(selected_usable, [False])


def test_artifact_rejects_geometry_outside_its_frame_contract() -> None:
    source, contract = _contracts((3, 3))
    site_axis, sites_valid = _site_domain()

    def artifact(coordinate_xy, box_xywh, coordinate_frame):
        site_map = SiteMap(
            site_axis,
            np.asarray([coordinate_xy], dtype="<f8"),
            (1, 1),
            GridOrder.ROW_MAJOR,
            coordinate_frame,
            sites_valid,
        )
        feature = BoxFeature(
            site_axis,
            np.asarray([box_xywh], dtype="<i8"),
            BoxReducer.MEAN,
            sites_valid,
        )
        return CalibrationArtifact(
            source,
            contract,
            _physical_context(contract),
            site_map,
            (ReadoutModel(feature, np.array([1.0]), sites_valid),),
            ReadoutModelKind.BOX,
        )

    with pytest.raises(ValueError, match="coordinates fall outside"):
        artifact((3.0, 1.0), (0, 0, 1, 1), contract.coordinate_frame)
    with pytest.raises(ValueError, match="boxes fall outside"):
        artifact((1.0, 1.0), (2, 2, 2, 2), contract.coordinate_frame)
    with pytest.raises(ValueError, match="coordinate frames differ"):
        artifact(
            (1.0, 1.0),
            (0, 0, 1, 1),
            CoordinateFrameId("some-other-camera-pixels"),
        )


def test_chance_level_short_signal_is_not_runtime_usable() -> None:
    result = _constant_short_calibration()
    report = result.report.model(ReadoutModelKind.BOX)
    model = result.artifact.select_model(ReadoutModelKind.BOX)

    assert report.site_fidelity[0].fidelity == pytest.approx(0.5)
    np.testing.assert_array_equal(model.usable_sites.mask, [False])


def test_public_apply_is_non_authoritative_but_rejects_structural_schema_drift() -> None:
    result = _constant_short_calibration()
    contract = result.artifact.frame_contract
    frame = _frame(
        contract,
        np.zeros(contract.frame_schema.data_shape, dtype="<f8"),
        np.ones(contract.frame_schema.data_shape, dtype=bool),
    )

    apply_calibration(result.artifact, frame)
    drifted = replace(
        frame,
        schema=replace(frame.schema, value_unit="different-count-unit"),
    )
    with pytest.raises(ValueError, match="schema differs"):
        apply_calibration(
            result.artifact,
            drifted,
        )


def test_result_rejects_artifact_thresholds_not_supported_by_report() -> None:
    result = _constant_short_calibration()
    original_model = result.artifact.select_model(ReadoutModelKind.BOX)
    mismatched_model = replace(
        original_model,
        thresholds=original_model.thresholds + 1000.0,
    )
    mismatched_artifact = replace(
        result.artifact,
        models=(mismatched_model,),
    )

    with pytest.raises(TypeError, match="returned by a committed calibration Run"):
        CalibrationAnalysisResult(mismatched_artifact, result.report)
    with pytest.raises(ValueError, match="thresholds differ"):
        CalibrationComputation(mismatched_artifact, result.report)


def test_invalid_psf_site_placeholder_does_not_pollute_uniform_kernel() -> None:
    source, contract = _contracts((15, 15))
    groups = 24
    references = np.zeros((groups, 2, 15, 15), dtype="<f8")
    short = np.zeros((groups, 15, 15), dtype="<f8")
    spot = np.array(
        [[1.0, 2.0, 1.0], [2.0, 9.0, 2.0], [1.0, 2.0, 1.0]],
        dtype="<f8",
    )
    for group in range(groups):
        occupied = group % 2 == 0
        for center_x in (4, 10):
            references[group, :, 6:9, center_x - 1 : center_x + 2] = (
                spot * (10.0 if occupied else 1.0)
            )
            short[group, 7, center_x] = 20.0 if occupied else 2.0
    reference_validity = np.ones(references.shape, dtype=bool)
    # One missing corner invalidates only the second site's PSF geometry while
    # preserving its bright detector pixel and the first site's fitted kernel.
    reference_validity[:, :, 6, 9] = False
    request = CalibrationAnalysisRequest(
        source.layout,
        (1, 2),
        box_radius=0,
        psf_half_width=1,
        psf_background=BackgroundMode.NONE,
        model_kinds=(
            ReadoutModelKind.PER_SITE_PSF,
            ReadoutModelKind.UNIFORM_PSF,
        ),
        default_model_kind=ReadoutModelKind.PER_SITE_PSF,
        train_fraction=0.5,
        histogram_bins=20,
        max_drop=0,
    )

    result = _calibrate_readout_frames(
        references,
        short,
        source_binding=source,
        frame_contract=contract,
        readout_physical_context=_physical_context(contract),
        request=request,
        reference_validity=reference_validity,
    )
    per_site = result.artifact.select_model(
        ReadoutModelKind.PER_SITE_PSF
    ).feature
    uniform = result.artifact.select_model(ReadoutModelKind.UNIFORM_PSF).feature

    assert isinstance(per_site, PerSitePsfFeature)
    np.testing.assert_array_equal(per_site.valid_sites.mask, [True, False])
    np.testing.assert_allclose(uniform.kernel, per_site.kernels[0])


def test_box_only_calibration_ignores_inapplicable_psf_geometry() -> None:
    result = _constant_short_calibration(psf_half_width=5)

    assert tuple(model.kind for model in result.artifact.models) == (
        ReadoutModelKind.BOX,
    )
    assert result.report.psf_fits == ()
