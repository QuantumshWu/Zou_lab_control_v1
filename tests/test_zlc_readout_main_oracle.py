"""Frozen same-frame oracle from the physical algorithms in ``main@6c337d49``.

The expected arrays are immutable evidence generated from the exact ``main``
implementation.  This test deliberately exercises only the current
Calibration capability APIs; it neither imports the reference tree nor
regenerates expected values from the implementation under test.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from zlc_data import (
    VALID,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.readout.calibration.analysis import (
    CalibrationAnalysisRequest,
    CalibrationComputation,
    _validate_site_center_admission,
    find_site_centers,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    BackgroundMode,
    BoxFeature,
    BoxReducer,
    CalibrationSourceBinding,
    GridOrder,
    PerSitePsfFeature,
    ReadoutModelKind,
    UniformPsfFeature,
    apply_readout_model,
    extract_readout_features,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import (
    CalibrationCaptureLayout,
    FrameContract,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.readout.physical_context import ReadoutPhysicalContext
from tests.calibration_physics_oracle import calibrate_readout_arrays_for_test


_FIXTURES = Path(__file__).with_name("fixtures")
_MANIFEST = _FIXTURES / "main_readout_oracle.json"
_ORACLE = _FIXTURES / "main_readout_oracle.npz"

_MODEL_CASES = (
    (ReadoutModelKind.BOX, "box", BoxFeature),
    (ReadoutModelKind.PER_SITE_PSF, "psf", PerSitePsfFeature),
    (ReadoutModelKind.UNIFORM_PSF, "uniform_psf", UniformPsfFeature),
)
_MODEL_FIELD_PREFIXES = (
    "quick_thresholds",
    "short_signals",
    "bin_edges",
    "thresholds",
    "pred",
    "site_fidelity",
    "site_model_fidelity",
    "site_fidelity_dark",
    "site_fidelity_bright",
    "site_mu_dark",
    "site_sigma_dark",
    "site_mu_bright",
    "site_sigma_bright",
    "site_bright_above",
    "site_n_test",
    "site_n_train_dark",
    "site_n_train_bright",
    "aggregate_fidelity",
    "global_threshold",
    "global_fidelity",
    "split_train",
    "split_test",
    "ablation_drop_k",
    "ablation_excluded",
    "ablation_fidelity",
    "ablation_errors",
    "ablation_n_valid",
    "runtime_signals",
    "runtime_occupied",
    "runtime_rate",
)
_COMMON_ORACLE_FIELDS = {
    "input_reference_frames",
    "input_short_frames",
    "input_latent_occupancy",
    "reference_average",
    "centers_row_major",
    "centers_serpentine",
    "centers_column_major",
    "centers_column_serpentine",
    "box_boxes_xywh",
    "psf_boxes_xywh",
    "psf_kernels",
    "psf_fit_centers_xy",
    "psf_fit_sigma_xy",
    "psf_fit_ok",
    "uniform_boxes_xywh",
    "uniform_kernel",
    "reference_box_signals",
    "labels_occupied",
    "labels_dark",
    "labels_valid",
    "reference_fit_threshold",
    "reference_fit_fidelity",
    "reference_fit_dark_mean",
    "reference_fit_dark_sigma",
    "reference_fit_bright_mean",
    "reference_fit_bright_sigma",
    "reference_fit_bright_above",
    "reference_fit_ok",
    "runtime_probe_indices",
}
_EXPECTED_ORACLE_FIELDS = _COMMON_ORACLE_FIELDS | {
    f"{prefix}_{name}"
    for _kind, name, _feature_type in _MODEL_CASES
    for prefix in _MODEL_FIELD_PREFIXES
}


def _physical_context(contract: FrameContract) -> ReadoutPhysicalContext:
    return ReadoutPhysicalContext(
        50e6,
        "a" * 64,
        0.0,
        contract.exposure_seconds,
        (),
        (),
    )


def _assert_close(actual: object, expected: object) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=2e-12,
        equal_nan=True,
    )


@pytest.fixture(scope="module")
def oracle() -> dict[str, np.ndarray]:
    with np.load(_ORACLE, allow_pickle=False) as archive:
        assert set(archive.files) == _EXPECTED_ORACLE_FIELDS
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _calibration_contracts(
    image_shape_yx: tuple[int, int],
    *,
    namespace: str = "main-oracle",
    grid_shape_yx: tuple[int, int] = (2, 3),
    box_radius: int = 1,
) -> tuple[CalibrationSourceBinding, FrameContract, CalibrationAnalysisRequest]:
    height, width = image_shape_yx
    coordinate_frame = CoordinateFrameId(f"{namespace}-camera-pixels")
    y_axis = AxisSpec(
        AxisId(f"{namespace}-y"),
        "main oracle y",
        SPATIAL_Y,
        height,
        coordinates=tuple(range(height)),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId(f"{namespace}-x"),
        "main oracle x",
        SPATIAL_X,
        width,
        coordinates=tuple(range(width)),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    frame_schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<f8"),
        "camera-count",
    )
    layout = CalibrationCaptureLayout(AxisId(f"{namespace}-event"), (0, 2), 1)
    source_binding = CalibrationSourceBinding(
        CaptureArtifactRef(namespace, "0" * 64),
        layout,
    )
    frame_contract = FrameContract(
        ReadoutBindingKey(f"{namespace}-readout"),
        f"{namespace}-camera",
        f"{namespace}-sensor",
        f"{namespace}-optical-path",
        image_shape_yx,
        (0, 0),
        image_shape_yx,
        (1, 1),
        y_axis.axis_id,
        x_axis.axis_id,
        coordinate_frame,
        np.dtype("<f8"),
        "camera-count",
        0.005,
        1.0,
        "external-trigger",
        None,
        frame_schema,
    )
    request = CalibrationAnalysisRequest(
        layout,
        grid_shape_yx,
        ordering=GridOrder.ROW_MAJOR,
        box_radius=box_radius,
        box_reducer=BoxReducer.MEAN,
        psf_half_width=3,
        psf_background=BackgroundMode.ANNULUS_MEDIAN,
        psf_background_padding=3,
        model_kinds=tuple(case[0] for case in _MODEL_CASES),
        default_model_kind=ReadoutModelKind.BOX,
        train_fraction=0.9,
        split_seed=0,
        histogram_bins=120,
        detector_min_distance=None,
        detector_threshold_rel=0.35,
        detector_refine_half=2,
    )
    return source_binding, frame_contract, request


def test_main_quick_threshold_remains_runtime_fallback_when_label_fit_is_nan() -> None:
    references = np.zeros((10, 2, 9, 9), dtype="<f8")
    short = np.zeros((10, 9, 9), dtype="<f8")
    references[:2, :, 4, 4] = 100.0
    references[2:, :, 4, 4] = 10.0
    short[:2, 4, 4] = 5.0
    source_binding, frame_contract, request = _calibration_contracts(
        (9, 9),
        namespace="main-quick-fallback",
        grid_shape_yx=(1, 1),
        box_radius=0,
    )

    result = calibrate_readout_arrays_for_test(
        references,
        short,
        source_binding=source_binding,
        frame_contract=frame_contract,
        readout_physical_context=_physical_context(frame_contract),
        request=request,
    )

    np.testing.assert_allclose(
        result.artifact.site_map.coordinates_xy,
        [[4.0, 4.0]],
        rtol=0.0,
        atol=2e-7,
    )
    expected_quick = {
        ReadoutModelKind.BOX: 2.4739583333333335,
        ReadoutModelKind.PER_SITE_PSF: 2.3150113691884893,
        ReadoutModelKind.UNIFORM_PSF: 2.3150113691884893,
    }
    for kind, expected_threshold in expected_quick.items():
        report = result.report.model(kind)
        model = result.artifact.select_model(kind)
        assert np.isnan(report.thresholds[0])
        _assert_close(report.quick_thresholds, [expected_threshold])
        _assert_close(model.thresholds, [expected_threshold])
        np.testing.assert_array_equal(model.usable_sites.mask, [True])

        model = result.artifact.select_model(kind)
        bright = apply_readout_model(
            model,
            _frame(short[0], frame_contract),
            expected_frame_schema=frame_contract.frame_schema,
        )
        dark = apply_readout_model(
            model,
            _frame(short[-1], frame_contract),
            expected_frame_schema=frame_contract.frame_schema,
        )
        np.testing.assert_array_equal(bright.occupied.values, [True])
        np.testing.assert_array_equal(dark.occupied.values, [False])


@pytest.fixture(scope="module")
def calibrated(
    oracle: dict[str, np.ndarray],
) -> tuple[CalibrationComputation, FrameContract]:
    references = oracle["input_reference_frames"]
    short = oracle["input_short_frames"]
    source_binding, frame_contract, request = _calibration_contracts(
        tuple(int(size) for size in short.shape[-2:])
    )
    request = replace(
        request,
        expected_centers_xy=oracle["centers_row_major"],
        maximum_site_residual_px=0.1,
    )
    result = calibrate_readout_arrays_for_test(
        references,
        short,
        source_binding=source_binding,
        frame_contract=frame_contract,
        readout_physical_context=_physical_context(frame_contract),
        request=request,
    )
    return result, frame_contract


def _regular_grid_template(
    grid_shape_yx: tuple[int, int],
    image_shape_yx: tuple[int, int],
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = grid_shape_yx
    height, width = image_shape_yx
    y_positions = np.linspace(0.18 * height, 0.82 * height, rows)
    x_positions = np.linspace(0.18 * width, 0.82 * width, columns)
    yy, xx = np.mgrid[:height, :width]
    image = np.full(image_shape_yx, 200.0)
    truth = []
    for amplitude, y, x in zip(
        np.asarray(amplitudes, dtype=float).reshape(-1),
        np.repeat(y_positions, columns),
        np.tile(x_positions, rows),
        strict=True,
    ):
        image += amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.6**2)
        )
        truth.append((x, y))
    return image, np.asarray(truth, dtype="<f8")


@pytest.mark.parametrize("dark_mode", ("corners", "row"))
def test_site_admission_preserves_main_dark_site_lattice_repair(dark_mode):
    grid = (5, 7)
    shape = (96, 128)
    amplitudes = np.full(grid, 700.0)
    if dark_mode == "corners":
        amplitudes.flat[[0, 6, 28, 34]] = 0.0
        seed = 0
    else:
        amplitudes[2, :] = 0.0
        seed = 1
    image, truth = _regular_grid_template(grid, shape, amplitudes)
    image += np.random.default_rng(seed).normal(0.0, 4.0, image.shape)
    centers = find_site_centers(image, grid)
    _source, _contract, request = _calibration_contracts(
        shape,
        namespace=f"dark-{dark_mode}",
        grid_shape_yx=grid,
    )
    request = replace(
        request,
        expected_centers_xy=truth,
        maximum_site_residual_px=0.1,
    )

    _validate_site_center_admission(centers, request)
    np.testing.assert_allclose(centers, truth, rtol=0.0, atol=0.1)


def test_site_admission_rejects_a_self_consistent_but_wrong_regular_grid():
    expected = np.asarray(
        ((8.0, 8.0), (24.0, 8.0), (8.0, 24.0), (24.0, 24.0)),
        dtype="<f8",
    )
    wrong_but_regular = expected + np.asarray((5.0, 3.0))
    _source, _contract, request = _calibration_contracts(
        (32, 32),
        namespace="wrong-regular-grid",
        grid_shape_yx=(2, 2),
    )
    request = replace(
        request,
        expected_centers_xy=expected,
        maximum_site_residual_px=1.0,
    )

    with pytest.raises(ValueError, match="differs from expected_centers_xy"):
        _validate_site_center_admission(wrong_but_regular, request)


def test_brighter_extra_peak_cannot_enter_an_authoritative_site_map():
    shape = (64, 64)
    truth = np.asarray(
        ((15.0, 15.0), (45.0, 15.0), (15.0, 45.0), (45.0, 45.0)),
        dtype="<f8",
    )
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = np.zeros(shape, dtype="<f8")
    for amplitude, (x, y) in zip((100.0, 90.0, 80.0, 10.0), truth, strict=True):
        image += amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.5**2)
        )
    image += 200.0 * np.exp(
        -((xx - 31.0) ** 2 + (yy - 31.0) ** 2) / (2.0 * 1.5**2)
    )
    image += np.random.default_rng(1).normal(0.0, 0.2, image.shape)
    detected = find_site_centers(image, (2, 2))
    assert np.max(np.linalg.norm(detected - truth, axis=1)) > 10.0

    source, contract, request = _calibration_contracts(
        shape,
        namespace="brighter-extra-peak",
        grid_shape_yx=(2, 2),
    )
    request = replace(
        request,
        expected_centers_xy=truth,
        maximum_site_residual_px=2.0,
    )
    references = np.broadcast_to(image, (4, 2, *shape))
    short = np.zeros((4, *shape), dtype="<f8")

    with pytest.raises(
        ValueError,
        match="differs from expected_centers_xy",
    ):
        calibrate_readout_arrays_for_test(
            references,
            short,
            source_binding=source,
            frame_contract=contract,
            readout_physical_context=_physical_context(contract),
            request=request,
        )


def _frame(values: np.ndarray, frame_contract: FrameContract) -> Value:
    return Value(np.asarray(values, dtype="<f8"), VALID, frame_contract.frame_schema)


def test_oracle_has_only_the_pinned_main_authority() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["format"] == "main-readout-oracle"
    assert manifest["authority_commit"] == (
        "6c337d49c7086fa0ff21f879cd159bdf0e753f51"
    )
    assert manifest["input"]["origin"] == (
        "deterministic synthetic raw camera frames, frozen in the NPZ"
    )
    assert all(
        owner.startswith("Zou_lab_control.neutral_atom.")
        for owner in manifest["authority_functions"]
    )


@pytest.mark.parametrize(
    ("ordering", "expected_field"),
    (
        (GridOrder.ROW_MAJOR, "centers_row_major"),
        (GridOrder.SERPENTINE, "centers_serpentine"),
        (GridOrder.COLUMN_MAJOR, "centers_column_major"),
        (GridOrder.COLUMN_SERPENTINE, "centers_column_serpentine"),
    ),
)
def test_main_detector_centers_and_ordering(
    oracle: dict[str, np.ndarray],
    ordering: GridOrder,
    expected_field: str,
) -> None:
    observed = find_site_centers(
        oracle["reference_average"],
        (2, 3),
        ordering=ordering,
    )
    _assert_close(observed, oracle[expected_field])


def test_main_box_reducers_share_one_frozen_raw_frame_oracle(
    oracle: dict[str, np.ndarray],
    calibrated: tuple[CalibrationComputation, FrameContract],
) -> None:
    result, frame_contract = calibrated
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    reducer_oracle = manifest["box_reducer_oracle"]
    assert reducer_oracle["source"] == (
        "input_reference_frames[0,0] with box_boxes_xywh"
    )
    base = result.artifact.select_model(ReadoutModelKind.BOX).feature
    assert isinstance(base, BoxFeature)
    frame = _frame(oracle["input_reference_frames"][0, 0], frame_contract)

    for reducer in BoxReducer:
        feature = BoxFeature(
            base.site_axis,
            base.boxes_xywh,
            reducer,
            base.valid_sites,
        )
        observed = extract_readout_features(feature, frame)
        _assert_close(observed.values, reducer_oracle[reducer.value])
        np.testing.assert_array_equal(observed.validity.mask, True)


def test_complete_main_same_frame_calibration_oracle(
    oracle: dict[str, np.ndarray],
    calibrated: tuple[CalibrationComputation, FrameContract],
) -> None:
    result, _frame_contract = calibrated
    artifact = result.artifact
    report = result.report

    # This is provenance, not a numerically frozen environmental expectation.
    lineage = dict(report.software_lineage)
    assert set(lineage) == {"numpy", "scipy"}
    assert all(lineage.values())

    # Every reference shot contributes to the detector image; no latest-frame
    # shortcut is allowed here.
    all_reference_average = np.mean(
        oracle["input_reference_frames"],
        axis=(0, 1),
        dtype=np.float64,
    )
    _assert_close(report.reference_average, all_reference_average)
    _assert_close(report.reference_average, oracle["reference_average"])
    _assert_close(
        artifact.site_map.coordinates_xy,
        oracle["centers_row_major"],
    )
    assert artifact.site_map.grid_shape_yx == (2, 3)
    assert artifact.site_map.ordering is GridOrder.ROW_MAJOR
    np.testing.assert_array_equal(artifact.site_map.validity.mask, True)

    _assert_close(
        report.reference_box_signals,
        oracle["reference_box_signals"],
    )
    labels = report.labels
    np.testing.assert_array_equal(labels.occupied, oracle["labels_occupied"])
    np.testing.assert_array_equal(labels.dark, oracle["labels_dark"])
    np.testing.assert_array_equal(labels.valid, oracle["labels_valid"])
    np.testing.assert_array_equal(
        labels.occupied[labels.valid],
        oracle["input_latent_occupancy"][labels.valid],
    )
    reference_fit_fields = {
        "threshold": "reference_fit_threshold",
        "fidelity": "reference_fit_fidelity",
        "dark_mean": "reference_fit_dark_mean",
        "dark_sigma": "reference_fit_dark_sigma",
        "bright_mean": "reference_fit_bright_mean",
        "bright_sigma": "reference_fit_bright_sigma",
    }
    for attribute, field in reference_fit_fields.items():
        _assert_close(
            np.asarray([getattr(fit, attribute) for fit in labels.fits]),
            oracle[field],
        )
    np.testing.assert_array_equal(
        [fit.bright_above for fit in labels.fits],
        oracle["reference_fit_bright_above"],
    )
    np.testing.assert_array_equal(
        [fit.ok for fit in labels.fits],
        oracle["reference_fit_ok"],
    )

    for _kind, name, _feature_type in _MODEL_CASES:
        np.testing.assert_array_equal(
            report.split.train,
            oracle[f"split_train_{name}"],
        )
        np.testing.assert_array_equal(
            report.split.test,
            oracle[f"split_test_{name}"],
        )

    psf_centers = np.asarray([item.center_xy for item in report.psf_fits])
    psf_sigmas = np.asarray([item.sigma_xy for item in report.psf_fits])
    psf_ok = np.asarray([item.fit_ok for item in report.psf_fits])
    _assert_close(psf_centers, oracle["psf_fit_centers_xy"])
    _assert_close(psf_sigmas, oracle["psf_fit_sigma_xy"])
    np.testing.assert_array_equal(psf_ok, oracle["psf_fit_ok"])

    for kind, name, feature_type in _MODEL_CASES:
        model = artifact.select_model(kind)
        model_report = report.model(kind)
        feature = model.feature
        assert isinstance(feature, feature_type)
        assert model.kind is kind
        assert model_report.kind is kind
        assert feature.site_axis == artifact.site_map.site_axis
        np.testing.assert_array_equal(feature.valid_sites.mask, True)

        boxes_field = {
            "box": "box_boxes_xywh",
            "psf": "psf_boxes_xywh",
            "uniform_psf": "uniform_boxes_xywh",
        }[name]
        np.testing.assert_array_equal(feature.boxes_xywh, oracle[boxes_field])
        if isinstance(feature, BoxFeature):
            assert feature.reducer is BoxReducer.MEAN
        elif isinstance(feature, PerSitePsfFeature):
            _assert_close(feature.kernels, oracle["psf_kernels"])
            assert feature.background is BackgroundMode.ANNULUS_MEDIAN
            assert feature.background_padding == 3
        else:
            assert isinstance(feature, UniformPsfFeature)
            _assert_close(feature.kernel, oracle["uniform_kernel"])
            assert feature.background is BackgroundMode.ANNULUS_MEDIAN
            assert feature.background_padding == 3

        _assert_close(
            model_report.quick_thresholds,
            oracle[f"quick_thresholds_{name}"],
        )
        _assert_close(model_report.short_signals, oracle[f"short_signals_{name}"])
        np.testing.assert_array_equal(model_report.short_validity, True)
        _assert_close(model_report.bin_edges, oracle[f"bin_edges_{name}"])
        _assert_close(model_report.thresholds, oracle[f"thresholds_{name}"])
        _assert_close(model.thresholds, oracle[f"thresholds_{name}"])
        np.testing.assert_array_equal(
            model_report.predictions,
            oracle[f"pred_{name}"],
        )

        metric_float_fields = {
            "fidelity": "site_fidelity",
            "model_fidelity": "site_model_fidelity",
            "fidelity_dark": "site_fidelity_dark",
            "fidelity_bright": "site_fidelity_bright",
            "dark_mean": "site_mu_dark",
            "dark_sigma": "site_sigma_dark",
            "bright_mean": "site_mu_bright",
            "bright_sigma": "site_sigma_bright",
        }
        for attribute, prefix in metric_float_fields.items():
            _assert_close(
                np.asarray(
                    [getattr(item, attribute) for item in model_report.site_fidelity]
                ),
                oracle[f"{prefix}_{name}"],
            )
        np.testing.assert_array_equal(
            [item.bright_above for item in model_report.site_fidelity],
            oracle[f"site_bright_above_{name}"],
        )
        for attribute, prefix in (
            ("n_test", "site_n_test"),
            ("n_train_dark", "site_n_train_dark"),
            ("n_train_bright", "site_n_train_bright"),
        ):
            np.testing.assert_array_equal(
                [getattr(item, attribute) for item in model_report.site_fidelity],
                oracle[f"{prefix}_{name}"],
            )
        expected_usable = np.asarray(
            [
                item.bright_above
                and np.isfinite(item.threshold)
                and np.isfinite(item.fidelity)
                for item in model_report.site_fidelity
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(model.usable_sites.mask, expected_usable)

        _assert_close(
            model_report.aggregate_fidelity,
            oracle[f"aggregate_fidelity_{name}"],
        )
        _assert_close(
            model_report.global_threshold,
            oracle[f"global_threshold_{name}"],
        )
        _assert_close(
            model_report.global_fidelity,
            oracle[f"global_fidelity_{name}"],
        )
        np.testing.assert_array_equal(
            [item.drop_worst_k for item in model_report.ablation],
            oracle[f"ablation_drop_k_{name}"],
        )
        np.testing.assert_array_equal(
            np.stack([item.excluded_sites for item in model_report.ablation]),
            oracle[f"ablation_excluded_{name}"],
        )
        _assert_close(
            [item.fidelity for item in model_report.ablation],
            oracle[f"ablation_fidelity_{name}"],
        )
        np.testing.assert_array_equal(
            [item.errors for item in model_report.ablation],
            oracle[f"ablation_errors_{name}"],
        )
        np.testing.assert_array_equal(
            [item.n_valid for item in model_report.ablation],
            oracle[f"ablation_n_valid_{name}"],
        )


def test_runtime_uses_training_features_and_preserves_repeat_point_site_shape(
    oracle: dict[str, np.ndarray],
    calibrated: tuple[CalibrationComputation, FrameContract],
) -> None:
    result, frame_contract = calibrated
    short_frames = oracle["input_short_frames"]
    probe_grid = oracle["runtime_probe_indices"].astype(np.int64).reshape(2, 3)

    for kind, name, _feature_type in _MODEL_CASES:
        model = result.artifact.select_model(kind)
        training_report = result.report.model(kind)

        # Applying every training frame through the public runtime path must
        # reproduce the feature matrix and decisions used during calibration.
        all_signals = np.empty_like(training_report.short_signals)
        all_occupied = np.empty_like(training_report.predictions)
        for group in range(short_frames.shape[0]):
            applied = apply_readout_model(
                result.artifact.select_model(kind),
                _frame(short_frames[group], frame_contract),
                expected_frame_schema=frame_contract.frame_schema,
            )
            all_signals[group, :] = applied.signals.values
            all_occupied[group, :] = applied.occupied.values
        _assert_close(all_signals, training_report.short_signals)
        np.testing.assert_array_equal(all_occupied, training_report.predictions)

        # Map by the two named leading indices.  The site/data dimension is
        # appended intact; it is never flattened into an R/P item list.
        site_count = model.feature.site_axis.size
        runtime_signals = np.empty((*probe_grid.shape, site_count), dtype="<f8")
        runtime_occupied = np.empty((*probe_grid.shape, site_count), dtype=bool)
        for repeat, point in np.ndindex(probe_grid.shape):
            frame_index = int(probe_grid[repeat, point])
            applied = apply_readout_model(
                result.artifact.select_model(kind),
                _frame(short_frames[frame_index], frame_contract),
                expected_frame_schema=frame_contract.frame_schema,
            )
            runtime_signals[repeat, point, :] = applied.signals.values
            runtime_occupied[repeat, point, :] = applied.occupied.values

        assert runtime_signals.shape == (2, 3, site_count)
        assert runtime_occupied.shape == (2, 3, site_count)
        _assert_close(runtime_signals, oracle[f"runtime_signals_{name}"])
        np.testing.assert_array_equal(
            runtime_occupied,
            oracle[f"runtime_occupied_{name}"],
        )
        _assert_close(
            np.mean(runtime_occupied, axis=-1, keepdims=True),
            oracle[f"runtime_rate_{name}"],
        )
