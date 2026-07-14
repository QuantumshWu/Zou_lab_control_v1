"""Independent readout oracle generated only by main@6c337d49."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zlc_data import (
    VALID,
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
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReadoutModel,
    BoxReducer,
    PerSitePsfReadoutModel,
    ReadoutFeatureSpec,
    ReadoutModelHeader,
    ReadoutModelKind,
    ReadoutModelQuality,
    UniformPsfReadoutModel,
    classify_occupancy,
    extract_readout_features,
)


_FIXTURES = Path(__file__).with_name("fixtures")
_MANIFEST = _FIXTURES / "main_readout_oracle.json"
_ORACLE = _FIXTURES / "main_readout_oracle.npz"


def _frame(values: np.ndarray) -> Value:
    coordinate_frame = CoordinateFrameId("main-oracle-camera-pixels")
    y_axis = AxisSpec(
        AxisId("main-oracle-y"),
        "main oracle y",
        SPATIAL_Y,
        values.shape[0],
        coordinates=tuple(range(values.shape[0])),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId("main-oracle-x"),
        "main oracle x",
        SPATIAL_X,
        values.shape[1],
        coordinates=tuple(range(values.shape[1])),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    return Value(
        np.asarray(values, dtype="<f8"),
        VALID,
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            np.dtype("<f8"),
            "camera-count",
        ),
    )


def _quality(site_axis_id: AxisId, site_count: int) -> ReadoutModelQuality:
    valid = ComponentValidity((site_axis_id,), np.ones(site_count, dtype=bool))
    training = np.full(site_count, 20, dtype="<u8")
    successes = np.full(site_count, 10, dtype="<u8")
    totals = np.full(site_count, 10, dtype="<u8")
    lower_bounds = np.full(site_count, 0.8, dtype="<f8")
    fidelity = np.ones(site_count, dtype="<f8")
    return ReadoutModelQuality(
        site_axis_id,
        valid,
        training,
        training,
        successes,
        totals,
        totals,
        successes,
        totals,
        totals,
        lower_bounds,
        lower_bounds,
        fidelity,
        valid,
        "main-oracle",
    )


def _header(
    site_axis_id: AxisId,
    thresholds: np.ndarray,
) -> ReadoutModelHeader:
    values = np.asarray(thresholds, dtype="<f8")
    return ReadoutModelHeader(
        "0" * 64,
        "1" * 64,
        site_axis_id,
        values,
        np.ones(values.shape, dtype=bool),
        _quality(site_axis_id, values.size),
    )


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


def test_main_same_frames_match_three_feature_models_and_runtime_occupancy() -> None:
    with np.load(_ORACLE, allow_pickle=False) as oracle:
        site_count = int(oracle["centers_row_major"].shape[0])
        site_axis_id = AxisId("main-oracle-site")
        site_validity = ComponentValidity(
            (site_axis_id,), np.ones(site_count, dtype=bool)
        )
        model_specs = {
            "box": (
                BoxReadoutModel(
                    _header(site_axis_id, oracle["thresholds_box"]),
                    oracle["box_boxes_xywh"].astype("<i8"),
                    BoxReducer.MEAN,
                ),
                ReadoutFeatureSpec(
                    ReadoutModelKind.BOX,
                    site_axis_id,
                    oracle["box_boxes_xywh"].astype("<i8"),
                    site_validity,
                    box_reducer=BoxReducer.MEAN,
                ),
            ),
            "psf": (
                PerSitePsfReadoutModel(
                    _header(site_axis_id, oracle["thresholds_psf"]),
                    oracle["psf_boxes_xywh"].astype("<i8"),
                    oracle["psf_kernels"].astype("<f8"),
                    BackgroundMode.ANNULUS_MEDIAN,
                    3,
                ),
                ReadoutFeatureSpec(
                    ReadoutModelKind.PER_SITE_PSF,
                    site_axis_id,
                    oracle["psf_boxes_xywh"].astype("<i8"),
                    site_validity,
                    per_site_kernels=oracle["psf_kernels"].astype("<f8"),
                    background=BackgroundMode.ANNULUS_MEDIAN,
                    background_padding=3,
                ),
            ),
            "uniform_psf": (
                UniformPsfReadoutModel(
                    _header(site_axis_id, oracle["thresholds_uniform_psf"]),
                    oracle["uniform_boxes_xywh"].astype("<i8"),
                    oracle["uniform_kernel"].astype("<f8"),
                    BackgroundMode.ANNULUS_MEDIAN,
                    3,
                ),
                ReadoutFeatureSpec(
                    ReadoutModelKind.UNIFORM_PSF,
                    site_axis_id,
                    oracle["uniform_boxes_xywh"].astype("<i8"),
                    site_validity,
                    uniform_kernel=oracle["uniform_kernel"].astype("<f8"),
                    background=BackgroundMode.ANNULUS_MEDIAN,
                    background_padding=3,
                ),
            ),
        }
        short_frames = oracle["input_short_frames"]
        probe_indices = oracle["runtime_probe_indices"].astype(np.int64)
        for name, (model, feature_spec) in model_specs.items():
            observed_signals = []
            observed_occupied = []
            for index in probe_indices:
                signals = extract_readout_features(
                    feature_spec,
                    _frame(short_frames[int(index)]),
                )
                observed_signals.append(signals.values)
                observed_occupied.append(classify_occupancy(model, signals).occupied)
            np.testing.assert_allclose(
                np.asarray(observed_signals).reshape(2, 3, site_count),
                oracle[f"runtime_signals_{name}"],
                rtol=1e-12,
                atol=2e-12,
            )
            np.testing.assert_array_equal(
                np.asarray(observed_occupied).reshape(2, 3, site_count),
                oracle[f"runtime_occupied_{name}"],
            )
