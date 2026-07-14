"""Independent same-frame readout golden against the legacy physical owners."""

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
    classify_occupancy,
    extract_readout_features,
)


_FIXTURE = Path(__file__).with_name("fixtures") / "legacy_readout_same_frame.json"


def _golden() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _frame(payload: dict[str, object]) -> Value:
    values = np.asarray(payload["frame"], dtype="<f8")
    coordinate_frame = CoordinateFrameId("legacy-qcmos-crop-pixels")
    y_axis = AxisSpec(
        AxisId("legacy-qcmos-y"),
        "legacy qCMOS crop y",
        SPATIAL_Y,
        values.shape[0],
        coordinates=tuple(range(values.shape[0])),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId("legacy-qcmos-x"),
        "legacy qCMOS crop x",
        SPATIAL_X,
        values.shape[1],
        coordinates=tuple(range(values.shape[1])),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<f8"),
        "photon-count",
    )
    return Value(values, VALID, schema)


def _site_contract(payload: dict[str, object]) -> tuple[AxisId, ComponentValidity]:
    site_count = len(payload["centers_xy"])
    axis_id = AxisId("legacy-readout-site")
    return axis_id, ComponentValidity(
        (axis_id,),
        np.ones(site_count, dtype=bool),
    )


def _quality(axis_id: AxisId, site_count: int) -> ReadoutModelQuality:
    valid = ComponentValidity((axis_id,), np.ones(site_count, dtype=bool))
    training = np.full(site_count, 20, dtype="<u8")
    successes = np.full(site_count, 10, dtype="<u8")
    totals = np.full(site_count, 10, dtype="<u8")
    lower_bounds = np.full(site_count, 0.8, dtype="<f8")
    fidelity = np.ones(site_count, dtype="<f8")
    return ReadoutModelQuality(
        axis_id,
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
        "legacy-same-frame-oracle",
    )


def _header(
    axis_id: AxisId,
    thresholds: object,
) -> ReadoutModelHeader:
    threshold_array = np.asarray(thresholds, dtype="<f8")
    return ReadoutModelHeader(
        "0" * 64,
        "1" * 64,
        axis_id,
        threshold_array,
        np.ones(threshold_array.shape, dtype=bool),
        _quality(axis_id, threshold_array.size),
    )


def test_real_qcmos_frame_matches_legacy_box_and_psf_feature_values() -> None:
    """The fixed expected values come from the exact legacy functions, not this tree.

    Fixture provenance pins the old commit, function owners, archive names, and source
    hashes.  The tracked crop is sufficient to reproduce both sites' full padded PSF
    windows, so the ignored multi-megabyte source archives are not a test dependency.
    """

    payload = _golden()
    provenance = payload["provenance"]
    assert provenance["legacy_commit"] == (
        "6c337d49c7086fa0ff21f879cd159bdf0e753f51"
    )
    assert provenance["source_site_indices"] == [0, 1]
    assert provenance["crop_origin_xy"] == [17, 5]

    frame = _frame(payload)
    axis_id, site_validity = _site_contract(payload)
    expected = payload["expected"]

    box_arguments = dict(
        kind=ReadoutModelKind.BOX,
        site_axis_id=axis_id,
        boxes_xywh=np.asarray(payload["box_xywh"], dtype="<i8"),
        site_validity=site_validity,
    )
    box_mean = extract_readout_features(
        ReadoutFeatureSpec(**box_arguments, box_reducer=BoxReducer.MEAN),
        frame,
    )
    box_sum = extract_readout_features(
        ReadoutFeatureSpec(**box_arguments, box_reducer=BoxReducer.SUM),
        frame,
    )

    psf_arguments = dict(
        kind=ReadoutModelKind.PER_SITE_PSF,
        site_axis_id=axis_id,
        boxes_xywh=np.asarray(payload["psf_boxes_xywh"], dtype="<i8"),
        site_validity=site_validity,
        per_site_kernels=np.asarray(payload["psf_kernels"], dtype="<f8"),
    )
    psf_none = extract_readout_features(
        ReadoutFeatureSpec(
            **psf_arguments,
            background=BackgroundMode.NONE,
            background_padding=0,
        ),
        frame,
    )
    psf_annulus = extract_readout_features(
        ReadoutFeatureSpec(
            **psf_arguments,
            background=BackgroundMode.ANNULUS_MEDIAN,
            background_padding=3,
        ),
        frame,
    )

    for observed, expected_key in (
        (box_mean, "box_mean"),
        (box_sum, "box_sum"),
        (psf_none, "psf_none"),
        (psf_annulus, "psf_annulus_median"),
    ):
        assert observed.validity.mask.tolist() == expected["valid"]
        np.testing.assert_allclose(
            observed.values,
            np.asarray(expected[expected_key], dtype="<f8"),
            rtol=0.0,
            atol=2e-15,
        )


def test_real_qcmos_frame_matches_legacy_occupancy_direction_and_site_order() -> None:
    payload = _golden()
    expected = payload["expected"]
    frame = _frame(payload)
    axis_id, site_validity = _site_contract(payload)

    box_model = BoxReadoutModel(
        _header(axis_id, expected["box_thresholds"]),
        np.asarray(payload["box_xywh"], dtype="<i8"),
        BoxReducer.MEAN,
    )
    box_signals = extract_readout_features(
        ReadoutFeatureSpec(
            ReadoutModelKind.BOX,
            axis_id,
            np.asarray(payload["box_xywh"], dtype="<i8"),
            site_validity,
            box_reducer=BoxReducer.MEAN,
        ),
        frame,
    )

    psf_model = PerSitePsfReadoutModel(
        _header(axis_id, expected["psf_thresholds"]),
        np.asarray(payload["psf_boxes_xywh"], dtype="<i8"),
        np.asarray(payload["psf_kernels"], dtype="<f8"),
        BackgroundMode.ANNULUS_MEDIAN,
        3,
    )
    psf_signals = extract_readout_features(
        ReadoutFeatureSpec(
            ReadoutModelKind.PER_SITE_PSF,
            axis_id,
            np.asarray(payload["psf_boxes_xywh"], dtype="<i8"),
            site_validity,
            per_site_kernels=np.asarray(payload["psf_kernels"], dtype="<f8"),
            background=BackgroundMode.ANNULUS_MEDIAN,
            background_padding=3,
        ),
        frame,
    )

    for decision in (
        classify_occupancy(box_model, box_signals),
        classify_occupancy(psf_model, psf_signals),
    ):
        assert decision.validity.mask.tolist() == expected["valid"]
        assert decision.occupied.tolist() == expected["occupied"]
        assert decision.occupied.tolist() == [True, False]
