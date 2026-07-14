"""Adversarial tests for the one headless calibration-analysis path."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version as distribution_version
from itertools import permutations, product
import math
from types import SimpleNamespace

import numpy as np
import pytest
import zlc_neutral_atom.readout.analysis as analysis_impl
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
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
    ValidityContract,
    ValueSchema,
    Value,
)
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.readout import (
    BackgroundMode,
    BoxAnalysisConfig,
    BoxReadoutModel,
    BoxReducer,
    CalibrationAnalysisError,
    CalibrationAnalysisPlanningAssumption,
    CalibrationAnalysisRequest,
    CalibrationAnalysisResult,
    CalibrationAnalysisResourcePolicy,
    CalibrationBracketSamplingAssumption,
    CalibrationWorkPlan,
    CalibrationCaptureLayout,
    CalibrationResourceExceeded,
    CalibrationParameter,
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    GridOrder,
    PerSitePsfReadoutModel,
    PsfAnalysisConfig,
    ReferenceClassOrientation,
    ReferenceLabelSource,
    ReadoutBindingKey,
    ReadoutModelKind,
    SiteDetectionPolicy,
    UniformPsfReadoutModel,
    UsableSiteAcceptance,
    analyze_calibration,
    build_calibration_work_plan,
    decode_calibration_artifact,
    encode_calibration_artifact,
    encode_readout_model,
    validate_calibration_analysis_contract,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress


_CENTERS = ((7, 7), (24, 7), (7, 24), (24, 24))
_SPOT_PROFILE = np.array(
    [[0.42, 0.60, 0.42], [0.60, 1.00, 0.60], [0.42, 0.60, 0.42]],
    dtype=np.float64,
)


def _paint_spot(frame, x, y, level):
    frame[y - 1 : y + 2, x - 1 : x + 2] = level * _SPOT_PROFILE


def _request(**changes):
    base = CalibrationAnalysisRequest(
        CalibrationCaptureLayout(AxisId("readout-event"), (0, 2), 1),
        (2, 2),
        ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY,
        ReferenceClassOrientation.ABOVE_IS_OCCUPIED,
        CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS,
        CalibrationAnalysisPlanningAssumption.PRECOMMITTED_BEFORE_SOURCE_INSPECTION,
        box=BoxAnalysisConfig(1, BoxReducer.SUM),
        model_kinds=(
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
        ),
        default_model_kind=ReadoutModelKind.BOX,
        psf=PsfAnalysisConfig(1, BackgroundMode.NONE, 0),
        train_fraction=0.35,
        minimum_held_out_class_accuracy_lower_bound=0.60,
    )
    if "model_kinds" in changes and all(
        kind is ReadoutModelKind.BOX for kind in changes["model_kinds"]
    ) and "psf" not in changes:
        changes["psf"] = None
    return replace(base, **changes)


def _capture(
    *,
    mapping_seed: int = 4,
    repeats: int = 24,
    inverted_readout: bool = False,
    mixed_readout_directions: bool = False,
    ambiguous_reference: bool = False,
    one_class: bool = False,
    sparse_context: bool = False,
    context_count: int | None = None,
    invalid_sample: bool = True,
    reference_scale_mismatch: bool = False,
    bad_readout_site: int | None = None,
):
    height = width = 32
    coordinate_frame = CoordinateFrameId("qcam-roi-local-output-pixels")
    y_axis = AxisSpec(
        AxisId("camera-y"),
        "ROI-local y",
        SPATIAL_Y,
        height,
        tuple(range(height)),
        "pixel",
        coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId("camera-x"),
        "ROI-local x",
        SPATIAL_X,
        width,
        tuple(range(width)),
        "pixel",
        coordinate_frame,
    )
    cell_schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<f8"),
        "count",
    )
    event_axis = AxisSpec(
        AxisId("readout-event"),
        "readout event",
        READOUT_EVENT,
        3,
    )
    detuning_axis = AxisSpec(
        AxisId("detuning"),
        "detuning",
        SCAN_POINT,
        2,
        (-1.0, 1.0),
        "MHz",
    )
    phase_axis = AxisSpec(
        AxisId("phase"),
        "phase",
        SCAN_POINT,
        2,
        (0, 1),
    )
    contexts = list(product(range(2), range(2)))
    if context_count is not None:
        contexts = (
            [(0, 0), (1, 0)]
            if context_count == 2
            else contexts[:context_count]
        )
    logical = [(event, detuning, phase) for event in range(3) for detuning, phase in contexts]
    if sparse_context:
        logical = [item for item in logical if item[1:] != (1, 1)]
    rng = np.random.default_rng(mapping_seed)
    rng.shuffle(logical)
    point_layout = PointLayout.explicit((3, 2, 2), tuple(logical))
    schema = DatasetSchema(
        AxisSpec(AxisId("repeat"), "repeat", REPEAT, repeats),
        (event_axis, detuning_axis, phase_axis),
        point_layout,
        cell_schema,
    )
    values = np.zeros(schema.physical_shape, dtype="<f8")
    validity = np.ones(schema.physical_shape, dtype=bool)

    def occupied(repeat, detuning, phase, site):
        return True if one_class else (repeat + detuning + 2 * phase + site) % 2 == 0

    for storage_row in range(point_layout.storage_size):
        event, detuning, phase = point_layout.multi_index(storage_row)
        for repeat in range(repeats):
            frame = values[repeat, storage_row]
            for site, (x, y) in enumerate(_CENTERS):
                atom = occupied(repeat, detuning, phase, site)
                if event in (0, 2):
                    level = 80.0 if atom else 8.0
                    if reference_scale_mismatch and event == 2:
                        level *= 10.0
                    if (
                        ambiguous_reference
                        and event == 2
                        and repeat == 0
                        and detuning == 0
                        and phase == 0
                        and site == 0
                    ):
                        level = 8.0 if atom else 80.0
                elif inverted_readout or (mixed_readout_directions and site % 2 == 1):
                    level = 3.0 if atom else 30.0
                else:
                    level = 30.0 if atom else 3.0
                if event == 1 and site == bad_readout_site:
                    level = 10.0
                _paint_spot(frame, x, y, level + 0.01 * repeat)

    if invalid_sample:
        row = point_layout.storage_index((1, 0, 0))
        x, y = _CENTERS[0]
        validity[1, row, y, x] = False
        values[1, row, y, x] = np.nan

    block = DataBlock(
        BlockId("raw-calibration"),
        DatasetRevision(1),
        values,
        ComponentValidity((y_axis.axis_id, x_axis.axis_id), validity),
        schema,
    )
    binding = ReadoutBindingKey("primary-readout")
    descriptor = CameraCaptureDescriptor(
        camera_identity="qcam-serial-1",
        sensor_identity="sensor-1",
        optical_path="objective-a",
        sensor_shape_yx=(height, width),
        roi_origin_yx=(0, 0),
        roi_shape_yx=(height, width),
        binning_yx=(1, 1),
        spatial_y_axis_id=y_axis.axis_id,
        spatial_x_axis_id=x_axis.axis_id,
        coordinate_frame=coordinate_frame,
        dtype=np.dtype("<f8"),
        count_unit="count",
        readout_event_axis_id=event_axis.axis_id,
        event_settings=(
            CameraEventReadoutSetting(0, 0.020, 1.0, "reference"),
            CameraEventReadoutSetting(1, 0.002, 1.0, "readout"),
            CameraEventReadoutSetting(2, 0.020, 1.0, "reference"),
        ),
        camera_arm_spec_fingerprint="a" * 64,
    )
    schedule = tuple(
        reversed(
            tuple(
                DatasetCellAddress(repeat, row)
                for repeat in range(repeats)
                for row in range(point_layout.storage_size)
            )
        )
    )
    return SimpleNamespace(
        ref=CaptureArtifactRef("captures", "b" * 64),
        block=block,
        camera_provenance=SimpleNamespace(
            descriptor=descriptor,
            binding=binding,
        ),
        source_cell_schedule=schedule,
    )


def _replace_block(capture, values):
    return SimpleNamespace(
        **{
            **capture.__dict__,
            "block": DataBlock(
                capture.block.block_id,
                capture.block.revision,
                values,
                capture.block.validity,
                capture.block.schema,
            ),
        }
    )


def _replace_block_and_validity(capture, values, validity):
    return SimpleNamespace(
        **{
            **capture.__dict__,
            "block": DataBlock(
                capture.block.block_id,
                capture.block.revision,
                values,
                ComponentValidity(capture.block.validity.axis_ids, validity),
                capture.block.schema,
            ),
        }
    )


def _mutate_partition_test_only(capture, request):
    brackets = request.layout.brackets(capture.block.schema)
    partition = analysis_impl._freeze_partition(brackets, request)
    values = capture.block.values.copy()
    repeat_axis = capture.block.schema.repeat_axis.axis_id
    for bracket_index in partition.test_indices:
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis]
        for _event, row in bracket.reference_point_storage_rows:
            frame = values[repeat, row]
            frame[15, 15] = 20_000.0  # a test-only extra candidate must not enter SiteMap
            for x, y in _CENTERS:
                frame[y, x] = frame[y, x] * (20.0 if frame[y, x] > 40.0 else 0.1)
        frame = values[repeat, bracket.readout_point_storage_row]
        for x, y in _CENTERS:
            frame[y, x] = frame[y, x] * (20.0 if frame[y, x] > 10.0 else 0.1)
    return _replace_block(capture, values)


def _mutate_partition_reference_evidence_only(capture, request):
    brackets = request.layout.brackets(capture.block.schema)
    partition = analysis_impl._freeze_partition(brackets, request)
    values = capture.block.values.copy()
    repeat_axis = capture.block.schema.repeat_axis.axis_id
    for bracket_index in partition.reference_evidence_indices:
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis]
        for _event, row in bracket.reference_point_storage_rows:
            for x, y in _CENTERS:
                _paint_spot(values[repeat, row], x, y, 44.0)
    return _replace_block(capture, values)


def _replace_all_reference_images(capture, painter):
    values = capture.block.values.copy()
    schema = capture.block.schema
    event_position = next(
        index for index, axis in enumerate(schema.point_axes) if axis.role == READOUT_EVENT
    )
    for row in range(schema.point_layout.storage_size):
        event = schema.point_layout.multi_index(row)[event_position]
        if event not in (0, 2):
            continue
        for repeat in range(schema.repeat_axis.size):
            values[repeat, row].fill(0.0)
            painter(values[repeat, row])
    return _replace_block(capture, values)


def _replace_bracket_levels(capture, request, levels):
    values = capture.block.values.copy()
    brackets = request.layout.brackets(capture.block.schema)
    repeat_axis_id = capture.block.schema.repeat_axis.axis_id
    levels = tuple(levels)
    assert len(levels) == len(brackets)
    for bracket, (reference_levels, readout_level) in zip(
        brackets,
        levels,
        strict=True,
    ):
        repeat = dict(bracket.context_key)[repeat_axis_id]
        assert len(reference_levels) == len(bracket.reference_point_storage_rows)
        for (_event, row), level in zip(
            bracket.reference_point_storage_rows,
            reference_levels,
            strict=True,
        ):
            for x, y in _CENTERS:
                _paint_spot(values[repeat, row], x, y, level)
        for x, y in _CENTERS:
            _paint_spot(
                values[repeat, bracket.readout_point_storage_row],
                x,
                y,
                readout_level,
            )
    return _replace_block(capture, values)


def test_one_path_builds_all_closed_models_with_shared_map_and_real_quality_evidence():
    capture = _capture(invalid_sample=False)
    result = analyze_calibration(capture, _request())
    artifact = result.artifact

    assert artifact.source_binding.source_capture_ref == capture.ref
    assert artifact.source_binding.layout == _request().layout
    assert artifact.default_model_kind is ReadoutModelKind.BOX
    assert artifact.select_model().kind is ReadoutModelKind.BOX
    assert {type(model) for model in artifact.models} == {
        BoxReadoutModel,
        PerSitePsfReadoutModel,
        UniformPsfReadoutModel,
    }
    assert all(
        model.header.site_map_fingerprint == artifact.site_map.fingerprint
        for model in artifact.models
    )
    assert result.diagnostics.bracket_count == 96
    assert result.diagnostics.train_bracket_count == 33
    assert result.diagnostics.reference_evidence_bracket_count == 33
    assert result.diagnostics.test_bracket_count == 30
    assert result.diagnostics.reference_frame_count == 192
    parameters = {item.name: item.value for item in artifact.parameters}
    assert parameters["reference-label-source"] == "UNSUPERVISED_REFERENCE_VALLEY"
    assert parameters["reference-class-orientation"] == "ABOVE_IS_OCCUPIED"
    assert parameters["reference-statistical-unit"] == "BRACKET"
    assert parameters["reference-evidence-assumption"] == (
        "INDEPENDENT_STATIONARY_BRACKETS_COMPLETE_REFERENCE_FEATURES"
    )
    assert parameters["bracket-sampling-assumption"] == (
        "INDEPENDENT_STATIONARY_BRACKETS"
    )
    assert parameters["analysis-planning-assumption"] == (
        "PRECOMMITTED_BEFORE_SOURCE_INSPECTION"
    )
    assert parameters["held-out-family-scope"] == "ARTIFACT_MODEL_SITE"
    assert parameters["held-out-family-model-count"] == 3
    assert parameters["held-out-family-hypothesis-count"] == 12
    assert parameters["reference-evidence-bracket-count"] == 33
    assert "reference-valley-gate-version" not in parameters
    assert "reference-ambiguity-gate-version" not in parameters
    for model in artifact.models:
        usable = model.header.quality.usable_sites.mask
        assert not hasattr(model.header, "model_id")
        assert not hasattr(model.header, "model_version")
        assert not hasattr(model.header.quality, "quality_gate_version")
        assert not hasattr(model.header.quality, "gate_passed")
        model_parameters = {
            item.name: item.value for item in model.header.parameters
        }
        assert model_parameters["reference-evidence-bracket-count"] == 33
        assert np.all(model.header.quality.dark_training_sample_counts[usable] > 0)
        assert np.all(model.header.quality.bright_training_sample_counts[usable] > 0)
        assert np.all(model.header.quality.held_out_fidelity[usable] == 1.0)
    per_site = artifact.select_model(kind=ReadoutModelKind.PER_SITE_PSF)
    uniform = artifact.select_model(kind=ReadoutModelKind.UNIFORM_PSF)
    assert np.all(per_site.kernels >= 0.0)
    assert np.allclose(np.sum(per_site.kernels, axis=(1, 2)), 1.0)
    assert uniform.kernel.ndim == 2
    assert not hasattr(uniform, "kernels")
    assert np.all(uniform.kernel >= 0.0)
    assert np.isclose(np.sum(uniform.kernel), 1.0)


@pytest.mark.parametrize("mapping_seed", range(32))
def test_named_brackets_ignore_non_row_major_storage_and_reversed_source_ordinals(mapping_seed):
    result = analyze_calibration(_capture(mapping_seed=mapping_seed), _request())
    coordinates = result.artifact.site_map.coordinates_xy
    assert np.allclose(coordinates[:, 0], [7, 24, 7, 24], atol=0.3)
    assert np.allclose(coordinates[:, 1], [7, 7, 24, 24], atol=0.3)
    for model in result.artifact.models:
        usable = model.header.quality.usable_sites.mask
        assert np.count_nonzero(usable) == 4
        assert np.all(model.header.occupied_above_thresholds[usable])


def test_sparse_multiaxis_context_is_preserved_as_independent_brackets():
    result = analyze_calibration(
        _capture(sparse_context=True),
        _request(
            minimum_held_out_class_accuracy_lower_bound=0.50,
            reference_valley_familywise_error_rate=0.05,
        ),
    )
    assert result.diagnostics.bracket_count == 72
    assert result.artifact.source_binding.bracket_count == 72
    assert result.diagnostics.consensus_dark_counts == (36, 36, 36, 36)
    assert result.diagnostics.consensus_bright_counts == (36, 36, 36, 36)


def test_component_invalid_nan_and_reference_disagreement_remove_samples_not_make_dark_labels():
    clean = analyze_calibration(_capture(ambiguous_reference=False), _request())
    ambiguous = analyze_calibration(_capture(ambiguous_reference=True), _request())
    assert (
        ambiguous.diagnostics.consensus_bright_counts[0]
        + ambiguous.diagnostics.consensus_dark_counts[0]
        == clean.diagnostics.consensus_bright_counts[0]
        + clean.diagnostics.consensus_dark_counts[0]
        - 1
    )
    # Invalid short-readout evidence is an adverse class failure whenever it
    # lands in heldout; it is never converted to a dark label or a NaN score.
    box = ambiguous.artifact.select_model(kind=ReadoutModelKind.BOX)
    assert box.header.quality.usable_sites.mask[0]
    assert box.header.quality.held_out_validity.mask[0]
    assert np.isfinite(box.header.quality.held_out_fidelity[0])


def test_reference_events_are_classified_on_their_own_declared_scale_before_consensus():
    result = analyze_calibration(
        _capture(reference_scale_mismatch=True),
        _request(model_kinds=(ReadoutModelKind.BOX,), default_model_kind=ReadoutModelKind.BOX),
    )
    assert result.diagnostics.consensus_dark_counts == (48, 48, 48, 48)
    assert result.diagnostics.consensus_bright_counts == (48, 48, 48, 48)
    assert np.all(result.artifact.select_model().header.quality.usable_sites.mask)


def test_inverted_readout_direction_is_explicit_per_site():
    result = analyze_calibration(
        _capture(inverted_readout=True, invalid_sample=False),
        _request(model_kinds=(ReadoutModelKind.BOX,), default_model_kind=ReadoutModelKind.BOX),
    )
    model = result.artifact.select_model()
    usable = model.header.quality.usable_sites.mask
    assert np.all(~model.header.occupied_above_thresholds[usable])
    assert np.all(model.header.quality.held_out_fidelity[usable] == 1.0)

    mixed = analyze_calibration(
        _capture(mixed_readout_directions=True),
        _request(model_kinds=(ReadoutModelKind.BOX,), default_model_kind=ReadoutModelKind.BOX),
    ).artifact.select_model()
    assert mixed.header.occupied_above_thresholds.tolist() == [True, False, True, False]


def test_low_quality_site_uses_domain_canonical_fillers_in_every_closed_model():
    artifact = analyze_calibration(
        _capture(bad_readout_site=0),
        _request(
            usable_site_acceptance=UsableSiteAcceptance.MINIMUM_FRACTION,
            minimum_usable_site_fraction=0.75,
        ),
    ).artifact
    for model in artifact.models:
        quality = model.header.quality
        assert quality.usable_sites.mask.tolist() == [False, True, True, True]
        assert model.header.thresholds[0] == 0.0
        assert not model.header.occupied_above_thresholds[0]
        assert quality.dark_training_sample_counts[0] > 0
        assert quality.bright_training_sample_counts[0] > 0
        assert quality.held_out_validity.mask[0]
        assert quality.held_out_fidelity[0] < 0.6
        assert min(
            quality.held_out_dark_accuracy_lower_bounds[0],
            quality.held_out_bright_accuracy_lower_bounds[0],
        ) < _request().minimum_held_out_class_accuracy_lower_bound
        if isinstance(model, BoxReadoutModel):
            assert model.boxes_xywh[0].tolist() == [0, 0, 1, 1]
        else:
            assert model.boxes_xywh[0].tolist() == [0, 0, 3, 3]
    per_site = artifact.select_model(kind=ReadoutModelKind.PER_SITE_PSF)
    expected = np.zeros((3, 3), dtype=float)
    expected[0, 0] = 1.0
    assert np.array_equal(per_site.kernels[0], expected)


def test_missing_class_or_heldout_evidence_fails_closed_instead_of_admitting_fillers():
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(
            _capture(one_class=True),
            _request(model_kinds=(ReadoutModelKind.BOX,), default_model_kind=ReadoutModelKind.BOX),
        )
    with pytest.raises(
        (CalibrationAnalysisError, CalibrationResourceExceeded),
        match="samples per class|populate train|prominence|admitted",
    ):
        analyze_calibration(
            _capture(repeats=1, context_count=2, invalid_sample=False),
            _request(model_kinds=(ReadoutModelKind.BOX,), default_model_kind=ReadoutModelKind.BOX),
        )


def test_reference_valley_uses_independent_proposal_and_exact_evidence():
    separated = np.array(
        (-1.4, -1.3, -1.2, -1.1, -1.0, -0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
    )
    proposal = analysis_impl._reference_threshold_proposal(separated, 4, 2.0)
    assert proposal is not None
    assert -0.9 < proposal.threshold < 0.9

    evidence = np.repeat((-1.1, 1.1), 20)
    valley = analysis_impl._reference_valley_evidence(evidence, proposal)
    assert valley.middle_count == 0
    assert valley.outside_count == 0
    assert valley.valley_pvalue == pytest.approx(2.0**-20)

    transformed = 17.0 + 9.0 * separated
    transformed_proposal = analysis_impl._reference_threshold_proposal(
        np.concatenate((transformed, (np.nan,))),
        4,
        2.0,
    )
    assert transformed_proposal is not None
    assert transformed_proposal.threshold == pytest.approx(
        17.0 + 9.0 * proposal.threshold
    )
    transformed_valley = analysis_impl._reference_valley_evidence(
        17.0 + 9.0 * evidence,
        transformed_proposal,
    )
    assert transformed_valley == valley


def test_reference_valley_rejects_continuous_middle_mass():
    proposal = analysis_impl._reference_threshold_proposal(
        np.repeat((-1.0, 1.0), 12),
        4,
        2.0,
    )
    assert proposal is not None
    evidence = np.concatenate(
        (np.repeat(-1.0, 5), np.repeat(0.0, 20), np.repeat(1.0, 5))
    )
    valley = analysis_impl._reference_valley_evidence(evidence, proposal)
    assert valley.valley_pvalue > 0.5


def test_reference_valley_treats_evidence_only_third_population_as_adverse():
    proposal = analysis_impl._reference_threshold_proposal(
        np.repeat((-1.0, 1.0), 20),
        4,
        2.0,
    )
    assert proposal is not None
    evidence = np.concatenate(
        (np.repeat(-1.0, 20), np.repeat(1.0, 20), np.repeat(10.0, 100))
    )
    valley = analysis_impl._reference_valley_evidence(evidence, proposal)
    assert valley.outside_count == 100
    assert valley.valley_pvalue > 0.99


def test_reference_valley_rejects_standard_single_mode_families():
    proposal = analysis_impl._reference_threshold_proposal(
        np.repeat((-1.0, 1.0), 30),
        4,
        2.0,
    )
    assert proposal is not None
    rng = np.random.default_rng(123)
    distributions = {
        "normal": rng.normal(0.0, 0.7, 400),
        "uniform": rng.uniform(-1.5, 1.5, 400),
        "exponential": rng.exponential(0.5, 400) - 0.5,
        "lognormal": rng.lognormal(0.0, 0.4, 400) - math.exp(0.08),
    }
    for name, evidence in distributions.items():
        valley = analysis_impl._reference_valley_evidence(evidence, proposal)
        assert valley.valley_pvalue > 0.05, name


def test_full_pipeline_rejects_one_class_common_mode_drift():
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        minimum_reference_cluster_separation_rss=1.0,
        reference_valley_familywise_error_rate=0.05,
    )
    capture = _capture(one_class=True, invalid_sample=False)
    bracket_count = len(request.layout.brackets(capture.block.schema))
    rng = np.random.default_rng(3)
    drift = rng.uniform(-1.0, 1.0, size=bracket_count)
    mutated = _replace_bracket_levels(
        capture,
        request,
        (
            ((80.0 + 20.0 * value,) * 2, 30.0 + 7.5 * value)
            for value in drift
        ),
    )
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(mutated, request)


def test_full_pipeline_rejects_three_level_reference_and_readout_state():
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        reference_valley_familywise_error_rate=0.05,
    )
    capture = _capture(invalid_sample=False)
    bracket_count = len(request.layout.brackets(capture.block.schema))
    reference_levels = (8.0, 44.0, 80.0)
    readout_levels = (3.0, 16.0, 30.0)
    mutated = _replace_bracket_levels(
        capture,
        request,
        (
            (
                (reference_levels[index % 3],) * 2,
                readout_levels[index % 3],
            )
            for index in range(bracket_count)
        ),
    )
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(mutated, request)


def test_full_pipeline_requires_every_reference_event_to_pass_the_valley_gate():
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        reference_valley_familywise_error_rate=0.05,
    )
    capture = _capture(invalid_sample=False)
    bracket_count = len(request.layout.brackets(capture.block.schema))
    mutated = _replace_bracket_levels(
        capture,
        request,
        (
            (
                ((8.0 if index % 2 == 0 else 80.0), 44.0),
                (3.0 if index % 2 == 0 else 30.0),
            )
            for index in range(bracket_count)
        ),
    )
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(mutated, request)


def test_selectively_invalid_reference_middle_mass_remains_adverse_full_pipeline(
    monkeypatch,
):
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        usable_site_acceptance=UsableSiteAcceptance.MINIMUM_FRACTION,
        minimum_usable_site_fraction=0.75,
        reference_valley_familywise_error_rate=0.05,
    )
    capture = _capture(invalid_sample=False)
    brackets = request.layout.brackets(capture.block.schema)
    partition = analysis_impl._freeze_partition(brackets, request)
    train_positions = {index: position for position, index in enumerate(partition.train_indices)}
    evidence_positions = {
        index: position
        for position, index in enumerate(partition.reference_evidence_indices)
    }

    def levels(index):
        if index in train_positions:
            level = 8.0 if train_positions[index] % 2 == 0 else 80.0
        elif index in evidence_positions:
            level = (8.0, 44.0, 80.0)[evidence_positions[index] % 3]
        else:
            level = 8.0 if index % 2 == 0 else 80.0
        return (level, level), (3.0 if level < 44.0 else 30.0)

    complete = _replace_bracket_levels(
        capture,
        request,
        (levels(index) for index in range(len(brackets))),
    )
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(complete, request)

    values = complete.block.values.copy()
    validity = complete.block.validity.mask.copy()
    repeat_axis_id = complete.block.schema.repeat_axis.axis_id
    for bracket_index, position in evidence_positions.items():
        if position % 3 != 1:
            continue
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis_id]
        for _event, row in bracket.reference_point_storage_rows:
            for x, y in _CENTERS:
                validity[repeat, row, y - 1 : y + 2, x - 1 : x + 2] = False
    selectively_invalid = _replace_block_and_validity(complete, values, validity)
    recorded = []
    original = analysis_impl._learn_reference_thresholds

    def capture_diagnostics(*args, **kwargs):
        result = original(*args, **kwargs)
        recorded.extend(result[2])
        return result

    monkeypatch.setattr(
        analysis_impl,
        "_learn_reference_thresholds",
        capture_diagnostics,
    )
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(selectively_invalid, request)
    assert recorded
    assert all(
        item.evidence.sample_count == len(partition.reference_evidence_indices)
        for item in recorded
    )
    assert all(item.evidence.invalid_count > 0 for item in recorded)


def test_selectively_invalid_third_reference_cluster_cannot_remove_nested_rejection():
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        reference_valley_familywise_error_rate=0.05,
    )
    capture = _capture(invalid_sample=False)
    brackets = request.layout.brackets(capture.block.schema)
    levels = (8.0, 44.0, 80.0)
    mutated = _replace_bracket_levels(
        capture,
        request,
        (
            ((levels[index % 3],) * 2, (3.0, 16.0, 30.0)[index % 3])
            for index in range(len(brackets))
        ),
    )
    partition = analysis_impl._freeze_partition(brackets, request)
    validity = mutated.block.validity.mask.copy()
    repeat_axis_id = mutated.block.schema.repeat_axis.axis_id
    for bracket_index in partition.reference_evidence_indices:
        if bracket_index % 3 != 2:
            continue
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis_id]
        for _event, row in bracket.reference_point_storage_rows:
            for x, y in _CENTERS:
                validity[repeat, row, y - 1 : y + 2, x - 1 : x + 2] = False
    selectively_invalid = _replace_block_and_validity(
        mutated,
        mutated.block.values.copy(),
        validity,
    )
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(selectively_invalid, request)


def test_selectively_invalid_heldout_misclassifications_remain_class_failures():
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        usable_site_acceptance=UsableSiteAcceptance.MINIMUM_FRACTION,
        minimum_usable_site_fraction=0.75,
    )
    capture = _capture(invalid_sample=False)
    brackets = request.layout.brackets(capture.block.schema)
    partition = analysis_impl._freeze_partition(brackets, request)
    repeat_axis_id = capture.block.schema.repeat_axis.axis_id
    values = capture.block.values.copy()
    for bracket_index in partition.test_indices:
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis_id]
        x, y = _CENTERS[0]
        reference_row = bracket.reference_point_storage_rows[0][1]
        occupied = values[repeat, reference_row, y, x] > 40.0
        _paint_spot(
            values[repeat, bracket.readout_point_storage_row],
            x,
            y,
            3.0 if occupied else 30.0,
        )
    wrong = _replace_block(capture, values)
    wrong_quality = analyze_calibration(wrong, request).artifact.select_model().header.quality
    assert wrong_quality.usable_sites.mask.tolist() == [False, True, True, True]

    validity = wrong.block.validity.mask.copy()
    x, y = _CENTERS[0]
    for bracket_index in partition.test_indices:
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis_id]
        validity[
            repeat,
            bracket.readout_point_storage_row,
            y - 1 : y + 2,
            x - 1 : x + 2,
        ] = False
    masked = _replace_block_and_validity(wrong, wrong.block.values.copy(), validity)
    masked_quality = (
        analyze_calibration(masked, request).artifact.select_model().header.quality
    )
    assert masked_quality.usable_sites.mask.tolist() == [False, True, True, True]
    for name in (
        "held_out_dark_success_counts",
        "held_out_dark_total_counts",
        "held_out_dark_labeled_counts",
        "held_out_bright_success_counts",
        "held_out_bright_total_counts",
        "held_out_bright_labeled_counts",
    ):
        assert np.array_equal(getattr(masked_quality, name), getattr(wrong_quality, name))


def test_reference_valley_accepts_quantized_two_level_statistical_evidence_only():
    values = np.repeat((-3.0, 4.0), 12)
    proposal = analysis_impl._reference_threshold_proposal(values, 4, 2.0)
    assert proposal is not None
    assert proposal.threshold == 0.5
    assert proposal.cluster_separation_rss == math.inf
    assert (
        _request().reference_label_source
        is ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY
    )


def test_holm_step_down_is_deterministic_for_order_and_ties():
    pvalues = np.array((0.002, 0.002, 0.02, 0.8))
    expected = np.array((True, True, False, False))
    assert np.array_equal(analysis_impl._holm_rejections(pvalues, 0.01), expected)
    order = np.array((2, 0, 3, 1))
    permuted = analysis_impl._holm_rejections(pvalues[order], 0.01)
    restored = np.zeros_like(permuted)
    restored[order] = permuted
    assert np.array_equal(restored, expected)


def test_held_out_holm_controls_one_artifact_model_site_family():
    pvalues = np.full((2, 2), 0.02, dtype=np.float64)
    assert np.all(
        analysis_impl._holm_rejections(pvalues[0], 0.05)
    )
    assert np.all(
        analysis_impl._holm_rejections(pvalues[1], 0.05)
    )
    assert not np.any(
        analysis_impl._artifact_wide_held_out_rejections(pvalues, 0.05)
    )


def test_full_pipeline_rejects_models_that_only_pass_separate_holm_families():
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(
            _capture(repeats=22, invalid_sample=False),
            _request(),
        )


def test_result_replay_rejects_per_model_holm_evidence_after_model_shopping():
    result = analyze_calibration(_capture(invalid_sample=False), _request())
    site_count = result.artifact.site_map.site_axis.size
    success_count = 14
    total_count = 15
    lower = analysis_impl._one_sided_clopper_pearson_lower_bound(
        success_count,
        total_count,
        0.95,
    )
    fidelity = success_count / total_count
    forged_models = []
    forged_diagnostics = []
    for model, diagnostic in zip(
        result.artifact.models,
        result.diagnostics.models,
        strict=True,
    ):
        quality = model.header.quality
        counts = np.full(site_count, total_count, dtype="<u8")
        successes = np.full(site_count, success_count, dtype="<u8")
        lower_bounds = np.full(site_count, lower, dtype="<f8")
        fidelities = np.full(site_count, fidelity, dtype="<f8")
        forged_quality = replace(
            quality,
            held_out_dark_success_counts=successes,
            held_out_dark_total_counts=counts,
            held_out_dark_labeled_counts=counts,
            held_out_bright_success_counts=successes,
            held_out_bright_total_counts=counts,
            held_out_bright_labeled_counts=counts,
            held_out_dark_accuracy_lower_bounds=lower_bounds,
            held_out_bright_accuracy_lower_bounds=lower_bounds,
            held_out_fidelity=fidelities,
        )
        forged_models.append(
            replace(model, header=replace(model.header, quality=forged_quality))
        )
        forged_diagnostics.append(
            replace(
                diagnostic,
                minimum_fidelity=fidelity,
                mean_fidelity=fidelity,
                minimum_class_accuracy_lower_bound=lower,
                mean_class_accuracy_lower_bound=lower,
            )
        )

    with pytest.raises(ValueError, match="artifact-wide familywise evidence"):
        CalibrationAnalysisResult(
            replace(result.artifact, models=tuple(forged_models)),
            replace(result.diagnostics, models=tuple(forged_diagnostics)),
        )


def test_reference_split_policy_is_fingerprinted_and_round_trips_current_only():
    from zlc_neutral_atom.readout.analysis_codec import (
        decode_calibration_analysis_request,
        encode_calibration_analysis_request,
    )

    request = replace(
        _request(),
        minimum_reference_cluster_separation_rss=2.75,
        reference_valley_familywise_error_rate=0.025,
        reference_evidence_fraction=0.30,
    )
    decoded = decode_calibration_analysis_request(
        encode_calibration_analysis_request(request)
    )
    assert decoded == request
    assert decoded.fingerprint == request.fingerprint
    assert decoded.minimum_reference_cluster_separation_rss == 2.75
    assert decoded.reference_valley_familywise_error_rate == 0.025
    assert decoded.reference_evidence_fraction == 0.30
    assert (
        decoded.reference_label_source
        is ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY
    )


def test_same_request_is_byte_deterministic_and_declared_order_is_canonical():
    capture = _capture()
    first = analyze_calibration(capture, _request())
    reordered = _request(
        model_kinds=(
            ReadoutModelKind.PER_SITE_PSF,
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
        )
    )
    second = analyze_calibration(capture, reordered)
    assert first.artifact.fingerprint == second.artifact.fingerprint
    assert first.diagnostics == second.diagnostics
    assert _request().fingerprint == reordered.fingerprint
    decoded = decode_calibration_artifact(encode_calibration_artifact(first.artifact))
    assert decoded.fingerprint == first.artifact.fingerprint


def test_repository_contract_rejects_coordinated_request_and_geometry_drift():
    capture = _capture(invalid_sample=False)
    request = _request()
    result = analyze_calibration(capture, request)
    plan = build_calibration_work_plan(capture, request)
    brackets = request.layout.brackets(capture.block.schema)
    assert (
        validate_calibration_analysis_contract(
            result,
            request,
            plan,
            source_brackets=brackets,
        )
        is result
    )

    def changed_parameters(parameters, name, value):
        return tuple(
            CalibrationParameter(item.name, value if item.name == name else item.value)
            for item in parameters
        )

    artifact_parameter_drifts = (
        ("grid-rows", 99),
        ("reference-class-orientation", "BELOW_IS_OCCUPIED"),
        ("reference-valley-familywise-error-rate", 0.25),
    )
    for name, value in artifact_parameter_drifts:
        artifact = replace(
            result.artifact,
            parameters=changed_parameters(result.artifact.parameters, name, value),
        )
        forged = CalibrationAnalysisResult(artifact, result.diagnostics)
        with pytest.raises(ValueError, match="parameters differ"):
            validate_calibration_analysis_contract(forged, request, plan)

    default_drift = CalibrationAnalysisResult(
        replace(
            result.artifact,
            default_model_kind=ReadoutModelKind.PER_SITE_PSF,
        ),
        result.diagnostics,
    )
    with pytest.raises(ValueError, match="default model kind differs"):
        validate_calibration_analysis_contract(default_drift, request, plan)

    sampling_drift = replace(
        result.artifact,
        parameters=changed_parameters(
            result.artifact.parameters,
            "bracket-sampling-assumption",
            "UNDECLARED_CONTEXT_MIXTURE",
        ),
    )
    with pytest.raises(ValueError, match="bracket-sampling-assumption differs"):
        CalibrationAnalysisResult(sampling_drift, result.diagnostics)

    for name, value in (
        ("analysis-planning-assumption", "POST_HOC_MODEL_SHOPPING"),
        ("held-out-family-scope", "PER_MODEL"),
        ("held-out-family-model-count", 1),
        ("held-out-family-hypothesis-count", request.site_count),
    ):
        coordinated_models = tuple(
            replace(
                model,
                header=replace(
                    model.header,
                    parameters=changed_parameters(
                        model.header.parameters,
                        name,
                        value,
                    ),
                ),
            )
            for model in result.artifact.models
        )
        coordinated_artifact = replace(
            result.artifact,
            models=coordinated_models,
            parameters=changed_parameters(result.artifact.parameters, name, value),
        )
        with pytest.raises(ValueError, match="artifact analysis parameter"):
            CalibrationAnalysisResult(coordinated_artifact, result.diagnostics)

    box_result = analyze_calibration(
        capture,
        _request(
            model_kinds=(ReadoutModelKind.BOX,),
            default_model_kind=ReadoutModelKind.BOX,
        ),
    )
    box_model = box_result.artifact.models[0]
    bool_count_artifact = replace(
        box_result.artifact,
        models=(
            replace(
                box_model,
                header=replace(
                    box_model.header,
                    parameters=changed_parameters(
                        box_model.header.parameters,
                        "held-out-family-model-count",
                        True,
                    ),
                ),
            ),
        ),
        parameters=changed_parameters(
            box_result.artifact.parameters,
            "held-out-family-model-count",
            True,
        ),
    )
    with pytest.raises(ValueError, match="artifact analysis parameter"):
        CalibrationAnalysisResult(bool_count_artifact, box_result.diagnostics)

    box_index = next(
        index
        for index, model in enumerate(result.artifact.models)
        if isinstance(model, BoxReadoutModel)
    )
    box_model = result.artifact.models[box_index]
    changed_header = replace(
        box_model.header,
        parameters=changed_parameters(
            box_model.header.parameters,
            "minimum-train-samples-per-class",
            999,
        ),
    )
    changed_model = replace(box_model, header=changed_header)
    models = list(result.artifact.models)
    models[box_index] = changed_model
    with pytest.raises(ValueError, match="inconsistent calibration gate policies"):
        CalibrationAnalysisResult(
            replace(result.artifact, models=tuple(models)),
            result.diagnostics,
        )

    models = list(result.artifact.models)
    shifted_boxes = box_model.boxes_xywh.copy()
    shifted_boxes[:, 0] -= 1
    models[box_index] = replace(box_model, boxes_xywh=shifted_boxes)
    forged = CalibrationAnalysisResult(
        replace(result.artifact, models=tuple(models)),
        result.diagnostics,
    )
    with pytest.raises(ValueError, match="extraction geometry differs"):
        validate_calibration_analysis_contract(forged, request, plan)

    models[box_index] = replace(box_model, reducer=BoxReducer.MEAN)
    forged = CalibrationAnalysisResult(
        replace(result.artifact, models=tuple(models)),
        result.diagnostics,
    )
    with pytest.raises(ValueError, match="BOX reducer differs"):
        validate_calibration_analysis_contract(forged, request, plan)


def test_explicit_no_default_policy_keeps_multi_model_selection_fail_closed():
    artifact = analyze_calibration(
        _capture(),
        _request(default_model_kind=None),
    ).artifact
    assert artifact.default_model_kind is None
    with pytest.raises(ValueError, match="default"):
        artifact.select_model()


def test_column_major_order_is_an_explicit_site_identity_choice():
    result = analyze_calibration(
        _capture(),
        _request(grid_order=GridOrder.COLUMN_MAJOR),
    )
    coordinates = result.artifact.site_map.coordinates_xy
    assert np.allclose(coordinates[:, 0], [7, 7, 24, 24], atol=0.3)
    assert np.allclose(coordinates[:, 1], [7, 24, 7, 24], atol=0.3)


def test_missing_typed_cell_address_and_resource_overruns_are_rejected_before_analysis():
    capture = _capture()
    capture.source_cell_schedule = capture.source_cell_schedule[1:]
    with pytest.raises(CalibrationAnalysisError, match="cardinality"):
        analyze_calibration(capture, _request())

    policy = CalibrationAnalysisResourcePolicy(max_brackets=3)
    with pytest.raises(CalibrationResourceExceeded, match="bracket"):
        analyze_calibration(
            _capture(),
            _request(resource_policy=policy),
        )
    pixel_policy = CalibrationAnalysisResourcePolicy(
        max_sampled_pixel_operations=100,
    )
    with pytest.raises(CalibrationResourceExceeded, match="sampled.pixel"):
        analyze_calibration(
            _capture(),
            _request(resource_policy=pixel_policy),
        )

    # Every proposal/evidence/model pass and exact-test work unit is admitted
    # before layout resolution or detector allocation.
    with pytest.raises(CalibrationResourceExceeded, match="signal-evaluation"):
        analyze_calibration(
            _capture(),
            _request(
                resource_policy=CalibrationAnalysisResourcePolicy(
                    max_signal_evaluations=900,
                )
            ),
        )
    with pytest.raises(CalibrationResourceExceeded, match="sampled-pixel"):
        analyze_calibration(
            _capture(),
            _request(
                resource_policy=CalibrationAnalysisResourcePolicy(
                    max_sampled_pixel_operations=50_000,
                )
            ),
        )
    with pytest.raises(CalibrationResourceExceeded, match="modality"):
        analyze_calibration(
            _capture(),
            _request(
                resource_policy=CalibrationAnalysisResourcePolicy(
                    max_modality_test_work_units=1,
                )
            ),
        )
    with pytest.raises(CalibrationResourceExceeded, match="diagnostics"):
        analyze_calibration(
            _capture(),
            _request(
                resource_policy=CalibrationAnalysisResourcePolicy(
                    max_reference_valley_diagnostics=1,
                )
            ),
        )


def test_preflight_rejects_an_evidence_partition_that_cannot_pass_holm():
    capture = _capture(repeats=10, invalid_sample=False)
    with pytest.raises(CalibrationResourceExceeded, match="cannot possibly pass"):
        build_calibration_work_plan(capture, _request())


def test_preflight_accounts_for_every_persisted_model_site_hypothesis():
    capture = _capture(repeats=60, context_count=1, invalid_sample=False)
    one_model = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
    )
    build_calibration_work_plan(capture, one_model)

    with pytest.raises(CalibrationResourceExceeded, match="held-out familywise"):
        build_calibration_work_plan(capture, _request())


def test_box_only_analysis_has_canonical_absent_psf_configuration():
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
    )
    assert request.psf is None
    result = analyze_calibration(
        _capture(),
        request,
    )
    assert len(result.artifact.models) == 1
    assert result.artifact.models[0].kind is ReadoutModelKind.BOX


def test_request_rejects_implicit_model_or_background_ambiguity():
    with pytest.raises(ValueError, match="default_model_kind"):
        CalibrationAnalysisRequest(
            _request().layout,
            (2, 2),
            ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY,
            ReferenceClassOrientation.ABOVE_IS_OCCUPIED,
            CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS,
            CalibrationAnalysisPlanningAssumption.PRECOMMITTED_BEFORE_SOURCE_INSPECTION,
            model_kinds=(ReadoutModelKind.BOX,),
            default_model_kind=ReadoutModelKind.UNIFORM_PSF,
        )
    with pytest.raises(ValueError, match="canonical zero"):
        PsfAnalysisConfig(1, BackgroundMode.NONE, 1)
    with pytest.raises(ValueError, match="canonical absent"):
        CalibrationAnalysisRequest(
            _request().layout,
            (2, 2),
            ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY,
            ReferenceClassOrientation.ABOVE_IS_OCCUPIED,
            CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS,
            CalibrationAnalysisPlanningAssumption.PRECOMMITTED_BEFORE_SOURCE_INSPECTION,
            model_kinds=(ReadoutModelKind.BOX,),
            psf=PsfAnalysisConfig(),
        )
    with pytest.raises(ValueError, match="cluster_separation_rss must be positive"):
        replace(_request(), minimum_reference_cluster_separation_rss=0.0)
    with pytest.raises(ValueError, match="familywise_error_rate must be finite"):
        replace(_request(), reference_valley_familywise_error_rate=float("nan"))
    with pytest.raises(ValueError, match="must be below one"):
        replace(_request(), reference_evidence_fraction=0.70)


def test_frozen_test_partition_cannot_influence_any_learned_parameter(monkeypatch):
    """Held-out frames may change evidence, never the learned calibration state."""

    learned_reference_splits = []
    learned_reference_diagnostics = []
    original = analysis_impl._learn_reference_thresholds

    def record_reference_split(*args, **kwargs):
        thresholds, validity, diagnostics = original(*args, **kwargs)
        learned_reference_splits.append((thresholds.copy(), validity.copy()))
        learned_reference_diagnostics.append(diagnostics)
        return thresholds, validity, diagnostics

    monkeypatch.setattr(
        analysis_impl,
        "_learn_reference_thresholds",
        record_reference_split,
    )
    request = _request()
    capture = _capture(invalid_sample=False)
    baseline = analyze_calibration(capture, request)
    mutated = analyze_calibration(_mutate_partition_test_only(capture, request), request)

    assert len(learned_reference_splits) == 2
    assert np.array_equal(
        learned_reference_splits[0][0],
        learned_reference_splits[1][0],
    )
    assert np.array_equal(
        learned_reference_splits[0][1],
        learned_reference_splits[1][1],
    )
    assert learned_reference_diagnostics[0] == learned_reference_diagnostics[1]
    assert baseline.diagnostics.partition_digest == mutated.diagnostics.partition_digest
    assert baseline.diagnostics.train_bracket_count == mutated.diagnostics.train_bracket_count
    assert baseline.diagnostics.test_bracket_count == mutated.diagnostics.test_bracket_count


    assert baseline.artifact.site_map.fingerprint == mutated.artifact.site_map.fingerprint
    assert np.array_equal(
        baseline.artifact.site_map.coordinates_xy,
        mutated.artifact.site_map.coordinates_xy,
    )

    for first, second in zip(
        baseline.artifact.models,
        mutated.artifact.models,
        strict=True,
    ):
        assert type(first) is type(second)
        assert np.array_equal(first.header.thresholds, second.header.thresholds)
        assert np.array_equal(
            first.header.occupied_above_thresholds,
            second.header.occupied_above_thresholds,
        )
        assert np.array_equal(
            first.header.quality.dark_training_sample_counts,
            second.header.quality.dark_training_sample_counts,
        )
        assert np.array_equal(
            first.header.quality.bright_training_sample_counts,
            second.header.quality.bright_training_sample_counts,
        )
        first_parameters = {item.name: item.value for item in first.header.parameters}
        second_parameters = {item.name: item.value for item in second.header.parameters}
        request_partition = baseline.diagnostics.partition_digest
        assert first_parameters["bracket-partition-digest"] == request_partition
        assert second_parameters["bracket-partition-digest"] == request_partition
        if isinstance(first, PerSitePsfReadoutModel):
            assert np.array_equal(first.kernels, second.kernels)
        elif isinstance(first, UniformPsfReadoutModel):
            assert np.array_equal(first.kernel, second.kernel)
        else:
            assert first.reducer is second.reducer
        assert np.array_equal(first.boxes_xywh, second.boxes_xywh)


def test_reference_evidence_can_reject_but_cannot_relearn_sites_or_proposals(
    monkeypatch,
):
    request = _request(
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
    )
    capture = _capture(invalid_sample=False)
    proposals = []
    detected_sites = []
    original_learn = analysis_impl._learn_reference_thresholds
    original_detect = analysis_impl._detect_sites

    def record_learn(*args, **kwargs):
        result = original_learn(*args, **kwargs)
        proposals.append(tuple(item.proposal_threshold for item in result[2]))
        return result

    def record_detect(*args, **kwargs):
        result = original_detect(*args, **kwargs)
        detected_sites.append(result.coordinates_xy.copy())
        return result

    monkeypatch.setattr(analysis_impl, "_learn_reference_thresholds", record_learn)
    monkeypatch.setattr(analysis_impl, "_detect_sites", record_detect)
    analyze_calibration(capture, request)
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(
            _mutate_partition_reference_evidence_only(capture, request),
            request,
        )
    assert proposals[0] == proposals[1]
    assert np.array_equal(detected_sites[0], detected_sites[1])


def test_site_detector_rejects_one_plateau_instead_of_manufacturing_a_grid():
    def one_plateau(frame):
        frame[7:25, 7:25] = 100.0

    capture = _replace_all_reference_images(
        _capture(invalid_sample=False),
        one_plateau,
    )
    with pytest.raises(CalibrationAnalysisError, match="candidates; grid requires"):
        analyze_calibration(capture, _request())


def test_site_detector_rejects_non_affine_random_scatter():
    def random_scatter(frame):
        for x, y in ((4, 4), (26, 7), (8, 24), (27, 29)):
            _paint_spot(frame, x, y, 100.0)

    capture = _replace_all_reference_images(
        _capture(invalid_sample=False),
        random_scatter,
    )
    with pytest.raises(
        CalibrationAnalysisError,
        match="candidates|lattice residual|lattice bands|orientation",
    ):
        analyze_calibration(capture, _request())


def test_site_detector_second_best_assignment_gate_is_explicit_and_fail_closed():
    strict = replace(
        _request().detection,
        minimum_assignment_cost_gap_pixels_squared=20.0,
    )
    peaks = analysis_impl._PeakCandidates(
        np.array([[10.0, 5.0], [13.0, 5.0], [10.0, 25.0], [13.0, 25.0]]),
        np.ones(4),
        np.full(4, 5, dtype=np.int64),
    )
    with pytest.raises(CalibrationAnalysisError, match="second-best"):
        analysis_impl._assign_unique_affine_lattice(
            peaks,
            _request(detection=strict),
        )


def test_resource_preflight_precedes_layout_and_layout_is_resolved_exactly_once(monkeypatch):
    bracket_calls = []
    layout_scan_calls = []
    axis_index_calls = []
    original_brackets = CalibrationCaptureLayout.brackets
    original_layout_scan = CalibrationCaptureLayout._rows_by_event_and_context
    original_axis_indices = PointLayout.axis_indices

    def counted(self, schema):
        bracket_calls.append((self, schema))
        return original_brackets(self, schema)

    def counted_layout_scan(self, schema):
        layout_scan_calls.append((self, schema))
        return original_layout_scan(self, schema)

    def counted_axis_indices(self, position):
        axis_index_calls.append((self, position))
        return original_axis_indices(self, position)

    monkeypatch.setattr(CalibrationCaptureLayout, "brackets", counted)
    monkeypatch.setattr(
        CalibrationCaptureLayout,
        "_rows_by_event_and_context",
        counted_layout_scan,
    )
    monkeypatch.setattr(PointLayout, "axis_indices", counted_axis_indices)
    capture = _capture(invalid_sample=False)

    with pytest.raises(CalibrationResourceExceeded, match="source cells"):
        analyze_calibration(
            capture,
            _request(
                resource_policy=CalibrationAnalysisResourcePolicy(max_source_cells=1)
            ),
        )
    with pytest.raises(CalibrationResourceExceeded, match="working-byte"):
        analyze_calibration(
            capture,
            _request(
                resource_policy=CalibrationAnalysisResourcePolicy(max_working_bytes=1)
            ),
        )
    assert bracket_calls == []
    assert layout_scan_calls == []
    assert axis_index_calls == []

    result = analyze_calibration(capture, _request())
    assert result.diagnostics.bracket_count == 96
    assert len(bracket_calls) == 1
    assert len(layout_scan_calls) == 1
    assert axis_index_calls == []


def test_schedule_validation_never_materializes_an_unbounded_iterable():
    capture = _capture(invalid_sample=False)
    capture.source_cell_schedule = iter(capture.source_cell_schedule)
    with pytest.raises(TypeError, match="bounded tuple"):
        analyze_calibration(capture, _request())


def test_analysis_routes_reference_and_every_model_through_shared_feature_owner(monkeypatch):
    calls = []
    original = analysis_impl._extract_readout_features_arrays

    def counted(spec, image, validity):
        calls.append(spec.kind)
        return original(spec, image, validity)

    monkeypatch.setattr(analysis_impl, "_extract_readout_features_arrays", counted)
    analyze_calibration(_capture(invalid_sample=False), _request())
    assert set(calls) == {
        ReadoutModelKind.BOX,
        ReadoutModelKind.PER_SITE_PSF,
        ReadoutModelKind.UNIFORM_PSF,
    }


def test_frozen_work_plan_accounts_every_streaming_pass_and_dense_phase():
    capture = _capture(invalid_sample=False)
    request = _request()
    plan = build_calibration_work_plan(capture, request)

    assert isinstance(plan, CalibrationWorkPlan)
    assert plan.full_frame_read_count == 678
    assert plan.reference_evidence_bracket_upper_bound == 33
    assert plan.reference_valley_diagnostic_count == 8
    assert plan.modality_test_work_units > 0
    assert plan.planned_kernel_elements == 45
    assert plan.assignment_scratch_bytes == 64 * request.site_count**2
    assert plan.dense_assignment_work_units == 10 * request.site_count**3
    assert plan.working_peak_bytes >= (
        plan.detector_working_bytes + plan.assignment_scratch_bytes
    )
    assert plan.working_peak_bytes >= (
        plan.psf_working_bytes + plan.feature_working_bytes
    )
    assert len(plan.fingerprint) == 64
    assert plan == build_calibration_work_plan(capture, request)
    with pytest.raises(AttributeError):
        plan.full_frame_read_count = 0


def test_resource_rejection_precedes_layout_detector_assignment_and_templates(monkeypatch):
    capture = _capture(invalid_sample=False)
    kernel_heavy = _request(
        grid_shape_yx=(40, 50),
        reference_valley_familywise_error_rate=0.99,
        minimum_held_out_class_accuracy_lower_bound=0.0,
        model_kinds=(ReadoutModelKind.PER_SITE_PSF,),
        default_model_kind=ReadoutModelKind.PER_SITE_PSF,
        psf=PsfAnalysisConfig(50, BackgroundMode.NONE, 0),
    )
    dense_heavy = _request(
        grid_shape_yx=(8, 8),
        minimum_held_out_class_accuracy_lower_bound=0.0,
        model_kinds=(ReadoutModelKind.BOX,),
        default_model_kind=ReadoutModelKind.BOX,
        resource_policy=CalibrationAnalysisResourcePolicy(
            max_dense_assignment_work_units=100,
        ),
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("heavy operation ran before work-plan admission")

    monkeypatch.setattr(CalibrationCaptureLayout, "brackets", forbidden)
    monkeypatch.setattr(analysis_impl, "gaussian_filter", forbidden)
    monkeypatch.setattr(analysis_impl, "linear_sum_assignment", forbidden)
    monkeypatch.setattr(analysis_impl, "_training_reference_template", forbidden)

    with pytest.raises(CalibrationResourceExceeded, match="kernels"):
        analyze_calibration(capture, kernel_heavy)
    with pytest.raises(CalibrationResourceExceeded, match="dense lattice"):
        analyze_calibration(capture, dense_heavy)


def test_analysis_borrows_every_frame_without_constructing_a_value_snapshot(monkeypatch):
    capture = _capture(invalid_sample=False)
    request = _request()
    plan = build_calibration_work_plan(capture, request)
    borrowed = 0
    original = analysis_impl._FrameAccessor.arrays

    def counted(self, bracket, storage_row):
        nonlocal borrowed
        image, validity = original(self, bracket, storage_row)
        assert np.shares_memory(image, capture.block.values)
        assert not image.flags.writeable
        assert validity.shape == image.shape
        borrowed += 1
        return image, validity

    def forbid_value_snapshot(*_args, **_kwargs):
        pytest.fail("analysis constructed a full-frame zlc_data.Value snapshot")

    monkeypatch.setattr(analysis_impl._FrameAccessor, "arrays", counted)
    monkeypatch.setattr(Value, "__post_init__", forbid_value_snapshot)
    analyze_calibration(capture, request)
    assert borrowed == plan.full_frame_read_count


def _brute_force_second_assignment(matrix):
    costs = sorted(
        math.fsum(float(matrix[row, column]) for row, column in enumerate(order))
        for order in permutations(range(len(matrix)))
    )
    return costs[0], costs[1] - costs[0]


def _edge_exclusion_second_assignment(matrix, rows, columns):
    best = math.fsum(float(matrix[row, column]) for row, column in zip(rows, columns))
    alternatives = []
    for row, column in zip(rows, columns):
        excluded = matrix.copy()
        excluded[row, column] = np.inf
        alternative_rows, alternative_columns = linear_sum_assignment(excluded)
        alternatives.append(
            math.fsum(
                float(excluded[r, c])
                for r, c in zip(alternative_rows, alternative_columns)
            )
        )
    return best, min(alternatives) - best


def test_exact_second_assignment_matches_brute_force_and_old_edge_exclusion():
    rng = np.random.default_rng(4815162342)
    for size in range(2, 8):
        for _case in range(4):
            matrix = rng.integers(0, 30, size=(size, size)).astype(np.float64)
            rows, columns = linear_sum_assignment(matrix)
            order = rng.permutation(size)
            observed = analysis_impl._second_best_assignment_cost_delta(
                matrix,
                rows[order],
                columns[order],
            )
            brute_force = _brute_force_second_assignment(matrix)
            exclusion = _edge_exclusion_second_assignment(matrix, rows, columns)
            assert np.allclose(observed, brute_force, rtol=0.0, atol=1e-12)
            assert np.allclose(observed, exclusion, rtol=0.0, atol=1e-12)

    best, delta = analysis_impl._second_best_assignment_cost_delta(
        np.array([[0.0, 9.0], [9.0, 0.0]]),
        np.array([0, 1]),
        np.array([0, 1]),
    )
    assert best == 0.0
    assert delta == 18.0  # raw objective margin, in pixels squared
    assert analysis_impl._second_best_assignment_cost_delta(
        np.zeros((2, 2)),
        np.array([0, 1]),
        np.array([0, 1]),
    ) == (0.0, 0.0)
    assert analysis_impl._second_best_assignment_cost_delta(
        np.array([[3.0]]),
        np.array([0]),
        np.array([0]),
    ) == (3.0, None)


@pytest.mark.parametrize(
    ("spacing", "angle", "shear", "noise", "amplitudes"),
    (
        (12.0, 0.0, 0.0, 0.3, (80.0, 80.0, 80.0, 80.0)),
        (11.0, 0.22, 0.12, 0.8, (65.0, 95.0, 75.0, 105.0)),
        (10.0, -0.18, -0.10, 1.2, (90.0, 65.0, 105.0, 75.0)),
    ),
)
def test_physical_gaussian_psf_oracle_accepts_independent_affine_sites(
    spacing,
    angle,
    shear,
    noise,
    amplitudes,
):
    yy, xx = np.indices((64, 64), dtype=np.float64)
    column_step = spacing * np.array([math.cos(angle), math.sin(angle)])
    row_step = spacing * np.array(
        [
            shear * math.cos(angle) - math.sin(angle),
            shear * math.sin(angle) + math.cos(angle),
        ]
    )
    origin = np.array([18.0, 18.0])
    expected = np.array(
        [
            origin + column * column_step + row * row_step
            for row in range(2)
            for column in range(2)
        ]
    )
    image = np.full((64, 64), 3.0, dtype=np.float64)
    for center, amplitude in zip(expected, amplitudes):
        image += amplitude * np.exp(
            -(
                (xx - center[0]) ** 2 + (yy - center[1]) ** 2
            )
            / (2.0 * 1.25**2)
        )
    seed = int(round(1000 * (spacing + angle + shear + noise))) & 0xFFFFFFFF
    image += np.random.default_rng(seed).normal(0.0, noise, image.shape)

    detected = analysis_impl._detect_sites(
        image,
        np.ones(image.shape, dtype=bool),
        _request(),
    )
    assert np.allclose(detected.coordinates_xy, expected, atol=0.35)
    assert detected.diagnostic.minimum_peak_to_saddle_prominence > 0.0
    assert detected.diagnostic.minimum_half_prominence_basin_area_pixels >= 4


@pytest.mark.parametrize(("radius", "ripple_amplitude"), ((5.0, 15.0), (7.0, 25.0), (9.0, 25.0)))
def test_broad_single_psf_with_shallow_ripples_cannot_manufacture_a_site_grid(
    radius,
    ripple_amplitude,
):
    yy, xx = np.indices((64, 64), dtype=np.float64)
    image = 3.0 + 100.0 * np.exp(
        -((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2.0 * 10.0**2)
    )
    for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
        x = 32.0 + radius * math.cos(angle)
        y = 32.0 + radius * math.sin(angle)
        image += ripple_amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.2**2)
        )
    with pytest.raises(CalibrationAnalysisError, match="candidates; grid requires"):
        analysis_impl._detect_sites(
            image,
            np.ones(image.shape, dtype=bool),
            _request(),
        )


def test_disconnected_valid_island_cannot_borrow_another_components_floor():
    image = np.zeros((24, 24), dtype=np.float64)
    validity = np.zeros(image.shape, dtype=bool)
    validity[1:8, 1:8] = True
    image[3:6, 3:6] = 30.0
    validity[13:22, 13:22] = True
    image[13:22, 13:22] = 96.0
    image[16:19, 16:19] = 100.0

    candidates = analysis_impl._collapse_prominent_maxima(
        image,
        validity,
        _request().detection,
    )
    assert candidates.coordinates_xy.shape == (1, 2)
    assert candidates.prominences.shape == (1,)
    assert np.all(np.isfinite(candidates.prominences))
    assert candidates.half_prominence_basin_areas.shape == (1,)
    assert np.all(candidates.half_prominence_basin_areas > 0)
    assert np.allclose(candidates.coordinates_xy[0], (4.0, 4.0), atol=0.2)
    assert not np.any(np.linalg.norm(candidates.coordinates_xy - (17.0, 17.0), axis=1) < 3.0)


def test_connected_double_gaussian_prominence_matches_independent_saddle_oracle():
    yy, xx = np.indices((64, 64), dtype=np.float64)
    centers = ((28.5, 32.0), (35.5, 32.0))
    amplitudes = (100.0, 95.0)
    psf_sigma = 2.0
    image = sum(
        amplitude
        * np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2)
            / (2.0 * psf_sigma**2)
        )
        for (center_x, center_y), amplitude in zip(centers, amplitudes)
    )
    smoothing_sigma = 1.0
    policy = replace(
        _request().detection,
        smoothing_sigma_pixels=smoothing_sigma,
        minimum_prominence_fraction=0.01,
        minimum_peak_separation_pixels=0.1,
        minimum_half_prominence_basin_area_pixels=1,
        reject_touching_half_prominence_basins=False,
    )

    candidates = analysis_impl._collapse_prominent_maxima(
        image,
        np.ones(image.shape, dtype=bool),
        policy,
    )

    # For this symmetric two-Gaussian geometry both regional maxima and their
    # maximin connecting saddle lie on y=32.  Compute that physical oracle
    # independently from the detector's graph/watershed machinery.
    smoothed = gaussian_filter(image, smoothing_sigma)
    profile = smoothed[32]
    left_peak = int(np.argmax(profile[25:32]) + 25)
    right_peak = int(np.argmax(profile[33:40]) + 33)
    saddle = float(np.min(profile[left_peak : right_peak + 1]))
    weaker_peak = min(float(profile[left_peak]), float(profile[right_peak]))
    oracle_weaker_prominence = weaker_peak - saddle

    weaker_candidate = int(
        np.argmin(np.linalg.norm(candidates.coordinates_xy - centers[1], axis=1))
    )
    assert candidates.coordinates_xy.shape == (2, 2)
    assert oracle_weaker_prominence == pytest.approx(29.4676612, abs=1e-6)
    assert candidates.prominences[weaker_candidate] == pytest.approx(
        oracle_weaker_prominence,
        rel=1e-10,
        abs=1e-10,
    )


def test_sparse_topographic_graph_keeps_zero_equivalent_edges_and_rejects_flat_input():
    # scipy sparse graphs use numeric zero to mean "no edge".  The detector's
    # positive spacing offset must therefore preserve even an equal-height,
    # negative-intensity edge; for a flat image that offset is the smallest
    # positive subnormal float.
    smooth = np.array([[-3.0, -3.0]], dtype=np.float64)
    prominences, basins = analysis_impl._topographic_prominences(
        smooth,
        np.ones(smooth.shape, dtype=bool),
        np.array([smooth.size, 0], dtype=np.int64),
        np.array([-np.inf, -3.0]),
        np.array([-np.inf, -3.0]),
        1,
    )
    assert prominences[1] == 0.0
    assert basins.tolist() == [[1, 1]]

    policy = replace(
        _request().detection,
        smoothing_sigma_pixels=0.0,
        minimum_prominence_fraction=0.01,
    )
    with pytest.raises(CalibrationAnalysisError, match="no site contrast"):
        analysis_impl._collapse_prominent_maxima(
            np.array([[0.0, 1e-310]], dtype=np.float64),
            np.ones((1, 2), dtype=bool),
            policy,
        )


def test_equal_height_plateau_seeds_have_deterministic_survivor_and_catchments():
    image = np.zeros((5, 9), dtype=np.float64)
    image[1:3, 1:3] = 10.0
    image[1:3, 6:8] = 10.0
    image[2, 3:6] = 2.0
    policy = replace(
        _request().detection,
        smoothing_sigma_pixels=0.0,
        minimum_prominence_fraction=0.05,
        minimum_peak_separation_pixels=0.1,
        minimum_half_prominence_basin_area_pixels=1,
        reject_touching_half_prominence_basins=False,
    )

    observed = [
        analysis_impl._collapse_prominent_maxima(
            image,
            np.ones(image.shape, dtype=bool),
            policy,
        )
        for _ in range(5)
    ]
    first = observed[0]
    assert first.coordinates_xy.tolist() == [[1.5, 1.5], [6.5, 1.5]]
    assert first.prominences.tolist() == [10.0, 8.0]
    assert first.half_prominence_basin_areas.tolist() == [4, 4]
    for repeated in observed[1:]:
        assert np.array_equal(repeated.coordinates_xy, first.coordinates_xy)
        assert np.array_equal(repeated.prominences, first.prominences)
        assert np.array_equal(
            repeated.half_prominence_basin_areas,
            first.half_prominence_basin_areas,
        )


def test_near_collinear_affine_lattice_is_rejected_before_band_heuristics():
    peaks = analysis_impl._PeakCandidates(
        np.array(
            [[10.0, 10.0], [20.0, 10.0], [19.8, 10.1], [29.8, 10.1]],
            dtype=np.float64,
        ),
        np.full(4, 20.0),
        np.full(4, 5, dtype=np.int64),
    )
    with pytest.raises(CalibrationAnalysisError, match="nearly collinear"):
        analysis_impl._assign_unique_affine_lattice(peaks, _request())


def test_single_site_lattice_has_canonical_absent_multisite_diagnostics():
    lattice = analysis_impl._assign_unique_affine_lattice(
        analysis_impl._PeakCandidates(
            np.array([[12.5, 9.5]], dtype=np.float64),
            np.array([20.0]),
            np.array([9], dtype=np.int64),
        ),
        _request(grid_shape_yx=(1, 1)),
    )
    assert np.array_equal(lattice.coordinates_xy, np.array([[12.5, 9.5]]))
    assert lattice.diagnostic.minimum_band_separation_pixels is None
    assert lattice.diagnostic.affine_sin_angle is None
    assert lattice.diagnostic.affine_condition_number is None
    assert lattice.diagnostic.assignment_cost_gap_pixels_squared is None


def test_clopper_pearson_gate_uses_per_class_evidence_and_defaults_to_all_sites():
    expected_perfect_five = 0.05 ** (1.0 / 5.0)
    assert analysis_impl._one_sided_clopper_pearson_lower_bound(5, 5, 0.95) == pytest.approx(
        expected_perfect_five
    )
    assert analysis_impl._one_sided_clopper_pearson_lower_bound(0, 5, 0.95) == 0.0

    with pytest.raises(CalibrationAnalysisError, match="requires at least 4"):
        analyze_calibration(_capture(bad_readout_site=0), _request())

    capture = _capture(invalid_sample=False)
    request = _request()
    brackets = request.layout.brackets(capture.block.schema)
    partition = analysis_impl._freeze_partition(brackets, request)
    values = capture.block.values.copy()
    repeat_axis = capture.block.schema.repeat_axis.axis_id
    for bracket_index in partition.test_indices:
        bracket = brackets[bracket_index]
        repeat = dict(bracket.context_key)[repeat_axis]
        frame = values[repeat, bracket.readout_point_storage_row]
        for x, y in _CENTERS:
            _paint_spot(frame, x, y, 16.0)
    with pytest.raises(CalibrationAnalysisError, match="admitted 0/4"):
        analyze_calibration(_replace_block(capture, values), request)


def test_numeric_package_versions_are_bounded_passive_artifact_notes(monkeypatch):
    capture = _capture(invalid_sample=False)
    request = _request()
    plan = build_calibration_work_plan(capture, request)
    result = analyze_calibration(capture, request)
    parameters = {item.name: item.value for item in result.artifact.parameters}
    assert "numeric-backend-digest" not in parameters
    assert parameters["numpy-version"] == distribution_version("numpy")
    assert parameters["scipy-version"] == distribution_version("scipy")
    assert all(
        len(str(parameters[name])) <= 64
        for name in ("numpy-version", "scipy-version")
    )
    assert all(
        "numpy-version" not in {item.name for item in model.header.parameters}
        and "scipy-version" not in {item.name for item in model.header.parameters}
        for model in result.artifact.models
    )

    monkeypatch.setattr(analysis_impl.np, "__version__", "numpy/" + "x" * 200)
    monkeypatch.setattr(analysis_impl.scipy, "__version__", "scipy path " + "y" * 200)
    assert build_calibration_work_plan(capture, request) == plan
    changed_notes = analyze_calibration(capture, request)
    assert changed_notes.diagnostics == result.diagnostics
    assert (
        changed_notes.artifact.site_map.fingerprint
        == result.artifact.site_map.fingerprint
    )
    assert tuple(map(encode_readout_model, changed_notes.artifact.models)) == tuple(
        map(encode_readout_model, result.artifact.models)
    )
    assert changed_notes.artifact.fingerprint != result.artifact.fingerprint

    without_notes = replace(
        result.artifact,
        parameters=tuple(
            item
            for item in result.artifact.parameters
            if item.name not in {"numpy-version", "scipy-version"}
        ),
    )
    stripped = CalibrationAnalysisResult(without_notes, result.diagnostics)
    assert validate_calibration_analysis_contract(stripped, request, plan) is stripped


def test_analysis_result_rejects_diagnostic_lineage_and_evidence_drift():
    result = analyze_calibration(_capture(invalid_sample=False), _request())
    diagnostics = result.diagnostics

    with pytest.raises(ValueError, match="reference-frame"):
        analysis_impl.CalibrationAnalysisResult(
            result.artifact,
            replace(diagnostics, reference_frame_count=diagnostics.reference_frame_count + 1),
        )
    with pytest.raises(ValueError, match="partition lineage"):
        analysis_impl.CalibrationAnalysisResult(
            result.artifact,
            replace(diagnostics, partition_digest="0" * 64),
        )
    with pytest.raises(ValueError, match="site-detection"):
        analysis_impl.CalibrationAnalysisResult(
            result.artifact,
            replace(
                diagnostics,
                detection=replace(
                    diagnostics.detection,
                    candidate_count=diagnostics.detection.candidate_count + 1,
                ),
            ),
        )
    first = diagnostics.models[0]
    drifted = replace(
        first,
        minimum_fidelity=0.99 * first.minimum_fidelity,
        mean_fidelity=0.99 * first.mean_fidelity,
    )
    with pytest.raises(ValueError, match="evidence diagnostic"):
        analysis_impl.CalibrationAnalysisResult(
            result.artifact,
            replace(diagnostics, models=(drifted, *diagnostics.models[1:])),
        )

    forged_models = []
    forged_diagnostics = []
    for model, diagnostic in zip(
        result.artifact.models,
        diagnostics.models,
        strict=True,
    ):
        quality = model.header.quality
        forged_quality = replace(
            quality,
            held_out_dark_accuracy_lower_bounds=np.full(
                quality.held_out_dark_accuracy_lower_bounds.shape,
                0.99,
                dtype="<f8",
            ),
            held_out_bright_accuracy_lower_bounds=np.full(
                quality.held_out_bright_accuracy_lower_bounds.shape,
                0.99,
                dtype="<f8",
            ),
        )
        forged_models.append(
            replace(model, header=replace(model.header, quality=forged_quality))
        )
        forged_diagnostics.append(
            replace(
                diagnostic,
                minimum_class_accuracy_lower_bound=0.99,
                mean_class_accuracy_lower_bound=0.99,
            )
        )
    with pytest.raises(ValueError, match="Clopper-Pearson"):
        analysis_impl.CalibrationAnalysisResult(
            replace(result.artifact, models=tuple(forged_models)),
            replace(diagnostics, models=tuple(forged_diagnostics)),
        )

    first_model = result.artifact.models[0]
    drifted_parameters = tuple(
        replace(parameter, value=parameter.value - 1)
        if parameter.name == "minimum-train-samples-per-class"
        else parameter
        for parameter in first_model.header.parameters
    )
    policy_drifted_model = replace(
        first_model,
        header=replace(first_model.header, parameters=drifted_parameters),
    )
    with pytest.raises(ValueError, match="inconsistent calibration gate policies"):
        analysis_impl.CalibrationAnalysisResult(
            replace(
                result.artifact,
                models=(policy_drifted_model, *result.artifact.models[1:]),
            ),
            diagnostics,
        )
