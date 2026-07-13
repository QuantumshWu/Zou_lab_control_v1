from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

from zlc_data import AxisId
from zlc_neutral_atom.readout.analysis import (
    BoxAnalysisConfig,
    CalibrationAnalysisDiagnostics,
    CalibrationAnalysisPlanningAssumption,
    CalibrationAnalysisRequest,
    CalibrationAnalysisResourcePolicy,
    CalibrationBracketSamplingAssumption,
    CalibrationWorkPlan,
    GridOrder,
    ModelAnalysisDiagnostic,
    PsfAnalysisConfig,
    ReferenceClassOrientation,
    ReferenceLabelSource,
    ReferenceValleyDiagnostic,
    ReferenceValleyEvidence,
    SiteDetectionDiagnostic,
    SiteDetectionPolicy,
    UsableSiteAcceptance,
)
import zlc_neutral_atom.readout.analysis_codec as analysis_codec
from zlc_neutral_atom.readout.analysis_codec import (
    CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
    CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
    CALIBRATION_WORK_PLAN_SCHEMA,
    REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA,
    CalibrationAnalysisCodecError,
    calibration_analysis_diagnostics_encoding_upper_bound,
    calibration_analysis_diagnostics_to_tree,
    calibration_analysis_request_to_tree,
    calibration_work_plan_to_tree,
    decode_calibration_analysis_diagnostics,
    decode_calibration_analysis_request,
    decode_calibration_work_plan,
    encode_calibration_analysis_diagnostics,
    encode_calibration_analysis_request,
    encode_calibration_work_plan,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    ReadoutModelKind,
)
from zlc_neutral_atom.readout.codec import calibration_capture_layout_to_tree
from zlc_neutral_atom.readout.contracts import CalibrationCaptureLayout
from zlc_storage import CanonicalEncodingError, decode, encode


def _resource_policy(*, max_sites: int = 64) -> CalibrationAnalysisResourcePolicy:
    return CalibrationAnalysisResourcePolicy(
        artifact_policy=CalibrationResourcePolicy(
            max_manifest_bytes=32_768,
            max_artifact_blob_bytes=8 * 1024 * 1024,
            max_models=3,
            max_sites=max_sites,
            max_kernel_elements=100_000,
            max_sampled_pixels_per_model=200_000,
            max_total_sampled_pixels_all_models=400_000,
        ),
        max_source_cells=10_000,
        max_brackets=1_000,
        max_reference_frames=2_000,
        max_image_pixels=100_000,
        max_signal_evaluations=1_000_000,
        max_modality_test_work_units=2_000_000,
        max_reference_valley_diagnostics=2_000,
        max_sampled_pixel_operations=2_000_000,
        max_working_bytes=4_000_000,
        max_lattice_sites=max_sites,
        max_detector_graph_work_units=8_000_000,
        max_dense_assignment_work_units=16_000_000,
    )


def _request() -> CalibrationAnalysisRequest:
    return CalibrationAnalysisRequest(
        layout=CalibrationCaptureLayout(
            AxisId("readout-event"),
            (4, 0, 2),
            3,
        ),
        grid_shape_yx=(2, 3),
        reference_label_source=(
            ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY
        ),
        reference_class_orientation=(
            ReferenceClassOrientation.BELOW_IS_OCCUPIED
        ),
        bracket_sampling_assumption=(
            CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS
        ),
        analysis_planning_assumption=(
            CalibrationAnalysisPlanningAssumption.PRECOMMITTED_BEFORE_SOURCE_INSPECTION
        ),
        grid_order=GridOrder.COLUMN_MAJOR,
        box=BoxAnalysisConfig(3, BoxReducer.MEAN),
        model_kinds=(
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
        ),
        default_model_kind=ReadoutModelKind.PER_SITE_PSF,
        psf=PsfAnalysisConfig(4, BackgroundMode.ANNULUS_MEDIAN, 2),
        detection=SiteDetectionPolicy(
            smoothing_sigma_pixels=0.5,
            minimum_prominence_fraction=0.2,
            minimum_peak_separation_pixels=2.5,
            minimum_half_prominence_basin_area_pixels=3,
            reject_touching_half_prominence_basins=False,
            maximum_lattice_rms_residual_pixels=0.6,
            minimum_lattice_step_pixels=2.2,
            minimum_band_separation_pixels=0.8,
            minimum_affine_sin_angle=0.3,
            maximum_affine_condition_number=8.0,
            minimum_assignment_cost_gap_pixels_squared=5.0,
        ),
        train_fraction=0.30,
        reference_evidence_fraction=0.30,
        minimum_train_samples_per_class=5,
        minimum_test_samples_per_class=6,
        held_out_confidence_level=0.975,
        minimum_held_out_class_accuracy_lower_bound=0.7,
        usable_site_acceptance=UsableSiteAcceptance.MINIMUM_FRACTION,
        minimum_usable_site_fraction=0.8,
        resource_policy=_resource_policy(),
    )


def _work_plan() -> CalibrationWorkPlan:
    return CalibrationWorkPlan(
        source_cell_count=60,
        bracket_upper_bound=20,
        train_bracket_upper_bound=6,
        reference_evidence_bracket_upper_bound=6,
        reference_frame_upper_bound=60,
        image_pixel_count=4_096,
        full_frame_read_count=120,
        feature_pixel_operations=200_000,
        signal_evaluations=50_000,
        modality_test_work_units=60_000,
        reference_valley_diagnostic_count=18,
        diagnostics_encoding_upper_bound_bytes=8_000,
        planned_kernel_elements=1_000,
        maximum_model_sampled_pixels=4_000,
        total_model_sampled_pixels=9_000,
        artifact_metadata_encoding_upper_bound_bytes=100_000,
        artifact_encoding_upper_bound_bytes=2_000_000,
        layout_working_bytes=10_000,
        detector_working_bytes=20_000,
        assignment_scratch_bytes=5_000,
        feature_working_bytes=30_000,
        psf_working_bytes=40_000,
        artifact_array_bytes=20_000,
        canonical_encoding_scratch_bytes=3_000_000,
        working_peak_bytes=3_100_000,
        detector_graph_work_units=80_000,
        dense_assignment_work_units=90_000,
    )


def _reference_valleys(
    reference_count: int,
    site_count: int,
    evidence_count: int = 2,
) -> tuple[ReferenceValleyDiagnostic, ...]:
    return tuple(
        ReferenceValleyDiagnostic(
            reference_index=reference,
            site_index=site,
            proposal_threshold=None,
            proposal_lower_sample_count=0,
            proposal_upper_sample_count=0,
            cluster_separation_rss=None,
            evidence=ReferenceValleyEvidence(
                evidence_count,
                0,
                0,
                0,
                evidence_count,
                0,
            ),
            lower_cluster_evidence=None,
            upper_cluster_evidence=None,
            site_accepted=False,
        )
        for reference in range(reference_count)
        for site in range(site_count)
    )


def _diagnostics() -> CalibrationAnalysisDiagnostics:
    return CalibrationAnalysisDiagnostics(
        bracket_count=6,
        train_bracket_count=2,
        reference_evidence_bracket_count=2,
        test_bracket_count=2,
        partition_digest="a" * 64,
        reference_frame_count=18,
        valid_training_reference_pixel_fraction=0.875,
        consensus_dark_counts=(1, 2, 3),
        consensus_bright_counts=(4, 3, 2),
        reference_valleys=_reference_valleys(3, 3),
        detection=SiteDetectionDiagnostic(
            candidate_count=3,
            minimum_peak_to_saddle_prominence=12.5,
            minimum_half_prominence_basin_area_pixels=7,
            lattice_rms_residual_pixels=0.2,
            minimum_band_separation_pixels=2.0,
            affine_sin_angle=0.8,
            affine_condition_number=1.25,
            assignment_cost_gap_pixels_squared=6.0,
        ),
        models=(
            ModelAnalysisDiagnostic(
                ReadoutModelKind.BOX,
                3,
                0,
                0.8,
                0.9,
                0.7,
                0.85,
            ),
            ModelAnalysisDiagnostic(
                ReadoutModelKind.PER_SITE_PSF,
                2,
                1,
                0.82,
                0.92,
                0.72,
                0.87,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("value", "encoder", "decoder", "schema"),
    (
        (
            _request(),
            encode_calibration_analysis_request,
            decode_calibration_analysis_request,
            CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
        ),
        (
            _work_plan(),
            encode_calibration_work_plan,
            decode_calibration_work_plan,
            CALIBRATION_WORK_PLAN_SCHEMA,
        ),
        (
            _diagnostics(),
            encode_calibration_analysis_diagnostics,
            decode_calibration_analysis_diagnostics,
            CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
        ),
    ),
)
def test_current_values_round_trip_to_deterministic_canonical_bytes(
    value,
    encoder,
    decoder,
    schema,
):
    first = encoder(value)
    second = encoder(value)

    assert first == second
    assert decode(first)["schema"] == schema
    restored = decoder(bytearray(first))
    assert restored == value
    assert encoder(restored) == first


def test_work_plan_rejects_wire_only_canonical_scratch_budget():
    plan = _work_plan()
    artifact_array_bytes = 32 * 1024 * 1024
    wire_only_scratch = 2 * artifact_array_bytes

    with pytest.raises(ValueError, match="owner encoding working bound"):
        replace(
            plan,
            artifact_array_bytes=artifact_array_bytes,
            artifact_encoding_upper_bound_bytes=wire_only_scratch,
            canonical_encoding_scratch_bytes=wire_only_scratch,
            working_peak_bytes=artifact_array_bytes + wire_only_scratch,
        )


def test_request_delegates_layout_to_its_owner_codec(monkeypatch):
    request = _request()
    expected = calibration_capture_layout_to_tree(request.layout)
    assert calibration_analysis_request_to_tree(request)["layout"] == expected

    to_calls = 0
    from_calls = 0
    original_to = analysis_codec.calibration_capture_layout_to_tree
    original_from = analysis_codec.calibration_capture_layout_from_tree

    def counted_to(value):
        nonlocal to_calls
        to_calls += 1
        return original_to(value)

    def counted_from(tree):
        nonlocal from_calls
        from_calls += 1
        return original_from(tree)

    monkeypatch.setattr(analysis_codec, "calibration_capture_layout_to_tree", counted_to)
    monkeypatch.setattr(analysis_codec, "calibration_capture_layout_from_tree", counted_from)
    payload = encode_calibration_analysis_request(request)
    assert decode_calibration_analysis_request(payload) == request
    assert to_calls >= 2
    assert from_calls == 1


@pytest.mark.parametrize(
    ("tree_factory", "path"),
    (
        (lambda: decode(encode_calibration_analysis_request(_request())), ()),
        (lambda: decode(encode_calibration_analysis_request(_request())), ("box",)),
        (lambda: decode(encode_calibration_analysis_request(_request())), ("psf",)),
        (
            lambda: decode(encode_calibration_analysis_request(_request())),
            ("detection",),
        ),
        (
            lambda: decode(encode_calibration_analysis_request(_request())),
            ("resource_policy",),
        ),
        (
            lambda: decode(encode_calibration_analysis_request(_request())),
            ("resource_policy", "artifact_policy"),
        ),
        (lambda: decode(encode_calibration_work_plan(_work_plan())), ()),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            (),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("detection",),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("models", 0),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("reference_valleys", 0),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("reference_valleys", 0, "evidence"),
        ),
    ),
)
def test_every_current_schema_rejects_unknown_and_missing_fields(tree_factory, path):
    tree = tree_factory()
    target = tree
    for part in path:
        target = target[part]
    removable = next(key for key in target if key != "schema")

    missing = deepcopy(tree)
    missing_target = missing
    for part in path:
        missing_target = missing_target[part]
    del missing_target[removable]

    unknown = deepcopy(tree)
    unknown_target = unknown
    for part in path:
        unknown_target = unknown_target[part]
    unknown_target["future_field"] = 1

    decoder = (
        decode_calibration_work_plan
        if tree["schema"] == CALIBRATION_WORK_PLAN_SCHEMA
        else decode_calibration_analysis_diagnostics
        if tree["schema"] == CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA
        else decode_calibration_analysis_request
    )
    with pytest.raises(ValueError, match="exactly|unexpected list"):
        decoder(encode(missing))
    with pytest.raises(ValueError, match="exactly|unexpected list"):
        decoder(encode(unknown))


@pytest.mark.parametrize(
    ("tree_factory", "schema_path"),
    (
        (lambda: decode(encode_calibration_analysis_request(_request())), ()),
        (lambda: decode(encode_calibration_analysis_request(_request())), ("box",)),
        (lambda: decode(encode_calibration_analysis_request(_request())), ("psf",)),
        (
            lambda: decode(encode_calibration_analysis_request(_request())),
            ("detection",),
        ),
        (
            lambda: decode(encode_calibration_analysis_request(_request())),
            ("resource_policy",),
        ),
        (
            lambda: decode(encode_calibration_analysis_request(_request())),
            ("resource_policy", "artifact_policy"),
        ),
        (lambda: decode(encode_calibration_work_plan(_work_plan())), ()),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            (),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("detection",),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("models", 0),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("reference_valleys", 0),
        ),
        (
            lambda: decode(encode_calibration_analysis_diagnostics(_diagnostics())),
            ("reference_valleys", 0, "evidence"),
        ),
    ),
)
def test_every_schema_rejects_an_unknown_version(tree_factory, schema_path):
    tree = tree_factory()
    target = tree
    for part in schema_path:
        target = target[part]
    target["schema"] += ".future"
    decoder = (
        decode_calibration_work_plan
        if tree["schema"].startswith(CALIBRATION_WORK_PLAN_SCHEMA)
        else decode_calibration_analysis_diagnostics
        if tree["schema"].startswith(CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA)
        else decode_calibration_analysis_request
    )
    with pytest.raises(ValueError, match="expected schema"):
        decoder(encode(tree))


def test_request_rejects_noncanonical_sequence_order_and_scalar_spelling():
    tree = decode(encode_calibration_analysis_request(_request()))
    tree["model_kinds"].reverse()
    with pytest.raises(CalibrationAnalysisCodecError, match="non-canonical"):
        decode_calibration_analysis_request(encode(tree))

    tree = decode(encode_calibration_analysis_request(_request()))
    tree["layout"]["reference_event_indices"].reverse()
    with pytest.raises(ValueError, match="non-canonical"):
        decode_calibration_analysis_request(encode(tree))

    tree = decode(encode_calibration_analysis_request(_request()))
    tree["train_fraction"] = 1
    with pytest.raises(ValueError, match="canonical float"):
        decode_calibration_analysis_request(encode(tree))

    tree = decode(encode_calibration_analysis_request(_request()))
    tree["detection"]["smoothing_sigma_pixels"] = -0.0
    with pytest.raises(CalibrationAnalysisCodecError, match="non-canonical"):
        decode_calibration_analysis_request(encode(tree))


def test_request_rejects_unknown_analysis_planning_assumption():
    tree = decode(encode_calibration_analysis_request(_request()))
    tree["analysis_planning_assumption"] = "POST_HOC"

    with pytest.raises(ValueError, match="unknown value"):
        decode_calibration_analysis_request(encode(tree))


def test_work_plan_and_diagnostics_reject_bool_integer_and_nonfinite_float():
    plan = decode(encode_calibration_work_plan(_work_plan()))
    plan["source_cell_count"] = True
    with pytest.raises(ValueError, match="canonical integer"):
        decode_calibration_work_plan(encode(plan))

    diagnostics = decode(encode_calibration_analysis_diagnostics(_diagnostics()))
    diagnostics["valid_training_reference_pixel_fraction"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decode_calibration_analysis_diagnostics(encode(diagnostics))


def test_present_infinite_reference_separation_round_trips_and_legacy_schemas_reject():
    evidence = ReferenceValleyEvidence(2, 1, 0, 1, 0, 0)
    present = ReferenceValleyDiagnostic(
        reference_index=0,
        site_index=0,
        proposal_threshold=0.5,
        proposal_lower_sample_count=1,
        proposal_upper_sample_count=1,
        cluster_separation_rss=math.inf,
        evidence=evidence,
        lower_cluster_evidence=ReferenceValleyEvidence(1, 1, 0, 0, 0, 0),
        upper_cluster_evidence=ReferenceValleyEvidence(1, 0, 0, 1, 0, 0),
        site_accepted=False,
    )
    diagnostics = replace(
        _diagnostics(),
        reference_valleys=(present,) + _diagnostics().reference_valleys[1:],
    )
    payload = encode_calibration_analysis_diagnostics(diagnostics)
    restored = decode_calibration_analysis_diagnostics(payload)
    assert restored == diagnostics
    assert math.isinf(restored.reference_valleys[0].cluster_separation_rss)

    tree = decode(payload)
    missing_separation = deepcopy(tree)
    missing_separation["reference_valleys"][0]["cluster_separation_rss"] = None
    with pytest.raises(ValueError, match="requires cluster_separation_rss"):
        decode_calibration_analysis_diagnostics(encode(missing_separation))

    absent_proposal = deepcopy(tree)
    absent_proposal["reference_valleys"][0]["proposal_threshold"] = None
    absent_proposal["reference_valleys"][0]["proposal_lower_sample_count"] = 0
    absent_proposal["reference_valleys"][0]["proposal_upper_sample_count"] = 0
    with pytest.raises(ValueError, match="missing proposal"):
        decode_calibration_analysis_diagnostics(encode(absent_proposal))

    legacy_nested = deepcopy(tree)
    legacy_nested["reference_valleys"][0]["schema"] = (
        "zlc_neutral_atom.reference-valley-diagnostic.v2"
    )
    with pytest.raises(ValueError, match="expected schema"):
        decode_calibration_analysis_diagnostics(encode(legacy_nested))

    legacy_outer = deepcopy(tree)
    legacy_outer["schema"] = "zlc_neutral_atom.calibration-analysis-diagnostics.v3"
    with pytest.raises(ValueError, match="expected schema"):
        decode_calibration_analysis_diagnostics(encode(legacy_outer))

    assert tree["reference_valleys"][0]["schema"] == (
        REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA
    )


def test_diagnostics_wire_estimator_covers_maximal_optional_valley_shape():
    evidence = ReferenceValleyEvidence(2, 1, 0, 1, 0, 0)
    valleys = tuple(
        ReferenceValleyDiagnostic(
            reference,
            site,
            0.5,
            1,
            1,
            math.inf,
            evidence,
            ReferenceValleyEvidence(1, 1, 0, 0, 0, 0),
            ReferenceValleyEvidence(1, 0, 0, 1, 0, 0),
            False,
        )
        for reference in range(3)
        for site in range(3)
    )
    diagnostics = replace(_diagnostics(), reference_valleys=valleys)
    payload = encode_calibration_analysis_diagnostics(diagnostics)
    upper = calibration_analysis_diagnostics_encoding_upper_bound(
        site_count=3,
        reference_count=3,
        bracket_upper_bound=6,
        train_bracket_upper_bound=2,
        reference_evidence_bracket_upper_bound=2,
        model_count=2,
    )
    assert len(payload) <= upper


def test_closed_enums_reject_type_names_and_future_members():
    tree = decode(encode_calibration_analysis_request(_request()))
    tree["grid_order"] = "zlc_neutral_atom.readout.analysis.GridOrder.ROW_MAJOR"
    with pytest.raises(ValueError, match="unknown value"):
        decode_calibration_analysis_request(encode(tree))

    tree = decode(encode_calibration_analysis_request(_request()))
    tree["model_kinds"][0] = "FUTURE_MODEL"
    with pytest.raises(ValueError, match="unknown value"):
        decode_calibration_analysis_request(encode(tree))

    tree = decode(encode_calibration_analysis_request(_request()))
    tree["reference_label_source"] = "KNOWN_OCCUPIED"
    with pytest.raises(ValueError, match="unknown value"):
        decode_calibration_analysis_request(encode(tree))

    with pytest.raises(CanonicalEncodingError, match="unsupported canonical value"):
        encode({"callable": _request})


def test_constructors_and_decoders_isolate_caller_owned_sequences():
    references = [0, 2]
    model_kinds = [ReadoutModelKind.BOX]
    layout = CalibrationCaptureLayout(AxisId("event"), references, 1)
    request = CalibrationAnalysisRequest(
        layout,
        [1, 1],
        ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY,
        ReferenceClassOrientation.ABOVE_IS_OCCUPIED,
        CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS,
        CalibrationAnalysisPlanningAssumption.PRECOMMITTED_BEFORE_SOURCE_INSPECTION,
        model_kinds=model_kinds,
        default_model_kind=ReadoutModelKind.BOX,
    )
    references.append(4)
    model_kinds.append(ReadoutModelKind.UNIFORM_PSF)
    assert request.layout.reference_event_indices == (0, 2)
    assert request.model_kinds == (ReadoutModelKind.BOX,)

    dark = [1, 2]
    bright = [2, 1]
    models = [
        ModelAnalysisDiagnostic(ReadoutModelKind.BOX, 2, 0, 0.8, 0.9, 0.7, 0.8)
    ]
    diagnostics = CalibrationAnalysisDiagnostics(
        3,
        1,
        1,
        1,
        "b" * 64,
        6,
        1.0,
        dark,
        bright,
        _reference_valleys(2, 2, 1),
        SiteDetectionDiagnostic(2, 1.0, 1, 0.0, None, None, None, None),
        models,
    )
    dark.append(9)
    bright[0] = 99
    models.clear()
    assert diagnostics.consensus_dark_counts == (1, 2)
    assert diagnostics.consensus_bright_counts == (2, 1)
    assert len(diagnostics.models) == 1

    wire = bytearray(encode_calibration_analysis_diagnostics(diagnostics))
    restored = decode_calibration_analysis_diagnostics(wire)
    wire[:] = b"x" * len(wire)
    assert restored == diagnostics


def test_diagnostic_encode_applies_request_resource_policy_before_projection(
    monkeypatch,
):
    policy = _resource_policy(max_sites=2)
    diagnostics = _diagnostics()
    projected = False
    original = analysis_codec._site_detection_diagnostic_to_tree

    def counted(value):
        nonlocal projected
        projected = True
        return original(value)

    monkeypatch.setattr(analysis_codec, "_site_detection_diagnostic_to_tree", counted)
    with pytest.raises(CalibrationResourceExceeded, match="vectors"):
        encode_calibration_analysis_diagnostics(
            diagnostics,
            resource_policy=policy,
        )
    assert not projected


def test_diagnostic_decode_rejects_oversized_vector_before_domain_materialization(
    monkeypatch,
):
    policy = _resource_policy(max_sites=2)
    tree = calibration_analysis_diagnostics_to_tree(_diagnostics())
    parser_calls = 0

    def forbidden(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        pytest.fail("oversized vectors reached typed list materialization")

    monkeypatch.setattr(
        analysis_codec,
        "calibration_analysis_diagnostics_from_tree",
        forbidden,
    )
    with pytest.raises(CalibrationResourceExceeded, match="resource policy"):
        decode_calibration_analysis_diagnostics(
            encode(tree),
            resource_policy=policy,
        )
    assert parser_calls == 0


def test_unknown_list_rejects_before_typed_exact_field_parser(monkeypatch):
    tree = calibration_analysis_diagnostics_to_tree(_diagnostics())
    tree["future_vector"] = [1, 2]
    parser_calls = 0

    def forbidden(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        pytest.fail("unknown list reached the typed parser")

    monkeypatch.setattr(
        analysis_codec,
        "calibration_analysis_diagnostics_from_tree",
        forbidden,
    )
    with pytest.raises(CalibrationResourceExceeded, match="unexpected list"):
        decode_calibration_analysis_diagnostics(encode(tree))
    assert parser_calls == 0


def test_ndarray_rejects_before_any_numpy_materialization(monkeypatch):
    tree = calibration_analysis_diagnostics_to_tree(_diagnostics())
    tree["consensus_dark_counts"] = np.arange(3, dtype="<u2")
    materializations = 0
    import zlc_storage.canonical as canonical

    original = canonical._decode_array

    def counted(payload, *, path):
        nonlocal materializations
        materializations += 1
        return original(payload, path=path)

    monkeypatch.setattr(canonical, "_decode_array", counted)
    with pytest.raises(CanonicalEncodingError, match="ndarray count"):
        decode_calibration_analysis_diagnostics(encode(tree))
    assert materializations == 0


def test_request_layout_budget_rejects_before_typed_parser(monkeypatch):
    tree = calibration_analysis_request_to_tree(_request())
    monkeypatch.setattr(analysis_codec, "MAX_LAYOUT_REFERENCE_EVENT_INDICES", 2)
    parser_calls = 0

    def forbidden(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        pytest.fail("oversized layout reached typed parser")

    monkeypatch.setattr(
        analysis_codec,
        "calibration_analysis_request_from_tree",
        forbidden,
    )
    with pytest.raises(CalibrationResourceExceeded, match="resource policy"):
        decode_calibration_analysis_request(encode(tree))
    assert parser_calls == 0


def test_byte_budgets_and_resource_policy_types_are_admitted_before_decode():
    with pytest.raises(CalibrationResourceExceeded, match="byte budget"):
        decode_calibration_work_plan(
            memoryview(b"x" * (analysis_codec.MAX_ANALYSIS_WORK_PLAN_BYTES + 1))
        )
    with pytest.raises(TypeError, match="bytes-like"):
        decode_calibration_analysis_request("not bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resource_policy"):
        decode_calibration_analysis_diagnostics(
            encode_calibration_analysis_diagnostics(_diagnostics()),
            resource_policy=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="resource_policy"):
        encode_calibration_analysis_diagnostics(
            _diagnostics(),
            resource_policy=object(),  # type: ignore[arg-type]
        )


def test_tree_projectors_are_closed_primitive_data_without_type_metadata():
    trees = (
        calibration_analysis_request_to_tree(_request()),
        calibration_work_plan_to_tree(_work_plan()),
        calibration_analysis_diagnostics_to_tree(_diagnostics()),
    )
    forbidden_keys = {"fqcn", "module", "qualname", "callable", "factory"}

    def walk(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert value is None or type(value) in {bool, int, float, str}

    for tree in trees:
        walk(tree)
