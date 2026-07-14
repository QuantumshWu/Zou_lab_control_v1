"""Strict current-format codecs for readout-calibration analysis evidence.

The calibration repository persists these values inside one derivation blob.  This
module is their sole serialization owner.  It deliberately stores closed scalar
schemas and enum values; it never stores Python type names, callables, or import
paths.  ``CalibrationCaptureLayout`` remains owned by :mod:`.codec` and is always
projected and parsed through that owner's tree codec.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalListEvent,
    canonical_text as _text,
    decode,
    encode,
    exact_mapping as _exact_map,
    integer as _integer,
)
from zlc_storage.canonical import CanonicalDecodeLimits

from .analysis import (
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
from .calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    ReadoutModelKind,
)
from .codec import (
    calibration_capture_layout_from_tree,
    calibration_capture_layout_to_tree,
)


CALIBRATION_ANALYSIS_REQUEST_SCHEMA = (
    "zlc_neutral_atom.calibration-analysis-request"
)
CALIBRATION_WORK_PLAN_SCHEMA = "zlc_neutral_atom.calibration-work-plan"
CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA = (
    "zlc_neutral_atom.calibration-analysis-diagnostics"
)

BOX_ANALYSIS_CONFIG_SCHEMA = "zlc_neutral_atom.box-analysis-config"
PSF_ANALYSIS_CONFIG_SCHEMA = "zlc_neutral_atom.psf-analysis-config"
SITE_DETECTION_POLICY_SCHEMA = "zlc_neutral_atom.site-detection-policy"
CALIBRATION_RESOURCE_POLICY_SCHEMA = "zlc_neutral_atom.calibration-resource-policy"
CALIBRATION_ANALYSIS_RESOURCE_POLICY_SCHEMA = (
    "zlc_neutral_atom.calibration-analysis-resource-policy"
)
SITE_DETECTION_DIAGNOSTIC_SCHEMA = (
    "zlc_neutral_atom.site-detection-diagnostic"
)
MODEL_ANALYSIS_DIAGNOSTIC_SCHEMA = (
    "zlc_neutral_atom.model-analysis-diagnostic"
)
REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA = (
    "zlc_neutral_atom.reference-valley-diagnostic"
)
REFERENCE_VALLEY_EVIDENCE_SCHEMA = (
    "zlc_neutral_atom.reference-valley-evidence"
)


class CalibrationAnalysisCodecError(ValueError):
    """A payload does not have one canonical current analysis meaning."""


# These are wire/object-amplification ceilings, not scientific analysis limits.
# Scientific limits remain encoded in CalibrationAnalysisResourcePolicy.  The
# codec ceilings keep hostile current-format payloads bounded even when they
# claim an arbitrarily permissive embedded policy.
MAX_ANALYSIS_REQUEST_BYTES = 16 * 1024 * 1024
MAX_ANALYSIS_WORK_PLAN_BYTES = 64 * 1024
MAX_ANALYSIS_DIAGNOSTICS_BYTES = 64 * 1024 * 1024
MAX_LAYOUT_REFERENCE_EVENT_INDICES = 200_000
MAX_DIAGNOSTIC_VECTOR_ENTRIES = 100_000


def calibration_analysis_diagnostics_encoding_upper_bound(
    *,
    site_count: int,
    reference_count: int,
    bracket_upper_bound: int,
    train_bracket_upper_bound: int,
    reference_evidence_bracket_upper_bound: int,
    model_count: int,
) -> int:
    """Conservative current-schema canonical wire bound without materializing it.

    This codec-owned estimate follows the current diagnostics projection.  It
    budgets the largest optional reference-valley form (proposal plus both
    nested evidence records), fixed JSON/tag/key overhead, all scalar digit
    widths, site vectors, model diagnostics, and a generous top-level envelope.
    Tests compare boundary-shaped real encodings against this formula so a
    future schema change cannot silently invalidate preflight.
    """

    values = {
        "site_count": site_count,
        "reference_count": reference_count,
        "bracket_upper_bound": bracket_upper_bound,
        "train_bracket_upper_bound": train_bracket_upper_bound,
        "reference_evidence_bracket_upper_bound": (
            reference_evidence_bracket_upper_bound
        ),
        "model_count": model_count,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    def digits(value: int) -> int:
        # ceil(bit_length * log10(2)); this is an allocation-free upper bound
        # and remains defined beyond Python's decimal-string safety limit.
        return max(
            1,
            (max(1, value).bit_length() * 30_103 + 99_999) // 100_000,
        )

    valley_count = site_count * reference_count
    per_valley = (
        2_048
        + digits(reference_count - 1)
        + digits(site_count - 1)
        + 2 * digits(train_bracket_upper_bound)
        + 18 * digits(reference_evidence_bracket_upper_bound)
    )
    return (
        64 * 1024
        + 2 * site_count * (32 + digits(bracket_upper_bound))
        + model_count * 2_048
        + valley_count * per_valley
    )


def calibration_analysis_diagnostics_encoding_working_upper_bound(
    wire_upper_bound: int,
) -> int:
    """Bound the current non-streaming projection/tag/JSON encode phase.

    The encoder simultaneously retains the domain graph, primitive projection,
    canonical tagged tree, and final bytes.  Measurements of the maximal current
    valley shape exceed eight times payload size; twelve times the conservative
    wire bound plus a fixed envelope leaves headroom across Python allocators.
    """

    if isinstance(wire_upper_bound, bool) or not isinstance(
        wire_upper_bound, int
    ):
        raise TypeError("wire_upper_bound must be an integer")
    if wire_upper_bound < 0:
        raise ValueError("wire_upper_bound must be non-negative")
    return 12 * wire_upper_bound + 1024 * 1024


_DEFAULT_ANALYSIS_RESOURCE_POLICY = CalibrationAnalysisResourcePolicy()
_MODEL_KIND_COUNT = len(ReadoutModelKind)
T = TypeVar("T")


def _exact_nested_map(tree: Any, fields: set[str], schema: str) -> dict[str, Any]:
    return _exact_map(tree, fields | {"schema"}, schema)


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a canonical list")
    return value


def _float(value: Any, field: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{field} must use the canonical float representation")
    return value


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be bool")
    return value


def _enum(enum_type, value: Any, field: str):
    try:
        return enum_type(_text(value, field))
    except ValueError as exc:
        raise ValueError(f"{field} has an unknown value {value!r}") from exc


def _canonical_tree(original: Any, projected: Any, schema: str) -> None:
    if encode(original) != encode(projected):
        raise CalibrationAnalysisCodecError(
            f"{schema} tree is typed but non-canonical"
        )


def _checked_payload(
    payload: bytes | bytearray | memoryview,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} payload must be bytes-like")
    size = payload.nbytes if isinstance(payload, memoryview) else len(payload)
    if size > max_bytes:
        raise CalibrationResourceExceeded(f"{label} payload exceeds codec byte budget")
    return bytes(payload)


def _encode_checked(tree: dict[str, Any], *, max_bytes: int, label: str) -> bytes:
    payload = encode(tree)
    if len(payload) > max_bytes:
        raise CalibrationResourceExceeded(f"{label} payload exceeds codec byte budget")
    return payload


def _decode_typed(
    payload: bytes | bytearray | memoryview,
    *,
    parser: Callable[[Any], T],
    projector: Callable[[T], dict[str, Any]],
    schema: str,
    max_bytes: int,
    limits: CanonicalDecodeLimits,
    admit_structure: Callable[[tuple[CanonicalListEvent | CanonicalArrayEvent, ...]], None],
) -> T:
    raw = _checked_payload(payload, max_bytes=max_bytes, label=schema)
    value = parser(
        decode(raw, limits=limits, admit_structure=admit_structure)
    )
    if encode(projector(value)) != raw:
        raise CalibrationAnalysisCodecError(
            f"{schema} payload uses a non-canonical typed representation"
        )
    return value


def _reject_unexpected_structure(
    events: tuple[CanonicalListEvent | CanonicalArrayEvent, ...],
    *,
    allowed_lists: dict[tuple[str | int, ...], int],
    label: str,
) -> None:
    for event in events:
        if isinstance(event, CanonicalArrayEvent):
            raise CalibrationResourceExceeded(f"{label} forbids ndarray fields")
        maximum = allowed_lists.get(event.path)
        if maximum is None:
            raise CalibrationResourceExceeded(
                f"{label} contains an unexpected list at {event.path!r}"
            )
        if event.length > maximum:
            raise CalibrationResourceExceeded(
                f"{label} list at {event.path!r} exceeds codec resource policy"
            )


def _box_config_to_tree(value: BoxAnalysisConfig) -> dict[str, Any]:
    if not isinstance(value, BoxAnalysisConfig):
        raise TypeError("value must be BoxAnalysisConfig")
    return {
        "schema": BOX_ANALYSIS_CONFIG_SCHEMA,
        "half_width": value.half_width,
        "reducer": value.reducer.value,
    }


def _box_config_from_tree(tree: Any) -> BoxAnalysisConfig:
    data = _exact_nested_map(
        tree,
        {"half_width", "reducer"},
        BOX_ANALYSIS_CONFIG_SCHEMA,
    )
    value = BoxAnalysisConfig(
        _integer(data["half_width"], "box half_width"),
        _enum(BoxReducer, data["reducer"], "box reducer"),
    )
    _canonical_tree(tree, _box_config_to_tree(value), BOX_ANALYSIS_CONFIG_SCHEMA)
    return value


def _psf_config_to_tree(value: PsfAnalysisConfig) -> dict[str, Any]:
    if not isinstance(value, PsfAnalysisConfig):
        raise TypeError("value must be PsfAnalysisConfig")
    return {
        "schema": PSF_ANALYSIS_CONFIG_SCHEMA,
        "half_width": value.half_width,
        "background": value.background.value,
        "background_padding": value.background_padding,
    }


def _psf_config_from_tree(tree: Any) -> PsfAnalysisConfig:
    data = _exact_nested_map(
        tree,
        {"half_width", "background", "background_padding"},
        PSF_ANALYSIS_CONFIG_SCHEMA,
    )
    value = PsfAnalysisConfig(
        _integer(data["half_width"], "PSF half_width"),
        _enum(BackgroundMode, data["background"], "PSF background"),
        _integer(data["background_padding"], "PSF background_padding"),
    )
    _canonical_tree(tree, _psf_config_to_tree(value), PSF_ANALYSIS_CONFIG_SCHEMA)
    return value


def _site_detection_policy_to_tree(value: SiteDetectionPolicy) -> dict[str, Any]:
    if not isinstance(value, SiteDetectionPolicy):
        raise TypeError("value must be SiteDetectionPolicy")
    return {
        "schema": SITE_DETECTION_POLICY_SCHEMA,
        "smoothing_sigma_pixels": value.smoothing_sigma_pixels,
        "minimum_prominence_fraction": value.minimum_prominence_fraction,
        "minimum_peak_separation_pixels": value.minimum_peak_separation_pixels,
        "minimum_half_prominence_basin_area_pixels": (
            value.minimum_half_prominence_basin_area_pixels
        ),
        "reject_touching_half_prominence_basins": (
            value.reject_touching_half_prominence_basins
        ),
        "maximum_lattice_rms_residual_pixels": (
            value.maximum_lattice_rms_residual_pixels
        ),
        "minimum_lattice_step_pixels": value.minimum_lattice_step_pixels,
        "minimum_band_separation_pixels": value.minimum_band_separation_pixels,
        "minimum_affine_sin_angle": value.minimum_affine_sin_angle,
        "maximum_affine_condition_number": value.maximum_affine_condition_number,
        "minimum_assignment_cost_gap_pixels_squared": (
            value.minimum_assignment_cost_gap_pixels_squared
        ),
    }


def _site_detection_policy_from_tree(tree: Any) -> SiteDetectionPolicy:
    fields = {
        "smoothing_sigma_pixels",
        "minimum_prominence_fraction",
        "minimum_peak_separation_pixels",
        "minimum_half_prominence_basin_area_pixels",
        "reject_touching_half_prominence_basins",
        "maximum_lattice_rms_residual_pixels",
        "minimum_lattice_step_pixels",
        "minimum_band_separation_pixels",
        "minimum_affine_sin_angle",
        "maximum_affine_condition_number",
        "minimum_assignment_cost_gap_pixels_squared",
    }
    data = _exact_nested_map(tree, fields, SITE_DETECTION_POLICY_SCHEMA)
    value = SiteDetectionPolicy(
        smoothing_sigma_pixels=_float(
            data["smoothing_sigma_pixels"], "smoothing_sigma_pixels"
        ),
        minimum_prominence_fraction=_float(
            data["minimum_prominence_fraction"], "minimum_prominence_fraction"
        ),
        minimum_peak_separation_pixels=_float(
            data["minimum_peak_separation_pixels"],
            "minimum_peak_separation_pixels",
        ),
        minimum_half_prominence_basin_area_pixels=_integer(
            data["minimum_half_prominence_basin_area_pixels"],
            "minimum_half_prominence_basin_area_pixels",
        ),
        reject_touching_half_prominence_basins=_bool(
            data["reject_touching_half_prominence_basins"],
            "reject_touching_half_prominence_basins",
        ),
        maximum_lattice_rms_residual_pixels=_float(
            data["maximum_lattice_rms_residual_pixels"],
            "maximum_lattice_rms_residual_pixels",
        ),
        minimum_lattice_step_pixels=_float(
            data["minimum_lattice_step_pixels"], "minimum_lattice_step_pixels"
        ),
        minimum_band_separation_pixels=_float(
            data["minimum_band_separation_pixels"],
            "minimum_band_separation_pixels",
        ),
        minimum_affine_sin_angle=_float(
            data["minimum_affine_sin_angle"], "minimum_affine_sin_angle"
        ),
        maximum_affine_condition_number=_float(
            data["maximum_affine_condition_number"],
            "maximum_affine_condition_number",
        ),
        minimum_assignment_cost_gap_pixels_squared=_float(
            data["minimum_assignment_cost_gap_pixels_squared"],
            "minimum_assignment_cost_gap_pixels_squared",
        ),
    )
    _canonical_tree(
        tree,
        _site_detection_policy_to_tree(value),
        SITE_DETECTION_POLICY_SCHEMA,
    )
    return value


def _artifact_resource_policy_to_tree(
    value: CalibrationResourcePolicy,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationResourcePolicy):
        raise TypeError("value must be CalibrationResourcePolicy")
    return {
        "schema": CALIBRATION_RESOURCE_POLICY_SCHEMA,
        "max_manifest_bytes": value.max_manifest_bytes,
        "max_artifact_blob_bytes": value.max_artifact_blob_bytes,
        "max_models": value.max_models,
        "max_sites": value.max_sites,
        "max_kernel_elements": value.max_kernel_elements,
        "max_sampled_pixels_per_model": value.max_sampled_pixels_per_model,
        "max_total_sampled_pixels_all_models": (
            value.max_total_sampled_pixels_all_models
        ),
    }


def _artifact_resource_policy_from_tree(tree: Any) -> CalibrationResourcePolicy:
    fields = {
        "max_manifest_bytes",
        "max_artifact_blob_bytes",
        "max_models",
        "max_sites",
        "max_kernel_elements",
        "max_sampled_pixels_per_model",
        "max_total_sampled_pixels_all_models",
    }
    data = _exact_nested_map(tree, fields, CALIBRATION_RESOURCE_POLICY_SCHEMA)
    value = CalibrationResourcePolicy(
        **{name: _integer(data[name], name) for name in fields}
    )
    _canonical_tree(
        tree,
        _artifact_resource_policy_to_tree(value),
        CALIBRATION_RESOURCE_POLICY_SCHEMA,
    )
    return value


def _analysis_resource_policy_to_tree(
    value: CalibrationAnalysisResourcePolicy,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationAnalysisResourcePolicy):
        raise TypeError("value must be CalibrationAnalysisResourcePolicy")
    return {
        "schema": CALIBRATION_ANALYSIS_RESOURCE_POLICY_SCHEMA,
        "artifact_policy": _artifact_resource_policy_to_tree(value.artifact_policy),
        "max_source_cells": value.max_source_cells,
        "max_brackets": value.max_brackets,
        "max_reference_frames": value.max_reference_frames,
        "max_image_pixels": value.max_image_pixels,
        "max_signal_evaluations": value.max_signal_evaluations,
        "max_modality_test_work_units": value.max_modality_test_work_units,
        "max_reference_valley_diagnostics": (
            value.max_reference_valley_diagnostics
        ),
        "max_sampled_pixel_operations": value.max_sampled_pixel_operations,
        "max_working_bytes": value.max_working_bytes,
        "max_lattice_sites": value.max_lattice_sites,
        "max_detector_graph_work_units": value.max_detector_graph_work_units,
        "max_dense_assignment_work_units": value.max_dense_assignment_work_units,
    }


def _analysis_resource_policy_from_tree(
    tree: Any,
) -> CalibrationAnalysisResourcePolicy:
    fields = {
        "artifact_policy",
        "max_source_cells",
        "max_brackets",
        "max_reference_frames",
        "max_image_pixels",
        "max_signal_evaluations",
        "max_modality_test_work_units",
        "max_reference_valley_diagnostics",
        "max_sampled_pixel_operations",
        "max_working_bytes",
        "max_lattice_sites",
        "max_detector_graph_work_units",
        "max_dense_assignment_work_units",
    }
    data = _exact_nested_map(
        tree,
        fields,
        CALIBRATION_ANALYSIS_RESOURCE_POLICY_SCHEMA,
    )
    integer_fields = fields - {"artifact_policy"}
    value = CalibrationAnalysisResourcePolicy(
        artifact_policy=_artifact_resource_policy_from_tree(data["artifact_policy"]),
        **{name: _integer(data[name], name) for name in integer_fields},
    )
    _canonical_tree(
        tree,
        _analysis_resource_policy_to_tree(value),
        CALIBRATION_ANALYSIS_RESOURCE_POLICY_SCHEMA,
    )
    return value


def calibration_analysis_request_to_tree(
    value: CalibrationAnalysisRequest,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationAnalysisRequest):
        raise TypeError("value must be CalibrationAnalysisRequest")
    if len(value.layout.reference_event_indices) > MAX_LAYOUT_REFERENCE_EVENT_INDICES:
        raise CalibrationResourceExceeded(
            "calibration layout reference-event count exceeds codec resource policy"
        )
    return {
        "schema": CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
        "layout": calibration_capture_layout_to_tree(value.layout),
        "grid_shape_yx": list(value.grid_shape_yx),
        "reference_label_source": value.reference_label_source.value,
        "reference_class_orientation": value.reference_class_orientation.value,
        "bracket_sampling_assumption": value.bracket_sampling_assumption.value,
        "analysis_planning_assumption": value.analysis_planning_assumption.value,
        "grid_order": value.grid_order.value,
        "box": _box_config_to_tree(value.box),
        "model_kinds": [kind.value for kind in value.model_kinds],
        "default_model_kind": (
            None if value.default_model_kind is None else value.default_model_kind.value
        ),
        "psf": None if value.psf is None else _psf_config_to_tree(value.psf),
        "detection": _site_detection_policy_to_tree(value.detection),
        "train_fraction": value.train_fraction,
        "reference_evidence_fraction": value.reference_evidence_fraction,
        "minimum_train_samples_per_class": value.minimum_train_samples_per_class,
        "minimum_test_samples_per_class": value.minimum_test_samples_per_class,
        "minimum_reference_cluster_separation_rss": (
            value.minimum_reference_cluster_separation_rss
        ),
        "reference_valley_familywise_error_rate": (
            value.reference_valley_familywise_error_rate
        ),
        "held_out_confidence_level": value.held_out_confidence_level,
        "minimum_held_out_class_accuracy_lower_bound": (
            value.minimum_held_out_class_accuracy_lower_bound
        ),
        "usable_site_acceptance": value.usable_site_acceptance.value,
        "minimum_usable_site_fraction": value.minimum_usable_site_fraction,
        "resource_policy": _analysis_resource_policy_to_tree(value.resource_policy),
    }


def calibration_analysis_request_from_tree(tree: Any) -> CalibrationAnalysisRequest:
    fields = {
        "schema",
        "layout",
        "grid_shape_yx",
        "reference_label_source",
        "reference_class_orientation",
        "bracket_sampling_assumption",
        "analysis_planning_assumption",
        "grid_order",
        "box",
        "model_kinds",
        "default_model_kind",
        "psf",
        "detection",
        "train_fraction",
        "reference_evidence_fraction",
        "minimum_train_samples_per_class",
        "minimum_test_samples_per_class",
        "minimum_reference_cluster_separation_rss",
        "reference_valley_familywise_error_rate",
        "held_out_confidence_level",
        "minimum_held_out_class_accuracy_lower_bound",
        "usable_site_acceptance",
        "minimum_usable_site_fraction",
        "resource_policy",
    }
    data = _exact_map(tree, fields, CALIBRATION_ANALYSIS_REQUEST_SCHEMA)
    shape = _list(data["grid_shape_yx"], "grid_shape_yx")
    kinds = _list(data["model_kinds"], "model_kinds")
    default_kind = data["default_model_kind"]
    psf = data["psf"]
    value = CalibrationAnalysisRequest(
        layout=calibration_capture_layout_from_tree(data["layout"]),
        grid_shape_yx=tuple(
            _integer(item, "grid_shape_yx entry") for item in shape
        ),
        reference_label_source=_enum(
            ReferenceLabelSource,
            data["reference_label_source"],
            "reference_label_source",
        ),
        reference_class_orientation=_enum(
            ReferenceClassOrientation,
            data["reference_class_orientation"],
            "reference_class_orientation",
        ),
        bracket_sampling_assumption=_enum(
            CalibrationBracketSamplingAssumption,
            data["bracket_sampling_assumption"],
            "bracket_sampling_assumption",
        ),
        analysis_planning_assumption=_enum(
            CalibrationAnalysisPlanningAssumption,
            data["analysis_planning_assumption"],
            "analysis_planning_assumption",
        ),
        grid_order=_enum(GridOrder, data["grid_order"], "grid_order"),
        box=_box_config_from_tree(data["box"]),
        model_kinds=tuple(
            _enum(ReadoutModelKind, item, "model kind") for item in kinds
        ),
        default_model_kind=(
            None
            if default_kind is None
            else _enum(ReadoutModelKind, default_kind, "default_model_kind")
        ),
        psf=None if psf is None else _psf_config_from_tree(psf),
        detection=_site_detection_policy_from_tree(data["detection"]),
        train_fraction=_float(data["train_fraction"], "train_fraction"),
        reference_evidence_fraction=_float(
            data["reference_evidence_fraction"],
            "reference_evidence_fraction",
        ),
        minimum_train_samples_per_class=_integer(
            data["minimum_train_samples_per_class"],
            "minimum_train_samples_per_class",
        ),
        minimum_test_samples_per_class=_integer(
            data["minimum_test_samples_per_class"],
            "minimum_test_samples_per_class",
        ),
        minimum_reference_cluster_separation_rss=_float(
            data["minimum_reference_cluster_separation_rss"],
            "minimum_reference_cluster_separation_rss",
        ),
        reference_valley_familywise_error_rate=_float(
            data["reference_valley_familywise_error_rate"],
            "reference_valley_familywise_error_rate",
        ),
        held_out_confidence_level=_float(
            data["held_out_confidence_level"], "held_out_confidence_level"
        ),
        minimum_held_out_class_accuracy_lower_bound=_float(
            data["minimum_held_out_class_accuracy_lower_bound"],
            "minimum_held_out_class_accuracy_lower_bound",
        ),
        usable_site_acceptance=_enum(
            UsableSiteAcceptance,
            data["usable_site_acceptance"],
            "usable_site_acceptance",
        ),
        minimum_usable_site_fraction=_float(
            data["minimum_usable_site_fraction"],
            "minimum_usable_site_fraction",
        ),
        resource_policy=_analysis_resource_policy_from_tree(data["resource_policy"]),
    )
    _canonical_tree(
        tree,
        calibration_analysis_request_to_tree(value),
        CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
    )
    return value


def encode_calibration_analysis_request(value: CalibrationAnalysisRequest) -> bytes:
    return _encode_checked(
        calibration_analysis_request_to_tree(value),
        max_bytes=MAX_ANALYSIS_REQUEST_BYTES,
        label=CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
    )


def decode_calibration_analysis_request(
    payload: bytes | bytearray | memoryview,
) -> CalibrationAnalysisRequest:
    list_limits = {
        ("layout", "reference_event_indices"): MAX_LAYOUT_REFERENCE_EVENT_INDICES,
        ("grid_shape_yx",): 2,
        ("model_kinds",): _MODEL_KIND_COUNT,
    }
    limits = CanonicalDecodeLimits(
        max_depth=32,
        max_nodes=MAX_LAYOUT_REFERENCE_EVENT_INDICES + 512,
        max_container_entries=MAX_LAYOUT_REFERENCE_EVENT_INDICES + 512,
        max_arrays=0,
        max_total_array_bytes=0,
    )
    return _decode_typed(
        payload,
        parser=calibration_analysis_request_from_tree,
        projector=calibration_analysis_request_to_tree,
        schema=CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
        max_bytes=MAX_ANALYSIS_REQUEST_BYTES,
        limits=limits,
        admit_structure=lambda events: _reject_unexpected_structure(
            events,
            allowed_lists=list_limits,
            label=CALIBRATION_ANALYSIS_REQUEST_SCHEMA,
        ),
    )


_WORK_PLAN_FIELDS = (
    "source_cell_count",
    "bracket_upper_bound",
    "train_bracket_upper_bound",
    "reference_evidence_bracket_upper_bound",
    "reference_frame_upper_bound",
    "image_pixel_count",
    "full_frame_read_count",
    "feature_pixel_operations",
    "signal_evaluations",
    "modality_test_work_units",
    "reference_valley_diagnostic_count",
    "diagnostics_encoding_upper_bound_bytes",
    "planned_kernel_elements",
    "maximum_model_sampled_pixels",
    "total_model_sampled_pixels",
    "artifact_metadata_encoding_upper_bound_bytes",
    "artifact_encoding_upper_bound_bytes",
    "layout_working_bytes",
    "detector_working_bytes",
    "assignment_scratch_bytes",
    "feature_working_bytes",
    "psf_working_bytes",
    "artifact_array_bytes",
    "canonical_encoding_scratch_bytes",
    "working_peak_bytes",
    "detector_graph_work_units",
    "dense_assignment_work_units",
)


def calibration_work_plan_to_tree(value: CalibrationWorkPlan) -> dict[str, Any]:
    if not isinstance(value, CalibrationWorkPlan):
        raise TypeError("value must be CalibrationWorkPlan")
    return {
        "schema": CALIBRATION_WORK_PLAN_SCHEMA,
        **{name: getattr(value, name) for name in _WORK_PLAN_FIELDS},
    }


def calibration_work_plan_from_tree(tree: Any) -> CalibrationWorkPlan:
    data = _exact_map(
        tree,
        {"schema", *_WORK_PLAN_FIELDS},
        CALIBRATION_WORK_PLAN_SCHEMA,
    )
    value = CalibrationWorkPlan(
        **{name: _integer(data[name], name) for name in _WORK_PLAN_FIELDS}
    )
    _canonical_tree(
        tree,
        calibration_work_plan_to_tree(value),
        CALIBRATION_WORK_PLAN_SCHEMA,
    )
    return value


def encode_calibration_work_plan(value: CalibrationWorkPlan) -> bytes:
    return _encode_checked(
        calibration_work_plan_to_tree(value),
        max_bytes=MAX_ANALYSIS_WORK_PLAN_BYTES,
        label=CALIBRATION_WORK_PLAN_SCHEMA,
    )


def decode_calibration_work_plan(
    payload: bytes | bytearray | memoryview,
) -> CalibrationWorkPlan:
    limits = CanonicalDecodeLimits(
        max_depth=8,
        max_nodes=64,
        max_container_entries=64,
        max_arrays=0,
        max_total_array_bytes=0,
    )
    return _decode_typed(
        payload,
        parser=calibration_work_plan_from_tree,
        projector=calibration_work_plan_to_tree,
        schema=CALIBRATION_WORK_PLAN_SCHEMA,
        max_bytes=MAX_ANALYSIS_WORK_PLAN_BYTES,
        limits=limits,
        admit_structure=lambda events: _reject_unexpected_structure(
            events,
            allowed_lists={},
            label=CALIBRATION_WORK_PLAN_SCHEMA,
        ),
    )


def _site_detection_diagnostic_to_tree(
    value: SiteDetectionDiagnostic,
) -> dict[str, Any]:
    if not isinstance(value, SiteDetectionDiagnostic):
        raise TypeError("value must be SiteDetectionDiagnostic")
    return {
        "schema": SITE_DETECTION_DIAGNOSTIC_SCHEMA,
        "candidate_count": value.candidate_count,
        "minimum_peak_to_saddle_prominence": (
            value.minimum_peak_to_saddle_prominence
        ),
        "minimum_half_prominence_basin_area_pixels": (
            value.minimum_half_prominence_basin_area_pixels
        ),
        "lattice_rms_residual_pixels": value.lattice_rms_residual_pixels,
        "minimum_band_separation_pixels": value.minimum_band_separation_pixels,
        "affine_sin_angle": value.affine_sin_angle,
        "affine_condition_number": value.affine_condition_number,
        "assignment_cost_gap_pixels_squared": (
            value.assignment_cost_gap_pixels_squared
        ),
    }


def _optional_float(value: Any, field: str) -> float | None:
    return None if value is None else _float(value, field)


def _site_detection_diagnostic_from_tree(tree: Any) -> SiteDetectionDiagnostic:
    fields = {
        "candidate_count",
        "minimum_peak_to_saddle_prominence",
        "minimum_half_prominence_basin_area_pixels",
        "lattice_rms_residual_pixels",
        "minimum_band_separation_pixels",
        "affine_sin_angle",
        "affine_condition_number",
        "assignment_cost_gap_pixels_squared",
    }
    data = _exact_nested_map(tree, fields, SITE_DETECTION_DIAGNOSTIC_SCHEMA)
    value = SiteDetectionDiagnostic(
        candidate_count=_integer(data["candidate_count"], "candidate_count"),
        minimum_peak_to_saddle_prominence=_float(
            data["minimum_peak_to_saddle_prominence"],
            "minimum_peak_to_saddle_prominence",
        ),
        minimum_half_prominence_basin_area_pixels=_integer(
            data["minimum_half_prominence_basin_area_pixels"],
            "minimum_half_prominence_basin_area_pixels",
        ),
        lattice_rms_residual_pixels=_float(
            data["lattice_rms_residual_pixels"],
            "lattice_rms_residual_pixels",
        ),
        minimum_band_separation_pixels=_optional_float(
            data["minimum_band_separation_pixels"],
            "minimum_band_separation_pixels",
        ),
        affine_sin_angle=_optional_float(
            data["affine_sin_angle"], "affine_sin_angle"
        ),
        affine_condition_number=_optional_float(
            data["affine_condition_number"], "affine_condition_number"
        ),
        assignment_cost_gap_pixels_squared=_optional_float(
            data["assignment_cost_gap_pixels_squared"],
            "assignment_cost_gap_pixels_squared",
        ),
    )
    _canonical_tree(
        tree,
        _site_detection_diagnostic_to_tree(value),
        SITE_DETECTION_DIAGNOSTIC_SCHEMA,
    )
    return value


def _model_diagnostic_to_tree(value: ModelAnalysisDiagnostic) -> dict[str, Any]:
    if not isinstance(value, ModelAnalysisDiagnostic):
        raise TypeError("value must be ModelAnalysisDiagnostic")
    return {
        "schema": MODEL_ANALYSIS_DIAGNOSTIC_SCHEMA,
        "kind": value.kind.value,
        "usable_site_count": value.usable_site_count,
        "rejected_site_count": value.rejected_site_count,
        "minimum_fidelity": value.minimum_fidelity,
        "mean_fidelity": value.mean_fidelity,
        "minimum_class_accuracy_lower_bound": (
            value.minimum_class_accuracy_lower_bound
        ),
        "mean_class_accuracy_lower_bound": value.mean_class_accuracy_lower_bound,
    }


def _model_diagnostic_from_tree(tree: Any) -> ModelAnalysisDiagnostic:
    fields = {
        "kind",
        "usable_site_count",
        "rejected_site_count",
        "minimum_fidelity",
        "mean_fidelity",
        "minimum_class_accuracy_lower_bound",
        "mean_class_accuracy_lower_bound",
    }
    data = _exact_nested_map(tree, fields, MODEL_ANALYSIS_DIAGNOSTIC_SCHEMA)
    value = ModelAnalysisDiagnostic(
        kind=_enum(ReadoutModelKind, data["kind"], "diagnostic model kind"),
        usable_site_count=_integer(
            data["usable_site_count"], "usable_site_count"
        ),
        rejected_site_count=_integer(
            data["rejected_site_count"], "rejected_site_count"
        ),
        minimum_fidelity=_float(data["minimum_fidelity"], "minimum_fidelity"),
        mean_fidelity=_float(data["mean_fidelity"], "mean_fidelity"),
        minimum_class_accuracy_lower_bound=_float(
            data["minimum_class_accuracy_lower_bound"],
            "minimum_class_accuracy_lower_bound",
        ),
        mean_class_accuracy_lower_bound=_float(
            data["mean_class_accuracy_lower_bound"],
            "mean_class_accuracy_lower_bound",
        ),
    )
    _canonical_tree(
        tree,
        _model_diagnostic_to_tree(value),
        MODEL_ANALYSIS_DIAGNOSTIC_SCHEMA,
    )
    return value


def _reference_valley_evidence_to_tree(
    value: ReferenceValleyEvidence,
) -> dict[str, Any]:
    if not isinstance(value, ReferenceValleyEvidence):
        raise TypeError("value must be ReferenceValleyEvidence")
    return {
        "schema": REFERENCE_VALLEY_EVIDENCE_SCHEMA,
        "sample_count": value.sample_count,
        "left_count": value.left_count,
        "middle_count": value.middle_count,
        "right_count": value.right_count,
        "outside_count": value.outside_count,
        "invalid_count": value.invalid_count,
    }


def _reference_valley_evidence_from_tree(tree: Any) -> ReferenceValleyEvidence:
    fields = {
        "sample_count",
        "left_count",
        "middle_count",
        "right_count",
        "outside_count",
        "invalid_count",
    }
    data = _exact_nested_map(
        tree,
        fields,
        REFERENCE_VALLEY_EVIDENCE_SCHEMA,
    )
    value = ReferenceValleyEvidence(
        sample_count=_integer(data["sample_count"], "sample_count"),
        left_count=_integer(data["left_count"], "left_count"),
        middle_count=_integer(data["middle_count"], "middle_count"),
        right_count=_integer(data["right_count"], "right_count"),
        outside_count=_integer(data["outside_count"], "outside_count"),
        invalid_count=_integer(data["invalid_count"], "invalid_count"),
    )
    _canonical_tree(
        tree,
        _reference_valley_evidence_to_tree(value),
        REFERENCE_VALLEY_EVIDENCE_SCHEMA,
    )
    return value


def _reference_valley_diagnostic_to_tree(
    value: ReferenceValleyDiagnostic,
) -> dict[str, Any]:
    if not isinstance(value, ReferenceValleyDiagnostic):
        raise TypeError("value must be ReferenceValleyDiagnostic")
    return {
        "schema": REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA,
        "reference_index": value.reference_index,
        "site_index": value.site_index,
        "proposal_threshold": value.proposal_threshold,
        "proposal_lower_sample_count": value.proposal_lower_sample_count,
        "proposal_upper_sample_count": value.proposal_upper_sample_count,
        "cluster_separation_rss": value.cluster_separation_rss,
        "evidence": _reference_valley_evidence_to_tree(value.evidence),
        "lower_cluster_evidence": (
            None
            if value.lower_cluster_evidence is None
            else _reference_valley_evidence_to_tree(value.lower_cluster_evidence)
        ),
        "upper_cluster_evidence": (
            None
            if value.upper_cluster_evidence is None
            else _reference_valley_evidence_to_tree(value.upper_cluster_evidence)
        ),
        "site_accepted": value.site_accepted,
    }


def _reference_valley_diagnostic_from_tree(
    tree: Any,
) -> ReferenceValleyDiagnostic:
    fields = {
        "reference_index",
        "site_index",
        "proposal_threshold",
        "proposal_lower_sample_count",
        "proposal_upper_sample_count",
        "cluster_separation_rss",
        "evidence",
        "lower_cluster_evidence",
        "upper_cluster_evidence",
        "site_accepted",
    }
    data = _exact_nested_map(
        tree,
        fields,
        REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA,
    )
    lower = data["lower_cluster_evidence"]
    upper = data["upper_cluster_evidence"]
    value = ReferenceValleyDiagnostic(
        reference_index=_integer(data["reference_index"], "reference_index"),
        site_index=_integer(data["site_index"], "site_index"),
        proposal_threshold=_optional_float(
            data["proposal_threshold"],
            "proposal_threshold",
        ),
        proposal_lower_sample_count=_integer(
            data["proposal_lower_sample_count"],
            "proposal_lower_sample_count",
        ),
        proposal_upper_sample_count=_integer(
            data["proposal_upper_sample_count"],
            "proposal_upper_sample_count",
        ),
        cluster_separation_rss=_optional_float(
            data["cluster_separation_rss"],
            "cluster_separation_rss",
        ),
        evidence=_reference_valley_evidence_from_tree(data["evidence"]),
        lower_cluster_evidence=(
            None if lower is None else _reference_valley_evidence_from_tree(lower)
        ),
        upper_cluster_evidence=(
            None if upper is None else _reference_valley_evidence_from_tree(upper)
        ),
        site_accepted=_bool(data["site_accepted"], "site_accepted"),
    )
    _canonical_tree(
        tree,
        _reference_valley_diagnostic_to_tree(value),
        REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA,
    )
    return value


def _diagnostic_site_limit(policy: CalibrationAnalysisResourcePolicy) -> int:
    return min(
        policy.artifact_policy.max_sites,
        policy.max_lattice_sites,
        MAX_DIAGNOSTIC_VECTOR_ENTRIES,
    )


def _diagnostic_model_limit(policy: CalibrationAnalysisResourcePolicy) -> int:
    return min(policy.artifact_policy.max_models, _MODEL_KIND_COUNT)


def _reference_valley_diagnostic_limit(
    policy: CalibrationAnalysisResourcePolicy,
) -> int:
    return min(
        policy.max_reference_valley_diagnostics,
        MAX_DIAGNOSTIC_VECTOR_ENTRIES,
    )


def _validate_diagnostics_resources(
    value: CalibrationAnalysisDiagnostics,
    policy: CalibrationAnalysisResourcePolicy,
) -> None:
    if not isinstance(policy, CalibrationAnalysisResourcePolicy):
        raise TypeError("resource_policy must be CalibrationAnalysisResourcePolicy")
    site_limit = _diagnostic_site_limit(policy)
    model_limit = _diagnostic_model_limit(policy)
    valley_limit = _reference_valley_diagnostic_limit(policy)
    if value.bracket_count > policy.max_brackets:
        raise CalibrationResourceExceeded(
            "calibration diagnostic bracket count exceeds resource policy"
        )
    if value.reference_frame_count > policy.max_reference_frames:
        raise CalibrationResourceExceeded(
            "calibration diagnostic reference-frame count exceeds resource policy"
        )
    if len(value.consensus_dark_counts) > site_limit:
        raise CalibrationResourceExceeded(
            "calibration diagnostic vectors exceed site resource policy"
        )
    if len(value.models) > model_limit:
        raise CalibrationResourceExceeded(
            "calibration diagnostic models exceed model resource policy"
        )
    if len(value.reference_valleys) > valley_limit:
        raise CalibrationResourceExceeded(
            "reference-valley diagnostics exceed resource policy"
        )
    if value.detection.candidate_count > site_limit:
        raise CalibrationResourceExceeded(
            "site-detection diagnostic exceeds site resource policy"
        )
    if any(
        item.usable_site_count > site_limit
        or item.rejected_site_count > site_limit
        for item in value.models
    ):
        raise CalibrationResourceExceeded(
            "model diagnostic site counts exceed site resource policy"
        )


def calibration_analysis_diagnostics_to_tree(
    value: CalibrationAnalysisDiagnostics,
    *,
    resource_policy: CalibrationAnalysisResourcePolicy = (
        _DEFAULT_ANALYSIS_RESOURCE_POLICY
    ),
) -> dict[str, Any]:
    if not isinstance(value, CalibrationAnalysisDiagnostics):
        raise TypeError("value must be CalibrationAnalysisDiagnostics")
    _validate_diagnostics_resources(value, resource_policy)
    return {
        "schema": CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
        "bracket_count": value.bracket_count,
        "train_bracket_count": value.train_bracket_count,
        "reference_evidence_bracket_count": (
            value.reference_evidence_bracket_count
        ),
        "test_bracket_count": value.test_bracket_count,
        "partition_digest": value.partition_digest,
        "reference_frame_count": value.reference_frame_count,
        "valid_training_reference_pixel_fraction": (
            value.valid_training_reference_pixel_fraction
        ),
        "consensus_dark_counts": list(value.consensus_dark_counts),
        "consensus_bright_counts": list(value.consensus_bright_counts),
        "reference_valleys": [
            _reference_valley_diagnostic_to_tree(item)
            for item in value.reference_valleys
        ],
        "detection": _site_detection_diagnostic_to_tree(value.detection),
        "models": [_model_diagnostic_to_tree(item) for item in value.models],
    }


def calibration_analysis_diagnostics_from_tree(
    tree: Any,
    *,
    resource_policy: CalibrationAnalysisResourcePolicy = (
        _DEFAULT_ANALYSIS_RESOURCE_POLICY
    ),
) -> CalibrationAnalysisDiagnostics:
    fields = {
        "schema",
        "bracket_count",
        "train_bracket_count",
        "reference_evidence_bracket_count",
        "test_bracket_count",
        "partition_digest",
        "reference_frame_count",
        "valid_training_reference_pixel_fraction",
        "consensus_dark_counts",
        "consensus_bright_counts",
        "reference_valleys",
        "detection",
        "models",
    }
    data = _exact_map(tree, fields, CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA)
    dark = _list(data["consensus_dark_counts"], "consensus_dark_counts")
    bright = _list(data["consensus_bright_counts"], "consensus_bright_counts")
    models = _list(data["models"], "models")
    reference_valleys = _list(data["reference_valleys"], "reference_valleys")
    value = CalibrationAnalysisDiagnostics(
        bracket_count=_integer(data["bracket_count"], "bracket_count"),
        train_bracket_count=_integer(
            data["train_bracket_count"], "train_bracket_count"
        ),
        reference_evidence_bracket_count=_integer(
            data["reference_evidence_bracket_count"],
            "reference_evidence_bracket_count",
        ),
        test_bracket_count=_integer(
            data["test_bracket_count"], "test_bracket_count"
        ),
        partition_digest=_text(data["partition_digest"], "partition_digest"),
        reference_frame_count=_integer(
            data["reference_frame_count"], "reference_frame_count"
        ),
        valid_training_reference_pixel_fraction=_float(
            data["valid_training_reference_pixel_fraction"],
            "valid_training_reference_pixel_fraction",
        ),
        consensus_dark_counts=tuple(
            _integer(item, "consensus_dark_counts entry") for item in dark
        ),
        consensus_bright_counts=tuple(
            _integer(item, "consensus_bright_counts entry") for item in bright
        ),
        reference_valleys=tuple(
            _reference_valley_diagnostic_from_tree(item)
            for item in reference_valleys
        ),
        detection=_site_detection_diagnostic_from_tree(data["detection"]),
        models=tuple(_model_diagnostic_from_tree(item) for item in models),
    )
    _validate_diagnostics_resources(value, resource_policy)
    _canonical_tree(
        tree,
        calibration_analysis_diagnostics_to_tree(
            value,
            resource_policy=resource_policy,
        ),
        CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
    )
    return value


def encode_calibration_analysis_diagnostics(
    value: CalibrationAnalysisDiagnostics,
    *,
    resource_policy: CalibrationAnalysisResourcePolicy = (
        _DEFAULT_ANALYSIS_RESOURCE_POLICY
    ),
) -> bytes:
    if not isinstance(resource_policy, CalibrationAnalysisResourcePolicy):
        raise TypeError("resource_policy must be CalibrationAnalysisResourcePolicy")
    max_bytes = min(
        MAX_ANALYSIS_DIAGNOSTICS_BYTES,
        resource_policy.artifact_policy.max_artifact_blob_bytes,
    )
    return _encode_checked(
        calibration_analysis_diagnostics_to_tree(
            value,
            resource_policy=resource_policy,
        ),
        max_bytes=max_bytes,
        label=CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
    )


def decode_calibration_analysis_diagnostics(
    payload: bytes | bytearray | memoryview,
    *,
    resource_policy: CalibrationAnalysisResourcePolicy = (
        _DEFAULT_ANALYSIS_RESOURCE_POLICY
    ),
) -> CalibrationAnalysisDiagnostics:
    if not isinstance(resource_policy, CalibrationAnalysisResourcePolicy):
        raise TypeError("resource_policy must be CalibrationAnalysisResourcePolicy")
    site_limit = _diagnostic_site_limit(resource_policy)
    model_limit = _diagnostic_model_limit(resource_policy)
    valley_limit = _reference_valley_diagnostic_limit(resource_policy)
    max_bytes = min(
        MAX_ANALYSIS_DIAGNOSTICS_BYTES,
        resource_policy.artifact_policy.max_artifact_blob_bytes,
    )
    limits = CanonicalDecodeLimits(
        max_depth=24,
        max_nodes=(
            2 * site_limit
            + 16 * model_limit
            + 64 * valley_limit
            + 128
        ),
        max_container_entries=(
            2 * site_limit
            + 16 * model_limit
            + 64 * valley_limit
            + 128
        ),
        max_arrays=0,
        max_total_array_bytes=0,
    )
    allowed_lists = {
        ("consensus_dark_counts",): site_limit,
        ("consensus_bright_counts",): site_limit,
        ("models",): model_limit,
        ("reference_valleys",): valley_limit,
    }
    parser = lambda tree: calibration_analysis_diagnostics_from_tree(
        tree,
        resource_policy=resource_policy,
    )
    projector = lambda value: calibration_analysis_diagnostics_to_tree(
        value,
        resource_policy=resource_policy,
    )
    return _decode_typed(
        payload,
        parser=parser,
        projector=projector,
        schema=CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
        max_bytes=max_bytes,
        limits=limits,
        admit_structure=lambda events: _reject_unexpected_structure(
            events,
            allowed_lists=allowed_lists,
            label=CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA,
        ),
    )


__all__ = [
    "BOX_ANALYSIS_CONFIG_SCHEMA",
    "CALIBRATION_ANALYSIS_DIAGNOSTICS_SCHEMA",
    "CALIBRATION_ANALYSIS_REQUEST_SCHEMA",
    "CALIBRATION_ANALYSIS_RESOURCE_POLICY_SCHEMA",
    "CALIBRATION_RESOURCE_POLICY_SCHEMA",
    "CALIBRATION_WORK_PLAN_SCHEMA",
    "MAX_ANALYSIS_DIAGNOSTICS_BYTES",
    "MAX_ANALYSIS_REQUEST_BYTES",
    "MAX_ANALYSIS_WORK_PLAN_BYTES",
    "MAX_DIAGNOSTIC_VECTOR_ENTRIES",
    "MAX_LAYOUT_REFERENCE_EVENT_INDICES",
    "MODEL_ANALYSIS_DIAGNOSTIC_SCHEMA",
    "PSF_ANALYSIS_CONFIG_SCHEMA",
    "REFERENCE_VALLEY_DIAGNOSTIC_SCHEMA",
    "REFERENCE_VALLEY_EVIDENCE_SCHEMA",
    "SITE_DETECTION_DIAGNOSTIC_SCHEMA",
    "SITE_DETECTION_POLICY_SCHEMA",
    "CalibrationAnalysisCodecError",
    "calibration_analysis_diagnostics_encoding_upper_bound",
    "calibration_analysis_diagnostics_encoding_working_upper_bound",
    "calibration_analysis_diagnostics_from_tree",
    "calibration_analysis_diagnostics_to_tree",
    "calibration_analysis_request_from_tree",
    "calibration_analysis_request_to_tree",
    "calibration_work_plan_from_tree",
    "calibration_work_plan_to_tree",
    "decode_calibration_analysis_diagnostics",
    "decode_calibration_analysis_request",
    "decode_calibration_work_plan",
    "encode_calibration_analysis_diagnostics",
    "encode_calibration_analysis_request",
    "encode_calibration_work_plan",
]
