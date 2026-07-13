"""Strict current-version codecs for neutral-atom calibration values."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np

from zlc_data import AxisId, ComponentValidity, CoordinateFrameId
from zlc_data.codec import axis_from_tree, axis_to_tree, validity_from_tree, validity_to_tree
from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalListEvent,
    canonical_text as _text,
    decode,
    encode,
    exact_mapping as _exact_map,
)
from zlc_neutral_atom.capture_reference import (
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)

from .calibration import (
    BackgroundMode,
    BoxReadoutModel,
    BoxReducer,
    CalibrationArtifact,
    CalibrationCapability,
    CalibrationParameter,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    CalibrationSourceBinding,
    CalibrationStage,
    DEFAULT_CALIBRATION_RESOURCE_POLICY,
    DefaultModelPolicy,
    PerSitePsfReadoutModel,
    ReadoutModel,
    ReadoutModelHeader,
    ReadoutModelKind,
    ReadoutModelQuality,
    SiteMap,
    UniformPsfReadoutModel,
    validate_calibration_artifact_resources,
)
from .codec import (
    calibration_capture_layout_from_tree,
    calibration_capture_layout_to_tree,
    frame_contract_from_tree,
    frame_contract_to_tree,
)
from .contracts import FrameContract


CALIBRATION_SOURCE_BINDING_SCHEMA = "zlc_neutral_atom.calibration-source-binding"
SITE_MAP_SCHEMA = "zlc_neutral_atom.site-map"
READOUT_MODEL_QUALITY_SCHEMA = "zlc_neutral_atom.readout-model-quality"
READOUT_MODEL_HEADER_SCHEMA = "zlc_neutral_atom.readout-model-header"
READOUT_MODEL_SCHEMA = "zlc_neutral_atom.readout-model"
DEFAULT_MODEL_POLICY_SCHEMA = "zlc_neutral_atom.default-model-policy"
CALIBRATION_ARTIFACT_SCHEMA = "zlc_neutral_atom.calibration-artifact"


class CalibrationCodecError(ValueError):
    """A payload does not have one canonical current calibration meaning."""


T = TypeVar("T")


def _list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")
    return value


def _enum(enum_type, value: Any, field_name: str):
    try:
        return enum_type(_text(value, field_name))
    except ValueError as exc:
        raise ValueError(f"{field_name} has an unknown value {value!r}") from exc


def _ndarray(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must be a canonical ndarray")
    return value


def _canonical_tree(original: Any, projected: Any, schema: str) -> None:
    if encode(original) != encode(projected):
        raise CalibrationCodecError(f"{schema} tree is typed but non-canonical")


def _encode_typed(value: T, projector: Callable[[T], dict[str, Any]]) -> bytes:
    return encode(projector(value))


def _decode_typed(
    payload: bytes | bytearray | memoryview,
    parser: Callable[[Any], T],
    projector: Callable[[T], dict[str, Any]],
    schema: str,
    *,
    admit_structure=None,
) -> T:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("typed calibration payload must be bytes-like")
    raw = bytes(payload)
    value = parser(decode(raw, admit_structure=admit_structure))
    if encode(projector(value)) != raw:
        raise CalibrationCodecError(
            f"{schema} payload uses a non-canonical typed representation"
        )
    return value


def calibration_source_binding_to_tree(
    value: CalibrationSourceBinding,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationSourceBinding):
        raise TypeError("value must be CalibrationSourceBinding")
    return {
        "schema": CALIBRATION_SOURCE_BINDING_SCHEMA,
        "source_capture_ref": capture_artifact_ref_to_tree(
            value.source_capture_ref
        ),
        "layout": calibration_capture_layout_to_tree(value.layout),
        "source_schema_fingerprint": value.source_schema_fingerprint,
        "frame_contract_fingerprint": value.frame_contract_fingerprint,
        "bracket_count": value.bracket_count,
        "bracket_witness_digest": value.bracket_witness_digest,
    }


def calibration_source_binding_from_tree(tree: Any) -> CalibrationSourceBinding:
    data = _exact_map(
        tree,
        {
            "schema",
            "source_capture_ref",
            "layout",
            "source_schema_fingerprint",
            "frame_contract_fingerprint",
            "bracket_count",
            "bracket_witness_digest",
        },
        CALIBRATION_SOURCE_BINDING_SCHEMA,
    )
    if type(data["bracket_count"]) is not int:
        raise ValueError("bracket_count must be a canonical integer")
    value = CalibrationSourceBinding(
        capture_artifact_ref_from_tree(data["source_capture_ref"]),
        calibration_capture_layout_from_tree(data["layout"]),
        _text(data["source_schema_fingerprint"], "source_schema_fingerprint"),
        _text(data["frame_contract_fingerprint"], "frame_contract_fingerprint"),
        data["bracket_count"],
        _text(data["bracket_witness_digest"], "bracket_witness_digest"),
    )
    _canonical_tree(
        tree,
        calibration_source_binding_to_tree(value),
        CALIBRATION_SOURCE_BINDING_SCHEMA,
    )
    return value


def site_map_to_tree(value: SiteMap) -> dict[str, Any]:
    if not isinstance(value, SiteMap):
        raise TypeError("value must be SiteMap")
    return {
        "schema": SITE_MAP_SCHEMA,
        "site_axis": axis_to_tree(value.site_axis),
        "coordinates_xy": value.coordinates_xy,
        "coordinate_frame": value.coordinate_frame.value,
        "validity": validity_to_tree(value.validity),
        "detection_lineage_digest": value.detection_lineage_digest,
    }


def site_map_from_tree(tree: Any) -> SiteMap:
    data = _exact_map(
        tree,
        {
            "schema",
            "site_axis",
            "coordinates_xy",
            "coordinate_frame",
            "validity",
            "detection_lineage_digest",
        },
        SITE_MAP_SCHEMA,
    )
    validity = validity_from_tree(data["validity"])
    if not isinstance(validity, ComponentValidity):
        raise ValueError("SiteMap validity must decode as ComponentValidity")
    value = SiteMap(
        axis_from_tree(data["site_axis"]),
        _ndarray(data["coordinates_xy"], "coordinates_xy"),
        CoordinateFrameId(_text(data["coordinate_frame"], "coordinate_frame")),
        validity,
        _text(data["detection_lineage_digest"], "detection_lineage_digest"),
    )
    _canonical_tree(tree, site_map_to_tree(value), SITE_MAP_SCHEMA)
    return value


def _parameter_to_tree(value: CalibrationParameter) -> dict[str, Any]:
    if not isinstance(value, CalibrationParameter):
        raise TypeError("value must be CalibrationParameter")
    scalar = value.value
    tag = (
        "bool"
        if isinstance(scalar, bool)
        else "int"
        if isinstance(scalar, int)
        else "float"
        if isinstance(scalar, float)
        else "text"
    )
    return {"name": value.name, "type": tag, "value": scalar}


def _parameter_from_tree(tree: Any) -> CalibrationParameter:
    if not isinstance(tree, dict) or set(tree) != {"name", "type", "value"}:
        raise ValueError("CalibrationParameter has an unknown field set")
    tag = _text(tree["type"], "parameter type")
    value = tree["value"]
    expected = {
        "bool": bool,
        "int": int,
        "float": float,
        "text": str,
    }.get(tag)
    if expected is None or type(value) is not expected:
        raise ValueError("CalibrationParameter value differs from its scalar type tag")
    result = CalibrationParameter(_text(tree["name"], "parameter name"), value)
    _canonical_tree(tree, _parameter_to_tree(result), "CalibrationParameter")
    return result


def _component_validity(tree: Any, field_name: str) -> ComponentValidity:
    value = validity_from_tree(tree)
    if not isinstance(value, ComponentValidity):
        raise ValueError(f"{field_name} must decode as ComponentValidity")
    return value


def readout_model_quality_to_tree(value: ReadoutModelQuality) -> dict[str, Any]:
    if not isinstance(value, ReadoutModelQuality):
        raise TypeError("value must be ReadoutModelQuality")
    return {
        "schema": READOUT_MODEL_QUALITY_SCHEMA,
        "site_axis_id": value.site_axis_id.value,
        "usable_sites": validity_to_tree(value.usable_sites),
        "dark_training_sample_counts": value.dark_training_sample_counts,
        "bright_training_sample_counts": value.bright_training_sample_counts,
        "held_out_dark_success_counts": value.held_out_dark_success_counts,
        "held_out_dark_total_counts": value.held_out_dark_total_counts,
        "held_out_dark_labeled_counts": value.held_out_dark_labeled_counts,
        "held_out_bright_success_counts": value.held_out_bright_success_counts,
        "held_out_bright_total_counts": value.held_out_bright_total_counts,
        "held_out_bright_labeled_counts": value.held_out_bright_labeled_counts,
        "held_out_dark_accuracy_lower_bounds": (
            value.held_out_dark_accuracy_lower_bounds
        ),
        "held_out_bright_accuracy_lower_bounds": (
            value.held_out_bright_accuracy_lower_bounds
        ),
        "held_out_fidelity": value.held_out_fidelity,
        "held_out_validity": validity_to_tree(value.held_out_validity),
        "quality_gate_id": value.quality_gate_id,
        "quality_gate_version": value.quality_gate_version,
        "gate_passed": value.gate_passed,
    }


def readout_model_quality_from_tree(tree: Any) -> ReadoutModelQuality:
    data = _exact_map(
        tree,
        {
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
        },
        READOUT_MODEL_QUALITY_SCHEMA,
    )
    value = ReadoutModelQuality(
        AxisId(_text(data["site_axis_id"], "site_axis_id")),
        _component_validity(data["usable_sites"], "usable_sites"),
        _ndarray(data["dark_training_sample_counts"], "dark_training_sample_counts"),
        _ndarray(data["bright_training_sample_counts"], "bright_training_sample_counts"),
        _ndarray(data["held_out_dark_success_counts"], "held_out_dark_success_counts"),
        _ndarray(data["held_out_dark_total_counts"], "held_out_dark_total_counts"),
        _ndarray(data["held_out_dark_labeled_counts"], "held_out_dark_labeled_counts"),
        _ndarray(data["held_out_bright_success_counts"], "held_out_bright_success_counts"),
        _ndarray(data["held_out_bright_total_counts"], "held_out_bright_total_counts"),
        _ndarray(
            data["held_out_bright_labeled_counts"],
            "held_out_bright_labeled_counts",
        ),
        _ndarray(
            data["held_out_dark_accuracy_lower_bounds"],
            "held_out_dark_accuracy_lower_bounds",
        ),
        _ndarray(
            data["held_out_bright_accuracy_lower_bounds"],
            "held_out_bright_accuracy_lower_bounds",
        ),
        _ndarray(data["held_out_fidelity"], "held_out_fidelity"),
        _component_validity(data["held_out_validity"], "held_out_validity"),
        _text(data["quality_gate_id"], "quality_gate_id"),
        _text(data["quality_gate_version"], "quality_gate_version"),
        _bool(data["gate_passed"], "gate_passed"),
    )
    _canonical_tree(
        tree,
        readout_model_quality_to_tree(value),
        READOUT_MODEL_QUALITY_SCHEMA,
    )
    return value


def readout_model_header_to_tree(value: ReadoutModelHeader) -> dict[str, Any]:
    if not isinstance(value, ReadoutModelHeader):
        raise TypeError("value must be ReadoutModelHeader")
    return {
        "schema": READOUT_MODEL_HEADER_SCHEMA,
        "model_id": value.model_id,
        "model_version": value.model_version,
        "frame_contract_fingerprint": value.frame_contract_fingerprint,
        "site_map_fingerprint": value.site_map_fingerprint,
        "site_axis_id": value.site_axis_id.value,
        "thresholds": value.thresholds,
        "occupied_above_thresholds": value.occupied_above_thresholds,
        "quality": readout_model_quality_to_tree(value.quality),
        "parameters": [_parameter_to_tree(item) for item in value.parameters],
    }


def readout_model_header_from_tree(tree: Any) -> ReadoutModelHeader:
    data = _exact_map(
        tree,
        {
            "schema",
            "model_id",
            "model_version",
            "frame_contract_fingerprint",
            "site_map_fingerprint",
            "site_axis_id",
            "thresholds",
            "occupied_above_thresholds",
            "quality",
            "parameters",
        },
        READOUT_MODEL_HEADER_SCHEMA,
    )
    value = ReadoutModelHeader(
        _text(data["model_id"], "model_id"),
        _text(data["model_version"], "model_version"),
        _text(data["frame_contract_fingerprint"], "frame_contract_fingerprint"),
        _text(data["site_map_fingerprint"], "site_map_fingerprint"),
        AxisId(_text(data["site_axis_id"], "site_axis_id")),
        _ndarray(data["thresholds"], "thresholds"),
        _ndarray(data["occupied_above_thresholds"], "occupied_above_thresholds"),
        readout_model_quality_from_tree(data["quality"]),
        tuple(
            _parameter_from_tree(item)
            for item in _list(data["parameters"], "parameters")
        ),
    )
    _canonical_tree(
        tree,
        readout_model_header_to_tree(value),
        READOUT_MODEL_HEADER_SCHEMA,
    )
    return value


def readout_model_to_tree(value: ReadoutModel) -> dict[str, Any]:
    common = {
        "schema": READOUT_MODEL_SCHEMA,
        "kind": value.kind.value,
        "header": readout_model_header_to_tree(value.header),
        "boxes_xywh": value.boxes_xywh,
    }
    if isinstance(value, BoxReadoutModel):
        return {**common, "reducer": value.reducer.value}
    if isinstance(value, PerSitePsfReadoutModel):
        return {
            **common,
            "kernels": value.kernels,
            "background": value.background.value,
            "background_padding": value.background_padding,
        }
    if isinstance(value, UniformPsfReadoutModel):
        return {
            **common,
            "kernel": value.kernel,
            "background": value.background.value,
            "background_padding": value.background_padding,
        }
    raise TypeError("value must be a closed ReadoutModel")


def readout_model_from_tree(tree: Any) -> ReadoutModel:
    if not isinstance(tree, dict):
        raise ValueError("ReadoutModel must be a mapping")
    kind = _enum(ReadoutModelKind, tree.get("kind"), "model kind")
    common = {"schema", "kind", "header", "boxes_xywh"}
    if kind is ReadoutModelKind.BOX:
        data = _exact_map(tree, common | {"reducer"}, READOUT_MODEL_SCHEMA)
        value: ReadoutModel = BoxReadoutModel(
            readout_model_header_from_tree(data["header"]),
            _ndarray(data["boxes_xywh"], "boxes_xywh"),
            _enum(BoxReducer, data["reducer"], "reducer"),
        )
    elif kind is ReadoutModelKind.PER_SITE_PSF:
        data = _exact_map(
            tree,
            common | {"kernels", "background", "background_padding"},
            READOUT_MODEL_SCHEMA,
        )
        value = PerSitePsfReadoutModel(
            readout_model_header_from_tree(data["header"]),
            _ndarray(data["boxes_xywh"], "boxes_xywh"),
            _ndarray(data["kernels"], "kernels"),
            _enum(BackgroundMode, data["background"], "background"),
            data["background_padding"],
        )
    else:
        data = _exact_map(
            tree,
            common | {"kernel", "background", "background_padding"},
            READOUT_MODEL_SCHEMA,
        )
        value = UniformPsfReadoutModel(
            readout_model_header_from_tree(data["header"]),
            _ndarray(data["boxes_xywh"], "boxes_xywh"),
            _ndarray(data["kernel"], "kernel"),
            _enum(BackgroundMode, data["background"], "background"),
            data["background_padding"],
        )
    _canonical_tree(tree, readout_model_to_tree(value), READOUT_MODEL_SCHEMA)
    return value


def default_model_policy_to_tree(value: DefaultModelPolicy) -> dict[str, Any]:
    if not isinstance(value, DefaultModelPolicy):
        raise TypeError("value must be DefaultModelPolicy")
    return {
        "schema": DEFAULT_MODEL_POLICY_SCHEMA,
        "policy_id": value.policy_id,
        "policy_version": value.policy_version,
        "default_model_id": value.default_model_id,
        "default_kind": None if value.default_kind is None else value.default_kind.value,
    }


def default_model_policy_from_tree(tree: Any) -> DefaultModelPolicy:
    data = _exact_map(
        tree,
        {"schema", "policy_id", "policy_version", "default_model_id", "default_kind"},
        DEFAULT_MODEL_POLICY_SCHEMA,
    )
    model_id = data["default_model_id"]
    kind = data["default_kind"]
    value = DefaultModelPolicy(
        _text(data["policy_id"], "policy_id"),
        _text(data["policy_version"], "policy_version"),
        None if model_id is None else _text(model_id, "default_model_id"),
        None if kind is None else _enum(ReadoutModelKind, kind, "default_kind"),
    )
    _canonical_tree(
        tree,
        default_model_policy_to_tree(value),
        DEFAULT_MODEL_POLICY_SCHEMA,
    )
    return value


def calibration_artifact_to_tree(value: CalibrationArtifact) -> dict[str, Any]:
    if not isinstance(value, CalibrationArtifact):
        raise TypeError("value must be CalibrationArtifact")
    return {
        "schema": CALIBRATION_ARTIFACT_SCHEMA,
        "source_binding": calibration_source_binding_to_tree(value.source_binding),
        "frame_contract": frame_contract_to_tree(value.frame_contract),
        "site_map": site_map_to_tree(value.site_map),
        "models": [readout_model_to_tree(model) for model in value.models],
        "stage": value.stage.value,
        "capabilities": [capability.value for capability in value.capabilities],
        "required_model_kinds": [kind.value for kind in value.required_model_kinds],
        "default_model_policy": default_model_policy_to_tree(
            value.default_model_policy
        ),
        "algorithm_id": value.algorithm_id,
        "algorithm_version": value.algorithm_version,
        "parameters": [_parameter_to_tree(item) for item in value.parameters],
    }


def calibration_artifact_from_tree(tree: Any) -> CalibrationArtifact:
    data = _exact_map(
        tree,
        {
            "schema",
            "source_binding",
            "frame_contract",
            "site_map",
            "models",
            "stage",
            "capabilities",
            "required_model_kinds",
            "default_model_policy",
            "algorithm_id",
            "algorithm_version",
            "parameters",
        },
        CALIBRATION_ARTIFACT_SCHEMA,
    )
    value = CalibrationArtifact(
        calibration_source_binding_from_tree(data["source_binding"]),
        frame_contract_from_tree(data["frame_contract"]),
        site_map_from_tree(data["site_map"]),
        tuple(readout_model_from_tree(item) for item in _list(data["models"], "models")),
        _enum(CalibrationStage, data["stage"], "stage"),
        tuple(
            _enum(ReadoutModelKind, item, "required model kind")
            for item in _list(data["required_model_kinds"], "required_model_kinds")
        ),
        default_model_policy_from_tree(data["default_model_policy"]),
        _text(data["algorithm_id"], "algorithm_id"),
        _text(data["algorithm_version"], "algorithm_version"),
        tuple(
            _parameter_from_tree(item)
            for item in _list(data["parameters"], "parameters")
        ),
    )
    capabilities = tuple(
        _enum(CalibrationCapability, item, "capability")
        for item in _list(data["capabilities"], "capabilities")
    )
    if capabilities != value.capabilities:
        raise ValueError("serialized calibration capabilities differ from derived capabilities")
    _canonical_tree(
        tree,
        calibration_artifact_to_tree(value),
        CALIBRATION_ARTIFACT_SCHEMA,
    )
    return value


def encode_site_map(value: SiteMap) -> bytes:
    return _encode_typed(value, site_map_to_tree)


def encode_calibration_source_binding(value: CalibrationSourceBinding) -> bytes:
    return _encode_typed(value, calibration_source_binding_to_tree)


def decode_calibration_source_binding(
    payload: bytes | bytearray | memoryview,
) -> CalibrationSourceBinding:
    return _decode_typed(
        payload,
        calibration_source_binding_from_tree,
        calibration_source_binding_to_tree,
        CALIBRATION_SOURCE_BINDING_SCHEMA,
    )


def _checked_payload(
    payload: bytes | bytearray | memoryview,
    resource_policy: CalibrationResourcePolicy,
) -> bytes:
    if not isinstance(resource_policy, CalibrationResourcePolicy):
        raise TypeError("resource_policy must be CalibrationResourcePolicy")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("calibration payload must be bytes-like")
    size = payload.nbytes if isinstance(payload, memoryview) else len(payload)
    if size > resource_policy.max_artifact_blob_bytes:
        raise CalibrationResourceExceeded("calibration payload exceeds resource policy")
    return bytes(payload)


def _resource_admission(resource_policy: CalibrationResourcePolicy):
    """Return a canonical-structure admission hook with no wire-parser knowledge."""

    def admit(events) -> None:
        model_count = next(
            (
                event.length
                for event in events
                if isinstance(event, CanonicalListEvent) and event.path == ("models",)
            ),
            None,
        )
        if model_count is not None and model_count > resource_policy.max_models:
            raise CalibrationResourceExceeded(
                "calibration model count exceeds resource policy"
            )
        kernel_elements = 0
        site_array_suffixes = (
            "coordinates_xy",
            "thresholds",
            "occupied_above_thresholds",
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
            "boxes_xywh",
            "kernels",
            "mask",
        )
        arrays = tuple(
            event for event in events if isinstance(event, CanonicalArrayEvent)
        )
        # A closed calibration model has a small fixed number of arrays.  This
        # generic upper bound prevents unknown-field arrays from amplifying a
        # compact wire payload into thousands of NumPy objects before the typed
        # exact-field decoder rejects them.
        if len(arrays) > 16 + 16 * resource_policy.max_models:
            raise CalibrationResourceExceeded(
                "calibration ndarray count exceeds resource policy"
            )
        if sum(event.nbytes for event in arrays) > resource_policy.max_artifact_blob_bytes:
            raise CalibrationResourceExceeded(
                "calibration ndarray bytes exceed resource policy"
            )
        for event in arrays:
            field_name = event.path[-1] if event.path else None
            if event.shape and field_name in site_array_suffixes:
                if event.shape[0] > resource_policy.max_sites:
                    raise CalibrationResourceExceeded(
                        "calibration site count exceeds resource policy"
                    )
            if field_name in ("kernels", "kernel"):
                kernel_elements += event.nbytes // np.dtype(event.dtype).itemsize
        if kernel_elements > resource_policy.max_kernel_elements:
            raise CalibrationResourceExceeded(
                "calibration kernel elements exceed resource policy"
            )

    return admit


def decode_site_map(
    payload: bytes | bytearray | memoryview,
    *,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> SiteMap:
    raw = _checked_payload(payload, resource_policy)
    value = _decode_typed(
        raw,
        site_map_from_tree,
        site_map_to_tree,
        SITE_MAP_SCHEMA,
        admit_structure=_resource_admission(resource_policy),
    )
    if value.site_axis.size > resource_policy.max_sites:
        raise CalibrationResourceExceeded("SiteMap site count exceeds resource policy")
    return value


def encode_readout_model(value: ReadoutModel) -> bytes:
    return _encode_typed(value, readout_model_to_tree)


def decode_readout_model(
    payload: bytes | bytearray | memoryview,
    *,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> ReadoutModel:
    raw = _checked_payload(payload, resource_policy)
    value = _decode_typed(
        raw,
        readout_model_from_tree,
        readout_model_to_tree,
        READOUT_MODEL_SCHEMA,
        admit_structure=_resource_admission(resource_policy),
    )
    if value.header.site_count > resource_policy.max_sites:
        raise CalibrationResourceExceeded("readout model site count exceeds resource policy")
    kernel_elements = (
        int(value.kernels.size)
        if isinstance(value, PerSitePsfReadoutModel)
        else int(value.kernel.size)
        if isinstance(value, UniformPsfReadoutModel)
        else 0
    )
    if kernel_elements > resource_policy.max_kernel_elements:
        raise CalibrationResourceExceeded("readout model kernels exceed resource policy")
    return value


def encode_calibration_artifact(value: CalibrationArtifact) -> bytes:
    return _encode_typed(value, calibration_artifact_to_tree)


def calibration_artifact_metadata_encoding_upper_bound(
    *,
    source_binding: CalibrationSourceBinding,
    frame_contract: FrameContract,
    artifact_parameters: tuple[CalibrationParameter, ...],
    model_parameters: tuple[tuple[CalibrationParameter, ...], ...],
    model_kinds: tuple[ReadoutModelKind, ...],
    default_model_policy: DefaultModelPolicy,
    algorithm_id: str,
    algorithm_version: str,
) -> int:
    """Bound every non-array artifact byte from its frozen owner values.

    External camera/schema/source text is deliberately measured from the
    owner projections instead of hidden behind a fixed allowance.  The small
    fixed envelopes cover only codec-owned field names, schemas, fixed model
    identities, quality-gate identifiers, and scalar wrappers.
    """

    if not isinstance(source_binding, CalibrationSourceBinding):
        raise TypeError("source_binding must be CalibrationSourceBinding")
    if not isinstance(frame_contract, FrameContract):
        raise TypeError("frame_contract must be FrameContract")
    if not isinstance(default_model_policy, DefaultModelPolicy):
        raise TypeError("default_model_policy must be DefaultModelPolicy")
    artifact_parameters = tuple(artifact_parameters)
    model_parameters = tuple(tuple(items) for items in model_parameters)
    model_kinds = tuple(model_kinds)
    if any(not isinstance(item, CalibrationParameter) for item in artifact_parameters):
        raise TypeError("artifact_parameters must contain CalibrationParameter")
    if not model_kinds or any(
        not isinstance(item, ReadoutModelKind) for item in model_kinds
    ):
        raise TypeError("model_kinds must contain ReadoutModelKind")
    if len(model_parameters) != len(model_kinds) or any(
        any(not isinstance(item, CalibrationParameter) for item in items)
        for items in model_parameters
    ):
        raise TypeError(
            "model_parameters must contain one CalibrationParameter tuple per model"
        )
    algorithm_id = _text(algorithm_id, "algorithm_id")
    algorithm_version = _text(algorithm_version, "algorithm_version")
    projected = {
        "source_binding": calibration_source_binding_to_tree(source_binding),
        "frame_contract": frame_contract_to_tree(frame_contract),
        "site_coordinate_frame": frame_contract.coordinate_frame.value,
        "artifact_parameters": [
            _parameter_to_tree(item) for item in artifact_parameters
        ],
        "models": [
            {
                "kind": kind.value,
                "parameters": [
                    _parameter_to_tree(item) for item in parameters
                ],
            }
            for kind, parameters in zip(
                model_kinds,
                model_parameters,
                strict=True,
            )
        ],
        "required_model_kinds": [kind.value for kind in model_kinds],
        "default_model_policy": default_model_policy_to_tree(
            default_model_policy
        ),
        "algorithm_id": algorithm_id,
        "algorithm_version": algorithm_version,
    }
    return len(encode(projected)) + 16 * 1024 + 8 * 1024 * len(model_kinds)


def calibration_artifact_encoding_upper_bound(
    *,
    site_count: int,
    model_count: int,
    kernel_elements: int,
    metadata_encoding_upper_bound_bytes: int,
) -> int:
    """Conservatively bound the current artifact wire representation.

    The calibration codec owns this estimate because it owns both the closed
    artifact tree and its canonical encoding.  The raw-array term includes the
    SiteMap, every per-site model/quality vector, extraction boxes, and all PSF
    kernels.  Two wire bytes per raw byte cover base64 expansion plus canonical
    array framing.
    The fixed and per-model envelopes cover axes, parameters, schemas, and
    scalar metadata.  Boundary tests compare real encodings with
    this estimate so schema growth cannot silently outrun repository preflight.
    """

    values = {
        "site_count": site_count,
        "model_count": model_count,
        "kernel_elements": kernel_elements,
        "metadata_encoding_upper_bound_bytes": (
            metadata_encoding_upper_bound_bytes
        ),
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    # SiteMap coordinates+validity: 17 bytes/site.  Each model carries
    # thresholds, direction, eight uint64 evidence vectors, three float64
    # quality vectors, two validity masks, and four int64 box coordinates:
    # 131 bytes/site/model.  Kernels are canonical little-endian float64.
    raw_array_bytes = (
        17 * site_count
        + 131 * site_count * model_count
        + 8 * kernel_elements
    )
    return metadata_encoding_upper_bound_bytes + 2 * raw_array_bytes


def calibration_artifact_encoding_working_upper_bound(
    retained_array_bytes: int,
    metadata_encoding_upper_bound_bytes: int,
) -> int:
    """Bound transient canonical-encoding memory above retained artifact arrays.

    Canonical ndarray encoding can simultaneously hold a normalized byte copy,
    base64 text, the tagged JSON tree, its rendered string, and final UTF-8
    bytes.  Profiling across power-of-two float64 kernels measures roughly
    4.67x raw transient memory.  Six times retained bytes plus a fixed envelope
    is the current fail-closed owner contract.
    """

    if isinstance(retained_array_bytes, bool) or not isinstance(
        retained_array_bytes,
        int,
    ):
        raise TypeError("retained_array_bytes must be an integer")
    if retained_array_bytes < 0:
        raise ValueError("retained_array_bytes must be non-negative")
    if isinstance(metadata_encoding_upper_bound_bytes, bool) or not isinstance(
        metadata_encoding_upper_bound_bytes,
        int,
    ):
        raise TypeError("metadata_encoding_upper_bound_bytes must be an integer")
    if metadata_encoding_upper_bound_bytes < 0:
        raise ValueError(
            "metadata_encoding_upper_bound_bytes must be non-negative"
        )
    return (
        6 * retained_array_bytes
        + 12 * metadata_encoding_upper_bound_bytes
        + 1024 * 1024
    )


def decode_calibration_artifact(
    payload: bytes | bytearray | memoryview,
    *,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> CalibrationArtifact:
    raw = _checked_payload(payload, resource_policy)
    value = _decode_typed(
        raw,
        calibration_artifact_from_tree,
        calibration_artifact_to_tree,
        CALIBRATION_ARTIFACT_SCHEMA,
        admit_structure=_resource_admission(resource_policy),
    )
    validate_calibration_artifact_resources(value, resource_policy)
    return value


__all__ = [
    "CALIBRATION_ARTIFACT_SCHEMA",
    "CALIBRATION_SOURCE_BINDING_SCHEMA",
    "DEFAULT_MODEL_POLICY_SCHEMA",
    "READOUT_MODEL_HEADER_SCHEMA",
    "READOUT_MODEL_QUALITY_SCHEMA",
    "READOUT_MODEL_SCHEMA",
    "SITE_MAP_SCHEMA",
    "CalibrationCodecError",
    "calibration_artifact_from_tree",
    "calibration_artifact_metadata_encoding_upper_bound",
    "calibration_artifact_encoding_upper_bound",
    "calibration_artifact_encoding_working_upper_bound",
    "calibration_artifact_to_tree",
    "calibration_source_binding_from_tree",
    "calibration_source_binding_to_tree",
    "decode_calibration_artifact",
    "decode_calibration_source_binding",
    "decode_readout_model",
    "decode_site_map",
    "default_model_policy_from_tree",
    "default_model_policy_to_tree",
    "encode_calibration_artifact",
    "encode_calibration_source_binding",
    "encode_readout_model",
    "encode_site_map",
    "readout_model_from_tree",
    "readout_model_header_from_tree",
    "readout_model_header_to_tree",
    "readout_model_quality_from_tree",
    "readout_model_quality_to_tree",
    "readout_model_to_tree",
    "site_map_from_tree",
    "site_map_to_tree",
]
