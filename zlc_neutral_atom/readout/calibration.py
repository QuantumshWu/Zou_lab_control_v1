"""Immutable readout calibration values and pure model application.

The values in this module are the complete, closed calibration domain.  They
contain no camera, runtime, GUI, plugin hook, or training callable.  A model is
applicable only to the exact :class:`FrameContract` and :class:`SiteMap` named
by its header.  Invalid or non-finite image components remain invalid data;
``False`` is only a storage filler and is never interpreted as a dark atom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np

from zlc_data import (
    SITE,
    AxisId,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    Value,
    ValueSchema,
    expand_value_validity,
)
from zlc_data.codec import validity_to_tree
from zlc_storage import canonical_digest

from zlc_neutral_atom.capture_reference import CaptureArtifactRef

from .contracts import (
    CalibrationCaptureLayout,
    CameraCaptureDescriptor,
    FrameContract,
    ReadoutBindingKey,
)


def _canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _immutable_array(
    values: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
    field_name: str,
) -> np.ndarray:
    source = np.asarray(values)
    expected_dtype = np.dtype(dtype).newbyteorder("<")
    if source.dtype.newbyteorder("<") != expected_dtype:
        raise TypeError(
            f"{field_name} dtype {source.dtype} does not match {expected_dtype}"
        )
    if shape is not None and source.shape != shape:
        raise ValueError(f"{field_name} shape {source.shape} does not match {shape}")
    if (
        source.dtype == expected_dtype
        and source.flags.c_contiguous
        and _is_bytes_backed_read_only(source)
    ):
        return source
    normalized = np.ascontiguousarray(source.astype(expected_dtype, copy=False))
    result = np.frombuffer(normalized.tobytes(order="C"), dtype=expected_dtype).reshape(
        normalized.shape
    )
    result.setflags(write=False)
    return result


def _is_bytes_backed_read_only(array: np.ndarray) -> bool:
    """Recognize the intrinsically immutable representation owned by this domain."""

    owner: object = array
    while isinstance(owner, np.ndarray):
        if owner.flags.writeable:
            return False
        owner = owner.base
    return isinstance(owner, bytes)


def _float64_array(
    values: object,
    *,
    shape: tuple[int, ...] | None = None,
    field_name: str,
) -> np.ndarray:
    result = _immutable_array(
        values,
        dtype=np.dtype("<f8"),
        shape=shape,
        field_name=field_name,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain only finite values")
    if np.any((result == 0.0) & np.signbit(result)):
        normalized = result.copy()
        normalized[normalized == 0.0] = 0.0
        result = _immutable_array(
            normalized,
            dtype=np.dtype("<f8"),
            shape=result.shape,
            field_name=field_name,
        )
    return result


def _validity(
    value: object,
    *,
    site_axis_id: AxisId,
    site_count: int,
    field_name: str,
) -> ComponentValidity:
    if not isinstance(value, ComponentValidity):
        raise TypeError(f"{field_name} must be ComponentValidity")
    if value.axis_ids != (site_axis_id,):
        raise ValueError(f"{field_name} must name exactly the site axis")
    if value.mask.shape != (site_count,):
        raise ValueError(f"{field_name} mask must have shape ({site_count},)")
    # ComponentValidity itself canonicalizes onto immutable bytes.  Returning
    # the already validated value avoids a second O(site) copy at every model
    # bind while preserving the same ownership guarantee.
    return value


@dataclass(frozen=True)
class CalibrationSourceBinding:
    """Exact source lineage from which a calibration FrameContract was derived."""

    source_capture_ref: CaptureArtifactRef
    layout: CalibrationCaptureLayout
    source_schema_fingerprint: str
    frame_contract_fingerprint: str
    bracket_count: int
    bracket_witness_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.layout, CalibrationCaptureLayout):
            raise TypeError("layout must be CalibrationCaptureLayout")
        _sha256(self.source_schema_fingerprint, "source_schema_fingerprint")
        _sha256(self.frame_contract_fingerprint, "frame_contract_fingerprint")
        count = _nonnegative_integer(self.bracket_count, "bracket_count")
        if count == 0:
            raise ValueError("calibration source binding requires at least one bracket")
        object.__setattr__(self, "bracket_count", count)
        _sha256(self.bracket_witness_digest, "bracket_witness_digest")


def derive_calibration_source_binding(
    capture: object,
    layout: CalibrationCaptureLayout,
) -> tuple[CalibrationSourceBinding, FrameContract]:
    """Derive an exact source-compatibility binding from a resolved capture.

    ``capture`` is a resolved CaptureArtifact-shaped value.  This domain checks
    every consumed owner value exactly without importing the artifact repository
    back into readout and creating a package cycle.
    """

    return _derive_calibration_source_binding_with_resolved_brackets(
        capture,
        layout,
        resolved_brackets=None,
    )


def _derive_calibration_source_binding_with_resolved_brackets(
    capture: object,
    layout: CalibrationCaptureLayout,
    *,
    resolved_brackets: tuple[object, ...] | None,
) -> tuple[CalibrationSourceBinding, FrameContract]:
    """Internal one-join path for a caller that just resolved ``layout`` itself."""

    if not isinstance(layout, CalibrationCaptureLayout):
        raise TypeError("layout must be CalibrationCaptureLayout")
    try:
        reference = capture.ref  # type: ignore[attr-defined]
        block = capture.block  # type: ignore[attr-defined]
        provenance = capture.camera_provenance  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("capture must be a resolved CaptureArtifact") from exc
    if not isinstance(reference, CaptureArtifactRef):
        raise TypeError("resolved capture ref must be CaptureArtifactRef")
    if not isinstance(block, DataBlock):
        raise TypeError("resolved capture block must be DataBlock")
    try:
        descriptor = provenance.descriptor
        binding = provenance.binding
    except AttributeError as exc:
        raise TypeError("resolved capture omits camera provenance") from exc
    if not isinstance(descriptor, CameraCaptureDescriptor):
        raise TypeError("capture camera descriptor must be CameraCaptureDescriptor")
    if not isinstance(binding, ReadoutBindingKey):
        raise TypeError("capture readout binding must be ReadoutBindingKey")
    if resolved_brackets is None:
        brackets = layout.brackets(block.schema)
    else:
        from .contracts import CalibrationCaptureBracket

        brackets = tuple(resolved_brackets)
        if not brackets or any(
            not isinstance(bracket, CalibrationCaptureBracket)
            for bracket in brackets
        ):
            raise TypeError(
                "resolved_brackets must be non-empty CalibrationCaptureBracket values"
            )
        if len({bracket.context_key for bracket in brackets}) != len(brackets):
            raise ValueError("resolved calibration brackets have duplicate context keys")
    if descriptor.readout_event_axis_id != layout.readout_event_axis_id:
        raise ValueError(
            "capture descriptor and calibration layout name different event axes"
        )
    # ``layout.brackets`` (or the caller-supplied resolved brackets) is the one
    # authoritative join.  Re-entering ``from_calibration_capture`` here would
    # scan the entire sparse PointLayout a second time merely to validate the
    # same join.  ``from_schema`` still validates every camera/schema fact and
    # the selected readout-event index without rebuilding the bracket map.
    frame_contract = FrameContract._from_witnessed_schema(
        binding,
        descriptor,
        block.schema,
        readout_event_index=layout.readout_event_index,
    )
    witness = canonical_digest(
        {
            "schema": "zlc_neutral_atom.calibration-bracket-witness.v1",
            "brackets": [
                {
                    "context_key": [
                        [axis_id.value, index] for axis_id, index in bracket.context_key
                    ],
                    "reference_rows": [
                        [event_index, storage_row]
                        for event_index, storage_row in (
                            bracket.reference_point_storage_rows
                        )
                    ],
                    "readout_row": bracket.readout_point_storage_row,
                }
                for bracket in brackets
            ],
        }
    )
    return (
        CalibrationSourceBinding(
            reference,
            layout,
            block.schema.fingerprint,
            frame_contract.fingerprint,
            len(brackets),
            witness,
        ),
        frame_contract,
    )


def validate_calibration_artifact_source_compatibility(
    artifact: "CalibrationArtifact",
    capture_resolver,
) -> object:
    """Resolve and exact-rederive declared source compatibility.

    This proves schema/layout/FrameContract identity only.  It deliberately
    does not prove that the capture committed as a hardware run or that model
    coefficients were computed from its pixels.  Those authority claims belong
    to the trusted calibration Task's final-commit evidence, not this pure value
    domain or its standalone CAS.
    """

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    if not callable(capture_resolver):
        raise TypeError("capture_resolver must be callable")
    capture = capture_resolver(artifact.source_binding.source_capture_ref)
    binding, frame_contract = derive_calibration_source_binding(
        capture,
        artifact.source_binding.layout,
    )
    if binding != artifact.source_binding:
        raise ValueError("calibration source binding differs from resolved capture")
    from .codec import encode_frame_contract

    if encode_frame_contract(frame_contract) != encode_frame_contract(
        artifact.frame_contract
    ):
        raise ValueError("calibration FrameContract differs from resolved capture")
    return capture


@dataclass(frozen=True, eq=False)
class SiteMap:
    """Stable sites in ROI-local output-pixel XY coordinates.

    Sensor-global coordinates require an explicit future coordinate transform;
    equal array shape or a remembered ROI offset is never used to guess one.
    """

    site_axis: AxisSpec
    coordinates_xy: np.ndarray
    coordinate_frame: CoordinateFrameId
    validity: ComponentValidity
    detection_lineage_digest: str
    _fingerprint: str = field(init=False, repr=False, compare=False)
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role 'site'")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        coordinates = _float64_array(
            self.coordinates_xy,
            shape=(self.site_axis.size, 2),
            field_name="coordinates_xy",
        )
        validity = _validity(
            self.validity,
            site_axis_id=self.site_axis.axis_id,
            site_count=self.site_axis.size,
            field_name="site-map validity",
        )
        if not np.any(validity.mask):
            raise ValueError("SiteMap must contain at least one valid site")
        if np.any(coordinates[~validity.mask] != 0.0):
            raise ValueError("invalid SiteMap coordinates require canonical zero fillers")
        valid_coordinates = coordinates[validity.mask]
        if len({tuple(item) for item in valid_coordinates}) != len(valid_coordinates):
            raise ValueError("valid SiteMap sites must have unique XY coordinates")
        _sha256(self.detection_lineage_digest, "detection_lineage_digest")
        object.__setattr__(self, "coordinates_xy", coordinates)
        object.__setattr__(self, "validity", validity)
        from .calibration_codec import site_map_to_tree

        object.__setattr__(self, "_fingerprint", canonical_digest(site_map_to_tree(self)))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


@dataclass(frozen=True, order=True)
class CalibrationParameter:
    """One closed scalar calibration parameter retained in lineage."""

    name: str
    value: bool | int | float | str

    def __post_init__(self) -> None:
        _canonical_text(self.name, "calibration parameter name")
        value = self.value
        if isinstance(value, np.generic):
            value = value.item()
            object.__setattr__(self, "value", value)
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError("calibration parameter value must be bool/int/float/text")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("calibration parameter float must be finite")
        if isinstance(value, float) and value == 0.0:
            object.__setattr__(self, "value", 0.0)
        if isinstance(value, str):
            _canonical_text(value, "calibration parameter text")


@dataclass(frozen=True, eq=False)
class ReadoutModelQuality:
    """Per-site quality evidence and the gate that admitted one model."""

    site_axis_id: AxisId
    usable_sites: ComponentValidity
    dark_training_sample_counts: np.ndarray
    bright_training_sample_counts: np.ndarray
    held_out_dark_success_counts: np.ndarray
    held_out_dark_total_counts: np.ndarray
    held_out_bright_success_counts: np.ndarray
    held_out_bright_total_counts: np.ndarray
    held_out_dark_accuracy_lower_bounds: np.ndarray
    held_out_bright_accuracy_lower_bounds: np.ndarray
    held_out_fidelity: np.ndarray
    held_out_validity: ComponentValidity
    quality_gate_id: str
    quality_gate_version: str
    gate_passed: bool
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis_id, AxisId):
            raise TypeError("site_axis_id must be AxisId")
        if not isinstance(self.gate_passed, bool):
            raise TypeError("gate_passed must be bool")
        _canonical_text(self.quality_gate_id, "quality_gate_id")
        _canonical_text(self.quality_gate_version, "quality_gate_version")
        dark_source = np.asarray(self.dark_training_sample_counts)
        bright_source = np.asarray(self.bright_training_sample_counts)
        if dark_source.ndim != 1 or bright_source.ndim != 1:
            raise ValueError("per-class training counts must be one-dimensional site vectors")
        dark_counts = _immutable_array(
            dark_source,
            dtype=np.dtype("<u8"),
            shape=dark_source.shape,
            field_name="dark_training_sample_counts",
        )
        bright_counts = _immutable_array(
            bright_source,
            dtype=np.dtype("<u8"),
            shape=dark_source.shape,
            field_name="bright_training_sample_counts",
        )
        site_count = dark_counts.shape[0]
        if site_count == 0:
            raise ValueError("readout model quality requires at least one site")
        usable = _validity(
            self.usable_sites,
            site_axis_id=self.site_axis_id,
            site_count=site_count,
            field_name="usable_sites",
        )
        fidelity = _float64_array(
            self.held_out_fidelity,
            shape=(site_count,),
            field_name="held_out_fidelity",
        )
        fidelity_validity = _validity(
            self.held_out_validity,
            site_axis_id=self.site_axis_id,
            site_count=site_count,
            field_name="held_out_validity",
        )
        def evidence_counts(values: object, field_name: str) -> np.ndarray:
            return _immutable_array(
                values,
                dtype=np.dtype("<u8"),
                shape=(site_count,),
                field_name=field_name,
            )

        dark_success = evidence_counts(
            self.held_out_dark_success_counts,
            "held_out_dark_success_counts",
        )
        dark_total = evidence_counts(
            self.held_out_dark_total_counts,
            "held_out_dark_total_counts",
        )
        bright_success = evidence_counts(
            self.held_out_bright_success_counts,
            "held_out_bright_success_counts",
        )
        bright_total = evidence_counts(
            self.held_out_bright_total_counts,
            "held_out_bright_total_counts",
        )
        dark_lower = _float64_array(
            self.held_out_dark_accuracy_lower_bounds,
            shape=(site_count,),
            field_name="held_out_dark_accuracy_lower_bounds",
        )
        bright_lower = _float64_array(
            self.held_out_bright_accuracy_lower_bounds,
            shape=(site_count,),
            field_name="held_out_bright_accuracy_lower_bounds",
        )
        if np.any(usable.mask & ~fidelity_validity.mask):
            raise ValueError("usable sites require held-out evidence")
        if np.any((fidelity < 0.0) | (fidelity > 1.0)):
            raise ValueError("held_out_fidelity must lie in [0, 1]")
        if np.any((dark_lower < 0.0) | (dark_lower > 1.0)) or np.any(
            (bright_lower < 0.0) | (bright_lower > 1.0)
        ):
            raise ValueError("held-out class lower bounds must lie in [0, 1]")
        if np.any(dark_success > dark_total) or np.any(bright_success > bright_total):
            raise ValueError("held-out successes cannot exceed held-out totals")
        evidence = fidelity_validity.mask
        if np.any(evidence & ((dark_total == 0) | (bright_total == 0))):
            raise ValueError("valid held-out evidence requires both class totals")
        invalid_evidence = ~evidence
        for values, field_name in (
            (dark_success, "dark successes"),
            (dark_total, "dark totals"),
            (bright_success, "bright successes"),
            (bright_total, "bright totals"),
            (dark_lower, "dark lower bounds"),
            (bright_lower, "bright lower bounds"),
            (fidelity, "held-out fidelity"),
        ):
            if np.any(values[invalid_evidence] != 0):
                raise ValueError(
                    f"invalid held-out evidence requires canonical zero {field_name}"
                )
        if np.any(evidence):
            dark_empirical = dark_success[evidence] / dark_total[evidence]
            bright_empirical = bright_success[evidence] / bright_total[evidence]
            expected_fidelity = 0.5 * (dark_empirical + bright_empirical)
            if not np.allclose(
                fidelity[evidence],
                expected_fidelity,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("held_out_fidelity differs from per-class evidence")
            if np.any(dark_lower[evidence] > dark_empirical + 1e-12) or np.any(
                bright_lower[evidence] > bright_empirical + 1e-12
            ):
                raise ValueError("class lower bound exceeds empirical class accuracy")
        if np.any(dark_counts[usable.mask] == 0) or np.any(
            bright_counts[usable.mask] == 0
        ):
            raise ValueError("usable sites require both dark and bright training samples")
        if self.gate_passed and not np.any(usable.mask):
            raise ValueError("a passed quality gate requires at least one usable site")
        if self.gate_passed and np.any(usable.mask & ~fidelity_validity.mask):
            raise ValueError(
                "a passed quality gate requires held-out evidence for every usable site"
            )
        object.__setattr__(self, "dark_training_sample_counts", dark_counts)
        object.__setattr__(self, "bright_training_sample_counts", bright_counts)
        object.__setattr__(self, "usable_sites", usable)
        object.__setattr__(self, "held_out_dark_success_counts", dark_success)
        object.__setattr__(self, "held_out_dark_total_counts", dark_total)
        object.__setattr__(self, "held_out_bright_success_counts", bright_success)
        object.__setattr__(self, "held_out_bright_total_counts", bright_total)
        object.__setattr__(self, "held_out_dark_accuracy_lower_bounds", dark_lower)
        object.__setattr__(self, "held_out_bright_accuracy_lower_bounds", bright_lower)
        object.__setattr__(self, "held_out_fidelity", fidelity)
        object.__setattr__(self, "held_out_validity", fidelity_validity)


class ReadoutModelKind(str, Enum):
    BOX = "BOX"
    PER_SITE_PSF = "PER_SITE_PSF"
    UNIFORM_PSF = "UNIFORM_PSF"


class BoxReducer(str, Enum):
    SUM = "SUM"
    MEAN = "MEAN"


class BackgroundMode(str, Enum):
    NONE = "NONE"
    ANNULUS_MEDIAN = "ANNULUS_MEDIAN"


@dataclass(frozen=True, eq=False)
class ReadoutModelHeader:
    """Fields shared by every member of the closed readout-model union."""

    model_id: str
    model_version: str
    frame_contract_fingerprint: str
    site_map_fingerprint: str
    site_axis_id: AxisId
    thresholds: np.ndarray
    occupied_above_thresholds: np.ndarray
    quality: ReadoutModelQuality
    parameters: tuple[CalibrationParameter, ...] = ()
    __hash__ = None

    def __post_init__(self) -> None:
        _canonical_text(self.model_id, "model_id")
        _canonical_text(self.model_version, "model_version")
        _sha256(self.frame_contract_fingerprint, "frame_contract_fingerprint")
        _sha256(self.site_map_fingerprint, "site_map_fingerprint")
        if not isinstance(self.site_axis_id, AxisId):
            raise TypeError("site_axis_id must be AxisId")
        if not isinstance(self.quality, ReadoutModelQuality):
            raise TypeError("quality must be ReadoutModelQuality")
        if self.quality.site_axis_id != self.site_axis_id:
            raise ValueError("model header and quality name different site axes")
        site_count = self.quality.usable_sites.mask.shape[0]
        thresholds = _float64_array(
            self.thresholds,
            shape=(site_count,),
            field_name="thresholds",
        )
        if np.any(thresholds[~self.quality.usable_sites.mask] != 0.0):
            raise ValueError("unusable sites require canonical zero threshold payloads")
        occupied_above = _immutable_array(
            self.occupied_above_thresholds,
            dtype=np.dtype(bool),
            shape=(site_count,),
            field_name="occupied_above_thresholds",
        )
        if np.any(occupied_above[~self.quality.usable_sites.mask]):
            raise ValueError(
                "unusable sites require canonical False occupied-direction fillers"
            )
        parameters = tuple(self.parameters)
        if any(not isinstance(item, CalibrationParameter) for item in parameters):
            raise TypeError("parameters must contain CalibrationParameter values")
        if len({item.name for item in parameters}) != len(parameters):
            raise ValueError("model parameter names must be unique")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "occupied_above_thresholds", occupied_above)
        object.__setattr__(self, "parameters", tuple(sorted(parameters)))

    @property
    def site_count(self) -> int:
        return self.thresholds.shape[0]


def _boxes(values: object, *, site_count: int) -> np.ndarray:
    boxes = _immutable_array(
        values,
        dtype=np.dtype("<i8"),
        shape=(site_count, 4),
        field_name="boxes_xywh",
    )
    if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] <= 0):
        raise ValueError("boxes_xywh requires non-negative origins and positive extents")
    return boxes


def _require_unusable_box_fillers(
    boxes: np.ndarray,
    usable_sites: ComponentValidity,
    *,
    extent_xy: tuple[int, int] = (1, 1),
) -> None:
    expected = np.array([0, 0, *extent_xy], dtype="<i8")
    if np.any(boxes[~usable_sites.mask] != expected):
        raise ValueError("unusable sites require canonical [0,0,1,1] box fillers")


@dataclass(frozen=True, eq=False)
class BoxReadoutModel:
    header: ReadoutModelHeader
    boxes_xywh: np.ndarray
    reducer: BoxReducer
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.header, ReadoutModelHeader):
            raise TypeError("header must be ReadoutModelHeader")
        if not isinstance(self.reducer, BoxReducer):
            raise TypeError("reducer must be BoxReducer")
        object.__setattr__(
            self,
            "boxes_xywh",
            _boxes(self.boxes_xywh, site_count=self.header.site_count),
        )
        _require_unusable_box_fillers(
            self.boxes_xywh,
            self.header.quality.usable_sites,
        )

    @property
    def kind(self) -> ReadoutModelKind:
        return ReadoutModelKind.BOX


@dataclass(frozen=True, eq=False)
class PerSitePsfReadoutModel:
    header: ReadoutModelHeader
    boxes_xywh: np.ndarray
    kernels: np.ndarray
    background: BackgroundMode = BackgroundMode.ANNULUS_MEDIAN
    background_padding: int = 3
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.header, ReadoutModelHeader):
            raise TypeError("header must be ReadoutModelHeader")
        boxes = _boxes(self.boxes_xywh, site_count=self.header.site_count)
        if len({tuple(item) for item in boxes[:, 2:]}) != 1:
            raise ValueError("per-site PSF kernels require one explicit common kernel shape")
        width, height = (int(value) for value in boxes[0, 2:])
        _require_unusable_box_fillers(
            boxes,
            self.header.quality.usable_sites,
            extent_xy=(width, height),
        )
        kernels = _float64_array(
            self.kernels,
            shape=(self.header.site_count, height, width),
            field_name="kernels",
        )
        _validate_normalized_kernels(kernels, "kernels")
        if np.any(~self.header.quality.usable_sites.mask):
            filler = np.zeros((height, width), dtype="<f8")
            filler[0, 0] = 1.0
            if np.any(kernels[~self.header.quality.usable_sites.mask] != filler):
                raise ValueError(
                    "unusable sites require a canonical unit-impulse PSF kernel filler"
                )
        if not isinstance(self.background, BackgroundMode):
            raise TypeError("background must be BackgroundMode")
        padding = _nonnegative_integer(self.background_padding, "background_padding")
        if self.background is BackgroundMode.ANNULUS_MEDIAN and padding == 0:
            raise ValueError("ANNULUS_MEDIAN requires positive background_padding")
        if self.background is BackgroundMode.NONE and padding != 0:
            raise ValueError("NONE background requires canonical zero background_padding")
        object.__setattr__(self, "boxes_xywh", boxes)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "background_padding", padding)

    @property
    def kind(self) -> ReadoutModelKind:
        return ReadoutModelKind.PER_SITE_PSF


@dataclass(frozen=True, eq=False)
class UniformPsfReadoutModel:
    """One shared kernel with per-site absolute boxes; never N copied kernels."""

    header: ReadoutModelHeader
    boxes_xywh: np.ndarray
    kernel: np.ndarray
    background: BackgroundMode = BackgroundMode.ANNULUS_MEDIAN
    background_padding: int = 3
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.header, ReadoutModelHeader):
            raise TypeError("header must be ReadoutModelHeader")
        boxes = _boxes(self.boxes_xywh, site_count=self.header.site_count)
        if len({tuple(item) for item in boxes[:, 2:]}) != 1:
            raise ValueError("uniform PSF boxes must share one explicit kernel shape")
        width, height = (int(value) for value in boxes[0, 2:])
        _require_unusable_box_fillers(
            boxes,
            self.header.quality.usable_sites,
            extent_xy=(width, height),
        )
        kernel = _float64_array(
            self.kernel,
            shape=(height, width),
            field_name="kernel",
        )
        _validate_normalized_kernels(kernel[None, ...], "kernel")
        if not isinstance(self.background, BackgroundMode):
            raise TypeError("background must be BackgroundMode")
        padding = _nonnegative_integer(self.background_padding, "background_padding")
        if self.background is BackgroundMode.ANNULUS_MEDIAN and padding == 0:
            raise ValueError("ANNULUS_MEDIAN requires positive background_padding")
        if self.background is BackgroundMode.NONE and padding != 0:
            raise ValueError("NONE background requires canonical zero background_padding")
        object.__setattr__(self, "boxes_xywh", boxes)
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "background_padding", padding)

    @property
    def kind(self) -> ReadoutModelKind:
        return ReadoutModelKind.UNIFORM_PSF


def _validate_normalized_kernels(kernels: np.ndarray, field_name: str) -> None:
    if np.any(kernels < 0.0):
        raise ValueError(f"{field_name} entries must be non-negative")
    sums = np.sum(kernels, axis=(-2, -1), dtype=np.float64)
    if not np.all(np.isclose(sums, 1.0, rtol=1e-12, atol=1e-12)):
        raise ValueError(f"every {field_name} entry must have a finite unit sum")


ReadoutModel: TypeAlias = (
    BoxReadoutModel | PerSitePsfReadoutModel | UniformPsfReadoutModel
)


@dataclass(frozen=True, eq=False)
class ReadoutFeatureSpec:
    """Closed, threshold-free frame feature extraction contract.

    Calibration training and runtime application both bind one of these values
    and execute :func:`extract_readout_features`.  The pixel/background/
    validity math therefore has exactly one owner and cannot drift between the
    learned and applied signal scales.
    """

    kind: ReadoutModelKind
    site_axis_id: AxisId
    boxes_xywh: np.ndarray
    site_validity: ComponentValidity
    box_reducer: BoxReducer | None = None
    per_site_kernels: np.ndarray | None = None
    uniform_kernel: np.ndarray | None = None
    background: BackgroundMode = BackgroundMode.NONE
    background_padding: int = 0
    _fingerprint: str = field(init=False, repr=False, compare=False)
    __hash__ = None

    @classmethod
    def _from_validated_model(cls, model: ReadoutModel) -> "ReadoutFeatureSpec":
        """Zero-copy bind after the closed model's invariants were already checked."""

        result = object.__new__(cls)
        object.__setattr__(result, "kind", model.kind)
        object.__setattr__(result, "site_axis_id", model.header.site_axis_id)
        object.__setattr__(result, "boxes_xywh", model.boxes_xywh)
        object.__setattr__(result, "site_validity", model.header.quality.usable_sites)
        object.__setattr__(
            result,
            "box_reducer",
            model.reducer if isinstance(model, BoxReadoutModel) else None,
        )
        object.__setattr__(
            result,
            "per_site_kernels",
            model.kernels if isinstance(model, PerSitePsfReadoutModel) else None,
        )
        object.__setattr__(
            result,
            "uniform_kernel",
            model.kernel if isinstance(model, UniformPsfReadoutModel) else None,
        )
        object.__setattr__(
            result,
            "background",
            model.background
            if isinstance(model, (PerSitePsfReadoutModel, UniformPsfReadoutModel))
            else BackgroundMode.NONE,
        )
        object.__setattr__(
            result,
            "background_padding",
            model.background_padding
            if isinstance(model, (PerSitePsfReadoutModel, UniformPsfReadoutModel))
            else 0,
        )
        object.__setattr__(
            result,
            "_fingerprint",
            canonical_digest(_readout_feature_spec_to_tree(result)),
        )
        return result

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind")
        if not isinstance(self.site_axis_id, AxisId):
            raise TypeError("site_axis_id must be AxisId")
        source_boxes = np.asarray(self.boxes_xywh)
        if source_boxes.ndim != 2 or source_boxes.shape[1:] != (4,):
            raise ValueError("boxes_xywh must have shape (site, 4)")
        boxes = _boxes(source_boxes, site_count=source_boxes.shape[0])
        validity = _validity(
            self.site_validity,
            site_axis_id=self.site_axis_id,
            site_count=boxes.shape[0],
            field_name="feature site_validity",
        )
        padding = _nonnegative_integer(
            self.background_padding,
            "background_padding",
        )
        if not isinstance(self.background, BackgroundMode):
            raise TypeError("background must be BackgroundMode")
        if self.kind is ReadoutModelKind.BOX:
            if not isinstance(self.box_reducer, BoxReducer):
                raise TypeError("BOX feature spec requires BoxReducer")
            if self.per_site_kernels is not None or self.uniform_kernel is not None:
                raise ValueError("BOX feature spec cannot carry PSF kernels")
            if self.background is not BackgroundMode.NONE or padding != 0:
                raise ValueError("BOX feature spec requires canonical no-background state")
        else:
            if self.box_reducer is not None:
                raise ValueError("PSF feature spec cannot carry a box reducer")
            if len({tuple(item) for item in boxes[:, 2:]}) != 1:
                raise ValueError("PSF feature boxes must share one kernel shape")
            width, height = (int(value) for value in boxes[0, 2:])
            if self.kind is ReadoutModelKind.PER_SITE_PSF:
                if self.uniform_kernel is not None:
                    raise ValueError("per-site PSF feature spec cannot carry uniform kernel")
                kernels = _float64_array(
                    self.per_site_kernels,
                    shape=(boxes.shape[0], height, width),
                    field_name="per_site_kernels",
                )
                _validate_normalized_kernels(kernels, "per_site_kernels")
                object.__setattr__(self, "per_site_kernels", kernels)
            else:
                if self.per_site_kernels is not None:
                    raise ValueError("uniform PSF feature spec cannot carry per-site kernels")
                kernel = _float64_array(
                    self.uniform_kernel,
                    shape=(height, width),
                    field_name="uniform_kernel",
                )
                _validate_normalized_kernels(kernel[None, ...], "uniform_kernel")
                object.__setattr__(self, "uniform_kernel", kernel)
            if self.background is BackgroundMode.ANNULUS_MEDIAN and padding == 0:
                raise ValueError("ANNULUS_MEDIAN requires positive background_padding")
            if self.background is BackgroundMode.NONE and padding != 0:
                raise ValueError("NONE background requires canonical zero padding")
        object.__setattr__(self, "boxes_xywh", boxes)
        object.__setattr__(self, "site_validity", validity)
        object.__setattr__(self, "background_padding", padding)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(_readout_feature_spec_to_tree(self)),
        )

    @property
    def fingerprint(self) -> str:
        """Canonical identity of every field that changes extracted signals."""

        current = canonical_digest(_readout_feature_spec_to_tree(self))
        if current != self._fingerprint:
            raise ValueError("readout feature spec changed after construction")
        return current


def _readout_feature_spec_to_tree(spec: ReadoutFeatureSpec) -> dict[str, object]:
    """Project the closed feature-math contract to its owner-defined tree."""

    if not isinstance(spec, ReadoutFeatureSpec):
        raise TypeError("spec must be ReadoutFeatureSpec")
    return {
        "schema": "zlc_neutral_atom.readout-feature-spec.v1",
        "kind": spec.kind.value,
        "site_axis_id": spec.site_axis_id.value,
        "boxes_xywh": spec.boxes_xywh,
        "site_validity": validity_to_tree(spec.site_validity),
        "box_reducer": None if spec.box_reducer is None else spec.box_reducer.value,
        "per_site_kernels": spec.per_site_kernels,
        "uniform_kernel": spec.uniform_kernel,
        "background": spec.background.value,
        "background_padding": spec.background_padding,
    }


def validate_readout_feature_spec_model(
    spec: ReadoutFeatureSpec,
    model: ReadoutModel,
) -> None:
    """Fail closed unless ``spec`` is exactly the model's feature-math contract."""

    if not isinstance(spec, ReadoutFeatureSpec):
        raise TypeError("spec must be ReadoutFeatureSpec")
    model = _model(model)
    expected = ReadoutFeatureSpec._from_validated_model(model)
    if spec.fingerprint != expected.fingerprint:
        raise ValueError("readout feature spec does not match the selected model")


class CalibrationResourceExceeded(ValueError):
    """A calibration value exceeds an explicit decode/application budget."""


@dataclass(frozen=True)
class CalibrationResourcePolicy:
    max_manifest_bytes: int = 64 * 1024
    max_artifact_blob_bytes: int = 256 * 1024 * 1024
    max_models: int = 32
    max_sites: int = 100_000
    max_kernel_elements: int = 20_000_000
    max_sampled_pixels_per_model: int = 50_000_000
    max_total_sampled_pixels_all_models: int = 200_000_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_manifest_bytes",
            "max_artifact_blob_bytes",
            "max_models",
            "max_sites",
            "max_kernel_elements",
            "max_sampled_pixels_per_model",
            "max_total_sampled_pixels_all_models",
        ):
            value = _nonnegative_integer(getattr(self, field_name), field_name)
            if value == 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)


DEFAULT_CALIBRATION_RESOURCE_POLICY = CalibrationResourcePolicy()


@dataclass(frozen=True)
class CalibrationResourceSummary:
    site_count: int
    model_count: int
    kernel_elements: int
    max_sampled_pixels_per_model: int
    total_sampled_pixels_all_models: int

    def __post_init__(self) -> None:
        for field_name in (
            "site_count",
            "model_count",
            "kernel_elements",
            "max_sampled_pixels_per_model",
            "total_sampled_pixels_all_models",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(getattr(self, field_name), field_name),
            )


def _resource_policy(value: object) -> CalibrationResourcePolicy:
    if not isinstance(value, CalibrationResourcePolicy):
        raise TypeError("resource_policy must be CalibrationResourcePolicy")
    return value


def _checked_box_slices(
    box: np.ndarray,
    *,
    image_shape_yx: tuple[int, int],
    site_index: int,
) -> tuple[slice, slice]:
    """Convert to Python ints before checked addition; never trust int64 wraparound."""

    values = tuple(int(value) for value in np.asarray(box))
    if len(values) != 4:
        raise ValueError(f"model site {site_index} box must have four XYWH entries")
    x0, y0, width, height = values
    if x0 < 0 or y0 < 0 or width <= 0 or height <= 0:
        raise ValueError(f"model site {site_index} box has invalid origin or extent")
    image_height, image_width = (int(value) for value in image_shape_yx)
    x1 = x0 + width
    y1 = y0 + height
    if x1 > image_width or y1 > image_height:
        raise ValueError(
            f"model site {site_index} box lies outside the FrameContract image geometry"
        )
    return slice(y0, y1), slice(x0, x1)


def _validate_site_map_geometry(
    site_map: SiteMap,
    frame_contract: FrameContract,
) -> None:
    if site_map.coordinate_frame != frame_contract.coordinate_frame:
        raise ValueError("SiteMap and FrameContract use different coordinate frames")
    image_height, image_width = frame_contract.frame_schema.data_shape
    coordinates = site_map.coordinates_xy
    if np.any(
        (coordinates[:, 0] < 0.0)
        | (coordinates[:, 0] >= image_width)
        | (coordinates[:, 1] < 0.0)
        | (coordinates[:, 1] >= image_height)
    ):
        raise ValueError("SiteMap coordinates lie outside the FrameContract image")


def _validate_model_geometry(
    model: ReadoutModel,
    site_map: SiteMap,
    frame_contract: FrameContract,
) -> None:
    if model.header.site_count != site_map.site_axis.size:
        raise ValueError("readout model and SiteMap have different site counts")
    _validate_site_map_geometry(site_map, frame_contract)
    image_shape = frame_contract.frame_schema.data_shape
    coordinates = site_map.coordinates_xy
    for site_index, box in enumerate(model.boxes_xywh):
        y_slice, x_slice = _checked_box_slices(
            box,
            image_shape_yx=image_shape,
            site_index=site_index,
        )
        if model.header.quality.usable_sites.mask[site_index]:
            center_x, center_y = (float(value) for value in coordinates[site_index])
            if not (
                x_slice.start <= center_x < x_slice.stop
                and y_slice.start <= center_y < y_slice.stop
            ):
                raise ValueError(
                    f"model {model.header.model_id!r} site {site_index} center "
                    "does not lie inside its extraction box"
                )


def _model_kernel_elements(model: ReadoutModel) -> int:
    if isinstance(model, PerSitePsfReadoutModel):
        return int(model.kernels.size)
    if isinstance(model, UniformPsfReadoutModel):
        return int(model.kernel.size)
    return 0


def _model_sampled_pixels(
    model: ReadoutModel,
    image_shape_yx: tuple[int, int],
) -> int:
    image_height, image_width = image_shape_yx
    total = 0
    usable = model.header.quality.usable_sites.mask
    for site_index, box in enumerate(model.boxes_xywh):
        y_slice, x_slice = _checked_box_slices(
            box,
            image_shape_yx=image_shape_yx,
            site_index=site_index,
        )
        if not usable[site_index]:
            continue
        if isinstance(model, BoxReadoutModel) or model.background is BackgroundMode.NONE:
            sampled = (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)
        else:
            padding = int(model.background_padding)
            outer_y0 = max(0, y_slice.start - padding)
            outer_x0 = max(0, x_slice.start - padding)
            outer_y1 = min(image_height, y_slice.stop + padding)
            outer_x1 = min(image_width, x_slice.stop + padding)
            sampled = (outer_y1 - outer_y0) * (outer_x1 - outer_x0)
        total += int(sampled)
    return total


def validate_readout_model_resources(
    model: ReadoutModel,
    *,
    image_shape_yx: tuple[int, int],
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> None:
    model = _model(model)
    policy = _resource_policy(resource_policy)
    if model.header.site_count > policy.max_sites:
        raise CalibrationResourceExceeded("readout model site count exceeds resource policy")
    if _model_kernel_elements(model) > policy.max_kernel_elements:
        raise CalibrationResourceExceeded("readout model kernel elements exceed resource policy")
    if _model_sampled_pixels(model, image_shape_yx) > policy.max_sampled_pixels_per_model:
        raise CalibrationResourceExceeded("readout model sampled pixels exceed resource policy")


def _model(value: object) -> ReadoutModel:
    if not isinstance(
        value,
        (BoxReadoutModel, PerSitePsfReadoutModel, UniformPsfReadoutModel),
    ):
        raise TypeError("models must contain a closed ReadoutModel value")
    return value


class CalibrationCapability(str, Enum):
    SITE_MAP = "SITE_MAP"
    THRESHOLDS = "THRESHOLDS"
    BOX_READOUT = "BOX_READOUT"
    PER_SITE_PSF_READOUT = "PER_SITE_PSF_READOUT"
    UNIFORM_PSF_READOUT = "UNIFORM_PSF_READOUT"


class CalibrationStage(str, Enum):
    SITE_MAP_ONLY = "SITE_MAP_ONLY"
    THRESHOLDS_READY = "THRESHOLDS_READY"
    COMPLETE = "COMPLETE"


_CAPABILITY_BY_KIND = {
    ReadoutModelKind.BOX: CalibrationCapability.BOX_READOUT,
    ReadoutModelKind.PER_SITE_PSF: CalibrationCapability.PER_SITE_PSF_READOUT,
    ReadoutModelKind.UNIFORM_PSF: CalibrationCapability.UNIFORM_PSF_READOUT,
}


@dataclass(frozen=True)
class DefaultModelPolicy:
    """Versioned, order-independent default selection policy."""

    policy_id: str
    policy_version: str
    default_model_id: str | None = None
    default_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.policy_id, "default policy id")
        _canonical_text(self.policy_version, "default policy version")
        if self.default_model_id is not None:
            _canonical_text(self.default_model_id, "default_model_id")
        if self.default_kind is not None and not isinstance(
            self.default_kind, ReadoutModelKind
        ):
            raise TypeError("default_kind must be ReadoutModelKind or None")
        if self.default_model_id is not None and self.default_kind is not None:
            raise ValueError("default policy cannot name both a model id and a kind")


@dataclass(frozen=True, eq=False)
class CalibrationArtifact:
    """Immutable calibration result; repository identity lives in its typed ref."""

    source_binding: CalibrationSourceBinding
    frame_contract: FrameContract
    site_map: SiteMap
    models: tuple[ReadoutModel, ...]
    stage: CalibrationStage
    required_model_kinds: tuple[ReadoutModelKind, ...]
    default_model_policy: DefaultModelPolicy
    algorithm_id: str
    algorithm_version: str
    parameters: tuple[CalibrationParameter, ...] = ()
    _capabilities: tuple[CalibrationCapability, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _fingerprint: str = field(init=False, repr=False, compare=False)
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_binding, CalibrationSourceBinding):
            raise TypeError("source_binding must be CalibrationSourceBinding")
        if not isinstance(self.frame_contract, FrameContract):
            raise TypeError("frame_contract must be FrameContract")
        if (
            self.source_binding.frame_contract_fingerprint
            != self.frame_contract.fingerprint
        ):
            raise ValueError(
                "calibration source binding and FrameContract fingerprints differ"
            )
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        if not isinstance(self.stage, CalibrationStage):
            raise TypeError("stage must be CalibrationStage")
        if not isinstance(self.default_model_policy, DefaultModelPolicy):
            raise TypeError("default_model_policy must be DefaultModelPolicy")
        _canonical_text(self.algorithm_id, "algorithm_id")
        _canonical_text(self.algorithm_version, "algorithm_version")
        models = tuple(_model(model) for model in self.models)
        if len({model.header.model_id for model in models}) != len(models):
            raise ValueError("calibration model ids must be unique")
        models = tuple(sorted(models, key=lambda model: model.header.model_id))
        required = tuple(self.required_model_kinds)
        if any(not isinstance(kind, ReadoutModelKind) for kind in required):
            raise TypeError("required_model_kinds must contain ReadoutModelKind values")
        if len(set(required)) != len(required):
            raise ValueError("required_model_kinds must be unique")
        required = tuple(sorted(required, key=lambda kind: kind.value))
        parameters = tuple(self.parameters)
        if any(not isinstance(item, CalibrationParameter) for item in parameters):
            raise TypeError("parameters must contain CalibrationParameter values")
        if len({item.name for item in parameters}) != len(parameters):
            raise ValueError("artifact parameter names must be unique")
        parameters = tuple(sorted(parameters))

        frame_fingerprint = self.frame_contract.fingerprint
        site_fingerprint = self.site_map.fingerprint
        site_axis_id = self.site_map.site_axis.axis_id
        site_valid = self.site_map.validity.mask
        _validate_site_map_geometry(self.site_map, self.frame_contract)
        for model in models:
            header = model.header
            if header.frame_contract_fingerprint != frame_fingerprint:
                raise ValueError(f"model {header.model_id!r} belongs to another FrameContract")
            if header.site_map_fingerprint != site_fingerprint:
                raise ValueError(f"model {header.model_id!r} belongs to another SiteMap")
            if header.site_axis_id != site_axis_id:
                raise ValueError(f"model {header.model_id!r} names another site axis")
            if header.site_count != self.site_map.site_axis.size:
                raise ValueError(f"model {header.model_id!r} has the wrong site count")
            if np.any(header.quality.usable_sites.mask & ~site_valid):
                raise ValueError(f"model {header.model_id!r} marks an invalid map site usable")
            if not header.quality.gate_passed:
                raise ValueError(f"model {header.model_id!r} did not pass its quality gate")
            _validate_model_geometry(model, self.site_map, self.frame_contract)

        present_kinds = {model.kind for model in models}
        if self.stage is CalibrationStage.SITE_MAP_ONLY:
            if models or required:
                raise ValueError("SITE_MAP_ONLY cannot contain models or required model kinds")
        elif self.stage is CalibrationStage.THRESHOLDS_READY:
            if not models or required:
                raise ValueError(
                    "THRESHOLDS_READY requires models and no completeness requirement"
                )
        else:
            if not required:
                raise ValueError("COMPLETE requires at least one required model kind")
            missing = set(required) - present_kinds
            if missing:
                raise ValueError(
                    "COMPLETE artifact is missing required model kinds: "
                    + ", ".join(sorted(kind.value for kind in missing))
                )

        capabilities = {CalibrationCapability.SITE_MAP}
        if models:
            capabilities.add(CalibrationCapability.THRESHOLDS)
            capabilities.update(_CAPABILITY_BY_KIND[model.kind] for model in models)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "required_model_kinds", required)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "_capabilities",
            tuple(sorted(capabilities, key=lambda capability: capability.value)),
        )
        # Validate a declared default at artifact construction, not first use.
        if models and (
            self.default_model_policy.default_model_id is not None
            or self.default_model_policy.default_kind is not None
        ):
            self.select_model()
        elif (
            self.default_model_policy.default_model_id is not None
            or self.default_model_policy.default_kind is not None
        ):
            raise ValueError("a site-map-only artifact cannot name a default readout model")
        from .calibration_codec import calibration_artifact_to_tree

        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(calibration_artifact_to_tree(self)),
        )

    @property
    def capabilities(self) -> tuple[CalibrationCapability, ...]:
        return self._capabilities

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def select_model(
        self,
        *,
        model_id: str | None = None,
        kind: ReadoutModelKind | None = None,
    ) -> ReadoutModel:
        """Resolve explicit selection or the stable default; ambiguity is fatal."""

        if model_id is not None:
            _canonical_text(model_id, "model_id")
        if kind is not None and not isinstance(kind, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind or None")
        if model_id is not None and kind is not None:
            raise ValueError("select_model accepts either model_id or kind, not both")
        if model_id is not None:
            candidates = tuple(
                model for model in self.models if model.header.model_id == model_id
            )
        elif kind is not None:
            candidates = tuple(model for model in self.models if model.kind is kind)
        else:
            policy = self.default_model_policy
            if policy.default_model_id is not None:
                candidates = tuple(
                    model
                    for model in self.models
                    if model.header.model_id == policy.default_model_id
                )
            elif policy.default_kind is not None:
                candidates = tuple(
                    model for model in self.models if model.kind is policy.default_kind
                )
            else:
                candidates = self.models
        if len(candidates) != 1:
            description = model_id or (kind.value if kind is not None else "default policy")
            raise ValueError(
                f"model selection {description!r} resolved to {len(candidates)} models; "
                "selection must be unique"
            )
        return candidates[0]


def validate_calibration_artifact_resources(
    artifact: CalibrationArtifact,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> None:
    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    policy = _resource_policy(resource_policy)
    validate_calibration_resource_summary(
        calibration_resource_summary(artifact),
        policy,
    )


def calibration_resource_summary(
    artifact: CalibrationArtifact,
) -> CalibrationResourceSummary:
    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    image_shape = artifact.frame_contract.frame_schema.data_shape
    sampled = tuple(
        _model_sampled_pixels(model, image_shape) for model in artifact.models
    )
    return CalibrationResourceSummary(
        artifact.site_map.site_axis.size,
        len(artifact.models),
        sum(_model_kernel_elements(model) for model in artifact.models),
        max(sampled, default=0),
        sum(sampled),
    )


def validate_calibration_resource_summary(
    summary: CalibrationResourceSummary,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> None:
    if not isinstance(summary, CalibrationResourceSummary):
        raise TypeError("summary must be CalibrationResourceSummary")
    policy = _resource_policy(resource_policy)
    if summary.site_count > policy.max_sites:
        raise CalibrationResourceExceeded("calibration site count exceeds resource policy")
    if summary.model_count > policy.max_models:
        raise CalibrationResourceExceeded("calibration model count exceeds resource policy")
    if summary.kernel_elements > policy.max_kernel_elements:
        raise CalibrationResourceExceeded(
            "calibration total kernel elements exceed resource policy"
        )
    if summary.max_sampled_pixels_per_model > policy.max_sampled_pixels_per_model:
        raise CalibrationResourceExceeded(
            "calibration sampled pixels exceed resource policy"
        )
    if (
        summary.total_sampled_pixels_all_models
        > policy.max_total_sampled_pixels_all_models
    ):
        raise CalibrationResourceExceeded(
            "calibration total sampled pixels exceed resource policy"
        )


@dataclass(frozen=True, eq=False)
class ReadoutSignals:
    site_axis_id: AxisId
    values: np.ndarray
    validity: ComponentValidity
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis_id, AxisId):
            raise TypeError("site_axis_id must be AxisId")
        source = np.asarray(self.values)
        if source.ndim != 1:
            raise ValueError("readout signal values must be a one-dimensional site vector")
        values = _float64_array(
            source,
            shape=source.shape,
            field_name="readout signal values",
        )
        validity = _validity(
            self.validity,
            site_axis_id=self.site_axis_id,
            site_count=values.shape[0],
            field_name="readout signal validity",
        )
        if np.any(values[~validity.mask] != 0.0):
            raise ValueError("invalid readout signals require canonical zero payloads")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True, eq=False)
class OccupancyDecision:
    signals: ReadoutSignals
    thresholds: np.ndarray
    occupied: np.ndarray
    validity: ComponentValidity
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.signals, ReadoutSignals):
            raise TypeError("signals must be ReadoutSignals")
        count = self.signals.values.shape[0]
        thresholds = _float64_array(
            self.thresholds,
            shape=(count,),
            field_name="occupancy thresholds",
        )
        occupied = _immutable_array(
            self.occupied,
            dtype=np.dtype(bool),
            shape=(count,),
            field_name="occupied",
        )
        validity = _validity(
            self.validity,
            site_axis_id=self.signals.site_axis_id,
            site_count=count,
            field_name="occupancy validity",
        )
        if not np.array_equal(validity.mask, self.signals.validity.mask):
            raise ValueError("occupancy and signal validity must be identical")
        if np.any(occupied[~validity.mask]):
            raise ValueError("invalid occupancy entries require canonical False fillers")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "validity", validity)


def bind_readout_feature_spec(
    model: ReadoutModel,
    site_map: SiteMap,
) -> ReadoutFeatureSpec:
    """Bind model geometry and admitted sites without copying classifier state."""

    model = _model(model)
    if not isinstance(site_map, SiteMap):
        raise TypeError("site_map must be SiteMap")
    if not model.header.quality.gate_passed:
        raise ValueError("readout model did not pass its quality gate")
    if model.header.site_map_fingerprint != site_map.fingerprint:
        raise ValueError("readout model does not apply to this SiteMap")
    if model.header.site_axis_id != site_map.site_axis.axis_id:
        raise ValueError("readout model and SiteMap name different site axes")
    usable = model.header.quality.usable_sites
    if np.any(usable.mask & ~site_map.validity.mask):
        raise ValueError("readout model marks an invalid SiteMap site usable")
    return ReadoutFeatureSpec._from_validated_model(model)


def readout_application_scratch_nbytes(
    spec: ReadoutFeatureSpec,
    frame_schema: ValueSchema,
) -> int:
    """Bound transient numerical buffers for the serial readout operator.

    The bound includes feature/classifier vectors, immutable-freeze copies,
    masked-index workspaces, and the largest one-site BOX/PSF/background window.
    It excludes the retained input frame, immutable calibration arrays, the final
    occupancy payload, and fixed interpreter/allocator overhead; those have
    separate owners in pipeline admission.  The maximum window is correct only
    while feature extraction remains the serial site loop below.
    """

    if not isinstance(spec, ReadoutFeatureSpec):
        raise TypeError("spec must be ReadoutFeatureSpec")
    if not isinstance(frame_schema, ValueSchema):
        raise TypeError("frame_schema must be ValueSchema")
    # Re-derive the frozen feature identity so reflective mutation cannot lower
    # a safety bound without first failing the same contract used by execution.
    _ = spec.fingerprint
    if len(frame_schema.data_shape) != 2:
        raise ValueError("readout scratch requires one two-dimensional frame schema")
    if frame_schema.dtype.kind not in "iuf":
        raise TypeError("readout frame dtype must be a real integer or floating dtype")

    image_height, image_width = (
        int(value) for value in frame_schema.data_shape
    )
    frame_itemsize = int(frame_schema.dtype.itemsize)
    index_itemsize = int(np.dtype(np.intp).itemsize)
    site_count = int(spec.boxes_xywh.shape[0])
    maximum_window = 0
    for site_index, box in enumerate(spec.boxes_xywh):
        y_slice, x_slice = _checked_box_slices(
            box,
            image_shape_yx=(image_height, image_width),
            site_index=site_index,
        )
        if not spec.site_validity.mask[site_index]:
            continue
        kernel_pixels = int(
            (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)
        )
        if spec.kind is ReadoutModelKind.BOX:
            window = 8 * kernel_pixels
        elif spec.background is BackgroundMode.NONE:
            window = 24 * kernel_pixels
        else:
            padding = int(spec.background_padding)
            outer_y0 = max(0, y_slice.start - padding)
            outer_x0 = max(0, x_slice.start - padding)
            outer_y1 = min(image_height, y_slice.stop + padding)
            outer_x1 = min(image_width, x_slice.stop + padding)
            outer_pixels = int(
                (outer_y1 - outer_y0) * (outer_x1 - outer_x0)
            )
            ring_pixels = outer_pixels - kernel_pixels
            annulus = outer_pixels + (
                2 * frame_itemsize + 2 + 2 * index_itemsize
            ) * ring_pixels
            window = max(24 * kernel_pixels, annulus)
        maximum_window = max(maximum_window, window)

    # 96 bytes/site covers the current maximum live set of signal values,
    # validity/classifier masks, immutable freezes, and advanced-index work.
    return 96 * site_count + maximum_window


def extract_readout_features(
    spec: ReadoutFeatureSpec,
    frame: Value,
) -> ReadoutSignals:
    """Strict public Value boundary over the one feature-math implementation."""

    if not isinstance(spec, ReadoutFeatureSpec):
        raise TypeError("spec must be ReadoutFeatureSpec")
    if not isinstance(frame, Value):
        raise TypeError(
            "frame must be zlc_data.Value so named axes and validity cannot be discarded"
        )
    return _extract_readout_features_arrays(
        spec,
        frame.values,
        expand_value_validity(frame.validity, frame.schema),
    )


def _extract_readout_features_arrays(
    spec: ReadoutFeatureSpec,
    image: np.ndarray,
    pixel_validity: np.ndarray,
) -> ReadoutSignals:
    """Package-private array core shared by strict runtime and borrowed analysis."""

    if not isinstance(spec, ReadoutFeatureSpec):
        raise TypeError("spec must be ReadoutFeatureSpec")
    image = np.asarray(image)
    validity_source = np.asarray(pixel_validity)
    if image.ndim != 2:
        raise ValueError("readout features require one two-dimensional Y,X Value")
    if validity_source.dtype != np.dtype(bool) or validity_source.shape != image.shape:
        raise ValueError("pixel_validity must be a bool mask matching the Y,X image")
    valid = spec.site_validity.mask.copy()
    values = np.zeros(spec.boxes_xywh.shape[0], dtype="<f8")
    for index, box in enumerate(spec.boxes_xywh):
        if not valid[index]:
            continue
        y_slice, x_slice = _checked_box_slices(
            box,
            image_shape_yx=image.shape,
            site_index=index,
        )
        cut = image[y_slice, x_slice]
        cut_validity = validity_source[y_slice, x_slice]
        if not np.all(cut_validity) or not np.all(np.isfinite(cut)):
            valid[index] = False
            continue
        if spec.kind is ReadoutModelKind.BOX:
            result = (
                np.sum(cut, dtype=np.float64)
                if spec.box_reducer is BoxReducer.SUM
                else np.mean(cut, dtype=np.float64)
            )
        else:
            background = _background_value(
                image,
                validity_source,
                y_slice,
                x_slice,
                mode=spec.background,
                padding=spec.background_padding,
            )
            if background is None:
                valid[index] = False
                continue
            kernel = (
                spec.per_site_kernels[index]
                if spec.kind is ReadoutModelKind.PER_SITE_PSF
                else spec.uniform_kernel
            )
            assert kernel is not None
            result = np.sum(
                kernel * (cut.astype(np.float64) - background),
                dtype=np.float64,
            )
        if not np.isfinite(result):
            valid[index] = False
            continue
        values[index] = float(result)
    values[~valid] = 0.0
    return ReadoutSignals(
        spec.site_axis_id,
        values,
        ComponentValidity((spec.site_axis_id,), valid),
    )


def readout_background_value(
    frame: Value,
    box_xywh: np.ndarray | tuple[int, int, int, int],
    *,
    mode: BackgroundMode,
    padding: int,
) -> float | None:
    """Public owner primitive for calibration-time PSF template preparation."""

    if not isinstance(frame, Value):
        raise TypeError("frame must be zlc_data.Value")
    return _readout_background_from_arrays(
        frame.values,
        expand_value_validity(frame.validity, frame.schema),
        box_xywh,
        mode=mode,
        padding=padding,
    )


def _readout_background_from_arrays(
    image: np.ndarray,
    pixel_validity: np.ndarray,
    box_xywh: np.ndarray | tuple[int, int, int, int],
    *,
    mode: BackgroundMode,
    padding: int,
) -> float | None:
    """Package-private array owner used by training-only averaged templates."""

    image = np.asarray(image)
    validity = np.asarray(pixel_validity, dtype=bool)
    if image.ndim != 2 or validity.shape != image.shape:
        raise ValueError("background image and validity must share one Y,X shape")
    if not isinstance(mode, BackgroundMode):
        raise TypeError("mode must be BackgroundMode")
    pad = _nonnegative_integer(padding, "padding")
    if mode is BackgroundMode.ANNULUS_MEDIAN and pad == 0:
        raise ValueError("ANNULUS_MEDIAN requires positive padding")
    if mode is BackgroundMode.NONE and pad != 0:
        raise ValueError("NONE requires canonical zero padding")
    y_slice, x_slice = _checked_box_slices(
        np.asarray(box_xywh),
        image_shape_yx=image.shape,
        site_index=0,
    )
    return _background_value(
        image,
        validity,
        y_slice,
        x_slice,
        mode=mode,
        padding=pad,
    )


def extract_readout_signals(
    model: ReadoutModel,
    *,
    frame_contract: FrameContract,
    site_map: SiteMap,
    frame: Value,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> ReadoutSignals:
    """Apply one model to exactly one Y,X frame without implicit batch/axis guessing."""

    model = _model(model)
    if not isinstance(frame_contract, FrameContract):
        raise TypeError("frame_contract must be FrameContract")
    validate_readout_model_resources(
        model,
        image_shape_yx=frame_contract.frame_schema.data_shape,
        resource_policy=resource_policy,
    )
    _assert_applicable(model, frame_contract=frame_contract, site_map=site_map)
    if not isinstance(frame, Value):
        raise TypeError(
            "frame must be zlc_data.Value so named axes and validity cannot be discarded"
        )
    if frame.schema.fingerprint != frame_contract.frame_schema.fingerprint:
        raise ValueError("frame ValueSchema differs from the FrameContract schema")
    return extract_readout_features(bind_readout_feature_spec(model, site_map), frame)


def classify_occupancy(model: ReadoutModel, signals: ReadoutSignals) -> OccupancyDecision:
    """Threshold valid signals; invalid fillers remain explicitly invalid, never dark."""

    model = _model(model)
    if not isinstance(signals, ReadoutSignals):
        raise TypeError("signals must be ReadoutSignals")
    if not model.header.quality.gate_passed:
        raise ValueError("readout model did not pass its quality gate")
    if signals.site_axis_id != model.header.site_axis_id:
        raise ValueError("signals and model name different site axes")
    if signals.values.shape != model.header.thresholds.shape:
        raise ValueError("signals and model have different site counts")
    if np.any(signals.validity.mask & ~model.header.quality.usable_sites.mask):
        raise ValueError("signals claim validity for a site the model declares unusable")
    valid = signals.validity.mask.copy()
    occupied = np.zeros(signals.values.shape, dtype=bool)
    above = valid & model.header.occupied_above_thresholds
    below = valid & ~model.header.occupied_above_thresholds
    occupied[above] = signals.values[above] > model.header.thresholds[above]
    occupied[below] = signals.values[below] < model.header.thresholds[below]
    return OccupancyDecision(
        signals,
        model.header.thresholds,
        occupied,
        ComponentValidity((signals.site_axis_id,), valid),
    )


def apply_readout_model(
    model: ReadoutModel,
    *,
    frame_contract: FrameContract,
    site_map: SiteMap,
    frame: Value,
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
) -> OccupancyDecision:
    """Extract and classify one frame through the single exact application path."""

    signals = extract_readout_signals(
        model,
        frame_contract=frame_contract,
        site_map=site_map,
        frame=frame,
        resource_policy=resource_policy,
    )
    return classify_occupancy(model, signals)


def _assert_applicable(
    model: ReadoutModel,
    *,
    frame_contract: FrameContract,
    site_map: SiteMap,
) -> None:
    if not isinstance(frame_contract, FrameContract):
        raise TypeError("frame_contract must be FrameContract")
    if not isinstance(site_map, SiteMap):
        raise TypeError("site_map must be SiteMap")
    header = model.header
    if not header.quality.gate_passed:
        raise ValueError("readout model did not pass its quality gate")
    if header.frame_contract_fingerprint != frame_contract.fingerprint:
        raise ValueError("readout model does not apply to this FrameContract")
    if header.site_map_fingerprint != site_map.fingerprint:
        raise ValueError("readout model does not apply to this SiteMap")
    if header.site_axis_id != site_map.site_axis.axis_id:
        raise ValueError("readout model and SiteMap name different site axes")
    _validate_model_geometry(model, site_map, frame_contract)
    if np.any(header.quality.usable_sites.mask & ~site_map.validity.mask):
        raise ValueError("readout model marks an invalid SiteMap site usable")


def _background_value(
    image: np.ndarray,
    pixel_validity: np.ndarray,
    y_slice: slice,
    x_slice: slice,
    *,
    mode: BackgroundMode,
    padding: int,
) -> float | None:
    if mode is BackgroundMode.NONE:
        return 0.0
    outer_x0 = max(0, x_slice.start - padding)
    outer_y0 = max(0, y_slice.start - padding)
    outer_x1 = min(image.shape[1], x_slice.stop + padding)
    outer_y1 = min(image.shape[0], y_slice.stop + padding)
    outer = image[outer_y0:outer_y1, outer_x0:outer_x1]
    outer_validity = pixel_validity[outer_y0:outer_y1, outer_x0:outer_x1]
    ring_mask = np.ones(outer.shape, dtype=bool)
    inner_y0 = y_slice.start - outer_y0
    inner_x0 = x_slice.start - outer_x0
    ring_mask[
        inner_y0 : inner_y0 + (y_slice.stop - y_slice.start),
        inner_x0 : inner_x0 + (x_slice.stop - x_slice.start),
    ] = False
    ring = outer[ring_mask]
    ring_validity = outer_validity[ring_mask]
    if ring.size == 0 or not np.all(ring_validity) or not np.all(np.isfinite(ring)):
        return None
    result = float(np.median(ring))
    return result if math.isfinite(result) else None


__all__ = [
    "BackgroundMode",
    "BoxReadoutModel",
    "BoxReducer",
    "CalibrationArtifact",
    "CalibrationCapability",
    "CalibrationParameter",
    "CalibrationResourceExceeded",
    "CalibrationResourcePolicy",
    "CalibrationResourceSummary",
    "CalibrationSourceBinding",
    "CalibrationStage",
    "DEFAULT_CALIBRATION_RESOURCE_POLICY",
    "DefaultModelPolicy",
    "OccupancyDecision",
    "PerSitePsfReadoutModel",
    "ReadoutModel",
    "ReadoutModelHeader",
    "ReadoutModelKind",
    "ReadoutModelQuality",
    "ReadoutFeatureSpec",
    "ReadoutSignals",
    "SiteMap",
    "UniformPsfReadoutModel",
    "apply_readout_model",
    "bind_readout_feature_spec",
    "calibration_resource_summary",
    "derive_calibration_source_binding",
    "classify_occupancy",
    "extract_readout_signals",
    "extract_readout_features",
    "readout_application_scratch_nbytes",
    "readout_background_value",
    "validate_calibration_artifact_resources",
    "validate_calibration_artifact_source_compatibility",
    "validate_calibration_resource_summary",
    "validate_readout_feature_spec_model",
    "validate_readout_model_resources",
]
