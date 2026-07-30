"""Immutable readout calibration values and the one runtime application path.

Calibration stores only facts needed to reproduce readout: the source capture,
the complete camera frame contract, one site map, and a closed set of feature
models.  Statistical diagnostics belong to :mod:`.analysis`; direct record-last
durability belongs to :mod:`.repository`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
import math
from typing import TypeAlias

import numpy as np

from zlc_data import (
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    Value,
    ValueSchema,
    ValidityContract,
)
from zlc_data._arrays import immutable_array
from zlc_data.value import expand_value_validity
from zlc_neutral_atom.capture.frames import CaptureFrameSource
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_storage import (
    finite_real as _finite_float,
    nonnegative_integer as _nonnegative_integer,
    positive_integer as _positive_integer,
)
from zlc_neutral_atom.authoring import (
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import (
    ReadoutModelKind as _ReadoutModelKind,
)

from .reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    ReadoutPhysicalContext,
    _derive_readout_physical_context_from_evidence,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import (
    CalibrationCaptureLayout,
    FrameContract,
    _minimum_coordinate_separation,
    _CalibrationCaptureJoin,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureDescriptor,
    ReadoutBindingKey,
)


class GridOrder(str, Enum):
    ROW_MAJOR = "row-major"
    SERPENTINE = "serpentine"
    COLUMN_MAJOR = "column-major"
    COLUMN_SERPENTINE = "column-serpentine"


def site_grid_positions_yx(
    grid_shape_yx: tuple[int, int],
    ordering: GridOrder,
) -> tuple[tuple[int, int], ...]:
    """Return physical ``(row, column)`` positions in canonical site order.

    Site-valued arrays always remain one-dimensional and follow ``ordering``;
    this projection is the one domain-owned bridge to a physical grid view.
    """

    try:
        raw_grid = tuple(grid_shape_yx)
    except TypeError as exc:
        raise ValueError("grid_shape_yx must contain two positive integers") from exc
    if len(raw_grid) != 2:
        raise ValueError("grid_shape_yx must contain two positive integers")
    rows = _positive_integer(raw_grid[0], "grid_shape_yx[0]")
    columns = _positive_integer(raw_grid[1], "grid_shape_yx[1]")
    if not isinstance(ordering, GridOrder):
        raise TypeError("ordering must be GridOrder")

    if ordering in (GridOrder.ROW_MAJOR, GridOrder.SERPENTINE):
        return tuple(
            (row, column)
            for row in range(rows)
            for column in (
                range(columns - 1, -1, -1)
                if ordering is GridOrder.SERPENTINE and row % 2
                else range(columns)
            )
        )
    return tuple(
        (row, column)
        for column in range(columns)
        for row in (
            range(rows - 1, -1, -1)
            if ordering is GridOrder.COLUMN_SERPENTINE and column % 2
            else range(rows)
        )
    )


class BoxReducer(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    MAX = "max"


class BackgroundMode(str, Enum):
    NONE = "none"
    ANNULUS_MEDIAN = "annulus"


class ThresholdMethod(str, Enum):
    """Fallback threshold estimator when bracket labels cannot train a site.

    Long-short-long reference labels remain the authoritative threshold source
    whenever they are usable.  This choice controls the quick ``otsu`` versus
    ``bimodal`` fallback; it is analysis intent, not a GUI-only preference.
    """

    OTSU = "otsu"
    BIMODAL = "bimodal"


# Public request constraints.  Presentation projects these values; the
# request constructor below remains the only validator.
CALIBRATION_MINIMUM_BOX_RADIUS = 0
CALIBRATION_MINIMUM_PSF_HALF_WIDTH = 0
CALIBRATION_MINIMUM_PSF_BACKGROUND_PADDING = 1
CALIBRATION_MINIMUM_SPLIT_SEED = 0
CALIBRATION_MINIMUM_HISTOGRAM_BINS = 2
CALIBRATION_MINIMUM_SITE_FIDELITY = 0.5
CALIBRATION_MAXIMUM_SITE_FIDELITY = 1.0
CALIBRATION_MINIMUM_MAX_DROP = 0
CALIBRATION_MINIMUM_DETECTOR_DISTANCE = 1
CALIBRATION_MINIMUM_DETECTOR_THRESHOLD_REL = 0.0
CALIBRATION_MAXIMUM_DETECTOR_THRESHOLD_REL = 1.0
CALIBRATION_MINIMUM_DETECTOR_REFINE_HALF = 0


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...] | None = None,
    field_name: str,
) -> np.ndarray:
    target_dtype = np.dtype(dtype).newbyteorder("<")
    array = np.asarray(value, dtype=target_dtype)
    expected_shape = array.shape if shape is None else shape
    if array.shape != expected_shape:
        raise ValueError(
            f"{field_name} must have shape {expected_shape}, got {array.shape}"
        )
    return immutable_array(array, dtype=target_dtype, shape=expected_shape)


@dataclass(frozen=True, eq=False)
class CalibrationAnalysisRequest:
    """Explicit physical and statistical intent for one calibration artifact.

    ``expected_centers_xy`` is independent spatial admission evidence in the
    declared ``ordering``.  It constrains authority; detector output can never
    fill or replace it.
    """

    layout: CalibrationCaptureLayout
    grid_shape_yx: tuple[int, int]
    ordering: GridOrder = GridOrder.ROW_MAJOR
    box_radius: int = 1
    box_reducer: BoxReducer = BoxReducer.MEAN
    psf_half_width: int = 3
    psf_background: BackgroundMode = BackgroundMode.ANNULUS_MEDIAN
    psf_background_padding: int = 3
    model_kinds: tuple[_ReadoutModelKind, ...] = (
        _ReadoutModelKind.BOX,
        _ReadoutModelKind.PER_SITE_PSF,
        _ReadoutModelKind.UNIFORM_PSF,
    )
    default_model_kind: _ReadoutModelKind = _ReadoutModelKind.BOX
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU
    train_fraction: float = 0.9
    split_seed: int = 0
    histogram_bins: int = 120
    minimum_site_fidelity: float = 0.5
    max_drop: int | None = None
    detector_min_distance: int | None = None
    detector_threshold_rel: float = 0.35
    detector_refine_half: int = 2
    expected_centers_xy: np.ndarray | None = None
    maximum_site_residual_px: float | None = None

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CalibrationAnalysisRequest):
            return NotImplemented
        for item in fields(self):
            if item.name == "expected_centers_xy":
                continue
            if getattr(self, item.name) != getattr(other, item.name):
                return False
        if self.expected_centers_xy is None or other.expected_centers_xy is None:
            return self.expected_centers_xy is other.expected_centers_xy
        return bool(np.array_equal(self.expected_centers_xy, other.expected_centers_xy))

    def __post_init__(self) -> None:
        if not isinstance(self.layout, CalibrationCaptureLayout):
            raise TypeError("layout must be CalibrationCaptureLayout")
        try:
            raw_grid = tuple(self.grid_shape_yx)
        except TypeError as exc:
            raise ValueError("grid_shape_yx must contain two positive integers") from exc
        if len(raw_grid) != 2:
            raise ValueError("grid_shape_yx must contain two positive integers")
        grid = (
            _positive_integer(raw_grid[0], "grid_shape_yx[0]"),
            _positive_integer(raw_grid[1], "grid_shape_yx[1]"),
        )
        if not isinstance(self.ordering, GridOrder):
            raise TypeError("ordering must be GridOrder")
        radius = _nonnegative_integer(self.box_radius, "box_radius")
        if not isinstance(self.box_reducer, BoxReducer):
            raise TypeError("box_reducer must be BoxReducer")
        psf_half = _nonnegative_integer(self.psf_half_width, "psf_half_width")
        if not isinstance(self.psf_background, BackgroundMode):
            raise TypeError("psf_background must be BackgroundMode")
        padding = _positive_integer(
            self.psf_background_padding,
            "psf_background_padding",
        )
        kinds = tuple(self.model_kinds)
        if not kinds or any(not isinstance(kind, _ReadoutModelKind) for kind in kinds):
            raise TypeError("model_kinds must contain ReadoutModelKind values")
        if len(set(kinds)) != len(kinds):
            raise ValueError("model_kinds must be unique")
        kinds = tuple(kind for kind in _ReadoutModelKind if kind in kinds)
        if not isinstance(self.default_model_kind, _ReadoutModelKind):
            raise TypeError("default_model_kind must be ReadoutModelKind")
        if self.default_model_kind not in kinds:
            raise ValueError("default_model_kind must be present in model_kinds")
        if not isinstance(self.threshold_method, ThresholdMethod):
            raise TypeError("threshold_method must be ThresholdMethod")
        fraction = _finite_float(self.train_fraction, "train_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        seed = _nonnegative_integer(self.split_seed, "split_seed")
        bins = _positive_integer(self.histogram_bins, "histogram_bins")
        if bins < CALIBRATION_MINIMUM_HISTOGRAM_BINS:
            raise ValueError(
                "histogram_bins must be at least "
                f"{CALIBRATION_MINIMUM_HISTOGRAM_BINS}"
            )
        minimum_site_fidelity = _finite_float(
            self.minimum_site_fidelity,
            "minimum_site_fidelity",
        )
        if not (
            CALIBRATION_MINIMUM_SITE_FIDELITY
            <= minimum_site_fidelity
            <= CALIBRATION_MAXIMUM_SITE_FIDELITY
        ):
            raise ValueError(
                "minimum_site_fidelity must be in "
                f"[{CALIBRATION_MINIMUM_SITE_FIDELITY}, "
                f"{CALIBRATION_MAXIMUM_SITE_FIDELITY}]"
            )
        site_count = grid[0] * grid[1]
        max_drop = (
            min(5, site_count)
            if self.max_drop is None
            else _nonnegative_integer(self.max_drop, "max_drop")
        )
        if max_drop > site_count:
            raise ValueError(
                f"max_drop must not exceed the {site_count} declared sites"
            )
        min_distance = self.detector_min_distance
        if min_distance is not None:
            min_distance = _positive_integer(min_distance, "detector_min_distance")
        threshold_rel = _finite_float(
            self.detector_threshold_rel,
            "detector_threshold_rel",
        )
        if not (
            CALIBRATION_MINIMUM_DETECTOR_THRESHOLD_REL
            <= threshold_rel
            <= CALIBRATION_MAXIMUM_DETECTOR_THRESHOLD_REL
        ):
            raise ValueError(
                "detector_threshold_rel must be in "
                f"[{CALIBRATION_MINIMUM_DETECTOR_THRESHOLD_REL}, "
                f"{CALIBRATION_MAXIMUM_DETECTOR_THRESHOLD_REL}]"
            )
        refine_half = _nonnegative_integer(
            self.detector_refine_half,
            "detector_refine_half",
        )
        expected_centers = self.expected_centers_xy
        maximum_residual = self.maximum_site_residual_px
        if (expected_centers is None) != (maximum_residual is None):
            raise ValueError(
                "expected_centers_xy and maximum_site_residual_px must be "
                "provided together"
            )
        if expected_centers is not None:
            expected_centers = _immutable_array(
                expected_centers,
                dtype="<f8",
                shape=(site_count, 2),
                field_name="expected_centers_xy",
            )
            if not np.all(np.isfinite(expected_centers)):
                raise ValueError("expected_centers_xy must be finite")
            maximum_residual = _finite_float(
                maximum_residual,
                "maximum_site_residual_px",
            )
            if maximum_residual <= 0.0:
                raise ValueError("maximum_site_residual_px must be positive")
            minimum_separation = _minimum_coordinate_separation(expected_centers)
            if site_count > 1:
                if (
                    not math.isfinite(minimum_separation)
                    or minimum_separation <= 0.0
                ):
                    raise ValueError("expected_centers_xy must contain unique sites")
                if 2.0 * maximum_residual >= minimum_separation:
                    raise ValueError(
                        "maximum_site_residual_px must be less than half the minimum "
                        "expected site-center separation"
                    )
        object.__setattr__(self, "grid_shape_yx", grid)
        object.__setattr__(self, "box_radius", radius)
        object.__setattr__(self, "psf_half_width", psf_half)
        object.__setattr__(self, "psf_background_padding", padding)
        object.__setattr__(self, "model_kinds", kinds)
        object.__setattr__(self, "train_fraction", fraction)
        object.__setattr__(self, "split_seed", seed)
        object.__setattr__(self, "histogram_bins", bins)
        object.__setattr__(
            self,
            "minimum_site_fidelity",
            minimum_site_fidelity,
        )
        object.__setattr__(self, "max_drop", max_drop)
        object.__setattr__(self, "detector_min_distance", min_distance)
        object.__setattr__(self, "detector_threshold_rel", threshold_rel)
        object.__setattr__(self, "detector_refine_half", refine_half)
        object.__setattr__(self, "expected_centers_xy", expected_centers)
        object.__setattr__(self, "maximum_site_residual_px", maximum_residual)

    @property
    def site_count(self) -> int:
        return self.grid_shape_yx[0] * self.grid_shape_yx[1]


def calibration_analysis_authoring_schema(
    request: CalibrationAnalysisRequest,
) -> AuthoringSchema:
    """Declare the ordered ordinary fields editable for one frozen request."""

    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    model_fields = tuple(
        AuthoringField(
            f"model.{kind.value}.enabled",
            "bool",
            f"Enable {kind.value}",
            default=kind in request.model_kinds,
            description=(
                "All enabled models are calibrated and committed atomically."
            ),
        )
        for kind in _ReadoutModelKind
    )
    return AuthoringSchema(
        (
            *model_fields,
            AuthoringField(
                "default_model_kind",
                "choice",
                "Default model",
                default=request.default_model_kind,
                choices=tuple(
                    AuthoringChoice(kind, kind.value) for kind in _ReadoutModelKind
                ),
            ),
            AuthoringField(
                "box_radius",
                "int",
                "Box radius",
                default=request.box_radius,
                required=True,
                unit="px",
                minimum=CALIBRATION_MINIMUM_BOX_RADIUS,
            ),
            AuthoringField(
                "box_reducer",
                "choice",
                "Box reducer",
                default=request.box_reducer,
                choices=tuple(
                    AuthoringChoice(item, item.value) for item in BoxReducer
                ),
            ),
            AuthoringField(
                "psf_half_width",
                "int",
                "PSF half width",
                default=request.psf_half_width,
                required=True,
                unit="px",
                minimum=CALIBRATION_MINIMUM_PSF_HALF_WIDTH,
            ),
            AuthoringField(
                "psf_background",
                "choice",
                "PSF background",
                default=request.psf_background,
                choices=tuple(
                    AuthoringChoice(item, item.value) for item in BackgroundMode
                ),
            ),
            AuthoringField(
                "psf_background_padding",
                "int",
                "PSF background padding",
                default=request.psf_background_padding,
                required=True,
                unit="px",
                minimum=CALIBRATION_MINIMUM_PSF_BACKGROUND_PADDING,
            ),
            AuthoringField(
                "train_fraction",
                "float",
                "Train fraction",
                default=request.train_fraction,
                required=True,
                description="Must remain strictly between zero and one.",
            ),
            AuthoringField(
                "split_seed",
                "int",
                "Split seed",
                default=request.split_seed,
                required=True,
                minimum=CALIBRATION_MINIMUM_SPLIT_SEED,
            ),
            AuthoringField(
                "histogram_bins",
                "int",
                "Histogram bins",
                default=request.histogram_bins,
                required=True,
                minimum=CALIBRATION_MINIMUM_HISTOGRAM_BINS,
            ),
            AuthoringField(
                "minimum_site_fidelity",
                "float",
                "Minimum site fidelity",
                default=request.minimum_site_fidelity,
                required=True,
                minimum=CALIBRATION_MINIMUM_SITE_FIDELITY,
                maximum=CALIBRATION_MAXIMUM_SITE_FIDELITY,
            ),
            AuthoringField(
                "max_drop",
                "int",
                "Maximum dropped sites",
                default=request.max_drop,
                required=True,
                minimum=CALIBRATION_MINIMUM_MAX_DROP,
                maximum=request.site_count,
            ),
            AuthoringField(
                "detector_min_distance",
                "int",
                "Detector minimum distance",
                default=request.detector_min_distance,
                unit="px",
                minimum=CALIBRATION_MINIMUM_DETECTOR_DISTANCE,
                allow_blank=True,
            ),
            AuthoringField(
                "detector_threshold_rel",
                "float",
                "Detector relative threshold",
                default=request.detector_threshold_rel,
                required=True,
                minimum=CALIBRATION_MINIMUM_DETECTOR_THRESHOLD_REL,
                maximum=CALIBRATION_MAXIMUM_DETECTOR_THRESHOLD_REL,
            ),
            AuthoringField(
                "detector_refine_half",
                "int",
                "Detector refine half width",
                default=request.detector_refine_half,
                required=True,
                unit="px",
                minimum=CALIBRATION_MINIMUM_DETECTOR_REFINE_HALF,
            ),
        )
    )


def build_calibration_analysis_request_from_authoring(
    request: CalibrationAnalysisRequest,
    values: Mapping[str, object],
) -> CalibrationAnalysisRequest:
    """Rebuild editable leaves while preserving frozen spatial authority."""

    authored = calibration_analysis_authoring_schema(request).freeze(values)
    model_kinds = tuple(
        kind
        for kind in _ReadoutModelKind
        if authored[f"model.{kind.value}.enabled"] is True
    )
    default_model_kind = authored["default_model_kind"]
    if default_model_kind not in model_kinds:
        raise ValueError("default model must remain enabled")
    return replace(
        request,
        model_kinds=model_kinds,
        default_model_kind=default_model_kind,
        box_radius=authored["box_radius"],
        box_reducer=authored["box_reducer"],
        psf_half_width=authored["psf_half_width"],
        psf_background=authored["psf_background"],
        psf_background_padding=authored["psf_background_padding"],
        train_fraction=authored["train_fraction"],
        split_seed=authored["split_seed"],
        histogram_bins=authored["histogram_bins"],
        minimum_site_fidelity=authored["minimum_site_fidelity"],
        max_drop=authored["max_drop"],
        detector_min_distance=authored["detector_min_distance"],
        detector_threshold_rel=authored["detector_threshold_rel"],
        detector_refine_half=authored["detector_refine_half"],
    )


def _site_validity(
    value: ComponentValidity,
    site_axis: AxisSpec,
    field_name: str,
) -> ComponentValidity:
    if not isinstance(value, ComponentValidity):
        raise TypeError(f"{field_name} must be ComponentValidity")
    if value.axis_ids != (site_axis.axis_id,):
        raise ValueError(f"{field_name} must name exactly the site axis")
    if value.mask.shape != (site_axis.size,):
        raise ValueError(
            f"{field_name} must have shape ({site_axis.size},), got {value.mask.shape}"
        )
    return value


def _boxes(value: object, site_count: int, field_name: str) -> np.ndarray:
    boxes = _immutable_array(
        value,
        dtype="<i8",
        shape=(site_count, 4),
        field_name=field_name,
    )
    if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] <= 0):
        raise ValueError(f"{field_name} must contain non-negative x/y and positive w/h")
    return boxes


def _normalized_kernel(value: object, shape: tuple[int, ...], field_name: str) -> np.ndarray:
    kernel = _immutable_array(value, dtype="<f8", shape=shape, field_name=field_name)
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0):
        raise ValueError(f"{field_name} must be finite and non-negative")
    totals = np.sum(kernel, axis=(-2, -1))
    if not np.allclose(totals, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{field_name} must be L1-normalized")
    return kernel


@dataclass(frozen=True)
class CalibrationSourceBinding:
    """The raw capture and logical event join used to derive a calibration."""

    source_capture_ref: CaptureArtifactRef
    layout: CalibrationCaptureLayout

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.layout, CalibrationCaptureLayout):
            raise TypeError("layout must be CalibrationCaptureLayout")


@dataclass(frozen=True, slots=True)
class _ResolvedCalibrationSource:
    source_binding: CalibrationSourceBinding
    frame_contract: FrameContract
    readout_physical_context: ReadoutPhysicalContext
    join: _CalibrationCaptureJoin


def _resolve_calibration_source(
    capture: object,
    layout: CalibrationCaptureLayout,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> _ResolvedCalibrationSource:
    """Resolve source lineage, physical contract, and sparse event join once."""

    if not isinstance(layout, CalibrationCaptureLayout):
        raise TypeError("layout must be CalibrationCaptureLayout")
    try:
        reference = capture.ref  # type: ignore[attr-defined]
        source = capture.frame_source  # type: ignore[attr-defined]
        provenance = capture.camera_provenance  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("capture must be a resolved raw CaptureArtifact") from exc
    if not isinstance(reference, CaptureArtifactRef):
        raise TypeError("capture.ref must be CaptureArtifactRef")
    if not isinstance(source, CaptureFrameSource):
        raise TypeError("capture.frame_source must be CaptureFrameSource")
    try:
        descriptor = provenance.descriptor
        binding = provenance.binding
    except AttributeError as exc:
        raise TypeError("capture omits camera provenance") from exc
    if not isinstance(descriptor, CameraCaptureDescriptor):
        raise TypeError("capture camera descriptor must be CameraCaptureDescriptor")
    if not isinstance(binding, ReadoutBindingKey):
        raise TypeError("capture readout binding must be ReadoutBindingKey")
    contract, join = FrameContract._resolve_calibration_capture(
        binding,
        descriptor,
        source.schema,
        layout,
    )
    source_binding = CalibrationSourceBinding(reference, layout)
    return _ResolvedCalibrationSource(
        source_binding,
        contract,
        derive_calibration_readout_physical_context(
            capture,
            layout,
            contract,
            checkpoint=checkpoint,
        ),
        join,
    )


def _validate_calibration_artifact_source_compatibility(
    artifact: "CalibrationArtifact",
    capture: object,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> _ResolvedCalibrationSource:
    """Compare one loaded source while honoring cancellation/resource bounds."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    resolved = _resolve_calibration_source(
        capture,
        artifact.source_binding.layout,
        checkpoint=checkpoint,
    )
    if resolved.source_binding != artifact.source_binding:
        raise ValueError("calibration source differs from the resolved capture")
    if resolved.frame_contract != artifact.frame_contract:
        raise ValueError("calibration FrameContract differs from the resolved capture")
    if resolved.readout_physical_context != artifact.readout_physical_context:
        raise ValueError(
            "calibration readout physical context differs from the resolved capture"
        )
    return resolved


def derive_calibration_readout_physical_context(
    capture: object,
    layout: CalibrationCaptureLayout,
    frame_contract: FrameContract,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> ReadoutPhysicalContext:
    """Derive calibration applicability only from loaded pulse/camera lineage."""

    if not isinstance(layout, CalibrationCaptureLayout):
        raise TypeError("layout must be CalibrationCaptureLayout")
    if not isinstance(frame_contract, FrameContract):
        raise TypeError("frame_contract must be FrameContract")
    try:
        evidence = capture.pulse_evidence  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("capture must be a resolved raw CaptureArtifact") from exc
    if evidence is None:
        raise ValueError(
            "authoritative calibration requires pulse-trigger lineage"
        )
    try:
        physical_facts = capture.camera_capability_evidence.physical_facts  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("capture omits camera capability evidence") from exc
    integration_offset = (
        physical_facts.external_trigger_integration_start_offset_seconds
    )
    frame_source = capture.frame_source  # type: ignore[attr-defined]
    return _derive_readout_physical_context_from_evidence(
        evidence,
        frame_source.schema,
        frame_source.iter_cell_schedule(),
        readout_event_index=layout.readout_event_index,
        integration_start_offset_seconds=integration_offset,
        integration_seconds=frame_contract.exposure_seconds,
        checkpoint=checkpoint,
    )


@dataclass(frozen=True, eq=False)
class SiteMap:
    """Stable sites in ROI-local output-pixel ``(x, y)`` coordinates."""

    site_axis: AxisSpec
    coordinates_xy: np.ndarray
    grid_shape_yx: tuple[int, int]
    ordering: GridOrder
    coordinate_frame: CoordinateFrameId
    validity: ComponentValidity

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role 'site'")
        coordinates = _immutable_array(
            self.coordinates_xy,
            dtype="<f8",
            shape=(self.site_axis.size, 2),
            field_name="coordinates_xy",
        )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates_xy must be finite")
        try:
            raw_grid = tuple(self.grid_shape_yx)
        except TypeError as exc:
            raise ValueError("grid_shape_yx must contain two positive integers") from exc
        if len(raw_grid) != 2:
            raise ValueError("grid_shape_yx must contain two positive integers")
        grid = (
            _positive_integer(raw_grid[0], "grid_shape_yx[0]"),
            _positive_integer(raw_grid[1], "grid_shape_yx[1]"),
        )
        if grid[0] * grid[1] != self.site_axis.size:
            raise ValueError("grid_shape_yx product must equal site-axis size")
        if not isinstance(self.ordering, GridOrder):
            raise TypeError("ordering must be GridOrder")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        validity = _site_validity(self.validity, self.site_axis, "validity")
        object.__setattr__(self, "coordinates_xy", coordinates)
        object.__setattr__(self, "grid_shape_yx", grid)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True, eq=False)
class BoxFeature:
    site_axis: AxisSpec
    boxes_xywh: np.ndarray
    reducer: BoxReducer
    valid_sites: ComponentValidity

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must have role 'site'")
        if not isinstance(self.reducer, BoxReducer):
            raise TypeError("reducer must be BoxReducer")
        object.__setattr__(
            self,
            "boxes_xywh",
            _boxes(self.boxes_xywh, self.site_axis.size, "boxes_xywh"),
        )
        object.__setattr__(
            self,
            "valid_sites",
            _site_validity(self.valid_sites, self.site_axis, "valid_sites"),
        )

    @property
    def kind(self) -> _ReadoutModelKind:
        return _ReadoutModelKind.BOX


@dataclass(frozen=True, eq=False)
class PerSitePsfFeature:
    site_axis: AxisSpec
    boxes_xywh: np.ndarray
    kernels: np.ndarray
    background: BackgroundMode
    background_padding: int
    valid_sites: ComponentValidity

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must have role 'site'")
        boxes = _boxes(self.boxes_xywh, self.site_axis.size, "boxes_xywh")
        shapes = {(int(h), int(w)) for _x, _y, w, h in boxes}
        if len(shapes) != 1:
            raise ValueError("per-site PSF boxes must share one shape")
        kernel_shape = next(iter(shapes))
        kernels = _normalized_kernel(
            self.kernels,
            (self.site_axis.size, *kernel_shape),
            "kernels",
        )
        if not isinstance(self.background, BackgroundMode):
            raise TypeError("background must be BackgroundMode")
        padding = _positive_integer(self.background_padding, "background_padding")
        validity = _site_validity(self.valid_sites, self.site_axis, "valid_sites")
        object.__setattr__(self, "boxes_xywh", boxes)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "background_padding", padding)
        object.__setattr__(self, "valid_sites", validity)

    @property
    def kind(self) -> _ReadoutModelKind:
        return _ReadoutModelKind.PER_SITE_PSF


@dataclass(frozen=True, eq=False)
class UniformPsfFeature:
    site_axis: AxisSpec
    boxes_xywh: np.ndarray
    kernel: np.ndarray
    background: BackgroundMode
    background_padding: int
    valid_sites: ComponentValidity

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must have role 'site'")
        boxes = _boxes(self.boxes_xywh, self.site_axis.size, "boxes_xywh")
        shapes = {(int(h), int(w)) for _x, _y, w, h in boxes}
        if len(shapes) != 1:
            raise ValueError("uniform PSF boxes must share one shape")
        kernel = _normalized_kernel(self.kernel, next(iter(shapes)), "kernel")
        if not isinstance(self.background, BackgroundMode):
            raise TypeError("background must be BackgroundMode")
        padding = _positive_integer(self.background_padding, "background_padding")
        validity = _site_validity(self.valid_sites, self.site_axis, "valid_sites")
        object.__setattr__(self, "boxes_xywh", boxes)
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "background_padding", padding)
        object.__setattr__(self, "valid_sites", validity)

    @property
    def kind(self) -> _ReadoutModelKind:
        return _ReadoutModelKind.UNIFORM_PSF


ReadoutFeature: TypeAlias = BoxFeature | PerSitePsfFeature | UniformPsfFeature


def _feature(value: object) -> ReadoutFeature:
    if not isinstance(value, (BoxFeature, PerSitePsfFeature, UniformPsfFeature)):
        raise TypeError("feature must be a closed readout feature")
    return value


@dataclass(frozen=True, eq=False)
class ReadoutModel:
    """One feature extractor plus fluorescence thresholds and usable sites."""

    feature: ReadoutFeature
    thresholds: np.ndarray
    usable_sites: ComponentValidity

    def __post_init__(self) -> None:
        feature = _feature(self.feature)
        thresholds = _immutable_array(
            self.thresholds,
            dtype="<f8",
            shape=(feature.site_axis.size,),
            field_name="thresholds",
        )
        usable = _site_validity(
            self.usable_sites,
            feature.site_axis,
            "usable_sites",
        )
        if np.any(usable.mask & ~np.isfinite(thresholds)):
            raise ValueError("usable sites require finite thresholds")
        if np.any(usable.mask & ~feature.valid_sites.mask):
            raise ValueError("usable sites must be a subset of feature-valid sites")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "usable_sites", usable)

    @property
    def kind(self) -> _ReadoutModelKind:
        return self.feature.kind


@dataclass(frozen=True, eq=False)
class CalibrationArtifact:
    """Runtime calibration bound to source, camera, pulse context, sites, and models."""

    source_binding: CalibrationSourceBinding
    frame_contract: FrameContract
    readout_physical_context: ReadoutPhysicalContext
    site_map: SiteMap
    models: tuple[ReadoutModel, ...]
    default_model_kind: _ReadoutModelKind

    def __post_init__(self) -> None:
        if not isinstance(self.source_binding, CalibrationSourceBinding):
            raise TypeError("source_binding must be CalibrationSourceBinding")
        if not isinstance(self.frame_contract, FrameContract):
            raise TypeError("frame_contract must be FrameContract")
        if not isinstance(self.readout_physical_context, ReadoutPhysicalContext):
            raise TypeError(
                "readout_physical_context must be ReadoutPhysicalContext"
            )
        if (
            self.readout_physical_context.integration_seconds
            != self.frame_contract.exposure_seconds
        ):
            raise ValueError(
                "readout physical context integration differs from FrameContract exposure"
            )
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        models = tuple(self.models)
        if not models or any(not isinstance(model, ReadoutModel) for model in models):
            raise TypeError("models must contain at least one ReadoutModel")
        kinds = tuple(model.kind for model in models)
        if len(set(kinds)) != len(kinds):
            raise ValueError("models must contain at most one model of each kind")
        if kinds != tuple(kind for kind in _ReadoutModelKind if kind in kinds):
            raise ValueError("models must follow ReadoutModelKind declaration order")
        if not isinstance(self.default_model_kind, _ReadoutModelKind):
            raise TypeError("default_model_kind must be ReadoutModelKind")
        if self.default_model_kind not in kinds:
            raise ValueError("default_model_kind must name a stored model")
        frame_height, frame_width = self.frame_contract.frame_schema.data_shape
        coordinates = self.site_map.coordinates_xy
        if np.any(
            (coordinates[:, 0] < 0.0)
            | (coordinates[:, 0] >= frame_width)
            | (coordinates[:, 1] < 0.0)
            | (coordinates[:, 1] >= frame_height)
        ):
            raise ValueError("SiteMap coordinates fall outside the FrameContract image")
        for model in models:
            if model.feature.site_axis != self.site_map.site_axis:
                raise ValueError("every model must use the SiteMap axis")
            if np.any(model.feature.valid_sites.mask & ~self.site_map.validity.mask):
                raise ValueError("model feature validity must be a subset of SiteMap validity")
            boxes = model.feature.boxes_xywh
            if np.any(boxes[:, 0] + boxes[:, 2] > frame_width) or np.any(
                boxes[:, 1] + boxes[:, 3] > frame_height
            ):
                raise ValueError("model feature boxes fall outside the FrameContract image")
        if self.site_map.coordinate_frame != self.frame_contract.coordinate_frame:
            raise ValueError("SiteMap and FrameContract coordinate frames differ")
        object.__setattr__(self, "models", models)

    def select_model(self, kind: _ReadoutModelKind | None = None) -> ReadoutModel:
        selected = self.default_model_kind if kind is None else kind
        if not isinstance(selected, _ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind or None")
        for model in self.models:
            if model.kind is selected:
                return model
        raise KeyError(selected)


@dataclass(frozen=True, slots=True)
class ResolvedCalibration:
    """One cold-opened Calibration value and its persisted Run provenance."""

    reference: CalibrationArtifactRef
    artifact: CalibrationArtifact
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if not isinstance(self.artifact, CalibrationArtifact):
            raise TypeError("artifact must be CalibrationArtifact")
        from zlc_storage import canonical_text

        canonical_text(self.run_id, "run_id")


@dataclass(frozen=True)
class ReadoutResult:
    """Signals and occupancy produced atomically from one camera frame."""

    signals: Value
    occupied: Value

    def __post_init__(self) -> None:
        if not isinstance(self.signals, Value) or not isinstance(self.occupied, Value):
            raise TypeError("signals and occupied must be Value")
        if self.signals.schema.data_axes != self.occupied.schema.data_axes:
            raise ValueError("signals and occupied must share the site axis")
        if not isinstance(self.signals.validity, ComponentValidity) or not isinstance(
            self.occupied.validity,
            ComponentValidity,
        ):
            raise TypeError("readout values require component validity")
        if (
            self.signals.validity.axis_ids != self.occupied.validity.axis_ids
            or not np.array_equal(
                self.signals.validity.mask,
                self.occupied.validity.mask,
            )
        ):
            raise ValueError("signals and occupied validity differ")


def _validate_frame(frame: Value) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frame, Value):
        raise TypeError("frame must be Value")
    axes = frame.schema.data_axes
    if len(axes) != 2 or axes[0].role != SPATIAL_Y or axes[1].role != SPATIAL_X:
        raise ValueError("frame must have exactly named spatial-y, spatial-x axes")
    # Preserve the camera dtype and storage.  Converting a full 2304x2304
    # uint16 qCMOS frame to float64 here allocates another ~42 MiB before any
    # site work begins; the numeric core converts only small site windows.
    values = np.asarray(frame.values)
    validity = np.asarray(expand_value_validity(frame.validity, frame.schema), dtype=bool)
    if validity.shape != values.shape:
        raise ValueError("frame validity does not align with frame pixels")
    return values, validity


def _checked_box(
    box: np.ndarray,
    image_shape: tuple[int, int],
    site: int,
) -> tuple[int, int, int, int]:
    x0, y0, width, height = (int(value) for value in box)
    if x0 + width > image_shape[1] or y0 + height > image_shape[0]:
        raise ValueError(
            f"site {site} box {(x0, y0, width, height)} is outside image shape "
            f"{image_shape}"
        )
    return x0, y0, width, height


def _annulus_background(
    image: np.ndarray,
    pixel_validity: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int,
) -> float:
    x, y, width, height = box
    y0 = max(0, y - padding)
    y1 = min(image.shape[0], y + height + padding)
    x0 = max(0, x - padding)
    x1 = min(image.shape[1], x + width + padding)
    region = image[y0:y1, x0:x1]
    region_valid = pixel_validity[y0:y1, x0:x1]
    if np.issubdtype(region.dtype, np.inexact):
        region_valid = region_valid & np.isfinite(region)
    ring = np.array(region_valid, copy=True)
    ring[y - y0 : y - y0 + height, x - x0 : x - x0 + width] = False
    values = region[ring]
    if values.size:
        return float(np.median(values))
    fallback_valid = pixel_validity
    if np.issubdtype(image.dtype, np.inexact):
        fallback_valid = fallback_valid & np.isfinite(image)
    fallback = image[fallback_valid]
    return float(np.median(fallback)) if fallback.size else 0.0


def _feature_schema(feature: ReadoutFeature, value_unit: str | None) -> ValueSchema:
    return ValueSchema(
        (feature.site_axis,),
        ValidityContract.components(feature.site_axis.axis_id),
        np.dtype("<f8"),
        value_unit,
    )


def _extract_readout_arrays(
    feature: ReadoutFeature,
    image: np.ndarray,
    pixel_validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared numeric core; callers retain ownership of frame storage."""

    feature = _feature(feature)
    image = np.asarray(image)
    pixel_validity = np.asarray(pixel_validity, dtype=bool)
    if image.ndim != 2 or pixel_validity.shape != image.shape:
        raise ValueError("readout image and pixel validity must be matching 2D arrays")
    output = np.zeros(feature.site_axis.size, dtype="<f8")
    valid = np.array(feature.valid_sites.mask, dtype=bool, copy=True)

    if isinstance(feature, BoxFeature):
        for site, raw_box in enumerate(feature.boxes_xywh):
            if not valid[site]:
                continue
            x0, y0, width, height = _checked_box(raw_box, image.shape, site)
            cut = image[y0 : y0 + height, x0 : x0 + width]
            mask = pixel_validity[y0 : y0 + height, x0 : x0 + width]
            if np.issubdtype(cut.dtype, np.inexact):
                mask = mask & np.isfinite(cut)
            values = np.asarray(cut[mask], dtype=np.float64)
            if not values.size:
                valid[site] = False
                continue
            if feature.reducer is BoxReducer.SUM:
                output[site] = float(np.sum(values))
            elif feature.reducer is BoxReducer.MEDIAN:
                output[site] = float(np.median(values))
            elif feature.reducer is BoxReducer.MAX:
                output[site] = float(np.max(values))
            else:
                output[site] = float(np.mean(values))
    else:
        kernels = (
            feature.kernels
            if isinstance(feature, PerSitePsfFeature)
            else np.broadcast_to(
                feature.kernel,
                (feature.site_axis.size, *feature.kernel.shape),
            )
        )
        for site, (raw_box, kernel) in enumerate(
            zip(feature.boxes_xywh, kernels, strict=True)
        ):
            if not valid[site]:
                continue
            x0, y0, width, height = _checked_box(raw_box, image.shape, site)
            cut = image[y0 : y0 + height, x0 : x0 + width]
            cut_valid = pixel_validity[y0 : y0 + height, x0 : x0 + width]
            # Dropping or renormalizing a PSF pixel changes the calibrated signal
            # scale.  Mark that site invalid instead of silently changing physics.
            if np.issubdtype(cut.dtype, np.inexact):
                complete = np.all(cut_valid & np.isfinite(cut))
            else:
                complete = np.all(cut_valid)
            if not complete:
                valid[site] = False
                continue
            cut = np.asarray(cut, dtype=np.float64)
            background = (
                _annulus_background(
                    image,
                    pixel_validity,
                    (x0, y0, width, height),
                    feature.background_padding,
                )
                if feature.background is BackgroundMode.ANNULUS_MEDIAN
                else 0.0
            )
            output[site] = float(np.sum(kernel * (cut - background)))

    output[~valid] = 0.0
    return output, valid


def extract_readout_features(feature: ReadoutFeature, frame: Value) -> Value:
    """Extract one scalar per site using the calibration/runtime single source."""

    feature = _feature(feature)
    image, pixel_validity = _validate_frame(frame)
    output, valid = _extract_readout_arrays(feature, image, pixel_validity)
    return Value(
        output,
        ComponentValidity((feature.site_axis.axis_id,), valid),
        _feature_schema(feature, frame.schema.value_unit),
    )


def classify_occupancy(model: ReadoutModel, signals: Value) -> Value:
    """Apply the fluorescence invariant: occupied means ``signal > threshold``."""

    if not isinstance(model, ReadoutModel):
        raise TypeError("model must be ReadoutModel")
    if not isinstance(signals, Value):
        raise TypeError("signals must be Value")
    if signals.schema.data_axes != (model.feature.site_axis,):
        raise ValueError("signals do not use the model site axis")
    signal_validity = np.asarray(
        expand_value_validity(signals.validity, signals.schema),
        dtype=bool,
    )
    usable = (
        signal_validity
        & model.usable_sites.mask
        & np.isfinite(signals.values)
        & np.isfinite(model.thresholds)
    )
    occupied = np.asarray(signals.values > model.thresholds, dtype=bool)
    occupied[~usable] = False
    schema = ValueSchema(
        (model.feature.site_axis,),
        ValidityContract.components(model.feature.site_axis.axis_id),
        np.dtype("bool"),
        "occupation",
    )
    return Value(
        occupied,
        ComponentValidity((model.feature.site_axis.axis_id,), usable),
        schema,
    )


def apply_readout_model(
    model: ReadoutModel,
    frame: Value,
    *,
    expected_frame_schema: ValueSchema,
) -> ReadoutResult:
    """Apply an already-bound calibration model to one compatible frame.

    This is the public, side-effect-free evaluator shared by every readout
    consumer.  Calibration owns the fitted model value; Occupancy and other
    processors reuse this evaluator instead of importing a private fitting
    implementation or reproducing feature extraction.
    """

    if not isinstance(expected_frame_schema, ValueSchema):
        raise TypeError("expected_frame_schema must be ValueSchema")
    if not isinstance(frame, Value):
        raise TypeError("frame must be Value")
    if frame.schema != expected_frame_schema:
        raise ValueError("frame schema differs from the bound readout schema")
    extracted = extract_readout_features(model.feature, frame)
    occupied = classify_occupancy(model, extracted)
    assert isinstance(occupied.validity, ComponentValidity)
    values = np.array(extracted.values, dtype="<f8", copy=True)
    values[~occupied.validity.mask] = 0.0
    signals = Value(values, occupied.validity, extracted.schema)
    return ReadoutResult(signals, occupied)


__all__ = [
    "BackgroundMode",
    "BoxFeature",
    "BoxReducer",
    "CalibrationAnalysisRequest",
    "CalibrationArtifact",
    "CalibrationSourceBinding",
    "CALIBRATION_MAXIMUM_DETECTOR_THRESHOLD_REL",
    "CALIBRATION_MAXIMUM_SITE_FIDELITY",
    "CALIBRATION_MINIMUM_BOX_RADIUS",
    "CALIBRATION_MINIMUM_DETECTOR_DISTANCE",
    "CALIBRATION_MINIMUM_DETECTOR_REFINE_HALF",
    "CALIBRATION_MINIMUM_DETECTOR_THRESHOLD_REL",
    "CALIBRATION_MINIMUM_HISTOGRAM_BINS",
    "CALIBRATION_MINIMUM_MAX_DROP",
    "CALIBRATION_MINIMUM_PSF_BACKGROUND_PADDING",
    "CALIBRATION_MINIMUM_PSF_HALF_WIDTH",
    "CALIBRATION_MINIMUM_SITE_FIDELITY",
    "CALIBRATION_MINIMUM_SPLIT_SEED",
    "GridOrder",
    "PerSitePsfFeature",
    "ReadoutFeature",
    "ReadoutModel",
    "ReadoutResult",
    "ResolvedCalibration",
    "SiteMap",
    "ThresholdMethod",
    "UniformPsfFeature",
    "apply_readout_model",
    "build_calibration_analysis_request_from_authoring",
    "calibration_analysis_authoring_schema",
    "classify_occupancy",
    "derive_calibration_readout_physical_context",
    "extract_readout_features",
    "site_grid_positions_yx",
]
