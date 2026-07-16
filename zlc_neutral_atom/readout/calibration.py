"""Immutable readout calibration values and the one runtime application path.

Calibration stores only facts needed to reproduce readout: the source capture,
the complete camera frame contract, one site map, and a closed set of feature
models.  Statistical diagnostics belong to :mod:`.analysis`; content identity
and durability belong to the repository/CAS.  Keeping those responsibilities
out of these values prevents the same fact being re-hashed and re-validated at
every layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
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
    expand_value_validity,
    immutable_array,
)
from zlc_neutral_atom.artifacts.capture_frames import CaptureFrameSource
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_storage import (
    finite_real as _finite_float,
    nonnegative_integer as _nonnegative_integer,
    positive_integer as _positive_integer,
)

from .calibration_reference import CalibrationArtifactRef
from .physical_context import (
    ReadoutPhysicalContext,
    _derive_readout_physical_context_from_evidence,
)
from .contracts import (
    CalibrationCaptureLayout,
    CameraCaptureDescriptor,
    FrameContract,
    ReadoutBindingKey,
    _minimum_coordinate_separation,
    _CalibrationCaptureJoin,
)


class GridOrder(str, Enum):
    ROW_MAJOR = "row-major"
    SERPENTINE = "serpentine"
    COLUMN_MAJOR = "column-major"
    COLUMN_SERPENTINE = "column-serpentine"


class ReadoutModelKind(str, Enum):
    BOX = "box"
    PER_SITE_PSF = "psf"
    UNIFORM_PSF = "uniform_psf"


class BoxReducer(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    MAX = "max"


class BackgroundMode(str, Enum):
    NONE = "none"
    ANNULUS_MEDIAN = "annulus"


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
    model_kinds: tuple[ReadoutModelKind, ...] = (
        ReadoutModelKind.BOX,
        ReadoutModelKind.PER_SITE_PSF,
        ReadoutModelKind.UNIFORM_PSF,
    )
    default_model_kind: ReadoutModelKind = ReadoutModelKind.BOX
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
        if not kinds or any(not isinstance(kind, ReadoutModelKind) for kind in kinds):
            raise TypeError("model_kinds must contain ReadoutModelKind values")
        if len(set(kinds)) != len(kinds):
            raise ValueError("model_kinds must be unique")
        kinds = tuple(kind for kind in ReadoutModelKind if kind in kinds)
        if not isinstance(self.default_model_kind, ReadoutModelKind):
            raise TypeError("default_model_kind must be ReadoutModelKind")
        if self.default_model_kind not in kinds:
            raise ValueError("default_model_kind must be present in model_kinds")
        fraction = _finite_float(self.train_fraction, "train_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        seed = _nonnegative_integer(self.split_seed, "split_seed")
        bins = _positive_integer(self.histogram_bins, "histogram_bins")
        if bins < 2:
            raise ValueError("histogram_bins must be at least two")
        minimum_site_fidelity = _finite_float(
            self.minimum_site_fidelity,
            "minimum_site_fidelity",
        )
        if not 0.5 <= minimum_site_fidelity <= 1.0:
            raise ValueError("minimum_site_fidelity must be in [0.5, 1.0]")
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
        if not 0.0 <= threshold_rel <= 1.0:
            raise ValueError("detector_threshold_rel must be in [0, 1]")
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
    physical_memory_limit_bytes: int | None = None,
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
            physical_memory_limit_bytes=physical_memory_limit_bytes,
        ),
        join,
    )


def _validate_calibration_artifact_source_compatibility(
    artifact: "CalibrationArtifact",
    capture: object,
    *,
    checkpoint: Callable[[], None] | None = None,
    physical_memory_limit_bytes: int | None = None,
) -> _ResolvedCalibrationSource:
    """Compare one admitted source while honoring cancellation/resource bounds."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    resolved = _resolve_calibration_source(
        capture,
        artifact.source_binding.layout,
        checkpoint=checkpoint,
        physical_memory_limit_bytes=physical_memory_limit_bytes,
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
    physical_memory_limit_bytes: int | None = None,
) -> ReadoutPhysicalContext:
    """Derive calibration applicability only from admitted pulse/camera lineage."""

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
        physical_memory_limit_bytes=physical_memory_limit_bytes,
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
    def kind(self) -> ReadoutModelKind:
        return ReadoutModelKind.BOX


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
    def kind(self) -> ReadoutModelKind:
        return ReadoutModelKind.PER_SITE_PSF


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
    def kind(self) -> ReadoutModelKind:
        return ReadoutModelKind.UNIFORM_PSF


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
    def kind(self) -> ReadoutModelKind:
        return self.feature.kind


@dataclass(frozen=True, eq=False)
class CalibrationArtifact:
    """Runtime calibration bound to source, camera, pulse context, sites, and models."""

    source_binding: CalibrationSourceBinding
    frame_contract: FrameContract
    readout_physical_context: ReadoutPhysicalContext
    site_map: SiteMap
    models: tuple[ReadoutModel, ...]
    default_model_kind: ReadoutModelKind

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
        if kinds != tuple(kind for kind in ReadoutModelKind if kind in kinds):
            raise ValueError("models must follow ReadoutModelKind declaration order")
        if not isinstance(self.default_model_kind, ReadoutModelKind):
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

    def select_model(self, kind: ReadoutModelKind | None = None) -> ReadoutModel:
        selected = self.default_model_kind if kind is None else kind
        if not isinstance(selected, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind or None")
        for model in self.models:
            if model.kind is selected:
                return model
        raise KeyError(selected)


_RESOLVED_CALIBRATION_TOKEN = object()


class ResolvedCalibration:
    """Process-local proof that one exact calibration target was committed."""

    __slots__ = (
        "_token",
        "_repository_token",
        "_reference",
        "_artifact",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("ResolvedCalibration is final and cannot be subclassed")

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError(
            "ResolvedCalibration is returned by CalibrationRepository.admit; "
            "reference/artifact pairs cannot be assembled by callers"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ResolvedCalibration is immutable")

    def __reduce__(self):
        raise TypeError("ResolvedCalibration is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("ResolvedCalibration is process-local and cannot be serialized")

    @classmethod
    def _from_admission(
        cls,
        token: object,
        *,
        repository_token: object,
        reference: CalibrationArtifactRef,
        artifact: CalibrationArtifact,
    ) -> "ResolvedCalibration":
        if token is not _RESOLVED_CALIBRATION_TOKEN:
            raise PermissionError(
                "ResolvedCalibration can only be minted by CalibrationRepository.admit"
            )
        if repository_token is None:
            raise ValueError("ResolvedCalibration repository authority is absent")
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if not isinstance(artifact, CalibrationArtifact):
            raise TypeError("artifact must be CalibrationArtifact")
        resolved = object.__new__(cls)
        object.__setattr__(resolved, "_token", token)
        object.__setattr__(resolved, "_repository_token", repository_token)
        object.__setattr__(resolved, "_reference", reference)
        object.__setattr__(resolved, "_artifact", artifact)
        return resolved

    def _require_authority(self) -> None:
        if (
            type(self) is not ResolvedCalibration
            or self._token is not _RESOLVED_CALIBRATION_TOKEN
            or self._repository_token is None
        ):
            raise PermissionError("ResolvedCalibration authority is invalid")

    @property
    def reference(self) -> CalibrationArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def artifact(self) -> CalibrationArtifact:
        self._require_authority()
        return self._artifact

    def _matches_admission(self, other: object) -> bool:
        self._require_authority()
        if type(other) is not ResolvedCalibration:
            return False
        other._require_authority()
        return (
            self._repository_token is other._repository_token
            and self._reference == other._reference
        )


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


def _apply_readout_model(model: ReadoutModel, frame: Value) -> ReadoutResult:
    """Apply an already-bound model; physical compatibility is caller-owned."""

    extracted = extract_readout_features(model.feature, frame)
    occupied = classify_occupancy(model, extracted)
    assert isinstance(occupied.validity, ComponentValidity)
    values = np.array(extracted.values, dtype="<f8", copy=True)
    values[~occupied.validity.mask] = 0.0
    signals = Value(values, occupied.validity, extracted.schema)
    return ReadoutResult(signals, occupied)


def apply_calibration(
    artifact: CalibrationArtifact,
    frame: Value,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> ReadoutResult:
    """Apply the numeric model to a caller-provided frame value.

    This raw-``Value`` API is deliberately non-authoritative: a ``ValueSchema``
    cannot prove exposure, ROI origin, optical path, camera identity, or pulse
    conditions.  Capture-backed pipelines establish those facts from the
    admitted current capture and compiled pulse lineage before using the
    internal hot path.  This function only prevents structural schema drift.
    """

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    if not isinstance(frame, Value):
        raise TypeError("frame must be Value")
    if frame.schema != artifact.frame_contract.frame_schema:
        raise ValueError("frame schema differs from the calibration FrameContract")
    return _apply_readout_model(artifact.select_model(model_kind), frame)


def calibration_retained_array_nbytes(artifact: CalibrationArtifact) -> int:
    """Return a codec-stable upper bound for retained logical array payloads.

    In-memory analysis may share one immutable validity array between several
    fields, while a canonical decode is free to reconstruct equal fields as
    distinct arrays.  Resource admission therefore counts every persisted
    logical field instead of depending on process-local ndarray identity.
    """

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    arrays: list[np.ndarray] = [
        artifact.site_map.coordinates_xy,
        artifact.site_map.validity.mask,
    ]
    for model in artifact.models:
        arrays.extend(
            [
                model.thresholds,
                model.usable_sites.mask,
                model.feature.boxes_xywh,
                model.feature.valid_sites.mask,
            ]
        )
        if isinstance(model.feature, PerSitePsfFeature):
            arrays.append(model.feature.kernels)
        elif isinstance(model.feature, UniformPsfFeature):
            arrays.append(model.feature.kernel)
    return sum(int(array.nbytes) for array in arrays)


def readout_runtime_scratch_nbytes(
    artifact: CalibrationArtifact,
    model_kind: ReadoutModelKind | None = None,
) -> int:
    """Conservative transient-memory bound for one bound frame evaluation.

    This estimate is owned beside the numeric implementation because it must
    change with its allocation behavior.  In particular, it accounts for the
    observed qCMOS failure mode where a full frame was converted to float64,
    while the normal path now converts only one site window at a time.  A PSF
    annulus with no usable local pixels retains main's whole-frame median
    fallback, so that rare path remains part of the bound.
    """

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    model = artifact.select_model(model_kind)
    feature = model.feature
    dtype = artifact.frame_contract.dtype
    dtype_bytes = int(dtype.itemsize)
    float_temporary = 8
    bool_temporary = 1
    site_scratch = feature.site_axis.size * float_temporary
    box_areas = feature.boxes_xywh[:, 2] * feature.boxes_xywh[:, 3]
    largest_box = int(np.max(box_areas))

    if isinstance(feature, BoxFeature):
        # Boolean finite mask + boolean-indexed source values + float64 reducer
        # input.  The latter two can overlap during dtype conversion.
        return site_scratch + largest_box * (
            bool_temporary + dtype_bytes + float_temporary
        )

    padding = feature.background_padding
    largest_region = int(
        np.max(
            (feature.boxes_xywh[:, 2] + 2 * padding)
            * (feature.boxes_xywh[:, 3] + 2 * padding)
        )
    )
    local_background = largest_region * (
        bool_temporary + 2 * dtype_bytes
    )
    frame_pixels = int(np.prod(artifact.frame_contract.frame_schema.data_shape))
    finite_masks = 2 * bool_temporary if np.issubdtype(dtype, np.inexact) else 0
    global_background_fallback = frame_pixels * (
        2 * dtype_bytes + finite_masks
    )
    weighted_product = largest_box * float_temporary
    # The float64 cutout remains live while background or weighted-product
    # scratch is allocated.
    return site_scratch + largest_box * float_temporary + max(
        local_background,
        global_background_fallback,
        weighted_product,
    )


__all__ = [
    "BackgroundMode",
    "BoxFeature",
    "BoxReducer",
    "CalibrationAnalysisRequest",
    "CalibrationArtifact",
    "CalibrationSourceBinding",
    "GridOrder",
    "PerSitePsfFeature",
    "ReadoutFeature",
    "ReadoutModel",
    "ReadoutModelKind",
    "ReadoutResult",
    "ResolvedCalibration",
    "SiteMap",
    "UniformPsfFeature",
    "apply_calibration",
    "calibration_retained_array_nbytes",
    "classify_occupancy",
    "derive_calibration_readout_physical_context",
    "extract_readout_features",
    "readout_runtime_scratch_nbytes",
]
