"""Installation-owned spatial and acquisition intent for site-map calibration.

The detector must never infer its own authority.  This module therefore joins
one independently configured grid in camera output-pixel coordinates with one
frozen three-event pulse recipe.  The installation gives the same grid value to
the apparatus model and to this profile; notebook convenience code only copies
the already-validated intent into ordinary Capture and Calibration requests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.resources import files
import json
import math

import numpy as np

from zlc_data import AxisId, CoordinateFrameId, immutable_array
from zlc_neutral_atom.capture_application import CaptureRequest
from zlc_neutral_atom.catalog import DefinitionKey, TaskDefinition
from zlc_neutral_atom.runtime.capture import CameraPhysicalFacts
from zlc_pulse import (
    FIELD_DURATION,
    PORT_DIGITAL,
    PulseDocument,
    RepeatRegion,
    pulse_document_from_tree,
    resolve_api_parameters,
)
from zlc_storage import (
    canonical_text,
    positive_integer,
    positive_real,
)

from .calibration import (
    CalibrationAnalysisRequest,
    GridOrder,
)
from .contracts import (
    CalibrationCaptureLayout,
    ReadoutBindingKey,
    _minimum_coordinate_separation,
    validate_camera_spatial_axes,
)


_PACKAGED_SITEMAP_PULSE = "assets/imaging_template.json"
_REFERENCE_BEFORE = "reference_probe_duration_before"
_READOUT = "readout_probe_duration"
_REFERENCE_AFTER = "reference_probe_duration_after"
_EVENT_PARAMETER_IDS = (_REFERENCE_BEFORE, _READOUT, _REFERENCE_AFTER)
SITEMAP_CALIBRATION_TASK_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "calibrate-readout",
)
SITEMAP_CALIBRATION_TASK_DEFINITION = TaskDefinition(
    SITEMAP_CALIBRATION_TASK_KEY,
    "Calibrate readout",
    "zlc_neutral_atom.SitemapCalibrationRequest",
)
SITEMAP_CALIBRATION_TASK_DEFINITIONS = (
    SITEMAP_CALIBRATION_TASK_DEFINITION,
)


@dataclass(frozen=True, slots=True)
class SitemapCalibrationRequest:
    """Freeze the two ordinary Runs that create one readout calibration.

    The capture request is complete before execution.  The calibration request
    itself can only be constructed after that capture has committed its exact
    ``CaptureArtifactRef``, so this value carries the already-frozen analysis
    intent and the internal analysis deadline for the second stage.
    """

    capture_request: CaptureRequest
    analysis: CalibrationAnalysisRequest
    calibration_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.capture_request, CaptureRequest):
            raise TypeError("capture_request must be CaptureRequest")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        object.__setattr__(
            self,
            "calibration_timeout_seconds",
            positive_real(
                self.calibration_timeout_seconds,
                "calibration_timeout_seconds",
            ),
        )


def _pair(value: object, field: str) -> tuple[int, int]:
    try:
        pair = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field} must be a two-integer tuple") from exc
    if len(pair) != 2:
        raise ValueError(f"{field} must have two entries in Y,X order")
    return tuple(
        positive_integer(item, f"{field}[{index}]")
        for index, item in enumerate(pair)
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class ReadoutGridGeometry:
    """Independent ordered site locations in one ROI-local output-pixel frame."""

    frame_shape_yx: tuple[int, int]
    spatial_y_axis_id: AxisId
    spatial_x_axis_id: AxisId
    coordinate_frame: CoordinateFrameId
    grid_shape_yx: tuple[int, int]
    ordering: GridOrder
    expected_centers_xy: np.ndarray

    __hash__ = None

    def __post_init__(self) -> None:
        frame_shape = _pair(self.frame_shape_yx, "frame_shape_yx")
        grid_shape = _pair(self.grid_shape_yx, "grid_shape_yx")
        validate_camera_spatial_axes(
            self.spatial_y_axis_id,
            self.spatial_x_axis_id,
            self.coordinate_frame,
        )
        if not isinstance(self.ordering, GridOrder):
            raise TypeError("ordering must be GridOrder")
        site_count = grid_shape[0] * grid_shape[1]
        centers = immutable_array(
            self.expected_centers_xy,
            dtype="<f8",
            shape=(site_count, 2),
        )
        if not np.all(np.isfinite(centers)):
            raise ValueError("expected_centers_xy must be finite")
        height, width = frame_shape
        if np.any(centers[:, 0] < 0.0) or np.any(centers[:, 0] >= width):
            raise ValueError("site X coordinates lie outside the output frame")
        if np.any(centers[:, 1] < 0.0) or np.any(centers[:, 1] >= height):
            raise ValueError("site Y coordinates lie outside the output frame")
        minimum_separation = _minimum_coordinate_separation(centers)
        if not math.isfinite(minimum_separation) or minimum_separation <= 0.0:
            if site_count > 1:
                raise ValueError("expected site centers must be unique")
            minimum_separation = math.inf
        object.__setattr__(self, "frame_shape_yx", frame_shape)
        object.__setattr__(self, "grid_shape_yx", grid_shape)
        object.__setattr__(self, "expected_centers_xy", centers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReadoutGridGeometry):
            return NotImplemented
        return (
            self.frame_shape_yx == other.frame_shape_yx
            and self.spatial_y_axis_id == other.spatial_y_axis_id
            and self.spatial_x_axis_id == other.spatial_x_axis_id
            and self.coordinate_frame == other.coordinate_frame
            and self.grid_shape_yx == other.grid_shape_yx
            and self.ordering is other.ordering
            and bool(np.array_equal(self.expected_centers_xy, other.expected_centers_xy))
        )

@dataclass(frozen=True, slots=True)
class SitemapAcquisitionProfile:
    """One camera binding's complete, finite three-event calibration recipe."""

    readout_binding: ReadoutBindingKey
    sequencer_role: str
    camera_facts: CameraPhysicalFacts
    geometry: ReadoutGridGeometry
    maximum_site_residual_px: float
    pulse_document: PulseDocument
    trigger_channel: str

    def __post_init__(self) -> None:
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        sequencer_role = canonical_text(self.sequencer_role, "sequencer_role")
        if not isinstance(self.camera_facts, CameraPhysicalFacts):
            raise TypeError("camera_facts must be CameraPhysicalFacts")
        if not isinstance(self.geometry, ReadoutGridGeometry):
            raise TypeError("geometry must be ReadoutGridGeometry")
        maximum_residual = positive_real(
            self.maximum_site_residual_px,
            "maximum_site_residual_px",
        )
        minimum_separation = _minimum_coordinate_separation(
            self.geometry.expected_centers_xy
        )
        if 2.0 * maximum_residual >= minimum_separation:
            raise ValueError(
                "maximum_site_residual_px must be less than half the minimum "
                "site-center separation"
            )
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        trigger = canonical_text(self.trigger_channel, "trigger_channel")
        self.camera_facts.require_single_capture_trigger_channel(trigger)
        if self.geometry.frame_shape_yx != self.camera_facts.output_shape_yx:
            raise ValueError(
                "sitemap geometry differs from the frozen camera output shape"
            )
        geometry_spatial_identity = (
            self.geometry.spatial_y_axis_id,
            self.geometry.spatial_x_axis_id,
            self.geometry.coordinate_frame,
        )
        camera_spatial_identity = (
            self.camera_facts.spatial_y_axis_id,
            self.camera_facts.spatial_x_axis_id,
            self.camera_facts.coordinate_frame,
        )
        if geometry_spatial_identity != camera_spatial_identity:
            raise ValueError(
                "sitemap geometry differs from the frozen camera spatial identity"
            )
        document = self.pulse_document
        if document.repeat is not None:
            raise ValueError("the sitemap base pulse must not contain a repeat region")
        if document.scan_parameters or document.scan_table is not None:
            raise ValueError("the sitemap base pulse must not contain a scan")
        parameters = document.api_parameter_by_id
        if set(parameters) != set(_EVENT_PARAMETER_IDS):
            raise ValueError(
                "sitemap pulse API parameters must be exactly "
                f"{_EVENT_PARAMETER_IDS!r}"
            )
        event_period_ids = tuple(
            parameters[parameter_id].field.period_id
            for parameter_id in _EVENT_PARAMETER_IDS
        )
        if any(
            parameters[parameter_id].field.kind != FIELD_DURATION
            for parameter_id in _EVENT_PARAMETER_IDS
        ):
            raise ValueError(
                "sitemap probe-duration parameters must name period durations"
            )
        period_positions = {
            period.period_id: index for index, period in enumerate(document.periods)
        }
        event_positions = tuple(period_positions[period_id] for period_id in event_period_ids)
        if tuple(sorted(event_positions)) != event_positions or len(set(event_positions)) != 3:
            raise ValueError("sitemap event periods must be distinct and ordered")
        port = document.target.by_key.get(trigger)
        if (
            port is None
            or port.kind != PORT_DIGITAL
            or len(port.lanes) != 1
            or port.safe_value != 0
        ):
            raise ValueError("sitemap trigger must be a one-lane digital port")
        lane_position = document.target.raw_lanes.index(port.lanes[0])
        previous = 0
        rising_period_ids: list[str] = []
        for period in document.periods:
            state = int(period.states[lane_position])
            if state and not previous:
                rising_period_ids.append(period.period_id)
            previous = state
        if tuple(rising_period_ids) != event_period_ids:
            raise ValueError(
                "sitemap reference/readout API periods must be the three trigger edges"
            )
        object.__setattr__(self, "sequencer_role", sequencer_role)
        object.__setattr__(self, "maximum_site_residual_px", maximum_residual)
        object.__setattr__(self, "trigger_channel", trigger)

    @property
    def event_count(self) -> int:
        return 3

    def document_for_repeats(self, repeat_count: int) -> PulseDocument:
        """Repeat the complete three-event hardware sequence, never its data shape."""

        repeats = self._repeat_count(repeat_count)
        if repeats == 1:
            return self.pulse_document
        periods = self.pulse_document.periods
        return replace(
            self.pulse_document,
            repeat=RepeatRegion(periods[0].period_id, periods[-1].period_id, repeats),
        )

    def configured_document_for_repeats(
        self,
        repeat_count: int,
        *,
        reference_exposure_s: float,
        readout_exposure_s: float,
        pulse_document: PulseDocument | None = None,
    ) -> PulseDocument:
        """Freeze Main's long-short-long exposure intent into the fired pulse.

        A caller-selected pulse is admitted through this profile's complete
        three-event/trigger validation before any API value is resolved.  The
        three exposure parameters are then consumed, leaving a fully explicit
        immutable execution document whose repeat encloses the whole bracket.
        """

        reference = positive_real(reference_exposure_s, "reference_exposure_s")
        readout = positive_real(readout_exposure_s, "readout_exposure_s")
        base = self.pulse_document
        if pulse_document is not None:
            if not isinstance(pulse_document, PulseDocument):
                raise TypeError("pulse_document must be PulseDocument or None")
            base = replace(self, pulse_document=pulse_document).pulse_document
        resolved = resolve_api_parameters(
            base,
            {
                _REFERENCE_BEFORE: reference,
                _READOUT: readout,
                _REFERENCE_AFTER: reference,
            },
        )
        repeats = self._repeat_count(repeat_count)
        if repeats == 1:
            return resolved
        periods = resolved.periods
        return replace(
            resolved,
            repeat=RepeatRegion(periods[0].period_id, periods[-1].period_id, repeats),
        )

    def repeat_major_grouping(self, repeat_count: int) -> tuple[tuple[int, int], ...]:
        repeats = self._repeat_count(repeat_count)
        return tuple(
            (repeat, event)
            for repeat in range(repeats)
            for event in range(self.event_count)
        )

    def _repeat_count(self, value: int) -> int:
        return positive_integer(value, "repeat_count")

    def analysis_request(
        self,
        readout_event_axis_id: AxisId,
    ) -> CalibrationAnalysisRequest:
        if not isinstance(readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        return CalibrationAnalysisRequest(
            CalibrationCaptureLayout(readout_event_axis_id, (0, 2), 1),
            self.geometry.grid_shape_yx,
            ordering=self.geometry.ordering,
            expected_centers_xy=self.geometry.expected_centers_xy,
            maximum_site_residual_px=self.maximum_site_residual_px,
        )


def load_packaged_sitemap_pulse() -> PulseDocument:
    """Load the neutral-atom-owned pulse asset without a source-tree/CWD path."""

    resource = files("zlc_neutral_atom").joinpath(_PACKAGED_SITEMAP_PULSE)
    return pulse_document_from_tree(json.loads(resource.read_text(encoding="utf-8")))


__all__ = [
    "ReadoutGridGeometry",
    "SITEMAP_CALIBRATION_TASK_DEFINITION",
    "SITEMAP_CALIBRATION_TASK_DEFINITIONS",
    "SITEMAP_CALIBRATION_TASK_KEY",
    "SitemapAcquisitionProfile",
    "SitemapCalibrationRequest",
    "load_packaged_sitemap_pulse",
]
