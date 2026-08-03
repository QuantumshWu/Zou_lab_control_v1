"""Domain-neutral overlays and inputs for specialised plot kinds.

Application code projects point annotations and pulse programs into these
records.  The plotting package never imports those domain layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from zlc_data import OwnedSnapshot

from ._validation import finite_real as _finite
from ._validation import integer, nonempty_text as _nonempty_text


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _nonnegative_time(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _scan_number(
    value: int | None,
    name: str,
    *,
    optional: bool,
) -> int | None:
    return integer(value, name, minimum=1, optional=optional)


class PointStatus(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class PointMarker:
    point_id: str
    x: float
    y: float
    status: PointStatus = PointStatus.UNKNOWN
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _nonempty_text(self.point_id, "point_id"))
        object.__setattr__(self, "x", _finite(self.x, "point x"))
        object.__setattr__(self, "y", _finite(self.y, "point y"))
        if not isinstance(self.status, PointStatus):
            raise TypeError("status must be PointStatus")
        if self.label is not None:
            object.__setattr__(self, "label", _text(self.label, "point label"))


@dataclass(frozen=True, slots=True)
class ImagePointOverlay:
    """One immutable revision of canonical x/y annotations for an image.

    ``coordinates`` has shape ``(N, 2)`` and stores canonical x then y.  The
    optional metadata arrays, when present, must contain exactly ``N`` items.
    Ordering is stable within a revision; ``point_ids`` provide identity
    across revisions when an application needs it.  Monotonicity is enforced
    by the receiving plot session.
    """

    revision: int
    coordinates: np.ndarray
    point_ids: tuple[str, ...] | None = None
    labels: tuple[str | None, ...] | None = None
    statuses: tuple[PointStatus, ...] | None = None

    def __post_init__(self) -> None:
        revision = integer(self.revision, "revision", minimum=0)
        assert revision is not None
        coordinates = np.asarray(self.coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
            raise ValueError("coordinates must have shape (N, 2)")
        if coordinates.dtype.kind not in "biuf":
            raise TypeError("coordinates must be numeric")
        canonical = np.asarray(coordinates, dtype=np.float64)
        if not np.all(np.isfinite(canonical)):
            raise ValueError("coordinates must be finite")
        frozen = np.frombuffer(
            np.ascontiguousarray(canonical).tobytes(order="C"),
            dtype=np.float64,
        ).reshape(canonical.shape)
        count = int(frozen.shape[0])

        point_ids = self.point_ids
        if point_ids is not None:
            point_ids = tuple(
                _nonempty_text(point_id, "point_id") for point_id in point_ids
            )
            if len(point_ids) != count:
                raise ValueError("point_ids must have one item per coordinate")
            if len(point_ids) != len(set(point_ids)):
                raise ValueError("point_ids must be unique")

        labels = self.labels
        if labels is not None:
            labels = tuple(
                None if label is None else _text(label, "point label")
                for label in labels
            )
            if len(labels) != count:
                raise ValueError("labels must have one item per coordinate")

        statuses = self.statuses
        if statuses is not None:
            statuses = tuple(statuses)
            if len(statuses) != count:
                raise ValueError("statuses must have one item per coordinate")
            if any(not isinstance(status, PointStatus) for status in statuses):
                raise TypeError("statuses must contain PointStatus values")

        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "coordinates", frozen)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "statuses", statuses)

    @property
    def count(self) -> int:
        return int(self.coordinates.shape[0])

    @classmethod
    def empty(cls, revision: int) -> "ImagePointOverlay":
        """Create a revisioned empty layer for an explicit clear operation."""

        return cls(revision=revision, coordinates=np.empty((0, 2), dtype=np.float64))

    @classmethod
    def from_markers(
        cls,
        revision: int,
        markers: Iterable[PointMarker],
    ) -> "ImagePointOverlay":
        """Build the array contract from ergonomic typed point records."""

        values = tuple(markers)
        if any(not isinstance(marker, PointMarker) for marker in values):
            raise TypeError("markers must contain PointMarker values")
        coordinates = np.asarray(
            tuple((marker.x, marker.y) for marker in values),
            dtype=np.float64,
        ).reshape((-1, 2))
        return cls(
            revision=revision,
            coordinates=coordinates,
            point_ids=tuple(marker.point_id for marker in values),
            labels=tuple(marker.label for marker in values),
            statuses=tuple(marker.status for marker in values),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ImageFrame:
    """Immutable image-data and point-overlay presentation transaction.

    ``snapshot`` owns frame ordering. ``overlay`` remains independently
    revisioned, so unchanged layers may be reused and clears use
    :meth:`ImagePointOverlay.empty` rather than an unversioned sentinel.
    """

    snapshot: OwnedSnapshot
    overlay: ImagePointOverlay

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.overlay, ImagePointOverlay):
            raise TypeError("overlay must be ImagePointOverlay")

    @property
    def revision(self) -> int:
        return self.snapshot.ref.revision.value


@dataclass(frozen=True, slots=True)
class PulseChannel:
    channel_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _nonempty_text(self.channel_id, "channel_id"))
        object.__setattr__(self, "label", _nonempty_text(self.label, "channel label"))


@dataclass(frozen=True, slots=True)
class PulseBlock:
    channel_id: str
    start: float
    stop: float
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _nonempty_text(self.channel_id, "channel_id"))
        start = _nonnegative_time(self.start, "pulse start")
        stop = _nonnegative_time(self.stop, "pulse stop")
        if stop <= start:
            raise ValueError("pulse stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "label", _text(self.label, "pulse label"))


@dataclass(frozen=True, slots=True)
class PulseScanRegion:
    """One highlighted scan interval with a positive integer badge."""

    start: float
    stop: float
    number: int

    def __post_init__(self) -> None:
        start = _nonnegative_time(self.start, "scan-region start")
        stop = _nonnegative_time(self.stop, "scan-region stop")
        if stop <= start:
            raise ValueError("scan-region stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(
            self,
            "number",
            _scan_number(self.number, "scan-region number", optional=False),
        )


@dataclass(frozen=True, slots=True)
class PulseAnalogTrace:
    name: str
    label: str
    minimum: float
    maximum: float
    starts: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "analog trace name"))
        object.__setattr__(self, "label", _nonempty_text(self.label, "analog trace label"))
        minimum = _finite(self.minimum, "analog trace minimum")
        maximum = _finite(self.maximum, "analog trace maximum")
        if maximum <= minimum:
            raise ValueError("analog trace maximum must be greater than minimum")
        starts = tuple(
            _nonnegative_time(value, "analog trace start") for value in self.starts
        )
        values = tuple(_finite(value, "analog trace value") for value in self.values)
        if bool(starts) != bool(values):
            raise ValueError(
                "analog trace starts and values must either both be empty or both be present"
            )
        if values and len(starts) != len(values) + 1:
            raise ValueError("analog trace starts must contain one more item than values")
        if any(right <= left for left, right in zip(starts, starts[1:])):
            raise ValueError("analog trace starts must be strictly increasing")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PulseRepeatMarker:
    start: float
    stop: float
    label: str

    def __post_init__(self) -> None:
        start = _nonnegative_time(self.start, "repeat start")
        stop = _nonnegative_time(self.stop, "repeat stop")
        if stop <= start:
            raise ValueError("repeat stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "label", _nonempty_text(self.label, "repeat label"))


@dataclass(frozen=True, slots=True)
class PulseDacScanSegment:
    trace_name: str
    start: float
    stop: float
    value: float
    number: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_name", _nonempty_text(self.trace_name, "DAC trace name"))
        start = _nonnegative_time(self.start, "DAC scan start")
        stop = _nonnegative_time(self.stop, "DAC scan stop")
        if stop <= start:
            raise ValueError("DAC scan stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "value", _finite(self.value, "DAC scan value"))
        object.__setattr__(
            self,
            "number",
            _scan_number(self.number, "DAC scan number", optional=True),
        )


@dataclass(frozen=True, slots=True)
class PulseTimelineData:
    channels: tuple[PulseChannel, ...]
    blocks: tuple[PulseBlock, ...]
    scan_regions: tuple[PulseScanRegion, ...] = ()
    time_unit: str = "s"
    total_duration: float | None = None
    analog_traces: tuple[PulseAnalogTrace, ...] = ()
    repeat_markers: tuple[PulseRepeatMarker, ...] = ()
    repeat_notation: str = ""
    scan_dac_segments: tuple[PulseDacScanSegment, ...] = ()

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if any(not isinstance(item, PulseChannel) for item in channels):
            raise TypeError("channels must contain PulseChannel values")
        ids = tuple(item.channel_id for item in channels)
        if len(ids) != len(set(ids)):
            raise ValueError("pulse channel ids must be unique")
        blocks = tuple(self.blocks)
        if any(not isinstance(item, PulseBlock) for item in blocks):
            raise TypeError("blocks must contain PulseBlock values")
        unknown = {item.channel_id for item in blocks} - set(ids)
        if unknown:
            raise ValueError(f"pulse blocks reference unknown channels: {sorted(unknown)}")
        regions = tuple(self.scan_regions)
        if any(not isinstance(item, PulseScanRegion) for item in regions):
            raise TypeError("scan_regions must contain PulseScanRegion values")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "scan_regions", regions)
        object.__setattr__(self, "time_unit", _nonempty_text(self.time_unit, "time_unit"))
        analog_traces = tuple(self.analog_traces)
        if any(not isinstance(item, PulseAnalogTrace) for item in analog_traces):
            raise TypeError("analog_traces must contain PulseAnalogTrace values")
        if not channels and not analog_traces:
            raise ValueError(
                "PulseTimeline requires at least one digital channel or analog trace"
            )
        names = tuple(item.name for item in analog_traces)
        if len(names) != len(set(names)):
            raise ValueError("analog trace names must be unique")
        repeat_markers = tuple(self.repeat_markers)
        if any(not isinstance(item, PulseRepeatMarker) for item in repeat_markers):
            raise TypeError("repeat_markers must contain PulseRepeatMarker values")
        scan_dac_segments = tuple(self.scan_dac_segments)
        if any(not isinstance(item, PulseDacScanSegment) for item in scan_dac_segments):
            raise TypeError("scan_dac_segments must contain PulseDacScanSegment values")
        unknown_traces = {item.trace_name for item in scan_dac_segments} - set(names)
        if unknown_traces:
            raise ValueError(f"DAC scan segments reference unknown traces: {sorted(unknown_traces)}")
        scan_numbers = [item.number for item in regions]
        scan_numbers.extend(
            item.number for item in scan_dac_segments if item.number is not None
        )
        if len(scan_numbers) != len(set(scan_numbers)):
            raise ValueError("scan numbers must be unique across the timeline")
        total_duration = self.total_duration
        if total_duration is not None:
            total_duration = _finite(total_duration, "total_duration")
            if total_duration <= 0.0:
                raise ValueError("total_duration must be positive")
            latest = max(
                [0.0]
                + [item.stop for item in blocks]
                + [item.stop for item in regions]
                + [item.stop for item in repeat_markers]
                + [item.stop for item in scan_dac_segments]
                + [item.starts[-1] for item in analog_traces if item.starts]
            )
            if total_duration < latest:
                raise ValueError("total_duration cannot end before pulse content")
        object.__setattr__(self, "total_duration", total_duration)
        object.__setattr__(self, "analog_traces", analog_traces)
        object.__setattr__(self, "repeat_markers", repeat_markers)
        object.__setattr__(
            self,
            "repeat_notation",
            _text(self.repeat_notation, "repeat_notation"),
        )
        object.__setattr__(self, "scan_dac_segments", scan_dac_segments)


PlotInput: TypeAlias = OwnedSnapshot | ImageFrame | PulseTimelineData


__all__ = [
    "ImageFrame",
    "ImagePointOverlay",
    "PlotInput",
    "PointMarker",
    "PointStatus",
    "PulseBlock",
    "PulseAnalogTrace",
    "PulseChannel",
    "PulseDacScanSegment",
    "PulseRepeatMarker",
    "PulseScanRegion",
    "PulseTimelineData",
]
