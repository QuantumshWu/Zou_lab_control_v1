"""Record-preserving contract for finite and free-running camera adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from zlc_storage import canonical_text, integer, sha256_text


@dataclass(frozen=True, eq=False)
class CameraFrameRecord:
    """One adapter-owned frame copied out of a reusable driver buffer."""

    image: np.ndarray
    source_ordinal: int
    produced_count: int | None
    frame_stamp: int | None
    camera_stamp: int | None
    timestamp_seconds: int | None
    timestamp_microseconds: int | None
    host_received_at_ns: int
    driver_buffer_index: int | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        ordinal = integer(self.source_ordinal, "source_ordinal", nonnegative=True)
        assert ordinal is not None
        object.__setattr__(self, "source_ordinal", ordinal)
        for name in (
            "produced_count",
            "timestamp_seconds",
            "timestamp_microseconds",
            "driver_buffer_index",
        ):
            object.__setattr__(
                self,
                name,
                integer(getattr(self, name), name, optional=True, nonnegative=True),
            )
        for name in ("frame_stamp", "camera_stamp"):
            object.__setattr__(
                self,
                name,
                integer(getattr(self, name), name, optional=True),
            )
        host_received_at_ns = integer(
            self.host_received_at_ns,
            "host_received_at_ns",
            nonnegative=True,
        )
        assert host_received_at_ns is not None
        if host_received_at_ns == 0:
            raise ValueError("host_received_at_ns must be positive")
        object.__setattr__(self, "host_received_at_ns", host_received_at_ns)
        if (
            self.timestamp_microseconds is not None
            and self.timestamp_microseconds >= 1_000_000
        ):
            raise ValueError("timestamp_microseconds must be less than 1_000_000")
        if (self.timestamp_seconds is None) != (self.timestamp_microseconds is None):
            raise ValueError("camera timestamp seconds and microseconds must appear together")
        image = np.array(self.image, copy=True, order="C")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)


@dataclass(frozen=True)
class CameraCaptureTerminalRecord:
    """Adapter readback proving a finite source has stopped and drained."""

    produced_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool

    def __post_init__(self) -> None:
        produced_count = integer(
            self.produced_count,
            "produced_count",
            nonnegative=True,
        )
        assert produced_count is not None
        object.__setattr__(self, "produced_count", produced_count)
        if any(
            type(getattr(self, name)) is not bool
            for name in ("source_stopped", "no_more_frames", "joined")
        ):
            raise TypeError("terminal proof flags must be bool")


@dataclass(frozen=True)
class CameraWorkingPoint:
    """One adapter-read physical working point frozen for capability minting.

    The adapter owns ``settings_fingerprint`` and physical readback only.  It
    cannot grant itself exact-capture qualification; installation/Q0 composition
    supplies that separate authority to the endpoint.  The endpoint converts
    these primitive facts once into its authoritative camera-domain values.
    """

    settings_fingerprint: str
    acquisition_mode: str
    frame_shape_yx: tuple[int, int]
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    dtype: np.dtype
    count_unit: str
    capture_trigger_channels: tuple[str, ...]
    exposure_seconds: float
    required_external_trigger_interval_seconds: float | None
    external_trigger_integration_start_offset_seconds: float | None
    gain: float
    readout_mode: str

    def __post_init__(self) -> None:
        sha256_text(self.settings_fingerprint, "settings_fingerprint")
        object.__setattr__(
            self,
            "acquisition_mode",
            canonical_text(self.acquisition_mode, "acquisition_mode"),
        )
        for name in (
            "frame_shape_yx",
            "sensor_shape_yx",
            "roi_origin_yx",
            "roi_shape_yx",
            "binning_yx",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(
            self,
            "count_unit",
            canonical_text(self.count_unit, "count_unit"),
        )
        if not isinstance(self.capture_trigger_channels, tuple):
            raise TypeError("capture_trigger_channels must be a tuple")
        object.__setattr__(
            self,
            "readout_mode",
            canonical_text(self.readout_mode, "readout_mode"),
        )


@runtime_checkable
class CameraAdapter(Protocol):
    """Record interface consumed by the composition-owned camera endpoint.

    Runtime structural checks only reject missing members; they do not prove
    thread safety, hardware identity, or exact-trigger qualification.  Each
    concrete adapter must pass its contract kit before a composition may bind it.
    This first seam deliberately does not make a real camera READY.
    """

    @property
    def max_pending_records(self) -> int:
        """Hard upper bound on adapter-owned records retained after ``arm``."""

        ...

    @property
    def timeout(self) -> float: ...

    def capture_working_point(self) -> CameraWorkingPoint: ...

    def arm(
        self,
        frames: int | None,
        *,
        max_inflight_frames: int,
        timeout: float,
    ) -> None:
        """Arm with bounded retention; ``None`` means hardware-paced monitor."""

        ...

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> Sequence[CameraFrameRecord]:
        """Read ordered records; terminalization from another thread must unblock it."""

        ...

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """Stop, drain, and freeze one stable terminal record.

        The endpoint may first call this from its bounded terminal worker while
        the arm-owner is blocked in a read, then call it again from the arm-owner
        to complete owner-affine teardown.  Both calls must be thread-safe and
        return the same frozen record; the first call must unblock any read.
        Adapters that cannot meet this two-phase contract require an owner-lane
        host and are not eligible for this endpoint yet.
        """

        ...

    def capture_state(self) -> tuple[bool, int]: ...


__all__ = [
    "CameraAdapter",
    "CameraCaptureTerminalRecord",
    "CameraFrameRecord",
    "CameraWorkingPoint",
]
