"""Production Hamamatsu DCAM adapter behind the neutral camera contract.

The adapter owns one DCAM handle and one non-daemon SDK thread.  It copies
every accepted driver frame into :class:`CameraFrameRecord` before publishing
it, preserves the DCAM count/stamp/index metadata, and treats an unprovable
ring window as a capture failure.  Hardware identity and one-trigger/one-frame
qualification intentionally remain installation responsibilities.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from zlc_storage import (
    canonical_digest,
    canonical_text,
    finite_real,
    integer,
    positive_integer,
    positive_real,
)

from ._dcam_driver import DcamProperty, DcamSdkDriver, DcamValue
from ._owner_lane import CameraSdkOwnerLane
from .contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)


_CAPTURE_STATUS_BUSY = 1
_WAIT_SLICE_SECONDS = 0.05


class DcamCaptureInterrupted(RuntimeError):
    """A blocked DCAM read was superseded by capture terminalization."""


@dataclass(frozen=True)
class DcamCameraConfig:
    """Requested qCMOS working point.

    ``roi_xywh`` is expressed in unbinned sensor pixels as
    ``(x, y, width, height)``.  Driver-ring cardinality belongs to each arm:
    complete finite cardinality for exact capture, or the declared monitor
    history geometry for continuous acquisition.
    """

    capture_trigger_channels: tuple[str, ...]
    exposure_seconds: float = 0.02
    readout_speed: int = 1
    binning: int = 1
    roi_xywh: tuple[int, int, int, int] | None = None
    device_index: int = 0
    timeout_seconds: float = 10.0
    sensor_mode: int | None = None
    trigger_global_exposure: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capture_trigger_channels, tuple):
            raise TypeError("capture_trigger_channels must be a tuple")
        channels = tuple(
            canonical_text(value, "capture_trigger_channels item")
            for value in self.capture_trigger_channels
        )
        if not channels:
            raise ValueError("capture_trigger_channels cannot be empty")
        if len(set(channels)) != len(channels):
            raise ValueError("capture_trigger_channels cannot contain duplicates")
        object.__setattr__(self, "capture_trigger_channels", channels)
        object.__setattr__(
            self,
            "exposure_seconds",
            positive_real(self.exposure_seconds, "exposure_seconds"),
        )
        object.__setattr__(
            self,
            "readout_speed",
            positive_integer(self.readout_speed, "readout_speed"),
        )
        binning = positive_integer(self.binning, "binning")
        if binning not in (1, 2, 4, 8, 16):
            raise ValueError("binning must be one of 1, 2, 4, 8, or 16")
        object.__setattr__(self, "binning", binning)
        device_index = integer(self.device_index, "device_index", nonnegative=True)
        assert device_index is not None
        object.__setattr__(self, "device_index", device_index)
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )
        for name in ("sensor_mode", "trigger_global_exposure"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, positive_integer(value, name))
        roi = self.roi_xywh
        if roi is not None:
            if not isinstance(roi, tuple) or len(roi) != 4:
                raise TypeError("roi_xywh must be a four-item tuple or None")
            normalized: list[int] = []
            for index, value in enumerate(roi):
                minimum = 0 if index < 2 else 1
                item = integer(value, f"roi_xywh[{index}]", minimum=minimum)
                assert item is not None
                normalized.append(item)
            object.__setattr__(self, "roi_xywh", tuple(normalized))


class DcamCameraAdapter:
    """External-trigger qCMOS implementation of the CameraAdapter protocol."""

    def __init__(
        self,
        config: DcamCameraConfig,
        *,
        driver: object | None = None,
        library_path: str | None = None,
    ) -> None:
        if not isinstance(config, DcamCameraConfig):
            raise TypeError("config must be DcamCameraConfig")
        if driver is not None and library_path is not None:
            raise ValueError("library_path cannot accompany an injected driver")
        self._config = config
        self._driver = (
            DcamSdkDriver(library_path=library_path) if driver is None else driver
        )
        self._lane = CameraSdkOwnerLane("zlc-dcam-camera-owner")
        self._state_lock = threading.RLock()
        self._finish_requested = threading.Event()
        self._device: object | None = None
        self._runtime_owned = False
        self._open = False
        self._armed = False
        self._capture_running = False
        self._ring_size = 0
        self._expected_frames: int | None = 0
        self._copied_count = 0
        self._last_transfer_count = 0
        self._last_transfer_newest: int | None = None
        self._pending: deque[CameraFrameRecord] = deque()
        self._terminal: CameraCaptureTerminalRecord | None = None
        self._terminal_failure: RuntimeError | None = None
        try:
            self._lane.call(self._open_on_owner)
        except BaseException:
            self._lane.close()
            raise

    @property
    def timeout(self) -> float:
        return self._config.timeout_seconds

    def _require_device(self) -> object:
        if self._device is None or not self._open:
            raise RuntimeError("DCAM camera is closed")
        return self._device

    def _open_on_owner(self) -> None:
        owned = bool(self._driver.initialize())
        self._runtime_owned = owned
        try:
            self._device = self._driver.open_device(self._config.device_index)
            self._open = True
            self._apply_settings_on_owner(self._config)
        except BaseException as primary:
            device = self._device
            self._device = None
            self._open = False
            if device is not None:
                try:
                    device.close()
                except BaseException as secondary:
                    primary.add_note(f"DCAM close after open failure also failed: {secondary}")
            if owned:
                try:
                    self._driver.uninitialize()
                except BaseException as secondary:
                    primary.add_note(
                        f"DCAM uninitialize after open failure also failed: {secondary}"
                    )
            self._runtime_owned = False
            raise

    @staticmethod
    def _integral(value: float, field: str, *, positive: bool = False) -> int:
        result = finite_real(value, field)
        if not result.is_integer():
            raise RuntimeError(f"DCAM returned non-integral {field}: {result!r}")
        normalized = int(result)
        if positive and normalized <= 0:
            raise RuntimeError(f"DCAM returned non-positive {field}: {normalized!r}")
        return normalized

    @staticmethod
    def _set_exact(device: object, property_id: DcamProperty, value: int) -> None:
        applied = float(device.set_get_property(property_id, value))
        if not math.isclose(applied, float(value), rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"DCAM property {property_id.name} read back {applied!r}, "
                f"expected {value!r}"
            )

    def _sensor_grid_on_owner(self) -> tuple[int, int, int, int]:
        device = self._require_device()
        h_step, h_max = device.property_attributes(DcamProperty.SUBARRAY_HSIZE)
        v_step, v_max = device.property_attributes(DcamProperty.SUBARRAY_VSIZE)
        width = self._integral(h_max, "sensor width", positive=True)
        height = self._integral(v_max, "sensor height", positive=True)
        horizontal_step = self._integral(h_step, "horizontal ROI step", positive=True)
        vertical_step = self._integral(v_step, "vertical ROI step", positive=True)
        return horizontal_step, vertical_step, width, height

    def _apply_settings_on_owner(self, config: DcamCameraConfig) -> CameraWorkingPoint:
        if self._armed:
            raise RuntimeError("qCMOS settings cannot change while armed")
        device = self._require_device()
        device.set_get_property(DcamProperty.EXPOSURE_TIME, config.exposure_seconds)
        self._set_exact(
            device,
            DcamProperty.TRIGGER_SOURCE,
            int(DcamValue.TRIGGER_SOURCE_EXTERNAL),
        )
        self._set_exact(
            device,
            DcamProperty.TRIGGER_ACTIVE,
            int(DcamValue.TRIGGER_ACTIVE_EDGE),
        )
        self._set_exact(
            device,
            DcamProperty.TRIGGER_POLARITY,
            int(DcamValue.TRIGGER_POLARITY_POSITIVE),
        )
        self._set_exact(device, DcamProperty.READOUT_SPEED, config.readout_speed)
        self._set_exact(device, DcamProperty.BINNING, config.binning)
        if config.sensor_mode is not None:
            self._set_exact(device, DcamProperty.SENSOR_MODE, config.sensor_mode)
        if config.trigger_global_exposure is not None:
            self._set_exact(
                device,
                DcamProperty.TRIGGER_GLOBAL_EXPOSURE,
                config.trigger_global_exposure,
            )

        self._set_exact(device, DcamProperty.SUBARRAY_MODE, int(DcamValue.MODE_OFF))
        h_step, v_step, sensor_width, sensor_height = self._sensor_grid_on_owner()
        roi = config.roi_xywh
        if roi is not None:
            x, y, width, height = roi
            if x + width > sensor_width or y + height > sensor_height:
                raise ValueError("roi_xywh extends beyond the DCAM sensor")
            if any(
                value % step
                for value, step in (
                    (x, h_step),
                    (width, h_step),
                    (y, v_step),
                    (height, v_step),
                )
            ):
                raise ValueError("roi_xywh is not aligned to the DCAM subarray grid")
            device.set_get_property(DcamProperty.SUBARRAY_HPOS, 0)
            device.set_get_property(DcamProperty.SUBARRAY_VPOS, 0)
            device.set_get_property(DcamProperty.SUBARRAY_HSIZE, width)
            device.set_get_property(DcamProperty.SUBARRAY_VSIZE, height)
            device.set_get_property(DcamProperty.SUBARRAY_HPOS, x)
            device.set_get_property(DcamProperty.SUBARRAY_VPOS, y)
            self._set_exact(device, DcamProperty.SUBARRAY_MODE, int(DcamValue.MODE_ON))
        return self._read_working_point_on_owner()

    def _read_working_point_on_owner(self) -> CameraWorkingPoint:
        device = self._require_device()

        def prop(property_id: DcamProperty) -> float:
            return float(device.get_property(property_id))

        exposure = positive_real(prop(DcamProperty.EXPOSURE_TIME), "applied exposure")
        interval = positive_real(
            prop(DcamProperty.TIMING_MIN_TRIGGER_INTERVAL),
            "minimum external trigger interval",
        )
        integration_offset = finite_real(
            prop(DcamProperty.TIMING_GLOBAL_EXPOSURE_DELAY),
            "global exposure delay",
            minimum=0.0,
        )
        trigger_source = self._integral(prop(DcamProperty.TRIGGER_SOURCE), "trigger source")
        trigger_active = self._integral(prop(DcamProperty.TRIGGER_ACTIVE), "trigger active")
        trigger_polarity = self._integral(
            prop(DcamProperty.TRIGGER_POLARITY), "trigger polarity"
        )
        expected_triggers = (
            int(DcamValue.TRIGGER_SOURCE_EXTERNAL),
            int(DcamValue.TRIGGER_ACTIVE_EDGE),
            int(DcamValue.TRIGGER_POLARITY_POSITIVE),
        )
        if (trigger_source, trigger_active, trigger_polarity) != expected_triggers:
            raise RuntimeError("qCMOS trigger readback is not external rising-edge mode")
        readout_speed = self._integral(
            prop(DcamProperty.READOUT_SPEED), "readout speed", positive=True
        )
        sensor_mode = self._integral(
            prop(DcamProperty.SENSOR_MODE), "sensor mode", positive=True
        )
        global_exposure = self._integral(
            prop(DcamProperty.TRIGGER_GLOBAL_EXPOSURE),
            "trigger global exposure",
            positive=True,
        )
        binning = self._integral(prop(DcamProperty.BINNING), "binning", positive=True)
        if binning not in (1, 2, 4, 8, 16):
            raise RuntimeError(f"qCMOS returned unsupported binning {binning}")
        _h_step, _v_step, sensor_width, sensor_height = self._sensor_grid_on_owner()
        subarray_mode = self._integral(
            prop(DcamProperty.SUBARRAY_MODE), "subarray mode", positive=True
        )
        if subarray_mode == int(DcamValue.MODE_OFF):
            origin_yx = (0, 0)
            roi_shape_yx = (sensor_height, sensor_width)
        elif subarray_mode == int(DcamValue.MODE_ON):
            x = self._integral(prop(DcamProperty.SUBARRAY_HPOS), "subarray x")
            y = self._integral(prop(DcamProperty.SUBARRAY_VPOS), "subarray y")
            width = self._integral(
                prop(DcamProperty.SUBARRAY_HSIZE), "subarray width", positive=True
            )
            height = self._integral(
                prop(DcamProperty.SUBARRAY_VSIZE), "subarray height", positive=True
            )
            if x < 0 or y < 0 or x + width > sensor_width or y + height > sensor_height:
                raise RuntimeError("qCMOS returned an invalid subarray readback")
            origin_yx = (y, x)
            roi_shape_yx = (height, width)
        else:
            raise RuntimeError(f"qCMOS returned unsupported subarray mode {subarray_mode}")
        frame_width = self._integral(
            prop(DcamProperty.IMAGE_WIDTH), "image width", positive=True
        )
        frame_height = self._integral(
            prop(DcamProperty.IMAGE_HEIGHT), "image height", positive=True
        )
        pixel_type = self._integral(
            prop(DcamProperty.IMAGE_PIXEL_TYPE), "image pixel type", positive=True
        )
        if pixel_type == int(DcamValue.PIXEL_MONO16):
            dtype = np.dtype("<u2")
        elif pixel_type == int(DcamValue.PIXEL_MONO8):
            dtype = np.dtype("u1")
        else:
            raise RuntimeError(f"qCMOS returned unsupported pixel type {pixel_type}")
        frame_shape_yx = (frame_height, frame_width)
        if frame_shape_yx != (
            roi_shape_yx[0] // binning,
            roi_shape_yx[1] // binning,
        ):
            raise RuntimeError(
                "qCMOS image geometry differs from ROI/binning readback"
            )
        primitive = {
            "acquisition_mode": "EXTERNAL_TRIGGERED",
            "frame_shape_yx": frame_shape_yx,
            "sensor_shape_yx": (sensor_height, sensor_width),
            "roi_origin_yx": origin_yx,
            "roi_shape_yx": roi_shape_yx,
            "binning_yx": (binning, binning),
            "dtype": dtype.str,
            "count_unit": "count",
            "capture_trigger_channels": self._config.capture_trigger_channels,
            "exposure_seconds": exposure,
            "required_external_trigger_interval_seconds": interval,
            "external_trigger_integration_start_offset_seconds": integration_offset,
            "readout_speed": readout_speed,
            "sensor_mode": sensor_mode,
            "trigger_global_exposure": global_exposure,
            "trigger_source": trigger_source,
            "trigger_active": trigger_active,
            "trigger_polarity": trigger_polarity,
            "subarray_mode": subarray_mode,
        }
        return CameraWorkingPoint(
            settings_fingerprint=canonical_digest(primitive),
            acquisition_mode="EXTERNAL_TRIGGERED",
            frame_shape_yx=frame_shape_yx,
            sensor_shape_yx=(sensor_height, sensor_width),
            roi_origin_yx=origin_yx,
            roi_shape_yx=roi_shape_yx,
            binning_yx=(binning, binning),
            dtype=dtype,
            count_unit="count",
            capture_trigger_channels=self._config.capture_trigger_channels,
            exposure_seconds=exposure,
            required_external_trigger_interval_seconds=interval,
            external_trigger_integration_start_offset_seconds=integration_offset,
            gain=1.0,
            readout_mode=(
                "dcam:"
                f"readout={readout_speed};sensor={sensor_mode};"
                f"global_exposure={global_exposure}"
            ),
        )

    def capture_working_point(self) -> CameraWorkingPoint:
        return self._lane.call(self._read_working_point_on_owner)

    def configure_exposure_seconds(self, exposure_seconds: float) -> None:
        candidate = replace(
            self._config,
            exposure_seconds=positive_real(exposure_seconds, "exposure_seconds"),
        )

        def apply() -> None:
            self._apply_settings_on_owner(candidate)
            self._config = candidate

        self._lane.call(apply)

    @staticmethod
    def _finite_groups(
        frames: int | None,
        source_group_sizes: tuple[int, ...] | None,
    ) -> tuple[int | None, tuple[int, ...] | None]:
        if frames is None:
            if source_group_sizes is not None:
                raise ValueError("monitor arm cannot declare source_group_sizes")
            return None, None
        expected = positive_integer(frames, "frames")
        if not isinstance(source_group_sizes, tuple):
            raise TypeError("finite arm requires tuple source_group_sizes")
        groups = tuple(
            positive_integer(value, "source_group_sizes item")
            for value in source_group_sizes
        )
        if not groups or sum(groups) != expected:
            raise ValueError("source_group_sizes must exactly cover frames")
        return expected, groups

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        expected, _groups = self._finite_groups(frames, source_group_sizes)
        buffer_count = positive_integer(buffer_frame_count, "buffer_frame_count")
        positive_real(timeout, "timeout")
        if expected is not None and buffer_count != expected:
            raise ValueError(
                "finite buffer_frame_count must equal the complete frame count"
            )

        def arm_on_owner() -> None:
            if self._armed:
                raise RuntimeError("qCMOS is already armed")
            device = self._require_device()
            before = self._read_working_point_on_owner()
            device.allocate_buffer(buffer_count)
            started = False
            try:
                device.start_capture()
                started = True
                self._capture_running = True
                after = self._read_working_point_on_owner()
                if after != before:
                    raise RuntimeError("qCMOS working point changed across arm")
                count, newest = device.transfer_info()
                if int(count) != 0 or int(newest) != -1:
                    raise RuntimeError(
                        "qCMOS transfer counter did not reset at the arm boundary"
                    )
            except BaseException as primary:
                stopped = not started
                if started:
                    try:
                        device.stop_capture()
                    except BaseException as secondary:
                        primary.add_note(f"DCAM stop after arm failure also failed: {secondary}")
                    else:
                        stopped = True
                        self._capture_running = False
                if stopped:
                    try:
                        device.release_buffer()
                    except BaseException as secondary:
                        primary.add_note(
                            f"DCAM release after arm failure also failed: {secondary}"
                        )
                        stopped = False
                if not stopped:
                    # The driver still owns storage or may still be capturing.
                    # Preserve that fact so explicit terminalization can retry;
                    # never hide it behind a failed arm result.
                    with self._state_lock:
                        self._finish_requested.set()
                        self._armed = True
                        self._ring_size = buffer_count
                        self._expected_frames = expected
                        self._copied_count = 0
                        self._last_transfer_count = 0
                        self._last_transfer_newest = None
                        self._pending.clear()
                        self._terminal = None
                raise
            with self._state_lock:
                self._finish_requested.clear()
                self._armed = True
                self._capture_running = True
                self._ring_size = buffer_count
                self._expected_frames = expected
                self._copied_count = 0
                self._last_transfer_count = 0
                self._last_transfer_newest = None
                self._pending.clear()
                self._terminal = None
                self._terminal_failure = None

        self._lane.call(arm_on_owner)

    def _check_not_finishing(self) -> None:
        if self._finish_requested.is_set():
            raise DcamCaptureInterrupted("qCMOS capture is terminalizing")

    def _observe_transfer_on_owner(self) -> tuple[int, int | None]:
        device = self._require_device()
        count_raw, newest_raw = device.transfer_info()
        count = int(count_raw)
        newest = int(newest_raw)
        if count < 0:
            raise RuntimeError("qCMOS produced count is negative")
        if count < self._last_transfer_count or count < self._copied_count:
            raise RuntimeError("qCMOS produced count moved backwards")
        if count == 0:
            if newest != -1:
                raise RuntimeError("qCMOS empty transfer window has a newest frame index")
            newest_value: int | None = None
        else:
            if not 0 <= newest < self._ring_size:
                raise RuntimeError("qCMOS newest frame index is outside the allocated ring")
            newest_value = newest
            if self._last_transfer_count > 0 and self._last_transfer_newest is not None:
                expected_newest = (
                    self._last_transfer_newest + count - self._last_transfer_count
                ) % self._ring_size
                if newest != expected_newest:
                    raise RuntimeError(
                        "qCMOS transfer count/newest-index pair is inconsistent"
                    )
        if self._expected_frames is not None and count > self._expected_frames:
            raise RuntimeError("qCMOS produced more frames than the finite arm declared")
        self._last_transfer_count = count
        self._last_transfer_newest = newest_value
        return count, newest_value

    def _drain_snapshot_on_owner(
        self,
        available: int,
        newest: int,
        deadline: float,
    ) -> None:
        if available - self._copied_count > self._ring_size:
            raise RuntimeError("qCMOS ring overrun overwrote an unread frame")
        device = self._require_device()
        snapshot_available = available
        snapshot_newest = newest
        while self._copied_count < snapshot_available:
            self._check_not_finishing()
            if time.monotonic() >= deadline:
                raise TimeoutError("qCMOS read deadline expired while draining the ring")
            ordinal = self._copied_count
            distance = snapshot_available - 1 - ordinal
            ring_index = (snapshot_newest - distance) % self._ring_size
            image, frame_stamp, camera_stamp, seconds, microseconds = device.copy_frame(
                ring_index
            )
            self._check_not_finishing()
            post_count, _post_newest = self._observe_transfer_on_owner()
            if post_count - ordinal > self._ring_size:
                raise RuntimeError(
                    "qCMOS ring advanced far enough to overwrite a frame during copy"
                )
            record = CameraFrameRecord(
                image=image,
                source_ordinal=ordinal,
                produced_count=ordinal + 1,
                frame_stamp=frame_stamp,
                camera_stamp=camera_stamp,
                timestamp_seconds=seconds,
                timestamp_microseconds=microseconds,
                host_received_at_ns=time.time_ns(),
                driver_buffer_index=ring_index,
            )
            self._check_not_finishing()
            if time.monotonic() >= deadline:
                raise TimeoutError("qCMOS read deadline expired before frame commit")
            with self._state_lock:
                self._pending.append(record)
                self._copied_count += 1

    def _read_on_owner(
        self,
        wanted: int,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        if not self._armed:
            raise RuntimeError("qCMOS read requires an armed capture")
        deadline = time.monotonic() + timeout
        result: list[CameraFrameRecord] = []
        device = self._require_device()
        while len(result) < wanted:
            self._check_not_finishing()
            with self._state_lock:
                while self._pending and len(result) < wanted:
                    result.append(self._pending.popleft())
            if len(result) == wanted:
                break
            available, newest = self._observe_transfer_on_owner()
            if available > self._copied_count:
                assert newest is not None
                self._drain_snapshot_on_owner(
                    available,
                    newest,
                    deadline,
                )
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            ready = device.wait_frame_ready(
                max(1, int(min(_WAIT_SLICE_SECONDS, remaining) * 1000.0))
            )
            self._check_not_finishing()
            if not ready and time.monotonic() >= deadline:
                break
        if len(result) != wanted and exact:
            raise TimeoutError(
                f"qCMOS returned {len(result)} of {wanted} exact frame(s)"
            )
        return result

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        wanted = positive_integer(n, "n")
        bounded_timeout = finite_real(timeout, "timeout", minimum=0.0)
        if type(exact) is not bool:
            raise TypeError("exact must be bool")
        return self._lane.call(
            lambda: self._read_on_owner(wanted, bounded_timeout, exact)
        )

    def _finish_on_owner(self) -> CameraCaptureTerminalRecord:
        if self._terminal is not None:
            return self._terminal
        prior_terminal_failure = self._terminal_failure
        if not self._armed:
            if prior_terminal_failure is not None:
                raise prior_terminal_failure
            self._terminal = CameraCaptureTerminalRecord(0, True, True, True)
            return self._terminal
        device = self._require_device()
        if self._capture_running:
            device.stop_capture()
            if int(device.capture_status()) == _CAPTURE_STATUS_BUSY:
                raise RuntimeError("qCMOS remained busy after cap_stop; ring retained")
            self._capture_running = False

        def stable_snapshot() -> tuple[int, int]:
            count_raw, newest_raw = device.transfer_info()
            count = int(count_raw)
            newest = int(newest_raw)
            if count < max(
                self._copied_count,
                self._last_transfer_count,
            ):
                raise RuntimeError("qCMOS final produced count moved backwards")
            if count == 0:
                if newest != -1:
                    raise RuntimeError("qCMOS final empty window has a newest index")
            elif not 0 <= newest < self._ring_size:
                raise RuntimeError("qCMOS final newest index is outside the ring")
            return count, newest

        final_error: BaseException | None = None
        produced_count = 0
        try:
            first = stable_snapshot()
            second = stable_snapshot()
            if first != second:
                raise RuntimeError("qCMOS transfer state did not stabilize after cap_stop")
            produced_count, newest = second
            if self._last_transfer_count > 0 and self._last_transfer_newest is not None:
                expected_newest = (
                    self._last_transfer_newest
                    + produced_count
                    - self._last_transfer_count
                ) % self._ring_size
                if newest != expected_newest:
                    raise RuntimeError(
                        "qCMOS final count/newest-index pair is inconsistent"
                    )
        except BaseException as error:
            final_error = error
        try:
            device.release_buffer()
        except BaseException:
            raise RuntimeError("qCMOS buffer release failed after cap_stop")
        with self._state_lock:
            self._armed = False
            self._capture_running = False
            self._ring_size = 0
            self._expected_frames = 0
            self._pending.clear()
        if final_error is not None:
            if prior_terminal_failure is not None:
                raise prior_terminal_failure
            terminal_failure = RuntimeError(
                "qCMOS final transfer state is not authoritative"
            )
            self._terminal_failure = terminal_failure
            raise terminal_failure from final_error
        if prior_terminal_failure is not None:
            raise prior_terminal_failure
        self._terminal = CameraCaptureTerminalRecord(
            produced_count,
            True,
            True,
            True,
        )
        return self._terminal

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self._finish_requested.set()
        return self._lane.call(self._finish_on_owner)

    def capture_state(self) -> tuple[bool, int]:
        with self._state_lock:
            return self._armed, len(self._pending)

    def observed_produced_count(self) -> int:
        """Read the DCAM transfer counter without consuming a frame."""

        def observe() -> int:
            if not self._armed:
                raise RuntimeError("qCMOS produced count requires an armed capture")
            count, _newest = self._observe_transfer_on_owner()
            return count

        return self._lane.call(observe)

    def _close_on_owner(self) -> None:
        device = self._device
        if device is not None:
            device.close()
            self._device = None
        if self._runtime_owned:
            self._driver.uninitialize()
            self._runtime_owned = False
        self._open = False

    def close(self) -> None:
        if not self._open and self._device is None and not self._runtime_owned:
            self._lane.close()
            return
        if self.capture_state()[0]:
            self.finish_record_capture()
        self._lane.call(self._close_on_owner)
        self._lane.close()


__all__ = [
    "DcamCameraAdapter",
    "DcamCameraConfig",
    "DcamCaptureInterrupted",
]
