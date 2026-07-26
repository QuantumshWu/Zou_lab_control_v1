"""Current-contract Basler pylon adapter for the MOT camera.

Finite acquisition is external-triggered and ordered.  Monitor acquisition is
free-running with pylon's latest-image strategy.  Every accepted Mono8 frame is
copied before the driver result is released; the adapter never widens it to a
floating dtype and never grants itself exact-trigger qualification.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from zlc_storage import canonical_digest, canonical_text, integer, positive_integer, positive_real

from ._owner_lane import CameraSdkOwnerLane
from .contract import CameraCaptureTerminalRecord, CameraFrameRecord, CameraWorkingPoint


_RETRIEVE_SLICE_SECONDS = 0.05


class PylonCaptureInterrupted(RuntimeError):
    """A blocked pylon read was superseded by capture terminalization."""


@dataclass(frozen=True, slots=True)
class PylonCameraConfig:
    """Requested dual-mode Basler working point.

    ``roi_xywh`` is ``(x, y, width, height)`` in unbinned sensor pixels.
    Finite captures always use ``trigger_source``; monitor arms temporarily
    disable TriggerMode and use the same exposure/ROI/Mono8 payload contract.
    """

    serial: str
    capture_trigger_channels: tuple[str, ...]
    exposure_seconds: float = 0.005
    trigger_source: str = "Line1"
    roi_xywh: tuple[int, int, int, int] | None = None
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "serial", canonical_text(self.serial, "pylon serial"))
        channels = tuple(
            canonical_text(value, "capture_trigger_channels item")
            for value in self.capture_trigger_channels
        )
        if not channels or len(channels) != len(set(channels)):
            raise ValueError("capture_trigger_channels must be non-empty and unique")
        object.__setattr__(self, "capture_trigger_channels", channels)
        object.__setattr__(
            self,
            "exposure_seconds",
            positive_real(self.exposure_seconds, "exposure_seconds"),
        )
        object.__setattr__(
            self,
            "trigger_source",
            canonical_text(self.trigger_source, "pylon trigger_source"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )
        roi = self.roi_xywh
        if roi is not None:
            if not isinstance(roi, tuple) or len(roi) != 4:
                raise TypeError("roi_xywh must be a four-item tuple or None")
            normalized = tuple(
                integer(value, f"roi_xywh[{index}]", minimum=0 if index < 2 else 1)
                for index, value in enumerate(roi)
            )
            object.__setattr__(self, "roi_xywh", normalized)


def _node_value(node: object, field: str) -> object:
    getter = getattr(node, "GetValue", None)
    if not callable(getter):
        raise RuntimeError(f"pylon node {field} has no GetValue()")
    return getter()


def _set_node(node: object, value: object, field: str) -> None:
    setter = getattr(node, "SetValue", None)
    if not callable(setter):
        raise RuntimeError(f"pylon node {field} has no SetValue()")
    setter(value)
    if _node_value(node, field) != value:
        raise RuntimeError(f"pylon {field} readback differs from the requested value")


def _integer_node(node: object, field: str, method: str) -> int:
    getter = getattr(node, method, None)
    if not callable(getter):
        raise RuntimeError(f"pylon node {field} has no {method}()")
    value = integer(getter(), field)
    assert value is not None
    return value


def _snap_node(value: int, node: object, field: str) -> int:
    lower = _integer_node(node, field, "GetMin")
    upper = _integer_node(node, field, "GetMax")
    increment = max(1, _integer_node(node, field, "GetInc"))
    bounded = max(lower, min(upper, value))
    return lower + ((bounded - lower) // increment) * increment


class PylonCameraAdapter:
    """Basler adapter implementing finite external-trigger and live monitor arms."""

    def __init__(self, config: PylonCameraConfig, *, pylon_module: object | None = None) -> None:
        if not isinstance(config, PylonCameraConfig):
            raise TypeError("config must be PylonCameraConfig")
        if pylon_module is None:
            try:
                from pypylon import pylon as pylon_module
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError("Basler hardware requires pypylon") from exc
        self._config = config
        self._pylon = pylon_module
        self._lane = CameraSdkOwnerLane("zlc-pylon-camera-owner")
        self._state_lock = threading.RLock()
        self._finish_requested = threading.Event()
        self._camera: object | None = None
        self._open = False
        self._armed = False
        self._monitor_mode = False
        self._expected_frames: int | None = 0
        self._delivered_count = 0
        self._terminal: CameraCaptureTerminalRecord | None = None
        try:
            self._lane.call(self._open_on_owner)
        except BaseException:
            self._lane.close()
            raise

    @property
    def timeout(self) -> float:
        return self._config.timeout_seconds

    def _require_camera(self) -> object:
        if self._camera is None or not self._open:
            raise RuntimeError("pylon camera is closed")
        return self._camera

    def _open_on_owner(self) -> None:
        factory = self._pylon.TlFactory.GetInstance()
        infos = tuple(factory.EnumerateDevices())
        match = next(
            (
                info
                for info in infos
                if str(info.GetSerialNumber()) == self._config.serial
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"pylon camera with serial {self._config.serial!r} was not found"
            )
        camera = self._pylon.InstantCamera(factory.CreateDevice(match))
        camera.Open()
        self._camera = camera
        self._open = True
        try:
            self._apply_settings_on_owner(self._config)
        except BaseException as primary:
            try:
                camera.Close()
            except BaseException as secondary:
                primary.add_note(f"pylon close after open failure also failed: {secondary}")
            self._camera = None
            self._open = False
            raise

    def _apply_trigger_on_owner(self, *, monitor: bool) -> None:
        camera = self._require_camera()
        _set_node(camera.TriggerSelector, "FrameStart", "TriggerSelector")
        if monitor:
            _set_node(camera.TriggerMode, "Off", "TriggerMode")
        else:
            _set_node(camera.TriggerMode, "On", "TriggerMode")
            _set_node(camera.TriggerSource, self._config.trigger_source, "TriggerSource")
            activation = getattr(camera, "TriggerActivation", None)
            if activation is not None:
                _set_node(activation, "RisingEdge", "TriggerActivation")

    def _apply_roi_on_owner(self, roi: tuple[int, int, int, int] | None) -> None:
        camera = self._require_camera()
        _set_node(camera.OffsetX, _integer_node(camera.OffsetX, "OffsetX", "GetMin"), "OffsetX")
        _set_node(camera.OffsetY, _integer_node(camera.OffsetY, "OffsetY", "GetMin"), "OffsetY")
        if roi is None:
            width = _integer_node(camera.Width, "Width", "GetMax")
            height = _integer_node(camera.Height, "Height", "GetMax")
            _set_node(camera.Width, width, "Width")
            _set_node(camera.Height, height, "Height")
            return
        x, y, width, height = roi
        width = _snap_node(width, camera.Width, "Width")
        height = _snap_node(height, camera.Height, "Height")
        _set_node(camera.Width, width, "Width")
        _set_node(camera.Height, height, "Height")
        x = _snap_node(x, camera.OffsetX, "OffsetX")
        y = _snap_node(y, camera.OffsetY, "OffsetY")
        if x + width > _integer_node(camera.Width, "Width", "GetMax"):
            raise ValueError("pylon ROI exceeds the sensor width")
        if y + height > _integer_node(camera.Height, "Height", "GetMax"):
            raise ValueError("pylon ROI exceeds the sensor height")
        _set_node(camera.OffsetX, x, "OffsetX")
        _set_node(camera.OffsetY, y, "OffsetY")

    def _apply_settings_on_owner(self, config: PylonCameraConfig) -> CameraWorkingPoint:
        if self._armed:
            raise RuntimeError("pylon settings cannot change while armed")
        camera = self._require_camera()
        _set_node(camera.PixelFormat, "Mono8", "PixelFormat")
        camera.ExposureTime.SetValue(config.exposure_seconds * 1e6)
        observed_exposure = positive_real(
            float(_node_value(camera.ExposureTime, "ExposureTime")) * 1e-6,
            "pylon exposure readback",
        )
        self._apply_trigger_on_owner(monitor=False)
        self._apply_roi_on_owner(config.roi_xywh)
        return self._read_working_point_on_owner()

    def _read_optional_positive_rate_on_owner(self) -> float | None:
        camera = self._require_camera()
        for name in (
            "BslResultingAcquisitionFrameRate",
            "ResultingFrameRate",
            "ResultingFrameRateAbs",
        ):
            node = getattr(camera, name, None)
            if node is None:
                continue
            try:
                return positive_real(_node_value(node, name), name)
            except (RuntimeError, TypeError, ValueError):
                continue
        return None

    def _read_optional_delay_seconds_on_owner(self) -> float | None:
        camera = self._require_camera()
        for name in ("BslExposureStartDelay", "ExposureStartDelay"):
            node = getattr(camera, name, None)
            if node is None:
                continue
            try:
                value = float(_node_value(node, name)) * 1e-6
            except (RuntimeError, TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= 0.0:
                return value
        return None

    def _read_working_point_on_owner(self) -> CameraWorkingPoint:
        camera = self._require_camera()
        pixel_format = str(_node_value(camera.PixelFormat, "PixelFormat"))
        if pixel_format != "Mono8":
            raise RuntimeError(f"pylon pixel format is {pixel_format!r}, expected 'Mono8'")
        trigger_mode = str(_node_value(camera.TriggerMode, "TriggerMode"))
        expected_mode = "Off" if self._armed and self._monitor_mode else "On"
        if trigger_mode != expected_mode:
            raise RuntimeError("pylon trigger mode changed outside the adapter")
        if expected_mode == "On":
            source = str(_node_value(camera.TriggerSource, "TriggerSource"))
            if source != self._config.trigger_source:
                raise RuntimeError("pylon trigger source changed outside the adapter")
        width = positive_integer(_node_value(camera.Width, "Width"), "pylon Width")
        height = positive_integer(_node_value(camera.Height, "Height"), "pylon Height")
        x = integer(_node_value(camera.OffsetX, "OffsetX"), "pylon OffsetX", nonnegative=True)
        y = integer(_node_value(camera.OffsetY, "OffsetY"), "pylon OffsetY", nonnegative=True)
        assert x is not None and y is not None
        width_max = getattr(camera, "WidthMax", None)
        height_max = getattr(camera, "HeightMax", None)
        sensor_width = (
            _integer_node(camera.Width, "Width", "GetMax") + x
            if width_max is None
            else positive_integer(_node_value(width_max, "WidthMax"), "pylon WidthMax")
        )
        sensor_height = (
            _integer_node(camera.Height, "Height", "GetMax") + y
            if height_max is None
            else positive_integer(_node_value(height_max, "HeightMax"), "pylon HeightMax")
        )
        exposure = positive_real(
            float(_node_value(camera.ExposureTime, "ExposureTime")) * 1e-6,
            "pylon exposure readback",
        )
        rate = self._read_optional_positive_rate_on_owner()
        interval = None if rate is None else 1.0 / rate
        integration_offset = self._read_optional_delay_seconds_on_owner()
        primitive = {
            "camera": "pylon",
            "serial": self._config.serial,
            "finite_trigger_source": self._config.trigger_source,
            "monitor_trigger_mode": "Off",
            "pixel_format": pixel_format,
            "frame_shape_yx": (height, width),
            "sensor_shape_yx": (sensor_height, sensor_width),
            "roi_origin_yx": (y, x),
            "exposure_seconds": exposure,
            "required_external_trigger_interval_seconds": interval,
            "external_trigger_integration_start_offset_seconds": integration_offset,
            "capture_trigger_channels": self._config.capture_trigger_channels,
        }
        return CameraWorkingPoint(
            settings_fingerprint=canonical_digest(primitive),
            acquisition_mode="EXTERNAL_TRIGGERED",
            frame_shape_yx=(height, width),
            sensor_shape_yx=(sensor_height, sensor_width),
            roi_origin_yx=(y, x),
            roi_shape_yx=(height, width),
            binning_yx=(1, 1),
            dtype=np.dtype("u1"),
            count_unit="count",
            capture_trigger_channels=self._config.capture_trigger_channels,
            exposure_seconds=exposure,
            required_external_trigger_interval_seconds=interval,
            external_trigger_integration_start_offset_seconds=integration_offset,
            gain=1.0,
            readout_mode=f"pylon:Mono8;external={self._config.trigger_source};monitor=free-run",
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
    ) -> int | None:
        if frames is None:
            if source_group_sizes is not None:
                raise ValueError("monitor arm cannot declare source_group_sizes")
            return None
        expected = positive_integer(frames, "frames")
        if not isinstance(source_group_sizes, tuple):
            raise TypeError("finite arm requires tuple source_group_sizes")
        groups = tuple(positive_integer(value, "source_group_sizes item") for value in source_group_sizes)
        if not groups or sum(groups) != expected:
            raise ValueError("source_group_sizes must exactly cover frames")
        return expected

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        expected = self._finite_groups(frames, source_group_sizes)
        buffer_count = positive_integer(buffer_frame_count, "buffer_frame_count")
        positive_real(timeout, "timeout")
        if expected is not None and buffer_count != expected:
            raise ValueError("finite buffer_frame_count must equal the complete frame count")

        def arm_on_owner() -> None:
            if self._armed:
                raise RuntimeError("pylon camera is already armed")
            camera = self._require_camera()
            if camera.IsGrabbing():
                camera.StopGrabbing()
            monitor = expected is None
            self._apply_trigger_on_owner(monitor=monitor)
            if monitor:
                camera.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)
            else:
                camera.StartGrabbingMax(expected, self._pylon.GrabStrategy_OneByOne)
            with self._state_lock:
                self._finish_requested.clear()
                self._armed = True
                self._monitor_mode = monitor
                self._expected_frames = expected
                self._delivered_count = 0
                self._terminal = None
            self._read_working_point_on_owner()

        self._lane.call(arm_on_owner)

    @staticmethod
    def _optional_result_integer(result: object, method: str) -> int | None:
        getter = getattr(result, method, None)
        if not callable(getter):
            return None
        try:
            value = int(getter())
        except (TypeError, ValueError, OverflowError):
            return None
        return value if value >= 0 else None

    def _read_on_owner(self, wanted: int, timeout: float, exact: bool) -> list[CameraFrameRecord]:
        if not self._armed:
            raise RuntimeError("pylon read requires an armed capture")
        camera = self._require_camera()
        deadline = time.monotonic() + timeout
        records: list[CameraFrameRecord] = []
        while len(records) < wanted:
            if self._finish_requested.is_set():
                raise PylonCaptureInterrupted("pylon capture is terminalizing")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            timeout_ms = max(1, int(min(_RETRIEVE_SLICE_SECONDS, remaining) * 1000.0))
            result = camera.RetrieveResult(timeout_ms, self._pylon.TimeoutHandling_Return)
            if result is None:
                continue
            try:
                if not result.GrabSucceeded():
                    raise RuntimeError("pylon returned an unsuccessful grab result")
                image = np.array(result.Array, copy=True)
                if image.dtype != np.dtype("u1"):
                    raise RuntimeError(
                        f"pylon Mono8 capture returned dtype {image.dtype}, expected uint8"
                    )
                with self._state_lock:
                    ordinal = self._delivered_count
                record = CameraFrameRecord(
                    image=image,
                    source_ordinal=ordinal,
                    produced_count=None if self._monitor_mode else ordinal + 1,
                    frame_stamp=self._optional_result_integer(result, "GetBlockID"),
                    camera_stamp=self._optional_result_integer(result, "GetImageNumber"),
                    timestamp_seconds=None,
                    timestamp_microseconds=None,
                    host_received_at_ns=time.time_ns(),
                    driver_buffer_index=None,
                )
            finally:
                result.Release()
            with self._state_lock:
                if self._delivered_count != ordinal:
                    raise RuntimeError("pylon delivered ordinal changed during frame copy")
                self._delivered_count += 1
            records.append(record)
        if len(records) != wanted and exact:
            raise TimeoutError(f"pylon returned {len(records)} of {wanted} exact frame(s)")
        return records

    def read_frame_records(self, n: int, *, timeout: float, exact: bool) -> list[CameraFrameRecord]:
        wanted = positive_integer(n, "n")
        bounded_timeout = float(timeout)
        if not np.isfinite(bounded_timeout) or bounded_timeout < 0.0:
            raise ValueError("timeout must be finite and non-negative")
        if type(exact) is not bool:
            raise TypeError("exact must be bool")
        return self._lane.call(lambda: self._read_on_owner(wanted, bounded_timeout, exact))

    def _finish_on_owner(self) -> CameraCaptureTerminalRecord:
        if self._terminal is not None:
            return self._terminal
        if not self._armed:
            self._terminal = CameraCaptureTerminalRecord(0, True, True, True)
            return self._terminal
        camera = self._require_camera()
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsGrabbing():
            raise RuntimeError("pylon remained grabbing after StopGrabbing")
        with self._state_lock:
            produced = self._delivered_count
        self._apply_trigger_on_owner(monitor=False)
        with self._state_lock:
            self._armed = False
            self._monitor_mode = False
            self._expected_frames = 0
        self._terminal = CameraCaptureTerminalRecord(produced, True, True, True)
        return self._terminal

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self._finish_requested.set()
        return self._lane.call(self._finish_on_owner)

    def capture_state(self) -> tuple[bool, int]:
        with self._state_lock:
            return self._armed, self._delivered_count

    def _close_on_owner(self) -> None:
        camera = self._camera
        if camera is not None:
            camera.Close()
        self._camera = None
        self._open = False

    def close(self) -> None:
        if not self._open and self._camera is None:
            self._lane.close()
            return
        if self.capture_state()[0]:
            self.finish_record_capture()
        self._lane.call(self._close_on_owner)
        self._lane.close()


__all__ = ["PylonCameraAdapter", "PylonCameraConfig", "PylonCaptureInterrupted"]
