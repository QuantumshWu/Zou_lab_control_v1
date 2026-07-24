"""Minimal target-owned DCAM-API v4 driver used by the qCMOS adapter.

The module defines the subset of the Hamamatsu ABI that the current adapter
actually consumes.  Importing it does not load ``dcamapi.dll``; the library is
opened only when :class:`DcamSdkDriver` is constructed by a hardware
composition root.
"""

from __future__ import annotations

import ctypes
import os
import platform
from enum import IntEnum

import numpy as np


_DCAMERR_TIMEOUT = -2147483386
_DCAMERR_ALREADY_INITIALIZED = -520093695
_DCAMCAP_START_SEQUENCE = -1
_DCAMWAIT_CAPEVENT_FRAMEREADY = 0x0002


class DcamProperty(IntEnum):
    TRIGGER_SOURCE = 1_048_848
    TRIGGER_ACTIVE = 1_048_864
    TRIGGER_POLARITY = 1_049_120
    EXPOSURE_TIME = 2_031_888
    TRIGGER_GLOBAL_EXPOSURE = 2_032_384
    READOUT_SPEED = 4_194_576
    SENSOR_MODE = 4_194_832
    BINNING = 4_198_672
    SUBARRAY_HPOS = 4_202_768
    SUBARRAY_HSIZE = 4_202_784
    SUBARRAY_VPOS = 4_202_800
    SUBARRAY_VSIZE = 4_202_816
    SUBARRAY_MODE = 4_202_832
    TIMING_MIN_TRIGGER_INTERVAL = 4_206_672
    TIMING_GLOBAL_EXPOSURE_DELAY = 4_206_736
    IMAGE_PIXEL_TYPE = 4_326_000
    IMAGE_WIDTH = 4_325_904
    IMAGE_HEIGHT = 4_325_920
    IMAGE_ROW_BYTES = 4_325_936


class DcamValue(IntEnum):
    MODE_OFF = 1
    MODE_ON = 2
    TRIGGER_SOURCE_EXTERNAL = 2
    TRIGGER_ACTIVE_EDGE = 1
    TRIGGER_POLARITY_POSITIVE = 2
    PIXEL_MONO8 = 1
    PIXEL_MONO16 = 2


class DcamDriverError(RuntimeError):
    """One failed DCAM call with its signed SDK error code."""

    def __init__(self, operation: str, code: int) -> None:
        self.operation = str(operation)
        self.code = int(code)
        super().__init__(f"{self.operation} failed with DCAM error {self.code}")


class _DcamApiInit(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("device_count", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("init_option_bytes", ctypes.c_int32),
        ("init_option", ctypes.POINTER(ctypes.c_int32)),
        ("guid", ctypes.c_void_p),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))


class _DcamDeviceOpen(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("index", ctypes.c_int32),
        ("handle", ctypes.c_void_p),
    ]

    def __init__(self, index: int) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.index = int(index)


class _DcamPropertyAttributes(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("property_id", ctypes.c_int32),
        ("option", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("attribute", ctypes.c_int32),
        ("group", ctypes.c_int32),
        ("unit", ctypes.c_int32),
        ("attribute2", ctypes.c_int32),
        ("value_min", ctypes.c_double),
        ("value_max", ctypes.c_double),
        ("value_step", ctypes.c_double),
        ("value_default", ctypes.c_double),
        ("max_channel", ctypes.c_int32),
        ("reserved3", ctypes.c_int32),
        ("max_view", ctypes.c_int32),
        ("number_of_elements", ctypes.c_int32),
        ("array_base_property_id", ctypes.c_int32),
        ("element_property_step", ctypes.c_int32),
    ]

    def __init__(self, property_id: int) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.property_id = int(property_id)


class _DcamTimestamp(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("seconds", ctypes.c_uint32),
        ("microseconds", ctypes.c_int32),
    ]


class _DcamTransferInfo(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("kind", ctypes.c_int32),
        ("newest_frame_index", ctypes.c_int32),
        ("frame_count", ctypes.c_int32),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.newest_frame_index = -1


class _DcamBufferFrame(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("kind", ctypes.c_int32),
        ("option", ctypes.c_int32),
        ("frame_index", ctypes.c_int32),
        ("buffer", ctypes.c_void_p),
        ("row_bytes", ctypes.c_int32),
        ("pixel_type", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("timestamp", _DcamTimestamp),
        ("frame_stamp", ctypes.c_int32),
        ("camera_stamp", ctypes.c_int32),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))


class _DcamWaitOpen(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("supported_events", ctypes.c_int32),
        ("wait_handle", ctypes.c_void_p),
        ("camera_handle", ctypes.c_void_p),
    ]

    def __init__(self, camera_handle: ctypes.c_void_p) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.camera_handle = camera_handle


class _DcamWaitStart(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("event_happened", ctypes.c_int32),
        ("event_mask", ctypes.c_int32),
        ("timeout_milliseconds", ctypes.c_int32),
    ]

    def __init__(self, timeout_milliseconds: int) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.event_mask = _DCAMWAIT_CAPEVENT_FRAMEREADY
        self.timeout_milliseconds = int(timeout_milliseconds)


def _checked(operation: str, code: object) -> int:
    value = int(code)
    if value < 0:
        raise DcamDriverError(operation, value)
    return value


class DcamSdkDriver:
    """Thin ctypes owner for the production DCAM library."""

    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        if platform.system() != "Windows":
            raise RuntimeError("the deployed qCMOS DCAM driver requires Windows")
        requested = "dcamapi.dll" if library_path is None else os.fspath(library_path)
        try:
            self._dll = ctypes.WinDLL(requested)
        except OSError as exc:
            raise RuntimeError(f"could not load DCAM library {requested!r}") from exc
        self._bind()

    def _bind(self) -> None:
        dll = self._dll
        dll.dcamapi_init.argtypes = [ctypes.POINTER(_DcamApiInit)]
        dll.dcamapi_init.restype = ctypes.c_int32
        dll.dcamapi_uninit.argtypes = []
        dll.dcamapi_uninit.restype = ctypes.c_int32
        dll.dcamdev_open.argtypes = [ctypes.POINTER(_DcamDeviceOpen)]
        dll.dcamdev_open.restype = ctypes.c_int32
        dll.dcamdev_close.argtypes = [ctypes.c_void_p]
        dll.dcamdev_close.restype = ctypes.c_int32
        dll.dcamprop_getattr.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_DcamPropertyAttributes),
        ]
        dll.dcamprop_getattr.restype = ctypes.c_int32
        dll.dcamprop_getvalue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.dcamprop_getvalue.restype = ctypes.c_int32
        dll.dcamprop_setgetvalue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
        ]
        dll.dcamprop_setgetvalue.restype = ctypes.c_int32
        dll.dcambuf_alloc.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        dll.dcambuf_alloc.restype = ctypes.c_int32
        dll.dcambuf_release.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        dll.dcambuf_release.restype = ctypes.c_int32
        dll.dcambuf_copyframe.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_DcamBufferFrame),
        ]
        dll.dcambuf_copyframe.restype = ctypes.c_int32
        dll.dcamcap_start.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        dll.dcamcap_start.restype = ctypes.c_int32
        dll.dcamcap_stop.argtypes = [ctypes.c_void_p]
        dll.dcamcap_stop.restype = ctypes.c_int32
        dll.dcamcap_status.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.dcamcap_status.restype = ctypes.c_int32
        dll.dcamcap_transferinfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_DcamTransferInfo),
        ]
        dll.dcamcap_transferinfo.restype = ctypes.c_int32
        dll.dcamwait_open.argtypes = [ctypes.POINTER(_DcamWaitOpen)]
        dll.dcamwait_open.restype = ctypes.c_int32
        dll.dcamwait_close.argtypes = [ctypes.c_void_p]
        dll.dcamwait_close.restype = ctypes.c_int32
        dll.dcamwait_start.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_DcamWaitStart),
        ]
        dll.dcamwait_start.restype = ctypes.c_int32

    def initialize(self) -> bool:
        """Initialize the process runtime; return whether this call owns it."""

        init = _DcamApiInit()
        code = int(self._dll.dcamapi_init(ctypes.byref(init)))
        if code == _DCAMERR_ALREADY_INITIALIZED:
            return False
        _checked("dcamapi_init", code)
        return True

    def uninitialize(self) -> None:
        _checked("dcamapi_uninit", self._dll.dcamapi_uninit())

    def open_device(self, index: int) -> "DcamSdkDevice":
        opened = _DcamDeviceOpen(index)
        _checked("dcamdev_open", self._dll.dcamdev_open(ctypes.byref(opened)))
        if not opened.handle:
            raise RuntimeError("dcamdev_open returned an empty camera handle")
        return DcamSdkDevice(self._dll, opened.handle)


class DcamSdkDevice:
    """One open DCAM handle.  The caller owns all thread serialization."""

    def __init__(self, dll: object, handle: ctypes.c_void_p) -> None:
        self._dll = dll
        self._handle = handle
        self._wait_handle: ctypes.c_void_p | None = None
        self._closed = False
        self._frame_shape_yx: tuple[int, int] | None = None
        self._frame_dtype: np.dtype | None = None
        self._row_bytes: int | None = None
        self._pixel_type: int | None = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DCAM device is closed")

    def get_property(self, property_id: int) -> float:
        self._require_open()
        value = ctypes.c_double()
        _checked(
            f"dcamprop_getvalue({int(property_id)})",
            self._dll.dcamprop_getvalue(
                self._handle,
                int(property_id),
                ctypes.byref(value),
            ),
        )
        return float(value.value)

    def set_get_property(self, property_id: int, value: float) -> float:
        self._require_open()
        applied = ctypes.c_double(float(value))
        _checked(
            f"dcamprop_setgetvalue({int(property_id)})",
            self._dll.dcamprop_setgetvalue(
                self._handle,
                int(property_id),
                ctypes.byref(applied),
                0,
            ),
        )
        return float(applied.value)

    def property_attributes(self, property_id: int) -> tuple[float, float]:
        self._require_open()
        attributes = _DcamPropertyAttributes(property_id)
        _checked(
            f"dcamprop_getattr({int(property_id)})",
            self._dll.dcamprop_getattr(self._handle, ctypes.byref(attributes)),
        )
        return float(attributes.value_step), float(attributes.value_max)

    def allocate_buffer(self, frame_count: int) -> None:
        self._require_open()
        # Establish all fallible geometry facts before asking DCAM to retain a
        # ring.  Once dcambuf_alloc succeeds, the remaining assignments cannot
        # strand an untracked driver allocation.
        pixel_type = int(round(self.get_property(DcamProperty.IMAGE_PIXEL_TYPE)))
        width = int(round(self.get_property(DcamProperty.IMAGE_WIDTH)))
        height = int(round(self.get_property(DcamProperty.IMAGE_HEIGHT)))
        row_bytes = int(round(self.get_property(DcamProperty.IMAGE_ROW_BYTES)))
        if pixel_type == int(DcamValue.PIXEL_MONO16):
            dtype = np.dtype("<u2")
        elif pixel_type == int(DcamValue.PIXEL_MONO8):
            dtype = np.dtype("u1")
        else:
            raise RuntimeError(f"unsupported DCAM pixel type {pixel_type}")
        if width <= 0 or height <= 0 or row_bytes < width * dtype.itemsize:
            raise RuntimeError("DCAM returned invalid frame geometry")
        _checked(
            "dcambuf_alloc",
            self._dll.dcambuf_alloc(self._handle, int(frame_count)),
        )
        self._frame_shape_yx = (height, width)
        self._frame_dtype = dtype
        self._row_bytes = row_bytes
        self._pixel_type = pixel_type

    def release_buffer(self) -> None:
        self._require_open()
        _checked("dcambuf_release", self._dll.dcambuf_release(self._handle, 0))
        self._frame_shape_yx = None
        self._frame_dtype = None
        self._row_bytes = None
        self._pixel_type = None

    def start_capture(self) -> None:
        self._require_open()
        _checked(
            "dcamcap_start",
            self._dll.dcamcap_start(self._handle, _DCAMCAP_START_SEQUENCE),
        )

    def stop_capture(self) -> None:
        self._require_open()
        _checked("dcamcap_stop", self._dll.dcamcap_stop(self._handle))

    def capture_status(self) -> int:
        self._require_open()
        status = ctypes.c_int32()
        _checked(
            "dcamcap_status",
            self._dll.dcamcap_status(self._handle, ctypes.byref(status)),
        )
        return int(status.value)

    def transfer_info(self) -> tuple[int, int]:
        self._require_open()
        info = _DcamTransferInfo()
        _checked(
            "dcamcap_transferinfo",
            self._dll.dcamcap_transferinfo(self._handle, ctypes.byref(info)),
        )
        return int(info.frame_count), int(info.newest_frame_index)

    def copy_frame(
        self,
        ring_index: int,
    ) -> tuple[np.ndarray, int, int, int, int]:
        self._require_open()
        if (
            self._frame_shape_yx is None
            or self._frame_dtype is None
            or self._row_bytes is None
            or self._pixel_type is None
        ):
            raise RuntimeError("DCAM frame buffer has not been allocated")
        height, width = self._frame_shape_yx
        # DCAM reports the physical destination row stride.  Do not assume it
        # equals width * itemsize: a padded row written into a compact ndarray
        # would overrun the allocation.  CameraFrameRecord makes the eventual
        # immutable compact copy from this possibly-strided view.
        storage = np.empty((height, self._row_bytes), dtype=np.uint8, order="C")
        image = storage[:, : width * self._frame_dtype.itemsize].view(
            self._frame_dtype
        )
        image = image.reshape(height, width)
        frame = _DcamBufferFrame()
        frame.frame_index = int(ring_index)
        frame.buffer = storage.ctypes.data_as(ctypes.c_void_p)
        frame.row_bytes = self._row_bytes
        frame.pixel_type = self._pixel_type
        frame.width = width
        frame.height = height
        _checked(
            f"dcambuf_copyframe({int(ring_index)})",
            self._dll.dcambuf_copyframe(self._handle, ctypes.byref(frame)),
        )
        return (
            image,
            int(frame.frame_stamp),
            int(frame.camera_stamp),
            int(frame.timestamp.seconds),
            int(frame.timestamp.microseconds),
        )

    def wait_frame_ready(self, timeout_milliseconds: int) -> bool:
        self._require_open()
        if self._wait_handle is None:
            opened = _DcamWaitOpen(self._handle)
            _checked("dcamwait_open", self._dll.dcamwait_open(ctypes.byref(opened)))
            if not opened.wait_handle:
                raise RuntimeError("dcamwait_open returned an empty wait handle")
            self._wait_handle = opened.wait_handle
        wait = _DcamWaitStart(timeout_milliseconds)
        code = int(self._dll.dcamwait_start(self._wait_handle, ctypes.byref(wait)))
        if code == _DCAMERR_TIMEOUT:
            return False
        _checked("dcamwait_start", code)
        return bool(wait.event_happened & _DCAMWAIT_CAPEVENT_FRAMEREADY)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        if self._wait_handle is not None:
            try:
                _checked(
                    "dcamwait_close",
                    self._dll.dcamwait_close(self._wait_handle),
                )
            except BaseException as error:
                errors.append(error)
            self._wait_handle = None
        device_closed = False
        try:
            _checked("dcamdev_close", self._dll.dcamdev_close(self._handle))
        except BaseException as error:
            errors.append(error)
        else:
            device_closed = True
        self._closed = device_closed
        if errors:
            primary = errors[0]
            for secondary in errors[1:]:
                primary.add_note(f"additional DCAM close failure: {secondary}")
            raise primary


__all__ = [
    "DcamDriverError",
    "DcamProperty",
    "DcamSdkDevice",
    "DcamSdkDriver",
    "DcamValue",
]
