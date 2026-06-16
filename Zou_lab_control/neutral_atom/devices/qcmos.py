"""Camera boundary for the lightweight neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import time
from typing import Any, Sequence

import numpy as np

from ..core.analysis import finite_float, positive_int
from ..timing import DEFAULT_CAMERA_TRIGGER_CHANNELS, exposure_from_sequence, imaging_channel_kwargs
from .base import AcquisitionCancelled, CameraDevice, snap_subarray
from .sequencer import PulseController, finite_frame_sequence


@dataclass
class QCMOSConfig:
    """Configuration for the thin real qCMOS adapter."""

    exposure: float = 20e-3
    readout_speed: int = 1
    roi: tuple[int, int, int, int] | None = None
    device_index: int = 0
    timeout_ms: int = 10_000
    # Shutter/exposure scheme.  Left None = the camera's power-on default
    # (ORCA-qCMOS is rolling-shutter AREA), which staggers row exposure relative
    # to the single FPGA trigger.  Synchronous atom imaging usually wants a
    # consciously chosen mode -- set these (raw DCAMPROP enum ints) in the camera
    # config to pin it; they are written + read back only when not None.
    sensor_mode: int | None = None
    trigger_global_exposure: int | None = None

    def __post_init__(self) -> None:
        self.exposure = positive_float(self.exposure, "exposure")
        self.readout_speed = nonnegative_int(self.readout_speed, "readout_speed")
        self.device_index = nonnegative_int(self.device_index, "device_index")
        self.timeout_ms = positive_int(self.timeout_ms, "timeout_ms")
        if self.roi is not None:
            self.roi = normalize_roi(self.roi)
        if self.sensor_mode is not None:
            self.sensor_mode = nonnegative_int(self.sensor_mode, "sensor_mode")
        if self.trigger_global_exposure is not None:
            self.trigger_global_exposure = nonnegative_int(self.trigger_global_exposure, "trigger_global_exposure")


DEFAULT_DCAM_MODULE = "Zou_lab_control.neutral_atom.devices.drivers.dcam.dcam"


class QCMOSCamera(CameraDevice):
    """Thin external-trigger Hamamatsu qCMOS adapter.

    The offline session uses ``VirtualCamera``. This class is intentionally
    small: it only owns DCAM open/configure/acquire/close. FPGA trigger timing
    still belongs to the sequencer.
    """

    def __init__(self, config: QCMOSConfig | dict[str, Any] | None = None, *, dcam_module: str = DEFAULT_DCAM_MODULE):
        self.config = config if isinstance(config, QCMOSConfig) else QCMOSConfig(**dict(config or {}))
        self.dcam_module_name = str(dcam_module)
        self._module = None
        self._api = None
        self._dcam = None
        # The sub-array the camera ACTUALLY applied (read back from DCAM after the
        # hardware snaps the request to its valid grid).  None until an open camera
        # has applied a ROI; the `roi` property reports this -- not the raw request
        # -- so a consumer (the Edit 'now:' reference, the 2D panel's axes) reflects
        # the region truly being imaged.
        self._applied_roi: tuple[int, int, int, int] | None = None

    @property
    def exposure(self) -> float:
        return self.config.exposure

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        # Report what the camera is ACTUALLY reading out (post hardware snap) when
        # open; the requested config ROI only when not yet applied.
        if self._applied_roi is not None:
            return self._applied_roi
        return self.config.roi

    def configure(self, *, exposure: float | None = None, readout_speed: int | None = None, roi: Sequence[int] | None | object = None) -> None:
        if exposure is not None:
            self.config.exposure = positive_float(exposure, "exposure")
        if readout_speed is not None:
            self.config.readout_speed = nonnegative_int(readout_speed, "readout_speed")
        if roi is not None:
            self.config.roi = normalize_roi(roi)
        if self._dcam is not None:
            self._write_settings()

    def open(self) -> "QCMOSCamera":
        if self._dcam is not None:
            return self
        mod = importlib.import_module(self.dcam_module_name)
        api = mod.Dcamapi
        api.init()
        dcam = mod.Dcam(self.config.device_index)
        if dcam.dev_open() is False:
            api.uninit()
            raise RuntimeError(f"failed to open qCMOS device {self.config.device_index}: {dcam.lasterr()}")
        self._module = mod
        self._api = api
        self._dcam = dcam
        self._write_settings()
        return self

    def acquire(self, frames: int = 1, *, sequence=None, sequencer=None, timeout_ms: int | None = None, stop=None) -> list[np.ndarray]:
        frames = positive_int(frames, "frames")
        self.open()
        if sequencer is None and isinstance(sequence, PulseController):
            sequencer = sequence.sequencer
        runtime_sequence = self._sequence_for_frames(sequence, frames=frames, sequencer=sequencer)
        if sequence is not None:
            # Read exposure off the SAME channel the imaging sequence put the probe
            # pulse on: imaging_channel_kwargs maps probe -> ch03 on a real chNN
            # streamer, so inferring from the placeholder "probe" name would miss
            # it and silently pin the camera at config exposure (a flat scan).
            probe_channel = imaging_channel_kwargs(sequencer).get("probe_channel", "probe")
            sequence_exposure = exposure_from_sequence(runtime_sequence, default=self.config.exposure, channel=probe_channel)
            if sequence_exposure != self.config.exposure:
                self.config.exposure = sequence_exposure
                self._write_settings()
        if sequencer is not None and sequence is not None:
            prepare = getattr(sequencer, "prepare", None)
            if callable(prepare):
                prepare(runtime_sequence)
        dcam = self._dcam
        if dcam.buf_alloc(frames) is False:
            raise RuntimeError(f"qCMOS buf_alloc({frames}) failed: {dcam.lasterr()}")
        images: list[np.ndarray] = []
        acquisition_error = False
        try:
            if dcam.cap_start(bSequence=True) is False:
                raise RuntimeError(f"qCMOS cap_start failed: {dcam.lasterr()}")
            if sequencer is not None and sequence is not None:
                fire = getattr(sequencer, "fire", None)
                if not callable(fire):
                    raise RuntimeError("sequencer must expose fire(sequence) for real qCMOS acquire.")
                fire(runtime_sequence)
            timeout = self.config.timeout_ms if timeout_ms is None else positive_int(timeout_ms, "timeout_ms")
            # When a live feed passes a stop event, wait in short slices and check
            # it between slices so Stop interrupts a wedged trigger within ~one
            # slice instead of blocking the full timeout.  Without a stop event,
            # one wait of the full timeout (the original behaviour).
            poll_ms = min(timeout, 200) if stop is not None else timeout
            next_frame = 0
            deadline = time.monotonic() + timeout / 1000.0
            while next_frame < frames:
                if stop is not None and stop.is_set():
                    raise AcquisitionCancelled(f"qCMOS acquire cancelled while waiting for frame {next_frame}.")
                slice_ms = poll_ms
                if stop is not None:
                    slice_ms = max(1, min(poll_ms, int((deadline - time.monotonic()) * 1000)))
                if dcam.wait_capevent_frameready(slice_ms) is False:
                    if stop is not None and not stop.is_set() and time.monotonic() < deadline:
                        continue  # slice expired but neither stopped nor timed out -- keep polling
                    if stop is not None and stop.is_set():
                        raise AcquisitionCancelled(f"qCMOS acquire cancelled while waiting for frame {next_frame}.")
                    raise TimeoutError(f"qCMOS timed out after {timeout} ms waiting for frame {next_frame}.")
                info = dcam.cap_transferinfo()
                available = int(getattr(info, "nFrameCount", next_frame + 1)) if info is not False else next_frame + 1
                while next_frame < min(available, frames):
                    data = dcam.buf_getframedata(next_frame)
                    if data is False:
                        raise RuntimeError(f"qCMOS buf_getframedata({next_frame}) failed: {dcam.lasterr()}")
                    images.append(np.asarray(data[1]).copy())
                    next_frame += 1
            if sequencer is not None and sequence is not None:
                wait_done = getattr(sequencer, "wait_done", None)
                if callable(wait_done):
                    wait_timeout = max(timeout / 1000.0, getattr(runtime_sequence, "duration", 0.0) * 2.0 + 1.0)
                    if not wait_done(wait_timeout):
                        raise TimeoutError("sequencer did not report done after qCMOS acquisition.")
            return images
        except Exception:
            acquisition_error = True
            raise
        finally:
            if acquisition_error and sequencer is not None:
                abort = getattr(sequencer, "abort", None)
                if callable(abort):
                    try:
                        abort()
                    except Exception:
                        pass
            try:
                dcam.cap_stop()
            finally:
                try:
                    dcam.buf_release()
                except Exception:
                    pass

    def _sequence_for_frames(self, sequence, *, frames: int, sequencer=None):
        if sequence is None:
            return None
        if isinstance(sequence, PulseController):
            return sequence.frame_sequence(frames)
        trigger_channels = getattr(sequencer, "trigger_channels", None)
        return finite_frame_sequence(
            sequence,
            frames,
            trigger_channels=trigger_channels if trigger_channels is not None else DEFAULT_CAMERA_TRIGGER_CHANNELS,
        )

    def _set_prop(self, idprop, value, name: str, *, verify: bool = False) -> None:
        """Write a DCAM property and FAIL LOUDLY if the camera rejects it.

        ``prop_setvalue`` returns False (and sets ``lasterr``) on a rejected
        write, so an unchecked write leaves the camera silently in its prior
        state -- e.g. a rejected external-trigger write keeps internal trigger,
        and the next acquire just times out with a confusing message.  ``verify``
        additionally reads the value back (critical trigger props) so a clamp /
        silent substitution also fails here, not at first light."""
        dcam = self._dcam
        if dcam.prop_setvalue(idprop, value) is False:
            raise RuntimeError(f"qCMOS rejected {name} = {value}: {dcam.lasterr()}")
        if verify:
            read = dcam.prop_getvalue(idprop)
            if read is False:
                raise RuntimeError(f"qCMOS could not read back {name}: {dcam.lasterr()}")
            # DCAM round-trips properties as doubles; compare numerically with a
            # tolerance, falling back to equality for non-numeric enum sentinels.
            try:
                mismatch = abs(float(read) - float(value)) > 1e-9
            except (TypeError, ValueError):
                mismatch = read != value
            if mismatch:
                raise RuntimeError(f"qCMOS {name} read back {read!r}, expected {value!r} "
                                   "(property clamped/substituted by the camera).")

    def _write_settings(self) -> None:
        mod = self._module
        self._set_prop(mod.DCAM_IDPROP.EXPOSURETIME, self.config.exposure, "exposure")
        # External rising-edge trigger is the imaging scheme -- verify the camera
        # actually accepted it (these are the writes that fail silently on a
        # mis-set / unsupported camera and then hang acquire).
        self._set_prop(mod.DCAM_IDPROP.TRIGGERSOURCE, mod.DCAMPROP.TRIGGERSOURCE.EXTERNAL, "trigger_source", verify=True)
        self._set_prop(mod.DCAM_IDPROP.TRIGGERACTIVE, mod.DCAMPROP.TRIGGERACTIVE.EDGE, "trigger_active", verify=True)
        self._set_prop(mod.DCAM_IDPROP.TRIGGERPOLARITY, mod.DCAMPROP.TRIGGERPOLARITY.POSITIVE, "trigger_polarity", verify=True)
        self._set_prop(mod.DCAM_IDPROP.READOUTSPEED, self.config.readout_speed, "readout_speed")
        if self.config.sensor_mode is not None:
            self._set_prop(mod.DCAM_IDPROP.SENSORMODE, self.config.sensor_mode, "sensor_mode", verify=True)
        if self.config.trigger_global_exposure is not None:
            self._set_prop(mod.DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE, self.config.trigger_global_exposure, "trigger_global_exposure", verify=True)
        self._apply_subarray()

    def _subarray_grid(self) -> tuple[int, int, int, int]:
        """The camera's valid sub-array grid (step, max_w, max_h) queried with
        SUBARRAYMODE OFF, where HSIZE/VSIZE max report the FULL sensor.  Falls back
        to a step of 4 (the qCMOS hardware requirement) if the attribute query
        is unavailable, so a request is always snapped to a legal window."""
        mod, dcam = self._module, self._dcam
        self._set_prop(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.OFF, "subarray_mode")

        def _attr(idprop, field, default):
            attr = dcam.prop_getattr(idprop)
            if attr is False:
                return default
            value = getattr(attr, field, 0.0)
            return int(value) if value else default

        step = _attr(mod.DCAM_IDPROP.SUBARRAYHSIZE, "valuestep", 4) or 4
        max_w = _attr(mod.DCAM_IDPROP.SUBARRAYHSIZE, "valuemax", 0)
        max_h = _attr(mod.DCAM_IDPROP.SUBARRAYVSIZE, "valuemax", 0)
        return step, max_w, max_h

    def _set_get_prop(self, idprop, value, name: str) -> int:
        """Write a sub-array property and return the value the camera ACTUALLY
        applied (DCAM ``prop_setgetvalue`` snaps to the hardware grid and reports
        it back), failing loudly on rejection -- so a clamped ROI is observed here
        rather than silently imaging the wrong region."""
        actual = self._dcam.prop_setgetvalue(idprop, value)
        if actual is False:
            raise RuntimeError(f"qCMOS rejected {name} = {value}: {self._dcam.lasterr()}")
        return int(round(float(actual)))

    def _apply_subarray(self) -> None:
        """Apply the requested ROI as a valid sub-array and record what the camera
        actually reads out.  The plot/acquisition layer hands a window in raw
        sensor-pixel coordinates; here it is SNAPPED to the camera's grid
        (``snap_subarray``) and written in a SAFE ORDER (positions to 0 first, then
        sizes, then positions) so an intermediate ``pos+size`` never exceeds the
        sensor, with SUBARRAYMODE ON last.  ``prop_setgetvalue`` gives the camera
        the final say; the read-back tuple becomes ``self._applied_roi``."""
        mod = self._module
        if self.config.roi is None:
            self._set_prop(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.OFF, "subarray_mode")
            self._applied_roi = None
            return
        step, max_w, max_h = self._subarray_grid()
        x, width, y, height = snap_subarray(
            self.config.roi, step=step,
            max_w=max_w or self.config.roi[0] + self.config.roi[1],
            max_h=max_h or self.config.roi[2] + self.config.roi[3],
        )
        self._set_prop(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.ON, "subarray_mode")
        self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHPOS, 0, "subarray_hpos")
        self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVPOS, 0, "subarray_vpos")
        applied_w = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHSIZE, width, "subarray_hsize")
        applied_h = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVSIZE, height, "subarray_vsize")
        applied_x = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHPOS, x, "subarray_hpos")
        applied_y = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVPOS, y, "subarray_vpos")
        self._applied_roi = (applied_x, applied_w, applied_y, applied_h)

    def close(self) -> None:
        if self._dcam is not None:
            try:
                self._dcam.dev_close()
            finally:
                self._dcam = None
        if self._api is not None:
            try:
                self._api.uninit()
            finally:
                self._api = None

    def stop(self) -> None:
        if self._dcam is not None:
            try:
                self._dcam.cap_stop()
            except Exception:
                pass

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.config.exposure,
            "readout_speed": self.config.readout_speed,
            "roi": self.config.roi,
            "device_index": self.config.device_index,
            "timeout_ms": self.config.timeout_ms,
            "open": self._dcam is not None,
        }


def normalize_roi(roi: Sequence[int]) -> tuple[int, int, int, int]:
    try:
        raw = tuple(roi)
    except TypeError as exc:
        raise ValueError("roi must be (x, width, y, height).") from exc
    if len(raw) != 4:
        raise ValueError("roi must be (x, width, y, height).")
    out = tuple(nonnegative_int(v, f"roi[{i}]") for i, v in enumerate(raw))
    if out[1] <= 0 or out[3] <= 0:
        raise ValueError("roi width and height must be positive.")
    return out


def nonnegative_int(value, name: str) -> int:
    out = finite_float(value, name)
    if int(out) != out or out < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(out)


def positive_float(value, name: str) -> float:
    out = finite_float(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be > 0.")
    return out


__all__ = ["DEFAULT_DCAM_MODULE", "QCMOSCamera", "QCMOSConfig", "normalize_roi"]
