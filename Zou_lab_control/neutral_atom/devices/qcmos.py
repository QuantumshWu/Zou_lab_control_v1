"""Camera boundary for the lightweight neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Sequence

import numpy as np

from ..core.analysis import finite_float, positive_int
from ..timing import exposure_from_sequence, sequence_for_frame_count
from .base import CameraDevice


@dataclass
class QCMOSConfig:
    """Configuration for the thin real qCMOS adapter."""

    exposure: float = 20e-3
    readout_speed: int = 1
    roi: tuple[int, int, int, int] | None = None
    device_index: int = 0
    timeout_ms: int = 10_000

    def __post_init__(self) -> None:
        self.exposure = positive_float(self.exposure, "exposure")
        self.readout_speed = nonnegative_int(self.readout_speed, "readout_speed")
        self.device_index = nonnegative_int(self.device_index, "device_index")
        self.timeout_ms = positive_int(self.timeout_ms, "timeout_ms")
        if self.roi is not None:
            self.roi = normalize_roi(self.roi)


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

    @property
    def exposure(self) -> float:
        return self.config.exposure

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

    def acquire(self, frames: int = 1, *, sequence=None, sequencer=None, timeout_ms: int | None = None) -> list[np.ndarray]:
        frames = positive_int(frames, "frames")
        self.open()
        runtime_sequence = sequence
        if sequence is not None:
            sequence_exposure = exposure_from_sequence(sequence, default=self.config.exposure)
            if sequence_exposure != self.config.exposure:
                self.config.exposure = sequence_exposure
                self._write_settings()
        if sequencer is not None and sequence is not None:
            trigger_channels = getattr(sequencer, "trigger_channels", None)
            runtime_sequence = (
                sequence_for_frame_count(sequence, frames, trigger_channels=trigger_channels)
                if trigger_channels is not None
                else sequence_for_frame_count(sequence, frames)
            )
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
            next_frame = 0
            while next_frame < frames:
                if dcam.wait_capevent_frameready(timeout) is False:
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

    def _write_settings(self) -> None:
        mod = self._module
        dcam = self._dcam
        dcam.prop_setvalue(mod.DCAM_IDPROP.EXPOSURETIME, self.config.exposure)
        dcam.prop_setvalue(mod.DCAM_IDPROP.TRIGGERSOURCE, mod.DCAMPROP.TRIGGERSOURCE.EXTERNAL)
        dcam.prop_setvalue(mod.DCAM_IDPROP.TRIGGERACTIVE, mod.DCAMPROP.TRIGGERACTIVE.EDGE)
        dcam.prop_setvalue(mod.DCAM_IDPROP.TRIGGERPOLARITY, mod.DCAMPROP.TRIGGERPOLARITY.POSITIVE)
        dcam.prop_setvalue(mod.DCAM_IDPROP.READOUTSPEED, self.config.readout_speed)
        if self.config.roi is not None:
            x, width, y, height = self.config.roi
            dcam.prop_setvalue(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.ON)
            dcam.prop_setvalue(mod.DCAM_IDPROP.SUBARRAYHSIZE, width)
            dcam.prop_setvalue(mod.DCAM_IDPROP.SUBARRAYHPOS, x)
            dcam.prop_setvalue(mod.DCAM_IDPROP.SUBARRAYVSIZE, height)
            dcam.prop_setvalue(mod.DCAM_IDPROP.SUBARRAYVPOS, y)

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
