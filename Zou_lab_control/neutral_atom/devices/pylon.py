"""Basler pylon MONITOR camera -- the real MOT-viewer sensor.

A thin :class:`~.base.CameraDevice` wrapper over ``pypylon`` with the SAME pure-grabber contract
the qCMOS keeps (H4f): ``acquire(N, sequence=..., on_armed=...)`` arms the camera, invokes the
fire hook (the measurement fires the FPGA -- arm-before-fire), then collects ``N`` externally- or
software-triggered frames.  ``sequence`` is accepted and ignored (a real sensor images the real
world; only the virtual :class:`~.virtual.VirtualMotCamera` consumes it to simulate the MOT), so
the virtual and real monitor cameras are drop-in swaps: switching changes only the device config.

``pypylon`` is imported lazily in :meth:`open` -- a machine without the Basler runtime can still
import the package, run the virtual backend and the full test suite.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import CameraDevice, snap_subarray


class PylonCamera(CameraDevice):
    """Externally-triggerable Basler (pylon) camera as a pure frame grabber.

    Parameters mirror the virtual monitor camera where they overlap (``exposure`` in seconds);
    ``serial`` selects a specific camera when several are attached (empty = first found --
    :func:`~.discovery.discover_devices` prints every attached serial);
    ``trigger_source`` is the Basler trigger line name (e.g. ``"Line1"``) or ``"Software"``
    (free-run: no trigger wiring needed, every ``acquire``/``capture`` just grabs frames --
    the out-of-the-box mode for an acA1920-155um on USB3);
    ``pixel_format`` is the Basler pixel format name (``"Mono8"`` / ``"Mono12"`` on the
    acA1920-155um; ``Mono12`` for photon-counting-ish dynamic range, ``Mono8`` for speed)."""

    def __init__(self, *, exposure: float = 5e-3, serial: str = "",
                 trigger_source: str = "Line1", pixel_format: str = "Mono8",
                 timeout: float = 2.0, subarray_step: int = 2):
        self._exposure = float(exposure)
        self.serial = str(serial)
        self.trigger_source = str(trigger_source)
        self.pixel_format = str(pixel_format)
        self.timeout = float(timeout)
        self.subarray_step = int(subarray_step)
        self._camera = None                     # the pypylon InstantCamera, created in open()
        self._roi: tuple[int, int, int, int] | None = None

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> None:
        from pypylon import pylon               # lazy: only a machine with the Basler runtime needs it

        factory = pylon.TlFactory.GetInstance()
        if self.serial:
            info = next((d for d in factory.EnumerateDevices()
                         if d.GetSerialNumber() == self.serial), None)
            if info is None:
                raise RuntimeError(f"pylon camera with serial {self.serial!r} not found.")
            self._camera = pylon.InstantCamera(factory.CreateDevice(info))
        else:
            self._camera = pylon.InstantCamera(factory.CreateFirstDevice())
        self._camera.Open()
        self._camera.PixelFormat.SetValue(self.pixel_format)
        self._apply_exposure()
        self._apply_trigger()

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.Close()
            finally:
                self._camera = None

    def stop(self) -> None:
        if self._camera is not None and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    # ------------------------------------------------------------------ configuration
    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = float(value)
        if self._camera is not None:
            self._apply_exposure()

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    @property
    def sensor_shape(self) -> tuple[int, int]:
        if self._camera is None:
            return (0, 0)
        return (int(self._camera.HeightMax.GetValue()), int(self._camera.WidthMax.GetValue()))

    def configure(self, *, exposure: float | None = None, roi: object = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure", "roi"}, kwargs)
        if exposure is not None:
            self.exposure = float(exposure)
        if roi is not None:
            if roi in ("", "None"):
                self._roi = None
            else:
                h, w = self.sensor_shape
                self._roi = snap_subarray(tuple(roi), step=self.subarray_step,
                                          max_w=w or 10**6, max_h=h or 10**6)
            self._apply_roi()

    def _apply_exposure(self) -> None:
        # Basler exposes ExposureTime in microseconds
        self._camera.ExposureTime.SetValue(float(self._exposure) * 1e6)

    def _apply_trigger(self) -> None:
        cam = self._camera
        cam.TriggerSelector.SetValue("FrameStart")
        if self.trigger_source.lower() == "software":
            cam.TriggerMode.SetValue("Off")     # free-run: each RetrieveResult grabs the next frame
        else:
            cam.TriggerMode.SetValue("On")
            cam.TriggerSource.SetValue(self.trigger_source)

    def _apply_roi(self) -> None:
        if self._camera is None or self._roi is None:
            return
        x, w, y, h = self._roi
        cam = self._camera
        cam.Width.SetValue(int(w))
        cam.Height.SetValue(int(h))
        cam.OffsetX.SetValue(int(x))
        cam.OffsetY.SetValue(int(y))

    # ------------------------------------------------------------------ acquisition
    def acquire(self, frames: int = 1, *, sequence=None, on_armed=None,
                stop=None, **_) -> list[np.ndarray]:
        """Arm, invoke the fire hook, then grab ``frames`` frames (the H4f grabber contract).
        ``sequence`` is ignored -- the real MOT answers through the real coils."""
        del sequence
        if self._camera is None:
            raise RuntimeError("PylonCamera.acquire before open() -- the device set opens cameras last.")
        from pypylon import pylon

        cam = self._camera
        cam.StartGrabbingMax(int(frames), pylon.GrabStrategy_OneByOne)
        if on_armed is not None:
            on_armed()                          # the measurement fires the FPGA now (arm-before-fire)
        out: list[np.ndarray] = []
        timeout_ms = int(self.timeout * 1000)
        while cam.IsGrabbing():
            if stop is not None and stop.is_set():
                cam.StopGrabbing()
                break
            result = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
            if result is None or not result.GrabSucceeded():
                if result is not None:
                    result.Release()
                break
            out.append(np.array(result.Array, copy=True))
            result.Release()
        return self._retain(out)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self._roi,
            "serial": self.serial,
            "trigger_source": self.trigger_source,
            "pixel_format": self.pixel_format,
        }
