"""Basler pylon MONITOR camera -- the real MOT-viewer sensor.

A thin :class:`~.base.CameraDevice` wrapper over ``pypylon`` with the SAME pure-grabber contract
every camera keeps: :meth:`~.base.CameraDevice.arm` readies the sensor, the frames its trigger
gates are consumed with :meth:`~.base.CameraDevice.read_frames`, and
:meth:`~.base.CameraDevice.disarm` stands it down (a measurement fires the FPGA between arm and
read through ``operations.measurement.triggered_frames``; a free-running ``Software`` sensor just
calls ``acquire``).  The camera never touches a sequencer, so the virtual
:class:`~.virtual.VirtualMotCamera` and this real monitor camera are drop-in swaps: switching
changes only the device config.

In ``Software`` (free-run) mode the grab stream is RESIDENT (like the qCMOS ``cap_start`` once /
poll thereafter): arming starts ``StartGrabbing`` on first use and disarm leaves it running --
restarting the USB3 stream per frame costs tens of milliseconds and was the live-monitor stutter.
Anything that must reconfigure the sensor (ROI, pixel format) pauses the stream and resumes it
afterwards.  A HARDWARE-triggered session (e.g. ``Line1``) instead spans exactly one arm..disarm:
one grab session per armed request, stopped when done, STRICTLY (see :meth:`_arm`).

``pypylon`` is imported lazily in :meth:`open` -- a machine without the Basler runtime can still
import the package, run the virtual backend and the full test suite.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .base import ROI_CLEAR_SENTINELS, SOFTWARE_TRIGGER, CameraDevice, snap_subarray
from .camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS


def _snap_to_inc(value: int, lo: int, hi: int, inc: int) -> int:
    """Clamp ``value`` into the camera's own [lo, hi] and align it to the GenICam increment --
    the hardware, not a hardcoded step, owns the legal grid."""
    inc = max(1, int(inc))
    v = max(int(lo), min(int(hi), int(value)))
    return int(lo) + ((v - int(lo)) // inc) * inc


class PylonCamera(CameraDevice):
    """Externally-triggerable Basler (pylon) camera as a pure frame grabber.

    Parameters mirror the virtual monitor camera where they overlap (``exposure`` in seconds);
    ``serial`` selects a specific camera when several are attached (empty = first found --
    :func:`~.discovery.discover_devices` prints every attached serial);
    ``trigger_source`` is the Basler trigger line name (e.g. ``"Line1"``) or ``"Software"``
    (free-run: no trigger wiring needed, every ``acquire``/``capture`` just grabs frames --
    the out-of-the-box mode for an acA1920-155um on USB3);
    ``pixel_format`` is the Basler pixel format name (``"Mono8"`` / ``"Mono12"`` on the
    acA1920-155um; ``Mono12`` for photon-counting-ish dynamic range, ``Mono8`` for speed);
    ``capture_trigger_channels`` is which SEQUENCER line the camera's hardware trigger input is
    physically wired to -- the passive wiring fact the measurement layer counts trigger edges on
    (``triggered_frames``/``_program_for_frames``).  In ``Software`` free-run the declaration is
    inert (no edge gates a frame), so the conservative base default is fine; in a HARDWARE-triggered
    session it MUST name the real cable's sequencer channel (e.g. ``("ch05",)``), or the edge count
    reads the wrong line and a multi-frame acquisition mis-repeats its program.  Note the two
    namespaces: ``trigger_source`` is the Basler-side GenICam line name (``"Line1"``),
    ``capture_trigger_channels`` the FPGA-side channel driving that line."""

    def __init__(self, *, exposure: float = 5e-3, serial: str = "",
                 trigger_source: str = "Line1", pixel_format: str = "Mono8",
                 timeout: float = 2.0, subarray_step: int = 2,
                 capture_trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS):
        self._exposure = float(exposure)
        self.serial = str(serial)
        self.trigger_source = str(trigger_source)
        self.capture_trigger_channels = tuple(str(c) for c in capture_trigger_channels)
        self.pixel_format = str(pixel_format)
        self.timeout = float(timeout)
        # Fallback ROI grid when the camera is not open yet; once open, the camera's OWN GenICam
        # increments (Width.Inc / OffsetX.Inc) own the legal grid -- never a hardcoded step.
        self.subarray_step = int(subarray_step)
        self._camera = None                     # the pypylon InstantCamera, created in open()
        self._roi: tuple[int, int, int, int] | None = None
        # Armed-session bookkeeping so a triggered-mode fault names WHICH frame of HOW MANY was
        # awaited (the qCMOS message style); reset per arm, meaningless in free-run.
        self._armed_total: int | None = None
        self._grabbed = 0

    # ------------------------------------------------------------------ discovery (self-describing)
    @classmethod
    def discover(cls):
        """Enumerate attached Basler cameras -- each hit carries a READY config entry for THIS
        class (serial-pinned, free-running so first light needs no trigger wiring).  The device
        class owns its own discovery: :func:`~.discovery.discover_devices` only aggregates."""
        from .discovery import DiscoveredDevice, discovery_note

        try:
            from pypylon import pylon
        except ImportError:
            return [discovery_note("basler", "pypylon not installed -- pip install pypylon")]
        try:
            infos = pylon.TlFactory.GetInstance().EnumerateDevices()
        except Exception as exc:                # runtime present but transport layer unhappy
            return [discovery_note("basler", f"enumerate failed: {exc}")]
        if not infos:
            return [discovery_note("basler", "no Basler camera attached")]
        return cls.discovery_rows(infos)

    @classmethod
    def discovery_rows(cls, infos):
        """Pure enumeration->rows mapping (tests feed fakes): each attached camera becomes a row
        whose config is a ready entry for THIS class -- serial-pinned, free-running."""
        from .discovery import DiscoveredDevice

        return [DiscoveredDevice(
            kind="basler", ident=str(i.GetSerialNumber()), label=str(i.GetModelName()),
            config={"type": cls.__name__,
                    "params": {"serial": str(i.GetSerialNumber()),
                               "trigger_source": SOFTWARE_TRIGGER,
                               # Inert in Software free-run; switching trigger_source to a hardware
                               # line means setting this to the sequencer channel wired to it.
                               "capture_trigger_channels": list(DEFAULT_CAMERA_TRIGGER_CHANNELS)}})
            for i in infos]

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
        # A single "push every configured setting" path -- so no setting made BEFORE open (a
        # configure(roi=) before the device set opens cameras, or a close->reopen) can be dropped.
        # ROI is applied LAST: it snaps against the now-live GenICam Width/Offset increments.
        self._camera.PixelFormat.SetValue(self.pixel_format)
        self._apply_exposure()
        self._apply_trigger()
        self._apply_roi()

    def close(self) -> None:
        if self._camera is not None:
            try:
                self.stop()
            finally:
                try:
                    self._camera.Close()
                finally:
                    self._camera = None

    def stop(self) -> None:
        if self._camera is not None and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    def _paused_stream(self):
        """Context manager: stop the resident grab stream around a sensor reconfiguration
        (GenICam rejects ROI/format writes while grabbing) and resume it afterwards."""
        camera = self._camera

        class _Pause:
            def __enter__(self_inner):
                self_inner.was_grabbing = camera is not None and camera.IsGrabbing()
                if self_inner.was_grabbing:
                    camera.StopGrabbing()
                return self_inner

            def __exit__(self_inner, *exc):
                # the stream restarts lazily on the next acquire (no eager restart on error)
                return False

        return _Pause()

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
            if roi in ROI_CLEAR_SENTINELS:       # FULL_FRAME (or a blank box) -> the full sensor
                self._roi = None
            else:
                h, w = self.sensor_shape
                self._roi = snap_subarray(tuple(roi), step=self.subarray_step,
                                          max_w=w or 10**6, max_h=h or 10**6)
            self._apply_roi()

    def _apply_exposure(self) -> None:
        # Basler exposes ExposureTime in microseconds; legal while grabbing (no stream pause)
        self._camera.ExposureTime.SetValue(float(self._exposure) * 1e6)

    def _apply_trigger(self) -> None:
        cam = self._camera
        cam.TriggerSelector.SetValue("FrameStart")
        if self._free_run:
            cam.TriggerMode.SetValue("Off")     # free-run: each RetrieveResult grabs the next frame
        else:
            cam.TriggerMode.SetValue("On")
            cam.TriggerSource.SetValue(self.trigger_source)

    def _apply_roi(self) -> None:
        """Push the ROI in the GenICam-safe order: pause the stream, zero the offsets, size the
        window, then place it.  Setting Width/Height while a stale OffsetX/Y is still active can
        violate ``Offset + Width <= SensorWidth`` and the camera rejects the write (the same
        zero-offsets-first dance the qCMOS subarray code does).  Every value is clamped/aligned
        to the CAMERA'S own Min/Max/Inc -- the hardware owns its legal grid."""
        if self._camera is None:
            return
        cam = self._camera
        with self._paused_stream():
            cam.OffsetX.SetValue(int(cam.OffsetX.GetMin()))
            cam.OffsetY.SetValue(int(cam.OffsetY.GetMin()))
            if self._roi is None:               # blank -> the FULL sensor, not a stale window
                cam.Width.SetValue(int(cam.WidthMax.GetValue()))
                cam.Height.SetValue(int(cam.HeightMax.GetValue()))
                return
            x, w, y, h = self._roi
            w = _snap_to_inc(w, cam.Width.GetMin(), cam.Width.GetMax(), cam.Width.GetInc())
            h = _snap_to_inc(h, cam.Height.GetMin(), cam.Height.GetMax(), cam.Height.GetInc())
            cam.Width.SetValue(w)
            cam.Height.SetValue(h)
            x = _snap_to_inc(x, cam.OffsetX.GetMin(), cam.OffsetX.GetMax(), cam.OffsetX.GetInc())
            y = _snap_to_inc(y, cam.OffsetY.GetMin(), cam.OffsetY.GetMax(), cam.OffsetY.GetInc())
            cam.OffsetX.SetValue(x)
            cam.OffsetY.SetValue(y)
            self._roi = (x, w, y, h)            # what the sensor actually granted

    # ------------------------------------------------------------------ acquisition hooks
    # The public surface (arm / read_frames / disarm / acquire) lives on the CameraDevice
    # base; this adapter implements only the pylon-facing hooks.

    @property
    def _free_run(self) -> bool:
        return self.trigger_source.lower() == SOFTWARE_TRIGGER.lower()

    def _arm(self, frames: int | None) -> None:
        """Start the grab session for an armed request.  Two modes, each with the strategy its
        semantics need:

        * ``Software`` (free-run) -- the stream is RESIDENT: it starts on the first arm and
          stays running across arm/disarm (a live monitor calls ``acquire(1)`` per shot;
          restarting the USB3 stream each time costs tens of ms and was the stutter).
          ``GrabStrategy_LatestImageOnly`` keeps the newest frame so a slow consumer always
          sees the live image, never a stale backlog.
        * hardware trigger (e.g. ``Line1``) -- one ``StartGrabbingMax`` per armed session,
          stopped by :meth:`_disarm`: one trigger = one frame = one shot, STRICTLY.  A
          resident latest-only stream could hand point K+1 a late frame from point K (a
          timed-out trigger whose frame arrives after the retry fired); per-point scans are
          slow anyway, so the restart cost is irrelevant there."""
        if self._camera is None:
            raise RuntimeError("PylonCamera.arm before open() -- the device set opens cameras last.")
        from pypylon import pylon

        cam = self._camera
        if self._free_run:
            if not cam.IsGrabbing():
                cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        else:
            if cam.IsGrabbing():
                cam.StopGrabbing()              # no resident stream in triggered mode (see above)
            if frames is None:
                cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            else:
                cam.StartGrabbingMax(int(frames), pylon.GrabStrategy_OneByOne)
        self._armed_total = None if frames is None else int(frames)
        self._grabbed = 0

    def _grab(self, n: int, *, timeout: float | None = None, stop=None) -> bool:
        """Retrieve ONE frame from the running grab session into the base frame queue.
        Fault LOUDNESS follows the trigger mode (this is the single source for both):

        * ``Software`` (free-run, the live MOT monitor): no frame within the timeout, or a
          failed grab, returns False -- ``read_frames`` returns short and the live view simply
          FREEZES, the right behaviour for a passive viewer that just sees no light;
        * hardware trigger (e.g. ``Line1``, a finite point-by-point acquisition): a trigger
          that never comes raises ``TimeoutError`` and a failed grab raises ``RuntimeError`` --
          the same loud fault model the qCMOS keeps, so a missing trigger can never silently
          leave a hole in a scan grid (the MOT-field optimiser would argmax a partial block).

        A cooperative Stop keeps its cancel semantics in BOTH modes (return False -> a clean
        short read, never a fault).  When a stop event is passed the wait is SLICED
        (``min(timeout, 200 ms)`` per ``RetrieveResult``) with the event re-checked between
        slices, so a Stop pressed mid-wait interrupts within ~one slice instead of sitting
        inside RetrieveResult for the full timeout (the same cooperative-cancel granularity the
        qCMOS grab uses).  The TOTAL timeout budget is unchanged -- only a genuinely absent
        trigger for the whole timeout faults/freezes."""
        from pypylon import pylon

        cam = self._camera
        timeout_ms = int((self.timeout if timeout is None else float(timeout)) * 1000)
        slice_ms = min(timeout_ms, 200) if stop is not None else timeout_ms
        deadline = time.monotonic() + timeout_ms / 1000.0
        # Which frame of how many this armed session is waiting for (the qCMOS message style).
        which = (f"frame {self._grabbed}" if self._armed_total is None
                 else f"frame {self._grabbed} of {self._armed_total}")
        while True:
            if stop is not None and stop.is_set():
                return False                    # cooperative Stop: clean cancel, never a fault
            wait_ms = slice_ms
            if stop is not None:
                wait_ms = max(1, min(slice_ms, int((deadline - time.monotonic()) * 1000)))
            result = cam.RetrieveResult(wait_ms, pylon.TimeoutHandling_Return)
            if result is None:
                # This slice saw no frame: keep polling until stopped or the total timeout expires.
                if stop is not None and not stop.is_set() and time.monotonic() < deadline:
                    continue
                if stop is not None and stop.is_set():
                    return False                # cooperative Stop: clean cancel, never a fault
                if self._free_run:
                    return False                # free-run: no light is normal -> the view freezes
                raise TimeoutError(
                    f"pylon timed out after {timeout_ms} ms waiting for {which} "
                    f"(no edge on trigger {self.trigger_source!r}).")
            if not result.GrabSucceeded():
                result.Release()
                if self._free_run:
                    return False                # free-run: a bad grab freezes the view (partial ok)
                raise RuntimeError(
                    f"pylon grab failed for {which} "
                    f"(GrabSucceeded() is False on trigger {self.trigger_source!r}).")
            self._deliver([np.array(result.Array, copy=True)])
            result.Release()
            self._grabbed += 1
            return True

    def _disarm(self) -> None:
        # Free-run keeps its resident stream; a triggered session ends STRICTLY here so a
        # late frame can never leak into the next armed request (the Y-round misalignment fix).
        if not self._free_run and self._camera is not None and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self._roi,
            "serial": self.serial,
            "trigger_source": self.trigger_source,
            "capture_trigger_channels": list(self.capture_trigger_channels),
            "pixel_format": self.pixel_format,
        }
