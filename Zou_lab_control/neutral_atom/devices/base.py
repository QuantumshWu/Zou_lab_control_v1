"""Base device contracts for hardware and virtual neutral-atom devices."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np

from .camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS


class AcquisitionCancelled(Exception):
    """Raised by ``acquire`` when its optional ``stop`` event fires mid-wait.

    Distinct from ``TimeoutError`` (a real fault): cancellation is an
    intentional Stop, so the logic node loop treats it as a clean exit rather than a
    recorded error / banner."""


class BaseDevice(ABC):
    """Common device lifecycle.

    Concrete hardware adapters must inherit this class or one of its typed
    subclasses.  This is intentionally stricter than duck typing: missing
    methods should be caught when the class is written, not halfway through a
    long experiment.
    """

    def open(self):
        return self

    def close(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"type": type(self).__name__}


class CameraDevice(BaseDevice):
    """Required contract for a camera used by ``NeutralAtomSession``.

    Owns a small RECENT-FRAMES ring (``recent_capacity`` frames).  The camera is
    externally triggered, so a single fired shot can yield MORE frames than a
    consumer requested (e.g. two ``emCCD`` triggers inside one pulse); every
    ``acquire`` retains its frames here so a live consumer that polls :meth:`drain`
    never misses the extra ones and :meth:`latest` always holds the newest.  This
    lives on the base class, so the virtual and real cameras retain identically
    (virtual == real); subclasses only call :meth:`_retain` at the end of acquire.

    A camera is a PURE frame-grabber: it arms for N frames and reads N externally
    triggered frames (a real qCMOS: ``buf_alloc(N)`` -> ``cap_start`` -> wait for the
    FPGA's hardware trigger edges -> ``buf_getframedata``).  It NEVER drives a sequencer,
    NEVER counts triggers in a pulse, NEVER infers exposure -- the MEASUREMENT orchestrates
    the shot (fire the sequencer, then read the camera).  The one thing a camera owns about
    the wider system is ``capture_trigger_channels`` -- which sequencer line its hardware
    trigger input is wired to -- a PASSIVE fact it exposes upward (so a measurement / the
    virtual atom simulation can read the fired pulse), never used to control the sequencer.
    """

    recent_capacity: int = 16
    #: Which sequencer channel(s) the camera's external-trigger input is wired to.  A passive
    #: device property (the camera is the source of this wiring fact), default the conventional
    #: ``emCCD`` line; a real camera is configured with its actual chNN at construction.
    capture_trigger_channels: tuple[str, ...] = DEFAULT_CAMERA_TRIGGER_CHANNELS

    def bind_experiment(self, session) -> "CameraDevice":
        """Attach experiment defaults used by convenience methods like capture."""

        self._zlc_session = session
        return self

    @property
    @abstractmethod
    def exposure(self) -> float:
        """Current default exposure in seconds."""

    @exposure.setter
    def exposure(self, value: float) -> None:
        """Write-through to :meth:`configure` -- so ``cam.exposure = 3e-3`` works uniformly on
        every backend.  Exposure is the ONE intrinsic scalar where ``=`` reads naturally; it
        routes through ``configure`` (the single hardware-write path) and never hides more than
        setting the exposure.  Multi-field / ROI changes still go through ``configure(...)``
        directly (an ROI snap / multi-subsystem write must not hide behind ``=``).  A backend
        that overrides the ``exposure`` getter must re-declare this setter (it would otherwise be
        shadowed); the concrete cameras (VirtualCamera, QCMOSCamera) do."""
        self.configure(exposure=float(value))

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """Sub-array readout window ``(x, width, y, height)``, or None for full frame.

        Part of the contract so a consumer (e.g. a raw-frame logic node's Edit panel)
        reads ROI without reaching into a backend's private ``config``.  Default
        None: a camera with no sub-array concept (the virtual renderer) honestly
        has no ROI; the real qCMOS overrides this."""
        return None

    @property
    def sensor_shape(self) -> tuple[int, int] | None:
        """The FULL sensor size ``(height, width)`` in pixels, or None if unknown.

        Part of the contract so a consumer (a raw-frame Edit panel) can show the ROI as
        the full-frame window even when no sub-array is set (``roi is None``) -- without
        reaching into a backend's internals.  Default None (size unknown until a frame is
        read); a backend that knows its sensor up front (the virtual renderer, the real
        qCMOS) overrides this."""
        return None

    @abstractmethod
    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        """Configure camera settings that are stable across an acquisition."""

    @abstractmethod
    def acquire(self, frames: int = 1, *, sequence=None, on_armed=None, stop=None, **kwargs) -> list[np.ndarray]:
        """Arm for ``frames`` frames and return one numpy array per externally-triggered frame.

        The camera is a PURE grabber: ``frames`` is the count the MEASUREMENT wants (it knows
        how many camera triggers its pulse carries -- e.g. ``frames_per_cycle`` for a live cycle,
        or the bracket length for a calibration shot).  The camera does NOT count the sequence's
        triggers, does NOT prepare/fire any sequencer, does NOT infer exposure.

        ``on_armed`` (optional) is a callback the camera invokes AFTER it has armed (a real
        ``cap_start``) and BEFORE it waits -- so the measurement can fire the FPGA exactly when
        the camera is ready for triggers (arm-before-fire), without the camera knowing anything
        about the sequencer.  ``sequence`` (optional) is the fired pulse, passed purely so the
        VIRTUAL camera can SIMULATE the frames it would have captured (a real camera ignores it --
        its frames come from the sensor); ``None`` with no firing pulse -> no frame (the live image
        freezes, exactly as a real externally-triggered camera produces nothing without triggers).

        Optional ``stop``: a ``threading.Event`` a live logic node can set to interrupt a blocking
        wait (a camera awaiting an external trigger that never comes).  Implementations that honour
        it poll the event and raise :class:`AcquisitionCancelled` when set; others may ignore it."""

    # ----------------------------------------------------------- recent frames
    def _recent_state(self) -> dict:
        """Lazily create the recent-frames ring (subclasses define their own
        ``__init__`` and need not call ``super().__init__()``)."""
        state = self.__dict__.get("_zlc_recent")
        if state is None:
            state = {
                "frames": deque(maxlen=max(1, int(self.recent_capacity))),
                "lock": threading.Lock(),
                "seq": 0,       # total frames ever retained
                "cursor": 0,    # drain watermark
            }
            self.__dict__["_zlc_recent"] = state
        return state

    def _retain(self, images) -> Any:
        """Append freshly-acquired frames to the recent-frames ring (thread-safe).

        Subclasses call this once at the end of ``acquire`` and return ``images``
        unchanged.  Returns ``images`` so ``return self._retain(images)`` reads
        cleanly."""
        state = self._recent_state()
        with state["lock"]:
            for image in images:
                state["frames"].append(np.asarray(image))
                state["seq"] += 1
        return images

    def recent_frames(self, n: int | None = None) -> list[np.ndarray]:
        """The most-recent retained frames (newest last); all of them when ``n`` is None."""
        state = self._recent_state()
        with state["lock"]:
            frames = list(state["frames"])
        return frames if n is None else frames[-int(n):]

    def latest(self) -> np.ndarray | None:
        """The single newest retained frame, or None if nothing has been acquired."""
        state = self._recent_state()
        with state["lock"]:
            return state["frames"][-1] if state["frames"] else None

    def drain(self) -> list[np.ndarray]:
        """Every frame retained since the previous ``drain`` (lossless up to capacity).

        A live consumer polling this gets ALL frames captured between its polls --
        so a multi-trigger shot's extra frames are never dropped on the floor."""
        state = self._recent_state()
        with state["lock"]:
            frames = list(state["frames"])
            n_new = max(0, min(len(frames), state["seq"] - state["cursor"]))
            state["cursor"] = state["seq"]
            return frames[-n_new:] if n_new else []

    def clear_recent(self) -> None:
        """Drop retained frames and reset the drain watermark (e.g. on reconfigure)."""
        state = self._recent_state()
        with state["lock"]:
            state["frames"].clear()
            state["cursor"] = state["seq"]

    def capture(self, *, frames: int = 1, exposure: float | None = None, sequence=None, display: bool = True, **kwargs):
        """Acquire images and return a notebook-friendly ``CaptureResult``.

        This is a camera-device method, not a session wrapper.  When the
        camera is attached to a ``NeutralAtomSession``, the session supplies
        default timing, sequencer, history, and frontend plotting.  A standalone
        camera can still call this method by passing an explicit
        sequence/sequencer.  ``capture`` always shows raw camera data; site
        overlays belong to calibrated readout/detection, not to capture.
        """

        from ..core.results import CaptureResult
        from ..timing import imaging_sequence
        from ..views.plots import plot_image

        explicit_sequencer = kwargs.pop("sequencer", None)
        session = getattr(self, "_zlc_session", None)
        if session is not None:
            sequence = sequence or (session._configure_imaging(exposure=exposure) if exposure is not None else session.sequence)
            sequencer = explicit_sequencer if explicit_sequencer is not None else getattr(session.devices, "sequencer", None)
        else:
            if exposure is not None:
                self.configure(exposure=exposure)
            sequence = sequence or imaging_sequence(exposure=self.exposure, load=True)
            sequencer = explicit_sequencer

        # Orchestrate the shot the decoupled way: the camera arms, then (on_armed) we fire the
        # sequencer -- the camera never drives it.  ``sequence`` is handed in only so the virtual
        # camera can simulate; a real camera reads the frames its hardware trigger gates.
        on_armed = arm_then_fire(sequencer if sequence is not None else None, sequence)
        images = self.acquire(frames, sequence=sequence, on_armed=on_armed, **kwargs)
        plot = plot_image(images[-1], display=display)
        result = CaptureResult(images=images, sequence=sequence, plot=plot)
        if session is not None:
            session.history.append(result)
        return result


class SequencerDevice(BaseDevice):
    """Required contract for a timing/sequencer backend."""

    channels: list[str] | tuple[str, ...]
    clock_hz: float

    @abstractmethod
    def prepare(self, sequence) -> Any:
        """Validate/compile/arm a pulse sequence."""

    @abstractmethod
    def fire(self, sequence=None) -> None:
        """Start a previously prepared sequence."""

    @property
    def firing(self) -> "Any | None":
        """The ``repeat_forever`` program the streamer is continuously playing, or None when
        idle/safe.  The live (no-sequence) camera read is gated on this -- it is the software
        model of "the FPGA is emitting camera triggers right now".

        Part of the contract (like :meth:`scan_progress`) so the camera/measurement layer reads
        the live firing state through the abstraction, not via a duck-typed ``getattr`` on a
        concrete backend.  Default None: a real/remote streamer keeps no local firing flag -- the
        camera learns "is it triggering" from the PRESENCE/ABSENCE of hardware trigger edges, not
        from a host-side handle -- so None is the correct real-hardware semantics.  The in-process
        VirtualSequencer overrides this to expose the program it is simulating."""

        return None

    def wait_done(self, timeout: float | None = None) -> bool:
        """Wait until the prepared finite sequence is done, when supported."""

        return True

    def scan_progress(self) -> dict:
        """Where the running scan is now: the SINGLE-source dict {scanning, point, n_points,
        sweep, n_repeats} (see ``sequencer.scan_progress_fields``).  A backend with no live
        scan -- or one that does not track progress -- reports idle.  The GUI polls this to show
        "point K / N · sweep r / R"; virtual and real backends return the same shape."""

        from .sequencer import SCAN_PROGRESS_IDLE
        return dict(SCAN_PROGRESS_IDLE)

    def settle(self, seconds: float, *, stop: "threading.Event | None" = None) -> None:
        """Idle for ``seconds`` after a finite pulse before the next load+fire.

        The DEVICE owns this inter-shot wait so a software-stepped sweep (e.g. an API-slot
        pulse-scan: load -> on_pulse -> wait the pulse done -> settle -> next) does not hand-roll
        timing in the caller.  The hardware simply sits in its idle/safe state during this window;
        the default is a plain host-side wait (a real backend is idle between fires).  ``stop``
        (when given) makes the wait COOPERATIVELY cancellable -- it returns early once the event is
        set, so a Stop pressed mid-settle does not block teardown for the full delay.  A virtual
        backend scales the delay by ``sleep_scale`` so tests fast-forward."""

        self._sleep_interruptible(float(seconds), stop)

    @staticmethod
    def _sleep_interruptible(seconds: float, stop: "threading.Event | None") -> None:
        """Sleep ``seconds`` in small slices, returning early if ``stop`` is set."""
        if seconds <= 0.0:
            return
        if stop is None:
            time.sleep(seconds)
            return
        deadline = time.monotonic() + seconds
        while not stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            stop.wait(min(0.05, remaining))

    def abort(self) -> None:
        """Abort the current sequence, when supported."""

        self.stop()

    def set_safe_state(self) -> None:
        """Drive outputs to a safe idle state, when supported."""

        self.stop()


def arm_then_fire(sequencer, sequence):
    """The single ``on_armed`` callback factory for the arm-before-fire shot.

    A camera is a pure grabber: a consumer arms it (:meth:`CameraDevice.acquire`),
    and the camera invokes ``on_armed`` AFTER it is armed and BEFORE it waits, so
    the measurement fires the sequencer at exactly the moment the camera is ready
    for triggers -- the camera never drives the sequencer.  Every readout / scan /
    capture path needs the same callback ("prepare then fire this sequence on this
    sequencer"), so it lives here next to the :class:`SequencerDevice` contract
    rather than being hand-copied as an inline lambda at each call site.

    Returns ``None`` when there is no bound sequencer (a notebook-composed readout
    that leans on the virtual atom array -- nothing to fire), otherwise a no-arg
    callable that prepares then fires ``sequence``.  The returned closure captures
    ``sequencer``/``sequence`` directly (they are this call's arguments, so there is
    no late-binding loop-variable hazard)."""
    if sequencer is None:
        return None
    return lambda: (sequencer.prepare(sequence), sequencer.fire(sequence))


class TrapArrayDevice(BaseDevice):
    """Required contract for a trap-array state source.

    Device implementations intentionally do not expose camera-space site
    centers.  Those are experimental calibration data and must enter the
    readout stack through sitemap calibration, not through simulator or hardware
    internals.
    """

    @property
    @abstractmethod
    def n_sites(self) -> int:
        """Number of trap sites."""


ROLE_BASES = {
    "camera": CameraDevice,
    "sequencer": SequencerDevice,
    "trap_array": TrapArrayDevice,
}


def validate_device_contract(name: str, device: Any) -> None:
    """Raise if a configured device does not inherit its required base class."""

    expected = ROLE_BASES.get(name, BaseDevice)
    if not isinstance(device, expected):
        raise TypeError(
            f"device {name!r} ({type(device).__name__}) must inherit {expected.__name__}. "
            "Implement the appropriate BaseDevice subclass instead of relying on duck typing."
        )


def snap_subarray(roi, *, step: int, max_w: int, max_h: int):
    """Snap a REQUESTED sub-array window to a camera's valid sub-array grid.

    A plot selection (scroll / area-select) is in continuous source-pixel
    coordinates; a real sensor only reads sub-arrays whose origin AND size are
    multiples of a hardware ``step`` (the Hamamatsu qCMOS requires multiples of
    4 -- see the DCAM ``SUBARRAY*`` properties), within the sensor.  This is the
    SINGLE source of truth for that adaptation, so the GUI/measurement layer can
    stay in plain plot coordinates and every camera (real or virtual) snaps the
    SAME way -- a window written raw would otherwise be silently clamped by the
    hardware and the camera would image the wrong region.

    ``roi`` is ``(x, width, y, height)``.  Returns the snapped ``(x, w, y, h)``
    with every field a multiple of ``step``, ``w``/``h`` >= ``step`` and the
    window fully inside ``(max_w, max_h)``.
    """
    step = int(step)
    if step <= 0:
        step = 1
    x, w, y, h = (int(round(float(v))) for v in roi)

    def _snap(v: int) -> int:
        return int(round(v / step)) * step

    max_w = (int(max_w) // step) * step
    max_h = (int(max_h) // step) * step
    w = min(max(step, _snap(w)), max_w)
    h = min(max(step, _snap(h)), max_h)
    x = min(max(0, _snap(x)), max_w - w)
    y = min(max(0, _snap(y)), max_h - h)
    return (x, w, y, h)


__all__ = [
    "BaseDevice",
    "CameraDevice",
    "SequencerDevice",
    "TrapArrayDevice",
    "arm_then_fire",
    "snap_subarray",
    "validate_device_contract",
]
