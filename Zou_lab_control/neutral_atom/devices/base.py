"""Base device contracts for hardware and virtual neutral-atom devices."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np

from ..core.analysis import positive_int
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


#: The ONE "clear the ROI back to the full sensor" sentinel every camera ``configure(roi=...)``
#: accepts.  ``roi=None`` means "leave unchanged" (so a multi-field configure can skip it), which
#: is why clearing needs an explicit value: the measurement layer sends this canonical spelling.
FULL_FRAME = "full"

#: Every spelling ``configure(roi=...)`` treats as "clear to full frame": the canonical
#: ``FULL_FRAME`` plus ``""`` / ``"None"`` (what a GUI's blank text box naturally produces).
#: Single source -- each backend tests membership HERE, so the accepted spellings can never
#: drift between cameras (the old per-backend ("", "None") tuples were an unreachable chain:
#: the measurement layer mapped a blank region to None, which means "leave unchanged").
ROI_CLEAR_SENTINELS = (FULL_FRAME, "", "None")


class CameraDevice(BaseDevice):
    """Required contract for a camera used by ``NeutralAtomSession``.

    A camera is a PURE frame-grabber with THREE acquisition primitives:

    * :meth:`arm` -- ready the hardware for externally triggered frames;
    * :meth:`read_frames` -- blockingly consume frames from the device's own buffer;
    * :meth:`disarm` -- stand the hardware down.

    (:meth:`acquire` is the free-run convenience composing the three.)  The camera NEVER
    drives a sequencer, NEVER counts triggers in a pulse, NEVER infers exposure, and knows
    NOTHING about the experiment session -- once armed it simply waits for triggers.  The
    MEASUREMENT layer orchestrates the shot (arm the camera, fire the sequencer, read the
    frames back): see ``operations.measurement.triggered_frames``, the single arm-before-fire
    helper.  The one thing a camera owns about the wider system is
    ``capture_trigger_channels`` -- which sequencer line its hardware trigger input is wired
    to -- a PASSIVE fact it exposes upward, never used to control the sequencer.

    Frame-loss protection is a DEVICE-OWNED buffer: every frame that arrives while the
    camera is armed is queued LOSSLESSLY (unbounded ``pending`` queue) until
    :meth:`read_frames` consumes it, so a fire that lands before the consumer starts
    reading ("late consumption") drops nothing.  Independently, a small RECENT-FRAMES ring
    (``recent_capacity``) keeps the newest frames for live viewers (:meth:`drain` /
    :meth:`latest`) -- lossy by design, a convenience view, never the acquisition path.
    Both live on this base class so the virtual and real cameras buffer identically
    (virtual == real); subclasses feed frames through :meth:`_deliver`.
    """

    recent_capacity: int = 16
    #: Which sequencer channel(s) the camera's external-trigger input is wired to.  A passive
    #: device property (the camera is the source of this wiring fact), default the conventional
    #: ``emCCD`` line; a real camera is configured with its actual chNN at construction.
    capture_trigger_channels: tuple[str, ...] = DEFAULT_CAMERA_TRIGGER_CHANNELS

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

    @property
    def frame_shape(self) -> tuple[int, int] | None:
        """The ``(height, width)`` of the NEXT frame this camera will deliver -- DERIVED
        (never stored) from the two contract facts above: the sub-array window when an
        ROI is set, else the full :attr:`sensor_shape`; ``None`` when the backend knows
        neither yet (shape is then learnt from the first frame).

        This is what lets a consumer DECLARE its output structure at BUILD time instead
        of waiting for a frame: a ``CameraMeasurement`` seeds its ``data_shape`` from
        here, so a panel bound to ``frame_0`` can render the hub's lingering block the
        moment the measurement (re)starts -- even while no trigger is firing yet."""
        r = self.roi
        if r is not None:
            _x, width, _y, height = (int(v) for v in r)   # ROI contract order: (x, width, y, height)
            return (height, width)
        return self.sensor_shape

    @abstractmethod
    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        """Configure camera settings that are stable across an acquisition.

        Backends declare the keyword arguments they accept and reject any unknown
        key via :meth:`_reject_unknown_configure_keys` -- so a mistyped or
        backend-specific option fails LOUDLY and identically on every backend
        (virtual == real), never silently ignored on one and a ``TypeError`` on
        another."""

    @staticmethod
    def _reject_unknown_configure_keys(allowed: set[str], got) -> None:
        """Raise ``ValueError`` listing the configurable options if ``got`` carries an
        unknown key.

        Shared so ``configure`` enforces the SAME contract on every backend: an
        option a backend does not recognise is an explicit error (naming the keys it
        DOES accept), not a silent no-op on one camera and a ``TypeError`` on the
        next.  ``allowed`` is the backend's own configurable set (always includes
        ``exposure``); ``got`` is its ``**kwargs`` of leftover keys."""
        unknown = sorted(set(got) - allowed)
        if unknown:
            raise ValueError(
                f"unknown configure option(s) {unknown}; configurable: {sorted(allowed)}")

    # ------------------------------------------------------------- acquisition
    def arm(self, frames: int | None = None) -> None:
        """Ready the camera for ``frames`` externally triggered frames (None = continuous).

        When this method RETURNS the hardware is armed and waiting for triggers, so a fire
        issued after ``arm()`` can NEVER outrun the camera -- this ordering IS the
        arm-before-fire guarantee every measurement relies on (arm the camera, THEN fire the
        sequencer, then :meth:`read_frames`).  Frames arriving while armed are queued
        losslessly until read.  Arming also takes the camera's acquisition lock, serializing
        concurrent consumers (a live monitor polling :meth:`acquire` waits while a
        measurement holds an armed session); :meth:`disarm` releases it."""
        if frames is not None:
            frames = positive_int(frames, "frames")
        self._acquire_lock().acquire()
        state = self._recent_state()
        with state["cond"]:
            state["pending"].clear()
            state["armed"] = True
            state["armed_frames"] = frames
        try:
            self._arm(frames)
        except BaseException:
            with state["cond"]:
                state["armed"] = False
                state["pending"].clear()
            self._acquire_lock().release()
            raise

    def read_frames(self, n: int = 1, *, timeout: float | None = None, stop=None, **kwargs) -> list[np.ndarray]:
        """Blockingly consume ``n`` frames from the device's own buffer (one numpy array each).

        Frames queued while armed (pushed by the data source, or fetched by the backend's own
        grab hook) are drained first -- so a fire that happened BEFORE this call loses nothing;
        when the buffer runs dry the backend's ``_grab`` hook produces/awaits more.  ``timeout``
        (seconds, backend default when None) bounds the wait for a trigger that never comes;
        ``stop`` (a ``threading.Event``) cancels a blocking wait cooperatively.  Returns what
        arrived (possibly fewer than ``n``); backends whose fault model is loud (the qCMOS)
        raise ``TimeoutError`` / :class:`AcquisitionCancelled` from their hook instead.

        Requires the camera to be ARMED first: the three primitives are ordered arm -> read ->
        disarm, and only :meth:`arm` fills the lossless ``pending`` queue the data source pushes
        into.  Reading unarmed would consume from an empty queue while a push-fed source's
        ``_grab`` (e.g. the virtual trigger wire during a continuous firing) reports "more will
        arrive" -- an unbounded live-lock.  So an unarmed read is a programming error, raised
        loudly; use :meth:`acquire` for a one-shot (it arms internally)."""
        n = positive_int(n, "n")
        state = self._recent_state()
        if not state["armed"]:
            raise RuntimeError(
                "read_frames() requires arm() first (the primitives are arm -> read -> disarm); "
                "use acquire() for a one-shot that arms internally.")
        out: list[np.ndarray] = []
        while len(out) < n:
            with state["cond"]:
                while state["pending"] and len(out) < n:
                    out.append(state["pending"].pop(0))
            if len(out) >= n:
                break
            if not self._grab(n - len(out), timeout=timeout, stop=stop, **kwargs):
                break
        return out

    def disarm(self) -> None:
        """Stand the camera down and release the acquisition lock taken by :meth:`arm`."""
        state = self._recent_state()
        try:
            self._disarm()
        finally:
            with state["cond"]:
                state["armed"] = False
                state["pending"].clear()
            try:
                self._acquire_lock().release()
            except RuntimeError:
                pass  # disarm without a matching arm -- defensive no-op

    def acquire(self, frames: int = 1, *, stop=None, **kwargs) -> list[np.ndarray]:
        """Convenience one-shot: ``arm(frames)`` + ``read_frames(frames)`` + ``disarm()``.

        For a camera that needs no external fire -- a free-running sensor (Basler
        ``Software`` mode), or a virtual camera watching an already-firing continuous
        pulse -- this is the whole story.  An externally TRIGGERED measurement must fire
        its sequencer between arm and read: use the single measurement-layer helper
        ``operations.measurement.triggered_frames`` (never hand-roll prepare+fire)."""
        frames = positive_int(frames, "frames")
        self.arm(frames)
        try:
            return self.read_frames(frames, stop=stop, **kwargs)
        finally:
            self.disarm()

    # --------------------------------------------------- backend hooks (subclass seam)
    def _arm(self, frames: int | None) -> None:
        """Backend hook: ready the hardware (``buf_alloc`` + ``cap_start`` on a qCMOS,
        ``StartGrabbing`` on a pylon camera).  Default: nothing to do (a push-fed source)."""

    def _grab(self, n: int, *, timeout: float | None = None, stop=None) -> bool:
        """Backend hook: produce or await at least one more frame for :meth:`read_frames`.

        Pull-based hardware fetches from its driver and feeds :meth:`_deliver`; a push-fed
        source may simply wait on the buffer condition.  Return True when new frames were
        (or will now be found) queued; False when none will arrive within ``timeout``
        (``read_frames`` then returns what it has).  Default: wait for a pushed frame."""
        state = self._recent_state()
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with state["cond"]:
            while not state["pending"]:
                if stop is not None and stop.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return False
                state["cond"].wait(0.05 if remaining is None else min(0.05, remaining))
            return True

    def _disarm(self) -> None:
        """Backend hook: stand the hardware down (``cap_stop`` / ``StopGrabbing``)."""

    # ----------------------------------------------------------- frame buffers
    def _recent_state(self) -> dict:
        """Lazily create the frame-buffer state (subclasses define their own
        ``__init__`` and need not call ``super().__init__()``).

        ``dict.setdefault`` is the ATOMIC create-or-get: two threads first touching an
        unbuilt camera (a measurement arming while a live monitor drains) both build a
        candidate, but the GIL makes ``setdefault`` a single winner -- both then read back
        the SAME state, never two divergent buffers with acquisitions serialised against
        different locks.  The loser's freshly-built dict is discarded (harmless -- it holds
        no frames yet)."""
        lock = threading.Lock()
        fresh = {
            "frames": deque(maxlen=max(1, int(self.recent_capacity))),
            "lock": lock,
            "cond": threading.Condition(lock),
            "seq": 0,        # total frames ever retained
            "cursor": 0,     # drain watermark
            "pending": [],   # LOSSLESS armed-session queue read_frames consumes
            "armed": False,
            "armed_frames": None,
        }
        return self.__dict__.setdefault("_zlc_recent", fresh)

    def _acquire_lock(self) -> "threading.RLock":
        """The per-camera acquisition lock :meth:`arm`/:meth:`disarm` hold across an armed
        session, so two consumers (a measurement + a live monitor) never interleave their
        armed state on one sensor.  Atomically create-or-get via ``setdefault`` (like
        :meth:`_recent_state`) so concurrent first touches share ONE lock -- a check-then-set
        would let two threads build two locks and defeat the mutual exclusion."""
        return self.__dict__.setdefault("_zlc_acquire_lock", threading.RLock())

    def _deliver(self, images) -> list[np.ndarray]:
        """Queue freshly captured frames: into the lossless armed ``pending`` queue (when
        armed) AND the recent-frames ring, then wake any waiting reader.  The ONE entry
        point every frame source uses (a real backend's grab hook, the virtual trigger
        wire), so buffering behaviour cannot drift between backends."""
        state = self._recent_state()
        arrs = [np.asarray(image) for image in images]
        with state["cond"]:
            if state["armed"]:
                state["pending"].extend(arrs)
            self._retain_locked(state, arrs)
            state["cond"].notify_all()
        return arrs

    @staticmethod
    def _retain_locked(state: dict, images) -> None:
        """Append frames to the recent ring; caller holds ``state['lock']``."""
        for image in images:
            state["frames"].append(np.asarray(image))
            state["seq"] += 1

    def _retain(self, images) -> Any:
        """Append frames to the recent-frames ring only (thread-safe); returns ``images``."""
        state = self._recent_state()
        with state["lock"]:
            self._retain_locked(state, images)
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
        idle/safe -- the software model of "the FPGA is emitting camera triggers right now".

        Part of the contract (like :meth:`scan_progress`) so a DEVICE-layer consumer (the
        virtual trigger wire inside ``devices.virtual``) reads the live firing state through
        the abstraction, not via a duck-typed ``getattr`` on a concrete backend.  Default
        None: a real/remote streamer keeps no local firing flag -- a real camera learns "is it
        triggering" from the PRESENCE/ABSENCE of hardware trigger edges, not from a host-side
        handle -- so None is the correct real-hardware semantics.  The in-process
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
    "FULL_FRAME",
    "ROI_CLEAR_SENTINELS",
    "SequencerDevice",
    "TrapArrayDevice",
    "snap_subarray",
]
