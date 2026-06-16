"""Base device contracts for hardware and virtual neutral-atom devices."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np


class AcquisitionCancelled(Exception):
    """Raised by ``acquire`` when its optional ``stop`` event fires mid-wait.

    Distinct from ``TimeoutError`` (a real fault): cancellation is an
    intentional Stop, so the feed loop treats it as a clean exit rather than a
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
    """

    recent_capacity: int = 16

    def bind_experiment(self, session) -> "CameraDevice":
        """Attach experiment defaults used by convenience methods like capture."""

        self._zlc_session = session
        return self

    @property
    @abstractmethod
    def exposure(self) -> float:
        """Current default exposure in seconds."""

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """Sub-array readout window ``(x, width, y, height)``, or None for full frame.

        Part of the contract so a consumer (e.g. a raw-frame feed's Edit panel)
        reads ROI without reaching into a backend's private ``config``.  Default
        None: a camera with no sub-array concept (the virtual renderer) honestly
        has no ROI; the real qCMOS overrides this."""
        return None

    @abstractmethod
    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        """Configure camera settings that are stable across an acquisition."""

    @abstractmethod
    def acquire(self, frames: int = 1, *, sequence=None, sequencer=None, **kwargs) -> list[np.ndarray]:
        """Acquire ``frames`` images and return one numpy array per frame.

        Optional ``stop`` kwarg: a ``threading.Event`` a live feed can set to
        interrupt a blocking wait (e.g. a camera awaiting an external trigger
        that never comes).  Implementations that honour it poll the event while
        waiting and raise :class:`AcquisitionCancelled` when it is set, instead
        of blocking for the full timeout; implementations that cannot interrupt
        may ignore it."""

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

        images = self.acquire(frames, sequence=sequence, sequencer=sequencer, **kwargs)
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

    def wait_done(self, timeout: float | None = None) -> bool:
        """Wait until the prepared finite sequence is done, when supported."""

        return True

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
    "snap_subarray",
    "validate_device_contract",
]
