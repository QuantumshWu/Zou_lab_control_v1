"""Base device contracts for hardware and virtual neutral-atom devices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


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
    """Required contract for a camera used by ``NeutralAtomSession``."""

    def bind_experiment(self, session) -> "CameraDevice":
        """Attach experiment defaults used by convenience methods like capture."""

        self._zlc_session = session
        return self

    @property
    @abstractmethod
    def exposure(self) -> float:
        """Current default exposure in seconds."""

    @abstractmethod
    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        """Configure camera settings that are stable across an acquisition."""

    @abstractmethod
    def acquire(self, frames: int = 1, *, sequence=None, sequencer=None, **kwargs) -> list[np.ndarray]:
        """Acquire ``frames`` images and return one numpy array per frame."""

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


__all__ = ["BaseDevice", "CameraDevice", "SequencerDevice", "TrapArrayDevice", "validate_device_contract"]
