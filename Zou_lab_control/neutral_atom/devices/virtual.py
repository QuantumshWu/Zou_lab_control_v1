"""Virtual devices for offline notebook tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import time

import numpy as np

from ..core.analysis import finite_float, grid_shape_tuple, positive_int
from .base import CameraDevice, SequencerDevice, TrapArrayDevice
from ..timing import PulseSequence, exposure_from_sequence, sequence_for_frame_count


DEFAULT_CHANNELS = (
    "trap",
    "cooling",
    "probe",
    "qcm_trigger",
    "camera_trigger",
    "trig",
    "pushout",
    "microwave",
)


@dataclass
class VirtualTrapArray(TrapArrayDevice):
    grid_shape: tuple[int, int] = (5, 7)
    image_shape: tuple[int, int] = (96, 128)
    spacing_px: float = 12.0
    origin_px: tuple[float, float] | None = None
    loading_probability: float = 0.55
    atom_rate: float = 3_000.0
    background_rate: float = 8.0
    dark_current_e_per_s: float = 0.006
    offset_counts: float = 200.0
    conversion_e_per_count: float = 0.107
    read_noise_e: float = 0.43
    atom_sigma_px: float = 1.35
    detection_lifetime: float = 10.0
    seed: int | None = 7
    occupancy: np.ndarray | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grid_shape = grid_shape_tuple(self.grid_shape)
        self.image_shape = grid_shape_tuple(self.image_shape, "image_shape")
        self.spacing_px = positive_float(self.spacing_px, "spacing_px")
        self.loading_probability = probability(self.loading_probability, "loading_probability")
        self.atom_rate = positive_float(self.atom_rate, "atom_rate")
        self.background_rate = positive_float(self.background_rate, "background_rate")
        self.dark_current_e_per_s = nonnegative_float(self.dark_current_e_per_s, "dark_current_e_per_s")
        self.offset_counts = nonnegative_float(self.offset_counts, "offset_counts")
        self.conversion_e_per_count = positive_float(self.conversion_e_per_count, "conversion_e_per_count")
        self.read_noise_e = nonnegative_float(self.read_noise_e, "read_noise_e")
        self.atom_sigma_px = positive_float(self.atom_sigma_px, "atom_sigma_px")
        self.detection_lifetime = positive_float(self.detection_lifetime, "detection_lifetime")
        self.rng = np.random.default_rng(self.seed)
        if self.origin_px is None:
            ny, nx = self.grid_shape
            h, w = self.image_shape
            self.origin_px = ((w - (nx - 1) * self.spacing_px) / 2.0, (h - (ny - 1) * self.spacing_px) / 2.0)
        else:
            self.origin_px = point_tuple(self.origin_px, "origin_px")
        if self.occupancy is None:
            self.reload()
        else:
            self.set_occupancy(self.occupancy)

    @property
    def n_sites(self) -> int:
        return int(np.prod(self.grid_shape))

    def _site_centers(self) -> np.ndarray:
        ny, nx = self.grid_shape
        x0, y0 = self.origin_px
        return np.asarray([[x0 + ix * self.spacing_px, y0 + iy * self.spacing_px] for iy in range(ny) for ix in range(nx)], dtype=float)

    def reload(self) -> np.ndarray:
        self.occupancy = self.rng.random(self.n_sites) < self.loading_probability
        return self.occupancy.copy()

    def set_occupancy(self, occupied: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(occupied)
        if arr.dtype == bool:
            flat = arr.reshape(-1)
            if flat.size != self.n_sites:
                raise ValueError(f"boolean occupancy must have length {self.n_sites}.")
            self.occupancy = flat.astype(bool, copy=True)
            return
        out = np.zeros(self.n_sites, dtype=bool)
        for value in np.asarray(occupied).reshape(-1):
            index = site_index(value, self.n_sites)
            out[index] = True
        self.occupancy = out

    def render_image(self, *, exposure: float, all_sites: bool = False) -> np.ndarray:
        exposure = positive_float(exposure, "exposure")
        h, w = self.image_shape
        yy, xx = np.mgrid[0:h, 0:w]
        expected_e = np.full((h, w), (self.background_rate + self.dark_current_e_per_s) * exposure, dtype=float)
        occupancy_for_frame = np.ones(self.n_sites, dtype=bool) if all_sites else self.occupancy.copy()
        next_occupancy = self.occupancy.copy()
        for site, ((cx, cy), occupied) in enumerate(zip(self._site_centers(), occupancy_for_frame)):
            if not occupied:
                continue
            if all_sites:
                signal_time = exposure
            else:
                lifetime = self.rng.exponential(self.detection_lifetime)
                signal_time = min(exposure, lifetime)
                if lifetime < exposure:
                    next_occupancy[site] = False
            amplitude = self.atom_rate * signal_time
            expected_e += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * self.atom_sigma_px**2))
        photoelectrons = self.rng.poisson(np.clip(expected_e, 0, None)).astype(float)
        noisy = photoelectrons / self.conversion_e_per_count + self.offset_counts
        if self.read_noise_e > 0:
            noisy += self.rng.normal(0.0, self.read_noise_e / self.conversion_e_per_count, size=noisy.shape)
        if not all_sites:
            self.occupancy = next_occupancy
        return np.clip(noisy, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "grid_shape": self.grid_shape,
            "image_shape": self.image_shape,
            "offset_counts": self.offset_counts,
            "conversion_e_per_count": self.conversion_e_per_count,
            "read_noise_e": self.read_noise_e,
        }

    def close(self) -> None:
        pass


class VirtualCamera(CameraDevice):
    def __init__(self, trap_array: VirtualTrapArray, exposure: float = 20e-3, timeout: float = 2.0):
        self.trap_array = trap_array
        self._exposure = positive_float(exposure, "exposure")
        self.timeout = positive_float(timeout, "timeout")
        self.last_sequence: str | None = None

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = positive_float(value, "exposure")

    def configure(self, *, exposure: float | None = None, **_) -> None:
        if exposure is not None:
            self.exposure = positive_float(exposure, "exposure")

    def acquire(self, frames: int = 1, *, sequence: PulseSequence | None = None, sequencer=None, **_) -> list[np.ndarray]:
        frames = positive_int(frames, "frames")
        runtime_sequence = sequence
        if sequencer is not None and sequence is not None:
            trigger_channels = getattr(sequencer, "trigger_channels", None)
            runtime_sequence = (
                sequence_for_frame_count(sequence, frames, trigger_channels=trigger_channels)
                if trigger_channels is not None
                else sequence_for_frame_count(sequence, frames)
            )
            sequencer.prepare(runtime_sequence)
            sequencer.fire(runtime_sequence)
        exposure = exposure_from_sequence(sequence, default=self.exposure)
        reload_each = sequence_requests_load(sequence)
        images: list[np.ndarray] = []
        all_sites = sequence is not None and sequence.name == "sitemap"
        for _ in range(frames):
            if reload_each:
                self.trap_array.reload()
            image = self.trap_array.render_image(exposure=exposure, all_sites=all_sites)
            images.append(image)
        if sequencer is not None and sequence is not None:
            wait_done = getattr(sequencer, "wait_done", None)
            if callable(wait_done) and not wait_done(max(self.timeout, getattr(runtime_sequence, "duration", 0.0) * 2.0 + 1.0)):
                raise TimeoutError("virtual sequencer did not report done.")
        self.last_sequence = None if sequence is None else sequence.name
        return images

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "timeout": self.timeout,
            "last_sequence": self.last_sequence,
        }

    def close(self) -> None:
        pass

    def stop(self) -> None:
        pass


class VirtualSequencer(SequencerDevice):
    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS, clock_hz: float = 250e6, sleep_scale: float = 0.0):
        self.channels = tuple(str(channel) for channel in channels)
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        self.history: list[dict[str, object]] = []
        self._prepared: PulseSequence | None = None

    def prepare(self, sequence: PulseSequence) -> None:
        sequence.validate(clock_hz=self.clock_hz, channels=self.channels).raise_if_failed()
        self._prepared = sequence
        self.history.append({"action": "prepare", "sequence": sequence.name, "duration": sequence.duration})

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self._prepared is None:
            raise RuntimeError("VirtualSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self._prepared:
            raise RuntimeError("VirtualSequencer.fire() received a sequence that was not prepared.")
        self.history.append({"action": "fire", "sequence": self._prepared.name, "duration": self._prepared.duration})
        if self.sleep_scale > 0:
            time.sleep(self._prepared.duration * self.sleep_scale)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "channels": list(self.channels),
            "clock_hz": self.clock_hz,
            "runs": sum(1 for row in self.history if row["action"] == "fire"),
        }

    def close(self) -> None:
        pass


def virtual_config() -> dict[str, object]:
    return {
        "trap_array": {"type": "VirtualTrapArray"},
        "camera": {"type": "VirtualCamera", "params": {"trap_array": "$device:trap_array"}},
        "sequencer": {"type": "VirtualSequencer"},
    }


def sequence_requests_load(sequence: PulseSequence | None) -> bool:
    if sequence is None:
        return False
    return any(pulse.channel in {"cooling", "mot", "load"} and pulse.value for pulse in sequence.effective_pulses())


def site_index(value, n_sites: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("site index must be an integer, not a boolean.")
    numeric = finite_float(value, "site index")
    if int(numeric) != numeric or numeric < 0 or numeric >= n_sites:
        raise ValueError(f"site index must be in [0, {n_sites}).")
    return int(numeric)


def point_tuple(value, name: str) -> tuple[float, float]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two finite numbers.") from exc
    if len(raw) != 2:
        raise ValueError(f"{name} must contain two finite numbers.")
    return finite_float(raw[0], f"{name}[0]"), finite_float(raw[1], f"{name}[1]")


def positive_float(value, name: str) -> float:
    out = finite_float(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be > 0.")
    return out


def nonnegative_float(value, name: str) -> float:
    out = finite_float(value, name)
    if out < 0:
        raise ValueError(f"{name} must be >= 0.")
    return out


def probability(value, name: str) -> float:
    out = finite_float(value, name)
    if out < 0 or out > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return out


__all__ = [
    "DEFAULT_CHANNELS",
    "VirtualCamera",
    "VirtualSequencer",
    "VirtualTrapArray",
    "virtual_config",
]
