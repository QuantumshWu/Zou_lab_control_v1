"""Small target-owned virtual apparatus used by the target composition root.

This is deliberately not a second public device framework.  It is the one
lowest-layer fake needed by the current finite pulse/camera vertical slice:
an immutable pulse target, a trigger wire, one bounded camera source, and the
atom-array physics observed through that source.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from zlc_neutral_atom.readout.sitemap import ReadoutGridGeometry
from zlc_pulse import CompiledPulseArtifact, PulsePlayback, PulseTarget


_K_B = 1.380649e-23
_RB87_MASS_KG = 86.909180527 * 1.66053906660e-27


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _channel_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of channel names")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty channel names")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique channel names")
    return result


@dataclass(frozen=True, eq=False)
class CameraFrameRecord:
    """One immutable virtual frame with the same metadata shape as capture input."""

    image: np.ndarray
    source_ordinal: int
    produced_count: int | None
    frame_stamp: int | None
    camera_stamp: int | None
    timestamp_seconds: int | None
    timestamp_microseconds: int | None
    host_received_at_ns: int
    driver_buffer_index: int | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_ordinal, bool)
            or not isinstance(self.source_ordinal, (int, np.integer))
            or int(self.source_ordinal) < 0
        ):
            raise ValueError("source_ordinal must be a non-negative integer")
        ordinal = int(self.source_ordinal)
        object.__setattr__(self, "source_ordinal", ordinal)
        for name in (
            "produced_count",
            "frame_stamp",
            "camera_stamp",
            "timestamp_seconds",
            "timestamp_microseconds",
            "driver_buffer_index",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer or None")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.host_received_at_ns, bool)
            or not isinstance(self.host_received_at_ns, (int, np.integer))
            or int(self.host_received_at_ns) <= 0
        ):
            raise ValueError("host_received_at_ns must be a positive integer")
        object.__setattr__(self, "host_received_at_ns", int(self.host_received_at_ns))
        image = np.array(self.image, copy=True, order="C")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)

    @property
    def captured_at(self) -> float:
        if self.timestamp_seconds is not None and self.timestamp_microseconds is not None:
            return float(self.timestamp_seconds) + float(self.timestamp_microseconds) * 1e-6
        return float(self.host_received_at_ns) * 1e-9


@dataclass(frozen=True)
class CameraCaptureTerminalRecord:
    produced_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.produced_count, bool)
            or not isinstance(self.produced_count, (int, np.integer))
            or int(self.produced_count) < 0
        ):
            raise ValueError("produced_count must be a non-negative integer")
        object.__setattr__(self, "produced_count", int(self.produced_count))
        if any(
            type(getattr(self, name)) is not bool
            for name in ("source_stopped", "no_more_frames", "joined")
        ):
            raise TypeError("terminal proof flags must be bool")


def _pulse_intervals(
    playback: PulsePlayback,
    channels: tuple[str, ...],
) -> list[tuple[float, float]]:
    selected = set(channels)
    return sorted(
        (pulse.start, pulse.stop)
        for pulse in playback.effective_pulses()
        if pulse.value and pulse.channel in selected
    )


def _trigger_starts(
    playback: PulsePlayback,
    channels: tuple[str, ...],
) -> list[float]:
    selected = set(channels)
    return sorted(
        pulse.start
        for pulse in playback.effective_pulses()
        if pulse.value and pulse.channel in selected
    )


def _overlap(intervals: Sequence[tuple[float, float]], lo: float, hi: float) -> float:
    return float(
        sum(
            max(0.0, min(stop, hi) - max(start, lo))
            for start, stop in intervals
            if start < hi and stop > lo
        )
    )


def _merged(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], stop))
        else:
            result.append((start, stop))
    return result


def _frame_timing(
    playback: PulsePlayback,
    frames: int,
    *,
    trigger_channels: tuple[str, ...],
    cooling_channels: tuple[str, ...],
    probe_channels: tuple[str, ...],
    trap_channels: tuple[str, ...],
    default_exposure: float,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    triggers = _trigger_starts(playback, trigger_channels)
    if len(triggers) != frames:
        raise RuntimeError(
            f"virtual trigger wire emitted {len(triggers)} edges for {frames} frames"
        )
    cooling = _merged(_pulse_intervals(playback, cooling_channels))
    probe = _pulse_intervals(playback, probe_channels)
    trap = _merged(_pulse_intervals(playback, trap_channels))
    camera_exposures = [float(default_exposure)] * frames
    probe_exposures = [0.0] * frames
    cooling_times = [0.0] * frames
    trap_off_times = [0.0] * frames
    trap_hold_times = [0.0] * frames
    for index, trigger in enumerate(triggers):
        before = 0.0 if index == 0 else triggers[index - 1]
        integration_stop = trigger + float(default_exposure)
        probe_exposures[index] = _overlap(probe, trigger, integration_stop)
        cooling_times[index] = _overlap(cooling, before, trigger)
        if index:
            cursor = before
            largest_gap = 0.0
            for start, stop in trap:
                if stop <= before or start >= trigger:
                    continue
                clipped_start = max(start, before)
                largest_gap = max(largest_gap, clipped_start - cursor)
                cursor = max(cursor, min(stop, trigger))
            trap_off_times[index] = max(largest_gap, trigger - cursor)
        trap_hold_times[index] = max(
            0.0,
            _overlap(trap, before, trigger) - _overlap(probe, before, trigger),
        )
    return (
        triggers,
        camera_exposures,
        probe_exposures,
        cooling_times,
        trap_off_times,
        trap_hold_times,
    )


class VirtualAtomArray:
    """Compact pulse-driven Rb87 array model retained from the deployed simulator."""

    def __init__(
        self,
        *,
        geometry: ReadoutGridGeometry,
        seed: int | None,
        cooling_channels: Sequence[str],
        probe_channels: Sequence[str],
        trap_channels: Sequence[str],
    ) -> None:
        if not isinstance(geometry, ReadoutGridGeometry):
            raise TypeError("geometry must be ReadoutGridGeometry")
        self.geometry = geometry
        self.grid_shape = geometry.grid_shape_yx
        self.image_shape = geometry.frame_shape_yx
        self.cooling_channels = _channel_tuple(cooling_channels, "cooling_channels")
        self.probe_channels = _channel_tuple(probe_channels, "probe_channels")
        self.trap_channels = _channel_tuple(trap_channels, "trap_channels")
        self.loading_probability = 0.5
        self.load_time_constant_s = 0.5e-3
        self.atom_rate = 1_100.0
        self.site_efficiency_sigma = 0.18
        self.background_rate = 300.0
        self.dark_current_e_per_s = 0.006
        self.offset_counts = 200.0
        self.conversion_e_per_count = 0.107
        self.read_noise_e = 0.43
        self.atom_sigma_px = 0.7
        self.atom_psf_aspect = 1.25
        self.atom_psf_angle_deg = 18.0
        self.atom_psf_skew = 0.45
        self.site_shape_sigma = 0.10
        self.detection_lifetime = 2.0
        self.trap_lifetime_s = 30.0
        self.cooled_temperature_K = 50e-6
        self.probe_heating_K_per_s = 0.0
        self.pgc_cool_tau_s = 0.3e-3
        self.capture_radius_m = 6.0e-6
        self.recapture_mass_kg = _RB87_MASS_KG
        self.rng = np.random.default_rng(seed)
        self._site_efficiency_cache: np.ndarray | None = None
        self._site_shape_cache: tuple[np.ndarray, np.ndarray] | None = None
        self.occupancy = np.zeros(self.n_sites, dtype=bool)
        self.temperature_K = np.full(self.n_sites, self.cooled_temperature_K)
        self.reload()

    @property
    def n_sites(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

    def _site_centers(self) -> np.ndarray:
        return self.geometry.expected_centers_xy

    def _site_efficiency(self) -> np.ndarray:
        if self._site_efficiency_cache is None:
            self._site_efficiency_cache = np.exp(
                self.rng.normal(0.0, self.site_efficiency_sigma, self.n_sites)
            )
        return self._site_efficiency_cache

    def _site_shapes(self) -> tuple[np.ndarray, np.ndarray]:
        if self._site_shape_cache is None:
            aspect = math.sqrt(self.atom_psf_aspect)
            sigma_x = self.atom_sigma_px / aspect
            sigma_y = self.atom_sigma_px * aspect
            self._site_shape_cache = (
                sigma_x
                * np.exp(self.rng.normal(0.0, self.site_shape_sigma, self.n_sites)),
                sigma_y
                * np.exp(self.rng.normal(0.0, self.site_shape_sigma, self.n_sites)),
            )
        return self._site_shape_cache

    def loading_fraction(self, duration: float | None) -> float:
        if duration is None:
            return self.loading_probability
        return self.loading_probability * (
            1.0 - math.exp(-max(0.0, float(duration)) / self.load_time_constant_s)
        )

    def reload(self, cooling_duration: float | None = None) -> None:
        self.occupancy = self.rng.random(self.n_sites) < self.loading_fraction(
            cooling_duration
        )
        self.temperature_K = np.full(self.n_sites, self.cooled_temperature_K)

    def cool(self, duration: float) -> None:
        if duration <= 0.0:
            return
        decay = math.exp(-duration / self.pgc_cool_tau_s)
        self.temperature_K = self.cooled_temperature_K + (
            self.temperature_K - self.cooled_temperature_K
        ) * decay

    def _render(self, occupancy: np.ndarray, signal_time: np.ndarray, exposure: float) -> np.ndarray:
        height, width = self.image_shape
        yy, xx = np.mgrid[0:height, 0:width]
        expected = np.full(
            (height, width),
            (self.background_rate + self.dark_current_e_per_s) * exposure,
            dtype=float,
        )
        aspect = math.sqrt(self.atom_psf_aspect)
        base_x = self.atom_sigma_px / aspect
        base_y = self.atom_sigma_px * aspect
        angle = math.radians(self.atom_psf_angle_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        efficiency = self._site_efficiency()
        site_x, site_y = self._site_shapes()
        for center, occupied, bright_time, gain, sigma_x, sigma_y in zip(
            self._site_centers(),
            occupancy,
            signal_time,
            efficiency,
            site_x,
            site_y,
        ):
            if not occupied or bright_time <= 0.0:
                continue
            dx = (xx - center[0]) * cosine + (yy - center[1]) * sine
            dy = -(xx - center[0]) * sine + (yy - center[1]) * cosine
            core = np.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
            spot = np.clip(
                core * (1.0 + self.atom_psf_skew * dx / sigma_x),
                0.0,
                None,
            )
            expected += (
                self.atom_rate
                * float(bright_time)
                * float(gain)
                * (base_x * base_y / (float(sigma_x) * float(sigma_y)))
                * spot
            )
        electrons = self.rng.poisson(np.clip(expected, 0.0, None)).astype(float)
        counts = electrons / self.conversion_e_per_count + self.offset_counts
        counts += self.rng.normal(
            0.0,
            self.read_noise_e / self.conversion_e_per_count,
            counts.shape,
        )
        return np.clip(counts, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def render_image(
        self,
        camera_exposure: float,
        probe_exposure: float,
    ) -> np.ndarray:
        camera_exposure = _positive(camera_exposure, "camera_exposure")
        if (
            isinstance(probe_exposure, bool)
            or not isinstance(probe_exposure, (int, float))
            or not math.isfinite(float(probe_exposure))
            or float(probe_exposure) < 0.0
        ):
            raise ValueError("probe_exposure must be finite and non-negative")
        probe_exposure = float(probe_exposure)
        if probe_exposure > camera_exposure and not math.isclose(
            probe_exposure,
            camera_exposure,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("probe exposure cannot exceed camera integration")
        probe_exposure = min(probe_exposure, camera_exposure)
        current = self.occupancy.copy()
        signal_time = np.zeros(self.n_sites)
        next_occupancy = current.copy()
        for index, occupied in enumerate(current):
            if occupied:
                lifetime = self.rng.exponential(self.detection_lifetime)
                signal_time[index] = min(probe_exposure, lifetime)
                if lifetime < probe_exposure:
                    next_occupancy[index] = False
        image = self._render(current, signal_time, camera_exposure)
        self.occupancy = next_occupancy
        if self.probe_heating_K_per_s > 0.0:
            self.temperature_K += self.probe_heating_K_per_s * probe_exposure
        return image

    def apply_recapture_loss(self, duration: float) -> None:
        if duration <= 0.0:
            return
        velocity = np.sqrt(
            _K_B * np.maximum(self.temperature_K, 1e-12) / self.recapture_mass_kg
        )
        argument = self.capture_radius_m / (
            math.sqrt(2.0) * velocity * duration
        )
        survival = np.fromiter(
            (math.erf(float(value)) ** 3 for value in argument),
            dtype=float,
            count=self.n_sites,
        )
        self.occupancy &= self.rng.random(self.n_sites) < survival

    def apply_trap_loss(self, duration: float) -> None:
        if duration <= 0.0:
            return
        survival = math.exp(-duration / self.trap_lifetime_s)
        self.occupancy &= self.rng.random(self.n_sites) < survival

    def iter_frames(
        self,
        playback: PulsePlayback,
        frames: int,
        *,
        trigger_channels: tuple[str, ...],
        default_exposure: float,
    ) -> Iterator[np.ndarray]:
        (
            _triggers,
            camera_exposures,
            probe_exposures,
            cooling,
            trap_off,
            trap_hold,
        ) = _frame_timing(
            playback,
            frames,
            trigger_channels=trigger_channels,
            cooling_channels=self.cooling_channels,
            probe_channels=self.probe_channels,
            trap_channels=self.trap_channels,
            default_exposure=default_exposure,
        )
        groups = playback.trigger_group_sizes(trigger_channels)
        if sum(groups) != frames:
            raise RuntimeError("pulse trigger groups do not exactly cover the camera arm")
        group_starts: set[int] = set()
        cursor = 0
        for size in groups:
            group_starts.add(cursor)
            cursor += size
        for index in range(frames):
            group_start = index in group_starts
            if group_start:
                self.reload(cooling[index] if cooling[index] > 0.0 else None)
            elif cooling[index] > 0.0:
                self.cool(cooling[index])
            self.apply_recapture_loss(trap_off[index])
            if not group_start:
                self.apply_trap_loss(trap_hold[index])
            yield self.render_image(
                camera_exposures[index],
                probe_exposures[index],
            )

class VirtualSequencer:
    """Finite target-IR player and the sole owner of the virtual trigger wire."""

    def __init__(
        self,
        target: PulseTarget,
        *,
        clock_hz: float,
        sleep_scale: float = 1.0,
    ) -> None:
        if not isinstance(target, PulseTarget):
            raise TypeError("virtual sequencer target must be PulseTarget")
        self.target = target
        self.clock_hz = _positive(clock_hz, "clock_hz")
        if isinstance(sleep_scale, bool) or not isinstance(sleep_scale, (int, float)):
            raise TypeError("sleep_scale must be a number")
        self.sleep_scale = float(sleep_scale)
        if not math.isfinite(self.sleep_scale) or self.sleep_scale < 0.0:
            raise ValueError("sleep_scale must be finite and non-negative")
        self._lock = threading.RLock()
        self._listeners: list[object] = []
        self._prepared: PulsePlayback | None = None
        self._artifact_digest: str | None = None
        self._firing: PulsePlayback | None = None
        self._last_fired: PulsePlayback | None = None
        self._logical_deadline: float | None = None
        self._state = "safe"
        self._open = True

    @property
    def firing(self) -> PulsePlayback | None:
        with self._lock:
            return self._firing

    @property
    def last_fired(self) -> PulsePlayback | None:
        with self._lock:
            return self._last_fired

    def ensure_open(self) -> "VirtualSequencer":
        if not self._open:
            raise RuntimeError("virtual sequencer is closed")
        return self

    def add_fire_listener(self, listener: object) -> None:
        if not callable(listener):
            raise TypeError("fire listener must be callable")
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_fire_listener(self, listener: object) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def prepare_compiled_playback(
        self,
        artifact: CompiledPulseArtifact,
        playback: PulsePlayback,
    ):
        self.ensure_open()
        if not isinstance(artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        if not isinstance(playback, PulsePlayback):
            raise TypeError("playback must be PulsePlayback")
        if artifact.target_abi_fingerprint != self.target.abi_fingerprint:
            raise ValueError("compiled pulse target differs from virtual sequencer")
        with self._lock:
            self._prepared = playback
            self._artifact_digest = artifact.fingerprint
            self._firing = None
            self._last_fired = None
            self._logical_deadline = None
            self._state = "prepared"
        return artifact.target_ir

    def fire_compiled_playback(self, artifact_digest: str) -> None:
        with self._lock:
            if self._prepared is None or artifact_digest != self._artifact_digest:
                raise RuntimeError("compiled FIRE does not match the prepared artifact")
            playback = self._prepared
            self._last_fired = playback
            self._firing = playback if playback.repeat_forever else None
            self._logical_deadline = (
                None
                if playback.repeat_forever
                else time.monotonic() + playback.logical_duration * self.sleep_scale
            )
            self._state = "running"
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(playback)

    def wait_compiled_playback(self, artifact_digest: str, timeout: float | None) -> bool:
        with self._lock:
            if self._prepared is None or artifact_digest != self._artifact_digest:
                raise RuntimeError("compiled wait does not match the prepared artifact")
            if self._prepared.repeat_forever:
                raise RuntimeError("cannot wait for cyclic virtual playback")
            deadline = self._logical_deadline
        if deadline is None:
            raise RuntimeError("compiled wait requires FIRE first")
        remaining = max(0.0, deadline - time.monotonic())
        if timeout is not None and remaining > float(timeout):
            return False
        if remaining:
            time.sleep(remaining)
        with self._lock:
            self._state = "done"
        return True

    def set_safe_state(self) -> None:
        with self._lock:
            self._prepared = None
            self._artifact_digest = None
            self._firing = None
            self._last_fired = None
            self._logical_deadline = None
            self._state = "safe"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "type": type(self).__name__,
                "state": self._state,
                "target_abi_fingerprint": self.target.abi_fingerprint,
                "clock_hz": self.clock_hz,
                "sleep_scale": self.sleep_scale,
                "prepared_program": (
                    None if self._prepared is None else self._prepared.name
                ),
            }

    def close(self) -> None:
        self.set_safe_state()
        with self._lock:
            self._listeners.clear()
            self._open = False


class VirtualCamera:
    """Finite externally-triggered camera with one bounded producer queue."""

    recent_capacity = 16

    def __init__(
        self,
        atoms: VirtualAtomArray,
        sequencer: VirtualSequencer,
        *,
        capture_trigger_channels: Sequence[str],
        exposure: float = 20e-3,
        timeout: float = 2.0,
    ) -> None:
        if not isinstance(atoms, VirtualAtomArray):
            raise TypeError("virtual camera requires VirtualAtomArray")
        if not isinstance(sequencer, VirtualSequencer):
            raise TypeError("virtual camera requires VirtualSequencer")
        self.atoms = atoms
        self.sequencer = sequencer
        self.capture_trigger_channels = _channel_tuple(
            capture_trigger_channels,
            "capture_trigger_channels",
        )
        self.exposure = _positive(exposure, "exposure")
        self.timeout = _positive(timeout, "timeout")
        self.roi = None
        self._condition = threading.Condition(threading.RLock())
        self._pending: deque[CameraFrameRecord] = deque()
        self._armed = False
        self._accepting = False
        self._expected = 0
        self._capacity = self.recent_capacity
        self._produced = 0
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._worker_error: BaseException | None = None
        self._terminal: CameraCaptureTerminalRecord | None = None
        self._open = True
        sequencer.add_fire_listener(self._on_fire)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.atoms.image_shape

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return self.atoms.image_shape

    @property
    def effective_trigger_channels(self) -> tuple[str, ...]:
        return self.capture_trigger_channels

    def ensure_open(self) -> "VirtualCamera":
        if not self._open:
            raise RuntimeError("virtual camera is closed")
        return self

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self.roi,
            "timeout": self.timeout,
        }

    def arm(
        self,
        frames: int,
        *,
        max_inflight_frames: int | None = None,
        timeout: float | None = None,
        stop: object | None = None,
    ) -> None:
        del timeout, stop
        self.ensure_open()
        expected = _positive_int(frames, "frames")
        capacity = (
            expected
            if max_inflight_frames is None
            else _positive_int(max_inflight_frames, "max_inflight_frames")
        )
        if capacity > expected:
            raise ValueError("max_inflight_frames cannot exceed frame budget")
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual camera already owns an armed capture")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous virtual camera producer is still running")
            self._pending.clear()
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._capacity = capacity
            self._produced = 0
            self._worker = None
            self._worker_stop = None
            self._worker_error = None
            self._terminal = None

    def _on_fire(self, playback: PulsePlayback) -> None:
        if playback.repeat_forever:
            return
        with self._condition:
            if not self._armed or not self._accepting:
                return
            if self._worker is not None:
                raise RuntimeError("virtual camera received overlapping finite FIRE")
            trigger_offsets = _trigger_starts(
                playback,
                self.capture_trigger_channels,
            )
            if len(trigger_offsets) != self._expected:
                self._worker_error = RuntimeError(
                    f"virtual pulse emitted {len(trigger_offsets)} camera edges for "
                    f"an arm budget of {self._expected}"
                )
                self._condition.notify_all()
                return
            stop = threading.Event()
            self._worker_stop = stop
            fired_at = time.monotonic()

            def produce() -> None:
                try:
                    frames = self.atoms.iter_frames(
                        playback,
                        self._expected,
                        trigger_channels=self.capture_trigger_channels,
                        default_exposure=self.exposure,
                    )
                    frame_iterator = iter(frames)
                    for ordinal, offset in enumerate(trigger_offsets):
                        # A frame becomes available only after its fixed sensor
                        # integration.  Waiting merely for the trigger edge made
                        # persisted host-receipt timing precede the exposure it
                        # claimed to contain, and advancing the generator before
                        # that wait mutated atom state too early.
                        deadline = fired_at + (
                            offset + self.exposure
                        ) * self.sequencer.sleep_scale
                        while not stop.is_set():
                            remaining = deadline - time.monotonic()
                            if remaining <= 0.0:
                                break
                            stop.wait(remaining)
                        if stop.is_set():
                            return
                        try:
                            image = next(frame_iterator)
                        except StopIteration as exc:
                            raise RuntimeError(
                                "virtual atom source ended before the trigger budget"
                            ) from exc
                        record = CameraFrameRecord(
                            image,
                            ordinal,
                            ordinal + 1,
                            None,
                            None,
                            None,
                            None,
                            time.time_ns(),
                        )
                        with self._condition:
                            if not self._armed or not self._accepting or stop.is_set():
                                return
                            if len(self._pending) >= self._capacity:
                                raise RuntimeError(
                                    "virtual camera retention exhausted before host drain"
                                )
                            self._pending.append(record)
                            self._produced = ordinal + 1
                            self._condition.notify_all()
                    sentinel = object()
                    if next(frame_iterator, sentinel) is not sentinel:
                        raise RuntimeError(
                            "virtual atom source exceeded the trigger budget"
                        )
                except BaseException as error:
                    with self._condition:
                        self._worker_error = error
                        self._condition.notify_all()
                finally:
                    with self._condition:
                        self._condition.notify_all()

            worker = threading.Thread(
                target=produce,
                name="zlc-target-virtual-camera",
                daemon=False,
            )
            self._worker = worker
            try:
                worker.start()
            except BaseException as error:
                self._worker = None
                self._worker_stop = None
                self._worker_error = error
                self._condition.notify_all()
                raise

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float | None = None,
        stop: object | None = None,
        exact: bool = False,
    ) -> list[CameraFrameRecord]:
        del exact
        wanted = _positive_int(n, "n")
        deadline = time.monotonic() + (
            self.timeout if timeout is None else max(0.0, float(timeout))
        )
        result: list[CameraFrameRecord] = []
        with self._condition:
            if not self._armed:
                raise RuntimeError("read_frame_records requires an armed camera")
            while len(result) < wanted:
                while self._pending and len(result) < wanted:
                    result.append(self._pending.popleft())
                    self._condition.notify_all()
                if len(result) == wanted:
                    break
                if self._worker_error is not None:
                    raise RuntimeError("virtual camera source failed") from self._worker_error
                if stop is not None and getattr(stop, "is_set", lambda: False)():
                    break
                if self._worker is not None and not self._worker.is_alive():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(min(0.05, remaining))
        if len(result) != wanted:
            raise TimeoutError(
                f"virtual camera returned {len(result)} of {wanted} exact frame(s)"
            )
        return result

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            worker = self._worker
            stop = self._worker_stop
            self._accepting = False
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(self.timeout)
            if worker.is_alive():
                raise TimeoutError("virtual camera producer did not join")
        with self._condition:
            terminal = CameraCaptureTerminalRecord(
                self._produced,
                True,
                True,
                worker is None or not worker.is_alive(),
            )
            self._armed = False
            self._pending.clear()
            self._terminal = terminal
            self._condition.notify_all()
            return terminal

    def capture_state(self) -> tuple[bool, int]:
        with self._condition:
            return self._armed, len(self._pending)

    def close(self) -> None:
        try:
            with self._condition:
                armed = self._armed
            if armed:
                self.finish_record_capture()
        finally:
            self.sequencer.remove_fire_listener(self._on_fire)
            self._open = False


__all__: list[str] = []
