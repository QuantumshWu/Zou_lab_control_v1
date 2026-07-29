"""Small target-owned virtual apparatus used by the target composition root.

This is deliberately not a second public device framework.  It is the one
lowest-layer fake needed by the current pulse/camera vertical slices: an
immutable pulse target, a trigger wire, bounded exact and free-running camera
sources, and the atom-array physics observed through the exact source.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Mapping, NoReturn, Sequence

import numpy as np

from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
    validate_camera_external_trigger_spacing,
)
from zlc_neutral_atom.runtime.signal_source import SignalAssociationRequest
from zlc_pulse import (
    CompiledPulseArtifact,
    PulsePlayback,
    PulseTarget,
    sample_compiled_bus_codes,
)
from zlc_storage import canonical_digest, canonical_text, sha256_text
from zlc_storage.canonical import positive_integer as _positive_int


_K_B = 1.380649e-23
_RB87_MASS_KG = 86.909180527 * 1.66053906660e-27
_GREY_MOLASSES_HOT_FACTOR = 6.0


def grey_molasses_cooling_factor(detuning_gamma: float) -> float:
    """Virtual D1 grey-molasses floor at the default saturation of 3."""

    detuning = float(detuning_gamma)
    if not math.isfinite(detuning):
        raise ValueError("two-photon detuning must be finite")
    half_width = 0.05 + 0.02 * 3.0
    wing = 3.0 if detuning >= 0.0 else 0.3
    return float(
        min(
            _GREY_MOLASSES_HOT_FACTOR,
            max(1.0, 1.0 + wing * (detuning / half_width) ** 2),
        )
    )


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
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
        frame_shape_yx: tuple[int, int],
        grid_shape_yx: tuple[int, int],
        site_centers_xy: Sequence[tuple[float, float]],
        seed: int | None,
        cooling_channels: Sequence[str],
        probe_channels: Sequence[str],
        trap_channels: Sequence[str],
        rf: "VirtualRfSource | None" = None,
    ) -> None:
        self.image_shape = tuple(
            _positive_int(size, "virtual frame dimension")
            for size in frame_shape_yx
        )
        self.grid_shape = tuple(
            _positive_int(size, "virtual grid dimension")
            for size in grid_shape_yx
        )
        if len(self.image_shape) != 2 or len(self.grid_shape) != 2:
            raise ValueError("virtual frame and grid shapes must be two-dimensional")
        centers = np.asarray(tuple(site_centers_xy), dtype=np.float64)
        expected_shape = (self.grid_shape[0] * self.grid_shape[1], 2)
        if centers.shape != expected_shape or not np.all(np.isfinite(centers)):
            raise ValueError(
                "site_centers_xy must contain one finite (x, y) pair per site"
            )
        self._site_centers_xy = centers.copy()
        self._site_centers_xy.setflags(write=False)
        self.cooling_channels = _channel_tuple(cooling_channels, "cooling_channels")
        self.probe_channels = _channel_tuple(probe_channels, "probe_channels")
        self.trap_channels = _channel_tuple(trap_channels, "trap_channels")
        if rf is not None and not isinstance(rf, VirtualRfSource):
            raise TypeError("rf must be VirtualRfSource or None")
        self.rf = rf
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
        self._cooling_floor_factor = 1.0
        self.reload()

    @property
    def n_sites(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

    @property
    def frame_dtype(self) -> np.dtype:
        return np.dtype("<u2")

    def _site_centers(self) -> np.ndarray:
        return self._site_centers_xy

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
        self.temperature_K = np.full(
            self.n_sites,
            self.cooled_temperature_K * self._cooling_floor_factor,
        )

    def cool(self, duration: float) -> None:
        if duration <= 0.0:
            return
        decay = math.exp(-duration / self.pgc_cool_tau_s)
        floor = self.cooled_temperature_K * self._cooling_floor_factor
        self.temperature_K = floor + (
            self.temperature_K - floor
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
        trigger_group_sizes: tuple[int, ...],
        trigger_group_point_indices: tuple[int | None, ...] | None = None,
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
        groups = tuple(
            _positive_int(size, "trigger group size")
            for size in trigger_group_sizes
        )
        if sum(groups) != frames:
            raise RuntimeError("pulse trigger groups do not exactly cover the camera arm")
        group_points: tuple[int | None, ...]
        if trigger_group_point_indices is not None:
            group_points = tuple(trigger_group_point_indices)
            if len(group_points) != len(groups):
                raise RuntimeError(
                    "pulse point coordinates differ from frozen camera groups"
                )
        else:
            group_points = tuple(None for _group in groups)
        group_starts: set[int] = set()
        point_by_start: dict[int, int | None] = {}
        cursor = 0
        for size, point_index in zip(groups, group_points):
            group_starts.add(cursor)
            point_by_start[cursor] = point_index
            cursor += size
        for index in range(frames):
            group_start = index in group_starts
            if group_start:
                point_index = point_by_start[index]
                self._cooling_floor_factor = (
                    1.0
                    if self.rf is None
                    else self.rf.cooling_factor_for_point(playback, point_index)
                )
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
        self._prepared_artifact: CompiledPulseArtifact | None = None
        self._artifact_digest: str | None = None
        self._firing: PulsePlayback | None = None
        self._last_fired: PulsePlayback | None = None
        self._firing_artifact: CompiledPulseArtifact | None = None
        self._last_fired_artifact: CompiledPulseArtifact | None = None
        self._fire_started_monotonic: float | None = None
        self._fire_generation = 0
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

    @property
    def output_artifact(self) -> CompiledPulseArtifact | None:
        """Compiled program currently driving, or last latched on, the outputs."""

        with self._lock:
            return self._firing_artifact or self._last_fired_artifact

    def active_fire(
        self,
        playback: PulsePlayback,
    ) -> tuple[int, float] | None:
        """Return the exact running FIRE generation and its hardware epoch."""

        if not isinstance(playback, PulsePlayback):
            raise TypeError("playback must be PulsePlayback")
        with self._lock:
            if self._firing is not playback or self._fire_started_monotonic is None:
                return None
            return self._fire_generation, self._fire_started_monotonic

    def is_active_fire(self, playback: PulsePlayback, generation: int) -> bool:
        with self._lock:
            return (
                self._firing is playback
                and self._fire_generation == generation
            )

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
            self._prepared_artifact = artifact
            self._artifact_digest = artifact.fingerprint
            self._firing = None
            self._last_fired = None
            self._firing_artifact = None
            self._last_fired_artifact = None
            self._fire_started_monotonic = None
            self._logical_deadline = None
            self._state = "prepared"
        return artifact.target_ir

    def fire_compiled_playback(self, artifact_digest: str) -> None:
        with self._lock:
            if self._prepared is None or artifact_digest != self._artifact_digest:
                raise RuntimeError("compiled FIRE does not match the prepared artifact")
            playback = self._prepared
            artifact = self._prepared_artifact
            if artifact is None:
                raise RuntimeError("compiled FIRE has no prepared artifact")
            self._last_fired = playback
            self._firing = playback if playback.repeat_forever else None
            self._last_fired_artifact = artifact
            self._firing_artifact = artifact if playback.repeat_forever else None
            self._fire_generation += 1
            self._fire_started_monotonic = time.monotonic()
            self._logical_deadline = (
                None
                if playback.repeat_forever
                else self._fire_started_monotonic
                + playback.logical_duration * self.sleep_scale
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

    def observe_scan_cursor(
        self,
        artifact_digest: str,
    ) -> tuple[int | None, str, str | None]:
        """Return the simulated hardware cursor, or an explicit unavailability."""

        with self._lock:
            state = self._state.upper()
            artifact = self._last_fired_artifact
            started = self._fire_started_monotonic
            if artifact_digest != self._artifact_digest or artifact is None:
                return None, state, "virtual sequencer does not own this artifact"
            if self._state != "running":
                return None, state, "virtual sequencer is not running"
            durations = tuple(artifact.target_ir.scan_point_durations)
            if not durations:
                return None, state, "compiled pulse has no scan table"
            if started is None:
                return None, state, "virtual FIRE timestamp is unavailable"
            scale = self.sleep_scale
        if scale <= 0.0:
            return (
                None,
                state,
                "zero-time virtual playback has no meaningful realtime cursor",
            )
        cycle_seconds = sum(durations)
        if not math.isfinite(cycle_seconds) or cycle_seconds <= 0.0:
            return None, state, "compiled scan duration is invalid"
        elapsed_logical = max(0.0, (time.monotonic() - started) / scale)
        phase = elapsed_logical % cycle_seconds
        boundary = 0.0
        for point_index, duration in enumerate(durations):
            boundary += duration
            if phase < boundary:
                return point_index, state, None
        return len(durations) - 1, state, None

    def set_safe_state(self) -> None:
        with self._lock:
            self._prepared = None
            self._prepared_artifact = None
            self._artifact_digest = None
            self._firing = None
            self._last_fired = None
            self._firing_artifact = None
            self._last_fired_artifact = None
            self._fire_started_monotonic = None
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


class VirtualRfSource:
    """Preloaded RF table advanced only by the virtual sequencer scan clock."""

    def __init__(self, sequencer: VirtualSequencer) -> None:
        if not isinstance(sequencer, VirtualSequencer):
            raise TypeError("virtual RF requires VirtualSequencer")
        self.sequencer = sequencer
        self._lock = threading.RLock()
        self._open = True
        self._session_id: str | None = None
        self._pulse_artifact_digest: str | None = None
        self._table_digest: str | None = None
        self._table: tuple[float, ...] = ()

    def ensure_open(self) -> "VirtualRfSource":
        if not self._open:
            raise RuntimeError("virtual RF is closed")
        return self

    def prepare_table(
        self,
        session_id: str,
        pulse_artifact_digest: str,
        table_digest: str,
        values: tuple[float, ...],
    ) -> None:
        self.ensure_open()
        table = tuple(float(value) for value in values)
        if not table or any(not math.isfinite(value) for value in table):
            raise ValueError("virtual RF table must contain finite values")
        with self._lock:
            if self._session_id is not None:
                raise RuntimeError("virtual RF already owns a prepared table")
            self._session_id = str(session_id)
            self._pulse_artifact_digest = str(pulse_artifact_digest)
            self._table_digest = str(table_digest)
            self._table = table

    def cooling_factor_for_point(
        self,
        playback: PulsePlayback,
        point_index: int | None,
    ) -> float:
        with self._lock:
            if self._session_id is None:
                return 1.0
            artifact = self.sequencer.output_artifact
            if (
                playback is not self.sequencer.last_fired
                or artifact is None
                or artifact.fingerprint != self._pulse_artifact_digest
            ):
                raise RuntimeError("virtual RF observed another pulse playback")
            if point_index is None:
                raise RuntimeError("virtual RF scan point is unavailable")
            if point_index >= len(self._table):
                raise RuntimeError("virtual RF scan clock exceeded its table")
            return grey_molasses_cooling_factor(self._table[point_index])

    def complete_table(
        self,
        session_id: str,
        table_digest: str,
    ) -> tuple[int, str]:
        with self._lock:
            if (
                self._session_id != session_id
                or self._table_digest != table_digest
            ):
                raise RuntimeError("virtual RF terminal belongs to another prepared table")
            playback = self.sequencer.last_fired
            artifact = self.sequencer.output_artifact
            if (
                playback is None
                or artifact is None
                or artifact.fingerprint != self._pulse_artifact_digest
            ):
                raise RuntimeError("virtual RF terminal has no matching sequencer FIRE")
            point_indices = tuple(
                group.point_index for group in playback.trigger_groups
            )
            if point_indices != tuple(range(len(self._table))):
                raise RuntimeError(
                    "virtual RF table does not exactly cover completed scan points"
                )
            return len(point_indices), canonical_digest(
                {
                    "pulse_artifact_digest": self._pulse_artifact_digest,
                    "table_digest": table_digest,
                    "advanced_points": point_indices,
                }
            )

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            if self._session_id not in (None, session_id):
                raise RuntimeError("virtual RF close belongs to another session")
            self._session_id = None
            self._pulse_artifact_digest = None
            self._table_digest = None
            self._table = ()
            return True

    def set_safe_state(self) -> None:
        with self._lock:
            session_id = self._session_id
        if session_id is not None:
            self.close_session(session_id)

    def close(self) -> None:
        self.set_safe_state()
        with self._lock:
            self._open = False


class VirtualMotFrameSource:
    """MOT fluorescence computed from the compiled DAC point being played."""

    def __init__(
        self,
        sequencer: VirtualSequencer,
        *,
        frame_shape: tuple[int, int] = (1200, 1920),
        seed: int | None = 19,
        coil_ports: Mapping[str, str] | None = None,
        optimum_levels: Mapping[str, float] | None = None,
        level_sigmas: Mapping[str, float] | None = None,
        peak_counts: float = 93.0,
        offset_counts: float = 7.0,
        read_noise: float = 1.5,
        spot_size_px: tuple[float, float] = (40.0, 20.0),
    ) -> None:
        if not isinstance(sequencer, VirtualSequencer):
            raise TypeError("virtual MOT frame source requires VirtualSequencer")
        if not isinstance(frame_shape, tuple) or len(frame_shape) != 2:
            raise TypeError("MOT frame_shape must be a (height, width) tuple")
        self.sequencer = sequencer
        self.image_shape = tuple(
            _positive_int(size, "MOT frame dimension") for size in frame_shape
        )
        self.coil_ports = {
            str(name): str(port)
            for name, port in dict(
                coil_ports
                or {
                    "da_x": "da_bias_x",
                    "da_y": "da_bias_y",
                    "da_z": "da_bias_z",
                }
            ).items()
        }
        if not self.coil_ports or any(
            not name or not port for name, port in self.coil_ports.items()
        ):
            raise ValueError("MOT coil_ports must map non-empty names to ports")
        target_ports = sequencer.target.by_key
        for port in self.coil_ports.values():
            spec = target_ports.get(port)
            if spec is None or spec.signed_range is None:
                raise ValueError(f"MOT coil port {port!r} is not a target DAC")
        self.optimum_levels = {
            str(name): float(value)
            for name, value in dict(
                optimum_levels or {"da_x": 7.0, "da_y": -5.0, "da_z": 11.0}
            ).items()
        }
        self.level_sigmas = {
            str(name): _positive(value, f"MOT level sigma {name!r}")
            for name, value in dict(
                level_sigmas or {"da_x": 6.0, "da_y": 6.0, "da_z": 6.0}
            ).items()
        }
        if (
            set(self.optimum_levels) != set(self.coil_ports)
            or set(self.level_sigmas) != set(self.coil_ports)
        ):
            raise ValueError("MOT coil, optimum, and sigma axes must match")
        self.peak_counts = _nonnegative(peak_counts, "MOT peak_counts")
        self.offset_counts = _nonnegative(offset_counts, "MOT offset_counts")
        self.read_noise = _nonnegative(read_noise, "MOT read_noise")
        if not isinstance(spot_size_px, tuple) or len(spot_size_px) != 2:
            raise TypeError("MOT spot_size_px must be a two-item tuple")
        self.spot_size_px = (
            _positive(spot_size_px[0], "MOT spot width"),
            _positive(spot_size_px[1], "MOT spot height"),
        )
        self.last_levels: dict[str, float] | None = None
        self._rng = np.random.default_rng(seed)

    @property
    def frame_dtype(self) -> np.dtype:
        return np.dtype("<u1")

    def snapshot(self) -> dict[str, object]:
        return {
            "frame_shape": self.image_shape,
            "coil_ports": dict(self.coil_ports),
            "optimum_levels": dict(self.optimum_levels),
            "level_sigmas": dict(self.level_sigmas),
            "peak_counts": self.peak_counts,
            "offset_counts": self.offset_counts,
            "read_noise": self.read_noise,
            "spot_size_px": self.spot_size_px,
        }

    def mot_efficiency(self, levels: Mapping[str, float]) -> float:
        distance_squared = sum(
            (
                (float(levels.get(name, 0.0)) - self.optimum_levels[name])
                / self.level_sigmas[name]
            )
            ** 2
            for name in self.coil_ports
        )
        return float(math.exp(-0.5 * distance_squared))

    def levels_for_point(
        self,
        artifact: CompiledPulseArtifact,
        point_index: int,
    ) -> dict[str, float]:
        codes = dict(
            sample_compiled_bus_codes(
                artifact,
                point_index=point_index,
                phase=0.5,
            )
        )
        target_ports = self.sequencer.target.by_key
        levels: dict[str, float] = {}
        for name, port in self.coil_ports.items():
            spec = target_ports[port]
            assert spec.signed_range is not None
            levels[name] = float(
                codes.get(port, spec.safe_value) + spec.signed_range[0]
            )
        return levels

    def _render_levels(self, levels: Mapping[str, float]) -> np.ndarray:
        self.last_levels = dict(levels)
        efficiency = self.mot_efficiency(levels)
        height, width = self.image_shape
        center_x, center_y = width / 2.0, height / 2.0
        fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0))
        sigma_x = self.spot_size_px[0] / fwhm
        sigma_y = self.spot_size_px[1] / fwhm
        x0 = max(0, int(center_x - 3.0 * sigma_x))
        x1 = min(width, int(math.ceil(center_x + 3.0 * sigma_x)))
        y0 = max(0, int(center_y - 3.0 * sigma_y))
        y1 = min(height, int(math.ceil(center_y + 3.0 * sigma_y)))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        spot = np.exp(
            -0.5
            * (
                ((xx - center_x) / sigma_x) ** 2
                + ((yy - center_y) / sigma_y) ** 2
            )
        )
        noise = self._rng.normal(
            self.offset_counts,
            self.read_noise,
            size=(height, width),
        )
        frame = np.clip(noise, 0, 255).astype(np.uint8)
        signal = self._rng.poisson(
            self.peak_counts * efficiency * spot
        ).astype(np.int32, copy=False)
        region = frame[y0:y1, x0:x1].astype(np.int32)
        region += signal
        np.clip(region, 0, 255, out=region)
        frame[y0:y1, x0:x1] = region
        return frame

    def render_current_output(self) -> np.ndarray:
        artifact = self.sequencer.output_artifact
        levels = (
            {name: 0.0 for name in self.coil_ports}
            if artifact is None
            else self.levels_for_point(artifact, 0)
        )
        if artifact is not None and len(artifact.target_ir.scan_points or ((),)) != 1:
            raise RuntimeError(
                "free-running monitor frames have no declared association with "
                "multi-point sequencer output"
            )
        return self._render_levels(levels)

    def iter_frames(
        self,
        playback: PulsePlayback,
        frames: int,
        *,
        trigger_channels: tuple[str, ...],
        default_exposure: float,
        trigger_group_sizes: tuple[int, ...],
        trigger_group_point_indices: tuple[int | None, ...] | None = None,
    ) -> Iterator[np.ndarray]:
        del playback, default_exposure
        if len(trigger_channels) != 1:
            raise ValueError(
                "virtual MOT point association requires exactly one trigger channel"
            )
        artifact = self.sequencer.output_artifact
        if artifact is None:
            raise RuntimeError("virtual MOT camera observed FIRE without an artifact")
        groups = tuple(
            _positive_int(size, "trigger group size")
            for size in trigger_group_sizes
        )
        if sum(groups) != frames:
            raise RuntimeError(
                "frozen trigger groups do not exactly cover MOT frames"
            )
        if trigger_group_point_indices is None:
            raise RuntimeError(
                "virtual MOT frames require the camera adapter's frozen point association"
            )
        group_points = tuple(trigger_group_point_indices)
        if len(group_points) != len(groups):
            raise RuntimeError(
                "frozen MOT point coordinates differ from frozen camera groups"
            )
        if any(point_index is None for point_index in group_points):
            raise RuntimeError(
                "frozen MOT camera groups are missing scan-point association"
            )
        point_indices = tuple(
            point_index
            for size, point_index in zip(groups, group_points)
            for _frame in range(size)
        )
        for point_index in point_indices:
            yield self._render_levels(
                self.levels_for_point(artifact, point_index)
            )


@dataclass
class _VirtualCameraSignalAssociation:
    """One pre-FIRE reservation owned by the virtual trigger wire."""

    association_id: str
    cause_digest: str
    trigger_schedule_fingerprint: str
    expected_trigger_count: int
    trigger_group_size: int
    expected_group_count: int
    physical_start_ordinal: int
    physical_end_ordinal: int | None = None
    terminal_evidence_digest: str | None = None
    error: BaseException | None = None


class VirtualCamera:
    """One installed camera supporting live and finite acquisition.

    A finite arm observes the sequencer's hardware-timed trigger edges.  A
    ``None`` arm uses the camera's own exposure cadence for the live view.
    These modes are mutually exclusive because they are two operations on the
    same physical camera, not two public camera roles.
    """

    def __init__(
        self,
        frame_source: VirtualAtomArray | VirtualMotFrameSource,
        sequencer: VirtualSequencer,
        *,
        capture_trigger_channels: Sequence[str],
        exposure: float = 20e-3,
        timeout: float = 2.0,
        free_running_live: bool = False,
    ) -> None:
        if not isinstance(frame_source, (VirtualAtomArray, VirtualMotFrameSource)):
            raise TypeError("virtual camera requires a supported frame source")
        if not isinstance(sequencer, VirtualSequencer):
            raise TypeError("virtual camera requires VirtualSequencer")
        self.frame_source = frame_source
        self.sequencer = sequencer
        self.capture_trigger_channels = _channel_tuple(
            capture_trigger_channels,
            "capture_trigger_channels",
        )
        self.exposure = _positive(exposure, "exposure")
        self.timeout = _positive(timeout, "timeout")
        if not isinstance(free_running_live, bool):
            raise TypeError("free_running_live must be bool")
        self.free_running_live = free_running_live
        self.roi = None
        self._condition = threading.Condition(threading.RLock())
        self._pending: deque[CameraFrameRecord] = deque()
        self._armed = False
        self._armed_at_monotonic: float | None = None
        self._accepting = False
        self._expected: int | None = 0
        self._source_group_sizes: tuple[int, ...] | None = None
        self._source_group_cursor = 0
        self._produced = 0
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._continuous_fire_generation: int | None = None
        self._queued_continuous_playback: PulsePlayback | None = None
        self._active_fire_end_ordinal = 0
        self._worker_error: BaseException | None = None
        self._terminal: CameraCaptureTerminalRecord | None = None
        self._signal_association: _VirtualCameraSignalAssociation | None = None
        self._open = True
        sequencer.add_fire_listener(self._on_fire)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.frame_source.image_shape

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return self.frame_source.image_shape

    @property
    def frame_dtype(self) -> np.dtype:
        return self.frame_source.frame_dtype

    @property
    def effective_trigger_channels(self) -> tuple[str, ...]:
        return self.capture_trigger_channels

    def ensure_open(self) -> "VirtualCamera":
        if not self._open:
            raise RuntimeError("virtual camera is closed")
        return self

    def arm_signal_event_association(
        self,
        request: SignalAssociationRequest,
        trigger_group_size: int,
        expected_group_count: int,
    ) -> tuple[object, int]:
        """Reserve the next exact finite FIRE on this camera's trigger wire.

        This is deliberately a virtual-apparatus seam, not part of the generic
        CameraAdapter contract.  Only this object observes the in-process FIRE
        callback and the physical frame ordinal counter under the same lock.
        """

        if not isinstance(request, SignalAssociationRequest):
            raise TypeError("request must be SignalAssociationRequest")
        identity = request.association_id
        digest = request.cause_digest
        schedule_fingerprint = request.trigger_schedule_fingerprint
        channel = request.trigger_channel
        if channel != self.capture_trigger_channels[0]:
            raise ValueError(
                "virtual signal association schedule belongs to another trigger wire"
            )
        trigger_count = request.trigger_count
        required_interval = (
            self.capture_working_point().required_external_trigger_interval_seconds
        )
        if required_interval is None:
            raise RuntimeError(
                "virtual camera lacks external-trigger interval readback"
            )
        validate_camera_external_trigger_spacing(
            minimum_trigger_interval_ticks=(
                request.minimum_trigger_interval_ticks
            ),
            clock_hz=request.clock_hz,
            required_interval_seconds=required_interval,
        )
        group_size = _positive_int(trigger_group_size, "trigger_group_size")
        group_count = _positive_int(
            expected_group_count,
            "expected_group_count",
        )
        if trigger_count != group_size * group_count:
            raise ValueError(
                "virtual signal association trigger groups do not cover its count"
            )
        deadline = time.monotonic() + self.timeout
        with self._condition:
            if self.free_running_live:
                raise ValueError(
                    "a free-running virtual camera has no pulse association"
                )
            if self._signal_association is not None:
                raise RuntimeError(
                    "virtual camera already owns a signal association"
                )
            while (
                (self._worker is not None and self._worker.is_alive())
                or self._pending
            ):
                if self._worker_error is not None:
                    raise RuntimeError(
                        "virtual camera source failed before association arm"
                    ) from self._worker_error
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "virtual camera did not reach a drained pre-FIRE boundary"
                    )
                self._condition.wait(min(0.05, remaining))
            if not self._armed or self._expected is not None or not self._accepting:
                raise RuntimeError(
                    "virtual camera association requires a running external-trigger monitor"
                )
            if self._worker_error is not None:
                raise RuntimeError(
                    "virtual camera source failed before association arm"
                ) from self._worker_error
            start = self._produced
            if start % group_size:
                raise RuntimeError(
                    "virtual camera monitor is not at a complete readout-cycle boundary"
                )
            association = _VirtualCameraSignalAssociation(
                identity,
                digest,
                schedule_fingerprint,
                trigger_count,
                group_size,
                group_count,
                start,
            )
            self._signal_association = association
            return association, start

    def bind_signal_event_association(
        self,
        token: object,
        *,
        artifact_digest: str,
        trigger_counts: tuple[tuple[str, int], ...],
        terminal_evidence_digest: str,
        terminal_evidence_kind: str,
    ) -> tuple[str, int, int]:
        """Bind the observed virtual FIRE group to its exact terminal receipt."""

        if terminal_evidence_kind != "SIMULATED":
            raise ValueError(
                "virtual camera association requires a simulated pulse terminal"
            )
        artifact = sha256_text(artifact_digest, "artifact_digest")
        terminal_digest = sha256_text(
            terminal_evidence_digest,
            "terminal_evidence_digest",
        )
        counts = tuple(trigger_counts)
        with self._condition:
            association = self._require_signal_association(token)
            if association.error is not None:
                raise RuntimeError(
                    "virtual camera signal association failed during FIRE"
                ) from association.error
            if association.physical_end_ordinal is None:
                raise RuntimeError(
                    "virtual camera did not observe the associated FIRE"
                )
            if artifact != association.cause_digest:
                raise ValueError(
                    "virtual camera terminal belongs to another pulse artifact"
                )
            channel = self.capture_trigger_channels[0]
            if counts != ((channel, association.expected_trigger_count),):
                raise RuntimeError(
                    "virtual pulse terminal trigger count differs from camera association"
                )
            association.terminal_evidence_digest = terminal_digest
            return (
                channel,
                association.physical_start_ordinal,
                association.physical_end_ordinal,
            )

    def finish_signal_event_association(
        self,
        token: object,
    ) -> tuple[str, int, int, str]:
        """Prove the bound ordinal interval was produced completely and exactly."""

        deadline = time.monotonic() + self.timeout
        with self._condition:
            association = self._require_signal_association(token)
            while True:
                if association.error is not None:
                    self._signal_association = None
                    self._condition.notify_all()
                    raise RuntimeError(
                        "virtual camera signal association failed"
                    ) from association.error
                end = association.physical_end_ordinal
                worker_running = (
                    self._worker is not None and self._worker.is_alive()
                )
                if end is not None and self._produced >= end and not worker_running:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "virtual camera association did not finish its frame interval"
                    )
                self._condition.wait(min(0.05, remaining))
            assert association.physical_end_ordinal is not None
            if self._produced != association.physical_end_ordinal:
                raise RuntimeError(
                    "virtual camera produced frames outside the associated FIRE interval"
                )
            terminal_digest = association.terminal_evidence_digest
            if terminal_digest is None:
                raise RuntimeError(
                    "virtual camera association has no bound pulse terminal"
                )
            result = (
                self.capture_trigger_channels[0],
                association.physical_start_ordinal,
                association.physical_end_ordinal,
                terminal_digest,
            )
            self._signal_association = None
            self._condition.notify_all()
            return result

    def cancel_signal_event_association(self, token: object) -> None:
        """Release an uncommitted association without changing camera ownership."""

        with self._condition:
            association = self._signal_association
            if association is None:
                return
            if association is not token:
                raise RuntimeError(
                    "virtual camera association token belongs to another request"
                )
            self._signal_association = None
            self._condition.notify_all()

    def _require_signal_association(
        self,
        token: object,
    ) -> _VirtualCameraSignalAssociation:
        association = self._signal_association
        if association is None or association is not token:
            raise RuntimeError(
                "virtual camera association token is not current"
            )
        return association

    def snapshot(self) -> dict[str, object]:
        source_snapshot = getattr(self.frame_source, "snapshot", None)
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self.roi,
            "timeout": self.timeout,
            "free_running_live": self.free_running_live,
            "frame_source_type": (
                f"{type(self.frame_source).__module__}."
                f"{type(self.frame_source).__qualname__}"
            ),
            "frame_source": (
                None if not callable(source_snapshot) else source_snapshot()
            ),
        }

    def capture_working_point(self) -> CameraWorkingPoint:
        """Freeze the same deterministic readback used by exact capability minting."""

        self.ensure_open()
        shape = tuple(int(size) for size in self.frame_shape)
        sensor = tuple(int(size) for size in self.sensor_shape)
        settings = dict(self.snapshot())
        settings.update(
            {
                "adapter_type": f"{type(self).__module__}.{type(self).__qualname__}",
                "frame_shape": shape,
                "sensor_shape": sensor,
                "frame_dtype": self.frame_dtype.str,
                "acquisition_mode": "EXTERNAL_TRIGGERED",
                "capture_trigger_channels": tuple(self.capture_trigger_channels),
                "effective_trigger_channels": tuple(self.effective_trigger_channels),
                "applied_exposure_seconds": float(self.exposure),
                "required_external_trigger_interval_seconds": float(self.exposure),
                "external_trigger_integration_start_offset_seconds": 0.0,
            }
        )
        if self.roi is None:
            origin_yx = (0, 0)
            roi_shape_yx = sensor
        else:
            x, width, y, height = (int(value) for value in self.roi)
            origin_yx = (y, x)
            roi_shape_yx = (height, width)
        return CameraWorkingPoint(
            settings_fingerprint=canonical_digest(settings),
            acquisition_mode="EXTERNAL_TRIGGERED",
            frame_shape_yx=shape,
            sensor_shape_yx=sensor,
            roi_origin_yx=origin_yx,
            roi_shape_yx=roi_shape_yx,
            binning_yx=(1, 1),
            dtype=self.frame_dtype,
            count_unit="count",
            capture_trigger_channels=tuple(self.capture_trigger_channels),
            exposure_seconds=float(self.exposure),
            required_external_trigger_interval_seconds=float(self.exposure),
            external_trigger_integration_start_offset_seconds=0.0,
            gain=1.0,
            readout_mode="target-virtual:mode=EXTERNAL_TRIGGERED",
        )

    def configure_exposure_seconds(self, exposure_seconds: float) -> None:
        """Apply the same exposure setting a real camera adapter must read back."""

        value = _positive(exposure_seconds, "exposure_seconds")
        with self._condition:
            if self._armed or (
                self._worker is not None and self._worker.is_alive()
            ):
                raise RuntimeError(
                    "virtual camera exposure cannot change while armed"
                )
            self.exposure = value

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float | None = None,
        stop: object | None = None,
    ) -> None:
        del timeout, stop
        self.ensure_open()
        expected = None if frames is None else _positive_int(frames, "frames")
        if expected is None:
            if source_group_sizes is not None:
                raise ValueError(
                    "monitor arm cannot declare finite source_group_sizes"
                )
            groups = None
        else:
            if source_group_sizes is None:
                raise ValueError("finite arm requires source_group_sizes")
            groups = tuple(
                _positive_int(size, "source_group_sizes item")
                for size in source_group_sizes
            )
            if not groups or sum(groups) != expected:
                raise ValueError(
                    "source_group_sizes must exactly cover finite arm frames"
                )
        buffer_count = _positive_int(buffer_frame_count, "buffer_frame_count")
        if expected is not None and buffer_count != expected:
            raise ValueError(
                "finite buffer_frame_count must equal the complete frame count"
            )
        continuous_playback = None
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual camera already owns an armed capture")
            if self._signal_association is not None:
                raise RuntimeError(
                    "virtual camera retained an unfinished signal association"
                )
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous virtual camera producer is still running")
            self._pending.clear()
            self._armed = True
            self._armed_at_monotonic = time.monotonic()
            self._accepting = True
            self._expected = expected
            self._source_group_sizes = groups
            self._source_group_cursor = 0
            self._produced = 0
            self._worker = None
            self._worker_stop = None
            self._continuous_fire_generation = None
            self._queued_continuous_playback = None
            self._active_fire_end_ordinal = 0
            self._worker_error = None
            self._terminal = None
            if expected is None and self.free_running_live:
                live_stop = threading.Event()
                self._worker_stop = live_stop

                def produce_live() -> None:
                    next_frame = time.monotonic() + self.exposure
                    try:
                        while not live_stop.is_set():
                            if live_stop.wait(
                                max(0.0, next_frame - time.monotonic())
                            ):
                                break
                            with self._condition:
                                if not self._armed or not self._accepting:
                                    break
                                ordinal = self._produced
                            if isinstance(self.frame_source, VirtualMotFrameSource):
                                image = self.frame_source.render_current_output()
                            else:
                                image = self.frame_source.render_image(
                                    self.exposure,
                                    self.exposure,
                                )
                            record = CameraFrameRecord(
                                image=image,
                                source_ordinal=ordinal,
                                produced_count=ordinal + 1,
                                frame_stamp=ordinal + 1,
                                camera_stamp=ordinal + 1,
                                timestamp_seconds=None,
                                timestamp_microseconds=None,
                                host_received_at_ns=time.time_ns(),
                            )
                            with self._condition:
                                if (
                                    not self._armed
                                    or not self._accepting
                                    or live_stop.is_set()
                                ):
                                    break
                                self._pending.append(record)
                                self._produced = ordinal + 1
                                self._condition.notify_all()
                            next_frame += self.exposure
                    except BaseException as error:
                        with self._condition:
                            self._worker_error = error
                            self._accepting = False
                            self._condition.notify_all()
                    finally:
                        with self._condition:
                            self._condition.notify_all()

                worker = threading.Thread(
                    target=produce_live,
                    name="zlc-target-virtual-camera-live",
                    daemon=False,
                )
                self._worker = worker
                try:
                    worker.start()
                except BaseException:
                    self._worker = None
                    self._worker_stop = None
                    self._armed = False
                    self._armed_at_monotonic = None
                    self._accepting = False
                    raise
            elif expected is None:
                continuous_playback = self.sequencer.firing
        # A qCMOS-like passive monitor may be armed either before or after the
        # continuous pulse starts.  In both orders the same hardware-trigger
        # listener owns frame production; arming never fires the sequencer.
        if continuous_playback is not None:
            self._on_fire(continuous_playback)

    def _on_fire(self, playback: PulsePlayback) -> None:
        if playback.repeat_forever:
            with self._condition:
                association = self._signal_association
                if association is not None:
                    error = RuntimeError(
                        "a finite signal association observed a continuous virtual FIRE"
                    )
                    association.error = error
                    self._worker_error = error
                    self._accepting = False
                    self._condition.notify_all()
                    raise error
            self._start_continuous_triggered_live(playback)
            return
        with self._condition:
            def reject(message: str, cause: BaseException | None = None) -> NoReturn:
                error = RuntimeError(message)
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
                if cause is not None:
                    raise error from cause
                raise error

            if not self._armed:
                return
            if self._expected is None and self.free_running_live:
                return
            if self._worker_error is not None:
                raise RuntimeError(
                    "virtual camera rejects FIRE after a source failure"
                ) from self._worker_error
            if not self._accepting:
                if self._expected is not None and self._produced >= self._expected:
                    raise RuntimeError(
                        "virtual camera received FIRE after its expected frame count was complete"
                    )
                return
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("virtual camera received overlapping finite FIRE")
            self._worker = None
            self._worker_stop = None
            trigger_offsets = _trigger_starts(
                playback,
                self.capture_trigger_channels,
            )
            trigger_count = len(trigger_offsets)
            association = self._signal_association
            if association is not None:
                try:
                    artifact = self.sequencer.output_artifact
                    if artifact is None:
                        raise RuntimeError(
                            "virtual camera observed FIRE without a compiled artifact"
                        )
                    if artifact.fingerprint != association.cause_digest:
                        raise RuntimeError(
                            "virtual camera observed another artifact after association arm"
                        )
                    matching_schedules = tuple(
                        schedule
                        for schedule in artifact.trigger_schedules
                        if schedule.channel == self.capture_trigger_channels[0]
                    )
                    if (
                        len(matching_schedules) != 1
                        or matching_schedules[0].fingerprint
                        != association.trigger_schedule_fingerprint
                    ):
                        raise RuntimeError(
                            "virtual FIRE trigger schedule differs from association preflight"
                        )
                    if trigger_count != association.expected_trigger_count:
                        raise RuntimeError(
                            "virtual FIRE trigger count differs from the associated group"
                        )
                    if self._produced != association.physical_start_ordinal:
                        raise RuntimeError(
                            "virtual camera advanced after association arm but before FIRE"
                        )
                    association.physical_end_ordinal = (
                        association.physical_start_ordinal + trigger_count
                    )
                except BaseException as error:
                    association.error = error
                    self._worker_error = error
                    self._accepting = False
                    self._condition.notify_all()
                    raise
            remaining = (
                None
                if self._expected is None
                else self._expected - self._produced
            )
            if trigger_count < 1:
                if self._expected is None:
                    return
                error = RuntimeError(
                    "finite FIRE emitted no camera trigger while an exact arm was active"
                )
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
                raise error
            if remaining is not None and trigger_count > remaining:
                error = RuntimeError(
                    f"virtual pulse emitted {trigger_count} camera edges with only "
                    f"{remaining} expected frames remaining"
                )
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
                raise error
            if self._expected is None:
                fire_group_sizes = (trigger_count,)
                fire_group_point_indices: tuple[int | None, ...] = (None,)
            else:
                groups = self._source_group_sizes
                if groups is None:
                    raise RuntimeError("finite camera arm lost its source grouping")
                selected: list[int] = []
                selected_count = 0
                while selected_count < trigger_count:
                    group_index = self._source_group_cursor + len(selected)
                    if group_index >= len(groups):
                        reject(
                            "finite FIRE exceeds the frozen camera source groups"
                        )
                    group_size = groups[group_index]
                    if selected_count + group_size > trigger_count:
                        reject(
                            "finite FIRE splits a frozen camera source group"
                        )
                    selected.append(group_size)
                    selected_count += group_size
                fire_group_sizes = tuple(selected)
                selected_channels = set(self.capture_trigger_channels)
                compiled_group_records = tuple(
                    group
                    for group in playback.trigger_groups
                    if any(
                        channel in selected_channels
                        for channel, _count in group.channel_counts
                    )
                )
                if compiled_group_records:
                    try:
                        compiled_groups = playback.trigger_group_sizes(
                            self.capture_trigger_channels
                        )
                    except (TypeError, ValueError) as cause:
                        reject(
                            "compiled pulse grouping cannot verify the frozen camera request",
                            cause,
                        )
                    if compiled_groups != fire_group_sizes:
                        reject(
                            "compiled pulse grouping differs from the frozen camera request"
                        )
                    if len(compiled_group_records) != len(fire_group_sizes):
                        reject(
                            "compiled pulse points differ from the frozen camera groups"
                        )
                    fire_group_point_indices = tuple(
                        group.point_index for group in compiled_group_records
                    )
                else:
                    fire_group_point_indices = tuple(
                        None for _group in fire_group_sizes
                    )
                self._source_group_cursor += len(fire_group_sizes)
            start_ordinal = self._produced
            self._active_fire_end_ordinal = start_ordinal + trigger_count
            stop = threading.Event()
            self._worker_stop = stop
            fired_at = time.monotonic()

            def produce() -> None:
                try:
                    frames = self.frame_source.iter_frames(
                        playback,
                        trigger_count,
                        trigger_channels=self.capture_trigger_channels,
                        default_exposure=self.exposure,
                        trigger_group_sizes=fire_group_sizes,
                        trigger_group_point_indices=fire_group_point_indices,
                    )
                    frame_iterator = iter(frames)
                    for local_ordinal, offset in enumerate(trigger_offsets):
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
                                "virtual atom source ended before the expected trigger count"
                            ) from exc
                        source_ordinal = start_ordinal + local_ordinal
                        produced_count = source_ordinal + 1
                        record = CameraFrameRecord(
                            image,
                            source_ordinal,
                            produced_count,
                            produced_count,
                            produced_count,
                            None,
                            None,
                            time.time_ns(),
                        )
                        with self._condition:
                            if not self._armed or not self._accepting or stop.is_set():
                                return
                            self._pending.append(record)
                            self._produced = produced_count
                            self._condition.notify_all()
                    sentinel = object()
                    if next(frame_iterator, sentinel) is not sentinel:
                        raise RuntimeError(
                            "virtual atom source exceeded the expected trigger count"
                        )
                except BaseException as error:
                    with self._condition:
                        self._worker_error = error
                        self._accepting = False
                        self._condition.notify_all()
                finally:
                    with self._condition:
                        if (
                            self._expected is not None
                            and self._produced >= self._expected
                        ):
                            self._accepting = False
                        if self._worker is threading.current_thread():
                            self._worker = None
                            self._worker_stop = None
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

    def _start_continuous_triggered_live(self, playback: PulsePlayback) -> None:
        """Follow one already-running cyclic trigger wire until SAFE or disarm."""

        active_fire = self.sequencer.active_fire(playback)
        if active_fire is None:
            return
        fire_generation, fired_at = active_fire
        trigger_offsets = tuple(
            _trigger_starts(playback, self.capture_trigger_channels)
        )
        if not trigger_offsets:
            return
        with self._condition:
            if not self._armed or self._expected is not None:
                return
            if self.free_running_live:
                # The MOT camera has its own sensor cadence; pulse output does
                # not become a second producer for the same physical camera.
                return
            if self._worker_error is not None:
                raise RuntimeError(
                    "virtual camera rejects continuous FIRE after a source failure"
                ) from self._worker_error
            if not self._accepting:
                return
            if self._worker is not None and self._worker.is_alive():
                if self._continuous_fire_generation == fire_generation:
                    return
                self._queued_continuous_playback = playback
                if self._worker_stop is not None:
                    self._worker_stop.set()
                self._condition.notify_all()
                return
            stop = threading.Event()
            self._worker_stop = stop
            self._continuous_fire_generation = fire_generation
            self._queued_continuous_playback = None
            armed_at = self._armed_at_monotonic
            if armed_at is None:
                raise RuntimeError("virtual camera lost its arm epoch")

            def produce() -> None:
                scale = self.sequencer.sleep_scale
                cycle = (
                    0
                    if scale <= 0.0
                    else max(
                        0,
                        int(
                            math.floor(
                                max(0.0, (armed_at - fired_at) / scale)
                                / playback.repeat_period
                            )
                        ),
                    )
                )
                replacement = None
                try:
                    while not stop.is_set():
                        if not self.sequencer.is_active_fire(
                            playback,
                            fire_generation,
                        ):
                            break
                        frames = iter(
                            self.frame_source.iter_frames(
                                playback,
                                len(trigger_offsets),
                                trigger_channels=self.capture_trigger_channels,
                                default_exposure=self.exposure,
                                trigger_group_sizes=(len(trigger_offsets),),
                            )
                        )
                        for offset in trigger_offsets:
                            logical_edge = cycle * playback.repeat_period + offset
                            edge_at = fired_at + logical_edge * scale
                            if edge_at < armed_at:
                                # The sensor was not armed at this hardware edge.
                                # Advance the deterministic virtual source, but
                                # never turn a pre-arm trigger into a host frame.
                                try:
                                    next(frames)
                                except StopIteration as exc:
                                    raise RuntimeError(
                                        "virtual atom source ended before a cyclic trigger"
                                    ) from exc
                                continue
                            deadline = edge_at + self.exposure * scale
                            while not stop.is_set():
                                if not self.sequencer.is_active_fire(
                                    playback,
                                    fire_generation,
                                ):
                                    return
                                remaining = deadline - time.monotonic()
                                if remaining <= 0.0:
                                    break
                                stop.wait(min(remaining, 0.05))
                            if stop.is_set():
                                return
                            try:
                                image = next(frames)
                            except StopIteration as exc:
                                raise RuntimeError(
                                    "virtual atom source ended before a cyclic trigger"
                                ) from exc
                            with self._condition:
                                if (
                                    not self._armed
                                    or not self._accepting
                                    or not self.sequencer.is_active_fire(
                                        playback,
                                        fire_generation,
                                    )
                                ):
                                    return
                                ordinal = self._produced
                                self._pending.append(
                                    CameraFrameRecord(
                                        image,
                                        ordinal,
                                        ordinal + 1,
                                        ordinal + 1,
                                        ordinal + 1,
                                        None,
                                        None,
                                        time.time_ns(),
                                    )
                                )
                                self._produced = ordinal + 1
                                self._condition.notify_all()
                        sentinel = object()
                        if next(frames, sentinel) is not sentinel:
                            raise RuntimeError(
                                "virtual atom source exceeded cyclic trigger count"
                            )
                        cycle += 1
                        # Zero-time simulation still needs a bounded sensor
                        # cadence so a monitor cannot spin and starve its host.
                        if self.sequencer.sleep_scale == 0.0:
                            stop.wait(min(self.exposure, 0.02))
                except BaseException as error:
                    with self._condition:
                        self._worker_error = error
                        self._accepting = False
                        self._condition.notify_all()
                finally:
                    with self._condition:
                        if self._worker is threading.current_thread():
                            self._worker = None
                            self._worker_stop = None
                            self._continuous_fire_generation = None
                            replacement = self._queued_continuous_playback
                            self._queued_continuous_playback = None
                        self._condition.notify_all()
                    if replacement is not None:
                        self._start_continuous_triggered_live(replacement)

            worker = threading.Thread(
                target=produce,
                name="zlc-target-virtual-camera-continuous-trigger",
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
        if not isinstance(exact, bool):
            raise TypeError("exact must be bool")
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
                    # The final frame of one FIRE is not authoritative until the
                    # producer has advanced the source once more and proved EOF.
                    # That post-frame probe detects both an extra frame and a
                    # source exception raised after the expected final yield.
                    # Keep the proof within the caller's original deadline:
                    # an ill-behaved iterator must not turn an exact read into an
                    # unbounded join, and cancellation must not return a frame
                    # whose finite-source cardinality was never established.
                    while exact and (
                        self._worker is not None
                        and self._worker.is_alive()
                        and self._produced >= self._active_fire_end_ordinal
                    ):
                        if self._worker_error is not None:
                            raise RuntimeError(
                                "virtual camera source failed"
                            ) from self._worker_error
                        if stop is not None and getattr(
                            stop,
                            "is_set",
                            lambda: False,
                        )():
                            raise TimeoutError(
                                "virtual camera exact source validation was cancelled"
                            )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            raise TimeoutError(
                                "virtual camera exact source validation timed out"
                            )
                        self._condition.wait(min(0.05, remaining))
                    if self._worker_error is not None:
                        raise RuntimeError(
                            "virtual camera source failed"
                        ) from self._worker_error
                    break
                if self._worker_error is not None:
                    raise RuntimeError("virtual camera source failed") from self._worker_error
                if stop is not None and getattr(stop, "is_set", lambda: False)():
                    break
                if not self._accepting:
                    break
                if self._worker is not None and not self._worker.is_alive():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(min(0.05, remaining))
        if len(result) != wanted and exact:
            raise TimeoutError(
                f"virtual camera returned {len(result)} of {wanted} exact frame(s)"
            )
        return result

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            association = self._signal_association
            if association is not None:
                association.error = RuntimeError(
                    "virtual camera was disarmed during signal association"
                )
                self._signal_association = None
            worker = self._worker
            stop = self._worker_stop
            self._accepting = False
            self._queued_continuous_playback = None
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
            self._armed_at_monotonic = None
            self._source_group_sizes = None
            self._source_group_cursor = 0
            self._continuous_fire_generation = None
            self._pending.clear()
            self._terminal = terminal
            self._condition.notify_all()
            return terminal

    def capture_state(self) -> tuple[bool, int]:
        with self._condition:
            return self._armed, self._produced

    def observed_produced_count(self) -> int:
        """Return the exact virtual sensor production ordinal without draining."""

        with self._condition:
            if not self._armed:
                raise RuntimeError(
                    "virtual camera produced count requires an armed capture"
                )
            return self._produced

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
