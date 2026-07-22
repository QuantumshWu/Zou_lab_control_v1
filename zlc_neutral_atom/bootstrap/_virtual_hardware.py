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
from typing import Iterator, Mapping, Sequence

import numpy as np

from zlc_neutral_atom.adapter_sdk import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.readout.sitemap import ReadoutGridGeometry
from zlc_pulse import (
    CompiledPulseArtifact,
    PulsePlayback,
    PulseTarget,
    sample_compiled_bus_codes,
)
from zlc_storage import canonical_digest
from zlc_storage.canonical import positive_integer as _positive_int


_K_B = 1.380649e-23
_RB87_MASS_KG = 86.909180527 * 1.66053906660e-27


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
        self._prepared_artifact: CompiledPulseArtifact | None = None
        self._artifact_digest: str | None = None
        self._firing: PulsePlayback | None = None
        self._last_fired: PulsePlayback | None = None
        self._firing_artifact: CompiledPulseArtifact | None = None
        self._last_fired_artifact: CompiledPulseArtifact | None = None
        self._fire_started_monotonic: float | None = None
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


class VirtualCamera:
    """Finite externally-triggered camera with one bounded producer queue."""

    max_pending_records = 16

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
        self._capacity = self.max_pending_records
        self._produced = 0
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._active_fire_end_ordinal = 0
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
                "frame_dtype": np.dtype("<u2").str,
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
            dtype=np.dtype("<u2"),
            count_unit="count",
            capture_trigger_channels=tuple(self.capture_trigger_channels),
            exposure_seconds=float(self.exposure),
            required_external_trigger_interval_seconds=float(self.exposure),
            external_trigger_integration_start_offset_seconds=0.0,
            gain=1.0,
            readout_mode="target-virtual:mode=EXTERNAL_TRIGGERED",
        )

    def arm(
        self,
        frames: int | None,
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
        if capacity > self.max_pending_records:
            raise ValueError("max_inflight_frames exceeds camera max_pending_records")
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
            self._active_fire_end_ordinal = 0
            self._worker_error = None
            self._terminal = None

    def _on_fire(self, playback: PulsePlayback) -> None:
        if playback.repeat_forever:
            return
        with self._condition:
            if not self._armed:
                return
            if self._worker_error is not None:
                raise RuntimeError(
                    "virtual camera rejects FIRE after a source failure"
                ) from self._worker_error
            if not self._accepting:
                if self._produced >= self._expected:
                    raise RuntimeError(
                        "virtual camera received FIRE after its arm budget was complete"
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
            remaining = self._expected - self._produced
            if trigger_count < 1:
                error = RuntimeError(
                    "finite FIRE emitted no camera trigger while an exact arm was active"
                )
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
                raise error
            if trigger_count > remaining:
                error = RuntimeError(
                    f"virtual pulse emitted {trigger_count} camera edges with only "
                    f"{remaining} remaining in the arm budget"
                )
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
                raise error
            start_ordinal = self._produced
            self._active_fire_end_ordinal = start_ordinal + trigger_count
            stop = threading.Event()
            self._worker_stop = stop
            fired_at = time.monotonic()

            def produce() -> None:
                try:
                    frames = self.atoms.iter_frames(
                        playback,
                        trigger_count,
                        trigger_channels=self.capture_trigger_channels,
                        default_exposure=self.exposure,
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
                                "virtual atom source ended before the trigger budget"
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
                            if len(self._pending) >= self._capacity:
                                raise RuntimeError(
                                    "virtual camera retention exhausted before host drain"
                                )
                            self._pending.append(record)
                            self._produced = produced_count
                            self._condition.notify_all()
                    sentinel = object()
                    if next(frame_iterator, sentinel) is not sentinel:
                        raise RuntimeError(
                            "virtual atom source exceeded the trigger budget"
                        )
                except BaseException as error:
                    with self._condition:
                        self._worker_error = error
                        self._accepting = False
                        self._condition.notify_all()
                finally:
                    with self._condition:
                        if self._produced >= self._expected:
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
                    # The final frame of one FIRE is not authoritative until the
                    # producer has advanced the source once more and proved EOF.
                    # That post-frame probe detects both an extra frame and a
                    # source exception raised after the expected final yield.
                    # Keep the proof on the caller's original blocking budget:
                    # an ill-behaved iterator must not turn an exact read into an
                    # unbounded join, and cancellation must not return a frame
                    # whose finite-source cardinality was never established.
                    while (
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


class VirtualMonitorCamera:
    """Hardware-clocked virtual Basler MOT monitor with a bounded record queue.

    The producer thread models the sensor exposure clock.  Host reads never
    schedule exposures and the queue never silently overwrites: if the current
    runtime cannot drain the admitted queue, the source fails and the live
    front is invalidated instead of relabelling pre-broker loss as GUI loss.
    Frames preserve the deployed simulator's MOT physics: the sensor observes
    the compiled three-axis DAC outputs, not task set-points, and renders a
    noisy Mono8 fluorescence spot whose efficiency is Gaussian in coil space.
    """

    max_pending_records = 4

    def __init__(
        self,
        sequencer: VirtualSequencer,
        *,
        frame_shape: tuple[int, int] = (1200, 1920),
        exposure: float = 0.05,
        timeout: float = 2.0,
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
            raise TypeError("virtual monitor camera requires VirtualSequencer")
        if (
            not isinstance(frame_shape, tuple)
            or len(frame_shape) != 2
        ):
            raise TypeError("monitor frame_shape must be a (height, width) tuple")
        self._frame_shape = tuple(
            _positive_int(size, "monitor frame dimension") for size in frame_shape
        )
        self.exposure = _positive(exposure, "monitor exposure")
        self.timeout = _positive(timeout, "monitor timeout")
        self.sequencer = sequencer
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
        if not self.coil_ports or any(not name or not port for name, port in self.coil_ports.items()):
            raise ValueError("monitor coil_ports must map non-empty names to ports")
        target_ports = sequencer.target.by_key
        for port in self.coil_ports.values():
            spec = target_ports.get(port)
            if spec is None or spec.signed_range is None:
                raise ValueError(f"monitor coil port {port!r} is not a target DAC")
        self.optimum_levels = {
            str(name): float(value)
            for name, value in dict(
                optimum_levels or {"da_x": 7.0, "da_y": -5.0, "da_z": 11.0}
            ).items()
        }
        self.level_sigmas = {
            str(name): _positive(value, f"monitor level sigma {name!r}")
            for name, value in dict(
                level_sigmas or {"da_x": 6.0, "da_y": 6.0, "da_z": 6.0}
            ).items()
        }
        if set(self.optimum_levels) != set(self.coil_ports) or set(self.level_sigmas) != set(
            self.coil_ports
        ):
            raise ValueError("monitor coil, optimum, and sigma axes must match")
        self.peak_counts = _nonnegative(peak_counts, "monitor peak_counts")
        self.offset_counts = _nonnegative(offset_counts, "monitor offset_counts")
        self.read_noise = _nonnegative(read_noise, "monitor read_noise")
        if not isinstance(spot_size_px, tuple) or len(spot_size_px) != 2:
            raise TypeError("monitor spot_size_px must be a two-item tuple")
        self.spot_size_px = (
            _positive(spot_size_px[0], "monitor spot width"),
            _positive(spot_size_px[1], "monitor spot height"),
        )
        self.last_levels: dict[str, float] | None = None
        self._rng = np.random.default_rng(seed)
        self._condition = threading.Condition(threading.RLock())
        self._pending: deque[CameraFrameRecord] = deque()
        self._armed = False
        self._accepting = False
        self._capacity = self.max_pending_records
        self._produced = 0
        self._producer: threading.Thread | None = None
        self._producer_stop: threading.Event | None = None
        self._worker_error: BaseException | None = None
        self._terminal: CameraCaptureTerminalRecord | None = None
        self._open = True

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._frame_shape

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return self._frame_shape

    def ensure_open(self) -> "VirtualMonitorCamera":
        if not self._open:
            raise RuntimeError("virtual monitor camera is closed")
        return self

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "frame_shape": self.frame_shape,
            "acquisition_mode": "FREE_RUNNING",
            "coil_ports": dict(self.coil_ports),
            "optimum_levels": dict(self.optimum_levels),
            "level_sigmas": dict(self.level_sigmas),
            "peak_counts": self.peak_counts,
            "offset_counts": self.offset_counts,
            "read_noise": self.read_noise,
            "spot_size_px": self.spot_size_px,
        }

    def capture_working_point(self) -> CameraWorkingPoint:
        self.ensure_open()
        settings = canonical_digest(self.snapshot())
        return CameraWorkingPoint(
            settings_fingerprint=settings,
            acquisition_mode="FREE_RUNNING",
            frame_shape_yx=self.frame_shape,
            sensor_shape_yx=self.sensor_shape,
            roi_origin_yx=(0, 0),
            roi_shape_yx=self.frame_shape,
            binning_yx=(1, 1),
            dtype=np.dtype("<u1"),
            count_unit="count",
            capture_trigger_channels=(),
            exposure_seconds=self.exposure,
            required_external_trigger_interval_seconds=None,
            external_trigger_integration_start_offset_seconds=None,
            gain=1.0,
            readout_mode="target-virtual-mot:mode=FREE_RUNNING;pixel=Mono8",
        )

    def arm(
        self,
        frames: int | None,
        *,
        max_inflight_frames: int,
        timeout: float,
    ) -> None:
        del timeout
        self.ensure_open()
        if frames is not None:
            raise ValueError("virtual monitor camera only accepts a continuous arm")
        capacity = _positive_int(max_inflight_frames, "max_inflight_frames")
        if capacity > self.max_pending_records:
            raise ValueError("monitor inflight budget exceeds camera capacity")
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual monitor camera is already armed")
            if self._producer is not None and self._producer.is_alive():
                raise RuntimeError("previous monitor producer is still running")
            self._pending.clear()
            self._armed = True
            self._accepting = True
            self._capacity = capacity
            self._produced = 0
            self._worker_error = None
            self._terminal = None
            stop = threading.Event()
            self._producer_stop = stop

            def produce() -> None:
                next_frame = time.monotonic() + self.exposure
                try:
                    while not stop.is_set():
                        if stop.wait(max(0.0, next_frame - time.monotonic())):
                            break
                        with self._condition:
                            if not self._armed or not self._accepting:
                                break
                            if len(self._pending) >= self._capacity:
                                raise RuntimeError(
                                    "virtual monitor camera record queue overflowed"
                                )
                            ordinal = self._produced
                        image = self._render(ordinal)
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
                            if not self._armed or not self._accepting or stop.is_set():
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
                        # Keep the Thread object until the arm owner explicitly
                        # joins it in finish_record_capture().  Clearing it from
                        # inside the producer would let cleanup claim "joined"
                        # during the tiny interval between this finally block
                        # and the OS thread's actual exit.
                        self._condition.notify_all()

            producer = threading.Thread(
                target=produce,
                name="zlc-target-virtual-monitor-camera",
                daemon=False,
            )
            self._producer = producer
            try:
                producer.start()
            except BaseException:
                self._producer = None
                self._producer_stop = None
                self._armed = False
                self._accepting = False
                raise

    def mot_efficiency(self, levels: Mapping[str, float]) -> float:
        z = sum(
            (
                (float(levels.get(name, 0.0)) - self.optimum_levels[name])
                / self.level_sigmas[name]
            )
            ** 2
            for name in self.coil_ports
        )
        return float(math.exp(-0.5 * z))

    def _sense_levels(self, _ordinal: int) -> dict[str, float]:
        artifact = self.sequencer.output_artifact
        if artifact is None:
            return {name: 0.0 for name in self.coil_ports}
        points = artifact.target_ir.scan_points or ((),)
        if len(points) != 1:
            raise RuntimeError(
                "free-running monitor frames have no declared association with "
                "multi-point sequencer output"
            )
        codes = dict(
            sample_compiled_bus_codes(
                artifact,
                point_index=0,
                phase=0.5,
            )
        )
        target_ports = self.sequencer.target.by_key
        levels: dict[str, float] = {}
        for name, port in self.coil_ports.items():
            spec = target_ports[port]
            assert spec.signed_range is not None
            levels[name] = float(codes.get(port, spec.safe_value) + spec.signed_range[0])
        return levels

    def _render(self, ordinal: int) -> np.ndarray:
        levels = self._sense_levels(ordinal)
        self.last_levels = dict(levels)
        efficiency = self.mot_efficiency(levels)
        height, width = self.frame_shape
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
        frame = np.empty((height, width), dtype=np.uint8)
        # Keep the noise scratch at or below one retained frame.  A full-frame
        # float64 normal array would make the adapter exceed the memory peak it
        # attests before arm, even though the delivered Mono8 frame is small.
        values_per_chunk = max(1, min(1 << 15, frame.size // 8))
        for start in range(0, frame.size, values_per_chunk):
            stop = min(frame.size, start + values_per_chunk)
            noise = self._rng.normal(
                self.offset_counts,
                self.read_noise,
                size=stop - start,
            )
            np.clip(noise, 0, 255, out=noise)
            frame.flat[start:stop] = noise
        signal = self._rng.poisson(
            self.peak_counts * efficiency * spot
        ).astype(np.int32, copy=False)
        region = frame[y0:y1, x0:x1].astype(np.int32)
        region += signal
        np.clip(region, 0, 255, out=region)
        frame[y0:y1, x0:x1] = region
        return frame

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        if exact:
            raise ValueError("free-running monitor records are not exact capture data")
        wanted = _positive_int(n, "n")
        deadline = time.monotonic() + _positive(timeout, "timeout")
        result: list[CameraFrameRecord] = []
        with self._condition:
            if not self._armed:
                raise RuntimeError("monitor read requires an armed camera")
            while len(result) < wanted:
                while self._pending and len(result) < wanted:
                    result.append(self._pending.popleft())
                    self._condition.notify_all()
                if len(result) == wanted:
                    break
                if self._worker_error is not None:
                    raise RuntimeError("virtual monitor source failed") from self._worker_error
                if not self._accepting:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("virtual monitor camera read timed out")
                self._condition.wait(min(0.05, remaining))
        return result

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            producer = self._producer
            stop = self._producer_stop
            self._accepting = False
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if producer is not None and producer is not threading.current_thread():
            producer.join(self.timeout)
            if producer.is_alive():
                raise TimeoutError("virtual monitor producer did not join")
        with self._condition:
            terminal = CameraCaptureTerminalRecord(
                self._produced,
                True,
                True,
                producer is None or not producer.is_alive(),
            )
            self._armed = False
            self._pending.clear()
            self._producer = None
            self._producer_stop = None
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
            self._open = False


__all__: list[str] = []
