"""Virtual devices for offline notebook tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from pathlib import Path
from typing import Sequence
import time

import numpy as np

from ..core.analysis import finite_float, grid_shape_tuple, positive_int
from ..core.utils import site_index
from .base import CameraDevice, SequencerDevice, TrapArrayDevice
from ..timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    PulseSequence,
    exposure_from_sequence,
    imaging_channel_kwargs,
    sequence_for_frame_count,
)


# Boltzmann constant, atomic mass unit, and the Rb87 mass (SI).  Used ONLY by the
# virtual data source's release-recapture loss model (below); the analysis layer
# never sees these -- it only receives the resulting camera frames.
_K_B = 1.380649e-23
_AMU = 1.66053906660e-27
_RB87_MASS = 86.909180527 * _AMU


DEFAULT_CHANNELS = (
    "trap",
    "cooling",
    "probe",
    "emCCD",
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
    # --- Release-recapture loss model (data-source side only) -----------------
    # When a fired sequence switches the trap OFF for ``t_off`` seconds between
    # two camera exposures, atoms fly ballistically and are lost unless they stay
    # within the capture region; the survival probability falls off with ``t_off``.
    # The model is the standard ballistic release-recapture survival (3-D
    # isotropic, point source): per axis an atom is recaptured if its speed obeys
    # ``|v| < r_c / t_off``, so for a Maxwell-Boltzmann spread
    # ``sigma_v = sqrt(k_B T / m)`` the per-axis fraction is ``erf(r_c / (sqrt2
    # sigma_v t_off))`` and 3-D survival is that cubed.  ``recapture_temperature_K``
    # sets the velocity spread (hotter -> faster decay) and ``capture_radius_m``
    # the trap geometry; together they fix the characteristic ``t_off`` scale.
    # Defaults give a clearly non-flat demo curve over ~0..300 us: survival is
    # ~1 near t_off=0 and falls smoothly through ~0.5 around 75 us to a few % by
    # 300 us (the half-survival scale is r_c / sqrt(2) sigma_v).
    recapture_temperature_K: float = 50e-6
    capture_radius_m: float = 6.0e-6
    recapture_mass_kg: float = _RB87_MASS
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
        self.recapture_temperature_K = positive_float(self.recapture_temperature_K, "recapture_temperature_K")
        self.capture_radius_m = positive_float(self.capture_radius_m, "capture_radius_m")
        self.recapture_mass_kg = positive_float(self.recapture_mass_kg, "recapture_mass_kg")
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

    def recapture_survival_probability(self, t_off: float) -> float:
        """Ballistic release-recapture survival probability after a trap-off of ``t_off`` s.

        Per axis an atom is recaptured if its ballistic displacement stays inside
        the capture radius ``r_c``, i.e. ``|v| < r_c / t_off``; for a 1-D thermal
        spread ``sigma_v = sqrt(k_B T / m)`` that fraction is
        ``erf(r_c / (sqrt(2) sigma_v t_off))``, and the isotropic 3-D survival is
        the cube.  ``t_off <= 0`` -> 1.0 (the atom cannot move).  This is the SAME
        physics the analysis-side fit assumes, so a fitted temperature recovers a
        value near ``recapture_temperature_K`` -- but it lives ONLY here, in the
        data source; the analysis layer never reads it.
        """

        t = float(t_off)
        if not np.isfinite(t) or t <= 0.0:
            return 1.0
        sigma_v = sqrt(_K_B * self.recapture_temperature_K / self.recapture_mass_kg)
        arg = self.capture_radius_m / (sqrt(2.0) * sigma_v * t)
        p_axis = erf(arg)
        return float(p_axis**3)

    def apply_recapture_loss(self, t_off: float) -> np.ndarray:
        """Randomly drop currently-occupied atoms with the release-recapture model.

        Mutates ``self.occupancy`` in place (the same atom loading is imaged again
        afterward) and returns the new occupancy.  Each occupied site survives
        independently with probability :meth:`recapture_survival_probability`.
        """

        p_survive = self.recapture_survival_probability(t_off)
        if p_survive >= 1.0:
            return self.occupancy.copy()
        survive = self.rng.random(self.n_sites) < p_survive
        self.occupancy = self.occupancy & survive
        return self.occupancy.copy()

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "grid_shape": self.grid_shape,
            "image_shape": self.image_shape,
            "offset_counts": self.offset_counts,
            "conversion_e_per_count": self.conversion_e_per_count,
            "read_noise_e": self.read_noise_e,
            "recapture_temperature_K": self.recapture_temperature_K,
            "capture_radius_m": self.capture_radius_m,
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

    def acquire(
        self,
        frames: int = 1,
        *,
        sequence: PulseSequence | None = None,
        sequencer=None,
        force_all_sites: bool | None = None,
        **_,
    ) -> list[np.ndarray]:
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
        # Infer from the channel the imaging sequence actually used (probe -> ch03
        # on a chNN sequencer); same single source as the real adapter so virtual
        # and real track exposure identically.
        probe_channel = imaging_channel_kwargs(sequencer).get("probe_channel", "probe")
        exposure = exposure_from_sequence(sequence, default=self.exposure, channel=probe_channel)
        reload_each = sequence_requests_load(sequence)
        images: list[np.ndarray] = []
        # "All sites loaded" (for sitemap calibration) is best requested
        # explicitly via ``force_all_sites``.  The legacy fallback keys off the
        # sequence *name* == "sitemap"; an explicit value always wins so callers
        # are not surprised by a hidden, virtual-only string match.
        all_sites = (
            bool(force_all_sites)
            if force_all_sites is not None
            else (sequence is not None and sequence.name == "sitemap")
        )
        # RELEASE-RECAPTURE LOSS (data-source side): if the fired sequence holds
        # the trap OFF between two camera triggers (the release-recapture pulse),
        # atoms fly ballistically during that gap and some are lost before the
        # next frame.  We parse the per-frame trap-off time from the SAME sequence
        # the analysis layer built, then apply the loss to the shared occupancy
        # right before rendering the frame that follows the gap.  No analysis code
        # learns of this: it only receives the resulting frames.
        trigger_channels = getattr(sequencer, "trigger_channels", None)
        trap_off_per_frame = trap_off_durations_per_frame(
            sequence, frames, trigger_channels=trigger_channels
        )
        # One ``acquire`` is one experimental shot.  A load-each-frame imaging
        # sequence reloads atoms before EVERY frame (its repeated cooling pulse =
        # separate loadings).  A release-recapture sequence instead has ONE loading
        # imaged twice with a trap-off between -- it carries no cooling pulse, so
        # reload_each is False; we must still reload ONCE at the start of the shot
        # so survival is measured against a FRESH loading each shot (otherwise the
        # trap-off loss would compound across shots and drive survival to zero).
        has_trap_off = any(t > 0.0 for t in trap_off_per_frame)
        if has_trap_off and not reload_each and not all_sites:
            self.trap_array.reload()
        for frame_index in range(frames):
            if reload_each:
                self.trap_array.reload()
            elif trap_off_per_frame[frame_index] > 0.0:
                self.trap_array.apply_recapture_loss(trap_off_per_frame[frame_index])
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
    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS, clock_hz: float = 50_000_000.0, sleep_scale: float = 0.0):
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


def virtual_loading_feed(
    hub,
    *,
    prefix: str = "",
    grid_shape: tuple[int, int] = (5, 7),
    loading_probability: float = 0.55,
    exposure: float = 0.02,
    roi_radius: int = 1,
    ema: float = 0.05,
    seed: int | None = None,
    calibration_frames: int = 4,
    threshold_frames: int = 24,
    trap_array: "VirtualTrapArray | None" = None,
):
    """Build a :class:`~..operations.feeds.LoadingFeed` driven by a virtual camera.

    This is the ONLY virtual-specific glue for the task-console producer: it wires
    a ``VirtualTrapArray`` behind a ``VirtualCamera`` and hands that camera to the
    backend-agnostic ``LoadingFeed``.  The feed itself is identical on real
    hardware -- there you build ``LoadingFeed(hub, exp.camera,
    sequencer=exp.devices.sequencer, grid_shape=...)`` instead.  Faking lives only
    here, at the data source; the feed's analysis path does not change.
    """

    from ..operations.feeds import LoadingFeed  # lazy: keep devices->operations off the import graph

    trap = trap_array if trap_array is not None else VirtualTrapArray(
        grid_shape=tuple(grid_shape), loading_probability=float(loading_probability), seed=seed
    )
    camera = VirtualCamera(trap, exposure=float(exposure))
    return LoadingFeed(
        hub,
        camera,
        prefix=prefix,
        grid_shape=trap.grid_shape,
        exposure=float(exposure),
        roi_radius=roi_radius,
        ema=ema,
        calibration_frames=calibration_frames,
        threshold_frames=threshold_frames,
    )


def write_virtual_run(
    data_dir: str | Path,
    prefix: str = "img",
    *,
    groups: int = 120,
    shots_per_group: int = 4,
    short_shot: int = 3,
    ref_shots: Sequence[int] = (1, 2, 4),
    short_exposure: float = 3e-3,
    reference_exposure: float = 20e-3,
    grid_shape: tuple[int, int] = (5, 7),
    loading_probability: float = 0.55,
    seed: int | None = None,
    suffix: str = ".npy",
    trap_array: "VirtualTrapArray | None" = None,
) -> dict[str, object]:
    """Render a virtual atom-loading RUN to ``data_dir`` as ``PREFIX<n>`` frames.

    This is the FAKE DATA SOURCE: it stands in for the real experiment + camera
    that would write these raw frames to disk.  Each of ``groups`` atom loadings
    is imaged ``shots_per_group`` times (the SAME atoms, no reload between shots,
    so the per-shot loss model makes the reference frames a meaningful ground
    truth); the ``short_shot`` frame uses ``short_exposure`` (the readout being
    characterized) and the ``ref_shots`` use ``reference_exposure`` (high SNR).
    Frames are numbered contiguously ``prefix1..prefixN`` so
    ``operations.index_run`` regroups them exactly.  The downstream analysis is
    the SAME on these files as on a real run -- only who wrote them differs.
    """

    from ..operations.imageio import save_frame  # lazy: keep devices->operations off the import graph

    trap = trap_array if trap_array is not None else VirtualTrapArray(
        grid_shape=tuple(grid_shape), loading_probability=float(loading_probability), seed=seed
    )
    data_dir = Path(data_dir).expanduser()
    refs = tuple(int(s) for s in ref_shots)
    spg = positive_int(shots_per_group, "shots_per_group")
    n_groups = positive_int(groups, "groups")
    n = 0
    for _ in range(n_groups):
        trap.reload()  # one atom loading, imaged shots_per_group times below
        for shot in range(1, spg + 1):
            exposure = float(short_exposure) if shot == int(short_shot) else float(reference_exposure)
            frame = trap.render_image(exposure=exposure)
            n += 1
            save_frame(data_dir / f"{prefix}{n}{suffix}", frame)
    return {
        "folder": str(data_dir), "prefix": str(prefix), "n_frames": n, "groups": n_groups,
        "shots_per_group": spg, "short_shot": int(short_shot), "ref_shots": refs,
        "grid_shape": tuple(trap.grid_shape),
    }


def virtual_config() -> dict[str, object]:
    return {
        "trap_array": {"type": "VirtualTrapArray"},
        "camera": {"type": "VirtualCamera", "params": {"trap_array": "$device:trap_array"}},
        "sequencer": {"type": "VirtualSequencer"},
    }


def virtual_config_with_overrides(
    *,
    trap_array: dict[str, object] | None = None,
    sitemap: dict[str, object] | None = None,
    camera: dict[str, object] | None = None,
    sequencer: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Translate ``connect("virtual", ...)`` convenience kwargs into a device
    config + inferred session defaults.

    The virtual backend OWNS this mapping (its own field names, aliases and the
    ``loss_rate``/``sitemap`` conveniences) so the orchestration layer never has
    to know about ``VirtualTrapArray`` internals -- it just asks the registry to
    resolve the named backend.  Returns ``(config_dict, defaults_dict)``.
    """

    cfg = virtual_config()
    trap_params = dict(trap_array or {})
    sitemap_params = dict(sitemap or {})
    camera_params = dict(camera or {})
    sequencer_params = dict(sequencer or {})
    defaults: dict[str, object] = {}
    trap_fields = set(VirtualTrapArray.__dataclass_fields__)
    aliases = {
        "bright_count_rate": "atom_rate",
        "atom_bright_rate": "atom_rate",
        "background_count_rate": "background_rate",
        "load_probability": "loading_probability",
        # Release-recapture loss model conveniences (the demo temperature curve).
        "temperature_K": "recapture_temperature_K",
        "recapture_temperature": "recapture_temperature_K",
        "capture_radius": "capture_radius_m",
    }
    for key, value in sitemap_params.items():
        target = aliases.get(key, key)
        if target in trap_fields:
            trap_params[target] = value
        elif key in {"roi_radius", "sitemap_exposure", "detection_times"}:
            defaults[key] = value
        else:
            raise TypeError(f"unknown sitemap configuration parameter {key!r}.")
    for key, value in dict(params or {}).items():
        if key == "loss_rate":
            loss_rate = float(value)
            if not np.isfinite(loss_rate) or loss_rate <= 0:
                raise ValueError("loss_rate must be positive and finite.")
            trap_params["detection_lifetime"] = 1.0 / loss_rate
        elif key in {"sitemap_exposure", "detection_times", "roi_radius"}:
            defaults[key] = value
        else:
            target = aliases.get(key, key)
            if target in trap_fields:
                trap_params[target] = value
            elif key in {"exposure", "timeout"}:
                camera_params[key] = value
            elif key in {"clock_hz", "sleep_scale", "channels"}:
                sequencer_params[key] = value
            else:
                raise TypeError(f"unknown virtual configuration parameter {key!r}.")
    cfg["trap_array"].setdefault("params", {}).update(trap_params)
    cfg["camera"].setdefault("params", {}).update(camera_params)
    cfg["sequencer"].setdefault("params", {}).update(sequencer_params)
    return cfg, defaults


def sequence_requests_load(sequence: PulseSequence | None) -> bool:
    if sequence is None:
        return False
    return any(pulse.channel in {"cooling", "mot", "load"} and pulse.value for pulse in sequence.effective_pulses())


# The trap channel whose OFF gaps drive the release-recapture loss model.  The
# release-recapture pulse (operations/temperature.build_release_recapture_pulse)
# and the virtual config both name it "trap"; aliases cover common variants.
DEFAULT_TRAP_CHANNELS = ("trap", "tweezer", "dipole")


def trap_off_durations_per_frame(
    sequence: PulseSequence | None,
    frames: int,
    *,
    trigger_channels: Sequence[str] | None = None,
    trap_channels: Sequence[str] = DEFAULT_TRAP_CHANNELS,
) -> list[float]:
    """Trap-off duration (s) that PRECEDES each acquired frame.

    Parses the SAME ``PulseSequence`` the analysis layer fired.  Frames are bounded
    by camera-trigger rises (``trigger_channels``); within the window between
    trigger ``k-1`` and trigger ``k`` we look at the trap channel
    (``trap_channels``) ON intervals and return the largest contiguous OFF gap --
    that is ``t_off`` for the release-recapture pulse, whose trap-off period sits
    exactly there.  The returned list has length ``frames``; entry 0 is always 0.0
    (nothing precedes the first frame), and any frame with no intervening trap-off
    (ordinary imaging, where the trap is held ON the whole time) is 0.0.

    This is the data source's view of the hardware timing -- the only thing the
    virtual backend reads to reproduce trap-off atom loss.  On real hardware the
    physics does this for free; here we model it from the fired sequence so the
    virtual survival curve is non-flat.
    """

    out = [0.0] * int(frames)
    if sequence is None or frames < 2 or not hasattr(sequence, "effective_pulses"):
        return out

    trig_set = {str(c) for c in (DEFAULT_CAMERA_TRIGGER_CHANNELS if trigger_channels is None else trigger_channels)}
    trap_set = {str(c) for c in trap_channels}
    pulses = sequence.effective_pulses()

    # Camera-trigger rise times define the frame boundaries.
    trigger_starts = sorted(p.start for p in pulses if p.value and p.channel in trig_set)
    if len(trigger_starts) < 2:
        return out

    # Trap ON intervals (merged is unnecessary: gaps are computed from sorted edges).
    trap_intervals = sorted((p.start, p.stop) for p in pulses if p.value and p.channel in trap_set)

    n = min(int(frames), len(trigger_starts))
    for k in range(1, n):
        window_lo = trigger_starts[k - 1]
        window_hi = trigger_starts[k]
        # Largest trap-off gap strictly inside (window_lo, window_hi): the union of
        # trap-ON intervals leaves gaps; the trap-off period is the biggest one.
        cursor = window_lo
        biggest_gap = 0.0
        for start, stop in trap_intervals:
            if stop <= window_lo or start >= window_hi:
                continue
            clipped_start = max(start, window_lo)
            if clipped_start > cursor:
                biggest_gap = max(biggest_gap, clipped_start - cursor)
            cursor = max(cursor, min(stop, window_hi))
        if window_hi > cursor:
            biggest_gap = max(biggest_gap, window_hi - cursor)
        out[k] = float(biggest_gap)
    return out


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
    "DEFAULT_TRAP_CHANNELS",
    "VirtualCamera",
    "VirtualSequencer",
    "VirtualTrapArray",
    "trap_off_durations_per_frame",
    "virtual_config",
    "virtual_config_with_overrides",
    "virtual_loading_feed",
    "write_virtual_run",
]
