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
from .base import CameraDevice, SequencerDevice, TrapArrayDevice, snap_subarray
from ..timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    PulseSequence,
    exposure_from_sequence,
    imaging_channel_kwargs,
    imaging_sequence,
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
    """A PULSE-DRIVEN virtual atom array -- the fake data source.

    Every camera shot is one experimental cycle: the atoms are first (re)loaded by
    the cooling/MOT light (stochastic ~50% single-atom loading per tweezer) and
    cooled by PGC to ``cooled_temperature_K``; the fired :class:`PulseSequence`
    then EVOLVES per-site occupancy + temperature frame by frame -- cooling pulses
    re-cool, the probe heats while it scatters the photons the camera sees, a
    trap-off gap releases the atoms ballistically (release-recapture loss that
    depends on the CURRENT temperature), and a pushout pulse ejects them.  So
    loading-rate, temperature (release-recapture) and readout-fidelity measurements
    all emerge from the SAME physics a real run would show -- recovered only through
    the rendered camera frames (the analysis layer never reads this state, exactly
    as on hardware).  Every field below is a tunable constant: edit it here, or pass
    it through ``na.connect("virtual", sitemap={...})`` / the device config.
    """

    grid_shape: tuple[int, int] = (5, 7)
    image_shape: tuple[int, int] = (96, 128)
    spacing_px: float = 12.0
    origin_px: tuple[float, float] | None = None
    # --- Loading + camera signal/noise model ---------------------------------
    # Loading SATURATES with the cooling/MOT time: a tweezer fills (up to the
    # light-assisted-collision ceiling ``loading_probability`` ~ 50%) with
    # probability ``loading_probability * (1 - exp(-t_cool / load_time_constant_s))``,
    # so a TOO-SHORT cooling pulse loads few atoms and a long one saturates -- editing
    # the virtual pulse's cooling duration genuinely changes the camera image.
    loading_probability: float = 0.5      # MAX single-atom loading per tweezer (collisional ceiling)
    load_time_constant_s: float = 0.5e-3  # cooling time to reach 1-1/e of the loading ceiling
    # A real experiment cycle is dominated by the MOT/PGC LOAD (hundreds of ms), not the
    # ~ms readout: the default live-monitor cycle uses this as its cooling/MOT-load
    # duration so one virtual shot REPRESENTS a realistic ~mot_load_s cycle (with
    # ``sleep_scale>0`` it also TAKES that long on the wall clock).  A short cooling pulse
    # still loads fewer atoms (loading_fraction), but a full MOT load saturates the
    # ~loading_probability ceiling.
    mot_load_s: float = 0.30
    atom_rate: float = 3_000.0            # bright-atom photon (count) rate during the probe
    background_rate: float = 8.0          # stray-light count rate (always present)
    dark_current_e_per_s: float = 0.006
    offset_counts: float = 200.0
    conversion_e_per_count: float = 0.107
    read_noise_e: float = 0.43
    atom_sigma_px: float = 1.35           # imaging PSF width (Gaussian sigma, px)
    # Atom 1/e lifetime UNDER the probe (s): a longer readout exposure scatters MORE
    # photons (higher SNR) but also loses MORE atoms mid-readout, so the readout
    # duration sets the real SNR-vs-survival trade-off a detection-time scan measures.
    # ~2 s is a realistic trap-imaging lifetime: a 20 ms readout loses ~1% (a clean
    # bimodal readout, fidelity ~99%), while a detection-time scan out to ~100 ms still
    # shows a visible survival roll-off -- the real SNR-vs-loss trade-off.
    detection_lifetime: float = 2.0
    # --- Cooling / heating model (sets the temperature release-recapture sees) -
    # An atom is loaded warm (``mot_temperature_K``) and PGC-cooled toward
    # ``cooled_temperature_K``; the probe (imaging light) heats it at
    # ``probe_heating_K_per_s`` whenever it is on WITHOUT cooling, and any cooling
    # phase pulls the temperature back toward the cooled floor with time constant
    # ``cooling_tau_s``.  A temperature scan (release-recapture vs trap-off time)
    # therefore recovers ``cooled_temperature_K`` -- raise the heating rate and you
    # see a hotter fitted temperature, exactly as a real heating pulse would give.
    mot_temperature_K: float = 250e-6
    cooled_temperature_K: float = 50e-6
    cooling_tau_s: float = 1.0e-3
    probe_heating_K_per_s: float = 0.0    # default 0 -> readout does not bias the temperature
    # --- Release-recapture loss model (data-source side only) -----------------
    # When a fired sequence switches the trap OFF for ``t_off`` seconds between
    # two camera exposures, atoms fly ballistically and are lost unless they stay
    # within the capture region; the survival probability falls off with ``t_off``.
    # The model is the standard ballistic release-recapture survival (3-D
    # isotropic, point source): per axis an atom is recaptured if its speed obeys
    # ``|v| < r_c / t_off``, so for a Maxwell-Boltzmann spread
    # ``sigma_v = sqrt(k_B T / m)`` the per-axis fraction is ``erf(r_c / (sqrt2
    # sigma_v t_off))`` and 3-D survival is that cubed.  The velocity spread uses
    # the atom's CURRENT temperature (``cooled_temperature_K`` after PGC, raised by
    # any probe heating) and ``capture_radius_m`` the trap geometry; together they
    # fix the characteristic ``t_off`` scale.  Defaults give a clearly non-flat demo
    # curve over ~0..300 us: survival is ~1 near t_off=0 and falls smoothly through
    # ~0.5 around 75 us to a few % by 300 us (half-survival scale = r_c / sqrt(2) sigma_v).
    capture_radius_m: float = 6.0e-6
    recapture_mass_kg: float = _RB87_MASS
    seed: int | None = 7
    occupancy: np.ndarray | None = None
    rng: np.random.Generator = field(init=False, repr=False)
    # Per-atom temperature (K), evolved by the pulse (cooled on load + cooling
    # phases, heated by the probe); seeded to ``cooled_temperature_K`` on reload.
    temperature_K: np.ndarray = field(init=False, repr=False)

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
        self.load_time_constant_s = positive_float(self.load_time_constant_s, "load_time_constant_s")
        self.mot_load_s = positive_float(self.mot_load_s, "mot_load_s")
        self.detection_lifetime = positive_float(self.detection_lifetime, "detection_lifetime")
        self.mot_temperature_K = positive_float(self.mot_temperature_K, "mot_temperature_K")
        self.cooled_temperature_K = positive_float(self.cooled_temperature_K, "cooled_temperature_K")
        self.cooling_tau_s = positive_float(self.cooling_tau_s, "cooling_tau_s")
        self.probe_heating_K_per_s = nonnegative_float(self.probe_heating_K_per_s, "probe_heating_K_per_s")
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

    def loading_fraction(self, cooling_duration: float | None = None) -> float:
        """Single-atom loading probability per tweezer for a cooling/MOT pulse of
        ``cooling_duration`` s: it SATURATES toward the collisional ceiling
        ``loading_probability`` as ``1 - exp(-t / load_time_constant_s)``, so a short
        cooling pulse loads few atoms.  ``None`` = assume a fully-saturated load (the
        bare ``reload()`` / file-writer path)."""
        if cooling_duration is None:
            return self.loading_probability
        t = max(0.0, float(cooling_duration))
        return self.loading_probability * (1.0 - float(np.exp(-t / self.load_time_constant_s)))

    def reload(self, *, cooling_duration: float | None = None) -> np.ndarray:
        """Load a FRESH atom array -- the cooling/MOT light + PGC at the start of a
        shot.  Each tweezer independently captures a single atom with probability
        :meth:`loading_fraction` (which grows with ``cooling_duration``), and every
        loaded atom starts PGC-cooled to ``cooled_temperature_K``."""
        self.occupancy = self.rng.random(self.n_sites) < self.loading_fraction(cooling_duration)
        self.temperature_K = np.full(self.n_sites, self.cooled_temperature_K, dtype=float)
        self._pinned = False                   # a fresh stochastic loading clears any manual pin
        return self.occupancy.copy()

    def set_occupancy(self, occupied: Sequence[int] | np.ndarray) -> None:
        """Force a SPECIFIC loading (a deterministic test / debug override).  The next
        shot images exactly this occupancy instead of a fresh random loading -- a
        ONE-SHOT pin (the shot after that reloads stochastically as usual)."""
        arr = np.asarray(occupied)
        if arr.dtype == bool:
            flat = arr.reshape(-1)
            if flat.size != self.n_sites:
                raise ValueError(f"boolean occupancy must have length {self.n_sites}.")
            self.occupancy = flat.astype(bool, copy=True)
        else:
            out = np.zeros(self.n_sites, dtype=bool)
            for value in np.asarray(occupied).reshape(-1):
                index = site_index(value, self.n_sites)
                out[index] = True
            self.occupancy = out
        self.temperature_K = np.full(self.n_sites, self.cooled_temperature_K, dtype=float)
        self._pinned = True                    # image THIS loading on the next shot, then resume

    def consume_pin(self) -> bool:
        """Whether the next shot should image the manually-:meth:`set_occupancy` loading
        instead of reloading; consumes the one-shot pin."""
        pinned = bool(getattr(self, "_pinned", False))
        self._pinned = False
        return pinned

    def cool(self, duration: float) -> None:
        """A cooling/PGC phase of ``duration`` s relaxes every atom's temperature
        toward the cooled floor with time constant ``cooling_tau_s`` (so a long
        cooling pulse fully re-cools an atom heated by a prior probe)."""
        dt = float(duration)
        if dt <= 0.0:
            return
        factor = float(np.exp(-dt / self.cooling_tau_s))
        self.temperature_K = self.cooled_temperature_K + (self.temperature_K - self.cooled_temperature_K) * factor

    def heat(self, duration: float) -> None:
        """A probe (imaging-light) phase of ``duration`` s recoil-heats every atom
        at ``probe_heating_K_per_s`` (0 by default -> readout is temperature-neutral)."""
        dt = float(duration)
        if dt <= 0.0 or self.probe_heating_K_per_s <= 0.0:
            return
        self.temperature_K = self.temperature_K + self.probe_heating_K_per_s * dt

    def _render(self, occupancy: np.ndarray, signal_time: np.ndarray, exposure: float) -> np.ndarray:
        """Render ONE camera frame from a site occupancy + per-site bright-scatter
        time (s).  Background scales with the frame ``exposure``; each occupied
        site adds a Gaussian PSF of amplitude ``atom_rate * signal_time[site]``;
        Poisson shot noise + read noise + offset as a real EMCCD/qCMOS frame."""
        h, w = self.image_shape
        yy, xx = np.mgrid[0:h, 0:w]
        expected_e = np.full((h, w), (self.background_rate + self.dark_current_e_per_s) * exposure, dtype=float)
        occ = np.asarray(occupancy, dtype=bool).reshape(-1)
        st = np.asarray(signal_time, dtype=float).reshape(-1)
        for (cx, cy), occupied, t_sig in zip(self._site_centers(), occ, st):
            if not occupied or t_sig <= 0.0:
                continue
            amplitude = self.atom_rate * float(t_sig)
            expected_e += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * self.atom_sigma_px**2))
        photoelectrons = self.rng.poisson(np.clip(expected_e, 0, None)).astype(float)
        noisy = photoelectrons / self.conversion_e_per_count + self.offset_counts
        if self.read_noise_e > 0:
            noisy += self.rng.normal(0.0, self.read_noise_e / self.conversion_e_per_count, size=noisy.shape)
        return np.clip(noisy, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def render_image(self, *, exposure: float, all_sites: bool = False) -> np.ndarray:
        """Image the CURRENT atom array under the probe for ``exposure`` s.

        Each occupied atom scatters for ``min(exposure, lifetime)`` where the
        light-assisted ``lifetime ~ Exp(detection_lifetime)``; an atom whose
        lifetime runs out mid-exposure is lost (drops from the occupancy that a
        FOLLOWING frame of the SAME loading sees) and the probe recoil-heats the
        survivors.  ``all_sites=True`` forces every site bright for the full
        exposure (the sitemap template) and leaves the occupancy untouched."""
        exposure = positive_float(exposure, "exposure")
        occupancy_for_frame = np.ones(self.n_sites, dtype=bool) if all_sites else self.occupancy.copy()
        signal_time = np.zeros(self.n_sites, dtype=float)
        next_occupancy = self.occupancy.copy()
        for site, occupied in enumerate(occupancy_for_frame):
            if not occupied:
                continue
            if all_sites:
                signal_time[site] = exposure
            else:
                lifetime = self.rng.exponential(self.detection_lifetime)
                signal_time[site] = min(exposure, lifetime)
                if lifetime < exposure:
                    next_occupancy[site] = False
        image = self._render(occupancy_for_frame, signal_time, exposure)
        if not all_sites:
            self.occupancy = next_occupancy
            self.heat(exposure)               # the probe recoil-heats the imaged atoms
        return image

    def recapture_survival_probability(self, t_off: float, *, temperature_K: float | None = None) -> float:
        """Ballistic release-recapture survival probability after a trap-off of ``t_off`` s.

        Per axis an atom is recaptured if its ballistic displacement stays inside
        the capture radius ``r_c``, i.e. ``|v| < r_c / t_off``; for a 1-D thermal
        spread ``sigma_v = sqrt(k_B T / m)`` that fraction is
        ``erf(r_c / (sqrt(2) sigma_v t_off))``, and the isotropic 3-D survival is
        the cube.  ``t_off <= 0`` -> 1.0 (the atom cannot move).  ``temperature_K``
        defaults to the cooled floor; the per-atom :meth:`apply_recapture_loss` uses
        each atom's CURRENT temperature.  This is the SAME physics the analysis-side
        fit assumes, so a fitted temperature recovers a value near the modelled
        temperature -- but it lives ONLY here, in the data source."""
        t = float(t_off)
        if not np.isfinite(t) or t <= 0.0:
            return 1.0
        temperature = self.cooled_temperature_K if temperature_K is None else float(temperature_K)
        sigma_v = sqrt(_K_B * max(temperature, 1e-12) / self.recapture_mass_kg)
        arg = self.capture_radius_m / (sqrt(2.0) * sigma_v * t)
        return float(erf(arg) ** 3)

    def apply_recapture_loss(self, t_off: float) -> np.ndarray:
        """Randomly drop currently-occupied atoms with the release-recapture model,
        using EACH atom's current temperature (a hotter atom is lost more readily).

        Mutates ``self.occupancy`` in place (the same atom loading is imaged again
        afterward) and returns the new occupancy."""
        t = float(t_off)
        if not np.isfinite(t) or t <= 0.0:
            return self.occupancy.copy()
        sigma_v = np.sqrt(_K_B * np.maximum(self.temperature_K, 1e-12) / self.recapture_mass_kg)
        arg = self.capture_radius_m / (sqrt(2.0) * sigma_v * t)
        p_survive = np.array([erf(a) for a in arg], dtype=float) ** 3
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
            "cooled_temperature_K": self.cooled_temperature_K,
            "capture_radius_m": self.capture_radius_m,
        }

    def close(self) -> None:
        pass


class VirtualCamera(CameraDevice):
    def __init__(self, trap_array: VirtualTrapArray, exposure: float = 20e-3, timeout: float = 2.0,
                 subarray_step: int = 4):
        self.trap_array = trap_array
        self._exposure = positive_float(exposure, "exposure")
        self.timeout = positive_float(timeout, "timeout")
        self.last_sequence: str | None = None
        # Mirror the real qCMOS sub-array: a ROI is snapped to a hardware grid
        # (the Hamamatsu step is 4) and the rendered frame is CROPPED to it, so the
        # virtual path exercises the SAME ROI contract a real camera does -- a ROI
        # bug then shows up in a virtual test, and switching to real changes only
        # connect().  None = full frame (the default; all existing behaviour).
        self.subarray_step = int(subarray_step)
        self._roi: tuple[int, int, int, int] | None = None

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = positive_float(value, "exposure")

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        # the ACTUALLY-applied (snapped) window, like the real camera's read-back
        return self._roi

    def configure(self, *, exposure: float | None = None, roi: object = None, **_) -> None:
        if exposure is not None:
            self.exposure = positive_float(exposure, "exposure")
        if roi is not None:
            if roi in ("", "None"):
                self._roi = None
            else:
                h, w = self.trap_array.image_shape
                self._roi = snap_subarray(tuple(roi), step=self.subarray_step, max_w=w, max_h=h)

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
        trigger_channels = getattr(sequencer, "trigger_channels", None)
        # Infer the imaging channels the same single source the real adapter uses (probe
        # -> ch03 / cooling -> ch00 on a chNN sequencer), so virtual and real track the
        # SAME timing.
        channel_kwargs = imaging_channel_kwargs(sequencer)
        probe_channel = channel_kwargs.get("probe_channel", "probe")
        # The live monitor calls acquire() with NO sequence: a real camera is always
        # triggered by SOME pulse, so synthesize the default "cool + image" cycle and
        # run the SAME load->image physics (a fresh ~50% loading imaged once).
        effective_sequence = (
            sequence if sequence is not None
            else imaging_sequence(exposure=self.exposure, load=True, name="live",
                                  cooling=self.trap_array.mot_load_s, **channel_kwargs)
        )
        # Expand to the requested frame count (one trigger -> N repeats) so the per-frame
        # analysis below sees ONE trigger window per frame -- the exact runtime sequence
        # the sequencer fires.
        kw = {"trigger_channels": trigger_channels} if trigger_channels is not None else {}
        runtime_sequence = sequence_for_frame_count(effective_sequence, frames, **kw)
        if sequence is not None and sequencer is not None:
            sequencer.prepare(runtime_sequence)
            sequencer.fire(runtime_sequence)
        exposure = exposure_from_sequence(effective_sequence, default=self.exposure, channel=probe_channel)
        # "All sites loaded" (the sitemap template) is requested explicitly via
        # ``force_all_sites``; the legacy fallback keys off the sequence name.
        all_sites = (
            bool(force_all_sites) if force_all_sites is not None
            else (sequence is not None and sequence.name == "sitemap")
        )
        # ---- PULSE-DRIVEN ATOM PHYSICS (data-source side only) ----------------
        # Each camera frame is bounded by a camera-trigger rise.  In the window BEFORE
        # a frame the fired pulse decides what happens to the atoms:
        #   * a cooling/MOT pulse (re)LOADS a fresh ~50% array, PGC-cooled to the cooled
        #     floor -- so an imaging shot, and each repeat of a threshold-calibration
        #     sequence, is an INDEPENDENT loading;
        #   * a trap-off gap RELEASES the current atoms ballistically (release-recapture
        #     loss against their CURRENT temperature) -- the SAME loading imaged again,
        #     as a temperature scan needs;
        #   * the FIRST frame always starts from a fresh shot loading.
        # render_image then images the survivors (probe scatter + readout loss + recoil
        # heating).  No analysis code learns of this -- it only receives the frames.
        # Cooling DURATION per frame (not just presence): a longer cooling/MOT pulse
        # loads more atoms (loading_fraction), so editing the pulse's cooling time
        # genuinely changes the loading you image.
        cooling_durations = cooling_durations_per_frame(runtime_sequence, frames, trigger_channels=trigger_channels)
        trap_off_per_frame = trap_off_durations_per_frame(runtime_sequence, frames, trigger_channels=trigger_channels)
        images: list[np.ndarray] = []
        for frame_index in range(frames):
            if not all_sites:
                cool_dt = cooling_durations[frame_index]
                if frame_index == 0:
                    # Each shot starts from a fresh loading -- UNLESS the caller pinned a
                    # specific occupancy (set_occupancy, a deterministic test/debug), which
                    # this one shot images instead of reloading.  The loading scales with the
                    # shot's cooling time (None -> saturated when the pulse has no cooling phase).
                    if not self.trap_array.consume_pin():
                        self.trap_array.reload(cooling_duration=(cool_dt if cool_dt > 0.0 else None))
                elif cool_dt > 0.0:
                    self.trap_array.reload(cooling_duration=cool_dt)  # fresh independent loading (cooling-time scaled)
                elif trap_off_per_frame[frame_index] > 0.0:
                    self.trap_array.apply_recapture_loss(trap_off_per_frame[frame_index])  # same atoms, released
            image = self.trap_array.render_image(exposure=exposure, all_sites=all_sites)
            if self._roi is not None:
                # crop to the applied sub-array, exactly as a real camera reads out
                # only its ROI -- so the displayed frame IS the ROI region (x, w, y, h)
                x, w, y, h = self._roi
                image = image[y:y + h, x:x + w]
            images.append(image)
        if sequence is not None and sequencer is not None:
            wait_done = getattr(sequencer, "wait_done", None)
            if callable(wait_done) and not wait_done(max(self.timeout, getattr(runtime_sequence, "duration", 0.0) * 2.0 + 1.0)):
                raise TimeoutError("virtual sequencer did not report done.")
        self.last_sequence = None if sequence is None else sequence.name
        return self._retain(images)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self._roi,
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
        # Atom-temperature conveniences: the cooled (PGC) floor the atoms reach, which
        # the release-recapture / temperature scan recovers.
        "temperature_K": "cooled_temperature_K",
        "recapture_temperature": "cooled_temperature_K",
        "pgc_temperature": "cooled_temperature_K",
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


# Channels whose ON pulse (re)LOADS a fresh, PGC-cooled atom array -- the cooling /
# MOT / PGC light.  A frame window that contains one of these starts an INDEPENDENT
# loading (aliases cover the names a user might give the loading light in a pulse table).
COOLING_CHANNELS = ("cooling", "mot", "pgc", "load")

# The trap channel whose OFF gaps drive the release-recapture loss model.  The
# release-recapture pulse (operations/temperature.build_release_recapture_pulse)
# and the virtual config both name it "trap"; aliases cover common variants.
DEFAULT_TRAP_CHANNELS = ("trap", "tweezer", "dipole")


def cooling_durations_per_frame(
    sequence: PulseSequence | None,
    frames: int,
    *,
    trigger_channels: Sequence[str] | None = None,
    cooling_channels: Sequence[str] = COOLING_CHANNELS,
) -> list[float]:
    """Cooling/MOT ON-duration (s) in the window that precedes each frame.

    Frames are bounded by camera-trigger rises (``trigger_channels``); entry ``k`` is
    the total cooling-channel ON time in the window ending at trigger ``k`` (for
    ``k == 0`` the window runs from the sequence start to the first trigger).  A
    non-zero entry means the pulse (re)LOADS a fresh array before that frame, and the
    DURATION sets how full that loading is (:meth:`VirtualTrapArray.loading_fraction`)
    -- so a longer cooling pulse loads more atoms.  Data-source side only; the analysis
    layer never reads it (it only receives the rendered frames)."""
    out = [0.0] * int(frames)
    if sequence is None or not hasattr(sequence, "effective_pulses"):
        return out
    trig_set = {str(c) for c in (DEFAULT_CAMERA_TRIGGER_CHANNELS if trigger_channels is None else trigger_channels)}
    cool_set = {str(c) for c in cooling_channels}
    pulses = sequence.effective_pulses()
    trigger_starts = sorted(p.start for p in pulses if p.value and p.channel in trig_set)
    if not trigger_starts:
        return out
    cooling_intervals = sorted((p.start, p.stop) for p in pulses if p.value and p.channel in cool_set)
    n = min(int(frames), len(trigger_starts))
    for k in range(n):
        window_lo = 0.0 if k == 0 else trigger_starts[k - 1]
        window_hi = trigger_starts[k]
        # sum the cooling-ON overlap with this frame's window
        out[k] = float(sum(max(0.0, min(stop, window_hi) - max(start, window_lo))
                           for start, stop in cooling_intervals
                           if start < window_hi and stop > window_lo))
    return out


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
    "COOLING_CHANNELS",
    "DEFAULT_CHANNELS",
    "DEFAULT_TRAP_CHANNELS",
    "VirtualCamera",
    "VirtualSequencer",
    "VirtualTrapArray",
    "cooling_durations_per_frame",
    "trap_off_durations_per_frame",
    "virtual_config",
    "virtual_config_with_overrides",
    "write_virtual_run",
]
