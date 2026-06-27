"""Virtual devices for offline notebook tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, erf, radians, sin, sqrt
from pathlib import Path
from typing import Sequence
import time

import numpy as np

from ..core.analysis import finite_float, grid_shape_tuple, positive_int
from ..core.utils import site_index
from .base import AODDevice, CameraDevice, SequencerDevice, TrapArrayDevice, snap_subarray
from ..timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    PulseSequence,
    PulseTableState,
    count_trigger_pulses,
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

# The virtual backend is a REAL-TIME hardware simulator: firing a pulse program TAKES its
# real wall-clock duration, exactly like the FPGA + externally-triggered camera.  So a live
# 2D monitor updates at the pulse cadence -- set the imaging period to several seconds and the
# image visibly slows to one frame every several seconds (``sleep_scale=1.0``).  The pytest
# suite fast-forwards this (conftest sets ``DEFAULT_SLEEP_SCALE = 0.0``) so the SAME data path
# runs without the wall-clock waits; only the time pacing is skipped, never the physics/data.
DEFAULT_SLEEP_SCALE = 1.0


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
    # Site pitch (px).  The real Rb87 v16 3 ms data has a tweezer pitch of ~9.2 px on the
    # 5x7 grid (dx 8.9 / dy 9.5 px); 9.0 matches it to <2%.
    spacing_px: float = 9.0
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
    # Bright-atom photoelectron rate (peak amplitude/s) during the probe.  Tuned to the REAL
    # Rb87 qCMOS dataset (references/.../rb87_readout_v16): a bright site's 3x3-box signal is
    # ~18-19 detected photons at the 3 ms reference readout (35-site mean 18.7 ph), and with
    # atom_sigma_px=0.7 this rate renders an all-bright 3 ms frame at that box level.
    atom_rate: float = 2_150.0
    # Per-site COLLECTION-efficiency variation (relative standard deviation, 1-sigma):
    # a real Rb87 tweezer array has different effective NA / aberration / vacuum-window
    # transmission per site, so each site's bright count rate scales by an independently
    # drawn ``Lognormal(0, sigma)`` multiplier around 1.0.  This is what makes a per-site
    # PSF kernel materially better than one shared box / one uniform PSF: the per-site
    # spot brightness varies, so a fixed box averages out signal a per-site kernel can
    # match.  Default 0.18 (=18% RSD) is a realistic spread on a moderate tweezer array.
    site_efficiency_sigma: float = 0.18
    # Stray-light + scatter floor (detected photons/s/pixel, always present).  The real v16
    # 3 ms corner-median is ~16.5 counts/px above the 200-count offset -> ~1.8 ph/px/3 ms ->
    # ~600 ph/s/px, so dark sites carry a realistic shot-noise floor (not a near-zero one).
    background_rate: float = 600.0
    dark_current_e_per_s: float = 0.006
    offset_counts: float = 200.0          # qCMOS nominal offset (reference CameraConfig)
    conversion_e_per_count: float = 0.107  # electrons per count (reference qCMOS gain)
    read_noise_e: float = 0.43            # read noise e-rms (reference qCMOS)
    # Imaging PSF (px).  A REAL atom image is NOT a perfect symmetric Gaussian: the Rb87 v16
    # per-site PSF fits give sigma_x ~0.61 / sigma_y ~0.76 px (anisotropic, geometric mean
    # ~0.68 px) and the empirical spot is skewed by optical aberration (coma).  So the virtual
    # spot is an ANISOTROPIC, ROTATED, SKEWED kernel: ``atom_sigma_px`` is the geometric-mean
    # spot size (it ANCHORS the total brightness -- the kernel is renormalized to the same
    # integral as an isotropic Gaussian of this sigma, so box counts are unchanged);
    # ``atom_psf_aspect`` = sigma_y/sigma_x (=0.76/0.61), ``atom_psf_angle_deg`` tilts the
    # ellipse, and ``atom_psf_skew`` adds the asymmetric (non-Gaussian) coma tail along the
    # major axis.  Set ``atom_psf_skew=0``, ``atom_psf_aspect=1`` to recover the old symmetric
    # Gaussian.
    atom_sigma_px: float = 0.7
    atom_psf_aspect: float = 1.25         # sigma_y / sigma_x (anisotropic elliptical spot)
    atom_psf_angle_deg: float = 18.0      # tilt of the elliptical/skewed spot (deg)
    atom_psf_skew: float = 0.45           # asymmetry (coma tail) along the major axis; 0 = symmetric
    # Atom 1/e lifetime UNDER the probe (s): a longer readout exposure scatters MORE
    # photons (higher SNR) but also loses MORE atoms mid-readout, so the readout
    # duration sets the real SNR-vs-survival trade-off a detection-time scan measures.
    # ~2 s is a realistic trap-imaging lifetime: a 20 ms readout loses ~1% (a clean
    # bimodal readout, fidelity ~99%), while a detection-time scan out to ~100 ms still
    # shows a visible survival roll-off -- the real SNR-vs-loss trade-off.
    detection_lifetime: float = 2.0
    # Vacuum-limited trap (1/e) lifetime in the DARK (s): background-gas collisions
    # eject a trapped atom over seconds even with no probe light, so any trap-ON hold
    # between two images loses atoms at ``exp(-t_hold / trap_lifetime_s)``.  This is the
    # signal a TRAP-LIFETIME measurement (hold-time scan, no release) recovers, and it
    # adds a realistic slow loss to long dark holds.  Much longer than the imaging
    # lifetime, so a few-ms readout window loses essentially nothing to it (~0.01%).
    trap_lifetime_s: float = 30.0
    # --- Cooling / heating model (sets the temperature release-recapture sees) -
    # A freshly loaded atom starts PGC-cooled at ``cooled_temperature_K``; the probe
    # (imaging light) recoil-heats it at ``probe_heating_K_per_s`` per second of
    # exposure (0 by default -> readout is temperature-neutral).  A temperature scan
    # (release-recapture vs trap-off time) therefore recovers ``cooled_temperature_K``
    # -- raise the heating rate and you see a hotter fitted temperature, exactly as a
    # real heating pulse would give.
    cooled_temperature_K: float = 50e-6
    # Probe recoil heating rate (K/s of probe exposure).  Default 0 models MOLASSES-COOLED
    # fluorescence imaging: the cooling light on during the probe balances photon-recoil
    # heating, so a standard readout is temperature-neutral (this is WHY real imaging uses
    # molasses -- un-cooled imaging at the true recoil rate would heat Rb87 to ~mK in a few
    # ms and eject it).  Raise it to model un-cooled / imperfect-molasses imaging: a probe
    # phase then heats the atoms, which a following release-recapture reads as a hotter
    # temperature (and an un-recooled readout loses them).  Fully wired -- ``render_image``
    # heats by this rate over its exposure; see [[virtual-atom-physics-model]].
    probe_heating_K_per_s: float = 0.0
    # PGC re-cooling 1/e time (s): a cooling pulse applied to ALREADY-loaded atoms (a PGC
    # phase mid-cycle, e.g. re-cool after an un-cooled probe heated them, before release)
    # relaxes their temperature toward ``cooled_temperature_K`` as
    # ``T -> floor + (T-floor) exp(-t_cool / pgc_cool_tau_s)`` -- WITHOUT reloading new
    # atoms (the initial MOT load still reloads).  So "heat then re-cool then release"
    # recovers the cooled floor, exactly as on hardware.
    pgc_cool_tau_s: float = 0.3e-3
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
        self.atom_psf_aspect = positive_float(self.atom_psf_aspect, "atom_psf_aspect")
        self.atom_psf_angle_deg = float(self.atom_psf_angle_deg)
        self.atom_psf_skew = nonnegative_float(self.atom_psf_skew, "atom_psf_skew")
        self.load_time_constant_s = positive_float(self.load_time_constant_s, "load_time_constant_s")
        self.mot_load_s = positive_float(self.mot_load_s, "mot_load_s")
        self.detection_lifetime = positive_float(self.detection_lifetime, "detection_lifetime")
        self.trap_lifetime_s = positive_float(self.trap_lifetime_s, "trap_lifetime_s")
        self.cooled_temperature_K = positive_float(self.cooled_temperature_K, "cooled_temperature_K")
        self.probe_heating_K_per_s = nonnegative_float(self.probe_heating_K_per_s, "probe_heating_K_per_s")
        self.pgc_cool_tau_s = positive_float(self.pgc_cool_tau_s, "pgc_cool_tau_s")
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

    def _site_efficiency(self) -> np.ndarray:
        """Per-site collection-efficiency multiplier (mean ~1.0).  Drawn once per session
        from ``Lognormal(0, site_efficiency_sigma)`` so each site has a stable bright count
        rate -- a uniform PSF kernel averages across this spread while a per-site PSF
        kernel matches each site's own profile, which is precisely why per-site PSF beats
        uniform PSF on a real Rb87 array.  Cached on first call (re-seeded by ``rng``)."""
        cache = getattr(self, "_site_eff_cache", None)
        if cache is not None and cache.size == self.n_sites:
            return cache
        sigma = float(getattr(self, "site_efficiency_sigma", 0.0) or 0.0)
        if sigma <= 0.0:
            eff = np.ones(self.n_sites, dtype=float)
        else:
            # Lognormal with median 1 stays strictly positive; geometric mean = 1, so the
            # average bright count rate is preserved (only the distribution widens).
            eff = np.exp(self.rng.normal(0.0, sigma, size=self.n_sites))
        self._site_eff_cache = eff
        return eff

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

    def apply_moves(self, moves, *, survival: float = 0.98) -> np.ndarray:
        """Execute an AOD rearrangement PLAN on the atom array -- the lowest-layer physics fake for
        ``exp.devices.aod.apply_moves`` (the real AOD instead chirps its RF / DAC ramps to drag the atoms;
        SAME plan, device-swapped, so virtual==real).  Each move drags its atom from ``src`` to ``dst``
        (flat row-major site indices); the atom carries its temperature with it and is lost in transit
        with probability ``1 - survival`` (≈1-5 % per move on a real array).  Pins the result so the NEXT
        image shows the rearranged array (like :meth:`set_occupancy`).  The occupancy remap matches the
        host planner's ``apply_moves_to_occupancy`` oracle, so the predicted defect-free array agrees.

        ``moves`` is a sequence of ``rearrangement.Move`` (or any object with ``.src`` / ``.dst``)."""
        survival = probability(survival, "survival")
        occ = self.occupancy.copy()
        temp = self.temperature_K.copy()
        for m in moves:
            src, dst = int(m.src), int(m.dst)
            if not (0 <= src < self.n_sites and 0 <= dst < self.n_sites):
                raise ValueError(f"move site out of range for {self.n_sites}-site array: {src}->{dst}")
            if not occ[src]:
                continue                                       # nothing to drag (lost on an earlier move)
            occ[src] = False
            if self.rng.random() < survival:
                occ[dst] = True
                temp[dst] = temp[src]                          # the atom keeps its temperature
        self.occupancy = occ
        self.temperature_K = temp
        self._pinned = True                                    # image THIS rearranged array next shot
        return self.occupancy.copy()

    def heat(self, duration: float) -> None:
        """A probe (imaging-light) phase of ``duration`` s recoil-heats every atom
        at ``probe_heating_K_per_s`` (0 by default -> molasses-cooled readout is
        temperature-neutral; raise it to model un-cooled imaging)."""
        dt = float(duration)
        if dt <= 0.0 or self.probe_heating_K_per_s <= 0.0:
            return
        self.temperature_K = self.temperature_K + self.probe_heating_K_per_s * dt

    def cool(self, duration: float) -> None:
        """A PGC cooling phase of ``duration`` s applied to the ALREADY-loaded atoms:
        relax each atom's temperature toward ``cooled_temperature_K`` as
        ``T -> floor + (T - floor) * exp(-duration / pgc_cool_tau_s)`` -- WITHOUT
        reloading.  This is how a re-cool between an (un-cooled) probe and a release
        brings the cloud back to the cooled floor, so "heat then re-cool then release"
        recovers ``cooled_temperature_K`` exactly as on hardware."""
        dt = float(duration)
        if dt <= 0.0:
            return
        floor = self.cooled_temperature_K
        decay = float(np.exp(-dt / self.pgc_cool_tau_s))
        self.temperature_K = floor + (self.temperature_K - floor) * decay

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
        # ANISOTROPIC + ROTATED + SKEWED spot (a real atom image is not a perfect symmetric
        # Gaussian).  geometric-mean sigma = atom_sigma_px is held fixed so the total photons (box
        # counts) are unchanged: sx*sy == atom_sigma_px**2, and the linear skew term integrates to
        # ~0 over the Gaussian, so it only ASYMMETRIZES the shape (coma tail) without adding flux.
        asp = sqrt(max(self.atom_psf_aspect, 1e-6))
        sx = self.atom_sigma_px / asp
        sy = self.atom_sigma_px * asp
        th = radians(self.atom_psf_angle_deg)
        cos_t, sin_t = cos(th), sin(th)
        eff = self._site_efficiency()
        for (cx, cy), occupied, t_sig, e in zip(self._site_centers(), occ, st, eff):
            if not occupied or t_sig <= 0.0:
                continue
            # per-site collection-efficiency multiplier: each tweezer has its OWN bright
            # count rate (NA / window transmission / aberration vary), so a per-site PSF
            # kernel materially beats a uniform one on a real array.
            amplitude = self.atom_rate * float(t_sig) * float(e)
            du = (xx - cx) * cos_t + (yy - cy) * sin_t          # along the major axis
            dv = -(xx - cx) * sin_t + (yy - cy) * cos_t         # minor axis
            core = np.exp(-0.5 * ((du / sx) ** 2 + (dv / sy) ** 2))
            spot = np.clip(core * (1.0 + self.atom_psf_skew * (du / sx)), 0.0, None)
            expected_e += amplitude * spot
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

    def apply_trap_loss(self, hold_duration: float) -> np.ndarray:
        """Drop currently-occupied atoms by VACUUM-LIMITED trap loss over a dark trap-ON
        hold of ``hold_duration`` s: each atom survives with probability
        ``exp(-hold_duration / trap_lifetime_s)`` (background-gas collisions, independent
        of light/temperature).  Mutates ``self.occupancy`` and returns it.  Over a few-ms
        readout this is negligible; over a long dark hold it is the trap-lifetime signal."""
        t = float(hold_duration)
        if not np.isfinite(t) or t <= 0.0:
            return self.occupancy.copy()
        p_survive = float(np.exp(-t / self.trap_lifetime_s))
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
            "probe_heating_K_per_s": self.probe_heating_K_per_s,
            "pgc_cool_tau_s": self.pgc_cool_tau_s,
            "trap_lifetime_s": self.trap_lifetime_s,
            "capture_radius_m": self.capture_radius_m,
        }

    def close(self) -> None:
        pass


def _sequence_triggers_camera(sequence, trigger_channels) -> bool:
    """True if ``sequence`` pulses a camera-trigger (emCCD) channel -- i.e. firing it would
    actually trigger the camera and produce a frame.  A pulse with NO camera trigger leaves
    the camera dark, exactly as on hardware (the camera only reads out on a trigger edge)."""
    trig = {str(c) for c in (DEFAULT_CAMERA_TRIGGER_CHANNELS if trigger_channels is None else trigger_channels)}
    return any(getattr(p, "channel", None) in trig for p in getattr(sequence, "pulses", ()))


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

    @property
    def sensor_shape(self) -> tuple[int, int]:
        # the full sensor (height, width) -- the trap array's image size, known up front,
        # so a raw-frame Edit shows the ROI as the full window even before any sub-array is set
        return tuple(self.trap_array.image_shape)

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
        frames: int | None = None,
        *,
        sequence: PulseSequence | None = None,
        sequencer=None,
        force_all_sites: bool | None = None,
        **_,
    ) -> list[np.ndarray]:
        # frames=None -> one frame per camera trigger the sequence carries (the camera is told
        # the PULSE, never a frame count; see CameraDevice.acquire).  An explicit count is the
        # repeat/live path.
        frames = self._resolve_frame_count(frames, sequence, sequencer)
        trigger_channels = getattr(sequencer, "trigger_channels", None)
        # Infer the imaging channels the same single source the real adapter uses (probe
        # -> ch03 / cooling -> ch00 on a chNN sequencer), so virtual and real track the
        # SAME timing.
        channel_kwargs = imaging_channel_kwargs(sequencer)
        probe_channel = channel_kwargs.get("probe_channel", "probe")
        # The camera is PURELY TRIGGER-DRIVEN, exactly like the real qCMOS: a frame exists
        # only when a camera-trigger edge gates a readout.  There are two -- and only two --
        # trigger sources, and NOTHING is ever fabricated:
        #   * an EXPLICIT sequence (a measurement / capture() fires it and reads), or
        #   * the streamer's CONTINUOUS firing state (the live monitor reads whatever the
        #     FPGA is currently firing).
        # With neither -- no streamer, or the streamer is not firing a camera-triggering pulse
        # (the user hit "Stop Pulse" -> set_safe_state) -- there is NO trigger, so NO frame.
        # This is what makes the live 2D image FREEZE on Stop Pulse instead of fabricating a
        # fresh frame every tick (the bug this fixes).
        if sequence is not None:
            effective_sequence = sequence
        else:
            firing = getattr(sequencer, "firing", None)
            if firing is None or not _sequence_triggers_camera(firing, trigger_channels):
                return self._retain([])          # no streamer firing a camera trigger -> no frame
            effective_sequence = firing
        # Expand to the requested frame count (one trigger -> N repeats) so the per-frame
        # analysis below sees ONE trigger window per frame -- the exact runtime sequence
        # the sequencer fires.
        kw = {"trigger_channels": trigger_channels} if trigger_channels is not None else {}
        runtime_sequence = sequence_for_frame_count(effective_sequence, frames, **kw)
        if sequence is not None and sequencer is not None:
            sequencer.prepare(runtime_sequence)
            sequencer.fire(runtime_sequence)
        # Per-frame integration time: a real externally-triggered camera integrates each
        # frame for the window ITS trigger gates, so a heterogeneous bracket (long-short-long
        # reference) images successive frames for different durations.  For a uniform repeated
        # sequence every entry equals the one exposure (the legacy single-exposure behaviour).
        probe_set = [probe_channel] + (["ch02"] if probe_channel == "probe" else [])
        exposures = exposures_per_frame(runtime_sequence, frames, default=self.exposure,
                                        trigger_channels=trigger_channels, probe_channels=probe_set)
        # "All sites loaded" is an EXPLICIT device-boundary request (``force_all_sites``), NEVER
        # inferred from the sequence NAME.  The old ``sequence.name == "sitemap"`` fallback let the
        # ANALYSIS layer (it chooses that name) secretly switch the sim to an idealized all-bright
        # frame -- a virtual!=real divergence: a real sitemap calibration sees ~50% loading and finds
        # the sites by AVERAGING many frames (calibrate_sitemap_from_images, reducer='mean'), the SAME
        # path virtual now takes.  So no analysis-set name controls what the camera renders.
        all_sites = bool(force_all_sites)
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
        trap_hold_per_frame = trap_hold_durations_per_frame(runtime_sequence, frames, trigger_channels=trigger_channels)
        # A SINGLE multi-trigger cycle (the BASE sequence already carries every trigger -- e.g. a
        # release-recapture bracket) holds ONE atom loading across all its frames, so a cooling
        # phase mid-cycle RE-COOLS those held atoms (no reload).  A single-trigger sequence
        # REPEATED to reach the frame count (e.g. the threshold-calibration pass) is a fresh shot
        # per frame, so its cooling phase RELOADS a new independent array.  Tell them apart by
        # whether the base already has at least ``frames`` camera triggers.
        _trig = DEFAULT_CAMERA_TRIGGER_CHANNELS if trigger_channels is None else trigger_channels
        base_triggers = count_trigger_pulses(effective_sequence, trigger_channels=_trig) if hasattr(effective_sequence, "effective_pulses") else 0
        single_cycle = base_triggers >= frames
        # REAL-TIME live cadence: the live monitor (no explicit sequence -- it reads the
        # streamer's continuous firing) gets ONE frame per trigger CYCLE, and that cycle takes
        # the firing program's per-cycle wall-clock time.  So editing the imaging pulse's period
        # to several seconds genuinely slows the displayed image to that cadence -- the camera
        # is paced BY the pulse, not by the polling worker.  Read the scale off the sequencer
        # (one knob; the test suite sets it to 0 to fast-forward).  The explicit-sequence path
        # instead blocks in ``wait_done`` below, so the wall-clock is never double-counted.
        live_path = sequence is None
        seq_scale = float(getattr(sequencer, "sleep_scale", 0.0) or 0.0)
        per_frame_wall = (float(getattr(runtime_sequence, "duration", 0.0)) / max(frames, 1)) * seq_scale
        images: list[np.ndarray] = []
        for frame_index in range(frames):
            if not all_sites:
                cool_dt = cooling_durations[frame_index]
                trap_off = trap_off_per_frame[frame_index]
                trap_hold = trap_hold_per_frame[frame_index]
                if frame_index == 0:
                    # Each shot starts from a fresh loading -- UNLESS the caller pinned a
                    # specific occupancy (set_occupancy, a deterministic test/debug), which
                    # this one shot images instead of reloading.  The loading scales with the
                    # shot's cooling time (None -> saturated when the pulse has no cooling phase).
                    if not self.trap_array.consume_pin():
                        self.trap_array.reload(cooling_duration=(cool_dt if cool_dt > 0.0 else None))
                elif cool_dt > 0.0:
                    # Mid-cycle cooling RE-COOLS the held atoms (single multi-trigger cycle) or
                    # RELOADS a fresh array (a repeated single-trigger imaging sequence).
                    if single_cycle:
                        self.trap_array.cool(cool_dt)
                    else:
                        self.trap_array.reload(cooling_duration=cool_dt)
                # A trap-off gap RELEASES the current atoms ballistically (recapture loss vs their
                # CURRENT temperature).  Applied AFTER any cooling in the SAME frame (cool THEN
                # release), so "heat -> re-cool -> release" drops the re-cooled cloud, not silently
                # only one effect.  The standard release-recapture bracket has no cooling channel,
                # so there this is just the release.
                if trap_off > 0.0:
                    self.trap_array.apply_recapture_loss(trap_off)
                # A DARK trap-ON hold (trap on, probe off, BETWEEN images) loses atoms to vacuum
                # collisions over its duration -- the trap-lifetime signal.  Skipped on frame 0
                # (that window is the load, accounted for by the loading fraction, not a dark hold).
                if trap_hold > 0.0 and frame_index >= 1:
                    self.trap_array.apply_trap_loss(trap_hold)
            image = self.trap_array.render_image(exposure=exposures[frame_index], all_sites=all_sites)
            if self._roi is not None:
                # crop to the applied sub-array, exactly as a real camera reads out
                # only its ROI -- so the displayed frame IS the ROI region (x, w, y, h)
                x, w, y, h = self._roi
                image = image[y:y + h, x:x + w]
            images.append(image)
            if live_path and per_frame_wall > 0:
                # one frame per trigger cycle takes the cycle's real wall-clock time
                time.sleep(per_frame_wall)
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
    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS, clock_hz: float = 50_000_000.0, sleep_scale: float | None = None):
        self.channels = tuple(str(channel) for channel in channels)
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        # REAL-TIME by default (DEFAULT_SLEEP_SCALE=1.0): a fired program takes its real
        # duration, so the live camera paces with the pulse.  ``None`` -> the module default
        # (the pytest suite flips that default to 0 to fast-forward; see conftest).
        if sleep_scale is None:
            sleep_scale = DEFAULT_SLEEP_SCALE
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        self.history: list[dict[str, object]] = []
        self._prepared: PulseSequence | None = None
        # The compiled program of the last prepare() (parity with RuntimeSequencer.prepare,
        # which on_pulse / sync-to-device read for .sequence_name / .trigger_count / etc.).
        self.last_program = None
        # The SOURCE payload (PulseTableState/PulseSequence as a JSON string) of the last
        # prepare -- the sync-to-device handle the pulse GUI's "Sync" pulls back, exactly like
        # the real SequencerService records.  None until something is prepared.
        self.last_payload_json: str | None = None
        # The program the streamer is CONTINUOUSLY firing (a repeat_forever pulse), or None
        # when idle/safe.  A real FPGA streamer plays a repeat_forever program until it is
        # set to a safe state; the externally-triggered camera produces frames ONLY while
        # this is set.  This is what makes "Stop Pulse" (set_safe_state) freeze the live
        # image: no streamer firing -> no camera trigger -> no frame (exactly like hardware).
        self._firing: PulseSequence | None = None
        # Scan-progress mirror of the real streamer: when a streamed scan is firing, this holds
        # {"n_points": N, "scan_repeats": K, "base_dt": per-point seconds}; scan_progress() derives
        # the current (point, sweep) from the wall-clock elapsed since fire, and a finite scan (K>0)
        # stops itself once K sweeps are played -- exactly like the host issues CMD_SAFE on the real
        # streamer.  None when no scan is firing.
        self._scan_info: dict[str, float] | None = None
        self._scan_fire_time: float = 0.0
        # Latched once a finite scan has played its K sweeps: scan_progress() then keeps returning the
        # SATURATED done reading (not idle) until the next prepare/stop -- matching the real streamer's
        # _scan_finished latch, so virtual == real at the scan's terminal reading too.
        self._scan_done: bool = False

    @property
    def firing(self) -> PulseSequence | None:
        """The repeat_forever program the streamer is continuously playing, or None when
        idle/safe.  The live (no-sequence) camera read is gated on this -- it is the
        software model of "the FPGA is emitting camera triggers right now"."""
        return self._firing

    def prepare(self, sequence: PulseSequence | PulseTableState):
        # Accept either a PulseSequence or a GUI PulseTableState (what bind_pulse/on_pulse and a
        # loaded pulse .json pass).  Compile a table to a PulseSequence so fire()/firing/acquire
        # all work on ONE type, carrying repeat_forever for a continuous (On Pulse) program.
        from .sequencer import compile_runtime_program
        if isinstance(sequence, PulseTableState):
            channels = list(sequence.channels)
            program = sequence.to_sequence(clock_hz=self.clock_hz)
            if bool(getattr(sequence, "repeat_forever", False)):
                program = program.forever()
        else:
            program = sequence
            # The program defines its OWN channels (friendly imaging names, or a saved pulse
            # table's board chNN) -- derive them from the pulses rather than this sequencer's
            # default list, so a virtual streamer plays whatever channel set the pulse uses.
            channels = sorted({p.channel for p in program.base_pulses()}) or list(self.channels)
        # Validate the TIMING against the clock grid (the channel set is intrinsic to the
        # program), then compile to a RuntimeSequenceProgram so prepare() returns the SAME object
        # the real RuntimeSequencer does (on_pulse / sync-to-device read it).
        program.validate(clock_hz=self.clock_hz).raise_if_failed()
        self._prepared = program
        self.last_program = compile_runtime_program(
            program, channels=channels, clock_hz=self.clock_hz,
            trigger_channels=getattr(self, "trigger_channels", DEFAULT_CAMERA_TRIGGER_CHANNELS))
        # Record the SOURCE timing as a syncable PulseTableState JSON (always carries
        # ``periods``), EXACTLY like SequencerService -- the pulse GUI's "Sync" reads it back
        # from snapshot()["last_payload_json"].  A bare PulseSequence (a Task / measurement
        # firing a compiled bracket) is reconstructed into a period table via
        # PulseTableState.from_sequence so it syncs too (virtual == real: no fired timing the
        # GUI "cannot sync").
        self._record_source_payload(sequence, channels=channels)
        # Capture scan-progress info from the SOURCE table (the compiled PulseSequence drops the
        # scan_table): N points + the requested sweep count, plus a per-point wall-clock estimate
        # (the program's one-frame duration) so scan_progress() can report "point K / N · sweep r".
        self._scan_done = False          # a fresh prepare clears any prior finite-scan done latch
        rows = list(getattr(sequence, "scan_table", None) or []) if isinstance(sequence, PulseTableState) else []
        if rows:
            self._scan_info = {
                "n_points": float(len(rows)),
                "scan_repeats": float(max(0, int(getattr(sequence, "scan_repeats", 0)))),
                "base_dt": float(max(program.duration, 1e-9)),
            }
        else:
            self._scan_info = None
        self.history.append({"action": "prepare", "sequence": program.name, "duration": program.duration})
        return self.last_program

    def _record_source_payload(self, payload, *, channels) -> None:
        import json
        from .sequencer import timing_from_payload
        step = 1e9 / float(self.clock_hz)
        timing = timing_from_payload(payload)
        if isinstance(timing, PulseTableState):
            table = timing.snapped(time_step_ns=step)
        else:
            table = PulseTableState.from_sequence(timing, channels=channels, clock_hz=self.clock_hz)
        self.last_payload_json = json.dumps(table.to_dict())

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self._prepared is None:
            raise RuntimeError("VirtualSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self._prepared:
            raise RuntimeError("VirtualSequencer.fire() received a sequence that was not prepared.")
        self.history.append({"action": "fire", "sequence": self._prepared.name, "duration": self._prepared.duration})
        # fire() is NON-BLOCKING, exactly like the real FPGA (a quick register write that
        # STARTS playback).  The program's real wall-clock is consumed where the hardware
        # actually blocks: ``wait_done`` (a finite ``on_pulse(wait=True)``) or the camera's
        # per-frame readout (the live monitor) -- never here.  So a continuous (repeat_forever)
        # On Pulse returns at once and the live camera then paces at the firing cadence.
        #
        # A repeat_forever program keeps the streamer firing (continuous camera triggers)
        # until set_safe_state.  A finite program plays ONCE -- a measurement's own acquire
        # fires + reads it within that call -- so it does NOT leave the streamer continuously
        # firing (and must not make a later live read see a stale finite shot).
        if bool(getattr(self._prepared, "repeat_forever", False)):
            self._firing = self._prepared
            self._scan_fire_time = time.monotonic()

    def scan_progress(self) -> dict:
        """Where the streamed scan is now -- mirrors the real streamer (virtual==real).  The
        current monotonic played-point count is the wall-clock elapsed since fire divided by the
        per-point estimate (scaled by ``sleep_scale`` like every other virtual time, so the test
        suite's ``sleep_scale=0`` fast-forwards a finite scan straight to done).  A finite scan
        (scan_repeats>0) that has played its K sweeps stops firing here, exactly like the host
        issues the engine stop on the real backend."""
        from .sequencer import SCAN_PROGRESS_IDLE, scan_progress_fields
        if self._scan_info is None:
            return dict(SCAN_PROGRESS_IDLE)
        n = int(self._scan_info["n_points"])
        k = int(self._scan_info["scan_repeats"])
        if self._scan_done:              # finite scan finished -> latch the SATURATED done reading (== real)
            return scan_progress_fields(max(1, k) * n, n, k)
        if self._firing is None:
            return dict(SCAN_PROGRESS_IDLE)
        dt = float(self._scan_info["base_dt"]) * self.sleep_scale
        if dt <= 0:                      # fast-forward: a finite scan is instantly done, an infinite one sits at point 0
            total = k * n if k > 0 else 0
        else:
            total = int((time.monotonic() - self._scan_fire_time) / dt)
        fields = scan_progress_fields(total, n, k)
        if not fields["scanning"]:       # finite scan reached K sweeps -> the streamer halts (camera stops),
            self._scan_done = True       # but the reading stays latched at done until the next prepare/stop
            self._firing = None
        return fields

    def wait_done(self, timeout: float | None = None) -> bool:
        """Block until the prepared FINITE program finishes on the wall clock
        (``duration x sleep_scale``), mirroring :meth:`SequencerService.wait_done` -- this IS
        the real time a measurement's ``on_pulse(wait=True)`` takes (real-time by default;
        the test suite sets sleep_scale=0 to fast-forward).  A repeat_forever (On Pulse)
        program never finishes, so it returns False (the caller stops it instead)."""
        if self._prepared is None:
            raise RuntimeError("VirtualSequencer.wait_done() called before prepare().")
        if bool(getattr(self._prepared, "repeat_forever", False)):
            # A FINITE scan (scan_repeats=K>0) DOES finish -- after K whole sweeps the streamer
            # halts (the host stops it on the real backend).  Block the K-sweep wall-clock, then
            # report done.  An infinite scan / continuous On Pulse never finishes -> False.
            info = self._scan_info
            if info is not None and int(info["scan_repeats"]) > 0:
                delay = int(info["scan_repeats"]) * int(info["n_points"]) * float(info["base_dt"]) * self.sleep_scale
                if timeout is not None and delay > float(timeout):
                    return False
                if delay > 0:
                    time.sleep(delay)
                self._scan_done = True       # latch the saturated done reading (matches the real backend)
                self._firing = None
                return True
            return False
        delay = float(self._prepared.duration) * self.sleep_scale
        if timeout is not None and delay > float(timeout):
            return False
        if delay > 0:
            time.sleep(delay)
        return True

    def settle(self, seconds: float, *, stop=None) -> None:
        """Idle ``seconds`` between software-stepped fires, scaled by ``sleep_scale`` like
        :meth:`wait_done` -- so the virtual backend takes the same proportional wall-clock as the
        rest of its timing (and the test suite's ``sleep_scale=0`` fast-forwards it to nothing).
        ``stop`` makes the wait cooperatively cancellable (same contract as the base)."""
        self._sleep_interruptible(float(seconds) * self.sleep_scale, stop)

    def stop(self) -> None:
        """Drive the streamer to a safe idle state -- it stops firing, so the camera sees no
        more triggers.  ``abort`` / ``set_safe_state`` (base) route here, so "Stop Pulse"
        from the GUI clears the firing state and the live image freezes."""
        if self._firing is not None:
            self.history.append({"action": "safe_state"})
        self._firing = None
        self._scan_info = None
        self._scan_done = False

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "channels": list(self.channels),
            "clock_hz": self.clock_hz,
            "sleep_scale": self.sleep_scale,
            "runs": sum(1 for row in self.history if row["action"] == "fire"),
            "firing": None if self._firing is None else self._firing.name,
            # the sync-to-device handle: the GUI's "Sync" reconstructs the editor state from
            # this (PulseTableState JSON of the last prepare), same as the real sequencer.
            "last_payload_json": self.last_payload_json,
        }

    def close(self) -> None:
        pass


class VirtualAOD(AODDevice):
    """Virtual crossed-AOD rearrangement actuator -- the lowest-layer fake for ``exp.devices.aod``.

    On real hardware the AOD drags atoms by chirping its X/Y RF tones; here it simply applies the move
    plan to the shared :class:`VirtualTrapArray` occupancy (``trap_array.apply_moves``), so the next
    camera image of that array shows the rearranged, defect-free pattern.  The rearrangement task /
    subsystem call ``aod.apply_moves(plan.moves)`` identically against this or a real AOD -- the only
    backend-specific thing is that the real AOD fires a waveform while this mutates the simulated atoms
    (virtual==real, branch at the device).  ``trap_array`` is wired by the config (``$device:trap_array``,
    the SAME array the camera images)."""

    def __init__(self, trap_array: VirtualTrapArray, move_survival: float = 0.98):
        self.trap_array = trap_array
        self.move_survival = probability(move_survival, "move_survival")

    def apply_moves(self, moves, *, survival: float | None = None) -> None:
        self.trap_array.apply_moves(moves, survival=self.move_survival if survival is None else survival)

    def snapshot(self) -> dict[str, object]:
        return {"type": type(self).__name__, "move_survival": self.move_survival,
                "n_sites": self.trap_array.n_sites}

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
        # the AOD rearrangement actuator drags atoms in the SAME array the camera images
        "aod": {"type": "VirtualAOD", "params": {"trap_array": "$device:trap_array"}},
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
    aod_params: dict[str, object] = {}
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
        elif key == "move_survival":
            aod_params["move_survival"] = float(value)        # per-move AOD transit survival
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
    cfg["aod"].setdefault("params", {}).update(aod_params)
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


def trap_hold_durations_per_frame(
    sequence: PulseSequence | None,
    frames: int,
    *,
    trigger_channels: Sequence[str] | None = None,
    trap_channels: Sequence[str] = DEFAULT_TRAP_CHANNELS,
    probe_channels: Sequence[str] = ("probe",),
) -> list[float]:
    """DARK trap-ON hold (s) -- trap ON but probe OFF -- in the window preceding each frame.

    Parses the SAME fired ``PulseSequence``.  For each frame's window (camera trigger
    ``k-1`` -> ``k``) it is the trap-channel ON time minus the probe-channel ON time
    (the probe always fires while the trap is on, so this is the dark hold: settle +
    recapture + any explicit trap-on hold).  The virtual backend applies vacuum-limited
    trap loss over this duration, so a TRAP-LIFETIME measurement (hold-time scan, no
    release) shows the survival roll-off.  Data-source side only; the analysis layer never
    reads it."""
    out = [0.0] * int(frames)
    if sequence is None or not hasattr(sequence, "effective_pulses"):
        return out
    trig_set = {str(c) for c in (DEFAULT_CAMERA_TRIGGER_CHANNELS if trigger_channels is None else trigger_channels)}
    trap_set = {str(c) for c in trap_channels}
    probe_set = {str(c) for c in probe_channels}
    pulses = sequence.effective_pulses()
    trigger_starts = sorted(p.start for p in pulses if p.value and p.channel in trig_set)
    if not trigger_starts:
        return out
    trap_intervals = sorted((p.start, p.stop) for p in pulses if p.value and p.channel in trap_set)
    probe_intervals = sorted((p.start, p.stop) for p in pulses if p.value and p.channel in probe_set)
    duration = float(getattr(sequence, "duration", 0.0))
    n = min(int(frames), len(trigger_starts))

    def overlap(intervals, lo, hi):
        return float(sum(max(0.0, min(stop, hi) - max(start, lo)) for start, stop in intervals if start < hi and stop > lo))

    for k in range(n):
        window_lo = 0.0 if k == 0 else trigger_starts[k - 1]
        window_hi = trigger_starts[k] if k < len(trigger_starts) else max(duration, window_lo)
        dark = overlap(trap_intervals, window_lo, window_hi) - overlap(probe_intervals, window_lo, window_hi)
        out[k] = max(0.0, dark)
    return out


def exposures_per_frame(
    sequence: PulseSequence | None,
    frames: int,
    *,
    default: float,
    trigger_channels: Sequence[str] | None = None,
    probe_channels: Sequence[str] = ("probe",),
) -> list[float]:
    """Probe-ON integration time (s) for each acquired frame, frame by frame.

    A real externally-triggered camera integrates for the window its OWN trigger gates,
    so a sequence may image successive frames for DIFFERENT durations (e.g. a
    long-short-long reference bracket: two 20 ms ground-truth images around a 5 ms
    readout).  Entry ``k`` is the probe-channel ON time in the window from camera trigger
    ``k`` to trigger ``k+1`` (the last frame runs to the sequence end).  For a uniform
    single-trigger sequence repeated per frame, every entry equals the one exposure -- so
    this is a no-op generalisation of :func:`exposure_from_sequence` for the render loop.
    Data-source side only; the analysis layer never reads it."""
    out = [float(default)] * int(frames)
    if sequence is None or not hasattr(sequence, "effective_pulses"):
        return out
    trig_set = {str(c) for c in (DEFAULT_CAMERA_TRIGGER_CHANNELS if trigger_channels is None else trigger_channels)}
    probe_set = {str(c) for c in probe_channels}
    pulses = sequence.effective_pulses()
    trigger_starts = sorted(p.start for p in pulses if p.value and p.channel in trig_set)
    if not trigger_starts:
        return out
    probe_intervals = sorted((p.start, p.stop) for p in pulses if p.value and p.channel in probe_set)
    duration = float(getattr(sequence, "duration", 0.0)) or max((e for _, e in probe_intervals), default=0.0)
    n = min(int(frames), len(trigger_starts))
    for k in range(n):
        lo = trigger_starts[k]
        hi = trigger_starts[k + 1] if k + 1 < len(trigger_starts) else max(duration, lo)
        total = sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in probe_intervals if s < hi and e > lo)
        if total > 0.0:
            out[k] = float(total)
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
    "exposures_per_frame",
    "trap_off_durations_per_frame",
    "virtual_config",
    "virtual_config_with_overrides",
    "write_virtual_run",
]
