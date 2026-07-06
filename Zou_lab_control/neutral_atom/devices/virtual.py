"""Virtual devices for offline notebook tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin, sqrt
from pathlib import Path
from typing import Mapping, Sequence
import time

import numpy as np

from ..core.analysis import (
    finite_float, grid_shape_tuple, nonnegative_float, point_tuple, positive_float, positive_int, probability)
from Zou_lab_control._clock import DEFAULT_CLOCK_HZ
# The device fabricates its release-recapture "truth" through the SAME error-function law the
# analysis layer FITS (operations.temperature.release_recapture_survival), so editing the law in
# ONE place keeps the sim and the fitter in agreement.  ``normal_cdf`` is the single source for that
# CDF/erf kernel (Zou_lab_control/_readout_math.py, a dependency-free leaf like ``_clock``); the
# physical constants K_B / RB87_MASS have their single source in operations.temperature and are
# imported lazily where used (keeping devices->operations off the import graph, as elsewhere here).
from Zou_lab_control._readout_math import normal_cdf
from ..core.utils import site_index
from .base import CameraDevice, SequencerDevice, TrapArrayDevice, snap_subarray
from .camera_trigger import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    base_cycle_trigger_pulses,
    count_trigger_pulses,
)
from ..timing import (
    PulseSequence,
    PulseTableState,
    probe_channel_set,
)


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
    # Bright-atom photoelectron rate (peak amplitude/s) during the probe.  ALIGNED to the REAL Rb87
    # qCMOS dataset's DETECTED PHOTON RATE (references/.../rb87_readout_v16): the v16 bright atom
    # delivers ~9.3 box / ~7.9 psf detected photons in a 2 ms readout (= ~4.6 / 4.0 kph/s), so this
    # rate makes the virtual per-atom photon count match v16 at the SAME exposure -- NOT higher (a
    # too-high rate gave an unrealistically clean, over-separated histogram: the readout looked far
    # easier than real Rb87).  With the photon count aligned, the bright distribution width is pure
    # Poisson(signal+background) -- it sharpens with exposure, so single-shot fidelity RISES from the
    # hard short-readout regime (~0.8-0.9) to ~0.99 at a long readout, exactly like the real data.
    atom_rate: float = 1_100.0
    # Per-site COLLECTION-efficiency variation (relative standard deviation, 1-sigma):
    # a real Rb87 tweezer array has different effective NA / aberration / vacuum-window
    # transmission per site, so each site's bright count rate scales by an independently
    # drawn ``Lognormal(0, sigma)`` multiplier around 1.0.  This is what makes a per-site
    # PSF kernel materially better than one shared box / one uniform PSF: the per-site
    # spot brightness varies, so a fixed box averages out signal a per-site kernel can
    # match.  Default 0.18 (=18% RSD) is a realistic spread on a moderate tweezer array.
    site_efficiency_sigma: float = 0.18
    # OPTIONAL extra super-Poisson shot-to-shot brightness jitter (1-sigma of a per-shot, per-atom
    # mean-preserving lognormal).  DEFAULT 0: the v16 bright-distribution width is fully explained by
    # Poisson(signal + background) once the photon RATE and background are aligned (a 2 ms bright box
    # ~9 ph over ~29 ph background -> Poisson sigma ~5.8 ph, the measured width) -- there is NO unexplained
    # excess, and a FIXED multiplicative jitter is WRONG anyway because it would cap the fidelity (it does
    # not sharpen with exposure, so single-shot fidelity could never rise to ~0.99 at a long readout the
    # way the real data does).  Left as a knob only to model a genuinely un-cooled / hopping atom; keep 0
    # for the v16-faithful pure-Poisson readout.
    shot_brightness_jitter: float = 0.0
    # Stray-light + scatter floor (detected photons/s/pixel, always present).  ALIGNED to the v16
    # background: its 2 ms box carries ~29 detected photons over a 49 px (7x7) box -> ~0.6 ph/px/2 ms
    # -> ~300 ph/s/px.  This sets the dark-site Poisson floor that the bright signal competes with;
    # a too-high floor (the old 600) added background noise that CAPPED the achievable fidelity (it
    # could not reach ~0.99 even at a long readout), so aligning it to v16 is what lets the readout
    # recover the real ~0.99 at the reference exposure.
    background_rate: float = 300.0
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
    # PER-SITE PSF SHAPE scatter (fractional RSD).  On a real array EACH tweezer has its OWN spot
    # shape (the Rb87 v16 per-site fits give sigma_x / sigma_y ~10 % RSD across the 35 sites), because
    # field-dependent aberration varies the kernel site to site.  Each site's sigma_x / sigma_y is
    # drawn ONCE per session (cached, like site_efficiency) as ``base * Lognormal(0, sigma)``, and the
    # spot is renormalized to keep that site's total flux (so box counts still track site_efficiency).
    # This per-site SHAPE diversity -- not just amplitude -- is exactly what lets a per-site PSF matched
    # filter beat a single uniform PSF / a box on a real array (set 0 to recover one shared shape).
    site_shape_sigma: float = 0.10
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
    # Recaptured-atom mass (kg).  ``None`` -> the Rb87 mass from the analysis layer's single source
    # (operations.temperature.RB87_MASS), resolved in __post_init__ via a lazy import so a dataclass
    # field default never re-types the physical constant AND devices->operations stays off the
    # import graph (that top-level dependency would cycle -- see the module-level import note).
    recapture_mass_kg: float | None = None
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
        # Validate like every sibling tunable (it feeds a lognormal sigma; a bad value otherwise
        # only blows up deep in _site_efficiency on the first render, not at construction).
        self.site_efficiency_sigma = nonnegative_float(self.site_efficiency_sigma, "site_efficiency_sigma")
        self.read_noise_e = nonnegative_float(self.read_noise_e, "read_noise_e")
        self.atom_sigma_px = positive_float(self.atom_sigma_px, "atom_sigma_px")
        self.atom_psf_aspect = positive_float(self.atom_psf_aspect, "atom_psf_aspect")
        self.atom_psf_angle_deg = float(self.atom_psf_angle_deg)
        self.atom_psf_skew = nonnegative_float(self.atom_psf_skew, "atom_psf_skew")
        self.site_shape_sigma = nonnegative_float(self.site_shape_sigma, "site_shape_sigma")
        self.shot_brightness_jitter = nonnegative_float(self.shot_brightness_jitter, "shot_brightness_jitter")
        self.load_time_constant_s = positive_float(self.load_time_constant_s, "load_time_constant_s")
        self.mot_load_s = positive_float(self.mot_load_s, "mot_load_s")
        self.detection_lifetime = positive_float(self.detection_lifetime, "detection_lifetime")
        self.trap_lifetime_s = positive_float(self.trap_lifetime_s, "trap_lifetime_s")
        self.cooled_temperature_K = positive_float(self.cooled_temperature_K, "cooled_temperature_K")
        self.probe_heating_K_per_s = nonnegative_float(self.probe_heating_K_per_s, "probe_heating_K_per_s")
        self.pgc_cool_tau_s = positive_float(self.pgc_cool_tau_s, "pgc_cool_tau_s")
        self.capture_radius_m = positive_float(self.capture_radius_m, "capture_radius_m")
        if self.recapture_mass_kg is None:
            from ..operations.temperature import RB87_MASS  # lazy: keep devices->operations off the import graph
            self.recapture_mass_kg = RB87_MASS
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

    def _site_shapes(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-site PSF (sigma_x, sigma_y) in px -- each tweezer's OWN spot shape, drawn ONCE per session
        (cached) as the global ``atom_sigma_px`` / aspect scaled by independent ``Lognormal(0,
        site_shape_sigma)`` factors (geometric mean 1, so the array-average spot size is unchanged).  This
        gives each site a distinct kernel like the real Rb87 v16 per-site fits (~10 % RSD), which is what
        lets a per-site PSF matched filter beat one shared (uniform) PSF.  ``site_shape_sigma=0`` -> every
        site shares the global shape."""
        cache = getattr(self, "_site_shape_cache", None)
        if cache is not None and cache[0].size == self.n_sites:
            return cache
        asp = sqrt(max(self.atom_psf_aspect, 1e-6))
        base_sx = self.atom_sigma_px / asp
        base_sy = self.atom_sigma_px * asp
        sigma = float(getattr(self, "site_shape_sigma", 0.0) or 0.0)
        if sigma <= 0.0:
            sx = np.full(self.n_sites, base_sx, dtype=float)
            sy = np.full(self.n_sites, base_sy, dtype=float)
        else:
            sx = base_sx * np.exp(self.rng.normal(0.0, sigma, size=self.n_sites))
            sy = base_sy * np.exp(self.rng.normal(0.0, sigma, size=self.n_sites))
        self._site_shape_cache = (sx, sy)
        return self._site_shape_cache

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
        base_sx = self.atom_sigma_px / asp                     # array-average (nominal) spot size
        base_sy = self.atom_sigma_px * asp
        th = radians(self.atom_psf_angle_deg)
        cos_t, sin_t = cos(th), sin(th)
        eff = self._site_efficiency()
        site_sx, site_sy = self._site_shapes()                 # per-site spot shape (cached, ~10 % RSD)
        # SHOT-TO-SHOT brightness jitter: a fresh per-atom mean-1 lognormal EACH frame (not cached) so
        # the same atom scatters a different photon count shot to shot -- this is what gives the bright
        # distribution its realistic (super-Poisson) width and a non-saturating per-site fidelity.
        js = float(self.shot_brightness_jitter)
        jit = (np.exp(self.rng.normal(-0.5 * js * js, js, size=self.n_sites)) if js > 0.0
               else np.ones(self.n_sites, dtype=float))
        for (cx, cy), occupied, t_sig, e, sx, sy, j in zip(self._site_centers(), occ, st, eff, site_sx, site_sy, jit):
            if not occupied or t_sig <= 0.0:
                continue
            # Each tweezer has its OWN bright count rate (collection efficiency: NA / window / aberration)
            # AND its OWN spot SHAPE.  ``amplitude`` is the array-nominal PEAK; ``norm`` rescales the
            # per-site spot so its INTEGRAL (total photons) is unchanged by the shape draw -- a real atom
            # scatters a fixed photon count set by eff, just spread over its own kernel.  So box counts
            # track ``eff`` (amplitude) while the per-site SHAPE differs, which is precisely why a per-site
            # PSF matched filter beats one uniform PSF / a box on a real array.
            amplitude = self.atom_rate * float(t_sig) * float(e) * float(j)
            norm = (base_sx * base_sy) / (float(sx) * float(sy))
            du = (xx - cx) * cos_t + (yy - cy) * sin_t          # along the major axis
            dv = -(xx - cx) * sin_t + (yy - cy) * cos_t         # minor axis
            core = np.exp(-0.5 * ((du / sx) ** 2 + (dv / sy) ** 2))
            spot = np.clip(core * (1.0 + self.atom_psf_skew * (du / sx)), 0.0, None)
            expected_e += amplitude * norm * spot
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
        afterward) and returns the new occupancy.

        The per-axis recaptured fraction is the SAME error-function law the analysis layer FITS
        (operations.temperature.release_recapture_survival): ``erf(arg)`` written through the shared
        ``normal_cdf`` single source as ``2*Phi(arg) - 1`` with a sqrt(2)-scaled standard normal
        (``normal_cdf(arg, 0, 1/sqrt2) == 0.5 (1 + erf(arg))``).  So this device fabricates its
        "truth" through the identical kernel the fitter uses -- edit the law once, sim and fit agree."""
        from ..operations.temperature import K_B  # lazy: keep devices->operations off the import graph

        t = float(t_off)
        if not np.isfinite(t) or t <= 0.0:
            return self.occupancy.copy()
        sigma_v = np.sqrt(K_B * np.maximum(self.temperature_K, 1e-12) / self.recapture_mass_kg)
        arg = self.capture_radius_m / (sqrt(2.0) * sigma_v * t)
        p_axis = 2.0 * np.asarray(normal_cdf(arg, 0.0, 1.0 / sqrt(2.0))) - 1.0  # == erf(arg), shared kernel
        p_survive = p_axis ** 3
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

    def image_frames(self, sequence, frames: int, *, capture_trigger_channels, exposure: float,
                     all_sites: bool = False) -> list[np.ndarray]:
        """The virtual 'physical world': evolve THIS atom array under the fired ``sequence`` and
        render the ``frames`` camera images its capture triggers gate.

        This is the ATOM device's job (mirroring real hardware, where the FPGA outputs drive the
        atoms and the camera merely images them): each camera-trigger window the pulse carries
        (re)LOADS / re-COOLS / RELEASES / dark-holds the atoms before its frame, then the survivors
        are imaged (probe scatter + readout loss + recoil).  The CAMERA passes its OWN
        ``capture_trigger_channels`` (it owns 'which line triggers me') + its exposure; it does not
        parse the pulse itself.  A pulse carrying NO camera trigger leaves the array dark -> no
        frame (the live image freezes), exactly like an externally-triggered sensor.  The analysis
        layer never sees any of this -- it only receives the rendered frames."""
        frames = positive_int(frames, "frames")
        trig = capture_trigger_channels or DEFAULT_CAMERA_TRIGGER_CHANNELS
        if sequence is None:
            return []
        # The camera-trigger gate is a LIVE-monitor concept: a CONTINUOUS (``repeat_forever``)
        # firing renders only while it pulses a camera-trigger line, and FREEZES the moment it
        # does not (Stop Pulse -> a non-imaging safe state), exactly like a real externally-
        # triggered sensor.  A FINITE measurement sequence (sitemap / threshold / detect /
        # detection-time scan / reference bracket) is one the measurement deliberately fired to
        # read ``frames`` windows, so it is rendered UNCONDITIONALLY -- its per-frame physics
        # (cooling/trap-off/trap-hold) is parsed against the camera's trigger line, defaulting
        # to independent re-loads when the bound pulse carries no trigger of its own.
        if getattr(sequence, "repeat_forever", False) and not _sequence_triggers_camera(sequence, trig):
            return []
        probe_set = probe_channel_set("probe")
        exposures = exposures_per_frame(sequence, frames, default=exposure,
                                        trigger_channels=trig, probe_channels=probe_set)
        cooling_durations = cooling_durations_per_frame(sequence, frames, trigger_channels=trig)
        trap_off_per_frame = trap_off_durations_per_frame(sequence, frames, trigger_channels=trig)
        trap_hold_per_frame = trap_hold_durations_per_frame(sequence, frames, trigger_channels=trig)
        # Does ONE BASE CYCLE carry a camera-trigger window for EVERY frame?  If so this shot is a
        # single-loading BRACKET -- a release-recapture bracket (2 windows around a readout) or a
        # long-short-long imaging bracket (3 windows) -- and the SAME atoms are imaged through each
        # window (a mid-cycle cooling phase RE-COOLS the held atoms), so every frame uses ITS OWN
        # window's exposure / cooling.  Otherwise the frames come from REPEATS of a SHORTER base cycle:
        # a ``.repeated(N)`` single-window sitemap / threshold / detect, or a continuous single-window
        # live monitor -- each frame a FRESH independent loading through the base window.
        #
        # The criterion is the BASE-CYCLE window count vs frames -- NOT repeat_count / repeat_forever.
        # A ``repeat_forever`` imaging bracket (a long-short-long On-Pulse looped live) is STILL a
        # bracket per cycle, so keying off ``repeat_forever`` collapsed frames 1..n onto window 0 and
        # EVERY emCCD frame came out at the FIRST window's (long) exposure -- the reported "看不到
        # long-short-long exposure" bug.  (The raw FIRED trigger total cannot distinguish a repeated
        # single-trigger from an N-window bracket -- both total N -- but the per-BASE-CYCLE count can:
        # 1 for the repeated single-trigger, N for the bracket.)
        single_cycle = base_cycle_trigger_pulses(sequence, trigger_channels=trig) >= frames
        # A repeated SINGLE-trigger sequence images a fresh independent loading every frame; each
        # repeat is one base cycle, so every frame's cooling window equals the base cooling
        # (``cooling_durations[0]``; None when the pulse has no cooling phase -> saturated load).
        repeated_shot_cooling = cooling_durations[0] if cooling_durations else 0.0
        # A repeated single-trigger shot images EVERY frame through the SAME one trigger window, so
        # every frame uses THAT window's exposure -- not the camera-default `exposures_per_frame`
        # falls back to for frames past the (single) trigger.  Mixing a short window-0 with
        # default-length repeats skews a per-site threshold learnt across the frames ABOVE the real
        # readout brightness, so a freshly loaded atom at the readout exposure reads as EMPTY
        # (release-recapture survival / readout fidelity collapse to 0).  Mirrors repeated_shot_cooling.
        repeated_shot_exposure = exposures[0] if exposures else float(exposure)
        images: list[np.ndarray] = []
        for frame_index in range(frames):
            frame_exposure = (
                repeated_shot_exposure if (not single_cycle and frame_index >= 1) else exposures[frame_index])
            if not all_sites:
                cool_dt = cooling_durations[frame_index]
                trap_off = trap_off_per_frame[frame_index]
                trap_hold = trap_hold_per_frame[frame_index]
                if frame_index == 0:
                    # Each shot starts from a fresh loading -- unless a deterministic test pinned a
                    # specific occupancy (consume_pin), which this one shot images instead.
                    if not self.consume_pin():
                        self.reload(cooling_duration=(cool_dt if cool_dt > 0.0 else None))
                elif not single_cycle:
                    # A repeated single-trigger shot: a fresh independent loading every frame (the same
                    # base cooling each time), so threshold/scan frames are independent shots -- NOT one
                    # decaying loading imaged N times.
                    self.reload(cooling_duration=(repeated_shot_cooling if repeated_shot_cooling > 0.0 else None))
                elif cool_dt > 0.0:
                    self.cool(cool_dt)       # mid-cycle cooling re-cools the held single-cycle atoms
                # Cool THEN release (so heat -> re-cool -> release drops the re-cooled cloud).
                if trap_off > 0.0:
                    self.apply_recapture_loss(trap_off)
                # Dark trap-ON hold loses atoms to vacuum collisions; skipped on frame 0 (the load).
                if trap_hold > 0.0 and frame_index >= 1:
                    self.apply_trap_loss(trap_hold)
            images.append(self.render_image(exposure=frame_exposure, all_sites=all_sites))
        return images

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


class _TriggerWiredCamera(CameraDevice):
    """Virtual camera plumbing for the TRIGGER CABLE that physically exists on the real rig.

    On hardware the FPGA's trigger line is a real wire into the camera's trigger input: the
    armed sensor reads out exactly when edges arrive.  A virtual camera simulates THAT WIRE --
    the one lowest-layer fake -- as a ``sequencer`` constructor parameter (injected from the
    device config via ``"$device:sequencer"``):

    * a FINITE fire on the wired :class:`VirtualSequencer` notifies the camera; while ARMED it
      renders the frames those trigger edges would have gated straight into the device buffer
      (an unarmed camera misses the trigger, exactly like real hardware);
    * while the streamer is CONTINUOUSLY firing (``repeat_forever``, the live On Pulse),
      :meth:`_grab` renders frames on demand at the firing cadence -- the simulation of an
      endless trigger train;
    * no wire, or a wire that is idle -> no trigger edges -> no frame, and the live view
      FREEZES instantly (the software mirror of "no hardware edges arrive").

    Everything above the device contract is untouched: measurements arm/fire/read through the
    SAME code path as on a real camera (virtual == real; swapping in real hardware changes only
    the device config)."""

    def _wire_to(self, sequencer) -> None:
        """Plug the trigger cable in (constructor-time): remember the wired streamer and
        subscribe to its fire events when it supports them."""
        self.sequencer = sequencer
        if sequencer is not None and hasattr(sequencer, "add_fire_listener"):
            sequencer.add_fire_listener(self._on_wire_fired)

    def close(self) -> None:
        """Unplug the trigger cable: stop listening to the wired streamer's fires (the mirror of
        the physical trigger wire being disconnected).  Symmetric with :meth:`_wire_to` -- so a
        closed camera stops rendering frames on every fire (and does not linger as a fire listener
        holding a reference).  Concrete cameras override ``close`` for their own teardown but call
        ``super().close()`` to unwire."""
        wire = getattr(self, "sequencer", None)
        if wire is not None and hasattr(wire, "remove_fire_listener"):
            wire.remove_fire_listener(self._on_wire_fired)

    @property
    def sleep_scale(self) -> float:
        """The virtual time scale, OWNED by the wired VirtualSequencer and read through the wire.

        A real externally-triggered sensor's wait-for-trigger cadence is dictated by the FPGA's
        firing rate, so the camera has no independent time scale -- it paces with the streamer.  The
        virtual camera reads the SAME ``sleep_scale`` the sequencer plays at (the one the pytest
        suite flips to 0 via ``DEFAULT_SLEEP_SCALE`` to fast-forward), through the wire it already
        consults for ``.firing`` -- so the two can never drift out of lockstep (whichever config
        path set the sequencer's scale, the camera honours it with no separate mirror to keep in
        sync).  No wire -> the module default."""
        wire = getattr(self, "sequencer", None)
        scale = getattr(wire, "sleep_scale", None)
        return float(DEFAULT_SLEEP_SCALE if scale is None else scale)

    def _render_frames(self, sequence, frames: int) -> list[np.ndarray]:
        raise NotImplementedError

    def _pace(self, sequence, stop) -> None:
        """Optional per-frame wall-clock pacing for a continuous firing (subclass hook)."""

    def _arm(self, frames: int | None) -> None:
        # Fresh armed session: forget any finite fire recorded under a previous arm, so the
        # loud-fault check in read_frames only ever sees THIS session's trigger edges.
        state = self._recent_state()
        with state["cond"]:
            state["_finite_fire_seen"] = False

    def _on_wire_fired(self, sequence) -> None:
        # A FINITE fire = the burst of trigger edges this camera's wire carries.  A continuous
        # (repeat_forever) program is instead pulled on demand in _grab, paced per frame.
        if sequence is None or bool(getattr(sequence, "repeat_forever", False)):
            return
        state = self._recent_state()
        with state["cond"]:
            if not state["armed"]:
                return              # an unarmed camera misses the trigger, exactly like hardware
            armed_frames = state["armed_frames"]
            # Mark that a FINITE fire reached this armed session: read_frames uses this to tell a
            # genuine trigger DEFICIT (a finite program that under-delivered its edges -> raise,
            # like the real qCMOS) apart from the frozen live monitor (a continuous/idle wire that
            # legitimately produced no frame -> return [] and hold the last image).
            state["_finite_fire_seen"] = True
        # EDGE-FAITHFUL: the number of frames this fire delivers is the number of camera-trigger
        # EDGES the fired program carries on THIS camera's own trigger line -- exactly what a real
        # sensor reads out (one frame per rising edge), never the count the consumer happened to arm
        # for.  ``armed_frames`` is only an UPPER BOUND (arm(N) reads at most N); the truth is the
        # edge count.  So a single-trigger pulse fired once delivers ONE frame even if armed for 20
        # -- the measurement layer that wants N frames fires N edges (``triggered_frames`` repeats a
        # single-trigger sequence to N triggers; a real FPGA emits N edges the same way).  Zero edges
        # -> zero frames delivered (nothing rendered), like a fire on a line this camera isn't wired to.
        trig = getattr(self, "capture_trigger_channels", None)
        edges = count_trigger_pulses(sequence, **({"trigger_channels": trig} if trig else {}))
        wanted = edges if armed_frames is None else min(int(armed_frames), edges)
        if wanted <= 0:
            return
        self._deliver(self._render_frames(sequence, int(wanted)))
        # REAL-TIME pacing of the FINITE readout: on real hardware read_frames() blocks until the
        # triggered exposures have been read out, which takes ~= the fired sequence's play duration
        # (the FPGA plays the pulse in real time, the camera exposes/reads out per trigger edge).
        # Stamp when that readout would COMPLETE so read_frames() below blocks the same wall-clock --
        # so a LONGER pulse template genuinely slows each shot (and the whole scan), the device-layer
        # mirror of the qCMOS wait_capevent block.  Single-sourced with _pace / wait_done / the scan
        # base_dt: per-shot wall-clock = duration * sleep_scale, so sleep_scale=0 (pytest) stamps
        # nothing and the suite stays instant.
        pace = float(getattr(sequence, "duration", 0.0)) * self.sleep_scale
        if pace > 0.0:
            with state["cond"]:
                state["_finite_pace_until"] = time.monotonic() + pace

    def _grab(self, n: int, *, timeout: float | None = None, stop=None) -> bool:
        # The wire is authoritative about trigger edges: an idle (or absent) wire means no
        # edge can arrive, so the read returns immediately and the live view freezes -- the
        # data-source gate the live monitor relies on (no fabricated frames, no dead wait).
        wire = getattr(self, "sequencer", None)
        firing = None if wire is None else wire.firing
        if firing is None:
            return False
        # EDGE-FAITHFUL in whole cycles: a continuous firing delivers its trigger train one BASE
        # CYCLE at a time, so render the cycle's WHOLE block -- the same block path a finite fire
        # takes (_on_wire_fired) and the one place the per-window physics lives (image_frames: a
        # long-short-long bracket is ONE loading imaged through its 3 windows at 3 exposures).
        # Rendering frame-by-frame here is what LOST the window phase (every grab re-parsed the
        # sequence from window 0, so a live 3-window bracket came out as 3 identical long frames
        # from 3 independent loadings).  read_frames drains the delivered block at the consumer's
        # own cadence, so a frames_per_cycle=N reader stays phase-aligned window-for-window.  A
        # firing whose base cycle carries no edge on THIS camera's line falls back to one frame
        # per grab (the MOT monitor's sequence-driven sensing; the atom camera's image_frames
        # gates that case to [] itself and the read stays frozen).
        trig = getattr(self, "capture_trigger_channels", None)
        cycle = base_cycle_trigger_pulses(firing, **({"trigger_channels": trig} if trig else {}))
        frames = self._render_frames(firing, max(1, cycle))
        if not frames:
            return False            # the firing pulse carries no camera trigger -> dark
        self._deliver(frames)
        # One pace per CYCLE (the block IS one cycle): the previous per-frame pace slept a full
        # cycle for every frame, so a 3-window live bracket ran 3x slower than the pulse it rode.
        self._pace(firing, stop)
        return True

    def read_frames(self, n: int = 1, *, timeout: float | None = None, stop=None, **kwargs) -> list[np.ndarray]:
        """Drain the armed frames, THEN block the fired sequence's remaining play time -- the
        device-layer mirror of a real externally-triggered sensor whose read blocks until the
        exposures the trigger burst gated have been read out.  The finite frames were pushed at
        fire (lossless, unchanged); the deadline stamped there (``_finite_pace_until``) makes the
        read consume ``duration * sleep_scale``, so a longer pulse template really slows each shot
        (virtual == real).  ``sleep_scale=0`` (pytest) stamps no deadline, so tests stay instant;
        ``stop`` cancels the wall-clock wait cooperatively.  The continuous live path sets no
        deadline (repeat_forever is paced per frame in :meth:`_grab`), so it is untouched.

        LOUD FAULT MODEL (mirrors the real qCMOS): a FINITE armed read that ends with FEWER than
        ``n`` frames raises :class:`TimeoutError` instead of returning short.  On real hardware
        ``read_frames(n)`` waits for ``n`` trigger edges and the qCMOS raises ``TimeoutError`` when
        an edge never comes (``qcmos.py``: "qCMOS timed out ... waiting for frame N"); a virtual
        camera that quietly returned a short list would let a trigger/edge DEFICIT pass in
        simulation that aborts the whole chain on the real sensor.  The armed session's budget is
        the camera's own ``timeout`` (the L6 knob), scaled by ``sleep_scale`` so a finite deficit is
        surfaced without a real wall-clock wait under the pytest fast-forward.  A CONTINUOUS live
        read (``armed_frames is None``) is unbounded and never raises -- it legitimately returns
        whatever the still-firing wire produced this poll."""
        out = super().read_frames(n, timeout=timeout, stop=stop, **kwargs)
        state = self._recent_state()
        with state["cond"]:
            until = state.pop("_finite_pace_until", None)   # pop unconditionally: never leak a stale deadline
            armed_frames = state["armed_frames"]
            finite_fire_seen = bool(state.get("_finite_fire_seen", False))
        if out and until is not None:
            wall = until - time.monotonic()
            if wall > 0.0:
                stop.wait(wall) if stop is not None else time.sleep(wall)
        # A FINITE armed read (armed_frames is not None) whose wire ALREADY carried a finite fire this
        # session, yet delivered fewer than ``n`` frames, is a genuine trigger/edge DEFICIT: the fired
        # program under-delivered its camera-trigger edges, so the sensor is still waiting for an edge
        # that never comes -- exactly the condition the real qCMOS raises TimeoutError for.  Fail LOUD
        # (never a silent short list) so a deficit that aborts a real run is caught the same way in
        # simulation.  The ``_finite_fire_seen`` gate is what separates this from the PASSIVE live
        # monitor, whose continuous/idle wire legitimately produces no frame (it never fires a finite
        # program) and must keep returning [] to hold its last image and freeze -- never raise.  The
        # wait budget is the camera's own ``timeout`` (the one L6 knob), reported in ms like the qCMOS
        # message; scaled by ``sleep_scale`` so the pytest fast-forward (sleep_scale=0) is instant.
        if armed_frames is not None and finite_fire_seen and len(out) < n:
            budget_s = float(getattr(self, "timeout", 0.0)) * self.sleep_scale
            raise TimeoutError(
                f"virtual camera timed out after {budget_s * 1000.0:.0f} ms waiting for frame "
                f"{len(out)} of {n} (only {len(out)} trigger edge(s) arrived).")
        return out


class VirtualCamera(_TriggerWiredCamera):
    def __init__(self, trap_array: VirtualTrapArray, exposure: float = 20e-3, timeout: float = 2.0,
                 subarray_step: int = 4, capture_trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
                 sequencer=None):
        self.trap_array = trap_array
        self._exposure = positive_float(exposure, "exposure")
        self.timeout = positive_float(timeout, "timeout")
        # Which line the camera's trigger is wired to -- a PASSIVE device property the camera owns
        # and exposes; it images the atoms the FPGA drives, never touching the sequencer (virtual==real).
        self.capture_trigger_channels = tuple(str(c) for c in capture_trigger_channels)
        # The virtual time scale is OWNED by the wired sequencer and read through the wire (the
        # ``sleep_scale`` property on _TriggerWiredCamera) -- a real triggered sensor paces with the
        # FPGA's firing, so there is NO independent camera time scale to fall out of sync.
        self.last_sequence: str | None = None
        # Mirror the real qCMOS sub-array: a ROI is snapped to a hardware grid
        # (the Hamamatsu step is 4) and the rendered frame is CROPPED to it, so the
        # virtual path exercises the SAME ROI contract a real camera does -- a ROI
        # bug then shows up in a virtual test, and switching to real changes only
        # connect().  None = full frame (the default; all existing behaviour).
        self.subarray_step = int(subarray_step)
        self._roi: tuple[int, int, int, int] | None = None
        # SIMULATION-ONLY knob: render every site as loaded (an idealized full array for a
        # deterministic device-level test).  Real production never sets it -- a sitemap
        # calibration finds sites by AVERAGING realistic ~50% loadings, same as hardware.
        self.force_all_sites: bool = False
        # The virtual TRIGGER CABLE (see _TriggerWiredCamera): the wired in-process streamer
        # whose fire events / continuous firing gate this camera's frames.  None = a bare
        # bench camera with no cable -- it then never sees a trigger and reads no frames.
        self._wire_to(sequencer)

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

    def configure(self, *, exposure: float | None = None, roi: object = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure", "roi"}, kwargs)
        if exposure is not None:
            self.exposure = positive_float(exposure, "exposure")
        if roi is not None:
            if roi in ("", "None"):
                self._roi = None
            else:
                h, w = self.trap_array.image_shape
                self._roi = snap_subarray(tuple(roi), step=self.subarray_step, max_w=w, max_h=h)

    def _render_frames(self, sequence: PulseSequence | None, frames: int) -> list[np.ndarray]:
        # Image what the atoms (the VirtualTrapArray) are doing under the fired pulse.  The
        # pulse-driven physics lives in the ATOM device -- mirroring real hardware where the FPGA
        # outputs drive the atoms and the camera merely images them; the camera owns ONLY its
        # capture-trigger channel + imaging (exposure / ROI).  ``sequence`` arrives ONLY through
        # the trigger wire (a finite fire event, or the continuous firing _grab pulls from) --
        # nothing above the device layer ever hands the camera a pulse.
        frames = positive_int(frames, "frames")
        images = self.trap_array.image_frames(
            sequence, frames,
            capture_trigger_channels=self.capture_trigger_channels,
            exposure=self.exposure,
            all_sites=bool(self.force_all_sites),
        )
        out: list[np.ndarray] = []
        for image in images:
            if self._roi is not None:
                # crop to the applied sub-array, exactly as a real camera reads out only its ROI
                x, w, y, h = self._roi
                image = image[y:y + h, x:x + w]
            out.append(image)
        self.last_sequence = None if sequence is None else getattr(sequence, "name", None)
        return out

    def _pace(self, sequence, stop) -> None:
        # Block for the trigger cycle's wall-clock when imaging a CONTINUOUS (live) firing, exactly
        # as a real qCMOS's wait_capevent blocks until the next trigger -- so lengthening the
        # imaging period genuinely slows the live image.  Only for a continuous (repeat_forever)
        # program; a finite shot's timing is the measurement's wait_done, not a per-frame camera
        # block.  Interruptible by ``stop`` so Stop Pulse / teardown never blocks a full cycle.
        if self.sleep_scale > 0.0 and bool(getattr(sequence, "repeat_forever", False)):
            wall = float(getattr(sequence, "duration", 0.0)) * self.sleep_scale
            if wall > 0.0:
                stop.wait(wall) if stop is not None else time.sleep(wall)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self._roi,
            "timeout": self.timeout,
            "last_sequence": self.last_sequence,
        }

    def close(self) -> None:
        super().close()          # unplug the trigger cable (stop rendering on the streamer's fires)

    def stop(self) -> None:
        pass


class VirtualMotCamera(_TriggerWiredCamera):
    """The virtual MOT MONITOR camera (the pylon-viewer stand-in): a second, independent sensor
    watching the MOT fluorescence, whose brightness depends on the THREE coil-field DACs.

    Physics: the MOT capture efficiency is a 3-D Gaussian in the coil space --
    ``I = peak * exp(-sum(((L_i - b0_i)/sigma_i)^2)/2)`` where ``L_i`` is the SIGNED level each coil
    bus actually drives in the FIRED sequence.  The camera SENSES those levels from the compiled
    bit-channel pulses through :func:`~..timing.sequence.decode_analog_bus` (the encoder's exact
    inverse, pinned by a round-trip contract test) -- never from a task's set-points, so the virtual
    sensing chain goes through the same artefact the hardware plays (virtual == real; swapping in a
    real pylon camera changes only ``connect()``).  The fired sequence reaches the camera ONLY over
    the virtual trigger wire (the ``sequencer`` constructor parameter -- see
    :class:`_TriggerWiredCamera`).  The frame is a Gaussian fluorescence spot at the sensor centre
    with Poisson photon noise + Gaussian read noise on a constant offset, mirroring the qCMOS noise
    chain."""

    def __init__(self, *, width: int = 64, height: int = 64, exposure: float = 5e-3,
                 coil_buses: Mapping[str, Sequence[str]] | None = None,
                 b0: Mapping[str, float] | None = None,
                 b_sigma: Mapping[str, float] | None = None,
                 peak_rate: float = 4.0e5, background_rate: float = 2.0e3,
                 offset_counts: float = 100.0, read_noise: float = 2.0,
                 spot_sigma_px: float = 6.0, timeout: float = 2.0, seed: int | None = None,
                 capture_trigger_channels: Sequence[str] = ("mot_trigger",),
                 sequencer=None):
        self.width, self.height = int(width), int(height)
        self._exposure = positive_float(exposure, "exposure")
        self.timeout = positive_float(timeout, "timeout")
        # Which sequencer line the MOT monitor camera's external trigger is wired to -- its OWN
        # capture-trigger channel (the coil/probe template pulses ``mot_trigger`` once per cycle),
        # NOT the readout qCMOS's ``emCCD``.  A PASSIVE device property (the camera owns the wiring
        # fact): the edge-faithful frame count reads THIS line, so a MOT probe fired once delivers
        # one monitor frame.  A real MOT camera is configured with its actual chNN here.
        self.capture_trigger_channels = tuple(str(c) for c in capture_trigger_channels)
        # The coil DAC buses this camera's MOT responds to: {bus name: member bit channels LSB..MSB}.
        # Defaults mirror pulses/mot_field_template.json (three 6-bit buses) so the virtual demo is
        # zero-config; a real setup names its own buses in the device config.
        self.coil_buses = {str(k): tuple(str(c) for c in v) for k, v in dict(
            coil_buses or {"da_x": [f"dx{i}" for i in range(6)],
                           "da_y": [f"dy{i}" for i in range(6)],
                           "da_z": [f"dz{i}" for i in range(6)]}).items()}
        self.b0 = {str(k): float(v) for k, v in dict(b0 or {"da_x": 7.0, "da_y": -5.0, "da_z": 11.0}).items()}
        self.b_sigma = {str(k): positive_float(v, "b_sigma") for k, v in dict(
            b_sigma or {"da_x": 6.0, "da_y": 6.0, "da_z": 6.0}).items()}
        self.peak_rate = nonnegative_float(peak_rate, "peak_rate")
        self.background_rate = nonnegative_float(background_rate, "background_rate")
        self.offset_counts = nonnegative_float(offset_counts, "offset_counts")
        self.read_noise = nonnegative_float(read_noise, "read_noise")
        self.spot_sigma_px = positive_float(spot_sigma_px, "spot_sigma_px")
        self._rng = np.random.default_rng(seed)
        self._roi: tuple[int, int, int, int] | None = None
        self.last_levels: dict[str, int] | None = None    # what the last frame sensed (snapshot/debug)
        # The virtual TRIGGER CABLE (see _TriggerWiredCamera): fired coil sequences reach this
        # sensor only through the wired in-process streamer.  None = no cable, never a frame.
        self._wire_to(sequencer)

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = positive_float(value, "exposure")

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def configure(self, *, exposure: float | None = None, roi: object = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure", "roi"}, kwargs)
        if exposure is not None:
            self.exposure = positive_float(exposure, "exposure")
        if roi is not None:
            self._roi = None if roi in ("", "None") else snap_subarray(
                tuple(roi), step=1, max_w=self.width, max_h=self.height)

    def mot_efficiency(self, levels: Mapping[str, float]) -> float:
        """The 3-D Gaussian capture efficiency at the given coil levels -- THE virtual MOT model,
        exposed so the end-to-end optimum test asserts against the same rule the frames obey."""
        z = 0.0
        for bus, b0 in self.b0.items():
            z += ((float(levels.get(bus, 0.0)) - b0) / self.b_sigma[bus]) ** 2
        return float(np.exp(-0.5 * z))

    def _render_frames(self, sequence: PulseSequence | None, frames: int) -> list[np.ndarray]:
        # A pure externally-triggered sensor: image what the FIRED sequence (arriving over the
        # trigger wire) actually drives -- no fired pulses -> no frame, exactly like real hardware.
        frames = positive_int(frames, "frames")
        if sequence is None or not getattr(sequence, "pulses", None):
            return []
        from ..timing.sequence import decode_analog_bus
        t_end = max(p.start + p.duration for p in sequence.pulses)
        t_sense = 0.5 * t_end                       # the steady mid-frame level (edges settle at t=0)
        levels = {bus: decode_analog_bus(sequence, members, t_sense)
                  for bus, members in self.coil_buses.items()}
        self.last_levels = dict(levels)
        eff = self.mot_efficiency(levels)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        cx, cy = self.width / 2.0, self.height / 2.0
        spot = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * self.spot_sigma_px ** 2)))
        rate = self.background_rate + self.peak_rate * eff * spot
        out: list[np.ndarray] = []
        for _k in range(frames):
            photons = self._rng.poisson(rate * self.exposure).astype(float)
            frame = self.offset_counts + photons + self._rng.normal(0.0, self.read_noise, size=photons.shape)
            frame = np.clip(frame, 0, 65535).astype(np.uint16)
            if self._roi is not None:
                x, w, y, h = self._roi
                frame = frame[y:y + h, x:x + w]
            out.append(frame)
        return out

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "exposure": self.exposure,
            "roi": self._roi,
            "coil_buses": {k: list(v) for k, v in self.coil_buses.items()},
            "last_levels": self.last_levels,
        }

    def close(self) -> None:
        super().close()          # unplug the trigger cable (stop rendering on the streamer's fires)

    def stop(self) -> None:
        pass


class VirtualSequencer(SequencerDevice):
    """In-process streamer = a :class:`SequencerService` software state machine + the virtual
    camera's ``firing`` handle.

    The service IS the single state machine (the one shared by the real ``SequencerService`` on
    the FPGA computer): ``VirtualSequencer`` COMPOSES one and delegates prepare/fire/state +
    source-payload recording to it, so there is no second copy of that protocol to drift.  The
    only state this class adds on top is the REAL-TIME pacing the bare service cannot express:

    * ``firing`` -- the ``repeat_forever`` :class:`PulseSequence` the streamer is playing right
      now (or None when idle/safe).  The externally-triggered virtual camera images ONLY while
      this is set, so "Stop Pulse" freezes the live image exactly like hardware.  A real/remote
      streamer keeps no host-side firing flag (it learns it from trigger edges), hence the base
      default is None and this in-process backend overrides it.
    * the per-point wall-clock scan pacing (``base_dt = program.duration * sleep_scale``) the
      scan-progress reading + the finite-scan ``wait_done`` use, so lengthening the pulse really
      slows the live image and the pytest suite's ``sleep_scale=0`` fast-forwards.
    """

    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS, clock_hz: float = DEFAULT_CLOCK_HZ, sleep_scale: float | None = None):
        from .sequencer import SequencerService
        self.channels = tuple(str(channel) for channel in channels)
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        # REAL-TIME by default (DEFAULT_SLEEP_SCALE=1.0): a fired program takes its real
        # duration, so the live camera paces with the pulse.  ``None`` -> the module default
        # (the pytest suite flips that default to 0 to fast-forward; see conftest).
        if sleep_scale is None:
            sleep_scale = DEFAULT_SLEEP_SCALE
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        # The shared software state machine: prepare (with the FPGA-geometry backstop), fire,
        # source-payload recording and last_payload_json all live HERE, one copy.  Its built-in
        # pure-software wall-clock scan-progress is BYPASSED by wiring our real-time-paced
        # ``_scan_progress`` as the scan_progress_callback (the same callback seam the real AXI
        # backend uses for its hardware cursor) -- so the live-scan reading is single-sourced too.
        self.service = SequencerService(
            channels=self.channels,
            clock_hz=self.clock_hz,
            sleep_scale=self.sleep_scale,
            scan_progress_callback=self._scan_progress,
        )
        # The compiled program of the last prepare() the camera reads as ``firing`` -- a
        # PulseSequence (not the service's compiled RuntimeSequenceProgram), because the camera
        # needs its ``base_pulses()`` to detect the capture-trigger channel + its physics.
        self._prepared: PulseSequence | None = None
        # The compiled RuntimeSequenceProgram of the last prepare() (parity with
        # RuntimeSequencer.prepare, which on_pulse / sync-to-device read for .sequence_name etc.).
        self.last_program = None
        # The program the streamer is CONTINUOUSLY firing (a repeat_forever pulse), or None when
        # idle/safe -- the virtual camera's gate; see the class docstring.
        self._firing: PulseSequence | None = None
        # Per-point wall-clock pacing of the firing streamed scan (None when no scan is firing):
        # {"n_points": N, "scan_repeats": K, "base_dt": per-point seconds}.  ``_scan_progress``
        # derives the current (point, sweep) from the wall clock; a finite scan (K>0) stops once
        # K sweeps are played, exactly like the host issues the engine stop on the real backend.
        self._scan_info: dict[str, float] | None = None
        self._scan_fire_time: float = 0.0
        # Latched once a finite scan has played its K sweeps: the scan reading then keeps returning
        # the SATURATED done value (not idle) until the next prepare/stop -- == the real backend.
        self._scan_done: bool = False
        # The virtual TRIGGER CABLES plugged into this streamer's output lines: each wired
        # camera registers a callback (see _TriggerWiredCamera._wire_to) that fire() invokes
        # with the fired program -- the software mirror of the electrical trigger edges the
        # FPGA would emit.  Purely a device-layer artefact (the one legal lowest-layer fake).
        self._fire_listeners: list = []

    @property
    def history(self) -> list[dict[str, object]]:
        """The service's action log (prepare/fire/wait_done/safe...), so tests + snapshot read the
        single state machine's history rather than a second copy kept here."""
        return self.service.history

    @property
    def last_payload_json(self) -> str | None:
        """The sync-to-device handle the service recorded on the last prepare/fire (PulseTableState
        JSON with ``periods``) -- delegated, so there is ONE record seam (no virtual copy to drift)."""
        return self.service.last_payload_json

    @property
    def firing(self) -> PulseSequence | None:
        """The repeat_forever program this in-process streamer is continuously playing, or None
        when idle/safe -- the override of :attr:`SequencerDevice.firing` (which defaults None for a
        real/remote streamer with no host-side firing flag)."""
        return self._firing

    def prepare(self, sequence: PulseSequence | PulseTableState):
        # Uploading a new program REPLACES whatever the streamer was playing: on real hardware a
        # prepare loads the next sequence, so the previous continuous firing is no longer what the
        # streamer will emit.  Clear ``_firing`` here (the ONE place upload happens) so a measurement
        # that prepare+fires a FINITE program right after a continuous On Pulse -- without an explicit
        # Stop -- does not leave the wired camera rendering against the STALE continuous program.
        # ``fire`` re-sets ``_firing`` iff the freshly prepared program is itself repeat_forever, so
        # the firing lifecycle is owned entirely by prepare (clear) / fire (set) / stop (clear).
        self._firing = None
        # Accept either a PulseSequence or a GUI PulseTableState (what bind_pulse/on_pulse and a
        # loaded pulse .json pass).  Compile a table to a PulseSequence so fire()/firing/acquire
        # all work on ONE type, carrying repeat_forever for a continuous (On Pulse) program.
        if isinstance(sequence, PulseTableState):
            channels = list(sequence.channels)
            program = sequence.to_sequence(clock_hz=self.clock_hz)
            # Carry the SOURCE state's cyclic intent onto the camera's firing handle: an On Pulse
            # (continuous until Stop) or a GUI/notebook scan arrives as repeat_forever=True on the
            # PulseTableState, so the in-process streamer keeps firing.  Derived from the state the
            # caller handed in (the fire seam owns the cyclic intent), NOT re-derived from the compiled
            # flag -- the same source the real backend reads.
            if bool(getattr(sequence, "repeat_forever", False)):
                program = program.forever()
        else:
            program = sequence
            # The program defines its OWN channels (friendly imaging names, or a saved pulse
            # table's board chNN) -- derive them from the pulses rather than this sequencer's
            # default list, so a virtual streamer plays whatever channel set the pulse uses.
            channels = sorted({p.channel for p in program.base_pulses()}) or list(self.channels)
        # Validate the TIMING against the clock grid (the channel set is intrinsic to the program).
        program.validate(clock_hz=self.clock_hz).raise_if_failed()
        # Delegate the state machine to the service on the per-prepare channel set: it compiles
        # the RuntimeSequenceProgram, runs the FPGA-geometry backstop, records the SOURCE timing as
        # syncable last_payload_json, and advances to "prepared" -- exactly the seam the GUI's Sync
        # and the real SequencerService share, with no virtual copy.  The channel set is intrinsic
        # to the program (a saved table's chNN, the imaging names), so point the service at it.
        from .sequencer import RuntimeSequenceProgram
        self.service.channels = list(channels)
        self.last_program = RuntimeSequenceProgram.from_dict(self.service.prepare(sequence))
        self._prepared = program
        # Capture scan-progress pacing from the SOURCE table (the compiled program drops the
        # scan_table): N points + the requested sweep count, plus a per-point wall-clock estimate
        # (the program's one-frame duration) so the scan reading reports "point K / N · sweep r".
        self._scan_done = False          # a fresh prepare clears any prior finite-scan done latch
        self._scan_fire_time = 0.0       # prepared but not yet fired -> scan_progress idles until fire()
        rows = list(getattr(sequence, "scan_table", None) or []) if isinstance(sequence, PulseTableState) else []
        if rows:
            self._scan_info = {
                "n_points": float(len(rows)),
                "scan_repeats": float(max(0, int(getattr(sequence, "scan_repeats", 0)))),
                "base_dt": float(max(program.duration, 1e-9)),
            }
        else:
            self._scan_info = None
        return self.last_program

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self._prepared is None:
            raise RuntimeError("VirtualSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self._prepared:
            raise RuntimeError("VirtualSequencer.fire() received a sequence that was not prepared.")
        # Delegate the state transition (idle->running + history) to the shared service.  fire() is
        # NON-BLOCKING, exactly like the real FPGA (a quick register write that STARTS playback): the
        # program's real wall-clock is consumed where the hardware actually blocks -- ``wait_done``
        # (a finite ``on_pulse(wait=True)``) or the camera's per-frame readout (the live monitor) --
        # never here.  So a continuous (repeat_forever) On Pulse returns at once and the live camera
        # then paces at the firing cadence.
        self.service.fire()
        # A repeat_forever program keeps the streamer firing (continuous camera triggers) until
        # set_safe_state.  A finite program plays ONCE -- the measurement arms the camera, fires
        # here, then reads the frames back -- so it does NOT leave the streamer continuously
        # firing (and must not make a later live read see a stale finite shot).
        if bool(getattr(self._prepared, "repeat_forever", False)):
            self._firing = self._prepared
        # Stamp the wall-clock fire time for ANY streamed scan (a prepared scan_table), whether it is
        # a cyclic repeat_forever sweep OR a finite single-pass one (repeat_forever=False + scan_points):
        # ``_scan_progress`` derives the played-point count from this timestamp, so single-pass progress
        # advances too (keying scanning off the scan_table, not repeat_forever -- virtual == real, which
        # the real backend already does for a single-pass streamed scan).
        if self._scan_info is not None:
            self._scan_fire_time = time.monotonic()
        # Emit the trigger edges: every wired camera (the virtual trigger cables) sees this
        # fire.  An ARMED camera renders the frames a finite program's edges gate; a
        # continuous program is instead consumed at the firing cadence via ``firing``.
        for listener in tuple(self._fire_listeners):
            listener(self._prepared)

    # ------------------------------------------------------------------ trigger cables
    def add_fire_listener(self, listener) -> None:
        """Plug a virtual trigger cable in: ``listener(program)`` is invoked on every fire.
        Registered by a wired camera at construction (``sequencer="$device:sequencer"`` in the
        device config) -- the simulation of the physical trigger wire, device layer only."""
        if listener not in self._fire_listeners:
            self._fire_listeners.append(listener)

    def remove_fire_listener(self, listener) -> None:
        """Unplug a trigger cable (no error if it was never plugged in)."""
        try:
            self._fire_listeners.remove(listener)
        except ValueError:
            pass

    def _scan_progress(self) -> dict:
        """Where the streamed scan is now -- the real-time-paced reading wired into the service as
        its scan_progress_callback (mirrors the real streamer; virtual==real).  The monotonic
        played-point count is the wall-clock elapsed since fire divided by the per-point estimate
        (scaled by ``sleep_scale`` like every other virtual time, so ``sleep_scale=0`` fast-forwards
        a finite scan straight to done).  The point/sweep math + idle/done shape are single-sourced
        in ``scan_progress_fields``; only the per-point ``base_dt`` pacing is virtual-specific.  A
        finite scan (scan_repeats>0) that has played its K sweeps stops firing here, exactly like the
        host issues the engine stop on the real backend."""
        from .sequencer import SCAN_PROGRESS_IDLE, scan_progress_fields
        if self._scan_info is None:
            return dict(SCAN_PROGRESS_IDLE)
        n = int(self._scan_info["n_points"])
        k = int(self._scan_info["scan_repeats"])
        if self._scan_done:              # finite scan finished -> latch the SATURATED done reading (== real)
            return scan_progress_fields(max(1, k) * n, n, k)
        # Scanning is judged by whether a streamed scan has been FIRED (``_scan_fire_time``), NOT by the
        # ``_firing`` (repeat_forever) handle: a finite SINGLE-PASS streamed scan (repeat_forever=False +
        # scan_points) advances its progress too, exactly like the real backend, and ``_firing`` is only
        # set for a cyclic repeat_forever sweep.  Prepared-but-not-yet-fired (fire time 0) reads idle.
        if self._scan_fire_time <= 0.0:
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

    def scan_progress(self) -> dict:
        """The live scan reading -- delegated to the service (which calls our real-time-paced
        ``_scan_progress`` callback), so virtual and real expose the one ``scan_progress`` seam."""
        return self.service.scan_progress()

    def wait_done(self, timeout: float | None = None) -> bool:
        """Block until the prepared FINITE program finishes on the wall clock
        (``duration x sleep_scale``) -- this IS the real time a measurement's ``on_pulse(wait=True)``
        takes (real-time by default; the test suite sets sleep_scale=0 to fast-forward).

        Kept on the virtual backend rather than delegated because a finite STREAMED scan
        (``repeat_forever`` + ``scan_repeats=K>0``) DOES finish after K sweeps, yet the bare
        ``SequencerService.wait_done`` -- having no hardware wait_done_callback -- reports any
        repeat_forever program as never-done and even refuses to wait without a timeout.  This
        override is the real-time pacing the service cannot express on its own."""
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
                self.service.history.append({"action": "wait_done", "ok": True})
                return True
            self.service.history.append({"action": "wait_done", "ok": False})
            return False
        delay = float(self._prepared.duration) * self.sleep_scale
        if timeout is not None and delay > float(timeout):
            self.service.history.append({"action": "wait_done", "ok": False})
            return False
        if delay > 0:
            time.sleep(delay)
        self.service.history.append({"action": "wait_done", "ok": True})
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
        from the GUI clears the firing state and the live image freezes.

        ALWAYS drives the service to its safe state (unconditionally, like real hardware: Stop
        parks the outputs whatever was running) -- not only when a repeat_forever ``_firing`` was
        set.  A finite single-pass streamed scan leaves ``_firing`` None yet still must be parked
        on Stop, and a redundant safe-state on an already-idle streamer is a harmless no-op."""
        self.service.set_safe_state()
        self._firing = None
        self._scan_info = None
        self._scan_done = False
        self._scan_fire_time = 0.0

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
    # Both cameras are WIRED to the streamer ("$device:sequencer" = the virtual trigger
    # cable): fired programs reach a camera only over that wire, exactly like the physical
    # trigger line on the real rig (see _TriggerWiredCamera).
    return {
        "trap_array": {"type": "VirtualTrapArray"},
        "camera": {"type": "VirtualCamera",
                   "params": {"trap_array": "$device:trap_array", "sequencer": "$device:sequencer"}},
        "monitor_camera": {"type": "VirtualMotCamera", "params": {"sequencer": "$device:sequencer"}},
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
                # ``sleep_scale`` is set ONCE on the sequencer; the camera reads it through the wire
                # (its ``sleep_scale`` property), so there is no camera-side mirror to keep in sync --
                # every connect path (params or the sequencer-override dict) lands in one place.
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
