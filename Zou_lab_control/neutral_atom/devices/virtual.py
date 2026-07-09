"""Virtual devices for offline notebook tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin, sqrt
from pathlib import Path
from typing import Mapping, Sequence
import threading
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
from .base import (
    ROI_CLEAR_SENTINELS, SOFTWARE_TRIGGER, CameraDevice, DeviceProperty, LaserDevice, RFSourceDevice,
    SequencerDevice, TrapArrayDevice, snap_subarray)
from .camera_trigger import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    base_cycle_camera_trigger_pulses,
    base_cycle_trigger_pulses,
    count_camera_trigger_pulses,
    normalize_trigger_channels,
)
from ..timing import (
    BUS_SAFE_SIGNED_LEVEL,
    DEFAULT_EXPOSURE_S,
    PulseSequence,
    PulseTableState,
    probe_channel_set,
)


# The MOT coil DAC buses -- {bus name: bit channels LSB..MSB} -- THE single source both the
# virtual channel catalog (DEFAULT_CHANNELS below) and the monitor camera's default coil wiring
# (VirtualMotCamera.coil_buses) derive from, mirroring pulses/mot_field_template.json (three
# 6-bit buses).  Keeping the bit names in ONE place is what lets the shipped MOT template load
# onto a stock VirtualSequencer as a SUBSET of the device catalog (the same direction the real
# rig enforces: board.xdc defines the catalog, a template may only use part of it).
MOT_COIL_BUSES: dict[str, tuple[str, ...]] = {
    "da_x": tuple(f"dx{i}" for i in range(6)),
    "da_y": tuple(f"dy{i}" for i in range(6)),
    "da_z": tuple(f"dz{i}" for i in range(6)),
}

# Display LABELS for the coil bits -- ``dx0 -> "da_x[0]"`` -- so the GUI folds the 18 coil channels
# into three ``da_x``/``da_y``/``da_z`` bus rows exactly the way the real rig does: the streamer
# carries CHANNEL names (``dx0`` here, ``chNN`` on hardware) plus a LABEL per channel written in
# ``base[bit]`` syntax; ``infer_bus_channels`` folds on the label regex.  Derived from
# MOT_COIL_BUSES (never a second hand-typed table) so name<->label stay in lockstep, and put on the
# VirtualSequencer (not the template) so the fold works straight off the DEVICE with NO template
# loaded: whenever the coil bits are SHOWN (they are not in the editor's first-four default visible
# set), they draw as three bus rows, matching real == virtual (real labels come off the board XDC).
MOT_COIL_LABELS: dict[str, str] = {
    channel: f"{bus}[{bit}]"
    for bus, members in MOT_COIL_BUSES.items()
    for bit, channel in enumerate(members)
}

# The virtual device's FULL channel catalog -- every simulated signal the fake rig owns, exactly
# like the real catalog is the FULL board.xdc pin list (62 channels), not whatever subset one
# saved template happens to drive.  A pulse template must be a SUBSET of this (virtual == real:
# the real service.prepare raises on unknown channels), and the GUI's "Show All" can only show
# rows that exist in the loaded state -- so the catalog carries ALL the roles the virtual
# physics understands: the atom-array imaging lines, the MOT monitor trigger, and the three
# coil DAC buses (derived from MOT_COIL_BUSES, never a second hand-typed copy).
DEFAULT_CHANNELS = (
    "trap",
    "cooling",
    "probe",
    "emCCD",
    "pushout",
    "microwave",
    "mot_trigger",
    *(channel for members in MOT_COIL_BUSES.values() for channel in members),
)

# The virtual backend is a REAL-TIME hardware simulator: firing a pulse program TAKES its
# real wall-clock duration, exactly like the FPGA + externally-triggered camera.  So a live
# 2D monitor updates at the pulse cadence -- set the imaging period to several seconds and the
# image visibly slows to one frame every several seconds (``sleep_scale=1.0``).  The pytest
# suite fast-forwards this (conftest sets ``DEFAULT_SLEEP_SCALE = 0.0``) so the SAME data path
# runs without the wall-clock waits; only the time pacing is skipped, never the physics/data.
DEFAULT_SLEEP_SCALE = 1.0


# --- Grey-molasses (Lambda-enhanced sub-Doppler) cooling of Rb87, on the D1 line -------------------
# Physical anchors (Steck, Rb87 D line data; Rosi et al. Sci. Rep. 2018): the D1 line is 794.98 nm with
# natural linewidth Gamma/2pi = 5.746 MHz; the ground-state hyperfine splitting is 6.834682 GHz (the RF
# sideband that makes the repumper, i.e. the two-photon / Raman reference).  The cooling light is a
# single beam LOCKED blue on D1 -- there is no separately-tunable laser "detuning"; the detuning that
# matters is the TWO-PHOTON (Raman) detuning delta between the cooling and repump beams, which the RF
# sideband produces.  Grey molasses cools strongest at delta = 0 (a coherent dark state) on the D1 line.
# We model the RESULT as a multiplier on the cooling floor: 1.0 at that optimum, rising as delta departs
# from 0 -- so the calibrated ``cooled_temperature_K`` is preserved at delta = 0 and a wrong RF detuning
# (delta != 0) or a wrong laser wavelength (off D1) makes the atoms warmer / lost.
RB87_D1_WAVELENGTH_NM = 794.98
RB87_D1_LINEWIDTH_HZ = 5.746e6
RB87_HYPERFINE_HZ = 6.834682610e9
GM_HOT_FACTOR = 6.0                    # floor multiplier for "no cooling" (off D1 / far mis-tuned delta)


def grey_molasses_cooling_factor(two_photon_detuning_gamma: float, saturation: float,
                                 on_d1: bool = True) -> float:
    """The multiplier on the sub-Doppler cooling floor from the grey-molasses tuning -- ``1.0`` at the
    optimum (on the D1 line, two-photon delta = 0), growing as delta departs from 0 (warmer atoms), up to
    :data:`GM_HOT_FACTOR` (effectively no cooling -> atoms hot / lost).  The single source the virtual
    atom array multiplies its ``cooled_temperature_K`` by.

    The ONLY detuning knob is the RF-produced two-photon detuning ``delta`` (the laser has none -- it is
    locked blue on D1).  Failure modes: OFF the D1 line (wrong wavelength) -> no cooling; wrong RF
    (delta != 0) -> a Fano feature about delta = 0: a SHARP heating rise for delta > 0 and a GENTLE
    warming for delta < 0 (the coherent dark state is spoiled asymmetrically).  Higher ``saturation``
    power-broadens the dark resonance (a wider delta window still cools)."""
    if not on_d1:
        return GM_HOT_FACTOR
    delta_2p = float(two_photon_detuning_gamma)
    half_width = 0.05 + 0.02 * max(float(saturation), 0.0)  # power-broadened dark-state half-width (Gamma)
    wing = 3.0 if delta_2p >= 0.0 else 0.3                  # heating (>0) vs weak-cooling (<0) asymmetry
    fano = 1.0 + wing * (delta_2p / half_width) ** 2
    return float(min(GM_HOT_FACTOR, max(1.0, fano)))


class VirtualLaser(LaserDevice):
    """A pure set-point grey-molasses cooling laser for the virtual rig -- a beam locked blue on the Rb87
    D1 line, so it just HOLDS its wavelength + saturation (no hardware), which :class:`VirtualTrapArray`
    reads.  It has no "detuning" knob (the D1 lock is fixed; the grey-molasses detuning is the RF two-
    photon delta).  Defaults are the D1 optimum (794.98 nm, s ~ 3), so a fresh rig cools well and a wrong
    wavelength an operator dials in takes it off the line (no cooling)."""

    D1_WINDOW_NM = 0.02        # within this of the D1 line -> "on the line" (cools); beyond -> no GM

    # Each live knob is declared ONCE as a DeviceProperty: the descriptor IS the property AND
    # auto-injects its device-viewer control (confocal ``ManagedProperty`` -> ``gui_dict``), so
    # ``runtime_controls`` is derived, never hand-typed.  Two of the three device-param kinds show here
    # -- FLOAT (wavelength / saturation, scrollable spin boxes) and BOOL (``beam_on`` writable switch,
    # ``on_d1`` a read-only derived read-back).
    wavelength_nm = DeviceProperty(
        "float", label="wavelength", unit="nm", lo=700.0, hi=900.0, default=RB87_D1_WAVELENGTH_NM,
        tooltip="Laser wavelength; the Rb87 D1 line is 794.98 nm.  Off the line the grey molasses does "
                "not cool (atoms stay hot / are lost).")
    saturation = DeviceProperty(
        "float", label="saturation I/Isat", lo=0.0, hi=100.0, default=3.0,
        tooltip="Total beam saturation parameter I/I_sat (~1..10 for grey molasses); higher power "
                "broadens the two-photon dark resonance.")
    beam_on = DeviceProperty(
        "bool", label="beam on", default=True,
        tooltip="Whether the cooling beam is on -- a pure on/off state flag.")

    def __init__(self, *, wavelength_nm: float = RB87_D1_WAVELENGTH_NM, saturation: float = 3.0):
        self.wavelength_nm = wavelength_nm    # through the descriptor (validates: clamps to [700, 900])
        self.saturation = saturation          # clamps to [0, 100]

    on_d1 = DeviceProperty(
        "bool", label="on D1 line",
        tooltip="Whether the wavelength is close enough to the Rb87 D1 line to cool (a derived read-back).")

    @on_d1.getter
    def on_d1(self) -> bool:
        return abs(self.wavelength_nm - RB87_D1_WAVELENGTH_NM) <= self.D1_WINDOW_NM

    # No snapshot override: BaseDevice.snapshot auto-dumps every DeviceProperty knob above
    # (wavelength_nm / saturation / beam_on / on_d1) -- the knob is declared once, dumped from that.


class VirtualRF(RFSourceDevice):
    """A pure set-point RF/EOM source for the virtual rig -- it PRODUCES the two-photon Raman detuning
    for grey molasses.  ``two_photon_detuning_gamma`` (delta in linewidths) is the primary knob and is
    read/write: the getter derives it from ``frequency_hz`` relative to the Rb87 ground hyperfine
    splitting, the setter moves ``frequency_hz`` to realise a requested delta (so a detuning scan just
    writes delta in Gamma).  The default is exactly on the splitting (delta = 0, the dark-state
    resonance), so a fresh rig cools well and a wrong detuning heats."""

    # Each live knob is declared ONCE as a DeviceProperty (confocal ``ManagedProperty``): the
    # descriptor IS the property AND auto-injects its control, so ``runtime_controls`` is derived.
    # ALL THREE device-param kinds show here -- FLOAT (delta / frequency / power spin boxes), BOOL
    # (``drive_on`` switch) and CHOICE (``waveform`` combo).  ``two_photon_detuning_gamma`` is declared
    # FIRST so the grey-molasses knob leads the catalog; its getter / setter derive it from
    # ``frequency_hz`` (bidirectional), the rest auto-store.
    two_photon_detuning_gamma = DeviceProperty(
        "float", label="two-photon δ", unit="Γ", lo=-50.0, hi=50.0,
        tooltip="Two-photon (Raman) detuning δ in linewidths Γ -- the grey-molasses knob; δ = 0 (RF on "
                "the 6.834682 GHz hyperfine line) is the dark-state resonance (best cooling).  Setting it "
                "moves the RF frequency.")

    @two_photon_detuning_gamma.getter
    def two_photon_detuning_gamma(self) -> float:
        return (self.frequency_hz - RB87_HYPERFINE_HZ) / RB87_D1_LINEWIDTH_HZ

    @two_photon_detuning_gamma.setter
    def two_photon_detuning_gamma(self, delta_gamma: float) -> None:
        # delta is realised by moving the drive frequency off the hyperfine splitting by delta linewidths.
        self.frequency_hz = RB87_HYPERFINE_HZ + float(delta_gamma) * RB87_D1_LINEWIDTH_HZ

    frequency_hz = DeviceProperty(
        "float", label="frequency", unit="Hz", lo=0.0, hi=2e10, default=RB87_HYPERFINE_HZ,
        tooltip="RF/EOM drive frequency generating the repump sideband; the Rb87 ground hyperfine "
                "splitting is 6.834682 GHz (δ = 0).  The raw set-point behind δ.")
    power_dbm = DeviceProperty(
        "float", label="power", unit="dBm", lo=-100.0, hi=40.0, default=0.0,
        tooltip="RF drive power.")
    drive_on = DeviceProperty(
        "bool", label="drive on", default=True,
        tooltip="Whether the RF drive is on -- a pure on/off state flag.")
    waveform = DeviceProperty(
        "choice", label="waveform", choices=("sine", "square", "triangle"), default="sine",
        tooltip="RF drive waveform.")

    def __init__(self, *, frequency_hz: float = RB87_HYPERFINE_HZ, power_dbm: float = 0.0):
        self.frequency_hz = frequency_hz      # through the descriptor (clamps to [0, 2e10])
        self.power_dbm = power_dbm            # clamps to [-100, 40]

    # No snapshot override: BaseDevice.snapshot auto-dumps every DeviceProperty knob above
    # (two_photon_detuning_gamma / frequency_hz / power_dbm / drive_on / waveform) from the one decl.


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
    # Optional grey-molasses cooling devices (wired via ``$device:laser`` / ``$device:rf`` in the config).
    # When BOTH are bound, the cooling floor becomes ``cooled_temperature_K`` scaled by the laser + RF
    # tuning (``_cooling_floor_K``): at the D1 grey-molasses optimum the factor is 1 (floor unchanged), and
    # a wrong RF two-photon detuning / a laser off the D1 line warms or loses the atoms.  ``None`` (no such
    # devices) keeps the fixed floor, so a rig without a laser/RF behaves exactly as before.
    laser: "LaserDevice | None" = None
    rf: "RFSourceDevice | None" = None
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

    def _cooling_floor_K(self) -> float:
        """The sub-Doppler cooling floor the atoms relax toward: ``cooled_temperature_K`` at the optimum,
        scaled by the grey-molasses tuning of the BOUND laser + RF (both must be present).  With no such
        devices it is the fixed floor, so a rig without a laser/RF is unchanged.  A wrong RF two-photon
        detuning (delta != 0) or a laser off the D1 line raises the floor -- warmer atoms, then lower
        release-recapture survival -- via :func:`grey_molasses_cooling_factor` (the ONE physics source)."""
        laser, rf = self.laser, self.rf
        if laser is None or rf is None:
            return self.cooled_temperature_K
        factor = grey_molasses_cooling_factor(
            getattr(rf, "two_photon_detuning_gamma", 0.0),
            getattr(laser, "saturation", 3.0),
            bool(getattr(laser, "on_d1", True)))
        return float(self.cooled_temperature_K) * factor

    def reload(self, *, cooling_duration: float | None = None) -> np.ndarray:
        """Load a FRESH atom array -- the cooling/MOT light + PGC at the start of a
        shot.  Each tweezer independently captures a single atom with probability
        :meth:`loading_fraction` (which grows with ``cooling_duration``), and every
        loaded atom starts cooled to the grey-molasses / PGC floor (:meth:`_cooling_floor_K`)."""
        self.occupancy = self.rng.random(self.n_sites) < self.loading_fraction(cooling_duration)
        self.temperature_K = np.full(self.n_sites, self._cooling_floor_K(), dtype=float)
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
        self.temperature_K = np.full(self.n_sites, self._cooling_floor_K(), dtype=float)
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
        floor = self._cooling_floor_K()
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
        trig = normalize_trigger_channels(capture_trigger_channels)
        if sequence is None:
            return []
        # How many camera-trigger windows does ONE base cycle carry on this camera's line?  The
        # ONE authoritative "does/how much does this pulse trigger the camera" reading
        # (camera_trigger.base_cycle_trigger_pulses: rising edges only -- ``pulse.value`` -- on
        # the delivered base cycle), shared by the live gate below AND the bracket criterion.
        base_windows = base_cycle_trigger_pulses(sequence, trigger_channels=trig)
        # The camera-trigger gate is a LIVE-monitor concept: a CONTINUOUS (``repeat_forever``)
        # firing renders only while it pulses a camera-trigger line (a RISING edge -- an
        # explicit value=0 entry drives the line LOW and gates nothing, exactly like hardware),
        # and FREEZES the moment it does not (Stop Pulse -> a non-imaging safe state).  A FINITE
        # measurement sequence (sitemap / threshold / detect / detection-time scan / reference
        # bracket) is one the measurement deliberately fired to read ``frames`` windows, so it
        # is rendered UNCONDITIONALLY -- its per-frame physics (cooling/trap-off/trap-hold) is
        # parsed against the camera's trigger line, defaulting to independent re-loads when the
        # bound pulse carries no trigger of its own.
        if getattr(sequence, "repeat_forever", False) and base_windows <= 0:
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
        single_cycle = base_windows >= frames
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
        out = super().snapshot()          # the ``type`` key has ONE producer: BaseDevice.snapshot
        out.update({
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
        })
        return out

    def close(self) -> None:
        pass


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
        edges = count_camera_trigger_pulses(self, sequence)
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
        cycle = base_cycle_camera_trigger_pulses(self, firing)
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
    def __init__(self, trap_array: VirtualTrapArray, exposure: float = DEFAULT_EXPOSURE_S, timeout: float = 2.0,
                 subarray_step: int = 4, capture_trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
                 sequencer=None):
        self.trap_array = trap_array
        self._exposure = self._validated_exposure(exposure)
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
        self._exposure = self._validated_exposure(value)

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
            self.exposure = exposure          # the setter validates (_validated_exposure)
        if roi is not None:
            if roi in ROI_CLEAR_SENTINELS:
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
            capture_trigger_channels=self.effective_trigger_channels,
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
        out = super().snapshot()          # the ``type`` key has ONE producer: BaseDevice.snapshot
        out.update({
            "exposure": self.exposure,
            "roi": self._roi,
            "timeout": self.timeout,
            "last_sequence": self.last_sequence,
        })
        return out

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
    chain.

    ``trigger_source`` selects the SAME two acquisition modes the real :class:`~.pylon.PylonCamera`
    keeps (one shared vocabulary, :data:`~.base.SOFTWARE_TRIGGER`):

    * ``"Software"`` (the default -- matching the real monitor camera's discovery config): FREE-RUN.
      The sensor exposes on its own clock and EVERY ``acquire`` yields a frame with no pulse wiring
      at all; the trigger wire's edges are ignored (Basler ``TriggerMode Off``).  It images whatever
      the streamer's outputs drive right now: the continuously firing program, else the steady state
      the last finite fire left latched on the DACs, else the safe state (all-zero levels -> the dim
      background MOT) -- a free-running sensor delivers dark frames, it never freezes.
    * a sequencer channel name (e.g. ``"mot_trigger"``): HARDWARE trigger.  One frame per capture
      edge on THAT line, the pure externally-triggered grabber -- an idle wire yields no frame and
      the live view freezes (``capture_trigger_channels`` is then ``(trigger_source,)``)."""

    def __init__(self, *, width: int = 1920, height: int = 1200, exposure: float = 0.05,
                 coil_buses: Mapping[str, Sequence[str]] | None = None,
                 b0: Mapping[str, float] | None = None,
                 b_sigma: Mapping[str, float] | None = None,
                 peak_counts: float = 93.0, offset_counts: float = 7.0, read_noise: float = 1.5,
                 spot_size_px: tuple[float, float] = (40.0, 20.0),
                 timeout: float = 2.0, seed: int | None = None,
                 trigger_source: str = SOFTWARE_TRIGGER,
                 sequencer=None):
        # FAITHFUL to the real monitor camera (Basler acA1920-155um defaults): the full 1920x1200
        # sensor, 50 ms exposure, and 8-bit frames whose MOT is a FLAT ~40x20 px bright blob
        # (peak ~offset+peak_counts ~= 100 counts at the capture optimum) over a 5-10 count noisy
        # background -- so the virtual pipeline carries the SAME 2.3 MP uint8 payload per frame the
        # real pylon stream does, and profiling/console behaviour transfer 1:1.
        self.width, self.height = int(width), int(height)
        self._exposure = self._validated_exposure(exposure)
        self.timeout = positive_float(timeout, "timeout")
        self.trigger_source = str(trigger_source)
        # The WIRING declaration mirrors the real PylonCamera exactly (virtual == real): a concrete
        # line name IS the wire the edge-faithful frame count reads (a MOT probe pulsing
        # ``mot_trigger`` once per cycle delivers one monitor frame); in ``Software`` free-run the
        # declaration is the same inert conservative default the real discovery config carries.
        # The ACTIVE counting line is the base-class ``effective_trigger_channels`` -- ``()`` while
        # free-running (the sensor never consults its trigger input), derived ONCE in devices.base.
        self.capture_trigger_channels = (
            DEFAULT_CAMERA_TRIGGER_CHANNELS if self._free_run else (self.trigger_source,))
        # The coil DAC buses this camera's MOT responds to: {bus name: member bit channels LSB..MSB}.
        # The default is MOT_COIL_BUSES -- the ONE module-level source the virtual channel catalog
        # (DEFAULT_CHANNELS) also derives from, so the sensor's wiring and the streamer's catalog
        # can never drift apart (they used to be two hand-typed mirror copies).  A real setup
        # names its own buses in the device config.
        self.coil_buses = {str(k): tuple(str(c) for c in v)
                           for k, v in dict(coil_buses or MOT_COIL_BUSES).items()}
        self.b0 = {str(k): float(v) for k, v in dict(b0 or {"da_x": 7.0, "da_y": -5.0, "da_z": 11.0}).items()}
        self.b_sigma = {str(k): positive_float(v, "b_sigma") for k, v in dict(
            b_sigma or {"da_x": 6.0, "da_y": 6.0, "da_z": 6.0}).items()}
        self.peak_counts = nonnegative_float(peak_counts, "peak_counts")
        self.offset_counts = nonnegative_float(offset_counts, "offset_counts")
        self.read_noise = nonnegative_float(read_noise, "read_noise")
        sx, sy = spot_size_px
        self.spot_size_px = (positive_float(sx, "spot_size_px[0]"), positive_float(sy, "spot_size_px[1]"))
        self._rng = np.random.default_rng(seed)
        self._roi: tuple[int, int, int, int] | None = None
        self.last_levels: dict[str, int] | None = None    # what the last frame sensed (snapshot/debug)
        # FREE-RUN producer state (the faithful LatestImageOnly stream, see _arm/_producer_loop):
        # the sensor thread exposes on its own clock into a ONE-deep latest slot; a consumer slower
        # than the frame period simply misses the overwritten frames, exactly like the real Basler.
        self._producer: threading.Thread | None = None
        self._producer_stop = threading.Event()
        # The virtual TRIGGER CABLE (see _TriggerWiredCamera): fired coil sequences reach this
        # sensor only through the wired in-process streamer.  None = no cable, never a frame.
        self._wire_to(sequencer)

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = self._validated_exposure(value)

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def configure(self, *, exposure: float | None = None, roi: object = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure", "roi"}, kwargs)
        if exposure is not None:
            self.exposure = exposure          # the setter validates (_validated_exposure)
        if roi is not None:
            self._roi = None if roi in ROI_CLEAR_SENTINELS else snap_subarray(
                tuple(roi), step=1, max_w=self.width, max_h=self.height)

    def mot_efficiency(self, levels: Mapping[str, float]) -> float:
        """The 3-D Gaussian capture efficiency at the given coil levels -- THE virtual MOT model,
        exposed so the end-to-end optimum test asserts against the same rule the frames obey."""
        z = 0.0
        for bus, b0 in self.b0.items():
            z += ((float(levels.get(bus, 0.0)) - b0) / self.b_sigma[bus]) ** 2
        return float(np.exp(-0.5 * z))

    # ``_free_run`` (software-trigger predicate) is inherited from CameraDevice -- the ONE copy both
    # this monitor camera and the real PylonCamera share, so the two backends can never drift.

    def _sense_levels(self, sequence: PulseSequence | None) -> dict[str, float]:
        """The coil levels the sensor sees -- THE one sense rule for both acquisition modes.

        A DRIVEN sequence is decoded from its compiled bit-channel pulses at the steady mid-point
        of the DELAYED base-cycle timeline (base_duration is delay-inclusive) -- the same timeline
        ``decode_analog_bus`` reads and ``edges()`` plays, so a ``.delay()`` on the coil bit
        channels moves the sensed window WITH the hardware output (the raw max(p.start+p.duration)
        put the sense time on the pre-delay timeline).  NO program at all (None) senses the SAFE
        state: every DAC parked at ``BUS_SAFE_SIGNED_LEVEL`` (the hardware mid-code = 0 V).  A
        program that never touches a coil bus needs NO special case here: ``decode_analog_bus``
        itself returns the safe level for an undriven bus -- the one safe-state rule lives there,
        never as a second copy in this gate (a TTL-only fire used to fall through to the decoder's
        old all-bits-low word and the MOT spot vanished)."""
        if sequence is None:
            return {bus: float(BUS_SAFE_SIGNED_LEVEL) for bus in self.coil_buses}
        from ..timing.sequence import decode_analog_bus
        t_sense = 0.5 * sequence.base_duration
        return {bus: decode_analog_bus(sequence, members, t_sense)
                for bus, members in self.coil_buses.items()}

    def _render_at_levels(self, levels: Mapping[str, float], frames: int) -> list[np.ndarray]:
        """THE one rendering core (levels -> efficiency -> spot + noise) BOTH acquisition modes
        share -- free-run and hardware trigger differ only in where ``levels`` come from, never in
        how a frame is made (no forked physics).

        Faithful 8-BIT frames at the real pylon payload: a flat elliptical MOT blob
        (``spot_size_px`` = FWHM ~40x20 px, peak ~``offset+peak_counts`` ~= 100 counts at the
        capture optimum) on a 5-10 count noisy background, ``uint8`` like the Basler Mono8 stream.
        PERFORMANCE: at 2.3 MP / 20 fps a full-frame Poisson draw (~80 ms) cannot keep the real
        frame period, and physically the ~7-count background is READ-NOISE dominated anyway -- so
        the background is one Gaussian field (offset + read noise) and the Poisson shot noise is
        drawn only inside the small spot window where the signal actually lives (~120x60 px)."""
        frames = positive_int(frames, "frames")
        self.last_levels = dict(levels)
        eff = self.mot_efficiency(levels)
        h, w = self.height, self.width
        cx, cy = w / 2.0, h / 2.0
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sx, sy = self.spot_size_px[0] / fwhm, self.spot_size_px[1] / fwhm
        # The spot's local window (+-3 sigma, clamped to the sensor): the ONLY region with signal.
        x0, x1 = max(0, int(cx - 3 * sx)), min(w, int(np.ceil(cx + 3 * sx)))
        y0, y1 = max(0, int(cy - 3 * sy)), min(h, int(np.ceil(cy + 3 * sy)))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        spot = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
        spot_rate = self.peak_counts * eff * spot            # mean SIGNAL counts inside the window
        out: list[np.ndarray] = []
        for _k in range(frames):
            frame = self.offset_counts + self._rng.normal(0.0, self.read_noise, size=(h, w))
            frame[y0:y1, x0:x1] += self._rng.poisson(spot_rate)
            frame = np.clip(frame, 0, 255).astype(np.uint8)  # Mono8, exactly the real pylon dtype
            if self._roi is not None:
                x, w_roi, y, h_roi = self._roi
                frame = frame[y:y + h_roi, x:x + w_roi]
            out.append(frame)
        return out

    def _render_frames(self, sequence: PulseSequence | None, frames: int) -> list[np.ndarray]:
        # The HARDWARE-TRIGGER path (edge-gated frames over the wire): a trigger edge exists only
        # when a sequence actually fired pulses -- no fired pulses -> no edge -> no frame, exactly
        # like a real externally-triggered sensor.  (Free-run never gates on this: its frames come
        # from the exposure clock in _grab, through the same _render_at_levels core.)
        frames = positive_int(frames, "frames")
        if sequence is None or not getattr(sequence, "pulses", None):
            return []
        return self._render_at_levels(self._sense_levels(sequence), frames)

    def _on_wire_fired(self, sequence) -> None:
        # ``Software`` mode ignores its trigger input entirely (the real Basler sets TriggerMode
        # Off): a fire's edges push no frames -- the free-running exposure clock in _grab is the
        # only frame source.  Hardware trigger keeps the edge-faithful push path.
        if self._free_run:
            return
        super()._on_wire_fired(sequence)

    def _scene_source(self):
        """What the free-running sensor images RIGHT NOW: the continuously firing program if there
        is one, else the STEADY state the last finite fire left latched on the DAC pins, else the
        safe state (all-zero levels -> the dim background MOT + noise)."""
        wire = getattr(self, "sequencer", None)
        if wire is None:
            return None
        return wire.firing if wire.firing is not None else getattr(wire, "last_fired", None)

    def _producer_loop(self, pace: float) -> None:
        """The free-running SENSOR clock (its own thread), faithful to the real Basler
        LatestImageOnly stream: expose one frame per ``exposure`` period into a ONE-deep latest
        slot.  A consumer slower than the frame period simply never sees the overwritten frames --
        the SAME drop semantics (and therefore the same console "display fell behind" advisory)
        the real pylon free-run produces.  The producer runs for the armed session only."""
        while not self._producer_stop.wait(pace):
            frame = self._render_at_levels(self._sense_levels(self._scene_source()), 1)[0]
            with self._latest_cond:
                self._latest = frame
                self._latest_seq += 1
                self._latest_cond.notify_all()

    def _arm(self, frames: int | None) -> None:
        super()._arm(frames)
        # FREE-RUN with real pacing: start the resident sensor stream (the virtual StartGrabbing).
        # pace == 0 (pytest fast-forward, sleep_scale = 0) keeps the deterministic render-on-demand
        # path in _grab -- same physics core, only WHO ticks the clock differs.
        pace = float(self.exposure) * self.sleep_scale
        if self._free_run and pace > 0.0 and self._producer is None:
            self._latest, self._latest_seq, self._consumed_seq = None, 0, 0
            self._latest_cond = threading.Condition()
            self._producer_stop.clear()
            self._producer = threading.Thread(
                target=self._producer_loop, args=(pace,), name="virtual-mot-sensor", daemon=True)
            self._producer.start()

    def _disarm(self) -> None:
        if self._producer is not None:
            self._producer_stop.set()
            self._producer.join(timeout=2.0)
            self._producer = None
        super()._disarm()

    def _grab(self, n: int, *, timeout: float | None = None, stop=None) -> bool:
        if not self._free_run:
            return super()._grab(n, timeout=timeout, stop=stop)
        if stop is not None and stop.is_set():
            return False
        if self._producer is not None:
            # REAL pacing: consume the producer's LATEST frame (never a backlog) -- block until a
            # frame NEWER than the last consumed one lands, exactly like RetrieveResult on a
            # LatestImageOnly stream.  Frames the producer overwrote while we were away are LOST,
            # which is the honest free-run behaviour the console's amber advisory reports.
            deadline = time.monotonic() + (self.timeout if timeout is None else float(timeout))
            with self._latest_cond:
                while self._latest is None or self._latest_seq == self._consumed_seq:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or (stop is not None and stop.is_set()):
                        return False
                    self._latest_cond.wait(min(0.05, remaining))
                self._consumed_seq = self._latest_seq
                frame = self._latest
            self._deliver([frame])
            return True
        # pace == 0 (tests): render on demand -- the exposure clock is fast-forwarded away.
        self._deliver(self._render_at_levels(self._sense_levels(self._scene_source()), 1))
        return True

    def snapshot(self) -> dict[str, object]:
        out = super().snapshot()          # the ``type`` key has ONE producer: BaseDevice.snapshot
        out.update({
            "exposure": self.exposure,
            "roi": self._roi,
            "trigger_source": self.trigger_source,
            "coil_buses": {k: list(v) for k, v in self.coil_buses.items()},
            "last_levels": self.last_levels,
        })
        return out

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

    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS, clock_hz: float = DEFAULT_CLOCK_HZ,
                 sleep_scale: float | None = None, channel_labels: dict | None = None):
        from .sequencer import SequencerService
        self.channels = tuple(str(channel) for channel in channels)
        # Display labels (channel -> ``base[bit]``) so the pulse GUI folds the coil bits into bus
        # rows -- the virtual analogue of the real sequencer reading its labels off the board XDC.
        # Restricted to the channels this instance actually carries (a custom channel list drops the
        # coil labels it does not use), so it never advertises a bus whose members are absent.
        labels = MOT_COIL_LABELS if channel_labels is None else dict(channel_labels)
        self.channel_labels = {ch: str(labels[ch]) for ch in self.channels if ch in labels}
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        # REAL-TIME by default (DEFAULT_SLEEP_SCALE=1.0): a fired program takes its real
        # duration, so the live camera paces with the pulse.  ``None`` -> the module default
        # (the pytest suite flips that default to 0 to fast-forward; see conftest).
        if sleep_scale is None:
            sleep_scale = DEFAULT_SLEEP_SCALE
        # The shared software state machine: prepare (with the FPGA-geometry backstop), fire,
        # source-payload recording and last_payload_json all live HERE, one copy.  Its built-in
        # pure-software wall-clock scan-progress is BYPASSED by wiring our real-time-paced
        # ``_scan_progress`` as the scan_progress_callback (the same callback seam the real AXI
        # backend uses for its hardware cursor) -- so the live-scan reading is single-sourced too.
        # ``sleep_scale`` is handed to the service and OWNED there (this adapter's ``sleep_scale``
        # is a delegating property) -- one copy, so settle / wait_done pacing / the wired cameras
        # all read the same value even when a caller flips it mid-session.
        self.service = SequencerService(
            channels=self.channels,
            clock_hz=self.clock_hz,
            sleep_scale=nonnegative_float(sleep_scale, "sleep_scale"),
            scan_progress_callback=self._scan_progress,
        )
        # The compiled program of the last prepare() the camera reads as ``firing`` -- a
        # PulseSequence (not the service's compiled RuntimeSequenceProgram), because the camera
        # needs its ``base_pulses()`` to detect the capture-trigger channel + its physics.
        self._prepared: PulseSequence | None = None
        # The compiled RuntimeSequenceProgram of the last prepare() (what on_pulse /
        # sync-to-device read for .sequence_name etc.).
        self.last_program = None
        # The program the streamer is CONTINUOUSLY firing (a repeat_forever pulse), or None when
        # idle/safe -- the virtual camera's gate; see the class docstring.
        self._firing: PulseSequence | None = None
        # The most recently FIRED program, whatever its kind -- the model of "what word the output
        # pins still HOLD" (see the ``last_fired`` property).  Cleared only by stop()/safe state.
        self._last_fired: PulseSequence | None = None
        # Per-point wall-clock pacing of the firing streamed scan (None when no scan is firing):
        # {"n_points": N, "scan_repeats": K, "base_dt": per-point seconds}.  ``_scan_progress``
        # derives the current (point, sweep) from the wall clock; a finite scan (K>0) stops once
        # K sweeps are played, exactly like the host issues the engine stop on the real backend.
        self._scan_info: dict[str, float] | None = None
        self._scan_fire_time: float = 0.0
        # Latched once a finite scan has played its K sweeps: the scan reading then keeps returning
        # the SATURATED done value (not idle) until the next prepare/stop -- == the real backend.
        self._scan_done: bool = False
        # When the FIRED finite program's playback completes on the wall clock (monotonic seconds),
        # stamped at fire().  The ONE anchor every waiter converges to: the wired camera's finite
        # read pacing (_finite_pace_until) and wait_done both wait to this same absolute instant,
        # so read_frames + wait_done back to back cost the play time ONCE, never twice.  None =
        # not fired yet (wait_done then anchors at its own call, the bare prepare+wait semantics).
        self._finite_play_deadline: float | None = None
        # The virtual TRIGGER CABLES plugged into this streamer's output lines: each wired
        # camera registers a callback (see _TriggerWiredCamera._wire_to) that fire() invokes
        # with the fired program -- the software mirror of the electrical trigger edges the
        # FPGA would emit.  Purely a device-layer artefact (the one legal lowest-layer fake).
        self._fire_listeners: list = []

    @property
    def sleep_scale(self) -> float:
        """The virtual time scale, OWNED by the composed service (the ONE copy every consumer
        reads: ``settle``/``wait_done`` pacing, the scan ``base_dt``, and the wired cameras
        through :attr:`_TriggerWiredCamera.sleep_scale`) -- a delegating view, so flipping it
        mid-session reaches every timing path at once, never a stale adapter-side mirror."""
        return float(self.service.sleep_scale)

    @sleep_scale.setter
    def sleep_scale(self, value) -> None:
        self.service.sleep_scale = nonnegative_float(value, "sleep_scale")

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

    @property
    def last_fired(self) -> PulseSequence | None:
        """The most recently FIRED program whose outputs the hardware still HOLDS, or None when
        the streamer sits in its safe state.  Physics: after a finite program finishes playing, the
        DAC/TTL words it drove stay LATCHED on the output pins -- nothing re-drives them until the
        next fire or until ``set_safe_state`` parks them (DAC mid-code = signed level 0).  A
        free-running (``Software``) monitor camera images exactly that steady state between fires.
        Distinct from :attr:`firing` (a repeat_forever program still PLAYING now) and from the
        merely-prepared program (uploaded, driving nothing yet): prepare does NOT clear this -- an
        upload replaces the next program but leaves the pins holding the last fired word."""
        return self._last_fired

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
        # Delegate the state machine to the composed service: it compiles the RuntimeSequenceProgram,
        # runs the FPGA-geometry backstop, records the SOURCE timing as syncable last_payload_json,
        # and advances to "prepared" -- exactly the seam the GUI's Sync and the real SequencerService
        # share, with no virtual copy.
        # CONVERGED with the real backend on the channel contract (was a KNOWN DIRECTION FORK): the
        # service keeps its FIXED device catalog (``self.channels``) and compiles against it, exactly
        # as the real ``SequencerService`` does -- it does NOT adopt whatever channels the program
        # uses.  A template whose channels fall OUTSIDE the catalog is REJECTED here, the SAME
        # rejection the real rig gives, instead of the old accommodation that silently widened the
        # catalog to the program (letting a virtual run "succeed" on a template real hardware would
        # refuse).  Catalog-respecting callers -- the imaging sequences, a GUI/notebook load aligned
        # through ``resolve_fireable_template`` -- present a SUBSET, so this is transparent to them;
        # only an unaligned out-of-catalog template now fails loud (virtual == real on channels).
        from .sequencer import RuntimeSequenceProgram
        unknown = [ch for ch in channels if ch not in self.channels]
        if unknown:
            raise ValueError(
                f"sequence uses channels {unknown} outside this sequencer's catalog "
                f"{list(self.channels)}; align the template to the device channels before firing "
                "(the real backend rejects out-of-catalog channels identically).")
        self.last_program = RuntimeSequenceProgram.from_dict(self.service.prepare(sequence))
        self._prepared = program
        # Capture scan-progress pacing from the SOURCE table (the compiled program drops the
        # scan_table): N points + the requested sweep count, plus a per-point wall-clock estimate
        # (the program's one-frame duration) so the scan reading reports "point K / N · sweep r".
        self._scan_done = False          # a fresh prepare clears any prior finite-scan done latch
        self._scan_fire_time = 0.0       # prepared but not yet fired -> scan_progress idles until fire()
        self._finite_play_deadline = None  # fresh program: no playback in flight
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
        # Whatever fired now owns the output pins: record it as the held steady state a
        # free-running monitor images between fires (see ``last_fired``).
        self._last_fired = self._prepared
        if bool(getattr(self._prepared, "repeat_forever", False)):
            self._firing = self._prepared
        else:
            # Anchor the finite program's play deadline AT FIRE: every waiter (the wired camera's
            # finite read pacing and wait_done below) waits to this same absolute instant, so a
            # measurement that reads its frames and THEN waits for program completion pays the
            # play time once -- exactly like real hardware, where the program ends when it ends,
            # not "duration after whoever happens to ask".
            self._finite_play_deadline = (
                time.monotonic() + float(self._prepared.duration) * self.sleep_scale)
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

    def wait_done(self, timeout: float | None = None, *, stop=None) -> bool:
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
            # report done.  An infinite scan / continuous On Pulse never finishes: the protocol's
            # deadlock guard applies exactly as on the bare service (virtual == real) -- waiting
            # forever raises, a bounded wait reports not-done.
            info = self._scan_info
            if info is not None and int(info["scan_repeats"]) > 0:
                delay = int(info["scan_repeats"]) * int(info["n_points"]) * float(info["base_dt"]) * self.sleep_scale
                if timeout is not None and delay > float(timeout):
                    return self._record_wait(False)
                if delay > 0:
                    self._sleep_interruptible(delay, stop)          # cooperatively cancellable, like settle
                if stop is not None and stop.is_set():
                    return self._record_wait(False)                 # cancelled mid-sweep -> not done, do NOT latch
                self._scan_done = True       # latch the saturated done reading (matches the real backend)
                self._firing = None
                return self._record_wait(True)
            if timeout is None:
                from .sequencer import WAIT_FOREVER_MESSAGE
                raise RuntimeError(WAIT_FOREVER_MESSAGE)
            return self._record_wait(False)
        # Wait to the fire-anchored deadline, not "duration from now": read_frames' finite pacing
        # already consumed the play time up to the SAME absolute instant, so a shot that reads its
        # frames and then wait_done()s pays the remainder (≈0), never the duration twice.  A
        # prepared-but-never-fired program keeps the bare prepare+wait semantics (full duration
        # from this call).
        if self._finite_play_deadline is None:
            self._finite_play_deadline = time.monotonic() + float(self._prepared.duration) * self.sleep_scale
        delay = max(0.0, self._finite_play_deadline - time.monotonic())
        if timeout is not None and delay > float(timeout):
            return self._record_wait(False)
        if delay > 0:
            self._sleep_interruptible(delay, stop)                 # cooperatively cancellable, like settle
        ok = not (stop is not None and stop.is_set())             # cancelled mid-play -> not done
        return self._record_wait(ok)

    def _record_wait(self, ok: bool) -> bool:
        """Land a wait_done outcome with the SAME protocol bookkeeping the bare service does
        (``state`` done/timeout + a history row), so a composed virtual wait reads identically in
        snapshots -- the real-time pacing is this backend's only divergence, never the record."""
        with self.service._lock:
            self.service.state = "done" if ok else "timeout"
            self.service.history.append({"action": "wait_done", "ok": ok})
        return ok

    def settle(self, seconds: float, *, stop=None) -> None:
        """Idle ``seconds`` between software-stepped fires -- DELEGATED to the composed service,
        which owns ``sleep_scale`` and therefore the one "settle fast-forwards with sleep_scale"
        rule (``SequencerService.settle``); ``stop`` stays cooperatively cancellable."""
        self.service.settle(seconds, stop=stop)

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
        # Safe state PARKS the output pins (DAC mid-code = signed 0): the held word of the last
        # fire is gone, so a free-running monitor now images the safe-state (all-zero) levels.
        self._last_fired = None
        self._scan_info = None
        self._scan_done = False
        self._scan_fire_time = 0.0

    def snapshot(self) -> dict[str, object]:
        # The composed service's snapshot carries the PROTOCOL state (state / cache_prepared /
        # prepared_program / the sync-to-device ``last_payload_json``); the base snapshot merges
        # AFTER it so the ``type`` key still has ONE producer per class (BaseDevice.snapshot),
        # and the virtual-specific fields land last (``channels`` shows the DAC coil bits folded
        # into da_x/da_y/da_z buses -- ONE display source).
        out = {**self.service.snapshot(), **super().snapshot()}
        out.update({
            "channels": self.display_channels(),
            "clock_hz": self.clock_hz,
            "sleep_scale": self.sleep_scale,
            "runs": sum(1 for row in self.history if row["action"] == "fire"),
            "firing": None if self._firing is None else self._firing.name,
        })
        return out

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
    reference_exposure: float = DEFAULT_EXPOSURE_S,
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
    # The grey-molasses cooling light (D1) and the F=1<->F=2 microwave source are WIRED
    # into the trap array: its cooled (PGC) temperature floor is set by how well those two
    # are tuned (laser detuning + two-photon RF resonance), exactly like the real bench where
    # a mis-set detuning or off-resonant RF warms the cloud (see _cooling_floor_K).
    return {
        "laser": {"type": "VirtualLaser"},
        "rf": {"type": "VirtualRF"},
        "trap_array": {"type": "VirtualTrapArray",
                       "params": {"laser": "$device:laser", "rf": "$device:rf"}},
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
    trig_set = set(normalize_trigger_channels(trigger_channels))
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

    trig_set = set(normalize_trigger_channels(trigger_channels))
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
    trig_set = set(normalize_trigger_channels(trigger_channels))
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
    trig_set = set(normalize_trigger_channels(trigger_channels))
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
    "GM_HOT_FACTOR",
    "MOT_COIL_BUSES",
    "RB87_D1_LINEWIDTH_HZ",
    "RB87_D1_WAVELENGTH_NM",
    "RB87_HYPERFINE_HZ",
    "VirtualCamera",
    "VirtualLaser",
    "VirtualMotCamera",
    "VirtualRF",
    "VirtualSequencer",
    "VirtualTrapArray",
    "cooling_durations_per_frame",
    "exposures_per_frame",
    "grey_molasses_cooling_factor",
    "trap_off_durations_per_frame",
    "virtual_config",
    "virtual_config_with_overrides",
    "write_virtual_run",
]
