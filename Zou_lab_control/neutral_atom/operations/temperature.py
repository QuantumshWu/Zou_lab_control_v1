"""Release-recapture thermometry: pulse, survival reducer, and the fit.

Release-recapture measures the temperature of trapped atoms by switching the trap
OFF for a variable time ``t_off``, letting the atoms fly ballistically, then
switching it back ON and asking how many are still close enough to be recaptured.
Hotter atoms have a wider thermal velocity spread, so they escape the capture
region faster; the survival-vs-``t_off`` curve therefore encodes the temperature.

Three pieces live here and compose with the generic engine in ``measurement.py``:

  * :func:`build_release_recapture_pulse` -- the 6-period :class:`PulseTableState`
    with TWO camera triggers and a trap-off period whose duration is the bound
    scan slot (so the SAME slot mechanism that scans a readout duration scans
    ``t_off``).
  * :class:`ReleaseRecapturePlan` -- the two-frame ``ShotPlan`` (one acquire ->
    ``[image1, image2]``); deliberately NOT a repeated single-frame sequence,
    because the two images must share one atom loading with a trap-off in
    between (see the docstring).
  * :class:`SurvivalReducer` -- per-point survival from ``calibration.detect`` on
    the two frames; mean (scalar) or one column per site (``per_site=True``).

:func:`fit_temperature` is a PURE post-processing function on the finished
``ScanResult`` (it never runs inside the acquire loop), fitting the standard
ballistic release-recapture model to recover ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.calibration import TrapCalibration

if TYPE_CHECKING:  # pragma: no cover
    from ..timing import PulseTableState


# Boltzmann constant and atomic mass unit (SI); defaults below are Rb87.
K_B = 1.380649e-23
AMU = 1.66053906660e-27
RB87_MASS = 86.909180527 * AMU


# --------------------------------------------------------------------------- pulse


def build_release_recapture_pulse(
    *,
    channels=("trap", "probe", "emCCD"),
    trap_channel: str = "trap",
    probe_channel: str = "probe",
    trigger_channel: str = "emCCD",
    exposure: float = 20e-3,
    settle: float = 1e-3,
    recapture: float = 1e-3,
    trap_off_nominal: float = 20e-6,
    time_step_ns: float = 20.0,
    name: str = "release_recapture",
) -> "PulseTableState":
    """Build the release-recapture :class:`PulseTableState` (t_off = scan slot 0).

    The period layout (durations in seconds at the API; stored as ns on the
    table) is::

        # | period           | trap | probe | emCCD | duration
        0 | image1_expose    |  1   |   1   |   1   | exposure   (1st trigger rise)
        1 | image1_settle    |  1   |   0   |   0   | settle
        2 | trap_off         |  0   |   0   |   0   | t_off  <- bound scan slot s0
        3 | trap_recapture   |  1   |   0   |   0   | recapture
        4 | image2_expose    |  1   |   1   |   1   | exposure   (2nd trigger rise)
        5 | image2_settle    |  1   |   0   |   0   | settle

    Period 2's duration is bound to a duration scan slot via ``bind_field`` --
    the SAME mechanism a scanned readout duration uses -- so ``pulse.set_time``
    and ``frame_sequence(time_ns=...)`` drive ``t_off`` per point, and a hardware
    scan can stream a whole ``t_off`` table.

    This must be a SINGLE sequence with two triggers, NOT a repeated single-frame
    sequence: ``finite_frame_sequence`` reloads atoms between frames, so the two
    images would be different loadings with no trap-off between them.  Release-
    recapture survival requires the SAME atoms imaged before and after one
    trap-off, which only the explicit trap-off period provides.
    """

    from ..timing import PulseTableState
    from ..timing.pulse_table import PulsePeriod

    channels = list(channels)
    for required in (trap_channel, probe_channel, trigger_channel):
        if required not in channels:
            raise ValueError(f"channel {required!r} must be in channels {channels}.")
    ti, pi, ei = (channels.index(trap_channel), channels.index(probe_channel), channels.index(trigger_channel))

    def states(trap: int, probe: int, trig: int) -> tuple[int, ...]:
        row = [0] * len(channels)
        row[ti], row[pi], row[ei] = trap, probe, trig
        return tuple(row)

    def secs_to_ns(value: float) -> float:
        return float(value) * 1e9

    periods = [
        PulsePeriod(secs_to_ns(exposure), states(1, 1, 1), unit="ns", name="image1_expose"),
        PulsePeriod(secs_to_ns(settle), states(1, 0, 0), unit="ns", name="image1_settle"),
        PulsePeriod(secs_to_ns(trap_off_nominal), states(0, 0, 0), unit="ns", name="trap_off"),
        PulsePeriod(secs_to_ns(recapture), states(1, 0, 0), unit="ns", name="trap_recapture"),
        PulsePeriod(secs_to_ns(exposure), states(1, 1, 1), unit="ns", name="image2_expose"),
        PulsePeriod(secs_to_ns(settle), states(1, 0, 0), unit="ns", name="image2_settle"),
    ]
    state = PulseTableState(
        channels=channels,
        periods=periods,
        time_step_ns=time_step_ns,
        repeat_forever=False,
        name=name,
    )
    # Bind the trap-off period (index 2) duration to scan slot s0 -- the t_off axis.
    state.bind_field("duration", "2", label="t_off", unit="ns")
    return state


# ---------------------------------------------------------------------- shot plan


@dataclass(frozen=True)
class ReleaseRecapturePlan:
    """Two-frame plan: one acquire returns ``[image1, image2]`` (no reload between).

    ``n_frames`` is fixed at 2 (the two triggers the release-recapture pulse
    emits).  ``sequence_for`` sets ``t_off`` via the time slot and returns the
    two-trigger sequence -- the camera then captures both images from one atom
    loading with the trap-off period in between, which is what makes the two
    frames comparable for survival.
    """

    n_frames: int = 2

    def __post_init__(self) -> None:
        if int(self.n_frames) != 2:
            raise ValueError("release-recapture acquires exactly 2 frames (image1, image2).")

    def sequence_for(self, pulse, axis, value: float):
        # t_off enters through the pulse's primary time slot, exactly like a
        # scanned readout duration; the sequence keeps its two camera triggers.
        return pulse.frame_sequence(2, time_ns=float(value) * float(axis.scale_to_ns))


# ------------------------------------------------------------------------ reducer


@dataclass
class SurvivalReducer:
    """Per-point survival from two frames via ``calibration.detect``.

    ``occ1``/``occ2`` are the bright/dark classifications of image1/image2;
    an atom "survives" if it is occupied in BOTH.  The scalar form returns the
    survival FRACTION among initially-occupied sites
    (``(occ1 & occ2).sum() / max(occ1.sum(), 1)``); ``per_site=True`` returns the
    per-site survival vector (1.0 survived, 0.0 lost, ``nan`` where the site was
    not occupied in frame 1 so it carries no survival information).  Uses ONLY
    ``calibration.detect`` -- never any simulation ground truth.
    """

    per_site: bool = False
    labels: tuple[str, str, str] = ("Trap-off time (s)", "Survival", "Survival")
    _n_sites: int | None = field(default=None, repr=False)

    @property
    def n_series(self) -> int:
        if self.per_site:
            if self._n_sites is None:
                raise RuntimeError("SurvivalReducer.bind_calibration must be called for per-site output.")
            return int(self._n_sites)
        return 1

    def bind_calibration(self, calibration: TrapCalibration) -> "SurvivalReducer":
        """Record the site count so a per-site reducer can size its output array."""

        self._n_sites = int(calibration.n_sites)
        return self

    def reduce(self, frames, calibration: TrapCalibration):
        if len(frames) != 2:
            raise ValueError(f"SurvivalReducer needs exactly 2 frames, got {len(frames)}.")
        occ1 = np.asarray(calibration.detect(frames[0]).occupied, dtype=bool)
        occ2 = np.asarray(calibration.detect(frames[1]).occupied, dtype=bool)
        survived = occ1 & occ2
        if self.per_site:
            out = np.full(occ1.shape, np.nan, dtype=float)
            out[occ1] = survived[occ1].astype(float)
            return out
        return float(survived.sum()) / float(max(int(occ1.sum()), 1))


# ---------------------------------------------------------------------------- fit


@dataclass
class TemperatureFit:
    """Result of a release-recapture temperature fit."""

    temperature_K: float
    temperature_uK: float
    capture_radius: float
    baseline: float
    t_off: np.ndarray
    survival: np.ndarray
    model: np.ndarray
    success: bool

    def summary(self) -> dict[str, float | bool]:
        return {
            "temperature_uK": float(self.temperature_uK),
            "capture_radius_m": float(self.capture_radius),
            "baseline": float(self.baseline),
            "success": bool(self.success),
        }


def release_recapture_survival(t_off, temperature_K, *, capture_radius, mass=RB87_MASS, baseline=1.0):
    """Ballistic release-recapture survival model (3-D isotropic, point source).

    During ``t_off`` an atom flies freely; it is recaptured only if its ballistic
    displacement stays inside the capture region of radius ``r_c =
    capture_radius`` (metres).  Per axis the atom is recaptured if its speed obeys
    ``|v| < r_c / t``, so for a Maxwell-Boltzmann velocity distribution with 1-D
    spread ``sigma_v = sqrt(k_B T / m)`` the per-axis recapture probability is the
    Gaussian fraction within that window -- the error function of
    ``arg = r_c / (sqrt(2) * sigma_v * t)`` -- and the isotropic 3-D survival is
    ``baseline * P_axis(t)**3`` (initial position spread neglected -- the standard
    short-time, tight-trap approximation).  This is the form used for tweezer/
    dipole-trap release-recapture thermometry; see e.g. Tuchendler et al.,
    PRA 78, 033425 (2008).

    Note the degeneracy: the curve fixes only the ratio ``r_c / sqrt(T)``, so the
    capture radius MUST be supplied (from the trap geometry) to pin down the
    temperature -- which is why ``fit_temperature`` takes ``capture_radius`` as a
    known input and fits T alone.

    The per-axis Gaussian fraction is written via the shared ``normal_cdf`` (the
    single source for the normal CDF: ``2*Phi(arg) - 1`` with a sqrt(2)-scaled
    standard normal) so this physics model REUSES that kernel rather than pulling
    in its own copy of the error function.
    """

    from Zou_lab_control._readout_math import normal_cdf

    t = np.asarray(t_off, dtype=float)
    sigma_v = np.sqrt(K_B * float(temperature_K) / float(mass))
    # At t->0 the atom cannot move, so survival -> baseline (the window covers the
    # whole velocity distribution -> P_axis -> 1).
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = float(capture_radius) / (np.sqrt(2.0) * sigma_v * t)
    # P_axis is the Gaussian fraction within +/- r_c/t, i.e. 2*Phi(arg) - 1 for a
    # standard normal whose sqrt(2)-scaled argument is exactly ``arg``
    # (normal_cdf(arg, 0, 1/sqrt2) = 0.5 (1 + the error function of arg)).
    p_axis = np.where(t > 0, 2.0 * np.asarray(normal_cdf(arg, 0.0, 1.0 / np.sqrt(2.0))) - 1.0, 1.0)
    return float(baseline) * p_axis**3


def fit_temperature(
    t_off,
    survival,
    *,
    capture_radius: float,
    mass: float = RB87_MASS,
    fit_baseline: bool = True,
    T0: float = 50e-6,
) -> TemperatureFit:
    """Fit ballistic release-recapture survival(t_off) -> temperature.

    ``t_off`` is in seconds, ``survival`` the measured fraction (scalar reducer)
    or a per-site survival already reduced to one value per point.  Because the
    survival curve only constrains ``r_c / sqrt(T)`` (see
    :func:`release_recapture_survival`), the capture radius ``capture_radius``
    (metres, from the trap geometry: roughly the trap waist for a tweezer) MUST be
    given; only the temperature ``T`` -- and optionally a ``baseline`` that
    absorbs imaging/loss inefficiency at ``t->0`` -- is fit.

    A pure post-processing step on a finished ``ScanResult``: it never runs inside
    the acquire loop, so the live engine stays free of physics models.
    """

    from scipy.optimize import curve_fit

    if not np.isfinite(capture_radius) or float(capture_radius) <= 0:
        raise ValueError("capture_radius must be a positive finite length in metres.")
    t = np.asarray(t_off, dtype=float).reshape(-1)
    y = np.asarray(survival, dtype=float).reshape(-1)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 3:
        raise ValueError("fit_temperature needs at least 3 finite (t_off, survival) points.")

    baseline_guess = float(np.nanmax(y)) if y.size else 1.0
    rc = float(capture_radius)

    def model_fixed_baseline(tt, T):
        return release_recapture_survival(tt, T, capture_radius=rc, mass=mass, baseline=baseline_guess)

    def model_free_baseline(tt, T, base):
        return release_recapture_survival(tt, T, capture_radius=rc, mass=mass, baseline=base)

    success = True
    try:
        if fit_baseline:
            popt, _ = curve_fit(
                model_free_baseline, t, y, p0=[float(T0), baseline_guess],
                bounds=([0.0, 0.0], [np.inf, 1.0]), maxfev=20000,
            )
            T_fit, base_fit = float(popt[0]), float(popt[1])
        else:
            popt, _ = curve_fit(
                model_fixed_baseline, t, y, p0=[float(T0)],
                bounds=([0.0], [np.inf]), maxfev=20000,
            )
            T_fit, base_fit = float(popt[0]), baseline_guess
    except Exception:
        success = False
        T_fit, base_fit = float(T0), baseline_guess

    model_curve = release_recapture_survival(t, T_fit, capture_radius=rc, mass=mass, baseline=base_fit)
    return TemperatureFit(
        temperature_K=float(T_fit),
        temperature_uK=float(T_fit) * 1e6,
        capture_radius=rc,
        baseline=float(base_fit),
        t_off=t,
        survival=y,
        model=np.asarray(model_curve, dtype=float),
        success=success,
    )


__all__ = [
    "K_B",
    "RB87_MASS",
    "ReleaseRecapturePlan",
    "SurvivalReducer",
    "TemperatureFit",
    "build_release_recapture_pulse",
    "fit_temperature",
    "release_recapture_survival",
]
