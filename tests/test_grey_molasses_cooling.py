"""Contract: the virtual backend models grey-molasses (D1) cooling of Rb87.

A ``VirtualLaser`` (the D1 cooling light) and a ``VirtualRF`` (the F=1<->F=2 microwave
that closes the Raman/dark-state condition) are ordinary devices in the virtual config,
WIRED into the trap array.  The cloud's cooled (PGC) temperature floor is set by how well
those two are tuned:

* blue single-photon detuning near the optimum (~+6 Gamma) AND two-photon (RF) resonance
  (delta = 0, RF exactly on the 6.834 GHz hyperfine splitting) AND light on the D1 line
  => full cooling => the calibrated floor is preserved (factor 1.0, so nothing else in the
  physics suite moves);
* a red single-photon detuning, an off-resonant RF, or light off the D1 line => cooling
  fails => the cloud stays hot (the MOT/Doppler ceiling).

Everything is derived from the single-source constants in ``virtual`` -- the test never
re-types a magic number the model owns.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.virtual import (
    GM_DETUNING_OPTIMUM_GAMMA,
    GM_HOT_FACTOR,
    RB87_D1_LINEWIDTH_HZ,
    RB87_D1_WAVELENGTH_NM,
    RB87_HYPERFINE_HZ,
    VirtualLaser,
    VirtualRF,
    VirtualTrapArray,
    grey_molasses_cooling_factor,
)


def test_cooling_factor_is_unity_only_at_the_grey_molasses_optimum():
    """On the D1 line, blue-detuned at the optimum, RF exactly on two-photon resonance =>
    the degradation factor is exactly 1.0 (full cooling, calibrated floor preserved)."""
    factor = grey_molasses_cooling_factor(
        GM_DETUNING_OPTIMUM_GAMMA, two_photon_detuning_gamma=0.0, saturation=3.0, on_d1=True)
    assert factor == pytest.approx(1.0, abs=1e-9)


def test_red_single_photon_detuning_defeats_grey_molasses():
    """Grey molasses needs BLUE detuning; a red (negative) detuning gives no sub-Doppler
    cooling at all -> the hot ceiling."""
    factor = grey_molasses_cooling_factor(-2.0, 0.0, 3.0, on_d1=True)
    assert factor == pytest.approx(GM_HOT_FACTOR)


def test_off_d1_wavelength_defeats_grey_molasses():
    """Light not on the Rb87 D1 line cools nothing -> the hot ceiling, regardless of detuning."""
    factor = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, 0.0, 3.0, on_d1=False)
    assert factor == pytest.approx(GM_HOT_FACTOR)


def test_detuning_from_the_optimum_warms_monotonically_up_to_the_ceiling():
    """Moving the single-photon detuning away from the optimum (either side) warms the cloud,
    a smooth cooling-vs-detuning curve that saturates at the hot ceiling."""
    at_opt = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, 0.0, 3.0, True)
    near = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA - 1.5, 0.0, 3.0, True)
    far = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA - 4.0, 0.0, 3.0, True)
    assert at_opt < near < far
    assert far <= GM_HOT_FACTOR


def test_two_photon_rf_detuning_has_a_resolvable_dark_resonance():
    """A fine RF sweep resolves the narrow dark resonance: on resonance the factor is 1.0 and
    it climbs smoothly as |delta| grows -- so a user scanning RF frequency sees a dip to find."""
    on_res = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, 0.0, 3.0, True)
    just_off = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, 0.04, 3.0, True)
    further = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, 0.10, 3.0, True)
    assert on_res == pytest.approx(1.0, abs=1e-9)
    assert on_res < just_off < further


def test_blue_side_rf_detuning_heats_more_sharply_than_the_red_side():
    """The Fano lineshape of a grey molasses is asymmetric: a positive two-photon detuning heats
    more sharply than the same-magnitude negative one."""
    blue = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, +0.05, 3.0, True)
    red = grey_molasses_cooling_factor(GM_DETUNING_OPTIMUM_GAMMA, -0.05, 3.0, True)
    assert blue > red > 1.0


def test_virtual_laser_and_rf_report_their_physical_state():
    """The virtual laser knows whether it sits on the D1 line; the virtual RF reports its
    two-photon detuning in linewidths, derived from its frequency vs the hyperfine splitting."""
    laser = VirtualLaser()
    assert laser.wavelength_nm == pytest.approx(RB87_D1_WAVELENGTH_NM)
    assert laser.on_d1 is True
    laser.wavelength_nm = RB87_D1_WAVELENGTH_NM + 5.0     # 5 nm off -> not the D1 line
    assert laser.on_d1 is False

    rf = VirtualRF()
    assert rf.two_photon_detuning_gamma == pytest.approx(0.0)   # default = on the hyperfine line
    rf.frequency_hz = RB87_HYPERFINE_HZ + RB87_D1_LINEWIDTH_HZ
    assert rf.two_photon_detuning_gamma == pytest.approx(1.0)   # one linewidth above resonance


def test_trap_floor_follows_the_wired_laser_and_rf():
    """Wired into a trap array, the laser+rf drive its cooled floor: at the optimum the floor
    equals the calibrated ``cooled_temperature_K``; a red-detuned laser or off-resonant RF warms
    it toward the hot ceiling.  Without a laser/rf the trap keeps its plain calibrated floor."""
    base = 50e-6
    laser = VirtualLaser(detuning_gamma=GM_DETUNING_OPTIMUM_GAMMA)
    rf = VirtualRF(frequency_hz=RB87_HYPERFINE_HZ)
    trap = VirtualTrapArray(grid_shape=(4, 4), seed=7, cooled_temperature_K=base, laser=laser, rf=rf)

    assert trap._cooling_floor_K() == pytest.approx(base)          # optimum -> calibrated floor

    rf.frequency_hz = RB87_HYPERFINE_HZ + 2.0 * RB87_D1_LINEWIDTH_HZ  # far off two-photon resonance
    assert trap._cooling_floor_K() == pytest.approx(base * GM_HOT_FACTOR)

    rf.frequency_hz = RB87_HYPERFINE_HZ
    laser.detuning_gamma = -3.0                                    # red-detuned -> hot
    assert trap._cooling_floor_K() == pytest.approx(base * GM_HOT_FACTOR)

    unwired = VirtualTrapArray(grid_shape=(4, 4), seed=7, cooled_temperature_K=base)
    assert unwired._cooling_floor_K() == pytest.approx(base)       # no GM light -> plain floor


def test_connect_virtual_wires_laser_and_rf_into_the_trap_array():
    """``na.connect("virtual")`` builds a laser + rf and wires them into the trap array, and both
    expose WRITABLE runtime controls (edit like an API) that route through the device's own setter."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        laser, rf, trap = exp.devices.laser, exp.devices.rf, exp.devices.trap_array
        assert isinstance(laser, VirtualLaser) and isinstance(rf, VirtualRF)
        assert trap.laser is laser and trap.rf is rf          # wired, one object each

        laser_ctrls = {c.decl.key: c for c in laser.runtime_controls()}
        assert laser_ctrls["detuning_gamma"].writable and laser_ctrls["wavelength_nm"].writable
        laser_ctrls["detuning_gamma"].setter(laser, -3.0)     # write like the API
        assert laser.detuning_gamma == pytest.approx(-3.0)

        rf_ctrls = {c.decl.key: c for c in rf.runtime_controls()}
        assert rf_ctrls["frequency_hz"].writable
        # the derived two-photon detuning is a read-only read-back (no setter)
        assert not rf_ctrls["two_photon_detuning_gamma"].writable
    finally:
        exp.close()
