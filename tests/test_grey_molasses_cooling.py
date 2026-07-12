"""Contract: the virtual backend models grey-molasses (D1) cooling of Rb87, and a detuning scan finds
the optimum.

Two devices produce grey molasses: a ``VirtualLaser`` (a beam LOCKED blue on the Rb87 D1 line -- it has
NO tunable detuning of its own) and a ``VirtualRF`` (the microwave/EOM sideband that PRODUCES the
two-photon Raman detuning delta between the cooling and repump beams).  The ONLY detuning knob is the
RF's ``two_photon_detuning_gamma``:

* delta = 0 (RF on the 6.834682 GHz hyperfine line), on the D1 line => full cooling => the calibrated
  cooled floor is preserved (factor 1.0, so the physics / calibration suites do not move);
* off the D1 line (wrong wavelength) or delta != 0 => cooling fails => the cloud stays hot (a Fano
  feature about delta = 0).

The ``Grey molasses detuning`` measurement sweeps the RF two-photon detuning (a DeviceControlAxis) and
release-recaptures at a fixed trap-off, so the recapture rate PEAKS at the optimum delta -- exactly how
an operator finds the best grey-molasses detuning.
"""

from __future__ import annotations

from conftest import raw_device_set

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.virtual import (
    GM_HOT_FACTOR,
    RB87_D1_LINEWIDTH_HZ,
    RB87_D1_WAVELENGTH_NM,
    RB87_HYPERFINE_HZ,
    VirtualLaser,
    VirtualRF,
    VirtualTrapArray,
    grey_molasses_cooling_factor,
)


def test_cooling_factor_is_unity_only_on_the_two_photon_resonance():
    """On the D1 line with the RF exactly on the hyperfine line (delta = 0) the degradation factor is
    exactly 1.0 (full cooling, calibrated floor preserved)."""
    assert grey_molasses_cooling_factor(0.0, saturation=3.0, on_d1=True) == pytest.approx(1.0, abs=1e-9)


def test_off_d1_wavelength_defeats_grey_molasses():
    """Light not on the Rb87 D1 line cools nothing -> the hot ceiling, whatever the detuning."""
    assert grey_molasses_cooling_factor(0.0, 3.0, on_d1=False) == pytest.approx(GM_HOT_FACTOR)


def test_two_photon_detuning_has_a_resolvable_dark_resonance():
    """A fine RF sweep resolves the narrow dark resonance: on resonance the factor is 1.0 and it climbs
    as |delta| grows -- so scanning the detuning shows a dip (a survival peak) to find."""
    on_res = grey_molasses_cooling_factor(0.0, 3.0, True)
    just_off = grey_molasses_cooling_factor(0.04, 3.0, True)
    further = grey_molasses_cooling_factor(0.10, 3.0, True)
    assert on_res == pytest.approx(1.0, abs=1e-9)
    assert on_res < just_off < further


def test_blue_side_detuning_heats_more_sharply_than_the_red_side():
    """The grey-molasses Fano lineshape is asymmetric: a positive two-photon detuning heats more
    sharply than the same-magnitude negative one."""
    blue = grey_molasses_cooling_factor(+0.05, 3.0, True)
    red = grey_molasses_cooling_factor(-0.05, 3.0, True)
    assert blue > red > 1.0


def test_higher_saturation_broadens_the_dark_resonance():
    """More beam power broadens the two-photon dark resonance, so the same off-resonant detuning heats
    LESS at higher saturation (a wider window still cools)."""
    lo = grey_molasses_cooling_factor(0.08, saturation=1.0, on_d1=True)
    hi = grey_molasses_cooling_factor(0.08, saturation=8.0, on_d1=True)
    assert hi < lo


def test_virtual_laser_has_no_detuning_and_the_rf_owns_it():
    """The laser exposes only wavelength (on-D1) + saturation -- NO detuning control; the RF's
    ``two_photon_detuning_gamma`` is a read/write control whose setter moves the drive frequency."""
    laser = VirtualLaser()
    laser_keys = {c.decl.key for c in laser.runtime_controls()}
    assert "detuning_gamma" not in laser_keys and not hasattr(laser, "detuning_gamma")
    # the grey-molasses knobs (no detuning): wavelength / saturation floats + a beam-on switch + the
    # derived on-D1 read-back -- all auto-injected from the laser's DeviceProperty declarations.
    assert laser_keys == {"wavelength_nm", "saturation", "beam_on", "on_d1"}
    assert laser.on_d1 is True
    laser.wavelength_nm = RB87_D1_WAVELENGTH_NM + 5.0
    assert laser.on_d1 is False

    rf = VirtualRF()
    ctrl = {c.decl.key: c for c in rf.runtime_controls()}["two_photon_detuning_gamma"]
    assert ctrl.writable                                            # the RF OWNS the detuning knob
    assert rf.two_photon_detuning_gamma == pytest.approx(0.0)
    ctrl.setter(rf, 1.0)                                            # write delta = +1 linewidth
    assert rf.frequency_hz == pytest.approx(RB87_HYPERFINE_HZ + RB87_D1_LINEWIDTH_HZ)
    assert rf.two_photon_detuning_gamma == pytest.approx(1.0)       # round-trips


def test_trap_floor_follows_the_rf_detuning_and_laser_line():
    """Wired into a trap array, the laser+RF drive its cooled floor: at delta = 0 on D1 the floor equals
    the calibrated ``cooled_temperature_K``; an off-resonant RF or an off-D1 laser warms it to the hot
    ceiling.  Without a laser/RF the trap keeps its plain floor."""
    base = 50e-6
    laser = VirtualLaser()
    rf = VirtualRF(frequency_hz=RB87_HYPERFINE_HZ)
    trap = VirtualTrapArray(grid_shape=(4, 4), seed=7, cooled_temperature_K=base, laser=laser, rf=rf)

    assert trap._cooling_floor_K() == pytest.approx(base)          # delta = 0 -> calibrated floor
    rf.two_photon_detuning_gamma = 1.0                             # far off two-photon resonance
    assert trap._cooling_floor_K() == pytest.approx(base * GM_HOT_FACTOR)
    rf.two_photon_detuning_gamma = 0.0
    laser.wavelength_nm = RB87_D1_WAVELENGTH_NM + 5.0             # off the D1 line -> hot
    assert trap._cooling_floor_K() == pytest.approx(base * GM_HOT_FACTOR)

    unwired = VirtualTrapArray(grid_shape=(4, 4), seed=7, cooled_temperature_K=base)
    assert unwired._cooling_floor_K() == pytest.approx(base)


def test_connect_virtual_wires_laser_and_rf_into_the_trap_array():
    """``na.connect("virtual")`` builds a laser + rf and wires them into the trap array; the RF's
    two-photon detuning is a writable runtime control (the grey-molasses knob edited in the viewer)."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        laser, rf, trap = raw_device_set(exp).laser, raw_device_set(exp).rf, raw_device_set(exp).trap_array
        assert isinstance(laser, VirtualLaser) and isinstance(rf, VirtualRF)
        assert trap.laser is laser and trap.rf is rf
        rf_ctrls = {c.decl.key: c for c in rf.runtime_controls()}
        assert rf_ctrls["two_photon_detuning_gamma"].writable
    finally:
        exp.close()


def test_detuning_scan_recapture_rate_peaks_at_the_optimum():
    """The ``Grey molasses detuning`` measurement sweeps the RF two-photon detuning and release-
    recaptures at a fixed trap-off; the recapture rate (survival) PEAKS at delta = 0 -- so the argmax IS
    the optimum grey-molasses detuning, obtained purely from camera frames (virtual == real path)."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5)})
    try:
        exp.readout.sitemap(frames=4, display=False)
        exp.readout.thresholds(frames=30, display=False)
        spec = {s.key: s for s in exp.readout.measurement_specs()}["gm_detuning"]
        assert spec.x_key == "detuning" and spec.y_key == "recapture"    # distinct from Temperature
        assert any(p.key == "rf" for p in spec.params)                   # declares the rf device role
        meas = spec.build(rf="rf", detuning=(-0.4, 0.4, 9), t_off=25.0, shots=32)
        result = meas.run(live=False, display=False)
        x = np.asarray(result.x)
        y = np.asarray(result.data_y)[:, 0]
        assert x[int(np.nanargmax(y))] == pytest.approx(0.0, abs=1e-9)   # coldest = best cooling at delta=0
        assert y[len(y) // 2] > y[0] and y[len(y) // 2] > y[-1]          # a real peak, not monotonic
    finally:
        exp.close()
