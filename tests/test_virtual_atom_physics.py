"""Contract: the virtual backend is a PULSE-DRIVEN atom model, not a fixed image.

The fake data source must reproduce the BASIC atom-physics effects an experiment
shows -- a fresh ~50% stochastic loading every shot, ballistic release-recapture
loss set by the (cooled) temperature, and independent loadings that build a bimodal
readout histogram -- so loading-rate / temperature / readout measurements are
meaningful in virtual.  These pin the data-source physics directly (the analysis
layer reading only frames is guarded by test_virtual_equals_real_contract.py).
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
    VirtualCamera, VirtualSequencer, VirtualTrapArray,
)
from Zou_lab_control.neutral_atom.operations.measurement import triggered_frames
from Zou_lab_control.neutral_atom.timing import imaging_sequence

from conftest import fire_imaging_pulse   # the live "On Pulse" the trigger-driven camera needs


def _rig(**trap):
    """A trap/camera/sequencer rig with the streamer already FIRING the live imaging pulse,
    so the trigger-driven camera streams -- the exact state the live monitor runs in (the
    user has hit "On Pulse").  ``set_safe_state()`` would stop it (the camera then sees no
    trigger and freezes)."""
    trap_array = VirtualTrapArray(grid_shape=(4, 5), seed=11, **trap)
    seqr = VirtualSequencer()
    cam = VirtualCamera(trap_array, sequencer=seqr)   # the trigger cable, wired at construction
    fire_imaging_pulse(seqr, exposure=cam.exposure, cooling=trap_array.mot_load_s)
    return trap_array, cam, seqr


def test_each_shot_is_a_fresh_random_loading():
    """The bug this fixes: the live monitor must show a NEW ~loading_probability
    random loading every shot, not one fixed loading slowly losing atoms."""
    trap, cam, seqr = _rig(loading_probability=0.5)
    fractions, patterns = [], set()
    for _ in range(40):
        cam.acquire(1)                               # the live-monitor path (the wire senses the firing)
        fractions.append(float(trap.occupancy.mean()))
        patterns.add(trap.occupancy.tobytes())
    assert 0.4 <= np.mean(fractions) <= 0.6      # ~50% loading on average
    assert len(patterns) >= 30                   # each shot a DIFFERENT loading (not frozen)


def test_set_occupancy_is_a_one_shot_pin():
    """set_occupancy images THAT loading on the next shot, then resumes reloading."""
    trap, cam, seqr = _rig()
    trap.set_occupancy([0, 1, 4, 9])
    cam.acquire(1)
    assert list(np.flatnonzero(trap.occupancy)) == [0, 1, 4, 9]   # pinned loading imaged
    cam.acquire(1)
    # the shot after the pin is a fresh stochastic loading again (extremely unlikely
    # to reproduce the exact 4-site pattern by chance)
    assert list(np.flatnonzero(trap.occupancy)) != [0, 1, 4, 9]


def test_threshold_frames_are_independent_loadings():
    """A load-each-frame imaging sequence (the threshold-calibration pass) reloads a
    fresh array per frame, so per-site counts are BIMODAL (bright when loaded, dark
    when empty) -- the spread an otsu threshold needs."""
    trap, cam, seqr = _rig(loading_probability=0.5)
    seq = imaging_sequence(exposure=20e-3, load=True, name="readout")
    frames = triggered_frames(cam, seqr, seq, 40)
    # central pixel of site 0 over the 40 independent shots: a clear bright/dark split
    cy, cx = (int(round(c)) for c in trap._site_centers()[0][::-1])
    centre = np.array([float(f[cy, cx]) for f in frames])
    bright = centre > (centre.min() + centre.max()) / 2.0
    assert 0.2 <= bright.mean() <= 0.8           # both populations present (not all one)
    assert centre.max() - centre.min() > 50.0    # a real bimodal gap, not just noise


def test_repeated_single_trigger_frames_all_use_the_window_exposure():
    """Every frame of a repeated single-trigger acquisition is imaged through the SAME one trigger
    window, so ALL frames share that window's (short) exposure -- NOT a mix where frame 0 uses the
    2 ms window and the repeats fall back to the camera-default (20 ms) exposure.  A mixed-brightness
    set teaches a per-site threshold ABOVE the true short-exposure readout brightness, so a freshly
    loaded atom reads as EMPTY and release-recapture survival / readout fidelity collapse to 0 -- the
    regression this guards (the threshold floated 209 -> 286 above the ~210 atom signal at 2 ms)."""
    trap, cam, seqr = _rig(loading_probability=1.0)            # every site loaded -> brightness == exposure
    seq = imaging_sequence(exposure=2e-3, load=True, name="readout")   # a SHORT 2 ms readout window
    frames = triggered_frames(cam, seqr, seq, 8)
    over_bg = np.array([float(np.asarray(f).max()) - 200.0 for f in frames])   # peak atom signal per frame
    assert over_bg.min() > 0                                   # every frame imaged a real atom
    # all frames at the SAME short-exposure brightness; the bug made the repeats ~6x brighter
    # (camera-default 20 ms) than the 2 ms window-0 frame, blowing this ratio far past 2.
    assert over_bg.max() <= 2.0 * over_bg.min()


def test_release_recapture_survival_decays_with_trap_off_temperature():
    """Survival after a trap-off gap falls from ~1 (no time to escape) toward 0 as the
    gap grows, at a rate set by the COOLED temperature -- the temperature-scan signal."""
    trap, cam, seqr = _rig(loading_probability=1.0, cooled_temperature_K=50e-6, capture_radius_m=6e-6)

    def survival(t_off: float, shots: int = 60) -> float:
        kept = total = 0
        for _ in range(shots):
            trap.reload()
            trap.apply_recapture_loss(t_off)
            kept += int(trap.occupancy.sum())
            total += trap.occupancy.size
        return kept / total

    near = survival(0.0)
    far = survival(300e-6)
    assert near >= 0.9                            # atoms cannot move in zero time
    assert far <= 0.3                             # mostly lost after a long release
    assert near - far >= 0.5                      # a big, monotone-ish drop


def test_hotter_atoms_survive_a_release_less():
    """A hotter cloud (faster atoms) is recaptured less for the same trap-off."""
    cold = VirtualTrapArray(grid_shape=(6, 6), seed=3, loading_probability=1.0, cooled_temperature_K=20e-6)
    hot = VirtualTrapArray(grid_shape=(6, 6), seed=3, loading_probability=1.0, cooled_temperature_K=200e-6)
    t_off = 80e-6

    def kept(trap):
        k = n = 0
        for _ in range(50):
            trap.reload(); trap.apply_recapture_loss(t_off)
            k += int(trap.occupancy.sum()); n += trap.occupancy.size
        return k / n

    assert kept(cold) > kept(hot) + 0.1


def test_cool_recools_existing_atoms_toward_the_floor_without_reloading():
    """A PGC cooling phase on ALREADY-loaded atoms relaxes their temperature toward the
    cooled floor WITHOUT replacing them -- so 'heat then re-cool' returns to the floor on
    the SAME atom loading (what a release after re-cool then sees)."""
    trap = VirtualTrapArray(grid_shape=(4, 4), seed=4, cooled_temperature_K=30e-6, probe_heating_K_per_s=10.0)
    trap.reload()
    occ_before = trap.occupancy.copy()
    trap.heat(0.02)                                  # a 20 ms un-cooled probe -> +0.2 K
    assert trap.temperature_K.mean() > 1e-3          # genuinely hot
    trap.cool(5e-3)                                  # many tau of PGC -> back to the floor
    assert abs(float(trap.temperature_K.mean()) - 30e-6) < 5e-6
    assert np.array_equal(trap.occupancy, occ_before)   # re-cool, NOT a fresh reload


def test_vacuum_trap_loss_decays_exponentially_with_dark_hold():
    """A dark trap-ON hold loses atoms to background-gas collisions at exp(-t/trap_lifetime_s),
    independent of light/temperature -- the trap-lifetime signal a hold-time scan recovers."""
    lifetime = 10.0
    for hold, expect in [(1.0, np.exp(-1.0 / lifetime)), (10.0, np.exp(-1.0))]:
        kept = total = 0
        for _ in range(25):
            trap = VirtualTrapArray(grid_shape=(8, 8), loading_probability=1.0, seed=10, trap_lifetime_s=lifetime)
            trap.reload(); total += int(trap.occupancy.sum())
            trap.apply_trap_loss(hold)
            kept += int(trap.occupancy.sum())
        assert abs(kept / total - expect) < 0.05


def test_probe_heating_increases_release_recapture_loss():
    """Probe (imaging) recoil heating raises the cloud temperature, so the SAME trap-off
    releases MORE atoms -- the bias an un-cooled readout puts on a temperature scan (and the
    proof the heating mechanism is live, not a no-op).  Default heating is 0 (molasses)."""
    t_off = 60e-6

    def survival(heating):
        kept = total = 0
        for _ in range(40):
            trap = VirtualTrapArray(grid_shape=(8, 8), loading_probability=1.0, seed=11,
                                    cooled_temperature_K=30e-6, probe_heating_K_per_s=heating)
            trap.reload(); trap.heat(20e-3)          # a 20 ms probe at this rate
            total += int(trap.occupancy.sum()); trap.apply_recapture_loss(t_off)
            kept += int(trap.occupancy.sum())
        return kept / total

    assert survival(0.0) > survival(2.0) + 0.1       # heating -> hotter -> recaptured less


def test_acquire_applies_vacuum_loss_over_a_dark_hold_between_frames():
    """End-to-end through ``acquire``: a two-trigger sequence with a long DARK trap-ON hold
    between the images loses atoms to vacuum decay, while a short hold does not -- the
    trap-lifetime physics emerges from the fired pulse (data-source side)."""
    from Zou_lab_control.neutral_atom.timing import PulseSequence

    def hold_seq(hold: float, expo: float = 20e-3, settle: float = 1e-3) -> PulseSequence:
        total = expo + settle + hold + expo
        s = PulseSequence(name="trap_hold").pulse("trap", 0.0, total, name="trap_on")
        s = s.pulse("probe", 0.0, expo, name="probe1").pulse("emCCD", 0.0, 20e-6, name="trig1")
        t2 = expo + settle + hold
        return s.pulse("probe", t2, expo, name="probe2").pulse("emCCD", t2, 20e-6, name="trig2")

    def survival(hold: float) -> float:
        trap = VirtualTrapArray(grid_shape=(10, 10), loading_probability=1.0, seed=12, trap_lifetime_s=8.0)
        seqr = VirtualSequencer()
        cam = VirtualCamera(trap, sequencer=seqr)
        kept = total = 0
        for _ in range(8):
            triggered_frames(cam, seqr, hold_seq(hold), 2)  # frame0 loads, dark hold, frame1
            # the survivors after the hold ARE the current occupancy (data-source physics test)
            total += trap.n_sites
            kept += int(trap.occupancy.sum())
        return kept / total

    assert survival(0.05) > 0.9                       # a 50 ms hold barely loses anything
    assert survival(8.0) < 0.6                        # a one-lifetime dark hold loses ~1/e
