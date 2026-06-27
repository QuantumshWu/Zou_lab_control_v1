"""#H3s-F10: the scan-TIER boundary is a HARD, mechanically-pinned contract.

The user (#6): "temperature 和 fidelity 都是 pulse scan 的特例".  That is TRUE at the model level --
every scan sweeps one or more NAMED pulse slots and reduces each point (MAINTAINER_NOTES §19) -- and
temperature / detection-time already share the ONE slot-scan spine (`_build_slot_scan`).  But there are
two TIERS by WHERE the per-point reduce lives, and the boundary between them is physics, not taste:

* DECOUPLED tier (`PulseScanNode`): per-point acquisition is ``camera.acquire(1)`` -- ONE camera
  trigger, ONE frame -- and y is a SINGLE-frame ``signal_expr`` read off another node.
* COUPLED tier (`ScannedMeasurement` + inline reducer): release-recapture survival needs EXACTLY two
  frames from ONE atom loading (a 2-trigger sequence, trap-off between the triggers), and Otsu fidelity
  needs the per-point frame SET.  Neither can be a single published frame.

So temperature / detection-time are IRREDUCIBLY coupled: you cannot route them through the decoupled
node.  This file pins that boundary so a future "let's finish the unification" refactor fails loudly.
The reducer/plan unit facts (2 triggers, reducer needs 2 frames, plan rejects n!=2) live in
``test_measurement_temperature.py``; here we pin the CROSS-TIER seam those facts imply.  ``fidelity.py``
(`characterize_readout`) is NOT a scan at all (no swept axis), so it is out of scope by construction.
"""

import inspect

import pytest

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.operations.temperature import build_release_recapture_pulse
from Zou_lab_control.neutral_atom.timing.sequence import sequence_for_frame_count


def _release_recapture_sequence():
    """The release-recapture pulse the temperature measurement fires, as a PulseSequence: TWO camera
    triggers (emCCD) with the trap-off period between them = ONE atom loading read twice."""
    state = build_release_recapture_pulse(
        channels=["trap", "probe", "emCCD"], exposure=2e-3, settle=2e-4, recapture=2e-4)
    return state.to_sequence(slots=state.reference_slots(),
                             time_step_ns=state.time_step_ns, expand_repeat=False)


def test_release_recapture_is_two_trigger_single_loading_not_acquire_one():
    """The coupled physics: the release-recapture pulse carries 2 camera triggers from ONE loading.
    The device boundary REFUSES to acquire it as the decoupled tier's one-frame-per-point: a 2-trigger
    sequence asked for 1 frame is a hard error.  Acquired as 2 frames it is returned UNCHANGED (the
    single loading, both reads sharing it) -- never repeated into two independent reloads."""
    seq = _release_recapture_sequence()
    assert na.count_trigger_pulses(seq, trigger_channels=["emCCD"]) == 2

    # The decoupled node's per-point contract is acquire(1); feeding it this pulse is impossible.
    with pytest.raises(ValueError):
        sequence_for_frame_count(seq, 1, trigger_channels=["emCCD"])

    # The coupled path acquires both triggers from the SAME sequence (one loading) -- unchanged.
    assert sequence_for_frame_count(seq, 2, trigger_channels=["emCCD"]) is seq


def test_one_trigger_imaging_acquired_as_two_is_independent_reloads():
    """The contrast that makes survival irreducible to the decoupled tier: a 1-trigger imaging pulse
    asked for 2 frames is REPEATED -> two independent loadings (a fresh reload between frames).  That
    is fine for averaging an image, but it is the WRONG physics for survival (no shared loading), which
    is exactly why release-recapture must stay a 2-trigger single sequence, not 2x acquire(1)."""
    img = na.imaging_sequence(exposure=2e-3, load=True)
    assert na.count_trigger_pulses(img) == 1
    repeated = sequence_for_frame_count(img, 2)
    assert repeated is not img                              # had to be expanded (reloaded), not shared
    assert na.count_trigger_pulses(repeated) == 2


def test_decoupled_pulse_scan_per_point_acquisition_is_acquire_one():
    """The decoupled tier's per-point acquisition is acquire(1): `PulseScanNode` images ONCE per scan
    point, then y is a single-frame signal_expr off another node.  Pin the one-frame contract -- the
    structural reason survival / fidelity (which need >=2 frames from ONE loading, or the per-point
    frame SET) cannot live in this tier.  A refactor that made pulse-scan multi-frame -- the exact
    encroachment on the coupled physics -- changes this line and must reckon with the boundary here."""
    from Zou_lab_control.neutral_atom.operations import logic as logic_mod

    node_src = inspect.getsource(logic_mod.PulseScanNode)
    assert "acquire(1" in node_src, (
        "decoupled PulseScanNode must acquire exactly ONE frame per scan point; a multi-frame "
        "acquisition would encroach on the coupled survival/fidelity tier (see MAINTAINER_NOTES §19).")


def test_temperature_and_detection_share_the_one_slot_scan_spine():
    """The POSITIVE half of the user's claim: temperature + detection-time ARE pulse-scans -- both
    coupled builders go through the ONE `_build_slot_scan` spine (a named-slot ScanAxis), differing only
    in (slot, plan, reducer).  So 'is temperature a pulse-scan?' is yes; it just lives in the coupled
    tier because its reduce is frame-structural (the boundary the tests above pin)."""
    from Zou_lab_control.neutral_atom.subsystems import readout as readout_mod

    spine = inspect.getsource(readout_mod.ReadoutSubsystem._build_slot_scan)
    assert "ScanAxis" in spine and "ScannedMeasurement" in spine
    # both temperature and detection-time builders route through the spine (not their own scan loop)
    for builder in ("build_temperature_scan", "build_detection_scan"):
        src = inspect.getsource(getattr(readout_mod.ReadoutSubsystem, builder))
        assert "_build_slot_scan" in src, f"{builder} must build on the shared slot-scan spine"
