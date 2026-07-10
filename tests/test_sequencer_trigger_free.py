"""Guard: the FPGA sequencer is a PURE pulse streamer -- it never owns a camera trigger.

A camera is externally triggered: which sequencer line gates a frame is the CAMERA device's
property (``capture_trigger_channels``), set when the camera is built and exposed upward.  The
sequencer/streamer compiles + streams whatever sequence it is handed and knows nothing about
which channel is a camera trigger.  These checks mechanically pin that decoupling so a refactor
cannot quietly re-introduce a ``trigger_channels`` knob or a ``trigger_count`` field on the
sequencer side (the camera_trigger seam + the camera device own that notion instead).
"""

from __future__ import annotations

import Zou_lab_control.neutral_atom as na


def test_runtime_sequencer_has_no_trigger_channels_attr():
    seq = na.VirtualSequencer(channels=["ch00", "ch11"], clock_hz=50e6)
    assert not hasattr(seq, "trigger_channels")


def test_runtime_program_has_no_trigger_count_and_carries_current_catalog_version():
    from Zou_lab_control.neutral_atom.devices.sequencer import _RUNTIME_PROGRAM_VERSION

    # Version 4 adds the PortCatalog fingerprint after version 3 removed trigger_count;
    # both old shapes fail fast instead of being guessed into the current wire contract.
    assert _RUNTIME_PROGRAM_VERSION == 4

    seq = na.imaging_sequence(exposure=1e-3, load=True)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=50e6)
    assert not hasattr(program, "trigger_count")
    assert "trigger_count" not in program.to_dict()


def test_camera_owns_the_capture_trigger_channel_not_the_sequencer():
    exp = na.connect("virtual", open_devices=True)
    try:
        # The camera exposes which line gates it; the sequencer does not.
        assert tuple(exp.devices.camera.capture_trigger_channels) == ("emCCD",)
        assert not hasattr(exp.devices.sequencer, "trigger_channels")
    finally:
        exp.close()


def test_firing_is_a_sequencer_contract_with_real_default_none():
    """``firing`` (the live "is the streamer continuously playing a program" handle the camera read
    is gated on) is declared on the SequencerDevice base, NOT a duck-typed attribute only on the
    virtual backend.  A real/remote streamer keeps no host-side firing flag -- the camera learns it
    from the presence/absence of hardware trigger edges -- so the contract default is None.  This
    pins that the abstraction owns the notion (callers read ``seqr.firing`` directly, no getattr)."""
    from Zou_lab_control.neutral_atom.devices.base import SequencerDevice

    assert "firing" in vars(SequencerDevice)               # declared on the abstract contract
    seq = na.VirtualSequencer(channels=["ch00", "ch11"], clock_hz=50e6)
    assert isinstance(seq, SequencerDevice)
    assert seq.firing is None                              # real backend default = correct semantics


def test_virtual_sequencer_overrides_firing_with_its_played_program():
    """The in-process VirtualSequencer overrides ``firing`` to expose the repeat_forever program it
    is simulating: None when idle, the program once fired forever, None again after safe-state."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer

    seqr = VirtualSequencer()
    assert seqr.firing is None                             # idle
    seq = na.imaging_sequence(exposure=1e-3, load=True, name="live").forever()
    seqr.prepare(seq)
    seqr.fire(seq)
    assert seqr.firing is not None                         # streaming a continuous program
    seqr.set_safe_state()
    assert seqr.firing is None                             # Stop Pulse -> cleared


def test_count_trigger_pulses_lives_in_the_camera_seam():
    # The trigger-counting util moved out of timing into the camera layer; na re-exports it.
    from Zou_lab_control.neutral_atom.devices.camera_trigger import (
        DEFAULT_CAMERA_TRIGGER_CHANNELS,
        count_trigger_pulses,
    )

    assert na.count_trigger_pulses is count_trigger_pulses
    assert na.DEFAULT_CAMERA_TRIGGER_CHANNELS == DEFAULT_CAMERA_TRIGGER_CHANNELS
    seq = na.imaging_sequence(exposure=1e-3, load=True)
    assert count_trigger_pulses(seq) == 1
    assert count_trigger_pulses(seq.repeated(3)) == 3

    # The 1->N expansion helpers were deleted with the new design (the pulse carries its own
    # triggers; no sequence is rewritten to add capture edges).
    assert not hasattr(na, "sequence_for_frame_count")
    assert not hasattr(na, "finite_frame_sequence")
