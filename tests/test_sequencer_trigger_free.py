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
    seq = na.RuntimeSequencer(channels=["ch00", "ch11"], clock_hz=50e6)
    assert not hasattr(seq, "trigger_channels")


def test_runtime_program_has_no_trigger_count_and_is_version_3():
    from Zou_lab_control.neutral_atom.devices.sequencer import _RUNTIME_PROGRAM_VERSION

    # version bumped to 3 when trigger_count was removed (old payloads fail-fast, not mis-decode).
    assert _RUNTIME_PROGRAM_VERSION == 3

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
