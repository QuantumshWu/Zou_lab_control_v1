"""Contract: on_pulse works on the virtual backend with a GUI PulseTableState.

The user's bug: ``exp.timing.bind_pulse(<a pulse .json / PulseTableState>).on_pulse(...)`` crashed
because ``VirtualSequencer.prepare`` called ``sequence.validate(clock_hz=, channels=)`` /
``.raise_if_failed()`` -- the PulseSequence contract -- on a PulseTableState (which has neither),
and returned None (so ``on_pulse`` returned None instead of a program).  ``VirtualSequencer.prepare``
now compiles a table to a PulseSequence and returns a RuntimeSequenceProgram, exactly like the real
VirtualSequencer -- so the SAME on_pulse code runs on both backends (only connect() differs).
"""

from __future__ import annotations

from conftest import raw_device_set

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.testing import bind_test_pulse


def test_on_pulse_finite_and_continuous_on_virtual():
    """The exact tutorial flow: bind a PulseTableState, fire it finite (wait) and continuous,
    then stop -- on the virtual backend, returning real programs with a sequence_name."""
    exp = na.connect("virtual")
    try:
        # build_release_recapture_pulse returns a PulseTableState (the GUI's pulse type).
        state = na.build_release_recapture_pulse(
            port_catalog=raw_device_set(exp).sequencer.port_catalog)
        pulse = bind_test_pulse(raw_device_set(exp).sequencer, state)

        # finite shot, wait for done -> a real program (NOT None) with a name.
        prog = pulse.on_pulse(wait=True, timeout=10.0, repeat_forever=False)
        assert prog is not None and prog.sequence_name == state.name

        # continuous (On Pulse) -> the streamer keeps firing (the live camera would stream).
        prog2 = pulse.on_pulse(wait=False, repeat_forever=True)
        assert prog2 is not None and prog2.sequence_name == state.name
        assert raw_device_set(exp).sequencer.firing is not None

        # Stop Pulse -> firing cleared.
        pulse.stop()
        assert raw_device_set(exp).sequencer.firing is None
    finally:
        exp.close()


def test_virtual_sequencer_prepare_accepts_pulse_table_state_and_returns_program():
    """Lowest level: VirtualSequencer.prepare(PulseTableState) compiles to a RuntimeSequenceProgram
    (the polymorphic prepare the real VirtualSequencer also satisfies) -- no TypeError on the
    'channels' kwarg, no None return."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer

    seqr = VirtualSequencer()
    state = na.build_release_recapture_pulse(port_catalog=seqr.port_catalog)
    prog = seqr.prepare(state)                      # PulseTableState -> compiled program
    assert prog is not None and prog.sequence_name == state.name
    assert seqr.last_program is prog

    # a plain PulseSequence still works (the measurement path) and returns a program.
    seq = na.imaging_sequence(exposure=20e-3, load=True, name="readout")
    prog2 = seqr.prepare(seq)
    assert prog2 is not None and prog2.sequence_name == "readout"
