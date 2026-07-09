"""Virtual DA SAFE state -- ONE convention: an undriven bus rests at the mid-code (signed 0 = 0 V).

Every hardware-side layer already speaks it: the RTL parks buses at ``BUS_SAFE_VALUE`` (power-up,
reset/CMD_SAFE, each DA-bit delay FIFO's idle ``SAFE_BIT``), the host engine model rests an
un-segmented bus at the same mid code, and the encoder marks an untouched bus by leaving its member
bits all-0 (never projecting the mid code into TTL pulses).  ``decode_analog_bus`` was the ONE
decoder that broke the convention: it read "no pulse on any member" as offset-binary code 0 and
returned the full-negative word (-2^(B-1)) -- so firing a TTL-only program made the virtual MOT
monitor sense every coil at -32 and the spot vanished (eff ~ 1e-25), while explicitly driving the
DACs to 0 rendered the correct spot.  The single source is now
``timing.pulse_table.BUS_SAFE_SIGNED_LEVEL``; these contracts pin:

* decoder: a TTL-only sequence decodes every coil bus to the safe level -- and the known
  UN-DECODABLE corner (a bus driven to the full-negative code everywhere is bit-identical to an
  undriven bus) unifies to the same safe level, while an EXPLICIT signed 0 (MSB pulse present)
  still round-trips exactly through the driven path;
* delayed window: a DRIVEN bus carrying a bus delay rests at the SAME safe level BEFORE its
  delayed window -- the RTL's per-DA-bit delay scheduler idles each bit at ``SAFE_BIT`` (the
  mid-code bits) until the first scheduled change, ``out[t] = in[t-d]`` holding safe before
  ``t == d`` -- pinned against the exact hardware shift semantics
  ``engine_model.bus_delay_line_reference`` (itself proven == the RTL register mirror);
* end to end: a TTL-only fire and an explicit all-zero-DA fire sense the SAME levels and render
  statistically identical frames (the regression scene);
* no regression: an explicit NONZERO drive (the coil optimum) still senses exactly and hits the
  model's peak efficiency.

Expected values derive from the device/model objects (``cam.b0``, ``mot_efficiency``,
``BUS_SAFE_SIGNED_LEVEL``) -- never re-typed literals.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.virtual import VirtualMotCamera, VirtualSequencer
from Zou_lab_control.neutral_atom.operations.tasks.mot_field import DEFAULT_MOT_TEMPLATE
from Zou_lab_control.neutral_atom.timing import (
    BUS_SAFE_SIGNED_LEVEL, PulseSequence, resolve_fireable_template, single_imaging_template)
from Zou_lab_control.neutral_atom.timing.pulse_table import bus_signed_range
from Zou_lab_control.neutral_atom.timing.sequence import decode_analog_bus

MOT_TEMPLATE = str(REPO_ROOT / DEFAULT_MOT_TEMPLATE)


def _mot_state():
    return resolve_fireable_template(MOT_TEMPLATE, default_name=DEFAULT_MOT_TEMPLATE,
                                     default_factory=single_imaging_template)


def _sequence_at(state, values_by_slot):
    slots = [s.name for s in state.api_slots if s.kind == "dac"]
    return state.with_api_resolved(dict(zip(slots, values_by_slot))).to_sequence()


def _ttl_only_sequence() -> PulseSequence:
    # A program that never touches any coil bus member -- the regression trigger.
    return PulseSequence().pulse("probe", 0.0, 1e-4)


# --------------------------------------------------------------------------- decoder contract
def test_decoder_reads_an_undriven_bus_as_the_safe_level():
    """No pulse on ANY member (the encoder's "unused" marker) -> BUS_SAFE_SIGNED_LEVEL, never the
    all-bits-low full-negative word.  Pinned for all three coil buses of the MOT template."""
    state = _mot_state()
    seq = _ttl_only_sequence()
    t = 0.5 * seq.base_duration
    for members in state.analog_buses.values():
        assert decode_analog_bus(seq, members, t) == BUS_SAFE_SIGNED_LEVEL


def test_explicit_zero_and_full_negative_corner():
    """EXPLICIT signed 0 is a DRIVEN word (the mid code sets the MSB pulse) and round-trips
    exactly; the full-negative code projects NO bits anywhere -- bit-identical to an undriven
    bus -- so it unifies to the safe level (the documented un-decodable corner, matching the
    hardware whose unused buses genuinely rest at 0 V)."""
    state = _mot_state()
    n_slots = sum(1 for s in state.api_slots if s.kind == "dac")

    zero = _sequence_at(state, (0,) * n_slots)
    t = 0.5 * zero.base_duration
    for members in state.analog_buses.values():
        # the driven path really is exercised: the mid code puts a pulse on the MSB member
        assert any(p.channel == members[-1] for p in zero.base_pulses())
        assert decode_analog_bus(zero, members, t) == 0

    lo = [bus_signed_range(len(m))[0] for m in state.analog_buses.values()]
    negfull = _sequence_at(state, lo)
    for members in state.analog_buses.values():
        assert not any(p.channel in members for p in negfull.base_pulses())   # nothing projected
        assert decode_analog_bus(negfull, members, 0.5 * negfull.base_duration) \
            == BUS_SAFE_SIGNED_LEVEL


def test_delayed_driven_bus_rests_at_the_safe_level_before_its_window():
    """A DRIVEN bus with a bus delay rests at the SAFE level BEFORE its delayed window: the
    hardware's per-DA-bit delay scheduler idles each bit at its ``SAFE_BIT`` -- the bits of the
    mid-scale ``BUS_SAFE_VALUE`` code -- until the first scheduled change, i.e.
    ``out[t] = in[t-d]`` holding safe before ``t == d``.  The exact shift semantics is
    ``engine_model.bus_delay_line_reference`` (proven == the RTL register mirror in
    test_neutral_atom_lightweight), so decoding the DELAYED sequence on a uniform sample grid
    must equal that reference applied to the decode of the UNDELAYED sequence: the safe hold
    before the window (never the all-bits-low full-negative word), the programmed levels shifted
    after it."""
    from fpga.pulse_streamer.host.engine_model import bus_delay_line_reference
    from Zou_lab_control.neutral_atom.timing.pulse_table import bus_zero_code

    state = _mot_state()
    n_slots = sum(1 for s in state.api_slots if s.kind == "dac")
    values = tuple(range(3, 3 + n_slots))              # nonzero driven levels (inputs, in range)
    seq = _sequence_at(state, values)

    n_samples = 16
    step = seq.base_duration / n_samples
    d_steps = n_samples // 4
    delayed = seq
    for members in state.analog_buses.values():
        for channel in members:
            delayed = delayed.delay(channel, d_steps * step)

    times = [(k + 0.5) * step for k in range(n_samples)]   # mid-step samples, whole base cycle
    for members in state.analog_buses.values():
        zero = bus_zero_code(len(members))                 # wire mid code = the safe rest value
        undelayed = [decode_analog_bus(seq, members, t) + zero for t in times]
        expected = bus_delay_line_reference(undelayed, d_steps, safe_value=zero)
        got = [decode_analog_bus(delayed, members, t) + zero for t in times]
        assert got == expected
        # the pre-window region really is exercised: it holds the mid code, i.e. signed safe 0
        assert got[:d_steps] == [zero] * d_steps
        assert decode_analog_bus(delayed, members, 0.5 * d_steps * step) == BUS_SAFE_SIGNED_LEVEL


# --------------------------------------------------------------------------- end to end (virtual)
def test_ttl_only_fire_images_the_same_scene_as_explicit_zero_da():
    """USER SCENE: fire a pulse WITHOUT any DA -> the free-running monitor must image the SAME
    MOT spot as firing the template with every DAC explicitly at 0 (both drive 0 V on the coils).
    Before the fix the TTL-only fire sensed every coil at -2^(B-1) and the spot vanished."""
    state = _mot_state()
    n_slots = sum(1 for s in state.api_slots if s.kind == "dac")
    seqr = VirtualSequencer()
    cam = VirtualMotCamera(sequencer=seqr, seed=0)

    def fire_and_grab(seq):
        seqr.prepare(seq)
        seqr.fire(seq)
        frame = cam.acquire(1)[0].astype(float)
        return dict(cam.last_levels), frame

    levels_ttl, f_ttl = fire_and_grab(_ttl_only_sequence())
    levels_zero, f_zero = fire_and_grab(_sequence_at(state, (0,) * n_slots))

    # the ONE sense rule gives identical levels for both scenes -- the safe level everywhere
    assert levels_ttl == levels_zero == {bus: BUS_SAFE_SIGNED_LEVEL for bus in cam.coil_buses}

    # both frames show the SAME spot: contrast (centre disc vs border) follows the model, and the
    # two frames agree within photon/read noise (identical rate maps, independent noise draws)
    expected_extra = cam.peak_counts * cam.mot_efficiency(levels_ttl)   # spot counts above background
    cy, cx = cam.height // 2, cam.width // 2

    def contrast(frame):
        return frame[cy - 2:cy + 3, cx - 2:cx + 3].mean() - frame[:4, :].mean()

    assert contrast(f_ttl) > 0.5 * expected_extra      # the regression: this was ~0 (spot gone)
    assert contrast(f_zero) > 0.5 * expected_extra
    assert f_ttl.mean() == pytest.approx(f_zero.mean(), abs=2.0)
    assert abs(float(f_ttl.max()) - float(f_zero.max())) < 0.5 * expected_extra


def test_explicit_nonzero_da_still_senses_exactly():
    """No regression on the DRIVEN path: firing at the coil optimum senses exactly ``b0`` and
    sits at the model's peak efficiency."""
    state = _mot_state()
    seqr = VirtualSequencer()
    cam = VirtualMotCamera(sequencer=seqr, seed=0)
    slots = [s.name for s in state.api_slots if s.kind == "dac"]
    at_peak = [int(cam.b0[bus]) for bus in cam.coil_buses]
    seq = _sequence_at(state, at_peak)
    seqr.prepare(seq)
    seqr.fire(seq)
    cam.acquire(1)
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}
    assert cam.mot_efficiency(cam.last_levels) == pytest.approx(1.0)
    assert len(slots) == len(cam.coil_buses)           # template and camera agree on the bus count
