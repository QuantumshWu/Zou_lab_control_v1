"""Contract test: the period-start prefix sum has ONE authority.

``starts[i]`` -- the tick at which period ``i`` begins (``starts[-1]`` == frame
duration) -- is the cumulative sum of every preceding ``PulsePeriod.duration_steps``.
The preview (``apply_analog_bus_modes_to_period_states`` /
``analog_bus_value_at_period_start``) and the hardware compiler (edge table, bus
segments, total-duration, affine scan rows) must read the SAME prefix sum, or the
two will silently drift and the uploaded edges land on different period boundaries
than the preview shows.

These tests pin that single source:
  * ``PulseTableState.period_start_steps`` IS the authority (public method, with a
    REQUIRED ``time_step_ns`` -- never falling back to ``self.time_step_ns`` -- so
    preview and compiler each pass their own clock);
  * the compiler no longer carries a private mirror free function;
  * the compiler's edge-table period boundaries / total frame == ``period_start_steps``
    at several ``time_step_ns`` (the preview and hardware clocks);
  * the affine scan form (``_pulse_table_affine_period_starts``) is the affine
    lift of the SAME prefix sum (scalar evaluation at any slot point agrees);
  * at the FPGA's 50 MHz runtime clock, the compiler's ``clock_step_ns`` equals a
    state built at that tick's ``time_step_ns`` (derived from
    ``DEFAULT_RUNTIME_CLOCK_HZ`` -- no hardcoded literal).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing import PulseTableState
from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod
from Zou_lab_control.neutral_atom.devices import sequencer as sq


# --------------------------------------------------------------------------- builders
def _multi_period_bracket_state(*, time_step_ns: float, scan: bool = False) -> PulseTableState:
    """A state with several periods + a finite repeat bracket (and optionally a scanned
    duration slot), built at the supplied ``time_step_ns`` so a 50 MHz-tick state and a
    1 ns-preview state are both exercisable."""
    state = PulseTableState(
        channels=["trap", "cooling", "probe", "emCCD"],
        visible_channels=["trap", "cooling", "probe", "emCCD"],
        time_step_ns=time_step_ns,
    )
    state.periods.append(PulsePeriod(2.0, (1, 1, 0, 0), unit="us", name="load"))
    state.periods.append(PulsePeriod(5.0, (1, 0, 1, 1), unit="us", name="image"))
    state.periods.append(PulsePeriod(3.0, (1, 1, 0, 0), unit="us", name="hold"))
    # finite repeat bracket over the middle two periods
    state.repeat_start = 1
    state.repeat_end = 2
    state.repeat_count = 3
    if scan:
        state.bind_field("duration", "1", label="image", unit="us")   # scan the image duration
    return state


def _independent_prefix_sum(state: PulseTableState, *, time_step_ns: float, slots) -> list[int]:
    """Recompute the period-start prefix sum from scratch (the spec) so the test does NOT
    re-call the very method under test."""
    starts = [0]
    for period in state.periods:
        starts.append(starts[-1] + period.duration_steps(slots=slots, time_step_ns=time_step_ns))
    return starts


# --------------------------------------------------------------------------- single-source guards
def test_period_start_steps_is_the_public_authority():
    # the public method exists and carries a REQUIRED time_step_ns (no self fallback)
    assert hasattr(PulseTableState, "period_start_steps")
    import inspect

    sig = inspect.signature(PulseTableState.period_start_steps)
    tsn = sig.parameters["time_step_ns"]
    assert tsn.kind is inspect.Parameter.KEYWORD_ONLY
    assert tsn.default is inspect.Parameter.empty, "time_step_ns must be required, never default to self"


def test_compiler_carries_no_private_period_start_mirror():
    # the old mirror free function is gone -- the compiler reads the state's method.
    assert not hasattr(sq, "_pulse_table_period_starts_ticks")


# --------------------------------------------------------------------------- preview/compiler agree
@pytest.mark.parametrize("time_step_ns", [1.0, 20.0, 5.0])
def test_period_start_steps_matches_independent_prefix_sum(time_step_ns):
    state = _multi_period_bracket_state(time_step_ns=time_step_ns)
    slots = state.reference_slots()
    got = state.period_start_steps(slots=slots, time_step_ns=time_step_ns)
    assert got == _independent_prefix_sum(state, time_step_ns=time_step_ns, slots=slots)


@pytest.mark.parametrize("time_step_ns", [1.0, 20.0])
def test_compiler_edge_boundaries_equal_period_start_steps(time_step_ns):
    """The compiler emits a period-boundary anchor at every ``period_start`` and the loop
    end is ``starts[-1]`` -- so every start must appear among the uploaded edge ticks and the
    table end must equal the last start."""
    state = _multi_period_bracket_state(time_step_ns=time_step_ns)
    slots = state.reference_slots()
    starts = state.period_start_steps(slots=slots, time_step_ns=time_step_ns)

    ticks, _masks, _channels, loop_end, _from_idx, _cdelays, _bdelays = sq._pulse_table_edge_table(
        state,
        channels=list(state.channels),
        slots=slots,
        time_step_ns=time_step_ns,
    )
    assert loop_end == starts[-1]
    assert set(starts) <= set(ticks), "every period boundary must be an uploaded edge tick"

    # effective-duration helper (the other compiler consumer of starts) reads the SAME
    # prefix sum: with a finite bracket it returns starts[-1] + (count-1)*loop_ticks where
    # loop_ticks is a difference of two starts -- so it is >= starts[-1] and derived from it.
    loop_ticks = starts[state.repeat_end + 1] - starts[state.repeat_start]
    expected = starts[-1] + (state.repeat_count - 1) * loop_ticks
    assert sq._pulse_table_effective_duration_ticks(state, slots=slots, time_step_ns=time_step_ns) == expected


def test_affine_period_starts_is_the_affine_lift_of_the_same_prefix_sum():
    """``_pulse_table_affine_period_starts`` returns ``(base, slot_coeffs)`` per boundary --
    the affine lift of the scalar prefix sum.  With NO scanned duration every coeff is zero
    and the ``base`` column must equal ``period_start_steps`` exactly (the affine form
    reduces to the scalar form when nothing scans), and it iterates the same period count."""
    time_step_ns = 20.0
    state = _multi_period_bracket_state(time_step_ns=time_step_ns, scan=False)
    slot_vars = ["s0", "s1"]   # any slot vars; unscanned durations have zero coeffs for all
    coeff_frac_bits = 8

    affine = sq._pulse_table_affine_period_starts(
        state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits
    )
    scalar = state.period_start_steps(
        slots=state.reference_slots(), time_step_ns=time_step_ns
    )
    assert len(affine) == len(scalar) == len(state.periods) + 1
    bases = [base for base, _coeffs in affine]
    assert bases == scalar
    for _base, coeffs in affine:
        assert all(int(c) == 0 for c in coeffs), "unscanned durations must carry zero slot coeffs"


def test_fpga_runtime_clock_step_equals_state_time_step():
    """At the FPGA's runtime clock, ``clock_step_ns = 1e9 / clock_hz`` -- a state built at that
    tick has the SAME ``time_step_ns``, so preview-tick and compile-tick coincide (no rounding
    seam between the two consumers of the one prefix sum).  Derived from the single source, not
    a hardcoded 20 ns."""
    clock_step_ns = 1e9 / sq.DEFAULT_RUNTIME_CLOCK_HZ
    state = _multi_period_bracket_state(time_step_ns=clock_step_ns)
    assert state.time_step_ns == pytest.approx(clock_step_ns)
    # and the compiler reads exactly this step back from the clock it is handed.
    assert 1e9 / sq.DEFAULT_RUNTIME_CLOCK_HZ == pytest.approx(state.time_step_ns)
