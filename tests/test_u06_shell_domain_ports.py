"""The seams the GUI shells CANNOT close by moving a module: object ports.

Most of a shell's domain tendrils are cut by relocation - the imported module is
stdlib+numpy and simply sinks into ``zlc_data`` (H1a/H1b, ledgered in
``test_u05_shell_salvage``).  Some cannot be, and this file owns those.

``frontend/live.py`` drew the DAC waveform by importing the pulse compiler
(``timing.pulse_table.analog_bus_ticks`` / ``_analog_bus_value_at_tick``).  That
module is 3k lines over the port catalog, the serializer and the streamer
geometry, so relocating it is not on the table - and the import DAG forbids the
destination outright: ``zlc_frontend`` may never depend on ``zlc_pulse``.

So the seam is cut the other way: the STATE answers for its own waveform.
``PulseTableState.analog_bus_samples`` is the one authority for "what levels does
this bus hold, and when", and the render surface asks the object it was already
handed.  No import crosses the boundary, and no second sampler exists to drift.

The tests below are an independent oracle for that method (hand-computed
expectations, not a re-run of the implementation) plus the single-source ratchet
that the render surface has not grown a private mirror of it.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState

BUS = "da_x"
TICK_NS = 20.0


def _state(plan, durations_us=(1.0, 2.0, 1.0)) -> PulseTableState:
    """A state whose LAST periods carry ``plan`` on ``da_x``.

    ``PulseTableState`` seeds its own periods, so the appended ones land at the
    end; the plan is padded in front with holds to line up with them.
    """

    catalog = VirtualSequencer().port_catalog
    state = PulseTableState(port_catalog=catalog, time_step_ns=TICK_NS)
    off = tuple([0] * len(catalog.raw_lanes))
    seeded = len(state.periods)
    for index, micros in enumerate(durations_us):
        state.periods.append(PulsePeriod(micros, off, unit="us", name=f"p{index}"))
    state.analog_bus_modes[BUS] = [
        {"mode": "hold", "value": None} for _ in range(seeded)
    ] + [dict(entry) for entry in plan]
    return state


def _appended_starts(state: PulseTableState) -> list[int]:
    slots = state._reference_slots()
    return state.period_start_steps(slots=slots, time_step_ns=state.time_step_ns)


def test_an_all_edge_plan_samples_exactly_at_the_period_boundaries():
    """With no ramp there is nothing to interpolate: one step per period start.

    Expectation written out by hand from the durations - 1/2/1 us at a 20 ns tick
    is 50/100/50 ticks - not read back from the sampler.
    """

    state = _state(
        [
            {"mode": "edge", "value": 100},
            {"mode": "hold", "value": None},
            {"mode": "edge", "value": -250},
        ]
    )
    starts = _appended_starts(state)
    assert starts[-4:] == [starts[-4], starts[-4] + 50, starts[-4] + 150, starts[-4] + 200]

    ticks, values = state.analog_bus_samples(BUS, looping=False)

    assert ticks == starts, "an all-edge bus breaks exactly at the period boundaries"
    assert len(values) == len(ticks) - 1, "the last tick closes the final segment"
    # The seeded periods hold at idle; then edge 100, a hold carrying it, then -250.
    assert values[-3:] == [100, 100, -250]
    assert set(values[:-3]) == {0}


def test_a_looping_ramp_then_hold_reads_flat_but_a_single_fire_climbs():
    """The documented steady state, checked as physics rather than as code.

    A program whose only DAC entries are ``ramp V`` then ``hold`` leaves the frame
    sitting at V, so the NEXT loop iteration starts there: the hardware output is a
    flat V, and the preview must show that.  Fired ONCE the same program starts from
    idle 0 V and climbs, ending at V.
    """

    plan = [
        {"mode": "ramp", "value": 400},
        {"mode": "hold", "value": None},
        {"mode": "hold", "value": None},
    ]

    _ticks, looping = _state(plan).analog_bus_samples(BUS, looping=True)
    assert set(looping) == {400}, "a looping ramp->hold converges to a flat level"

    _ticks, single = _state(plan).analog_bus_samples(BUS, looping=False)
    assert single[0] == 0, "a single fire enters at idle 0 V"
    assert single[-1] == 400, "and finishes on the ramp target"
    assert single == sorted(single), "the one-shot ramp is monotonic to the target"
    assert len(set(single)) > 2, "a ramp is sampled inside the period, not just at its ends"


def test_a_scanned_slot_value_previews_at_its_reference_instead_of_failing():
    """A scanned DAC level is the string ``s0`` until the scan runs; a preview must
    still draw something, so slot references resolve to the slot's nominal value.

    Bound through the real editor path (``bind_field``), which rewrites the plan
    entry to ``s0`` and remembers 180 as its nominal - so this is the state a user
    who ticked "scan this DAC" actually previews.
    """

    state = _state(
        [
            {"mode": "edge", "value": 180},
            {"mode": "hold", "value": None},
            {"mode": "hold", "value": None},
        ]
    )
    first_appended = len(state.periods) - 3
    state.bind_field("dac", f"{BUS}@{first_appended}", label="bias", unit="")
    assert state.analog_bus_modes[BUS][first_appended]["value"] == "s0", (
        "binding rewrites the plan entry to a slot reference"
    )

    _ticks, values = state.analog_bus_samples(BUS, looping=False)
    assert values[-3:] == [180, 180, 180], (
        "an unresolved slot previews at its nominal value, not as a crash on int('s0')"
    )


def test_the_render_surface_has_no_private_copy_of_the_sampler():
    """The single-source ratchet: what ``live.py`` draws IS what the state reports.

    ``analog_bus_traces`` used to import the compiler helpers and re-derive the
    period prefix sum itself.  Both are now the state's job; this pins that the
    render output stays identical to the state's own answer, so a future edit
    cannot quietly reintroduce a second sampler that drifts.
    """

    from Zou_lab_control.frontend.live import analog_bus_traces

    state = _state(
        [
            {"mode": "edge", "value": 120},
            {"mode": "ramp", "value": -80},
            {"mode": "hold", "value": None},
        ]
    )
    traces, _folded = analog_bus_traces(state)
    drawn = {trace["name"]: trace for trace in traces}
    assert BUS in drawn, "the DAC bus must fold into one analog trace row"

    ticks, values = state.analog_bus_samples(BUS, looping=True)
    assert drawn[BUS]["values"] == values
    assert drawn[BUS]["starts"] == [tick * state.time_step_ns * 1e-9 for tick in ticks], (
        "the render converts the state's ticks to seconds and nothing else"
    )
