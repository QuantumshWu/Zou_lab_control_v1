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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


from zlc_neutral_atom.timing.ports import PortCatalog
from zlc_neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState
from zlc_neutral_atom.timing.streamer_geometry import hardware_channel_names

BUS = "da_x"
TICK_NS = 20.0


def _state(plan, durations_us=(1.0, 2.0, 1.0)) -> PulseTableState:
    """A state whose LAST periods carry ``plan`` on ``da_x``.

    ``PulseTableState`` seeds its own periods, so the appended ones land at the
    end; the plan is padded in front with holds to line up with them.
    """

    lanes = tuple(f"dx{i}" for i in range(6))
    catalog = PortCatalog.from_channels(
        list(lanes) + hardware_channel_names(4), analog_buses={BUS: lanes})
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
            {"mode": "edge", "value": 20},
            {"mode": "hold", "value": None},
            {"mode": "edge", "value": -30},
        ]
    )
    starts = _appended_starts(state)
    assert starts[-4:] == [starts[-4], starts[-4] + 50, starts[-4] + 150, starts[-4] + 200]

    ticks, values = state.analog_bus_samples(BUS, looping=False)

    assert ticks == starts, "an all-edge bus breaks exactly at the period boundaries"
    assert len(values) == len(ticks) - 1, "the last tick closes the final segment"
    # The seeded periods hold at idle; then edge 100, a hold carrying it, then -250.
    assert values[-3:] == [20, 20, -30]
    assert set(values[:-3]) == {0}


def test_a_looping_ramp_then_hold_reads_flat_but_a_single_fire_climbs():
    """The documented steady state, checked as physics rather than as code.

    A program whose only DAC entries are ``ramp V`` then ``hold`` leaves the frame
    sitting at V, so the NEXT loop iteration starts there: the hardware output is a
    flat V, and the preview must show that.  Fired ONCE the same program starts from
    idle 0 V and climbs, ending at V.
    """

    plan = [
        {"mode": "ramp", "value": 31},
        {"mode": "hold", "value": None},
        {"mode": "hold", "value": None},
    ]

    _ticks, looping = _state(plan).analog_bus_samples(BUS, looping=True)
    assert set(looping) == {31}, "a looping ramp->hold converges to a flat level"

    _ticks, single = _state(plan).analog_bus_samples(BUS, looping=False)
    assert single[0] == 0, "a single fire enters at idle 0 V"
    assert single[-1] == 31, "and finishes on the ramp target"
    assert single == sorted(single), "the one-shot ramp is monotonic to the target"
    assert len(set(single)) > 2, "a ramp is sampled inside the period, not just at its ends"


def test_a_scanned_slot_value_previews_at_its_reference_instead_of_failing():
    """A scanned DAC level is the string ``s0`` until the scan runs; a preview must
    still draw something, so slot references resolve to the slot's nominal value.

    Bound through the real editor path (``bind_field``), which rewrites the plan
    entry to ``s0`` and remembers 25 as its nominal - so this is the state a user
    who ticked "scan this DAC" actually previews.
    """

    state = _state(
        [
            {"mode": "edge", "value": 25},
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
    assert values[-3:] == [25, 25, 25], (
        "an unresolved slot previews at its nominal value, not as a crash on int('s0')"
    )




# ------------------------------------------------- the pulse-replay object port


def test_the_render_layer_refuses_a_pulse_replay_it_cannot_build():
    """An unwired process must say so, not draw half a figure.

    Checked in a SEPARATE interpreter: this test suite imports the legacy
    frontend package, which registers the factory, so the unregistered state
    cannot be observed in-process without tearing down a global.
    """

    import subprocess
    import sys

    code = (
        "from zlc_frontend.domain_ports import PulseReplayUnavailable, "
        "pulse_state_from_dict, pulse_state_factory_is_registered\n"
        "assert not pulse_state_factory_is_registered()\n"
        "try:\n"
        "    pulse_state_from_dict({})\n"
        "except PulseReplayUnavailable as exc:\n"
        "    assert 'register_pulse_state_factory' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('an unwired replay must refuse')\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), check=True)


def test_the_product_surface_reaches_no_legacy_module():
    """The product surface is already legacy-free, and must stay that way.

    ``Zou_lab_control/notebook`` and ``Zou_lab_control/workbench`` survive Z0 as
    the user's entry points, so every legacy import they hold is one the deletion
    would have to resolve.  Today there are none - a property worth a ratchet
    rather than a coincidence worth discovering later.

    This guard was written after an attempt to wire the pulse-replay port from
    here.  ``zlc_pulse`` has no editor-state class yet, so such a factory must
    reach the legacy ``pulse_table``; wiring it at package import ALSO loaded the
    renderer and broke the headless guarantee
    (``test_headless_notebook_import_does_not_load_frontend_renderer``).  The
    port is not reachable from the product anyway: nothing in the product surface
    or ``zlc_workbench`` loads a saved npz figure, so ``SavedFigure.pulse_state``
    has no product caller until the saved-figure viewer is salvaged.  The
    registration moves then - lazily, on the render-touching path, never at
    import - and this set is where a premature attempt shows up.
    """

    import ast

    expected: set[Path] = set()
    found = set()
    for root in ("Zou_lab_control/notebook", "Zou_lab_control/workbench"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                if any(
                    name.startswith(("Zou_lab_control.frontend",
                                     "Zou_lab_control.neutral_atom"))
                    for name in names
                ):
                    found.add(path.relative_to(REPO_ROOT))

    assert found == expected, (
        "the product surface's legacy reach changed.\n"
        f"  now      : {sorted(str(p) for p in found)}\n"
        f"  ledgered : {sorted(str(p) for p in expected)}\n"
        "A new one is a regression; removing the last one means the pulse state "
        "has migrated - empty this set in the same commit."
    )


def test_importing_the_workbench_wires_the_pulse_replay_port():
    """Today's composition root: ``zlc_workbench`` (the call inherited from the
    deleted legacy frontend root).  Importing the package registers the factory,
    so any window composed from it can replay a saved pulse figure."""

    import zlc_workbench  # noqa: F401  - the import IS the wiring under test

    from zlc_frontend.domain_ports import (
        pulse_state_factory_is_registered,
        pulse_state_from_dict,
    )

    assert pulse_state_factory_is_registered()

    state = _state([{"mode": "edge", "value": 20}, {"mode": "hold", "value": None},
                    {"mode": "hold", "value": None}])
    rebuilt = pulse_state_from_dict(state.to_dict())
    assert type(rebuilt) is type(state), "the port rebuilds the real state class"
    assert rebuilt.analog_bus_samples(BUS) == state.analog_bus_samples(BUS), (
        "a round trip through the port preserves what the figure will draw"
    )


def test_a_second_conflicting_factory_is_refused():
    """Two constructors would mean two answers to 'what did this pulse look
    like'.  Re-registering the SAME callable stays fine - composition roots get
    imported more than once."""

    import pytest

    import zlc_workbench  # noqa: F401  - ensures the factory is wired
    from zlc_frontend import domain_ports

    current = domain_ports._PULSE_STATE_FACTORY
    assert current is not None
    domain_ports.register_pulse_state_factory(current)  # idempotent

    with pytest.raises(RuntimeError, match="ONE source"):
        domain_ports.register_pulse_state_factory(lambda data: None)
    assert domain_ports._PULSE_STATE_FACTORY is current


# ---------------------------------------------- the pulse-template row port


def test_an_unwired_process_refuses_to_read_a_pulse_template():
    """Refusal, not an empty form: a blank slots table would look like a template
    with no slots, which is a lie the operator cannot see through."""

    import subprocess
    import sys

    code = (
        "from zlc_frontend.domain_ports import (PulseTemplateUnavailable,\n"
        "    pulse_template_rows, pulse_template_reader_is_registered)\n"
        "assert not pulse_template_reader_is_registered()\n"
        "try:\n"
        "    pulse_template_rows('anything.json')\n"
        "except PulseTemplateUnavailable as exc:\n"
        "    assert 'register_pulse_template_reader' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('an unwired template read must refuse')\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), check=True)


def test_the_template_reader_returns_rows_the_gui_can_draw_without_pulse_types():
    """The port's whole point: what comes back is a DESCRIPTION.

    Every field is a plain str/float/tuple, so the slots form can be drawn by a
    process that never imported the pulse compiler - which is what lets the
    console stop reaching into a live PulseTableState.
    """

    import zlc_workbench  # noqa: F401  - ensures the reader is wired
    from zlc_frontend.domain_ports import (
        PulseTemplateRows,
        pulse_template_reader_is_registered,
        pulse_template_rows,
    )

    assert pulse_template_reader_is_registered()

    # Written by this test rather than read from pulses/probe_template.json: that
    # SHIPPED default is still in the pre-rename schema and cannot be loaded at
    # all (api_parameters / scan_recipe / target vs api_slots / scan_slots /
    # scan_code).  The console swallows the failure into an empty slots form, so
    # the breakage is invisible there; it is recorded in the design doc's S4 ledger and
    # is NOT frozen here as expected behaviour.
    import json
    import tempfile

    state = _state([{"mode": "edge", "value": 20},
                    {"mode": "hold", "value": None},
                    {"mode": "hold", "value": None}])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "template.json"
        path.write_text(json.dumps(state.to_dict(), default=str), encoding="utf-8")
        rows = pulse_template_rows(str(path))
    assert isinstance(rows, PulseTemplateRows)
    assert isinstance(rows.program, str) and isinstance(rows.program_id, str)
    for row in rows.api_rows:
        handle, coordinate, kind, target, unit, current = row
        assert all(isinstance(x, str) for x in (handle, coordinate, kind, target, unit))
        assert isinstance(current, float)
    for row in rows.scan_rows:
        coordinate, kind, target, unit, _label = row
        assert all(isinstance(x, str) for x in (coordinate, kind, target, unit))


