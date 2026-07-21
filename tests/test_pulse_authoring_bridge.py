"""What the editor authors must arrive intact at the pipeline that runs it.

The editor's state and the pipeline's document keep the same programme in two
shapes, and the two differences below are the ones that would corrupt a pulse
rather than merely look wrong -- so they are pinned here.

Everything is driven through the editor state's own API, the way the window's
buttons drive it: binding a scan field, setting a bus value, freezing a table.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from zlc_neutral_atom.pulse_authoring import pulse_document_from_table
from zlc_neutral_atom.timing.board_config import load_board_config
from zlc_neutral_atom.timing.pulse_table import PulsePeriod as EditorPeriod
from zlc_neutral_atom.timing.pulse_table import PulseTableState
from zlc_pulse.authoring import new_pulse_document


def _authored_state() -> PulseTableState:
    board = load_board_config()
    catalog = board.port_catalog()
    state = PulseTableState(
        port_catalog=catalog,
        visible_ports=[port.key for port in catalog.ports if port.kind != "clock"][:4],
    )
    state.bind_field("duration", "0", label="hold time", unit="us")
    state.bind_api_field("delay", "ch00", unit="ns")
    state.set_bus_value(0, "da_bias_x", -120)
    state.delays["ch00"] = 40.0
    state.delay_units["ch00"] = "ns"
    state.set_scan_table([[1000.0], [2000.0], [3000.0]])
    state.scan_code = "import numpy as np\nscan = np.linspace(1e3, 3e3, 3)"
    return state


def test_the_two_models_are_a_draft_and_a_validated_artefact_not_two_copies() -> None:
    """Why a translation exists at all -- and why it is not a duplicated source.

    A single source of truth forbids one fact being writable in two places that can
    drift.  These two are not that: the document is DERIVED from the editor state and
    never written back, and it cannot even represent what the editor must hold while a
    human is mid-edit.  Pinning that here so nobody later "unifies" them by handing a
    document to the editor, or a half-typed state to the hardware.
    """

    board = load_board_config()
    document = new_pulse_document(board.pulse_target(), time_step_ns=20.0, name="draft")
    period = document.periods[0]

    # An operator typing "1234.5" into a 20 ns machine is a legal editing state and an
    # illegal programme.  The document refuses it -- that refusal is its whole job.
    with pytest.raises(ValueError):
        replace(document, periods=(replace(period, duration=1234.5, unit="ns"),))
    with pytest.raises(TypeError):
        replace(document, periods=(replace(period, duration="s0", unit="ns"),))

    # The editor must accept both: the first is being typed, the second is a scan
    # binding that only becomes a number once a slot supplies one.
    states = (0,) * len(board.pulse_target().raw_lanes)
    assert EditorPeriod(duration=1234.5, states=states, unit="ns").duration == 1234.5
    assert EditorPeriod(duration="s0", states=states, unit="str (ns)").duration == "s0"


def test_an_unquantised_edit_is_refused_loudly_by_the_bridge() -> None:
    """The bridge must not round a half-typed duration onto the grid behind the operator."""

    board = load_board_config()
    catalog = board.port_catalog()
    state = PulseTableState(
        port_catalog=catalog,
        visible_ports=[port.key for port in catalog.ports if port.kind != "clock"][:2],
    )
    state.periods[0] = replace(state.periods[0], duration=1234.5, unit="ns")

    with pytest.raises((ValueError, TypeError)):
        pulse_document_from_table(state, board.pulse_target())


def test_a_dac_value_moves_out_of_the_bit_vector_into_an_analog_step() -> None:
    board = load_board_config()
    target = board.pulse_target()
    document = pulse_document_from_table(_authored_state(), target)

    dac_lanes = {lane for port in target.ports if port.kind == "dac" for lane in port.lanes}
    for period in document.periods:
        lit = [lane for lane, bit in zip(target.raw_lanes, period.states) if bit and lane in dac_lanes]
        assert not lit, (
            f"{period.period_id} still carries the DAC value as digital bits {lit}; a document "
            "states it once as an AnalogStep and keeps those lanes at 0")

    first = document.periods[0]
    assert [(step.port, step.mode, step.value) for step in first.analog_steps] == \
           [("da_bias_x", "edge", -120)], "the authored value must survive as-is (signed user layer)"


def test_a_scanned_duration_becomes_concrete_plus_a_typed_binding() -> None:
    document = pulse_document_from_table(_authored_state(), load_board_config().pulse_target())

    # The editor holds "s0"; a document must hold a number the hardware can quantise,
    # plus the binding that keeps the field scannable.
    assert isinstance(document.periods[0].duration, (int, float))
    assert document.periods[0].duration > 0
    bound = [(p.parameter_id, p.field.kind, p.field.period_id) for p in document.scan_parameters]
    assert bound == [("hold_time", "duration", document.periods[0].period_id)]

    api = [(p.parameter_id, p.field.kind, p.field.port) for p in document.api_parameters]
    assert api == [("a1", "delay", "ch00")]


def test_the_frozen_table_is_stamped_by_its_owner_not_by_the_bridge() -> None:
    document = pulse_document_from_table(_authored_state(), load_board_config().pulse_target())

    assert document.scan_table is not None
    assert document.scan_table.columns == ("hold_time",)
    assert len(document.scan_table.rows) == 3
    # A recipe only attaches when its source reproduces the frozen table exactly; that the
    # digest exists at all is the proof the owner did the freezing.
    assert document.scan_recipe is not None
    assert document.scan_recipe.frozen_definition_digest
    assert document.scan_recipe.columns == document.scan_table.columns
