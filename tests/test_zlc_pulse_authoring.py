"""Stable typed Pulse authoring cannot silently rebind physical fields."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from zlc_pulse import (
    AnalogStep,
    ApiParameter,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    PulsePeriod,
    RepeatRegion,
    ScanParameter,
    attach_scan_recipe,
    compile_pulse_document,
    freeze_scan_table,
    insert_period,
    load_pulse_document,
    move_period,
    new_period,
    pulse_document_from_tree,
    pulse_document_to_tree,
    replace_pulse_field,
    remove_period,
    resolve_api_parameters,
)
from zlc_pulse.authoring import DestructivePulseEditError


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


def test_freeze_scan_table_names_columns_and_reports_clock_normalization():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")

    table, report = freeze_scan_table(document, ("t_off",), ((21.0,), (39.0,)))

    assert table.columns == ("t_off",)
    assert table.rows == ((20.0,), (40.0,))
    assert report.adjusted_cells == 2
    assert report.by_parameter["t_off"].unit == "ns"
    assert report.by_parameter["t_off"].max_abs_adjustment == 1.0


def test_document_rejects_a_non_frozen_table_instead_of_snapping_at_compile():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")

    with pytest.raises(ValueError, match="not frozen"):
        replace(document, scan_table=FrozenScanTable(("t_off",), ((21.0,),)))


def test_one_physical_field_cannot_have_two_scan_owners_or_scan_and_api():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    original = document.scan_parameters[0]
    duplicate = replace(original, parameter_id="another_exposure")

    with pytest.raises(ValueError, match="multiple scan parameters"):
        replace(document, scan_parameters=(original, duplicate))

    api = ApiParameter("release_exposure", original.field, "ns")
    with pytest.raises(ValueError, match="both scan- and API-bound"):
        replace(document, api_parameters=(api,))


def test_free_form_scan_expressions_cannot_enter_authoring_values():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    period = document.periods[0]

    with pytest.raises(TypeError, match="numeric"):
        PulsePeriod(
            period_id="new_period",
            duration="s0 + 100",
            unit="ns",
            name="",
            states=period.states,
        )


def test_api_resolution_uses_semantic_identity_and_changes_only_its_field():
    document = load_pulse_document(IMAGING_TEMPLATE)
    before = tuple(period.duration for period in document.periods)

    resolved = resolve_api_parameters(document, {"readout_probe_duration": 0.007})

    after = tuple(period.duration for period in resolved.periods)
    assert before == tuple(period.duration for period in document.periods)
    assert after[:3] == before[:3]
    assert after[3] == 0.007
    assert after[4:] == before[4:]
    with pytest.raises(KeyError, match="unknown pulse API"):
        resolve_api_parameters(document, {"a2": 0.007})


def test_delay_api_requires_and_updates_one_explicit_logical_port_delay():
    document = load_pulse_document(IMAGING_TEMPLATE)
    delay_ref = PulseFieldRef("delay", None, "ch11")
    document = replace(
        document,
        delays=(OutputDelay("ch11", 0, "ns"),),
        api_parameters=(*document.api_parameters, ApiParameter("camera_delay", delay_ref, "ns")),
    )

    resolved = resolve_api_parameters(document, {"camera_delay": 40})

    assert resolved.field_value(delay_ref) == (40.0, "ns")
    assert document.field_value(delay_ref) == (0, "ns")


def test_period_insert_and_move_preserve_stable_field_references():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    field = document.scan_parameters[0].field
    added = new_period(document, duration=100, unit="ns", name="extra")

    inserted = insert_period(document, period=added, before="p2").document
    moved = move_period(inserted, period_id=added.period_id, before=None).document

    assert inserted.scan_parameters[0].field == field
    assert moved.scan_parameters[0].field == field
    assert moved.periods[-1].period_id == added.period_id
    with pytest.raises(FrozenInstanceError):
        moved.target = document.target


def test_period_removal_preflights_and_reports_every_cascade():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    table, _report = freeze_scan_table(document, ("t_off",), ((20.0,), (40.0,)))
    document = replace(document, scan_table=table)
    document = attach_scan_recipe(
        document,
        source="scan_columns = {'t_off': [20, 40]}\n",
        generated_columns={"t_off": (20.0, 40.0)},
    )

    with pytest.raises(DestructivePulseEditError) as caught:
        remove_period(document, "p3")

    impact = caught.value.impact
    assert impact.removed_period_ids == ("p3",)
    assert impact.removed_scan_parameters == ("t_off",)
    assert impact.removed_scan_columns == ("t_off",)
    assert impact.scan_provenance_removed

    result = remove_period(document, "p3", cascade=True)
    assert result.document.scan_parameters == ()
    assert result.document.scan_table is None
    assert result.document.scan_recipe is None
    assert document.scan_parameters[0].parameter_id == "t_off"


def test_api_bound_period_removal_is_never_silent():
    document = load_pulse_document(IMAGING_TEMPLATE)

    with pytest.raises(DestructivePulseEditError) as caught:
        remove_period(document, "p4")

    assert caught.value.impact.removed_api_parameters == ("readout_probe_duration",)
    result = remove_period(document, "p4", cascade=True)
    assert "readout_probe_duration" not in result.document.api_parameter_by_id


def test_repeat_endpoints_use_period_identity_across_front_insertions():
    document = load_pulse_document(IMAGING_TEMPLATE)
    repeated = replace(document, repeat=RepeatRegion("p2", "p4", 3))
    added = new_period(repeated, duration=100, unit="ns")

    inserted = insert_period(repeated, period=added, before="p1").document

    assert inserted.repeat == RepeatRegion("p2", "p4", 3)

    inside = new_period(repeated, duration=120, unit="ns")
    with pytest.raises(DestructivePulseEditError) as caught:
        insert_period(repeated, period=inside, before="p3")
    assert caught.value.impact.repeat_members_after == (
        "p2",
        inside.period_id,
        "p3",
        "p4",
    )
    edit = insert_period(repeated, period=inside, before="p3", cascade=True)
    assert edit.impact.repeat_members_before == ("p2", "p3", "p4")
    assert edit.impact.repeat_members_after == ("p2", inside.period_id, "p3", "p4")

    with pytest.raises(DestructivePulseEditError):
        move_period(repeated, period_id="p1", before="p3")
    moved_inside = move_period(
        repeated,
        period_id="p1",
        before="p3",
        cascade=True,
    )
    assert moved_inside.impact.repeat_members_after == ("p2", "p1", "p3", "p4")


def test_scan_recipe_must_reproduce_table_and_remains_bound_to_freeze_context():
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    table, _report = freeze_scan_table(document, ("t_off",), ((20.0,), (40.0,)))
    document = replace(document, scan_table=table)

    with pytest.raises(ValueError, match="does not reproduce"):
        attach_scan_recipe(
            document,
            source="scan_columns = {'t_off': [999999999]}\n",
            generated_columns={"t_off": (999999999.0,)},
        )

    attached = attach_scan_recipe(
        document,
        source="scan_columns = {'t_off': [20, 40]}\n",
        generated_columns={"t_off": (20.0, 40.0)},
    )
    assert pulse_document_from_tree(pulse_document_to_tree(attached)) == attached
    changed_unit = replace(attached.scan_parameters[0], unit="us")
    with pytest.raises(ValueError, match="frozen scan definition"):
        replace(attached, scan_parameters=(changed_unit,))


def test_authoritative_api_and_field_edits_never_snap_silently():
    document = load_pulse_document(IMAGING_TEMPLATE)
    reference = document.api_parameter_by_id["readout_probe_duration"].field
    half_tick_seconds = (1_000_000_000_000 + 0.5) * 20e-9

    with pytest.raises(ValueError, match="not frozen"):
        replace_pulse_field(
            document,
            reference,
            half_tick_seconds,
            unit="s",
        )

    dac = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    with pytest.raises(ValueError, match="unit 'value'"):
        replace_pulse_field(
            dac,
            dac.scan_parameters[0].field,
            0,
            unit="ns",
        )


def test_generated_period_identity_is_not_a_reusable_sequence_position():
    document = load_pulse_document(IMAGING_TEMPLATE)
    removed = remove_period(document, "p6", cascade=True).document

    created = new_period(removed)

    assert created.period_id != "p6"
    assert created.period_id.startswith("p_")


def test_analog_step_order_is_canonical_and_has_no_parallel_period_array():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    period = document.periods[0]
    reversed_steps = replace(period, analog_steps=tuple(reversed(period.analog_steps)))

    rebuilt = replace(document, periods=(reversed_steps, *document.periods[1:]))

    assert rebuilt == document
    assert rebuilt.fingerprint == document.fingerprint
    assert not hasattr(document, "analog_bus_programs")


def test_scan_table_columns_are_identity_bound_not_parameter_ordinal():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    table = document.scan_table
    assert table is not None
    reordered_parameters = tuple(reversed(document.scan_parameters))

    reordered = replace(document, scan_parameters=reordered_parameters)

    assert reordered.scan_table == table
    assert reordered.scan_table.columns == ("da_x", "da_y", "da_z")
    assert compile_pulse_document(
        reordered,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    ) == compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )


def test_dac_parameter_must_reference_an_explicit_step():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    missing = PulseFieldRef("dac", "p2", "da_bias_x")
    parameter = ScanParameter("missing_step", missing, "", "value")

    with pytest.raises(ValueError, match="explicit edge/ramp"):
        replace(document, scan_parameters=(*document.scan_parameters, parameter))


def test_period_states_cannot_encode_hidden_dac_or_clock_values():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    period = document.periods[0]
    dac_lane = next(
        lane
        for port in document.target.ports
        if port.kind == "dac"
        for lane in port.lanes
    )
    index = document.target.raw_lanes.index(dac_lane)
    states = list(period.states)
    states[index] = 1

    with pytest.raises(ValueError, match="non-digital lane"):
        replace(
            document,
            periods=(replace(period, states=tuple(states)), *document.periods[1:]),
        )
