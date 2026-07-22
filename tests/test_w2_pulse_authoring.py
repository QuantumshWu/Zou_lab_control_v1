"""Current Pulse editor commands preserve typed field identity and scan data."""

from __future__ import annotations

from dataclasses import replace

import pytest

from zlc_pulse import (
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    AnalogStep,
    ApiParameter,
    DestructivePulseEditError,
    OutputDelay,
    PulseDocument,
    PulseFieldRef,
    ScanParameter,
    attach_scan_recipe,
    freeze_scan_table,
    load_deployed_pulse_target,
    new_pulse_document,
    replace_field_binding,
    set_analog_action,
    set_digital_output,
    set_output_delay,
)
from zlc_pulse.authoring import clear_port, clear_pulse_schedule, rename_port_label


def _blank():
    return new_pulse_document(
        load_deployed_pulse_target(),
        time_step_ns=20,
        name="editor pulse",
    )


def _port(document, kind: str) -> str:
    return next(port.key for port in document.target.ports if port.kind == kind)


def test_new_document_exposes_only_editable_outputs_and_digital_command_is_logical():
    document = _blank()
    digital = _port(document, PORT_DIGITAL)
    dac = _port(document, PORT_DAC)
    clock = _port(document, PORT_CLOCK)

    assert document.visible_ports == tuple(
        port.key
        for port in document.target.ports
        if port.kind in (PORT_DIGITAL, PORT_DAC)
    )
    assert clock not in document.visible_ports
    assert document.periods[0].period_id == "p1"
    assert not any(document.periods[0].states)
    with pytest.raises(ValueError, match="digital/DAC"):
        replace(document, visible_ports=(clock,))

    changed = set_digital_output(document, "p1", digital, True)
    lane = document.target.by_key[digital].lanes[0]
    lane_index = document.target.raw_lanes.index(lane)
    assert changed.periods[0].states[lane_index] == 1
    assert sum(changed.periods[0].states) == 1
    assert not any(document.periods[0].states)

    with pytest.raises(ValueError, match="not a digital output"):
        set_digital_output(document, "p1", dac, True)


def test_unrelated_edit_reuses_validated_scan_table_and_cached_indexes(monkeypatch):
    """A scalar editor commit must not walk an unchanged large scan table."""

    document = _blank()
    field = PulseFieldRef("duration", "p1", None)
    document = replace_field_binding(
        document,
        field,
        ScanParameter("duration_p1", field, "Duration", "ns"),
    ).document
    document = replace(
        document,
        periods=(replace(document.periods[0], duration=60),),
    )
    table, _report = freeze_scan_table(
        document,
        ("duration_p1",),
        tuple((20 + 20 * index,) for index in range(64)),
    )
    document = replace(document, scan_table=table)
    document = attach_scan_recipe(
        document,
        source="scan_table = [(20,)]\n",
        generated_columns={
            "duration_p1": tuple(row[0] for row in table.rows),
        },
    )

    calls = 0
    original = PulseDocument._validate_frozen_scan_value

    def counted(self, parameter, value, *, field):
        nonlocal calls
        calls += 1
        return original(self, parameter, value, field=field)

    monkeypatch.setattr(PulseDocument, "_validate_frozen_scan_value", counted)
    changed = replace(document, name="renamed")

    assert calls == 0
    assert changed.scan_table is table
    assert changed.target.by_key is document.target.by_key
    assert changed.period_by_id is changed.period_by_id
    assert changed.scan_parameter_by_id is changed.scan_parameter_by_id
    assert changed.scan_definition_digest == document.scan_definition_digest

    with pytest.raises(ValueError, match="clock grid"):
        replace(changed, time_step_ns=30)
    assert calls > 0


def test_bound_dac_action_removal_requires_and_applies_the_whole_cascade():
    document = _blank()
    dac = _port(document, PORT_DAC)
    document = set_analog_action(
        document,
        "p1",
        dac,
        AnalogStep(dac, "edge", 7),
    ).document
    field = PulseFieldRef("dac", "p1", dac)
    document = replace_field_binding(
        document,
        field,
        ScanParameter("bias", field, "Bias", "value"),
    ).document
    table, _report = freeze_scan_table(document, ("bias",), ((3,), (5,)))
    document = replace(document, scan_table=table)
    document = attach_scan_recipe(
        document,
        source="scan_columns = {'bias': [3, 5]}\n",
        generated_columns={"bias": (3, 5)},
    )

    with pytest.raises(DestructivePulseEditError) as caught:
        set_analog_action(document, "p1", dac, None)
    assert caught.value.impact.removed_scan_parameters == ("bias",)
    assert caught.value.impact.removed_scan_columns == ("bias",)
    assert caught.value.impact.scan_provenance_removed

    removed = set_analog_action(document, "p1", dac, None, cascade=True)
    assert removed.document.periods[0].analog_steps == ()
    assert removed.document.scan_parameters == ()
    assert removed.document.scan_table is None
    assert removed.document.scan_recipe is None
    assert document.field_value(field) == (7, "value")


def test_bound_output_delay_removal_cannot_leave_an_api_reference():
    document = _blank()
    digital = _port(document, PORT_DIGITAL)
    document = set_output_delay(
        document,
        digital,
        OutputDelay(digital, 40, "ns"),
    ).document
    field = PulseFieldRef("delay", None, digital)
    document = replace_field_binding(
        document,
        field,
        ApiParameter("camera_delay", field, "ns"),
    ).document

    with pytest.raises(DestructivePulseEditError) as caught:
        set_output_delay(document, digital, None)
    assert caught.value.impact.removed_api_parameters == ("camera_delay",)

    removed = set_output_delay(document, digital, None, cascade=True)
    assert removed.document.delays == ()
    assert removed.document.api_parameters == ()


def test_clear_logical_row_preserves_its_separate_delay_and_cascades_dac_binding():
    document = _blank()
    digital = _port(document, PORT_DIGITAL)
    dac = _port(document, PORT_DAC)
    document = set_digital_output(document, "p1", digital, True)
    document = set_analog_action(
        document,
        "p1",
        dac,
        AnalogStep(dac, "edge", 7),
    ).document
    dac_field = PulseFieldRef("dac", "p1", dac)
    document = replace_field_binding(
        document,
        dac_field,
        ScanParameter("bias", dac_field, "Bias", "value"),
    ).document
    table, _report = freeze_scan_table(document, ("bias",), ((3,), (5,)))
    document = replace(document, scan_table=table)
    document = set_output_delay(
        document,
        dac,
        OutputDelay(dac, 40, "ns"),
    ).document
    delay_field = PulseFieldRef("delay", None, dac)
    document = replace_field_binding(
        document,
        delay_field,
        ApiParameter("dac_delay", delay_field, "ns"),
    ).document

    with pytest.raises(DestructivePulseEditError):
        clear_port(document, dac)

    cleared_dac = clear_port(document, dac, cascade=True).document
    assert cleared_dac.periods[0].analog_steps == ()
    assert cleared_dac.scan_parameters == ()
    assert cleared_dac.scan_table is None
    assert cleared_dac.delays == (OutputDelay(dac, 40, "ns"),)
    assert cleared_dac.api_parameters == (
        ApiParameter("dac_delay", delay_field, "ns"),
    )

    cleared_digital = clear_port(document, digital, cascade=True).document
    lane = document.target.by_key[digital].lanes[0]
    lane_index = document.target.raw_lanes.index(lane)
    assert cleared_digital.periods[0].states[lane_index] == 0
    assert cleared_digital.scan_table == document.scan_table


def test_clear_all_resets_only_schedule_authority_and_keeps_editor_presentation():
    document = _blank()
    digital = _port(document, PORT_DIGITAL)
    dac = _port(document, PORT_DAC)
    document = rename_port_label(document, digital, "Camera trigger")
    document = replace(document, visible_ports=(digital,), name="Readout pulse")
    document = set_digital_output(document, "p1", digital, True)
    document = set_analog_action(
        document,
        "p1",
        dac,
        AnalogStep(dac, "ramp", 13),
    ).document
    document = set_output_delay(
        document,
        digital,
        OutputDelay(digital, 40, "ns"),
    ).document
    duration = PulseFieldRef("duration", "p1")
    document = replace_field_binding(
        document,
        duration,
        ScanParameter("duration", duration, "Duration", "ns"),
    ).document
    table, _report = freeze_scan_table(document, ("duration",), ((1000,), (1020,)))
    document = replace(document, scan_table=table)

    cleared = clear_pulse_schedule(document, cascade=True).document

    assert cleared.name == "Readout pulse"
    assert cleared.target == document.target
    assert cleared.target.by_key[digital].label == "Camera trigger"
    assert cleared.visible_ports == (digital,)
    assert cleared.time_step_ns == document.time_step_ns
    assert len(cleared.periods) == 1
    assert cleared.periods[0].duration == 1
    assert cleared.periods[0].unit == "us"
    assert not any(cleared.periods[0].states)
    assert cleared.periods[0].analog_steps == ()
    assert cleared.delays == ()
    assert cleared.scan_parameters == ()
    assert cleared.scan_table is None
    assert cleared.scan_recipe is None
    assert cleared.api_parameters == ()
    assert cleared.repeat is None


def test_binding_switch_preserves_other_scan_columns_and_invalidates_recipe():
    document = _blank()
    dac = _port(document, PORT_DAC)
    document = set_analog_action(
        document,
        "p1",
        dac,
        AnalogStep(dac, "edge", 7),
    ).document
    duration = PulseFieldRef("duration", "p1")
    bias = PulseFieldRef("dac", "p1", dac)
    document = replace_field_binding(
        document,
        duration,
        ScanParameter("duration", duration, "Duration", "ns"),
    ).document
    table, _report = freeze_scan_table(
        document,
        ("duration",),
        ((40,), (60,)),
    )
    document = replace(document, scan_table=table)
    document = replace_field_binding(
        document,
        bias,
        ScanParameter("bias", bias, "Bias", "value"),
    ).document
    assert document.scan_table is not None
    assert document.scan_table.columns == ("duration", "bias")
    assert document.scan_table.rows == ((40, 7), (60, 7))
    document = attach_scan_recipe(
        document,
        source="scan_columns = {'duration': [40, 60], 'bias': [7, 7]}\n",
        generated_columns={"duration": (40, 60), "bias": (7, 7)},
    )

    api = ApiParameter("duration_api", duration, "ns")
    with pytest.raises(DestructivePulseEditError):
        replace_field_binding(document, duration, api)
    switched = replace_field_binding(document, duration, api, cascade=True).document
    assert switched.scan_table is not None
    assert switched.scan_table.columns == ("bias",)
    assert switched.scan_table.rows == ((7,), (7,))
    assert switched.scan_recipe is None
    assert switched.scan_parameters[0].parameter_id == "bias"
    assert switched.api_parameters == (api,)

    with pytest.raises(ValueError, match="identity namespace"):
        replace_field_binding(
            switched,
            duration,
            ApiParameter("bias", duration, "ns"),
            cascade=True,
        )
    assert switched.api_parameters == (api,)


def test_rebinding_duration_unit_preserves_physical_scan_values():
    document = _blank()
    field = PulseFieldRef("duration", "p1")
    document = replace_field_binding(
        document,
        field,
        ScanParameter("duration", field, "Duration", "ns"),
    ).document
    table, _report = freeze_scan_table(document, ("duration",), ((40,), (60,)))
    document = replace(document, scan_table=table)

    rebound = replace_field_binding(
        document,
        field,
        ScanParameter("duration", field, "Duration", "us"),
    ).document

    assert rebound.scan_table is not None
    assert rebound.scan_table.rows == ((0.04,), (0.06,))
