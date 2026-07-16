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
