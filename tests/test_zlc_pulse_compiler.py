from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    COMPILER_ID,
    PORT_DAC,
    PORT_DIGITAL,
    PulseExecutionForm,
    compile_pulse_artifact,
    compile_pulse_document,
    load_pulse_document,
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "form", "expected_ir_digest"),
    (
        ("camera_imaging_address_switch.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "1c8f05d640be640f22a72a516ddb2d6c1d08274cdf9c640b160328e0fd042d52"),
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "3b49211b15c50d0cbdb9f61e35dc8594070c1d6fe8daca058fb18e380a47e67d"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "36934f9e37e3ed785d4144e37b520e2d52e36229aa604229d3eeebc3589dad26"),
        ("probe_template.json", PulseExecutionForm.STATIC_ONCE, "376609a880f39cc65f9b928e38fdaeaa3ee0355d376330f83ad941b369575564"),
        ("pulse_test.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "4720f17f31aa6421da65dfff7b1ac05a236a7a26f93ccdb36577e98706724b0e"),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "0bacd7db575bb5e830ccb12872c13fb3231bb58be8f2c93e389a58abeee054a7"),
        ("T.json", PulseExecutionForm.STATIC_ONCE, "a1ec95ee70ac18ab421b865682bacf18251a1d71b766f2ba7cd0d4970b8aea2e"),
    ),
)
def test_native_compiler_matches_the_frozen_migration_golden(filename, form, expected_ir_digest):
    document = load_pulse_document(ROOT / "pulses" / filename)

    actual = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=form,
    )

    assert actual.fingerprint == expected_ir_digest


def test_native_compiler_preserves_compact_and_delay_unrolled_repeat_semantics():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    active_lane = next(
        lane
        for lane_index, lane in enumerate(document.target.raw_lanes)
        if any(period.states[lane_index] for period in document.periods)
    )
    compact = replace(
        document,
        repeat_start=1,
        repeat_end=3,
        repeat_count=3,
        repeat_forever=False,
    )
    delayed = replace(
        compact,
        delays=((active_lane, -40),),
        delay_units=((active_lane, "ns"),),
    )

    compact_ir = compile_pulse_document(
        compact,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    delayed_ir = compile_pulse_document(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )

    assert compact_ir.fingerprint == "38d96258e9bc011aea54ef270254a311b87e2168771b55d56389071bcfd9a0ec"
    assert delayed_ir.fingerprint == "5c43d993a8ea26f2fa686a2ae7899eca9f995885d9abac69c7de4cfc01505060"
    assert compact_ir.loop_count == 3
    assert delayed_ir.loop_count == 1


def test_native_compiler_folds_digital_and_dac_delays_in_one_physical_schedule():
    document = load_pulse_document(ROOT / "pulses" / "T.json")
    bus = next(port for port in document.target.ports if port.kind == PORT_DAC)
    ttl = next(
        lane
        for lane_index, lane in enumerate(document.target.raw_lanes)
        if lane not in bus.lanes and any(period.states[lane_index] for period in document.periods)
    )
    delays = tuple((lane, 100) for lane in bus.lanes) + ((ttl, -40),)
    delayed = replace(
        document,
        delays=delays,
        delay_units=tuple((lane, "ns") for lane, _value in delays),
    )

    actual = compile_pulse_document(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )

    assert actual.fingerprint == "6a1eac76d5f571e7b4302d4fec1cf26727da7a6c3c13c3e0759202365cfdd538"
    assert any(actual.channel_delays)
    assert any(delay.delay_ticks for delay in actual.bus_delays)


def test_artifact_uses_native_compiler_identity_and_one_geometry_for_ir_and_wire():
    document = load_pulse_document(ROOT / "pulses" / "pulse_test.json")
    trigger = next(
        port.lanes[0]
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )

    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=(trigger,),
    )

    assert artifact.compiler_id == COMPILER_ID
    assert artifact.compiler_version == "1"
    assert artifact.target_ir.scan_coeff_frac_bits == 8
    assert artifact.wire_image.source_ir_digest == artifact.target_ir.fingerprint


def test_compiler_rejects_scan_loss_and_historical_cursor_wrap_semantics():
    scan = load_pulse_document(ROOT / "pulses" / "pulse_test.json")
    with pytest.raises(ValueError, match="silently ignore"):
        compile_pulse_document(
            scan,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.STATIC_ONCE,
        )
    with pytest.raises(ValueError, match="scan_repeats"):
        compile_pulse_document(
            replace(scan, scan_repeats=2),
            clock_hz=50e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        )
