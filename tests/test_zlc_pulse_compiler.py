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
from zlc_workbench.pulse_compile_bridge import (
    compile_pulse_document as compile_with_migration_oracle,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "form"),
    (
        ("camera_imaging_address_switch.json", PulseExecutionForm.STATIC_REFERENCE_POINT),
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE),
        ("probe_template.json", PulseExecutionForm.STATIC_ONCE),
        ("pulse_test.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT),
        ("T.json", PulseExecutionForm.STATIC_ONCE),
    ),
)
def test_native_compiler_is_target_ir_identical_to_the_one_time_oracle(filename, form):
    document = load_pulse_document(ROOT / "pulses" / filename)

    expected = compile_with_migration_oracle(
        document,
        clock_hz=50e6,
        execution_form=form,
    )
    actual = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=form,
    )

    assert actual == expected


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

    assert compact_ir == compile_with_migration_oracle(
        compact,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    assert delayed_ir == compile_with_migration_oracle(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
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

    assert actual == compile_with_migration_oracle(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
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
