from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    COMPILER_ID,
    FIELD_DURATION,
    OutputDelay,
    PORT_DAC,
    PORT_DIGITAL,
    PulseExecutionForm,
    PulseFieldRef,
    RepeatRegion,
    ScanParameter,
    compile_pulse_artifact,
    compile_pulse_document,
    freeze_scan_table,
    load_pulse_document,
)


ROOT = Path(__file__).resolve().parents[1]


def test_native_compiler_preserves_compact_and_delay_unrolled_repeat_semantics():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    active_port = next(
        port
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )
    compact = replace(document, repeat=RepeatRegion("p2", "p4", 3))
    delayed = replace(compact, delays=(OutputDelay(active_port.key, -40, "ns"),))

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

    assert compact_ir.ticks == (
        0,
        100_000,
        1_100_000,
        1_105_000,
        1_355_000,
        1_360_000,
        2_360_000,
    )
    assert compact_ir.masks == (513, 2568, 512, 2568, 512, 2568, 0)
    assert compact_ir.loop_start_index == 1
    assert compact_ir.loop_end_tick == 1_355_000
    assert compact_ir.loop_count == 3
    assert not any(compact_ir.channel_delays)

    assert delayed_ir.ticks == (
        0,
        100_000,
        1_100_000,
        1_105_000,
        1_355_000,
        2_355_000,
        2_360_000,
        2_610_000,
        3_610_000,
        3_615_000,
        3_865_000,
        3_870_000,
        4_870_000,
    )
    assert delayed_ir.masks == (
        513,
        2568,
        512,
        2568,
        2568,
        512,
        2568,
        2568,
        512,
        2568,
        512,
        2568,
        0,
    )
    assert delayed_ir.loop_start_index == 0
    assert delayed_ir.loop_end_tick == 4_870_000
    assert delayed_ir.loop_count == 1
    assert [
        (index, delay)
        for index, delay in enumerate(delayed_ir.channel_delays)
        if delay
    ] == [(3, 2), (9, 2), (11, 2)]


def test_native_compiler_folds_digital_and_dac_delays_in_one_schedule():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    active_bus_keys = {
        step.port for period in document.periods for step in period.analog_steps
    }
    bus = next(
        port
        for port in document.target.ports
        if port.kind == PORT_DAC and port.key in active_bus_keys
    )
    digital = next(
        port
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )
    delayed = replace(
        document,
        delays=(
            OutputDelay(bus.key, 40, "ns"),
            OutputDelay(digital.key, 100, "ns"),
        ),
    )

    actual = compile_pulse_document(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )

    assert any(actual.channel_delays)
    assert any(delay.delay_ticks for delay in actual.bus_delays)


def test_scan_duration_includes_every_compact_inner_repeat_iteration():
    base = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    parameter = ScanParameter(
        "gap_duration",
        PulseFieldRef(FIELD_DURATION, "p3"),
        "gap duration",
        "s",
    )
    document = replace(
        base,
        scan_parameters=(parameter,),
        repeat=RepeatRegion("p2", "p4", 3),
    )
    table, _report = freeze_scan_table(
        document,
        (parameter.parameter_id,),
        ((0.0001,), (0.0002,)),
    )
    document = replace(document, scan_table=table)

    actual = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )

    assert actual.loop_count == 3
    assert actual.scan_point_durations == pytest.approx((0.0974, 0.0977))
    assert actual.duration_seconds == pytest.approx(0.1951)


def test_scan_rejects_dac_segments_inside_compact_inner_repeat():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    unsupported = replace(document, repeat=RepeatRegion("p2", "p2", 2))

    with pytest.raises(ValueError, match="DAC bus segments.*inner repeat"):
        compile_pulse_document(
            unsupported,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        )


def test_artifact_uses_stable_compiler_identity_and_one_geometry_for_ir_and_wire():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
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
    assert artifact.target_ir.scan_coeff_frac_bits == 8
    assert artifact.wire_image.source_ir_digest == artifact.target_ir.fingerprint


def test_compiler_rejects_scan_loss_and_clock_grid_drift():
    scan = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    with pytest.raises(ValueError, match="silently ignore"):
        compile_pulse_document(
            scan,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.STATIC_ONCE,
        )
    with pytest.raises(ValueError, match="clock grid differs"):
        compile_pulse_document(
            scan,
            clock_hz=40e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        )
    assert not hasattr(scan, "scan_repeats")
    assert not hasattr(scan, "repeat_forever")
