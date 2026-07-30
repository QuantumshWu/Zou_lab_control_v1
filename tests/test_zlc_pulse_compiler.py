from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import accumulate
from pathlib import Path

import pytest

from zlc_pulse import (
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
from zlc_pulse.compiler import COMPILER_ID


ROOT = Path(__file__).resolve().parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"
_SECONDS_PER_TIME_UNIT = {
    "ns": Fraction(1, 1_000_000_000),
    "us": Fraction(1, 1_000_000),
    "ms": Fraction(1, 1_000),
    "s": Fraction(1, 1),
}


def _time_seconds(value, unit):
    return float(Fraction(str(value)) * _SECONDS_PER_TIME_UNIT[unit])


def _time_ticks(value, unit, clock_hz):
    ticks = (
        Fraction(str(value))
        * _SECONDS_PER_TIME_UNIT[unit]
        * Fraction(str(clock_hz))
    )
    assert ticks.denominator == 1
    return ticks.numerator


def _period_ticks(periods, clock_hz):
    return tuple(_time_ticks(period.duration, period.unit, clock_hz) for period in periods)


def _cumulative_ticks(periods, clock_hz):
    return tuple(accumulate(_period_ticks(periods, clock_hz), initial=0))


def _period_mask(period):
    return sum(int(state) << index for index, state in enumerate(period.states))


def test_native_compiler_preserves_compact_and_delay_unrolled_repeat_semantics():
    document = load_pulse_document(IMAGING_TEMPLATE)
    clock_hz = 50e6
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
        clock_hz=clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    delayed_ir = compile_pulse_document(
        delayed,
        clock_hz=clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )

    periods = tuple(document.periods)
    repeat = compact.repeat
    assert repeat is not None
    period_index = {period.period_id: index for index, period in enumerate(periods)}
    repeat_start = period_index[repeat.start_period_id]
    repeat_end = period_index[repeat.end_period_id]
    compact_ticks = _cumulative_ticks(periods, clock_hz)

    assert compact_ir.ticks == compact_ticks
    assert compact_ir.masks == tuple(_period_mask(period) for period in periods) + (0,)
    assert compact_ir.loop_start_index == repeat_start
    assert compact_ir.loop_end_tick == compact_ticks[repeat_end + 1]
    assert compact_ir.loop_count == repeat.count
    assert not any(compact_ir.channel_delays)

    repeat_members = periods[repeat_start : repeat_end + 1]
    expanded_periods = (
        periods[:repeat_start]
        + repeat_members * repeat.count
        + periods[repeat_end + 1 :]
    )
    expanded_ticks = _cumulative_ticks(expanded_periods, clock_hz)
    assert delayed_ir.ticks == expanded_ticks
    assert delayed_ir.masks == tuple(
        _period_mask(period) for period in expanded_periods
    ) + (0,)
    assert delayed_ir.loop_start_index == 0
    assert delayed_ir.loop_end_tick == expanded_ticks[-1]
    assert delayed_ir.loop_count == 1
    active_lane_indices = tuple(
        index
        for index in range(len(document.target.raw_lanes))
        if any(period.states[index] for period in periods)
    )
    delayed_lane_index = document.target.raw_lanes.index(active_port.lanes[0])
    normalized_delay_ticks = _time_ticks(40, "ns", clock_hz)
    assert [
        (index, delay)
        for index, delay in enumerate(delayed_ir.channel_delays)
        if delay
    ] == [
        (index, normalized_delay_ticks)
        for index in active_lane_indices
        if index != delayed_lane_index
    ]


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
    base = load_pulse_document(IMAGING_TEMPLATE)
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
    scan_rows = ((0.0001,), (0.0002,))
    table, _report = freeze_scan_table(
        document,
        (parameter.parameter_id,),
        scan_rows,
    )
    document = replace(document, scan_table=table)

    actual = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )

    repeat = document.repeat
    assert repeat is not None
    period_index = {
        period.period_id: index for index, period in enumerate(document.periods)
    }
    repeat_start = period_index[repeat.start_period_id]
    repeat_end = period_index[repeat.end_period_id]
    scanned_period = period_index[parameter.field.period_id]
    base_durations = [
        _time_seconds(period.duration, period.unit) for period in document.periods
    ]
    expected_point_durations = []
    for (scanned_duration,) in table.rows:
        point_durations = list(base_durations)
        point_durations[scanned_period] = _time_seconds(
            scanned_duration,
            parameter.unit,
        )
        repeat_duration = sum(point_durations[repeat_start : repeat_end + 1])
        expected_point_durations.append(
            sum(point_durations) + (repeat.count - 1) * repeat_duration
        )

    assert actual.loop_count == repeat.count
    assert actual.scan_point_durations == pytest.approx(expected_point_durations)
    assert actual.duration_seconds == pytest.approx(sum(expected_point_durations))


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
