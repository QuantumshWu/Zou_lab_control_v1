"""Compiled trigger schedules match the cycle model and preserve point provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpga.pulse_streamer.host.engine_model import reference_play
from zlc_pulse import (
    PulseExecutionForm,
    build_digital_trigger_schedules,
    load_pulse_document,
)
from zlc_workbench.pulse_compile_bridge import compile_pulse_document


ROOT = Path(__file__).parents[1]


def _model_rising_ticks(ir, channel):
    carrier = SimpleNamespace(
        ticks=ir.ticks,
        masks=ir.masks,
        tick_slot_coeffs=ir.tick_slot_coeffs,
        scan_points=ir.scan_points,
        slot_count=len(ir.slot_kinds),
        scan_coeff_frac_bits=ir.scan_coeff_frac_bits,
        loop_start_index=ir.loop_start_index,
        loop_end_tick=ir.loop_end_tick,
        loop_end_slot_coeffs=ir.loop_end_slot_coeffs,
        loop_count=ir.loop_count,
        repeat_forever=False,
        repeat_from_index=0,
        channel_delays=ir.channel_delays,
    )
    point_spans = []
    for point in ir.scan_points or ((),):
        def effective(base, coeffs):
            return base + (
                sum(coefficient * value for coefficient, value in zip(coeffs, point))
                >> ir.scan_coeff_frac_bits
            )
        final = effective(ir.ticks[-1], ir.tick_slot_coeffs[-1])
        start = effective(ir.ticks[ir.loop_start_index], ir.tick_slot_coeffs[ir.loop_start_index])
        end = effective(ir.loop_end_tick, ir.loop_end_slot_coeffs)
        point_spans.append(final + (ir.loop_count - 1) * (end - start))
    duration = sum(point_spans) + max(ir.channel_delays) + 4
    values = reference_play(carrier, duration)
    bit = ir.channels.index(channel)
    previous = 0
    out = []
    for tick, mask in enumerate(values):
        current = (mask >> bit) & 1
        if current and not previous:
            out.append(tick)
        previous = current
    return out


@pytest.mark.parametrize(
    ("name", "form", "channel"),
    [
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "emCCD"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "emCCD"),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "emCCD"),
    ],
)
def test_schedule_rising_ticks_equal_cycle_accurate_model(name, form, channel):
    document = load_pulse_document(ROOT / "pulses" / name)
    ir = compile_pulse_document(document, clock_hz=1e6, execution_form=form)
    schedule = build_digital_trigger_schedules(ir, (channel,))[0]
    assert [edge.tick_from_run_start for edge in schedule.edges] == _model_rising_ticks(ir, channel)


def test_scan_schedule_keeps_point_identity_before_output_delay():
    document = replace(
        load_pulse_document(ROOT / "pulses" / "mot_field_template.json"),
        scan_table=((0.0, 0.0, 0.0), (20.0, -10.0, 5.0), (-30.0, 15.0, 10.0)),
    )
    ir = compile_pulse_document(
        document,
        clock_hz=1e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    trigger_channel = "mot_trigger"
    bit = ir.channels.index(trigger_channel)
    delayed = replace(
        ir,
        channel_delays=tuple(100 if index == bit else value for index, value in enumerate(ir.channel_delays)),
    )
    schedule = build_digital_trigger_schedules(delayed, (trigger_channel,))[0]
    assert schedule.point_count == len(ir.scan_points)
    assert [edge.point_index for edge in schedule.edges] == list(range(len(ir.scan_points)))
    assert all(edge.point_trigger_ordinal == 0 for edge in schedule.edges)
    assert [edge.tick_from_run_start for edge in schedule.edges] == _model_rising_ticks(
        delayed, trigger_channel
    )


def test_cyclic_and_clock_mux_channels_have_no_finite_schedule():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    cyclic = compile_pulse_document(
        document,
        clock_hz=1e6,
        execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    with pytest.raises(ValueError, match="cyclic"):
        build_digital_trigger_schedules(cyclic, ("emCCD",))
