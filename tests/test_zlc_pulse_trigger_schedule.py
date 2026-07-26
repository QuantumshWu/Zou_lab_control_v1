"""Compiled trigger schedules match the cycle model and preserve point provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpga.pulse_streamer.host.engine_model import reference_play
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    TargetIR,
    build_digital_trigger_schedules,
    compile_pulse_document,
    freeze_scan_table,
    load_pulse_document,
    pack_target_ir,
)


ROOT = Path(__file__).parents[1]
IMAGING_PULSE = ROOT / "pulses" / "imaging_template.json"


def _compact(document):
    return replace(
        document,
        periods=tuple(
            replace(period, duration=20, unit="ns") for period in document.periods
        ),
    )


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
    ("path", "form", "channel"),
    [
        (IMAGING_PULSE, PulseExecutionForm.STATIC_ONCE, "ch11"),
        (ROOT / "pulses" / "mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "ch06"),
        (ROOT / "pulses" / "release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "ch11"),
    ],
)
def test_schedule_rising_ticks_equal_cycle_accurate_model(path, form, channel):
    document = _compact(load_pulse_document(path))
    ir = compile_pulse_document(document, clock_hz=50e6, execution_form=form)
    schedule = build_digital_trigger_schedules(ir, (channel,))[0]
    assert [edge.tick_from_run_start for edge in schedule.iter_edges()] == _model_rising_ticks(ir, channel)


def test_scan_schedule_keeps_point_identity_before_output_delay():
    document = _compact(
        load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    )
    table, _report = freeze_scan_table(
        document,
        ("da_x", "da_y", "da_z"),
        ((0.0, 0.0, 0.0), (20.0, -10.0, 5.0), (-30.0, 15.0, 10.0)),
    )
    document = replace(document, scan_table=table)
    ir = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    trigger_channel = "ch06"
    bit = ir.channels.index(trigger_channel)
    delayed = replace(
        ir,
        channel_delays=tuple(100 if index == bit else value for index, value in enumerate(ir.channel_delays)),
    )
    schedule = build_digital_trigger_schedules(delayed, (trigger_channel,))[0]
    edges = tuple(schedule.iter_edges())
    assert schedule.point_count == len(ir.scan_points)
    assert [edge.point_index for edge in edges] == list(range(len(ir.scan_points)))
    assert all(edge.point_trigger_ordinal == 0 for edge in edges)
    assert [edge.tick_from_run_start for edge in edges] == _model_rising_ticks(
        delayed, trigger_channel
    )


def test_cyclic_and_clock_mux_channels_have_no_finite_schedule():
    document = _compact(load_pulse_document(IMAGING_PULSE))
    cyclic = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    with pytest.raises(ValueError, match="cyclic"):
        build_digital_trigger_schedules(cyclic, ("ch11",))
    assert build_digital_trigger_schedules(cyclic, ()) == ()


def _large_compact_ir(*, loop_count, rising_each_loop):
    if rising_each_loop:
        ticks = (0, 1, 3)
        masks = (1, 0, 0)
        loop_end = 2
    else:
        ticks = (0, 10)
        masks = (0, 0)
        loop_end = 10
    wall_ticks = ticks[-1] + (loop_count - 1) * loop_end
    return TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="8" * 64,
        channels=("trigger",),
        ticks=ticks,
        masks=masks,
        duration_seconds=wall_ticks / 50e6,
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=loop_end,
        loop_count=loop_count,
        tick_slot_coeffs=tuple(() for _ in ticks),
        channel_delays=(0,),
    )


def test_compact_trigger_projection_does_not_expand_a_loop_without_rises():
    no_rises = _large_compact_ir(
        loop_count=(1 << 32) - 1,
        rising_each_loop=False,
    )
    schedule = build_digital_trigger_schedules(no_rises, ("trigger",))[0]
    assert tuple(schedule.iter_edges()) == ()

    artifact = CompiledPulseArtifact(
        "9" * 64,
        "test-compiler",
        PulseExecutionForm.STATIC_ONCE,
        no_rises,
        pack_target_ir(no_rises),
        (),
    )
    assert artifact.trigger_schedules == ()
