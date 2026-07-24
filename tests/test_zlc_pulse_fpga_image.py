"""Pulse-owned TargetIR packs byte-for-byte like the installed host path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpga.pulse_streamer.host.image import (
    StreamerParams,
    default_slot_mul_width,
    pack_program,
)
from zlc_storage import canonical_digest
from zlc_pulse import (
    PulseExecutionForm,
    SLOT_MULTIPLIER_WIDTH,
    TargetBusDelay,
    TargetBusSegment,
    TargetIR,
    compile_pulse_document,
    load_pulse_document,
    pack_target_ir,
    validate_target_ir_for_target,
)


ROOT = Path(__file__).parents[1]


def _document_path(name: str) -> Path:
    if name == "imaging_template.json":
        return ROOT / "zlc_neutral_atom" / "assets" / name
    return ROOT / "pulses" / name


def test_current_validator_pins_the_frozen_rtl_slot_multiplier_width():
    assert SLOT_MULTIPLIER_WIDTH == default_slot_mul_width() == 25


@pytest.mark.parametrize(
    ("name", "form", "expected_physical_words_digest"),
    [
        ("camera_imaging_address_switch.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "5d4984bc20a7e635210903878e3b5c0dacd1f22d0ba0029ab7de0218db5fc946"),
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "929b45c22a81cb7071fbe480bc270d591bbc5fea96d79703d6370bdf1509cb40"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "763f7df74b083fc1cb971fd4b1354bd5bcdd226293c0813c255887fea437d16d"),
        ("probe_template.json", PulseExecutionForm.STATIC_ONCE, "52b9dd8fc0f4bcfb60fd7f22189904bdfcad8ccfcba0e192d8e668b80430d6fa"),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "d9a679a6d6118a817cd213fe908d5b27f51590675638941f3aa5656c7b5c253a"),
    ],
)
def test_target_ir_wire_image_matches_the_frozen_wire_golden(
    name,
    form,
    expected_physical_words_digest,
):
    document = load_pulse_document(_document_path(name))
    params = StreamerParams()
    ir = compile_pulse_document(document, clock_hz=50e6, execution_form=form)
    current = pack_target_ir(ir, params)

    assert canonical_digest([list(item) for item in current.words]) == (
        expected_physical_words_digest
    )


def test_wire_image_is_immutable_and_geometry_bound():
    document = load_pulse_document(ROOT / "pulses" / "probe_template.json")
    ir = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    first = pack_target_ir(ir, StreamerParams())
    second = pack_target_ir(ir, StreamerParams(max_edges=2048))
    assert first.geometry_fingerprint != second.geometry_fingerprint
    assert first.digest != second.digest
    with pytest.raises(TypeError):
        first.words[0] = (0, 0)


def _single_slot_scan(
    *,
    slot_count: int = 1,
    slot_value: int = 1,
    final_base: int = 2,
    final_coefficient: int = 0,
    point_count: int = 1,
    loop_count: int = 1,
) -> TargetIR:
    kinds = ("duration",) * slot_count
    point = (slot_value,) * slot_count
    final_coefficients = (final_coefficient, *([0] * (slot_count - 1)))
    final = final_base + ((final_coefficient * slot_value) >> 8)
    return TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="1" * 64,
        channels=("trigger",),
        ticks=(0, final_base),
        masks=(1, 0),
        duration_seconds=final * loop_count * point_count / 50e6,
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=final_base,
        loop_count=loop_count,
        slot_kinds=kinds,
        loop_end_slot_coeffs=final_coefficients,
        tick_slot_coeffs=((0,) * slot_count, final_coefficients),
        scan_points=(point,) * point_count,
        scan_point_durations=(final * loop_count / 50e6,) * point_count,
        scan_coeff_frac_bits=8,
        channel_delays=(0,),
    )


def test_wire_geometry_gate_rejects_every_silent_integer_truncation_class():
    document = load_pulse_document(ROOT / "pulses" / "probe_template.json")
    static = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    oversized_tick = (1 << 32) + 10
    with pytest.raises(ValueError, match="base tick"):
        pack_target_ir(
            replace(
                static,
                ticks=(*static.ticks[:-1], oversized_tick),
                loop_end_tick=oversized_tick,
                duration_seconds=oversized_tick / static.clock_hz,
            )
        )
    with pytest.raises(ValueError, match="loop_count"):
        pack_target_ir(
            replace(
                static,
                loop_start_index=0,
                loop_end_tick=static.ticks[-1],
                loop_count=(1 << 32) + 1,
                duration_seconds=(static.ticks[-1] * ((1 << 32) + 1))
                / static.clock_hz,
            )
        )
    with pytest.raises(ValueError, match="final mask"):
        pack_target_ir(replace(static, masks=(*static.masks[:-1], 1)))

    with pytest.raises(ValueError, match="scan-BRAM read latency"):
        pack_target_ir(_single_slot_scan(final_base=1, point_count=2))
    pack_target_ir(_single_slot_scan(final_base=1, point_count=1))
    pack_target_ir(_single_slot_scan(final_base=1, point_count=2, loop_count=3))
    with pytest.raises(ValueError, match="RTL multiplier input"):
        pack_target_ir(_single_slot_scan(slot_value=1 << 24))
    with pytest.raises(ValueError, match="coefficient"):
        pack_target_ir(_single_slot_scan(final_coefficient=1 << 15))
    with pytest.raises(ValueError, match="uses 5 scan slots"):
        pack_target_ir(_single_slot_scan(slot_count=5))

    too_many_channels = tuple(f"ch{index}" for index in range(63))
    with pytest.raises(ValueError, match="uses 63 channels"):
        pack_target_ir(
            replace(
                static,
                channels=too_many_channels,
                channel_delays=(0,) * len(too_many_channels),
                logical_digital_outputs=(),
            )
        )


def test_wire_geometry_gate_rejects_bus_row_and_literal_overflow():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    scan = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    zeros = (0,) * scan.slot_count
    segments = tuple(
        TargetBusSegment(
            0,
            scan.bus_names[0],
            tick,
            tick,
            0,
            0,
            "edge",
            0,
            0,
            zeros,
            zeros,
        )
        for tick in range(65)
    )
    with pytest.raises(ValueError, match="more than 64"):
        pack_target_ir(replace(scan, bus_segments=segments))

    # The low-level image serializer is also a public upload boundary.  It
    # must reject the same overflow even when a caller bypasses TargetIR's
    # higher-level geometry validator; otherwise row 64 aliases the next bus.
    carrier = SimpleNamespace(
        ticks=scan.ticks,
        masks=scan.masks,
        slot_count=scan.slot_count,
        tick_slot_coeffs=scan.tick_slot_coeffs,
        scan_points=scan.scan_points,
        bus_segments=segments,
        repeat_forever=scan.repeat_forever,
        repeat_from_index=0,
        loop_start_index=scan.loop_start_index,
        loop_count=scan.loop_count,
        loop_end_tick=scan.loop_end_tick,
        loop_end_slot_coeffs=scan.loop_end_slot_coeffs,
    )
    with pytest.raises(ValueError, match="more than 64"):
        pack_program(carrier)

    segment = scan.bus_segments[0]
    with pytest.raises(ValueError, match="literal value"):
        pack_target_ir(
            replace(
                scan,
                bus_segments=(
                    replace(
                        segment,
                        start_value=1 << 10,
                        stop_value=1 << 10,
                        value_select=0,
                        stop_value_select=0,
                    ),
                    *scan.bus_segments[1:],
                ),
            )
        )


def test_wire_geometry_gate_enforces_frozen_event_fifo_capacities():
    ticks = tuple(range(67))
    dense = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="2" * 64,
        channels=("trigger",),
        ticks=ticks,
        masks=tuple(index % 2 for index in ticks),
        duration_seconds=ticks[-1] / 50e6,
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=ticks[-1],
        loop_count=1,
        tick_slot_coeffs=tuple(() for _ in ticks),
        channel_delays=(65,),
    )
    with pytest.raises(ValueError, match="edges in flight"):
        pack_target_ir(dense)

    zeros: tuple[int, ...] = ()
    segments = tuple(
        TargetBusSegment(
            0,
            "dac",
            tick,
            tick,
            0,
            0,
            "edge",
            0,
            0,
            zeros,
            zeros,
        )
        for tick in range(1, 41)
    )
    cyclic_bus = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="3" * 64,
        channels=("trigger",),
        ticks=(0, 100),
        masks=(0, 0),
        duration_seconds=100 / 50e6,
        repeat_forever=True,
        loop_start_index=0,
        loop_end_tick=100,
        loop_count=1,
        tick_slot_coeffs=((), ()),
        bus_names=("dac",),
        bus_safe_values=(512,),
        bus_segments=segments,
        bus_delays=(
            # A 200-tick physical delay spans multiple 40-descriptor frames.
            TargetBusDelay(0, 200),
        ),
        channel_delays=(0,),
    )
    with pytest.raises(ValueError, match="descriptors in flight"):
        pack_target_ir(cyclic_bus)


def _bus_capacity_ir(
    starts,
    *,
    final_tick,
    delay,
    loop_end=None,
    loop_count=1,
    repeat_forever=False,
):
    end = final_tick if loop_end is None else loop_end
    wall_ticks = final_tick + (loop_count - 1) * end
    return TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="4" * 64,
        channels=("trigger",),
        ticks=(0, final_tick),
        masks=(0, 0),
        duration_seconds=wall_ticks / 50e6,
        repeat_forever=repeat_forever,
        loop_start_index=0,
        loop_end_tick=end,
        loop_count=loop_count,
        tick_slot_coeffs=((), ()),
        bus_names=("dac",),
        bus_safe_values=(512,),
        bus_segments=tuple(
            TargetBusSegment(
                0,
                "dac",
                tick,
                tick,
                0,
                0,
                "edge",
                0,
                0,
                (),
                (),
            )
            for tick in starts
        ),
        bus_delays=(TargetBusDelay(0, delay),),
        channel_delays=(0,),
    )


def test_bus_fifo_uses_the_frozen_rtl_half_open_delay_window_and_done_safe():
    exactly_full = _bus_capacity_ir(range(1, 65), final_tick=65, delay=64)
    pack_target_ir(exactly_full)
    with pytest.raises(ValueError, match="at least 65 descriptors"):
        pack_target_ir(replace(exactly_full, bus_delays=(TargetBusDelay(0, 65),)))


def test_compact_loop_bus_tail_runs_once_and_large_cyclic_tail_never_underbounds():
    # These 33 post-loop rows run only after the final compact iteration; DONE
    # contributes the 34th descriptor, so the 64-entry FIFO is safe.
    pack_target_ir(
        _bus_capacity_ir(
            range(10, 43),
            final_tick=100,
            loop_end=10,
            loop_count=2,
            delay=1000,
        )
    )

    period = 250_002
    safe = _bus_capacity_ir(
        (1,),
        final_tick=2,
        loop_end=1,
        loop_count=250_001,
        repeat_forever=True,
        delay=64 * period,
    )
    pack_target_ir(safe)
    with pytest.raises(ValueError, match="at least 65 descriptors"):
        pack_target_ir(
            replace(
                safe,
                bus_delays=(TargetBusDelay(0, 64 * period + 1),),
            )
        )


def test_channel_fifo_models_real_frame_boundaries_and_point_dependent_loops():
    constant_high = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="5" * 64,
        channels=("trigger",),
        ticks=(0, 10),
        masks=(1, 0),
        duration_seconds=10 / 50e6,
        repeat_forever=True,
        loop_start_index=0,
        loop_end_tick=10,
        loop_count=1,
        tick_slot_coeffs=((), ()),
        channel_delays=(1000,),
    )
    # The final zero row is a boundary sentinel.  A cyclic frame seeds edge0
    # directly, so only FIRE creates a toggle; it is not repeated every frame.
    pack_target_ir(constant_high)

    wrap_toggle = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="c" * 64,
        channels=("trigger",),
        ticks=(0, 5, 10),
        masks=(0, 1, 0),
        duration_seconds=10 / 50e6,
        repeat_forever=True,
        loop_start_index=0,
        loop_end_tick=10,
        loop_count=1,
        tick_slot_coeffs=((), (), ()),
        channel_delays=(320,),
    )
    pack_target_ir(wrap_toggle)
    with pytest.raises(ValueError, match="at least 65 edges"):
        pack_target_ir(replace(wrap_toggle, channel_delays=(321,)))

    points = ((1,),) * 100
    finite_scan = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="6" * 64,
        channels=("trigger",),
        ticks=(0, 10),
        masks=(1, 0),
        duration_seconds=1000 / 50e6,
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=10,
        loop_count=1,
        slot_kinds=("duration",),
        loop_end_slot_coeffs=(0,),
        tick_slot_coeffs=((0,), (0,)),
        scan_points=points,
        scan_point_durations=(10 / 50e6,) * len(points),
        scan_coeff_frac_bits=8,
        channel_delays=(1000,),
    )
    # Point boundaries likewise seed the next edge0 directly: one startup
    # toggle and one finite-DONE toggle, never a fabricated 1->0->1 seam.
    pack_target_ir(finite_scan)

    ticks = (0, *range(150, 182), 220)
    masks = (0, *(index % 2 for index in range(1, 33)), 0)
    scan_points = ((1,), (101,))
    durations = tuple(
        (220 + 199 * (99 + point[0])) / 50e6 for point in scan_points
    )
    point_dependent = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="7" * 64,
        channels=("trigger",),
        ticks=ticks,
        masks=masks,
        duration_seconds=sum(durations),
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=99,
        loop_count=200,
        slot_kinds=("duration",),
        loop_end_slot_coeffs=(256,),
        tick_slot_coeffs=tuple((0,) for _ in ticks),
        scan_points=scan_points,
        scan_point_durations=durations,
        scan_coeff_frac_bits=8,
        channel_delays=(450,),
    )
    with pytest.raises(ValueError, match="at least 96 edges"):
        pack_target_ir(point_dependent)


def test_target_gate_rejects_ir_that_lies_under_a_valid_abi_fingerprint():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    ir = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    target = document.target

    with pytest.raises(ValueError, match="logical digital outputs"):
        validate_target_ir_for_target(
            replace(
                ir,
                logical_digital_outputs=ir.logical_digital_outputs[:-1],
            ),
            target,
        )
    forged_safe_values = (
        ir.bus_safe_values[0] + 1,
        *ir.bus_safe_values[1:],
    )
    with pytest.raises(ValueError, match="DAC safe values"):
        validate_target_ir_for_target(
            replace(ir, bus_safe_values=forged_safe_values),
            target,
        )

    with pytest.raises(ValueError, match="channel order"):
        validate_target_ir_for_target(
            replace(
                ir,
                channels=tuple(reversed(ir.channels)),
                logical_digital_outputs=(),
            ),
            target,
        )
    clock_document = load_pulse_document(
        ROOT / "pulses" / "camera_imaging_address_switch.json"
    )
    clock_ir = compile_pulse_document(
        clock_document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_REFERENCE_POINT,
    )
    with pytest.raises(ValueError, match="clock-enable"):
        validate_target_ir_for_target(
            replace(clock_ir, clk_enable=0),
            clock_document.target,
        )

    dac_lane = next(
        port.lanes[0] for port in target.ports if port.kind == "dac"
    )
    dac_bit = 1 << target.raw_lanes.index(dac_lane)
    masks = list(ir.masks)
    masks[0] |= dac_bit
    with pytest.raises(ValueError, match="non-digital"):
        validate_target_ir_for_target(replace(ir, masks=tuple(masks)), target)

    first = ir.bus_segments[0]
    too_wide = replace(
        first,
        start_value=1 << 10,
        stop_value=1 << 10,
        value_select=0,
        stop_value_select=0,
    )
    with pytest.raises(ValueError, match="exceeds target port"):
        validate_target_ir_for_target(
            replace(ir, bus_segments=(too_wide, *ir.bus_segments[1:])),
            target,
        )

    selector = first.stop_value_select or first.value_select
    assert selector
    points = [list(point) for point in ir.scan_points]
    points[0][selector - 1] = 1 << 10
    with pytest.raises(ValueError, match="DAC scan slot exceeds"):
        validate_target_ir_for_target(
            replace(ir, scan_points=tuple(tuple(point) for point in points)),
            target,
        )
