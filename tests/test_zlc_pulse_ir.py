"""Pulse TargetIR is closed, immutable, and validates affine timing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from zlc_pulse import (
    TargetBusDelay,
    TargetBusSegment,
    TargetIR,
    target_ir_from_tree,
    target_ir_to_tree,
)


DIGEST = "2" * 64


def static_ir():
    return TargetIR(
        100e6,
        DIGEST,
        ("camera", "trap"),
        (0, 10, 20),
        (0, 1, 0),
        2e-7,
        False,
        0,
        20,
        1,
        (),
        (),
        ((), (), ()),
        (),
        (),
        0,
        (),
        (),
        (),
        (2, 0),
        0,
    )


def scan_ir():
    return TargetIR(
        100e6,
        DIGEST,
        ("camera", "d0", "d1"),
        (0, 10, 20),
        (0, 1, 0),
        5.1e-7,
        False,
        0,
        20,
        1,
        ("duration", "dac"),
        (256, 0),
        ((0, 0), (256, 0), (256, 0)),
        ((3, 2), (8, 3)),
        (2.3e-7, 2.8e-7),
        8,
        ("dac",),
        (
            TargetBusSegment(0, "dac", 5, 15, 0, 0, "ramp", 0, 2, (0, 0), (256, 0)),
        ),
        (TargetBusDelay(0, 4),),
        (0, 0, 0),
        0,
    )


@pytest.mark.parametrize("factory", [static_ir, scan_ir])
def test_target_ir_current_tree_round_trips(factory):
    value = factory()
    assert target_ir_from_tree(target_ir_to_tree(value)) == value
    assert len(value.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        value.loop_count = 2


def test_affine_scan_validation_rejects_non_increasing_physical_edges():
    value = scan_ir()
    with pytest.raises(ValueError, match="non-increasing"):
        replace(
            value,
            tick_slot_coeffs=((0, 0), (-1024, 0), (256, 0)),
        )


def test_target_ir_rejects_clk_engine_contention_and_wrong_shapes():
    with pytest.raises(ValueError, match="conflicts"):
        replace(static_ir(), clk_enable=1)
    with pytest.raises(ValueError, match="channel delay vector"):
        replace(static_ir(), channel_delays=(0,))
    with pytest.raises(ValueError, match="coefficient matrix"):
        replace(static_ir(), tick_slot_coeffs=((), ()))
    with pytest.raises(ValueError, match="duration disagrees"):
        replace(scan_ir(), duration_seconds=2e-7)
    with pytest.raises(ValueError, match="invalid loop end"):
        replace(static_ir(), loop_end_tick=30, loop_count=2, duration_seconds=5e-7)
    with pytest.raises(ValueError, match="invalid loop end"):
        replace(static_ir(), loop_end_tick=0)


def test_target_ir_rejects_unsupported_bus_inner_loops_and_dead_scan_repeat_state():
    value = scan_ir()
    with pytest.raises(ValueError, match="DAC bus segments.*inner repeat"):
        replace(value, loop_start_index=1, loop_count=2)

    assert not hasattr(value, "scan_repeats")
    tree = target_ir_to_tree(value)
    tree["scan_repeats"] = 2
    with pytest.raises(ValueError, match="unknown field set"):
        target_ir_from_tree(tree)


def test_bus_segments_bind_names_slots_and_delays_exactly():
    value = scan_ir()
    with pytest.raises(ValueError, match="index/name"):
        replace(
            value,
            bus_segments=(
                replace(value.bus_segments[0], bus_name="other"),
            ),
        )
    with pytest.raises(ValueError, match="selector"):
        replace(
            value,
            bus_segments=(
                replace(value.bus_segments[0], value_select=3),
            ),
        )


def test_bus_segments_cannot_cross_typed_slot_or_physical_time_boundaries():
    value = scan_ir()
    segment = value.bus_segments[0]

    with pytest.raises(ValueError, match="reference a DAC slot"):
        replace(
            value,
            bus_segments=(
                replace(segment, value_select=1, stop_value_select=1),
            ),
        )
    with pytest.raises(ValueError, match="duration slots"):
        replace(
            value,
            tick_slot_coeffs=((0, 0), (256, 1), (256, 0)),
        )
    with pytest.raises(ValueError, match="timing bounds"):
        replace(
            value,
            bus_segments=(
                replace(segment, start_tick=30, stop_tick=40),
            ),
        )
    with pytest.raises(ValueError, match="timing bounds"):
        replace(
            value,
            bus_segments=(
                replace(segment, start_tick=0, start_tick_coeffs=(-256, 0)),
            ),
        )
    with pytest.raises(ValueError, match="canonical zero literal"):
        replace(
            value,
            bus_segments=(replace(segment, stop_value=1),),
        )
    with pytest.raises(ValueError, match="live-state driven"):
        replace(
            value,
            bus_segments=(replace(segment, start_value=1),),
        )

    with pytest.raises(ValueError, match="timing bounds"):
        TargetIR(
            clock_hz=100e6,
            target_abi_fingerprint=DIGEST,
            channels=("camera",),
            ticks=(0, 20),
            masks=(0, 0),
            duration_seconds=2e-7,
            repeat_forever=False,
            loop_start_index=0,
            loop_end_tick=20,
            loop_count=1,
            tick_slot_coeffs=((), ()),
            bus_names=("dac",),
            bus_segments=(
                TargetBusSegment(
                    0,
                    "dac",
                    20,
                    20,
                    0,
                    0,
                    "edge",
                    0,
                    0,
                    (),
                    (),
                ),
            ),
            channel_delays=(0,),
        )

    first = TargetBusSegment(
        0,
        "dac",
        8,
        8,
        2,
        2,
        "edge",
        0,
        0,
        (0, 0),
        (0, 0),
    )
    second = replace(first, start_tick=7, stop_tick=7)
    with pytest.raises(ValueError, match="order overlaps or regresses"):
        replace(value, bus_segments=(first, second))
