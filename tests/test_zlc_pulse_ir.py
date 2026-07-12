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
        2e-7,
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
        0,
        ("dac",),
        (
            TargetBusSegment(0, "dac", 5, 15, 2, 3, "ramp", 0, 2, (0, 0), (256, 0)),
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
