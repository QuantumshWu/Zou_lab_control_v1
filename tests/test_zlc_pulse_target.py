"""PulseTarget owns the output ABI independently of neutral-atom code."""

from __future__ import annotations

import pytest

from zlc_pulse.target import (
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    PulsePortSpec,
    PulseTarget,
    load_deployed_pulse_target,
    pulse_target_from_tree,
    pulse_target_to_tree,
)


def target():
    return PulseTarget(
        ("ttl", "d0", "d1", "clk"),
        (
            PulsePortSpec("ttl", PORT_DIGITAL, ("ttl",), "TTL", None, 1, "binary", 0, None),
            PulsePortSpec("dac", PORT_DAC, ("d0", "d1"), "DAC", 0, 2, "offset_binary", 2, "clk"),
            PulsePortSpec("clk", PORT_CLOCK, ("clk",), "Clock", None, 1, "binary", 0, None),
        ),
    )


def test_target_current_tree_round_trips_and_labels_do_not_change_abi():
    value = target()
    assert pulse_target_from_tree(pulse_target_to_tree(value)) == value
    relabeled = PulseTarget(
        value.raw_lanes,
        tuple(
            PulsePortSpec(
                port.key,
                port.kind,
                port.lanes,
                "renamed" if port.key == "ttl" else port.label,
                port.bus_index,
                port.width,
                port.encoding,
                port.safe_value,
                port.latch_clock,
            )
            for port in value.ports
        ),
    )
    assert relabeled.abi_fingerprint == value.abi_fingerprint
    assert len(value.abi_fingerprint) == 64


def test_shipped_current_target_loads_without_neutral_import():
    value = load_deployed_pulse_target()
    assert len(value.raw_lanes) == 62
    assert value.by_key["ch11"].kind == PORT_DIGITAL
    assert value.by_key["ch11"].label == "emCCD"
    assert len(value.abi_fingerprint) == 64


def test_target_rejects_double_owned_lanes_and_bad_latch_clock():
    with pytest.raises(ValueError, match="belongs to both"):
        PulseTarget(
            ("x",),
            (
                PulsePortSpec("a", PORT_DIGITAL, ("x",), "a", None, 1, "binary", 0, None),
                PulsePortSpec("b", PORT_DIGITAL, ("x",), "b", None, 1, "binary", 0, None),
            ),
        )
    with pytest.raises(ValueError, match="missing clock"):
        PulseTarget(
            ("d0", "d1"),
            (
                PulsePortSpec("dac", PORT_DAC, ("d0", "d1"), "dac", 0, 2, "offset_binary", 2, "missing"),
            ),
        )


def test_port_safe_values_are_the_frozen_engine_states():
    with pytest.raises(ValueError, match="low safe state"):
        PulsePortSpec(
            "ttl",
            PORT_DIGITAL,
            ("ttl",),
            "TTL",
            None,
            1,
            "binary",
            1,
            None,
        )
    with pytest.raises(ValueError, match="offset-binary midpoint"):
        PulsePortSpec(
            "dac",
            PORT_DAC,
            ("d0", "d1"),
            "DAC",
            0,
            2,
            "offset_binary",
            0,
            "clk",
        )


def test_current_target_reader_rejects_legacy_shape():
    with pytest.raises(ValueError):
        pulse_target_from_tree({"schema": "Zou_lab_control.neutral_atom.PortCatalog"})
