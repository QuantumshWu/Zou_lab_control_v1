"""PulseTarget owns the output ABI independently of neutral-atom code."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy

import pytest
import zlc_pulse.manifest as pulse_manifest_module
import zlc_pulse.target as pulse_target_module

from zlc_pulse import (
    DestructivePulseTargetEditError,
    build_pulse_target_manifest,
    new_pulse_document,
    pulse_target_manifest_from_lanes,
    pulse_target_manifest_from_tree,
    pulse_target_manifest_to_tree,
    pulse_target_port_drafts,
    replace_pulse_document_target,
    set_digital_output,
)

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


def test_operator_rename_preserves_internal_reference_and_endpoint_binding():
    value = load_deployed_pulse_target()
    document = set_digital_output(
        new_pulse_document(value, time_step_ns=20),
        "p1",
        "ch00",
        True,
    )
    source_manifest = pulse_target_manifest_from_lanes(value)
    drafts = list(pulse_target_port_drafts(source_manifest))
    drafts[0] = replace(drafts[0], signal="renamed_ttl")

    result = replace_pulse_document_target(
        document,
        build_pulse_target_manifest(tuple(drafts)),
    )

    assert result.document.target.by_key["ch00"].label == "renamed_ttl"
    assert result.document.target.abi_fingerprint == value.abi_fingerprint
    assert result.document.periods == document.periods
    assert result.manifest.by_key["ch00"].endpoints == ("ch00",)
    assert not result.impact.destructive


def test_manifest_codec_and_fingerprint_include_backend_endpoint_facts():
    manifest = pulse_target_manifest_from_lanes(load_deployed_pulse_target())
    tree = pulse_target_manifest_to_tree(manifest)
    assert pulse_target_manifest_from_tree(tree) == manifest

    rebound_tree = deepcopy(tree)
    rebound_tree["ports"][0]["endpoints"] = ["F15"]
    rebound = pulse_target_manifest_from_tree(rebound_tree)
    assert rebound.target.abi_fingerprint == manifest.target.abi_fingerprint
    assert rebound.fingerprint != manifest.fingerprint


def test_manifest_fingerprint_is_bound_once_at_construction(monkeypatch):
    manifest = pulse_target_manifest_from_lanes(load_deployed_pulse_target())
    expected = manifest.fingerprint

    def unexpected_digest(*_args, **_kwargs):
        raise AssertionError("manifest getter recomputed its canonical digest")

    monkeypatch.setattr(pulse_manifest_module, "canonical_digest", unexpected_digest)
    assert manifest.fingerprint == expected
    assert manifest.fingerprint == expected


def test_used_port_removal_is_explicitly_destructive_and_cascade_clears_it():
    value = load_deployed_pulse_target()
    document = set_digital_output(
        new_pulse_document(value, time_step_ns=20),
        "p1",
        "ch00",
        True,
    )
    drafts = tuple(
        row
        for row in pulse_target_port_drafts(pulse_target_manifest_from_lanes(value))
        if row.key != "ch00"
    )
    candidate = build_pulse_target_manifest(drafts)

    with pytest.raises(DestructivePulseTargetEditError) as rejected:
        replace_pulse_document_target(document, candidate)
    assert rejected.value.impact.cleared_references == (
        "period p1: cooling is high",
    )

    result = replace_pulse_document_target(document, candidate, cascade=True)
    assert "ch00" not in result.document.target.by_key
    assert not any(result.document.periods[0].states)
    assert result.impact == rejected.value.impact


def test_target_abi_fingerprint_is_bound_once_at_construction(monkeypatch):
    value = target()
    expected = value.abi_fingerprint

    def unexpected_digest(*_args, **_kwargs):
        raise AssertionError("ABI getter recomputed its canonical digest")

    monkeypatch.setattr(pulse_target_module, "canonical_digest", unexpected_digest)
    assert value.abi_fingerprint == expected
    assert value.abi_fingerprint == expected


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
