"""Current PulseDocument compiles through one isolated migration bridge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    PulseExecutionForm,
    load_pulse_document,
    target_ir_from_tree,
    target_ir_to_tree,
)
from zlc_workbench.pulse_compile_bridge import compile_pulse_document


ROOT = Path(__file__).parents[1]


def test_static_current_document_compiles_to_target_ir():
    document = load_pulse_document(ROOT / "pulses" / "probe_template.json")
    ir = compile_pulse_document(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        live_target=document.target,
    )
    assert ir.channels == document.target.raw_lanes
    assert ir.target_abi_fingerprint == document.target.abi_fingerprint
    assert not ir.repeat_forever and not ir.scan_enabled
    assert ir.ticks[0] == 0
    assert target_ir_from_tree(target_ir_to_tree(ir)) == ir


def test_mot_scan_document_compiles_as_finite_autonomous_ir():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    ir = compile_pulse_document(
        document,
        clock_hz=100e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )
    assert ir.scan_enabled
    assert ir.slot_kinds == ("dac", "dac", "dac")
    assert len(ir.scan_points) == len(document.scan_table)
    assert not ir.repeat_forever
    assert ir.scan_repeats == 0


@pytest.mark.parametrize(
    "name",
    [
        "camera_imaging_address_switch.json",
        "imaging_template.json",
        "mot_field_template.json",
        "probe_template.json",
        "pulse_test.json",
        "release_recapture.json",
        "T.json",
    ],
)
def test_every_shipped_authoring_document_has_an_explicit_compile_form(name):
    document = load_pulse_document(ROOT / "pulses" / name)
    if document.scan_slots and document.scan_table:
        form = PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
    elif document.scan_slots:
        form = PulseExecutionForm.STATIC_REFERENCE_POINT
    else:
        form = PulseExecutionForm.STATIC_ONCE
    ir = compile_pulse_document(
        document,
        clock_hz=100e6,
        execution_form=form,
    )
    assert ir.target_abi_fingerprint == document.target.abi_fingerprint


def test_compile_rejects_live_target_drift_and_scan_ignored_by_static_mode():
    scan = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    other = load_pulse_document(ROOT / "pulses" / "probe_template.json")
    with pytest.raises(ValueError, match="target ABI"):
        compile_pulse_document(
            scan,
            clock_hz=100e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            live_target=other.target,
        )
    with pytest.raises(ValueError, match="cannot silently ignore"):
        compile_pulse_document(
            scan,
            clock_hz=100e6,
            execution_form=PulseExecutionForm.STATIC_ONCE,
        )


def test_formal_scan_rejects_legacy_cursor_wrap_repeats():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    with pytest.raises(ValueError, match="cursor-wrap"):
        compile_pulse_document(
            replace(document, scan_repeats=2),
            clock_hz=100e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        )
