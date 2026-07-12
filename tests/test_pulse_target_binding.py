"""PulseDocument target binding follows physical ownership, never names or shape."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from zlc_pulse import (
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_pulse.target import pulse_target_from_legacy_tree


ROOT = Path(__file__).parents[1]


def _live_target():
    sequencer = VirtualSequencer(sleep_scale=0)
    return sequencer, pulse_target_from_legacy_tree(sequencer.port_catalog.to_dict())


@pytest.mark.parametrize(
    ("filename", "form", "trigger_channel"),
    [
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "emCCD"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "mot_trigger"),
    ],
)
def test_shipped_authoring_documents_bind_to_the_verified_virtual_target(
    filename,
    form,
    trigger_channel,
):
    sequencer, target = _live_target()
    source = load_pulse_document(ROOT / "pulses" / filename)
    bound = bind_pulse_document_target(source, target)
    assert bound.target is target
    assert bound.target.abi_fingerprint == target.abi_fingerprint
    if filename == "mot_field_template.json":
        assert "mot_trigger" in bound.visible_ports
        assert "camera_trigger" not in bound.visible_ports
    artifact = compile_pulse_artifact(
        bound,
        clock_hz=sequencer.clock_hz,
        execution_form=form,
        trigger_channels=(trigger_channel,),
        live_target=target,
    )
    assert artifact.target_abi_fingerprint == target.abi_fingerprint


def test_referenced_lane_cannot_be_guessed_into_a_different_port_kind():
    _sequencer, target = _live_target()
    source = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    # dx0 is a standalone digital authoring port but belongs to a six-lane DAC
    # on the live installation; singleton shape must not silently reclassify it.
    source = replace(source, visible_ports=("dx0",))
    with pytest.raises(ValueError, match="no unique physically equivalent"):
        bind_pulse_document_target(source, target)
