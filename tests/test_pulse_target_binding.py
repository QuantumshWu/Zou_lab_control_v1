"""PulseDocument targets are deployment facts; rebinding is physical-only."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import tracked_repo_files

from zlc_pulse import (
    PORT_CLOCK,
    PulseExecutionForm,
    PulseTarget,
    bind_pulse_document_target,
    compile_pulse_artifact,
    load_deployed_pulse_target,
    load_pulse_document,
)
from zlc_pulse.deployment import APPROVED_DEPLOYED_TARGET_ABI


ROOT = Path(__file__).parents[1]
IMAGING_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


def test_every_shipped_document_matches_the_packaged_deployment_target():
    target = load_deployed_pulse_target()
    paths = tracked_repo_files("pulses/*.json")
    assert paths, "the repository must ship at least one pulse document"
    for path in sorted(paths):
        assert load_pulse_document(path).target == target, path.name


@pytest.mark.parametrize(
    ("path", "form", "trigger_channel"),
    [
        (IMAGING_PULSE, PulseExecutionForm.STATIC_ONCE, "ch11"),
        (ROOT / "pulses" / "mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "ch11"),
        (ROOT / "pulses" / "probe_template.json", PulseExecutionForm.STATIC_ONCE, "ch11"),
        (ROOT / "pulses" / "release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "ch11"),
    ],
)
def test_shipped_hardware_documents_embed_the_approved_deployment_target(
    path,
    form,
    trigger_channel,
):
    target = load_deployed_pulse_target()
    document = load_pulse_document(path)

    assert document.target == target
    assert target.abi_fingerprint == APPROVED_DEPLOYED_TARGET_ABI
    assert bind_pulse_document_target(document, target).target is target

    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=form,
        trigger_channels=(trigger_channel,),
        live_target=target,
    )
    assert artifact.target_abi_fingerprint == APPROVED_DEPLOYED_TARGET_ABI


def test_binding_rekeys_referenced_ports_only_by_exact_physical_ownership():
    live_target = load_deployed_pulse_target()
    document = load_pulse_document(IMAGING_PULSE)
    authored_target = PulseTarget(
        live_target.raw_lanes,
        tuple(
            replace(port, key="camera_trigger") if port.key == "ch11" else port
            for port in live_target.ports
        ),
    )
    authored = replace(
        document,
        target=authored_target,
        visible_ports=tuple(
            "camera_trigger" if key == "ch11" else key
            for key in document.visible_ports
        ),
    )

    bound = bind_pulse_document_target(authored, live_target)

    assert bound.target is live_target
    assert bound.visible_ports == document.visible_ports


def test_referenced_lane_cannot_be_guessed_into_a_different_port_kind():
    source = load_pulse_document(IMAGING_PULSE)
    ports = []
    for port in source.target.ports:
        if port.key == "ch11":
            ports.append(replace(port, kind=PORT_CLOCK))
        else:
            ports.append(port)
    incompatible = PulseTarget(source.target.raw_lanes, tuple(ports))

    with pytest.raises(ValueError, match="clock-mux outputs differ"):
        bind_pulse_document_target(source, incompatible)


def test_binding_cannot_inject_an_unreferenced_hardware_clock_output():
    source = load_pulse_document(IMAGING_PULSE)
    injected = PulseTarget(
        source.target.raw_lanes,
        tuple(
            replace(port, kind=PORT_CLOCK)
            if port.key == "ch04"
            else port
            for port in source.target.ports
        ),
    )

    with pytest.raises(ValueError, match="clock-mux outputs differ"):
        bind_pulse_document_target(source, injected)
