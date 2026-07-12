"""The new notebook API stays short without exposing the raw hardware graph."""

from __future__ import annotations

from pathlib import Path

import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.neutral_atom.devices.base import BaseDevice
from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from zlc_neutral_atom.artifacts import CaptureArtifactRef
from zlc_neutral_atom.runtime import BoundDevice, RunPlan


ROOT = Path(__file__).parents[1]


def test_virtual_connect_capture_load_is_a_short_current_api(tmp_path):
    exp = zlc.connect("virtual", repository=tmp_path / "captures")
    try:
        request = exp.readout.capture_request(
            ROOT / "pulses" / "imaging_template.json"
        )
        descriptor = exp.inspect(request)
        assert descriptor.camera_role == "camera"
        assert descriptor.sequencer_role == "sequencer"
        assert descriptor.trigger_channel == "emCCD"
        assert descriptor.expected_frames == 3
        assert descriptor.output_shape == (1, 3, 96, 128)
        assert descriptor.resource_claims == ("device/sequencer", "device/camera")
        assert descriptor.estimated_peak_bytes < request.pipeline_memory_limit_bytes

        reference = exp.run(request)
        assert isinstance(reference, CaptureArtifactRef)
        artifact = exp.readout.load(reference)
        assert artifact.block.values.shape == descriptor.output_shape
        assert artifact.pulse_lineage is not None
        assert artifact.pulse_lineage.compiled_artifact_digest == descriptor.compiled_pulse_digest
        assert artifact.pulse_lineage.expected_trigger_count == 3
    finally:
        exp.close()
        exp.close()


def test_readout_capture_convenience_uses_the_same_authoritative_path(tmp_path):
    with zlc.connect("virtual", repository=tmp_path / "captures") as exp:
        reference = exp.readout.capture(
            ROOT / "pulses" / "imaging_template.json"
        )
        assert exp.readout.load(reference).pulse_lineage is not None


def test_public_experiment_graph_contains_no_drive_capability(tmp_path):
    exp = zlc.connect("virtual", repository=tmp_path / "captures")
    forbidden = (BaseDevice, DeviceSet, BoundDevice, RunPlan)
    try:
        public_values = [
            exp.name,
            exp.device_catalog,
            exp.readout,
            exp.timing,
            exp.timing.target,
        ]
        assert not any(isinstance(value, forbidden) for value in public_values)
        assert not hasattr(exp, "devices")
        assert not hasattr(exp, "camera")
        assert not hasattr(exp, "sequencer")
        assert not hasattr(exp.readout, "camera")
        assert not hasattr(exp.timing, "sequencer")
    finally:
        exp.close()


def test_connect_requires_an_explicit_repository_root():
    with pytest.raises(TypeError):
        zlc.connect("virtual")  # type: ignore[call-arg]


def test_camera_wiring_mismatch_fails_before_start(tmp_path):
    exp = zlc.connect("virtual", repository=tmp_path / "captures")
    try:
        request = exp.readout.capture_request(
            ROOT / "pulses" / "imaging_template.json",
            trigger_channel="mot_trigger",
        )
        with pytest.raises(ValueError, match="not wired"):
            exp.inspect(request)
        assert exp.device_catalog.availability.value == "available"
    finally:
        exp.close()
