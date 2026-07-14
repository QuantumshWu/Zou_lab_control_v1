"""The new notebook API stays short without exposing the raw hardware graph."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import Zou_lab_control.notebook as zlc
import Zou_lab_control.notebook.facade as facade_impl
from Zou_lab_control.neutral_atom.devices.base import BaseDevice
from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.device_catalog import DeviceRef
from zlc_data import FitNumericPolicy, FitResultArtifactRef, SPATIAL_X, SPATIAL_Y
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureFitResultRepository,
    CaptureRepository,
)
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_storage import RepositoryRootBusy
from zlc_neutral_atom.runtime import BoundDevice, RunPlan


ROOT = Path(__file__).parents[1]


def test_virtual_connect_capture_load_is_a_short_current_api(tmp_path):
    exp = zlc.connect("virtual", repository=tmp_path / "workspace")
    try:
        request = exp.readout.capture_request(
            ROOT / "pulses" / "imaging_template.json"
        )
        assert request.camera_ref == exp.device_catalog["camera"].ref
        assert request.sequencer_ref == exp.device_catalog["sequencer"].ref
        descriptor = exp.inspect(request)
        assert descriptor.camera_role == "camera"
        assert descriptor.sequencer_role == "sequencer"
        assert descriptor.trigger_channel == "ch11"
        assert descriptor.expected_frames == 3
        assert descriptor.output_shape == (1, 3, 96, 128)
        assert descriptor.resource_claims == ("device/sequencer", "device/camera")
        assert descriptor.estimated_peak_bytes < request.pipeline_memory_limit_bytes

        reference = exp.run(request)
        assert isinstance(reference, CaptureArtifactRef)
        artifact = exp.readout.load_capture(reference)
        assert artifact.block.values.shape == descriptor.output_shape
        assert artifact.pulse_lineage is not None
        assert artifact.pulse_lineage.compiled_artifact_digest == descriptor.compiled_pulse_digest
        assert artifact.pulse_lineage.expected_trigger_count == 3
        assert artifact.source_cell_schedule == artifact.pulse_lineage.cell_plan.expected_cells
        assert tuple(
            setting.event_index
            for setting in artifact.camera_provenance.descriptor.event_settings
        ) == (0, 1, 2)
        assert not hasattr(artifact.camera_provenance, "readout_event_index")
        assert not hasattr(artifact.camera_provenance, "frame_contract")
    finally:
        exp.close()
        exp.close()


def test_readout_capture_convenience_uses_the_same_authoritative_path(tmp_path):
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        reference = exp.readout.capture(
            ROOT / "pulses" / "imaging_template.json"
        )
        assert exp.readout.load_capture(reference).pulse_lineage is not None


def test_capture_fit_save_load_is_short_headless_and_preserves_named_batch_axes(
    tmp_path,
):
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        capture_ref = exp.readout.capture(
            ROOT / "pulses" / "imaging_template.json"
        )
        execution = exp.fit(
            capture_ref,
            model="radial_gaussian_center",
            numeric_policy=FitNumericPolicy(
                max_evaluations=500,
                max_seconds_per_batch=1.0,
                max_total_seconds=5.0,
                sample_budget_per_batch=512,
                max_packed_observations=4_096,
            ),
        )
        assert tuple(axis.role for axis in execution.result.fit_axis_specs) == (
            SPATIAL_X,
            SPATIAL_Y,
        )
        assert execution.result.spec.batch_axis_ids == tuple(
            axis.axis_id for axis in execution.result.batch_axis_specs
        )
        assert len(execution.result.batch_axis_specs) == 2

        fit_ref = execution.save()
        assert isinstance(fit_ref, FitResultArtifactRef)
        admitted = exp.load_fit(fit_ref)
        assert admitted.reference == fit_ref
        assert admitted.source_capture_ref == capture_ref
        assert admitted.result.digest == execution.result.digest


def test_public_experiment_graph_contains_no_drive_capability(tmp_path):
    exp = zlc.connect("virtual", repository=tmp_path / "workspace")
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

    with pytest.raises(TypeError, match="workspace root"):
        zlc.connect(
            "virtual",
            repository=CaptureRepository,
        )


def test_camera_wiring_mismatch_fails_before_start(tmp_path):
    exp = zlc.connect("virtual", repository=tmp_path / "workspace")
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


def test_capture_request_rejects_wrong_domain_and_stale_device_generation(tmp_path):
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        with pytest.raises(ValueError, match="not 'camera'"):
            exp.readout.capture_request(
                ROOT / "pulses" / "imaging_template.json",
                camera_role="sequencer",
            )

        request = exp.readout.capture_request(
            ROOT / "pulses" / "imaging_template.json"
        )
        stale = DeviceRef(
            request.camera_ref.installation_id,
            request.camera_ref.installation_generation + 1,
            request.camera_ref.role,
        )
        with pytest.raises(RuntimeError, match="stale installation generation"):
            exp.inspect(replace(request, camera_ref=stale))


def test_experiment_owns_its_active_repositories_until_close(tmp_path):
    root = tmp_path / "experiment-workspace"
    exp = zlc.connect("virtual", repository=root)
    try:
        with pytest.raises(RepositoryRootBusy, match="live owner"):
            CaptureRepository(root / "captures")
        with pytest.raises(RepositoryRootBusy, match="live owner"):
            CaptureFitResultRepository(root / "fits")
    finally:
        exp.close()

    CaptureRepository(root / "captures").close()
    CaptureFitResultRepository(root / "fits").close()


def test_failed_composition_releases_all_repository_roots(tmp_path):
    root = tmp_path / "failed-workspace"
    with pytest.raises(TypeError, match="string indices"):
        zlc.connect(
            {"schema": "not-a-device-config"},
            repository=root,
        )
    CaptureRepository(root / "captures").close()
    CaptureFitResultRepository(root / "fits").close()


def test_failed_composition_reports_runtime_that_did_not_shutdown(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "failed-runtime-workspace"

    class _RuntimeThatWillNotShutdown:
        def __init__(self, _devices, **_kwargs):
            self.asset_map = SimpleNamespace(revision="a" * 64)

        def shutdown(self, *, timeout):
            assert timeout == 2.0
            return False

    monkeypatch.setattr(
        facade_impl,
        "LegacyNeutralAtomRuntime",
        _RuntimeThatWillNotShutdown,
    )

    def fail_catalog(*_args, **_kwargs):
        raise ValueError("catalog construction failed")

    monkeypatch.setattr(facade_impl, "_catalog_from_device_set", fail_catalog)
    with pytest.raises(RuntimeError, match="cleanup deadline") as caught:
        zlc.connect("virtual", repository=root)
    assert isinstance(caught.value.__cause__, ValueError)
    CaptureRepository(root / "captures").close()
    CaptureFitResultRepository(root / "fits").close()


def test_readout_binding_view_is_typed_without_session_calibration_state(tmp_path):
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        bound = exp.readout.for_binding(ReadoutBindingKey("camera"))
        assert not hasattr(bound, "current_calibration_ref")
        with pytest.raises(ValueError, match="cannot switch"):
            bound.for_binding("sequencer")
