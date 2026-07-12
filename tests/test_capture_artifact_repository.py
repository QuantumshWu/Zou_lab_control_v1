"""Exact camera datasets become durable current-schema CaptureArtifacts."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from zlc_data import AxisId, AxisSpec, BlockId, PointLayout, REPEAT, SCAN_POINT
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.runtime import (
    DatasetCellAddress,
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    RunFailed,
)
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


def _axis(name, role, size):
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _runtime_and_spec():
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(6, 8), seed=4),
        exposure=1e-3,
    )
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera},
            {"readout": {"type": "VirtualCamera", "params": {}}},
        )
    )
    measurement = runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            _axis("repeat", REPEAT, 1),
            (_axis("point", SCAN_POINT, 2),),
            PointLayout.rect_c((2,)),
            (DatasetCellAddress(0, 0), DatasetCellAddress(0, 1)),
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            4 << 20,
        )
    )
    spec = MinimalPipelineSpec(
        "persist exact camera capture",
        measurement,
        DatasetMaterializerSpec(
            BlockId("capture-artifact-test"),
            PipelineMemoryProfile.for_current_runtime(8 << 20),
        ),
    )
    return camera, runtime, spec


def _deliver_when_armed(camera, images):
    failure = []

    def source():
        try:
            deadline = time.monotonic() + 2.0
            state = camera._recent_state()
            with state["cond"]:
                while not state["armed"]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("camera was not armed")
                    state["cond"].wait(remaining)
            camera._deliver(images)
        except BaseException as error:  # surfaced on the test owner below
            failure.append(error)

    thread = threading.Thread(target=source, daemon=False)
    thread.start()
    return thread, failure


def test_exact_pipeline_commits_and_reloads_capture_artifact(tmp_path):
    camera, runtime, spec = _runtime_and_spec()
    repository = CaptureRepository(tmp_path / "captures")
    plan = compile_capture_artifact_pipeline(spec, repository)
    thread, source_failure = _deliver_when_armed(
        camera,
        [
            np.full((6, 8), 17, dtype=np.uint16),
            np.full((6, 8), 29, dtype=np.uint16),
        ],
    )
    try:
        handle = runtime.controller.start(plan)
        reference = handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive() and source_failure == []
        assert isinstance(reference, CaptureArtifactRef)
        assert handle.snapshot().final_committed

        artifact = repository.load(reference)
        assert artifact.ref == reference
        assert artifact.block.values.shape == (1, 2, 6, 8)
        assert np.all(artifact.block.values[0, 0] == 17)
        assert np.all(artifact.block.values[0, 1] == 29)
        assert [item.source_ordinal for item in artifact.event_metadata] == [0, 1]
        assert artifact.coverage.complete
        assert artifact.terminal.produced_count == 2
        assert artifact.terminal.drained_count == 2
        assert artifact.pulse_lineage is None
        assert artifact.provenance.trace_binding.run_id == handle.snapshot().run_id.value
        assert not hasattr(artifact, "camera")
        assert not hasattr(reference, "repository")

        reopened = CaptureRepository(tmp_path / "captures")
        assert reopened.startup_reconciliations == ()
        reloaded = reopened.load(reference)
        assert np.array_equal(reloaded.block.values, artifact.block.values)
        assert reloaded.event_metadata == artifact.event_metadata
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        assert runtime.shutdown(timeout=2.0)


def test_failed_capture_never_publishes_a_manifest(tmp_path):
    camera, runtime, spec = _runtime_and_spec()
    repository_root = tmp_path / "captures"
    repository = CaptureRepository(repository_root)
    plan = compile_capture_artifact_pipeline(spec, repository)
    # Capability was frozen while binding.  Drift before preflight must fail
    # before arm and therefore before any artifact commit authority is consumed.
    camera.exposure = 2e-3
    try:
        with pytest.raises(RunFailed):
            runtime.controller.start(plan).result(3.0)
        manifest_root = repository_root / "content" / "manifests" / "capture"
        assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_capture_ref_cannot_be_loaded_from_another_repository(tmp_path):
    first = CaptureRepository(tmp_path / "first", repository_id="first")
    second = CaptureRepository(tmp_path / "second", repository_id="second")
    reference = CaptureArtifactRef("first", "1" * 64)
    with pytest.raises(ValueError, match="another repository"):
        second.load(reference)
    assert first.startup_reconciliations == ()
