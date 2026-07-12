"""Production composition bridge from a raw camera to the exact runtime."""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    PointLayout,
    REPEAT,
    SCAN_POINT,
)
from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    CameraCaptureSpec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.runtime import (
    BoundMeasurement,
    CleanupStepAck,
    DatasetMaterializerSpec,
    DatasetCellAddress,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    MemoryQuarantineJournal,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    ResourceArbiter,
    ResourceKey,
    RunCancelled,
    RunController,
    RunFailed,
    SafeStateAck,
    SafetyOperation,
    compile_pipeline,
)
from zlc_storage import canonical_digest
from zlc_workbench.camera_capture import (
    CameraCaptureBindingRequest,
    CameraCaptureDescription,
    CameraCaptureEndpoint,
    bind_camera_measurement,
)
from zlc_workbench.legacy_runtime import (
    LegacyDeviceRegistration,
    LegacyDeviceRegistry,
)
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _camera() -> VirtualCamera:
    return VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(6, 8), seed=1),
        exposure=1e-3,
    )


def _bound_endpoint(camera: VirtualCamera):
    broker = DeviceBroker()
    endpoint = CameraCaptureEndpoint(camera, "readout", max_inflight_frames=2)
    key = ResourceKey.parse("device/camera/readout")

    def cleanup():
        camera.disarm()
        return CleanupStepAck(SafetyOperation.DISARM, "virtual-camera-disarmed")

    def verify():
        state = camera._recent_state()
        with state["cond"]:
            if state["armed"] or state["pending"]:
                raise RuntimeError("virtual camera is not terminal")
        return SafeStateAck("virtual-camera-safe")

    registry = LegacyDeviceRegistry(broker)
    registry.register(
        LegacyDeviceRegistration(
            camera,
            key,
            lambda: DeviceIdentityAck(
                "virtual-camera:readout",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
                "virtual-camera-connection",
                "test-assets-v1",
            ),
            {SafetyOperation.DISARM: cleanup},
            (SafetyOperation.DISARM,),
            verify,
            target_endpoint=endpoint.target_endpoint,
        )
    )
    binding = registry.binding_for(camera)
    attestation = broker.verify_capability(binding)
    description = endpoint.target_endpoint.describe(binding)
    assert isinstance(description, CameraCaptureDescription)
    capability = attestation.snapshot
    assert capability.driver_ring_bytes == 2 * 6 * 8 * 2
    assert capability.adapter_record_retention_bytes == 16 * 6 * 8 * 2
    return broker, registry, endpoint, binding, attestation, description


def _pipeline(camera: VirtualCamera):
    broker, registry, endpoint, binding, _attestation, _description = (
        _bound_endpoint(camera)
    )
    measurement = bind_camera_measurement(
        types.SimpleNamespace(devices={"readout": camera}),
        registry,
        CameraCaptureBindingRequest(
            "readout",
            _axis("repeat", REPEAT, 1),
            (_axis("point", SCAN_POINT, 2),),
            PointLayout.rect_c((2,)),
            (DatasetCellAddress(0, 0), DatasetCellAddress(0, 1)),
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            4 << 20,
        ),
    )
    spec = MinimalPipelineSpec(
        "camera endpoint integration",
        measurement,
        DatasetMaterializerSpec(
            BlockId("camera-endpoint"),
            PipelineMemoryProfile.for_current_runtime(8 << 20),
        ),
    )
    return broker, endpoint, binding, compile_pipeline(spec)


def test_camera_endpoint_runs_the_real_exact_pipeline_without_raw_device_escape():
    camera = _camera()
    _broker, _endpoint, _binding, plan = _pipeline(camera)
    source_done = threading.Event()

    def physical_source():
        deadline = time.monotonic() + 2.0
        state = camera._recent_state()
        with state["cond"]:
            while not state["armed"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("test camera was never armed")
                state["cond"].wait(remaining)
        camera._deliver(
            [
                np.full((6, 8), 11, dtype=np.uint16),
                np.full((6, 8), 22, dtype=np.uint16),
            ]
        )
        camera.last_sequence = "hardware-updated-observation"
        source_done.set()

    source = threading.Thread(target=physical_source, daemon=False)
    source.start()
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    result = controller.start(plan).result(3.0)
    source.join(2.0)

    assert source_done.is_set() and not source.is_alive()
    assert result.dataset.block.values.shape == (1, 2, 6, 8)
    assert np.all(result.dataset.block.values[0, 0] == 11)
    assert np.all(result.dataset.block.values[0, 1] == 22)
    assert [item.source_ordinal for item in result.dataset.event_metadata] == [0, 1]
    assert [item.produced_count for item in result.dataset.event_metadata] == [1, 2]
    assert result.capture_terminal.produced_count == 2
    assert result.capture_terminal.drained_count == 2

    # Public results contain domain values and immutable device references only.
    assert not hasattr(result, "camera")
    assert not hasattr(result.capture_terminal, "driver")


def test_camera_endpoint_rejects_settings_drift_before_hardware_arm():
    camera = _camera()
    broker, _registry, endpoint, binding, attestation, description = _bound_endpoint(camera)
    camera.exposure = 2e-3
    target = endpoint.target_endpoint
    capability = attestation.snapshot
    frozen = freeze_camera_capture_spec(
        CameraCaptureSpec(
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            1,
            description.settings_fingerprint,
        )
    )
    from zlc_neutral_atom.runtime import PrepareCaptureCommand

    command = PrepareCaptureCommand(
        "session",
        "run",
        "readout",
        frozen.payload,
        frozen.owner_fingerprint,
        frozen.digest,
        capability.capability_fingerprint,
        capability.settings_fingerprint,
        1,
        capability.max_blocking_call_seconds,
    )
    with pytest.raises(RuntimeError, match="settings changed"):
        target.execute_command(binding, command)
    assert camera._recent_state()["armed"] is False
    assert broker.current_binding(binding.key) is binding


def test_target_endpoint_description_contains_no_raw_camera():
    camera = _camera()
    _broker, _registry, endpoint, binding, _attestation, _description = _bound_endpoint(camera)
    description = endpoint.target_endpoint.describe(binding)
    assert isinstance(description, CameraCaptureDescription)
    assert description.source_id == "readout"
    assert description.payload_contract.value_schema.data_shape == (6, 8)
    assert not hasattr(description, "camera")
    assert len(canonical_digest(description.settings_fingerprint)) == 64


def test_installation_runtime_binds_camera_role_without_returning_raw_graph():
    camera = _camera()
    device_set = DeviceSet(
        {"readout": camera},
        {"readout": {"type": "VirtualCamera", "params": {}}},
    )
    runtime = LegacyNeutralAtomRuntime(device_set)
    try:
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
        assert isinstance(measurement, BoundMeasurement)
        assert measurement.capture_contract.dataset_schema.physical_shape == (
            1,
            2,
            6,
            8,
        )
        assert measurement.capture_port.device is runtime.registry.binding_for(camera)
        assert not hasattr(measurement, "camera")
        assert not hasattr(measurement.capture_port.device, "adapter")
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_cancel_interrupts_a_real_endpoint_exact_read_and_releases_owner_lock():
    camera = _camera()
    _broker, _endpoint, _binding, plan = _pipeline(camera)
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    handle = controller.start(plan)
    deadline = time.monotonic() + 2.0
    state = camera._recent_state()
    with state["cond"]:
        while not state["armed"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("camera endpoint never entered its exact read")
            state["cond"].wait(remaining)
    handle.cancel("test exact read cancellation")
    with pytest.raises((RunFailed, RunCancelled)):
        handle.result(3.0)
    with state["cond"]:
        assert not state["armed"]
        assert not state["pending"]
    camera.arm(1, timeout=0.1)
    camera.finish_record_capture()
