"""Production composition bridge from a raw camera to the exact runtime."""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    PointLayout,
    READOUT_EVENT,
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
    StartCaptureCommand,
    compile_pipeline,
)
from zlc_neutral_atom.runtime.capture import (
    camera_capability_evidence_from_tree,
    camera_capability_evidence_to_tree,
    camera_physical_facts_from_tree,
    camera_physical_facts_to_tree,
)
from zlc_storage import canonical_digest
from zlc_workbench.camera_capture import (
    CameraCaptureBindingRequest,
    CameraCaptureDescription,
    CameraCaptureEndpoint,
    _settings_tree,
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
    assert description.physical_facts is capability.camera_physical_facts
    assert description.physical_facts.camera_identity == binding.stable_device_identity
    assert description.physical_facts.output_shape_yx == camera.frame_shape
    assert description.physical_facts.exposure_seconds == camera.exposure
    assert (
        description.physical_facts.required_external_trigger_interval_seconds
        == 0.0
    )
    assert (
        description.physical_facts.external_trigger_integration_start_offset_seconds
        == 0.0
    )
    with pytest.raises(PermissionError):
        CameraCaptureDescription(
            object(),
            description.source_id,
            description.payload_contract,
            description.settings_fingerprint,
            description.physical_facts,
        )
    frame_bytes = int(np.prod(camera.frame_shape)) * 2
    assert capability.driver_ring_bytes == 2 * frame_bytes
    assert capability.adapter_record_retention_bytes == 16 * frame_bytes
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
            PipelineMemoryProfile(8 << 20),
        ),
    )
    return broker, endpoint, binding, compile_pipeline(spec)


def _one_frame_prepare_command(capability, description):
    from zlc_neutral_atom.runtime import PrepareCaptureCommand

    frozen = freeze_camera_capture_spec(
        CameraCaptureSpec(
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            1,
            description.settings_fingerprint,
        )
    )
    return PrepareCaptureCommand(
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
    command = _one_frame_prepare_command(capability, description)
    with pytest.raises(RuntimeError, match="settings changed"):
        target.execute_command(binding, command)
    assert camera._recent_state()["armed"] is False
    assert broker.current_binding(binding.key) is binding


def test_camera_endpoint_revalidates_after_arm_and_disarms_before_fire():
    camera = _camera()
    broker, _registry, endpoint, binding, attestation, description = _bound_endpoint(
        camera
    )
    target = endpoint.target_endpoint
    prepare = _one_frame_prepare_command(attestation.snapshot, description)
    target.execute_command(binding, prepare)
    original_arm = camera.arm

    def arm_then_drift(*args, **kwargs):
        original_arm(*args, **kwargs)
        camera.configure(exposure=2e-3)

    camera.arm = arm_then_drift
    start = StartCaptureCommand(prepare.session_id, 1.0)
    with pytest.raises(RuntimeError, match="between prepare and armed start"):
        target.execute_command(binding, start)

    state = camera._recent_state()
    with state["cond"]:
        assert state["armed"] is False
        assert state["arming"] is False
    assert broker.current_binding(binding.key) is binding


def test_camera_endpoint_rejects_physical_trigger_wiring_drift_before_arm():
    camera = _camera()
    broker, _registry, endpoint, binding, attestation, description = _bound_endpoint(
        camera
    )
    original = description.capture_trigger_channels
    camera.capture_trigger_channels = ("physically-rewired",)
    assert description.capture_trigger_channels == original
    command = _one_frame_prepare_command(attestation.snapshot, description)
    with pytest.raises(RuntimeError, match="settings changed"):
        endpoint.target_endpoint.execute_command(binding, command)
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


def test_capability_physical_facts_use_applied_roi_and_payload_geometry():
    camera = _camera()
    camera.configure(roi=(4, 4, 0, 4))
    _broker, _registry, _endpoint, _binding, attestation, description = (
        _bound_endpoint(camera)
    )
    facts = attestation.snapshot.camera_physical_facts
    assert facts is description.physical_facts
    assert facts.sensor_shape_yx == (6, 8)
    assert facts.roi_origin_yx == (0, 4)
    assert facts.roi_shape_yx == (4, 4)
    assert facts.binning_yx == (1, 1)
    assert facts.output_shape_yx == description.payload_contract.value_schema.data_shape
    y_axis, x_axis = description.payload_contract.value_schema.data_axes
    assert y_axis.name == "ROI-local output y"
    assert x_axis.name == "ROI-local output x"
    assert y_axis.coordinates == tuple(range(4))
    assert x_axis.coordinates == tuple(range(4))
    assert y_axis.coordinate_frame == x_axis.coordinate_frame == facts.coordinate_frame
    assert facts.coordinate_frame.value == "readout.roi-local-output-pixels"
    assert "sensor" not in facts.coordinate_frame.value
    evidence = attestation.snapshot.camera_capability_evidence
    assert evidence is not None
    assert evidence.physical_facts is facts
    assert evidence.fingerprint == attestation.snapshot.capability_fingerprint
    assert evidence.source_id == "readout"


def test_unqualified_pylon_integration_offset_is_not_coerced_to_zero():
    class _IntegerNode:
        def __init__(self, value):
            self._value = value

        def GetValue(self):
            return self._value

    camera = PylonCamera(trigger_source="Line1")
    camera._camera = types.SimpleNamespace(
        HeightMax=_IntegerNode(6),
        WidthMax=_IntegerNode(8),
    )
    try:
        settings = _settings_tree(camera)
        assert settings["external_trigger_integration_start_offset_seconds"] is None
        assert settings["required_external_trigger_interval_seconds"] == 0.0
    finally:
        camera._camera = None


def test_trigger_wiring_has_one_physical_owner_and_current_canonical_round_trip():
    camera = _camera()
    camera.capture_trigger_channels = ("ch11", "ch12")
    _broker, _registry, _endpoint, _binding, attestation, description = (
        _bound_endpoint(camera)
    )
    evidence = attestation.snapshot.camera_capability_evidence
    assert evidence is not None
    facts = evidence.physical_facts

    assert facts.capture_trigger_channels == ("ch11", "ch12")
    assert description.capture_trigger_channels is facts.capture_trigger_channels
    assert "trigger_channels" not in CameraCaptureDescription.__dataclass_fields__
    facts.validate_capture_trigger_channel("ch12")
    with pytest.raises(ValueError, match="not wired"):
        facts.validate_capture_trigger_channel("ch10")
    with pytest.raises(ValueError, match="exactly one"):
        facts.require_single_capture_trigger_channel("ch12")

    facts_tree = camera_physical_facts_to_tree(facts)
    assert facts_tree["schema"] == "zlc_neutral_atom.CameraPhysicalFacts"
    assert facts_tree["capture_trigger_channels"] == ["ch11", "ch12"]
    assert facts_tree["required_external_trigger_interval_seconds"] == 0.0
    assert facts_tree["external_trigger_integration_start_offset_seconds"] == 0.0
    assert camera_physical_facts_from_tree(facts_tree) == facts
    single_tree = dict(facts_tree)
    single_tree["capture_trigger_channels"] = ["ch11"]
    single_facts = camera_physical_facts_from_tree(single_tree)
    single_facts.require_single_capture_trigger_channel("ch11")

    evidence_tree = camera_capability_evidence_to_tree(evidence)
    assert evidence_tree["schema"] == "zlc_neutral_atom.CameraCapabilityEvidence"
    assert camera_capability_evidence_from_tree(evidence_tree) == evidence
    unknown_facts_tree = dict(facts_tree)
    unknown_facts_tree["schema"] = "unsupported-camera-physical-facts"
    with pytest.raises(ValueError, match="expected.*CameraPhysicalFacts"):
        camera_physical_facts_from_tree(unknown_facts_tree)
    unknown_evidence_tree = dict(evidence_tree)
    unknown_evidence_tree["schema"] = "unsupported-camera-capability-evidence"
    with pytest.raises(ValueError, match="expected.*CameraCapabilityEvidence"):
        camera_capability_evidence_from_tree(unknown_evidence_tree)

    empty_tree = dict(facts_tree)
    empty_tree["capture_trigger_channels"] = []
    empty_facts = camera_physical_facts_from_tree(empty_tree)
    assert empty_facts.capture_trigger_channels == ()
    assert camera_physical_facts_from_tree(
        camera_physical_facts_to_tree(empty_facts)
    ) == empty_facts
    with pytest.raises(ValueError, match="not wired"):
        empty_facts.require_single_capture_trigger_channel("ch11")
    duplicate_tree = dict(facts_tree)
    duplicate_tree["capture_trigger_channels"] = ["ch11", "ch11"]
    with pytest.raises(ValueError, match="unique"):
        camera_physical_facts_from_tree(duplicate_tree)


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


def test_multi_event_binding_requires_explicit_attested_event_settings():
    camera = _camera()
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera},
            {"readout": {"type": "VirtualCamera", "params": {}}},
        )
    )
    cells = (DatasetCellAddress(0, 0), DatasetCellAddress(0, 1))
    base = (
        "readout",
        _axis("repeat", REPEAT, 1),
        (_axis("event", READOUT_EVENT, 2),),
        PointLayout.rect_c((2,)),
        cells,
        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        0,
        4 << 20,
    )
    try:
        with pytest.raises(ValueError, match="explicit event_settings"):
            runtime.bind_camera_measurement(CameraCaptureBindingRequest(*base))
        description = runtime.describe_camera("readout")
        measurement = runtime.bind_camera_measurement(
            CameraCaptureBindingRequest(
                *base,
                tuple(description.event_setting(index) for index in range(2)),
            )
        )
        descriptor = measurement.capture_contract.camera_provenance.descriptor
        assert descriptor.event_settings == tuple(
            description.event_setting(index) for index in range(2)
        )
        assert not hasattr(measurement.capture_contract.camera_provenance, "frame_contract")
        assert not hasattr(
            measurement.capture_contract.camera_provenance,
            "readout_event_index",
        )
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
