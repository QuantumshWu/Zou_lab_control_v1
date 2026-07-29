"""Neutral camera-domain values remain multidimensional, immutable, and canonical."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_data.axis import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_data.schema import DatasetSchema, PointColumn, PointTable, ValueSchema
from zlc_data.validity import VALID, ValidityContract
from zlc_data.value import Value
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraCaptureSpec,
    CameraDatasetEventAdapter,
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraFrameRecord,
    CameraSample,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.capture.session import (
    CameraCaptureContract,
    CameraCaptureProvenance,
)
from zlc_neutral_atom.devices.camera.capture_port import CaptureCapabilitySnapshot
from zlc_neutral_atom.devices.simulation.installation import create_virtual_installation
from zlc_neutral_atom.devices.camera.contract import (
    CameraCapabilityEvidence,
    CameraPhysicalFacts,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureDescriptor,
    ReadoutBindingKey,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    FrozenDatasetEdge,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceBindingStamp,
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
)
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.capture.binding import (
    CameraCaptureBindingRequest,
    bind_camera_capture,
)
from zlc_neutral_atom.capture.pipeline import BoundCameraCapture
from zlc_storage import decode, encode


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _schemas() -> tuple[ValueSchema, DatasetSchema]:
    frame = CoordinateFrameId("camera.roi-local")
    value_schema = ValueSchema(
        (
            AxisSpec(
                AxisId("camera.y"),
                "camera.y",
                SPATIAL_Y,
                3,
                tuple(range(3)),
                "pixel",
                frame,
            ),
            AxisSpec(
                AxisId("camera.x"),
                "camera.x",
                SPATIAL_X,
                4,
                tuple(range(4)),
                "pixel",
                frame,
            ),
        ),
        ValidityContract.value(),
        np.dtype("<u2"),
        "count",
    )
    point = _axis("point", SCAN_POINT, 2)
    dataset_schema = DatasetSchema(
        _axis("repeat", REPEAT, 2),
        PointTable(
            point.size,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    point.coordinates,
                ),
            ),
        ),
        None,
        value_schema,
    )
    return value_schema, dataset_schema


def _metadata(**changes) -> CameraFrameMetadata:
    values = dict(
        source_ordinal=0,
        produced_count=4,
        frame_stamp=101,
        camera_stamp=201,
        timestamp_seconds=301,
        timestamp_microseconds=401,
        host_received_at_ns=501,
        driver_buffer_index=0,
        correlation_id="camera-session:0",
    )
    values.update(changes)
    return CameraFrameMetadata(**values)


def _sample(value_schema: ValueSchema, **metadata_changes) -> CameraSample:
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    return CameraSample(
        Value(pixels, VALID, value_schema),
        _metadata(**metadata_changes),
    )


def test_camera_capture_spec_has_one_current_canonical_encoding():
    spec = CameraCaptureSpec(
        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        4,
        (1, 1, 1, 1),
        "a" * 64,
    )
    frozen = freeze_camera_capture_spec(spec)
    assert frozen.owner_fingerprint == CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT
    assert decode_camera_capture_spec(frozen) == spec
    assert freeze_camera_capture_spec(decode_camera_capture_spec(frozen)).payload == frozen.payload

    tree = decode(frozen.payload)
    tree["unexpected"] = True
    with pytest.raises(ValueError, match="unknown field set"):
        decode_camera_capture_spec(encode(tree))
    with pytest.raises(ValueError, match="schema differs"):
        decode_camera_capture_spec(
            encode({**decode(frozen.payload), "schema": "unsupported-camera-spec"})
        )


def test_camera_sample_contract_preserves_all_data_axes_and_owned_pixels():
    value_schema, _dataset_schema = _schemas()
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    value = Value(source, VALID, value_schema)
    sample = CameraSample(value, _metadata())
    contract = CameraSampleContract(value_schema)
    frozen = contract.snapshot(sample)

    source[:] = 999
    assert frozen is sample
    assert frozen.image.values.shape == (3, 4)
    np.testing.assert_array_equal(
        frozen.image.values,
        np.arange(12, dtype=np.uint16).reshape(3, 4),
    )
    assert not frozen.image.values.flags.writeable
    assert contract.source_ordinal(frozen) == 0
    assert contract.captured_at(frozen) == pytest.approx(301.000401)
    assert contract.correlation_id(frozen) == "camera-session:0"


def test_camera_record_snapshots_mutable_driver_pixels_once_then_value_reuses():
    value_schema, _dataset_schema = _schemas()
    mutable = np.arange(12, dtype=np.uint16).reshape(3, 4)
    record = CameraFrameRecord(
        mutable,
        source_ordinal=0,
        produced_count=1,
        frame_stamp=101,
        camera_stamp=201,
        timestamp_seconds=301,
        timestamp_microseconds=401,
        host_received_at_ns=501,
        driver_buffer_index=0,
    )
    assert not np.shares_memory(record.image, mutable)
    assert not record.image.flags.writeable

    value = Value(record.image, VALID, value_schema)
    assert np.shares_memory(value.values, record.image)
    mutable[:] = 0
    assert np.any(value.values != 0)


def test_metadata_digest_covers_every_physical_observation():
    contract = CameraFrameMetadataContract()
    baseline = _metadata()
    baseline_digest = contract.digest(baseline)
    assert len(baseline_digest) == 64
    variants = (
        replace(baseline, source_ordinal=1),
        replace(baseline, produced_count=5),
        replace(baseline, frame_stamp=102),
        replace(baseline, camera_stamp=202),
        replace(baseline, timestamp_seconds=302),
        replace(baseline, timestamp_microseconds=402),
        replace(baseline, host_received_at_ns=502),
        replace(baseline, driver_buffer_index=1),
        replace(baseline, correlation_id="camera-session:1"),
    )
    assert all(contract.digest(value) != baseline_digest for value in variants)


def test_camera_sample_digest_binds_pixels_and_physical_metadata_atomically():
    value_schema, _dataset_schema = _schemas()
    contract = CameraSampleContract(value_schema)
    baseline = _sample(value_schema)
    baseline_digest = contract.digest(baseline)
    changed_pixels = np.array(baseline.image.values, copy=True)
    changed_pixels[1, 2] += 1
    pixel_variant = CameraSample(
        Value(changed_pixels, VALID, value_schema),
        baseline.metadata,
    )
    metadata_variant = CameraSample(
        baseline.image,
        replace(baseline.metadata, frame_stamp=baseline.metadata.frame_stamp + 1),
    )

    assert len(baseline_digest) == 64
    assert contract.digest(_sample(value_schema)) == baseline_digest
    assert contract.digest(pixel_variant) != baseline_digest
    assert contract.digest(metadata_variant) != baseline_digest


def test_camera_metadata_rejects_partial_timestamp():
    with pytest.raises(ValueError, match="must appear together"):
        _metadata(timestamp_microseconds=None)
    with pytest.raises(ValueError, match="less than"):
        _metadata(timestamp_microseconds=1_000_000)
    with pytest.raises(TypeError, match="source_ordinal"):
        _metadata(source_ordinal=0.5)


def test_camera_contract_plugs_into_exact_capture_without_anonymous_data_dim():
    value_schema, dataset_schema = _schemas()
    payload_contract = CameraSampleContract(value_schema)
    event_adapter = CameraDatasetEventAdapter(payload_contract)
    cells = tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(2)
        for point in range(2)
    )
    physical_facts = CameraPhysicalFacts(
        camera_identity="camera:test",
        sensor_identity="sensor:test",
        optical_path="test-path",
        capture_trigger_channels=("camera-trigger",),
        sensor_shape_yx=(3, 4),
        roi_origin_yx=(0, 0),
        roi_shape_yx=(3, 4),
        binning_yx=(1, 1),
        spatial_y_axis_id=AxisId("camera.y"),
        spatial_x_axis_id=AxisId("camera.x"),
        coordinate_frame=CoordinateFrameId("camera.roi-local"),
        dtype=np.dtype("<u2"),
        count_unit="count",
        exposure_seconds=1.0,
        required_external_trigger_interval_seconds=0.0,
        external_trigger_integration_start_offset_seconds=None,
        gain=0.0,
        readout_mode="test",
        opaque_frame_settings_fingerprint="a" * 64,
    )
    evidence = CameraCapabilityEvidence(
        adapter_type="tests.Camera",
        source_id="camera",
        payload_contract_fingerprint=payload_contract.fingerprint,
        capture_spec_owner_fingerprint=CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
        max_blocking_call_seconds=1.0,
        physical_facts=physical_facts,
    )
    binding_stamp = DeviceBindingStamp(
        PhysicalDeviceIdentity(
            "camera:test",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            "test-evidence",
            "test-assets-v1",
        ),
        "camera-binding",
    )
    capability = CaptureCapabilitySnapshot(
        binding_stamp=binding_stamp,
        payload_contract=payload_contract,
        camera_capability_evidence=evidence,
    )
    arm_spec = freeze_camera_capture_spec(
        CameraCaptureSpec(
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            len(cells),
            (1,) * len(cells),
            evidence.settings_fingerprint,
        )
    )
    descriptor = CameraCaptureDescriptor(
        camera_identity=physical_facts.camera_identity,
        sensor_identity=physical_facts.sensor_identity,
        optical_path=physical_facts.optical_path,
        sensor_shape_yx=physical_facts.sensor_shape_yx,
        roi_origin_yx=physical_facts.roi_origin_yx,
        roi_shape_yx=physical_facts.roi_shape_yx,
        binning_yx=physical_facts.binning_yx,
        spatial_y_axis_id=physical_facts.spatial_y_axis_id,
        spatial_x_axis_id=physical_facts.spatial_x_axis_id,
        coordinate_frame=physical_facts.coordinate_frame,
        dtype=physical_facts.dtype,
        count_unit=physical_facts.count_unit,
        readout_event_axis_id=None,
        event_settings=(physical_facts.event_setting(0),),
        camera_arm_spec_fingerprint=arm_spec.digest,
    )
    contract = CameraCaptureContract(
        stream_id=StreamId("camera.frames"),
        dataset_edge=FrozenDatasetEdge(
            dataset_schema,
            event_adapter,
            DatasetCellSchedule.from_cells(dataset_schema, cells),
        ),
        capability=capability,
        camera_provenance=CameraCaptureProvenance(
            descriptor,
            ReadoutBindingKey("camera"),
            binding_stamp,
            capability.capability_fingerprint,
        ),
    )

    assert contract.dataset_schema.physical_shape == (2, 2, 3, 4)
    sample = _sample(value_schema)
    projected = contract.event_adapter.value(sample)
    assert projected is sample.image
    assert projected.values.shape == (3, 4)
    assert np.array_equal(projected.values, sample.image.values)
    assert contract.total_events == 4


def test_camera_capture_owner_proof_belongs_to_the_bound_camera_contract():
    composition = create_virtual_installation(seed=7)
    runtime = composition.runtime
    try:
        camera_ref = runtime.device_catalog.require("camera").ref
        port = runtime.camera_port(camera_ref)
        repeat = _axis("binding.repeat", REPEAT, 1)
        event = _axis("binding.event", READOUT_EVENT, 1)
        schema = DatasetSchema(
            repeat,
            PointTable(
                1,
                (
                    PointColumn(
                        event.axis_id,
                        event.name,
                        event.role,
                        PointColumn.NUMERIC,
                        event.coordinates,
                    ),
                ),
            ),
            None,
            port.capability.payload_contract.value_schema,
        )
        schedule = DatasetCellSchedule.from_cells(
            schema,
            (DatasetCellAddress(0, 0),),
        )
        camera_capture = bind_camera_capture(
            port,
            CameraCaptureBindingRequest(
                "camera",
                schema,
                schedule,
                CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                (port.capability.camera_physical_facts.event_setting(0),),
            ),
        )

        assert camera_capture.capture_spec.owner_fingerprint == (
            camera_capture.capture_contract.capture_spec_owner_fingerprint
        )
        with pytest.raises(
            ValueError,
            match="capture spec and camera contract owner differ",
        ):
            BoundCameraCapture(
                camera_capture.capture_port,
                camera_capture.capture_contract,
                replace(
                    camera_capture.capture_spec,
                    owner_fingerprint="f" * 64,
                ),
            )
    finally:
        assert runtime.shutdown(timeout=2.0)
