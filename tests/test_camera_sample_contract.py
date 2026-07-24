"""Neutral camera-domain values remain multidimensional, immutable, and canonical."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraCaptureSpec,
    CameraDatasetEventAdapter,
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.logic_nodes.camera_capture.session import (
    CameraCaptureContract,
    CameraCaptureProvenance,
)
from zlc_neutral_atom.devices.camera.capture_port import CaptureCapabilitySnapshot
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
from zlc_neutral_atom.logic_nodes.camera_measurement.binding import (
    CameraCaptureBindingRequest,
    _source_group_sizes,
)
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
    dataset_schema = DatasetSchema(
        _axis("repeat", REPEAT, 2),
        (_axis("point", SCAN_POINT, 2),),
        PointLayout.rect_c((2,)),
        value_schema,
    )
    return value_schema, dataset_schema


def test_camera_source_groups_come_only_from_the_frozen_cell_schedule() -> None:
    value_schema, _ = _schemas()
    repeat = _axis("repeat.grouping", REPEAT, 2)
    scan = _axis("scan.grouping", SCAN_POINT, 3)

    scalar_layout = PointLayout.rect_c((scan.size,))
    scalar_schema = DatasetSchema(repeat, (scan,), scalar_layout, value_schema)
    scalar_cells = tuple(
        DatasetCellAddress(r, p)
        for r in range(repeat.size)
        for p in range(scan.size)
    )
    scalar_request = CameraCaptureBindingRequest(
        "camera",
        repeat,
        (scan,),
        scalar_layout,
        DatasetCellSchedule.from_cells(scalar_schema, scalar_cells),
        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
    )
    assert _source_group_sizes(scalar_request, scalar_schema) == (1,) * 6

    event = _axis("event.grouping", READOUT_EVENT, 2)
    layout = PointLayout.rect_c((scan.size, event.size))
    schema = DatasetSchema(repeat, (scan, event), layout, value_schema)

    def address(r: int, p: int, e: int) -> DatasetCellAddress:
        return DatasetCellAddress(r, layout.storage_index((p, e)))

    cells = tuple(
        address(r, p, e)
        for r in range(repeat.size)
        for p in range(scan.size)
        for e in range(event.size)
    )
    request = CameraCaptureBindingRequest(
        "camera",
        repeat,
        (scan, event),
        layout,
        DatasetCellSchedule.from_cells(schema, cells),
        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
    )
    assert _source_group_sizes(request, schema) == (2,) * 6

    split = (
        address(0, 0, 0),
        address(0, 1, 0),
        address(0, 0, 1),
        address(0, 1, 1),
        *(address(r, p, e) for r, p, e in (
            (0, 2, 0), (0, 2, 1),
            (1, 0, 0), (1, 0, 1),
            (1, 1, 0), (1, 1, 1),
            (1, 2, 0), (1, 2, 1),
        )),
    )
    split_request = replace(
        request,
        cell_schedule=DatasetCellSchedule.from_cells(schema, split),
    )
    with pytest.raises(ValueError, match="incomplete READOUT_EVENT group"):
        _source_group_sizes(split_request, schema)


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
