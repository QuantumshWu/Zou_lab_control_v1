"""Neutral camera-domain values remain multidimensional, immutable, and canonical."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    PointLayout,
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
from zlc_neutral_atom.acquisition.camera import (
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
from zlc_neutral_atom.runtime import (
    CaptureCapabilitySnapshot,
    CaptureRuntimeProfile,
    CaptureStreamContract,
    DatasetCellAddress,
    ProducerFlowControl,
    StreamId,
)
from zlc_storage import decode, encode


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _schemas() -> tuple[ValueSchema, DatasetSchema]:
    value_schema = ValueSchema(
        (
            _axis("camera.y", SPATIAL_Y, 3),
            _axis("camera.x", SPATIAL_X, 4),
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
            encode({**decode(frozen.payload), "schema": "camera-spec.v0"})
        )


def test_camera_sample_contract_preserves_all_data_axes_and_owned_bytes():
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
    assert contract.retained_nbytes(frozen) <= contract.max_retained_nbytes


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


def test_camera_metadata_rejects_partial_timestamp_and_unbounded_correlation():
    with pytest.raises(ValueError, match="must appear together"):
        _metadata(timestamp_microseconds=None)
    with pytest.raises(ValueError, match="less than"):
        _metadata(timestamp_microseconds=1_000_000)
    with pytest.raises(TypeError, match="source_ordinal"):
        _metadata(source_ordinal=0.5)

    contract = CameraFrameMetadataContract(correlation_id_max_bytes=4)
    with pytest.raises(ValueError, match="retained-byte bound"):
        contract.validate(_metadata(correlation_id="too-long"))


def test_camera_contract_plugs_into_exact_capture_without_anonymous_data_dim():
    value_schema, dataset_schema = _schemas()
    payload_contract = CameraSampleContract(value_schema)
    event_adapter = CameraDatasetEventAdapter(payload_contract)
    cells = tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(2)
        for point in range(2)
    )
    capability = CaptureCapabilitySnapshot(
        binding_id="camera-binding",
        stable_device_identity="camera:test",
        connection_generation="generation-1",
        capability_fingerprint="b" * 64,
        settings_fingerprint="a" * 64,
        payload_contract_fingerprint=payload_contract.fingerprint,
        capture_spec_owner_fingerprint=CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
        flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
        max_source_burst_events=2,
        driver_ring_bytes=1 << 20,
        adapter_record_retention_bytes=1 << 20,
        max_blocking_call_seconds=1.0,
        max_capture_spec_bytes=4096,
    )
    contract = CaptureStreamContract(
        StreamId("camera.frames"),
        "camera",
        dataset_schema,
        payload_contract,
        event_adapter,
        cells,
        capability,
        CaptureRuntimeProfile(0, 3 << 20),
        CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    )

    assert contract.dataset_schema.physical_shape == (2, 2, 3, 4)
    assert contract.event_adapter.value(_sample(value_schema)).values.shape == (3, 4)
    assert contract.payload_contract.max_retained_nbytes == (
        ValuePayloadContract(value_schema).max_retained_nbytes
        + contract.event_adapter.metadata_contract.max_retained_nbytes
    )
    assert contract.total_events == 4
    assert contract.max_inflight_events == 2
