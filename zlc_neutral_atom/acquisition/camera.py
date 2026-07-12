"""Immutable camera samples and frozen acquisition specifications.

This module owns neutral-domain camera values.  It deliberately knows nothing
about DCAM, Pylon, Qt, a concrete device registry, or a driver frame object.
Composition adapters translate their private driver records into these values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from zlc_data import VALID, Valid, Value, ValuePayloadContract, ValueSchema
from zlc_storage import canonical_digest, decode, encode

from zlc_neutral_atom.runtime.capture import FrozenCaptureSpec


_CAMERA_CAPTURE_SPEC_SCHEMA = "zlc_neutral_atom.camera-capture-spec.v1"
CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT = canonical_digest(
    {
        "owner": "zlc_neutral_atom.acquisition.camera",
        "schema": _CAMERA_CAPTURE_SPEC_SCHEMA,
    }
)


def _canonical_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: object,
    field: str,
    *,
    optional: bool = False,
    nonnegative: bool = False,
) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        suffix = " or None" if optional else ""
        raise TypeError(f"{field} must be an integer{suffix}")
    normalized = int(value)
    if nonnegative and normalized < 0:
        raise ValueError(f"{field} must be non-negative")
    return normalized


class CameraAcquisitionMode(str, Enum):
    EXTERNAL_TRIGGERED = "EXTERNAL_TRIGGERED"
    FREE_RUNNING = "FREE_RUNNING"


@dataclass(frozen=True)
class CameraCaptureSpec:
    """Request-owned settings that are frozen before runtime preflight."""

    mode: CameraAcquisitionMode
    expected_frames: int
    settings_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CameraAcquisitionMode):
            raise TypeError("mode must be CameraAcquisitionMode")
        frames = _integer(
            self.expected_frames,
            "expected_frames",
            nonnegative=True,
        )
        assert frames is not None
        if frames == 0:
            raise ValueError("expected_frames must be positive")
        object.__setattr__(self, "expected_frames", frames)
        _sha256(self.settings_fingerprint, "settings_fingerprint")


def freeze_camera_capture_spec(spec: CameraCaptureSpec) -> FrozenCaptureSpec:
    if not isinstance(spec, CameraCaptureSpec):
        raise TypeError("spec must be CameraCaptureSpec")
    payload = encode(
        {
            "schema": _CAMERA_CAPTURE_SPEC_SCHEMA,
            "mode": spec.mode.value,
            "expected_frames": spec.expected_frames,
            "settings_fingerprint": spec.settings_fingerprint,
        }
    )
    return FrozenCaptureSpec(CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT, payload)


def decode_camera_capture_spec(value: FrozenCaptureSpec | bytes) -> CameraCaptureSpec:
    """Decode exactly the current camera spec schema; no runtime legacy reader."""

    if isinstance(value, FrozenCaptureSpec):
        if value.owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
            raise ValueError("camera capture spec owner fingerprint differs")
        payload = value.payload
    elif isinstance(value, bytes):
        payload = value
    else:
        raise TypeError("value must be FrozenCaptureSpec or canonical bytes")
    decoded = decode(payload)
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema",
        "mode",
        "expected_frames",
        "settings_fingerprint",
    }:
        raise ValueError("camera capture spec has an unknown field set")
    if decoded["schema"] != _CAMERA_CAPTURE_SPEC_SCHEMA:
        raise ValueError("camera capture spec schema differs")
    try:
        mode = CameraAcquisitionMode(decoded["mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("camera capture mode is unknown") from exc
    return CameraCaptureSpec(
        mode,
        decoded["expected_frames"],
        decoded["settings_fingerprint"],
    )


@dataclass(frozen=True)
class CameraFrameMetadata:
    """One frame's frozen physical observations plus run correlation identity."""

    source_ordinal: int
    produced_count: int | None
    frame_stamp: int | None
    camera_stamp: int | None
    timestamp_seconds: int | None
    timestamp_microseconds: int | None
    host_received_at_ns: int
    driver_buffer_index: int | None
    correlation_id: str

    def __post_init__(self) -> None:
        for field in ("source_ordinal", "host_received_at_ns"):
            value = _integer(getattr(self, field), field, nonnegative=True)
            assert value is not None
            if field == "host_received_at_ns" and value == 0:
                raise ValueError("host_received_at_ns must be positive")
            object.__setattr__(self, field, value)
        for field in (
            "produced_count",
            "timestamp_seconds",
            "timestamp_microseconds",
            "driver_buffer_index",
        ):
            object.__setattr__(
                self,
                field,
                _integer(
                    getattr(self, field),
                    field,
                    optional=True,
                    nonnegative=True,
                ),
            )
        for field in ("frame_stamp", "camera_stamp"):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field, optional=True),
            )
        if (
            self.timestamp_microseconds is not None
            and self.timestamp_microseconds >= 1_000_000
        ):
            raise ValueError("timestamp_microseconds must be less than 1_000_000")
        if (self.timestamp_seconds is None) != (self.timestamp_microseconds is None):
            raise ValueError("camera timestamp seconds and microseconds must appear together")
        _canonical_text(self.correlation_id, "correlation_id")

    @property
    def captured_at(self) -> float:
        if self.timestamp_seconds is not None:
            assert self.timestamp_microseconds is not None
            return float(self.timestamp_seconds) + self.timestamp_microseconds * 1e-6
        return self.host_received_at_ns * 1e-9


@dataclass(frozen=True)
class CameraSample:
    image: Value
    metadata: CameraFrameMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.image, Value):
            raise TypeError("image must be a zlc_data.Value")
        if not isinstance(self.metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")


@dataclass(frozen=True)
class CameraFrameMetadataContract:
    correlation_id_max_bytes: int = 160

    def __post_init__(self) -> None:
        limit = _integer(
            self.correlation_id_max_bytes,
            "correlation_id_max_bytes",
            nonnegative=True,
        )
        assert limit is not None
        if limit == 0:
            raise ValueError("correlation_id_max_bytes must be positive")
        object.__setattr__(self, "correlation_id_max_bytes", limit)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc.camera-frame-metadata.v1",
                "correlation_id_max_bytes": self.correlation_id_max_bytes,
            }
        )

    @property
    def max_retained_nbytes(self) -> int:
        # Conservative scalar/object accounting plus bounded UTF-8 correlation id.
        return 256 + self.correlation_id_max_bytes

    def snapshot(self, payload: CameraSample) -> CameraFrameMetadata:
        if not isinstance(payload, CameraSample):
            raise TypeError("metadata snapshot requires CameraSample")
        self.validate(payload.metadata)
        return payload.metadata

    def validate(self, metadata: object) -> None:
        if not isinstance(metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")
        encoded_id = metadata.correlation_id.encode("utf-8")
        if len(encoded_id) > self.correlation_id_max_bytes:
            raise ValueError("correlation_id exceeds the declared retained-byte bound")
        if not math.isfinite(metadata.captured_at):
            raise ValueError("captured_at must be finite")

    def retained_nbytes(self, metadata: object) -> int:
        self.validate(metadata)
        assert isinstance(metadata, CameraFrameMetadata)
        return 256 + len(metadata.correlation_id.encode("utf-8"))

    def digest(self, metadata: object) -> str:
        self.validate(metadata)
        assert isinstance(metadata, CameraFrameMetadata)
        return canonical_digest(
            {
                "source_ordinal": metadata.source_ordinal,
                "produced_count": metadata.produced_count,
                "frame_stamp": metadata.frame_stamp,
                "camera_stamp": metadata.camera_stamp,
                "timestamp_seconds": metadata.timestamp_seconds,
                "timestamp_microseconds": metadata.timestamp_microseconds,
                "host_received_at_ns": metadata.host_received_at_ns,
                "driver_buffer_index": metadata.driver_buffer_index,
                "correlation_id": metadata.correlation_id,
            }
        )


@dataclass(frozen=True)
class CameraSampleContract:
    value_schema: ValueSchema
    metadata_contract: CameraFrameMetadataContract = CameraFrameMetadataContract()

    def __post_init__(self) -> None:
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema")
        if not isinstance(self.metadata_contract, CameraFrameMetadataContract):
            raise TypeError("metadata_contract must be CameraFrameMetadataContract")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc.camera-sample.v1",
                "value_schema_fingerprint": self.value_schema.fingerprint,
                "metadata_contract_fingerprint": self.metadata_contract.fingerprint,
            }
        )

    @property
    def max_retained_nbytes(self) -> int:
        return (
            ValuePayloadContract(self.value_schema).max_retained_nbytes
            + self.metadata_contract.max_retained_nbytes
        )

    def snapshot(self, payload: CameraSample) -> CameraSample:
        self.validate(payload)
        return payload

    def validate(self, payload: CameraSample) -> None:
        if not isinstance(payload, CameraSample):
            raise TypeError("payload must be CameraSample")
        ValuePayloadContract(self.value_schema).validate(payload.image)
        self.metadata_contract.validate(payload.metadata)

    def retained_nbytes(self, payload: CameraSample) -> int:
        self.validate(payload)
        return (
            ValuePayloadContract(self.value_schema).retained_nbytes(payload.image)
            + self.metadata_contract.retained_nbytes(payload.metadata)
        )

    @staticmethod
    def source_ordinal(payload: CameraSample) -> int:
        return payload.metadata.source_ordinal

    @staticmethod
    def captured_at(payload: CameraSample) -> float:
        return payload.metadata.captured_at

    @staticmethod
    def correlation_id(payload: CameraSample) -> str:
        return payload.metadata.correlation_id


@dataclass(frozen=True)
class CameraDatasetEventAdapter:
    payload_contract: CameraSampleContract

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, CameraSampleContract):
            raise TypeError("payload_contract must be CameraSampleContract")

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.value_schema

    @property
    def metadata_contract(self) -> CameraFrameMetadataContract:
        return self.payload_contract.metadata_contract

    def value(self, payload: CameraSample) -> Value:
        self.payload_contract.validate(payload)
        return payload.image


__all__ = [
    "CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT",
    "CameraAcquisitionMode",
    "CameraCaptureSpec",
    "CameraDatasetEventAdapter",
    "CameraFrameMetadata",
    "CameraFrameMetadataContract",
    "CameraSample",
    "CameraSampleContract",
    "decode_camera_capture_spec",
    "freeze_camera_capture_spec",
]
