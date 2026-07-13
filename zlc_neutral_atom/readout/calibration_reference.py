"""Leaf-owned calibration artifact identity and strict current codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_storage import decode, encode


CALIBRATION_ARTIFACT_REF_SCHEMA = "zlc_neutral_atom.calibration-artifact-ref.v1"
_CALIBRATION_NAMESPACE = "calibration"


def _canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, order=True)
class CalibrationArtifactRef:
    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _canonical_text(self.repository_id, "repository_id")
        _sha256(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{_CALIBRATION_NAMESPACE}/{self.manifest_digest}"


def calibration_artifact_ref_to_tree(value: CalibrationArtifactRef) -> dict[str, Any]:
    if not isinstance(value, CalibrationArtifactRef):
        raise TypeError("value must be CalibrationArtifactRef")
    return {
        "schema": CALIBRATION_ARTIFACT_REF_SCHEMA,
        "repository_id": value.repository_id,
        "manifest_digest": value.manifest_digest,
    }


def calibration_artifact_ref_from_tree(tree: Any) -> CalibrationArtifactRef:
    fields = {"schema", "repository_id", "manifest_digest"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CalibrationArtifactRef has an unknown field set")
    if tree["schema"] != CALIBRATION_ARTIFACT_REF_SCHEMA:
        raise ValueError("CalibrationArtifactRef schema is not current")
    value = CalibrationArtifactRef(
        _canonical_text(tree["repository_id"], "repository_id"),
        _sha256(tree["manifest_digest"], "manifest_digest"),
    )
    if calibration_artifact_ref_to_tree(value) != tree:
        raise ValueError("CalibrationArtifactRef tree is typed but non-canonical")
    return value


def encode_calibration_artifact_ref(value: CalibrationArtifactRef) -> bytes:
    return encode(calibration_artifact_ref_to_tree(value))


def decode_calibration_artifact_ref(
    payload: bytes | bytearray | memoryview,
) -> CalibrationArtifactRef:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("CalibrationArtifactRef payload must be bytes-like")
    raw = bytes(payload)
    value = calibration_artifact_ref_from_tree(decode(raw))
    if encode_calibration_artifact_ref(value) != raw:
        raise ValueError("CalibrationArtifactRef payload is typed but non-canonical")
    return value


__all__ = [
    "CALIBRATION_ARTIFACT_REF_SCHEMA",
    "CalibrationArtifactRef",
    "calibration_artifact_ref_from_tree",
    "calibration_artifact_ref_to_tree",
    "decode_calibration_artifact_ref",
    "encode_calibration_artifact_ref",
]
