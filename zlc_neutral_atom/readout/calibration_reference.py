"""Leaf-owned calibration artifact identity and strict current codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_storage import (
    canonical_text as _canonical_text,
    decode,
    encode,
    sha256_text as _sha256,
)


CALIBRATION_ARTIFACT_REF_SCHEMA = "zlc_neutral_atom.calibration-artifact-ref.v1"
_CALIBRATION_NAMESPACE = "calibration"


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


def calibration_artifact_input_ref(value: CalibrationArtifactRef):
    """Mint the runtime dependency edge through this reference owner's codec."""

    from zlc_neutral_atom.runtime.streams import ArtifactInputRef

    if not isinstance(value, CalibrationArtifactRef):
        raise TypeError("value must be CalibrationArtifactRef")
    return ArtifactInputRef(
        CALIBRATION_ARTIFACT_REF_SCHEMA,
        encode_calibration_artifact_ref(value),
        value.manifest_digest,
    )


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
    "calibration_artifact_input_ref",
    "calibration_artifact_ref_to_tree",
    "decode_calibration_artifact_ref",
    "encode_calibration_artifact_ref",
]
