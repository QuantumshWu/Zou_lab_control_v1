"""Leaf-owned identity and strict codec for durable camera captures.

This module intentionally imports neither ``artifacts`` nor ``readout`` so
calibration/result domains can name a source capture without creating a
repository import cycle or inventing a second reference type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_storage import decode, encode


CAPTURE_ARTIFACT_REF_SCHEMA = "zlc_neutral_atom.capture-artifact-ref.v1"
CAPTURE_ARTIFACT_NAMESPACE = "capture"


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


@dataclass(frozen=True, order=True)
class CaptureArtifactRef:
    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _canonical_text(self.repository_id, "repository_id")
        _sha256(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{CAPTURE_ARTIFACT_NAMESPACE}/{self.manifest_digest}"


def capture_artifact_ref_to_tree(value: CaptureArtifactRef) -> dict[str, object]:
    if not isinstance(value, CaptureArtifactRef):
        raise TypeError("value must be CaptureArtifactRef")
    return {
        "schema": CAPTURE_ARTIFACT_REF_SCHEMA,
        "repository_id": value.repository_id,
        "manifest_digest": value.manifest_digest,
    }


def capture_artifact_ref_from_tree(tree: Any) -> CaptureArtifactRef:
    fields = {"schema", "repository_id", "manifest_digest"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CaptureArtifactRef has an unknown field set")
    if tree["schema"] != CAPTURE_ARTIFACT_REF_SCHEMA:
        raise ValueError("CaptureArtifactRef schema is not current")
    value = CaptureArtifactRef(
        _canonical_text(tree["repository_id"], "repository_id"),
        _sha256(tree["manifest_digest"], "manifest_digest"),
    )
    if capture_artifact_ref_to_tree(value) != tree:
        raise ValueError("CaptureArtifactRef tree is typed but non-canonical")
    return value


def encode_capture_artifact_ref(value: CaptureArtifactRef) -> bytes:
    return encode(capture_artifact_ref_to_tree(value))


def decode_capture_artifact_ref(
    payload: bytes | bytearray | memoryview,
) -> CaptureArtifactRef:
    raw = bytes(payload)
    value = capture_artifact_ref_from_tree(decode(raw))
    if encode_capture_artifact_ref(value) != raw:
        raise ValueError("CaptureArtifactRef payload is typed but non-canonical")
    return value


__all__ = [
    "CAPTURE_ARTIFACT_REF_SCHEMA",
    "CAPTURE_ARTIFACT_NAMESPACE",
    "CaptureArtifactRef",
    "capture_artifact_ref_from_tree",
    "capture_artifact_ref_to_tree",
    "decode_capture_artifact_ref",
    "encode_capture_artifact_ref",
]
