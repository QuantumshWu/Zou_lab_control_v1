"""Data-owned identity for one formally persisted fit result.

The reference names a repository manifest without importing any repository
backend.  Storage and domain composition layers may persist or admit the
manifest, while the fit identity and its canonical bytes remain owned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_storage.canonical import (
    canonical_text,
    decode,
    encode,
    exact_mapping,
    sha256_text,
)


FIT_RESULT_ARTIFACT_REF_SCHEMA = "zlc_data.FitResultArtifactRef"
FIT_RESULT_ARTIFACT_NAMESPACE = "fit-result"


@dataclass(frozen=True, order=True)
class FitResultArtifactRef:
    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        canonical_text(self.repository_id, "repository_id")
        sha256_text(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{FIT_RESULT_ARTIFACT_NAMESPACE}/{self.manifest_digest}"


def fit_result_artifact_ref_to_tree(
    value: FitResultArtifactRef,
) -> dict[str, object]:
    if not isinstance(value, FitResultArtifactRef):
        raise TypeError("value must be FitResultArtifactRef")
    return {
        "schema": FIT_RESULT_ARTIFACT_REF_SCHEMA,
        "repository_id": value.repository_id,
        "manifest_digest": value.manifest_digest,
    }


def fit_result_artifact_ref_from_tree(tree: Any) -> FitResultArtifactRef:
    data = exact_mapping(
        tree,
        {"schema", "repository_id", "manifest_digest"},
        FIT_RESULT_ARTIFACT_REF_SCHEMA,
    )
    return FitResultArtifactRef(
        canonical_text(data["repository_id"], "repository_id"),
        sha256_text(data["manifest_digest"], "manifest_digest"),
    )


def encode_fit_result_artifact_ref(value: FitResultArtifactRef) -> bytes:
    return encode(fit_result_artifact_ref_to_tree(value))


def decode_fit_result_artifact_ref(
    payload: bytes | bytearray | memoryview,
) -> FitResultArtifactRef:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("FitResultArtifactRef payload must be bytes-like")
    raw = bytes(payload)
    value = fit_result_artifact_ref_from_tree(decode(raw))
    if encode_fit_result_artifact_ref(value) != raw:
        raise ValueError("FitResultArtifactRef payload is typed but non-canonical")
    return value


__all__ = [
    "FIT_RESULT_ARTIFACT_NAMESPACE",
    "FIT_RESULT_ARTIFACT_REF_SCHEMA",
    "FitResultArtifactRef",
    "decode_fit_result_artifact_ref",
    "encode_fit_result_artifact_ref",
    "fit_result_artifact_ref_from_tree",
    "fit_result_artifact_ref_to_tree",
]
