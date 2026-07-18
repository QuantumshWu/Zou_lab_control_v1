"""Neutral-atom-owned identity for one persisted capture-fit result."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text, sha256_text


CAPTURE_FIT_RESULT_ARTIFACT_NAMESPACE = "fit-result"


@dataclass(frozen=True, order=True)
class CaptureFitResultArtifactRef:
    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        canonical_text(self.repository_id, "repository_id")
        sha256_text(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{CAPTURE_FIT_RESULT_ARTIFACT_NAMESPACE}/{self.manifest_digest}"


__all__ = [
    "CAPTURE_FIT_RESULT_ARTIFACT_NAMESPACE",
    "CaptureFitResultArtifactRef",
]
