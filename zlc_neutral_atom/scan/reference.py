"""Leaf-owned content identity for durable scan authority."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text, sha256_text


SCAN_ARTIFACT_NAMESPACE = "scan"


@dataclass(frozen=True, order=True, slots=True)
class ScanArtifactRef:
    """Content-addressed identity of one current-format ScanArtifact manifest."""

    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        canonical_text(self.repository_id, "repository_id")
        sha256_text(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{SCAN_ARTIFACT_NAMESPACE}/{self.manifest_digest}"

__all__ = [
    "SCAN_ARTIFACT_NAMESPACE",
    "ScanArtifactRef",
]
