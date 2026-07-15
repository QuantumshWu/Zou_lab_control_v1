"""Leaf-owned identity for committed neutral-atom occupancy results."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text, sha256_text


OCCUPANCY_ARTIFACT_NAMESPACE = "occupancy"


@dataclass(frozen=True, order=True)
class OccupancyArtifactRef:
    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        canonical_text(self.repository_id, "repository_id")
        sha256_text(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{OCCUPANCY_ARTIFACT_NAMESPACE}/{self.manifest_digest}"


__all__ = [
    "OCCUPANCY_ARTIFACT_NAMESPACE",
    "OccupancyArtifactRef",
]
