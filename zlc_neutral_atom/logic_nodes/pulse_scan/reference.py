"""Leaf-owned content identity for durable scan authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_storage import canonical_text, sha256_text


SCAN_ARTIFACT_REF_SCHEMA = "zlc_neutral_atom.logic_nodes.pulse_scan.artifact-ref"
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


def scan_artifact_ref_to_tree(value: ScanArtifactRef) -> dict[str, object]:
    if not isinstance(value, ScanArtifactRef):
        raise TypeError("value must be ScanArtifactRef")
    return {
        "schema": SCAN_ARTIFACT_REF_SCHEMA,
        "repository_id": value.repository_id,
        "manifest_digest": value.manifest_digest,
    }


def scan_artifact_ref_from_tree(tree: Any) -> ScanArtifactRef:
    fields = {"schema", "repository_id", "manifest_digest"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("ScanArtifactRef has an unknown field set")
    if tree["schema"] != SCAN_ARTIFACT_REF_SCHEMA:
        raise ValueError("ScanArtifactRef schema is not current")
    value = ScanArtifactRef(
        canonical_text(tree["repository_id"], "repository_id"),
        sha256_text(tree["manifest_digest"], "manifest_digest"),
    )
    if scan_artifact_ref_to_tree(value) != tree:
        raise ValueError("ScanArtifactRef tree is typed but non-canonical")
    return value


__all__ = [
    "SCAN_ARTIFACT_REF_SCHEMA",
    "SCAN_ARTIFACT_NAMESPACE",
    "ScanArtifactRef",
    "scan_artifact_ref_from_tree",
    "scan_artifact_ref_to_tree",
]
