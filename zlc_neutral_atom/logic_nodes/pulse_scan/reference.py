"""Leaf-owned path identity for one durable scan record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from zlc_storage import canonical_text


SCAN_ARTIFACT_REF_SCHEMA = "zlc_neutral_atom.logic_nodes.pulse_scan.artifact-ref"
SCAN_ARTIFACT_NAMESPACE = "scan"


@dataclass(frozen=True, order=True, slots=True)
class ScanArtifactRef:
    """Path to ``scan.json`` relative to the configured scan output root."""

    record_path: str

    def __post_init__(self) -> None:
        value = canonical_text(self.record_path, "record_path")
        if "\\" in value:
            raise ValueError("record_path must use POSIX separators")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("record_path must stay beneath the scan output root")
        if path.name != "scan.json" or len(path.parts) != 2:
            raise ValueError("record_path must be '<run-name>/scan.json'")
        object.__setattr__(self, "record_path", path.as_posix())

    @property
    def target_ref(self) -> str:
        return f"{SCAN_ARTIFACT_NAMESPACE}/{self.record_path}"


def scan_artifact_ref_to_tree(value: ScanArtifactRef) -> dict[str, object]:
    if not isinstance(value, ScanArtifactRef):
        raise TypeError("value must be ScanArtifactRef")
    return {
        "schema": SCAN_ARTIFACT_REF_SCHEMA,
        "record_path": value.record_path,
    }


def scan_artifact_ref_from_tree(tree: Any) -> ScanArtifactRef:
    fields = {"schema", "record_path"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("ScanArtifactRef has an unknown field set")
    if tree["schema"] != SCAN_ARTIFACT_REF_SCHEMA:
        raise ValueError("ScanArtifactRef schema is not current")
    value = ScanArtifactRef(canonical_text(tree["record_path"], "record_path"))
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
