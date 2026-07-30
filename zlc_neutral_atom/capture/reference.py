"""Stable relative reference to one directly written camera capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from zlc_storage import canonical_text


CAPTURE_ARTIFACT_REF_SCHEMA = "zlc_neutral_atom.capture-artifact-ref"
CAPTURE_ARTIFACT_NAMESPACE = "capture"


def _capture_record_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("record_path must be str")
    canonical_text(value, "record_path")
    if "\\" in value:
        raise ValueError("record_path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("record_path must be a canonical relative path")
    if len(path.parts) != 2 or path.parts[1] != "capture.json":
        raise ValueError("record_path must be '<run-name>/capture.json'")
    if path.parts[0] in {"", ".", ".."}:
        raise ValueError("record_path requires one concrete run-name")
    canonical_text(path.parts[0], "capture run-name")
    return value


@dataclass(frozen=True, order=True)
class CaptureArtifactRef:
    """Location of one Capture record relative to its explicit captures root."""

    record_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_path", _capture_record_path(self.record_path))

    @property
    def target_ref(self) -> str:
        return f"{CAPTURE_ARTIFACT_NAMESPACE}/{self.record_path}"


def capture_artifact_ref_to_tree(value: CaptureArtifactRef) -> dict[str, object]:
    if not isinstance(value, CaptureArtifactRef):
        raise TypeError("value must be CaptureArtifactRef")
    return {
        "schema": CAPTURE_ARTIFACT_REF_SCHEMA,
        "record_path": value.record_path,
    }


def capture_artifact_ref_from_tree(tree: Any) -> CaptureArtifactRef:
    if not isinstance(tree, dict) or set(tree) != {"schema", "record_path"}:
        raise ValueError("CaptureArtifactRef has an unknown field set")
    if tree["schema"] != CAPTURE_ARTIFACT_REF_SCHEMA:
        raise ValueError("CaptureArtifactRef schema is not current")
    value = CaptureArtifactRef(tree["record_path"])
    if capture_artifact_ref_to_tree(value) != tree:
        raise ValueError("CaptureArtifactRef tree is typed but non-canonical")
    return value


__all__ = [
    "CAPTURE_ARTIFACT_NAMESPACE",
    "CAPTURE_ARTIFACT_REF_SCHEMA",
    "CaptureArtifactRef",
    "capture_artifact_ref_from_tree",
    "capture_artifact_ref_to_tree",
]
