"""Leaf-owned calibration artifact identity and strict current codec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from zlc_storage import canonical_text, encode


CALIBRATION_ARTIFACT_REF_FORMAT = (
    "zlc_neutral_atom.logic_nodes.readout.calibration.artifact-ref"
)
CALIBRATION_ARTIFACT_NAMESPACE = "calibration"


@dataclass(frozen=True, order=True, slots=True)
class CalibrationArtifactRef:
    """Path to ``calibration.json`` relative to the Calibration output root."""

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
            raise ValueError("record_path must stay beneath the Calibration output root")
        if path.name != "calibration.json" or len(path.parts) != 2:
            raise ValueError(
                "record_path must be '<run-name>/calibration.json'"
            )
        object.__setattr__(self, "record_path", path.as_posix())

    @property
    def target_ref(self) -> str:
        return f"{CALIBRATION_ARTIFACT_NAMESPACE}/{self.record_path}"


def calibration_artifact_ref_to_tree(value: CalibrationArtifactRef) -> dict[str, Any]:
    if not isinstance(value, CalibrationArtifactRef):
        raise TypeError("value must be CalibrationArtifactRef")
    return {
        "schema": CALIBRATION_ARTIFACT_REF_FORMAT,
        "record_path": value.record_path,
    }


def calibration_artifact_ref_from_tree(tree: Any) -> CalibrationArtifactRef:
    fields = {"schema", "record_path"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CalibrationArtifactRef has an unknown field set")
    if tree["schema"] != CALIBRATION_ARTIFACT_REF_FORMAT:
        raise ValueError("CalibrationArtifactRef format is not current")
    value = CalibrationArtifactRef(canonical_text(tree["record_path"], "record_path"))
    if calibration_artifact_ref_to_tree(value) != tree:
        raise ValueError("CalibrationArtifactRef tree is typed but non-canonical")
    return value


def encode_calibration_artifact_ref(value: CalibrationArtifactRef) -> bytes:
    return encode(calibration_artifact_ref_to_tree(value))


__all__ = [
    "CALIBRATION_ARTIFACT_NAMESPACE",
    "CALIBRATION_ARTIFACT_REF_FORMAT",
    "CalibrationArtifactRef",
    "calibration_artifact_ref_from_tree",
    "calibration_artifact_ref_to_tree",
    "encode_calibration_artifact_ref",
]
