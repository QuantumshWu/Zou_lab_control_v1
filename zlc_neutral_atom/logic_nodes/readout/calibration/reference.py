"""Leaf-owned path identity for durable Calibration results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from zlc_storage import canonical_text, resolve_under


CALIBRATION_ARTIFACT_REF_FORMAT = (
    "zlc_neutral_atom.logic_nodes.readout.calibration.artifact-ref"
)


@dataclass(frozen=True, order=True, slots=True)
class CalibrationArtifactRef:
    """Project-relative path to one readable Calibration record."""

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
            raise ValueError("record_path must stay beneath the project root")
        if (
            path.name != "calibration.json"
            or len(path.parts) != 4
            or path.parts[:2] != ("tasks", "calibration")
        ):
            raise ValueError(
                "record_path must be "
                "'tasks/calibration/<run-name>/calibration.json'"
            )
        canonical_text(path.parts[2], "calibration run-name")
        object.__setattr__(self, "record_path", path.as_posix())

    @property
    def target_ref(self) -> str:
        return self.record_path


def calibration_artifact_ref_from_input(
    project_root: str | Path,
    value: CalibrationArtifactRef | str | Path,
) -> CalibrationArtifactRef:
    """Resolve one host-frozen current, Task-output, or saved-record input."""

    if isinstance(value, CalibrationArtifactRef):
        return value
    if not isinstance(value, (str, Path)):
        raise TypeError("Calibration input must be an artifact ref or record path")
    project = Path(project_root).expanduser()
    if not project.is_absolute():
        raise ValueError("project_root must be absolute")
    project = project.resolve()
    authored = Path(value).expanduser()
    if authored.is_absolute():
        record = authored.resolve()
        try:
            record.relative_to(project)
        except ValueError as error:
            raise ValueError(
                "Calibration record must stay below project_root"
            ) from error
    else:
        record = resolve_under(project, authored)
    return CalibrationArtifactRef(record.relative_to(project).as_posix())


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
    return value


__all__ = [
    "CALIBRATION_ARTIFACT_REF_FORMAT",
    "CalibrationArtifactRef",
    "calibration_artifact_ref_from_input",
    "calibration_artifact_ref_from_tree",
    "calibration_artifact_ref_to_tree",
]
