"""Strict current pointer written by a successful calibration task.

The repositories own the actual capture and calibration artifacts.  A task
output folder contains only this small, human-portable pointer joining the two
references; both the writer and every later Occupancy reader use this codec.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from zlc_neutral_atom.logic_nodes.camera_capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)

from .reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_from_tree,
    calibration_artifact_ref_to_tree,
)


CALIBRATION_TASK_OUTPUT_FORMAT = (
    "zlc_neutral_atom.logic_nodes.calibration.task-output"
)


@dataclass(frozen=True, slots=True)
class CalibrationTaskOutput:
    calibration_ref: CalibrationArtifactRef
    source_capture_ref: CaptureArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")


def calibration_task_output_to_tree(value: CalibrationTaskOutput) -> dict[str, Any]:
    if not isinstance(value, CalibrationTaskOutput):
        raise TypeError("value must be CalibrationTaskOutput")
    return {
        "schema": CALIBRATION_TASK_OUTPUT_FORMAT,
        "calibration_ref": calibration_artifact_ref_to_tree(value.calibration_ref),
        "source_capture_ref": capture_artifact_ref_to_tree(value.source_capture_ref),
    }


def calibration_task_output_from_tree(tree: Any) -> CalibrationTaskOutput:
    fields = {"schema", "calibration_ref", "source_capture_ref"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CalibrationTaskOutput has an unknown field set")
    if tree["schema"] != CALIBRATION_TASK_OUTPUT_FORMAT:
        raise ValueError("CalibrationTaskOutput format is not current")
    value = CalibrationTaskOutput(
        calibration_artifact_ref_from_tree(tree["calibration_ref"]),
        capture_artifact_ref_from_tree(tree["source_capture_ref"]),
    )
    if calibration_task_output_to_tree(value) != tree:
        raise ValueError("CalibrationTaskOutput tree is typed but non-canonical")
    return value


def write_calibration_task_output(
    path: str | Path,
    value: CalibrationTaskOutput,
) -> None:
    destination = Path(path).expanduser().resolve()
    destination.write_text(
        json.dumps(
            calibration_task_output_to_tree(value),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_calibration_task_output(path: str | Path) -> CalibrationTaskOutput:
    source = Path(path).expanduser().resolve()
    try:
        tree = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("CalibrationTaskOutput is not valid JSON") from error
    return calibration_task_output_from_tree(tree)


__all__ = [
    "CALIBRATION_TASK_OUTPUT_FORMAT",
    "CalibrationTaskOutput",
    "calibration_task_output_from_tree",
    "calibration_task_output_to_tree",
    "read_calibration_task_output",
    "write_calibration_task_output",
]
