"""Authored request and post-run exports for the Calibration Task.

Runtime ownership is deliberately absent.  The discovered ``logic_node``
binds this value to the generic host and sequences the two ordinary Runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zlc_neutral_atom.authoring import (
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
    MINIMUM_POSITIVE_FLOAT,
)
from zlc_storage import integer, normalized_text, positive_real
from zlc_storage.durability import atomic_write_file
from zlc_storage.paths import resolve_under

from .analysis import CalibrationAnalysisResult, CalibrationComputation
from .calibration import ThresholdMethod
from .outputs import calibration_final_outputs
from .reference import CalibrationArtifactRef
from .sitemap import DEFAULT_CALIBRATION_PULSE_PATH


CALIBRATION_THRESHOLD_METHODS = tuple(item.value for item in ThresholdMethod)
DEFAULT_CALIBRATION_THRESHOLD_METHOD = "otsu"
DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S = 0.020
DEFAULT_CALIBRATION_READOUT_EXPOSURE_S = 0.005
DEFAULT_CALIBRATION_THRESHOLD_FRAMES = 100
MINIMUM_CALIBRATION_THRESHOLD_FRAMES = 2
DEFAULT_CALIBRATION_ROI_RADIUS = 1
MINIMUM_CALIBRATION_ROI_RADIUS = 1
DEFAULT_CALIBRATION_GRID_ROWS = 5
DEFAULT_CALIBRATION_GRID_COLUMNS = 7


@dataclass(frozen=True, slots=True)
class CalibrationTaskRequest:
    """Complete user intent before devices and installation facts are bound."""

    save_frames: bool
    pulse: str
    threshold_method: str
    reference_exposure_s: float
    readout_exposure_s: float
    threshold_frames: int
    roi_radius: int
    camera_instance_id: str
    sequencer_instance_id: str
    grid_rows: int = DEFAULT_CALIBRATION_GRID_ROWS
    grid_columns: int = DEFAULT_CALIBRATION_GRID_COLUMNS

    def __post_init__(self) -> None:
        if type(self.save_frames) is not bool:
            raise TypeError("save_frames must be bool")
        pulse = normalized_text(self.pulse, "pulse")
        method = normalized_text(
            self.threshold_method,
            "threshold_method",
        ).lower()
        if method not in CALIBRATION_THRESHOLD_METHODS:
            raise ValueError(
                "threshold_method must be one of "
                f"{CALIBRATION_THRESHOLD_METHODS}"
            )
        reference = positive_real(
            self.reference_exposure_s,
            "reference_exposure_s",
        )
        readout = positive_real(
            self.readout_exposure_s,
            "readout_exposure_s",
        )
        frames = integer(
            self.threshold_frames,
            "threshold_frames",
            minimum=MINIMUM_CALIBRATION_THRESHOLD_FRAMES,
        )
        radius = integer(
            self.roi_radius,
            "roi_radius",
            minimum=MINIMUM_CALIBRATION_ROI_RADIUS,
        )
        assert frames is not None and radius is not None
        camera_id = normalized_text(
            self.camera_instance_id,
            "camera_instance_id",
        )
        sequencer_id = normalized_text(
            self.sequencer_instance_id,
            "sequencer_instance_id",
        )
        rows = integer(self.grid_rows, "grid_rows", minimum=1)
        columns = integer(self.grid_columns, "grid_columns", minimum=1)
        assert rows is not None and columns is not None
        object.__setattr__(self, "pulse", pulse)
        object.__setattr__(self, "threshold_method", method)
        object.__setattr__(self, "reference_exposure_s", reference)
        object.__setattr__(self, "readout_exposure_s", readout)
        object.__setattr__(self, "threshold_frames", frames)
        object.__setattr__(self, "roi_radius", radius)
        object.__setattr__(self, "camera_instance_id", camera_id)
        object.__setattr__(self, "sequencer_instance_id", sequencer_id)
        object.__setattr__(self, "grid_rows", rows)
        object.__setattr__(self, "grid_columns", columns)

    @property
    def grid_shape_yx(self) -> tuple[int, int]:
        return self.grid_rows, self.grid_columns


_CALIBRATION_TASK_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "save_frames",
            "bool",
            "Save raw frames",
            default=False,
            required=True,
            description=(
                "Export source values and validity beside the calibration "
                "record."
            ),
        ),
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_CALIBRATION_PULSE_PATH,
            required=True,
            description="Imaging pulse for each long-short-long bracket.",
        ),
        AuthoringField(
            "threshold_method",
            "choice",
            "Threshold",
            default=DEFAULT_CALIBRATION_THRESHOLD_METHOD,
            required=True,
            choices=tuple(
                AuthoringChoice(value, value)
                for value in CALIBRATION_THRESHOLD_METHODS
            ),
            description="Per-site threshold estimator.",
        ),
        AuthoringField(
            "reference_exposure_s",
            "float",
            "Reference exposure (long)",
            default=DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S,
            required=True,
            unit="s",
            minimum=MINIMUM_POSITIVE_FLOAT,
            allow_blank=False,
        ),
        AuthoringField(
            "readout_exposure_s",
            "float",
            "Readout exposure (short)",
            default=DEFAULT_CALIBRATION_READOUT_EXPOSURE_S,
            required=True,
            unit="s",
            minimum=MINIMUM_POSITIVE_FLOAT,
            allow_blank=False,
        ),
        AuthoringField(
            "threshold_frames",
            "int",
            "Reference brackets",
            default=DEFAULT_CALIBRATION_THRESHOLD_FRAMES,
            required=True,
            minimum=MINIMUM_CALIBRATION_THRESHOLD_FRAMES,
            allow_blank=False,
        ),
        AuthoringField(
            "roi_radius",
            "int",
            "ROI radius",
            default=DEFAULT_CALIBRATION_ROI_RADIUS,
            required=True,
            unit="px",
            minimum=MINIMUM_CALIBRATION_ROI_RADIUS,
            allow_blank=False,
        ),
        AuthoringField(
            "grid_rows",
            "int",
            "Lattice rows",
            default=DEFAULT_CALIBRATION_GRID_ROWS,
            required=True,
            minimum=1,
            allow_blank=False,
            description="Number of site rows to detect in the calibration image.",
        ),
        AuthoringField(
            "grid_columns",
            "int",
            "Lattice columns",
            default=DEFAULT_CALIBRATION_GRID_COLUMNS,
            required=True,
            minimum=1,
            allow_blank=False,
            description="Number of site columns to detect in the calibration image.",
        ),
        AuthoringField(
            "camera_instance_id",
            "choice",
            "Camera",
            required=True,
            dynamic_choices=True,
            description="Stable installed device instance with camera.capture.",
        ),
        AuthoringField(
            "sequencer_instance_id",
            "choice",
            "Pulse sequencer",
            required=True,
            dynamic_choices=True,
            description="Stable installed device instance with pulse.execute.",
        ),
    )
)


def calibration_task_authoring_schema() -> AuthoringSchema:
    return _CALIBRATION_TASK_AUTHORING_SCHEMA


def build_calibration_task_request(
    values: Mapping[str, object],
) -> CalibrationTaskRequest:
    return CalibrationTaskRequest(
        **calibration_task_authoring_schema().freeze(values)  # type: ignore[arg-type]
    )


def write_calibration_post_final_exports(
    result: CalibrationAnalysisResult,
    reference: CalibrationArtifactRef,
    *,
    project_root: Path,
    export_plots: Callable,
    save_frames: bool,
    warn: Callable[[str], None],
) -> None:
    """Write optional raw frames and the human report from the live result.

    The report is intentionally not reconstructed from the runtime artifact:
    fidelity diagnostics belong to this completed Task result, while the
    artifact stores only facts required to classify future frames.
    """

    if not isinstance(result, CalibrationAnalysisResult):
        raise TypeError("result must be CalibrationAnalysisResult")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    root = Path(project_root).expanduser()
    if not root.is_absolute():
        raise ValueError("project_root must be absolute")
    root = root.resolve()
    if type(save_frames) is not bool:
        raise TypeError("save_frames must be bool")
    if not callable(export_plots):
        raise TypeError("export_plots must be callable")
    if not callable(warn):
        raise TypeError("warn must be callable")
    record_path = resolve_under(root, reference.record_path)
    run_directory = record_path.parent
    if save_frames:
        try:
            snapshot = result.source.materialize_snapshot()

            def save(path: Path, array: np.ndarray) -> None:
                atomic_write_file(
                    path,
                    lambda stream: np.save(stream, array, allow_pickle=False),
                )

            save(run_directory / "source_frames.npy", snapshot.block.values)
            save(
                run_directory / "source_frame_validity.npy",
                np.asarray(snapshot.block.validity.mask),
            )
        except Exception as error:
            warn(f"Calibration raw-frame export failed: {error}")
    computation = CalibrationComputation(result.artifact, result.report)
    try:
        export_plots(
            run_directory / "report",
            calibration_final_outputs(computation, reference),
        )
    except Exception as error:
        warn(f"Calibration report export failed: {error}")


__all__ = [
    "CALIBRATION_THRESHOLD_METHODS",
    "DEFAULT_CALIBRATION_PULSE_PATH",
    "DEFAULT_CALIBRATION_GRID_ROWS",
    "DEFAULT_CALIBRATION_GRID_COLUMNS",
    "CalibrationTaskRequest",
    "build_calibration_task_request",
    "calibration_task_authoring_schema",
    "write_calibration_post_final_exports",
]
