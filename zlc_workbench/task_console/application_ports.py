"""Explicit application capabilities consumed by the TaskConsole shell.

The outer composition root resolves installation/runtime services once and
supplies this closed set of operations and immutable installation facts.
Adding a console operation therefore changes this type visibly; the console
cannot discover capabilities through an opaque owning object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from zlc_neutral_atom.camera_measurement import (
    DEFAULT_CAMERA_FRAMES_PER_CYCLE,
    DEFAULT_CAMERA_MEASUREMENT_REPEAT,
    CameraMeasurementRequest,
)
from zlc_neutral_atom.capture_application import PreparedFiniteCameraMeasurement
from zlc_neutral_atom.monitor_application import PreparedLiveCameraMeasurement
from zlc_neutral_atom.mot_field_task import MotFieldTaskIntent, PreparedMotFieldTask
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.calibration_task import (
    CalibrationTaskIntent,
    PreparedCalibrationTask,
)
from zlc_neutral_atom.readout.coupled_application import (
    CoupledMeasurementApplicationCommand,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    GreyMolassesDetuningIntent,
    ReadoutDurationFidelityIntent,
    TemperatureReleaseRecaptureIntent,
)
from zlc_neutral_atom.readout.reactive_occupancy_application import (
    PreparedReactiveOccupancyMonitor,
)
from zlc_neutral_atom.scan import PulseScanProgram, ScanSourceBinding
from zlc_neutral_atom.scan.application import PreparedExactScan
from zlc_pulse import PulseTemplateDescription
from zlc_storage import canonical_text


class CameraMeasurementRequestBuilder(Protocol):
    def __call__(
        self,
        *,
        camera_role: str | None = None,
        repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT,
        frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE,
    ) -> CameraMeasurementRequest: ...


class ReactiveOccupancyPreparer(Protocol):
    def __call__(
        self,
        camera_request: CameraMeasurementRequest,
        camera_output_name: str,
        *,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
    ) -> PreparedReactiveOccupancyMonitor: ...


class ScanSourcePreparer(Protocol):
    def __call__(
        self,
        program: PulseScanProgram,
        source: ScanSourceBinding,
    ) -> PreparedExactScan: ...


def _roles(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    roles = tuple(values)
    if len(set(roles)) != len(roles):
        raise ValueError(f"{field} must contain unique roles")
    for role in roles:
        canonical_text(role, field)
    return roles


@dataclass(frozen=True, slots=True)
class TaskConsoleApplicationPorts:
    """Closed, non-locating capability set required by ``open_task_console``."""

    installed_camera_roles: tuple[str, ...]
    sitemap_camera_roles: tuple[str, ...]
    installed_rf_roles: tuple[str, ...]
    build_camera_measurement_request: CameraMeasurementRequestBuilder
    prepare_camera_measurement: Callable[
        [CameraMeasurementRequest],
        PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement,
    ]
    load_saved_calibration_reference: Callable[
        [str | Path], CalibrationArtifactRef
    ]
    prepare_temperature_release_recapture: Callable[
        [TemperatureReleaseRecaptureIntent, CalibrationArtifactRef],
        CoupledMeasurementApplicationCommand,
    ]
    prepare_readout_duration_fidelity: Callable[
        [ReadoutDurationFidelityIntent, CalibrationArtifactRef],
        CoupledMeasurementApplicationCommand,
    ]
    prepare_grey_molasses_detuning: Callable[
        [GreyMolassesDetuningIntent, CalibrationArtifactRef],
        CoupledMeasurementApplicationCommand,
    ]
    prepare_reactive_occupancy: ReactiveOccupancyPreparer
    prepare_calibration_task: Callable[
        [CalibrationTaskIntent], PreparedCalibrationTask
    ]
    prepare_mot_field_task: Callable[[MotFieldTaskIntent], PreparedMotFieldTask]
    prepare_scan_source: ScanSourcePreparer
    read_pulse_template: Callable[[str], PulseTemplateDescription]

    def __post_init__(self) -> None:
        cameras = _roles(self.installed_camera_roles, "installed_camera_roles")
        sitemap = _roles(self.sitemap_camera_roles, "sitemap_camera_roles")
        rf = _roles(self.installed_rf_roles, "installed_rf_roles")
        if not set(sitemap).issubset(cameras):
            raise ValueError(
                "sitemap_camera_roles must be installed camera roles"
            )
        object.__setattr__(self, "installed_camera_roles", cameras)
        object.__setattr__(self, "sitemap_camera_roles", sitemap)
        object.__setattr__(self, "installed_rf_roles", rf)
        for field in (
            "build_camera_measurement_request",
            "prepare_camera_measurement",
            "load_saved_calibration_reference",
            "prepare_temperature_release_recapture",
            "prepare_readout_duration_fidelity",
            "prepare_grey_molasses_detuning",
            "prepare_reactive_occupancy",
            "prepare_calibration_task",
            "prepare_mot_field_task",
            "prepare_scan_source",
            "read_pulse_template",
        ):
            if not callable(getattr(self, field)):
                raise TypeError(f"{field} must be callable")


__all__ = [
    "CameraMeasurementRequestBuilder",
    "ReactiveOccupancyPreparer",
    "ScanSourcePreparer",
    "TaskConsoleApplicationPorts",
]
