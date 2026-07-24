"""Typed physical-source binding for exact pulse scans.

The Workbench may resolve a selected row/output and an authoritative Figure
transform, but it does not own Camera or Occupancy semantics.  This module is
the single boundary that validates those resolved facts and constructs the
installed scan request.  Both pulse-program forms therefore follow one path;
no UI layer expands an autonomous or API-segmented program back into facade
arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zlc_data import DataTransformSpec
from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_storage import canonical_text

from .contracts import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseScanProgram,
)

if TYPE_CHECKING:
    from zlc_neutral_atom.readout.calibration import ReadoutModelKind


def _require_transform(
    value: DataTransformSpec | None,
) -> DataTransformSpec | None:
    if value is None:
        return None
    if not isinstance(value, DataTransformSpec):
        raise TypeError("transform must be DataTransformSpec or None")
    if not value.operations:
        raise ValueError("an empty transform must be None")
    return value


def _require_single_frame_camera(
    request: CameraMeasurementRequest,
) -> CameraMeasurementRequest:
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera_request must be CameraMeasurementRequest")
    if request.frames_per_cycle != 1:
        raise ValueError(
            "an exact pulse-scan source requires exactly one Camera frame "
            "per scan cell"
        )
    return request


def _require_model_kind(value: ReadoutModelKind | None) -> None:
    if value is None:
        return
    # Kept local so importing the public scan package while calibration itself
    # is loading does not create a calibration -> artifacts -> scan cycle.
    from zlc_neutral_atom.readout.calibration import ReadoutModelKind

    if not isinstance(value, ReadoutModelKind):
        raise TypeError("model_kind must be ReadoutModelKind or None")


def _require_occupancy_output_name(value: str) -> str:
    # Occupancy owns its output vocabulary.  The local import avoids copying
    # that vocabulary here while preserving calibration's acyclic import path.
    from zlc_neutral_atom.readout.occupancy import (
        OCCUPANCY_EXACT_SOURCE_OUTPUT_NAMES,
    )

    output_name = canonical_text(value, "output_name")
    if output_name not in OCCUPANCY_EXACT_SOURCE_OUTPUT_NAMES:
        raise ValueError("Occupancy scan output must be 'counts' or 'occupied'")
    return output_name


@dataclass(frozen=True, slots=True)
class DirectCameraScanSource:
    """One exact Camera output selected as the scan's physical y source."""

    camera_request: CameraMeasurementRequest
    output_name: str
    transform: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        camera_request = _require_single_frame_camera(self.camera_request)
        output_name = canonical_text(self.output_name, "output_name")
        if output_name not in camera_request.output_names:
            raise ValueError(
                "output_name is absent from the Camera Measurement request"
            )
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "transform", _require_transform(self.transform))


@dataclass(frozen=True, slots=True)
class OccupancyScanSource:
    """One Camera→Occupancy physical source selected as exact scan y.

    Capacity-one Camera capture makes its sole ``frame_0`` input unambiguous;
    ``output_name`` names the classified result and is intentionally limited
    to the two lossless exact outputs.  Monitor-only ``rate`` is not a formal
    scan source.
    """

    camera_request: CameraMeasurementRequest
    output_name: str
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    transform: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        _require_single_frame_camera(self.camera_request)
        output_name = _require_occupancy_output_name(self.output_name)
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        _require_model_kind(self.model_kind)
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "transform", _require_transform(self.transform))

    @property
    def camera_output_name(self) -> str:
        """Return the request-owned sole Camera output without guessing."""

        return self.camera_request.output_names[0]


ScanSourceBinding = DirectCameraScanSource | OccupancyScanSource


def _validate_scan_request_fields(
    program: PulseScanProgram,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    trigger_channel: str | None,
    output_transform_spec: DataTransformSpec | None,
) -> None:
    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a current pulse-scan program")
    if not isinstance(camera_ref, DeviceRef):
        raise TypeError("camera_ref must be DeviceRef")
    if not isinstance(sequencer_ref, DeviceRef):
        raise TypeError("sequencer_ref must be DeviceRef")
    if trigger_channel is not None:
        canonical_text(trigger_channel, "trigger_channel")
    _require_transform(output_transform_spec)


@dataclass(frozen=True, slots=True)
class ScanRequest:
    """Frozen exact pulse scan whose y is the sole Camera frame."""

    program: PulseScanProgram
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    trigger_channel: str | None = None
    output_transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        _validate_scan_request_fields(
            self.program,
            self.camera_ref,
            self.sequencer_ref,
            self.trigger_channel,
            self.output_transform_spec,
        )


@dataclass(frozen=True, slots=True)
class OccupancyScanRequest:
    """Frozen exact pulse scan whose y is Camera-derived Occupancy."""

    program: PulseScanProgram
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    output_name: str = "counts"
    trigger_channel: str | None = None
    output_transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        _require_model_kind(self.model_kind)
        output_name = _require_occupancy_output_name(self.output_name)
        object.__setattr__(self, "output_name", output_name)
        _validate_scan_request_fields(
            self.program,
            self.camera_ref,
            self.sequencer_ref,
            self.trigger_channel,
            self.output_transform_spec,
        )


def build_scan_request(
    program: PulseScanProgram,
    source: ScanSourceBinding,
    *,
    sequencer_ref: DeviceRef,
    trigger_channel: str | None = None,
) -> ScanRequest | OccupancyScanRequest:
    """Construct one installed request from a typed, already-resolved source.

    ``program`` stays intact.  Consequently autonomous SCAN_SLOT and the
    accepted API-slot segmented form share this exact construction path.
    Device lookup remains the composition root's responsibility; this pure
    function only binds resolved identities and domain intent.
    """

    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a current pulse-scan program")
    if not isinstance(sequencer_ref, DeviceRef):
        raise TypeError("sequencer_ref must be DeviceRef")
    if trigger_channel is not None:
        canonical_text(trigger_channel, "trigger_channel")

    if isinstance(source, DirectCameraScanSource):
        return ScanRequest(
            program=program,
            camera_ref=source.camera_request.camera_ref,
            sequencer_ref=sequencer_ref,
            trigger_channel=trigger_channel,
            output_transform_spec=source.transform,
        )
    if isinstance(source, OccupancyScanSource):
        return OccupancyScanRequest(
            program=program,
            camera_ref=source.camera_request.camera_ref,
            sequencer_ref=sequencer_ref,
            calibration_ref=source.calibration_ref,
            model_kind=source.model_kind,
            output_name=source.output_name,
            trigger_channel=trigger_channel,
            output_transform_spec=source.transform,
        )
    raise TypeError(
        "source must be DirectCameraScanSource or OccupancyScanSource"
    )


__all__ = [
    "DirectCameraScanSource",
    "OccupancyScanRequest",
    "OccupancyScanSource",
    "ScanRequest",
    "ScanSourceBinding",
    "build_scan_request",
]
