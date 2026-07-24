"""Prepared Camera -> Occupancy Processor application boundary.

The Workbench resolves which row/output the operator selected.  This module
owns every physical consequence of that choice: the Camera output contract,
calibration admission identity, model selection, output vocabulary, and the
classification of one immutable source revision.  A desktop shell may host
the pure evaluation on a worker and namespace the returned outputs; it never
reconstructs an occupancy schema or carries a calibration resolver callback.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.logic_nodes.camera_measurement import CameraMeasurementRequest
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.logic_nodes.calibration.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.occupancy.processor import (
    OCCUPANCY_CALIBRATION_INPUT_SPEC,
    OCCUPANCY_CAMERA_INPUT_SPEC,
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
    OccupancyProcessorConfig,
    OccupancyProcessorEvaluation,
    evaluate_occupancy_processor,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_storage import sha256_text


@dataclass(frozen=True, slots=True)
class OccupancyProcessorRequest:
    """One admitted Camera output and calibration identity.

    Console row names are deliberately absent.  They are Workbench routing
    identities, not neutral-atom physics.  The request instead freezes the
    Camera application's own typed request and exact output declaration.
    """

    camera_request: CameraMeasurementRequest
    camera_output: DatasetOutputDeclaration
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_request, CameraMeasurementRequest):
            raise TypeError("camera_request must be CameraMeasurementRequest")
        if not isinstance(self.camera_output, DatasetOutputDeclaration):
            raise TypeError("camera_output must be DatasetOutputDeclaration")
        if self.camera_output not in self.camera_request.output_declarations:
            raise ValueError(
                "camera_output is absent from the Camera Measurement request"
            )
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")

    @property
    def camera_output_name(self) -> str:
        return self.camera_output.name


def bind_occupancy_processor_request(
    config: OccupancyProcessorConfig,
    inputs: BoundNodeInputs,
) -> OccupancyProcessorRequest:
    """Bind owner-declared inputs without any TaskConsole field knowledge."""

    if not isinstance(config, OccupancyProcessorConfig):
        raise TypeError("config must be OccupancyProcessorConfig")
    if not isinstance(inputs, BoundNodeInputs):
        raise TypeError("inputs must be BoundNodeInputs")
    camera = inputs.dataset(OCCUPANCY_CAMERA_INPUT_SPEC)
    calibration = inputs.artifact(OCCUPANCY_CALIBRATION_INPUT_SPEC)
    from zlc_neutral_atom.logic_nodes.camera_measurement import CAMERA_MEASUREMENT_KEY

    if camera.producer_definition != CAMERA_MEASUREMENT_KEY:
        raise ValueError("Occupancy source must be a Camera Measurement output")
    if camera.transform_spec is not None:
        raise ValueError("Occupancy source must be an untransformed Camera frame")
    if not isinstance(camera.producer_request, CameraMeasurementRequest):
        raise TypeError("Camera source has another request type")
    if not isinstance(calibration.reference, CalibrationArtifactRef):
        raise TypeError("Calibration input has another artifact reference type")
    return OccupancyProcessorRequest(
        camera.producer_request,
        camera.output,
        calibration.reference,
        config.model_kind,
    )


class PreparedOccupancyProcessor:
    """Closed, replayable evaluation over immutable Camera revisions.

    Repository admission happens once while preparing this value.  ``evaluate``
    is then a deterministic application operation over an immutable source
    snapshot and its exact source coverage/event identity.  It owns no QWidget,
    executor, subscription, mutable latest state, or device session.
    """

    __slots__ = ("_calibration", "_request", "_selected_model_kind")

    def __init__(
        self,
        request: OccupancyProcessorRequest,
        calibration: ResolvedCalibration,
    ) -> None:
        if not isinstance(request, OccupancyProcessorRequest):
            raise TypeError("request must be OccupancyProcessorRequest")
        if type(calibration) is not ResolvedCalibration:
            raise TypeError("calibration must be an admitted ResolvedCalibration")
        calibration._require_authority()
        if calibration.reference != request.calibration_ref:
            raise ValueError(
                "admitted calibration differs from the frozen application request"
            )
        expected_binding = ReadoutBindingKey(
            request.camera_request.camera_ref.role
        )
        if calibration.artifact.frame_contract.binding != expected_binding:
            raise ValueError(
                "Camera Measurement role differs from the calibration readout binding"
            )
        selected = calibration.artifact.select_model(request.model_kind)
        self._request = request
        self._calibration = calibration
        self._selected_model_kind = selected.kind

    @property
    def request(self) -> OccupancyProcessorRequest:
        return self._request

    @property
    def output_declarations(self):
        return OCCUPANCY_LIVE_OUTPUT_DECLARATIONS

    @property
    def selected_model_kind(self) -> ReadoutModelKind:
        return self._selected_model_kind

    def evaluate(
        self,
        source: OwnedSnapshot,
        coverage: MonitorCoverage,
        *,
        source_event_digest: str,
    ) -> OccupancyProcessorEvaluation:
        """Classify one already-routed Camera revision atomically."""

        if not isinstance(source, OwnedSnapshot):
            raise TypeError("source must be OwnedSnapshot")
        if not isinstance(coverage, MonitorCoverage):
            raise TypeError(
                "Occupancy Processor requires Camera delivery coverage"
            )
        digest = sha256_text(source_event_digest, "source_event_digest")
        assert digest is not None
        evaluation = evaluate_occupancy_processor(
            source,
            self._calibration,
            coverage,
            model_kind=self._selected_model_kind,
            source_event_digest=digest,
        )
        declarations = tuple(
            output.declaration for output in evaluation.outputs.values()
        )
        if declarations != self.output_declarations:
            raise RuntimeError(
                "Occupancy Processor returned another output vocabulary"
            )
        if any(
            not isinstance(output, LiveDatasetOutput)
            for output in evaluation.outputs.values()
        ):
            raise RuntimeError(
                "Occupancy Processor returned a non-live output"
            )
        return evaluation


def prepare_occupancy_processor(
    request: OccupancyProcessorRequest,
    calibration: ResolvedCalibration,
) -> PreparedOccupancyProcessor:
    """Freeze one admitted Occupancy Processor application."""

    return PreparedOccupancyProcessor(request, calibration)


__all__ = [
    "OccupancyProcessorRequest",
    "PreparedOccupancyProcessor",
    "bind_occupancy_processor_request",
    "prepare_occupancy_processor",
]
