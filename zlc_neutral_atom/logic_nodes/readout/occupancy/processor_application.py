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

from zlc_data import OwnedSnapshot, ValueSchema
from zlc_neutral_atom.logic_nodes.camera_measurement.output_binding import (
    CameraFrameOutputBinding,
)
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import (
    OCCUPANCY_CALIBRATION_INPUT_SPEC,
    OCCUPANCY_CAMERA_INPUT_SPEC,
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
    OccupancyProcessorConfig,
    OccupancyProcessorEvaluation,
    _evaluate_occupancy_processor,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from zlc_storage import sha256_text

from .signal_source import (
    OccupancySignalProcessor,
    RunningOccupancySignalSource,
)


@dataclass(frozen=True, slots=True)
class OccupancyProcessorRequest:
    """One admitted Camera output and calibration identity.

    Console row names are deliberately absent.  They are Workbench routing
    identities, not neutral-atom physics.  The request instead freezes the
    Camera application's own typed request and exact output declaration.
    """

    camera_output_binding: CameraFrameOutputBinding
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_output_binding, CameraFrameOutputBinding):
            raise TypeError(
                "camera_output_binding must be CameraFrameOutputBinding"
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
        return self.camera_output_binding.output.name


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
    from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
        CAMERA_MEASUREMENT_KEY,
    )

    if camera.producer_definition != CAMERA_MEASUREMENT_KEY:
        raise ValueError("Occupancy source must be a Camera Measurement output")
    if camera.transform_spec is not None:
        raise ValueError("Occupancy source must be an untransformed Camera frame")
    if not isinstance(camera.output_binding, CameraFrameOutputBinding):
        raise ValueError(
            "Occupancy requires an active Camera output with endpoint-read "
            "physical binding"
        )
    if camera.output_binding.output != camera.output:
        raise ValueError("Camera output binding names another Dataset output")
    if not isinstance(calibration.reference, CalibrationArtifactRef):
        raise TypeError("Calibration input has another artifact reference type")
    return OccupancyProcessorRequest(
        camera.output_binding,
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

    __slots__ = (
        "_calibration",
        "_request",
        "_selected_model_kind",
        "_signal_processor",
    )

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
        source_binding = request.camera_output_binding
        calibration.artifact.frame_contract.assert_compatible_working_point(
            source_binding.readout_binding,
            source_binding.capability_evidence.physical_facts,
            source_binding.frame_schema,
        )
        selected = calibration.artifact.select_model(request.model_kind)
        self._request = request
        self._calibration = calibration
        self._selected_model_kind = selected.kind
        self._signal_processor = OccupancySignalProcessor(
            frame_contract=calibration.artifact.frame_contract,
            model=selected,
            artifact_input=calibration_artifact_input_ref(calibration.reference),
            source_binding=source_binding,
        )

    @property
    def request(self) -> OccupancyProcessorRequest:
        return self._request

    @property
    def output_declarations(self):
        return OCCUPANCY_LIVE_OUTPUT_DECLARATIONS

    @property
    def selected_model_kind(self) -> ReadoutModelKind:
        return self._selected_model_kind

    @property
    def site_map(self):
        return self._calibration.artifact.site_map

    def signal_value_schema(self, output_name: str) -> ValueSchema:
        """Return one declared per-event output schema before starting the worker."""

        return self._signal_processor.value_schema(output_name)

    def start_signal_events(
        self,
        source: SignalEventSource,
    ) -> RunningOccupancySignalSource:
        """Start lossless future-event classification without owning the Camera."""

        return self._signal_processor.start(
            source,
            self._request.camera_output_name,
        )

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
        source_binding = self._request.camera_output_binding
        if source.ref.stream_generation != source_binding.stream_generation:
            raise ValueError(
                "Camera snapshot belongs to another live stream generation"
            )
        if source.block.schema.cell_schema != source_binding.frame_schema:
            raise ValueError("Camera snapshot differs from its bound frame schema")
        digest = sha256_text(source_event_digest, "source_event_digest")
        assert digest is not None
        evaluation = _evaluate_occupancy_processor(
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
