"""Notebook surface owned by the Occupancy Processor and detection artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zlc_data import Selection
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.camera_measurement.output_binding import (
    CameraFrameOutputBinding,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.runtime.run import RunHandle

from .application import DetectionRequest
from .cell import OccupancyCellDomain
from .processor import ResolvedOccupancy
from .processor_application import (
    OccupancyProcessorRequest,
    PreparedOccupancyProcessor,
    prepare_occupancy_processor,
)
from .reference import OccupancyArtifactRef


class OccupancyNotebookHost(Protocol):
    def resolve_occupancy_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration: ...

    def load_saved_occupancy_calibration(
        self,
        calibration_ref_file: str | Path,
    ) -> ResolvedCalibration: ...

    def build_occupancy_detection_request(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        model_kind: ReadoutModelKind | None,
    ) -> DetectionRequest: ...

    def start_occupancy_detection(self, request: DetectionRequest) -> RunHandle: ...

    def run_occupancy_detection(
        self,
        request: DetectionRequest,
    ) -> OccupancyArtifactRef: ...

    def admit_occupancy_artifact(
        self,
        reference: OccupancyArtifactRef,
    ) -> ResolvedOccupancy: ...

    def inspect_occupancy_navigation(
        self,
        reference: OccupancyArtifactRef,
    ) -> OccupancyCellDomain: ...

    def compose_occupancy_cell_source(
        self,
        reference: OccupancyArtifactRef,
        selection: Selection | None,
        *,
        expected_navigation: OccupancyCellDomain | None,
    ): ...

    def open_occupancy_cell_gui(
        self,
        reference: OccupancyArtifactRef,
        selection: Selection | None,
    ): ...


class OccupancyNotebookAdapter:
    __slots__ = ()

    @property
    def _occupancy_notebook_host(self) -> OccupancyNotebookHost:
        raise NotImplementedError

    def prepare_occupancy_processor_request(
        self,
        request: OccupancyProcessorRequest,
    ) -> PreparedOccupancyProcessor:
        if not isinstance(request, OccupancyProcessorRequest):
            raise TypeError("request must be OccupancyProcessorRequest")
        resolved = self._occupancy_notebook_host.resolve_occupancy_calibration(
            request.calibration_ref
        )
        self._require_binding(resolved.artifact.frame_contract.binding)
        return prepare_occupancy_processor(request, resolved)

    def prepare_occupancy_processor(
        self,
        camera_output_binding: CameraFrameOutputBinding,
        *,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
    ) -> PreparedOccupancyProcessor:
        return self.prepare_occupancy_processor_request(
            OccupancyProcessorRequest(
                camera_output_binding,
                calibration_ref,
                model_kind,
            )
        )

    def prepare_saved_occupancy_processor(
        self,
        camera_output_binding: CameraFrameOutputBinding,
        *,
        calibration_ref_file: str | Path,
        model_kind: ReadoutModelKind | None = None,
    ) -> PreparedOccupancyProcessor:
        resolved = self._occupancy_notebook_host.load_saved_occupancy_calibration(
            calibration_ref_file
        )
        return self.prepare_occupancy_processor_request(
            OccupancyProcessorRequest(
                camera_output_binding,
                resolved.reference,
                model_kind,
            )
        )

    def detection_request(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        model_kind: ReadoutModelKind | None = None,
    ) -> DetectionRequest:
        request = self._occupancy_notebook_host.build_occupancy_detection_request(
            source,
            calibration,
            model_kind=model_kind,
        )
        self._require_binding(request.readout_binding)
        return request

    def start_detection(self, request: DetectionRequest) -> RunHandle:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return self._occupancy_notebook_host.start_occupancy_detection(request)

    def detect(self, request: DetectionRequest) -> OccupancyArtifactRef:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return self._occupancy_notebook_host.run_occupancy_detection(request)

    def load_occupancy(
        self,
        reference: OccupancyArtifactRef,
    ) -> ResolvedOccupancy:
        resolved = self._occupancy_notebook_host.admit_occupancy_artifact(reference)
        self._require_binding(resolved.readout_binding)
        return resolved

    def _inspect_occupancy_cell_navigation(
        self,
        reference: OccupancyArtifactRef,
    ) -> OccupancyCellDomain:
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        domain = self._occupancy_notebook_host.inspect_occupancy_navigation(reference)
        self._require_binding(domain.readout_binding)
        return domain

    def _load_occupancy_cell_source(
        self,
        reference: OccupancyArtifactRef,
        selection: Selection | None,
        *,
        expected_navigation: OccupancyCellDomain | None = None,
    ):
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        if expected_navigation is not None and not isinstance(
            expected_navigation,
            OccupancyCellDomain,
        ):
            raise TypeError("expected_navigation must be OccupancyCellDomain or None")
        return self._occupancy_notebook_host.compose_occupancy_cell_source(
            reference,
            selection,
            expected_navigation=expected_navigation,
        )

    def occupancy_cell_view(
        self,
        reference: OccupancyArtifactRef,
        *,
        selection: Selection | None = None,
    ):
        return self._load_occupancy_cell_source(reference, selection)

    def occupancy_cell_gui(
        self,
        reference: OccupancyArtifactRef,
        *,
        selection: Selection | None = None,
    ):
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        return self._occupancy_notebook_host.open_occupancy_cell_gui(
            reference,
            selection,
        )


__all__ = [
    "OccupancyNotebookAdapter",
    "OccupancyNotebookHost",
]
