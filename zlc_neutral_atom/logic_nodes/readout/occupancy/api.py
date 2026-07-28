"""Public Experiment API owned by Occupancy and its detection artifact."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.camera_measurement.output_binding import (
    CameraFrameOutputBinding,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

from .application import (
    DetectionRequest,
    build_detection_request,
)
from .cell import (
    OccupancyCellDomain,
)
from .processor import ResolvedOccupancy
from .processor_application import (
    OccupancyProcessorRequest,
    PreparedOccupancyProcessor,
    prepare_occupancy_processor,
)
from .reference import OccupancyArtifactRef
from .repository import OccupancyRepository


class OccupancyApi:
    __slots__ = (
        "_admit_capture",
        "_calibration",
        "_inspect_cell",
        "_load_cell",
        "_load_occupancy_operation",
        "_repository",
        "_repository_path",
        "_require_binding_operation",
        "_start_detection_operation",
        "_wait_run",
    )

    def __init__(
        self,
        calibration,
        *,
        repository_path: Path,
        require_binding: Callable,
        wait_run: Callable,
        admit_capture: Callable,
        start_detection: Callable,
        load_occupancy: Callable,
        inspect_cell: Callable,
        load_cell: Callable,
    ) -> None:
        if not isinstance(repository_path, Path):
            raise TypeError("repository_path must be Path")
        operations = (
            require_binding,
            wait_run,
            admit_capture,
            start_detection,
            load_occupancy,
            inspect_cell,
            load_cell,
        )
        if any(not callable(operation) for operation in operations):
            raise TypeError("Occupancy API operations must be callable")
        self._calibration = calibration
        self._repository_path = repository_path
        self._require_binding_operation = require_binding
        self._wait_run = wait_run
        self._admit_capture = admit_capture
        self._start_detection_operation = start_detection
        self._load_occupancy_operation = load_occupancy
        self._inspect_cell = inspect_cell
        self._load_cell = load_cell
        self._repository: OccupancyRepository | None = None

    def _occupancy_repository(self) -> OccupancyRepository:
        repository = self._repository
        if repository is None:
            repository = OccupancyRepository(self._repository_path)
            self._repository = repository
        return repository

    def close(self) -> tuple[Exception, ...]:
        repository = self._repository
        if repository is None:
            return ()
        try:
            repository.close()
        except Exception as error:
            return (error,)
        return ()

    def _require_binding(self, binding: ReadoutBindingKey) -> None:
        self._require_binding_operation(binding)

    def prepare_occupancy_processor_request(
        self,
        request: OccupancyProcessorRequest,
    ) -> PreparedOccupancyProcessor:
        if not isinstance(request, OccupancyProcessorRequest):
            raise TypeError("request must be OccupancyProcessorRequest")
        resolved = self._calibration.load_calibration(request.calibration_ref)
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
        resolved = self._calibration.load_saved_calibration(calibration_ref_file)
        return self.prepare_occupancy_processor_request(
            OccupancyProcessorRequest(
                camera_output_binding,
                resolved.reference,
                model_kind,
            )
        )

    def load_saved_calibration(self, path: str | Path) -> ResolvedCalibration:
        """Admit one saved Calibration pointer for Occupancy authoring."""

        return self._calibration.load_saved_calibration(path)

    def detection_request(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        model_kind: ReadoutModelKind | None = None,
    ) -> DetectionRequest:
        request = build_detection_request(
            self._admit_capture(source),
            self._calibration.load_calibration(calibration),
            model_kind=model_kind,
        )
        self._require_binding(request.readout_binding)
        return request

    def start_detection(self, request: DetectionRequest) -> RunHandle:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return self._start_detection_operation(
            request,
            self._occupancy_repository(),
        )

    def detect(self, request: DetectionRequest) -> OccupancyArtifactRef:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return self._wait_run(self.start_detection(request))

    def load_occupancy(
        self,
        reference: OccupancyArtifactRef,
    ) -> ResolvedOccupancy:
        resolved = self._load_occupancy_operation(
            reference,
            self._occupancy_repository(),
        )
        self._require_binding(resolved.readout_binding)
        return resolved

    def _project_figure(
        self,
        reference: OccupancyArtifactRef,
        *,
        output: str | None,
        materialize: bool,
    ):
        """Project an Occupancy artifact through its capability-owned UI leaf."""

        from .ui.view_projection import project_occupancy_figure

        return project_occupancy_figure(
            self.load_occupancy(reference),
            output=output,
            materialize=materialize,
        )

    def _inspect_occupancy_cell_navigation(
        self,
        reference: OccupancyArtifactRef,
    ) -> OccupancyCellDomain:
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        domain = self._inspect_cell(reference, self._occupancy_repository())
        self._require_binding(domain.readout_binding)
        return domain

    def _load_occupancy_cell_source(
        self,
        reference: OccupancyArtifactRef,
        address: DatasetCellAddress | None,
        *,
        expected_navigation: OccupancyCellDomain | None = None,
    ):
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if address is not None and not isinstance(address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress or None")
        if expected_navigation is not None and not isinstance(
            expected_navigation,
            OccupancyCellDomain,
        ):
            raise TypeError("expected_navigation must be OccupancyCellDomain or None")
        source = self._load_cell(
            reference,
            self._occupancy_repository(),
            address,
            expected_domain_identity=(
                None
                if expected_navigation is None
                else expected_navigation.identity
            ),
        )
        self._require_binding(source.domain.readout_binding)
        from .ui.view_projection import build_exact_occupancy_cell_view

        return build_exact_occupancy_cell_view(source)

    def occupancy_cell_view(
        self,
        reference: OccupancyArtifactRef,
        *,
        address: DatasetCellAddress | None = None,
    ):
        return self._load_occupancy_cell_source(reference, address)

    def occupancy_cell_gui(
        self,
        reference: OccupancyArtifactRef,
        *,
        address: DatasetCellAddress | None = None,
    ):
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if address is not None and not isinstance(address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress or None")
        from .ui.workbench import open_occupancy_cell_workbench

        return open_occupancy_cell_workbench(
            self._inspect_occupancy_cell_navigation,
            self._load_occupancy_cell_source,
            reference,
            address=address,
        )


__all__ = [
    "OccupancyApi",
]
