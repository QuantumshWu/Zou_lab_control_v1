"""Public Experiment API owned by Occupancy and its direct artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_neutral_atom.capture.artifact import load_capture_artifact
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.camera_measurement.output_binding import (
    CameraFrameOutputBinding,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
    load_calibration_artifact,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress
from zlc_neutral_atom.runtime.run import RunHandle

from .application import DetectionRequest, build_detection_request, prepare_detection_plan
from .cell import (
    OccupancyCellDomain,
    inspect_occupancy_cell_domain,
    load_exact_occupancy_cell_source,
)
from .processor import ResolvedOccupancy
from .processor_application import (
    OccupancyProcessorRequest,
    PreparedOccupancyProcessor,
    prepare_occupancy_processor,
)
from .reference import OccupancyArtifactRef
from .artifact import load_occupancy_artifact


class OccupancyApi:
    __slots__ = (
        "_calibration",
        "_calibrations_root",
        "_captures_root",
        "_occupancy_root",
        "_open_ui",
        "_start_run",
        "_wait_run",
    )

    def __init__(
        self,
        calibration,
        *,
        captures_root: Path,
        calibrations_root: Path,
        occupancy_root: Path,
        start_run: Callable,
        wait_run: Callable,
        open_ui: Callable,
    ) -> None:
        for field, value in (
            ("captures_root", captures_root),
            ("calibrations_root", calibrations_root),
            ("occupancy_root", occupancy_root),
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise TypeError(f"{field} must be an absolute Path")
        if any(not callable(operation) for operation in (start_run, wait_run, open_ui)):
            raise TypeError("Occupancy API operations must be callable")
        self._calibration = calibration
        self._captures_root = captures_root.resolve()
        self._calibrations_root = calibrations_root.resolve()
        self._occupancy_root = occupancy_root.resolve()
        self._start_run = start_run
        self._wait_run = wait_run
        self._open_ui = open_ui

    def close(self) -> tuple[Exception, ...]:
        return ()

    def prepare_occupancy_processor_request(
        self,
        request: OccupancyProcessorRequest,
    ) -> PreparedOccupancyProcessor:
        if not isinstance(request, OccupancyProcessorRequest):
            raise TypeError("request must be OccupancyProcessorRequest")
        resolved = load_calibration_artifact(
            self._calibrations_root,
            self._captures_root,
            request.calibration_ref,
        )
        return prepare_occupancy_processor(request, resolved)

    def prepare_occupancy_processor(
        self,
        camera_output_binding: CameraFrameOutputBinding,
        *,
        calibration_ref: CalibrationArtifactRef | None = None,
        model_kind: ReadoutModelKind | None = None,
    ) -> PreparedOccupancyProcessor:
        reference = self._resolve_calibration_ref(calibration_ref)
        return self.prepare_occupancy_processor_request(
            OccupancyProcessorRequest(
                camera_output_binding,
                reference,
                model_kind,
            )
        )

    def _resolve_calibration_ref(
        self,
        reference: CalibrationArtifactRef | None,
    ) -> CalibrationArtifactRef:
        if reference is None:
            reference = self._calibration.current_calibration_ref
            if reference is None:
                raise RuntimeError(
                    "no current Calibration; pass calibration_ref explicitly or set "
                    "exp.nodes.calibration.current_calibration_ref"
                )
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef or None")
        return reference

    def _reference_from_record_path(
        self,
        path: str | Path,
    ) -> CalibrationArtifactRef:
        """Validate one explicit ``calibration.json`` and freeze its typed ref."""

        record_path = Path(path).expanduser().resolve()
        try:
            relative = record_path.relative_to(self._calibrations_root)
        except ValueError as error:
            raise ValueError(
                "Calibration record must be inside the current project's "
                "_output/calibrations directory"
            ) from error
        reference = CalibrationArtifactRef(relative.as_posix())
        self._calibration.load_calibration(reference)
        return reference

    def detection_request(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef | None = None,
        *,
        model_kind: ReadoutModelKind | None = None,
    ) -> DetectionRequest:
        reference = self._resolve_calibration_ref(calibration)
        return build_detection_request(
            load_capture_artifact(self._captures_root, source),
            load_calibration_artifact(
                self._calibrations_root,
                self._captures_root,
                reference,
            ),
            model_kind=model_kind,
        )

    def start_detection(self, request: DetectionRequest) -> RunHandle:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        plan = prepare_detection_plan(
            request,
            captures_root=self._captures_root,
            calibrations_root=self._calibrations_root,
            occupancy_root=self._occupancy_root,
        )
        return self._start_run(plan)

    def detect(self, request: DetectionRequest) -> OccupancyArtifactRef:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        return self._wait_run(self.start_detection(request))

    def load_occupancy(
        self,
        reference: OccupancyArtifactRef,
    ) -> ResolvedOccupancy:
        return load_occupancy_artifact(
            self._occupancy_root,
            self._captures_root,
            self._calibrations_root,
            reference,
        )

    def _inspect_occupancy_cell_navigation(
        self,
        reference: OccupancyArtifactRef,
    ) -> OccupancyCellDomain:
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        return inspect_occupancy_cell_domain(
            reference,
            self._occupancy_root,
            self._captures_root,
            self._calibrations_root,
        )

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
        return load_exact_occupancy_cell_source(
            reference,
            self._occupancy_root,
            self._captures_root,
            self._calibrations_root,
            address,
            expected_domain_identity=(
                None
                if expected_navigation is None
                else expected_navigation.identity
            ),
        )

    def occupancy_cell_view(
        self,
        reference: OccupancyArtifactRef,
        *,
        address: DatasetCellAddress | None = None,
    ):
        source = self._load_occupancy_cell_source(reference, address)
        from .ui.plot import occupancy_cell_session

        return occupancy_cell_session(source)

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
        return self._open_ui(
            "cell",
            self._inspect_occupancy_cell_navigation,
            self._load_occupancy_cell_source,
            reference,
            address=address,
        )


__all__ = ["OccupancyApi"]
