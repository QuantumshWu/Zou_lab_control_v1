"""Explicit static composition of the built-in Readout notebook surface.

This is the sole ordinary-import list of notebook capability adapters.  It
performs no package scan, registration, string lookup, dynamic type creation,
or late replacement.
"""

from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.capture.application import (
    CaptureRequest,
    PreparedFiniteCapture,
    prepare_finite_capture,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CameraMeasurementRequest,
    DEFAULT_CAMERA_MEASUREMENT_ROLE,
    PreparedFiniteCameraMeasurement,
    PreparedLiveCameraMeasurement,
    prepare_finite_camera_measurement,
    prepare_live_camera_measurement,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.notebook_adapter import (
    CameraMeasurementNotebookAdapter,
    CameraMeasurementNotebookHost,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.application import (
    PreparedExactScan,
    prepare_exact_scan,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import PulseScanBoundRequest
from zlc_neutral_atom.logic_nodes.pulse_scan.notebook_adapter import (
    PulseScanNotebookAdapter,
    PulseScanNotebookHost,
)
from zlc_neutral_atom.logic_nodes.mot_field import (
    DEFAULT_MOT_FIELD_CAMERA_ROLE,
    MotFieldRequest,
    MotFieldTaskIntent,
    PreparedMotFieldAcquisition,
    PreparedMotFieldTask,
    prepare_mot_field_acquisition,
    prepare_mot_field_task,
)
from zlc_neutral_atom.logic_nodes.mot_field.notebook_adapter import (
    MotFieldNotebookAdapter,
    MotFieldNotebookHost,
)
from zlc_neutral_atom.logic_nodes.readout.duration_fidelity import (
    PreparedReadoutDurationFidelity,
    ReadoutDurationFidelityRequest,
    prepare_readout_duration_fidelity,
)
from zlc_neutral_atom.logic_nodes.readout.duration_fidelity.notebook_adapter import (
    ReadoutDurationFidelityNotebookAdapter,
    ReadoutDurationFidelityNotebookHost,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
    CalibrationArtifactRequest,
    prepare_calibration_artifact_plan,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.notebook_adapter import (
    CalibrationNotebookAdapter,
    CalibrationNotebookHost,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
    SitemapAcquisitionProfile,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.installation import (
    build_sitemap_acquisition_profile,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.task import (
    CalibrationTaskIntent,
    PreparedCalibrationTask,
    admit_calibration_capture_export,
    admit_calibration_task_output,
    prepare_calibration_task,
    write_calibration_task_outputs,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.application import (
    DetectionRequest,
    build_detection_request,
    prepare_detection_plan,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import (
    inspect_occupancy_cell_domain,
    load_exact_occupancy_cell_source,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.notebook_adapter import (
    OccupancyNotebookAdapter,
    OccupancyNotebookHost,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import (
    OccupancyArtifactRef,
)
from zlc_neutral_atom.logic_nodes.release_recapture.temperature import (
    PreparedReleaseRecapture,
    TemperatureReleaseRecaptureRequest,
    prepare_temperature_release_recapture,
)
from zlc_neutral_atom.logic_nodes.release_recapture.temperature.notebook_adapter import (
    TemperatureReleaseRecaptureNotebookAdapter,
    TemperatureReleaseRecaptureNotebookHost,
)
from zlc_neutral_atom.logic_nodes.release_recapture.grey_molasses_detuning import (
    AutonomousMeasurementUnavailable,
    GREY_MOLASSES_CAPABILITY_GAP,
    GreyMolassesDetuningRequest,
    prepare_grey_molasses_detuning,
)
from zlc_neutral_atom.logic_nodes.release_recapture.grey_molasses_detuning.notebook_adapter import (
    GreyMolassesDetuningNotebookAdapter,
    GreyMolassesDetuningNotebookHost,
)

from ._readout_core import ReadoutCoreFacade, ReadoutCoreHost
from ._application_services import (
    ExperimentServices,
    load_project_pulse,
    resolve_role,
    service_guard,
)
from ._readout_repositories import ReadoutApplicationResources


def compose_readout_resources(
    repository_root: Path,
    installation,
    *,
    catalog,
    runtime,
) -> ReadoutApplicationResources:
    """Compose readout-only installation facts and lazy repositories."""

    sitemap_profiles: dict[str, SitemapAcquisitionProfile] = {}
    for apparatus in installation.readout_apparatus_facts:
        camera_info = catalog.require(apparatus.camera_role)
        sequencer_info = catalog.require(apparatus.sequencer_role)
        profile = build_sitemap_acquisition_profile(
            apparatus,
            camera_port=runtime.camera_port(camera_info.ref),
            pulse_port=runtime.pulse_port(sequencer_info.ref),
        )
        binding = profile.readout_binding.value
        if binding in sitemap_profiles:
            raise ValueError("installation produced duplicate sitemap profile bindings")
        sitemap_profiles[binding] = profile
    from zlc_neutral_atom.logic_nodes.pulse_scan.repository import ScanRepository

    return ReadoutApplicationResources(
        calibration_repository_path=repository_root / "calibrations",
        occupancy_repository_path=repository_root / "occupancy",
        sitemap_profiles=sitemap_profiles,
        camera_signal_association_authorities=(
            installation.camera_signal_association_authorities
        ),
        scan_repository=ScanRepository(repository_root / "scans"),
    )


class ReadoutFacade(
    CameraMeasurementNotebookAdapter,
    TemperatureReleaseRecaptureNotebookAdapter,
    ReadoutDurationFidelityNotebookAdapter,
    GreyMolassesDetuningNotebookAdapter,
    MotFieldNotebookAdapter,
    PulseScanNotebookAdapter,
    CalibrationNotebookAdapter,
    OccupancyNotebookAdapter,
    ReadoutCoreFacade,
):
    """One statically composed notebook surface for the built-in installation."""

    __slots__ = ("_services", "_binding")

    def __init__(
        self,
        services: ExperimentServices,
        binding: ReadoutBindingKey | None = None,
    ) -> None:
        if not isinstance(services, ExperimentServices):
            raise TypeError("services must be ExperimentServices")
        if binding is not None and not isinstance(binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey or None")
        self._services = services
        self._binding = binding

    @property
    def _readout_core_host(self) -> ReadoutCoreHost:
        return self

    @property
    def readout_binding(self) -> ReadoutBindingKey | None:
        return self._binding

    def bind_readout(self, binding: ReadoutBindingKey) -> ReadoutCoreFacade:
        with self._borrow_services() as guarded:
            info = guarded.catalog.require(binding.value)
            if info.domain != "camera":
                raise ValueError(f"readout binding {binding.value!r} is not a camera")
        return ReadoutFacade(self._services, binding)

    def _borrow_services(self):
        return service_guard(self._services)

    def _resolve_role(self, requested, domain, preferred):
        with self._borrow_services() as guarded:
            return resolve_role(guarded.catalog, requested, domain, preferred)

    def _resolve_camera_role(self, requested):
        if self._binding is not None:
            if requested is not None and requested != self._binding.value:
                raise ValueError("bound readout facade cannot target another camera")
            requested = self._binding.value
        return self._resolve_role(
            requested,
            "camera",
            (DEFAULT_CAMERA_MEASUREMENT_ROLE, "readout", "camera"),
        )

    def load_readout_pulse(self, value):
        return load_project_pulse(value)

    def resolve_readout_camera_ref(self, requested):
        role = self._resolve_camera_role(requested)
        with self._borrow_services() as guarded:
            return guarded.catalog.require(role).ref

    def resolve_readout_sequencer_ref(self, requested):
        role = self._resolve_role(requested, "sequencer", ("sequencer",))
        with self._borrow_services() as guarded:
            return guarded.catalog.require(role).ref

    def bind_finite_capture(self, request: CaptureRequest) -> PreparedFiniteCapture:
        with self._borrow_services() as guarded:
            return prepare_finite_capture(
                request,
                pulse_port=guarded.runtime.pulse_port(request.sequencer_ref),
                camera_port=guarded.runtime.camera_port(request.camera_ref),
                repository=guarded.capture_repository,
                start_run=guarded.runtime.start,
            )

    def wait_readout_run(self, handle):
        with self._borrow_services() as guarded:
            runtime = guarded.runtime
        return runtime.wait(handle)

    def load_capture_artifact(self, reference):
        with self._borrow_services() as guarded:
            return guarded.capture_repository.load(reference)

    def materialize_capture_artifact(self, reference):
        with self._borrow_services() as guarded:
            return guarded.capture_repository.materialize_final(reference)

    @property
    def _pulse_scan_notebook_host(self) -> PulseScanNotebookHost:
        return self

    def bind_pulse_scan_source(
        self,
        request: PulseScanBoundRequest,
        source,
        *,
        sequencer_role: str | None,
    ) -> PreparedExactScan:
        role = self._resolve_role(
            sequencer_role,
            "sequencer",
            ("sequencer",),
        )
        with self._borrow_services() as guarded:
            sequencer_ref = guarded.catalog.require(role).ref
            return prepare_exact_scan(
                request,
                source,
                pulse_port=guarded.runtime.pulse_port(sequencer_ref),
                repository=guarded.readout_resources.scan_repository,
                start_run=guarded.runtime.start,
            )

    def load_pulse_scan(self, reference):
        with self._borrow_services() as guarded:
            return guarded.readout_resources.scan_repository.admit(reference)

    def materialize_pulse_scan(self, reference):
        with self._borrow_services() as guarded:
            return guarded.readout_resources.scan_repository.materialize(reference)

    @property
    def _camera_measurement_notebook_host(
        self,
    ) -> CameraMeasurementNotebookHost:
        return self

    def resolve_camera_measurement_ref(self, requested_role):
        return self.resolve_readout_camera_ref(requested_role)

    def bind_camera_measurement(
        self,
        request: CameraMeasurementRequest,
    ) -> PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement:
        with self._borrow_services() as guarded:
            if request.repeat == 0:
                return prepare_live_camera_measurement(
                    request,
                    monitor_port=guarded.runtime.camera_monitor_port(request.camera_ref),
                    start_run=guarded.runtime.start,
                    association_authority=(
                        guarded.readout_resources.camera_signal_association_authorities.get(
                            request.camera_ref.role
                        )
                    ),
                )
            return prepare_finite_camera_measurement(
                request,
                camera_port=guarded.runtime.camera_port(request.camera_ref),
                repository=guarded.capture_repository,
                start_run=guarded.runtime.start,
            )

    @property
    def _temperature_notebook_host(
        self,
    ) -> TemperatureReleaseRecaptureNotebookHost:
        return self

    def load_temperature_pulse(self, value):
        return self.load_readout_pulse(value)

    def resolve_temperature_camera_ref(self, requested):
        return self.resolve_readout_camera_ref(requested)

    def resolve_temperature_sequencer_ref(self, requested):
        return self.resolve_readout_sequencer_ref(requested)

    def bind_temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ) -> PreparedReleaseRecapture:
        with self._borrow_services() as guarded:
            calibration = guarded.readout_resources.calibration_repository().admit(
                request.calibration_ref,
                guarded.capture_repository,
            )
            return prepare_temperature_release_recapture(
                request,
                calibration,
                pulse_port=guarded.runtime.pulse_port(request.sequencer_ref),
                camera_port=guarded.runtime.camera_port(request.camera_ref),
                start_run=guarded.runtime.start,
            )

    def wait_temperature_release_recapture(self, handle):
        return self.wait_readout_run(handle)

    @property
    def _duration_fidelity_notebook_host(
        self,
    ) -> ReadoutDurationFidelityNotebookHost:
        return self

    def load_duration_fidelity_pulse(self, value):
        return self.load_readout_pulse(value)

    def resolve_duration_fidelity_camera_ref(self, requested):
        return self.resolve_readout_camera_ref(requested)

    def resolve_duration_fidelity_sequencer_ref(self, requested):
        return self.resolve_readout_sequencer_ref(requested)

    def bind_readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ) -> PreparedReadoutDurationFidelity:
        with self._borrow_services() as guarded:
            calibration = guarded.readout_resources.calibration_repository().admit(
                request.calibration_ref,
                guarded.capture_repository,
            )
            return prepare_readout_duration_fidelity(
                request,
                calibration,
                pulse_port=guarded.runtime.pulse_port(request.sequencer_ref),
                camera_port=guarded.runtime.camera_port(request.camera_ref),
                start_run=guarded.runtime.start,
            )

    def wait_readout_duration_fidelity(self, handle):
        return self.wait_readout_run(handle)

    @property
    def _grey_molasses_notebook_host(
        self,
    ) -> GreyMolassesDetuningNotebookHost:
        return self

    def load_grey_molasses_pulse(self, value):
        return self.load_readout_pulse(value)

    def resolve_grey_molasses_camera_ref(self, requested):
        return self.resolve_readout_camera_ref(requested)

    def resolve_grey_molasses_sequencer_ref(self, requested):
        return self.resolve_readout_sequencer_ref(requested)

    def resolve_grey_molasses_rf_role(self, requested):
        with self._borrow_services() as guarded:
            if not guarded.catalog.roles("rf"):
                return requested
        return self._resolve_role(requested, "rf", ("rf",))

    def bind_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> PreparedReleaseRecapture:
        with self._borrow_services() as guarded:
            rf_info = guarded.catalog.find(request.rf_role)
            if rf_info is None or rf_info.domain != "rf":
                raise AutonomousMeasurementUnavailable(
                    GREY_MOLASSES_CAPABILITY_GAP
                )
            calibration = guarded.readout_resources.calibration_repository().admit(
                request.calibration_ref,
                guarded.capture_repository,
            )
            return prepare_grey_molasses_detuning(
                request,
                calibration,
                pulse_port=guarded.runtime.pulse_port(request.sequencer_ref),
                camera_port=guarded.runtime.camera_port(request.camera_ref),
                rf_port=guarded.runtime.rf_port(rf_info.ref),
                start_run=guarded.runtime.start,
            )

    @property
    def _mot_field_notebook_host(self) -> MotFieldNotebookHost:
        return self

    def load_mot_field_pulse(self, value):
        return self.load_readout_pulse(value)

    def resolve_mot_camera_ref(self, requested):
        role = self._resolve_role(
            requested,
            "camera",
            (DEFAULT_MOT_FIELD_CAMERA_ROLE,),
        )
        if role != DEFAULT_MOT_FIELD_CAMERA_ROLE:
            raise ValueError(
                "MOT field optimization requires the installation's "
                "'mot_camera' role; an arbitrary camera is not a "
                "coil-sensitive exact-scan sensor"
            )
        with self._borrow_services() as guarded:
            return guarded.catalog.require(role).ref

    def resolve_mot_sequencer_ref(self, requested):
        return self.resolve_readout_sequencer_ref(requested)

    def bind_mot_field_acquisition(
        self,
        request: MotFieldRequest,
    ) -> PreparedMotFieldAcquisition:
        with self._borrow_services() as guarded:
            return prepare_mot_field_acquisition(
                request,
                pulse_port=guarded.runtime.pulse_port(request.sequencer_ref),
                camera_port=guarded.runtime.camera_port(request.camera_ref),
            )

    def bind_mot_field_task(
        self,
        intent: MotFieldTaskIntent,
        dependencies,
    ) -> PreparedMotFieldTask:
        with self._borrow_services() as guarded:
            return prepare_mot_field_task(
                intent,
                dependencies,
                capture_repository=guarded.capture_repository,
                start_run=guarded.runtime.start,
            )

    @property
    def _calibration_notebook_host(self) -> CalibrationNotebookHost:
        return self

    def resolve_sitemap_profile(
        self,
        camera_role,
    ) -> tuple[str, SitemapAcquisitionProfile]:
        selected = self._resolve_camera_role(camera_role)
        with self._borrow_services() as guarded:
            try:
                profile = guarded.readout_resources.sitemap_profiles[selected]
            except KeyError as exc:
                raise ValueError(
                    f"camera role {selected!r} has no sitemap profile"
                ) from exc
        if not isinstance(profile, SitemapAcquisitionProfile):
            raise TypeError("experiment composition contains an invalid sitemap profile")
        if profile.readout_binding != ReadoutBindingKey(selected):
            raise ValueError("composed sitemap profile differs from the selected camera")
        return selected, profile

    def available_sitemap_camera_roles(self) -> tuple[str, ...]:
        with self._borrow_services() as guarded:
            roles = tuple(guarded.readout_resources.sitemap_profiles)
            cameras = set(guarded.catalog.roles("camera"))
        if any(role not in cameras for role in roles):
            raise RuntimeError(
                "installation sitemap capabilities differ from its camera catalog"
            )
        return roles

    def load_calibration_pulse(self, value):
        return self.load_readout_pulse(value)

    def calibration_camera_ref(self, role):
        with self._borrow_services() as guarded:
            return guarded.catalog.require(role).ref

    def calibration_sequencer_ref(self, role):
        resolved = self._resolve_role(role, "sequencer", ("sequencer",))
        with self._borrow_services() as guarded:
            return guarded.catalog.require(resolved).ref

    def run_calibration_capture(self, request: CaptureRequest):
        return self.wait_readout_run(self.bind_finite_capture(request).start())

    def bind_calibration_task(
        self,
        intent: CalibrationTaskIntent,
        application,
    ) -> PreparedCalibrationTask:
        return prepare_calibration_task(intent, application)

    def admit_saved_calibration_capture_source(
        self,
        source_path,
        *,
        expected_camera_role,
    ):
        binding = ReadoutBindingKey(expected_camera_role)
        self._require_binding(binding)
        with self._borrow_services() as guarded:
            info = guarded.catalog.require(binding.value)
            if info.domain != "camera":
                raise ValueError(
                    f"saved calibration binding {binding.value!r} is not a camera"
                )
            return admit_calibration_capture_export(
                source_path,
                expected_camera_role=binding.value,
                capture_repository=guarded.capture_repository,
            )

    def write_calibration_outputs(
        self,
        source,
        calibration,
        *,
        folder,
        frame_export_policy,
        expected_camera_role,
    ) -> None:
        binding = (
            self._binding
            if expected_camera_role is None
            else ReadoutBindingKey(expected_camera_role)
        )
        if binding is not None:
            self._require_binding(binding)
        with self._borrow_services() as guarded:
            if binding is not None:
                info = guarded.catalog.require(binding.value)
                if info.domain != "camera":
                    raise ValueError(
                        f"calibration binding {binding.value!r} is not a camera"
                    )
            write_calibration_task_outputs(
                source,
                calibration,
                folder=folder,
                frame_export_policy=frame_export_policy,
                capture_repository=guarded.capture_repository,
                calibration_repository=(
                    guarded.readout_resources.calibration_repository()
                ),
                expected_camera_role=None if binding is None else binding.value,
                render_report=self._render_calibration_report,
            )

    @staticmethod
    def _render_calibration_report(view):
        from zlc_frontend import render_plot_report
        from zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_projection import (
            project_calibration_plot_report,
        )

        return render_plot_report(project_calibration_plot_report(view))

    def admit_calibration_capture(self, source):
        with self._borrow_services() as guarded:
            return guarded.capture_repository.admit(source)

    def start_calibration_request(
        self,
        request: CalibrationArtifactRequest,
    ):
        with self._borrow_services() as guarded:
            plan = prepare_calibration_artifact_plan(
                request,
                capture_repository=guarded.capture_repository,
                calibration_repository=(
                    guarded.readout_resources.calibration_repository()
                ),
            )
            return guarded.runtime.start(plan)

    def run_calibration_request(self, request: CalibrationArtifactRequest):
        handle = self.start_calibration_request(request)
        return self.wait_readout_run(handle)

    def open_calibration_request_gui(self, request):
        from Zou_lab_control.workbench import open_calibration_workbench

        return open_calibration_workbench(
            self.read_calibration_computation,
            self.start_calibration,
            request=request,
        )

    def open_calibration_edit_gui(self, reference):
        from Zou_lab_control.workbench import open_calibration_workbench

        return open_calibration_workbench(
            self.read_calibration_computation,
            self.start_calibration,
            reference=reference,
        )

    def admit_calibration_artifact(self, reference):
        with self._borrow_services() as guarded:
            resolved = guarded.readout_resources.calibration_repository().admit(
                reference,
                guarded.capture_repository,
            )
        self._require_binding(resolved.artifact.frame_contract.binding)
        return resolved

    def admit_saved_calibration_pointer(self, path):
        with self._borrow_services() as guarded:
            resolved = admit_calibration_task_output(
                path,
                capture_repository=guarded.capture_repository,
                calibration_repository=(
                    guarded.readout_resources.calibration_repository()
                ),
            )
        self._require_binding(resolved.artifact.frame_contract.binding)
        return resolved

    def read_calibration_computation(self, reference):
        with self._borrow_services() as guarded:
            computation = guarded.readout_resources.calibration_repository().load_computation(
                reference
            )
        self._require_binding(computation.artifact.frame_contract.binding)
        return computation

    def open_calibration_report_gui(self, reference):
        from Zou_lab_control.workbench import open_calibration_report_workbench

        return open_calibration_report_workbench(
            self.read_calibration_computation,
            reference,
        )

    @property
    def _occupancy_notebook_host(self) -> OccupancyNotebookHost:
        return self

    def resolve_occupancy_calibration(self, reference):
        with self._borrow_services() as guarded:
            return guarded.readout_resources.calibration_repository().admit(
                reference,
                guarded.capture_repository,
            )

    def load_saved_occupancy_calibration(self, path):
        """Explicit composition bridge to the Calibration pointer owner."""

        return self.admit_saved_calibration_pointer(path)

    def build_occupancy_detection_request(
        self,
        source,
        calibration,
        *,
        model_kind,
    ) -> DetectionRequest:
        with self._borrow_services() as guarded:
            return build_detection_request(
                guarded.capture_repository.admit(source),
                guarded.readout_resources.calibration_repository().admit(
                    calibration,
                    guarded.capture_repository,
                ),
                model_kind=model_kind,
            )

    def start_occupancy_detection(self, request: DetectionRequest):
        with self._borrow_services() as guarded:
            plan = prepare_detection_plan(
                request,
                capture_repository=guarded.capture_repository,
                calibration_repository=(
                    guarded.readout_resources.calibration_repository()
                ),
                occupancy_repository=(
                    guarded.readout_resources.occupancy_repository()
                ),
            )
            return guarded.runtime.start(plan)

    def run_occupancy_detection(self, request: DetectionRequest):
        return self.wait_readout_run(self.start_occupancy_detection(request))

    def admit_occupancy_artifact(self, reference: OccupancyArtifactRef):
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        with self._borrow_services() as guarded:
            return guarded.readout_resources.occupancy_repository().admit(
                reference,
                guarded.capture_repository,
                guarded.readout_resources.calibration_repository(),
            )

    def inspect_occupancy_navigation(self, reference):
        with self._borrow_services() as guarded:
            return inspect_occupancy_cell_domain(
                reference,
                guarded.readout_resources.occupancy_repository(),
                guarded.capture_repository,
                guarded.readout_resources.calibration_repository(),
            )

    def compose_occupancy_cell_source(
        self,
        reference,
        selection,
        *,
        expected_navigation,
    ):
        with self._borrow_services() as guarded:
            source = load_exact_occupancy_cell_source(
                reference,
                guarded.readout_resources.occupancy_repository(),
                guarded.capture_repository,
                guarded.readout_resources.calibration_repository(),
                selection,
                expected_domain_identity=(
                    None
                    if expected_navigation is None
                    else expected_navigation.identity
                ),
            )
        self._require_binding(source.domain.readout_binding)
        from zlc_neutral_atom.logic_nodes.readout.occupancy.ui.view_projection import (
            build_exact_occupancy_cell_view,
        )

        return build_exact_occupancy_cell_view(source)

    def open_occupancy_cell_gui(self, reference, selection):
        from Zou_lab_control.workbench import open_occupancy_cell_workbench

        return open_occupancy_cell_workbench(
            self._inspect_occupancy_cell_navigation,
            self._load_occupancy_cell_source,
            reference,
            selection=selection,
        )


__all__ = ["ReadoutFacade", "compose_readout_resources"]
