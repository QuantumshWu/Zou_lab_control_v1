"""Calibration's complete built-in capability package."""

from __future__ import annotations

from types import MappingProxyType

from zlc_neutral_atom.capture.application import prepare_finite_capture
from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    UiContributionDescriptor,
)
from zlc_pulse import PulseDocument, load_pulse_document
from zlc_storage.paths import resolve_under

from .api import CalibrationApi
from .application import prepare_calibration_artifact_plan
from .declaration import CALIBRATION_LOGIC_NODE
from .installation import build_sitemap_acquisition_profile
from .task import (
    admit_calibration_capture_export,
    admit_calibration_task_output,
    start_calibration_task_command,
    write_calibration_task_outputs,
)


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> CalibrationApi:
    (
        repository_root,
        pulses_root,
        output_root,
        capture_repository,
        apparatus_facts,
        resolve_camera_ref,
        resolve_sequencer_ref,
        camera_port,
        pulse_port,
        start_run,
        wait_run,
        open_ui,
    ) = facts

    def resolve_camera_role(requested):
        return resolve_camera_ref(requested).role

    def load_pulse(value):
        if isinstance(value, PulseDocument):
            return value
        return load_pulse_document(resolve_under(pulses_root, value))

    def prepare_capture(request):
        return prepare_finite_capture(
            request,
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            repository=capture_repository,
            start_run=start_run,
        )
    profiles = {}
    for apparatus in apparatus_facts:
        profile = build_sitemap_acquisition_profile(
            apparatus,
            camera_port=camera_port(resolve_camera_ref(apparatus.camera_role)),
            pulse_port=pulse_port(
                resolve_sequencer_ref(apparatus.sequencer_role)
            ),
            pulses_root=pulses_root,
        )
        binding = profile.readout_binding.value
        if binding in profiles:
            raise ValueError("installation produced duplicate sitemap profile bindings")
        profiles[binding] = profile

    def admit_saved_capture(path, expected_camera_role):
        return admit_calibration_capture_export(
            resolve_under(output_root, path),
            expected_camera_role=expected_camera_role,
            capture_repository=capture_repository,
        )

    def write_outputs(
        source,
        calibration,
        calibration_repository,
        **kwargs,
    ):
        from .ui.workbench_jobs import render_calibration_plot_report

        options = dict(kwargs)
        options["folder"] = str(resolve_under(output_root, options["folder"]))
        return write_calibration_task_outputs(
            source,
            calibration,
            capture_repository=capture_repository,
            calibration_repository=calibration_repository,
            render_report=render_calibration_plot_report,
            **options,
        )

    def start_calibration(
        request,
        calibration_repository,
        lifecycle_owner,
    ):
        plan = prepare_calibration_artifact_plan(
            request,
            capture_repository=capture_repository,
            calibration_repository=calibration_repository,
        )
        if lifecycle_owner is not None:
            plan = plan.with_lifecycle(
                owner=lifecycle_owner,
                preemptible=False,
            )
        return start_run(plan)

    def load_calibration(reference, calibration_repository):
        return calibration_repository.admit(reference, capture_repository)

    def admit_saved_calibration(path, calibration_repository):
        return admit_calibration_task_output(
            path,
            capture_repository=capture_repository,
            calibration_repository=calibration_repository,
        )

    return CalibrationApi(
        repository_path=repository_root / "calibrations",
        profiles=MappingProxyType(profiles),
        camera_roles=tuple(profiles),
        resolve_camera_role=resolve_camera_role,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        load_pulse=load_pulse,
        bind_capture=prepare_capture,
        wait_run=wait_run,
        admit_capture=capture_repository.admit,
        admit_saved_capture=admit_saved_capture,
        write_outputs=write_outputs,
        start_calibration=start_calibration,
        load_calibration=load_calibration,
        admit_saved_calibration=admit_saved_calibration,
        open_ui=open_ui,
    )


def _project_signal_presentation(node, output_name, publication):
    from .ui.workbench_jobs import project_calibration_signal_presentation

    return project_calibration_signal_presentation(node, output_name, publication)


def _prepare_hosted(api, request, event_source):
    if event_source is not None:
        raise ValueError("Calibration has no event-associated input")
    return api.prepare_calibration_task(request)


def _availability(_catalog, apparatus):
    return None if apparatus else "no installed readout apparatus"


def _close_api(api: CalibrationApi) -> tuple[Exception, ...]:
    return api.close()


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="calibration",
    declaration=CALIBRATION_LOGIC_NODE,
    api_requirements=(
        "repository_root",
        "pulses_root",
        "output_root",
        "capture_repository",
        "readout_apparatus_facts",
        "resolve_camera_ref",
        "resolve_sequencer_ref",
        "camera_port",
        "pulse_port",
        "start_run",
        "wait_run",
        "open_ui",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    availability=_availability,
    dynamic_choice_fact="readout_camera_roles",
    start_prepared=start_calibration_task_command,
    project_signal_presentation=_project_signal_presentation,
    ui_contributions=(
        UiContributionDescriptor(
            "create",
            "zlc_neutral_atom.logic_nodes.readout.calibration.ui."
            "workbench_window",
            "CalibrationWorkbenchWindow",
        ),
        UiContributionDescriptor(
            "report",
            "zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_window",
            "CalibrationReportWindow",
        ),
    ),
    close_api=_close_api,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
