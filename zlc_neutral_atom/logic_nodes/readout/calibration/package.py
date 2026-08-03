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
    start_calibration_task_command,
    write_calibration_post_final_exports,
)


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> CalibrationApi:
    (
        captures_root,
        calibrations_root,
        pulses_root,
        apparatus_facts,
        resolve_device_ref,
        resolve_camera_ref,
        resolve_sequencer_ref,
        camera_port,
        pulse_port,
        start_run,
        wait_run,
        operation_guard,
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
            captures_root=captures_root,
            start_run=start_run,
        )
    profiles = {}
    for apparatus in apparatus_facts:
        camera_ref = resolve_device_ref(
            apparatus.camera_instance_id,
            "camera.capture",
        )
        sequencer_ref = resolve_device_ref(
            apparatus.sequencer_instance_id,
            "pulse.execute",
        )
        profile = build_sitemap_acquisition_profile(
            apparatus,
            camera_ref=camera_ref,
            sequencer_ref=sequencer_ref,
            camera_port=camera_port(camera_ref),
            pulse_port=pulse_port(sequencer_ref),
        )
        binding = profile.readout_binding.value
        if binding in profiles:
            raise ValueError("installation produced duplicate sitemap profile bindings")
        profiles[binding] = profile

    def write_outputs(
        source,
        calibration,
        **kwargs,
    ):
        from .ui.plot_report import export_calibration_plot_pages

        return write_calibration_post_final_exports(
            source,
            calibration,
            captures_root=captures_root,
            calibrations_root=calibrations_root,
            export_plots=export_calibration_plot_pages,
            **kwargs,
        )

    def start_calibration(
        request,
        lifecycle_owner,
        on_committed,
    ):
        plan = prepare_calibration_artifact_plan(
            request,
            captures_root=captures_root,
            calibrations_root=calibrations_root,
            on_committed=on_committed,
        )
        if lifecycle_owner is not None:
            plan = plan.with_lifecycle(
                owner=lifecycle_owner,
                preemptible=False,
            )
        return start_run(plan)

    return CalibrationApi(
        captures_root=captures_root,
        calibrations_root=calibrations_root,
        profiles=MappingProxyType(profiles),
        camera_roles=tuple(profiles),
        resolve_camera_role=resolve_camera_role,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        load_pulse=load_pulse,
        bind_capture=prepare_capture,
        wait_run=wait_run,
        operation_guard=operation_guard,
        write_outputs=write_outputs,
        start_calibration=start_calibration,
        open_ui=open_ui,
    )


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
        "captures_root",
        "calibrations_root",
        "pulses_root",
        "readout_apparatus_facts",
        "resolve_device_ref",
        "resolve_camera_ref",
        "resolve_sequencer_ref",
        "camera_port",
        "pulse_port",
        "start_run",
        "wait_run",
        "operation_guard",
        "open_ui",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    availability=_availability,
    dynamic_choice_fact="readout_camera_roles",
    start_prepared=start_calibration_task_command,
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
