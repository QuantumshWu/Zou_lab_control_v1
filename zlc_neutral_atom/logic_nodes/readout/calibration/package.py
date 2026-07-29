"""Calibration's complete built-in capability package."""

from __future__ import annotations

from types import MappingProxyType

from zlc_neutral_atom.logic_node_package import LogicNodePackage
from zlc_storage.paths import resolve_under

from .api import CalibrationApi
from .application import prepare_calibration_artifact_plan
from .declaration import CALIBRATION_LOGIC_NODE
from .installation import build_sitemap_acquisition_profile
from .task import (
    admit_calibration_capture_export,
    admit_calibration_task_output,
    write_calibration_task_outputs,
)


def _bind_api(host: object, _dependencies: tuple[object, ...]) -> CalibrationApi:
    operations = host._logic_node_operations()
    output_root = operations.output_root
    capture_repository = operations.capture_repository
    start_run = operations.start_run
    profiles = {}
    for apparatus in operations.readout_apparatus_facts:
        profile = build_sitemap_acquisition_profile(
            apparatus,
            camera_port=operations.camera_port(
                operations.device_ref(apparatus.camera_role)
            ),
            pulse_port=operations.pulse_port(
                operations.device_ref(apparatus.sequencer_role)
            ),
            pulses_root=operations.pulses_root,
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
        from .ui.report_projection import render_calibration_plot_report

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

    def start_calibration(request, calibration_repository):
        plan = prepare_calibration_artifact_plan(
            request,
            capture_repository=capture_repository,
            calibration_repository=calibration_repository,
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
        repository_path=operations.repository_root / "calibrations",
        profiles=MappingProxyType(profiles),
        camera_roles=operations.roles("camera"),
        readout_binding=host.readout_binding,
        require_binding=host.require_readout_binding,
        resolve_camera_role=host._resolve_camera_role,
        resolve_camera_ref=host.resolve_readout_camera_ref,
        resolve_sequencer_ref=host.resolve_readout_sequencer_ref,
        load_pulse=host.load_readout_pulse,
        bind_capture=host.prepare_capture,
        wait_run=operations.wait_run,
        admit_capture=capture_repository.admit,
        admit_saved_capture=admit_saved_capture,
        write_outputs=write_outputs,
        start_calibration=start_calibration,
        load_calibration=load_calibration,
        admit_saved_calibration=admit_saved_calibration,
    )


def _close_api(api: CalibrationApi) -> tuple[Exception, ...]:
    return api.close()


def _bind_task_console(api: CalibrationApi, _catalog: object, projection):
    from zlc_neutral_atom.processing.signal_plane import SignalPublication
    from zlc_neutral_atom.runtime.hosted_run import HostedRun
    from .projection import CALIBRATION_FINAL_OUTPUT_DECLARATIONS
    from .ui.view_projection import calibration_site_map_figure
    from .workbench_adapter import start_calibration_task_command

    site_map_name = CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0].name

    def project_signal_presentation(
        node: object,
        output_name: str,
        publication: SignalPublication,
    ):
        if output_name != site_map_name:
            return None
        if not isinstance(node, HostedRun):
            raise TypeError("Calibration presentation requires HostedRun")
        command = node.prepared_command
        result = node.final_result
        value = publication.value(node.signal_key(output_name))
        if command is None or result is None or value is None:
            return None
        context = command.site_map_context(result)
        figure, source = calibration_site_map_figure(
            value.snapshot,
            context,
            coherence_identity=value.join_digest,
            run_id=value.run_id,
            provenance_epoch_id=value.epoch_id,
        )
        return figure, source.site_map

    return projection.run(
        CALIBRATION_LOGIC_NODE,
        prepare=api.prepare_calibration_task,
        dynamic_choice_context=api.sitemap_camera_roles(),
        start_with_live_output=start_calibration_task_command,
        project_signal_presentation=project_signal_presentation,
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="calibration",
    declaration=CALIBRATION_LOGIC_NODE,
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=60,
    close_api=_close_api,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
