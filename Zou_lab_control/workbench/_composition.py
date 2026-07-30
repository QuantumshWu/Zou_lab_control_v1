"""Application-owned adapters for desktop Workbench surfaces.

This module is imported only by the public functions in
``Zou_lab_control.workbench``.  It is the single place where an ``Experiment``
is decomposed into the closed ports that desktop shells are allowed to retain.
No object defined here is a service locator: every operation exposed to a
Workbench is named in its concrete port type.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from zlc_neutral_atom.installation_config import InstallationConfigDocument
from zlc_pulse import PulseDocument


def _require_experiment(value):
    from Zou_lab_control.api.facade import Experiment

    if not isinstance(value, Experiment):
        raise TypeError("experiment must be the current Experiment")
    return value


def task_console_ports(experiment):
    """Decompose one Experiment into TaskConsole's explicit application port."""

    experiment = _require_experiment(experiment)
    from zlc_workbench.task_console.application_ports import (
        TaskConsoleApplicationPorts,
    )
    from zlc_workbench.task_console.artifact_resolution import (
        resolve_final_or_saved_artifact,
    )
    from zlc_neutral_atom.catalog import ProcessorDefinition
    from Zou_lab_control.workbench import _build_logic_node_ui
    from zlc_workbench.task_console.declaration_projection import (
        project_processor_declaration,
        project_run_declaration,
    )

    workspace = experiment._workspace_paths()
    path_roots = {
        "pulses": workspace.pulses_root,
        "tasks": workspace.tasks_root,
        "output": workspace.output_root,
    }
    attachments = []
    for package, api, dynamic_choices in experiment.nodes._composition_entries():
        declaration = package.declaration

        bind_request = declaration.bind_request
        if package.bind_hosted_request is not None:
            bind_request = (
                lambda authored, inputs, owner=package.bind_hosted_request,
                current_api=api: owner(current_api, authored, inputs)
            )

        resolve_artifact = None
        if package.resolve_artifact_reference is not None:
            resolve_artifact = (
                lambda binding, owner=package.resolve_artifact_reference,
                current_api=api: owner(
                    current_api,
                    binding,
                    resolve_final_or_saved_artifact,
                )
            )

        editor_builder = None
        if any(
            value.purpose == "task_console_editor"
            for value in package.ui_contributions
        ):
            editor_builder = (
                lambda form, contributions=package.ui_contributions: _build_logic_node_ui(
                    contributions,
                    "task_console_editor",
                    form,
                    pulses_root=workspace.pulses_root,
                )
            )

        presenter = package.project_signal_presentation
        if isinstance(declaration.definition, ProcessorDefinition):
            attachments.append(
                project_processor_declaration(
                    declaration,
                    bind_request=bind_request,
                    prepare=(
                        lambda request, owner=package.prepare_hosted,
                        current_api=api: owner(current_api, request, None)
                    ),
                    project_signal_presentation=presenter,
                    dynamic_choices=dynamic_choices,
                    resolve_artifact_reference=resolve_artifact,
                    path_roots=path_roots,
                )
            )
            continue

        start_prepared = None
        if package.start_prepared is not None:
            start_prepared = (
                lambda command, live_host, command_context,
                owner=package.start_prepared: owner(
                    command,
                    live_host,
                    command_context,
                )
            )
        attachments.append(
            project_run_declaration(
                declaration,
                prepare=(
                    lambda request, event_source, owner=package.prepare_hosted,
                    current_api=api: owner(current_api, request, event_source)
                ),
                bind_request=bind_request,
                dynamic_choices=dynamic_choices,
                resolve_artifact_reference=resolve_artifact,
                start_prepared=start_prepared,
                editor_builder=editor_builder,
                path_roots=path_roots,
                project_signal_presentation=presenter,
            )
        )

    return TaskConsoleApplicationPorts(
        attachments=tuple(attachments),
        data_plane=experiment._signal_data_plane(),
        tasks_root=workspace.tasks_root,
        output_root=workspace.output_root,
    )


def open_fit_capable_figure_gui(
    experiment,
    display_source,
    fit_source,
    *,
    intent,
    point_ordinals,
    preferences,
    artifact_output,
    selected_model,
    initial_fit_spec,
    initial_selection,
    open_fit,
    timeout_seconds,
    initial_fit_result_identity,
    direct_fit_single_panel,
):
    """Compose the one Figure-owned Fit host from application-owned ports."""

    experiment = _require_experiment(experiment)
    artifact_operations = experiment._artifact_operations
    services_owner = experiment._services

    from Zou_lab_control.api._application_services import (
        fit_service_guard,
        service_guard,
    )
    from Zou_lab_control.api._dataset_sources import project_final_dataset_source
    from Zou_lab_control.api._figure_projection import data_figure_for_services
    from zlc_data import FitResultBatch, FitSpec
    from zlc_neutral_atom.artifacts import execute_fit

    def source_schema():
        with service_guard(services_owner):
            return project_final_dataset_source(
                artifact_operations,
                fit_source,
                materialize=False,
            ).schema

    def prepare_fit(
        visible_figure,
        authority_selection,
        histogram_projection,
    ):
        schema = source_schema()
        seed_spec = initial_fit_spec
        if (
            seed_spec is not None
            and seed_spec.committed_transform.source_schema_fingerprint
            != schema.fingerprint
        ):
            raise ValueError("initial FitSpec belongs to another source schema")
        from zlc_frontend import DataFigure, prepare_fit_authoring_options

        if not isinstance(visible_figure, DataFigure):
            raise TypeError("Figure Fit requires its exact visible DataFigure")
        layer = visible_figure.document.layers[0]
        visible_schema = visible_figure.datasets.resolve(layer.dataset_id).block.schema
        if visible_schema.fingerprint != schema.fingerprint:
            raise ValueError("visible Figure belongs to another Fit source schema")
        options = prepare_fit_authoring_options(
            visible_figure,
            authority_selection,
            seed_spec=seed_spec,
            histogram_projection=histogram_projection,
        )
        if not options:
            raise ValueError(
                "the displayed named axes and selection have no compatible Fit model"
            )
        if selected_model is not None and selected_model not in {
            option.spec.model_id for option in options
        }:
            raise ValueError(
                f"Fit model {selected_model!r} is not compatible with this panel"
            )
        return tuple(options)

    def execute_fit(spec, cancel_check, deadline_monotonic):
        if not isinstance(spec, FitSpec):
            raise TypeError("Figure Fit execution requires FitSpec")
        with fit_service_guard(services_owner) as services:

            def combined_cancel_check() -> bool:
                with services.operation_lock:
                    closing = services.state != "OPEN"
                return closing or bool(cancel_check())

            return execute_fit(
                artifact_operations,
                fit_source,
                spec,
                cancel_check=combined_cancel_check,
                deadline_monotonic=deadline_monotonic,
            )

    def save_fit_result(result):
        if not isinstance(result, FitResultBatch):
            raise TypeError("Fit save requires FitResultBatch")
        return experiment.save_fit(fit_source, result)

    def reload_fit_result(reference):
        admitted = experiment.load_fit(reference)
        if admitted.source_artifact_ref != fit_source:
            raise ValueError("saved Fit reopened against another source artifact")
        return admitted.result

    if direct_fit_single_panel and display_source != fit_source:
        raise ValueError(
            "direct Fit single-panel display requires its exact source artifact"
        )

    def figure_factory(source, *, intent, point_ordinals, preferences):
        if direct_fit_single_panel and source != fit_source:
            raise ValueError("direct Fit Figure loader received another source")
        with service_guard(services_owner) as services:
            figure, figure_intent = data_figure_for_services(
                services,
                artifact_operations,
                source,
                intent=intent,
                point_ordinals=point_ordinals,
                preferences=preferences,
                artifact_output=(
                    None if direct_fit_single_panel else artifact_output
                ),
            )
        if not direct_fit_single_panel:
            return figure, figure_intent
        from zlc_frontend.fit_projection import (
            evaluated_figure_panels,
            panel_focus_address,
        )

        panels = evaluated_figure_panels(figure.evaluated)
        if len(panels) <= 1:
            return figure, figure_intent
        layer, cell, series_group = panels[0]
        focused = figure.focused_typed_panel(
            0,
            expected_address=panel_focus_address(layer, cell, series_group),
            expected_intent=figure.document.layers[0].view.intent,
        )
        from zlc_frontend.plot_panel import figure_intent_from_view

        return focused, figure_intent_from_view(
            focused.document.layers[0].view,
            title=figure_intent.title,
            value_label=figure_intent.value_label,
        )

    def compose():
        from Zou_lab_control.workbench import open_figure_workbench

        return open_figure_workbench(
            figure_factory,
            display_source,
            output_root=experiment._workspace_paths().output_root,
            intent=intent,
            point_ordinals=point_ordinals,
            preferences=preferences,
            fit_preparer=prepare_fit,
            fit_executor=execute_fit,
            fit_saver=save_fit_result,
            fit_reloader=reload_fit_result,
            fit_selected_model=selected_model,
            fit_initial_selection=initial_selection,
            open_fit=open_fit,
            fit_timeout_seconds=timeout_seconds,
            initial_fit_result_identity=initial_fit_result_identity,
        )

    return experiment._open_workbench_handle(None, compose)


def standalone_pulse_connection_factory(workspace):
    """Build standalone virtual/remote connections outside ``zlc_workbench``."""

    from Zou_lab_control.api import WorkspacePaths

    if not isinstance(workspace, WorkspacePaths):
        raise TypeError("workspace must be WorkspacePaths")

    def compose(
        mode: str,
        host: str | None,
        port: int | None,
        required_document: PulseDocument,
    ):
        from Zou_lab_control.api.facade import connect
        from zlc_workbench.pulse_editor.controller import OwnedPulseConnection

        if not isinstance(required_document, PulseDocument):
            raise TypeError("required_document must be PulseDocument")
        if mode == "virtual":
            document = InstallationConfigDocument.from_parameters(
                "virtual",
                {},
            )
            connection = connect(
                document,
                workspace=workspace,
                name="pulse_gui",
            )
        elif mode == "remote":
            if host is None or port is None:
                raise ValueError("remote Pulse connection requires host and port")
            document = InstallationConfigDocument.from_parameters(
                "remote_pulse",
                {
                    "host": host,
                    "port": port,
                },
            )
            connection = connect(
                document,
                workspace=workspace,
                name="pulse_gui",
                required_pulse_document=required_document,
            )
        else:
            raise ValueError("Pulse connection mode must be virtual or remote")
        try:
            pulse = connection.pulse
            return OwnedPulseConnection(
                pulse=pulse,
                descriptor=pulse.target,
                close=connection.close,
            )
        except BaseException:
            connection.close()
            raise

    return compose


def bound_pulse_mode(experiment) -> str:
    """Project the immutable installation backend into PulseGUI's mode choice."""

    experiment = _require_experiment(experiment)
    from zlc_neutral_atom.installation_package import installation_package

    package = installation_package(experiment.installation_config.backend)
    mode = package.pulse_editor_mode
    if mode is None:
        raise ValueError(
            f"installation backend {package.backend!r} has no Pulse editor mode"
        )
    return mode


class ExperimentDeviceAdmin:
    """Composition-owned implementation of DeviceManager's narrow admin port.

    A standalone manager owns the Experiment it creates and closes it when its
    owner window is retired.  A manager opened from an existing Experiment only
    borrows that authority: disposing the window never closes the Experiment.
    The explicit ``shutdown_for_restart`` command remains the sole UI operation
    allowed to request installation shutdown.
    """

    __slots__ = (
        "_active",
        "_connect",
        "_disposed",
        "_initializing",
        "_lock",
        "_name",
        "_owns_active",
        "_workspace",
        "_terminated",
    )

    def __init__(
        self,
        *,
        workspace=None,
        name: str | None = None,
        active=None,
        owns_active: bool,
    ) -> None:
        from Zou_lab_control.api.facade import connect

        if active is not None:
            active = _require_experiment(active)
        if owns_active:
            if active is not None:
                raise ValueError("standalone DeviceManager starts without an Experiment")
            from Zou_lab_control.api import WorkspacePaths

            if not isinstance(workspace, WorkspacePaths):
                raise TypeError("standalone DeviceManager requires WorkspacePaths")
            resolved_name = "" if name is None else str(name).strip()
            if not resolved_name:
                raise ValueError("DeviceManager experiment name must be non-empty")
            self._workspace = workspace
            self._name = resolved_name
            self._connect: Callable | None = connect
        else:
            if active is None:
                raise ValueError("bound DeviceManager requires an Experiment")
            if workspace is not None or name is not None:
                raise ValueError(
                    "bound DeviceManager does not own repository or connection settings"
                )
            self._workspace = None
            self._name = active.name
            self._connect = None
        self._active = active
        self._owns_active = bool(owns_active)
        self._disposed = False
        self._initializing = False
        self._terminated = False
        self._lock = threading.RLock()

    @classmethod
    def standalone(
        cls,
        *,
        workspace,
        name: str,
    ) -> "ExperimentDeviceAdmin":
        return cls(
            workspace=workspace,
            name=name,
            active=None,
            owns_active=True,
        )

    @classmethod
    def bound(cls, experiment) -> "ExperimentDeviceAdmin":
        experiment = _require_experiment(experiment)
        return cls(
            active=experiment,
            owns_active=False,
        )

    def state(self):
        from zlc_workbench.device_manager.controller import DeviceAdminState

        with self._lock:
            active = self._active
            disposed = self._disposed
            terminated = self._terminated
        if active is None:
            return DeviceAdminState(
                None,
                None,
                None,
                can_initialize=not disposed and not terminated,
                closed=terminated or disposed,
            )
        catalog = active.device_catalog
        return DeviceAdminState(
            active.installation_config,
            catalog,
            catalog.runtime_instance_id,
            can_initialize=False,
        )

    def assess(self, candidate: InstallationConfigDocument):
        from zlc_workbench.device_manager.controller import ConfigChange

        if not isinstance(candidate, InstallationConfigDocument):
            raise TypeError("candidate must be InstallationConfigDocument")
        with self._lock:
            if self._disposed:
                raise RuntimeError("Device manager authority is closed")
            active = self._active
            if active is None:
                if self._terminated:
                    raise RuntimeError(
                        "a shut down installation must be restarted in a new process"
                    )
                return ConfigChange(
                    candidate.content_digest,
                    None,
                    initialization_required=True,
                    restart_required=False,
                )
            active_digest = active.installation_config.content_digest
        return ConfigChange(
            candidate.content_digest,
            active_digest,
            initialization_required=False,
            restart_required=candidate.content_digest != active_digest,
        )

    def initialize_once(self, candidate: InstallationConfigDocument):
        if not isinstance(candidate, InstallationConfigDocument):
            raise TypeError("candidate must be InstallationConfigDocument")
        with self._lock:
            connect = self._connect
            workspace = self._workspace
            if not self._owns_active or connect is None or workspace is None:
                raise RuntimeError("a bound DeviceManager cannot initialize a replacement")
            if self._disposed:
                raise RuntimeError("Device manager authority is closed")
            if self._terminated:
                raise RuntimeError(
                    "a shut down installation must be restarted in a new process"
                )
            if self._active is not None or self._initializing:
                raise RuntimeError("an installation is already active or initializing")
            self._initializing = True
        try:
            active = connect(
                candidate,
                workspace=workspace,
                name=self._name,
            )
        except BaseException:
            with self._lock:
                self._initializing = False
            raise
        close_unclaimed = False
        with self._lock:
            self._initializing = False
            if self._disposed:
                close_unclaimed = True
            else:
                self._active = active
        if close_unclaimed:
            active.close()
            raise RuntimeError("Device manager closed while installation initialized")
        return self.state()

    def experiment_for_runtime(self, runtime_instance_id: str):
        """Resolve a just-emitted state back to its exact application authority."""

        with self._lock:
            active = self._active
        if active is None:
            raise RuntimeError("no installation is active")
        if active.device_catalog.runtime_instance_id != str(runtime_instance_id):
            raise RuntimeError("DeviceManager state belongs to another installation")
        return active

    def shutdown_for_restart(self, expected_runtime_instance_id: str):
        from zlc_workbench.device_manager.controller import ShutdownReport

        with self._lock:
            if self._disposed:
                raise RuntimeError("Device manager authority is closed")
            active = self._active
            if active is None:
                raise RuntimeError("no installation is active")
            actual = active.device_catalog.runtime_instance_id
            if str(expected_runtime_instance_id) != actual:
                raise RuntimeError("installation generation changed before shutdown")
        if self._owns_active:
            active.close()
        else:
            active._close_for_device_restart()
        with self._lock:
            if self._active is active:
                self._active = None
            self._terminated = True
        return ShutdownReport(actual, True)

    def dispose(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            active = self._active if self._owns_active else None
            if active is not None:
                self._active = None
            self._terminated = self._terminated or active is not None
        if active is not None:
            active.close()


__all__ = [
    "ExperimentDeviceAdmin",
    "bound_pulse_mode",
    "standalone_pulse_connection_factory",
    "task_console_ports",
]
