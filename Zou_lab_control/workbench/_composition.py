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

from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    installation_template,
)
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

        if isinstance(declaration.definition, ProcessorDefinition):
            attachments.append(
                project_processor_declaration(
                    declaration,
                    bind_request=bind_request,
                    prepare=(
                        lambda request, owner=package.prepare_hosted,
                        current_api=api: owner(current_api, request, None)
                    ),
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
            )
        )

    return TaskConsoleApplicationPorts(
        attachments=tuple(attachments),
        data_plane=experiment._signal_data_plane(),
        tasks_root=workspace.tasks_root,
        output_root=workspace.output_root,
    )


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
            document = installation_template("virtual")
            connection = connect(
                document,
                workspace=workspace,
                name="pulse_gui",
            )
        elif mode == "remote":
            if host is None or port is None:
                raise ValueError("remote Pulse connection requires host and port")
            document = installation_template(
                "remote_pulse",
                host=host,
                port=port,
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
    """Project the bound pulse capability into PulseGUI's mode choice."""

    experiment = _require_experiment(experiment)
    from Zou_lab_control.api._application_services import service_guard

    target = experiment.pulse.target
    with service_guard(experiment._services) as services:
        port = services.runtime.require_capability(
            target.sequencer_ref,
            "pulse.execute",
        )
        mode = getattr(port, "editor_connection_mode", None)
    if mode not in {"virtual", "remote"}:
        raise ValueError(
            "the bound pulse capability does not declare a virtual/remote "
            "editor connection mode"
        )
    return mode


class ExperimentDeviceAdmin:
    """Composition-owned implementation of DeviceManager's narrow admin port."""

    __slots__ = (
        "_active",
        "_disposed",
        "_ever_active",
        "_applying",
        "_lock",
        "_name",
        "_owns_active",
        "_workspace",
    )

    def __init__(
        self,
        *,
        workspace=None,
        name: str | None = None,
        active=None,
        owns_active: bool,
    ) -> None:
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
        else:
            if active is None:
                raise ValueError("bound DeviceManager requires an Experiment")
            if workspace is not None or name is not None:
                raise ValueError(
                    "bound DeviceManager does not own repository or connection settings"
                )
            self._workspace = None
            self._name = active.name
        self._active = active
        self._owns_active = bool(owns_active)
        self._disposed = False
        self._ever_active = active is not None
        self._applying = False
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
            applying = self._applying
            can_apply = (
                not disposed
                and not applying
                and (active is not None or not self._ever_active)
            )
        if active is None:
            return DeviceAdminState(
                None,
                None,
                None,
                can_apply=can_apply,
                closed=disposed,
            )
        catalog = active.device_catalog
        return DeviceAdminState(
            active.installation_config,
            catalog,
            catalog.runtime_instance_id,
            can_apply=can_apply,
            closed=disposed,
        )

    def apply(self, candidate: InstallationConfigDocument):
        if not isinstance(candidate, InstallationConfigDocument):
            raise TypeError("candidate must be InstallationConfigDocument")
        from zlc_neutral_atom.device_types import validate_installation_graph

        # This preflight is pure: no old owner, device factory, or transport is
        # touched before the complete candidate graph has been accepted.
        validate_installation_graph(candidate)
        with self._lock:
            if self._disposed:
                raise RuntimeError("Device manager authority is closed")
            if self._applying:
                raise RuntimeError("an installation Apply is already running")
            active = self._active
            if active is None and self._ever_active:
                raise RuntimeError(
                    "the previous installation could not be restored; "
                    "no active session remains"
                )
            workspace = self._workspace
            self._applying = True

        try:
            if active is None:
                if not self._owns_active or workspace is None:
                    raise RuntimeError("bound DeviceManager has no active Experiment")
                from Zou_lab_control.api.facade import connect

                applied = connect(
                    candidate,
                    workspace=workspace,
                    name=self._name,
                )
            else:
                active._replace_installation(candidate)
                applied = active
        except BaseException:
            with self._lock:
                self._applying = False
                if active is not None:
                    try:
                        active._workspace_paths()
                    except RuntimeError:
                        if self._active is active:
                            self._active = None
            raise

        close_unclaimed = None
        with self._lock:
            self._applying = False
            if self._disposed:
                if self._owns_active:
                    close_unclaimed = applied
            else:
                self._active = applied
                self._ever_active = True
        if close_unclaimed is not None:
            close_unclaimed.close()
            raise RuntimeError("Device manager closed while installation Apply completed")
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

    def dispose(self) -> None:
        with self._lock:
            if self._disposed:
                return
            if self._applying:
                raise RuntimeError(
                    "cannot dispose device authority during installation Apply"
                )
            self._disposed = True
            active = self._active if self._owns_active else None
            if active is not None:
                self._active = None
        if active is not None:
            active.close()


__all__ = [
    "ExperimentDeviceAdmin",
    "bound_pulse_mode",
    "standalone_pulse_connection_factory",
    "task_console_ports",
]
