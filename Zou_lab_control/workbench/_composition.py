"""Application-owned adapters for desktop Workbench surfaces.

This module is imported only by the public functions in
``Zou_lab_control.workbench``.  It is the single place where an ``Experiment``
is decomposed into the closed ports that desktop shells are allowed to retain.
No object defined here is a service locator: every operation exposed to a
Workbench is named in its concrete port type.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

from zlc_neutral_atom.installation_config import InstallationConfigDocument
from zlc_pulse import PulseDocument


class _TaskConsoleProjection:
    """Explicit domain-neutral projectors supplied to capability packages."""

    __slots__ = ("_custom", "_processor", "_resolve_final_or_saved", "_run")

    def __init__(self, *, run, processor, custom, resolve_final_or_saved) -> None:
        for value, name in (
            (run, "run"),
            (processor, "processor"),
            (custom, "custom"),
            (resolve_final_or_saved, "resolve_final_or_saved"),
        ):
            if not callable(value):
                raise TypeError(f"TaskConsole {name} projector must be callable")
        self._run = run
        self._processor = processor
        self._custom = custom
        self._resolve_final_or_saved = resolve_final_or_saved

    def run(self, declaration, **kwargs):
        return self._run(declaration, **kwargs)

    def processor(self, declaration, **kwargs):
        return self._processor(declaration, **kwargs)

    def custom(self, declaration, **kwargs):
        return self._custom(declaration, **kwargs)

    def resolve_final_or_saved(self, binding, **kwargs):
        return self._resolve_final_or_saved(binding, **kwargs)


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
    from zlc_workbench.task_console.declaration_projection import (
        project_custom_declaration,
        project_processor_declaration,
        project_run_declaration,
    )
    from Zou_lab_control.api._logic_node_api import (
        compose_task_console_attachments,
    )

    return TaskConsoleApplicationPorts(
        attachments=compose_task_console_attachments(
            experiment.nodes,
            experiment.device_catalog,
            _TaskConsoleProjection(
                run=project_run_declaration,
                processor=project_processor_declaration,
                custom=project_custom_declaration,
                resolve_final_or_saved=resolve_final_or_saved_artifact,
            ),
        )
    )


def standalone_pulse_workspace(value: str | Path | None) -> Path:
    """Resolve the one durable workspace owned by a standalone PulseGUI."""

    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("ZLC_PULSE_WORKSPACE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".zlc" / "pulse-workbench").resolve()


def standalone_pulse_connection_factory(workspace: Path):
    """Build standalone virtual/remote connections outside ``zlc_workbench``."""

    root = Path(workspace).expanduser().resolve()

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
                repository=root,
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
                repository=root,
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
    backend = experiment.installation_config.backend
    if backend == "virtual":
        return "virtual"
    if backend == "remote_pulse":
        return "remote"
    raise ValueError(f"unsupported Pulse installation backend {backend!r}")


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
        "_repository",
        "_terminated",
    )

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
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
            if repository is None:
                raise ValueError("standalone DeviceManager requires a repository")
            resolved_name = "" if name is None else str(name).strip()
            if not resolved_name:
                raise ValueError("DeviceManager experiment name must be non-empty")
            self._repository = Path(repository).expanduser().resolve()
            self._name = resolved_name
            self._connect: Callable | None = connect
        else:
            if active is None:
                raise ValueError("bound DeviceManager requires an Experiment")
            if repository is not None or name is not None:
                raise ValueError(
                    "bound DeviceManager does not own repository or connection settings"
                )
            self._repository = None
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
        repository: str | Path | None,
        name: str,
    ) -> "ExperimentDeviceAdmin":
        root = (
            Path.home() / ".zlc" / "device-manager"
            if repository is None
            else Path(repository)
        )
        return cls(
            repository=root,
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
            repository = self._repository
            if not self._owns_active or connect is None or repository is None:
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
                repository=repository,
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
    "standalone_pulse_workspace",
    "task_console_ports",
]
