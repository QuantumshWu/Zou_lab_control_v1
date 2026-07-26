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
from functools import partial
from pathlib import Path
from typing import Callable

from zlc_neutral_atom.installation_config import InstallationConfigDocument
from zlc_pulse import PulseDocument, describe_pulse_template


def _require_experiment(value):
    from Zou_lab_control.notebook.facade import Experiment

    if not isinstance(value, Experiment):
        raise TypeError("experiment must be the current Experiment")
    return value


def task_console_ports(experiment):
    """Decompose one Experiment into TaskConsole's explicit application port."""

    experiment = _require_experiment(experiment)
    readout = experiment.readout
    catalog = experiment.device_catalog

    from zlc_workbench.task_console.application_ports import (
        TaskConsoleApplicationPorts,
    )
    from zlc_neutral_atom.logic_nodes.readout.calibration.declaration import (
        CALIBRATION_LOGIC_NODE,
    )
    from zlc_neutral_atom.logic_nodes.camera_measurement import (
        CAMERA_MEASUREMENT_LOGIC_NODE,
        bind_camera_measurement_intent,
    )
    from zlc_neutral_atom.logic_nodes.camera_measurement.workbench_adapter import (
        start_camera_measurement_command,
    )
    from zlc_neutral_atom.logic_nodes.release_recapture.grey_molasses_detuning import (
        GREY_MOLASSES_DETUNING_LOGIC_NODE,
        prepare_bound_grey_molasses_detuning,
    )
    from zlc_neutral_atom.logic_nodes.mot_field import (
        MOT_FIELD_LOGIC_NODE,
    )
    from zlc_neutral_atom.logic_nodes.mot_field.workbench_adapter import (
        start_mot_field_task_command,
    )
    from zlc_neutral_atom.logic_nodes.readout.occupancy.declaration import (
        OCCUPANCY_LOGIC_NODE,
    )
    from zlc_neutral_atom.logic_nodes.readout.occupancy.workbench_adapter import (
        resolve_occupancy_calibration_input,
    )
    from zlc_neutral_atom.logic_nodes.pulse_scan.ui.task_console import (
        pulse_scan_task_console_adapter,
    )
    from zlc_neutral_atom.logic_nodes.readout.duration_fidelity import (
        READOUT_DURATION_FIDELITY_LOGIC_NODE,
        prepare_bound_readout_duration_fidelity,
    )
    from zlc_neutral_atom.logic_nodes.release_recapture.temperature import (
        TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE,
        prepare_bound_temperature_release_recapture,
    )
    from zlc_workbench.task_console.declaration_projection import (
        project_processor_declaration,
        project_run_declaration,
    )
    from zlc_workbench.task_console.artifact_resolution import (
        resolve_final_or_saved_artifact,
    )
    from zlc_neutral_atom.logic_nodes.readout.calibration.workbench_adapter import (
        start_calibration_task_command,
    )
    from zlc_neutral_atom.logic_nodes.readout.calibration.ui.view_projection import (
        project_calibration_final_views,
    )
    from zlc_neutral_atom.logic_nodes.readout.occupancy.ui.view_projection import (
        project_occupancy_views,
    )

    camera_roles = catalog.roles("camera")
    rf_roles = catalog.roles("rf")
    calibration_roles = readout.sitemap_camera_roles()

    bind_camera_request = partial(
        bind_camera_measurement_intent,
        request_builder=readout.camera_measurement_request,
    )
    prepare_temperature = partial(
        prepare_bound_temperature_release_recapture,
        application=readout,
    )
    prepare_readout_duration = partial(
        prepare_bound_readout_duration_fidelity,
        application=readout,
    )
    prepare_grey_molasses = partial(
        prepare_bound_grey_molasses_detuning,
        application=readout,
    )
    resolve_occupancy_calibration = partial(
        resolve_occupancy_calibration_input,
        resolve_final_or_saved=resolve_final_or_saved_artifact,
        load_saved_calibration=readout.load_saved_calibration,
    )

    attachments = (
        project_run_declaration(
            CAMERA_MEASUREMENT_LOGIC_NODE,
            bind_request=bind_camera_request,
            prepare=readout.prepare_camera_measurement,
            dynamic_choice_context=camera_roles,
            start_with_live_output=start_camera_measurement_command,
        ),
        project_run_declaration(
            TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE,
            prepare=prepare_temperature,
        ),
        project_run_declaration(
            READOUT_DURATION_FIDELITY_LOGIC_NODE,
            prepare=prepare_readout_duration,
        ),
        project_run_declaration(
            GREY_MOLASSES_DETUNING_LOGIC_NODE,
            prepare=prepare_grey_molasses,
            dynamic_choice_context=rf_roles,
        ),
        project_processor_declaration(
            OCCUPANCY_LOGIC_NODE,
            prepare=readout.prepare_occupancy_processor_request,
            resolve_artifact_reference=resolve_occupancy_calibration,
            project_presentations=project_occupancy_views,
        ),
        project_run_declaration(
            CALIBRATION_LOGIC_NODE,
            prepare=readout.prepare_calibration_task,
            dynamic_choice_context=calibration_roles,
            start_with_live_output=start_calibration_task_command,
            materialize_final_presentations=project_calibration_final_views,
        ),
        project_run_declaration(
            MOT_FIELD_LOGIC_NODE,
            prepare=readout.prepare_mot_field_task,
            dynamic_choice_context=camera_roles,
            start_with_live_output=start_mot_field_task_command,
        ),
        pulse_scan_task_console_adapter(
            prepare=readout.prepare_scan_source,
            read_pulse_template=describe_pulse_template,
        ),
    )
    return TaskConsoleApplicationPorts(attachments=attachments)


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
        from Zou_lab_control.notebook.facade import connect
        from zlc_workbench.pulse_editor.controller import OwnedPulseConnection

        if not isinstance(required_document, PulseDocument):
            raise TypeError("required_document must be PulseDocument")
        if mode == "virtual":
            document = InstallationConfigDocument.virtual()
            connection = connect(
                document,
                repository=root,
                name="pulse_gui",
            )
        elif mode == "remote":
            if host is None or port is None:
                raise ValueError("remote Pulse connection requires host and port")
            document = InstallationConfigDocument.remote_pulse(
                host=host,
                port=port,
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
        from Zou_lab_control.notebook.facade import connect

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
