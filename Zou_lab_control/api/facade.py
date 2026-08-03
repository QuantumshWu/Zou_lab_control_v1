"""Experiment composition API with no public raw hardware graph."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import (
    Callable,
    Mapping,
    TYPE_CHECKING,
)

from zlc_neutral_atom.installation import (
    DeviceCatalogView,
)
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
    installation_template,
    load_installation_config,
)
from zlc_neutral_atom.installation_runtime import (
    create_installation,
)
from zlc_data import OwnedSnapshot
from zlc_neutral_atom.devices.sequencer.application import (
    AppliedPulseSnapshot,
    PreparedPulseExecution,
    PulseApplicationOwner,
    PulseRunDescriptor,
    PulseRunObservation,
    PulseRunRequest,
    PulseRunResult,
    PulseTargetDescriptor,
    prepare_pulse_execution,
)
from zlc_neutral_atom.devices.sequencer.port import PulseScanProgress
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_neutral_atom.runtime.run import CancelOutcome, RunHandle
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
)
from zlc_storage import canonical_text as _text
from zlc_storage import durable_makedirs

from ._application_services import (
    ExperimentCloseAttempt as _ExperimentCloseAttempt,
    ExperimentServices as _ExperimentServices,
    WorkspacePaths,
    application_start_run as _application_start_run,
    _prune_terminal_runs_locked,
    resolve_role as _resolve_role,
    service_guard as _service_guard,
    wait_for_close_attempt as _wait_for_close_attempt,
)
from ._logic_node_api import compose_logic_node_apis

if TYPE_CHECKING:
    from zlc_plot import PlotSpec


class _ResourceCleanupError(RuntimeError):
    """Report retaining every ordinary cleanup failure."""

    def __init__(
        self,
        message: str,
        failures: tuple[Exception, ...],
    ) -> None:
        if not failures:
            raise ValueError("cleanup error requires at least one failure")
        self.failures = failures
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in failures
        )
        super().__init__(f"{message}: {details}")


def _cleanup_failures(*actions) -> list[Exception]:
    failures: list[Exception] = []
    for action in actions:
        if action is None:
            continue
        try:
            action()
        except Exception as error:
            failures.append(error)
    return failures


def _require_runtime_shutdown(runtime, *, timeout: float) -> None:
    if not runtime.shutdown(timeout=timeout):
        diagnostics = tuple(getattr(runtime, "shutdown_diagnostics", ()))
        suffix = "" if not diagnostics else ": " + "; ".join(diagnostics)
        raise RuntimeError(
            "runtime did not terminate within the cleanup deadline" + suffix
        )


class PulseFacade:
    __slots__ = ("_services",)

    def __init__(self, services: _ExperimentServices) -> None:
        if not isinstance(services, _ExperimentServices):
            raise TypeError("services must be _ExperimentServices")
        self._services = services

    @property
    def target(self) -> PulseTargetDescriptor:
        with _service_guard(self._services) as services:
            role = _resolve_role(
                services.catalog,
                None,
                "sequencer",
                ("sequencer",),
                "pulse.execute",
            )
            reference = services.catalog.require_role(role).ref
            port = services.runtime.require_capability(
                reference,
                "pulse.execute",
            )
            capability = port.capability
            return PulseTargetDescriptor(
                reference,
                capability.manifest,
                capability.clock_hz,
                capability.geometry_fingerprint,
            )

    def request(
        self,
        document: PulseDocument,
        execution_form: PulseExecutionForm = PulseExecutionForm.STATIC_ONCE,
        *,
        api_values: Mapping[str, int | float] | None = None,
        scan_sweep_count: int = 1,
        sequencer_role: str | None = None,
        timeout_seconds: float | None = None,
    ) -> PulseRunRequest:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        with _service_guard(self._services) as services:
            role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
                "pulse.execute",
            )
            reference = services.catalog.require_role(role).ref
        if api_values is not None and not isinstance(api_values, Mapping):
            raise TypeError("api_values must be a mapping or None")
        requested_api_values = {} if api_values is None else dict(api_values)
        expected_api_ids = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        expected_api_id_set = set(expected_api_ids)
        if set(requested_api_values) != expected_api_id_set:
            missing = tuple(
                parameter_id
                for parameter_id in expected_api_ids
                if parameter_id not in requested_api_values
            )
            unknown = tuple(
                parameter_id
                for parameter_id in requested_api_values
                if parameter_id not in expected_api_id_set
            )
            raise ValueError(
                "Pulse Run API values must exactly cover declared parameters; "
                f"missing={missing}, unknown={unknown}"
            )
        return PulseRunRequest(
            document=document,
            execution_form=execution_form,
            sequencer_ref=reference,
            timeout_seconds=timeout_seconds,
            api_values=tuple(
                (parameter_id, requested_api_values[parameter_id])
                for parameter_id in expected_api_ids
            ),
            scan_sweep_count=scan_sweep_count,
        )

    def snapshot(self) -> AppliedPulseSnapshot | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.snapshot()

    def observe_active(self) -> PulseRunObservation | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.observe_active()

    def observe_scan_progress(self) -> PulseScanProgress | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.observe_scan_progress()

    def cancel_active(
        self,
        reason: str = "user requested pulse stop",
    ) -> CancelOutcome | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.cancel_active(reason)

    def inspect(self, request: PulseRunRequest) -> PulseRunDescriptor:
        with _service_guard(self._services) as services:
            return _prepare_pulse_for_services(services, request).descriptor

    def start(self, request: PulseRunRequest) -> RunHandle:
        return _start_pulse(self._services, request)

    def run(self, request: PulseRunRequest) -> PulseRunResult:
        if request.requires_cancellation:
            raise ValueError("continuous pulse execution must be started and cancelled")
        return _run_pulse(self._services, request)


class Experiment:
    """Public experiment root containing values, requests, and narrow APIs only."""

    __slots__ = (
        "_binding",
        "name",
    )

    def __init__(
        self,
        services: _ExperimentServices,
        *,
        name: str,
        device_catalog: DeviceCatalogView,
        installation_config: InstallationConfigDocument,
    ) -> None:
        if not isinstance(services, _ExperimentServices):
            raise TypeError("services must be _ExperimentServices")
        self.name = _text(name, "experiment name")
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        if not isinstance(installation_config, InstallationConfigDocument):
            raise TypeError(
                "installation_config must be InstallationConfigDocument"
            )
        nodes = compose_logic_node_apis(services)
        self._binding = (
            services,
            device_catalog,
            installation_config,
            nodes,
            PulseFacade(services),
        )

    @property
    def _services(self) -> _ExperimentServices:
        return self._binding[0]

    @property
    def device_catalog(self) -> DeviceCatalogView:
        return self._binding[1]

    @property
    def installation_config(self) -> InstallationConfigDocument:
        return self._binding[2]

    @property
    def nodes(self):
        return self._binding[3]

    @property
    def pulse(self) -> PulseFacade:
        return self._binding[4]

    def pulse_gui(
        self,
        *,
        document: PulseDocument | None = None,
        path: str | Path | None = None,
    ):
        """Open the current PulseDocument editor on this exact installation."""

        from Zou_lab_control.workbench import open_pulse_editor

        return open_pulse_editor(self, document=document, path=path)

    def task_console(self, *, task=None, state=None, **kwargs):
        """Lazily open the task console bound to this experiment.

        One composition root owns the window (``zlc_workbench.task_console.app``),
        so scripted and double-clickable launchers open the same console.
        """

        from Zou_lab_control.workbench import open_task_console

        return open_task_console(self, task=task, state=state, **kwargs)

    def device_manager(self):
        """Open the one config/admin window bound to this installation."""

        from Zou_lab_control.workbench import open_device_manager

        return open_device_manager(self)

    def _workspace_paths(self) -> WorkspacePaths:
        """Return the immutable roots only to application composition."""

        with _service_guard(self._services) as services:
            return services.workspace_paths

    def _signal_data_plane(self) -> SignalDataPlane:
        """Lend Workbench the one Experiment-owned signal authority."""

        with _service_guard(self._services) as services:
            return services.signal_plane

    def _open_workbench_handle(
        self,
        key: str | None,
        compose: Callable[[], object],
        *,
        existing_error: str | None = None,
    ):
        """Own one GUI handle without exposing the application service graph.

        Concrete Workbench construction lives in ``Zou_lab_control.workbench``;
        this method owns only Experiment-scoped reuse and retirement.
        """
        from ._application_services import open_workbench_handle

        return open_workbench_handle(
            self._services,
            None if key is None else _text(key, "workbench key"),
            compose,
            existing_error=existing_error,
        )

    def _replace_installation(
        self,
        candidate: InstallationConfigDocument,
    ) -> None:
        """Replace one idle installation while preserving this public object.

        The DeviceManager is the only caller.  This is the sole private
        replacement seam: it freezes admission, retires every runtime-bound
        Workbench except the initiating manager, reaches the old runtime's SAFE
        terminal, composes the candidate, and publishes one complete binding in
        a single assignment.  A failed candidate is never published.
        """

        if not isinstance(candidate, InstallationConfigDocument):
            raise TypeError("candidate must be InstallationConfigDocument")
        if candidate == self.installation_config:
            return

        previous = self.installation_config
        services = self._services
        caller_thread_id = threading.get_ident()
        retained_manager = None
        retained_artifacts: dict[str, object]
        with services.admission_lock:
            _prune_terminal_runs_locked(services)
            if services.active_runs:
                raise RuntimeError(
                    "installation Apply requires every active Run to finish or stop"
                )
            with services.operation_lock:
                if services.state != "OPEN":
                    raise RuntimeError("Experiment is closing or closed")
                if services.operation_thread_counts.get(caller_thread_id, 0):
                    raise RuntimeError(
                        "installation Apply cannot run inside an Experiment operation"
                    )
                retained_manager = services.gui_handles.pop(
                    "device-manager",
                    None,
                )
                retained_artifacts = dict(services.default_artifacts)
                services.closing_gui_handles = tuple(
                    services.gui_handles.values()
                )
                services.gui_handles.clear()
                services.state = "CLOSING"
                workspace = services.workspace_paths

        self.close()

        def publish(replacement: "Experiment") -> None:
            replacement_services = replacement._services
            with replacement_services.operation_lock:
                if replacement_services.state != "OPEN":
                    raise RuntimeError(
                        "replacement Experiment closed before publication"
                    )
                replacement_services.default_artifacts.update(
                    retained_artifacts
                )
                if retained_manager is not None:
                    replacement_services.gui_handles[
                        "device-manager"
                    ] = retained_manager
            self._binding = replacement._binding

        try:
            replacement = connect(
                candidate,
                workspace=workspace,
                name=self.name,
            )
        except BaseException as apply_error:
            try:
                restored = connect(
                    previous,
                    workspace=workspace,
                    name=self.name,
                )
                publish(restored)
            except BaseException as restore_error:
                raise BaseExceptionGroup(
                    "installation Apply failed and the previous installation "
                    "could not be restored",
                    [apply_error, restore_error],
                ) from apply_error
            raise
        publish(replacement)

    def figure_gui(
        self,
        snapshot: OwnedSnapshot | str | Path | None = None,
        *,
        spec: "PlotSpec | None" = None,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        archive_path: str | Path | None = None,
        metadata: Mapping[str, object] | None = None,
        open_fit: bool = False,
    ):
        """Open the viewer, or one explicit snapshot through ``zlc_plot``."""

        if not isinstance(open_fit, bool):
            raise TypeError("open_fit must be bool")
        if snapshot is None or isinstance(snapshot, (str, Path)):
            supplied_overrides = tuple(
                name
                for name, value in (
                    ("spec", spec),
                    ("size", size),
                    ("parameters", parameters),
                    ("archive_path", archive_path),
                    ("metadata", metadata),
                    ("open_fit", True if open_fit else None),
                )
                if value is not None
            )
            if supplied_overrides:
                raise ValueError(
                    "saved FigureViewer does not accept snapshot options: "
                    + ", ".join(supplied_overrides)
                )
            from zlc_workbench.figure_viewer.app import open_figure_viewer

            with _service_guard(self._services) as services:
                figures_root = services.workspace_paths.figures_root
            return open_figure_viewer(
                path=snapshot,
                output_root=figures_root,
            )

        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot, path, or None")
        if spec is None:
            raise ValueError("snapshot Figure requires explicit PlotSpec")
        from zlc_plot import PlotSpec as CurrentPlotSpec

        if not isinstance(spec, CurrentPlotSpec):
            raise TypeError("spec must be PlotSpec")
        from Zou_lab_control.workbench import open_figure_workbench
        with _service_guard(self._services) as services:
            figures_root = services.workspace_paths.figures_root

        return open_figure_workbench(
            snapshot,
            spec,
            output_root=figures_root,
            size=size,
            parameters=parameters,
            archive_path=archive_path,
            metadata=metadata,
            open_fit=open_fit,
        )

    def close(self) -> None:
        services = self._services
        caller_thread_id = threading.get_ident()
        with services.operation_lock:
            if services.state == "CLOSED":
                return
            if services.operation_thread_counts.get(caller_thread_id, 0):
                raise RuntimeError(
                    "Experiment cannot close reentrantly from an active operation"
                )
            attempt = services.close_attempt
            if attempt is not None:
                if attempt.owner_thread_id == caller_thread_id:
                    raise RuntimeError(
                        "Experiment cannot close reentrantly from its active close attempt"
                    )
                wait_for_attempt = attempt
                wait_handles = services.closing_gui_handles
                owns_attempt = False
            else:
                if services.state == "OPEN":
                    services.closing_gui_handles = tuple(
                        services.gui_handles.values()
                    )
                    services.gui_handles.clear()
                    services.state = "CLOSING"
                elif services.state != "CLOSING":
                    raise RuntimeError(
                        f"unknown Experiment lifecycle state {services.state!r}"
                    )
                attempt = _ExperimentCloseAttempt(caller_thread_id)
                services.close_attempt = attempt
                gui_handles = services.closing_gui_handles
                operations_drained = services.operations_drained
                wait_for_attempt = attempt
                owns_attempt = True

        if not owns_attempt:
            _wait_for_close_attempt(
                wait_for_attempt,
                wait_handles,
            )
            if wait_for_attempt.failure is not None:
                raise wait_for_attempt.failure
            return

        try:
            gui_close_failures = []
            for handle in gui_handles:
                try:
                    handle.request_owner_close()
                except Exception as error:
                    gui_close_failures.append(error)
            # Long application operations own service borrows outside the
            # short composition lock. Closing flips state first, then waits
            # without holding the lock needed by their finally blocks.
            operations_drained.wait()
            with services.operation_lock:
                if services.active_operations:
                    raise RuntimeError(
                        "operation drain event disagrees with operation count"
                    )

            node_close_failures = list(self.nodes.close())
            if node_close_failures:
                raise _ResourceCleanupError(
                    "Experiment Logic-node close failed",
                    tuple(node_close_failures),
                )

            # Only this attempt owner may tear down the runtime.
            # Other close callers wait on the same completion fact instead of
            # racing a second cleanup sequence.
            shutdown = services.runtime.shutdown(timeout=2.0)
            if not shutdown:
                diagnostics = tuple(
                    getattr(services.runtime, "shutdown_diagnostics", ())
                )
                suffix = (
                    ""
                    if not diagnostics
                    else ": " + "; ".join(diagnostics)
                )
                raise RuntimeError(
                    "Experiment close did not complete" + suffix
                )

            # Runtime shutdown is the terminal authority for active Runs. Give
            # each GUI handle one idempotent post-runtime turn, then wait for
            # its owner-thread terminal acknowledgement before data services
            # can be closed.
            for handle in gui_handles:
                try:
                    handle.request_owner_close()
                except Exception as error:
                    gui_close_failures.append(error)
            for handle in gui_handles:
                try:
                    if not handle.wait_owner_closed(10.0):
                        gui_close_failures.append(
                            RuntimeError(
                                f"{type(handle).__name__} did not complete owner close"
                            )
                        )
                except Exception as error:
                    gui_close_failures.append(error)

            # A GUI that has not acknowledged owner-thread retirement can
            # still hold a worker borrow into the data plane.  Preserve the
            # data plane and frozen handle set so a later close attempt
            # can retry; never turn an acknowledgement failure into use-after-
            # close by continuing down the teardown chain.
            if gui_close_failures:
                raise _ResourceCleanupError(
                    "Experiment Workbench close failed",
                    tuple(gui_close_failures),
                )

            failures = _cleanup_failures(services.signal_plane.close)
            if failures:
                raise _ResourceCleanupError(
                    "Experiment close failed",
                    tuple(failures),
                )
        except BaseException as error:
            with services.operation_lock:
                if services.close_attempt is attempt:
                    attempt.failure = error
                    services.close_attempt = None
                    attempt.completed.set()
            raise
        else:
            with services.operation_lock:
                if services.close_attempt is not attempt:
                    raise RuntimeError("Experiment close attempt ownership was lost")
                services.closing_gui_handles = ()
                services.state = "CLOSED"
                services.close_attempt = None
                attempt.completed.set()

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _prepare_pulse_for_services(
    services: _ExperimentServices,
    request: PulseRunRequest,
) -> PreparedPulseExecution:
    return prepare_pulse_execution(
        request,
        pulse_port=services.runtime.require_capability(
            request.sequencer_ref,
            "pulse.execute",
        ),
        start_run=lambda plan: _application_start_run(services, plan),
        on_applied=services.pulse_application._record_applied,
    )


def _start_pulse(services: _ExperimentServices, request: PulseRunRequest) -> RunHandle:
    with _service_guard(services) as guarded:
        prepared = _prepare_pulse_for_services(guarded, request)
        owner = guarded.pulse_application
    return owner.start(prepared)


def _run_pulse(services: _ExperimentServices, request: PulseRunRequest) -> PulseRunResult:
    with _service_guard(services) as guarded:
        prepared = _prepare_pulse_for_services(guarded, request)
        owner = guarded.pulse_application
        runtime = guarded.runtime
    handle = owner.start(prepared)
    result = runtime.wait(handle)
    if not isinstance(result, PulseRunResult):
        raise TypeError("pulse Run returned an unexpected result")
    return result


_CONNECT_SEED_UNSET = object()


def _resolve_installation_document(
    config: str | Path | InstallationConfigDocument,
    seed: int | None | object,
) -> InstallationConfigDocument:
    if isinstance(config, InstallationConfigDocument):
        if seed is not _CONNECT_SEED_UNSET:
            raise ValueError(
                "seed belongs inside an InstallationConfigDocument"
            )
        return config
    if isinstance(config, Path):
        if seed is not _CONNECT_SEED_UNSET:
            raise ValueError("seed cannot override a saved installation config")
        return load_installation_config(config)
    if not isinstance(config, str):
        raise TypeError(
            "config must be 'virtual', a config path, or "
            "InstallationConfigDocument"
        )
    if config == "virtual":
        parameters = {} if seed is _CONNECT_SEED_UNSET else {"seed": seed}
        return installation_template("virtual", **parameters)
    if config == "remote_pulse":
        raise ValueError(
            "remote_pulse requires a saved installation config or "
            "installation_template('remote_pulse', host=..., port=...)"
        )
    if seed is not _CONNECT_SEED_UNSET:
        raise ValueError("seed cannot override a saved installation config")
    return load_installation_config(config)


def connect(
    config: str | Path | InstallationConfigDocument = "virtual",
    *,
    workspace: WorkspacePaths,
    name: str = "neutral_atom",
    seed: int | None | object = _CONNECT_SEED_UNSET,
    required_pulse_document: PulseDocument | None = None,
) -> Experiment:
    """Compose one Experiment; raw devices remain authority-private."""

    installation_document = _resolve_installation_document(config, seed)
    if required_pulse_document is not None and not isinstance(
        required_pulse_document,
        PulseDocument,
    ):
        raise TypeError("required_pulse_document must be PulseDocument or None")
    if not isinstance(workspace, WorkspacePaths):
        raise TypeError("workspace must be WorkspacePaths")
    canonical_name = _text(name, "experiment name")
    for root in (
        workspace.project_root,
        workspace.pulses_root,
        workspace.tasks_root,
        workspace.runs_root,
        workspace.figures_root,
    ):
        durable_makedirs(root)
    runtime = None
    signal_plane = None
    try:
        installation = create_installation(
            installation_document,
            required_pulse_document=required_pulse_document,
        )
        runtime = installation.runtime
        catalog = runtime.device_catalog
        signal_plane = SignalDataPlane()
        operations_drained = threading.Event()
        operations_drained.set()
        services = _ExperimentServices(
            workspace_paths=workspace,
            installation=installation,
            runtime=runtime,
            catalog=catalog,
            installation_config=installation_document,
            pulse_application=PulseApplicationOwner(),
            signal_plane=signal_plane,
            operation_lock=threading.RLock(),
            admission_lock=threading.RLock(),
            operations_drained=operations_drained,
            operation_thread_counts={},
            active_runs={},
            gui_handles={},
        )
        experiment = Experiment(
            services,
            name=canonical_name,
            device_catalog=catalog,
            installation_config=installation_document,
        )
        return experiment
    except BaseException as error:
        failures = (
            _cleanup_failures(
                (
                    None
                    if runtime is None
                    else lambda: _require_runtime_shutdown(runtime, timeout=2.0)
                )
            )
            + _cleanup_failures(
                None if signal_plane is None else signal_plane.close,
            )
        )
        if failures and isinstance(error, Exception):
            raise _ResourceCleanupError(
                "Experiment composition cleanup failed",
                tuple(failures),
            ) from error
        raise


def device_manager(
    config: str | Path | InstallationConfigDocument = "virtual",
    *,
    workspace: WorkspacePaths,
    name: str = "neutral_atom",
    seed: int | None | object = _CONNECT_SEED_UNSET,
    on_initialized=None,
):
    """Open the standalone config editor before any installation is composed."""

    document = _resolve_installation_document(config, seed)
    config_path = None
    if isinstance(config, Path):
        config_path = config.expanduser().resolve()
    elif isinstance(config, str) and config not in {"virtual", "remote_pulse"}:
        config_path = Path(config).expanduser().resolve()
    from Zou_lab_control.workbench import open_device_manager

    return open_device_manager(
        document=document,
        config_path=config_path,
        workspace=workspace,
        name=name,
        on_initialized=on_initialized,
    )


__all__ = [
    "AppliedPulseSnapshot",
    "connect",
    "device_manager",
    "DeviceInstanceConfig",
    "Experiment",
    "InstallationConfigDocument",
    "installation_template",
    "PreparedPulseExecution",
    "PulseFacade",
    "PulseRunDescriptor",
    "PulseRunObservation",
    "PulseRunRequest",
    "PulseRunResult",
    "PulseTargetDescriptor",
    "WorkspacePaths",
]
