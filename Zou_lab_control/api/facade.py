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
    InstallationConfigDocument,
    load_installation_config,
)
from zlc_data import (
    AxisSourceRef,
    CommittedTransform,
    FitNumericPolicy,
    FitParameterConstraint,
    FitSpec,
    HistogramSpec,
    Selection,
)
from zlc_data.fit import fit_spec_for
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    FitExecution,
    FitResultArtifactRef,
    FitResultRepository,
)
from zlc_neutral_atom.capture.artifact import CaptureRepository
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.capture.application import (
    CaptureRequest,
    PlanDescriptor,
    PreparedFiniteCapture,
)
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
from zlc_storage import positive_real as _positive_real

from ._readout_core import ReadoutFacade
from ._application_services import (
    ExperimentCloseAttempt as _ExperimentCloseAttempt,
    ExperimentServices as _ExperimentServices,
    WorkspacePaths,
    application_start_run as _application_start_run,
    fit_service_guard as _fit_service_guard,
    resolve_role as _resolve_role,
    service_guard as _service_guard,
    wait_for_close_attempt as _wait_for_close_attempt,
)
from ._figure_projection import (
    data_figure_for_services as _data_figure_for_services,
    project_figure as _project_figure,
)
from ._dataset_sources import (
    project_final_dataset_source as _project_final_dataset_source,
)
from ._logic_node_api import compose_logic_node_apis

if TYPE_CHECKING:
    from zlc_frontend import DataFigure
    from zlc_frontend.figure import FigureDocument, ViewIntent, ViewPreferences


_DEFAULT_FIT_GUI_TIMEOUT_SECONDS = 30.0


def _fit_selection_authority(
    transform: CommittedTransform,
    *,
    context: str,
) -> Selection | None:
    operations = tuple(transform.spec.operations)
    if not operations:
        return None
    if len(operations) == 1 and isinstance(operations[0], Selection):
        return operations[0]
    if isinstance(operations[-1], HistogramSpec):
        return None
    raise ValueError(
        f"{context} supports only identity or one range-preserving "
        "Selection transform"
    )


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
            )
            reference = services.catalog.require(role).ref
            port = services.runtime.pulse_port(reference)
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
            )
            reference = services.catalog.require(role).ref
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
        "_artifact_operations",
        "_services",
        "name",
        "device_catalog",
        "installation_config",
        "pulse",
        "readout",
        "nodes",
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
        self._services = services
        self.name = _text(name, "experiment name")
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        if not isinstance(installation_config, InstallationConfigDocument):
            raise TypeError(
                "installation_config must be InstallationConfigDocument"
            )
        self.device_catalog = device_catalog
        self.installation_config = installation_config
        self.readout = ReadoutFacade(services)
        self.nodes = compose_logic_node_apis(
            self.readout,
            self.readout._artifact_capabilities(),
        )
        self._artifact_operations = self.nodes._artifact_operations
        self.pulse = PulseFacade(services)

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

    def _close_for_device_restart(self) -> None:
        """Close this installation while its DeviceManager remains the reporter.

        The bound DeviceManager initiated the explicit restart transition, so
        it must survive long enough to receive and display the terminal admin
        state.  Every other registered Workbench is retired by ``close()``.
        """

        with _service_guard(self._services) as services:
            services.gui_handles.pop("device-manager", None)
        self.close()

    def start(self, request: CaptureRequest) -> RunHandle:
        return self.readout.prepare_capture(request).start()

    def prepare_capture(self, request: CaptureRequest) -> PreparedFiniteCapture:
        """Bind one finite capture without exposing the runtime service graph."""

        return self.readout.prepare_capture(request)

    def run(self, request: CaptureRequest) -> CaptureArtifactRef:
        handle = self.readout.prepare_capture(request).start()
        with _service_guard(self._services) as services:
            runtime = services.runtime
        result = runtime.wait(handle)
        if not isinstance(result, CaptureArtifactRef):
            raise TypeError("capture Run returned an unexpected result")
        return result

    def inspect(self, request: CaptureRequest) -> PlanDescriptor:
        return self.readout.prepare_capture(request).descriptor

    def fit(
        self,
        source: object,
        spec: FitSpec | None = None,
        *,
        model: str | None = None,
        committed_transform: CommittedTransform | None = None,
        independent_sources: tuple[AxisSourceRef, ...] | None = None,
        batch_sources: tuple[AxisSourceRef, ...] | None = None,
        constraints: tuple[FitParameterConstraint, ...] = (),
        numeric_policy: FitNumericPolicy | None = None,
    ) -> FitExecution:
        """Fit one owner-admitted FINAL Dataset artifact without hidden reduction."""

        if not self._artifact_operations.can_project_dataset(source):
            raise TypeError("source must be a composed FINAL Dataset artifact")
        if (spec is None) == (model is None):
            raise ValueError("provide exactly one of spec or model")
        with _fit_service_guard(self._services) as services:
            def closing_cancel_check() -> bool:
                with services.operation_lock:
                    return services.state != "OPEN"

            if spec is None:
                assert model is not None
                if independent_sources is None:
                    raise ValueError(
                        "model convenience Fit requires explicit independent_sources"
                    )
                schema = _project_final_dataset_source(
                    self._artifact_operations,
                    source,
                    materialize=False,
                ).schema
                spec = fit_spec_for(
                    schema,
                    model,
                    committed_transform=committed_transform,
                    independent_sources=tuple(independent_sources),
                    batch_sources=(
                        () if batch_sources is None else tuple(batch_sources)
                    ),
                    constraints=constraints,
                    numeric_policy=(
                        FitNumericPolicy()
                        if numeric_policy is None
                        else numeric_policy
                    ),
                )
                del schema
            elif any(
                value is not None
                for value in (
                    committed_transform,
                    independent_sources,
                    batch_sources,
                    numeric_policy,
                )
            ) or constraints:
                raise ValueError(
                    "spec cannot be combined with model convenience arguments"
                )
            return services.fit_repository.execute(
                self._artifact_operations,
                source,
                spec,
                cancel_check=closing_cancel_check,
            )

    def load_fit(
        self,
        reference: FitResultArtifactRef,
    ) -> AdmittedFitResult:
        with _service_guard(self._services) as services:
            return services.fit_repository.load(
                reference,
                artifacts=self._artifact_operations,
            )

    def _open_fit_capable_figure_gui(
        self,
        display_source,
        fit_source: object,
        *,
        intent,
        point_ordinals: tuple[int, ...] | None,
        preferences,
        artifact_output: str | None,
        selected_model: str | None = None,
        initial_fit_spec: FitSpec | None = None,
        initial_selection: Selection | None = None,
        open_fit: bool = False,
        timeout_seconds: float = _DEFAULT_FIT_GUI_TIMEOUT_SECONDS,
        initial_fit_result_identity: str | None = None,
        direct_fit_single_panel: bool = False,
    ):
        """Validate one Figure request and delegate its Qt composition lazily."""

        if not self._artifact_operations.can_project_dataset(fit_source):
            raise TypeError("fit_source must be a composed FINAL Dataset artifact")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        chosen_model = (
            None if selected_model is None else _text(selected_model, "model")
        )
        if initial_fit_spec is not None and not isinstance(
            initial_fit_spec,
            FitSpec,
        ):
            raise TypeError("initial_fit_spec must be FitSpec or None")
        if initial_fit_spec is not None:
            if chosen_model is None:
                chosen_model = initial_fit_spec.model_id
            elif chosen_model != initial_fit_spec.model_id:
                raise ValueError("selected model differs from the initial FitSpec")
        from Zou_lab_control.workbench._composition import (
            open_fit_capable_figure_gui,
        )

        return open_fit_capable_figure_gui(
            self,
            display_source,
            fit_source,
            intent=intent,
            point_ordinals=point_ordinals,
            preferences=preferences,
            artifact_output=artifact_output,
            selected_model=chosen_model,
            initial_fit_spec=initial_fit_spec,
            initial_selection=initial_selection,
            open_fit=open_fit,
            timeout_seconds=timeout,
            initial_fit_result_identity=initial_fit_result_identity,
            direct_fit_single_panel=direct_fit_single_panel,
        )

    def fit_gui(
        self,
        source: object,
        *,
        model: str | None = None,
        committed_transform: CommittedTransform | None = None,
        timeout_seconds: float = _DEFAULT_FIT_GUI_TIMEOUT_SECONDS,
    ):
        """Open the same DataFigure host with its Fit tab selected."""

        if not self._artifact_operations.can_project_dataset(source):
            raise TypeError("source must be a composed FINAL Dataset artifact")
        initial_selection = None
        if committed_transform is not None:
            if not isinstance(committed_transform, CommittedTransform):
                raise TypeError(
                    "committed_transform must be CommittedTransform or None"
                )
            initial_selection = _fit_selection_authority(
                committed_transform,
                context="fit_gui",
            )
        return self._open_fit_capable_figure_gui(
            source,
            source,
            intent=None,
            point_ordinals=None,
            preferences=None,
            artifact_output=None,
            selected_model=model,
            initial_selection=initial_selection,
            open_fit=True,
            timeout_seconds=timeout_seconds,
            direct_fit_single_panel=True,
        )

    def figure_document(
        self,
        source: object,
        *,
        intent: "ViewIntent | None" = None,
        point_ordinals: tuple[int, ...] | None = None,
        preferences: "ViewPreferences | None" = None,
        output: str | None = None,
    ) -> "FigureDocument":
        """Project one committed source into a renderer-free document.

        A multi-output artifact may expose an owner-defined ``output`` name;
        ordinary Dataset artifacts and Fit results reject that argument.
        """

        with _service_guard(self._services) as services:
            document, _datasets, _figure_intent, _fit = _project_figure(
                services,
                self._artifact_operations,
                source,
                intent=intent,
                point_ordinals=point_ordinals,
                preferences=preferences,
                artifact_output=output,
                materialize=False,
            )
        return document

    def figure(
        self,
        source: object,
        *,
        intent: "ViewIntent | None" = None,
        point_ordinals: tuple[int, ...] | None = None,
        preferences: "ViewPreferences | None" = None,
        output: str | None = None,
    ) -> "DataFigure":
        """Resolve one frozen source and return its optional-render DataFigure."""

        with _service_guard(self._services) as services:
            figure, _figure_intent = _data_figure_for_services(
                services,
                self._artifact_operations,
                source,
                intent=intent,
                point_ordinals=point_ordinals,
                preferences=preferences,
                artifact_output=output,
            )
            return figure

    def figure_gui(
        self,
        source: object | str | Path | None = None,
        *,
        intent: "ViewIntent | None" = None,
        point_ordinals: tuple[int, ...] | None = None,
        preferences: "ViewPreferences | None" = None,
        output: str | None = None,
    ):
        """Open the saved-figure browser or show one typed frozen source.

        ``None`` opens the session-independent FigureViewer.  A filesystem path
        opens that viewer and commits the selected current archive; typed
        artifact/result inputs retain their existing interactive dispatch.
        """

        if source is None or isinstance(source, (str, Path)):
            supplied_overrides = tuple(
                name
                for name, value in (
                    ("intent", intent),
                    ("point_ordinals", point_ordinals),
                    ("preferences", preferences),
                    ("output", output),
                )
                if value is not None
            )
            if supplied_overrides:
                raise ValueError(
                    "saved FigureViewer does not accept typed-source view overrides: "
                    + ", ".join(supplied_overrides)
                )
            from zlc_workbench.figure_viewer.app import open_figure_viewer

            with _service_guard(self._services) as services:
                output_root = services.workspace_paths.output_root
            return open_figure_viewer(
                path=source,
                output_root=output_root,
            )

        if isinstance(source, FitResultArtifactRef):
            with _service_guard(self._services) as services:
                source = services.fit_repository.load(
                    source,
                    artifacts=self._artifact_operations,
                )

        if self._artifact_operations.can_project_dataset(source):
            return self._open_fit_capable_figure_gui(
                source,
                source,
                intent=intent,
                point_ordinals=point_ordinals,
                preferences=preferences,
                artifact_output=output,
            )

        if isinstance(source, (FitExecution, AdmittedFitResult)):
            result = source.result
            transform = result.spec.committed_transform
            initial_authority_selection = _fit_selection_authority(
                transform,
                context="Fit result analysis",
            )
            if isinstance(source, AdmittedFitResult):
                identity = (
                    f"{source.reference.repository_id}:"
                    f"{source.reference.manifest_digest}"
                )
                fit_source = source.source_artifact_ref
            else:
                identity = f"draft-execution:{id(source):x}"
                fit_source = source.source_artifact_ref
            return self._open_fit_capable_figure_gui(
                source,
                fit_source,
                intent=intent,
                point_ordinals=point_ordinals,
                preferences=preferences,
                artifact_output=output,
                selected_model=result.spec.model_id,
                initial_fit_spec=result.spec,
                initial_selection=initial_authority_selection,
                initial_fit_result_identity=identity,
            )

        from Zou_lab_control.workbench import open_figure_workbench
        with _service_guard(self._services) as services:
            output_root = services.workspace_paths.output_root

        def figure_factory(current_source, *, intent, point_ordinals, preferences):
            with _service_guard(self._services) as services:
                return _data_figure_for_services(
                    services,
                    self._artifact_operations,
                    current_source,
                    intent=intent,
                    point_ordinals=point_ordinals,
                    preferences=preferences,
                    artifact_output=output,
                )

        return open_figure_workbench(
            figure_factory,
            source,
            output_root=output_root,
            intent=intent,
            point_ordinals=point_ordinals,
            preferences=preferences,
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

            # Only this attempt owner may tear down the runtime or repositories.
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
            # its owner-thread terminal acknowledgement before repositories
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
            # still hold a worker borrow into the data plane.  Preserve every
            # repository and the frozen handle set so a later close attempt
            # can retry; never turn an acknowledgement failure into use-after-
            # close by continuing down the teardown chain.
            if gui_close_failures:
                raise _ResourceCleanupError(
                    "Experiment Workbench close failed",
                    tuple(gui_close_failures),
                )

            failures = (
                _cleanup_failures(services.signal_plane.close)
                + _cleanup_failures(services.fit_repository.close)
                + list(self.nodes.close())
                + _cleanup_failures(services.capture_repository.close)
            )
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
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
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
        return InstallationConfigDocument.from_parameters(
            "virtual",
            {} if seed is _CONNECT_SEED_UNSET else {"seed": seed},
        )
    if config == "remote_pulse":
        raise ValueError(
            "remote_pulse requires a saved installation config or "
            "InstallationConfigDocument.from_parameters('remote_pulse', ...)"
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
    repository_root = workspace.repository_root
    # The composition root owns the workspace hierarchy; each repository owns
    # exactly one child beneath it and never guesses missing ancestors.
    durable_makedirs(repository_root)
    capture_repository = None
    fit_repository = None
    runtime = None
    signal_plane = None
    try:
        capture_repository = CaptureRepository(repository_root / "captures")
        fit_repository = FitResultRepository(repository_root / "fits")
        from zlc_neutral_atom.installation_dispatch import create_installation

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
            capture_repository=capture_repository,
            fit_repository=fit_repository,
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
            + _cleanup_failures(
                None if fit_repository is None else fit_repository.close,
            )
            + _cleanup_failures(
                None if capture_repository is None else capture_repository.close,
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
    "AdmittedFitResult",
    "AppliedPulseSnapshot",
    "CaptureArtifactRef",
    "FitResultArtifactRef",
    "CaptureRequest",
    "connect",
    "device_manager",
    "Experiment",
    "FitExecution",
    "InstallationConfigDocument",
    "PlanDescriptor",
    "PreparedPulseExecution",
    "PulseFacade",
    "PulseRunDescriptor",
    "PulseRunObservation",
    "PulseRunRequest",
    "PulseRunResult",
    "PulseTargetDescriptor",
    "ReadoutFacade",
    "WorkspacePaths",
]
