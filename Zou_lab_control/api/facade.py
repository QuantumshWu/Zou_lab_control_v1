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
    AxisId,
    CommittedTransform,
    FitNumericPolicy,
    FitParameterConstraint,
    FitResultBatch,
    FitSpec,
    Selection,
    bind_fit,
    fit_model_catalog,
    fit_spec_for,
    suggest_fit_draft,
)
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
    WorkbenchHandle as _WorkbenchHandle,
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

    def _open_workbench_handle(
        self,
        key: str,
        compose: Callable[[], object],
        *,
        existing_error: str | None = None,
    ):
        """Own one GUI handle without exposing the application service graph.

        Concrete Workbench construction lives in ``Zou_lab_control.workbench``;
        this method owns only Experiment-scoped reuse and retirement.
        """

        name = _text(key, "workbench key")
        if not callable(compose):
            raise TypeError("compose must be callable")
        with _service_guard(self._services) as services:
            existing = services.gui_handles.get(name)
            if existing is not None:
                if existing.permanently_closed:
                    # A committed teardown can race a later API reopen by
                    # one owner turn.  Retire the exact dead handle instead of
                    # leaving an un-restorable registry tombstone forever.
                    services.gui_handles.pop(name)
                    existing = None
            if existing is not None:
                if existing_error is not None:
                    raise RuntimeError(existing_error)
                existing.restore_window()
                return existing
            body = compose()
            if not isinstance(body, _WorkbenchHandle):
                raise TypeError(
                    f"{name} composition did not return the application GUI handle port"
                )
            services.gui_handles[name] = body
            return body

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
        fit_axis_ids: tuple[AxisId, ...] | None = None,
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
                schema = _project_final_dataset_source(
                    self._artifact_operations,
                    source,
                    materialize=False,
                ).schema
                spec = fit_spec_for(
                    schema,
                    model,
                    committed_transform=committed_transform,
                    fit_axis_ids=fit_axis_ids,
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
                    fit_axis_ids,
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
        selection: Selection | None,
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
        """Compose the one Figure-owned Fit host without exposing repositories."""

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

        def source_schema(services):
            return _project_final_dataset_source(
                self._artifact_operations,
                fit_source,
                materialize=False,
            ).schema

        def prepare_fit(
            fit_axis_ids: tuple[AxisId, ...],
            authority_selection: Selection | None,
        ):
            if not fit_axis_ids or any(
                not isinstance(axis_id, AxisId) for axis_id in fit_axis_ids
            ):
                raise ValueError("Figure Fit requires exact named display axes")
            if authority_selection is not None and not isinstance(
                authority_selection,
                Selection,
            ):
                raise TypeError("Figure Fit selection must be Selection or None")
            with _service_guard(self._services) as services:
                schema = source_schema(services)
            seed_spec = initial_fit_spec
            if seed_spec is not None:
                if seed_spec.input_schema_fingerprint != schema.fingerprint:
                    raise ValueError("initial FitSpec belongs to another source schema")
                if seed_spec.fit_axis_ids != tuple(fit_axis_ids):
                    raise ValueError(
                        "displayed named Fit axes differ from the initial FitSpec"
                    )
            from zlc_frontend import fit_authoring_option

            options = []
            for definition in fit_model_catalog():
                try:
                    bound = suggest_fit_draft(
                        schema,
                        definition.model_id,
                        fit_axis_ids=tuple(fit_axis_ids),
                        selection=authority_selection,
                        constraints=(
                            seed_spec.constraints
                            if seed_spec is not None
                            and definition.model_id == seed_spec.model_id
                            else ()
                        ),
                        numeric_policy=(
                            seed_spec.numeric_policy
                            if seed_spec is not None
                            and definition.model_id == seed_spec.model_id
                            else FitNumericPolicy()
                        ),
                    )
                    if (
                        seed_spec is not None
                        and definition.model_id == seed_spec.model_id
                        and bound.spec.committed_transform
                        == seed_spec.committed_transform
                    ):
                        del bound
                        bound = bind_fit(seed_spec, schema)
                except ValueError:
                    continue
                option = fit_authoring_option(bound)
                options.append(option)
                del bound
            if not options:
                raise ValueError(
                    "the displayed named axes and selection have no compatible Fit model"
                )
            if chosen_model is not None and chosen_model not in {
                option.spec.model_id for option in options
            }:
                raise ValueError(
                    f"Fit model {chosen_model!r} is not compatible with this panel"
                )
            return tuple(options)

        def execute_fit(
            spec: FitSpec,
            cancel_check,
            deadline_monotonic: float,
        ) -> FitExecution:
            if not isinstance(spec, FitSpec):
                raise TypeError("Figure Fit execution requires FitSpec")
            with _fit_service_guard(self._services) as services:
                def combined_cancel_check() -> bool:
                    with services.operation_lock:
                        closing = services.state != "OPEN"
                    return closing or bool(cancel_check())

                return services.fit_repository.execute(
                    self._artifact_operations,
                    fit_source,
                    spec,
                    cancel_check=combined_cancel_check,
                    deadline_monotonic=deadline_monotonic,
                )

        def save_fit_execution(
            execution: FitExecution,
        ) -> FitResultArtifactRef:
            if not isinstance(execution, FitExecution):
                raise TypeError("Fit save requires FitExecution")
            if execution.source_artifact_ref != fit_source:
                raise ValueError("Fit execution belongs to another source artifact")
            with _service_guard(self._services):
                return execution.save()

        def reload_fit_result(
            reference: FitResultArtifactRef,
        ) -> FitResultBatch:
            with _service_guard(self._services) as services:
                admitted = services.fit_repository.load(
                    reference,
                    artifacts=self._artifact_operations,
                )
            if admitted.source_artifact_ref != fit_source:
                raise ValueError("saved Fit reopened against another source artifact")
            return admitted.result

        def figure_factory(source, *, intent, selection, preferences):
            return self.figure(
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                output=artifact_output,
            )

        if direct_fit_single_panel:
            if display_source != fit_source:
                raise ValueError(
                    "direct Fit single-panel display requires its exact source artifact"
                )

            def figure_factory(
                source,
                *,
                intent,
                selection,
                preferences,
            ):
                """Resolve the labelled display cell on the Figure worker."""

                if source != fit_source:
                    raise ValueError("direct Fit Figure loader received another source")
                with _service_guard(self._services) as services:
                    source_projection = _project_final_dataset_source(
                        self._artifact_operations,
                        fit_source,
                        materialize=False,
                    )
                    schema = source_projection.schema
                    seed_document, _datasets, _fit_result = (
                        _project_figure(
                            services,
                            self._artifact_operations,
                            source,
                            intent=intent,
                            selection=selection,
                            preferences=preferences,
                            artifact_output=None,
                            materialize=False,
                            preprojected_source=source_projection,
                        )
                    )
                    from zlc_frontend.figure import (
                        fit_single_panel_presentation,
                    )

                    display_selection, display_preferences = (
                        fit_single_panel_presentation(
                            schema,
                            seed_document.layers[0].view,
                            preferences,
                        )
                    )
                    del schema, source_projection, seed_document
                    return _data_figure_for_services(
                        services,
                        self._artifact_operations,
                        source,
                        intent=intent,
                        selection=display_selection,
                        preferences=display_preferences,
                        artifact_output=None,
                    )

        from Zou_lab_control.workbench import open_figure_workbench

        return open_figure_workbench(
            figure_factory,
            display_source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            fit_preparer=prepare_fit,
            fit_executor=execute_fit,
            fit_saver=save_fit_execution,
            fit_reloader=reload_fit_result,
            fit_selected_model=chosen_model,
            fit_initial_selection=initial_selection,
            open_fit=open_fit,
            fit_timeout_seconds=timeout,
            initial_fit_result_identity=initial_fit_result_identity,
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
            operations = committed_transform.spec.operations
            if len(operations) != 1 or not isinstance(operations[0], Selection):
                raise ValueError(
                    "fit_gui accepts only one range-preserving Selection transform"
                )
            initial_selection = operations[0]
        return self._open_fit_capable_figure_gui(
            source,
            source,
            intent=None,
            selection=None,
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
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
        output: str | None = None,
    ) -> "FigureDocument":
        """Project one committed source into a renderer-free document.

        A multi-output artifact may expose an owner-defined ``output`` name;
        ordinary Dataset artifacts and Fit results reject that argument.
        """

        with _service_guard(self._services) as services:
            document, _datasets, _fit = _project_figure(
                services,
                self._artifact_operations,
                source,
                intent=intent,
                selection=selection,
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
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
        output: str | None = None,
    ) -> "DataFigure":
        """Resolve one frozen source and return its optional-render DataFigure."""

        with _service_guard(self._services) as services:
            return _data_figure_for_services(
                services,
                self._artifact_operations,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                artifact_output=output,
            )

    def figure_gui(
        self,
        source: object | str | Path | None = None,
        *,
        intent: "ViewIntent | None" = None,
        selection: Selection | None = None,
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
                    ("selection", selection),
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

            return open_figure_viewer(path=source)

        if (
            isinstance(source, FitResultArtifactRef)
            and intent is None
            and selection is None
            and preferences is None
            and output is None
        ):
            experiment_services = self._services
            session_thread_id = None
            session_admitted = None
            session_source = None
            session_model = None

            def load_saved_fit_grid_view(
                reference,
                *,
                page_address,
                cell_selection,
            ):
                nonlocal session_thread_id
                nonlocal session_admitted
                nonlocal session_source
                nonlocal session_model
                if reference != source:
                    raise ValueError("saved-fit loader received another artifact ref")
                worker_thread_id = threading.get_ident()
                if session_thread_id is None:
                    session_thread_id = worker_thread_id
                elif session_thread_id != worker_thread_id:
                    raise RuntimeError(
                        "saved-fit view session changed worker thread"
                    )
                with _service_guard(experiment_services) as services:
                    from zlc_frontend import FitGridModel

                    if session_admitted is None:
                        admitted = services.fit_repository.load(
                            reference,
                            artifacts=self._artifact_operations,
                        )
                        model = FitGridModel.from_result(
                            reference.target_ref,
                            admitted.result,
                        )
                        source_projection = _project_final_dataset_source(
                            self._artifact_operations,
                            admitted.source_artifact_ref,
                            materialize=True,
                        )
                        session_admitted = admitted
                        session_source = source_projection
                        session_model = model
                    else:
                        admitted = session_admitted
                        source_projection = session_source
                        model = session_model
                        assert source_projection is not None and model is not None
                    if cell_selection is None:
                        page = model.page(page_address)
                        resolved_selection = page.selection
                        resolved_preferences = page.preferences
                        cell_summary = None
                    else:
                        if page_address is not None:
                            raise ValueError(
                                "saved-fit view cannot request a page and cell together"
                            )
                        model.resolve_selection(cell_selection)
                        page = None
                        resolved_selection = cell_selection
                        resolved_preferences = model.focus_preferences()
                        cell_summary = model.cell_summary(
                            admitted.result,
                            cell_selection,
                        )
                    figure = _data_figure_for_services(
                        services,
                        self._artifact_operations,
                        admitted,
                        intent=None,
                        selection=resolved_selection,
                        preferences=resolved_preferences,
                        artifact_output=None,
                        preprojected_source=source_projection,
                    )
                return figure, model, page, cell_summary

            def open_saved_fit_refit(reference, cell_selection):
                if reference != source:
                    raise ValueError("saved-fit refit received another artifact ref")
                admitted = session_admitted
                model = session_model
                if admitted is None or model is None:
                    raise RuntimeError("saved-fit refit requires an admitted grid session")
                model.resolve_selection(cell_selection)
                result = admitted.result
                fit_source = admitted.source_artifact_ref
                transform = result.spec.committed_transform
                authority_selection = None
                if transform is not None:
                    operations = transform.spec.operations
                    if len(operations) != 1 or not isinstance(
                        operations[0], Selection
                    ):
                        raise ValueError(
                            "saved Fit refit requires its exact range-preserving "
                            "Selection authority"
                        )
                    authority_selection = operations[0]
                return self._open_fit_capable_figure_gui(
                    fit_source,
                    fit_source,
                    intent=None,
                    # The focused batch cell chooses only the displayed source panel.
                    selection=cell_selection,
                    preferences=model.focus_preferences(),
                    artifact_output=None,
                    selected_model=result.spec.model_id,
                    initial_fit_spec=result.spec,
                    initial_selection=authority_selection,
                    open_fit=True,
                )

            from Zou_lab_control.workbench import open_saved_fit_grid_workbench

            return open_saved_fit_grid_workbench(
                load_saved_fit_grid_view,
                open_saved_fit_refit,
                source,
            )

        if self._artifact_operations.can_project_dataset(source):
            return self._open_fit_capable_figure_gui(
                source,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                artifact_output=output,
            )

        if isinstance(source, (FitExecution, AdmittedFitResult)):
            result = source.result
            transform = result.spec.committed_transform
            initial_authority_selection = None
            if transform is not None:
                operations = transform.spec.operations
                if len(operations) != 1 or not isinstance(
                    operations[0], Selection
                ):
                    raise ValueError(
                        "Fit result analysis requires its exact range-preserving "
                        "Selection authority"
                    )
                initial_authority_selection = operations[0]
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
                selection=selection,
                preferences=preferences,
                artifact_output=output,
                selected_model=result.spec.model_id,
                initial_fit_spec=result.spec,
                initial_selection=initial_authority_selection,
                initial_fit_result_identity=identity,
            )

        from Zou_lab_control.workbench import open_figure_workbench

        initial_identity = (
            f"{source.repository_id}:{source.manifest_digest}"
            if isinstance(source, FitResultArtifactRef)
            else None
        )

        def figure_factory(current_source, *, intent, selection, preferences):
            return self.figure(
                current_source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                output=output,
            )

        return open_figure_workbench(
            figure_factory,
            source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            initial_fit_result_identity=initial_identity,
        )

    def close(self) -> None:
        services = self._services
        caller_thread_id = threading.get_ident()
        with services.operation_lock:
            if services.state == "CLOSED":
                return
            if services.fit_operation_thread_counts.get(caller_thread_id, 0):
                raise RuntimeError(
                    "Experiment cannot close reentrantly from its active Fit operation"
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
                fit_operations_drained = services.fit_operations_drained
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
            # A Figure Fit owns repository borrows outside the short
            # composition lock. Closing flips the state first, then waits
            # without holding the lock needed by the Fit's finally block.
            fit_operations_drained.wait()
            with services.operation_lock:
                if services.active_fit_operations:
                    raise RuntimeError(
                        "Fit drain event disagrees with operation count"
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
                _cleanup_failures(services.fit_repository.close)
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
        start_run=services.runtime.start,
        on_applied=services.pulse_application._record_applied,
    )


def _start_pulse(services: _ExperimentServices, request: PulseRunRequest) -> RunHandle:
    with _service_guard(services) as guarded:
        prepared = _prepare_pulse_for_services(guarded, request)
        return guarded.pulse_application.start(prepared)


def _run_pulse(services: _ExperimentServices, request: PulseRunRequest) -> PulseRunResult:
    with _service_guard(services) as guarded:
        prepared = _prepare_pulse_for_services(guarded, request)
        handle = guarded.pulse_application.start(prepared)
        runtime = guarded.runtime
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
    repository: str | Path,
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
    if not isinstance(repository, (str, Path)):
        raise TypeError("repository must be an explicit experiment workspace root")
    canonical_name = _text(name, "experiment name")
    repository_root = Path(repository).expanduser().resolve()
    # The composition root owns the workspace hierarchy; each repository owns
    # exactly one child beneath it and never guesses missing ancestors.
    durable_makedirs(repository_root)
    capture_repository = None
    fit_repository = None
    runtime = None
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
        fit_operations_drained = threading.Event()
        fit_operations_drained.set()
        services = _ExperimentServices(
            repository_root=repository_root,
            installation=installation,
            runtime=runtime,
            capture_repository=capture_repository,
            fit_repository=fit_repository,
            catalog=catalog,
            installation_config=installation_document,
            pulse_application=PulseApplicationOwner(),
            operation_lock=threading.RLock(),
            fit_operations_drained=fit_operations_drained,
            fit_operation_thread_counts={},
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
    repository: str | Path | None = None,
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
        repository=repository,
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
]
