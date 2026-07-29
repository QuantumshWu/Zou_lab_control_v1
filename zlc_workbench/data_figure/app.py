"""DataFigure Workbench composition and public launch functions."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from zlc_data import FitResultBatch, Selection
from zlc_frontend import (
    DataFigure,
)
from zlc_frontend.figure import ViewIntent
from zlc_frontend.fit_histogram_projection import _histogram_fit_display_state
from zlc_frontend.data_figure_presentation import (
    DATA_FIGURE_PANEL_ID,
    DataFigureDisplayState,
    DataFigureGridDisplayState,
    DataFigureGridFocusRequest,
    classify_faceted_data_figure,
    classify_single_data_figure,
    data_figure_initial_size_name,
    default_data_figure_display_state,
)
from zlc_frontend.panel_size import DEFAULT_PANEL_SIZE
from zlc_frontend.panel_params import panel_display_state_intent
from zlc_frontend.data_figure_render import (
    render_data_figure_front,
    render_data_figure_grid_overview,
    render_encoded_data_figure,
)
from zlc_frontend.plot_layout import PanelSurfaceGeometry, panel_surface_geometry
from zlc_frontend.plot_panel import (
    FigureIntent,
    PlotPanelContract,
    PlotPanelSession,
    figure_intent_from_view,
)
from zlc_frontend.qt_widgets import FigureSurfaceContext, launch_qt_window

from .fit_contract import (
    DEFAULT_FIT_TIMEOUT_SECONDS,
    FitSaveReceipt,
    FitWorkbenchBindings,
)
from .worker_jobs import _require_not_cancelled
from .window import DataFigureWindow


def _figure_window_factory(
    loader,
    *,
    output_root: Path,
    fit_bindings: FitWorkbenchBindings | None = None,
    initial_fit_result_identity: str | None = None,
    initial_display: DataFigureDisplayState | None = None,
    initial_grid_display: DataFigureGridDisplayState | None = None,
    embedded: bool = False,
    size_name: str = DEFAULT_PANEL_SIZE,
):
    if initial_display is not None:
        panel_display_state_intent(initial_display)
    if initial_grid_display is not None:
        panel_display_state_intent(initial_grid_display)
    if not isinstance(embedded, bool):
        raise TypeError("embedded must be bool")
    initial_geometry = panel_surface_geometry(size_name)
    size_name = initial_geometry.size_name
    worker_thread_id: int | None = None
    cached_source: DataFigure | None = None
    cached_figure_intent: FigureIntent | None = None
    cached_surface_intent: FigureIntent | None = None
    cached_base: DataFigure | None = None
    cached_typed_grid: tuple[ViewIntent, DataFigure, str | None] | None = None
    cached_grid_histogram_home_x_limits: tuple[float, float] | None = None
    render_session: PlotPanelSession | None = None

    def plot_session(contract: PlotPanelContract) -> PlotPanelSession:
        nonlocal render_session
        if render_session is None or render_session.contract != contract:
            if render_session is not None:
                render_session.close()
            render_session = PlotPanelSession(contract)
        return render_session

    def close_render_session() -> None:
        nonlocal render_session
        session, render_session = render_session, None
        if session is not None:
            session.close()

    def typed_completion(result, display, contract, overlays=None):
        if result.frame is None or result.figure is None:
            raise RuntimeError("single-panel render returned no Figure/frame")
        context = FigureSurfaceContext.for_frame(
            result.frame,
            figure=result.figure,
            display=display,
            contract=contract,
        )
        return result.frame, context, overlays

    def render_front(figure, display, contract, sequence, cancelled, **options):
        try:
            return render_data_figure_front(
                figure,
                display,
                contract=contract,
                sequence=sequence,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
                _session=plot_session(contract),
                **options,
            )
        except BaseException:
            close_render_session()
            raise

    def saved_fit_result(figure: DataFigure) -> FitResultBatch | None:
        results = tuple(figure.fit_results.values())
        if not results:
            return None
        if len(results) != 1:
            raise ValueError("typed DataFigure requires one exact Fit result")
        return results[0]

    def require_worker_owner() -> None:
        nonlocal worker_thread_id
        current = threading.get_ident()
        if worker_thread_id is None:
            worker_thread_id = current
        elif worker_thread_id != current:
            raise RuntimeError("figure view session changed worker thread")

    def initial(
        sequence: int,
        cancelled: threading.Event,
        geometry: PanelSurfaceGeometry,
    ):
        nonlocal cached_source, cached_figure_intent, cached_surface_intent
        nonlocal cached_base, cached_typed_grid
        require_worker_owner()
        _require_not_cancelled(cancelled)
        if not isinstance(geometry, PanelSurfaceGeometry):
            raise TypeError("DataFigure render requires PanelSurfaceGeometry")
        if geometry.size_name != size_name:
            raise ValueError("DataFigure surface changed its authored size")
        if cached_source is None:
            loaded = loader()
            if not isinstance(loaded, tuple) or len(loaded) != 2:
                raise TypeError("figure loader must return (DataFigure, FigureIntent)")
            figure, figure_intent = loaded
            if not isinstance(figure, DataFigure):
                raise TypeError("figure loader lost its DataFigure")
            if not isinstance(figure_intent, FigureIntent):
                raise TypeError("figure loader lost its FigureIntent")
            if (
                len(figure.document.layers) != 1
                or figure_intent.view != figure.document.layers[0].view
            ):
                raise ValueError("FigureIntent differs from its frozen DataFigure")
            cached_source = figure
            cached_figure_intent = figure_intent
        else:
            figure = cached_source
            figure_intent = cached_figure_intent
        if figure_intent is None:
            raise RuntimeError("typed Figure intent is unavailable")
        contract = PlotPanelContract(
            DATA_FIGURE_PANEL_ID,
            figure_intent,
            size_name=size_name,
            pixel_ratio=geometry.pixel_ratio,
        )
        intent, unavailable_reason = classify_single_data_figure(figure)
        if intent is not None:
            if initial_grid_display is not None:
                raise ValueError(
                    "a single-panel figure does not accept a grid display state"
                )
            if figure.has_fit_overlays and initial_fit_result_identity is None:
                return render_encoded_data_figure(
                    figure,
                    unavailable_reason=(
                        "typed Fit replay requires an exact caller-supplied result identity"
                    ),
                    check_cancelled=lambda: _require_not_cancelled(cancelled),
                )
            if not figure.has_fit_overlays and initial_fit_result_identity is not None:
                raise ValueError("Fit result identity was supplied for a source-only Figure")
            display = (
                default_data_figure_display_state(intent)
                if initial_display is None
                else initial_display
            )
            if panel_display_state_intent(display) is not intent:
                raise ValueError(
                    "saved display state does not match the figure view intent"
                )
            if figure_intent.faceted or figure_intent.view_intent is not intent:
                raise ValueError("single-panel FigureIntent differs from its data")
            fit_result = saved_fit_result(figure)
            base = figure.with_fit_results(None) if fit_result is not None else figure
            histogram_home = None
            if intent is ViewIntent.HISTOGRAM and fit_result is not None:
                display, histogram_home = _histogram_fit_display_state(
                    figure,
                    display,
                    fit_result,
                )
            rendered = render_front(
                base,
                display,
                contract,
                sequence,
                cancelled,
                histogram_projection_value_range=histogram_home,
            )
            cached_surface_intent = figure_intent
            cached_base = rendered.figure
            if fit_result is None:
                return typed_completion(rendered, display, contract)
            overlays = base.materialize_transient_fit_overlays(
                fit_result,
                rendered.frame,
                result_identity=initial_fit_result_identity,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
            )
            return typed_completion(rendered, display, contract, overlays)
        grid_intent, grid_panel_count, grid_reason = classify_faceted_data_figure(figure)
        if grid_intent is not None and grid_panel_count is not None:
            if not figure_intent.faceted or figure_intent.view_intent is not grid_intent:
                raise ValueError("Grid FigureIntent differs from its data")
            if initial_display is not None:
                raise ValueError(
                    "a multi-panel figure does not accept one single-panel display state"
                )
            if figure.has_fit_overlays and initial_fit_result_identity is None:
                return render_encoded_data_figure(
                    figure,
                    unavailable_reason=(
                        "typed Fit replay requires an exact caller-supplied result identity"
                    ),
                    check_cancelled=lambda: _require_not_cancelled(cancelled),
                )
            if not figure.has_fit_overlays and initial_fit_result_identity is not None:
                raise ValueError(
                    "Fit result identity was supplied for a source-only typed grid"
                )
            try:
                overview = render_data_figure_grid_overview(
                    figure,
                    contract=contract,
                    display_state=initial_grid_display,
                    fit_result_identity=initial_fit_result_identity,
                    check_cancelled=lambda: _require_not_cancelled(cancelled),
                    _session=plot_session(contract),
                )
            except BaseException:
                close_render_session()
                raise
            cached_typed_grid = (
                grid_intent,
                overview.figure,
                initial_fit_result_identity,
            )
            return overview
        if initial_grid_display is not None:
            raise ValueError(
                "a grid display state requires a supported typed grid figure"
            )
        return render_encoded_data_figure(
            figure,
            unavailable_reason=unavailable_reason or grid_reason,
            check_cancelled=lambda: _require_not_cancelled(cancelled),
        )

    def rerender(
        state: DataFigureDisplayState | DataFigureGridFocusRequest,
        sequence: int,
        cancelled: threading.Event,
        geometry: PanelSurfaceGeometry,
    ):
        nonlocal cached_base, cached_surface_intent
        nonlocal cached_grid_histogram_home_x_limits
        require_worker_owner()
        if not isinstance(geometry, PanelSurfaceGeometry):
            raise TypeError("DataFigure render requires PanelSurfaceGeometry")
        if geometry.size_name != size_name:
            raise ValueError("DataFigure surface changed its authored size")
        if isinstance(state, DataFigureGridFocusRequest):
            if cached_typed_grid is None:
                raise RuntimeError("typed grid focus has no frozen source")
            grid_intent, grid, grid_fit_result_identity = cached_typed_grid
            display = state.display
            if panel_display_state_intent(display) is not grid_intent:
                raise ValueError("typed grid focus intent changed after overview")
            cached_base = None
            focused = grid.focused_typed_panel(
                state.panel_index,
                expected_address=state.expected_address,
                expected_intent=grid_intent,
            )
            if cached_figure_intent is None:
                raise RuntimeError("Grid FigureIntent is unavailable")
            focused_intent = figure_intent_from_view(
                focused.document.layers[0].view,
                title=cached_figure_intent.title,
                value_label=cached_figure_intent.value_label,
            )
            contract = PlotPanelContract(
                DATA_FIGURE_PANEL_ID,
                focused_intent,
                size_name=size_name,
                pixel_ratio=geometry.pixel_ratio,
            )
            fit_result = saved_fit_result(focused)
            base = (
                focused.with_fit_results(None)
                if fit_result is not None
                else focused
            )
            rendered = render_front(
                base,
                display,
                contract,
                sequence,
                cancelled,
                histogram_projection_value_range=state.histogram_home_x_limits,
            )
            cached_surface_intent = focused_intent
            cached_base = rendered.figure
            cached_grid_histogram_home_x_limits = state.histogram_home_x_limits
            if fit_result is None:
                return typed_completion(rendered, display, contract)
            overlays = base.materialize_transient_fit_overlays(
                fit_result,
                rendered.frame,
                result_identity=grid_fit_result_identity,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
            )
            return typed_completion(rendered, display, contract, overlays)
        base = cached_base
        if base is None:
            raise RuntimeError("typed session has no frozen DataFigure")
        if cached_surface_intent is None:
            raise RuntimeError("typed surface FigureIntent is unavailable")
        contract = PlotPanelContract(
            DATA_FIGURE_PANEL_ID,
            cached_surface_intent,
            size_name=size_name,
            pixel_ratio=geometry.pixel_ratio,
        )
        rendered = render_front(
            base,
            state,
            contract,
            sequence,
            cancelled,
            histogram_projection_value_range=(
                cached_grid_histogram_home_x_limits
                if cached_typed_grid is not None
                else None
            ),
        )
        return typed_completion(rendered, state, contract)

    return lambda: DataFigureWindow(
        initial,
        rerender,
        fit_bindings=fit_bindings,
        worker_release=close_render_session,
        initial_display=initial_display,
        embedded=embedded,
        surface_size_name=size_name,
        output_root=output_root,
    )

def create_data_figure_pane(
    figure: DataFigure,
    figure_intent: FigureIntent,
    *,
    output_root: Path,
    initial_display: DataFigureDisplayState | None = None,
    initial_grid_display: DataFigureGridDisplayState | None = None,
    initial_fit_result_identity: str | None = None,
    local_fit: bool = False,
    local_fit_initial_selection: Selection | None = None,
    local_fit_archive_path: str | Path | None = None,
    local_fit_archive_metadata: Mapping[str, object] | None = None,
    open_fit: bool = False,
    embedded: bool = True,
    size_name: str | None = None,
) -> DataFigureWindow:
    """Build the one DataFigure interaction owner for embedding or launch.

    This is construction only.  FigureViewer uses the same body as a child, so
    saved figures cannot grow a second selector, Setting, Fit, or export
    implementation.  Typed artifact entry points use ``open_figure_workbench``.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    if not isinstance(figure_intent, FigureIntent):
        raise TypeError("figure_intent must be FigureIntent")
    resolved_size_name = (
        data_figure_initial_size_name(figure)
        if size_name is None
        else str(size_name)
    )
    if not isinstance(local_fit, bool):
        raise TypeError("local_fit must be bool")
    if not local_fit and any(
        value is not None
        for value in (
            local_fit_initial_selection,
            local_fit_archive_path,
            local_fit_archive_metadata,
        )
    ):
        raise ValueError("local Fit options require local_fit=True")
    if not local_fit and open_fit:
        raise ValueError("open_fit requires local_fit=True")
    fit_bindings = None
    if local_fit:
        from .local_fit import local_fit_bindings

        fit_bindings = local_fit_bindings(
            figure,
            initial_selection=local_fit_initial_selection,
            open_fit=open_fit,
            archive_path=local_fit_archive_path,
            archive_figure_intent=figure_intent,
            archive_size_name=resolved_size_name,
            archive_metadata=local_fit_archive_metadata,
        )
    return _figure_window_factory(
        lambda: (figure, figure_intent),
        output_root=output_root,
        fit_bindings=fit_bindings,
        initial_display=initial_display,
        initial_grid_display=initial_grid_display,
        initial_fit_result_identity=initial_fit_result_identity,
        embedded=embedded,
        size_name=resolved_size_name,
    )()

def open_figure_workbench(
    figure_factory,
    source,
    *,
    output_root: Path,
    intent=None,
    point_ordinals=None,
    preferences=None,
    fit_preparer=None,
    fit_executor=None,
    fit_saver=None,
    fit_reloader=None,
    fit_selected_model: str | None = None,
    fit_initial_selection: Selection | None = None,
    open_fit: bool = False,
    fit_timeout_seconds: float = DEFAULT_FIT_TIMEOUT_SECONDS,
    initial_fit_result_identity: str | None = None,
) -> DataFigureWindow:
    """Resolve and display a current artifact entirely on the worker lane."""

    if not callable(figure_factory):
        raise TypeError("figure_factory must be callable")
    fit_calls = (fit_preparer, fit_executor, fit_saver, fit_reloader)
    if any(item is not None for item in fit_calls) and not all(
        callable(item) for item in fit_calls
    ):
        raise ValueError("all four Figure Fit capabilities must be supplied together")
    fit_bindings = None
    if any(item is not None for item in fit_calls):
        from zlc_neutral_atom.artifacts import FitExecution, FitResultArtifactRef

        def execution_result(execution) -> FitResultBatch:
            if not isinstance(execution, FitExecution):
                raise TypeError("artifact Fit executor must return FitExecution")
            return execution.result

        def save_execution(execution, destination, _display) -> FitSaveReceipt:
            if destination is not None:
                raise ValueError("artifact Fit save does not accept an archive path")
            reference = fit_saver(execution)
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("artifact Fit saver returned another reference type")
            identity = f"{reference.repository_id}:{reference.manifest_digest}"
            return FitSaveReceipt(
                reference,
                identity,
                identity,
                artifact_reference=reference,
            )

        fit_bindings = FitWorkbenchBindings(
            fit_preparer,
            fit_executor,
            execution_result,
            save_execution,
            fit_reloader,
            selected_model=fit_selected_model,
            initial_selection=fit_initial_selection,
            open_fit=bool(open_fit),
            timeout_seconds=fit_timeout_seconds,
        )
    options = {
        "intent": intent,
        "point_ordinals": point_ordinals,
        "preferences": preferences,
    }
    return launch_qt_window(
        _figure_window_factory(
            lambda: figure_factory(source, **options),
            output_root=output_root,
            fit_bindings=fit_bindings,
            initial_fit_result_identity=initial_fit_result_identity,
        )
    )


__all__ = [
    "create_data_figure_pane",
    "open_figure_workbench",
]
