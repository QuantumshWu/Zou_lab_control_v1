"""DataFigure Workbench composition and public launch functions."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from zlc_data import FitResultBatch, Selection
from zlc_frontend import (
    DataFigure,
    FigurePresentationContract,
)
from zlc_frontend.figure import ViewIntent
from zlc_frontend.data_figure_presentation import (
    DataFigureDisplayState,
    DataFigureFront,
    DataFigureGridDisplayState,
    DataFigureGridFocusRequest,
    classify_faceted_data_figure,
    classify_single_data_figure,
    data_figure_initial_size_name,
    default_data_figure_display_state,
    display_state_intent,
    grid_display_state_intent,
)
from zlc_frontend.panel_size import DEFAULT_PANEL_SIZE
from zlc_frontend.data_figure_render import (
    DataFigureRenderSession,
    render_data_figure_grid_overview,
    render_encoded_data_figure,
)
from zlc_frontend.plot_layout import PanelSurfaceGeometry, panel_surface_geometry
from zlc_workbench.window_runtime import open_workbench_window

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
    fit_bindings: FitWorkbenchBindings | None = None,
    initial_fit_result_identity: str | None = None,
    initial_display: DataFigureDisplayState | None = None,
    initial_grid_display: DataFigureGridDisplayState | None = None,
    embedded: bool = False,
    size_name: str = DEFAULT_PANEL_SIZE,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
):
    if initial_display is not None:
        display_state_intent(initial_display)
    if initial_grid_display is not None:
        grid_display_state_intent(initial_grid_display)
    if not isinstance(embedded, bool):
        raise TypeError("embedded must be bool")
    for name, value in (
        ("presentation_title", presentation_title),
        ("presentation_value_label", presentation_value_label),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be text or None")
    initial_geometry = panel_surface_geometry(size_name)
    size_name = initial_geometry.size_name
    worker_thread_id: int | None = None
    cached_source: DataFigure | None = None
    cached_typed: DataFigure | None = None
    cached_base: DataFigure | None = None
    cached_typed_grid: tuple[ViewIntent, DataFigure, str | None] | None = None
    cached_grid_histogram_home_x_limits: tuple[float, float] | None = None
    render_session = DataFigureRenderSession()

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
        nonlocal cached_source, cached_typed, cached_base, cached_typed_grid
        require_worker_owner()
        _require_not_cancelled(cancelled)
        if not isinstance(geometry, PanelSurfaceGeometry):
            raise TypeError("DataFigure render requires PanelSurfaceGeometry")
        if geometry.size_name != size_name:
            raise ValueError("DataFigure surface changed its authored size")
        if cached_source is None:
            figure = loader()
            if not isinstance(figure, DataFigure):
                raise TypeError("figure loader must return DataFigure")
            cached_source = figure
        else:
            figure = cached_source
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
            if display_state_intent(display) is not intent:
                raise ValueError(
                    "saved display state does not match the figure view intent"
                )
            front = render_session.render_front(
                figure,
                display,
                sequence=sequence,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
                fit_result_identity=initial_fit_result_identity,
                raster_size=geometry.raster_size,
                size_name=size_name,
                pixel_ratio=geometry.pixel_ratio,
                presentation_title=presentation_title,
                presentation_value_label=presentation_value_label,
            )
            cached_typed = front.figure
            cached_base = (
                front.figure.with_fit_results(None)
                if front.figure.has_fit_overlays
                else front.figure
            )
            return front
        grid_intent, grid_panel_count, grid_reason = classify_faceted_data_figure(figure)
        if grid_intent is not None and grid_panel_count is not None:
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
            overview = render_data_figure_grid_overview(
                figure,
                raster_size=geometry.raster_size,
                size_name=size_name,
                pixel_ratio=geometry.pixel_ratio,
                display_state=initial_grid_display,
                presentation_title=presentation_title,
                presentation_value_label=presentation_value_label,
                fit_result_identity=initial_fit_result_identity,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
            )
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
        fit_result: FitResultBatch | None,
        fit_result_identity: str | None,
        state: DataFigureDisplayState | DataFigureGridFocusRequest,
        sequence: int,
        cancelled: threading.Event,
        geometry: PanelSurfaceGeometry,
    ) -> DataFigureFront:
        nonlocal cached_typed, cached_base, cached_grid_histogram_home_x_limits
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
            if display_state_intent(display) is not grid_intent:
                raise ValueError("typed grid focus intent changed after overview")
            if fit_result is not None or fit_result_identity is not None:
                raise ValueError(
                    "typed grid focus does not accept a transient Fit result"
                )
            cached_typed = None
            cached_base = None
            focused = grid.focused_typed_panel(
                state.panel_index,
                expected_address=state.expected_address,
                expected_intent=grid_intent,
            )
            front = render_session.render_front(
                focused,
                display,
                sequence=sequence,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
                fit_result_identity=grid_fit_result_identity,
                histogram_projection_value_range=state.histogram_home_x_limits,
                raster_size=geometry.raster_size,
                size_name=size_name,
                pixel_ratio=geometry.pixel_ratio,
                presentation_title=presentation_title,
                presentation_value_label=presentation_value_label,
            )
            cached_typed = focused
            cached_base = (
                focused.with_fit_results(None)
                if focused.has_fit_overlays
                else focused
            )
            cached_grid_histogram_home_x_limits = state.histogram_home_x_limits
            return front
        figure = cached_typed
        base = cached_base
        if base is None:
            raise RuntimeError("typed session has no frozen DataFigure")
        if fit_result is not None:
            render_figure = base
        elif fit_result_identity is None:
            render_figure = base
        elif (
            figure is not None
            and figure.has_fit_overlays
            and fit_result_identity == initial_fit_result_identity
        ):
            render_figure = figure
        else:
            raise ValueError("typed renderer has no exact result for this identity")
        releases_canonical = bool(
            render_figure is base
            and cached_typed is not None
            and cached_typed is not base
        )
        return render_session.render_front(
            render_figure,
            state,
            sequence=sequence,
            check_cancelled=lambda: _require_not_cancelled(cancelled),
            fit_result=fit_result,
            fit_result_identity=fit_result_identity,
            histogram_projection_value_range=(
                cached_grid_histogram_home_x_limits
                if cached_typed_grid is not None
                else None
            ),
            release_initial_canonical_on_commit=releases_canonical,
            raster_size=geometry.raster_size,
            size_name=size_name,
            pixel_ratio=geometry.pixel_ratio,
            presentation_title=presentation_title,
            presentation_value_label=presentation_value_label,
        )

    def commit_front(release_initial_canonical: bool) -> None:
        nonlocal cached_typed
        if release_initial_canonical:
            cached_typed = None

    return lambda: DataFigureWindow(
        initial,
        rerender,
        rerender if fit_bindings is not None else None,
        fit_bindings=fit_bindings,
        typed_front_committed=commit_front,
        worker_release=render_session.close,
        initial_display=initial_display,
        embedded=embedded,
        surface_size_name=size_name,
    )

def create_data_figure_pane(
    figure: DataFigure,
    *,
    initial_display: DataFigureDisplayState | None = None,
    initial_grid_display: DataFigureGridDisplayState | None = None,
    initial_fit_result_identity: str | None = None,
    local_fit: bool = False,
    local_fit_initial_selection: Selection | None = None,
    local_fit_archive_path: str | Path | None = None,
    local_fit_archive_presentation: FigurePresentationContract | None = None,
    local_fit_archive_metadata: Mapping[str, object] | None = None,
    open_fit: bool = False,
    embedded: bool = True,
    size_name: str | None = None,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
) -> DataFigureWindow:
    """Build the one DataFigure interaction owner for embedding or launch.

    This is construction only.  FigureViewer uses the same body as a child, so
    saved figures cannot grow a second selector, Setting, Fit, or export
    implementation.  Typed artifact entry points use ``open_figure_workbench``.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
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
            local_fit_archive_presentation,
            local_fit_archive_metadata,
        )
    ):
        raise ValueError("local Fit options require local_fit=True")
    if not local_fit and open_fit:
        raise ValueError("open_fit requires local_fit=True")
    if local_fit and not isinstance(
        local_fit_archive_presentation,
        FigurePresentationContract,
    ):
        raise TypeError(
            "local_fit requires one exact FigurePresentationContract"
        )
    fit_bindings = None
    if local_fit:
        from .local_fit import local_fit_bindings

        fit_bindings = local_fit_bindings(
            figure,
            initial_selection=local_fit_initial_selection,
            open_fit=open_fit,
            archive_path=local_fit_archive_path,
            archive_presentation=local_fit_archive_presentation,
            archive_metadata=local_fit_archive_metadata,
        )
    return _figure_window_factory(
        lambda: figure,
        fit_bindings=fit_bindings,
        initial_display=initial_display,
        initial_grid_display=initial_grid_display,
        initial_fit_result_identity=initial_fit_result_identity,
        embedded=embedded,
        size_name=resolved_size_name,
        presentation_title=presentation_title,
        presentation_value_label=presentation_value_label,
    )()

def open_figure_workbench(
    figure_factory,
    source,
    *,
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
    return open_workbench_window(
        _figure_window_factory(
            lambda: figure_factory(source, **options),
            fit_bindings=fit_bindings,
            initial_fit_result_identity=initial_fit_result_identity,
        )
    )


__all__ = [
    "create_data_figure_pane",
    "open_figure_workbench",
]
