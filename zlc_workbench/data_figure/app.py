"""DataFigure Workbench composition and public launch functions."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from pathlib import Path

from zlc_data import FitResultBatch, Selection
from zlc_frontend import (
    CurvePanelPayload,
    DataFigure,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
)
from zlc_frontend.figure import ViewIntent
from zlc_frontend.plot_layout import panel_display_size
from zlc_workbench.window_runtime import open_workbench_window

from .projection import (
    _DEFAULT_FIT_TIMEOUT_SECONDS,
    _FitSaveReceipt,
    _FitWorkbenchBindings,
    _GridDisplayState,
    _GridFocusRequest,
    _NUMERIC_RASTER_SIZE,
    _TypedDisplayState,
    _TypedPanelPayload,
    _classify_single_typed,
    _classify_typed_grid,
    _default_typed_state,
    _grid_state_intent,
    _payload_intent,
    _state_intent,
    _validate_rendered_authored_payload,
)
from .render_lane import (
    _encoded_figure,
    _render_typed_front,
    _render_typed_grid_overview,
    _require_not_cancelled,
)
from .window import DataFigureWindow


def _surface_geometry(
    size_name: str | None,
    pixel_ratio: float,
) -> tuple[tuple[int, int] | None, tuple[int, int], float]:
    ratio = float(pixel_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("pixel_ratio must be positive and finite")
    if size_name is None:
        logical_size = None
        base_size = _NUMERIC_RASTER_SIZE
    else:
        if not isinstance(size_name, str):
            raise TypeError("size_name must be text or None")
        from zlc_data.panel_size import panel_size_cells

        panel_size_cells(size_name)
        logical_size = tuple(int(value) for value in panel_display_size(size_name))
        base_size = logical_size
    raster_size = tuple(
        max(1, math.floor(value * ratio + 0.5))
        for value in base_size
    )
    return logical_size, raster_size, ratio


def _initial_payload_context(
    figure: DataFigure,
    display: _TypedDisplayState,
    payload: _TypedPanelPayload | None,
    fit_result_identity: str | None,
):
    if payload is None:
        return None, None, None
    intent = _state_intent(display)
    if _payload_intent(payload) is not intent:
        raise ValueError("initial payload does not match the figure display intent")
    if payload.evaluated_input != figure.evaluated.inputs[0]:
        raise ValueError("initial payload belongs to another evaluated input")
    _validate_rendered_authored_payload(
        payload,
        display,
        fit_result_identity,
    )
    if isinstance(payload, ImagePanelPayload):
        return payload.color_limits, display.relim_mode, None
    if isinstance(payload, CurvePanelPayload):
        return payload.viewport.y_limits, display.relim_mode, None
    if isinstance(payload, HistogramPanelPayload):
        return (
            payload.viewport.count_limits,
            display.relim_mode,
            display.count_scale,
        )
    if not isinstance(payload, MeterPanelPayload):
        raise TypeError("initial payload has another typed panel kind")
    return None, None, None


def _figure_window_factory(
    loader,
    *,
    fit_bindings: _FitWorkbenchBindings | None = None,
    initial_fit_result_identity: str | None = None,
    initial_display: _TypedDisplayState | None = None,
    initial_grid_display: _GridDisplayState | None = None,
    embedded: bool = False,
    surface_only: bool = False,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
    initial_payload: _TypedPanelPayload | None = None,
):
    if initial_display is not None:
        _state_intent(initial_display)
    if initial_grid_display is not None:
        _grid_state_intent(initial_grid_display)
    if initial_payload is not None:
        _payload_intent(initial_payload)
    if not isinstance(embedded, bool):
        raise TypeError("embedded must be bool")
    if not isinstance(surface_only, bool):
        raise TypeError("surface_only must be bool")
    if surface_only and not embedded:
        raise ValueError("surface_only requires embedded=True")
    for name, value in (
        ("presentation_title", presentation_title),
        ("presentation_value_label", presentation_value_label),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be text or None")
    logical_size, raster_size, pixel_ratio = _surface_geometry(
        size_name,
        pixel_ratio,
    )
    worker_thread_id: int | None = None
    cached_typed: DataFigure | None = None
    cached_base: DataFigure | None = None
    cached_typed_grid: tuple[ViewIntent, DataFigure] | None = None
    cached_grid_histogram_home_x_limits: tuple[float, float] | None = None

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
    ):
        nonlocal cached_typed, cached_base, cached_typed_grid
        require_worker_owner()
        _require_not_cancelled(cancelled)
        figure = loader()
        if not isinstance(figure, DataFigure):
            raise TypeError("figure loader must return DataFigure")
        intent, unavailable_reason = _classify_single_typed(figure)
        if intent is not None:
            if initial_grid_display is not None:
                raise ValueError(
                    "a single-panel figure does not accept a grid display state"
                )
            if figure.has_fit_overlays and initial_fit_result_identity is None:
                return _encoded_figure(
                    figure,
                    cancelled,
                    unavailable_reason=(
                        "typed Fit replay requires an exact caller-supplied result identity"
                    ),
                )
            if not figure.has_fit_overlays and initial_fit_result_identity is not None:
                raise ValueError("Fit result identity was supplied for a source-only Figure")
            display = (
                _default_typed_state(intent)
                if initial_display is None
                else initial_display
            )
            if _state_intent(display) is not intent:
                raise ValueError(
                    "saved display state does not match the figure view intent"
                )
            (
                current_value_limits,
                previous_relim_mode,
                previous_count_scale,
            ) = _initial_payload_context(
                figure,
                display,
                initial_payload,
                initial_fit_result_identity,
            )
            front = _render_typed_front(
                figure,
                display,
                current_value_limits=current_value_limits,
                previous_relim_mode=previous_relim_mode,
                previous_count_scale=previous_count_scale,
                sequence=sequence,
                cancelled=cancelled,
                fit_result_identity=initial_fit_result_identity,
                raster_size=raster_size,
                size_name=size_name,
                pixel_ratio=pixel_ratio,
                presentation_title=presentation_title,
                presentation_value_label=presentation_value_label,
            )
            cached_typed = figure
            cached_base = (
                figure.with_fit_results(None)
                if figure.has_fit_overlays
                else figure
            )
            return front
        grid_intent, grid_panel_count, grid_reason = _classify_typed_grid(figure)
        if grid_intent is not None and grid_panel_count is not None:
            if initial_payload is not None:
                raise ValueError(
                    "a grid overview does not accept one focused panel payload"
                )
            if initial_display is not None:
                raise ValueError(
                    "a multi-panel figure does not accept one single-panel display state"
                )
            if initial_fit_result_identity is not None:
                raise ValueError("Fit result identity was supplied for a typed grid")
            overview = _render_typed_grid_overview(
                figure,
                cancelled,
                raster_size=raster_size,
                size_name=size_name,
                pixel_ratio=pixel_ratio,
                display_state=initial_grid_display,
                presentation_title=presentation_title,
                presentation_value_label=presentation_value_label,
            )
            cached_typed_grid = (grid_intent, figure)
            return overview
        if initial_grid_display is not None:
            raise ValueError(
                "a grid display state requires a supported typed grid figure"
            )
        if initial_payload is not None:
            raise ValueError(
                "an initial panel payload requires a supported single-panel figure"
            )
        return _encoded_figure(
            figure,
            cancelled,
            unavailable_reason=unavailable_reason or grid_reason,
        )

    def rerender(
        fit_result: FitResultBatch | None,
        fit_result_identity: str | None,
        state: _TypedDisplayState | _GridFocusRequest,
        current_value_limits,
        previous_relim_mode,
        previous_count_scale,
        sequence: int,
        cancelled: threading.Event,
    ) -> _TypedFigureFront:
        nonlocal cached_typed, cached_base, cached_grid_histogram_home_x_limits
        require_worker_owner()
        if isinstance(state, _GridFocusRequest):
            if cached_typed_grid is None:
                raise RuntimeError("typed grid focus has no frozen source")
            grid_intent, grid = cached_typed_grid
            display = state.display
            if _state_intent(display) is not grid_intent:
                raise ValueError("typed grid focus intent changed after overview")
            if fit_result is not None or fit_result_identity is not None:
                raise ValueError("typed grid focus cannot carry a Fit result")
            cached_typed = None
            cached_base = None
            focused = grid.focused_typed_panel(
                state.panel_index,
                expected_selection=state.expected_selection,
                expected_intent=grid_intent,
            )
            front = _render_typed_front(
                focused,
                display,
                current_value_limits=None,
                previous_relim_mode=None,
                previous_count_scale=None,
                sequence=sequence,
                cancelled=cancelled,
                histogram_projection_value_range=state.histogram_home_x_limits,
                raster_size=raster_size,
                size_name=size_name,
                pixel_ratio=pixel_ratio,
                presentation_title=presentation_title,
                presentation_value_label=presentation_value_label,
            )
            cached_typed = focused
            cached_base = focused
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
        return _render_typed_front(
            render_figure,
            state,
            current_value_limits=current_value_limits,
            previous_relim_mode=previous_relim_mode,
            previous_count_scale=previous_count_scale,
            sequence=sequence,
            cancelled=cancelled,
            fit_result=fit_result,
            fit_result_identity=fit_result_identity,
            histogram_projection_value_range=(
                cached_grid_histogram_home_x_limits
                if cached_typed_grid is not None
                else None
            ),
            release_initial_canonical_on_commit=releases_canonical,
            raster_size=raster_size,
            size_name=size_name,
            pixel_ratio=pixel_ratio,
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
        initial_display=initial_display,
        embedded=embedded,
        surface_only=surface_only,
        logical_panel_size=logical_size,
        size_name=size_name,
        pixel_ratio=pixel_ratio,
        presentation_title=presentation_title,
        presentation_value_label=presentation_value_label,
    )

def create_data_figure_pane(
    figure: DataFigure,
    *,
    initial_display: _TypedDisplayState | None = None,
    initial_grid_display: _GridDisplayState | None = None,
    initial_fit_result_identity: str | None = None,
    local_fit: bool = False,
    local_fit_initial_selection: Selection | None = None,
    local_fit_archive_path: str | Path | None = None,
    local_fit_archive_metadata: Mapping[str, object] | None = None,
    open_fit_analysis: bool = False,
    embedded: bool = True,
    surface_only: bool = False,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
    initial_payload: _TypedPanelPayload | None = None,
) -> DataFigureWindow:
    """Build the one DataFigure interaction owner for embedding or launch.

    This is construction only: callers that need a top-level window still go
    through :func:`open_data_figure_workbench`.  FigureViewer uses the same
    body as a child, so saved figures cannot grow a second selector, Setting,
    Fit, or export implementation.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
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
    if not local_fit and open_fit_analysis:
        raise ValueError("open_fit_analysis requires local_fit=True")
    fit_bindings = None
    if local_fit:
        from .local_fit import local_fit_bindings

        fit_bindings = local_fit_bindings(
            figure,
            initial_selection=local_fit_initial_selection,
            open_analysis=open_fit_analysis,
            archive_path=local_fit_archive_path,
            archive_metadata=local_fit_archive_metadata,
        )
    return _figure_window_factory(
        lambda: figure,
        fit_bindings=fit_bindings,
        initial_display=initial_display,
        initial_grid_display=initial_grid_display,
        initial_fit_result_identity=initial_fit_result_identity,
        embedded=embedded,
        surface_only=surface_only,
        size_name=size_name,
        pixel_ratio=pixel_ratio,
        presentation_title=presentation_title,
        presentation_value_label=presentation_value_label,
        initial_payload=initial_payload,
    )()

def open_data_figure_workbench(
    figure: DataFigure,
    *,
    initial_display: _TypedDisplayState | None = None,
) -> DataFigureWindow:
    """Open an already-resolved DataFigure on the shared raster lane."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    return open_workbench_window(
        _figure_window_factory(
            lambda: figure,
            initial_display=initial_display,
        )
    )


def open_local_data_figure_analysis(
    figure: DataFigure,
    *,
    initial_selection: Selection | None = None,
    archive_path: str | Path | None = None,
    archive_metadata: Mapping[str, object] | None = None,
    initial_display: _TypedDisplayState | None = None,
    open_analysis: bool = True,
) -> DataFigureWindow:
    """Open one frozen panel in the sole DataFigure/Fit host.

    Unlike Capture/Scan Fit this path has no neutral artifact authority.  Save
    therefore publishes a current ``DataFigure`` archive (asking for a path
    when the caller did not already open one).
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    from .local_fit import local_fit_bindings

    bindings = local_fit_bindings(
        figure,
        initial_selection=initial_selection,
        open_analysis=open_analysis,
        archive_path=archive_path,
        archive_metadata=archive_metadata,
    )
    return open_workbench_window(
        _figure_window_factory(
            lambda: figure,
            fit_bindings=bindings,
            initial_display=initial_display,
        )
    )


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    occupancy_output=None,
    fit_preparer=None,
    fit_executor=None,
    fit_saver=None,
    fit_reloader=None,
    fit_selected_model: str | None = None,
    fit_initial_selection: Selection | None = None,
    open_fit_analysis: bool = False,
    fit_timeout_seconds: float = _DEFAULT_FIT_TIMEOUT_SECONDS,
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

        def save_execution(execution, destination, _display) -> _FitSaveReceipt:
            if destination is not None:
                raise ValueError("artifact Fit save does not accept an archive path")
            reference = fit_saver(execution)
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("artifact Fit saver returned another reference type")
            identity = f"{reference.repository_id}:{reference.manifest_digest}"
            return _FitSaveReceipt(
                reference,
                identity,
                identity,
                artifact_reference=reference,
            )

        fit_bindings = _FitWorkbenchBindings(
            fit_preparer,
            fit_executor,
            execution_result,
            save_execution,
            fit_reloader,
            selected_model=fit_selected_model,
            initial_selection=fit_initial_selection,
            open_analysis=bool(open_fit_analysis),
            timeout_seconds=fit_timeout_seconds,
        )
    options = {
        "intent": intent,
        "selection": selection,
        "preferences": preferences,
    }
    if occupancy_output is not None:
        options["occupancy_output"] = occupancy_output
    return open_workbench_window(
        _figure_window_factory(
            lambda: figure_factory(source, **options),
            fit_bindings=fit_bindings,
            initial_fit_result_identity=initial_fit_result_identity,
        )
    )


__all__ = [
    "create_data_figure_pane",
    "open_data_figure_workbench",
    "open_figure_workbench",
    "open_local_data_figure_analysis",
]
