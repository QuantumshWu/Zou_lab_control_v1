"""Canonical DataFigure raster composition and immutable front construction."""

from __future__ import annotations

from collections.abc import Callable
from .data_figure import DataFigure
from .render import BoardFrame
from .display_range import validated_display_range
from .data_figure_presentation import (
    DATA_FIGURE_PANEL_ID,
    DataFigureDisplayState,
    DataFigureGridDisplayState,
    DataFigureGridOverview,
    DataFigurePanelPayload,
    classify_faceted_data_figure,
    classify_single_data_figure,
    data_figure_join_digest,
    data_figure_payload_intent,
    data_figure_summary,
    default_data_figure_display_state,
    validate_rendered_data_figure_payload,
)
from .encoded_raster import EncodedRasterDocument, EncodedRasterPage
from .encoded_raster import encode_raster_buffer_png
from .fit_histogram_projection import _histogram_fit_display_state
from .figure import ViewIntent
from .figure_source import FigureSource
from .histogram_display import histogram_projection_home_x_limits
from .plot_panel import (
    PlotPanelComposeRequest,
    PlotPanelComposeResult,
    PlotPanelContract,
    PlotPanelSession,
)
from .plot_kind import PlotKind
from .panel_render import PanelProvenance
from .panel_params import panel_display_state_intent


def _check(check_cancelled: Callable[[], None] | None) -> None:
    if check_cancelled is not None:
        check_cancelled()


def _plot_session(
    contract: PlotPanelContract,
    session: PlotPanelSession | None,
) -> tuple[PlotPanelSession, bool]:
    if session is None:
        return PlotPanelSession(contract), True
    if not isinstance(session, PlotPanelSession) or session.contract != contract:
        raise ValueError("provided PlotPanelSession has another contract")
    return session, False


def render_encoded_data_figure(
    figure: DataFigure,
    *,
    unavailable_reason: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> EncodedRasterDocument:
    _check(check_cancelled)
    payload = figure.to_png_bytes()
    _check(check_cancelled)
    summary = data_figure_summary(figure)
    if unavailable_reason is not None:
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            raise ValueError("unavailable_reason must be non-empty text or None")
        summary = f"{summary} · interaction unavailable: {unavailable_reason.strip()}"
    return EncodedRasterDocument(
        summary,
        (EncodedRasterPage("figure", "Figure", payload),),
    )

def render_data_figure_grid_overview(
    figure: DataFigure,
    *,
    contract: PlotPanelContract,
    display_state: DataFigureGridDisplayState | None = None,
    fit_result_identity: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
    _session: PlotPanelSession | None = None,
) -> DataFigureGridOverview:
    intent, panel_count, reason = classify_faceted_data_figure(figure)
    if intent is None or panel_count is None:
        raise ValueError(
            "typed grid overview requires a supported multi-cell figure"
            + ("" if reason is None else f": {reason}")
        )
    _check(check_cancelled)
    if display_state is None:
        display_state = default_data_figure_display_state(intent)
    elif panel_display_state_intent(display_state) is not intent:
        raise ValueError("typed grid display state does not match the figure intent")
    if figure.has_fit_overlays != (fit_result_identity is not None):
        raise ValueError(
            "typed grid Fit replay requires one exact result identity"
        )
    if not isinstance(contract, PlotPanelContract):
        raise TypeError("typed grid render requires PlotPanelContract")
    if contract.figure.kind is not PlotKind.GRID:
        raise ValueError("typed grid render requires a GRID FigureIntent")
    overlay_result = None
    base_figure = figure
    if figure.has_fit_overlays:
        results = tuple(figure.fit_results.values())
        if len(results) != 1:
            raise ValueError("typed grid requires one exact Fit result")
        overlay_result = results[0]
        base_figure = figure.with_fit_results(None)
    histogram_home = None
    if intent is ViewIntent.HISTOGRAM:
        if overlay_result is not None:
            display_state, histogram_home = _histogram_fit_display_state(
                figure,
                display_state,
                overlay_result,
            )
        else:
            histogram_display = getattr(display_state, "display", display_state)
            histogram_home = histogram_projection_home_x_limits(
                tuple(
                    series.data.samples
                    for cell in figure.evaluated.layers[0].cells
                    for series in cell.series
                ),
                bins=histogram_display.bin_count,
            )
    _check(check_cancelled)
    layer = base_figure.document.layers[0]
    if contract.figure.view != layer.view:
        raise ValueError("typed grid contract differs from its Figure view")
    snapshot = base_figure.datasets.resolve(layer.dataset_id)
    session, owns_session = _plot_session(contract, _session)
    try:
        rendered = session.compose_data_figure_grid(
            base_figure,
            PlotPanelComposeRequest(
                FigureSource(snapshot),
                display_state,
                PanelProvenance(
                    f"figure:{snapshot.ref.block_id.value}",
                    snapshot.ref.stream_generation.value,
                    data_figure_join_digest(
                        base_figure,
                        intent,
                        fit_result_identity,
                    ),
                ),
                fit_result=overlay_result,
                fit_result_identity=fit_result_identity,
                check_cancelled=check_cancelled,
            )
        )
    finally:
        if owns_session:
            session.close()
    if rendered.faceted is None or rendered.faceted.overview is None:
        raise RuntimeError("PlotPanel grid returned no overview")
    overview = rendered.faceted.overview
    visible_figure = rendered.figure
    if visible_figure is None:
        raise RuntimeError("PlotPanel grid returned no Figure authority")
    payload = encode_raster_buffer_png(overview.raster)
    regions = overview.regions
    _check(check_cancelled)
    if len(regions) != panel_count:
        raise RuntimeError("typed grid regions do not cover every canonical panel")
    bundle = EncodedRasterDocument(
        data_figure_summary(visible_figure),
        (
            EncodedRasterPage(
                f"{intent.value.lower()}-overview",
                "Overview",
                payload,
            ),
        ),
    )
    return DataFigureGridOverview(
        intent=intent,
        figure=visible_figure,
        bundle=bundle,
        regions=regions,
        histogram_home_x_limits=histogram_home,
        display_state=display_state,
    )

def render_data_figure_front(
    figure: DataFigure,
    state: DataFigureDisplayState,
    *,
    contract: PlotPanelContract,
    sequence: int,
    histogram_projection_value_range: tuple[float, float] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    _session: PlotPanelSession | None = None,
) -> PlotPanelComposeResult:
    intent, unavailable_reason = classify_single_data_figure(figure)
    if intent is None or intent is not panel_display_state_intent(state):
        raise ValueError(
            "typed render requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    _check(check_cancelled)
    if not isinstance(contract, PlotPanelContract):
        raise TypeError("typed render requires PlotPanelContract")
    if contract.figure.faceted or contract.figure.view_intent is not intent:
        raise ValueError("typed render contract has another Figure intent")
    if figure.has_fit_overlays:
        raise ValueError("base Figure render cannot contain Fit overlays")
    if histogram_projection_value_range is not None:
        if intent is not ViewIntent.HISTOGRAM:
            raise ValueError("only HISTOGRAM render can fix its projection value range")
        histogram_projection_value_range = validated_display_range(
            histogram_projection_value_range,
            "histogram projection value range",
        )
    _check(check_cancelled)
    layer = figure.document.layers[0]
    if contract.figure.view != layer.view:
        raise ValueError("typed render contract differs from its Figure view")
    snapshot = figure.datasets.resolve(layer.dataset_id)
    session, owns_session = _plot_session(contract, _session)
    try:
        result = session.compose_data_figure(
            figure,
            PlotPanelComposeRequest(
                FigureSource(snapshot),
                state,
                PanelProvenance(
                    f"figure:{snapshot.ref.block_id.value}",
                    snapshot.ref.stream_generation.value,
                    data_figure_join_digest(
                        figure,
                        intent,
                        None,
                    ),
                ),
                histogram_projection_value_range=(
                    histogram_projection_value_range
                ),
                check_cancelled=check_cancelled,
            )
        )
    finally:
        if owns_session:
            session.close()
    _check(check_cancelled)
    if result.frame is None or result.figure is None:
        raise RuntimeError("single-panel PlotPanel returned a faceted front")
    raw_frame = result.frame
    frame = BoardFrame(
        raw_frame.board_id,
        raw_frame.layout_generation,
        sequence,
        raw_frame.panels,
    )
    payload = frame.panels[0].display_payload
    validate_rendered_data_figure_payload(payload, state, None)
    return PlotPanelComposeResult(frame, None, result.figure)

def encode_data_figure_front_png(
    frame: BoardFrame,
    state: DataFigureDisplayState,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    """Encode one already-admitted Figure front using the canonical renderer."""

    _check(check_cancelled)
    if not isinstance(frame, BoardFrame) or len(frame.panels) != 1:
        raise TypeError("typed export requires one exact BoardFrame")
    panel = frame.panels[0]
    payload = panel.display_payload
    if (
        panel.panel_id != DATA_FIGURE_PANEL_ID
        or data_figure_payload_intent(payload) is not panel_display_state_intent(state)
    ):
        raise ValueError("typed export frame has another presentation")
    # Export the already-admitted Plot Panel surface.  Re-composing IMAGE here
    # used to create a second layout/style owner with unrelated dimensions;
    # every plot kind now saves the exact immutable pixels the user inspected.
    encoded = encode_raster_buffer_png(panel.raster)
    _check(check_cancelled)
    return encoded


__all__ = [
    "encode_data_figure_front_png",
    "render_data_figure_front",
    "render_data_figure_grid_overview",
    "render_encoded_data_figure",
]
