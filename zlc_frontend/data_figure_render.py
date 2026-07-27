"""Canonical DataFigure raster composition and immutable front construction."""

from __future__ import annotations

from collections.abc import Callable
from zlc_data import FitResultBatch

from .data_figure import DataFigure
from .render import BoardFrame
from .display_range import validated_display_range
from .data_figure_presentation import (
    DATA_FIGURE_PANEL_ID,
    DataFigureDisplayState,
    DataFigureFront,
    DataFigureGridDisplayState,
    DataFigureGridOverview,
    DataFigurePanelPayload,
    classify_faceted_data_figure,
    classify_single_data_figure,
    data_figure_front_contract,
    data_figure_initial_size_name,
    data_figure_join_digest,
    data_figure_payload_intent,
    data_figure_summary,
    default_data_figure_display_state,
    display_state_intent,
    grid_display_state_intent,
    validate_rendered_data_figure_payload,
)
from .encoded_raster import EncodedRasterDocument, EncodedRasterPage
from .encoded_raster import encode_raster_buffer_png
from .fit_editor import fit_projection_metadata
from .figure import ViewIntent
from .figure_source import FigureSource
from .histogram_display import histogram_projection_home_x_limits
from .panel_size import DEFAULT_PANEL_SIZE
from .plot_layout import panel_surface_geometry
from .plot_panel import (
    PlotPanelComposeRequest,
    PlotPanelContract,
    PlotPanelSession,
)
from .panel_render import PanelProvenance


class DataFigureRenderSession:
    """Figure adapter over the frontend's sole :class:`PlotPanelSession`.

    FigureViewer owns archive loading and Fit persistence, not another raster
    engine.  A supported single/faceted ``DataFigure`` is normalized to the
    exact Plot Panel contract and the resulting immutable front is handed back
    to the existing DataFigure workbench shell.  Renderer/style/Divider/DPR and
    display continuity therefore have one owner across live panels, Edit, and
    saved figures.
    """

    def __init__(self) -> None:
        self._key: tuple[object, ...] | None = None
        self._session: PlotPanelSession | None = None

    def plot_session(self, contract: PlotPanelContract) -> PlotPanelSession:
        if not isinstance(contract, PlotPanelContract):
            raise TypeError("DataFigure surface requires PlotPanelContract")
        key = contract.session_identity
        if self._session is None or self._key != key:
            self.close()
            self._session = PlotPanelSession(contract)
            self._key = key
        return self._session

    def render_front(self, figure: DataFigure, state, **options) -> DataFigureFront:
        try:
            return render_data_figure_front(
                figure,
                state,
                _session=self,
                **options,
            )
        except BaseException:
            # A partially-mutated Matplotlib artist graph is not reusable.
            self.close()
            raise

    def close(self) -> None:
        session, self._session = self._session, None
        self._key = None
        if session is not None:
            session.close()


def _check(check_cancelled: Callable[[], None] | None) -> None:
    if check_cancelled is not None:
        check_cancelled()


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

def _presentation_label(
    value: str | None,
    fallback: str,
    name: str,
) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text or None")
    return value

def render_data_figure_grid_overview(
    figure: DataFigure,
    *,
    raster_size: tuple[int, int] | None = None,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    display_state: DataFigureGridDisplayState | None = None,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
    fit_result_identity: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> DataFigureGridOverview:
    intent, panel_count, reason = classify_faceted_data_figure(figure)
    if intent is None or panel_count is None:
        raise ValueError(
            "typed grid overview requires a supported multi-cell figure"
            + ("" if reason is None else f": {reason}")
        )
    _check(check_cancelled)
    histogram_home = None
    if intent is ViewIntent.HISTOGRAM:
        histogram_home = histogram_projection_home_x_limits(
            tuple(
                series.data.samples
                for cell in figure.evaluated.layers[0].cells
                for series in cell.series
            )
        )
    _check(check_cancelled)
    if display_state is None:
        display_state = default_data_figure_display_state(intent)
    elif grid_display_state_intent(display_state) is not intent:
        raise ValueError("typed grid display state does not match the figure intent")
    if figure.has_fit_overlays != (fit_result_identity is not None):
        raise ValueError(
            "typed grid Fit replay requires one exact result identity"
        )
    title = _presentation_label(
        presentation_title,
        figure.document.layers[0].layer_id,
        "presentation_title",
    )
    value_label = _presentation_label(
        presentation_value_label,
        figure.document.datasets[0].label,
        "presentation_value_label",
    )
    surface_size_name = (
        data_figure_initial_size_name(figure)
        if size_name is None
        else str(size_name)
    )
    geometry = panel_surface_geometry(
        surface_size_name,
        pixel_ratio=pixel_ratio,
    )
    if raster_size is None:
        raster_size = geometry.raster_size
    if tuple(raster_size) != geometry.raster_size:
        raise ValueError(
            "DataFigure grid geometry must come from panel_surface_geometry"
        )
    overlay_result = None
    base_figure = figure
    if figure.has_fit_overlays:
        results = tuple(figure.fit_results.values())
        if len(results) != 1:
            raise ValueError("typed grid requires one exact Fit result")
        overlay_result = results[0]
        base_figure = figure.with_fit_results(None)
    layer = base_figure.document.layers[0]
    snapshot = base_figure.datasets.resolve(layer.dataset_id)
    contract = PlotPanelContract(
        DATA_FIGURE_PANEL_ID,
        "grid",
        title,
        value_label,
        size_name=geometry.size_name,
        pixel_ratio=geometry.pixel_ratio,
        view=layer.view,
    )
    session = PlotPanelSession(contract)
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
    sequence: int,
    fit_result: FitResultBatch | None = None,
    fit_result_identity: str | None = None,
    histogram_projection_value_range: tuple[float, float] | None = None,
    release_initial_canonical_on_commit: bool = False,
    raster_size: tuple[int, int] | None = None,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
    _session: DataFigureRenderSession | None = None,
) -> DataFigureFront:
    intent, unavailable_reason = classify_single_data_figure(figure)
    if intent is None or intent is not display_state_intent(state):
        raise ValueError(
            "typed render requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    _check(check_cancelled)
    if histogram_projection_value_range is not None:
        if intent is not ViewIntent.HISTOGRAM:
            raise ValueError("only HISTOGRAM render can fix its projection value range")
        histogram_projection_value_range = validated_display_range(
            histogram_projection_value_range,
            "histogram projection value range",
        )
    if not isinstance(release_initial_canonical_on_commit, bool):
        raise TypeError("release_initial_canonical_on_commit must be bool")

    if figure.has_fit_overlays:
        if fit_result is not None or fit_result_identity is None:
            raise ValueError(
                "canonical typed Fit replay requires one caller-supplied result identity"
            )
    elif (fit_result is None) != (fit_result_identity is None):
        raise ValueError("transient typed Fit result and identity must be present together")
    if intent is ViewIntent.METER and (
        figure.has_fit_overlays
        or fit_result is not None
        or fit_result_identity is not None
    ):
        raise ValueError("METER display cannot carry a Fit overlay")
    _check(check_cancelled)
    title = _presentation_label(
        presentation_title,
        figure.document.layers[0].layer_id,
        "presentation_title",
    )
    value_label = _presentation_label(
        presentation_value_label,
        figure.document.datasets[0].label,
        "presentation_value_label",
    )
    surface_size_name = DEFAULT_PANEL_SIZE if size_name is None else str(size_name)
    geometry = panel_surface_geometry(
        surface_size_name,
        pixel_ratio=pixel_ratio,
    )
    if raster_size is None:
        raster_size = geometry.raster_size
    if tuple(raster_size) != geometry.raster_size:
        raise ValueError(
            "DataFigure raster geometry must come from panel_surface_geometry"
        )

    overlay_result = fit_result
    base_figure = figure
    if figure.has_fit_overlays:
        results = tuple(figure.fit_results.values())
        if len(results) != 1:
            raise ValueError("single-panel Figure requires one exact Fit result")
        overlay_result = results[0]
        base_figure = figure.with_fit_results(None)

    layer = base_figure.document.layers[0]
    snapshot = base_figure.datasets.resolve(layer.dataset_id)
    kind = {
        ViewIntent.IMAGE: "2d",
        ViewIntent.CURVE: "1d",
        ViewIntent.HISTOGRAM: "hist",
        ViewIntent.METER: "meter",
    }[intent]
    contract = PlotPanelContract(
        DATA_FIGURE_PANEL_ID,
        kind,
        title,
        value_label,
        size_name=geometry.size_name,
        pixel_ratio=geometry.pixel_ratio,
        view=layer.view,
    )
    owner = DataFigureRenderSession() if _session is None else _session
    try:
        result = owner.plot_session(contract).compose_data_figure(
            base_figure,
            PlotPanelComposeRequest(
                FigureSource(snapshot),
                state,
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
                histogram_projection_value_range=(
                    histogram_projection_value_range
                ),
                check_cancelled=check_cancelled,
            )
        )
    finally:
        if _session is None:
            owner.close()
    _check(check_cancelled)
    if result.frame is None or result.figure is None:
        raise RuntimeError("single-panel PlotPanel returned a faceted front")
    visible_figure = (
        result.figure
        if overlay_result is None
        else result.figure.with_fit_results({layer.layer_id: overlay_result})
    )
    raw_frame = result.frame
    frame = BoardFrame(
        raw_frame.board_id,
        raw_frame.layout_generation,
        sequence,
        raw_frame.panels,
    )
    payload = frame.panels[0].display_payload
    validate_rendered_data_figure_payload(payload, state, fit_result_identity)
    fit_axis_ids, axis_roles = fit_projection_metadata(visible_figure, intent)
    data_contract = data_figure_front_contract(
        intent,
        frame,
    )
    return DataFigureFront(
        intent=intent,
        figure=visible_figure,
        state=state,
        summary=data_figure_summary(visible_figure),
        frame=frame,
        data_contract=data_contract,
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        fit_result_identity=fit_result_identity,
        transient_fit_result_owner=fit_result,
        release_initial_canonical_on_commit=(
            release_initial_canonical_on_commit
        ),
        raster_size=geometry.raster_size,
        surface_contract=contract,
    )

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
        or data_figure_payload_intent(payload) is not display_state_intent(state)
    ):
        raise ValueError("typed export frame has another presentation")
    # Export the already-admitted Plot Panel surface.  Re-composing IMAGE here
    # used to create a second layout/style owner with unrelated dimensions;
    # every plot kind now saves the exact immutable pixels the user inspected.
    encoded = encode_raster_buffer_png(panel.raster)
    _check(check_cancelled)
    return encoded


__all__ = [
    "DataFigureRenderSession",
    "encode_data_figure_front_png",
    "render_data_figure_front",
    "render_data_figure_grid_overview",
    "render_encoded_data_figure",
]
