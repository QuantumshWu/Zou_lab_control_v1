"""Canonical DataFigure raster composition and immutable front construction."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

from zlc_data import FitResultBatch

from .data_figure import DataFigure
from .render import (
    BoardFrame,
    CoherenceStamp,
    CurveFitOverlay,
    CurvePanelPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    RadialGaussianImageFitOverlay,
    SourceIdentity,
)
from .curve_display import CurveDisplayState
from .display_range import validated_display_range
from .data_figure_presentation import (
    DATA_FIGURE_BOARD_ID,
    DATA_FIGURE_JOIN_SCHEMA_DIGEST,
    DATA_FIGURE_PANEL_ID,
    DEFAULT_DATA_FIGURE_RASTER_SIZE,
    DataFigureDisplayState,
    DataFigureFront,
    DataFigureGridDisplayState,
    DataFigureGridOverview,
    DataFigurePanelPayload,
    classify_faceted_data_figure,
    classify_single_data_figure,
    data_figure_front_contract,
    data_figure_join_digest,
    data_figure_payload_intent,
    data_figure_summary,
    default_data_figure_display_state,
    display_state_intent,
    grid_display_state_intent,
    validate_rendered_data_figure_payload,
)
from .encoded_raster import EncodedRasterDocument, EncodedRasterPage
from .fit_curve_projection import (
    CurveFitOverlayPlan,
    materialize_curve_fit_overlay_plan,
)
from .fit_editor import fit_projection_metadata
from .figure import (
    EvaluatedImage,
    EvaluatedProjectionIdentity,
    ViewIntent,
)
from .histogram_display import (
    HistogramCountScale,
    HistogramDisplayState,
    histogram_projection_home_x_limits,
)
from .image_display import (
    ImageDisplayState,
    evaluated_image_data_range,
    image_viewport_for_display_state,
    resolve_image_color_limits_from_range,
)
from .image_view import image_viewport_for_evaluated_image
from .meter_display import MeterDisplayState
from .plot_layout import LIVE_PANEL_DPI


class DataFigureRenderSession:
    """Persistent Agg owner for one interactive DataFigure surface.

    Display revisions update the existing artist tree.  Only renderer geometry,
    authored labels, or the Figure document identity replace that tree.  The
    stateless public render function remains a one-shot export convenience.
    """

    def __init__(self) -> None:
        self._image_key: tuple[object, ...] | None = None
        self._image_renderer = None
        self._single_key: tuple[object, ...] | None = None
        self._single_renderer = None
        self._image_range_source = None
        self._image_range: tuple[float, float] | None = None

    def _image(self, *, width, height, dpi, size_name):
        key = (int(width), int(height), float(dpi), size_name)
        if self._image_renderer is None or self._image_key != key:
            self._close_image()
            from .matplotlib_render import ImagePanelAggRenderer

            self._image_renderer = ImagePanelAggRenderer(
                width=width,
                height=height,
                dpi=dpi,
                size_name=size_name,
            )
            self._image_key = key
        return self._image_renderer

    def _single(
        self,
        figure: DataFigure,
        *,
        width,
        height,
        dpi,
        size_name,
        title,
        value_label,
    ):
        document = figure.document
        key = (
            document.document_id,
            document.revision,
            int(width),
            int(height),
            float(dpi),
            size_name,
            title,
            value_label,
        )
        if self._single_renderer is None or self._single_key != key:
            self._close_single()
            from .matplotlib_render import SinglePanelAggRenderer

            self._single_renderer = SinglePanelAggRenderer(
                document,
                width=width,
                height=height,
                dpi=dpi,
                size_name=size_name,
                title=title,
                value_label=value_label,
            )
            self._single_key = key
        return self._single_renderer

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

    def _image_limits(
        self,
        image: EvaluatedImage,
        state: ImageDisplayState,
        *,
        current_color_limits,
        previous_relim_mode,
    ):
        if self._image_range_source is not image:
            self._image_range_source = image
            self._image_range = evaluated_image_data_range((image,))
        return resolve_image_color_limits_from_range(
            self._image_range,
            state,
            current_color_limits=current_color_limits,
            previous_relim_mode=previous_relim_mode,
        )

    def _close_image(self) -> None:
        renderer, self._image_renderer = self._image_renderer, None
        self._image_key = None
        if renderer is not None:
            renderer.close()

    def _close_single(self) -> None:
        renderer, self._single_renderer = self._single_renderer, None
        self._single_key = None
        if renderer is not None:
            renderer.close()

    def close(self) -> None:
        self._close_image()
        self._close_single()
        self._image_range_source = None
        self._image_range = None


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
    raster_size: tuple[int, int] = DEFAULT_DATA_FIGURE_RASTER_SIZE,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    display_state: DataFigureGridDisplayState | None = None,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
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
    if size_name is None:
        if (
            presentation_title is not None
            or presentation_value_label is not None
        ):
            raise ValueError(
                "authored grid labels require named panel geometry"
            )
        if display_state != default_data_figure_display_state(intent):
            raise ValueError(
                "an authored grid display requires named panel geometry"
            )
        payload, regions = figure.to_png_bytes_with_panel_regions()
        visible_display_state = None
    else:
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
        payload, regions = figure.to_panel_png_bytes_with_panel_regions(
            size=size_name,
            width=raster_size[0],
            height=raster_size[1],
            dpi=LIVE_PANEL_DPI * pixel_ratio,
            display_state=display_state,
            title=title,
            value_label=value_label,
        )
        visible_display_state = display_state
    _check(check_cancelled)
    if len(regions) != panel_count:
        raise RuntimeError("typed grid regions do not cover every canonical panel")
    bundle = EncodedRasterDocument(
        data_figure_summary(figure),
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
        figure=figure,
        bundle=bundle,
        regions=regions,
        histogram_home_x_limits=histogram_home,
        display_state=visible_display_state,
    )

def render_data_figure_front(
    figure: DataFigure,
    state: DataFigureDisplayState,
    *,
    current_value_limits: tuple[float, float] | None,
    previous_relim_mode,
    previous_count_scale: HistogramCountScale | None,
    sequence: int,
    fit_result: FitResultBatch | None = None,
    fit_result_identity: str | None = None,
    histogram_projection_value_range: tuple[float, float] | None = None,
    release_initial_canonical_on_commit: bool = False,
    raster_size: tuple[int, int] = DEFAULT_DATA_FIGURE_RASTER_SIZE,
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
    curve_fit_overlay_plan: CurveFitOverlayPlan | None = None
    image_fit_overlay: RadialGaussianImageFitOverlay | None = None
    if figure.has_fit_overlays:
        if intent is ViewIntent.CURVE:
            curve_fit_overlay_plan = figure.single_panel_curve_fit_overlay_plan(
                result_identity=fit_result_identity,
            )
        else:
            image_fit_overlay = figure.single_panel_radial_fit_overlay(
                result_identity=fit_result_identity,
            )
    elif fit_result is not None:
        if intent is ViewIntent.CURVE:
            curve_fit_overlay_plan = (
                figure.transient_single_panel_curve_fit_overlay_plan(
                    fit_result,
                    result_identity=fit_result_identity,
                )
            )
        else:
            image_fit_overlay = figure.transient_single_panel_radial_fit_overlay(
                fit_result,
                result_identity=fit_result_identity,
                check_cancelled=check_cancelled,
            )
    curve_fit_overlays: tuple[CurveFitOverlay, ...] = (
        ()
        if curve_fit_overlay_plan is None
        else materialize_curve_fit_overlay_plan(
            curve_fit_overlay_plan,
            check_cancelled=check_cancelled,
        )
    )
    _check(check_cancelled)

    if isinstance(state, ImageDisplayState):
        evaluated_input = figure.evaluated.inputs[0]
        image = figure.evaluated.layers[0].cells[0].series[0].data
        assert isinstance(image, EvaluatedImage)
        home_viewport = image_viewport_for_evaluated_image(image)
        viewport = image_viewport_for_display_state(state, home_viewport)
        session = DataFigureRenderSession() if _session is None else _session
        data_range, color_limits = session._image_limits(
            image,
            state,
            current_color_limits=current_value_limits,
            previous_relim_mode=previous_relim_mode,
        )
        width, height = raster_size
        renderer = session._image(
            width=width,
            height=height,
            dpi=LIVE_PANEL_DPI * pixel_ratio,
            size_name=size_name,
        )
        try:
            raster, raster_geometry = renderer.render(
                image,
                viewport,
                state,
                color_limits=color_limits,
                data_range=data_range,
                title=title,
                value_label=value_label,
                projection_identity=EvaluatedProjectionIdentity(
                    figure.evaluated.document_id,
                    figure.evaluated.document_revision,
                    evaluated_input,
                    figure.evaluated.layers[0].layer_id,
                    figure.evaluated.layers[0].resolutions,
                    figure.evaluated.layers[0].cells[0].facet_address,
                    figure.evaluated.layers[0].cells[0].series[0].batch_address,
                    image,
                ),
                fit_overlay=image_fit_overlay,
            )
        finally:
            if _session is None:
                session.close()
        payload: DataFigurePanelPayload = ImagePanelPayload(
            image,
            evaluated_input,
            viewport,
            data_range,
            state.colormap,
            color_limits,
            raster_geometry,
            image_fit_overlay,
        )
    else:
        width, height = raster_size
        session = DataFigureRenderSession() if _session is None else _session
        renderer = session._single(
            figure,
            width=width,
            height=height,
            dpi=LIVE_PANEL_DPI * pixel_ratio,
            size_name=size_name,
            title=title,
            value_label=value_label,
        )
        try:
            if isinstance(state, CurveDisplayState):
                raster, payload = renderer.render_interactive_curve(
                    figure.evaluated,
                    state,
                    current_y_limits=current_value_limits,
                    previous_relim_mode=previous_relim_mode,
                    fit_overlays=curve_fit_overlays,
                )
            elif isinstance(state, HistogramDisplayState):
                histogram_options = {}
                if histogram_projection_value_range is not None:
                    histogram_options["projection_value_range"] = (
                        histogram_projection_value_range
                    )
                raster, payload = renderer.render_interactive_histogram(
                    figure.evaluated,
                    state,
                    current_count_limits=current_value_limits,
                    previous_relim_mode=previous_relim_mode,
                    previous_count_scale=previous_count_scale,
                    **histogram_options,
                )
            else:
                assert isinstance(state, MeterDisplayState)
                raster, payload = renderer.render_meter(
                    figure.evaluated,
                    display_revision=state.revision,
                )
        finally:
            if _session is None:
                session.close()
    _check(check_cancelled)

    evaluated_input = payload.evaluated_input
    presentation = PanelPresentationIdentity(
        DATA_FIGURE_PANEL_ID,
        figure.document.document_id,
        figure.document.revision,
        0,
        state.revision,
    )
    ref = evaluated_input.ref
    stamp = CoherenceStamp(
        f"figure:{ref.block_id.value}",
        ref.stream_generation.value,
        "FrozenTypedFigureJoin",
        DATA_FIGURE_JOIN_SCHEMA_DIGEST,
        data_figure_join_digest(figure, intent, fit_result_identity),
        (evaluated_input,),
        (presentation,),
    )
    source = SourceIdentity(
        evaluated_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )
    frame = BoardFrame(
        DATA_FIGURE_BOARD_ID,
        0,
        sequence,
        (
            PanelFrame(
                DATA_FIGURE_PANEL_ID,
                f"frozen-{intent.value.lower()}",
                source,
                stamp,
                raster,
                payload,
            ),
        ),
    )
    validate_rendered_data_figure_payload(payload, state, fit_result_identity)
    fit_axis_ids, axis_roles = fit_projection_metadata(figure, intent)
    data_contract = data_figure_front_contract(
        intent,
        frame,
    )
    visible_figure = (
        figure
        if fit_result is None
        else figure.with_fit_results(
            {figure.document.layers[0].layer_id: fit_result}
        )
    )
    return DataFigureFront(
        intent=intent,
        figure=visible_figure,
        state=state,
        summary=data_figure_summary(figure),
        frame=frame,
        data_contract=data_contract,
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        fit_result_identity=fit_result_identity,
        transient_fit_result_owner=fit_result,
        release_initial_canonical_on_commit=(
            release_initial_canonical_on_commit
        ),
        raster_size=raster_size,
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
    if isinstance(payload, ImagePanelPayload):
        from .matplotlib_render import encode_image_panel_png

        encoded = encode_image_panel_png(payload, state)
        _check(check_cancelled)
        return encoded
    if not isinstance(
        payload,
        (CurvePanelPayload, HistogramPanelPayload, MeterPanelPayload),
    ):
        raise ValueError("typed export payload is unsupported")
    raster = panel.raster
    from PIL import Image

    image = Image.frombytes(
        "RGBA",
        (raster.width, raster.height),
        raster.pixels,
    )
    stream = BytesIO()
    try:
        image.save(stream, format="PNG")
        encoded = stream.getvalue()
    finally:
        image.close()
        stream.close()
    _check(check_cancelled)
    return encoded


__all__ = [
    "DataFigureRenderSession",
    "encode_data_figure_front_png",
    "render_data_figure_front",
    "render_data_figure_grid_overview",
    "render_encoded_data_figure",
]
