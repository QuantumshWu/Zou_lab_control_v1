"""Capacity-one DataFigure fit, render, and export worker jobs."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor
import math
from pathlib import Path
import threading
import time

from zlc_data import FitBatchStatus, FitDeadlineExceeded, FitResultBatch, FitSpec
from zlc_frontend import (
    BoardFrame,
    CoherenceStamp,
    CurveFitOverlay,
    CurvePanelPayload,
    DataFigure,
    FitAuthoringOption,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterDisplayState,
    MeterPanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    RadialGaussianImageFitOverlay,
    SourceIdentity,
)
from zlc_frontend.curve_display import CurveDisplayState
from zlc_frontend.display_range import RelimMode, validated_display_range
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_frontend.fit_curve_projection import CurveFitOverlayPlan, materialize_curve_fit_overlay_plan
from zlc_frontend.figure import EvaluatedImage, ViewIntent
from zlc_frontend.histogram_display import (
    HistogramCountScale,
    HistogramDisplayState,
    histogram_projection_home_x_limits,
)
from zlc_frontend.image_display import ImageDisplayState, image_viewport_for_display_state
from zlc_frontend.image_display import resolve_image_color_limits
from zlc_frontend.image_view import image_viewport_for_evaluated_image
from zlc_frontend.plot_layout import LIVE_PANEL_DPI
from zlc_workbench.fit import FitDraftAuthority, FitDraftResult
from zlc_workbench.window_runtime import stage_and_replace_export

from .projection import (
    _NUMERIC_RASTER_SIZE,
    _TYPED_BOARD_ID,
    _TYPED_JOIN_SCHEMA_DIGEST,
    _TYPED_PANEL_ID,
    _GridDisplayState,
    _TypedDisplayState,
    _TypedFigureFront,
    _TypedGridOverview,
    _build_typed_front_contract,
    _classify_single_typed,
    _classify_typed_grid,
    _figure_summary,
    _fit_projection_metadata,
    _payload_intent,
    _grid_state_intent,
    _state_intent,
    _require_not_cancelled,
    _validate_fit_replay_options,
    _validate_rendered_authored_payload,
    _typed_join_digest,
)

_FIT_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-data-figure-fit",
)

def _encoded_figure(
    figure: DataFigure,
    cancelled: threading.Event | None,
    *,
    unavailable_reason: str | None = None,
) -> EncodedRasterDocument:
    _require_not_cancelled(cancelled)
    payload = figure.to_png_bytes()
    _require_not_cancelled(cancelled)
    summary = _figure_summary(figure)
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

def _render_typed_grid_overview(
    figure: DataFigure,
    cancelled: threading.Event,
    *,
    raster_size: tuple[int, int] = _NUMERIC_RASTER_SIZE,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    display_state: _GridDisplayState | None = None,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
) -> _TypedGridOverview:
    intent, panel_count, reason = _classify_typed_grid(figure)
    if intent is None or panel_count is None:
        raise ValueError(
            "typed grid overview requires a supported multi-cell figure"
            + ("" if reason is None else f": {reason}")
        )
    _require_not_cancelled(cancelled)
    histogram_home = None
    if intent is ViewIntent.HISTOGRAM:
        histogram_home = histogram_projection_home_x_limits(
            tuple(
                series.data.samples
                for cell in figure.evaluated.layers[0].cells
                for series in cell.series
            )
        )
    _require_not_cancelled(cancelled)
    if display_state is None:
        display_state = _default_typed_state(intent)
    elif _grid_state_intent(display_state) is not intent:
        raise ValueError("typed grid display state does not match the figure intent")
    if size_name is None:
        if (
            presentation_title is not None
            or presentation_value_label is not None
        ):
            raise ValueError(
                "authored grid labels require named panel geometry"
            )
        if display_state != _default_typed_state(intent):
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
    _require_not_cancelled(cancelled)
    if len(regions) != panel_count:
        raise RuntimeError("typed grid regions do not cover every canonical panel")
    bundle = EncodedRasterDocument(
        _figure_summary(figure),
        (
            EncodedRasterPage(
                f"{intent.value.lower()}-overview",
                "Overview",
                payload,
            ),
        ),
    )
    return _TypedGridOverview(
        intent=intent,
        figure=figure,
        bundle=bundle,
        regions=regions,
        histogram_home_x_limits=histogram_home,
        display_state=visible_display_state,
    )

def _fit_summary(
    draft: FitDraftResult,
    *,
    cancelled=None,
) -> str:
    result = draft.result
    counts = Counter(status.value for status in result.statuses)
    status_text = ", ".join(
        f"{name.lower()}={count}" for name, count in sorted(counts.items())
    )
    quality_min = math.inf
    quality_max = -math.inf
    for index, (status, rss, used) in enumerate(
        zip(
            result.statuses,
            result.residual_sum_squares,
            result.used_observation_counts,
            strict=True,
        )
    ):
        if cancelled is not None and index % 1024 == 0 and cancelled():
            raise CancelledError()
        if status is not FitBatchStatus.CONVERGED or int(used) <= 0:
            continue
        value = math.sqrt(float(rss) / int(used))
        if math.isfinite(value):
            quality_min = min(quality_min, value)
            quality_max = max(quality_max, value)
    quality_text = (
        "no converged RMSE"
        if not math.isfinite(quality_min)
        else f"RMSE {quality_min:.4g}–{quality_max:.4g}"
    )
    return (
        f"{result.spec.model_id} · {len(result.statuses)} named batch cell(s) · "
        f"{status_text} · {quality_text} · draft is not saved"
    )

def _prepare_fit_options(
    prepare,
    fit_axis_ids: tuple[AxisId, ...],
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...],
    selection: Selection | None,
    allow_prepared_transform: bool = False,
) -> tuple[FitAuthoringOption, ...]:
    options = tuple(
        prepare(
            fit_axis_ids,
            selection,
        )
    )
    if not options or any(
        not isinstance(option, FitAuthoringOption) for option in options
    ):
        raise ValueError("Fit preparation produced no FitAuthoringOption")
    schemas = {option.spec.input_schema_fingerprint for option in options}
    models = tuple(option.spec.model_id for option in options)
    if len(schemas) != 1 or len(models) != len(set(models)):
        raise ValueError("Fit options require one source schema and unique models")
    if any(option.spec.fit_axis_ids != fit_axis_ids for option in options):
        raise ValueError("Fit option axes differ from the exact displayed axes")
    return _validate_fit_replay_options(
        options,
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        selection=selection,
        allow_prepared_transform=allow_prepared_transform,
    )

def _execute_fit_draft(
    authority: FitDraftAuthority,
    spec: FitSpec,
    deadline_monotonic: float,
    window_cancelled: threading.Event,
    analysis_cancelled: threading.Event,
) -> tuple[FitDraftResult, str]:
    def cancelled() -> bool:
        return window_cancelled.is_set() or analysis_cancelled.is_set()

    if cancelled():
        raise CancelledError()
    if time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired while waiting for its worker lane")
    draft = authority.execute(spec, cancelled, deadline_monotonic)
    try:
        return (
            draft,
            _fit_summary(draft, cancelled=cancelled),
        )
    except BaseException:
        # ``authority.execute`` has already installed the one live draft.  Any
        # failure in worker-only presentation/accounting must release that exact
        # generation or all later Fit submissions deadlock behind a hidden draft.
        authority.discard(draft)
        raise

def _reload_fit_result(
    reload_result,
    handle: object,
) -> FitResultBatch:
    result = reload_result(handle)
    if not isinstance(result, FitResultBatch):
        raise TypeError("saved Fit reload returned another result type")
    return result

def _render_typed_front(
    figure: DataFigure,
    state: _TypedDisplayState,
    *,
    current_value_limits: tuple[float, float] | None,
    previous_relim_mode,
    previous_count_scale: HistogramCountScale | None,
    sequence: int,
    cancelled: threading.Event,
    fit_result: FitResultBatch | None = None,
    fit_result_identity: str | None = None,
    histogram_projection_value_range: tuple[float, float] | None = None,
    release_initial_canonical_on_commit: bool = False,
    raster_size: tuple[int, int] = _NUMERIC_RASTER_SIZE,
    size_name: str | None = None,
    pixel_ratio: float = 1.0,
    presentation_title: str | None = None,
    presentation_value_label: str | None = None,
) -> _TypedFigureFront:
    intent, unavailable_reason = _classify_single_typed(figure)
    if intent is None or intent is not _state_intent(state):
        raise ValueError(
            "typed render requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    _require_not_cancelled(cancelled)
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
    _require_not_cancelled(cancelled)
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
                check_cancelled=lambda: _require_not_cancelled(cancelled),
            )
    curve_fit_overlays: tuple[CurveFitOverlay, ...] = (
        ()
        if curve_fit_overlay_plan is None
        else materialize_curve_fit_overlay_plan(
            curve_fit_overlay_plan,
            check_cancelled=lambda: _require_not_cancelled(cancelled),
        )
    )
    _require_not_cancelled(cancelled)

    if isinstance(state, ImageDisplayState):
        evaluated_input = figure.evaluated.inputs[0]
        image = figure.evaluated.layers[0].cells[0].series[0].data
        assert isinstance(image, EvaluatedImage)
        home_viewport = image_viewport_for_evaluated_image(image)
        viewport = image_viewport_for_display_state(state, home_viewport)
        data_range, color_limits = resolve_image_color_limits(
            image,
            state,
            current_color_limits=current_value_limits,
            previous_relim_mode=previous_relim_mode,
        )
        from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

        width, height = raster_size
        renderer = ImagePanelAggRenderer(
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
                fit_overlay=image_fit_overlay,
            )
        finally:
            renderer.close()
        payload: _TypedPanelPayload = ImagePanelPayload(
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
        from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

        width, height = raster_size
        renderer = SinglePanelAggRenderer(
            figure.document,
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
            renderer.close()
    _require_not_cancelled(cancelled)

    evaluated_input = payload.evaluated_input
    presentation = PanelPresentationIdentity(
        _TYPED_PANEL_ID,
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
        _TYPED_JOIN_SCHEMA_DIGEST,
        _typed_join_digest(figure, intent, fit_result_identity),
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
        _TYPED_BOARD_ID,
        0,
        sequence,
        (
            PanelFrame(
                _TYPED_PANEL_ID,
                f"frozen-{intent.value.lower()}",
                source,
                stamp,
                raster,
                payload,
            ),
        ),
    )
    _validate_rendered_authored_payload(payload, state, fit_result_identity)
    fit_axis_ids, axis_roles = _fit_projection_metadata(figure, intent)
    data_contract = _build_typed_front_contract(
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
    return _TypedFigureFront(
        intent=intent,
        figure=visible_figure,
        state=state,
        summary=_figure_summary(figure),
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

def _export_typed_png(
    frame: BoardFrame,
    state: _TypedDisplayState,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    if not isinstance(frame, BoardFrame) or len(frame.panels) != 1:
        raise TypeError("typed export requires one exact BoardFrame")
    panel = frame.panels[0]
    payload = panel.display_payload
    if panel.panel_id != _TYPED_PANEL_ID or _payload_intent(payload) is not _state_intent(state):
        raise ValueError("typed export frame has another presentation")
    if isinstance(payload, ImagePanelPayload):
        def write_staged(path: Path) -> None:
            _require_not_cancelled(cancelled)
            from zlc_frontend.matplotlib_render import save_image_panel_png

            save_image_panel_png(
                payload,
                state,
                path,
            )
            _require_not_cancelled(cancelled)

        result = stage_and_replace_export(
            Path(destination),
            write_staged=write_staged,
            cancelled=cancelled,
            commit_lock=commit_lock,
        )
        return revision, result
    if not isinstance(
        payload,
        (CurvePanelPayload, HistogramPanelPayload, MeterPanelPayload),
    ):
        raise ValueError("typed export payload is unsupported")
    raster = panel.raster
    def write_staged(path: Path) -> None:
        from PIL import Image

        image = Image.frombytes(
            "RGBA",
            (raster.width, raster.height),
            raster.pixels,
        )
        try:
            image.save(path, format="PNG")
        finally:
            image.close()

    result = stage_and_replace_export(
        Path(destination),
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, result

def _export_encoded_png(
    payload: bytes,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    if not isinstance(payload, bytes):
        raise TypeError("encoded export requires owned immutable bytes")

    def write_staged(path: Path) -> None:
        _require_not_cancelled(cancelled)
        path.write_bytes(payload)
        _require_not_cancelled(cancelled)

    result = stage_and_replace_export(
        Path(destination),
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, result
