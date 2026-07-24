"""Saved-fit grid load, reraster, and export worker jobs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

from zlc_data import Selection
from zlc_frontend import (
    BoardFrame,
    CoherenceStamp,
    DataFigure,
    FitGridCellSummary,
    FitGridModel,
    FitGridPage,
    ImageDisplayState,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    RadialGaussianImageFitPanel,
    SourceIdentity,
)
from zlc_frontend.display_range import RelimMode, deadband_display_range
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_frontend.image_display import (
    image_display_for_viewport,
    image_viewport_for_display_state,
    resolve_image_color_limits,
)
from zlc_frontend.plot_layout import panel_display_size
from zlc_neutral_atom.artifacts.fit_reference import FitResultArtifactRef
from zlc_workbench.window_runtime import stage_and_replace_export

from .projection import (
    _BOARD_ID,
    _FIT_GRID_JOIN_SCHEMA_DIGEST,
    _fit_grid_join_identity,
    _fit_panel_id,
    _grid_columns,
    _require_not_cancelled,
    _shared_color_limits,
    _validated_regions,
)

def _build_image_grid_frame(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    *,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    sequence: int,
    cancelled: threading.Event,
) -> tuple[BoardFrame, tuple[float, float]]:
    panels = tuple(panels)
    if not panels or len(panels) > 36:
        raise ValueError("saved-fit IMAGE view requires between 1 and 36 panels")
    if len({_fit_panel_id(panel) for panel in panels}) != len(panels):
        raise ValueError("saved-fit logical panels do not have unique identities")
    _require_not_cancelled(cancelled)
    color_limits = _shared_color_limits(
        panels,
        display,
        current_color_limits,
        previous_relim_mode,
        cancelled,
    )
    resolved_display = replace(
        display,
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=color_limits,
    )
    panel_ids = tuple(_fit_panel_id(panel) for panel in panels)
    presentations = tuple(
        PanelPresentationIdentity(
            panel_id,
            f"saved-fit:{panel.fit_overlay.artifact_identity}",
            0,
            sequence,
            display.revision,
        )
        for panel_id, panel in zip(panel_ids, panels, strict=True)
    )
    artifact_identity, inputs, join_key_digest = _fit_grid_join_identity(
        panels,
        panel_ids,
    )
    stamp = CoherenceStamp(
        artifact_identity,
        artifact_identity,
        "SavedFitGridJoin",
        _FIT_GRID_JOIN_SCHEMA_DIGEST,
        join_key_digest,
        inputs,
        presentations,
    )
    from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

    width, height = panel_display_size("2x2")
    renderer = ImagePanelAggRenderer(width=width, height=height)
    frames = []
    try:
        for panel_id, panel in zip(panel_ids, panels, strict=True):
            _require_not_cancelled(cancelled)
            viewport = image_viewport_for_display_state(display, panel.home_viewport)
            data_range, effective_limits = resolve_image_color_limits(
                panel.image,
                resolved_display,
                current_color_limits=color_limits,
                previous_relim_mode=RelimMode.FIXED,
            )
            if effective_limits != color_limits:
                raise RuntimeError("saved-fit cell escaped the shared color scale")
            raster, raster_geometry = renderer.render(
                panel.image,
                viewport,
                resolved_display,
                color_limits=color_limits,
                data_range=data_range,
                title="",
                value_label="Signal",
                fit_overlay=panel.fit_overlay,
            )
            payload = ImagePanelPayload(
                panel.image,
                panel.evaluated_input,
                viewport,
                data_range,
                display.colormap,
                color_limits,
                raster_geometry,
                panel.fit_overlay,
            )
            ref = panel.evaluated_input.ref
            source = SourceIdentity(
                panel.evaluated_input.dataset_id,
                ref.block_id,
                ref.stream_generation,
                ref.schema_fingerprint,
            )
            frames.append(
                PanelFrame(
                    panel_id,
                    "saved-fit-grid",
                    source,
                    stamp,
                    raster,
                    payload,
                )
            )
    finally:
        renderer.close()
    _require_not_cancelled(cancelled)
    return (
        BoardFrame(
            _BOARD_ID,
            layout_generation,
            sequence,
            tuple(frames),
        ),
        color_limits,
    )

def _reframe_existing_image_panels(
    frame: BoardFrame,
    panels: tuple[RadialGaussianImageFitPanel, ...],
    *,
    layout_generation: int,
    sequence: int,
) -> BoardFrame:
    """Atomically relayout immutable panels without copying samples or rasters."""

    if not isinstance(frame, BoardFrame):
        raise TypeError("saved-fit relayout source must be BoardFrame")
    panels = tuple(panels)
    requested = tuple(_fit_panel_id(panel) for panel in panels)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("saved-fit relayout requires unique panel ids")
    by_id = {panel.panel_id: panel for panel in frame.panels}
    try:
        selected = tuple(by_id[panel_id] for panel_id in requested)
    except KeyError as error:
        raise ValueError("saved-fit relayout names an absent panel") from error
    old_presentations = {
        item.panel_id: item
        for item in frame.panels[0].coherence_stamp.presentations
    }
    presentations = tuple(
        PanelPresentationIdentity(
            panel_id,
            old_presentations[panel_id].document_id,
            old_presentations[panel_id].document_revision,
            sequence,
            old_presentations[panel_id].panel_revision,
        )
        for panel_id in requested
    )
    old_stamp = frame.panels[0].coherence_stamp
    artifact_identity, inputs, join_key_digest = _fit_grid_join_identity(
        panels,
        requested,
    )
    if (
        old_stamp.run_id != artifact_identity
        or old_stamp.provenance_epoch_id != artifact_identity
        or old_stamp.join_key_type != "SavedFitGridJoin"
        or old_stamp.join_key_schema_fingerprint != _FIT_GRID_JOIN_SCHEMA_DIGEST
    ):
        raise ValueError("saved-fit relayout source has another join identity")
    for projected, selected_panel in zip(panels, selected, strict=True):
        payload = selected_panel.display_payload
        if (
            not isinstance(payload, ImagePanelPayload)
            or payload.image is not projected.image
            or payload.evaluated_input != projected.evaluated_input
            or payload.fit_overlay != projected.fit_overlay
        ):
            raise ValueError("saved-fit relayout projection differs from its front")
    stamp = CoherenceStamp(
        old_stamp.run_id,
        old_stamp.provenance_epoch_id,
        old_stamp.join_key_type,
        old_stamp.join_key_schema_fingerprint,
        join_key_digest,
        inputs,
        presentations,
    )
    return BoardFrame(
        frame.board_id,
        layout_generation,
        sequence,
        tuple(
            PanelFrame(
                panel.panel_id,
                panel.coherence_group,
                panel.source_identity,
                stamp,
                panel.raster,
                panel.display_payload,
            )
            for panel in selected
        ),
    )

def _rerasterize_grid_view(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    revision: int,
    cancelled: threading.Event,
):
    _require_not_cancelled(cancelled)
    frame, limits = _build_image_grid_frame(
        panels,
        display,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
        layout_generation=layout_generation,
        sequence=revision,
        cancelled=cancelled,
    )
    return revision, panels, display, frame, limits

def _load_grid_view(
    view_loader,
    reference: FitResultArtifactRef,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    revision: int,
    return_model: bool,
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    cancelled: threading.Event,
):
    _require_not_cancelled(cancelled)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 4:
        raise TypeError(
            "saved-fit loader must return figure/model/page/cell summary"
        )
    figure, model, page, cell_summary = loaded
    if not isinstance(figure, DataFigure):
        raise TypeError("saved-fit loader must return DataFigure")
    if not isinstance(model, FitGridModel):
        raise TypeError("saved-fit loader must return FitGridModel")
    if model.artifact_identity != reference.target_ref:
        raise ValueError("saved-fit loader names a different artifact")
    if cell_selection is None:
        if not isinstance(page, FitGridPage) or cell_summary is not None:
            raise TypeError("saved-fit page load returned invalid compact metadata")
        if page != model.page(page_address):
            raise ValueError("saved-fit loader returned another logical page")
        expected_page_selections = model.page_logical_selections(page)
        resolved_selection = page.selection
    else:
        if page is not None or not isinstance(cell_summary, FitGridCellSummary):
            raise TypeError("saved-fit cell load returned invalid compact metadata")
        if cell_summary.selection != cell_selection:
            raise ValueError("saved-fit cell summary belongs to another selection")
        model.resolve_selection(cell_selection)
        expected_page_selections = None
        resolved_selection = cell_selection
    summary = (
        f"{model.summary} · {page.label}"
        if page is not None
        else model.summary
    )
    if model.model_id == "radial_gaussian_center":
        layers = tuple(figure.document.layers)
        if len(layers) != 1:
            raise ValueError(
                "saved-fit typed IMAGE explorer requires exactly one layer"
            )
        panels = figure.radial_gaussian_image_fit_panels(
            layers[0].layer_id,
            artifact_identity=reference.target_ref,
        )
        if len(panels) > 36:
            raise ValueError("saved-fit grid page exceeded 36 logical panels")
        if expected_page_selections is not None and tuple(
            panel.selection for panel in panels
        ) != expected_page_selections:
            raise ValueError(
                "saved-fit typed panels omitted, reordered, or substituted a "
                "logical page cell"
            )
        for panel in panels:
            expected = model.storage_index_or_none(panel.selection)
            if panel.fit_storage_index != expected:
                raise ValueError(
                    "saved-fit typed panel disagrees with the exact batch layout"
                )
        if cell_selection is not None:
            if len(panels) != 1:
                raise ValueError("exact saved-fit focus must project one IMAGE panel")
            if panels[0].selection != cell_selection:
                raise ValueError("focused typed panel belongs to another selection")
            if panels[0].fit_storage_index != cell_summary.storage_index:
                raise ValueError("focused typed panel and stored cell summary disagree")
            if panels[0].summary != cell_summary.text:
                raise ValueError(
                    "focused typed panel summary diverged from grid metadata"
                )
        # The typed panels retain their exact arrays and overlay facts.  Drop
        # the broader DataFigure/document before allocating the raster front.
        del loaded, figure, layers
        frame, color_limits = _build_image_grid_frame(
            panels,
            display,
            current_color_limits=current_color_limits,
            previous_relim_mode=previous_relim_mode,
            layout_generation=layout_generation,
            sequence=revision,
            cancelled=cancelled,
        )
        projection = (
            "typed-image",
            panels,
            frame,
            color_limits,
        )
    else:
        # The five public 1D fit models remain real saved-fit consumers.  Keep
        # their exact generic GridPlot path until the CURVE cell family is
        # migrated to the typed board; never make the IMAGE slice a blackout.
        payload, regions = figure.to_png_bytes_with_panel_regions()
        bundle = EncodedRasterDocument(
            summary,
            (EncodedRasterPage("figure", "Fit grid", payload),),
        )
        prepared_regions = _validated_regions(model, regions)
        if expected_page_selections is not None and tuple(
            region.fit_selection for region in prepared_regions
        ) != expected_page_selections:
            raise ValueError(
                "saved-fit generic panels omitted, reordered, or substituted a "
                "logical page cell"
            )
        if cell_selection is not None:
            if len(prepared_regions) != 1:
                raise ValueError("exact saved-fit focus must render one panel")
            if prepared_regions[0].fit_storage_index != cell_summary.storage_index:
                raise ValueError(
                    "focused panel and stored cell summary disagree"
                )
        projection = ("encoded", bundle, prepared_regions)
        del loaded, figure
    _require_not_cancelled(cancelled)
    model_identity = model.identity
    returned_model = model if return_model else None
    if not return_model:
        del model
    return (
        revision,
        returned_model,
        model_identity,
        page,
        cell_summary,
        resolved_selection,
        summary,
        projection,
    )

def _export_grid_view(
    view_loader,
    reference: FitResultArtifactRef,
    expected_model_identity: object,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
):
    _require_not_cancelled(cancelled)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 4:
        raise TypeError("saved-fit export loader returned invalid values")
    figure, model, page, cell_summary = loaded
    if not isinstance(figure, DataFigure) or not isinstance(model, FitGridModel):
        raise TypeError("saved-fit export loader returned invalid values")
    if model.artifact_identity != reference.target_ref:
        raise ValueError("saved-fit export loader names another artifact")
    if model.identity != expected_model_identity:
        raise ValueError("saved-fit export metadata changed within one session")
    if cell_selection is None:
        if not isinstance(page, FitGridPage) or cell_summary is not None:
            raise TypeError("saved-fit page export metadata is invalid")
        if page != model.page(page_address):
            raise ValueError("saved-fit export substituted another logical page")
    elif page is not None or not isinstance(cell_summary, FitGridCellSummary):
        raise TypeError("saved-fit cell export metadata is invalid")
    else:
        if cell_summary.selection != cell_selection:
            raise ValueError("saved-fit export substituted another logical cell")
        model.resolve_selection(cell_selection)
    _require_not_cancelled(cancelled)
    target = Path(destination)
    image_format = target.suffix.lstrip(".") or "png"
    if not target.suffix:
        target = target.with_suffix(f".{image_format}")

    def write_staged(temporary: Path) -> None:
        temporary.write_bytes(
            figure.to_bytes(
                image_format=image_format,
            )
        )

    committed = stage_and_replace_export(
        target,
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, committed

def _export_typed_grid_view(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    color_limits: tuple[float, float],
    columns: int,
    expected_join_key_digest: str,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
):
    """Export one frozen typed front without reloading data or fit authority."""

    _require_not_cancelled(cancelled)
    prepared = tuple(panels)
    panel_ids = tuple(_fit_panel_id(panel) for panel in prepared)
    _artifact_identity, _inputs, join_key_digest = _fit_grid_join_identity(
        prepared,
        panel_ids,
    )
    if join_key_digest != expected_join_key_digest:
        raise ValueError("saved-fit typed export differs from its painted join")
    if (
        isinstance(columns, bool)
        or not isinstance(columns, int)
        or not 1 <= columns <= len(prepared)
    ):
        raise ValueError("saved-fit typed export columns are invalid")
    target = Path(destination)
    image_format = target.suffix.lstrip(".").lower() or "png"
    if not target.suffix:
        target = target.with_suffix(f".{image_format}")

    def write_staged(temporary: Path) -> None:
        _require_not_cancelled(cancelled)
        from zlc_frontend.matplotlib_render import (
            encode_radial_gaussian_image_fit_panels,
        )

        temporary.write_bytes(
            encode_radial_gaussian_image_fit_panels(
                prepared,
                display,
                color_limits,
                image_format=image_format,
                columns=columns,
            )
        )
        _require_not_cancelled(cancelled)

    committed = stage_and_replace_export(
        target,
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, committed
