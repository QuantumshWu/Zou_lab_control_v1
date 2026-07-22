"""Exact saved-fit GridPlot explorer over immutable current artifacts."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace
import math
from pathlib import Path
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import (
    Selection,
    dataset_revision_ref_to_tree,
    selection_to_tree,
)
from zlc_frontend import (
    BoardFrame,
    CoherenceStamp,
    DataFigure,
    FigurePanelRegion,
    FitGridCellSummary,
    FitGridModel,
    FitGridPage,
    ImageDisplayState,
    ImagePanelPayload,
    ImageViewportTransform,
    PanelFrame,
    PanelPresentationIdentity,
    RadialGaussianImageFitPanel,
    SourceIdentity,
    radial_gaussian_image_fit_panel_retained_upper_bound_nbytes,
)
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_frontend.display_range import RelimMode, deadband_display_range
from zlc_frontend.image_display import (
    image_display_for_viewport,
    image_display_form_spec,
    image_display_form_values,
    image_display_from_form,
    image_viewport_for_display_state,
)
from zlc_frontend.image_raster import (
    evaluated_image_data_range,
    estimate_evaluated_image_retained_nbytes,
    estimate_indexed8_raster_peak_nbytes,
    rasterize_image_indexed8,
)
from zlc_frontend.qt_widgets import (
    AxisLayoutNavigator,
    FluentButton,
    FluentLabel,
    FluentPopup,
    FluentRevisionedFormEditor,
    FluentSwitch,
    GREY,
    ORANGE,
    FrozenRasterView,
    QtRasterBoard,
    RectangleGesture,
    runtime_range_placeholders,
    FluentSettingsPopupAnchor,
    signals_blocked,
    sync_revisioned_form_editors,
)
from zlc_frontend.render_style import indexed_colormap
from zlc_frontend.selector import (
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportCommit,
    PanelInteractionOrigin,
)
from zlc_neutral_atom.fit_reference import FitResultArtifactRef
from zlc_storage import canonical_digest, positive_integer

from ._frozen_raster import (
    FrozenRasterWindow,
)
from ._window_runtime import (
    cancel_export_commits,
    error_summary,
    open_workbench_window,
    stage_and_replace_export,
)


_DEFAULT_FIT_GRID_MEMORY_LIMIT_BYTES = 512 << 20
_BOARD_ID = "saved-fit-grid"
_FIT_GRID_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.SavedFitGridJoin",
        "fields": (
            "artifact_identity",
            "source_inputs",
            "logical_panel_selection",
            "fit_storage_index",
        ),
    }
)


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _fit_panel_id(panel: RadialGaussianImageFitPanel) -> str:
    if not isinstance(panel, RadialGaussianImageFitPanel):
        raise TypeError("saved-fit grid panel has the wrong type")
    selection = (
        None if panel.selection is None else selection_to_tree(panel.selection)
    )
    identity = canonical_digest(
        {
            "selection": selection,
            "fit_storage_index": panel.fit_storage_index,
        }
    )
    return f"fit-cell-{identity[:20]}"


def _grid_columns(count: int) -> int:
    count = positive_integer(count, "saved-fit grid panel count")
    return min(6, max(1, math.ceil(math.sqrt(count))))


def _validated_regions(
    model: FitGridModel,
    regions: tuple[FigurePanelRegion, ...],
) -> tuple[FigurePanelRegion, ...]:
    """Validate the still-consumed generic 1D saved-fit hit map."""

    prepared = tuple(regions)
    if not prepared or any(
        not isinstance(region, FigurePanelRegion) for region in prepared
    ):
        raise TypeError("saved-fit generic renderer must return panel regions")
    if len(prepared) > 36:
        raise ValueError("saved-fit grid page exceeded 36 display panels")
    if len({region.key for region in prepared}) != len(prepared):
        raise ValueError("saved-fit grid panel keys must be unique")
    for region in prepared:
        if region.fit_storage_index != model.storage_index_or_none(region.selection):
            raise ValueError(
                "saved-fit panel hit map disagrees with the exact batch layout"
            )
    return prepared


def _image_data_range(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    cancelled: threading.Event,
) -> tuple[float, float] | None:
    """Pool exact valid ranges without concatenating cell-sized arrays."""

    def images():
        for panel in panels:
            _require_not_cancelled(cancelled)
            yield panel.image

    return evaluated_image_data_range(images())


def _shared_color_limits(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    cancelled: threading.Event,
) -> tuple[float, float]:
    data_range = _image_data_range(panels, cancelled)
    if data_range is None:
        if display.relim_mode is RelimMode.FIXED:
            assert display.fixed_color_limits is not None
            return display.fixed_color_limits
        return (0.0, 1.0) if current_color_limits is None else current_color_limits
    return deadband_display_range(
        display.relim_mode,
        current_color_limits,
        data_range[0],
        data_range[1],
        fixed_range=display.fixed_color_limits,
        force=(
            previous_relim_mode is None
            or previous_relim_mode is not display.relim_mode
        ),
    )


def _fit_grid_join_identity(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    panel_ids: tuple[str, ...],
) -> tuple[str, tuple[object, ...], str]:
    """Freeze the exact logical cells represented by one board frame."""

    if not panels or len(panels) != len(panel_ids):
        raise ValueError("saved-fit join requires one id per logical panel")
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("saved-fit join panel ids must be unique")
    input_by_id = {}
    for panel in panels:
        incumbent = input_by_id.setdefault(
            panel.evaluated_input.dataset_id,
            panel.evaluated_input,
        )
        if incumbent != panel.evaluated_input:
            raise ValueError("saved-fit grid panel inputs disagree by dataset id")
    inputs = tuple(input_by_id.values())
    artifact_identities = {
        panel.fit_overlay.artifact_identity for panel in panels
    }
    if len(artifact_identities) != 1:
        raise ValueError("saved-fit grid panels belong to different artifacts")
    artifact_identity = next(iter(artifact_identities))
    digest = canonical_digest(
        {
            "schema": "zlc_frontend.SavedFitGridJoin",
            "artifact_identity": artifact_identity,
            "inputs": tuple(
                {
                    "dataset_id": item.dataset_id.value,
                    "ref": dataset_revision_ref_to_tree(item.ref),
                }
                for item in inputs
            ),
            "panels": tuple(
                {
                    "panel_id": panel_id,
                    "selection": (
                        None
                        if panel.selection is None
                        else selection_to_tree(panel.selection)
                    ),
                    "fit_storage_index": panel.fit_storage_index,
                }
                for panel_id, panel in zip(panel_ids, panels, strict=True)
            ),
        }
    )
    return artifact_identity, inputs, digest


def _fit_grid_memory_bounds(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    *,
    samples_prebudgeted: bool = False,
) -> tuple[int, int]:
    """Return retained GUI-front and sequential worker peak upper bounds."""

    if not isinstance(samples_prebudgeted, bool):
        raise TypeError("samples_prebudgeted must be bool")
    retained = 0
    incremental_retained = 0
    largest_scratch = 0
    for panel in panels:
        height, width = panel.image.values.shape
        pixels = height * width
        sample = estimate_evaluated_image_retained_nbytes(
            height,
            width,
            value_itemsize=panel.image.values.dtype.itemsize,
        )
        panel_metadata = (
            radial_gaussian_image_fit_panel_retained_upper_bound_nbytes(panel)
        )
        raster_retained = sample + 2 * pixels + (64 << 10)
        panel_retained = raster_retained + panel_metadata
        panel_incremental = 2 * pixels + (64 << 10) + panel_metadata
        panel_peak = estimate_indexed8_raster_peak_nbytes(
            height,
            width,
            value_itemsize=panel.image.values.dtype.itemsize,
            retained_fronts=1,
            retained_sample_fronts=1,
        )
        retained += panel_retained
        incremental_retained += panel_incremental
        largest_scratch = max(largest_scratch, panel_peak - raster_retained)
    admitted_retained = (
        incremental_retained if samples_prebudgeted else retained
    )
    return retained, admitted_retained + largest_scratch


def _build_image_grid_frame(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    *,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    sequence: int,
    memory_limit_bytes: int,
    cancelled: threading.Event,
    samples_prebudgeted: bool = False,
) -> tuple[BoardFrame, tuple[float, float], int]:
    panels = tuple(panels)
    if not panels or len(panels) > 36:
        raise ValueError("saved-fit IMAGE view requires between 1 and 36 panels")
    if len({_fit_panel_id(panel) for panel in panels}) != len(panels):
        raise ValueError("saved-fit logical panels do not have unique identities")
    retained_bound, worker_peak_bound = _fit_grid_memory_bounds(
        panels,
        samples_prebudgeted=samples_prebudgeted,
    )
    if worker_peak_bound > positive_integer(
        memory_limit_bytes,
        "saved-fit image worker memory limit",
    ):
        raise MemoryError("saved-fit typed IMAGE front exceeds worker budget")
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
    palette = indexed_colormap(display.colormap.value)
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
    frames = []
    for panel_id, panel in zip(panel_ids, panels, strict=True):
        _require_not_cancelled(cancelled)
        viewport = image_viewport_for_display_state(display, panel.home_viewport)
        raster, data_range, histogram, effective_limits = rasterize_image_indexed8(
            panel.image,
            resolved_display,
            current_color_limits=color_limits,
            previous_relim_mode=RelimMode.FIXED,
        )
        if effective_limits != color_limits:
            raise RuntimeError("saved-fit cell escaped the shared color scale")
        payload = ImagePanelPayload(
            panel.image,
            panel.evaluated_input,
            viewport,
            data_range,
            histogram,
            palette,
            color_limits,
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
    _require_not_cancelled(cancelled)
    return (
        BoardFrame(
            _BOARD_ID,
            layout_generation,
            sequence,
            tuple(frames),
        ),
        color_limits,
        retained_bound,
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
    memory_limit_bytes: int,
    cancelled: threading.Event,
):
    _require_not_cancelled(cancelled)
    frame, limits, retained = _build_image_grid_frame(
        panels,
        display,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
        layout_generation=layout_generation,
        sequence=revision,
        memory_limit_bytes=memory_limit_bytes,
        cancelled=cancelled,
        samples_prebudgeted=True,
    )
    return revision, panels, display, frame, limits, retained


def _load_grid_view(
    view_loader,
    reference: FitResultArtifactRef,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    memory_limit_bytes: int,
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
        memory_limit_bytes=memory_limit_bytes,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 5:
        raise TypeError(
            "saved-fit loader must return figure/model/page/cell summary/session budget"
        )
    figure, model, page, cell_summary, session_retained_bytes = loaded
    if not isinstance(figure, DataFigure):
        raise TypeError("saved-fit loader must return DataFigure")
    if not isinstance(model, FitGridModel):
        raise TypeError("saved-fit loader must return FitGridModel")
    if model.artifact_identity != reference.target_ref:
        raise ValueError("saved-fit loader names a different artifact")
    session_retained_bytes = positive_integer(
        session_retained_bytes,
        "saved-fit session retained bytes",
    )
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
        projection_peak = (
            figure.radial_gaussian_image_fit_panels_preflight_nbytes(
                layers[0].layer_id,
                artifact_identity=reference.target_ref,
            )
        )
        projection_required = figure.retained_upper_bound_nbytes + projection_peak
        if return_model:
            projection_required += (
                session_retained_bytes + model.retained_upper_bound_bytes
            )
        if projection_required > memory_limit_bytes:
            raise MemoryError(
                "saved-fit typed panel projection exceeds worker budget"
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
        # the broader DataFigure/document before allocating the raster front so
        # unrelated metadata cannot overlap outside the worker budget.
        del loaded, figure, layers
        frame_limit = memory_limit_bytes
        if return_model:
            frame_limit -= (
                session_retained_bytes + model.retained_upper_bound_bytes
            )
        if frame_limit <= 0:
            raise MemoryError(
                "saved-fit session leaves no typed IMAGE front budget"
            )
        frame, color_limits, frame_retained_bytes = _build_image_grid_frame(
            panels,
            display,
            current_color_limits=current_color_limits,
            previous_relim_mode=previous_relim_mode,
            layout_generation=layout_generation,
            sequence=revision,
            memory_limit_bytes=frame_limit,
            cancelled=cancelled,
        )
        projection = (
            "typed-image",
            panels,
            frame,
            color_limits,
            frame_retained_bytes,
        )
    else:
        # The five public 1D fit models remain real saved-fit consumers.  Keep
        # their exact generic GridPlot path until the CURVE cell family is
        # migrated to the typed board; never make the IMAGE slice a blackout.
        payload, regions = figure.to_png_bytes_with_panel_regions(
            memory_limit_bytes=figure.render_memory_limit_bytes,
        )
        bundle = EncodedRasterDocument(
            summary,
            (EncodedRasterPage("figure", "Fit grid", payload),),
        )
        prepared_regions = _validated_regions(model, regions)
        if expected_page_selections is not None and tuple(
            region.selection for region in prepared_regions
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
        required = bundle.source_front_peak_nbytes
        if return_model:
            required += session_retained_bytes + model.retained_upper_bound_bytes
        if required > memory_limit_bytes:
            raise MemoryError("saved-fit session and raster exceed worker budget")
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
        session_retained_bytes,
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
    memory_limit_bytes: int,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
):
    _require_not_cancelled(cancelled)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
        memory_limit_bytes=memory_limit_bytes,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 5:
        raise TypeError("saved-fit export loader returned invalid values")
    figure, model, page, cell_summary, _session_retained_bytes = loaded
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
        exported = figure.export(
            temporary,
            image_format=image_format,
            memory_limit_bytes=figure.render_memory_limit_bytes,
        )
        if Path(exported) != temporary:
            raise RuntimeError("saved-fit export changed its staged destination")

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
    memory_limit_bytes: int,
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
    memory_limit_bytes = positive_integer(
        memory_limit_bytes,
        "saved-fit typed export memory limit",
    )
    target = Path(destination)
    image_format = target.suffix.lstrip(".").lower() or "png"
    if not target.suffix:
        target = target.with_suffix(f".{image_format}")

    def write_staged(temporary: Path) -> None:
        _require_not_cancelled(cancelled)
        from zlc_frontend.matplotlib_render import (
            save_radial_gaussian_image_fit_panels,
        )

        save_radial_gaussian_image_fit_panels(
            prepared,
            display,
            color_limits,
            temporary,
            image_format=image_format,
            columns=columns,
            memory_limit_bytes=memory_limit_bytes,
        )
        _require_not_cancelled(cancelled)

    committed = stage_and_replace_export(
        target,
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, committed


class SavedFitGridWindow(FrozenRasterWindow):
    """Browse one exact saved ``FitResultBatch`` without ever re-solving it."""

    def __init__(
        self,
        view_loader,
        refit_opener,
        reference: FitResultArtifactRef,
        *,
        memory_limit_bytes: int,
    ) -> None:
        if not callable(view_loader):
            raise TypeError("saved-fit view_loader must be callable")
        if not callable(refit_opener):
            raise TypeError("saved-fit refit_opener must be callable")
        if not isinstance(reference, FitResultArtifactRef):
            raise TypeError("reference must be FitResultArtifactRef")
        self._view_loader = view_loader
        self._refit_opener = refit_opener
        self._reference = reference
        self._model: FitGridModel | None = None
        self._session_retained_bytes = 0
        self._navigator: AxisLayoutNavigator | None = None
        self._page: FitGridPage | None = None
        self._page_panels: tuple[RadialGaussianImageFitPanel, ...] = ()
        self._page_frame: BoardFrame | None = None
        self._page_retained_bytes = 0
        self._page_color_limits: tuple[float, float] | None = None
        self._page_encoded_bundle: EncodedRasterDocument | None = None
        self._page_regions: tuple[FigurePanelRegion, ...] = ()
        self._current_panels: tuple[RadialGaussianImageFitPanel, ...] = ()
        self._current_frame: BoardFrame | None = None
        self._current_retained_bytes = 0
        self._current_encoded_bundle: EncodedRasterDocument | None = None
        self._regions: tuple[FigurePanelRegion, ...] = ()
        self._view_family: str | None = None
        self._current_selection: Selection | None = None
        self._showing_page = True
        self._requested_selection: Selection | None = None
        self._request_revision = 0
        self._active_revision = 0
        self._active_kind: str | None = "page"
        self._layout_generation = -1
        self._display = ImageDisplayState()
        self._current_color_limits: tuple[float, float] | None = None
        self._previous_relim_mode: RelimMode | None = None
        self._pending_image_interaction_origin: PanelInteractionOrigin | None = None
        self._display_rollback: tuple[
            ImageDisplayState,
            tuple[float, float] | None,
            RelimMode | None,
        ] | None = None
        self._area_candidates: dict[
            str,
            tuple[float, float, float, float],
        ] = {}
        self._bound_panel_ids: set[str] = set()
        self._export_commit_lock = threading.Lock()

        super().__init__(
            None,
            window_title="Saved Fit Grid",
            mode_text="EXACT SAVED FIT · GRID EXPLORER · EXPLICIT REFIT",
            loading_summary=f"Resolving {reference.target_ref}…",
            object_prefix="savedFitGrid",
            subject="SAVED FIT GRID",
            memory_limit_bytes=memory_limit_bytes,
        )

        self._retire_tab_pages()
        self._live_page = QtWidgets.QWidget(self._tabs)
        live_layout = QtWidgets.QVBoxLayout(self._live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self._board_widget = QtRasterBoard(
            ("saved-fit-loading",),
            self._live_page,
            columns=1,
            empty_text="Resolving exact saved fit…",
        )
        self._board_widget.setObjectName("savedFitGridBoard")
        self._board_widget.setMinimumSize(480, 320)
        live_layout.addWidget(self._board_widget, 1)
        self._encoded_board = FrozenRasterView(
            "saved-fit-generic",
            self._live_page,
            empty_text="Saved fit raster unavailable",
        )
        self._encoded_board.setObjectName("savedFitGridEncodedBoard")
        self._encoded_board.setMinimumSize(480, 320)
        self._encoded_board.hide()
        live_layout.addWidget(self._encoded_board, 1)
        self._edit_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._tabs,
        )
        self._edit_image_display.setObjectName("savedFitGridImageDisplayEditEditor")
        self._tabs.addTab(self._live_page, "Grid")
        self._tabs.addTab(self._edit_image_display, "Edit")
        self._tabs.tabBar().setVisible(True)

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("savedFitGridDisplaySettingsPopup")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._setting_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._settings_popup,
        )
        self._setting_image_display.setObjectName(
            "savedFitGridImageDisplaySettingEditor"
        )
        settings_layout.addWidget(self._setting_image_display)

        self._previous_page_button = FluentButton(
            "Previous page",
            self,
            color=GREY,
        )
        self._previous_page_button.setObjectName("savedFitGridPreviousPage")
        self._overview_button = FluentButton("Overview", self, color=GREY)
        self._overview_button.setObjectName("savedFitGridOverview")
        self._next_page_button = FluentButton("Next page", self, color=GREY)
        self._next_page_button.setObjectName("savedFitGridNextPage")
        self._selector_switch = FluentSwitch("Selectors", self)
        self._selector_switch.setObjectName("savedFitGridSelectorSwitch")
        self._selector_switch.setChecked(False)
        self._selector_switch.setEnabled(False)
        self._setting_button = FluentButton("Setting…", self, color=GREY)
        self._setting_button.setObjectName("savedFitGridDisplaySettingButton")
        self._setting_button.setEnabled(False)
        self._settings_anchor = FluentSettingsPopupAnchor(
            self._settings_popup,
            self._setting_button,
        )
        self._analyze_button = FluentButton(
            "Analyze → Fit/Refit",
            self,
            color=ORANGE,
        )
        self._analyze_button.setObjectName("savedFitGridAnalyzeFitButton")
        self._analyze_button.setEnabled(False)
        self._export_button = FluentButton("Export image…", self, color=ORANGE)
        self._export_button.setObjectName("savedFitGridExport")
        actions = QtWidgets.QHBoxLayout()
        for button in (
            self._previous_page_button,
            self._overview_button,
            self._next_page_button,
        ):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addWidget(self._selector_switch)
        actions.addWidget(self._setting_button)
        actions.addWidget(self._analyze_button)
        actions.addWidget(self._export_button)
        self._export_button.setEnabled(False)
        actions.addStretch(1)

        self._navigation_host = QtWidgets.QWidget(self)
        host_layout = QtWidgets.QVBoxLayout(self._navigation_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(6)
        host_layout.addLayout(actions)
        self._cell_detail = FluentLabel("", self._navigation_host)
        self._cell_detail.setObjectName("savedFitGridCellDetail")
        self._cell_detail.setWordWrap(True)
        self._cell_detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        host_layout.addWidget(self._cell_detail)
        self._layout.insertWidget(3, self._navigation_host)

        self._previous_page_button.clicked.connect(
            lambda: self._move_page(-1)
        )
        self._overview_button.clicked.connect(self._show_page)
        self._next_page_button.clicked.connect(lambda: self._move_page(1))
        self._export_button.clicked.connect(self._choose_export)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
        self._setting_button.clicked.connect(self._open_display_settings)
        self._analyze_button.clicked.connect(self._open_refit)
        self._board_widget.imagePanelLeftDoubleClicked.connect(
            self._focus_panel_id
        )
        self._encoded_board.normalizedDoubleClicked.connect(self._focus_at)
        self._edit_image_display.applyRequested.connect(
            lambda revision, values: self._apply_image_display_form(
                self._edit_image_display,
                revision,
                values,
            )
        )
        self._setting_image_display.applyRequested.connect(
            lambda revision, values: self._apply_image_display_form(
                self._setting_image_display,
                revision,
                values,
            )
        )
        self._edit_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._edit_image_display)
        )
        self._setting_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._setting_image_display)
        )
        self._sync_image_display_editors()
        self._submit_view(
            "page",
            page_address=None,
            cell_selection=None,
            memory_limit_bytes=self._memory_limit_bytes,
            layout_generation=0,
        )

    @property
    def raster_ready(self) -> bool:
        if self._view_family == "encoded":
            return (
                self._current_encoded_bundle is not None
                and self._encoded_board.has_front
            )
        return (
            self._view_family == "typed-image"
            and self._current_frame is not None
            and self._board_widget.has_front
            and self._board_widget.front_frame is self._current_frame
        )

    def _open_display_settings(self) -> None:
        self._settings_anchor.toggle(
            self._setting_image_display,
            prepare=lambda: self._reload_image_display_editor(
                self._setting_image_display
            ),
        )

    def _install_model(self, model: FitGridModel) -> None:
        if self._model is not None:
            raise RuntimeError("saved-fit compact metadata was installed twice")
        self._model = model
        if not model.axes:
            return
        navigator = AxisLayoutNavigator(
            model.axes,
            model.layout,
            object_prefix="savedFitGrid",
            action_text="Focus exact cell",
            parent=self._navigation_host,
        )
        navigator.candidateChanged.connect(self._candidate_changed)
        navigator.activated.connect(self._focus_indices)
        self._navigator = navigator
        self._navigation_host.layout().insertWidget(1, navigator)
        self._candidate_changed()

    def _retained_state_bytes(
        self,
        *,
        keep_page: bool,
        keep_current: bool,
    ) -> int:
        retained = (
            self._session_retained_bytes
            + (
                0
                if self._model is None
                else self._model.retained_upper_bound_bytes
            )
        )
        if self._view_family == "encoded":
            page = self._page_encoded_bundle
            if keep_page and page is not None:
                retained += page.source_front_peak_nbytes
            current = self._current_encoded_bundle
            if keep_current and current is not None and current is not page:
                retained += current.source_front_peak_nbytes
        else:
            if keep_page and self._page_frame is not None:
                retained += self._page_retained_bytes
            if (
                keep_current
                and self._current_frame is not None
                and self._current_frame is not self._page_frame
            ):
                retained += self._current_retained_bytes
        return retained

    def _available_worker_limit(
        self,
        *,
        keep_page: bool,
        keep_current: bool,
    ) -> int:
        available = self._memory_limit_bytes - self._retained_state_bytes(
            keep_page=keep_page,
            keep_current=keep_current,
        )
        if available <= 0:
            raise MemoryError("saved-fit retained fronts leave no worker budget")
        return available

    def _admit_cached_relayout(self, target: BoardFrame) -> None:
        """Admit the short old-QImage/new-QImage overlap of a local relayout."""

        current = self._current_frame
        page = self._page_frame
        if current is None or page is None:
            raise RuntimeError("saved-fit relayout requires current and page fronts")

        def qimage_bytes(frame: BoardFrame) -> int:
            return sum(len(panel.raster.pixels) for panel in frame.panels)

        reserved_page_front = qimage_bytes(page)
        overlap = max(
            0,
            qimage_bytes(current) + qimage_bytes(target) - reserved_page_front,
        )
        if overlap:
            overlap += 64 << 10
        required = self._retained_state_bytes(
            keep_page=True,
            keep_current=True,
        ) + overlap
        if required > self._memory_limit_bytes:
            raise MemoryError(
                "saved-fit cached relayout QImage overlap exceeds total budget"
            )

    def _frame_matches_display_revision(
        self,
        frame: BoardFrame | None = None,
    ) -> bool:
        candidate = self._current_frame if frame is None else frame
        if candidate is None:
            return False
        if candidate is self._current_frame and (
            not self._board_widget.has_front
            or self._board_widget.front_frame is not candidate
        ):
            return False
        for panel in candidate.panels:
            presentations = {
                item.panel_id: item
                for item in panel.coherence_stamp.presentations
            }
            presentation = presentations.get(panel.panel_id)
            if (
                not isinstance(panel.display_payload, ImagePanelPayload)
                or panel.display_payload.viewport.viewport_revision
                != self._display.revision
                or presentation is None
                or presentation.panel_revision != self._display.revision
            ):
                return False
        return True

    def _visible_image_color_limits(self) -> tuple[float, float] | None:
        if not self._frame_matches_display_revision():
            return None
        frame = self._current_frame
        assert frame is not None
        limits = {
            panel.display_payload.color_limits
            for panel in frame.panels
            if isinstance(panel.display_payload, ImagePanelPayload)
        }
        if len(limits) != 1:
            raise RuntimeError("saved-fit grid front escaped its shared color scale")
        return next(iter(limits))

    def _reload_image_display_editor(
        self,
        editor: FluentRevisionedFormEditor,
    ) -> None:
        if editor not in (
            self._edit_image_display,
            self._setting_image_display,
        ):
            raise ValueError("image display editor does not belong to this window")
        editor.load(
            revision=self._display.revision,
            semantic_identity=self._display,
            values=image_display_form_values(self._display),
            runtime_placeholders=runtime_range_placeholders(
                self._visible_image_color_limits(),
                "color_min",
                "color_max",
            ),
        )

    def _sync_image_display_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        sync_revisioned_form_editors(
            (self._edit_image_display, self._setting_image_display),
            revision=self._display.revision,
            semantic_identity=self._display,
            values=image_display_form_values(self._display),
            runtime_placeholders=runtime_range_placeholders(
                self._visible_image_color_limits(),
                "color_min",
                "color_max",
            ),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _apply_image_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        try:
            if editor not in (
                self._edit_image_display,
                self._setting_image_display,
            ):
                raise ValueError("image display editor does not belong to this window")
            if self._closing or self._future is not None:
                raise RuntimeError("saved-fit display already has active work")
            if not self._frame_matches_display_revision():
                raise RuntimeError("saved-fit display has no current exact front")
            if base_revision != self._display.revision:
                raise RuntimeError(
                    f"image display edit base r{base_revision} is stale; "
                    f"current revision is r{self._display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("image display form must emit one exact mapping")
            candidate = image_display_from_form(
                self._display,
                values,
                current_color_limits=self._visible_image_color_limits(),
            )
            self._commit_image_display(
                candidate,
                accepted_editor=editor,
                accepted_base_revision=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(
                f"Image display edit rejected: {error_summary(error)}"
            )

    def _visible_origin_is_current(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        return (
            isinstance(origin, PanelInteractionOrigin)
            and self._frame_matches_display_revision()
            and origin.panel_id in self._bound_panel_ids
            and self._board_widget.visible_image_origin(origin.panel_id) == origin
            and origin.presentation.panel_revision == self._display.revision
        )

    def _accept_rectangle_gesture(self, gesture: RectangleGesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("saved-fit area callback requires RectangleGesture")
        origin = self._board_widget.visible_image_origin(gesture.panel_id)
        if origin is None or not self._visible_origin_is_current(origin):
            raise RuntimeError("saved-fit area origin is stale")
        if (
            gesture.board_id,
            gesture.layout_generation,
            gesture.sequence,
            gesture.source_identity,
            gesture.viewport_revision,
        ) != (
            origin.board_id,
            origin.layout_generation,
            origin.sequence,
            origin.source_identity,
            self._display.revision,
        ):
            raise RuntimeError("saved-fit area differs from its painted origin")
        self._area_candidates[gesture.panel_id] = gesture.normalized_bounds
        self._board_widget.set_image_rectangle_candidate(
            gesture.normalized_bounds,
            panel_id=gesture.panel_id,
        )
        left, top, right, bottom = gesture.normalized_bounds
        self._diagnostic.setText(
            f"{gesture.panel_id}: DISPLAY ONLY area "
            f"({left:.4f}, {top:.4f})..({right:.4f}, {bottom:.4f})"
        )

    def _accept_image_interaction(
        self,
        command: ImageInteractionCommit,
    ) -> None:
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("saved-fit image callback received an unknown command")
        origin = command.origin
        if not self._visible_origin_is_current(origin):
            raise RuntimeError("saved-fit image interaction origin is stale")
        if isinstance(command, ImageViewportCommit):
            if command.viewport.viewport_revision != self._display.revision + 1:
                raise RuntimeError("saved-fit viewport must advance exactly once")
            candidate = image_display_for_viewport(
                self._display,
                command.viewport,
            )
        else:
            candidate = replace(
                self._display,
                revision=self._display.revision + 1,
                relim_mode=RelimMode.FIXED,
                fixed_color_limits=command.color_limits,
            )
        self._commit_image_display(candidate, interaction_origin=origin)

    def _commit_image_display(
        self,
        state: ImageDisplayState,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
        interaction_origin: PanelInteractionOrigin | None = None,
    ) -> None:
        current = self._display
        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        changed = state != current
        if state.revision != current.revision + int(changed):
            raise ValueError("saved-fit display commit has an invalid revision")
        if accepted_editor is not None and (
            accepted_base_revision != current.revision
            or accepted_editor.base_revision != accepted_base_revision
        ):
            raise ValueError("accepted display editor no longer owns its base")
        if interaction_origin is not None and not changed:
            raise ValueError("image interaction cannot commit a semantic no-op")
        if not changed:
            self._sync_image_display_editors(
                accepted_editor=accepted_editor,
                accepted_base_revision=accepted_base_revision,
            )
            return
        if self._future is not None or self._display_rollback is not None:
            raise RuntimeError("saved-fit display already has active work")
        targets = (*self._page_panels, *self._current_panels)
        seen = set()
        for panel in targets:
            identity = id(panel)
            if identity in seen:
                continue
            seen.add(identity)
            image_viewport_for_display_state(state, panel.home_viewport)
        worker_limit = self._available_worker_limit(
            keep_page=self._page_frame is not None,
            keep_current=self._current_frame is not None,
        )
        rollback = (
            current,
            self._current_color_limits,
            self._previous_relim_mode,
        )
        self._display = state
        self._display_rollback = rollback
        self._pending_image_interaction_origin = interaction_origin
        self._sync_image_display_editors(
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )
        self._status.setText(f"RENDERING SAVED FIT DISPLAY r{state.revision}")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        kind = "display-page" if self._showing_page else "display-focus"
        if not self._submit_grid_reraster(
            kind,
            self._current_panels,
            layout_generation=self._layout_generation,
            memory_limit_bytes=worker_limit,
            previous_relim_mode=current.relim_mode,
            baseline_color_limits=self._current_color_limits,
        ):
            self._restore_display_after_failure()
            self._set_controls_enabled(True)

    def _restore_display_after_failure(self) -> None:
        rollback = self._display_rollback
        if rollback is not None:
            self._display, self._current_color_limits, self._previous_relim_mode = (
                rollback
            )
        origin = self._pending_image_interaction_origin
        if origin is not None:
            self._board_widget.discard_pending_image_interaction(origin)
        self._pending_image_interaction_origin = None
        self._display_rollback = None
        self._sync_image_display_editors()

    def _sync_panel_bindings(self, frame: BoardFrame) -> None:
        intent = self._selector_switch.isChecked()
        for panel_id in tuple(self._bound_panel_ids):
            self._board_widget.unbind_rectangle_selector(panel_id)
        self._bound_panel_ids.clear()
        for panel in frame.panels:
            payload = panel.display_payload
            if not isinstance(payload, ImagePanelPayload):
                raise TypeError("saved-fit board admitted a non-IMAGE panel")
            self._board_widget.bind_rectangle_selector(
                panel.panel_id,
                payload.viewport,
                self._accept_rectangle_gesture,
                enabled=intent,
                interaction_callback=self._accept_image_interaction,
            )
            self._bound_panel_ids.add(panel.panel_id)
            bounds = self._area_candidates.get(panel.panel_id)
            if bounds is not None:
                self._board_widget.set_image_rectangle_candidate(
                    bounds,
                    panel_id=panel.panel_id,
                )

    def _present_typed_frame(
        self,
        frame: BoardFrame,
        panels: tuple[RadialGaussianImageFitPanel, ...],
        *,
        retained_bytes: int,
        color_limits: tuple[float, float],
    ) -> None:
        if self._view_family not in (None, "typed-image"):
            raise RuntimeError("saved-fit view family changed within one exact session")
        panel_ids = tuple(panel.panel_id for panel in frame.panels)
        if panel_ids != tuple(_fit_panel_id(panel) for panel in panels):
            raise ValueError("saved-fit frame order differs from projected panels")
        front = self._board_widget.front_frame
        needs_stage = tuple(self._board_widget.panel_ids) != panel_ids or (
            front is not None
            and (front.board_id, front.layout_generation)
            != (frame.board_id, frame.layout_generation)
        )
        if needs_stage:
            self._board_widget.stage_layout(
                panel_ids,
                board_id=frame.board_id,
                layout_generation=frame.layout_generation,
                columns=_grid_columns(len(panel_ids)),
            )
        self._board_widget.present(frame)
        self._view_family = "typed-image"
        self._encoded_board.hide()
        self._board_widget.show()
        self._current_panels = tuple(panels)
        self._current_frame = frame
        self._current_retained_bytes = int(retained_bytes)
        self._current_color_limits = color_limits
        self._layout_generation = frame.layout_generation
        self._sync_panel_bindings(frame)
        self._pending_image_interaction_origin = None
        self._sync_image_display_editors()

    def _present_encoded_bundle(
        self,
        bundle: EncodedRasterDocument,
    ) -> None:
        if self._view_family not in (None, "encoded"):
            raise RuntimeError("saved-fit view family changed within one exact session")
        if not isinstance(bundle, EncodedRasterDocument) or len(bundle.pages) != 1:
            raise TypeError("generic saved-fit view requires one encoded page")
        model_bytes = (
            0
            if self._model is None
            else self._model.retained_upper_bound_bytes
        )
        other = self._session_retained_bytes + model_bytes
        page = self._page_encoded_bundle
        if page is not None and page is not bundle:
            other += page.source_front_peak_nbytes
        required = self._presentation_peak(bundle, (self._encoded_board,))
        if other + required > self._memory_limit_bytes:
            raise MemoryError("generic saved-fit Qt front exceeds total budget")
        self._encoded_board.present_encoded(
            bundle.pages[0].png_bytes,
            image_format="PNG",
        )
        self._view_family = "encoded"
        self._board_widget.hide()
        self._encoded_board.show()
        self._current_encoded_bundle = bundle

    def _set_selector_enabled(self, enabled: bool) -> None:
        try:
            if enabled and not self._frame_matches_display_revision():
                raise RuntimeError("selector has no current exact IMAGE front")
            self._board_widget.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            self._board_widget.set_selectors_enabled(False)
            self._diagnostic.setText(
                f"Selector rejected: {error_summary(error)}"
            )

    def _set_controls_enabled(self, enabled: bool) -> None:
        page = self._page
        focused = False
        if enabled and not self._showing_page and self._model is not None:
            try:
                self._model.resolve_selection(self._current_selection)
            except (TypeError, ValueError):
                pass
            else:
                focused = True
        self._analyze_button.setEnabled(focused)
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(enabled)
        self._previous_page_button.setEnabled(
            enabled and page is not None and page.previous_address is not None
        )
        self._next_page_button.setEnabled(
            enabled and page is not None and page.next_address is not None
        )
        if self._view_family == "encoded":
            self._overview_button.setEnabled(
                enabled
                and not self._showing_page
                and self._page_encoded_bundle is not None
            )
            self._selector_switch.setEnabled(False)
            self._setting_button.setEnabled(False)
            self._edit_image_display.setEnabled(False)
            self._setting_image_display.setEnabled(False)
            self._export_button.setEnabled(
                enabled and self._current_encoded_bundle is not None
            )
            if self._board_widget.selectors_enabled:
                self._board_widget.set_selectors_enabled(False)
            return
        self._overview_button.setEnabled(
            enabled
            and not self._showing_page
            and self._page_frame is not None
        )
        front_ready = enabled and self._frame_matches_display_revision()
        healthy = False
        for panel_id in self._bound_panel_ids:
            ready = (
                front_ready
                and self._board_widget.image_selector_fault(panel_id) is None
            )
            self._board_widget.set_image_interaction_readiness(panel_id, ready)
            healthy = healthy or ready
        self._selector_switch.setEnabled(healthy)
        intended = self._selector_switch.isChecked() and healthy
        if self._board_widget.selectors_enabled != intended:
            self._board_widget.set_selectors_enabled(intended)
        self._setting_button.setEnabled(front_ready)
        self._edit_image_display.setEnabled(front_ready)
        self._setting_image_display.setEnabled(front_ready)
        self._export_button.setEnabled(front_ready)

    def _open_refit(self) -> None:
        if (
            self._future is not None
            or self._closing
            or self._showing_page
            or self._model is None
            or not self._analyze_button.isEnabled()
        ):
            return
        selection = self._current_selection
        try:
            self._model.resolve_selection(selection)
            self._refit_opener(self._reference, selection)
        except BaseException as error:
            self._status.setText("FIT/REFIT OPEN FAILED")
            self._summary.setText("The exact saved artifact remains unchanged")
            self._diagnostic.setText(error_summary(error))

    def _candidate_changed(self) -> None:
        navigator = self._navigator
        model = self._model
        if navigator is None or model is None:
            return
        indices = navigator.indices
        if indices is None:
            self._cell_detail.setText(
                "Choose every non-singleton batch axis to inspect one exact saved fit cell."
            )
            return
        try:
            selection = model.selection_for_indices(indices)
            storage, _multi, address = model.resolve_selection(selection)
        except BaseException as error:
            self._cell_detail.setText(error_summary(error))
            return
        self._cell_detail.setText(
            f"{address}\nstorage row {storage} · activate to load stored fit facts"
        )

    def _submit_view(
        self,
        kind: str,
        *,
        page_address: tuple[int, ...] | None,
        cell_selection: Selection | None,
        memory_limit_bytes: int,
        layout_generation: int,
    ) -> None:
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = kind
        self._requested_selection = cell_selection
        if not self._submit_future(
            _load_grid_view,
            self._view_loader,
            self._reference,
            page_address,
            cell_selection,
            memory_limit_bytes,
            self._active_revision,
            self._model is None,
            self._display,
            self._current_color_limits,
            self._previous_relim_mode,
            layout_generation,
            self._cancelled,
        ):
            self._active_kind = None
            self._set_controls_enabled(True)

    def _submit_grid_reraster(
        self,
        kind: str,
        panels: tuple[RadialGaussianImageFitPanel, ...],
        *,
        layout_generation: int,
        memory_limit_bytes: int,
        previous_relim_mode: RelimMode | None,
        baseline_color_limits: tuple[float, float] | None,
    ) -> bool:
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = kind
        submitted = self._submit_future(
            _rerasterize_grid_view,
            tuple(panels),
            self._display,
            baseline_color_limits,
            previous_relim_mode,
            layout_generation,
            self._active_revision,
            memory_limit_bytes,
            self._cancelled,
        )
        if not submitted:
            self._active_kind = None
        return submitted

    def _start_page(self, address: tuple[int, ...]) -> None:
        if self._future is not None or self._closing:
            return
        try:
            worker_limit = self._available_worker_limit(
                keep_page=self._page is not None,
                keep_current=(
                    self._current_frame is not None
                    or self._current_encoded_bundle is not None
                ),
            )
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))
            return
        self._status.setText("BUILDING FIT GRID PAGE")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        self._submit_view(
            "page",
            page_address=tuple(address),
            cell_selection=None,
            memory_limit_bytes=worker_limit,
            layout_generation=self._layout_generation + 1,
        )

    def _move_page(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("fit grid page direction must be -1 or 1")
        page = self._page
        if page is None:
            return
        address = (
            page.previous_address if direction < 0 else page.next_address
        )
        if address is not None:
            self._start_page(address)

    def _focus_panel_id(self, panel_id: str) -> None:
        if self._future is not None or self._closing:
            return
        if not self._showing_page:
            self._show_page()
            return
        panel = next(
            (
                candidate
                for candidate in self._page_panels
                if _fit_panel_id(candidate) == panel_id
            ),
            None,
        )
        if panel is None:
            self._diagnostic.setText("Ignored stale saved-fit panel activation")
            return
        if panel.fit_storage_index is None:
            self._status.setText("FIT CELL NOT PRESENT")
            self._cell_detail.setText(
                "This logical gallery position is a sparse-layout hole; "
                "no neighbouring stored fit row was substituted."
            )
            return
        self._start_focus(panel.selection)

    def _focus_at(self, x: float, y: float) -> None:
        if (
            self._future is not None
            or self._closing
            or self._view_family != "encoded"
        ):
            return
        if not self._showing_page:
            self._show_page()
            return
        for region in self._regions:
            if not region.contains(x, y):
                continue
            if region.fit_storage_index is None:
                self._status.setText("FIT CELL NOT PRESENT")
                self._cell_detail.setText(
                    "This logical gallery position is a sparse-layout hole; "
                    "no neighbouring stored fit row was substituted."
                )
                return
            self._start_focus(region.selection)
            return

    def _focus_indices(self, indices: object) -> None:
        model = self._model
        if model is None:
            return
        try:
            selection = model.selection_for_indices(tuple(indices))
            if selection is None:
                return
            self._start_focus(selection)
        except BaseException as error:
            self._status.setText("FIT CELL INVALID")
            self._diagnostic.setText(error_summary(error))

    def _start_focus(self, selection: Selection | None) -> None:
        if self._future is not None or self._closing:
            return
        model = self._model
        if model is None or (
            self._view_family == "typed-image" and self._page_frame is None
        ) or (
            self._view_family == "encoded"
            and self._page_encoded_bundle is None
        ):
            return
        try:
            model.resolve_selection(selection)
            cached = (
                next(
                    (
                        panel
                        for panel in self._page_panels
                        if panel.selection == selection
                        and panel.fit_storage_index is not None
                    ),
                    None,
                )
                if self._view_family == "typed-image"
                else None
            )
            if cached is not None:
                self._start_cached_focus(cached)
                return
            worker_limit = self._available_worker_limit(
                keep_page=True,
                keep_current=True,
            )
        except BaseException as error:
            self._status.setText("FIT CELL INVALID")
            self._diagnostic.setText(error_summary(error))
            return
        self._status.setText("BUILDING FIT CELL")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        self._submit_view(
            "focus",
            page_address=None,
            cell_selection=selection,
            memory_limit_bytes=worker_limit,
            layout_generation=self._layout_generation + 1,
        )

    def _start_cached_focus(
        self,
        panel: RadialGaussianImageFitPanel,
    ) -> None:
        page_frame = self._page_frame
        model = self._model
        if page_frame is None or model is None:
            return
        panel_id = _fit_panel_id(panel)
        if self._frame_matches_display_revision(page_frame):
            self._request_revision += 1
            frame = _reframe_existing_image_panels(
                page_frame,
                (panel,),
                layout_generation=self._layout_generation + 1,
                sequence=self._request_revision,
            )
            page_color_limits = self._page_color_limits
            payload = frame.panels[0].display_payload
            if (
                page_color_limits is None
                or not isinstance(payload, ImagePanelPayload)
                or payload.color_limits != page_color_limits
            ):
                self._diagnostic.setText(
                    "Cached fit cell differs from its exact page colour scale"
                )
                return
            try:
                self._admit_cached_relayout(frame)
                self._present_typed_frame(
                    frame,
                    (panel,),
                    retained_bytes=0,
                    color_limits=page_color_limits,
                )
            except BaseException as error:
                self._status.setText("FIT CELL INVALID")
                self._diagnostic.setText(error_summary(error))
                return
            self._showing_page = False
            self._current_selection = panel.selection
            self._requested_selection = panel.selection
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(model.summary)
            self._cell_detail.setText(panel.summary)
            self._diagnostic.setText("")
            if self._navigator is not None:
                _storage, multi, _label = model.resolve_selection(panel.selection)
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
            self._set_controls_enabled(True)
            return
        try:
            worker_limit = self._available_worker_limit(
                keep_page=True,
                keep_current=True,
            )
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))
            return
        self._requested_selection = panel.selection
        self._status.setText("RENDERING CACHED FIT CELL")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if not self._submit_grid_reraster(
            "cached-focus",
            (panel,),
            layout_generation=self._layout_generation + 1,
            memory_limit_bytes=worker_limit,
            previous_relim_mode=self._previous_relim_mode,
            baseline_color_limits=self._page_color_limits,
        ):
            self._set_controls_enabled(True)

    def _show_page(self) -> None:
        model = self._model
        page = self._page
        if self._view_family == "encoded":
            bundle = self._page_encoded_bundle
            if (
                self._future is not None
                or model is None
                or page is None
                or bundle is None
                or self._showing_page
            ):
                return
            try:
                self._present_encoded_bundle(bundle)
            except BaseException as error:
                self._status.setText("DISPLAY FAILED")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
                return
            self._regions = self._page_regions
            self._showing_page = True
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
            self._diagnostic.setText("")
            self._set_controls_enabled(True)
            return
        frame = self._page_frame
        if (
            self._future is not None
            or model is None
            or page is None
            or frame is None
            or self._showing_page
        ):
            return
        if self._frame_matches_display_revision(frame):
            self._request_revision += 1
            restored = _reframe_existing_image_panels(
                frame,
                self._page_panels,
                layout_generation=self._layout_generation + 1,
                sequence=self._request_revision,
            )
            try:
                self._admit_cached_relayout(restored)
                self._present_typed_frame(
                    restored,
                    self._page_panels,
                    retained_bytes=self._page_retained_bytes,
                    color_limits=self._page_color_limits or (0.0, 1.0),
                )
            except BaseException as error:
                self._status.setText("DISPLAY FAILED")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
                return
            self._page_frame = restored
            self._showing_page = True
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
            self._diagnostic.setText("")
            self._set_controls_enabled(True)
            return
        try:
            worker_limit = self._available_worker_limit(
                keep_page=True,
                keep_current=True,
            )
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))
            return
        self._requested_selection = None
        self._status.setText("RENDERING FIT GRID PAGE")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if not self._submit_grid_reraster(
            "return-page-display",
            self._page_panels,
            layout_generation=self._layout_generation + 1,
            memory_limit_bytes=worker_limit,
            previous_relim_mode=self._previous_relim_mode,
            baseline_color_limits=self._page_color_limits,
        ):
            self._set_controls_enabled(True)

    def _choose_export(self) -> None:
        if (
            self._future is not None
            or self._closing
            or (
                self._current_frame is None
                and self._current_encoded_bundle is None
            )
        ):
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export saved fit view",
            "saved_fit_grid.png",
            "Images (*.png *.pdf *.svg *.jpg *.jpeg)",
        )
        if path:
            self._start_export(Path(path))

    def _start_export(self, destination: Path) -> None:
        if self._future is not None or self._closing:
            return
        page = self._page
        model = self._model
        try:
            if model is None:
                raise RuntimeError("saved-fit export has no compact model")
            export_limit = self._available_worker_limit(
                keep_page=True,
                keep_current=True,
            )
            typed_export = self._view_family == "typed-image"
            if typed_export:
                frame = self._current_frame
                panels = self._current_panels
                color_limits = self._current_color_limits
                if (
                    frame is None
                    or not panels
                    or color_limits is None
                    or not self._frame_matches_display_revision(frame)
                ):
                    raise RuntimeError("saved-fit typed export has no exact current front")
                panel_ids = tuple(_fit_panel_id(panel) for panel in panels)
                if panel_ids != tuple(panel.panel_id for panel in frame.panels):
                    raise ValueError("saved-fit export panel order differs from its front")
                _artifact, _inputs, expected_join = _fit_grid_join_identity(
                    panels,
                    panel_ids,
                )
                painted_join = frame.panels[0].coherence_stamp.join_key_digest
                if expected_join != painted_join:
                    raise ValueError("saved-fit export join differs from its front")
                for projected, rendered in zip(panels, frame.panels, strict=True):
                    payload = rendered.display_payload
                    if (
                        not isinstance(payload, ImagePanelPayload)
                        or payload.image is not projected.image
                        or payload.evaluated_input != projected.evaluated_input
                        or payload.fit_overlay != projected.fit_overlay
                        or payload.color_limits != color_limits
                    ):
                        raise ValueError(
                            "saved-fit export projection differs from its painted front"
                        )
        except BaseException as error:
            self._status.setText("FIT VIEW EXPORT FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = "export"
        self._status.setText("EXPORTING FIT VIEW")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if typed_export:
            submitted = self._submit_future(
                _export_typed_grid_view,
                panels,
                self._display,
                color_limits,
                _grid_columns(len(panels)),
                expected_join,
                Path(destination),
                export_limit,
                self._active_revision,
                self._cancelled,
                self._export_commit_lock,
            )
        else:
            submitted = self._submit_future(
                _export_grid_view,
                self._view_loader,
                self._reference,
                model.identity,
                (
                    None
                    if self._current_selection is not None or page is None
                    else page.address
                ),
                self._current_selection,
                Path(destination),
                export_limit,
                self._active_revision,
                self._cancelled,
                self._export_commit_lock,
            )
        if not submitted:
            self._active_kind = None
            self._set_controls_enabled(True)

    def _accept_view_result(self, result, kind: str, revision: int) -> None:
        (
            result_revision,
            model,
            model_identity,
            session_retained_bytes,
            page,
            cell_summary,
            resolved_selection,
            summary,
            projection,
        ) = result
        if result_revision != revision or revision != self._request_revision:
            self._set_controls_enabled(True)
            return
        if self._model is None:
            if not isinstance(model, FitGridModel):
                raise TypeError("initial saved-fit page omitted compact metadata")
            if model.identity != model_identity:
                raise ValueError("saved-fit worker metadata identity changed")
            self._session_retained_bytes = positive_integer(
                session_retained_bytes,
                "saved-fit session retained bytes",
            )
            self._install_model(model)
        else:
            if (
                model is not None
                or model_identity != self._model.identity
                or session_retained_bytes != self._session_retained_bytes
            ):
                raise ValueError("saved-fit metadata changed during one exact-ref session")
        current_model = self._model
        assert current_model is not None
        if not isinstance(projection, tuple) or not projection:
            raise TypeError("saved-fit worker omitted its presentation family")
        family = projection[0]
        if family not in ("typed-image", "encoded"):
            raise ValueError("saved-fit worker returned an unknown presentation family")
        if self._view_family is not None and family != self._view_family:
            raise ValueError("saved-fit presentation family changed during one session")
        if kind == "page":
            if not isinstance(page, FitGridPage) or cell_summary is not None:
                raise TypeError("saved-fit page result is invalid")
            if family == "typed-image":
                if len(projection) != 5:
                    raise TypeError("typed saved-fit projection has invalid fields")
                _tag, panels, frame, color_limits, frame_retained_bytes = projection
                next_panels = tuple(panels)
                self._present_typed_frame(
                    frame,
                    next_panels,
                    retained_bytes=frame_retained_bytes,
                    color_limits=color_limits,
                )
                self._page_panels = next_panels
                self._page_frame = frame
                self._page_retained_bytes = frame_retained_bytes
                self._page_color_limits = color_limits
                self._current_frame = self._page_frame
                self._current_retained_bytes = self._page_retained_bytes
            else:
                if len(projection) != 3:
                    raise TypeError("encoded saved-fit projection has invalid fields")
                _tag, bundle, regions = projection
                self._present_encoded_bundle(bundle)
                self._page_encoded_bundle = bundle
                self._page_regions = tuple(regions)
                self._regions = self._page_regions
            self._page = page
            self._current_selection = None
            self._showing_page = True
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(summary)
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
        elif kind == "focus":
            if page is not None or not isinstance(cell_summary, FitGridCellSummary):
                raise TypeError("saved-fit focus result is invalid")
            if family == "typed-image":
                if len(projection) != 5:
                    raise TypeError("typed saved-fit projection has invalid fields")
                _tag, panels, frame, color_limits, frame_retained_bytes = projection
                self._present_typed_frame(
                    frame,
                    tuple(panels),
                    retained_bytes=frame_retained_bytes,
                    color_limits=color_limits,
                )
            else:
                if len(projection) != 3:
                    raise TypeError("encoded saved-fit projection has invalid fields")
                _tag, bundle, regions = projection
                self._present_encoded_bundle(bundle)
                self._regions = tuple(regions)
            self._current_selection = resolved_selection
            self._showing_page = False
            if self._navigator is not None:
                _storage, multi, _label = current_model.resolve_selection(
                    resolved_selection
                )
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(summary)
            self._cell_detail.setText(cell_summary.text)
        else:
            raise RuntimeError("unknown saved-fit view result")
        self._requested_selection = (
            None if kind == "page" else resolved_selection
        )
        self._previous_relim_mode = self._display.relim_mode
        self._diagnostic.setText("")
        self._set_controls_enabled(True)

    def _accept_reraster_result(
        self,
        result,
        kind: str,
        revision: int,
    ) -> None:
        (
            result_revision,
            panels,
            display,
            frame,
            color_limits,
            retained_bytes,
        ) = result
        if result_revision != revision or revision != self._request_revision:
            self._set_controls_enabled(True)
            return
        if display != self._display:
            raise ValueError("saved-fit reraster returned another display revision")
        model = self._model
        page = self._page
        if model is None:
            raise RuntimeError("saved-fit reraster has no compact model")
        if kind in ("display-page", "return-page-display"):
            if page is None:
                raise RuntimeError("saved-fit page reraster has no page metadata")
            candidate_panels = tuple(panels)
            self._present_typed_frame(
                frame,
                candidate_panels,
                retained_bytes=retained_bytes,
                color_limits=color_limits,
            )
            self._page_panels = candidate_panels
            self._page_frame = frame
            self._page_retained_bytes = retained_bytes
            self._page_color_limits = color_limits
            self._current_frame = self._page_frame
            self._current_retained_bytes = self._page_retained_bytes
            self._showing_page = True
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
        elif kind in ("display-focus", "cached-focus"):
            focused = tuple(panels)
            if len(focused) != 1:
                raise ValueError("saved-fit focus reraster must contain one panel")
            panel = focused[0]
            if kind == "display-focus":
                selection = self._current_selection
            else:
                selection = self._requested_selection
            if panel.selection != selection:
                raise ValueError("saved-fit focus reraster changed exact selection")
            self._present_typed_frame(
                frame,
                focused,
                retained_bytes=retained_bytes,
                color_limits=color_limits,
            )
            self._showing_page = False
            self._current_selection = selection
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(model.summary)
            self._cell_detail.setText(panel.summary)
            if self._navigator is not None:
                _storage, multi, _label = model.resolve_selection(selection)
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
        else:
            raise RuntimeError("unknown saved-fit reraster result")
        self._previous_relim_mode = self._display.relim_mode
        self._display_rollback = None
        self._pending_image_interaction_origin = None
        self._diagnostic.setText("")
        self._set_controls_enabled(True)

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        revision = self._active_revision
        self._active_kind = None
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                if kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._restore_display_after_failure()
                self._status.setText("SAVED FIT GRID CANCELLED")
                self._set_controls_enabled(True)
        except BaseException as error:
            if not self._closing:
                if kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._restore_display_after_failure()
                self._status.setText("SAVED FIT GRID FAILED")
                self._summary.setText("The exact saved artifact remains unchanged")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
        else:
            if self._closing:
                return
            try:
                if kind in ("page", "focus"):
                    self._accept_view_result(result, kind, revision)
                elif kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._accept_reraster_result(result, kind, revision)
                elif kind == "export":
                    result_revision, destination = result
                    if (
                        result_revision == revision == self._request_revision
                    ):
                        self._status.setText("FIT VIEW EXPORTED")
                        self._diagnostic.setText(str(destination))
                    self._set_controls_enabled(True)
                else:
                    raise RuntimeError("unknown saved-fit worker result")
            except BaseException as error:
                if kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._restore_display_after_failure()
                self._status.setText("SAVED FIT GRID FAILED")
                self._summary.setText("The exact saved artifact remains unchanged")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and not self._showing_page:
            self._show_page()
            event.accept()
            return
        super().keyPressEvent(event)

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        cancel_export_commits(
            cancelled=self._cancelled,
            commit_lock=self._export_commit_lock,
        )
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(False)
        for button in (
            self._previous_page_button,
            self._overview_button,
            self._next_page_button,
            self._selector_switch,
            self._setting_button,
            self._analyze_button,
            self._export_button,
        ):
            button.setEnabled(False)
        self._settings_popup.hide()
        super().shutdown()

    def _forget_presented_pages(self) -> None:
        """Drop every page and focus display fact this window still holds."""

        self._page = None
        self._page_panels = ()
        self._page_frame = None
        self._page_retained_bytes = 0
        self._page_color_limits = None
        self._page_encoded_bundle = None
        self._page_regions = ()
        self._current_panels = ()
        self._current_frame = None
        self._current_retained_bytes = 0
        self._current_encoded_bundle = None
        self._regions = ()

    def _clear_bundle(self) -> None:
        self._bundle = None
        self._boards = ()
        board = getattr(self, "_board_widget", None)
        if board is not None:
            origin = self._pending_image_interaction_origin
            if origin is not None:
                board.discard_pending_image_interaction(origin)
            for panel_id in tuple(self._bound_panel_ids):
                board.unbind_rectangle_selector(panel_id)
            self._bound_panel_ids.clear()
            board.clear()
        encoded_board = getattr(self, "_encoded_board", None)
        if encoded_board is not None:
            encoded_board.clear()
        self._forget_presented_pages()
        self._current_selection = None
        self._requested_selection = None
        self._showing_page = True
        self._pending_image_interaction_origin = None
        self._display_rollback = None
        self._area_candidates.clear()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._view_loader = None
            self._refit_opener = None
            self._model = None
            self._session_retained_bytes = 0
            self._navigator = None
            self._forget_presented_pages()
        super()._finish_close_if_ready()


def open_saved_fit_grid_workbench(
    view_loader,
    refit_opener,
    reference: FitResultArtifactRef,
    *,
    memory_limit_bytes: int = _DEFAULT_FIT_GRID_MEMORY_LIMIT_BYTES,
) -> SavedFitGridWindow:
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    return open_workbench_window(
        lambda: SavedFitGridWindow(
            view_loader,
            refit_opener,
            reference,
            memory_limit_bytes=limit,
        )
    )


__all__ = ["SavedFitGridWindow", "open_saved_fit_grid_workbench"]
