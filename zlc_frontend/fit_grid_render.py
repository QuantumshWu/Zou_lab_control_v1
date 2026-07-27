"""Canonical saved-Fit grid composition, identity, and display projection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import math
from typing import Literal, TypeAlias

from zlc_data import Selection, dataset_revision_ref_to_tree, selection_to_tree
from zlc_storage import canonical_digest, canonical_text, positive_integer

from .data_figure import DataFigure, FigurePanelRegion
from .display_range import (
    RelimMode,
    deadband_display_range,
    validated_display_range,
)
from .figure import EvaluatedProjectionIdentity
from .encoded_raster import EncodedRasterDocument, EncodedRasterPage
from .fit_grid import FitGridCellSummary, FitGridModel, FitGridPage
from .fit_image_projection import RadialGaussianImageFitPanel
from .image_display import (
    ImageDisplayState,
    evaluated_image_data_range,
    image_viewport_for_display_state,
    resolve_image_color_limits_from_range,
)
from .panel_size import DEFAULT_PANEL_SIZE
from .plot_layout import PanelSurfaceGeometry, panel_surface_geometry
from .render import (
    BoardFrame,
    CoherenceStamp,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
)


SAVED_FIT_GRID_BOARD_ID = "saved-fit-grid"

SAVED_FIT_GRID_JOIN_SCHEMA_DIGEST = canonical_digest(
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


FitGridTypedProjection: TypeAlias = tuple[
    Literal["typed-image"],
    tuple[RadialGaussianImageFitPanel, ...],
    "FitGridBoardFront",
]
FitGridEncodedProjection: TypeAlias = tuple[
    Literal["encoded"],
    EncodedRasterDocument,
    tuple[FigurePanelRegion, ...],
]
FitGridProjection: TypeAlias = FitGridTypedProjection | FitGridEncodedProjection


@dataclass(frozen=True, slots=True)
class FitGridBoardFront:
    """One coherent saved-Fit board and its exact logical presentation box.

    ``QtRasterBoard`` uses tight equal-cell tiling: each cell raster already
    owns its Matplotlib margins, so there is no second Qt gutter between cells.
    This contract is therefore the sole owner of rows, columns, and aggregate
    logical size.  Device-pixel ratio changes only ``raster_size`` and never
    silently resize the logical board.
    """

    frame: BoardFrame
    surface_geometry: PanelSurfaceGeometry
    color_limits: tuple[float, float]
    rows: int = field(init=False)
    columns: int = field(init=False)
    logical_size: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, BoardFrame):
            raise TypeError("saved-Fit front frame must be BoardFrame")
        if self.frame.board_id != SAVED_FIT_GRID_BOARD_ID:
            raise ValueError("saved-Fit front has another board identity")
        if not isinstance(self.surface_geometry, PanelSurfaceGeometry):
            raise TypeError(
                "saved-Fit front surface must be PanelSurfaceGeometry"
            )
        count = positive_integer(
            len(self.frame.panels),
            "saved-Fit front panel count",
        )
        columns = fit_grid_columns(count)
        rows = math.ceil(count / columns)
        panel_width, panel_height = self.surface_geometry.logical_size
        expected_raster = self.surface_geometry.raster_size
        if any(
            (panel.raster.width, panel.raster.height) != expected_raster
            for panel in self.frame.panels
        ):
            raise ValueError(
                "saved-Fit front raster geometry differs from its surface"
            )
        limits = validated_display_range(
            self.color_limits,
            "saved-Fit front color limits",
        )
        if any(
            not isinstance(panel.display_payload, ImagePanelPayload)
            or panel.display_payload.color_limits != limits
            for panel in self.frame.panels
        ):
            raise ValueError(
                "saved-Fit front panels escaped their shared color limits"
            )
        object.__setattr__(self, "color_limits", limits)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self,
            "logical_size",
            (columns * panel_width, rows * panel_height),
        )


class FitGridRenderSession:
    """Persistent per-cell Agg owner for one interactive saved-Fit explorer."""

    def __init__(
        self,
        *,
        size_name: str = DEFAULT_PANEL_SIZE,
        pixel_ratio: float = 1.0,
    ) -> None:
        self._surface_geometry = panel_surface_geometry(
            size_name,
            pixel_ratio=pixel_ratio,
        )
        self._renderers: dict[str, object] = {}
        self._range_key: tuple[object, ...] | None = None
        self._panel_ranges: dict[str, tuple[float, float] | None] = {}
        self._pooled_range: tuple[float, float] | None = None

    @property
    def surface_geometry(self) -> PanelSurfaceGeometry:
        return self._surface_geometry

    def use_surface_geometry(self, geometry: PanelSurfaceGeometry) -> bool:
        """Adopt one request-frozen surface on the worker-owned session."""

        if not isinstance(geometry, PanelSurfaceGeometry):
            raise TypeError("saved-Fit surface must be PanelSurfaceGeometry")
        if geometry == self._surface_geometry:
            return False
        self.close()
        self._surface_geometry = geometry
        return True

    @staticmethod
    def _projection_identity(panel: RadialGaussianImageFitPanel):
        return EvaluatedProjectionIdentity(
            "saved-fit-grid",
            0,
            panel.evaluated_input,
            "saved-fit-cell",
            (),
            (),
            (),
            panel.image,
        )

    def _prepare_panels(
        self,
        panels: tuple[RadialGaussianImageFitPanel, ...],
        check_cancelled: Callable[[], None] | None,
    ) -> tuple[float, float] | None:
        panel_ids = tuple(fit_grid_panel_id(panel) for panel in panels)
        retired = self._renderers.keys() - set(panel_ids)
        for panel_id in tuple(retired):
            self._renderers.pop(panel_id).close()
        key = tuple(
            (
                panel_id,
                panel.evaluated_input.ref,
            )
            for panel_id, panel in zip(panel_ids, panels, strict=True)
        )
        if key != self._range_key:
            ranges: dict[str, tuple[float, float] | None] = {}
            low = high = None
            for panel_id, panel in zip(panel_ids, panels, strict=True):
                _check(check_cancelled)
                current = evaluated_image_data_range((panel.image,))
                ranges[panel_id] = current
                if current is None:
                    continue
                low = current[0] if low is None else min(low, current[0])
                high = current[1] if high is None else max(high, current[1])
            self._range_key = key
            self._panel_ranges = ranges
            self._pooled_range = None if low is None else (low, high)
        return self._pooled_range

    def _panel_range(self, panel_id: str) -> tuple[float, float] | None:
        try:
            return self._panel_ranges[panel_id]
        except KeyError as error:
            raise RuntimeError("saved-Fit panel range was not prepared") from error

    def _renderer(self, panel_id: str):
        renderer = self._renderers.get(panel_id)
        if renderer is None:
            from .matplotlib_render import ImagePanelAggRenderer

            geometry = self._surface_geometry
            renderer = ImagePanelAggRenderer(
                width=geometry.raster_size[0],
                height=geometry.raster_size[1],
                dpi=geometry.dpi,
                size_name=geometry.size_name,
            )
            self._renderers[panel_id] = renderer
        return renderer

    def build_front(self, panels, display, **options):
        try:
            return build_fit_image_grid_front(
                panels,
                display,
                _session=self,
                **options,
            )
        except BaseException:
            self.close()
            raise

    def project_loaded(self, loaded, **options):
        try:
            return project_loaded_fit_grid(
                loaded,
                _session=self,
                **options,
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        renderers, self._renderers = tuple(self._renderers.values()), {}
        self._range_key = None
        self._panel_ranges.clear()
        self._pooled_range = None
        for renderer in renderers:
            renderer.close()


def _check(check_cancelled: Callable[[], None] | None) -> None:
    if check_cancelled is not None:
        check_cancelled()

def fit_grid_panel_id(panel: RadialGaussianImageFitPanel) -> str:
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

def fit_grid_columns(count: int) -> int:
    count = positive_integer(count, "saved-fit grid panel count")
    return min(6, max(1, math.ceil(math.sqrt(count))))

def validate_fit_grid_regions(
    model: FitGridModel,
    regions: tuple[FigurePanelRegion, ...],
) -> tuple[FigurePanelRegion, ...]:
    """Validate the still-consumed generic 1D saved-fit hit map."""

    prepared = tuple(regions)
    if not prepared or any(
        not isinstance(region, FigurePanelRegion) for region in prepared
    ):
        raise TypeError("saved-fit generic renderer must return panel regions")
    if len({region.key for region in prepared}) != len(prepared):
        raise ValueError("saved-fit grid panel keys must be unique")
    for region in prepared:
        if region.fit_storage_index != model.storage_index_or_none(
            region.fit_selection
        ):
            raise ValueError(
                "saved-fit panel hit map disagrees with the exact batch layout"
            )
    return prepared


def _validated_loaded_fit_grid(
    loaded: object,
    *,
    artifact_identity: str,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
) -> tuple[
    DataFigure,
    FitGridModel,
    FitGridPage | None,
    FitGridCellSummary | None,
    tuple[Selection | None, ...] | None,
    Selection | None,
    str,
]:
    """Admit one repository result against the exact requested logical view."""

    identity = canonical_text(artifact_identity, "fit artifact identity")
    if not isinstance(loaded, tuple) or len(loaded) != 4:
        raise TypeError(
            "saved-fit loader must return figure/model/page/cell summary"
        )
    figure, model, page, cell_summary = loaded
    if not isinstance(figure, DataFigure):
        raise TypeError("saved-fit loader must return DataFigure")
    if not isinstance(model, FitGridModel):
        raise TypeError("saved-fit loader must return FitGridModel")
    if model.artifact_identity != identity:
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
    summary = f"{model.summary} · {page.label}" if page is not None else model.summary
    return (
        figure,
        model,
        page,
        cell_summary,
        expected_page_selections,
        resolved_selection,
        summary,
    )

def _image_data_range(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    check_cancelled: Callable[[], None] | None,
) -> tuple[float, float] | None:
    """Pool exact valid ranges without concatenating cell-sized arrays."""

    def images():
        for panel in panels:
            _check(check_cancelled)
            yield panel.image

    return evaluated_image_data_range(images())

def shared_fit_grid_color_limits(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    check_cancelled: Callable[[], None] | None = None,
    _data_range_override: tuple[float, float] | None | object = ...,
) -> tuple[float, float]:
    data_range = (
        _image_data_range(panels, check_cancelled)
        if _data_range_override is ...
        else _data_range_override
    )
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

def fit_grid_join_identity(
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


def build_fit_image_grid_front(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    *,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    sequence: int,
    check_cancelled: Callable[[], None] | None = None,
    _session: FitGridRenderSession | None = None,
) -> FitGridBoardFront:
    """Compose one coherent saved-Fit image front with exact board geometry."""

    panels = tuple(panels)
    if not panels:
        raise ValueError("saved-fit IMAGE view requires at least one panel")
    if len({fit_grid_panel_id(panel) for panel in panels}) != len(panels):
        raise ValueError("saved-fit logical panels do not have unique identities")
    _check(check_cancelled)
    session = FitGridRenderSession() if _session is None else _session
    color_limits = shared_fit_grid_color_limits(
        panels,
        display,
        current_color_limits,
        previous_relim_mode,
        check_cancelled,
        _data_range_override=session._prepare_panels(panels, check_cancelled),
    )
    resolved_display = replace(
        display,
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=color_limits,
    )
    panel_ids = tuple(fit_grid_panel_id(panel) for panel in panels)
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
    artifact_identity, inputs, join_key_digest = fit_grid_join_identity(
        panels,
        panel_ids,
    )
    stamp = CoherenceStamp(
        artifact_identity,
        artifact_identity,
        "SavedFitGridJoin",
        SAVED_FIT_GRID_JOIN_SCHEMA_DIGEST,
        join_key_digest,
        inputs,
        presentations,
    )
    frames = []
    try:
        for panel_id, panel in zip(panel_ids, panels, strict=True):
            _check(check_cancelled)
            renderer = session._renderer(panel_id)
            viewport = image_viewport_for_display_state(display, panel.home_viewport)
            data_range, effective_limits = resolve_image_color_limits_from_range(
                session._panel_range(panel_id),
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
                projection_identity=session._projection_identity(panel),
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
        if _session is None:
            session.close()
    _check(check_cancelled)
    return FitGridBoardFront(
        BoardFrame(
            SAVED_FIT_GRID_BOARD_ID,
            layout_generation,
            sequence,
            tuple(frames),
        ),
        session.surface_geometry,
        color_limits,
    )


def project_loaded_fit_grid(
    loaded: object,
    *,
    artifact_identity: str,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    sequence: int,
    check_cancelled: Callable[[], None] | None = None,
    _session: FitGridRenderSession | None = None,
) -> tuple[
    FitGridModel,
    FitGridPage | None,
    FitGridCellSummary | None,
    Selection | None,
    str,
    FitGridProjection,
]:
    """Project one repository load through the sole saved-Fit view policy."""

    _check(check_cancelled)
    (
        figure,
        model,
        page,
        cell_summary,
        expected_page_selections,
        resolved_selection,
        summary,
    ) = _validated_loaded_fit_grid(
        loaded,
        artifact_identity=artifact_identity,
        page_address=page_address,
        cell_selection=cell_selection,
    )
    if model.model_id == "radial_gaussian_center":
        layers = tuple(figure.document.layers)
        if len(layers) != 1:
            raise ValueError(
                "saved-fit typed IMAGE explorer requires exactly one layer"
            )
        panels = figure.radial_gaussian_image_fit_panels(
            layers[0].layer_id,
            artifact_identity=artifact_identity,
        )
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
            assert cell_summary is not None
            if len(panels) != 1:
                raise ValueError("exact saved-fit focus must project one IMAGE panel")
            if panels[0].selection != cell_selection:
                raise ValueError("focused typed panel belongs to another selection")
            if panels[0].fit_storage_index != cell_summary.storage_index:
                raise ValueError(
                    "focused typed panel and stored cell summary disagree"
                )
            if panels[0].summary != cell_summary.text:
                raise ValueError(
                    "focused typed panel summary diverged from grid metadata"
                )
        _check(check_cancelled)
        front = build_fit_image_grid_front(
            panels,
            display,
            current_color_limits=current_color_limits,
            previous_relim_mode=previous_relim_mode,
            layout_generation=layout_generation,
            sequence=sequence,
            check_cancelled=check_cancelled,
            _session=_session,
        )
        projection: FitGridProjection = (
            "typed-image",
            panels,
            front,
        )
    else:
        payload, regions = figure.to_png_bytes_with_panel_regions()
        _check(check_cancelled)
        bundle = EncodedRasterDocument(
            summary,
            (EncodedRasterPage("figure", "Fit grid", payload),),
        )
        prepared_regions = validate_fit_grid_regions(model, regions)
        if expected_page_selections is not None and tuple(
            region.fit_selection for region in prepared_regions
        ) != expected_page_selections:
            raise ValueError(
                "saved-fit generic panels omitted, reordered, or substituted a "
                "logical page cell"
            )
        if cell_selection is not None:
            assert cell_summary is not None
            if len(prepared_regions) != 1:
                raise ValueError("exact saved-fit focus must render one panel")
            if prepared_regions[0].fit_storage_index != cell_summary.storage_index:
                raise ValueError(
                    "focused panel and stored cell summary disagree"
                )
        projection = ("encoded", bundle, prepared_regions)
    _check(check_cancelled)
    return (
        model,
        page,
        cell_summary,
        resolved_selection,
        summary,
        projection,
    )


def reframe_fit_image_grid_front(
    front: FitGridBoardFront,
    panels: tuple[RadialGaussianImageFitPanel, ...],
    *,
    layout_generation: int,
    sequence: int,
) -> FitGridBoardFront:
    """Relayout one immutable front without copying samples or rasters."""

    if not isinstance(front, FitGridBoardFront):
        raise TypeError("saved-fit relayout source must be FitGridBoardFront")
    frame = front.frame
    panels = tuple(panels)
    requested = tuple(fit_grid_panel_id(panel) for panel in panels)
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
    artifact_identity, inputs, join_key_digest = fit_grid_join_identity(
        panels,
        requested,
    )
    if (
        old_stamp.run_id != artifact_identity
        or old_stamp.provenance_epoch_id != artifact_identity
        or old_stamp.join_key_type != "SavedFitGridJoin"
        or old_stamp.join_key_schema_fingerprint
        != SAVED_FIT_GRID_JOIN_SCHEMA_DIGEST
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
    return FitGridBoardFront(
        BoardFrame(
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
        ),
        front.surface_geometry,
        front.color_limits,
    )


def encode_loaded_fit_grid(
    loaded: object,
    *,
    artifact_identity: str,
    expected_model_identity: object,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    image_format: str,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    """Validate and encode one reloaded generic saved-Fit Figure."""

    _check(check_cancelled)
    figure, model, _page, _cell, _expected, _resolved, _summary = (
        _validated_loaded_fit_grid(
            loaded,
            artifact_identity=artifact_identity,
            page_address=page_address,
            cell_selection=cell_selection,
        )
    )
    if model.identity != expected_model_identity:
        raise ValueError("saved-fit export metadata changed within one session")
    if not isinstance(image_format, str) or not image_format.strip():
        raise ValueError("saved-fit image format must be non-empty text")
    encoded = figure.to_bytes(image_format=image_format.strip().lower())
    _check(check_cancelled)
    return encoded


def encode_fit_image_grid(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    color_limits: tuple[float, float],
    *,
    columns: int,
    expected_join_key_digest: str,
    image_format: str,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    """Encode the exact painted saved-Fit grid through the sole renderer."""

    _check(check_cancelled)
    prepared = tuple(panels)
    panel_ids = tuple(fit_grid_panel_id(panel) for panel in prepared)
    _artifact_identity, _inputs, join_key_digest = fit_grid_join_identity(
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
    if not isinstance(image_format, str) or not image_format.strip():
        raise ValueError("saved-fit image format must be non-empty text")
    from .matplotlib_render import encode_radial_gaussian_image_fit_panels

    encoded = encode_radial_gaussian_image_fit_panels(
        prepared,
        display,
        color_limits,
        image_format=image_format.strip().lower(),
        columns=columns,
    )
    _check(check_cancelled)
    return encoded


__all__ = [
    "FitGridRenderSession",
    "FitGridBoardFront",
    "FitGridEncodedProjection",
    "FitGridProjection",
    "FitGridTypedProjection",
    "SAVED_FIT_GRID_BOARD_ID",
    "SAVED_FIT_GRID_JOIN_SCHEMA_DIGEST",
    "build_fit_image_grid_front",
    "encode_loaded_fit_grid",
    "encode_fit_image_grid",
    "fit_grid_columns",
    "fit_grid_join_identity",
    "fit_grid_panel_id",
    "project_loaded_fit_grid",
    "reframe_fit_image_grid_front",
    "shared_fit_grid_color_limits",
    "validate_fit_grid_regions",
]
