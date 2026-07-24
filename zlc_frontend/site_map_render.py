"""Headless typed values and the single interactive SiteMap render owner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zlc_data import (
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    DatasetRevisionRef,
    IndexSelection,
    OwnedSnapshot,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    Value,
    ValueSchema,
    dataset_cell_value,
    dataset_revision_ref_to_tree,
    expand_value_validity,
    selection_to_tree,
)
from zlc_storage import canonical_digest, canonical_text, positive_real

from .figure import DatasetId, EvaluatedImage, EvaluatedInput, evaluate_axis
from .image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
    resolve_image_color_limits,
)
from .image_view import ImageViewportTransform
from .render import (
    BoardFrame,
    CoherenceStamp,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    SITE_MAP_JOIN_SCHEMA_DIGEST,
    SiteMapPanelPayload,
    SourceIdentity,
)
from .site_map import immutable_site_state, site_ring_radius


def _evaluated_image_cell(
    values: np.ndarray,
    validity: np.ndarray,
    schema: ValueSchema,
) -> tuple[EvaluatedImage, ImageViewportTransform, AxisSpec, AxisSpec]:
    if not isinstance(schema, ValueSchema):
        raise TypeError("schema must be ValueSchema")
    axes = schema.data_axes
    x_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == SPATIAL_X
    )
    y_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == SPATIAL_Y
    )
    if len(axes) != 2 or len(x_positions) != 1 or len(y_positions) != 1:
        raise ValueError(
            "site-map background requires exactly one SPATIAL_X and "
            "SPATIAL_Y data axis"
        )
    x_position, y_position = x_positions[0], y_positions[0]
    x_axis, y_axis = axes[x_position], axes[y_position]
    order_yx = (y_position, x_position)
    image = EvaluatedImage(
        evaluate_axis(x_axis, tuple(range(x_axis.size))),
        evaluate_axis(y_axis, tuple(range(y_axis.size))),
        np.transpose(values, order_yx),
        np.transpose(validity, order_yx),
        schema.value_unit,
    )
    return image, ImageViewportTransform((y_axis, x_axis)), x_axis, y_axis


def _image_cell(
    snapshot: OwnedSnapshot,
    repeat_index: int,
    point_storage_index: int,
) -> tuple[EvaluatedImage, ImageViewportTransform, AxisSpec, AxisSpec]:
    return _image_value(
        dataset_cell_value(
            snapshot.block,
            repeat_index,
            point_storage_index,
        )
    )


def _image_value(
    value: Value,
) -> tuple[EvaluatedImage, ImageViewportTransform, AxisSpec, AxisSpec]:
    """Project one already selected image value without a dataset materialization."""

    if not isinstance(value, Value):
        raise TypeError("image cell must be Value")
    return _evaluated_image_cell(
        value.values,
        expand_value_validity(value.validity, value.schema),
        value.schema,
    )


def build_calibration_site_map_view(
    snapshot: OwnedSnapshot,
    *,
    site_axis: AxisSpec,
    coordinate_frame: CoordinateFrameId,
    centers_xy: np.ndarray,
    site_validity: np.ndarray,
    calibration_identity: str,
    run_id: str,
    provenance_epoch_id: str,
    summary: str,
) -> "CalibrationSiteMapView":
    """Build the canonical calibration presentation from declared typed data."""

    schema = snapshot.block.schema
    if schema.repeat_axis.size != 1 or schema.point_layout.storage_size != 1:
        raise ValueError("calibration SiteMap background must contain one cell")
    background, viewport, x_axis, y_axis = _image_cell(snapshot, 0, 0)
    if (
        x_axis.coordinate_frame != coordinate_frame
        or y_axis.coordinate_frame != coordinate_frame
    ):
        raise ValueError(
            "calibration background and site geometry use different coordinate frames"
        )
    identity = canonical_digest(
        {
            "owner": "zlc_frontend.calibration-site-map-view",
            "source": dataset_revision_ref_to_tree(snapshot.ref),
            "calibration_identity": calibration_identity,
        }
    )
    return CalibrationSiteMapView(
        background=background,
        background_input=EvaluatedInput(
            DatasetId(f"calibration-reference-{identity}"),
            snapshot.ref,
        ),
        calibration_input=EvaluatedInput(
            DatasetId(f"calibration-sites-{identity}"),
            snapshot.ref,
        ),
        home_viewport=viewport,
        site_axis=site_axis,
        coordinate_frame=coordinate_frame,
        centers_xy=centers_xy,
        site_radius=site_ring_radius(centers_xy),
        site_validity=site_validity,
        calibration_identity=calibration_identity,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=summary,
    )


def build_occupancy_cell_view(
    background_value: Value,
    background_ref: DatasetRevisionRef,
    occupied_value: Value,
    occupancy_ref: DatasetRevisionRef,
    selection: Selection,
    *,
    site_axis: AxisSpec,
    coordinate_frame: CoordinateFrameId,
    centers_xy: np.ndarray,
    calibration_site_validity: np.ndarray,
    calibration_identity: str,
    run_id: str,
    provenance_epoch_id: str,
    summary: str,
) -> "OccupancyCellView":
    """Build one same-cell Camera/SITE view from exact typed cell values.

    Domain admission and cell addressing happen before this presentation
    boundary.  In particular, the Camera input is one chunk-backed ``Value``;
    callers never need to materialize the complete capture artifact merely to
    display one physical cell.
    """

    if not isinstance(background_value, Value) or not isinstance(
        occupied_value,
        Value,
    ):
        raise TypeError("background_value and occupied_value must be Value")
    if not isinstance(background_ref, DatasetRevisionRef) or not isinstance(
        occupancy_ref,
        DatasetRevisionRef,
    ):
        raise TypeError("background_ref and occupancy_ref must be DatasetRevisionRef")
    if not isinstance(selection, Selection):
        raise TypeError("selection must be Selection")
    if background_ref.revision != occupancy_ref.revision:
        raise ValueError("occupancy revision differs from its Camera source")
    background, viewport, x_axis, y_axis = _image_value(background_value)
    if (
        x_axis.coordinate_frame != coordinate_frame
        or y_axis.coordinate_frame != coordinate_frame
    ):
        raise ValueError(
            "Camera background and calibration geometry use different coordinate frames"
        )
    data_axes = occupied_value.schema.data_axes
    if len(data_axes) != 1 or data_axes[0] != site_axis or site_axis.role != SITE:
        raise ValueError("occupancy data must follow the calibration SITE axis")
    if occupied_value.schema.dtype != np.dtype(bool):
        raise TypeError("occupancy cell values must be boolean")
    if not isinstance(occupied_value.validity, ComponentValidity) or (
        occupied_value.validity.axis_ids != (site_axis.axis_id,)
    ):
        raise ValueError("occupancy cell validity must name exactly the SITE axis")
    occupied_values = np.asarray(occupied_value.values, dtype=np.bool_)
    site_validity = np.asarray(occupied_value.validity.mask, dtype=np.bool_)
    admitted_sites = np.asarray(calibration_site_validity, dtype=np.bool_)
    if admitted_sites.shape != site_validity.shape:
        raise ValueError("calibration validity differs from the occupancy SITE axis")
    if np.any(site_validity & ~admitted_sites):
        raise ValueError("occupancy marks a calibration-invalid site as valid")
    identity = canonical_digest(
        {
            "owner": "zlc_frontend.occupancy-cell-view",
            "source": dataset_revision_ref_to_tree(background_ref),
            "occupied": dataset_revision_ref_to_tree(occupancy_ref),
            "calibration_identity": calibration_identity,
            "selection": selection_to_tree(selection),
        }
    )
    return OccupancyCellView(
        background=background,
        background_input=EvaluatedInput(
            DatasetId(f"occupancy-frame-{identity}"),
            background_ref,
        ),
        occupancy_input=EvaluatedInput(
            DatasetId(f"occupancy-sites-{identity}"),
            occupancy_ref,
        ),
        home_viewport=viewport,
        site_axis=site_axis,
        coordinate_frame=coordinate_frame,
        centers_xy=centers_xy,
        site_radius=site_ring_radius(centers_xy),
        occupied=occupied_values,
        site_validity=site_validity,
        calibration_identity=calibration_identity,
        cell_identity=identity,
        cell_selection=selection,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=summary,
    )

@dataclass(frozen=True, eq=False)
class OccupancyCellView:
    """Self-contained physical facts for one exact ``(repeat, point)`` cell."""

    background: EvaluatedImage
    background_input: EvaluatedInput
    occupancy_input: EvaluatedInput
    home_viewport: ImageViewportTransform
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    site_radius: float
    occupied: np.ndarray
    site_validity: np.ndarray
    calibration_identity: str
    cell_identity: str
    cell_selection: Selection
    run_id: str
    provenance_epoch_id: str
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.background, EvaluatedImage):
            raise TypeError("background must be EvaluatedImage")
        if not isinstance(self.background_input, EvaluatedInput):
            raise TypeError("background_input must be EvaluatedInput")
        if not isinstance(self.occupancy_input, EvaluatedInput):
            raise TypeError("occupancy_input must be EvaluatedInput")
        if self.background_input.dataset_id == self.occupancy_input.dataset_id:
            raise ValueError("background and occupancy inputs require distinct DatasetIds")
        if self.background_input.ref.revision != self.occupancy_input.ref.revision:
            raise ValueError("background and occupancy inputs must name one exact revision")
        if not isinstance(self.home_viewport, ImageViewportTransform):
            raise TypeError("home_viewport must be ImageViewportTransform")
        if self.home_viewport.viewport_revision != 0:
            raise ValueError("home_viewport must begin in authored revision zero")
        for evaluated, declared, role, name in (
            (self.background.x_axis, self.home_viewport.x_axis, SPATIAL_X, "x"),
            (self.background.y_axis, self.home_viewport.y_axis, SPATIAL_Y, "y"),
        ):
            if declared.role != role or evaluated.role != role:
                raise ValueError(f"occupancy background {name} axis has the wrong role")
            if (
                evaluated.axis_id != declared.axis_id
                or evaluated.name != declared.name
                or evaluated.unit != declared.unit
                or evaluated.indices != tuple(range(declared.size))
                or evaluated.coordinates
                != tuple(declared.coordinate_at(index) for index in range(declared.size))
            ):
                raise ValueError(
                    f"occupancy background {name} axis differs from its declared viewport"
                )
        if self.background.values.dtype.kind == "c":
            raise TypeError("occupancy background must contain real values")
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        if self.home_viewport.coordinate_frame != self.coordinate_frame:
            raise ValueError("background and site geometry coordinate frames differ")
        sites = self.site_axis.size
        centers, occupied, site_validity = immutable_site_state(
            self.centers_xy,
            self.occupied,
            self.site_validity,
            site_count=sites,
        )
        radius = positive_real(self.site_radius, "site_radius")
        if not isinstance(self.cell_selection, Selection):
            raise TypeError("cell_selection must be Selection")
        if any(
            not isinstance(term, IndexSelection)
            for term in self.cell_selection.terms
        ):
            raise TypeError("cell_selection must contain only exact IndexSelection terms")
        for value, name in (
            (self.calibration_identity, "calibration_identity"),
            (self.cell_identity, "cell_identity"),
            (self.run_id, "run_id"),
            (self.provenance_epoch_id, "provenance_epoch_id"),
        ):
            canonical_text(value, name)
        canonical_text(self.summary, "occupancy cell summary")
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "site_radius", radius)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "site_validity", site_validity)

    @property
    def site_state_input(self) -> EvaluatedInput:
        return self.occupancy_input

    @property
    def view_identity(self) -> str:
        return self.cell_identity

    @property
    def site_occupancy(self) -> np.ndarray:
        return self.occupied

    @property
    def valid_site_count(self) -> int:
        """Number of physically admitted SITE components in this view."""

        return int(np.count_nonzero(self.site_validity))

    @property
    def occupied_site_count(self) -> int:
        """Number of occupied components among the admitted SITE components."""

        return int(np.count_nonzero(self.occupied & self.site_validity))

    @property
    def invalid_site_count(self) -> int:
        return self.site_axis.size - self.valid_site_count

    @property
    def site_count_summary(self) -> str:
        """Canonical display text for component validity and occupancy counts."""

        return (
            f"occupied={self.occupied_site_count}/{self.valid_site_count} valid sites | "
            f"invalid={self.invalid_site_count}"
        )

    @property
    def presentation_kind(self) -> str:
        return "ExactOccupancyCell"


@dataclass(frozen=True, eq=False)
class CalibrationSiteMapView:
    """Committed calibration reference image with its exact calibrated sites.

    This is not occupancy.  It contains no occupied/empty state and therefore
    cannot be mistaken for a classified measurement.  The shared SiteMap
    renderer paints calibrated/invalid rings over the calibration-owned
    reference average.
    """

    background: EvaluatedImage
    background_input: EvaluatedInput
    calibration_input: EvaluatedInput
    home_viewport: ImageViewportTransform
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    site_radius: float
    site_validity: np.ndarray
    calibration_identity: str
    run_id: str
    provenance_epoch_id: str
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.background, EvaluatedImage):
            raise TypeError("background must be EvaluatedImage")
        if not isinstance(self.background_input, EvaluatedInput):
            raise TypeError("background_input must be EvaluatedInput")
        if not isinstance(self.calibration_input, EvaluatedInput):
            raise TypeError("calibration_input must be EvaluatedInput")
        if self.background_input.dataset_id == self.calibration_input.dataset_id:
            raise ValueError(
                "calibration background and site state require distinct DatasetIds"
            )
        if self.background_input.ref != self.calibration_input.ref:
            raise ValueError(
                "calibration background and site state must name one exact revision"
            )
        if not isinstance(self.home_viewport, ImageViewportTransform):
            raise TypeError("home_viewport must be ImageViewportTransform")
        if self.home_viewport.viewport_revision != 0:
            raise ValueError("home_viewport must begin in authored revision zero")
        for evaluated, declared, role, name in (
            (self.background.x_axis, self.home_viewport.x_axis, SPATIAL_X, "x"),
            (self.background.y_axis, self.home_viewport.y_axis, SPATIAL_Y, "y"),
        ):
            if declared.role != role or evaluated.role != role:
                raise ValueError(
                    f"calibration background {name} axis has the wrong role"
                )
            if (
                evaluated.axis_id != declared.axis_id
                or evaluated.name != declared.name
                or evaluated.unit != declared.unit
                or evaluated.indices != tuple(range(declared.size))
                or evaluated.coordinates
                != tuple(
                    declared.coordinate_at(index)
                    for index in range(declared.size)
                )
            ):
                raise ValueError(
                    f"calibration background {name} axis differs from its viewport"
                )
        if self.background.values.dtype.kind == "c":
            raise TypeError("calibration background must contain real values")
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        if self.home_viewport.coordinate_frame != self.coordinate_frame:
            raise ValueError("background and site geometry coordinate frames differ")
        centers, _empty, validity = immutable_site_state(
            self.centers_xy,
            np.zeros(self.site_axis.size, dtype=np.bool_),
            self.site_validity,
            site_count=self.site_axis.size,
        )
        radius = positive_real(self.site_radius, "site_radius")
        for value, name in (
            (self.calibration_identity, "calibration_identity"),
            (self.run_id, "run_id"),
            (self.provenance_epoch_id, "provenance_epoch_id"),
        ):
            canonical_text(value, name)
        canonical_text(self.summary, "calibration site-map summary")
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "site_radius", radius)
        object.__setattr__(self, "site_validity", validity)

    @property
    def site_state_input(self) -> EvaluatedInput:
        return self.calibration_input

    @property
    def view_identity(self) -> str:
        return self.calibration_identity

    @property
    def site_occupancy(self) -> None:
        return None

    @property
    def presentation_kind(self) -> str:
        return "CalibrationSiteMap"


@dataclass(frozen=True, eq=False)
class OccupancySummarySiteMapView(CalibrationSiteMapView):
    """Exact singleton occupancy over an explicitly calibration-owned background.

    The background is the calibration reference average, not the camera frame
    that produced ``occupied``.  Keeping this as a distinct type prevents a
    summary visualization from satisfying any same-shot ``OccupancyCellView``
    contract.
    """

    occupancy_input: EvaluatedInput
    occupied: np.ndarray
    occupancy_identity: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.occupancy_input, EvaluatedInput):
            raise TypeError("occupancy_input must be EvaluatedInput")
        if self.occupancy_input.dataset_id in {
            self.background_input.dataset_id,
            self.calibration_input.dataset_id,
        }:
            raise ValueError(
                "summary occupancy requires its own exact DatasetId"
            )
        centers, occupied, validity = immutable_site_state(
            self.centers_xy,
            self.occupied,
            self.site_validity,
            site_count=self.site_axis.size,
        )
        canonical_text(self.occupancy_identity, "occupancy_identity")
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "site_validity", validity)

    @property
    def site_state_input(self) -> EvaluatedInput:
        return self.occupancy_input

    @property
    def view_identity(self) -> str:
        return self.occupancy_identity

    @property
    def site_occupancy(self) -> np.ndarray:
        return self.occupied

    @property
    def presentation_kind(self) -> str:
        return "OccupancySummaryOnCalibrationBackground"


SiteMapView = (
    OccupancyCellView
    | CalibrationSiteMapView
    | OccupancySummarySiteMapView
)


def compose_site_map_front(
    view: SiteMapView,
    display: ImageDisplayState,
    *,
    panel_id: str,
    board_id: str,
    sequence: int,
    selection_revision: int = 0,
    current_color_limits: tuple[float, float] | None = None,
    previous_relim_mode=None,
    renderer=None,
    size: tuple[int, int] = (480, 420),
    size_name: str | None = None,
    title: str = "Site map",
    value_label: str = "Counts",
) -> tuple[BoardFrame, tuple[float, float]]:
    """Rasterize one typed physical SiteMap through the shared render owner.

    The caller chooses only hosting identity and display continuity.  The
    physical join is already closed by the view: background, site coordinates,
    validity, and optional occupancy are consumed together and never
    rediscovered from array shape or independently-latest signals.
    """

    if not isinstance(
        view,
        (
            OccupancyCellView,
            CalibrationSiteMapView,
            OccupancySummarySiteMapView,
        ),
    ):
        raise TypeError("view must be a typed SiteMap view")
    if not isinstance(display, ImageDisplayState):
        raise TypeError("display must be ImageDisplayState")
    panel_id = canonical_text(panel_id, "panel_id")
    board_id = canonical_text(board_id, "board_id")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if (
        isinstance(selection_revision, bool)
        or not isinstance(selection_revision, int)
        or selection_revision < 0
    ):
        raise ValueError("selection_revision must be a non-negative integer")

    viewport = image_viewport_for_display_state(display, view.home_viewport)
    data_range, effective_limits = resolve_image_color_limits(
        view.background,
        display,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
    )
    owned_renderer = renderer is None
    if renderer is None:
        from .matplotlib_render import ImagePanelAggRenderer

        renderer = ImagePanelAggRenderer(
            width=size[0],
            height=size[1],
            size_name=size_name,
            site_map=True,
        )
    try:
        raster, raster_geometry = renderer.render(
            view.background,
            viewport,
            display,
            color_limits=effective_limits,
            data_range=data_range,
            title=title,
            value_label=value_label,
            distribution_guides=False,
            distribution_bins=40,
            distribution_identity=view.background_input.ref,
            site_centers_xy=view.centers_xy,
            site_radius=view.site_radius,
            site_occupied=view.site_occupancy,
            site_validity=view.site_validity,
            colorbar_endpoints=False,
        )
    finally:
        if owned_renderer:
            renderer.close()
    background = ImagePanelPayload(
        view.background,
        view.background_input,
        viewport,
        data_range,
        display.colormap,
        effective_limits,
        raster_geometry,
    )
    payload = SiteMapPanelPayload(
        background,
        view.site_state_input,
        view.site_axis,
        view.coordinate_frame,
        view.centers_xy,
        view.site_occupancy,
        view.site_validity,
        view.calibration_identity,
        view.view_identity,
    )
    presentation = PanelPresentationIdentity(
        panel_id,
        f"site-map:{view.presentation_kind}:{view.view_identity}",
        0,
        selection_revision,
        display.revision,
    )
    stamp = CoherenceStamp(
        view.run_id,
        view.provenance_epoch_id,
        view.presentation_kind,
        SITE_MAP_JOIN_SCHEMA_DIGEST,
        payload.join_key_digest,
        (view.background_input, view.site_state_input),
        (presentation,),
    )
    ref = view.site_state_input.ref
    source = SourceIdentity(
        view.site_state_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )
    panel = PanelFrame(
        panel_id,
        panel_id,
        source,
        stamp,
        raster,
        payload,
    )
    return BoardFrame(board_id, 0, sequence, (panel,)), effective_limits


class SiteMapComposer:
    """Worker-owned display continuity for any typed SiteMap panel."""

    __slots__ = (
        "_board_id",
        "_color_limits",
        "_panel_id",
        "_pixel_ratio",
        "_previous_relim_mode",
        "_renderer",
        "_sequence",
        "_size",
        "_size_name",
        "_title",
        "_value_label",
    )

    def __init__(
        self,
        panel_id: str,
        *,
        board_id: str | None = None,
        size: tuple[int, int] = (480, 420),
        size_name: str | None = None,
        pixel_ratio: float = 1.0,
        title: str = "Site map",
        value_label: str = "Counts",
    ) -> None:
        self._panel_id = canonical_text(panel_id, "panel_id")
        self._board_id = canonical_text(
            board_id or f"site-map-board-{panel_id}",
            "board_id",
        )
        self._color_limits = None
        self._previous_relim_mode = None
        self._sequence = 0
        self._renderer = None
        self._size = (int(size[0]), int(size[1]))
        self._size_name = None if size_name is None else str(size_name)
        self._pixel_ratio = positive_real(pixel_ratio, "pixel_ratio")
        self._title = str(title)
        self._value_label = str(value_label)

    def compose(
        self,
        view: SiteMapView,
        *,
        display: ImageDisplayState,
    ) -> BoardFrame:
        self._sequence += 1
        if self._renderer is None:
            from .matplotlib_render import ImagePanelAggRenderer
            from .plot_layout import LIVE_PANEL_DPI

            self._renderer = ImagePanelAggRenderer(
                width=self._size[0],
                height=self._size[1],
                dpi=LIVE_PANEL_DPI * self._pixel_ratio,
                size_name=self._size_name,
                site_map=True,
            )
        frame, effective_limits = compose_site_map_front(
            view,
            display,
            panel_id=self._panel_id,
            board_id=self._board_id,
            sequence=self._sequence,
            current_color_limits=self._color_limits,
            previous_relim_mode=self._previous_relim_mode,
            renderer=self._renderer,
            size=self._size,
            size_name=self._size_name,
            title=self._title,
            value_label=self._value_label,
        )
        self._color_limits = effective_limits
        self._previous_relim_mode = display.relim_mode
        return frame

    def close(self) -> None:
        """Release display continuity when the panel source is replaced."""

        self._color_limits = None
        self._previous_relim_mode = None
        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            renderer.close()


__all__ = [
    "CalibrationSiteMapView",
    "OccupancyCellView",
    "OccupancySummarySiteMapView",
    "SiteMapComposer",
    "SiteMapView",
    "build_calibration_site_map_view",
    "build_occupancy_cell_view",
    "compose_site_map_front",
]
