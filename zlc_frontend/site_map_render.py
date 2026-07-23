"""Headless typed values and the single interactive SiteMap render owner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zlc_data import (
    AxisLayout,
    AxisSpec,
    CoordinateFrameId,
    IndexSelection,
    PointLayout,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    StreamGenerationId,
    resolve_selection_indices,
)
from zlc_storage import canonical_text, positive_real, sha256_text

from .figure import EvaluatedImage, EvaluatedInput
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
from .site_map import immutable_site_state


@dataclass(frozen=True)
class OccupancyCellNavigation:
    """Frozen outer-axis identity for navigating one committed occupancy artifact."""

    artifact_identity: str
    schema_fingerprint: str
    generation: StreamGenerationId
    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    cell_layout: AxisLayout

    def __post_init__(self) -> None:
        canonical_text(self.artifact_identity, "artifact_identity")
        sha256_text(self.schema_fingerprint, "schema_fingerprint")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if not isinstance(self.repeat_axis, AxisSpec):
            raise TypeError("repeat_axis must be AxisSpec")
        point_axes = tuple(self.point_axes)
        if any(not isinstance(axis, AxisSpec) for axis in point_axes):
            raise TypeError("point_axes must contain AxisSpec values")
        axes = (self.repeat_axis, *point_axes)
        if len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("occupancy navigation axes must have unique AxisId values")
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        expected_shape = tuple(axis.size for axis in point_axes)
        if self.point_layout.logical_shape != expected_shape:
            raise ValueError("point_layout logical shape differs from point axes")
        if not isinstance(self.cell_layout, AxisLayout):
            raise TypeError("cell_layout must be AxisLayout")
        expected_cell_shape = (self.repeat_axis.size, *expected_shape)
        if (
            self.cell_layout.logical_shape != expected_cell_shape
            or self.cell_layout.storage_size
            != self.repeat_axis.size * self.point_layout.storage_size
        ):
            raise ValueError("cell_layout differs from repeat and point layout")
        object.__setattr__(self, "point_axes", point_axes)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return (self.repeat_axis, *self.point_axes)

    @property
    def identity(self) -> tuple[str, str, StreamGenerationId]:
        return (
            self.artifact_identity,
            self.schema_fingerprint,
            self.generation,
        )

    @property
    def linear_cell_count(self) -> int:
        return self.cell_layout.storage_size

    def resolve_selection(
        self,
        selection: Selection | None,
    ) -> tuple[int, int, tuple[int, ...], str]:
        """Resolve one exact named cell without first/latest/reduce fallbacks."""

        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        by_axis = {} if selection is None else {
            term.axis_id: term for term in selection.terms
        }
        known = {axis.axis_id for axis in self.axes}
        if any(axis_id not in known for axis_id in by_axis):
            raise ValueError(
                "occupancy cell selection may name only repeat and point axes"
            )
        indices = []
        labels = []
        for axis in self.axes:
            term = by_axis.get(axis.axis_id)
            if term is None:
                if axis.size != 1:
                    raise ValueError(
                        f"occupancy cell requires an explicit index for axis {axis.axis_id}"
                    )
                index = 0
            else:
                if not isinstance(term, IndexSelection):
                    raise TypeError(
                        "occupancy cell selection accepts only exact IndexSelection terms"
                    )
                resolved, drop = resolve_selection_indices(axis, term)
                if not drop or len(resolved) != 1:
                    raise ValueError("occupancy cell selection must resolve one exact index")
                index = resolved.start
            coordinate = axis.coordinate_at(index)
            unit = "" if axis.unit is None else f" {axis.unit}"
            labels.append(f"{axis.name}={coordinate}{unit} [index {index}]")
            indices.append(index)
        logical_point = tuple(indices[1:])
        try:
            point_storage_index = self.point_layout.storage_index(logical_point)
        except KeyError as error:
            raise ValueError(
                f"selected logical point {logical_point} is absent from PointLayout"
            ) from error
        return (
            indices[0],
            point_storage_index,
            logical_point,
            " | ".join(labels),
        )

    def selection_for_indices(
        self,
        repeat_index: int,
        logical_point: tuple[int, ...],
    ) -> Selection:
        """Build the canonical all-axis exact selection for one logical cell."""

        logical = tuple(logical_point)
        self.point_layout.storage_index(logical)
        terms = [IndexSelection(self.repeat_axis.axis_id, repeat_index)]
        terms.extend(
            IndexSelection(axis.axis_id, index)
            for axis, index in zip(self.point_axes, logical, strict=True)
        )
        selection = Selection(tuple(terms))
        self.resolve_selection(selection)
        return selection

    def selection_at_linear(self, linear_index: int) -> Selection:
        """Map repeat-major physical order to a canonical exact selection."""

        if (
            isinstance(linear_index, bool)
            or not isinstance(linear_index, int)
            or not 0 <= linear_index < self.linear_cell_count
        ):
            raise IndexError("occupancy navigation index is out of range")
        multi = self.cell_layout.multi_index(linear_index)
        return self.selection_for_indices(multi[0], tuple(multi[1:]))

    def linear_index(self, selection: Selection) -> int:
        repeat_index, _point_storage_index, logical, _label = self.resolve_selection(
            selection
        )
        return self.cell_layout.storage_index((repeat_index, *logical))


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
    "OccupancyCellNavigation",
    "OccupancyCellView",
    "OccupancySummarySiteMapView",
    "SiteMapComposer",
    "SiteMapView",
    "compose_site_map_front",
]
