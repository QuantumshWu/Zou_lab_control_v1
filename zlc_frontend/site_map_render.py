"""Generic SiteMap plot-kind protocol and worker-owned renderer."""

from __future__ import annotations

from zlc_storage import canonical_text

from .image_display import (
    ImageDisplayState,
    evaluated_image_data_range,
    image_viewport_for_display_state,
    resolve_image_color_limits_from_range,
)
from .figure import EvaluatedProjectionIdentity
from .plot_layout import PanelSurfaceGeometry
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
from .site_map import SiteMapPresentation


def _compose_site_map_front(
    view: SiteMapPresentation,
    display: ImageDisplayState,
    *,
    panel_id: str,
    board_id: str,
    sequence: int,
    surface_revision: int,
    selection_revision: int,
    data_range: tuple[float, float] | None,
    current_color_limits: tuple[float, float] | None = None,
    previous_relim_mode=None,
    renderer,
    title: str = "Site map",
    value_label: str = "Counts",
) -> tuple[BoardFrame, tuple[float, float]]:
    """Rasterize one typed SiteMap through its persistent frontend owner.

    This is deliberately private: renderer lifetime, data-range continuity,
    and sequence ownership are inseparable and belong to :class:`SiteMapComposer`.
    The physical join is already closed by the view, so rendering never
    rediscovers background/site relations from shapes or latest values.
    """

    if not isinstance(view, SiteMapPresentation):
        raise TypeError("view must be a typed SiteMap view")
    if not isinstance(display, ImageDisplayState):
        raise TypeError("display must be ImageDisplayState")
    panel_id = canonical_text(panel_id, "panel_id")
    board_id = canonical_text(board_id, "board_id")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if (
        isinstance(surface_revision, bool)
        or not isinstance(surface_revision, int)
        or surface_revision < 0
    ):
        raise ValueError("surface_revision must be a non-negative integer")
    if (
        isinstance(selection_revision, bool)
        or not isinstance(selection_revision, int)
        or selection_revision < 0
    ):
        raise ValueError("selection_revision must be a non-negative integer")

    viewport = image_viewport_for_display_state(display, view.home_viewport)
    data_range, effective_limits = resolve_image_color_limits_from_range(
        data_range,
        display,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
    )
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
        projection_identity=EvaluatedProjectionIdentity(
            view.presentation_kind,
            0,
            view.background_input,
            "site-map-background",
            (),
            (),
            (),
            view.background,
        ),
        site_centers_xy=view.centers_xy,
        site_radius=view.site_radius,
        site_occupied=view.site_state,
        site_validity=view.site_validity,
        colorbar_endpoints=False,
    )
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
        view.site_state,
        view.site_validity,
        view.site_geometry_identity,
        view.view_identity,
        view.coherence_identity,
    )
    presentation = PanelPresentationIdentity(
        panel_id,
        f"site-map:{view.presentation_kind}:{view.view_identity}",
        surface_revision,
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
        "_previous_relim_mode",
        "_continuity_lineage",
        "_continuity_revisions",
        "_range_identity",
        "_range_value",
        "_renderer",
        "_sequence",
        "_surface_geometry",
        "_surface_revision",
        "_title",
        "_value_label",
    )

    def __init__(
        self,
        panel_id: str,
        *,
        surface_geometry: PanelSurfaceGeometry,
        board_id: str | None = None,
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
        self._continuity_lineage = None
        self._continuity_revisions = None
        self._range_identity = None
        self._range_value = None
        self._sequence = 0
        self._renderer = None
        if not isinstance(surface_geometry, PanelSurfaceGeometry):
            raise TypeError("surface_geometry must be PanelSurfaceGeometry")
        self._surface_geometry = surface_geometry
        self._surface_revision = 0
        self._title = str(title)
        self._value_label = str(value_label)

    def compose(
        self,
        view: SiteMapPresentation,
        *,
        display: ImageDisplayState,
        selection_revision: int = 0,
        surface_geometry: PanelSurfaceGeometry | None = None,
        surface_revision: int = 0,
    ) -> BoardFrame:
        if not isinstance(view, SiteMapPresentation):
            raise TypeError("view must be a typed SiteMap view")
        if (
            isinstance(selection_revision, bool)
            or not isinstance(selection_revision, int)
            or selection_revision < 0
        ):
            raise ValueError("selection_revision must be a non-negative integer")
        if surface_geometry is None:
            surface_geometry = self._surface_geometry
        if not isinstance(surface_geometry, PanelSurfaceGeometry):
            raise TypeError("surface_geometry must be PanelSurfaceGeometry")
        if (
            isinstance(surface_revision, bool)
            or not isinstance(surface_revision, int)
            or surface_revision < self._surface_revision
        ):
            raise ValueError("surface_revision must be monotonic")
        if (
            surface_geometry != self._surface_geometry
            and surface_revision == self._surface_revision
        ):
            raise ValueError("changed surface geometry requires a new revision")
        if surface_geometry != self._surface_geometry:
            renderer, self._renderer = self._renderer, None
            if renderer is not None:
                renderer.close()
            self._surface_geometry = surface_geometry
        self._surface_revision = surface_revision
        background_input = view.background_input
        state_input = view.site_state_input
        range_identity = (
            background_input,
            state_input,
            view.view_identity,
        )
        lineage = (
            background_input.dataset_id,
            background_input.ref.block_id,
            background_input.ref.stream_generation,
            background_input.ref.schema_fingerprint,
            state_input.dataset_id,
            state_input.ref.block_id,
            state_input.ref.stream_generation,
            state_input.ref.schema_fingerprint,
            view.presentation_kind,
            view.cell_selection,
            view.site_geometry_identity,
            view.coordinate_frame,
            view.home_viewport.x_axis.axis_id,
            view.home_viewport.y_axis.axis_id,
            view.site_axis.axis_id,
        )
        revisions = (
            background_input.ref.revision.value,
            state_input.ref.revision.value,
        )
        if range_identity != self._range_identity:
            old_revisions = self._continuity_revisions
            same_advancing_lineage = (
                lineage == self._continuity_lineage
                and old_revisions is not None
                and all(
                    current >= previous
                    for current, previous in zip(
                        revisions,
                        old_revisions,
                        strict=True,
                    )
                )
                and revisions != old_revisions
            )
            self._range_identity = range_identity
            self._range_value = evaluated_image_data_range((view.background,))
            # NORMAL deadband continuity follows one monotonically advancing
            # stream/view lineage.  Selecting another frozen cell changes the
            # exact projection without advancing its refs and therefore starts
            # from authored relim state; a newer revision keeps continuity.
            if not same_advancing_lineage:
                self._color_limits = None
                self._previous_relim_mode = None
            self._continuity_lineage = lineage
            self._continuity_revisions = revisions
        self._sequence += 1
        if self._renderer is None:
            from .matplotlib_render import ImagePanelAggRenderer

            geometry = self._surface_geometry
            self._renderer = ImagePanelAggRenderer(
                width=geometry.raster_size[0],
                height=geometry.raster_size[1],
                dpi=geometry.dpi,
                size_name=geometry.size_name,
                site_map=True,
            )
        try:
            frame, effective_limits = _compose_site_map_front(
                view,
                display,
                panel_id=self._panel_id,
                board_id=self._board_id,
                sequence=self._sequence,
                surface_revision=self._surface_revision,
                selection_revision=selection_revision,
                data_range=self._range_value,
                current_color_limits=self._color_limits,
                previous_relim_mode=self._previous_relim_mode,
                renderer=self._renderer,
                title=self._title,
                value_label=self._value_label,
            )
        except BaseException:
            # A failed Matplotlib mutation leaves no reusable artist graph.
            self.close()
            raise
        self._color_limits = effective_limits
        self._previous_relim_mode = display.relim_mode
        return frame

    def close(self) -> None:
        """Release display continuity when the panel source is replaced."""

        self._color_limits = None
        self._previous_relim_mode = None
        self._continuity_lineage = None
        self._continuity_revisions = None
        self._range_identity = None
        self._range_value = None
        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            renderer.close()

__all__ = [
    "SiteMapComposer",
]
