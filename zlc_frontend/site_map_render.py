"""Generic SiteMap plot-kind protocol and worker-owned renderer."""

from __future__ import annotations

from zlc_storage import canonical_text, positive_real

from .image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
    resolve_image_color_limits,
)
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


def compose_site_map_front(
    view: SiteMapPresentation,
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
    validity, and optional boolean site state are consumed together and never
    rediscovered from array shape or independently-latest signals.
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
            site_occupied=view.site_state,
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
        view.site_state,
        view.site_validity,
        view.site_geometry_identity,
        view.view_identity,
        view.coherence_identity,
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
        view: SiteMapPresentation,
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
    "SiteMapComposer",
    "compose_site_map_front",
]
