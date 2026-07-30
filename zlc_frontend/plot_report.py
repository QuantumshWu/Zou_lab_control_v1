"""Multi-page report composition through the canonical Plot Panel contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from zlc_storage import canonical_text

from .encoded_raster import (
    EncodedRasterDocument,
    EncodedRasterPage,
    encode_raster_buffer_png,
)
from .plot_panel import (
    FigureIntent,
    PlotPanelComposeRequest,
    PlotPanelContract,
    PlotPanelSession,
    PlotDisplayState,
    plot_panel_display_state,
)
from .plot_kind import PlotKind
from .figure_source import FigureSource
from .plot_layout import (
    PANEL_EXPORT_PIXEL_RATIO,
    optimal_grid_size_for_view,
    panel_surface_geometry,
)
from .panel_size import DEFAULT_PANEL_SIZE


@dataclass(frozen=True, slots=True)
class PlotReportPage:
    """One typed report page before backend rendering."""

    key: str
    contract: PlotPanelContract
    source: FigureSource
    display: PlotDisplayState

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", canonical_text(self.key, "report page key"))
        if not isinstance(self.contract, PlotPanelContract):
            raise TypeError("report page contract must be PlotPanelContract")
        if not isinstance(self.source, FigureSource):
            raise TypeError("report page source must be FigureSource")


@dataclass(frozen=True, slots=True)
class PlotReportDocument:
    """A report whose every plot is governed by the same frontend contract."""

    summary: str
    pages: tuple[PlotReportPage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "summary",
            canonical_text(self.summary, "plot report summary"),
        )
        pages = tuple(self.pages)
        if not pages or any(not isinstance(page, PlotReportPage) for page in pages):
            raise TypeError("plot report requires PlotReportPage values")
        if len({page.key for page in pages}) != len(pages):
            raise ValueError("plot report page keys must be unique")
        object.__setattr__(self, "pages", pages)


def plot_report_page(
    key: str,
    *,
    figure: FigureIntent,
    source: FigureSource,
    display: PlotDisplayState | None = None,
) -> PlotReportPage:
    """Create one report page using the frontend's single report surface."""

    key = canonical_text(key, "report page key")
    # Ordinary pages use the product's stock 2x2 surface.  A Grid uses the
    # exact same schema/view topology policy as an interactive Grid; export
    # resolution remains a separate terminal-render concern below.
    size_name = DEFAULT_PANEL_SIZE
    if not isinstance(figure, FigureIntent):
        raise TypeError("report page figure must be FigureIntent")
    if figure.kind is PlotKind.GRID:
        if figure.view is None:
            raise ValueError("Grid report page requires a resolved ViewSpec")
        size_name = optimal_grid_size_for_view(
            source.snapshot.block.schema,
            figure.view,
        )
    contract = PlotPanelContract(
        f"report-{key}",
        figure,
        size_name=size_name,
    )
    if display is None:
        display = plot_panel_display_state(contract, {}, revision=0)
    return PlotReportPage(key, contract, source, display)


def _one_panel_png(frame) -> bytes:
    panels = tuple(frame.panels)
    if len(panels) != 1:
        raise RuntimeError("one report page produced multiple panel rasters")
    return encode_raster_buffer_png(panels[0].raster)


def render_plot_report(
    document: PlotReportDocument,
    *,
    surface_pixel_ratio: float | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> EncodedRasterDocument:
    """Render one immutable report for the caller's runtime raster surface.

    ``PlotReportDocument`` carries scientific inputs and authored display
    intent only.  Its named logical panel size is therefore identical to an
    interactive Plot Panel.  Screen DPR is supplied at this terminal render
    boundary; a saved report defaults to the frontend's export pixel density.
    More output pixels never masquerade as a larger named panel, so fixed
    typography, margins, and data-box proportions cannot drift between a live
    card and a report.
    """

    if not isinstance(document, PlotReportDocument):
        raise TypeError("document must be PlotReportDocument")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    if surface_pixel_ratio is None:
        surface_pixel_ratio = PANEL_EXPORT_PIXEL_RATIO
    encoded: list[EncodedRasterPage] = []
    for page in document.pages:
        if checkpoint is not None:
            checkpoint()
        surface = panel_surface_geometry(
            page.contract.size_name,
            pixel_ratio=surface_pixel_ratio,
        )
        runtime_contract = replace(
            page.contract,
            size_name=surface.size_name,
            pixel_ratio=surface.pixel_ratio,
        )
        session = PlotPanelSession(runtime_contract)
        try:
            result = session.compose(
                PlotPanelComposeRequest(
                    page.source,
                    page.display,
                )
            )
            if result.frame is not None:
                payload = _one_panel_png(result.frame)
            else:
                faceted = result.faceted
                if faceted is None or faceted.overview is None:
                    raise RuntimeError("faceted report page omitted its overview raster")
                payload = encode_raster_buffer_png(faceted.overview.raster)
        finally:
            session.close()
        encoded.append(
            EncodedRasterPage(page.key, page.contract.figure.title, payload)
        )
    if checkpoint is not None:
        checkpoint()
    return EncodedRasterDocument(document.summary, tuple(encoded))


__all__ = [
    "PlotReportDocument",
    "PlotReportPage",
    "plot_report_page",
    "render_plot_report",
]
