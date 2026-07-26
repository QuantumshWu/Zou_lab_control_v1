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
    PlotPanelComposeRequest,
    PlotPanelContract,
    PlotPanelSession,
    PlotDisplayState,
)
from .figure_source import FigureSource
from .panel_render import PanelProvenance
from .plot_layout import panel_surface_geometry


_REPORT_PANEL_SIZE = "8x8"


@dataclass(frozen=True, slots=True)
class PlotReportPage:
    """One typed report page before backend rendering."""

    key: str
    title: str
    contract: PlotPanelContract
    source: FigureSource
    display: PlotDisplayState
    provenance: PanelProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", canonical_text(self.key, "report page key"))
        object.__setattr__(
            self,
            "title",
            canonical_text(self.title, "report page title"),
        )
        if not isinstance(self.contract, PlotPanelContract):
            raise TypeError("report page contract must be PlotPanelContract")
        if not isinstance(self.source, FigureSource):
            raise TypeError("report page source must be FigureSource")
        if not isinstance(self.provenance, PanelProvenance):
            raise TypeError("report page provenance must be PanelProvenance")


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
    title: str,
    *,
    kind: str,
    source: FigureSource,
    display: PlotDisplayState,
    provenance: PanelProvenance,
    value_label: str,
    view=None,
) -> PlotReportPage:
    """Create one report page using the frontend's single report surface."""

    key = canonical_text(key, "report page key")
    title = canonical_text(title, "report page title")
    return PlotReportPage(
        key,
        title,
        PlotPanelContract(
            f"report-{key}",
            kind,
            title,
            str(value_label),
            size_name=_REPORT_PANEL_SIZE,
            view=view,
        ),
        source,
        display,
        provenance,
    )


def _one_panel_png(frame) -> bytes:
    panels = tuple(frame.panels)
    if len(panels) != 1:
        raise RuntimeError("one report page produced multiple panel rasters")
    return encode_raster_buffer_png(panels[0].raster)


def render_plot_report(
    document: PlotReportDocument,
    *,
    surface_pixel_ratio: float = 1.0,
    checkpoint: Callable[[], None] | None = None,
) -> EncodedRasterDocument:
    """Render one immutable report for the caller's runtime raster surface.

    ``PlotReportDocument`` carries scientific inputs and authored display
    intent only.  Screen DPR is deliberately supplied at this terminal render
    boundary, so moving a window never rebuilds calibration physics or mutates
    the document.  Every runtime page geometry is derived by the frontend's
    sole :func:`panel_surface_geometry` owner.
    """

    if not isinstance(document, PlotReportDocument):
        raise TypeError("document must be PlotReportDocument")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
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
                    page.provenance,
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
        encoded.append(EncodedRasterPage(page.key, page.title, payload))
    if checkpoint is not None:
        checkpoint()
    return EncodedRasterDocument(document.summary, tuple(encoded))


__all__ = [
    "PlotReportDocument",
    "PlotReportPage",
    "plot_report_page",
    "render_plot_report",
]
