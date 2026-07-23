"""Stable facade for the dependency-closed Matplotlib render owners."""

from __future__ import annotations

from ._mpl_common import release_agg_figure
from ._mpl_document import (
    encode_evaluated_figure_with_panel_regions,
    encode_evaluated_panel_with_regions,
    render_evaluated_figure,
    save_evaluated_figure,
)
from ._mpl_image import (
    ImagePanelAggRenderer,
    render_radial_gaussian_image_fit_panels,
    save_image_panel_png,
    save_radial_gaussian_image_fit_panels,
)
from ._mpl_live import SinglePanelAggRenderer
from ._mpl_pulse import (
    render_pulse_timeline_panel,
    render_pulse_timeline_png,
)

__all__ = [
    "encode_evaluated_figure_with_panel_regions",
    "encode_evaluated_panel_with_regions",
    "ImagePanelAggRenderer",
    "render_radial_gaussian_image_fit_panels",
    "render_evaluated_figure",
    "release_agg_figure",
    "save_image_panel_png",
    "save_radial_gaussian_image_fit_panels",
    "save_evaluated_figure",
    "SinglePanelAggRenderer",
]
