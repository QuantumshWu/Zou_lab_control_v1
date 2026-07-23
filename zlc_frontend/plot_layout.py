"""Pure plot geometry shared by the Agg and Qt raster surfaces.

The established frontend layout has one fixed outer-margin/data-box model.
Matplotlib consumes it as an axes locator while Qt maps the same normalized
boxes into a raster-board cell.  This module deliberately imports neither
Matplotlib nor Qt so neither surface has to copy the geometry constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


DESIGN_DPI = 300
PANEL_DISPLAY_SCALE = 0.7
LIVE_PANEL_DPI = DESIGN_DPI * PANEL_DISPLAY_SCALE
STOCK_DATA_PX = (480, 360)
STOCK_MARGINS_PX = (110, 110, 100, 40)

TITLE_SLOT_PX = 70
PANEL_UNIT_PX = (180, 240)
PANEL_MARGINS_PX = (STOCK_MARGINS_PX[0], 96, 80, TITLE_SLOT_PX)
PULSE_LEFT_MARGIN_PX = 122

_IMAGE_WIDTHS = (0.75, 0.10, 0.10)
_IMAGE_GAPS = (0.025, 0.025)

SITE_GRID_MAX_COLUMNS = 7
_GRID_DOUBLE_ROWS_THRESHOLD = 4
_GRID_DOUBLE_COLUMNS_THRESHOLD = 5


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """One positive top-origin rectangle in normalized panel coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = tuple(
            float(value)
            for value in (self.left, self.top, self.right, self.bottom)
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("normalized plot box values must be finite")
        left, top, right, bottom = values
        if not (0.0 <= left < right <= 1.0):
            raise ValueError("normalized plot box requires 0 <= left < right <= 1")
        if not (0.0 <= top < bottom <= 1.0):
            raise ValueError("normalized plot box requires 0 <= top < bottom <= 1")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "bottom", bottom)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def matplotlib_bounds(self) -> tuple[float, float, float, float]:
        """Return Matplotlib's bottom-origin ``left,bottom,width,height``."""

        return (self.left, 1.0 - self.bottom, self.width, self.height)


@dataclass(frozen=True, slots=True)
class ImagePanelLayout:
    """The established image | distribution | colorbar split inside one data box."""

    data: NormalizedBox
    image: NormalizedBox
    distribution: NormalizedBox
    colorbar: NormalizedBox


@dataclass(frozen=True, slots=True)
class RollingPanelLayout:
    """The established history | side-distribution split."""

    data: NormalizedBox
    history: NormalizedBox
    distribution: NormalizedBox


def panel_margins_px(kind: str = "default") -> tuple[int, int, int, int]:
    """Return the only fixed ``left,right,bottom,top`` margin tuple."""

    left, right, bottom, top = PANEL_MARGINS_PX
    if str(kind).strip().lower() == "pulse":
        left = PULSE_LEFT_MARGIN_PX
    return (left, right, bottom, top)


def _panel_geometry_px(
    size: str,
    *,
    kind: str,
) -> tuple[int, int, int, int, int, int]:
    from zlc_data.panel_size import panel_size_cells

    rows, columns = panel_size_cells(size)
    left, right, bottom, top = panel_margins_px(kind)
    return (
        columns * PANEL_UNIT_PX[1],
        rows * PANEL_UNIT_PX[0],
        left,
        right,
        bottom,
        top,
    )


def panel_figure_size_inches(
    size: str = "2x2",
    *,
    kind: str = "default",
) -> tuple[float, float]:
    """Figure size derived from the fixed data box and margins."""

    data_width, data_height, left, right, bottom, top = _panel_geometry_px(
        size,
        kind=kind,
    )
    return (
        (left + data_width + right) / DESIGN_DPI,
        (bottom + data_height + top) / DESIGN_DPI,
    )


def panel_display_size(
    size: str = "2x2",
    *,
    kind: str = "default",
) -> tuple[int, int]:
    """Logical Qt size paired with :func:`panel_figure_size_inches`."""

    width, height = panel_figure_size_inches(size, kind=kind)
    return (
        round(width * DESIGN_DPI * PANEL_DISPLAY_SCALE),
        round(height * DESIGN_DPI * PANEL_DISPLAY_SCALE),
    )


def panel_axes_bounds(
    size: str = "2x2",
    *,
    kind: str = "default",
) -> tuple[float, float, float, float]:
    """Fixed Matplotlib axes box for a named panel size."""

    data_width, data_height, left, right, bottom, top = _panel_geometry_px(
        size,
        kind=kind,
    )
    figure_width = left + data_width + right
    figure_height = bottom + data_height + top
    return (
        left / figure_width,
        bottom / figure_height,
        data_width / figure_width,
        data_height / figure_height,
    )


def panel_data_box(
    size: str = "2x2",
    *,
    kind: str = "default",
) -> NormalizedBox:
    """Return Main's exact design-space DATA box for a named preset."""

    data_width, data_height, left, right, bottom, top = _panel_geometry_px(
        size,
        kind=kind,
    )
    figure_width = left + data_width + right
    figure_height = top + data_height + bottom
    return NormalizedBox(
        left / figure_width,
        top / figure_height,
        (left + data_width) / figure_width,
        (top + data_height) / figure_height,
    )


def panel_data_box_for_raster(
    width: int,
    height: int,
    *,
    kind: str = "default",
) -> NormalizedBox:
    """Fixed top-origin data box for an already-sized logical raster.

    Panel presets grow the data region while keeping the typography margins
    fixed.  The raster renderer therefore applies the established display
    scale to those margins rather than asking a layout engine to infer them
    from the current labels.
    """

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("panel raster dimensions must be positive")
    margin_left, margin_right, margin_bottom, margin_top = panel_margins_px(kind)
    margin_scale = min(
        PANEL_DISPLAY_SCALE,
        max(0.0, (width - 1.0) / (margin_left + margin_right)),
        max(0.0, (height - 1.0) / (margin_top + margin_bottom)),
    )
    left, right, bottom, top = (
        margin * margin_scale
        for margin in (
            margin_left,
            margin_right,
            margin_bottom,
            margin_top,
        )
    )
    # Main's Divider fixes the DATA box and lets the final subpixel figure
    # rounding land in the trailing margin.  Recomputing the data width as
    # ``raster - both margins`` steals 0.2 px from every standard-width panel
    # (686 design px × .7 rounds to 480), which then shifts all three image
    # bands.  Recover the exact half-unit data box whenever this raster is one
    # of the declared panel geometries; arbitrary export sizes keep the
    # residual box.
    available_width = width - left - right
    available_height = height - top - bottom
    unit_width = PANEL_UNIT_PX[1] * margin_scale
    unit_height = PANEL_UNIT_PX[0] * margin_scale

    def fixed_panel_span(available: float, unit: float) -> float:
        if unit <= 0.0:
            return available
        units = max(1, round(available / unit))
        declared = units * unit
        return declared if abs(declared - available) <= 1.0 else available

    data_width = fixed_panel_span(available_width, unit_width)
    data_height = fixed_panel_span(available_height, unit_height)
    return NormalizedBox(
        left / width,
        top / height,
        (left + data_width) / width,
        (top + data_height) / height,
    )


def grid_shape_for(
    cell_count: int,
    *,
    max_columns: int = SITE_GRID_MAX_COLUMNS,
) -> tuple[int, int]:
    """Return main's near-square, column-capped grid for ``cell_count``."""

    cell_count = max(1, int(cell_count))
    max_columns = max(1, int(max_columns))
    columns = min(max_columns, int(math.ceil(math.sqrt(cell_count))))
    return int(math.ceil(cell_count / columns)), columns


def grid_shape_for_aspect(
    cell_count: int,
    cell_aspect: float,
    region_px: tuple[int, int],
    *,
    max_columns: int = SITE_GRID_MAX_COLUMNS,
) -> tuple[int, int]:
    """Return main's best packing for fixed-aspect image thumbnails."""

    cell_count = max(1, int(cell_count))
    max_columns = max(1, int(max_columns))
    aspect = float(cell_aspect)
    if not math.isfinite(aspect) or aspect <= 0.0:
        aspect = 1.0
    region_width, region_height = (
        max(1.0, float(value)) for value in region_px
    )
    best_key = None
    best = (1, cell_count)
    for columns in range(1, min(max_columns, cell_count) + 1):
        rows = int(math.ceil(cell_count / columns))
        cell_width = region_width / columns
        cell_height = region_height / rows
        image_height = min(cell_height, cell_width / aspect)
        key = (
            image_height,
            -(rows * columns),
            -abs(rows - columns),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (rows, columns)
    return best


def optimal_grid_size(rows: int, columns: int) -> str:
    """Return main's independent per-axis default size recommendation."""

    rows = max(1, int(rows))
    columns = max(1, int(columns))
    row_units = 4 if rows > _GRID_DOUBLE_ROWS_THRESHOLD else 2
    column_units = 4 if columns > _GRID_DOUBLE_COLUMNS_THRESHOLD else 2
    return f"{row_units}x{column_units}"


def site_grid_geometry(
    size: str,
    recommended: str,
    *,
    tick_font_points: float,
) -> tuple[
    tuple[int, int],
    int,
    int,
    tuple[int, int, int, int],
    float,
]:
    """Return main's fixed data box, gutters, margins, and cell-font scale.

    Values remain in design pixels.  A live raster uses the resulting
    normalized axes boxes, so the one ``PANEL_DISPLAY_SCALE`` conversion that
    produced its Qt size also scales every margin and gutter.
    """

    from zlc_data.panel_size import panel_size_cells

    rows, columns = panel_size_cells(size)
    recommended_rows, recommended_columns = panel_size_cells(recommended)
    font_scale = (
        0.5
        if rows < recommended_rows or columns < recommended_columns
        else 1.0
    )
    data_width, data_height, left, right, bottom, top = _panel_geometry_px(
        size,
        kind="default",
    )
    row_gap = (
        round(float(tick_font_points) * font_scale * DESIGN_DPI / 72.0) + 4
    )
    column_gap = max(6, round(10 * font_scale))
    return (
        (data_width, data_height),
        column_gap,
        row_gap,
        (left, right, bottom, top),
        font_scale,
    )


def _image_panel_layout(data: NormalizedBox) -> ImagePanelLayout:
    cursor = data.left
    image_width = data.width * _IMAGE_WIDTHS[0]
    image = NormalizedBox(
        cursor,
        data.top,
        cursor + image_width,
        data.bottom,
    )
    cursor = image.right + data.width * _IMAGE_GAPS[0]
    distribution_width = data.width * _IMAGE_WIDTHS[1]
    distribution = NormalizedBox(
        cursor,
        data.top,
        cursor + distribution_width,
        data.bottom,
    )
    cursor = distribution.right + data.width * _IMAGE_GAPS[1]
    colorbar = NormalizedBox(
        cursor,
        data.top,
        cursor + data.width * _IMAGE_WIDTHS[2],
        data.bottom,
    )
    return ImagePanelLayout(data, image, distribution, colorbar)


def image_panel_layout(size: str = "2x2") -> ImagePanelLayout:
    """Return Main's exact design-space image split for a named preset."""

    return _image_panel_layout(panel_data_box(size))


def image_panel_layout_for_raster(width: int, height: int) -> ImagePanelLayout:
    """Return the image split for a genuinely arbitrary raster host."""

    return _image_panel_layout(panel_data_box_for_raster(width, height))


def _rolling_panel_layout(data: NormalizedBox) -> RollingPanelLayout:
    history = NormalizedBox(
        data.left,
        data.top,
        data.left + data.width * 0.825,
        data.bottom,
    )
    distribution = NormalizedBox(
        history.right + data.width * 0.025,
        data.top,
        data.right,
        data.bottom,
    )
    return RollingPanelLayout(data, history, distribution)


def rolling_panel_layout(size: str = "2x2") -> RollingPanelLayout:
    """Return Main's exact design-space rolling split for a named preset."""

    return _rolling_panel_layout(panel_data_box(size))


def rolling_panel_layout_for_raster(width: int, height: int) -> RollingPanelLayout:
    """Return the rolling split for a genuinely arbitrary raster host."""

    return _rolling_panel_layout(panel_data_box_for_raster(width, height))


_PULSE_ROW_MIN_PX = 26
_PULSE_PERIOD_MIN_PX = 46


def optimal_pulse_size(channel_count: int, period_count: int) -> str:
    """Return the smallest panel preset that keeps pulse rows legible."""

    from zlc_data.panel_size import PANEL_SIZES, panel_size_cells

    rows_needed = max(1, int(channel_count))
    periods_needed = max(1, int(period_count))
    by_area = sorted(
        PANEL_SIZES,
        key=lambda name: (
            lambda cells: cells[0] * cells[1]
        )(panel_size_cells(name)),
    )
    for name in by_area:
        rows, columns = panel_size_cells(name)
        data_width = columns * PANEL_UNIT_PX[1]
        data_height = rows * PANEL_UNIT_PX[0]
        if (
            data_height >= rows_needed * _PULSE_ROW_MIN_PX
            and data_width >= periods_needed * _PULSE_PERIOD_MIN_PX
        ):
            return name
    return by_area[-1]


__all__ = [
    "DESIGN_DPI",
    "ImagePanelLayout",
    "NormalizedBox",
    "RollingPanelLayout",
    "PANEL_DISPLAY_SCALE",
    "LIVE_PANEL_DPI",
    "PANEL_MARGINS_PX",
    "PANEL_UNIT_PX",
    "PULSE_LEFT_MARGIN_PX",
    "STOCK_DATA_PX",
    "STOCK_MARGINS_PX",
    "TITLE_SLOT_PX",
    "image_panel_layout",
    "image_panel_layout_for_raster",
    "grid_shape_for",
    "grid_shape_for_aspect",
    "optimal_pulse_size",
    "optimal_grid_size",
    "panel_axes_bounds",
    "panel_data_box",
    "panel_data_box_for_raster",
    "panel_display_size",
    "panel_figure_size_inches",
    "panel_margins_px",
    "rolling_panel_layout",
    "rolling_panel_layout_for_raster",
    "site_grid_geometry",
    "SITE_GRID_MAX_COLUMNS",
]
