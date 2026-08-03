"""Fixed preset geometry shared by GUI, notebook and export surfaces.

Only the nine declared panel presets are accepted.  A host may choose another
preset through ``PlotSession.set_size``; it may not inject an arbitrary
window size and thereby create backend-specific layout behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._validation import finite_real, integer
from .kinds import PlotKind
from .style import PlotStyleConfig


_PANEL_PRESET_CELLS = (
    ("1x2", 1, 2),
    ("2x2", 2, 2),
    ("4x2", 4, 2),
    ("1x4", 1, 4),
    ("2x4", 2, 4),
    ("4x4", 4, 4),
    ("4x8", 4, 8),
    ("8x4", 8, 4),
    ("8x8", 8, 8),
)
PANEL_SIZE_NAMES = tuple(name for name, _, _ in _PANEL_PRESET_CELLS)


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    result = integer(value, field, minimum=1 if positive else None)
    assert result is not None
    return result


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    result = finite_real(value, field)
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _round_pixel(value: float) -> int:
    return max(1, int(math.floor(float(value) + 0.5)))


@dataclass(frozen=True, slots=True)
class PixelSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _integer(self.width, "pixel width", positive=True))
        object.__setattr__(self, "height", _integer(self.height, "pixel height", positive=True))

@dataclass(frozen=True, slots=True)
class Margins:
    left: int
    right: int
    bottom: int
    top: int

    def __post_init__(self) -> None:
        for field in ("left", "right", "bottom", "top"):
            value = _integer(getattr(self, field), f"{field} margin")
            if value < 0:
                raise ValueError("margins must be non-negative")
            object.__setattr__(self, field, value)

@dataclass(frozen=True, slots=True)
class PanelPreset:
    name: str
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("panel preset name must be text")
        rows = _integer(self.rows, "preset rows", positive=True)
        columns = _integer(self.columns, "preset columns", positive=True)
        canonical = f"{rows}x{columns}"
        if self.name != canonical:
            raise ValueError(f"panel preset name must be {canonical!r}")


@dataclass(frozen=True, slots=True)
class ImageSplit:
    image: float
    distribution: float
    colorbar: float
    image_distribution_gap: float
    distribution_colorbar_gap: float

    def __post_init__(self) -> None:
        for field in (
            "image",
            "distribution",
            "colorbar",
            "image_distribution_gap",
            "distribution_colorbar_gap",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        total = (
            self.image
            + self.distribution
            + self.colorbar
            + self.image_distribution_gap
            + self.distribution_colorbar_gap
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("image split widths and gaps must sum to 1")


@dataclass(frozen=True, slots=True)
class RollingSplit:
    history: float
    gap: float
    distribution: float

    def __post_init__(self) -> None:
        for field in ("history", "gap", "distribution"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if not math.isclose(
            self.history + self.gap + self.distribution,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("rolling split widths and gap must sum to 1")


@dataclass(frozen=True, slots=True)
class PlotLayoutConfig:
    """All logical geometry and fixed-size policy."""

    presets: tuple[PanelPreset, ...]
    default_preset: str
    design_dpi: float
    display_scale: float
    export_dpi: float
    panel_unit: PixelSize
    panel_margins: Margins
    pulse_left_margin_px: int
    image_split: ImageSplit
    rolling_split: RollingSplit
    facet_max_columns: int
    facet_double_rows_threshold: int
    facet_double_columns_threshold: int
    facet_column_gap_px: int
    facet_min_column_gap_px: int
    facet_row_gap_extra_px: int
    pulse_row_min_px: int
    pulse_period_min_px: int

    def __post_init__(self) -> None:
        presets = tuple(self.presets)
        if not presets or any(not isinstance(item, PanelPreset) for item in presets):
            raise ValueError("presets must contain PanelPreset values")
        names = tuple(item.name for item in presets)
        if len(names) != len(set(names)):
            raise ValueError("panel preset names must be unique")
        if names != PANEL_SIZE_NAMES:
            raise ValueError(
                "presets must be the nine fixed panel sizes in canonical order"
            )
        object.__setattr__(self, "presets", presets)
        if not isinstance(self.default_preset, str) or self.default_preset not in names:
            raise ValueError("default_preset must name one declared preset")
        for field in ("design_dpi", "display_scale", "export_dpi"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if self.export_dpi < self.live_dpi:
            raise ValueError("export_dpi must not be lower than live_dpi")
        if not isinstance(self.panel_unit, PixelSize):
            raise TypeError("panel_unit must be PixelSize")
        if not isinstance(self.panel_margins, Margins):
            raise TypeError("panel_margins must be Margins")
        if not isinstance(self.image_split, ImageSplit):
            raise TypeError("image_split must be ImageSplit")
        if not isinstance(self.rolling_split, RollingSplit):
            raise TypeError("rolling_split must be RollingSplit")
        for field in (
            "pulse_left_margin_px",
            "facet_max_columns",
            "facet_double_rows_threshold",
            "facet_double_columns_threshold",
            "facet_column_gap_px",
            "facet_min_column_gap_px",
            "facet_row_gap_extra_px",
            "pulse_row_min_px",
            "pulse_period_min_px",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field, positive=True))
        if self.facet_min_column_gap_px > self.facet_column_gap_px:
            raise ValueError("minimum facet column gap cannot exceed its normal gap")

    @property
    def size_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.presets)

    @property
    def live_dpi(self) -> float:
        return self.design_dpi * self.display_scale

    @property
    def export_scale(self) -> float:
        return self.export_dpi / self.live_dpi

    def validate_preset(self, preset: str) -> str:
        if not isinstance(preset, str):
            raise TypeError("panel preset must be text")
        canonical = preset.strip().lower().replace(" ", "")
        if canonical not in self.size_names:
            raise ValueError(
                f"unknown panel preset {preset!r}; choose from {', '.join(self.size_names)}"
            )
        return canonical

    def preset(self, preset: str) -> PanelPreset:
        canonical = self.validate_preset(preset)
        return next(item for item in self.presets if item.name == canonical)


DEFAULT_LAYOUT = PlotLayoutConfig(
    presets=tuple(
        PanelPreset(name, rows, columns)
        for name, rows, columns in _PANEL_PRESET_CELLS
    ),
    default_preset="2x2",
    design_dpi=300.0,
    display_scale=0.7,
    export_dpi=600.0,
    panel_unit=PixelSize(240, 180),
    panel_margins=Margins(110, 96, 80, 70),
    pulse_left_margin_px=122,
    image_split=ImageSplit(0.75, 0.10, 0.10, 0.025, 0.025),
    rolling_split=RollingSplit(0.825, 0.025, 0.15),
    facet_max_columns=7,
    facet_double_rows_threshold=4,
    facet_double_columns_threshold=5,
    facet_column_gap_px=10,
    facet_min_column_gap_px=6,
    facet_row_gap_extra_px=4,
    pulse_row_min_px=26,
    pulse_period_min_px=46,
)


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Top-origin normalized rectangle used identically by every backend."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for field in ("left", "top", "right", "bottom"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if not 0.0 <= self.left < self.right <= 1.0:
            raise ValueError("box requires 0 <= left < right <= 1")
        if not 0.0 <= self.top < self.bottom <= 1.0:
            raise ValueError("box requires 0 <= top < bottom <= 1")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def matplotlib_bounds(self) -> tuple[float, float, float, float]:
        return (self.left, 1.0 - self.bottom, self.width, self.height)


@dataclass(frozen=True, slots=True)
class AxesPlan:
    role: str
    box: NormalizedBox
    cell_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("axes role must be non-empty text")
        object.__setattr__(self, "role", self.role.strip())
        if not isinstance(self.box, NormalizedBox):
            raise TypeError("axes box must be NormalizedBox")
        if self.cell_index is not None:
            object.__setattr__(self, "cell_index", _integer(self.cell_index, "cell_index"))
            if self.cell_index < 0:
                raise ValueError("cell_index must be non-negative")


@dataclass(frozen=True, slots=True)
class FacetTopology:
    """Render topology for homogeneous FacetGrid cells."""

    cell_count: int
    cell_aspect: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_count", _integer(self.cell_count, "cell_count", positive=True))
        if self.cell_aspect is not None:
            object.__setattr__(self, "cell_aspect", _finite(self.cell_aspect, "cell_aspect", positive=True))

@dataclass(frozen=True, slots=True)
class FacetTypographyPlan:
    """Cell chrome uses the reference compact/normal typography tier."""

    tier: str
    scale: float
    cell_title_pt: float
    tick_pt: float
    outer_axis_label_pt: float

    def __post_init__(self) -> None:
        if self.tier not in {"compact", "normal"}:
            raise ValueError("facet typography tier must be 'compact' or 'normal'")
        for field in (
            "scale",
            "cell_title_pt",
            "tick_pt",
            "outer_axis_label_pt",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))


@dataclass(frozen=True, slots=True)
class SurfacePlan:
    """Fully resolved immutable geometry for one surface revision."""

    preset: str
    kind: str
    logical_size: tuple[int, int]
    raster_size: tuple[int, int]
    figure_size_inches: tuple[float, float]
    logical_dpi: float
    dpi: float
    device_pixel_ratio: float
    export_scale: float
    axes: tuple[AxesPlan, ...]
    facet_topology: FacetTopology | None
    facet_shape: tuple[int, int] | None
    recommended_preset: str | None
    facet_typography: FacetTypographyPlan | None
    rolling_side_distribution: bool

    def __post_init__(self) -> None:
        for field in ("logical_size", "raster_size"):
            values = tuple(getattr(self, field))
            if len(values) != 2:
                raise ValueError(f"{field} must have width and height")
            values = tuple(_integer(value, field, positive=True) for value in values)
            object.__setattr__(self, field, values)
        inches = tuple(self.figure_size_inches)
        if len(inches) != 2:
            raise ValueError("figure_size_inches must have width and height")
        object.__setattr__(
            self,
            "figure_size_inches",
            tuple(_finite(value, "figure size", positive=True) for value in inches),
        )
        for field in ("logical_dpi", "dpi", "device_pixel_ratio", "export_scale"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        axes = tuple(self.axes)
        if not axes or any(not isinstance(item, AxesPlan) for item in axes):
            raise ValueError("surface must contain AxesPlan values")
        object.__setattr__(self, "axes", axes)
        if not isinstance(self.rolling_side_distribution, bool):
            raise TypeError("rolling_side_distribution must be bool")

def _kind_name(kind: object) -> str:
    value = getattr(kind, "value", kind)
    if not isinstance(value, str):
        raise TypeError("plot kind must be text or a string-valued enum")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return PlotKind(key).value
    except ValueError as error:
        raise ValueError(f"unknown plot kind {value!r}") from error


def _panel_design_geometry(
    preset: PanelPreset,
    kind: str,
    layout: PlotLayoutConfig,
) -> tuple[int, int, Margins]:
    margins = layout.panel_margins
    if kind == "pulse_timeline":
        margins = Margins(
            layout.pulse_left_margin_px,
            margins.right,
            margins.bottom,
            margins.top,
        )
    data_width = preset.columns * layout.panel_unit.width
    data_height = preset.rows * layout.panel_unit.height
    return data_width, data_height, margins


def _data_box(
    data_width: int,
    data_height: int,
    margins: Margins,
) -> NormalizedBox:
    width = margins.left + data_width + margins.right
    height = margins.top + data_height + margins.bottom
    return NormalizedBox(
        margins.left / width,
        margins.top / height,
        (margins.left + data_width) / width,
        (margins.top + data_height) / height,
    )


def facet_shape(cell_count: int, *, max_columns: int) -> tuple[int, int]:
    count = _integer(cell_count, "cell_count", positive=True)
    cap = _integer(max_columns, "max_columns", positive=True)
    columns = min(cap, int(math.ceil(math.sqrt(count))))
    return (int(math.ceil(count / columns)), columns)


def facet_shape_for_aspect(
    cell_count: int,
    cell_aspect: float,
    region_px: tuple[int, int],
    *,
    max_columns: int,
) -> tuple[int, int]:
    count = _integer(cell_count, "cell_count", positive=True)
    aspect = _finite(cell_aspect, "cell_aspect", positive=True)
    if len(region_px) != 2:
        raise ValueError("region_px must contain width and height")
    region_width, region_height = (
        _finite(value, "region dimension", positive=True) for value in region_px
    )
    cap = min(_integer(max_columns, "max_columns", positive=True), count)
    best_key: tuple[float, int, int] | None = None
    best = (1, count)
    for columns in range(1, cap + 1):
        rows = int(math.ceil(count / columns))
        cell_width = region_width / columns
        cell_height = region_height / rows
        image_height = min(cell_height, cell_width / aspect)
        key = (image_height, -(rows * columns), -abs(rows - columns))
        if best_key is None or key > best_key:
            best_key = key
            best = (rows, columns)
    return best


def facet_image_cell_aspect(x_count: int, y_count: int) -> float:
    """Return an image-cell aspect from x/y coordinate cardinalities."""

    x_size = _integer(x_count, "image facet x_count", positive=True)
    y_size = _integer(y_count, "image facet y_count", positive=True)
    return x_size / y_size


def facet_focus_box(plan: SurfacePlan) -> NormalizedBox:
    """Return the full data region occupied by a FacetGrid overview."""

    if not isinstance(plan, SurfacePlan):
        raise TypeError("plan must be SurfacePlan")
    if plan.kind != "facet_grid":
        raise TypeError("facet focus geometry requires a FacetGrid surface")
    boxes = tuple(item.box for item in plan.axes if item.role == "facet_cell")
    if not boxes:
        raise ValueError("FacetGrid surface has no cell geometry")
    return NormalizedBox(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def recommended_facet_preset(
    topology: FacetTopology,
    layout: PlotLayoutConfig = DEFAULT_LAYOUT,
) -> str:
    if not isinstance(topology, FacetTopology):
        raise TypeError("topology must be FacetTopology")
    if not isinstance(layout, PlotLayoutConfig):
        raise TypeError("layout must be PlotLayoutConfig")
    rows, columns = facet_shape(topology.cell_count, max_columns=layout.facet_max_columns)
    row_units = 4 if rows > layout.facet_double_rows_threshold else 2
    column_units = 4 if columns > layout.facet_double_columns_threshold else 2
    return layout.validate_preset(f"{row_units}x{column_units}")


def _facet_typography(
    selected: PanelPreset,
    recommended: PanelPreset,
    style: PlotStyleConfig,
) -> FacetTypographyPlan:
    compact = selected.rows < recommended.rows or selected.columns < recommended.columns
    scale = style.fonts.facet_compact_scale if compact else style.fonts.facet_normal_scale
    return FacetTypographyPlan(
        tier="compact" if compact else "normal",
        scale=scale,
        cell_title_pt=style.fonts.tick_pt * scale,
        tick_pt=style.fonts.tick_pt * scale,
        outer_axis_label_pt=style.fonts.axis_label_pt,
    )


def _split_image(data: NormalizedBox, split: ImageSplit) -> tuple[AxesPlan, ...]:
    cursor = data.left
    image = NormalizedBox(cursor, data.top, cursor + data.width * split.image, data.bottom)
    cursor = image.right + data.width * split.image_distribution_gap
    distribution = NormalizedBox(
        cursor,
        data.top,
        cursor + data.width * split.distribution,
        data.bottom,
    )
    cursor = distribution.right + data.width * split.distribution_colorbar_gap
    colorbar = NormalizedBox(cursor, data.top, data.right, data.bottom)
    return (
        AxesPlan("image", image),
        AxesPlan("distribution", distribution),
        AxesPlan("colorbar", colorbar),
    )


def _split_rolling(
    data: NormalizedBox,
    split: RollingSplit,
    show_distribution: bool,
) -> tuple[AxesPlan, ...]:
    if not show_distribution:
        return (AxesPlan("history", data),)
    history = NormalizedBox(
        data.left,
        data.top,
        data.left + data.width * split.history,
        data.bottom,
    )
    distribution = NormalizedBox(
        history.right + data.width * split.gap,
        data.top,
        data.right,
        data.bottom,
    )
    return (AxesPlan("history", history), AxesPlan("distribution", distribution))


def _facet_axes(
    count: int,
    shape: tuple[int, int],
    data_width: int,
    data_height: int,
    margins: Margins,
    typography: FacetTypographyPlan,
    layout: PlotLayoutConfig,
) -> tuple[AxesPlan, ...]:
    rows, columns = shape
    row_gap = _round_pixel(
        typography.tick_pt * layout.design_dpi / 72.0
    ) + layout.facet_row_gap_extra_px
    column_gap = max(
        layout.facet_min_column_gap_px,
        _round_pixel(layout.facet_column_gap_px * typography.scale),
    )
    cell_width = max((data_width - (columns - 1) * column_gap) / columns, 1.0)
    cell_height = max((data_height - (rows - 1) * row_gap) / rows, 1.0)
    figure_width = margins.left + data_width + margins.right
    figure_height = margins.bottom + data_height + margins.top
    result = []
    for index in range(count):
        row, column = divmod(index, columns)
        left = margins.left + column * (cell_width + column_gap)
        bottom_from_origin = margins.bottom + (rows - 1 - row) * (cell_height + row_gap)
        top = figure_height - (bottom_from_origin + cell_height)
        result.append(
            AxesPlan(
                "facet_cell",
                NormalizedBox(
                    left / figure_width,
                    top / figure_height,
                    (left + cell_width) / figure_width,
                    (top + cell_height) / figure_height,
                ),
                index,
            )
        )
    return tuple(result)


def resolve_surface(
    preset: str | None = None,
    kind: object = "curve",
    facet_topology: FacetTopology | None = None,
    *,
    device_pixel_ratio: float = 1.0,
    export_scale: float | None = None,
    rolling_side_distribution: bool | None = None,
    layout: PlotLayoutConfig,
    style: PlotStyleConfig,
) -> SurfacePlan:
    """Resolve one of the fixed presets into a backend-independent plan.

    ``device_pixel_ratio`` and ``export_scale`` affect only physical raster
    pixels and render DPI.  They never change logical size, normalized axes,
    facet packing or typography.
    """

    if not isinstance(layout, PlotLayoutConfig):
        raise TypeError("layout must be PlotLayoutConfig")
    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    canonical_kind = _kind_name(kind)
    if canonical_kind == PlotKind.ROLLING.value:
        if not isinstance(rolling_side_distribution, bool):
            raise TypeError(
                "Rolling surfaces require an explicit rolling_side_distribution"
            )
    elif rolling_side_distribution is not None:
        raise ValueError(
            "rolling_side_distribution is accepted only for Rolling surfaces"
        )
    if canonical_kind == "facet_grid":
        if not isinstance(facet_topology, FacetTopology):
            raise TypeError("FacetGrid requires facet_topology")
        recommended_name = recommended_facet_preset(facet_topology, layout)
        selected_name = recommended_name if preset is None else layout.validate_preset(preset)
    else:
        if facet_topology is not None:
            raise ValueError("facet_topology is accepted only for FacetGrid")
        recommended_name = None
        selected_name = layout.default_preset if preset is None else layout.validate_preset(preset)
    selected = layout.preset(selected_name)
    data_width, data_height, margins = _panel_design_geometry(selected, canonical_kind, layout)
    figure_design_width = margins.left + data_width + margins.right
    figure_design_height = margins.top + data_height + margins.bottom
    logical_size = (
        _round_pixel(figure_design_width * layout.display_scale),
        _round_pixel(figure_design_height * layout.display_scale),
    )
    dpr = _finite(device_pixel_ratio, "device_pixel_ratio", positive=True)
    physical_export_scale = 1.0 if export_scale is None else _finite(
        export_scale, "export_scale", positive=True
    )
    physical_scale = dpr * physical_export_scale
    physical_dpi = layout.live_dpi * physical_scale
    raster_size = tuple(
        _round_pixel(value * physical_scale) for value in logical_size
    )
    # Derive inches from the already-rounded device raster.  Matplotlib
    # otherwise truncates ``figsize * dpi`` on fractional screen DPR and can
    # miss the fixed Qt/notebook surface by one physical pixel.
    figure_size_inches = tuple(
        pixels / physical_dpi for pixels in raster_size
    )
    data = _data_box(data_width, data_height, margins)
    typography = None
    shape = None
    show_distribution = False
    if canonical_kind == "image":
        axes = _split_image(data, layout.image_split)
    elif canonical_kind == "rolling":
        assert rolling_side_distribution is not None
        show_distribution = rolling_side_distribution
        axes = _split_rolling(data, layout.rolling_split, show_distribution)
    elif canonical_kind == "facet_grid":
        assert facet_topology is not None and recommended_name is not None
        recommended = layout.preset(recommended_name)
        typography = _facet_typography(selected, recommended, style)
        shape = (
            facet_shape(
                facet_topology.cell_count,
                max_columns=layout.facet_max_columns,
            )
            if facet_topology.cell_aspect is None
            else facet_shape_for_aspect(
                facet_topology.cell_count,
                facet_topology.cell_aspect,
                (data_width, data_height),
                max_columns=layout.facet_max_columns,
            )
        )
        axes = _facet_axes(
            facet_topology.cell_count,
            shape,
            data_width,
            data_height,
            margins,
            typography,
            layout,
        )
    else:
        axes = (AxesPlan("main", data),)
    return SurfacePlan(
        preset=selected.name,
        kind=canonical_kind,
        logical_size=logical_size,
        raster_size=raster_size,
        figure_size_inches=figure_size_inches,
        logical_dpi=layout.live_dpi,
        dpi=physical_dpi,
        device_pixel_ratio=dpr,
        export_scale=physical_export_scale,
        axes=axes,
        facet_topology=facet_topology,
        facet_shape=shape,
        recommended_preset=recommended_name,
        facet_typography=typography,
        rolling_side_distribution=show_distribution,
    )


__all__ = [
    "AxesPlan",
    "DEFAULT_LAYOUT",
    "FacetTopology",
    "FacetTypographyPlan",
    "ImageSplit",
    "Margins",
    "NormalizedBox",
    "PANEL_SIZE_NAMES",
    "PanelPreset",
    "PixelSize",
    "PlotLayoutConfig",
    "RollingSplit",
    "SurfacePlan",
    "facet_shape",
    "facet_shape_for_aspect",
    "facet_image_cell_aspect",
    "facet_focus_box",
    "recommended_facet_preset",
    "resolve_surface",
]
