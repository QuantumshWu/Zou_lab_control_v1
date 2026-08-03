"""Typed semantic plot specifications and their editable display parameters.

The specification describes *what* data coordinates a plot uses.  It is
immutable for the lifetime of a session.  Frequently edited presentation
values live in :class:`~zlc_plot.state.DisplayStateStore`, so a GUI slider can
change (for example) ``bin_count`` without constructing another Figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, TypeAlias

from ._validation import finite_real, integer, nonempty_text
from .kinds import AxisRef, PlotKind
from .parameters import ParameterSchema, ParameterSpec, RenderEffect
from .style import PlotStyleConfig


_TEXT_EFFECTS = RenderEffect.TEXT
_DISPLAY_UNIT_EFFECTS = (
    RenderEffect.VIEW_PROJECTION
    | RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.TEXT
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_FACET_UNIT_EFFECTS = (
    RenderEffect.VIEW_PROJECTION
    | RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.TEXT
)
_PULSE_TIME_UNIT_EFFECTS = (
    RenderEffect.AXIS_TRANSFORM
    | RenderEffect.TEXT
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_AXIS_LIMIT_EFFECTS = (
    RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_HISTOGRAM_PROJECTION_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
)
_IMAGE_COLOR_EFFECTS = (
    RenderEffect.BASE_STYLE
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_ROLLING_DISTRIBUTION_BIN_EFFECTS = (
    RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
)
_ROLLING_WINDOW_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
)


class Reduction(str, Enum):
    """How repeated samples that share a plotted coordinate are combined."""

    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    FIRST = "first"

    def __str__(self) -> str:
        return self.value.title()


class RelimMode(str, Enum):
    TIGHT = "tight"
    NORMAL = "normal"
    FIXED = "fixed"


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be text")
    return value


def _nonempty_text(value: object) -> str:
    return nonempty_text(value, "value")


def _integer(value: object) -> int:
    result = integer(value, "value")
    assert result is not None
    return result


def _finite_float(value: object) -> float:
    return finite_real(value, "value")


def _finite_or_none(value: object) -> float | None:
    return None if value is None else _finite_float(value)


def _relim_mode(value: object) -> str:
    if isinstance(value, RelimMode):
        return value.value
    return _text(value).strip().lower()


def _unit_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    result = _text(value).strip()
    return result or None


def _label_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    return _text(value)


@dataclass(frozen=True, slots=True)
class PlotLabels:
    title: str | None = None
    x: str | None = None
    y: str | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        for name in ("title", "x", "y", "value"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"label {name!r} must be non-empty text or None")


@dataclass(frozen=True, slots=True)
class CurvePlot:
    x: AxisRef
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.CURVE

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisRef):
            raise TypeError("CurvePlot.x must be AxisRef")
        if self.group is not None and not isinstance(self.group, AxisRef):
            raise TypeError("CurvePlot.group must be AxisRef or None")
        if self.group == self.x:
            raise ValueError("CurvePlot.group must differ from x")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("CurvePlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("CurvePlot.labels must be PlotLabels")


@dataclass(frozen=True, slots=True)
class ImagePlot:
    x: AxisRef
    y: AxisRef
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.IMAGE

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisRef) or not isinstance(self.y, AxisRef):
            raise TypeError("ImagePlot x and y must be AxisRef")
        if self.x == self.y:
            raise ValueError("ImagePlot x and y must be different axes")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("ImagePlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("ImagePlot.labels must be PlotLabels")


@dataclass(frozen=True, slots=True)
class HistogramPlot:
    samples: tuple[AxisRef, ...] = ()
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.HISTOGRAM

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if any(not isinstance(sample, AxisRef) for sample in samples):
            raise TypeError("HistogramPlot.samples must contain AxisRef values")
        if len(set(samples)) != len(samples):
            raise ValueError("HistogramPlot.samples must be unique")
        object.__setattr__(self, "samples", samples)
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("HistogramPlot.labels must be PlotLabels")


@dataclass(frozen=True, slots=True)
class RollingPlot:
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.ROLLING

    def __post_init__(self) -> None:
        if self.group is not None and not isinstance(self.group, AxisRef):
            raise TypeError("RollingPlot.group must be AxisRef or None")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("RollingPlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("RollingPlot.labels must be PlotLabels")


CellPlot: TypeAlias = CurvePlot | ImagePlot | HistogramPlot


@dataclass(frozen=True, slots=True)
class FacetGridPlot:
    facet: AxisRef
    cell: CellPlot
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.FACET_GRID

    def __post_init__(self) -> None:
        if not isinstance(self.facet, AxisRef):
            raise TypeError("FacetGridPlot.facet must be AxisRef")
        if not isinstance(self.cell, (CurvePlot, ImagePlot, HistogramPlot)):
            raise TypeError("FacetGrid cells must be CurvePlot, ImagePlot or HistogramPlot")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("FacetGridPlot.labels must be PlotLabels")
        if self.facet in {
            getattr(self.cell, "x", None),
            getattr(self.cell, "y", None),
            getattr(self.cell, "group", None),
            *getattr(self.cell, "samples", ()),
        }:
            raise ValueError("facet source cannot also be a cell axis or group")


@dataclass(frozen=True, slots=True)
class PulseTimelinePlot:
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.PULSE_TIMELINE

    def __post_init__(self) -> None:
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("PulseTimelinePlot.labels must be PlotLabels")


PlotSpec: TypeAlias = (
    CurvePlot
    | ImagePlot
    | HistogramPlot
    | RollingPlot
    | FacetGridPlot
    | PulseTimelinePlot
)


def _title_parameter(default: str | None) -> ParameterSpec[object]:
    return ParameterSpec(
        "title",
        str,
        _TEXT_EFFECTS,
        default="" if default is None else default,
        normalizer=_text,
        label="Title",
    )


def _label_parameter(name: str, default: str | None, label: str) -> ParameterSpec[object]:
    return ParameterSpec(
        name,
        str,
        _TEXT_EFFECTS,
        default=default,
        normalizer=_label_text_or_none,
        allow_none=True,
        label=label,
    )


def _show_grid_parameter(*, default: bool = False) -> ParameterSpec[object]:
    return ParameterSpec(
        "show_grid",
        bool,
        RenderEffect.CHROME,
        default=default,
        label="Grid",
    )


def _unit_parameter(
    name: str,
    *,
    effects: RenderEffect = _DISPLAY_UNIT_EFFECTS,
) -> ParameterSpec[object]:
    return ParameterSpec(
        name,
        str,
        effects,
        default=None,
        normalizer=_unit_text_or_none,
        allow_none=True,
        label=name.removesuffix("_display_unit").replace("_", " ").title() + " unit",
    )


def _pulse_time_unit(value: object) -> str | None:
    unit = _unit_text_or_none(value)
    if unit is None:
        return None
    normalized = unit.lower().replace("μ", "µ")
    if normalized == "µs":
        normalized = "us"
    return normalized


_PULSE_X_UNIT_PARAMETER = ParameterSpec(
    "x_display_unit",
    str,
    _PULSE_TIME_UNIT_EFFECTS,
    default=None,
    normalizer=_pulse_time_unit,
    allow_none=True,
    label="Time unit",
    choices=("ns", "us", "ms", "s"),
)


def _curve_parameters() -> tuple[ParameterSpec[object], ...]:
    return (
        ParameterSpec(
            "relim_mode",
            str,
            _AXIS_LIMIT_EFFECTS,
            default=RelimMode.NORMAL.value,
            normalizer=_relim_mode,
            label="Limits",
            choices=tuple(item.value for item in RelimMode),
        ),
    )


def _bin_count_parameter(
    effects: RenderEffect = _HISTOGRAM_PROJECTION_EFFECTS,
) -> ParameterSpec[object]:
    return ParameterSpec(
        "bin_count",
        int,
        effects,
        default=60,
        normalizer=_integer,
        label="Bins",
        minimum=1,
        maximum=4096,
        step=1,
    )


def _histogram_parameters() -> tuple[ParameterSpec[object], ...]:
    return (
        _bin_count_parameter(),
        ParameterSpec(
            "density",
            bool,
            _HISTOGRAM_PROJECTION_EFFECTS | RenderEffect.TEXT,
            default=False,
            label="Density",
        ),
        ParameterSpec(
            "cumulative",
            bool,
            _HISTOGRAM_PROJECTION_EFFECTS,
            default=False,
            label="Cumulative",
        ),
        ParameterSpec(
            "relim_mode",
            str,
            _AXIS_LIMIT_EFFECTS,
            default=RelimMode.NORMAL.value,
            normalizer=_relim_mode,
            label="Count limits",
            choices=tuple(item.value for item in RelimMode),
        ),
        ParameterSpec(
            "log_y",
            bool,
            _AXIS_LIMIT_EFFECTS,
            default=False,
            label="Log count axis",
        ),
    )


def _image_parameters(
    style: PlotStyleConfig,
    *,
    default_interpolation: str,
    include_colorbar: bool = True,
) -> tuple[ParameterSpec[object], ...]:
    policy = style.render
    entries: list[ParameterSpec[object]] = [
        ParameterSpec(
            "relim_mode",
            str,
            _IMAGE_COLOR_EFFECTS,
            default=RelimMode.TIGHT.value,
            normalizer=_relim_mode,
            label="Color limits",
            choices=tuple(item.value for item in RelimMode),
        ),
        ParameterSpec(
            "colormap",
            str,
            _IMAGE_COLOR_EFFECTS,
            default=policy.image_default_colormap,
            normalizer=_nonempty_text,
            label="Colormap",
            choices=policy.image_colormaps,
        ),
        ParameterSpec(
            "color_min",
            (int, float),
            _IMAGE_COLOR_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Color minimum",
        ),
        ParameterSpec(
            "color_max",
            (int, float),
            _IMAGE_COLOR_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Color maximum",
        ),
        ParameterSpec(
            "interpolation",
            str,
            RenderEffect.BASE_STYLE,
            default=default_interpolation,
            normalizer=_nonempty_text,
            label="Interpolation",
            choices=policy.image_interpolations,
        ),
    ]
    if include_colorbar:
        entries.append(
            ParameterSpec(
                "show_colorbar",
                bool,
                RenderEffect.CHROME,
                default=True,
                label="Colorbar",
            )
        )
    return tuple(entries)


_LIMIT_PARAMETER_PAIRS = (
    ("color_min", "color_max"),
    ("y_min", "y_max"),
)


def _normalize_limit_transition(
    current: Mapping[str, object],
    updates: Mapping[str, object],
) -> Mapping[str, object]:
    normalized = dict(updates)
    if "relim_mode" in updates and updates["relim_mode"] != RelimMode.FIXED.value:
        for low_name, high_name in _LIMIT_PARAMETER_PAIRS:
            if low_name in current and high_name in current:
                normalized.setdefault(low_name, None)
                normalized.setdefault(high_name, None)
    return normalized


def _validate_limit_state(values: Mapping[str, object]) -> None:
    mode = values["relim_mode"]
    for low_name, high_name in _LIMIT_PARAMETER_PAIRS:
        if low_name not in values or high_name not in values:
            continue
        low = values[low_name]
        high = values[high_name]
        if low is not None and high is not None and float(low) >= float(high):
            raise ValueError(f"{low_name} must be smaller than {high_name}")
        if mode == RelimMode.FIXED.value and (low is None or high is None):
            raise ValueError(
                f"fixed relim_mode requires {low_name} and {high_name}"
            )
    if (
        values.get("log_y") is True
        and mode == RelimMode.FIXED.value
        and values.get("y_min") is not None
        and float(values["y_min"]) <= 0.0
    ):
        raise ValueError("log count limits require a positive y_min")


def parameter_schema_for(
    spec: PlotSpec,
    *,
    style: PlotStyleConfig,
    extra_parameters: tuple[ParameterSpec[object], ...] = (),
) -> ParameterSchema:
    """Return the introspectable, complete UI parameter contract for ``spec``."""

    if not isinstance(
        spec,
        (
            CurvePlot,
            ImagePlot,
            HistogramPlot,
            RollingPlot,
            FacetGridPlot,
            PulseTimelinePlot,
        ),
    ):
        raise TypeError("unsupported plot specification")
    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    if any(not isinstance(item, ParameterSpec) for item in extra_parameters):
        raise TypeError("extra_parameters must contain ParameterSpec values")
    entries: list[ParameterSpec[object]] = [
        _title_parameter(spec.labels.title),
        _label_parameter("x_label", spec.labels.x, "X label"),
        _label_parameter("y_label", spec.labels.y, "Y label"),
        _label_parameter("value_label", spec.labels.value, "Value label"),
        _show_grid_parameter(default=isinstance(spec, PulseTimelinePlot)),
    ]
    semantic = spec.cell if isinstance(spec, FacetGridPlot) else spec
    if isinstance(spec, PulseTimelinePlot):
        entries.append(_PULSE_X_UNIT_PARAMETER)
    else:
        if (
            isinstance(getattr(semantic, "x", None), AxisRef)
            and not isinstance(semantic, RollingPlot)
        ):
            entries.append(_unit_parameter("x_display_unit"))
        if isinstance(semantic, ImagePlot):
            entries.append(_unit_parameter("y_display_unit"))
        if isinstance(spec, FacetGridPlot):
            entries.append(
                _unit_parameter(
                    "facet_display_unit",
                    effects=_FACET_UNIT_EFFECTS,
                )
            )
        entries.append(_unit_parameter("value_display_unit"))
    if isinstance(semantic, (CurvePlot, RollingPlot)):
        entries.extend(_curve_parameters())
    if isinstance(semantic, HistogramPlot):
        entries.extend(_histogram_parameters())
    if isinstance(semantic, ImagePlot):
        entries.extend(
            _image_parameters(
                style,
                default_interpolation=(
                    style.render.facet_image_interpolation
                    if isinstance(spec, FacetGridPlot)
                    else style.render.image_default_interpolation
                ),
                include_colorbar=not isinstance(spec, FacetGridPlot),
            )
        )
    entries.extend(extra_parameters)
    if isinstance(spec, RollingPlot):
        entries.extend(
            (
                _bin_count_parameter(_ROLLING_DISTRIBUTION_BIN_EFFECTS),
                ParameterSpec(
                    "window",
                    int,
                    _ROLLING_WINDOW_EFFECTS,
                    default=100,
                    normalizer=_integer,
                    label="Window",
                    minimum=1,
                    maximum=1_000_000,
                    step=1,
                ),
                ParameterSpec(
                    "side_distribution",
                    bool,
                    RenderEffect.LAYOUT,
                    default=True,
                    label="Side distribution",
                ),
            )
        )
    if isinstance(spec, ImagePlot):
        entries.append(
            ParameterSpec(
                "show_point_labels",
                bool,
                RenderEffect.OVERLAY,
                default=False,
                label="Point labels",
            )
        )
    if isinstance(spec, PulseTimelinePlot):
        entries.append(
            ParameterSpec(
                "show_scan_regions",
                bool,
                RenderEffect.BASE_STYLE | RenderEffect.TEXT,
                default=True,
                label="Scan regions",
            )
        )
    has_limits = any(entry.name == "relim_mode" for entry in entries)
    return ParameterSchema(
        entries,
        transition_normalizer=(
            _normalize_limit_transition if has_limits else None
        ),
        full_state_validator=(
            _validate_limit_state if has_limits else None
        ),
    )


__all__ = [
    "CellPlot",
    "CurvePlot",
    "FacetGridPlot",
    "HistogramPlot",
    "ImagePlot",
    "PlotLabels",
    "PlotSpec",
    "PulseTimelinePlot",
    "Reduction",
    "RelimMode",
    "RollingPlot",
    "parameter_schema_for",
]
