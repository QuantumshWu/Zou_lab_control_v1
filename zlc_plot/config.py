"""Top-level immutable plotting configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import finite_real, integer
from .layout import DEFAULT_LAYOUT as _DEFAULT_LAYOUT, PlotLayoutConfig
from .style import PlotStyleConfig, build_plot_style


def _positive_integer(value: object, field: str) -> int:
    result = integer(value, field, minimum=1)
    assert result is not None
    return result


def _positive_float(value: object, field: str) -> float:
    result = finite_real(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _fraction(value: object, field: str) -> float:
    result = finite_real(value, field)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1)")
    return result


@dataclass(frozen=True, slots=True)
class LiveDefaults:
    """Allowed and default live-view refresh intervals."""

    refresh_intervals_ms: tuple[int, ...]
    default_refresh_interval_ms: int

    def __post_init__(self) -> None:
        intervals = tuple(
            _positive_integer(value, "refresh interval")
            for value in self.refresh_intervals_ms
        )
        if not intervals:
            raise ValueError("refresh_intervals_ms must not be empty")
        if tuple(sorted(set(intervals))) != intervals:
            raise ValueError("refresh_intervals_ms must be unique and ascending")
        default = _positive_integer(
            self.default_refresh_interval_ms,
            "default_refresh_interval_ms",
        )
        if default not in intervals:
            raise ValueError("default refresh interval must be one declared interval")
        object.__setattr__(self, "refresh_intervals_ms", intervals)
        object.__setattr__(self, "default_refresh_interval_ms", default)


@dataclass(frozen=True, slots=True)
class RuntimeDefaults:
    """Analysis worker and shutdown bounds."""

    analysis_worker_count: int
    shutdown_timeout_ms: int

    def __post_init__(self) -> None:
        for field in ("analysis_worker_count", "shutdown_timeout_ms"):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field),
            )


@dataclass(frozen=True, slots=True)
class InteractionDefaults:
    """Backend-independent pointer cadence and hit/zoom policy."""

    pointer_update_interval_ms: int
    raster_preview_interval_ms: int
    double_click_interval_ms: int
    double_click_radius_px: float
    selector_hit_radius_fraction: float
    selector_handle_radius_px: float
    selector_center_radius_px: float
    wheel_zoom_factor: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pointer_update_interval_ms",
            _positive_integer(
                self.pointer_update_interval_ms,
                "pointer_update_interval_ms",
            ),
        )
        object.__setattr__(
            self,
            "raster_preview_interval_ms",
            _positive_integer(
                self.raster_preview_interval_ms,
                "raster_preview_interval_ms",
            ),
        )
        object.__setattr__(
            self,
            "double_click_interval_ms",
            _positive_integer(
                self.double_click_interval_ms,
                "double_click_interval_ms",
            ),
        )
        object.__setattr__(
            self,
            "double_click_radius_px",
            _positive_float(
                self.double_click_radius_px,
                "double_click_radius_px",
            ),
        )
        object.__setattr__(
            self,
            "selector_hit_radius_fraction",
            _fraction(
                self.selector_hit_radius_fraction,
                "selector_hit_radius_fraction",
            ),
        )
        if self.selector_hit_radius_fraction == 0.0:
            raise ValueError("selector_hit_radius_fraction must be positive")
        handle_radius = _positive_float(
            self.selector_handle_radius_px,
            "selector_handle_radius_px",
        )
        center_radius = _positive_float(
            self.selector_center_radius_px,
            "selector_center_radius_px",
        )
        if center_radius < handle_radius:
            raise ValueError(
                "selector_center_radius_px must not be smaller than "
                "selector_handle_radius_px"
            )
        object.__setattr__(self, "selector_handle_radius_px", handle_radius)
        object.__setattr__(self, "selector_center_radius_px", center_radius)
        zoom = _positive_float(self.wheel_zoom_factor, "wheel_zoom_factor")
        if zoom <= 1.0:
            raise ValueError("wheel_zoom_factor must be greater than one")
        object.__setattr__(self, "wheel_zoom_factor", zoom)


@dataclass(frozen=True, slots=True)
class ProjectionDefaults:
    """Stable data-to-artist projection policy."""

    histogram_domain_padding_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "histogram_domain_padding_fraction",
            _fraction(
                self.histogram_domain_padding_fraction,
                "histogram_domain_padding_fraction",
            ),
        )


@dataclass(frozen=True, slots=True)
class PlotLibraryDefaults:
    """One configuration object for data-independent plotting policy."""

    style: PlotStyleConfig
    layout: PlotLayoutConfig
    live: LiveDefaults
    runtime: RuntimeDefaults
    interaction: InteractionDefaults
    projection: ProjectionDefaults

    def __post_init__(self) -> None:
        expected = (
            (self.style, PlotStyleConfig, "style"),
            (self.layout, PlotLayoutConfig, "layout"),
            (self.live, LiveDefaults, "live"),
            (self.runtime, RuntimeDefaults, "runtime"),
            (self.interaction, InteractionDefaults, "interaction"),
            (self.projection, ProjectionDefaults, "projection"),
        )
        for value, value_type, name in expected:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be {value_type.__name__}")


DEFAULTS = PlotLibraryDefaults(
    style=build_plot_style(),
    layout=_DEFAULT_LAYOUT,
    live=LiveDefaults((100, 200, 400, 800), 400),
    runtime=RuntimeDefaults(analysis_worker_count=1, shutdown_timeout_ms=5_000),
    interaction=InteractionDefaults(
        pointer_update_interval_ms=30,
        raster_preview_interval_ms=100,
        double_click_interval_ms=500,
        double_click_radius_px=6.0,
        selector_hit_radius_fraction=0.035,
        selector_handle_radius_px=10.0,
        selector_center_radius_px=20.0,
        wheel_zoom_factor=1.1,
    ),
    projection=ProjectionDefaults(histogram_domain_padding_fraction=0.05),
)


__all__ = [
    "DEFAULTS",
    "LiveDefaults",
    "InteractionDefaults",
    "PlotLibraryDefaults",
    "ProjectionDefaults",
    "RuntimeDefaults",
]
