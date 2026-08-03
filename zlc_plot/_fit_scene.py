"""Backend-neutral display-space values for presenting fitted models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import finite_real, integer, readonly_copy
from .fit import FitParameterDisplay


@dataclass(frozen=True, slots=True)
class FitPolyline:
    x: np.ndarray
    y: np.ndarray
    role: str = "primary"
    component_index: int = 0

    def __post_init__(self) -> None:
        x = readonly_copy(self.x, dtype=float).reshape(-1)
        y = readonly_copy(self.y, dtype=float).reshape(-1)
        if x.shape != y.shape or not x.size:
            raise ValueError("fit polyline x/y must be non-empty equal-length arrays")
        if self.role not in {"primary", "component", "total"}:
            raise ValueError("fit polyline role must be primary, component or total")
        component_index = integer(
            self.component_index,
            "fit component_index",
            minimum=0,
        )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "component_index", component_index)


@dataclass(frozen=True, slots=True)
class FitRadialGlyph:
    """One indivisible center-and-radius glyph in display coordinates."""

    center_x: float
    center_y: float
    radius_x: float
    radius_y: float

    def __post_init__(self) -> None:
        values = tuple(
            finite_real(value, f"fit radial glyph {name}")
            for name, value in zip(
                ("center_x", "center_y", "radius_x", "radius_y"),
                (self.center_x, self.center_y, self.radius_x, self.radius_y),
                strict=True,
            )
        )
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("fit radial glyph radii must be positive")
        for name, value in zip(
            ("center_x", "center_y", "radius_x", "radius_y"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class FitOverlay:
    """Complete display-space fit scene."""

    polylines: tuple[FitPolyline, ...] = ()
    radial_glyph: FitRadialGlyph | None = None
    suggested_threshold: float | None = None
    success: bool = True
    formula: str = ""
    parameter_display: tuple[FitParameterDisplay, ...] = ()
    diagnostic: str = ""
    facet_index: int | None = None

    def __post_init__(self) -> None:
        polylines = tuple(self.polylines)
        if any(not isinstance(item, FitPolyline) for item in polylines):
            raise TypeError("fit polylines must contain FitPolyline values")
        object.__setattr__(self, "polylines", polylines)
        if self.radial_glyph is not None and not isinstance(
            self.radial_glyph,
            FitRadialGlyph,
        ):
            raise TypeError("fit radial_glyph must be FitRadialGlyph or None")
        if self.suggested_threshold is not None:
            object.__setattr__(
                self,
                "suggested_threshold",
                finite_real(
                    self.suggested_threshold,
                    "fit suggested threshold",
                ),
            )
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("fit overlay success must be bool")
        object.__setattr__(self, "success", bool(self.success))
        if not isinstance(self.formula, str):
            raise TypeError("fit formula must be text")
        object.__setattr__(self, "formula", self.formula.strip())
        parameter_display = tuple(self.parameter_display)
        if any(not isinstance(item, FitParameterDisplay) for item in parameter_display):
            raise TypeError("parameter_display must contain FitParameterDisplay values")
        names = tuple(item.name for item in parameter_display)
        if len(names) != len(set(names)):
            raise ValueError("fit parameter names must be unique")
        object.__setattr__(self, "parameter_display", parameter_display)
        if not isinstance(self.diagnostic, str):
            raise TypeError("fit diagnostic must be text")
        object.__setattr__(self, "diagnostic", self.diagnostic.strip())
        object.__setattr__(
            self,
            "facet_index",
            integer(
                self.facet_index,
                "fit facet_index",
                minimum=0,
                optional=True,
            ),
        )


__all__ = ["FitOverlay"]
