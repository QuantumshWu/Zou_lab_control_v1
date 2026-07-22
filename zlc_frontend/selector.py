"""Headless, front-bound display gestures for raster panels.

Coordinate and viewport math has one owner in :mod:`zlc_frontend.image_view`.
This module only binds a completed selection gesture to the exact immutable
front on which it was drawn; it owns no Qt, Matplotlib, runtime control, or
analysis authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

from zlc_storage import canonical_text, nonnegative_integer

from .figure import EvaluatedInput
from .curve_display import CurveViewportTransform
from .display_range import optional_display_range, validated_display_range
from .histogram_display import HistogramViewportTransform
from .image_view import (
    ImageViewportTransform,
    NormalizedRectangle,
    validate_normalized_rectangle,
)
from .render import PanelPresentationIdentity, SourceIdentity


@dataclass(frozen=True, slots=True)
class RectangleGesture:
    """One immutable rectangle tied to the exact front on which it was drawn."""

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    source_identity: SourceIdentity
    normalized_bounds: NormalizedRectangle
    viewport_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", canonical_text(self.panel_id, "panel_id"))
        object.__setattr__(self, "board_id", canonical_text(self.board_id, "board_id"))
        for field in ("layout_generation", "sequence", "viewport_revision"):
            object.__setattr__(
                self,
                field,
                nonnegative_integer(getattr(self, field), field),
            )
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be zlc_frontend.render.SourceIdentity")
        object.__setattr__(
            self,
            "normalized_bounds",
            validate_normalized_rectangle(self.normalized_bounds),
        )


@dataclass(frozen=True, slots=True)
class PanelInteractionOrigin:
    """The exact painted front against which one display intent was authored."""

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    source_identity: SourceIdentity
    presentation: PanelPresentationIdentity
    evaluated_input: EvaluatedInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", canonical_text(self.panel_id, "panel_id"))
        object.__setattr__(self, "board_id", canonical_text(self.board_id, "board_id"))
        for field in ("layout_generation", "sequence"):
            object.__setattr__(
                self,
                field,
                nonnegative_integer(getattr(self, field), field),
            )
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be zlc_frontend.render.SourceIdentity")
        if not isinstance(self.presentation, PanelPresentationIdentity):
            raise TypeError(
                "presentation must be zlc_frontend.render.PanelPresentationIdentity"
            )
        if self.presentation.panel_id != self.panel_id:
            raise ValueError("interaction presentation belongs to another panel")
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("evaluated_input must be zlc_frontend.figure.EvaluatedInput")
        if self.evaluated_input.dataset_id != self.source_identity.dataset_id:
            raise ValueError("interaction input belongs to another dataset")
        if (
            self.evaluated_input.ref.block_id != self.source_identity.block_id
            or self.evaluated_input.ref.stream_generation
            != self.source_identity.stream_generation
            or self.evaluated_input.ref.schema_fingerprint
            != self.source_identity.schema_fingerprint
        ):
            raise ValueError("interaction input differs from its source identity")


@dataclass(frozen=True, slots=True)
class ImageViewportCommit:
    """Request one new committed viewport; the owner performs the state CAS."""

    origin: PanelInteractionOrigin
    viewport: ImageViewportTransform

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        if not isinstance(self.viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if self.viewport.viewport_revision <= self.origin.presentation.panel_revision:
            raise ValueError("viewport commit revision must exceed its painted origin")


@dataclass(frozen=True, slots=True)
class ImageColorLimitsCommit:
    """Request FIXED colour limits without mutating the painted QImage LUT."""

    origin: PanelInteractionOrigin
    color_limits: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        object.__setattr__(
            self,
            "color_limits",
            validated_display_range(self.color_limits, "color_limits"),
        )


ImageInteractionCommit: TypeAlias = ImageViewportCommit | ImageColorLimitsCommit


@dataclass(frozen=True, slots=True)
class CurveViewportCommit:
    """Request one display-only CURVE x viewport from an exact painted front."""

    origin: PanelInteractionOrigin
    viewport: CurveViewportTransform

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        if not isinstance(self.viewport, CurveViewportTransform):
            raise TypeError("viewport must be CurveViewportTransform")
        if self.viewport.display_revision <= self.origin.presentation.panel_revision:
            raise ValueError("curve viewport commit revision must exceed its painted origin")


@dataclass(frozen=True, slots=True)
class CurveRangeGesture:
    """Set or clear one display-only x span; never an authority Selection."""

    origin: PanelInteractionOrigin
    x_span: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        object.__setattr__(
            self, "x_span", optional_display_range(self.x_span, "curve x_span")
        )


CurveInteractionIntent: TypeAlias = CurveViewportCommit | CurveRangeGesture


@dataclass(frozen=True, slots=True)
class HistogramViewportCommit:
    """Request one display-only HISTOGRAM x viewport from an exact front."""

    origin: PanelInteractionOrigin
    viewport: HistogramViewportTransform

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        if not isinstance(self.viewport, HistogramViewportTransform):
            raise TypeError("viewport must be HistogramViewportTransform")
        if self.viewport.display_revision <= self.origin.presentation.panel_revision:
            raise ValueError(
                "histogram viewport commit revision must exceed its painted origin"
            )
        if self.viewport.x_limits_are_auto:
            raise ValueError(
                "histogram viewport commit must carry an explicit authored x pin"
            )


@dataclass(frozen=True, slots=True)
class HistogramRangeGesture:
    """Set or clear one display-only value span; never an analysis threshold."""

    origin: PanelInteractionOrigin
    x_span: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        object.__setattr__(
            self,
            "x_span",
            optional_display_range(self.x_span, "histogram x_span"),
        )


@dataclass(frozen=True, slots=True)
class HistogramThresholdCommit:
    """Author the display threshold cut lines from a live drag step.

    The reference's DragVLine calls back on EVERY motion and the value is
    pure display state on the histogram figure -- never an analysis
    authority.  ``thresholds`` is the COMPLETE authored set after this step.
    """

    origin: PanelInteractionOrigin
    thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        values = tuple(float(value) for value in self.thresholds)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("threshold commit values must be finite")
        object.__setattr__(self, "thresholds", values)


HistogramInteractionIntent: TypeAlias = (
    HistogramViewportCommit | HistogramRangeGesture | HistogramThresholdCommit
)


__all__ = [
    "CurveInteractionIntent",
    "CurveRangeGesture",
    "CurveViewportCommit",
    "HistogramInteractionIntent",
    "HistogramRangeGesture",
    "HistogramThresholdCommit",
    "HistogramViewportCommit",
    "ImageColorLimitsCommit",
    "ImageInteractionCommit",
    "ImageViewportCommit",
    "ImageViewportTransform",
    "NormalizedRectangle",
    "PanelInteractionOrigin",
    "RectangleGesture",
]
