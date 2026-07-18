"""Headless, front-bound selection gestures for regular pixel images.

Coordinate and viewport math has one owner in :mod:`zlc_frontend.image_view`.
This module only binds a completed selection gesture to the exact immutable
front on which it was drawn; it owns no Qt, Matplotlib, runtime control, or
analysis authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from zlc_storage import canonical_text, nonnegative_integer

from .figure import EvaluatedInput
from .image_display import validated_image_range
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
class ImageInteractionOrigin:
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

    origin: ImageInteractionOrigin
    viewport: ImageViewportTransform

    def __post_init__(self) -> None:
        if not isinstance(self.origin, ImageInteractionOrigin):
            raise TypeError("origin must be ImageInteractionOrigin")
        if not isinstance(self.viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if self.viewport.viewport_revision <= self.origin.presentation.panel_revision:
            raise ValueError("viewport commit revision must exceed its painted origin")


@dataclass(frozen=True, slots=True)
class ImageColorLimitsCommit:
    """Request FIXED colour limits without mutating the painted QImage LUT."""

    origin: ImageInteractionOrigin
    color_limits: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, ImageInteractionOrigin):
            raise TypeError("origin must be ImageInteractionOrigin")
        object.__setattr__(
            self,
            "color_limits",
            validated_image_range(self.color_limits, "color_limits"),
        )


ImageInteractionCommit: TypeAlias = ImageViewportCommit | ImageColorLimitsCommit


__all__ = [
    "ImageColorLimitsCommit",
    "ImageInteractionCommit",
    "ImageInteractionOrigin",
    "ImageViewportCommit",
    "ImageViewportTransform",
    "NormalizedRectangle",
    "RectangleGesture",
]
