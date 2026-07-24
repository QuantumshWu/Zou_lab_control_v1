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

from zlc_storage import canonical_text, finite_real, nonnegative_integer

from .figure import EvaluatedInput
from .curve_display import CurveViewportTransform, NumericViewportTransform
from .display_range import optional_display_range, validated_display_range
from .histogram_display import HistogramViewportTransform
from .image_view import (
    ImageViewportTransform,
    NormalizedRectangle,
    validate_normalized_rectangle,
)
from .render import (
    DocumentInputIdentity,
    PanelPresentationIdentity,
    SourceIdentity,
)


@dataclass(frozen=True, slots=True)
class RectangleGesture:
    """Set or clear one rectangle on the exact front where it was drawn.

    ``None`` is the image-family equivalent of the numeric range gestures'
    cleared span: a fresh left click that never forms a non-degenerate box.
    Existing-box move/resize gestures retain their non-degenerate initial
    rectangle, so an unmoved handle click is not misclassified as a clear.
    Non-``None`` bounds are complete-raster source-relative Area coordinates,
    never visible-window coordinates or an unbounded viewport.
    """

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    source_identity: SourceIdentity
    normalized_bounds: NormalizedRectangle | None
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
        if self.normalized_bounds is not None:
            object.__setattr__(
                self,
                "normalized_bounds",
                validate_normalized_rectangle(self.normalized_bounds),
            )


@dataclass(frozen=True, slots=True)
class PanelInteractionOrigin:
    """The exact painted front against which one display intent was authored.

    Dataset and document identity are a closed, matched pair.  There is no
    optional dataset provenance slot for document-backed surfaces to fake or
    leave half-populated.
    """

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    source_identity: SourceIdentity | DocumentInputIdentity
    presentation: PanelPresentationIdentity
    input_identity: EvaluatedInput | DocumentInputIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", canonical_text(self.panel_id, "panel_id"))
        object.__setattr__(self, "board_id", canonical_text(self.board_id, "board_id"))
        for field in ("layout_generation", "sequence"):
            object.__setattr__(
                self,
                field,
                nonnegative_integer(getattr(self, field), field),
            )
        if not isinstance(
            self.source_identity, (SourceIdentity, DocumentInputIdentity)
        ):
            raise TypeError(
                "source_identity must be SourceIdentity or DocumentInputIdentity"
            )
        if not isinstance(self.presentation, PanelPresentationIdentity):
            raise TypeError(
                "presentation must be zlc_frontend.render.PanelPresentationIdentity"
            )
        if self.presentation.panel_id != self.panel_id:
            raise ValueError("interaction presentation belongs to another panel")
        if isinstance(self.source_identity, DocumentInputIdentity):
            if not isinstance(self.input_identity, DocumentInputIdentity):
                raise TypeError(
                    "document interaction requires DocumentInputIdentity"
                )
            if self.input_identity != self.source_identity:
                raise ValueError(
                    "interaction document differs from its source identity"
                )
            if (
                self.presentation.document_id
                != self.source_identity.document_id
                or self.presentation.document_revision
                != self.source_identity.document_revision
            ):
                raise ValueError(
                    "interaction presentation differs from its document identity"
                )
            return
        if not isinstance(self.input_identity, EvaluatedInput):
            raise TypeError("dataset interaction requires EvaluatedInput")
        if self.input_identity.dataset_id != self.source_identity.dataset_id:
            raise ValueError("interaction input belongs to another dataset")
        if (
            self.input_identity.ref.block_id != self.source_identity.block_id
            or self.input_identity.ref.stream_generation
            != self.source_identity.stream_generation
            or self.input_identity.ref.schema_fingerprint
            != self.source_identity.schema_fingerprint
        ):
            raise ValueError("interaction input differs from its source identity")


@dataclass(frozen=True, slots=True)
class CrossGesture:
    """Set or clear one Cross cursor on an exact immutable painted front.

    ``point`` is the continuous physical coordinate at the completed right
    click.  ``None`` is authored only by a double-right-click and clears the
    Cross.  The gesture is never emitted from pointer motion and therefore is
    not a hover channel.
    """

    origin: PanelInteractionOrigin
    point: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        point = self.point
        if point is None:
            return
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError("cross point must be a pair or None")
        object.__setattr__(
            self,
            "point",
            (
                finite_real(point[0], "cross x"),
                finite_real(point[1], "cross y"),
            ),
        )


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
    """Request one display-only numeric x viewport from an exact painted front.

    Dataset curves retain their :class:`CurveViewportTransform`; a pulse
    document uses the authority-free :class:`NumericViewportTransform`.
    """

    origin: PanelInteractionOrigin
    viewport: NumericViewportTransform

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        if not isinstance(self.viewport, NumericViewportTransform):
            raise TypeError("viewport must be NumericViewportTransform")
        if isinstance(self.origin.source_identity, DocumentInputIdentity):
            if type(self.viewport) is not NumericViewportTransform:
                raise TypeError(
                    "document viewport commit requires NumericViewportTransform"
                )
        elif not isinstance(self.viewport, CurveViewportTransform):
            raise TypeError(
                "dataset curve viewport commit requires CurveViewportTransform"
            )
        if self.viewport.display_revision <= self.origin.presentation.panel_revision:
            raise ValueError(
                "numeric viewport commit revision must exceed its painted origin"
            )


@dataclass(frozen=True, slots=True)
class CurveRangeGesture:
    """Set or clear one display-only numeric x span; never an authority Selection."""

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
    "CrossGesture",
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
