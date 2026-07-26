"""Headless drawing geometry and immutable state for SiteMap presentation."""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

import numpy as np

from zlc_data import immutable_array

if TYPE_CHECKING:
    from zlc_data import AxisSpec, CoordinateFrameId, Selection
    from .figure import EvaluatedImage, EvaluatedInput
    from .figure_outputs import FigureDerivedSignal
    from .figure_source import FigureSource
    from .image_view import ImageViewportTransform


_SITE_RADIUS_BLOCK = 128

# Painter-neutral physical SiteMap vocabulary.  Agg and Qt own their drawing
# APIs, but neither is allowed to retype these domain-visible facts.
SITE_EMPTY_COLOR = "#FFFFFF"
SITE_EMPTY_ALPHA = 0.85
SITE_EMPTY_LINEWIDTH = 0.6
SITE_OCCUPIED_COLOR = "#D07850"
SITE_OCCUPIED_ALPHA = 0.95
SITE_OCCUPIED_LINEWIDTH = 0.9
SITE_INVALID_COLOR = "#CD7380"
SITE_INVALID_ALPHA = 0.95
SITE_INVALID_LINEWIDTH = SITE_OCCUPIED_LINEWIDTH


@runtime_checkable
class SiteMapPresentation(Protocol):
    """Generic Figure-ready site layer owned by the frontend."""

    background: "EvaluatedImage"
    background_input: "EvaluatedInput"
    home_viewport: "ImageViewportTransform"
    site_axis: "AxisSpec"
    coordinate_frame: "CoordinateFrameId"
    centers_xy: np.ndarray
    site_radius: float
    site_validity: np.ndarray
    run_id: str
    provenance_epoch_id: str
    coherence_identity: str
    summary: str

    @property
    def site_state_input(self) -> "EvaluatedInput": ...

    @property
    def cell_selection(self) -> "Selection": ...

    @property
    def site_geometry_identity(self) -> str: ...

    @property
    def view_identity(self) -> str: ...

    @property
    def site_state(self) -> np.ndarray | None: ...

    @property
    def presentation_kind(self) -> str: ...

    def materialize_area_outputs(
        self,
        source: "FigureSource",
        selection: "Selection",
    ) -> Mapping[str, "FigureDerivedSignal"]: ...


def site_ring_radius(centers_xy: np.ndarray) -> float:
    """Return the established 30%-nearest-neighbour site-ring radius.

    The block traversal avoids an ``(sites, sites, 2)`` temporary while
    preserving the established treatment of duplicate centers: zero-distance
    pairs are ignored and an entirely coincident map uses the 1.5 px floor.
    """

    centers = np.asarray(centers_xy, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1:] != (2,):
        raise ValueError("centers_xy must have shape (sites, 2)")
    if len(centers) < 2 or not np.all(np.isfinite(centers)):
        return 1.5
    block = min(_SITE_RADIUS_BLOCK, len(centers))
    dx = np.empty((block, block), dtype=np.float64)
    dy = np.empty((block, block), dtype=np.float64)
    squared = np.empty((block, block), dtype=np.float64)
    nonpositive = np.empty((block, block), dtype=bool)
    nearest_squared = math.inf
    for left_start in range(0, len(centers), block):
        left = centers[left_start : left_start + block]
        rows = len(left)
        for right_start in range(left_start, len(centers), block):
            right = centers[right_start : right_start + block]
            columns = len(right)
            current_dx = dx[:rows, :columns]
            current_dy = dy[:rows, :columns]
            current_squared = squared[:rows, :columns]
            current_nonpositive = nonpositive[:rows, :columns]
            np.subtract(left[:, None, 0], right[None, :, 0], out=current_dx)
            np.square(current_dx, out=current_squared)
            np.subtract(left[:, None, 1], right[None, :, 1], out=current_dy)
            np.square(current_dy, out=current_dy)
            np.add(current_squared, current_dy, out=current_squared)
            np.less_equal(current_squared, 0.0, out=current_nonpositive)
            current_squared[current_nonpositive] = math.inf
            nearest_squared = min(nearest_squared, float(np.min(current_squared)))
    nearest = math.sqrt(nearest_squared)
    return max(1.5, 0.3 * nearest) if math.isfinite(nearest) else 1.5


def immutable_site_state(
    centers_xy: np.ndarray,
    occupied: np.ndarray,
    site_validity: np.ndarray,
    *,
    site_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structurally freeze one already-admitted SiteMap projection.

    The upstream neutral publication decides whether a site state is
    physically admissible.  The frontend checks only the shape, dtype and
    finite drawing coordinates required by its render payload.
    """

    if isinstance(site_count, bool) or not isinstance(site_count, Integral):
        raise TypeError("site_count must be an integer")
    count = int(site_count)
    if count <= 0:
        raise ValueError("site_count must be positive")
    center_source = np.asarray(centers_xy)
    if center_source.dtype.kind not in "iuf":
        raise TypeError("centers_xy must contain real numeric values")
    centers = immutable_array(
        np.asarray(center_source, dtype=np.dtype("<f8"), order="C"),
        dtype=np.dtype("<f8"),
        shape=(count, 2),
    )
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers_xy must be finite")

    states = []
    for value, field in (
        (occupied, "occupied"),
        (site_validity, "site_validity"),
    ):
        source = np.asarray(value)
        if source.dtype != np.dtype(bool):
            raise TypeError(f"{field} must have bool dtype")
        states.append(
            immutable_array(
                source,
                dtype=np.dtype(bool),
                shape=(count,),
            )
        )
    occupied_array, validity_array = states
    return centers, occupied_array, validity_array


__all__ = [
    "SITE_EMPTY_ALPHA",
    "SITE_EMPTY_COLOR",
    "SITE_EMPTY_LINEWIDTH",
    "SITE_INVALID_ALPHA",
    "SITE_INVALID_COLOR",
    "SITE_INVALID_LINEWIDTH",
    "SITE_OCCUPIED_ALPHA",
    "SITE_OCCUPIED_COLOR",
    "SITE_OCCUPIED_LINEWIDTH",
    "SiteMapPresentation",
    "immutable_site_state",
    "site_ring_radius",
]
