"""Display-only physical site map for one exact committed occupancy cell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
from io import BytesIO
import math

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import EllipseCollection
from matplotlib.figure import Figure

from zlc_storage import canonical_text, positive_integer

from .encoded_raster import EncodedRasterDocument, EncodedRasterPage
from .image_raster import png_raster_size
from .matplotlib_render import release_agg_figure, site_ring_radius
from .render_style import (
    FIT_FAILURE_COLOR,
    PALETTE,
    SITE_OCCUPANCY_STYLE,
    apply_title,
    render_style_context,
)


_SCREEN_DPI = 150
_RASTER_WIDTH = 1300
_RASTER_HEIGHT = 1000
_RENDER_FIXED_BYTES = 8 << 20
_RASTER_PEAK_MULTIPLIER = 8
_ARRAY_PEAK_MULTIPLIER = 8
_VIEW_FIXED_BYTES = 1 << 20
_RADIUS_WORKSPACE_BYTES = 1 << 20
_SITE_COLLECTION_PEAK_BYTES = 512


def _owned_array(
    value,
    dtype,
    shape: tuple[int, ...],
    field: str,
    *,
    finite: bool = False,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


def estimate_occupancy_cell_view_retained_nbytes(
    frame_shape: tuple[int, int],
    frame_dtype: np.dtype | str,
    site_count: int,
) -> int:
    """Conservatively bound the self-contained exact-cell presentation value."""

    shape = tuple(frame_shape)
    if (
        len(shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise ValueError("frame_shape must contain two positive integers")
    dtype = np.dtype(frame_dtype)
    if dtype.hasobject or dtype.kind == "c":
        raise TypeError("physical site-map frames must be real numeric arrays")
    sites = positive_integer(site_count, "site_count")
    pixels = math.prod(shape)
    # Frame + expanded pixel validity.  Site vectors are retained once between
    # sequential repository phases and copied once into the cross-thread view.
    return (
        _VIEW_FIXED_BYTES
        + pixels * (dtype.itemsize + np.dtype(bool).itemsize)
        + 2 * sites * (2 * np.dtype("<f8").itemsize + 2)
    )


@dataclass(frozen=True, eq=False)
class OccupancyCellView:
    """Self-contained physical facts for one exact ``(repeat, point)`` cell."""

    frame: np.ndarray
    frame_validity: np.ndarray
    centers_xy: np.ndarray
    occupied: np.ndarray
    site_validity: np.ndarray
    cell_label: str
    summary: str
    value_unit: str | None = None

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame)
        if frame.ndim != 2 or frame.dtype.hasobject or frame.dtype.kind == "c":
            raise ValueError("frame must be one real two-dimensional image")
        frame = _owned_array(frame, frame.dtype, frame.shape, "frame")
        frame_validity = _owned_array(
            self.frame_validity,
            bool,
            frame.shape,
            "frame_validity",
        )
        centers = np.asarray(self.centers_xy)
        if centers.ndim != 2 or centers.shape[1:] != (2,):
            raise ValueError("centers_xy must have shape (sites, 2)")
        centers = _owned_array(
            centers,
            "<f8",
            centers.shape,
            "centers_xy",
            finite=True,
        )
        sites = centers.shape[0]
        occupied = _owned_array(self.occupied, bool, (sites,), "occupied")
        site_validity = _owned_array(
            self.site_validity,
            bool,
            (sites,),
            "site_validity",
        )
        if np.any(occupied[~site_validity]):
            raise ValueError("invalid sites require canonical False fillers")
        canonical_text(self.cell_label, "cell_label")
        canonical_text(self.summary, "occupancy cell summary")
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "frame_validity", frame_validity)
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "site_validity", site_validity)

    @property
    def array_nbytes(self) -> int:
        return sum(
            int(value.nbytes)
            for value in (
                self.frame,
                self.frame_validity,
                self.centers_xy,
                self.occupied,
                self.site_validity,
            )
        )


def _add_site_ring_collection(
    axis,
    centers_xy: np.ndarray,
    mask: np.ndarray,
    *,
    radius: float,
    color: str,
    alpha: float,
    linewidth: float,
    linestyle: str,
) -> None:
    offsets = centers_xy[mask]
    if not len(offsets):
        return
    collection = EllipseCollection(
        (2.0 * radius,),
        (2.0 * radius,),
        (0.0,),
        units="xy",
        offsets=offsets,
        offset_transform=axis.transData,
        facecolors="none",
        edgecolors=color,
        alpha=alpha,
        linewidths=linewidth,
        linestyles=linestyle,
        zorder=5,
    )
    axis.add_collection(collection)


def _build_site_map(view: OccupancyCellView, figure: Figure) -> None:
    axis = figure.add_subplot(1, 1, 1)
    cmap = colormaps[PALETTE["cmap_camera"]].copy()
    cmap.set_bad(FIT_FAILURE_COLOR)
    visible = np.isfinite(view.frame) & view.frame_validity
    image = np.ma.array(view.frame, mask=~visible)
    artist = axis.imshow(image, cmap=cmap, origin="upper", interpolation="nearest")
    radius = site_ring_radius(view.centers_xy)
    empty = view.site_validity & ~view.occupied
    occupied = view.site_validity & view.occupied
    invalid = ~view.site_validity
    for mask, style, linestyle in (
        (empty, SITE_OCCUPANCY_STYLE["empty"], "-"),
        (occupied, SITE_OCCUPANCY_STYLE["occupied"], "-"),
        (
            invalid,
            {
                "color": FIT_FAILURE_COLOR,
                "alpha": 0.95,
                "linewidth": SITE_OCCUPANCY_STYLE["occupied"]["linewidth"],
            },
            "--",
        ),
    ):
        _add_site_ring_collection(
            axis,
            view.centers_xy,
            mask,
            radius=radius,
            color=str(style["color"]),
            alpha=float(style["alpha"]),
            linewidth=float(style["linewidth"]),
            linestyle=linestyle,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("camera x [px]")
    axis.set_ylabel("camera y [px]")
    apply_title(axis, view.cell_label)
    colorbar = figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("camera signal" if view.value_unit is None else view.value_unit)
    valid_count = int(np.count_nonzero(view.site_validity))
    occupied_count = int(np.count_nonzero(view.occupied & view.site_validity))
    axis.text(
        0.01,
        0.99,
        f"occupied {occupied_count}/{valid_count} valid | invalid {len(view.site_validity) - valid_count}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
        zorder=6,
    )


def render_occupancy_cell(
    view: OccupancyCellView,
    *,
    memory_limit_bytes: int,
    source_retained_upper_bound_bytes: int,
    checkpoint: Callable[[], None] | None = None,
) -> EncodedRasterDocument:
    """Render one exact same-shot physical map without reducing any dataset axis."""

    if not isinstance(view, OccupancyCellView):
        raise TypeError("view must be OccupancyCellView")
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    source_retained = positive_integer(
        source_retained_upper_bound_bytes,
        "source_retained_upper_bound_bytes",
    )
    if source_retained < view.array_nbytes:
        raise ValueError("source retained bound is smaller than the projected arrays")
    check = (lambda: None) if checkpoint is None else checkpoint
    if not callable(check):
        raise TypeError("checkpoint must be callable or None")
    required = (
        _RENDER_FIXED_BYTES
        + source_retained
        + _RADIUS_WORKSPACE_BYTES
        + _SITE_COLLECTION_PEAK_BYTES * len(view.site_validity)
        + _ARRAY_PEAK_MULTIPLIER * view.array_nbytes
        + _RASTER_PEAK_MULTIPLIER * _RASTER_WIDTH * _RASTER_HEIGHT * 4
    )
    if required > limit:
        raise MemoryError(
            f"occupancy cell raster composition requires {required} bytes; limit is {limit}"
        )
    figure = None
    payload = b""
    with render_style_context():
        try:
            check()
            figure = Figure(
                figsize=(_RASTER_WIDTH / _SCREEN_DPI, _RASTER_HEIGHT / _SCREEN_DPI),
                dpi=_SCREEN_DPI,
                constrained_layout=True,
            )
            FigureCanvasAgg(figure)
            _build_site_map(view, figure)
            target = BytesIO()
            figure.savefig(target, format="png", dpi=_SCREEN_DPI)
            payload = target.getvalue()
            check()
        finally:
            if figure is not None:
                release_agg_figure(figure)
            figure = None
            gc.collect()
    png_raster_size(payload)
    valid = int(np.count_nonzero(view.site_validity))
    occupied = int(np.count_nonzero(view.occupied & view.site_validity))
    summary = (
        f"{view.summary}\n"
        f"occupied={occupied}/{valid} valid sites | invalid={len(view.site_validity) - valid}"
    )
    return EncodedRasterDocument(
        summary,
        (EncodedRasterPage("exact-cell", "Exact cell", payload),),
    )


__all__ = [
    "OccupancyCellView",
    "estimate_occupancy_cell_view_retained_nbytes",
    "render_occupancy_cell",
]
