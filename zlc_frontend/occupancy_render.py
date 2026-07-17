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

from zlc_data import (
    AxisLayout,
    AxisSpec,
    IndexSelection,
    PointLayout,
    Selection,
    StreamGenerationId,
    resolve_selection_indices,
)
from zlc_storage import canonical_text, positive_integer, sha256_text

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
_NAVIGATION_UI_FIXED_BYTES = 1 << 20
_NAVIGATION_UI_AXIS_BYTES = 64 << 10


def estimate_occupancy_navigation_retained_nbytes(
    inspection_retained_upper_bound_bytes: int,
    axis_count: int,
) -> int:
    """Bound the long-lived FINAL metadata and its bounded Qt axis controls."""

    inspection = positive_integer(
        inspection_retained_upper_bound_bytes,
        "inspection_retained_upper_bound_bytes",
    )
    if isinstance(axis_count, bool) or not isinstance(axis_count, int) or axis_count <= 0:
        raise ValueError("axis_count must be a positive integer")
    return inspection + _NAVIGATION_UI_FIXED_BYTES + axis_count * _NAVIGATION_UI_AXIS_BYTES


@dataclass(frozen=True)
class OccupancyCellNavigation:
    """Frozen outer-axis identity for navigating one committed occupancy artifact."""

    artifact_identity: str
    schema_fingerprint: str
    generation: StreamGenerationId
    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    cell_layout: AxisLayout
    retained_upper_bound_bytes: int

    def __post_init__(self) -> None:
        canonical_text(self.artifact_identity, "artifact_identity")
        sha256_text(self.schema_fingerprint, "schema_fingerprint")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if not isinstance(self.repeat_axis, AxisSpec):
            raise TypeError("repeat_axis must be AxisSpec")
        point_axes = tuple(self.point_axes)
        if any(not isinstance(axis, AxisSpec) for axis in point_axes):
            raise TypeError("point_axes must contain AxisSpec values")
        axes = (self.repeat_axis, *point_axes)
        if len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("occupancy navigation axes must have unique AxisId values")
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        expected_shape = tuple(axis.size for axis in point_axes)
        if self.point_layout.logical_shape != expected_shape:
            raise ValueError("point_layout logical shape differs from point axes")
        if not isinstance(self.cell_layout, AxisLayout):
            raise TypeError("cell_layout must be AxisLayout")
        expected_cell_shape = (self.repeat_axis.size, *expected_shape)
        if (
            self.cell_layout.logical_shape != expected_cell_shape
            or self.cell_layout.storage_size
            != self.repeat_axis.size * self.point_layout.storage_size
        ):
            raise ValueError("cell_layout differs from repeat and point layout")
        positive_integer(
            self.retained_upper_bound_bytes,
            "retained_upper_bound_bytes",
        )
        object.__setattr__(self, "point_axes", point_axes)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return (self.repeat_axis, *self.point_axes)

    @property
    def identity(self) -> tuple[str, str, StreamGenerationId]:
        return (
            self.artifact_identity,
            self.schema_fingerprint,
            self.generation,
        )

    @property
    def linear_cell_count(self) -> int:
        return self.cell_layout.storage_size

    def resolve_selection(
        self,
        selection: Selection | None,
    ) -> tuple[int, int, tuple[int, ...], str]:
        """Resolve one exact named cell without first/latest/reduce fallbacks."""

        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        by_axis = {} if selection is None else {
            term.axis_id: term for term in selection.terms
        }
        known = {axis.axis_id for axis in self.axes}
        if any(axis_id not in known for axis_id in by_axis):
            raise ValueError(
                "occupancy cell selection may name only repeat and point axes"
            )
        indices = []
        labels = []
        for axis in self.axes:
            term = by_axis.get(axis.axis_id)
            if term is None:
                if axis.size != 1:
                    raise ValueError(
                        f"occupancy cell requires an explicit index for axis {axis.axis_id}"
                    )
                index = 0
            else:
                if not isinstance(term, IndexSelection):
                    raise TypeError(
                        "occupancy cell selection accepts only exact IndexSelection terms"
                    )
                resolved, drop = resolve_selection_indices(axis, term)
                if not drop or len(resolved) != 1:
                    raise ValueError("occupancy cell selection must resolve one exact index")
                index = resolved.start
            coordinate = axis.coordinate_at(index)
            unit = "" if axis.unit is None else f" {axis.unit}"
            labels.append(f"{axis.name}={coordinate}{unit} [index {index}]")
            indices.append(index)
        logical_point = tuple(indices[1:])
        try:
            point_storage_index = self.point_layout.storage_index(logical_point)
        except KeyError as error:
            raise ValueError(
                f"selected logical point {logical_point} is absent from PointLayout"
            ) from error
        return (
            indices[0],
            point_storage_index,
            logical_point,
            " | ".join(labels),
        )

    def selection_for_indices(
        self,
        repeat_index: int,
        logical_point: tuple[int, ...],
    ) -> Selection:
        """Build the canonical all-axis exact selection for one logical cell."""

        logical = tuple(logical_point)
        self.point_layout.storage_index(logical)
        terms = [IndexSelection(self.repeat_axis.axis_id, repeat_index)]
        terms.extend(
            IndexSelection(axis.axis_id, index)
            for axis, index in zip(self.point_axes, logical, strict=True)
        )
        selection = Selection(tuple(terms))
        self.resolve_selection(selection)
        return selection

    def selection_at_linear(self, linear_index: int) -> Selection:
        """Map repeat-major physical order to a canonical exact selection."""

        if (
            isinstance(linear_index, bool)
            or not isinstance(linear_index, int)
            or not 0 <= linear_index < self.linear_cell_count
        ):
            raise IndexError("occupancy navigation index is out of range")
        multi = self.cell_layout.multi_index(linear_index)
        return self.selection_for_indices(multi[0], tuple(multi[1:]))

    def linear_index(self, selection: Selection) -> int:
        repeat_index, _point_storage_index, logical, _label = self.resolve_selection(
            selection
        )
        return self.cell_layout.storage_index((repeat_index, *logical))


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
    "OccupancyCellNavigation",
    "OccupancyCellView",
    "estimate_occupancy_cell_view_retained_nbytes",
    "estimate_occupancy_navigation_retained_nbytes",
    "render_occupancy_cell",
]
