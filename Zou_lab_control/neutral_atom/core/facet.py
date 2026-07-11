"""The grid-as-axis-expander facet slicer -- the SINGLE, headless data rule shared by the frontend
GridPlot (which lays a faceted axis out as N aligned cells) and the live FitProcessor (which fits each
of those cells on its worker).  A grid picks ONE axis of a canonical measurement block ``(R, P,
*data_shape)`` and expands it into N cells, each a standard ``sub_plot_kind`` panel of the REMAINING
axes; ``facet_cells`` below is the ONE place "which axis becomes the cells and what each cell shows" is
defined, so the plot and the fit node can never slice differently.

Pure numpy (+ the ``_readout_math`` gap-safe mean): no Qt / Matplotlib / frontend dependency, so the
neutral_atom fit node can import it without breaking the sealed frontend boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from Zou_lab_control._readout_math import finite_mean


_INDEXED_FACET_TOKEN = re.compile(r"(points|data):(0|[1-9][0-9]*)\Z")


def normalize_facet(facet: str) -> tuple[str, int]:
    """Parse the one facet grammar into its internal ``(axis_group, axis_index)`` pair.

    The serialized declaration has exactly three forms: ``"repeat"``, ``"points:k"`` and
    ``"data:k"``, where ``k`` is a canonical non-negative decimal integer.  Absence is represented
    by the caller not invoking this parser; empty strings, implicit axis zero, tuple/list forms and
    every other spelling are invalid rather than aliases for the same declaration.
    """
    if isinstance(facet, str) and facet == "repeat":
        return "repeat", 0
    match = _INDEXED_FACET_TOKEN.fullmatch(facet) if isinstance(facet, str) else None
    if match is None:
        raise ValueError(
            "facet must be exactly 'repeat', 'points:k' or 'data:k' with canonical k >= 0; "
            f"got {facet!r}.")
    return match.group(1), int(match.group(2))


@dataclass(frozen=True)
class _FacetCellLayout:
    """A facet cell's retained axis contract.

    ``image_shape`` is present only when the retained axes have one unambiguous 2-D
    interpretation.  ``curve_size`` is present only when a curve can be formed without
    merging P with non-scalar data or flattening a multidimensional ``data_shape``.
    The auto renderer and the concrete slicer consume this same classification.
    """

    points_shape: tuple[int, ...]
    data_shape: tuple[int, ...]
    image_shape: tuple[int, int] | None
    image_source: str | None
    curve_size: int | None

    @property
    def auto_kind(self) -> str:
        if self.image_shape is not None:
            return "2d"
        if self.curve_size is not None:
            return "1d" if self.curve_size > 1 else "hist"
        raise ValueError(
            "the retained facet axes have no unambiguous automatic renderer: "
            f"points left {self.points_shape}, data_shape left {self.data_shape}.  "
            "Choose another facet or reduce/select explicitly."
        )


def _classify_facet_cell(points_shape, data_shape) -> _FacetCellLayout:
    """Classify retained facet axes without deleting any declared axis."""

    pts = tuple(int(n) for n in points_shape)
    data = tuple(int(n) for n in data_shape)
    point_count = int(np.prod(pts or (1,), dtype=np.int64))
    data_items = int(np.prod(data or (1,), dtype=np.int64))
    scalar_data = data in ((), (1,))

    image_shape = None
    image_source = None
    if len(pts) == 2 and scalar_data:
        image_shape = (pts[0], pts[1])
        image_source = "points"
    elif point_count == 1 and len(data) == 2:
        image_shape = (data[0], data[1])
        image_source = "data"

    curve_size = None
    if len(data) <= 1 and not (point_count > 1 and data_items > 1):
        curve_size = point_count if data_items == 1 else data_items

    return _FacetCellLayout(
        points_shape=pts,
        data_shape=data,
        image_shape=image_shape,
        image_source=image_source,
        curve_size=curve_size,
    )


def _facet_remainder_shapes(facet, *, points_shape=(), data_shape=()):
    """Return the exact logical shapes left after one facet slice."""

    spec = None if facet is None else normalize_facet(facet)
    pts = tuple(int(n) for n in points_shape) or (1,)
    data = tuple(int(n) for n in data_shape) or (1,)
    if spec is None or spec[0] == "repeat":
        return pts, data
    group, index = spec
    shape = pts if group == "points" else data
    if index >= len(shape):
        raise ValueError(f"facet {group}:{index} out of range for {group}_shape {shape}.")
    remainder = tuple(n for i, n in enumerate(shape) if i != index)
    return (remainder, data) if group == "points" else (pts, remainder)


def _collapse_repeat(sliced, repeat_mode: str):
    """Collapse the leading REPEAT axis of a per-cell slice for an image/line cell -- the same display
    verbs the standalone kinds use (``average`` mean / ``add`` sum / ``replace`` latest).  A histogram
    cell never calls this: a distribution POOLS every repeat's samples by definition."""
    mode = str(repeat_mode)
    exact = np.asarray(sliced).dtype.kind in "iub"      # integer slice: no NaN sentinels possible
    if mode == "add":
        return np.asarray(sliced).sum(axis=0) if exact else np.nansum(sliced, axis=0)
    if mode == "replace":
        return np.asarray(sliced[-1])                   # NATIVE dtype passthrough (latest repeat)
    # 'average' (and the fallback for any other verb): an all-NaN cell -- the up-front EMPTY grid a
    # scanning task publishes before its first point, or a not-yet-filled facet plane -- collapses to
    # NaN (a blank cell) BY INTENT, through the ONE gap-safe mean so it never warns per empty cell.
    return np.asarray(sliced).mean(axis=0) if exact else finite_mean(sliced, 0)


def default_sub_plot_kind(facet, *, points_shape=(), data_shape=()) -> str:
    """The natural per-cell kind for a facet slice -- from what each cell has LEFT after the slice
    (the same by-dimensionality rule the console's reshape uses, #H3o): a remaining 2-D points grid
    OR a remaining 2-D ``data_shape`` (a camera frame) is an image cell, more than one remaining value is
    a curve, a bare sample set is a distribution.  The ONE auto rule the console's 'auto'
    sub_plot_kind and the notebook factory share."""
    pts, data = _facet_remainder_shapes(
        facet, points_shape=points_shape, data_shape=data_shape)
    return _classify_facet_cell(pts, data).auto_kind


def facet_cells(value, facet, *, sub_plot_kind: str, points_shape=(), repeat_mode: str = "average"):
    """Slice ONE ``(R, P, *data_shape)`` block along the facet axis into the per-cell inputs a
    ``GridPlot`` of ``sub_plot_kind`` cells displays -- the SINGLE data rule of the grid-as-facet
    design.  Returns ``[cell_0_data, cell_1_data, ...]``.

    ``points_shape`` is the MULTI-D shape of the points axis when the producing scan declared one (the
    node's ``grid_shape`` -- e.g. ``(n_outer, ny, nx)`` for a 3-level scan; the stored block keeps points
    FLAT).  Each cell keeps the iron law's remaining axes and is shaped for its kind:

    * ``hist``  -- the cell's SAMPLES: everything left after the slice, pooled (a distribution pools
      repeats by definition; ``repeat_mode`` is not consulted).
    * ``2d``    -- the cell's FRAME: repeats collapsed by ``repeat_mode``, then the remaining points
      grid (2-D) or the 2-D ``data_shape`` is the image.
    * ``1d``    -- the cell's CURVE: repeats collapsed by ``repeat_mode``; only the physical P axis
      or one remaining data axis may vary.  A P-by-data matrix or multidimensional ``data_shape``
      must be faceted/reduced explicitly and is never flattened into one ambiguous vector.
    """
    spec = None if facet is None else normalize_facet(facet)
    if spec is None:
        raise ValueError("facet_cells needs an explicit facet; None means a recipe (non-faceted) grid.")
    group, index = spec
    a = np.asarray(value)                     # NATIVE dtype (uint8 camera block stays uint8);
    if a.dtype.kind not in "iubf":            # only non-numeric values normalize to float
        a = np.asarray(a, dtype=float)
    if a.ndim < 2:
        raise ValueError(f"a facet grid slices a canonical (R,P,*data_shape) block; got shape {a.shape}.")
    n_repeat, n_points = int(a.shape[0]), int(a.shape[1])
    data_shape = tuple(int(n) for n in a.shape[2:])
    pts_shape = tuple(int(n) for n in (points_shape or (n_points,)))
    if int(np.prod(pts_shape)) != n_points:
        raise ValueError(f"points_shape {pts_shape} does not match the block's points axis ({n_points}).")
    # Expose only the physical P axis in its declared logical point geometry.  The complete trailing
    # data_shape tuple is appended unchanged.
    a = a.reshape((n_repeat, *pts_shape, *data_shape))
    if group == "repeat":
        slices = [a[r] for r in range(n_repeat)]                    # each: (*point_shape, *data_shape)
        pts_rest, data_rest, has_repeat = pts_shape, data_shape, False
    elif group == "points":
        if index >= len(pts_shape):
            raise ValueError(f"facet points:{index} out of range for points_shape {pts_shape}.")
        moved = np.moveaxis(a, 1 + index, 0)                        # (n_i, R, *points_rest, *data_shape)
        slices = [moved[j] for j in range(moved.shape[0])]
        pts_rest = tuple(n for i, n in enumerate(pts_shape) if i != index)
        data_rest, has_repeat = data_shape, True
    else:  # "data"
        if index >= len(data_shape):
            raise ValueError(f"facet data:{index} out of range for data_shape {data_shape}.")
        moved = np.moveaxis(a, 1 + len(pts_shape) + index, 0)       # (n_i, R, *points, *data_rest)
        slices = [moved[j] for j in range(moved.shape[0])]
        pts_rest = pts_shape
        data_rest = tuple(n for i, n in enumerate(data_shape) if i != index)
        has_repeat = True

    kind = str(sub_plot_kind)
    if kind == "hist":
        return [s.reshape(-1) for s in slices]                      # a distribution pools everything left
    layout = _classify_facet_cell(pts_rest, data_rest)
    cells = []
    for s in slices:
        core = _collapse_repeat(s, repeat_mode) if has_repeat else np.asarray(s, dtype=float)
        if kind == "2d":
            # Reshape only to the exact 2-D geometry identified by the shared classifier.  This
            # removes structural scalar P/data cells, never arbitrary singleton data axes.
            if layout.image_shape is None:
                raise ValueError(
                    f"a 2d facet cell needs an exact 2-D frame: a points grid with scalar data or an exact "
                    f"2-D data_shape with scalar P; got shape {core.shape} "
                    f"(points left {pts_rest}, data_shape left {data_rest}).")
            cells.append(core.reshape(layout.image_shape))
        else:  # "1d"
            if layout.curve_size is None:
                raise ValueError(
                    "a 1d facet cell cannot merge P with data_shape or flatten multidimensional "
                    f"data_shape; points left {pts_rest}, data_shape left {data_rest}.  "
                    "Choose data:k or reduce/select explicitly.")
            cells.append(core.reshape(layout.curve_size))
    return cells


__all__ = [
    "normalize_facet", "default_sub_plot_kind", "facet_cells",
    "_FacetCellLayout", "_classify_facet_cell", "_facet_remainder_shapes", "_collapse_repeat",
]
