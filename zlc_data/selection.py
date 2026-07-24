"""Serializable, axis-named selection semantics with no presentation state."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

from zlc_storage.canonical import (
    exact_mapping as _exact_map,
    integer,
    nonnegative_integer,
)

from ._diagnostic import exact_integer_text
from .axis import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    _canonical_numeric_coordinate,
)
from .layout import PointLayout


@dataclass(frozen=True)
class IndexSelection:
    """Select one logical index and remove its named axis."""

    axis_id: AxisId
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        object.__setattr__(
            self,
            "index",
            nonnegative_integer(self.index, "selection index"),
        )


@dataclass(frozen=True)
class IndexRangeSelection:
    """Retain the half-open logical index interval ``[start, stop)``."""

    axis_id: AxisId
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        for name, value in (("start", self.start), ("stop", self.stop)):
            normalized = integer(value, f"selection {name}")
            assert normalized is not None
            object.__setattr__(self, name, normalized)
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("index range must be a non-empty half-open interval")


@dataclass(frozen=True)
class CoordinateRangeSelection:
    """Retain coordinates in the closed interval ``[lower, upper]``."""

    axis_id: AxisId
    lower: int | float
    upper: int | float
    coordinate_frame: CoordinateFrameId | None

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            object.__setattr__(
                self,
                name,
                _canonical_numeric_coordinate(value, f"coordinate {name}"),
            )
        if self.lower > self.upper:
            raise ValueError("coordinate range lower bound cannot exceed upper bound")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")


SelectionTerm = IndexSelection | IndexRangeSelection | CoordinateRangeSelection
SELECTION_SCHEMA = "zlc_data.Selection"


@dataclass(frozen=True)
class Selection:
    """One immutable selection snapshot over one or more named axes.

    A rectangle is exactly two coordinate-range terms.  Facet scope, widget
    identity and processor bindings deliberately live in their respective
    adapters rather than in this value.
    """

    terms: tuple[SelectionTerm, ...]

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms:
            raise ValueError("Selection requires at least one term")
        if any(
            not isinstance(term, (IndexSelection, IndexRangeSelection, CoordinateRangeSelection))
            for term in terms
        ):
            raise TypeError("Selection contains an unsupported term")
        axis_ids = tuple(term.axis_id for term in terms)
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("Selection may name each AxisId only once")
        object.__setattr__(self, "terms", tuple(sorted(terms, key=lambda term: term.axis_id.value)))

    @classmethod
    def index(cls, axis_id: AxisId, index: int) -> "Selection":
        return cls((IndexSelection(axis_id, index),))

    @classmethod
    def index_range(
        cls,
        axis_id: AxisId,
        start: int,
        stop: int,
    ) -> "Selection":
        return cls((IndexRangeSelection(axis_id, start, stop),))

    @classmethod
    def coordinate_range(
        cls,
        axis_id: AxisId,
        lower: int | float,
        upper: int | float,
        *,
        coordinate_frame: CoordinateFrameId | None,
    ) -> "Selection":
        return cls(
            (CoordinateRangeSelection(axis_id, lower, upper, coordinate_frame),),
        )

    @classmethod
    def rectangle(
        cls,
        x_axis_id: AxisId,
        y_axis_id: AxisId,
        x_lower: int | float,
        x_upper: int | float,
        y_lower: int | float,
        y_upper: int | float,
        *,
        coordinate_frame: CoordinateFrameId | None,
    ) -> "Selection":
        return cls(
            (
                CoordinateRangeSelection(
                    x_axis_id, x_lower, x_upper, coordinate_frame
                ),
                CoordinateRangeSelection(
                    y_axis_id, y_lower, y_upper, coordinate_frame
                ),
            ),
        )


def resolve_selection_indices(
    axis: AxisSpec,
    term: SelectionTerm,
) -> tuple[range | tuple[int, ...], bool]:
    """Resolve one named term without expanding a contiguous logical range."""

    if not isinstance(axis, AxisSpec):
        raise TypeError("axis must be AxisSpec")
    if not isinstance(term, (IndexSelection, IndexRangeSelection, CoordinateRangeSelection)):
        raise TypeError(f"unsupported selection term {type(term).__name__}")
    if term.axis_id != axis.axis_id:
        raise ValueError("selection term axis does not match AxisSpec")
    if isinstance(term, IndexSelection):
        if term.index >= axis.size:
            raise IndexError(
                "selection index "
                f"{exact_integer_text(term.index)} is outside axis "
                f"{axis.axis_id}"
            )
        return range(term.index, term.index + 1), True
    if isinstance(term, IndexRangeSelection):
        if term.stop > axis.size:
            raise IndexError(
                "selection range stop "
                f"{exact_integer_text(term.stop)} is outside axis "
                f"{axis.axis_id}"
            )
        return range(term.start, term.stop), False
    if axis.coordinates is None:
        raise ValueError(f"axis {axis.axis_id} has no coordinates for coordinate selection")
    if axis.coordinate_frame != term.coordinate_frame:
        raise ValueError(f"coordinate frame mismatch for axis {axis.axis_id}")
    if any(
        value is None or isinstance(value, (bool, str)) or not isinstance(value, Real)
        for value in axis.coordinates
    ):
        raise TypeError(f"axis {axis.axis_id} coordinates are not entirely numeric")
    indices = tuple(
        index
        for index, value in enumerate(axis.coordinates)
        if term.lower <= value <= term.upper
    )
    if not indices:
        raise ValueError(f"coordinate selection is empty on axis {axis.axis_id}")
    if indices[-1] - indices[0] + 1 == len(indices):
        return range(indices[0], indices[-1] + 1), False
    return indices, False


def resolve_outer_cell_selection(
    repeat_axis: AxisSpec,
    point_axes: tuple[AxisSpec, ...],
    point_layout: PointLayout,
    selection: Selection | None,
) -> tuple[int, int, tuple[int, ...]]:
    """Resolve one exact named ``(repeat, point)`` cell.

    Missing terms are accepted only for singleton axes.  This is the single
    data-layer owner for mapping an axis-named selection onto the physical
    point storage layout; presentation and experiment packages must not infer
    the address from rank, tuple position, or a first/latest fallback.
    """

    if not isinstance(repeat_axis, AxisSpec):
        raise TypeError("repeat_axis must be AxisSpec")
    axes_after_repeat = tuple(point_axes)
    if any(not isinstance(axis, AxisSpec) for axis in axes_after_repeat):
        raise TypeError("point_axes must contain AxisSpec values")
    if not isinstance(point_layout, PointLayout):
        raise TypeError("point_layout must be PointLayout")
    if point_layout.logical_shape != tuple(axis.size for axis in axes_after_repeat):
        raise ValueError("point_layout logical shape differs from point_axes")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("selection must be Selection or None")
    axes = (repeat_axis, *axes_after_repeat)
    by_axis = {} if selection is None else {
        term.axis_id: term for term in selection.terms
    }
    known = {axis.axis_id for axis in axes}
    if any(axis_id not in known for axis_id in by_axis):
        raise ValueError("dataset cell selection may name only repeat and point axes")
    indices: list[int] = []
    for axis in axes:
        term = by_axis.get(axis.axis_id)
        if term is None:
            if axis.size != 1:
                raise ValueError(
                    f"dataset cell requires an explicit index for axis {axis.axis_id}"
                )
            indices.append(0)
            continue
        if not isinstance(term, IndexSelection):
            raise TypeError("dataset cell selection accepts only IndexSelection terms")
        resolved, drop = resolve_selection_indices(axis, term)
        if not drop or len(resolved) != 1:
            raise ValueError("dataset cell selection must resolve one exact index")
        indices.append(resolved.start)
    logical_point = tuple(indices[1:])
    try:
        point_storage_index = point_layout.storage_index(logical_point)
    except KeyError as error:
        raise ValueError(
            f"selected logical point {logical_point} is absent from PointLayout"
        ) from error
    return indices[0], point_storage_index, logical_point


def selection_for_outer_cell(
    repeat_axis: AxisSpec,
    point_axes: tuple[AxisSpec, ...],
    point_layout: PointLayout,
    repeat_index: int,
    logical_point: tuple[int, ...],
) -> Selection:
    """Build the canonical all-outer-axis selection for one physical cell."""

    if not isinstance(repeat_axis, AxisSpec):
        raise TypeError("repeat_axis must be AxisSpec")
    axes_after_repeat = tuple(point_axes)
    if any(not isinstance(axis, AxisSpec) for axis in axes_after_repeat):
        raise TypeError("point_axes must contain AxisSpec values")
    if not isinstance(point_layout, PointLayout):
        raise TypeError("point_layout must be PointLayout")
    if point_layout.logical_shape != tuple(axis.size for axis in axes_after_repeat):
        raise ValueError("point_layout logical shape differs from point_axes")
    normalized_repeat = integer(repeat_index, "repeat_index")
    assert normalized_repeat is not None
    repeat = normalized_repeat
    if not 0 <= repeat < repeat_axis.size:
        raise IndexError("repeat_index is outside the repeat axis")
    try:
        logical = tuple(logical_point)
    except TypeError as error:
        raise TypeError("logical_point must be an index tuple") from error
    if len(logical) != len(axes_after_repeat):
        raise ValueError("logical_point rank differs from point_axes")
    point_layout.storage_index(logical)
    terms = [IndexSelection(repeat_axis.axis_id, repeat)]
    terms.extend(
        IndexSelection(axis.axis_id, index)
        for axis, index in zip(axes_after_repeat, logical, strict=True)
    )
    selection = Selection(tuple(terms))
    resolved_repeat, _point_storage_index, resolved_logical = (
        resolve_outer_cell_selection(
            repeat_axis,
            axes_after_repeat,
            point_layout,
            selection,
        )
    )
    if resolved_repeat != repeat or resolved_logical != logical:
        raise RuntimeError("canonical dataset-cell selection changed its address")
    return selection


def selection_to_tree(selection: Selection) -> dict[str, Any]:
    if not isinstance(selection, Selection):
        raise TypeError("selection must be Selection")
    terms: list[dict[str, Any]] = []
    for term in selection.terms:
        if isinstance(term, IndexSelection):
            terms.append(
                {"kind": "INDEX", "axis_id": term.axis_id.value, "index": term.index}
            )
        elif isinstance(term, IndexRangeSelection):
            terms.append(
                {
                    "kind": "INDEX_RANGE",
                    "axis_id": term.axis_id.value,
                    "start": term.start,
                    "stop": term.stop,
                }
            )
        elif isinstance(term, CoordinateRangeSelection):
            terms.append(
                {
                    "kind": "COORDINATE_RANGE",
                    "axis_id": term.axis_id.value,
                    "lower": term.lower,
                    "upper": term.upper,
                    "coordinate_frame": None
                    if term.coordinate_frame is None
                    else term.coordinate_frame.value,
                }
            )
        else:  # pragma: no cover - Selection validates the closed union
            raise TypeError(f"unsupported selection term {type(term).__name__}")
    return {"schema": SELECTION_SCHEMA, "terms": terms}


def selection_from_tree(tree: Any) -> Selection:
    data = _exact_map(tree, {"schema", "terms"}, SELECTION_SCHEMA)
    raw_terms = data["terms"]
    if not isinstance(raw_terms, list):
        raise ValueError("Selection terms must be a list")
    terms: list[SelectionTerm] = []
    for raw in raw_terms:
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
            raise ValueError("selection term must be a tagged map")
        kind = raw["kind"]
        if kind == "INDEX":
            item = _exact_map(
                raw,
                {"kind", "axis_id", "index"},
                "INDEX",
                discriminator="kind",
            )
            terms.append(
                IndexSelection(AxisId(item["axis_id"]), item["index"])
            )
        elif kind == "INDEX_RANGE":
            item = _exact_map(
                raw,
                {"kind", "axis_id", "start", "stop"},
                "INDEX_RANGE",
                discriminator="kind",
            )
            terms.append(
                IndexRangeSelection(
                    AxisId(item["axis_id"]),
                    item["start"],
                    item["stop"],
                )
            )
        elif kind == "COORDINATE_RANGE":
            item = _exact_map(
                raw,
                {"kind", "axis_id", "lower", "upper", "coordinate_frame"},
                "COORDINATE_RANGE",
                discriminator="kind",
            )
            frame = item["coordinate_frame"]
            terms.append(
                CoordinateRangeSelection(
                    AxisId(item["axis_id"]),
                    item["lower"],
                    item["upper"],
                    None if frame is None else CoordinateFrameId(frame),
                )
            )
        else:
            raise ValueError(f"invalid selection term for kind {kind!r}")
    return Selection(tuple(terms))


__all__ = [
    "CoordinateRangeSelection",
    "IndexRangeSelection",
    "IndexSelection",
    "Selection",
    "SelectionTerm",
    "resolve_outer_cell_selection",
    "resolve_selection_indices",
    "selection_for_outer_cell",
    "selection_from_tree",
    "selection_to_tree",
]
