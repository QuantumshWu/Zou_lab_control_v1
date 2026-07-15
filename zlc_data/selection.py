"""Serializable, axis-named selection semantics with no presentation state."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from zlc_storage.canonical import exact_mapping as _exact_map

from .axis import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    _canonical_numeric_coordinate,
)


@dataclass(frozen=True)
class IndexSelection:
    """Select one logical index and remove its named axis."""

    axis_id: AxisId
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        if isinstance(self.index, bool) or not isinstance(self.index, Integral):
            raise TypeError("selection index must be an integer")
        if self.index < 0:
            raise ValueError("selection index must be non-negative")
        object.__setattr__(self, "index", int(self.index))


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
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"selection {name} must be an integer")
            object.__setattr__(self, name, int(value))
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
            raise IndexError(f"selection index {term.index} is outside axis {axis.axis_id}")
        return range(term.index, term.index + 1), True
    if isinstance(term, IndexRangeSelection):
        if term.stop > axis.size:
            raise IndexError(f"selection range stop {term.stop} is outside axis {axis.axis_id}")
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
    "resolve_selection_indices",
    "selection_from_tree",
    "selection_to_tree",
]
