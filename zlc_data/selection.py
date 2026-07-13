"""Serializable, axis-named selection semantics with no presentation state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

from zlc_storage.canonical import canonical_text as _text, decode, encode

from .axis import AxisId, CoordinateFrameId


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
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"coordinate {name} must be a real number")
            scalar = value.item() if hasattr(value, "item") else value
            if isinstance(scalar, Integral):
                normalized: int | float = int(scalar)
            else:
                numeric = float(scalar)
                normalized = int(numeric) if numeric.is_integer() else numeric
            if isinstance(normalized, float) and not math.isfinite(normalized):
                raise ValueError("coordinate range bounds must be finite")
            object.__setattr__(self, name, normalized)
        if self.lower > self.upper:
            raise ValueError("coordinate range lower bound cannot exceed upper bound")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")


SelectionTerm = IndexSelection | IndexRangeSelection | CoordinateRangeSelection
SELECTION_SCHEMA = "zlc_data.Selection/v1"


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
    if not isinstance(tree, dict) or set(tree) != {"schema", "terms"}:
        raise ValueError("Selection must contain exactly schema and terms")
    if tree["schema"] != SELECTION_SCHEMA:
        raise ValueError(f"expected schema {SELECTION_SCHEMA!r}")
    raw_terms = tree["terms"]
    if not isinstance(raw_terms, list):
        raise ValueError("Selection terms must be a list")
    terms: list[SelectionTerm] = []
    for raw in raw_terms:
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
            raise ValueError("selection term must be a tagged map")
        kind = raw["kind"]
        if kind == "INDEX" and set(raw) == {"kind", "axis_id", "index"}:
            terms.append(IndexSelection(AxisId(_text(raw["axis_id"], "axis_id")), raw["index"]))
        elif kind == "INDEX_RANGE" and set(raw) == {
            "kind",
            "axis_id",
            "start",
            "stop",
        }:
            terms.append(
                IndexRangeSelection(
                    AxisId(_text(raw["axis_id"], "axis_id")), raw["start"], raw["stop"]
                )
            )
        elif kind == "COORDINATE_RANGE" and set(raw) == {
            "kind",
            "axis_id",
            "lower",
            "upper",
            "coordinate_frame",
        }:
            frame = raw["coordinate_frame"]
            terms.append(
                CoordinateRangeSelection(
                    AxisId(_text(raw["axis_id"], "axis_id")),
                    raw["lower"],
                    raw["upper"],
                    None if frame is None else CoordinateFrameId(_text(frame, "coordinate_frame")),
                )
            )
        else:
            raise ValueError(f"invalid selection term for kind {kind!r}")
    return Selection(tuple(terms))


def encode_selection(selection: Selection) -> bytes:
    return encode(selection_to_tree(selection))


def decode_selection(payload: bytes) -> Selection:
    selection = selection_from_tree(decode(payload))
    if bytes(payload) != encode_selection(selection):
        raise ValueError("Selection payload uses a non-canonical typed representation")
    return selection


__all__ = [
    "CoordinateRangeSelection",
    "IndexRangeSelection",
    "IndexSelection",
    "Selection",
    "SelectionTerm",
    "decode_selection",
    "encode_selection",
    "selection_from_tree",
    "selection_to_tree",
]
