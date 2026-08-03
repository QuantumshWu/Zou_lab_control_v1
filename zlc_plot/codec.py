"""Plain-tree persistence for the plot specification owned by this package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .kinds import AxisDomain, AxisRef, PlotKind
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    PlotSpec,
    PulseTimelinePlot,
    Reduction,
    RollingPlot,
)


def _mapping(value: Any, keys: set[str], what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    if set(value) != keys:
        raise ValueError(
            f"{what} fields must be exactly {tuple(sorted(keys))!r}"
        )
    return value


def _axis_to_tree(axis: AxisRef | None) -> dict[str, Any] | None:
    if axis is None:
        return None
    if not isinstance(axis, AxisRef):
        raise TypeError("plot axis must be AxisRef or None")
    return {"domain": axis.domain.value, "axis_id": axis.axis_id}


def _axis_from_tree(tree: Any, *, optional: bool = False) -> AxisRef | None:
    if tree is None and optional:
        return None
    data = _mapping(tree, {"domain", "axis_id"}, "plot axis")
    domain = AxisDomain(data["domain"])
    axis_id = data["axis_id"]
    if axis_id is not None and not isinstance(axis_id, str):
        raise TypeError("plot axis_id must be text or None")
    return AxisRef(domain, axis_id)


def _labels_to_tree(labels: PlotLabels) -> dict[str, str | None]:
    return {
        "title": labels.title,
        "x": labels.x,
        "y": labels.y,
        "value": labels.value,
    }


def _labels_from_tree(tree: Any) -> PlotLabels:
    data = _mapping(tree, {"title", "x", "y", "value"}, "plot labels")
    values = {name: data[name] for name in ("title", "x", "y", "value")}
    if any(value is not None and not isinstance(value, str) for value in values.values()):
        raise TypeError("plot labels must be text or None")
    return PlotLabels(**values)


def plot_spec_to_tree(spec: PlotSpec) -> dict[str, Any]:
    """Return a JSON-compatible tree without hashes or duplicate DTOs."""

    common = {"labels": _labels_to_tree(spec.labels)}
    if isinstance(spec, CurvePlot):
        return {
            "kind": PlotKind.CURVE.value,
            "x": _axis_to_tree(spec.x),
            "group": _axis_to_tree(spec.group),
            "reduction": spec.reduction.value,
            **common,
        }
    if isinstance(spec, ImagePlot):
        return {
            "kind": PlotKind.IMAGE.value,
            "x": _axis_to_tree(spec.x),
            "y": _axis_to_tree(spec.y),
            "reduction": spec.reduction.value,
            **common,
        }
    if isinstance(spec, HistogramPlot):
        return {
            "kind": PlotKind.HISTOGRAM.value,
            "samples": [_axis_to_tree(axis) for axis in spec.samples],
            **common,
        }
    if isinstance(spec, RollingPlot):
        return {
            "kind": PlotKind.ROLLING.value,
            "group": _axis_to_tree(spec.group),
            "reduction": spec.reduction.value,
            **common,
        }
    if isinstance(spec, FacetGridPlot):
        return {
            "kind": PlotKind.FACET_GRID.value,
            "facet": _axis_to_tree(spec.facet),
            "cell": plot_spec_to_tree(spec.cell),
            **common,
        }
    if isinstance(spec, PulseTimelinePlot):
        return {"kind": PlotKind.PULSE_TIMELINE.value, **common}
    raise TypeError("unsupported PlotSpec")


def plot_spec_from_tree(tree: Any) -> PlotSpec:
    """Rebuild exactly one typed PlotSpec from its owner-authored tree."""

    if not isinstance(tree, Mapping) or not isinstance(tree.get("kind"), str):
        raise TypeError("plot specification must be a mapping with text kind")
    kind = PlotKind(tree["kind"])
    if kind is PlotKind.CURVE:
        data = _mapping(tree, {"kind", "x", "group", "reduction", "labels"}, "CurvePlot")
        return CurvePlot(
            _axis_from_tree(data["x"]),
            _axis_from_tree(data["group"], optional=True),
            Reduction(data["reduction"]),
            _labels_from_tree(data["labels"]),
        )
    if kind is PlotKind.IMAGE:
        data = _mapping(tree, {"kind", "x", "y", "reduction", "labels"}, "ImagePlot")
        return ImagePlot(
            _axis_from_tree(data["x"]),
            _axis_from_tree(data["y"]),
            Reduction(data["reduction"]),
            _labels_from_tree(data["labels"]),
        )
    if kind is PlotKind.HISTOGRAM:
        data = _mapping(tree, {"kind", "samples", "labels"}, "HistogramPlot")
        samples = data["samples"]
        if not isinstance(samples, list):
            raise TypeError("HistogramPlot samples must be a list")
        return HistogramPlot(
            tuple(_axis_from_tree(item) for item in samples),
            _labels_from_tree(data["labels"]),
        )
    if kind is PlotKind.ROLLING:
        data = _mapping(tree, {"kind", "group", "reduction", "labels"}, "RollingPlot")
        return RollingPlot(
            _axis_from_tree(data["group"], optional=True),
            Reduction(data["reduction"]),
            _labels_from_tree(data["labels"]),
        )
    if kind is PlotKind.FACET_GRID:
        data = _mapping(tree, {"kind", "facet", "cell", "labels"}, "FacetGridPlot")
        cell = plot_spec_from_tree(data["cell"])
        if not isinstance(cell, (CurvePlot, ImagePlot, HistogramPlot)):
            raise ValueError("FacetGrid cell must be Curve, Image, or Histogram")
        return FacetGridPlot(
            _axis_from_tree(data["facet"]),
            cell,
            _labels_from_tree(data["labels"]),
        )
    data = _mapping(tree, {"kind", "labels"}, "PulseTimelinePlot")
    return PulseTimelinePlot(_labels_from_tree(data["labels"]))


__all__ = ["plot_spec_from_tree", "plot_spec_to_tree"]
