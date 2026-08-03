"""Small convenience constructors over the typed public plotting API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from zlc_data import OwnedSnapshot

from .kinds import AxisRef
from .primitives import ImageFrame, ImagePointOverlay, PulseTimelineData
from .session import PlotSession
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    PulseTimelinePlot,
    Reduction,
    RollingPlot,
)

if TYPE_CHECKING:
    from .notebook import NotebookView


def _axis(value: AxisRef | str) -> AxisRef:
    if isinstance(value, AxisRef):
        return value
    if isinstance(value, str):
        return AxisRef.point(value)
    raise TypeError("axis must be AxisRef or a PointTable column name")


def _with_parameter_alias(
    parameters: Mapping[str, object] | None,
    name: str,
    value: object | None,
    *,
    alias: str,
) -> dict[str, object]:
    """Normalize one convenience keyword into the parameter state once."""

    if parameters is None:
        values: dict[str, object] = {}
    elif isinstance(parameters, Mapping):
        values = dict(parameters)
    else:
        raise TypeError("parameters must be a mapping or None")
    if value is None:
        return values
    if name in values:
        raise ValueError(
            f"{alias} conflicts with parameters[{name!r}]; provide only one value"
        )
    values[name] = value
    return values


def curve(
    data: OwnedSnapshot,
    x: AxisRef | str,
    *,
    group: AxisRef | str | None = None,
    reduction: Reduction = Reduction.MEAN,
    labels: PlotLabels | None = None,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    **session_options: Any,
) -> PlotSession:
    """Create a Curve session; text axes mean explicit PointTable columns."""

    return PlotSession(
        data,
        CurvePlot(
            _axis(x),
            None if group is None else _axis(group),
            reduction,
            labels or PlotLabels(),
        ),
        size=size,
        parameters=parameters,
        **session_options,
    )


def histogram(
    data: OwnedSnapshot,
    *,
    samples: Iterable[AxisRef | str] | None = None,
    bins: int | None = None,
    labels: PlotLabels | None = None,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    **session_options: Any,
) -> PlotSession:
    sample_refs = (
        (AxisRef.repeat(), AxisRef.point_rows())
        if samples is None
        else tuple(_axis(sample) for sample in samples)
    )
    values = _with_parameter_alias(parameters, "bin_count", bins, alias="bins")
    return PlotSession(
        data,
        HistogramPlot(sample_refs, labels or PlotLabels()),
        size=size,
        parameters=values,
        **session_options,
    )


def image(
    data: OwnedSnapshot | ImageFrame,
    x: AxisRef | str,
    y: AxisRef | str,
    *,
    reduction: Reduction = Reduction.MEAN,
    overlay: ImagePointOverlay | None = None,
    labels: PlotLabels | None = None,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    **session_options: Any,
) -> PlotSession:
    if isinstance(data, ImageFrame):
        if overlay is not None:
            raise ValueError("overlay is already owned by the supplied ImageFrame")
        source = data
    else:
        source = data if overlay is None else ImageFrame(data, overlay)
    return PlotSession(
        source,
        ImagePlot(_axis(x), _axis(y), reduction, labels or PlotLabels()),
        size=size,
        parameters=parameters,
        **session_options,
    )


def rolling(
    data: OwnedSnapshot,
    *,
    group: AxisRef | str | None = None,
    reduction: Reduction = Reduction.MEAN,
    side_distribution: bool | None = None,
    window: int | None = None,
    labels: PlotLabels | None = None,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    **session_options: Any,
) -> PlotSession:
    values = _with_parameter_alias(
        parameters,
        "side_distribution",
        side_distribution,
        alias="side_distribution",
    )
    values = _with_parameter_alias(values, "window", window, alias="window")
    return PlotSession(
        data,
        RollingPlot(
            None if group is None else _axis(group),
            reduction,
            labels=labels or PlotLabels(),
        ),
        size=size,
        parameters=values,
        **session_options,
    )


def facet_grid(
    data: OwnedSnapshot,
    facet: AxisRef | str,
    cell: CurvePlot | ImagePlot | HistogramPlot,
    *,
    labels: PlotLabels | None = None,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    **session_options: Any,
) -> PlotSession:
    return PlotSession(
        data,
        FacetGridPlot(_axis(facet), cell, labels or PlotLabels()),
        size=size,
        parameters=parameters,
        **session_options,
    )


def pulse_timeline(
    data: PulseTimelineData,
    *,
    labels: PlotLabels | None = None,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    **session_options: Any,
) -> PlotSession:
    return PlotSession(
        data,
        PulseTimelinePlot(labels or PlotLabels()),
        size=size,
        parameters=parameters,
        **session_options,
    )


def show(
    session: PlotSession,
    *,
    close_session_on_close: bool = False,
) -> "NotebookView":
    """Display one session in Jupyter without backend magic or pyplot state."""

    if not isinstance(session, PlotSession):
        raise TypeError("session must be PlotSession")
    from .notebook import NotebookView

    view = NotebookView(
        session,
        close_session_on_close=close_session_on_close,
    )
    view.display()
    return view


__all__ = [
    "curve",
    "facet_grid",
    "histogram",
    "image",
    "pulse_timeline",
    "rolling",
    "show",
]
