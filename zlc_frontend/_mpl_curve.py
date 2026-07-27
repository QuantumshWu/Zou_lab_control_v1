"""Internal Matplotlib implementation owner: curve."""

from __future__ import annotations

from decimal import Decimal
import math
from numbers import Integral, Number
import numpy as np
from .figure import (
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
)
from .fit_projection import fit_batch_storage_index as _batch_storage_index
from .curve_display import (
    CurveDisplayState,
    CurveViewportTransform,
    NumericDisplayAxis,
    NumericViewportTransform,
    curve_home_x_limits,
    numeric_curve_coordinates,
)
from .display_range import (
    RelimMode,
    deadband_display_range,
    validated_display_range,
)
from .axis_display import axis_label as _axis_label
from .render_style import (
    ANNOTATION_FONT_SIZE,
    CURVE_LINESTYLE,
    CURVE_MARKER,
    FIT_CONTOUR_COLOR,
    FIT_CONTOUR_LINEWIDTH,
    FIT_DIM_ALPHA,
    FIT_FAILURE_COLOR,
    HIST_FILL_ALPHA,
    LINE_CYCLE,
    PALETTE,
    SITE_OCCUPANCY_STYLE,
    apply_title,
    axis_label_fontsize,
    bimodal_fit_line_specs,
    curve_fit_line_kwargs,
    render_style_context,
    small_fontsize,
    threshold_line_kwargs,
    tick_fontsize,
)

from ._mpl_common import (
    _display_series_label,
    _fit_status,
    _series_label,
)

def _curve_values(series):
    data = series.data
    assert isinstance(data, EvaluatedCurve)
    if np.iscomplexobj(data.values):
        raise ValueError(
            "complex curves require an explicit real-valued display transform"
        )
    valid_values = np.asarray(data.values)[np.asarray(data.validity, dtype=bool)]
    if valid_values.size and not bool(np.all(np.isfinite(valid_values))):
        raise ValueError("valid curve values must all be finite")
    return np.ma.array(data.values, mask=~data.validity)

def _curve(axis, layer, cell, series_group, fit_result):
    multiple_series = len(series_group) > 1
    for index, series in enumerate(series_group):
        data = series.data
        assert isinstance(data, EvaluatedCurve)
        x = np.asarray(data.x_axis.coordinates)
        values = _curve_values(series)
        label = _display_series_label(
            layer.layer_id,
            series,
            index,
            multiple_series=multiple_series,
        )
        style = {"color": LINE_CYCLE[index % len(LINE_CYCLE)]}
        source_artist, = axis.plot(
            x,
            values,
            marker=CURVE_MARKER,
            linestyle=CURVE_LINESTYLE,
            label=label,
            **style,
        )
        if fit_result is not None:
            index = _batch_storage_index(fit_result, layer, cell, series)
            if _fit_status(axis, fit_result, index):
                coordinates = np.asarray(data.x_axis.coordinates, dtype=np.float64)
                predicted = fit_result.evaluate_batch(index, (coordinates,))
                axis.plot(
                    coordinates,
                    predicted,
                    label=("fit" if label is None else f"fit {label}"),
                    **curve_fit_line_kwargs(),
                )
                source_artist.set_alpha(FIT_DIM_ALPHA)
    data = series_group[0].data
    axis.set_xlabel(_axis_label(data.x_axis))
    axis.set_ylabel(
        "Signal" if data.value_unit is None else f"Signal ({data.value_unit})"
    )

def _shared_curve_limits(
    panels,
    fit_results,
    *,
    live_grid: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return comparison x/y ranges for a homogeneous source-only curve grid.

    The standalone focused panel deliberately does not consume this range: it
    keeps the ordinary Curve relim policy, matching main's LineCell/GridPlot
    split between comparable thumbnails and a detailed focus view.
    """

    if fit_results or len(panels) <= 1:
        return None
    if len({layer.layer_id for layer, _cell, _series in panels}) != 1:
        return None
    curves = tuple(
        series.data
        for _layer, _cell, series_group in panels
        for series in series_group
    )
    if not curves or any(not isinstance(curve, EvaluatedCurve) for curve in curves):
        return None
    first = curves[0]
    if any(
        curve.x_axis != first.x_axis or curve.value_unit != first.value_unit
        for curve in curves[1:]
    ):
        return None
    try:
        shared_x_limits = curve_home_x_limits(first.x_axis)
    except (TypeError, ValueError):
        # Numeric/strictly-monotonic X is an interactive-CURVE requirement,
        # not a generic Figure requirement.  Categorical/repeated/nonmonotonic
        # curves remain complete encoded panels with ordinary Matplotlib relim.
        return None

    data_min = math.inf
    data_max = -math.inf
    for curve in curves:
        values = np.asarray(curve.values)
        if np.iscomplexobj(values):
            raise ValueError(
                "complex curves require an explicit real-valued display transform"
            )
        valid = np.asarray(curve.validity, dtype=bool)
        if not bool(np.any(valid)):
            continue
        selected = values[valid]
        if not bool(np.all(np.isfinite(selected))):
            raise ValueError("valid curve values must all be finite")
        data_min = min(data_min, float(np.min(selected)))
        data_max = max(data_max, float(np.max(selected)))
    if not math.isfinite(data_min):
        y_limits = (0.0, 1.0)
    elif live_grid:
        padding = 0.08 * (
            (data_max - data_min)
            or abs(data_max)
            or 1.0
        )
        y_limits = (data_min - padding, data_max + padding)
    else:
        y_limits = deadband_display_range(
            RelimMode.TIGHT,
            None,
            data_min,
            data_max,
            force=True,
        )
    return shared_x_limits, y_limits

def _meter_text(series_group) -> str:
    lines = []
    multiple_series = len(series_group) > 1
    for series in series_group:
        data = series.data
        assert isinstance(data, EvaluatedMeter)
        label = _series_label(series, include_reductions=multiple_series) or ""
        if data.valid and not _finite_numeric_scalar(data.value):
            raise ValueError("valid meter values must be finite")
        if data.valid:
            value = (
                "true"
                if data.value is True
                else "false"
                if data.value is False
                else format(Decimal(int(data.value)), ".6g")
                if isinstance(data.value, Integral)
                else format(data.value, ".6g")
            )
            if data.value_unit is not None:
                value = f"{value} {data.value_unit}"
        else:
            value = "invalid"
        lines.append(f"{label}: {value}" if label else value)
    return "\n".join(lines)

def _finite_numeric_scalar(value: Number) -> bool:
    if isinstance(value, Integral):
        return True
    try:
        return bool(np.isfinite(value))
    except TypeError:
        predicate = getattr(value, "is_finite", None)
        if callable(predicate):
            return bool(predicate())
        return bool(math.isfinite(value))

def _meter(axis, series_group):
    axis.set_axis_off()
    axis.text(0.5, 0.5, _meter_text(series_group), ha="center", va="center")

__all__ = [
]
