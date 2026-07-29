"""Pure exact projections from one frozen CURVE fit to render DTOs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from numbers import Number

import numpy as np

from zlc_data import (
    DatasetSchema,
    DatasetRevisionRef,
    FitBatchStatus,
    FitResultBatch,
)
from zlc_data._arrays import immutable_array
from zlc_data.fit import validate_fit_result_source_binding
from zlc_storage import canonical_text

from .figure import (
    AxisViewRole,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedInput,
    EvaluatedSeries,
    FigureDocument,
    ViewIntent,
)
from .figure.contract import _fit_display_selection_indices
from .fit_projection import fit_batch_storage_index
from .render import CurveFitOverlay


@dataclass(frozen=True, slots=True)
class _CurveFitOverlayPlanEntry:
    series: EvaluatedSeries
    source_sample_span: tuple[int, int]
    batch_storage_index: int | None
    status: FitBatchStatus | None
    diagnostic: str


@dataclass(frozen=True, slots=True, eq=False)
class CurveFitOverlayPlan:
    """Validated exact overlay work with no allocated prediction arrays."""

    source_ref: DatasetRevisionRef
    result_identity: str
    _result: FitResultBatch = field(repr=False)
    _entries: tuple[_CurveFitOverlayPlanEntry, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("curve fit plan source_ref must be DatasetRevisionRef")
        identity = canonical_text(self.result_identity, "curve fit result identity")
        object.__setattr__(self, "result_identity", identity)
        if not isinstance(self._result, FitResultBatch):
            raise TypeError("curve fit plan result must be FitResultBatch")
        if self._result.source_ref != self.source_ref:
            raise ValueError("curve fit plan result belongs to another input")
        entries = tuple(self._entries)
        if not entries or any(
            not isinstance(item, _CurveFitOverlayPlanEntry) for item in entries
        ):
            raise ValueError("curve fit plan requires validated series entries")
        object.__setattr__(self, "_entries", entries)


def transient_single_panel_curve_fit_overlay_plan(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    source_schema: DatasetSchema,
    result: FitResultBatch,
    *,
    result_identity: str,
) -> CurveFitOverlayPlan:
    """Validate transient replay and freeze sparse/status/span work only."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")
    if len(document.layers) != 1 or len(evaluated.layers) != 1:
        raise ValueError("typed curve fit projection requires exactly one layer")
    if len(evaluated.inputs) != 1:
        raise ValueError("typed curve fit projection requires exactly one input")

    document_layer = document.layers[0]
    layer = evaluated.layers[0]
    if (
        layer.layer_id != document_layer.layer_id
        or layer.dataset_id != document_layer.dataset_id
    ):
        raise ValueError("evaluated curve layer differs from its document layer")
    if len(layer.cells) != 1:
        raise ValueError("typed curve fit projection requires exactly one cell")
    evaluated_input = evaluated.inputs[0]
    if (
        not isinstance(evaluated_input, EvaluatedInput)
        or evaluated_input.dataset_id != layer.dataset_id
    ):
        raise ValueError("typed curve fit projection input differs from its layer")
    if result.source_ref != evaluated_input.ref:
        raise ValueError("curve fit result belongs to another source revision")
    validate_fit_result_source_binding(result, evaluated_input.ref, source_schema)
    if len(result.fit_axis_specs) != 1:
        raise ValueError("typed CURVE projection requires exactly one fitted axis")
    if (
        document_layer.view.intent is not ViewIntent.CURVE
        or document_layer.view.binding(result.spec.independent_sources[0]).role
        is not AxisViewRole.X
    ):
        raise ValueError("cached CURVE view does not bind the fitted x axis")
    allowed_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
    }
    if any(
        document_layer.view.binding(source).role not in allowed_batch_roles
        for source in result.spec.batch_sources
    ):
        raise ValueError("cached CURVE view does not uniquely bind fit batch axes")

    cell = layer.cells[0]
    entries: list[_CurveFitOverlayPlanEntry] = []
    for series in cell.series:
        curve = series.data
        if not isinstance(curve, EvaluatedCurve):
            raise ValueError("typed curve fit projection requires only CURVE series")
        if curve.x_axis.source != result.spec.independent_sources[0]:
            raise ValueError("evaluated curve x axis differs from fitted axis")
        source_span = _curve_source_span(
            source_schema,
            document_layer.view,
            layer.resolutions,
            result,
            curve,
        )
        storage_index = fit_batch_storage_index(result, layer, cell, series)
        if storage_index is None:
            status = None
            diagnostic = "NOT_PRESENT"
        else:
            status = result.statuses[storage_index]
            if status is FitBatchStatus.CONVERGED:
                start, stop = source_span
                for coordinate in curve.x_axis.coordinates[start:stop]:
                    if (
                        isinstance(coordinate, (bool, np.bool_))
                        or not isinstance(coordinate, Number)
                        or not math.isfinite(float(coordinate))
                    ):
                        raise TypeError(
                            "fitted CURVE coordinates must be finite real numbers"
                        )
                diagnostic = ""
            else:
                diagnostic = status.value
                if result.errors[storage_index]:
                    diagnostic = f"{diagnostic}: {result.errors[storage_index]}"
        entries.append(
            _CurveFitOverlayPlanEntry(
                series,
                source_span,
                storage_index,
                status,
                diagnostic,
            )
        )
    if not entries:
        raise ValueError("typed curve fit projection produced no series")
    return CurveFitOverlayPlan(
        evaluated_input.ref,
        result_identity,
        result,
        tuple(entries),
    )


def _panel_prediction_point_share(
    maximum_prediction_points: int | None,
    painted_path_count: int,
) -> int | None:
    """Divide one optional panel budget among its painted prediction paths."""

    if maximum_prediction_points is not None:
        if (
            isinstance(maximum_prediction_points, bool)
            or not isinstance(maximum_prediction_points, (int, np.integer))
        ):
            raise TypeError("maximum_prediction_points must be an integer or None")
        maximum_prediction_points = int(maximum_prediction_points)
        if maximum_prediction_points < 2:
            raise ValueError("maximum_prediction_points must be at least two")
    if painted_path_count < 0:
        raise ValueError("painted_path_count must be non-negative")
    if maximum_prediction_points is None:
        return None
    if painted_path_count == 0:
        return 0
    if maximum_prediction_points < 2 * painted_path_count:
        raise ValueError(
            "panel prediction budget must provide two points per painted path"
        )
    return maximum_prediction_points // painted_path_count


def materialize_curve_fit_overlay_plan(
    plan: CurveFitOverlayPlan,
    *,
    maximum_prediction_points: int | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[CurveFitOverlay, ...]:
    """Evaluate predictions under one optional panel-wide point budget."""

    if not isinstance(plan, CurveFitOverlayPlan):
        raise TypeError("plan must be CurveFitOverlayPlan")
    point_share = _panel_prediction_point_share(
        maximum_prediction_points,
        sum(
            entry.status is FitBatchStatus.CONVERGED
            for entry in plan._entries
        ),
    )
    if check_cancelled is not None and not callable(check_cancelled):
        raise TypeError("check_cancelled must be callable or None")
    result = plan._result
    projected: list[CurveFitOverlay] = []
    for entry in plan._entries:
        if check_cancelled is not None:
            check_cancelled()
        predicted = np.empty((0,), dtype=np.dtype("<f8"))
        fit_coordinates = None
        if entry.status is FitBatchStatus.CONVERGED:
            assert entry.batch_storage_index is not None
            curve = entry.series.data
            assert isinstance(curve, EvaluatedCurve)
            start, stop = entry.source_sample_span
            coordinates = np.asarray(
                curve.x_axis.coordinates[start:stop],
                dtype=np.dtype("<f8"),
            )
            if point_share is not None and coordinates.size > point_share:
                positions = (
                    np.arange(point_share, dtype=np.int64)
                    * (coordinates.size - 1)
                    // (point_share - 1)
                )
                coordinates = coordinates[positions]
            evaluated_prediction = np.asarray(
                result.evaluate_batch(
                    entry.batch_storage_index,
                    (coordinates,),
                )
            )
            if np.iscomplexobj(evaluated_prediction):
                raise ValueError("fit prediction must be real-valued")
            evaluated_prediction = np.asarray(
                evaluated_prediction,
                dtype=np.dtype("<f8"),
            )
            prediction_shape = coordinates.shape
            if evaluated_prediction.shape != prediction_shape:
                raise ValueError("fit prediction shape differs from CURVE fit span")
            if not bool(np.all(np.isfinite(evaluated_prediction))):
                raise ValueError("converged fit prediction must be finite")
            predicted = immutable_array(
                evaluated_prediction,
                dtype=np.dtype("<f8"),
                shape=prediction_shape,
            )
            if maximum_prediction_points is not None:
                fit_coordinates = coordinates
        projected.append(
            CurveFitOverlay(
                plan.source_ref,
                plan.result_identity,
                entry.series.batch_address,
                entry.source_sample_span,
                entry.batch_storage_index,
                entry.status,
                entry.diagnostic,
                predicted,
                coordinates=fit_coordinates,
            )
        )
    return tuple(projected)


def _curve_source_span(
    source_schema: DatasetSchema,
    view,
    resolutions,
    result: FitResultBatch,
    curve: EvaluatedCurve,
) -> tuple[int, int]:
    """Resolve the authority ROI into one contiguous cached-curve span."""

    source = result.spec.independent_sources[0]
    selected = dict(
        _fit_display_selection_indices(
            source_schema,
            view,
            resolutions,
            result,
        )
    ).get(source)
    if selected is None:
        positions = tuple(range(curve.values.size))
    else:
        position_by_index = {
            source_index: position
            for position, source_index in enumerate(curve.x_axis.indices)
        }
        try:
            positions = tuple(position_by_index[index] for index in selected)
        except KeyError as exc:
            raise ValueError(
                "CURVE Figure does not contain every selected Fit sample"
            ) from exc
    if not positions or positions != tuple(range(positions[0], positions[-1] + 1)):
        raise ValueError(
            "CURVE Fit Selection is not one contiguous span in the displayed axis"
        )
    start, stop = positions[0], positions[-1] + 1
    fit_axis = result.fit_axis_specs[0]
    coordinates = curve.x_axis.coordinates[start:stop]
    if len(coordinates) != fit_axis.size or any(
        coordinate != fit_axis.coordinate_at(index)
        for index, coordinate in enumerate(coordinates)
    ):
        raise ValueError("CURVE Fit coordinates differ from the displayed source span")
    return (start, stop)


__all__ = [
    "CurveFitOverlayPlan",
    "materialize_curve_fit_overlay_plan",
    "transient_single_panel_curve_fit_overlay_plan",
]
