"""Pure exact projections from one frozen CURVE fit to render DTOs."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from numbers import Number

import numpy as np

from zlc_data import (
    DatasetSchema,
    DatasetRevisionRef,
    FitBatchStatus,
    FitResultBatch,
    immutable_array,
    resolve_selection_indices,
    validate_fit_result_source_binding,
)
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
from .figure.contract import _selection_fit_projection, dataset_axes
from .fit_image_projection import fit_batch_storage_index
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
        if len(identity) > 4096:
            raise ValueError("curve fit result identity exceeds its display bound")
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


def single_panel_curve_fit_overlay_plan(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: Mapping[str, FitResultBatch],
    source_schema: DatasetSchema,
    *,
    result_identity: str,
) -> CurveFitOverlayPlan:
    """Validate one canonical curve result without evaluating its model."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    supplied = dict(fit_results)
    if len(document.layers) != 1 or set(supplied) != {document.layers[0].layer_id}:
        raise ValueError("typed curve fit projection requires one exact layer result")
    result = supplied[document.layers[0].layer_id]
    return transient_single_panel_curve_fit_overlay_plan(
        document,
        evaluated,
        source_schema,
        result,
        result_identity=result_identity,
    )


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
        or document_layer.view.binding(result.fit_axis_specs[0].axis_id).role
        is not AxisViewRole.X
    ):
        raise ValueError("cached CURVE view does not bind the fitted x axis")
    allowed_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
        AxisViewRole.SLIDER,
    }
    if any(
        document_layer.view.binding(axis.axis_id).role not in allowed_batch_roles
        and not (
            axis.size == 1
            and document_layer.view.binding(axis.axis_id).role
            is AxisViewRole.REDUCED
        )
        for axis in result.batch_axis_specs
    ):
        raise ValueError("cached CURVE view does not uniquely bind fit batch axes")

    cell = layer.cells[0]
    entries: list[_CurveFitOverlayPlanEntry] = []
    for series in cell.series:
        curve = series.data
        if not isinstance(curve, EvaluatedCurve):
            raise ValueError("typed curve fit projection requires only CURVE series")
        if curve.x_axis.axis_id != result.fit_axis_specs[0].axis_id:
            raise ValueError("evaluated curve x axis differs from fitted axis")
        source_span = _curve_source_span(source_schema, result, curve)
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


def materialize_curve_fit_overlay_plan(
    plan: CurveFitOverlayPlan,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[CurveFitOverlay, ...]:
    """Evaluate predictions only after the caller has admitted ``plan``."""

    if not isinstance(plan, CurveFitOverlayPlan):
        raise TypeError("plan must be CurveFitOverlayPlan")
    if check_cancelled is not None and not callable(check_cancelled):
        raise TypeError("check_cancelled must be callable or None")
    result = plan._result
    projected: list[CurveFitOverlay] = []
    for entry in plan._entries:
        if check_cancelled is not None:
            check_cancelled()
        predicted = np.empty((0,), dtype=np.dtype("<f8"))
        if entry.status is FitBatchStatus.CONVERGED:
            assert entry.batch_storage_index is not None
            curve = entry.series.data
            assert isinstance(curve, EvaluatedCurve)
            start, stop = entry.source_sample_span
            coordinates = np.asarray(
                curve.x_axis.coordinates[start:stop],
                dtype=np.dtype("<f8"),
            )
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
            prediction_shape = (stop - start,)
            if evaluated_prediction.shape != prediction_shape:
                raise ValueError("fit prediction shape differs from CURVE fit span")
            if not bool(np.all(np.isfinite(evaluated_prediction))):
                raise ValueError("converged fit prediction must be finite")
            predicted = immutable_array(
                evaluated_prediction,
                dtype=np.dtype("<f8"),
                shape=prediction_shape,
            )
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
            )
        )
    return tuple(projected)


def _curve_source_span(
    source_schema: DatasetSchema,
    result: FitResultBatch,
    curve: EvaluatedCurve,
) -> tuple[int, int]:
    """Resolve the authority ROI into one contiguous cached-curve span."""

    transform = result.spec.committed_transform
    if transform is None:
        return (0, curve.values.size)
    _effective_schema, authority_selection = _selection_fit_projection(
        source_schema,
        result,
    )
    fit_axis_id = result.fit_axis_specs[0].axis_id
    terms = tuple(
        term for term in authority_selection.terms if term.axis_id == fit_axis_id
    )
    if len(terms) != 1:
        raise ValueError("transient CURVE fit requires one authority range on x")
    source_axis = next(
        axis for axis in dataset_axes(source_schema) if axis.axis_id == fit_axis_id
    )
    selected, _drop = resolve_selection_indices(source_axis, terms[0])
    selected_count = len(selected)
    first_selected = selected[0]
    start: int | None = None
    matched = 0

    def is_selected(raw_index: int) -> bool:
        if isinstance(selected, range):
            return raw_index in selected
        position = bisect_left(selected, raw_index)
        return position < selected_count and selected[position] == raw_index

    for position, raw_index in enumerate(curve.x_axis.indices):
        if start is None:
            if raw_index == first_selected:
                start = position
                matched = 1
            elif is_selected(raw_index):
                raise ValueError(
                    "committed fit range is not contiguous in cached CURVE view"
                )
            continue
        if matched < selected_count:
            if raw_index != selected[matched]:
                raise ValueError(
                    "committed fit range is not contiguous in cached CURVE view"
                )
            matched += 1
        elif is_selected(raw_index):
            raise ValueError("cached CURVE fit-range indices are not unique")
    if start is None or matched != selected_count:
        raise ValueError(
            "cached CURVE view does not contain the complete committed fit range"
        )
    stop = start + selected_count
    fit_axis = result.fit_axis_specs[0]
    if fit_axis.size != selected_count or any(
        curve.x_axis.coordinates[start + offset] != fit_axis.coordinate_at(offset)
        for offset in range(selected_count)
    ):
        raise ValueError("cached CURVE fit-range coordinates differ from FitResult")
    return (start, stop)


def curve_fit_overlays_retained_nbytes(
    overlays: tuple[CurveFitOverlay, ...],
) -> tuple[int, int]:
    """Return conservative retained DTO bytes and exact prediction-array bytes.

    The first value includes Python/container/text/address overhead and the
    immutable prediction buffers.  The second is separated so Agg admission
    can additionally charge the artist/model-to-line scratch term without
    double-counting the retained DTO ownership.
    """

    overlays = tuple(overlays)
    if any(not isinstance(item, CurveFitOverlay) for item in overlays):
        raise TypeError("overlays must contain CurveFitOverlay values")
    prediction_bytes = sum(int(item.predicted_y.nbytes) for item in overlays)
    retained = prediction_bytes
    for item in overlays:
        retained += 1024
        retained += len(item.result_identity.encode("utf-8"))
        retained += len(item.diagnostic.encode("utf-8"))
        retained += 256 * len(item.series_batch_address)
    return int(retained), int(prediction_bytes)


def estimate_curve_fit_overlay_plan_nbytes(
    plan: CurveFitOverlayPlan,
) -> tuple[int, int]:
    """Return retained plan/DTO bytes and prediction bytes before evaluation."""

    if not isinstance(plan, CurveFitOverlayPlan):
        raise TypeError("plan must be CurveFitOverlayPlan")
    prediction_bytes = sum(
        (entry.source_sample_span[1] - entry.source_sample_span[0])
        * np.dtype("<f8").itemsize
        for entry in plan._entries
        if entry.status is FitBatchStatus.CONVERGED
    )
    retained = prediction_bytes + 4096 + 512 * len(plan._entries)
    identity_bytes = len(plan.result_identity.encode("utf-8"))
    for entry in plan._entries:
        retained += 1024 + identity_bytes
        retained += len(entry.diagnostic.encode("utf-8"))
        retained += 256 * len(entry.series.batch_address)
    return int(retained), int(prediction_bytes)


__all__ = [
    "CurveFitOverlayPlan",
    "curve_fit_overlays_retained_nbytes",
    "estimate_curve_fit_overlay_plan_nbytes",
    "materialize_curve_fit_overlay_plan",
    "single_panel_curve_fit_overlay_plan",
    "transient_single_panel_curve_fit_overlay_plan",
]
