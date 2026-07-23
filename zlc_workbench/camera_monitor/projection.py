"""Headless typed projection for the camera-monitor window."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import MONITOR_HISTORY, SPATIAL_X, SPATIAL_Y, Selection
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    SuggestionStatus,
    ViewIntent,
    ViewSpec,
    suggest_view,
    validate_view_spec,
)
from zlc_frontend.histogram_display import DEFAULT_HISTOGRAM_BINS
from zlc_frontend.image_view import ImageViewportTransform
from zlc_neutral_atom.monitor_application import PreparedCameraMonitor
from zlc_neutral_atom.processing.roi_monitor import RoiScalarBinding


IMAGE_PANEL_ID = "camera-monitor-image"
CURVE_PANEL_ID = "camera-monitor-roi-curve"
HISTOGRAM_PANEL_ID = "camera-monitor-roi-histogram"
METER_PANEL_ID = "camera-monitor-roi-meter"
SCALAR_PANEL_IDS = (
    CURVE_PANEL_ID,
    HISTOGRAM_PANEL_ID,
    METER_PANEL_ID,
)
RAW_PROJECTION_TEXT = "latest raw frame · history slot 0 · DISPLAY ONLY"


def roi_scalar_view_specs(schema) -> tuple[ViewSpec, ViewSpec, ViewSpec]:
    axes = (schema.repeat_axis, *schema.point_axes, *schema.cell_schema.data_axes)
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    if len(history) != 1 or schema.cell_schema.data_axes:
        raise ValueError("ROI scalar views require one scalar MONITOR_HISTORY axis")
    repeat_binding = AxisViewBinding(
        schema.repeat_axis.axis_id,
        AxisViewRole.SELECTED,
        selector=FixedIndex(0),
    )
    curve = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(history[0].axis_id, AxisViewRole.X),
            repeat_binding,
        ),
    )
    histogram = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(history[0].axis_id, AxisViewRole.SAMPLE),
            repeat_binding,
        ),
    )
    meter = ViewSpec(
        schema.fingerprint,
        ViewIntent.METER,
        (
            AxisViewBinding(
                history[0].axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            repeat_binding,
        ),
    )
    views = (curve, histogram, meter)
    if any(len(view.axis_bindings) != len(axes) for view in views):
        raise ValueError("ROI scalar views refuse undeclared extra axes")
    for view in views:
        validate_view_spec(schema, view)
    return views


def roi_scalar_views(schema, binding):
    views = roi_scalar_view_specs(schema)
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    assert len(history) == 1
    input_axes = {
        axis.axis_id: axis for axis in binding.input_contract.value_schema.data_axes
    }
    terms = {term.axis_id: term for term in binding.selection.terms}
    description = ", ".join(
        f"{input_axes[axis_id].name}={term.lower}..{term.upper}"
        for axis_id, term in sorted(terms.items(), key=lambda item: item[0].value)
    )
    summary = (
        f"latest raw frame + ROI {binding.reduction.value.lower()} scalar "
        f"[{description}] · binding {binding.fingerprint[:12]} · "
        f"validity {binding.validity_policy.value.lower()} · "
        f"scalar history 0..{history[0].size - 1} (0 newest) · "
        f"curve + histogram (default {DEFAULT_HISTOGRAM_BINS} bins) + "
        "latest meter · "
        "MONITOR DERIVED / DISPLAY ONLY"
    )
    return views, summary


def scalar_documents(
    schema,
    binding,
    dataset_id: DatasetId,
    *,
    identity: str,
) -> tuple[tuple[FigureDocument, ...], str]:
    """Build the three admitted documents for one applied scalar generation."""

    views, projection_text = roi_scalar_views(schema, binding)
    descriptor = DatasetDescriptor(
        dataset_id,
        f"ROI {binding.reduction.value.lower()} scalar monitor",
        schema.fingerprint,
    )
    documents = tuple(
        FigureDocument(
            f"{panel_id}-{identity}",
            0,
            (descriptor,),
            (FigureLayer(panel_id, dataset_id, view),),
        )
        for panel_id, view in zip(SCALAR_PANEL_IDS, views, strict=True)
    )
    return documents, projection_text


@dataclass(frozen=True, slots=True)
class MonitorViewProjection:
    """One fully checked headless view product awaiting a single Run start."""

    command: PreparedCameraMonitor
    generation: int
    dataset_id: DatasetId
    scalar_dataset_id: DatasetId | None
    image_document: FigureDocument
    scalar_documents: tuple[FigureDocument, ...]
    viewport: ImageViewportTransform
    projection_text: str


def prepare_monitor_view_projection(
    command: PreparedCameraMonitor,
    generation: int,
) -> MonitorViewProjection:
    """Finish source-lifetime display admission before the camera is armed."""

    if not isinstance(command, PreparedCameraMonitor):
        raise TypeError("command must be PreparedCameraMonitor")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("monitor view generation must be a positive integer")
    schema = command.view_schema
    scalar_schema = command.scalar_view_schema
    roi_binding = command.roi_binding
    history = tuple(axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY)
    y_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_Y
    )
    x_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_X
    )
    if (
        len(history) != 1
        or len(y_axes) != 1
        or len(x_axes) != 1
        or len(schema.cell_schema.data_axes) != 2
    ):
        raise ValueError(
            "camera monitor requires one history axis and declared "
            "SPATIAL_Y/SPATIAL_X image axes"
        )
    selection = Selection.index(history[0].axis_id, 0)
    suggestion = suggest_view(schema, ViewIntent.IMAGE, selection)
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        raise ValueError("camera monitor IMAGE view needs an explicit axis choice")
    image_view = suggestion.spec
    scalar_views: tuple[ViewSpec, ...] = ()
    projection_text = RAW_PROJECTION_TEXT
    viewport = ImageViewportTransform(
        (y_axes[0], x_axes[0]),
        viewport_revision=0,
    )
    if not command.roi_control_schemas:
        raise RuntimeError("camera monitor exposes no ROI control schema")
    for possible_schema in command.roi_control_schemas:
        roi_scalar_view_specs(possible_schema)
    if scalar_schema is None:
        if roi_binding is not None or command.request.roi is not None:
            raise RuntimeError("ROI request has no scalar view product")
    else:
        if roi_binding is None or command.request.roi is None:
            raise RuntimeError("scalar view product has no admitted ROI binding")
        if (
            roi_binding.selection != command.request.roi
            or roi_binding.reduction is not command.request.roi_reduction
        ):
            raise RuntimeError("prepared ROI binding differs from its immutable request")
        scalar_views, projection_text = roi_scalar_views(scalar_schema, roi_binding)
        viewport.normalized_bounds_for_selection(roi_binding.selection)

    dataset_id = DatasetId(f"camera-monitor-{generation}")
    scalar_dataset_id = (
        None
        if scalar_schema is None
        else DatasetId(f"camera-monitor-roi-scalar-{generation}")
    )
    image_document = FigureDocument(
        f"camera-monitor-image-{generation}",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "Raw monitor camera frame",
                schema.fingerprint,
            ),
        ),
        (FigureLayer(IMAGE_PANEL_ID, dataset_id, image_view),),
    )
    documents: tuple[FigureDocument, ...] = ()
    if scalar_views:
        assert scalar_schema is not None and scalar_dataset_id is not None
        assert roi_binding is not None
        documents, checked_projection = scalar_documents(
            scalar_schema,
            roi_binding,
            scalar_dataset_id,
            identity=str(generation),
        )
        if checked_projection != projection_text:
            raise RuntimeError("ROI scalar projection summary changed during preparation")
    return MonitorViewProjection(
        command,
        generation,
        dataset_id,
        scalar_dataset_id,
        image_document,
        documents,
        viewport,
        projection_text,
    )


__all__ = [
    "CURVE_PANEL_ID",
    "HISTOGRAM_PANEL_ID",
    "IMAGE_PANEL_ID",
    "METER_PANEL_ID",
    "MonitorViewProjection",
    "RAW_PROJECTION_TEXT",
    "SCALAR_PANEL_IDS",
    "prepare_monitor_view_projection",
    "roi_scalar_view_specs",
    "roi_scalar_views",
    "scalar_documents",
]
