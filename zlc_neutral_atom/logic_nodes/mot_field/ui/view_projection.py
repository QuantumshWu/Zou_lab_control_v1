"""MOT-field adapter into the frontend-owned Figure contract."""

from __future__ import annotations

from zlc_data import AxisSourceRef, DatasetSchema
from zlc_frontend.figure import ViewIntent, ViewPreferences, suggest_view
from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import FigureIntent
from zlc_neutral_atom.runtime.hosted_run import HostedRun

from ..mot_field import (
    MOT_FIELD_FINAL_OUTPUT_DECLARATIONS,
    _MOT_SCAN_COORDINATE_IDS,
)
from ..mot_field_live import MOT_FIELD_LIVE_OUTPUT_DECLARATIONS
from ..mot_field_task import PreparedMotFieldTask


def mot_field_figure_intent(schema: DatasetSchema) -> FigureIntent:
    """Freeze Bx-by-By cells faceted by Bz for one prepared generation."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("MOT-field Figure schema must be DatasetSchema")
    grid_sources = tuple(
        AxisSourceRef.grid_dimension(axis_id)
        for axis_id in _MOT_SCAN_COORDINATE_IDS
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.IMAGE,
        preferences=ViewPreferences(
            image_x_source=grid_sources[0],
            image_y_source=grid_sources[1],
            facet_sources=(grid_sources[2],),
        ),
    )
    if suggestion.spec is None:
        detail = "; ".join(reason.message for reason in suggestion.reasons)
        raise ValueError(f"MOT-field Grid intent is unavailable: {detail}")
    return FigureIntent(
        PlotKind.GRID,
        "MOT field",
        schema.cell_schema.value_unit or "Counts",
        view=suggestion.spec,
    )


def project_mot_field_signal_presentation(
    node,
    output_name,
    _publication,
) -> FigureIntent | None:
    """Return the generation's single eager immutable Figure intent."""

    if output_name not in {
        MOT_FIELD_LIVE_OUTPUT_DECLARATIONS[0].name,
        MOT_FIELD_FINAL_OUTPUT_DECLARATIONS[0].name,
    }:
        return None
    if not isinstance(node, HostedRun):
        raise TypeError("MOT-field presentation requires HostedRun")
    command = node.prepared_command
    if not isinstance(command, PreparedMotFieldTask):
        raise RuntimeError("MOT-field presentation has no prepared generation")
    intent = command._bound_figure_intent()
    if not isinstance(intent, FigureIntent):
        raise TypeError("MOT-field generation retained another presentation type")
    return intent


__all__ = [
    "mot_field_figure_intent",
    "project_mot_field_signal_presentation",
]
