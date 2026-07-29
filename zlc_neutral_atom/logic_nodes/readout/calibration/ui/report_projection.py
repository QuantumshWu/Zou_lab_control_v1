"""Project Calibration facts into the generic frontend Plot Report contract.

This optional UI leaf chooses only domain-significant page intent and labels.
Size, DPR, style, renderer, raster encoding, and backend lifecycle remain owned
by :mod:`zlc_frontend`.
"""

from __future__ import annotations

import math

import numpy as np

from zlc_data.axis import AxisSourceRef
from zlc_data.value import OwnedSnapshot
from zlc_frontend import (
    AxisViewRole,
    CurveDisplayState,
    FacetedHistogramDisplayState,
    FixedIndex,
    HistogramCellThresholds,
    HistogramDisplayState,
    FigureSource,
    FigureIntent,
    PanelProvenance,
    PlotKind,
    PlotReportDocument,
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
    plot_report_page,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.histogram_display import histogram_display_with_thresholds
from zlc_storage import canonical_digest

from ..projection import (
    CalibrationModelReportProjection,
    CalibrationReportProjection,
    CalibrationSiteMapContext,
    materialize_calibration_report_datasets,
)
from .view_projection import calibration_site_map_figure


def _format_metric(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def _provenance(
    view: CalibrationReportProjection,
    key: str,
    snapshot: OwnedSnapshot,
) -> PanelProvenance:
    return PanelProvenance(
        f"calibration-report-{view.calibration_identity}",
        snapshot.ref.stream_generation.value,
        canonical_digest(
            {
                "owner": "zlc_neutral_atom.calibration-report-front",
                "calibration_identity": view.calibration_identity,
                "key": key,
                "block_id": snapshot.ref.block_id.value,
                "revision": snapshot.ref.revision.value,
                "stream_generation": snapshot.ref.stream_generation.value,
                "schema_fingerprint": snapshot.ref.schema_fingerprint,
            }
        ),
    )


def _site_map_page(view: CalibrationReportProjection, snapshot: OwnedSnapshot):
    figure, source = calibration_site_map_figure(
        snapshot,
        CalibrationSiteMapContext(
            view.site_axis,
            view.coordinate_frame,
            view.actual_centers_xy,
            view.site_validity,
            view.calibration_identity,
        ),
        coherence_identity=view.source_capture_identity,
        run_id=f"calibration-report-{view.calibration_identity}",
        provenance_epoch_id=snapshot.ref.stream_generation.value,
    )
    return plot_report_page(
        "overview",
        figure=figure,
        source=source,
        provenance=_provenance(view, "overview", snapshot),
    )


def _histogram_page(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    snapshot: OwnedSnapshot,
):
    schema = snapshot.block.schema
    if len(schema.point_table.columns) != 1 or len(schema.cell_schema.data_axes) != 1:
        raise ValueError("calibration histogram Dataset has an invalid declared shape")
    site_column = schema.point_table.columns[0]
    population_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(schema.repeat_axis.axis_id),
                AxisViewRole.SAMPLE,
            ),
            SourceViewBinding(
                AxisSourceRef.point_coordinate(site_column.coordinate_id),
                AxisViewRole.FACET,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(population_axis.axis_id),
                AxisViewRole.BATCH,
            ),
        ),
    )
    base_display = HistogramDisplayState(
        bin_count=max(5, int(np.asarray(model.bin_edges).size) - 1),
    )
    cell_thresholds = tuple(
        HistogramCellThresholds(
            (
                (
                    AxisSourceRef.point_coordinate(site_column.coordinate_id),
                    site,
                ),
            ),
            (
                (float(model.runtime_thresholds[site]),)
                if math.isfinite(float(model.runtime_thresholds[site]))
                else ()
            ),
        )
        for site in range(schema.point_table.row_count)
    )
    faceted = len(view.site_labels) > 1
    display = (
        FacetedHistogramDisplayState(base_display, cell_thresholds)
        if faceted
        else histogram_display_with_thresholds(
            base_display,
            cell_thresholds[0].thresholds,
        )
    )
    return plot_report_page(
        f"hist-{model.label}",
        figure=FigureIntent(
            PlotKind.GRID if faceted else PlotKind.HISTOGRAM,
            f"{model.label} | per-site readout distributions",
            view.frame_schema.value_unit or "Signal",
            view=spec,
        ),
        source=FigureSource(snapshot),
        display=display,
        provenance=_provenance(view, f"hist-{model.label}", snapshot),
    )


def _pooled_page(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    snapshot: OwnedSnapshot,
):
    schema = snapshot.block.schema
    if (
        schema.point_table.row_count != 1
        or schema.point_table.columns
        or len(schema.cell_schema.data_axes) != 1
    ):
        raise ValueError("calibration pooled Dataset has an invalid declared shape")
    population_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(schema.repeat_axis.axis_id),
                AxisViewRole.SAMPLE,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(population_axis.axis_id),
                AxisViewRole.BATCH,
            ),
        ),
    )
    return plot_report_page(
        f"pooled-{model.label}",
        figure=FigureIntent(
            PlotKind.HISTOGRAM,
            f"{model.label} | threshold-centred populations",
            (
                f"{view.frame_schema.value_unit} - runtime threshold"
                if view.frame_schema.value_unit
                else "Signal - runtime threshold"
            ),
            view=spec,
        ),
        source=FigureSource(snapshot),
        display=HistogramDisplayState(
            bin_count=60,
            thresholds=(0.0,),
        ),
        provenance=_provenance(view, f"pooled-{model.label}", snapshot),
    )


def _fidelity_page(view: CalibrationReportProjection, snapshot: OwnedSnapshot):
    schema = snapshot.block.schema
    if len(schema.point_table.columns) != 1 or len(schema.cell_schema.data_axes) != 1:
        raise ValueError("calibration fidelity Dataset has an invalid declared shape")
    site_column = schema.point_table.columns[0]
    model_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(schema.repeat_axis.axis_id),
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            SourceViewBinding(
                AxisSourceRef.point_coordinate(site_column.coordinate_id),
                AxisViewRole.X,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(model_axis.axis_id),
                AxisViewRole.BATCH,
            ),
        ),
    )
    return plot_report_page(
        "fidelity",
        figure=FigureIntent(
            PlotKind.CURVE,
            "Per-site model fidelity",
            "Fidelity",
            view=spec,
        ),
        source=FigureSource(snapshot),
        display=CurveDisplayState(
            relim_mode=RelimMode.FIXED,
            fixed_y_limits=(0.45, 1.01),
        ),
        provenance=_provenance(view, "fidelity", snapshot),
    )


def _psf_page(
    view: CalibrationReportProjection,
    snapshot: OwnedSnapshot | None,
):
    if snapshot is None:
        return None
    schema = snapshot.block.schema
    if len(schema.point_table.columns) != 1 or len(schema.cell_schema.data_axes) != 2:
        raise ValueError("calibration PSF Dataset has an invalid declared shape")
    site_column = schema.point_table.columns[0]
    y_axis, x_axis = schema.cell_schema.data_axes
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.IMAGE,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(schema.repeat_axis.axis_id),
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            SourceViewBinding(
                AxisSourceRef.point_coordinate(site_column.coordinate_id),
                AxisViewRole.FACET,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(y_axis.axis_id),
                AxisViewRole.IMAGE_Y,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(x_axis.axis_id),
                AxisViewRole.IMAGE_X,
            ),
        ),
    )
    return plot_report_page(
        "psf-kernels",
        figure=FigureIntent(
            PlotKind.GRID if len(view.site_labels) > 1 else PlotKind.IMAGE,
            "Empirical PSF kernels",
            "Normalized weight",
            view=spec,
        ),
        source=FigureSource(snapshot),
        provenance=_provenance(view, "psf-kernels", snapshot),
    )


def project_calibration_plot_report(
    view: CalibrationReportProjection,
) -> PlotReportDocument:
    """Return typed pages; rendering and encoding remain frontend-owned."""

    if not isinstance(view, CalibrationReportProjection):
        raise TypeError("view must be CalibrationReportProjection")
    datasets = materialize_calibration_report_datasets(view)
    pages = [
        _site_map_page(view, datasets["reference"]),
        _fidelity_page(view, datasets["fidelity"]),
    ]
    for model in view.models:
        pages.append(_histogram_page(view, model, datasets[f"hist-{model.label}"]))
        pages.append(_pooled_page(view, model, datasets[f"pooled-{model.label}"]))
    psf = _psf_page(view, datasets.get("psf-kernels"))
    if psf is not None:
        pages.append(psf)

    model_summaries = []
    for model in view.models:
        default = " default" if model.is_default else ""
        model_summaries.append(
            f"{model.label}{default}: model="
            f"{_format_metric(model.runtime_model_fidelity_mean)}, "
            f"held-out={_format_metric(model.aggregate_fidelity)}, "
            f"global={_format_metric(model.global_fidelity)}, "
            f"usable={int(np.count_nonzero(model.runtime_usable))}/{len(view.site_labels)}"
        )
    summary = (
        f"{len(view.site_labels)} sites · {len(view.models)} models\n"
        + "\n".join(model_summaries)
    )
    return PlotReportDocument(summary, tuple(pages))


def render_calibration_plot_report(view: CalibrationReportProjection):
    """Render through the frontend owner at the optional UI adapter boundary."""

    from zlc_frontend.plot_report import render_plot_report

    return render_plot_report(project_calibration_plot_report(view))


__all__ = [
    "project_calibration_plot_report",
    "render_calibration_plot_report",
]
