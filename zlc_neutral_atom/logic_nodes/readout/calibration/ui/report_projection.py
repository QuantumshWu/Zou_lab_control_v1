"""Project Calibration facts into the generic frontend Plot Report contract.

This optional UI leaf chooses only domain-significant page intent and labels.
Size, DPR, style, renderer, raster encoding, and backend lifecycle remain owned
by :mod:`zlc_frontend`.
"""

from __future__ import annotations

import math

import numpy as np

from zlc_data import IndexSelection, OwnedSnapshot, Selection
from zlc_frontend import (
    AxisViewBinding,
    AxisViewRole,
    CurveDisplayState,
    FacetedHistogramDisplayState,
    FixedIndex,
    HistogramCellThresholds,
    HistogramDisplayState,
    ImageColormap,
    ImageDisplayState,
    FigureSource,
    PanelProvenance,
    PlotReportDocument,
    ViewIntent,
    ViewSpec,
    plot_report_page,
    render_plot_report,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.histogram_display import histogram_display_with_thresholds
from zlc_frontend.site_map_view import build_site_map_snapshot_view
from zlc_storage import canonical_digest

from ..projection import (
    CalibrationModelReportProjection,
    CalibrationReportProjection,
    materialize_calibration_report_datasets,
)


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
    site_map = build_site_map_snapshot_view(
        snapshot,
        site_axis=view.site_axis,
        coordinate_frame=view.coordinate_frame,
        centers_xy=view.actual_centers_xy,
        site_validity=view.site_validity,
        site_geometry_identity=view.calibration_identity,
        coherence_identity=view.source_capture_identity,
        run_id=f"calibration-report-{view.calibration_identity}",
        provenance_epoch_id=snapshot.ref.stream_generation.value,
        summary="Reference average and calibrated sites",
        presentation_kind="calibration-report",
    )
    return plot_report_page(
        "overview",
        "Reference average | calibrated sites",
        kind="sites",
        source=FigureSource(snapshot, site_map),
        display=ImageDisplayState(colormap=ImageColormap.GRAY),
        provenance=_provenance(view, "overview", snapshot),
        value_label=view.frame_schema.value_unit or "Signal",
    )


def _histogram_page(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    snapshot: OwnedSnapshot,
):
    schema = snapshot.block.schema
    if len(schema.point_axes) != 1 or len(schema.cell_schema.data_axes) != 1:
        raise ValueError("calibration histogram Dataset has an invalid declared shape")
    site_axis = schema.point_axes[0]
    population_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(schema.repeat_axis.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(site_axis.axis_id, AxisViewRole.FACET),
            AxisViewBinding(population_axis.axis_id, AxisViewRole.BATCH),
        ),
    )
    base_display = HistogramDisplayState(
        bin_count=max(5, int(np.asarray(model.bin_edges).size) - 1),
    )
    cell_thresholds = tuple(
        HistogramCellThresholds(
            Selection(
                (
                    IndexSelection(site_axis.axis_id, site),
                )
            ),
            (
                (float(model.runtime_thresholds[site]),)
                if math.isfinite(float(model.runtime_thresholds[site]))
                else ()
            ),
        )
        for site in range(site_axis.size)
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
        f"{model.label} | per-site readout distributions",
        kind="grid" if faceted else "hist",
        source=FigureSource(snapshot),
        display=display,
        provenance=_provenance(view, f"hist-{model.label}", snapshot),
        value_label=view.frame_schema.value_unit or "Signal",
        view=spec,
    )


def _pooled_page(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    snapshot: OwnedSnapshot,
):
    schema = snapshot.block.schema
    if schema.point_axes or len(schema.cell_schema.data_axes) != 1:
        raise ValueError("calibration pooled Dataset has an invalid declared shape")
    population_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(schema.repeat_axis.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(population_axis.axis_id, AxisViewRole.BATCH),
        ),
    )
    return plot_report_page(
        f"pooled-{model.label}",
        f"{model.label} | threshold-centred populations",
        kind="hist",
        source=FigureSource(snapshot),
        display=HistogramDisplayState(
            bin_count=60,
            thresholds=(0.0,),
        ),
        provenance=_provenance(view, f"pooled-{model.label}", snapshot),
        value_label=(
            f"{view.frame_schema.value_unit} - runtime threshold"
            if view.frame_schema.value_unit
            else "Signal - runtime threshold"
        ),
        view=spec,
    )


def _fidelity_page(view: CalibrationReportProjection, snapshot: OwnedSnapshot):
    schema = snapshot.block.schema
    if len(schema.point_axes) != 1 or len(schema.cell_schema.data_axes) != 1:
        raise ValueError("calibration fidelity Dataset has an invalid declared shape")
    site_axis = schema.point_axes[0]
    model_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(
                schema.repeat_axis.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            AxisViewBinding(site_axis.axis_id, AxisViewRole.X),
            AxisViewBinding(model_axis.axis_id, AxisViewRole.BATCH),
        ),
    )
    return plot_report_page(
        "fidelity",
        "Per-site model fidelity",
        kind="1d",
        source=FigureSource(snapshot),
        display=CurveDisplayState(
            relim_mode=RelimMode.FIXED,
            fixed_y_limits=(0.45, 1.01),
        ),
        provenance=_provenance(view, "fidelity", snapshot),
        value_label="Fidelity",
        view=spec,
    )


def _psf_page(
    view: CalibrationReportProjection,
    snapshot: OwnedSnapshot | None,
):
    if snapshot is None:
        return None
    schema = snapshot.block.schema
    if len(schema.point_axes) != 1 or len(schema.cell_schema.data_axes) != 2:
        raise ValueError("calibration PSF Dataset has an invalid declared shape")
    site_axis = schema.point_axes[0]
    y_axis, x_axis = schema.cell_schema.data_axes
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.IMAGE,
        (
            AxisViewBinding(
                schema.repeat_axis.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            AxisViewBinding(site_axis.axis_id, AxisViewRole.FACET),
            AxisViewBinding(y_axis.axis_id, AxisViewRole.IMAGE_Y),
            AxisViewBinding(x_axis.axis_id, AxisViewRole.IMAGE_X),
        ),
    )
    return plot_report_page(
        "psf-kernels",
        "Empirical PSF kernels",
        kind="grid" if len(view.site_labels) > 1 else "2d",
        source=FigureSource(snapshot),
        display=ImageDisplayState(colormap=ImageColormap.INFERNO),
        provenance=_provenance(view, "psf-kernels", snapshot),
        value_label="Normalized weight",
        view=spec,
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
    """Render through the same frontend report owner used by every caller."""

    return render_plot_report(project_calibration_plot_report(view))


__all__ = [
    "project_calibration_plot_report",
    "render_calibration_plot_report",
]
