"""Calibration's sole outward adapter to frontend Figure/report services."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
import math
import threading

import numpy as np

from zlc_data.axis import AxisSourceRef
from zlc_data.value import OwnedSnapshot
from zlc_frontend import (
    AxisViewRole,
    CurveDisplayState,
    FacetedHistogramDisplayState,
    FigureIntent,
    FigureSource,
    FixedIndex,
    HistogramCellThresholds,
    HistogramDisplayState,
    PanelProvenance,
    PlotKind,
    PlotReportDocument,
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
    plot_report_page,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.histogram_display import histogram_display_with_thresholds
from zlc_frontend.plot_report import render_plot_report
from zlc_frontend.site_map_view import build_site_map_snapshot_view
from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
    calibration_request_from_computation,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.projection import (
    CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
    CalibrationModelReportProjection,
    CalibrationReportProjection,
    CalibrationSiteMapContext,
    materialize_calibration_report_datasets,
    project_calibration_report,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.processing.signal_plane import SignalPublication
from zlc_neutral_atom.runtime.hosted_run import HostedRun
from zlc_storage import canonical_digest


_SITE_MAP_FIGURE = FigureIntent(
    PlotKind.SITE_MAP,
    "Reference average | calibrated sites",
    "Counts",
)


def calibration_site_map_figure(
    snapshot: OwnedSnapshot,
    context: CalibrationSiteMapContext,
    *,
    coherence_identity: str,
    run_id: str,
    provenance_epoch_id: str,
) -> tuple[FigureIntent, FigureSource]:
    """Project the sole Calibration SiteMap intent from physical context."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if not isinstance(context, CalibrationSiteMapContext):
        raise TypeError("context must be CalibrationSiteMapContext")
    valid_count = int(context.site_validity.sum())
    view = build_site_map_snapshot_view(
        snapshot,
        site_axis=context.site_axis,
        coordinate_frame=context.coordinate_frame,
        centers_xy=context.centers_xy,
        site_validity=context.site_validity,
        site_geometry_identity=context.calibration_identity,
        coherence_identity=coherence_identity,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=(
            f"{context.calibration_identity} | reference average | "
            f"valid sites={valid_count}/{context.site_axis.size}"
        ),
        presentation_kind="calibration-sites",
    )
    return _SITE_MAP_FIGURE, FigureSource(snapshot, view)


def project_calibration_signal_presentation(
    node: object,
    output_name: str,
    publication: SignalPublication,
):
    """Project the Calibration SiteMap from one exact FINAL publication."""

    site_map_name = CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0].name
    if output_name != site_map_name:
        return None
    if not isinstance(node, HostedRun):
        raise TypeError("Calibration presentation requires HostedRun")
    command = node.prepared_command
    result = node.final_result
    value = publication.value(node.signal_key(output_name))
    if command is None or result is None or value is None:
        return None
    context = command.site_map_context(result)
    figure, source = calibration_site_map_figure(
        value.snapshot,
        context,
        coherence_identity=value.join_digest,
        run_id=value.run_id,
        provenance_epoch_id=value.epoch_id,
    )
    return figure, source.site_map


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
        display=HistogramDisplayState(bin_count=60, thresholds=(0.0,)),
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
    """Project one physical report through frontend's canonical page contract."""

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
    """Render the operator report through frontend's sole renderer owner."""

    return render_plot_report(project_calibration_plot_report(view))


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _project_calibration_computation(
    computation,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
) -> PlotReportDocument:
    """Project immutable report inputs once, independently of screen DPR."""

    if cancelled.is_set():
        raise CancelledError()
    view = project_calibration_report(computation, reference)
    if cancelled.is_set():
        raise CancelledError()
    document = project_calibration_plot_report(view)
    if cancelled.is_set():
        raise CancelledError()
    lineage = ", ".join(
        f"{name}={version}" for name, version in view.software_lineage
    )
    summary = (
        f"{view.calibration_identity} · source {view.source_capture_identity}\n"
        f"binding={view.binding} · camera={view.camera_identity} · "
        f"ROI={view.roi_shape_yx} · "
        f"exposure={1e3 * view.exposure_seconds:.4g} ms · "
        f"groups={view.group_count}\n"
        f"{document.summary}\n"
        f"numeric notes: {lineage}"
    )
    return replace(document, summary=summary)


def _load_calibration_computation(
    loader,
    reference: CalibrationArtifactRef,
):
    try:
        return loader(reference)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "calibration diagnostics are unavailable; this does not by itself "
            "invalidate the committed runtime calibration"
        ) from error


def _load_calibration_report_document(
    loader,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
) -> PlotReportDocument:
    if cancelled.is_set():
        raise CancelledError()
    computation = _load_calibration_computation(loader, reference)
    return _project_calibration_computation(
        computation,
        reference,
        cancelled,
    )


def _render_calibration_report(
    document: PlotReportDocument,
    surface_pixel_ratio: float,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    """Rasterize only an already-projected immutable report document."""

    if not isinstance(document, PlotReportDocument):
        raise TypeError("calibration report render requires PlotReportDocument")
    _require_not_cancelled(cancelled)
    rendered = render_plot_report(
        document,
        surface_pixel_ratio=surface_pixel_ratio,
        checkpoint=lambda: _require_not_cancelled(cancelled),
    )
    _require_not_cancelled(cancelled)
    return rendered


def _prepare_calibration_editor(
    computation_loader,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
):
    """Resolve one prior request and immutable report document exactly once."""

    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    if cancelled.is_set():
        raise CancelledError()
    computation = _load_calibration_computation(computation_loader, reference)
    request = calibration_request_from_computation(computation)
    try:
        document = _project_calibration_computation(
            computation,
            reference,
            cancelled,
        )
    except BaseException as error:
        if cancelled.is_set() or isinstance(error, CancelledError):
            raise CancelledError() from error
        from zlc_frontend.qt_widgets import error_summary

        return request, None, error_summary(error)
    return request, document, None
