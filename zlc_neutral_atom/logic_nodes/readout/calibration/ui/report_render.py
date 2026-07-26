"""Compose typed calibration report Datasets through shared frontend plots.

The calibration projection owns every value, validity mask, population split,
threshold-centred sample, and site-grid mapping.  This optional adapter selects
ordinary frontend view intents and display state.  It owns no Figure, Axes,
artist, binning implementation, layout algorithm, or image encoder.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np

from zlc_data import IndexSelection, OwnedSnapshot, Selection
from zlc_frontend.curve_display import CurveDisplayState
from zlc_frontend.display_range import RelimMode
from zlc_frontend.encoded_raster import (
    EncodedRasterDocument,
    EncodedRasterPage,
    encode_raster_buffer_png,
)
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    FixedIndex,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.histogram_display import (
    FacetedHistogramDisplayState,
    HistogramCellThresholds,
    HistogramDisplayState,
    HistogramFitMode,
)
from zlc_frontend.image_display import ImageColormap, ImageDisplayState
from zlc_frontend.panel_render import PanelComposer, PanelProvenance
from zlc_frontend.plot_layout import panel_display_size
from zlc_frontend.site_map_render import SiteMapComposer
from zlc_frontend.site_map_view import build_site_map_snapshot_view
from zlc_storage import canonical_digest

from ..projection import (
    CalibrationModelReportProjection,
    CalibrationReportProjection,
    materialize_calibration_report_datasets,
)


_REPORT_SIZE_NAME = "8x8"
_REPORT_RASTER_SIZE = panel_display_size(_REPORT_SIZE_NAME)


def _format_metric(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def _checkpoint(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _frame_png(frame) -> bytes:
    panels = tuple(frame.panels)
    if len(panels) != 1:
        raise RuntimeError("single report plot produced more than one frontend panel")
    return encode_raster_buffer_png(panels[0].raster)


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


def _site_map_page(
    view: CalibrationReportProjection,
    snapshot: OwnedSnapshot,
) -> EncodedRasterPage:
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
    composer = SiteMapComposer(
        "calibration-report-overview",
        size=(1800, 1100),
        title="Reference average | calibrated sites",
        value_label=view.frame_schema.value_unit or "Signal",
    )
    try:
        frame = composer.compose(
            site_map,
            display=ImageDisplayState(colormap=ImageColormap.GRAY),
        )
    finally:
        composer.close()
    return EncodedRasterPage("overview", "Overview", _frame_png(frame))


def _histogram_page(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    snapshot: OwnedSnapshot,
) -> EncodedRasterPage:
    schema = snapshot.block.schema
    if len(schema.point_axes) != 2 or len(schema.cell_schema.data_axes) != 1:
        raise ValueError("calibration histogram Dataset has an invalid declared shape")
    row_axis, column_axis = schema.point_axes
    population_axis = schema.cell_schema.data_axes[0]
    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(schema.repeat_axis.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(row_axis.axis_id, AxisViewRole.FACET),
            AxisViewBinding(column_axis.axis_id, AxisViewRole.FACET),
            AxisViewBinding(population_axis.axis_id, AxisViewRole.BATCH),
        ),
    )
    display = FacetedHistogramDisplayState(
        HistogramDisplayState(
            bin_count=max(5, int(np.asarray(model.bin_edges).size) - 1),
            fit_mode=HistogramFitMode.NONE,
        ),
        tuple(
            HistogramCellThresholds(
                Selection(
                    (
                        IndexSelection(row_axis.axis_id, row),
                        IndexSelection(column_axis.axis_id, column),
                    )
                ),
                (
                    (float(model.runtime_thresholds[site]),)
                    if math.isfinite(float(model.runtime_thresholds[site]))
                    else ()
                ),
            )
            for site, (row, column) in enumerate(view.site_grid_positions_yx)
        ),
    )
    composer = PanelComposer(
        f"calibration-report-hist-{model.label}",
        intent=ViewIntent.HISTOGRAM,
        size=_REPORT_RASTER_SIZE,
        size_name=_REPORT_SIZE_NAME,
        label=f"{model.label} | per-site readout distributions",
        value_label=view.frame_schema.value_unit or "Signal",
        view=spec,
    )
    try:
        if len(view.site_labels) == 1:
            payload = _frame_png(
                composer.compose(
                    snapshot,
                    display=display.display_for(display.cell_thresholds[0].selection),
                    provenance=_provenance(view, f"hist-{model.label}", snapshot),
                )
            )
        else:
            result = composer.compose_faceted(
                snapshot,
                display=display,
                provenance=_provenance(view, f"hist-{model.label}", snapshot),
            )
            if result.overview_png is None:
                raise RuntimeError("histogram grid omitted its frontend overview")
            payload = result.overview_png
    finally:
        composer.close()
    return EncodedRasterPage(f"hist-{model.label}", model.label, payload)


def _pooled_page(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    snapshot: OwnedSnapshot,
) -> EncodedRasterPage:
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
    composer = PanelComposer(
        f"calibration-report-pooled-{model.label}",
        intent=ViewIntent.HISTOGRAM,
        size=_REPORT_RASTER_SIZE,
        size_name=_REPORT_SIZE_NAME,
        label=f"{model.label} | threshold-centred populations",
        value_label=(
            f"{view.frame_schema.value_unit} - runtime threshold"
            if view.frame_schema.value_unit
            else "Signal - runtime threshold"
        ),
        view=spec,
    )
    try:
        frame = composer.compose(
            snapshot,
            display=HistogramDisplayState(
                bin_count=60,
                fit_mode=HistogramFitMode.NONE,
                thresholds=(0.0,),
            ),
            provenance=_provenance(view, f"pooled-{model.label}", snapshot),
        )
    finally:
        composer.close()
    return EncodedRasterPage(
        f"pooled-{model.label}",
        f"{model.label} pooled",
        _frame_png(frame),
    )


def _fidelity_page(
    view: CalibrationReportProjection,
    snapshot: OwnedSnapshot,
) -> EncodedRasterPage:
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
    composer = PanelComposer(
        "calibration-report-fidelity",
        intent=ViewIntent.CURVE,
        size=_REPORT_RASTER_SIZE,
        size_name=_REPORT_SIZE_NAME,
        label="Per-site model fidelity",
        value_label="Fidelity",
        view=spec,
    )
    try:
        frame = composer.compose(
            snapshot,
            display=CurveDisplayState(
                relim_mode=RelimMode.FIXED,
                fixed_y_limits=(0.45, 1.01),
            ),
            provenance=_provenance(view, "fidelity", snapshot),
        )
    finally:
        composer.close()
    return EncodedRasterPage("fidelity", "Per-site fidelity", _frame_png(frame))


def _psf_page(
    view: CalibrationReportProjection,
    snapshot: OwnedSnapshot | None,
) -> EncodedRasterPage | None:
    if snapshot is None:
        return None
    schema = snapshot.block.schema
    if len(schema.point_axes) != 2 or len(schema.cell_schema.data_axes) != 2:
        raise ValueError("calibration PSF Dataset has an invalid declared shape")
    row_axis, column_axis = schema.point_axes
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
            AxisViewBinding(row_axis.axis_id, AxisViewRole.FACET),
            AxisViewBinding(column_axis.axis_id, AxisViewRole.FACET),
            AxisViewBinding(y_axis.axis_id, AxisViewRole.IMAGE_Y),
            AxisViewBinding(x_axis.axis_id, AxisViewRole.IMAGE_X),
        ),
    )
    composer = PanelComposer(
        "calibration-report-psf",
        intent=ViewIntent.IMAGE,
        size=_REPORT_RASTER_SIZE,
        size_name=_REPORT_SIZE_NAME,
        label="Empirical PSF kernels",
        value_label="Normalized weight",
        view=spec,
    )
    display = ImageDisplayState(colormap=ImageColormap.INFERNO)
    try:
        if len(view.site_labels) == 1:
            payload = _frame_png(
                composer.compose(
                    snapshot,
                    display=display,
                    provenance=_provenance(view, "psf-kernels", snapshot),
                )
            )
        else:
            result = composer.compose_faceted(
                snapshot,
                display=display,
                provenance=_provenance(view, "psf-kernels", snapshot),
            )
            if result.overview_png is None:
                raise RuntimeError("PSF grid omitted its frontend overview")
            payload = result.overview_png
    finally:
        composer.close()
    return EncodedRasterPage("psf-kernels", "PSF kernels", payload)


def render_calibration_report(
    view: CalibrationReportProjection,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> EncodedRasterDocument:
    """Render one frozen report exclusively through frontend-owned plot kinds."""

    if not isinstance(view, CalibrationReportProjection):
        raise TypeError("view must be CalibrationReportProjection")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    datasets = materialize_calibration_report_datasets(view)

    pages: list[EncodedRasterPage] = []
    _checkpoint(checkpoint)
    pages.append(_site_map_page(view, datasets["reference"]))
    _checkpoint(checkpoint)
    pages.append(_fidelity_page(view, datasets["fidelity"]))
    for model in view.models:
        _checkpoint(checkpoint)
        pages.append(_histogram_page(view, model, datasets[f"hist-{model.label}"]))
        _checkpoint(checkpoint)
        pages.append(_pooled_page(view, model, datasets[f"pooled-{model.label}"]))
    _checkpoint(checkpoint)
    psf_page = _psf_page(view, datasets.get("psf-kernels"))
    if psf_page is not None:
        pages.append(psf_page)
    _checkpoint(checkpoint)

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
    return EncodedRasterDocument(summary, tuple(pages))


__all__ = ["render_calibration_report"]
