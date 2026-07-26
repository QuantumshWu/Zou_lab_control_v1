"""Headless projection, loading, and raster jobs for calibration windows."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
import threading

from zlc_frontend import PlotReportDocument, render_plot_report
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
    calibration_request_from_computation,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.projection import (
    project_calibration_report,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef


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
    from .report_projection import project_calibration_plot_report

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
        from zlc_workbench.window_runtime import error_summary

        return request, None, error_summary(error)
    return request, document, None
