"""Headless projection, loading, and raster jobs for calibration windows."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_neutral_atom.logic_nodes.calibration.application import (
    calibration_request_from_computation,
)
from zlc_neutral_atom.logic_nodes.calibration.projection import (
    project_calibration_report,
)
from zlc_neutral_atom.logic_nodes.calibration.reference import CalibrationArtifactRef


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _render_calibration_computation(
    computation,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    if cancelled.is_set():
        raise CancelledError()
    view = project_calibration_report(computation, reference)
    if cancelled.is_set():
        raise CancelledError()
    from zlc_frontend.calibration_render import render_calibration_report

    rendered = render_calibration_report(
        view,
        checkpoint=lambda: _require_not_cancelled(cancelled),
    )
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
        f"{rendered.summary}\n"
        f"numeric notes: {lineage}"
    )
    return EncodedRasterDocument(
        summary,
        rendered.pages,
    )


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


def _render_calibration(
    loader,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    if cancelled.is_set():
        raise CancelledError()
    computation = _load_calibration_computation(loader, reference)
    return _render_calibration_computation(
        computation,
        reference,
        cancelled,
    )


def _prepare_calibration_editor(
    computation_loader,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
):
    """Resolve one prior request and report off the Qt owner without double load."""

    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    if cancelled.is_set():
        raise CancelledError()
    computation = _load_calibration_computation(computation_loader, reference)
    request = calibration_request_from_computation(computation)
    try:
        bundle = _render_calibration_computation(
            computation,
            reference,
            cancelled,
        )
    except BaseException as error:
        if cancelled.is_set() or isinstance(error, CancelledError):
            raise CancelledError() from error
        from zlc_workbench.window_runtime import error_summary

        return request, None, error_summary(error)
    return request, bundle, None
