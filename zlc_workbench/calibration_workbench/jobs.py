"""Headless projection, loading, and raster jobs for calibration windows."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

import numpy as np

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_workbench.calibration import calibration_seed_from_computation


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _project_calibration(computation):
    from zlc_frontend.calibration_render import (
        CalibrationModelView,
        CalibrationReportView,
    )
    from zlc_neutral_atom.readout.analysis import (
        CalibrationComputation,
        calibration_runtime_threshold_sources,
    )
    from zlc_neutral_atom.readout.calibration import (
        PerSitePsfFeature,
        UniformPsfFeature,
        site_grid_positions_yx,
    )

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("calibration loader must return CalibrationComputation")
    artifact = computation.artifact
    report = computation.report
    models = []
    threshold_sources = calibration_runtime_threshold_sources(report)
    psf_kernels = None
    psf_caption = None
    uniform_kernel = None
    for artifact_model, model_report, model_threshold_sources in zip(
        artifact.models,
        report.models,
        threshold_sources,
        strict=True,
    ):
        if artifact_model.kind is not model_report.kind:
            raise ValueError("calibration artifact/report model order differs")
        if isinstance(artifact_model.feature, PerSitePsfFeature):
            psf_kernels = artifact_model.feature.kernels
            psf_caption = (
                "Empirical per-site PSF kernels "
                "(artifact values, not Gaussian redraw)"
            )
        elif isinstance(artifact_model.feature, UniformPsfFeature):
            uniform_kernel = artifact_model.feature.kernel
        site_fidelity = model_report.site_fidelity
        models.append(
            CalibrationModelView(
                label=artifact_model.kind.value,
                is_default=artifact_model.kind is artifact.default_model_kind,
                signals=model_report.short_signals,
                signal_validity=model_report.short_validity,
                bin_edges=model_report.bin_edges,
                quick_thresholds=model_report.quick_thresholds,
                formal_thresholds=model_report.thresholds,
                runtime_thresholds=artifact_model.thresholds,
                runtime_threshold_sources=model_threshold_sources,
                feature_validity=artifact_model.feature.valid_sites.mask,
                runtime_usable=artifact_model.usable_sites.mask,
                bright_above=np.asarray(
                    [item.bright_above for item in site_fidelity],
                    dtype=np.bool_,
                ),
                model_fidelity=np.asarray(
                    [item.model_fidelity for item in site_fidelity],
                    dtype=np.float64,
                ),
                heldout_fidelity=np.asarray(
                    [item.fidelity for item in site_fidelity],
                    dtype=np.float64,
                ),
                aggregate_fidelity=model_report.aggregate_fidelity,
                global_fidelity=model_report.global_fidelity,
            )
        )
    if psf_kernels is None and uniform_kernel is not None:
        psf_kernels = np.broadcast_to(
            uniform_kernel,
            (artifact.site_map.site_axis.size, *uniform_kernel.shape),
        )
        psf_caption = (
            "Empirical shared uniform PSF kernel "
            "(repeated by site for display)"
        )
    psf_fit_ok = psf_sigma = None
    if psf_kernels is not None:
        if len(report.psf_fits) != artifact.site_map.site_axis.size:
            raise ValueError("empirical PSF kernels lack aligned fit diagnostics")
        psf_fit_ok = np.asarray(
            [item.fit_ok for item in report.psf_fits],
            dtype=np.bool_,
        )
        psf_sigma = np.asarray(
            [item.sigma_xy for item in report.psf_fits],
            dtype=np.float64,
        )
    default_model = artifact.select_model()
    return CalibrationReportView(
        reference_average=report.reference_average,
        reference_average_validity=report.reference_average_validity,
        actual_centers_xy=artifact.site_map.coordinates_xy,
        expected_centers_xy=report.request.expected_centers_xy,
        site_validity=artifact.site_map.validity.mask,
        default_boxes_xywh=default_model.feature.boxes_xywh,
        grid_shape_yx=artifact.site_map.grid_shape_yx,
        site_grid_positions_yx=site_grid_positions_yx(
            artifact.site_map.grid_shape_yx,
            artifact.site_map.ordering,
        ),
        site_labels=tuple(
            str(value) for value in artifact.site_map.site_axis.coordinates
        ),
        occupied_labels=report.labels.occupied,
        dark_labels=report.labels.dark,
        label_validity=report.labels.valid,
        models=tuple(models),
        psf_kernels=psf_kernels,
        psf_caption=psf_caption,
        psf_fit_ok=psf_fit_ok,
        psf_sigma_xy=psf_sigma,
    )


def _render_calibration_computation(
    computation,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    if cancelled.is_set():
        raise CancelledError()
    view = _project_calibration(computation)
    artifact = computation.artifact
    report = computation.report
    if cancelled.is_set():
        raise CancelledError()
    from zlc_frontend.calibration_render import render_calibration_report

    rendered = render_calibration_report(
        view,
        checkpoint=lambda: _require_not_cancelled(cancelled),
    )
    if cancelled.is_set():
        raise CancelledError()
    frame = artifact.frame_contract
    source_ref = artifact.source_binding.source_capture_ref
    lineage = ", ".join(
        f"{name}={version}" for name, version in report.software_lineage
    )
    summary = (
        f"{reference.target_ref} · source {source_ref.target_ref}\n"
        f"binding={frame.binding.value} · camera={frame.camera_identity} · "
        f"ROI={frame.roi_shape_yx} · "
        f"exposure={1e3 * frame.exposure_seconds:.4g} ms · "
        f"groups={len(report.group_contexts)}\n"
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
    timeout_seconds: float,
    cancelled: threading.Event,
):
    """Resolve one prior seed and report off the Qt owner without double load."""

    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    if cancelled.is_set():
        raise CancelledError()
    computation = _load_calibration_computation(computation_loader, reference)
    prepared = calibration_seed_from_computation(
        computation,
        reference,
        timeout_seconds=timeout_seconds,
    )
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

        return prepared, None, error_summary(error)
    return prepared, bundle, None
