"""Runtime-DPR Qt surfaces for immutable calibration PlotReports."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
import math

from zlc_frontend import PlotReportDocument
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.qt_widgets import (
    FrozenRasterWindow,
    RasterPixelRatioObserver,
    error_summary,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from .workbench_jobs import (
    _load_calibration_report_document,
    _render_calibration_report,
)


class CalibrationReportSurfaceWindow(FrozenRasterWindow):
    """Own one immutable report document and its replaceable screen raster.

    Loading/projecting calibration physics is intentionally outside the
    runtime-surface loop.  A DPR change retires the painted PNG immediately,
    then submits only ``PlotReportDocument -> EncodedRasterDocument`` on the
    shared raster worker.  At most one render is active; repeated screen
    changes collapse to the newest authored surface revision.
    """

    def __init__(self, **window_options) -> None:
        self._report_document: PlotReportDocument | None = None
        self._report_document_revision = 0
        self._report_admitted_document_revision: int | None = None
        self._report_surface_revision = 0
        self._report_render_requested = False
        self._report_render_reason: str | None = None
        self._report_render_active: tuple[
            PlotReportDocument,
            int,
            int,
            str,
        ] | None = None
        super().__init__(None, **window_options)
        self._report_surface_observer = RasterPixelRatioObserver(
            self,
            self._apply_report_surface_pixel_ratio,
        )
        self._report_surface_pixel_ratio = (
            self._report_surface_observer.current_ratio
        )

    @property
    def report_document(self) -> PlotReportDocument | None:
        return self._report_document

    @property
    def report_surface_revision(self) -> int:
        return self._report_surface_revision

    @property
    def worker_idle(self) -> bool:
        return (
            super().worker_idle
            and not self._report_render_requested
            and self._report_render_active is None
        )

    def _apply_report_surface_pixel_ratio(self, ratio: float) -> None:
        if self._closing:
            return
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("report surface pixel ratio must be real")
        normalized = float(ratio)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError("report surface pixel ratio must be finite and positive")
        if normalized == self._report_surface_pixel_ratio:
            return
        self._report_surface_pixel_ratio = normalized
        self._report_surface_revision += 1
        # The old bitmap belongs to another physical surface.  Clear it before
        # queuing work so no event turn can continue presenting stale pixels.
        self._clear_bundle()
        if self._report_document is not None:
            self._report_render_requested = True
            reason = (
                "surface"
                if self._report_admitted_document_revision
                == self._report_document_revision
                else "document"
            )
            if reason == "document" or self._report_render_reason is None:
                self._report_render_reason = reason
        self._start_report_render_if_ready()

    def _install_report_document(self, document: PlotReportDocument) -> None:
        if not isinstance(document, PlotReportDocument):
            raise TypeError("calibration report projection must be PlotReportDocument")
        self._report_document = document
        self._report_document_revision += 1
        self._report_admitted_document_revision = None
        self._clear_bundle()
        self._report_render_requested = True
        self._report_render_reason = "document"
        self._start_report_render_if_ready()

    def _discard_report_document(self) -> None:
        """Retire a report whose artifact authority has been superseded."""

        self._report_document = None
        self._report_document_revision += 1
        self._report_admitted_document_revision = None
        self._report_render_requested = False
        self._report_render_reason = None
        self._clear_bundle()

    def _start_report_render_if_ready(self) -> None:
        if (
            self._closing
            or self._future is not None
            or not self._report_render_requested
            or self._report_document is None
        ):
            return
        document = self._report_document
        reason = self._report_render_reason
        if reason not in {"document", "surface"}:
            raise RuntimeError("report render request has no authored reason")
        key = (
            document,
            self._report_document_revision,
            self._report_surface_revision,
            reason,
        )
        self._report_render_requested = False
        self._report_render_reason = None
        self._report_render_active = key
        self._report_render_started(self._report_surface_revision, reason)
        if not self._submit_future(
            _render_calibration_report,
            document,
            self._report_surface_pixel_ratio,
            self._cancelled,
        ):
            self._report_render_active = None

    def _worker_submit_failed(self, error: BaseException) -> None:
        active = self._report_render_active
        if active is not None:
            self._report_render_active = None
            self._report_render_failed(error, reason=active[3])
            return
        super()._worker_submit_failed(error)

    def _accept_finished_future(self, future: Future) -> None:
        active = self._report_render_active
        if active is None:
            super()._accept_finished_future(future)
            return
        self._report_render_active = None
        document, document_revision, surface_revision, reason = active
        current = (
            document is self._report_document
            and document_revision == self._report_document_revision
            and surface_revision == self._report_surface_revision
        )
        try:
            bundle = future.result()
            if not isinstance(bundle, EncodedRasterDocument):
                raise TypeError("report worker returned an invalid raster")
        except CancelledError:
            if not self._closing and current:
                self._report_render_failed(
                    RuntimeError("report rendering was cancelled"),
                    reason=reason,
                )
        except BaseException as error:
            if not self._closing and current:
                self._report_render_failed(error, reason=reason)
        else:
            if not self._closing and current:
                displayed = self._present_bundle(bundle)
                if displayed:
                    self._report_admitted_document_revision = document_revision
                self._report_render_succeeded(
                    bundle,
                    displayed=displayed,
                    reason=reason,
                )
        if not current and self._report_document is not None:
            self._report_render_requested = True
            pending_reason = (
                "surface"
                if self._report_admitted_document_revision
                == self._report_document_revision
                else "document"
            )
            if (
                pending_reason == "document"
                or self._report_render_reason is None
            ):
                self._report_render_reason = pending_reason

    def _after_worker_completion(self) -> None:
        self._start_report_render_if_ready()

    def _report_render_started(
        self,
        surface_revision: int,
        reason: str,
    ) -> None:
        self._status.setText(f"BUILDING REPORT SURFACE r{surface_revision}")
        self._diagnostic.setText("")

    def _report_render_succeeded(
        self,
        bundle: EncodedRasterDocument,
        *,
        displayed: bool,
        reason: str,
    ) -> None:
        if not displayed:
            self._status.setText("REPORT DISPLAY FAILED")

    def _report_render_failed(self, error: BaseException, *, reason: str) -> None:
        self._status.setText("REPORT DISPLAY FAILED")
        self._summary.setText("The immutable calibration report remains valid")
        self._diagnostic.setText(error_summary(error))

    def _before_worker_shutdown(self) -> None:
        self._report_surface_observer.detach()
        self._report_document = None
        self._report_admitted_document_revision = None
        self._report_render_requested = False
        self._report_render_reason = None
        super()._before_worker_shutdown()


class CalibrationReportWindow(CalibrationReportSurfaceWindow):
    """Load one FINAL calibration document once and display it at live DPR."""

    def __init__(self, computation_loader, reference: CalibrationArtifactRef) -> None:
        if not callable(computation_loader):
            raise TypeError("computation_loader must be callable")
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        self._report_load_active = True
        super().__init__(
            window_title="Calibration Report",
            mode_text="FROZEN CALIBRATION REPORT · DISPLAY ONLY",
            loading_summary=f"Resolving {reference.target_ref}…",
            object_prefix="calibrationReport",
            subject="report",
        )
        if not self._submit_future(
            _load_calibration_report_document,
            computation_loader,
            reference,
            self._cancelled,
        ):
            self._report_load_active = False

    def _accept_finished_future(self, future: Future) -> None:
        if not self._report_load_active:
            super()._accept_finished_future(future)
            return
        self._report_load_active = False
        try:
            document = future.result()
            if not isinstance(document, PlotReportDocument):
                raise TypeError("report loader returned an invalid document")
        except CancelledError:
            if not self._closing:
                self._status.setText("REPORT CANCELLED")
        except BaseException as error:
            if not self._closing:
                self._status.setText("REPORT FAILED")
                self._summary.setText("No report document was admitted")
                self._diagnostic.setText(error_summary(error))
        else:
            if not self._closing:
                self._install_report_document(document)


__all__ = [
    "CalibrationReportSurfaceWindow",
    "CalibrationReportWindow",
]
