"""Qt composition entry points for calibration creation and reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
        CalibrationArtifactRef,
    )
    from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
        CalibrationArtifactRequest,
    )

    from .workbench_window import CalibrationWorkbenchWindow
    from .report_window import CalibrationReportWindow


def open_calibration_report_workbench(
    computation_loader,
    reference: CalibrationArtifactRef,
) -> CalibrationReportWindow:
    """Load and display one FINAL calibration report on the shared raster lane."""

    from zlc_frontend.qt_widgets import launch_qt_window
    from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
        CalibrationArtifactRef,
    )

    from .report_window import CalibrationReportWindow

    if not callable(computation_loader):
        raise TypeError("computation_loader must be callable")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    return launch_qt_window(
        lambda: CalibrationReportWindow(
            computation_loader,
            reference,
        ),
    )


def open_calibration_workbench(
    computation_loader,
    run_starter,
    *,
    request: CalibrationArtifactRequest | None = None,
    reference: CalibrationArtifactRef | None = None,
) -> CalibrationWorkbenchWindow:
    """Open formal creation/editing from one request or exact artifact ref."""

    from zlc_frontend.qt_widgets import launch_qt_window

    from .workbench_window import CalibrationWorkbenchWindow

    if (request is None) == (reference is None):
        raise ValueError("provide exactly one calibration request or reference")
    return launch_qt_window(
        lambda: CalibrationWorkbenchWindow(
            computation_loader,
            run_starter,
            request=request,
            reference=reference,
        )
    )
