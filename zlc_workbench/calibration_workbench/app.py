"""Qt composition entry points for calibration creation and reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zlc_workbench.frozen_raster import FrozenRasterWindow
    from zlc_neutral_atom.logic_nodes.calibration.reference import (
        CalibrationArtifactRef,
    )
    from zlc_neutral_atom.logic_nodes.calibration.application import (
        CalibrationArtifactRequest,
    )

    from .window import CalibrationWorkbenchWindow


def open_calibration_report_workbench(
    computation_loader,
    reference: CalibrationArtifactRef,
) -> FrozenRasterWindow:
    """Load and display one FINAL calibration report on the shared raster lane."""

    from zlc_workbench.frozen_raster import (
        open_frozen_raster_window,
    )
    from zlc_neutral_atom.logic_nodes.calibration.reference import (
        CalibrationArtifactRef,
    )

    from .jobs import _render_calibration

    if not callable(computation_loader):
        raise TypeError("computation_loader must be callable")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    return open_frozen_raster_window(
        lambda cancelled: _render_calibration(
            computation_loader,
            reference,
            cancelled,
        ),
        window_title="Calibration Report",
        mode_text="FROZEN CALIBRATION REPORT · DISPLAY ONLY",
        loading_summary=f"Resolving {reference.target_ref}…",
        object_prefix="calibrationReport",
        subject="report",
    )


def open_calibration_workbench(
    computation_loader,
    run_starter,
    *,
    request: CalibrationArtifactRequest | None = None,
    reference: CalibrationArtifactRef | None = None,
) -> CalibrationWorkbenchWindow:
    """Open formal creation/editing from one request or exact artifact ref."""

    from zlc_workbench.window_runtime import open_workbench_window

    from .window import CalibrationWorkbenchWindow

    if (request is None) == (reference is None):
        raise ValueError("provide exactly one calibration request or reference")
    return open_workbench_window(
        lambda: CalibrationWorkbenchWindow(
            computation_loader,
            run_starter,
            request=request,
            reference=reference,
        )
    )
