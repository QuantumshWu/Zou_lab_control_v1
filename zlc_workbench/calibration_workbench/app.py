"""Qt composition entry points for calibration creation and reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zlc_workbench.frozen_raster import FrozenRasterWindow
    from zlc_neutral_atom.readout.calibration_reference import (
        CalibrationArtifactRef,
    )
    from zlc_workbench.calibration import CalibrationEditorSeed

    from .window import CalibrationWorkbenchWindow


_DEFAULT_CALIBRATION_GUI_TIMEOUT_SECONDS = 300.0


def open_calibration_report_workbench(
    computation_loader,
    reference: CalibrationArtifactRef,
) -> FrozenRasterWindow:
    """Load and display one FINAL calibration report on the shared raster lane."""

    from zlc_workbench.frozen_raster import (
        open_frozen_raster_window,
    )
    from zlc_neutral_atom.readout.calibration_reference import (
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
    seed: CalibrationEditorSeed | None = None,
    reference: CalibrationArtifactRef | None = None,
    timeout_seconds: float = _DEFAULT_CALIBRATION_GUI_TIMEOUT_SECONDS,
) -> CalibrationWorkbenchWindow:
    """Open formal creation/editing from one request seed or exact artifact ref."""

    from zlc_workbench.window_runtime import open_workbench_window
    from zlc_storage import positive_real

    from .window import CalibrationWorkbenchWindow

    if (seed is None) == (reference is None):
        raise ValueError("provide exactly one calibration seed or reference")
    if seed is not None:
        timeout = seed.timeout_seconds
    else:
        timeout = positive_real(timeout_seconds, "timeout_seconds")
    return open_workbench_window(
        lambda: CalibrationWorkbenchWindow(
            computation_loader,
            run_starter,
            seed=seed,
            reference=reference,
            timeout_seconds=timeout,
        )
    )
