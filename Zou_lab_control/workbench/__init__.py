"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def open_camera_monitor_workbench(prepare):
    """Open one free-running camera monitor without eager Qt imports."""

    from ._camera_monitor import open_camera_monitor_workbench as _open

    return _open(prepare)


def open_capture_workbench(experiment, request):
    """Open the finite exact-capture Workbench without owning the Experiment."""

    from ._capture import open_capture_workbench as _open

    return _open(experiment, request)


def open_calibration_report_workbench(
    computation_loader,
    reference,
    *,
    memory_limit_bytes=None,
):
    """Open one committed calibration report without eager Qt imports."""

    from ._calibration import open_calibration_report_workbench as _open

    if memory_limit_bytes is None:
        return _open(computation_loader, reference)
    return _open(
        computation_loader,
        reference,
        memory_limit_bytes=memory_limit_bytes,
    )


def open_data_figure_workbench(figure, *, memory_limit_bytes=None):
    """Open one already-resolved DataFigure without eager Qt imports."""

    from ._figure import open_data_figure_workbench as _open

    if memory_limit_bytes is None:
        return _open(figure)
    return _open(figure, memory_limit_bytes=memory_limit_bytes)


def open_occupancy_cell_workbench(
    navigation_loader,
    cell_loader,
    reference,
    *,
    selection=None,
    memory_limit_bytes=None,
):
    """Open one exact same-shot occupancy map without eager Qt imports."""

    from ._occupancy import open_occupancy_cell_workbench as _open

    options = {"selection": selection}
    if memory_limit_bytes is not None:
        options["memory_limit_bytes"] = memory_limit_bytes
    return _open(navigation_loader, cell_loader, reference, **options)


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    occupancy_output=None,
    memory_limit_bytes=None,
):
    """Resolve and display one frozen artifact without blocking the Qt owner."""

    from ._figure import open_figure_workbench as _open

    options = {
        "intent": intent,
        "selection": selection,
        "preferences": preferences,
    }
    if occupancy_output is not None:
        options["occupancy_output"] = occupancy_output
    if memory_limit_bytes is not None:
        options["memory_limit_bytes"] = memory_limit_bytes
    return _open(figure_factory, source, **options)


def open_pulse_workbench(experiment, document=None, *, path=None):
    """Open the current PulseWorkbench without loading Qt at package import."""

    from ._pulse import open_pulse_workbench as _open

    return _open(experiment, document, path=path)


def open_scan_workbench(experiment, request):
    """Open the current typed autonomous scan panel lazily."""

    from ._scan import open_scan_workbench as _open

    return _open(experiment, request)


def open_task_console(experiment, initial_intent=None):
    """Open the current single-card SCAN_SLOT TaskConsole lazily."""

    from ._task_console import open_task_console as _open

    return _open(experiment, initial_intent)


def open_offline_pulse_workbench(
    target,
    *,
    time_step_ns,
    document=None,
    path=None,
):
    """Open current PulseDocument authoring/preview without a hardware facade."""

    from ._pulse import open_offline_pulse_workbench as _open

    return _open(
        target,
        time_step_ns=time_step_ns,
        document=document,
        path=path,
    )


__all__ = [
    "open_calibration_report_workbench",
    "open_camera_monitor_workbench",
    "open_capture_workbench",
    "open_data_figure_workbench",
    "open_figure_workbench",
    "open_occupancy_cell_workbench",
    "open_offline_pulse_workbench",
    "open_pulse_workbench",
    "open_scan_workbench",
    "open_task_console",
]
