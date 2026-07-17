"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def open_capture_workbench(experiment, request):
    """Open the finite exact-capture Workbench without owning the Experiment."""

    from ._capture import open_capture_workbench as _open

    return _open(experiment, request)


def open_data_figure_workbench(figure, *, memory_limit_bytes=None):
    """Open one already-resolved DataFigure without eager Qt imports."""

    from ._figure import open_data_figure_workbench as _open

    if memory_limit_bytes is None:
        return _open(figure)
    return _open(figure, memory_limit_bytes=memory_limit_bytes)


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    memory_limit_bytes=None,
):
    """Resolve and display one frozen artifact without blocking the Qt owner."""

    from ._figure import open_figure_workbench as _open

    options = {
        "intent": intent,
        "selection": selection,
        "preferences": preferences,
    }
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
    "open_capture_workbench",
    "open_data_figure_workbench",
    "open_figure_workbench",
    "open_offline_pulse_workbench",
    "open_pulse_workbench",
    "open_scan_workbench",
    "open_task_console",
]
