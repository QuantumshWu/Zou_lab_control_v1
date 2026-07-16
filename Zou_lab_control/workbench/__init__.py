"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def open_capture_workbench(experiment, request):
    """Open the finite exact-capture Workbench without owning the Experiment."""

    from ._capture import open_capture_workbench as _open

    return _open(experiment, request)


def open_pulse_workbench(experiment, document=None, *, path=None):
    """Open the current PulseWorkbench without loading Qt at package import."""

    from ._pulse import open_pulse_workbench as _open

    return _open(experiment, document, path=path)


def open_scan_workbench(experiment, request):
    """Open the current typed autonomous scan panel lazily."""

    from ._scan import open_scan_workbench as _open

    return _open(experiment, request)


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
    "open_offline_pulse_workbench",
    "open_pulse_workbench",
    "open_scan_workbench",
]
