"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def open_capture_workbench(experiment, request):
    """Open the finite exact-capture Workbench without owning the Experiment."""

    from ._capture import open_capture_workbench as _open

    return _open(experiment, request)


__all__ = ["open_capture_workbench"]
