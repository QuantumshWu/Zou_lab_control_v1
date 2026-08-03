"""Compose TaskConsole directly from current application facts."""

from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.logic_node import LogicNodeDescriptor
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane

__all__ = ["open_task_console"]


def open_task_console(
    *,
    descriptors: tuple[LogicNodeDescriptor, ...],
    device_catalog: DeviceCatalogView,
    host_factory,
    data_plane: SignalDataPlane,
    project_root: Path,
    pulses_root: Path,
    tasks_root: Path,
    figures_root: Path,
    state=None,
    task=None,
    **kwargs,
):
    """Open the generic shell without a second catalog or attachment layer."""

    values = tuple(descriptors)
    if any(not isinstance(value, LogicNodeDescriptor) for value in values):
        raise TypeError("descriptors must contain LogicNodeDescriptor values")
    if len({value.definition.key for value in values}) != len(values):
        raise ValueError("TaskConsole descriptor keys must be unique")
    if not isinstance(device_catalog, DeviceCatalogView):
        raise TypeError("device_catalog must be DeviceCatalogView")
    if not callable(host_factory):
        raise TypeError("host_factory must be callable")
    if not isinstance(data_plane, SignalDataPlane):
        raise TypeError("data_plane must be SignalDataPlane")

    from .window import show_task_console

    return show_task_console(
        descriptors=values,
        device_catalog=device_catalog,
        host_factory=host_factory,
        data_plane=data_plane,
        project_root=project_root,
        pulses_root=pulses_root,
        tasks_root=tasks_root,
        figures_root=figures_root,
        state=state,
        task=task,
        **kwargs,
    )
