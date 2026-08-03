"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def _build_logic_node_ui(contribution, *args, **kwargs):
    """Resolve and call one leaf-owned optional UI contribution lazily."""

    from importlib import import_module

    from zlc_neutral_atom.logic_node import UiContribution

    if not isinstance(contribution, UiContribution):
        raise TypeError("contribution must be UiContribution")
    symbol = getattr(
        import_module(contribution.module),
        contribution.symbol,
    )
    if not callable(symbol):
        raise TypeError(
            f"Logic-node UI symbol {contribution.module}:{contribution.symbol} "
            "must be callable"
        )
    return symbol(*args, **kwargs)


def _invoke_logic_node_ui(contribution, *args, **kwargs):
    """Launch one already-frozen optional contribution as an owned window."""

    from zlc_frontend.qt_widgets import launch_qt_window

    return launch_qt_window(
        lambda: _build_logic_node_ui(contribution, *args, **kwargs)
    )


def open_figure_workbench(
    snapshot,
    spec,
    *,
    output_root,
    size=None,
    parameters=None,
    archive_path=None,
    metadata=None,
    open_fit=False,
):
    """Display one explicit frozen snapshot through the current plot surface."""

    from zlc_workbench.data_figure.app import open_figure_workbench as _open

    return _open(
        snapshot,
        spec,
        output_root=output_root,
        size=size,
        parameters=parameters,
        archive_path=archive_path,
        metadata=metadata,
        open_fit=open_fit,
    )


def open_pulse_editor(
    experiment=None,
    *,
    document=None,
    path=None,
    remote_endpoint=None,
    workspace=None,
):
    """Open PulseGUI from explicit bound or standalone composition inputs."""

    from zlc_workbench.pulse_editor.app import open_pulse_editor as _open

    if experiment is None:
        from Zou_lab_control.api import WorkspacePaths
        from ._composition import standalone_pulse_connection_factory

        if not isinstance(workspace, WorkspacePaths):
            raise TypeError("standalone Pulse editor requires WorkspacePaths")
        return _open(
            connection_factory=standalone_pulse_connection_factory(workspace),
            pulses_root=workspace.pulses_root,
            output_root=workspace.project_root,
            initial_connection_mode="offline",
            document=document,
            path=path,
            remote_endpoint=remote_endpoint,
        )

    if remote_endpoint is not None or workspace is not None:
        raise ValueError(
            "a bound Experiment already owns its Pulse endpoint and workspace"
        )
    from ._composition import bound_pulse_mode, _require_experiment

    experiment = _require_experiment(experiment)
    roots = experiment._workspace_paths()

    def compose():
        pulse = experiment.pulse
        return _open(
            pulse=pulse,
            descriptor=pulse.target,
            initial_connection_mode=bound_pulse_mode(experiment),
            pulses_root=roots.pulses_root,
            output_root=roots.project_root,
            document=document,
            path=path,
        )

    existing_error = None
    if document is not None or path is not None:
        existing_error = (
            "this Experiment already owns a Pulse editor; use its Open control "
            "instead of replacing unsaved state"
        )
    return experiment._open_workbench_handle(
        "pulse-editor",
        compose,
        existing_error=existing_error,
    )


def open_device_manager(
    experiment=None,
    *,
    document=None,
    config_path=None,
    workspace=None,
    name="neutral_atom",
    on_initialized=None,
):
    """Open DeviceManager while application composition retains connect/close."""

    from zlc_workbench.device_manager.app import open_device_manager as _open
    from ._composition import ExperimentDeviceAdmin, _require_experiment

    if on_initialized is not None and not callable(on_initialized):
        raise TypeError("on_initialized must be callable or None")

    if experiment is not None:
        experiment = _require_experiment(experiment)
        if any(
            value is not None
            for value in (document, config_path, workspace, on_initialized)
        ):
            raise ValueError(
                "a bound DeviceManager takes its config from the Experiment"
            )

        def compose():
            return _open(ExperimentDeviceAdmin.bound(experiment))

        return experiment._open_workbench_handle("device-manager", compose)

    admin = ExperimentDeviceAdmin.standalone(
        workspace=workspace,
        name=name,
    )

    def runtime_changed(state):
        if on_initialized is None or state.runtime_instance_id is None:
            return
        on_initialized(admin.experiment_for_runtime(state.runtime_instance_id))

    try:
        return _open(
            admin,
            document=document,
            config_path=config_path,
            on_runtime_changed=runtime_changed,
        )
    except BaseException:
        admin.dispose()
        raise


def open_task_console(experiment, *, state=None, task=None, **kwargs):
    """Open the sole current Monitor/Logic TaskConsole lazily."""

    from zlc_workbench.task_console.app import open_task_console as _open
    from ._composition import _require_experiment, task_console_dependencies

    experiment = _require_experiment(experiment)
    if "hide_on_close" in kwargs:
        raise TypeError(
            "a bound TaskConsole always preserves its window on X; "
            "hide_on_close is not caller-configurable"
        )

    def compose():
        return _open(
            **task_console_dependencies(experiment),
            state=state,
            task=task,
            hide_on_close=True,
            **kwargs,
        )

    existing_error = None
    if state is not None or task is not None:
        existing_error = (
            "this Experiment already owns a TaskConsole; load another layout "
            "from that window"
        )
    return experiment._open_workbench_handle(
        "task-console",
        compose,
        existing_error=existing_error,
    )


__all__ = [
    "open_figure_workbench",
    "open_device_manager",
    "open_pulse_editor",
    "open_task_console",
]
