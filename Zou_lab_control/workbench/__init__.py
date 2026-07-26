"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    fit_preparer=None,
    fit_executor=None,
    fit_saver=None,
    fit_reloader=None,
    fit_selected_model=None,
    fit_initial_selection=None,
    open_fit=False,
    fit_timeout_seconds=None,
    initial_fit_result_identity=None,
):
    """Resolve and display one frozen artifact without blocking the Qt owner."""

    from zlc_workbench.data_figure.app import open_figure_workbench as _open

    options = {
        "intent": intent,
        "selection": selection,
        "preferences": preferences,
    }
    for name, value in (
        ("fit_preparer", fit_preparer),
        ("fit_executor", fit_executor),
        ("fit_saver", fit_saver),
        ("fit_reloader", fit_reloader),
        ("fit_selected_model", fit_selected_model),
        ("fit_initial_selection", fit_initial_selection),
        ("initial_fit_result_identity", initial_fit_result_identity),
    ):
        if value is not None:
            options[name] = value
    if open_fit:
        options["open_fit"] = True
    if fit_timeout_seconds is not None:
        options["fit_timeout_seconds"] = fit_timeout_seconds
    return _open(figure_factory, source, **options)


def open_saved_fit_grid_workbench(
    view_loader,
    refit_opener,
    reference,
):
    """Open one exact saved-fit GridPlot without eager Qt imports."""

    from zlc_workbench.fit_grid.app import (
        open_saved_fit_grid_workbench as _open,
    )

    return _open(view_loader, refit_opener, reference)


def open_pulse_editor(
    experiment=None,
    *,
    document=None,
    path=None,
    remote_endpoint=None,
    repository=None,
):
    """Open PulseGUI from explicit bound or standalone composition inputs."""

    from zlc_workbench.pulse_editor.app import open_pulse_editor as _open

    if experiment is None:
        from ._composition import (
            standalone_pulse_connection_factory,
            standalone_pulse_workspace,
        )

        workspace = standalone_pulse_workspace(repository)
        return _open(
            connection_factory=standalone_pulse_connection_factory(workspace),
            initial_connection_mode="offline",
            document=document,
            path=path,
            remote_endpoint=remote_endpoint,
        )

    if remote_endpoint is not None or repository is not None:
        raise ValueError(
            "a bound Experiment already owns its Pulse endpoint and workspace"
        )
    from ._composition import bound_pulse_mode, _require_experiment

    experiment = _require_experiment(experiment)

    def compose():
        pulse = experiment.pulse
        return _open(
            pulse=pulse,
            descriptor=pulse.target,
            initial_connection_mode=bound_pulse_mode(experiment),
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
    repository=None,
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
            for value in (document, config_path, repository, on_initialized)
        ):
            raise ValueError(
                "a bound DeviceManager takes its config from the Experiment"
            )

        def compose():
            return _open(ExperimentDeviceAdmin.bound(experiment))

        return experiment._open_workbench_handle("device-manager", compose)

    admin = ExperimentDeviceAdmin.standalone(
        repository=repository,
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
    from ._composition import _require_experiment, task_console_ports

    experiment = _require_experiment(experiment)
    if "hide_on_close" in kwargs:
        raise TypeError(
            "a bound TaskConsole always preserves its window on X; "
            "hide_on_close is not caller-configurable"
        )

    def compose():
        return _open(
            task_console_ports(experiment),
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
    "open_saved_fit_grid_workbench",
    "open_task_console",
]
