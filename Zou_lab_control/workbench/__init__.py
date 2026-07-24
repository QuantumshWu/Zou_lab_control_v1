"""Lazy desktop entry points; importing this module never imports Qt."""

from __future__ import annotations


def open_capture_workbench(experiment, request):
    """Open the finite exact-capture Workbench without owning the Experiment."""

    from zlc_workbench.capture.app import open_capture_workbench as _open

    return _open(experiment, request)


def open_calibration_report_workbench(
    computation_loader,
    reference,
):
    """Open one committed calibration report without eager Qt imports."""

    from zlc_workbench.calibration_workbench.app import (
        open_calibration_report_workbench as _open,
    )

    return _open(computation_loader, reference)


def open_calibration_workbench(
    computation_loader,
    run_starter,
    *,
    seed=None,
    reference=None,
    timeout_seconds=None,
):
    """Open formal calibration creation/editing without eager Qt imports."""

    from zlc_workbench.calibration_workbench.app import (
        open_calibration_workbench as _open,
    )

    options = {"seed": seed, "reference": reference}
    if timeout_seconds is not None:
        options["timeout_seconds"] = timeout_seconds
    return _open(
        computation_loader,
        run_starter,
        **options,
    )


def open_data_figure_workbench(figure):
    """Open one already-resolved DataFigure without eager Qt imports."""

    from zlc_workbench.data_figure.app import open_data_figure_workbench as _open

    return _open(figure)


def create_data_figure_pane(
    figure,
    *,
    initial_display=None,
    initial_fit_result_identity=None,
    embedded=True,
):
    """Build the shared DataFigure body for an owning Workbench shell."""

    from zlc_workbench.data_figure.app import create_data_figure_pane as _create

    return _create(
        figure,
        initial_display=initial_display,
        initial_fit_result_identity=initial_fit_result_identity,
        embedded=embedded,
    )


def open_occupancy_cell_workbench(
    navigation_loader,
    cell_loader,
    reference,
    *,
    selection=None,
):
    """Open one exact same-shot occupancy map without eager Qt imports."""

    from zlc_workbench.occupancy_viewer.app import (
        open_occupancy_cell_workbench as _open,
    )

    return _open(
        navigation_loader,
        cell_loader,
        reference,
        selection=selection,
    )


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    occupancy_output=None,
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
    if occupancy_output is not None:
        options["occupancy_output"] = occupancy_output
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


def open_scan_workbench(experiment, request):
    """Open the current typed autonomous scan panel lazily."""

    from zlc_workbench.scan_workbench.app import open_scan_workbench as _open

    return _open(experiment, request)


def open_task_console(experiment, *, state=None, task=None, **kwargs):
    """Open the sole current Monitor/Logic TaskConsole lazily."""

    from zlc_workbench.task_console.app import open_task_console as _open

    return _open(experiment, state=state, task=task, **kwargs)


__all__ = [
    "open_calibration_workbench",
    "open_calibration_report_workbench",
    "open_capture_workbench",
    "create_data_figure_pane",
    "open_data_figure_workbench",
    "open_figure_workbench",
    "open_occupancy_cell_workbench",
    "open_saved_fit_grid_workbench",
    "open_scan_workbench",
    "open_task_console",
]
