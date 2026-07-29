"""Qt composition entry point for the exact occupancy-cell viewer."""

from __future__ import annotations


def open_occupancy_cell_workbench(
    navigation_loader,
    cell_loader,
    reference,
    *,
    address=None,
):
    from zlc_frontend.qt_widgets import launch_qt_window

    from .workbench_window import OccupancyCellWindow

    return launch_qt_window(
        lambda: OccupancyCellWindow(
            navigation_loader,
            cell_loader,
            reference,
            address=address,
        )
    )
