"""Qt composition entry point for the exact occupancy-cell viewer."""

from __future__ import annotations


def open_occupancy_cell_workbench(
    navigation_loader,
    cell_loader,
    reference,
    *,
    selection=None,
):
    from zlc_workbench.window_runtime import open_workbench_window

    from .window import OccupancyCellWindow

    return open_workbench_window(
        lambda: OccupancyCellWindow(
            navigation_loader,
            cell_loader,
            reference,
            selection=selection,
        )
    )
