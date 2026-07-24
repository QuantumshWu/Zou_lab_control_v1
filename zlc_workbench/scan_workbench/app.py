"""Qt composition entry point for the pulse-scan Workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import (
        OccupancyScanRequest,
        ScanRequest,
    )

    from .application import ScanWorkbenchActions

    from .window import ScanWorkbenchWindow


def open_scan_workbench(
    actions: ScanWorkbenchActions,
    request: ScanRequest | OccupancyScanRequest,
) -> ScanWorkbenchWindow:
    from zlc_workbench.window_runtime import open_workbench_window

    from .window import ScanWorkbenchWindow

    return open_workbench_window(lambda: ScanWorkbenchWindow(actions, request))
