"""Saved-fit grid Workbench composition and launch function."""

from __future__ import annotations

from zlc_neutral_atom.fit_reference import FitResultArtifactRef
from zlc_workbench.window_runtime import open_workbench_window

from .window import SavedFitGridWindow

def open_saved_fit_grid_workbench(
    view_loader,
    refit_opener,
    reference: FitResultArtifactRef,
) -> SavedFitGridWindow:
    return open_workbench_window(
        lambda: SavedFitGridWindow(
            view_loader,
            refit_opener,
            reference,
        )
    )


__all__ = ["open_saved_fit_grid_workbench"]
