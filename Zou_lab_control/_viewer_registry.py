"""Process-wide viewer registry (dependency-free decoupling seam).

The experiment layer (``neutral_atom``) produces DATA and never imports the
frontend.  Optional plotting is routed through whatever viewer is registered
here; the ``frontend`` package registers itself on import.  This module imports
nothing, so neither package pulls the other at import time:

    neutral_atom  --(reads)-->  _viewer_registry  <--(registers)--  frontend

If no viewer is registered (headless use, or the frontend was never imported),
the plot adapters return ``None`` and the data on result objects is still
complete.  A registered plotter must provide ``plot``, ``display_figure`` and
``run`` matching the frontend's public functions of the same name.
"""

from __future__ import annotations

from typing import Any, Optional

_PLOTTER: Optional[Any] = None


def register_plotter(plotter: Any) -> None:
    """Register the active viewer (called by ``frontend`` on import)."""

    global _PLOTTER
    _PLOTTER = plotter


def active_plotter() -> Optional[Any]:
    """Return the registered viewer, or ``None`` when running headless."""

    return _PLOTTER


def has_plotter() -> bool:
    return _PLOTTER is not None


__all__ = ["register_plotter", "active_plotter", "has_plotter"]
