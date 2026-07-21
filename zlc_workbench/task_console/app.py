"""The task console's composition root -- the one place that opens the console window.

Every entry goes through :func:`open_task_console`: the double-clickable
``task_console.bat``, the root ``task_console.py`` launcher, and
``Experiment.task_console()`` from a notebook.

The window is the ORIGINAL console UI -- the Monitor/Logic tabbed board, panel
cards, Fluent chrome -- hosted in :mod:`.plot_bridge_console` (the UI skeleton is
kept BY DIRECTIVE 2026-07-21; it is never redesigned).  Its DATA plane is being
rewired onto the CURRENT architecture per the design document's section 10; the
four contracted seams this root assembles, in rewiring order:

1. CATALOG -- the ``zlc_neutral_atom`` DefinitionCatalog (measurement /
   stream-processor / task definitions), mapped through a local CatalogView
   adapter into the skeleton's Add-Panel / Logic-tab vocabulary.  No global
   registry: plain imports; duplicate keys fail at startup.
2. RUN -- panel/logic Start compiles an immutable PipelineSpec ->
   ``compile_pipeline`` -> one flat RunPlan under a single RunController; the
   skeleton never starts nested runs or owns terminal state.
3. MONITOR -- live panels consume admitted ``MonitorTap -> MonitorDataset ->
   LiveDatasetSlot``: the tick reads coalesced revision notifications and takes
   atomic MonitorDatasetSnapshots; no mutable signal hub returns.
4. RENDER -- panels draw through the worker-raster pipeline (``zlc_frontend``
   encoded_raster / image_raster / render DTOs onto the qt_widgets raster
   boards); no transitional matplotlib live stack.

Until a seam lands the corresponding skeleton members stay disconnected -- the
window may not fully operate yet, which is the accepted state of the rewiring
phase (the purge deliberately preceded the reconnect).
"""

from __future__ import annotations

__all__ = ["open_task_console"]


def open_task_console(experiment, *, state=None, task=None, **kwargs):
    """Open the console UI for ``experiment`` and return the console body.

    ``experiment`` is the current ``Zou_lab_control.notebook`` Experiment; the
    four seams above are derived from it HERE and nowhere else -- the skeleton
    never imports the domain."""

    raise NotImplementedError(
        "task_console rewiring in progress (purge b68fc81 landed; reconnect phase): "
        "the catalog/run/monitor/render seams -- see this module's docstring, "
        "contracts 1-4 -- are being assembled onto the current zlc_neutral_atom "
        "application layer.  The ORIGINAL UI skeleton is intact in "
        "plot_bridge_console and reopens the moment seams 1+3 land."
    )
