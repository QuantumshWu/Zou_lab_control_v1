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

    ``experiment`` is the current ``Zou_lab_control.notebook`` Experiment.  The
    seams are derived from it HERE and nowhere else: the skeleton is handed a
    catalog view, a run factory and a data plane, and never imports the domain.
    """

    from Zou_lab_control.notebook.facade import _prepare_camera_monitor_for_workbench
    from zlc_frontend.figure import DatasetId
    from zlc_workbench.live import LiveDatasetSlot

    from .catalog_bridge import ConsoleCatalogView
    from .data_plane import ConsoleDataPlane
    from .plot_bridge_console import show_task_console
    from .run_bridge import ConsoleRunNode

    catalog_view = ConsoleCatalogView(experiment)
    data_plane = ConsoleDataPlane()
    console: list = []            # filled below; the wake closure needs the body

    def request_owner_wake() -> None:
        """A worker finished something -- let the next tick pick it up.

        The console already polls every tick, so waking is a no-op here; the
        callback exists because the mailbox promises never to block a worker on
        the GUI thread, and a surface that DID need an immediate repaint would
        hook it here rather than inside the run bridge.
        """

    def run_factory(spec, values):
        node = ConsoleRunNode(
            spec, values,
            prepare=lambda request: _prepare_camera_monitor_for_workbench(experiment, request),
            request_owner_wake=request_owner_wake,
        )
        _bind_monitor_view(node, data_plane)
        return node

    body = show_task_console(
        state=state, task=task,
        catalog_view=catalog_view,
        run_factory=run_factory,
        data_plane=data_plane,
        **kwargs,
    )
    console.append(body)
    return body


def _bind_monitor_view(node, data_plane) -> None:
    """Teach one run node how to start WITH a live view, and register that view.

    A camera monitor starts only through ``start_with_view``: the factory it
    receives runs on the worker, builds this node's LiveDatasetSlot, and hands it
    to the data plane, which is what makes the node's frames reach the board.
    """

    from zlc_frontend.figure import DatasetId
    from zlc_workbench.live import LiveDatasetSlot

    dataset_id = DatasetId(f"console-{node.spec.key.stable_definition_id}-{id(node):x}")

    def start(command):
        if not hasattr(command, "start_with_view"):
            return command.start()

        def factory(view_spec):
            slot = LiveDatasetSlot(
                view_spec,
                dataset_id=dataset_id,
                retain_on_terminal=False,
            )
            data_plane.attach(node, slot)

            def source_changed() -> None:
                data_plane.mark_changed(node)
                request_owner_wake()

            slot.set_change_listener(source_changed)
            return slot

        return command.start_with_view(factory=factory)

    node.bind_starter(start)
