"""Compose the generic TaskConsole from explicit Logic-node attachments."""

from __future__ import annotations

from .application_ports import TaskConsoleApplicationPorts

__all__ = ["TaskConsoleApplicationPorts", "open_task_console"]


def open_task_console(
    ports: TaskConsoleApplicationPorts,
    *,
    state=None,
    task=None,
    **kwargs,
):
    """Open one shell that has no concrete Logic-node knowledge."""

    if not isinstance(ports, TaskConsoleApplicationPorts):
        raise TypeError("ports must be TaskConsoleApplicationPorts")

    from .capability import ConsoleNodeHost
    from .catalog_bridge import ConsoleCatalogView
    from .data_plane import ConsoleDataPlane
    from .window import show_task_console

    catalog_view = ConsoleCatalogView(
        tuple(attachment.spec for attachment in ports.attachments)
    )
    data_plane = ConsoleDataPlane()
    console: list[object] = []

    def request_owner_wake() -> None:
        # The shell polls one owner mailbox per GUI tick.  Keeping this callback
        # explicit means another host may schedule an immediate owner wake
        # without letting workers call QWidget methods.
        return None

    def resolve_inputs(spec, values):
        if not console:
            raise RuntimeError(
                "TaskConsole composition is not ready for input binding"
            )
        return console[0].resolve_node_inputs(spec, values)

    host = ConsoleNodeHost(
        data_plane=data_plane,
        resolve_inputs=resolve_inputs,
        request_owner_wake=request_owner_wake,
    )

    def run_factory(
        spec,
        values,
        *,
        instance_id: str,
        instance_label: str,
    ):
        attachment = ports.attachment_for(spec.key)
        if attachment is None or attachment.spec is not spec:
            raise RuntimeError(
                "TaskConsole catalog/attachment invariant was violated"
            )
        return attachment.create_node(
            host,
            spec,
            values,
            instance_id,
            instance_label,
        )

    body = show_task_console(
        state=state,
        task=task,
        catalog_view=catalog_view,
        run_factory=run_factory,
        data_plane=data_plane,
        **kwargs,
    )
    console.append(body)
    return body
