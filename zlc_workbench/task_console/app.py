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
    from .window import show_task_console

    catalog_view = ConsoleCatalogView(
        tuple(attachment.spec for attachment in ports.attachments)
    )
    console: list[object] = []

    def request_owner_wake() -> None:
        # Worker callbacks cross exactly one queued Qt-owner seam.  A node cannot
        # start before ``show_task_console`` has returned its composed body, so a
        # wake before that point is a broken composition order, not work to drop.
        if not console:
            raise RuntimeError("TaskConsole owner is not composed")
        console[0].request_owner_wake()

    data_plane = ports.data_plane

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

    def project_signal_presentation(node, output_name, publication):
        attachment = ports.attachment_for(node.definition_key)
        if attachment is None:
            return None
        projector = attachment.project_signal_presentation
        if projector is None:
            return None
        return projector(node, output_name, publication)

    def run_factory(
        spec,
        values,
        *,
        instance_id: str,
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
        )

    body = show_task_console(
        tasks_root=ports.tasks_root,
        output_root=ports.output_root,
        state=state,
        task=task,
        catalog_view=catalog_view,
        run_factory=run_factory,
        data_plane=data_plane,
        project_signal_presentation=project_signal_presentation,
        **kwargs,
    )
    console.append(body)
    return body
