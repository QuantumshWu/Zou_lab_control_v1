"""Small composition helpers shared by concrete TaskConsole attachments."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_workbench.task_console.capability import (
    ConsoleCapabilityAttachment,
    ConsoleNodeHost,
)
from zlc_workbench.task_console.processor_node import ConsoleProcessorNode
from zlc_workbench.task_console.run_bridge import ConsoleRunNode


def run_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object], object],
    start: Callable[[object, ConsoleRunNode, ConsoleNodeHost], object] | None = None,
    materialize_final_presentations: Callable[[object, object, object], object]
    | None = None,
) -> ConsoleCapabilityAttachment:
    """Attach one finite Task/Measurement through the common Run host."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if start is not None and not callable(start):
        raise TypeError("start must be callable or None")

    def create_node(
        host: ConsoleNodeHost,
        current_spec,
        values: Mapping[str, object],
        instance_id: str,
        instance_label: str,
    ):
        if current_spec is not spec:
            raise RuntimeError("attachment received another ConsoleNodeSpec")
        inputs = host.bind_inputs(spec, values)
        authored = spec.build_request(values)
        request = bind_request(authored, inputs.bound)
        node = ConsoleRunNode(
            spec,
            values,
            instance_id=instance_id,
            instance_label=instance_label,
            prepare=prepare,
            request_owner_wake=host.request_owner_wake,
            frozen_request=request,
            materialize_final_presentations=materialize_final_presentations,
        )

        def start_prepared(command):
            if start is None:
                starter = getattr(command, "start", None)
                if not callable(starter):
                    raise TypeError("prepared command exposes no start()")
                return starter()
            return start(command, node, host)

        node.bind_starter(start_prepared)
        return node

    return ConsoleCapabilityAttachment(spec, create_node)


def processor_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object], object],
    materialize_publication: Callable[[object, object], object],
) -> ConsoleCapabilityAttachment:
    """Attach a source-driven Processor to the host's internal live lane."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if not callable(materialize_publication):
        raise TypeError("materialize_publication must be callable")

    def create_node(
        host: ConsoleNodeHost,
        current_spec,
        values: Mapping[str, object],
        instance_id: str,
        instance_label: str,
    ):
        if current_spec is not spec:
            raise RuntimeError("attachment received another ConsoleNodeSpec")
        inputs = host.bind_inputs(spec, values)
        authored = spec.build_request(values)
        request = bind_request(authored, inputs.bound)
        source_input = inputs.only_dataset()
        source_node = source_input.producer.run_node
        if source_node is None or not bool(getattr(source_node, "running", False)):
            raise RuntimeError(
                "start the selected Dataset producer before its Processor"
            )
        initial_source = host.current_value(source_input)
        return ConsoleProcessorNode(
            spec,
            values,
            instance_id=instance_id,
            instance_label=instance_label,
            request=request,
            source_input=source_input,
            initial_source=initial_source,
            prepare_application=lambda: prepare(request),
            materialize_publication=materialize_publication,
            data_plane=host.data_plane,
            request_owner_wake=host.request_owner_wake,
        )

    return ConsoleCapabilityAttachment(spec, create_node)


def no_inputs(request: object, inputs: BoundNodeInputs) -> object:
    """Identity binder for an owner that declares no cross-node input."""

    if not isinstance(inputs, BoundNodeInputs):
        raise TypeError("inputs must be BoundNodeInputs")
    if inputs.values:
        raise ValueError("this Logic node declares no cross-node inputs")
    return request


__all__ = ["no_inputs", "processor_attachment", "run_attachment"]
