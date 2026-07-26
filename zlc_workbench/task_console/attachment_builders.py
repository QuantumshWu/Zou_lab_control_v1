"""Domain-neutral builders for explicit TaskConsole capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from zlc_workbench.task_console.capability import (
    ConsoleCapabilityAttachment,
    ConsoleNodeHost,
)
from zlc_workbench.task_console.processor_node import (
    ConsoleProcessorNode,
    ConsoleProcessorPublication,
)
from zlc_workbench.task_console.run_bridge import ConsoleRunNode


def run_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object], object],
    start: Callable[[object, ConsoleRunNode, ConsoleNodeHost], object] | None = None,
    start_with_live_output: Callable[[object, object], object] | None = None,
    materialize_final_presentations: Callable[[object, object, object], object]
    | None = None,
    resolve_artifact_reference: Callable[[object], object] | None = None,
) -> ConsoleCapabilityAttachment:
    """Attach one finite Task/Measurement through the common Run host."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if start is not None and not callable(start):
        raise TypeError("start must be callable or None")
    if start_with_live_output is not None and not callable(start_with_live_output):
        raise TypeError("start_with_live_output must be callable or None")
    if start is not None and start_with_live_output is not None:
        raise ValueError("Run attachment accepts only one custom start seam")

    def create_node(
        host: ConsoleNodeHost,
        current_spec,
        values: Mapping[str, object],
        instance_id: str,
        instance_label: str,
    ):
        if current_spec is not spec:
            raise RuntimeError("attachment received another ConsoleNodeSpec")
        inputs = host.bind_inputs(
            spec,
            values,
            resolve_artifact_reference=resolve_artifact_reference,
        )
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
            if start_with_live_output is not None:
                from .live_output import start_with_console_live_output

                return start_with_console_live_output(
                    command,
                    node,
                    host,
                    start=start_with_live_output,
                )
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
    project_presentations: Callable[..., Mapping[str, object]] | None = None,
    resolve_artifact_reference: Callable[[object], object] | None = None,
) -> ConsoleCapabilityAttachment:
    """Attach a source-driven Processor to the host's internal live lane."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if project_presentations is not None and not callable(project_presentations):
        raise TypeError("project_presentations must be callable or None")

    def materialize_publication(result, source):
        outputs = getattr(result, "outputs", None)
        if not isinstance(outputs, Mapping):
            raise TypeError("Processor evaluation exposes no typed output mapping")
        presentations = (
            {}
            if project_presentations is None
            else project_presentations(
                result,
                run_id=source.run_id,
                provenance_epoch_id=source.epoch_id,
            )
        )
        if not isinstance(presentations, Mapping):
            raise TypeError("Processor presentation projection must return a mapping")
        return ConsoleProcessorPublication(outputs, presentations)

    def create_node(
        host: ConsoleNodeHost,
        current_spec,
        values: Mapping[str, object],
        instance_id: str,
        instance_label: str,
    ):
        if current_spec is not spec:
            raise RuntimeError("attachment received another ConsoleNodeSpec")
        inputs = host.bind_inputs(
            spec,
            values,
            resolve_artifact_reference=resolve_artifact_reference,
        )
        authored = spec.build_request(values)
        request = bind_request(authored, inputs.bound)
        source_input = inputs.only_dataset()
        source_node = source_input.producer.run_node
        if source_node is None or not bool(getattr(source_node, "running", False)):
            raise RuntimeError(
                "start the selected Dataset producer before its Processor"
            )
        initial_source = host.current_value(source_input)
        source_event_source = (
            source_node if isinstance(source_node, SignalEventSource) else None
        )
        return ConsoleProcessorNode(
            spec,
            values,
            instance_id=instance_id,
            instance_label=instance_label,
            request=request,
            source_input=source_input,
            initial_source=initial_source,
            source_event_source=source_event_source,
            prepare_application=lambda: prepare(request),
            materialize_publication=materialize_publication,
            data_plane=host.data_plane,
            request_owner_wake=host.request_owner_wake,
        )

    return ConsoleCapabilityAttachment(spec, create_node)


__all__ = ["processor_attachment", "run_attachment"]
