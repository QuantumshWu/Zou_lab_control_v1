"""Domain-neutral builders for explicit TaskConsole capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.processing.hosted_processor import (
    HostedProcessor,
    ProcessorPublication,
)
from zlc_neutral_atom.processing.signal_plane import SignalPublication
from zlc_neutral_atom.runtime.hosted_run import HostedRun
from zlc_workbench.task_console.capability import (
    ConsoleCapabilityAttachment,
    ConsoleNodeHost,
)
from zlc_workbench.task_console.console_records import console_signal_key


def run_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object], object],
    start: Callable[[object, HostedRun, ConsoleNodeHost], object] | None = None,
    start_with_live_output: Callable[[object, object], object] | None = None,
    project_signal_presentation: Callable[
        [object, str, SignalPublication], object | None
    ]
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
    live_output_starter = start_with_live_output

    def create_node(
        host: ConsoleNodeHost,
        current_spec,
        values: Mapping[str, object],
        instance_id: str,
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
        output_presentations = tuple(spec.outputs_for(request))
        node = HostedRun(
            definition_key=spec.key,
            request=request,
            instance_id=instance_id,
            dataset_output_declarations=tuple(
                output.declaration for output in output_presentations
            ),
            artifact_output_declarations=tuple(
                output.declaration for output in spec.artifact_outputs
            ),
            prepare=prepare,
            qualify_output=lambda name: console_signal_key(instance_id, name),
            request_owner_wake=host.request_owner_wake,
        )
        def start_prepared(command):
            if live_output_starter is not None:
                from zlc_neutral_atom.runtime.live_output_host import (
                    start_with_live_output as host_live_output,
                )

                return host_live_output(
                    command,
                    node,
                    host.data_plane,
                    start=live_output_starter,
                )
            if start is None:
                starter = getattr(command, "start", None)
                if not callable(starter):
                    raise TypeError("prepared command exposes no start()")
                return starter()
            return start(command, node, host)

        node.bind_starter(start_prepared)
        return node

    return ConsoleCapabilityAttachment(
        spec,
        create_node,
        project_signal_presentation,
    )


def processor_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object], object],
    project_signal_presentation: Callable[
        [object, str, SignalPublication], object | None
    ]
    | None = None,
    resolve_artifact_reference: Callable[[object], object] | None = None,
) -> ConsoleCapabilityAttachment:
    """Attach a source-driven Processor to the host's internal live lane."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if project_signal_presentation is not None and not callable(
        project_signal_presentation
    ):
        raise TypeError("project_signal_presentation must be callable or None")

    def materialize_publication(result, source):
        outputs = getattr(result, "outputs", None)
        if not isinstance(outputs, Mapping):
            raise TypeError("Processor evaluation exposes no typed output mapping")
        return ProcessorPublication(outputs)

    def create_node(
        host: ConsoleNodeHost,
        current_spec,
        values: Mapping[str, object],
        instance_id: str,
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
        if source_input.transform_spec is not None:
            raise ValueError("latest-only Processor input must be a direct output")
        source_node = source_input.producer.run_node
        if source_node is None or not bool(getattr(source_node, "running", False)):
            raise RuntimeError(
                "start the selected Dataset producer before its Processor"
            )
        initial_publication = host.current_publication(source_input)

        return HostedProcessor(
            definition_key=spec.key,
            request=request,
            instance_id=instance_id,
            dataset_output_declarations=tuple(
                output.declaration for output in spec.outputs_for(request)
            ),
            source_signal=source_input.selection.signal_key,
            source_node=source_node,
            initial_publication=initial_publication,
            prepare_application=lambda: prepare(request),
            materialize_publication=materialize_publication,
            qualify_output=lambda name: console_signal_key(instance_id, name),
            data_plane=host.data_plane,
            request_owner_wake=host.request_owner_wake,
        )

    return ConsoleCapabilityAttachment(
        spec,
        create_node,
        project_signal_presentation,
    )


__all__ = ["processor_attachment", "run_attachment"]
