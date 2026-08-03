"""Domain-neutral builders for explicit TaskConsole capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.input_spec import DatasetInputSpec
from zlc_neutral_atom.processing.hosted_processor import (
    HostedProcessor,
)
from zlc_neutral_atom.runtime.hosted_run import HostedRun
from zlc_workbench.task_console.capability import (
    ConsoleCapabilityAttachment,
    ConsoleNodeHost,
)
from zlc_workbench.task_console.input_binding import ResolvedDatasetInput
from zlc_workbench.task_console.console_records import console_signal_key


def run_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object, object | None], object],
    start_prepared: Callable[[object, object, object], object]
    | None = None,
    resolve_artifact_reference: Callable[[object], object] | None = None,
) -> ConsoleCapabilityAttachment:
    """Attach one finite Task/Measurement through the common Run host."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if start_prepared is not None and not callable(start_prepared):
        raise TypeError("start_prepared must be callable or None")
    association_specs = tuple(
        value
        for value in spec.input_specs
        if isinstance(value, DatasetInputSpec) and value.requires_event_association
    )
    if len(association_specs) > 1:
        raise ValueError(
            "one hosted Run may consume at most one event-associated Dataset"
        )

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

        event_source = None
        event_generation = None
        event_signal = None
        event_output = None
        event_producer = None
        if association_specs:
            association = inputs.resolved[association_specs[0].key]
            if not isinstance(association, ResolvedDatasetInput):
                raise TypeError("event-associated input resolved as another type")
            event_producer = association.producer
            if not event_producer.running:
                raise RuntimeError(
                    "start the selected Dataset producer before this Logic node"
                )
            event_signal = association.selection.signal_key
            (
                event_generation,
                event_source,
                event_output,
            ) = (
                host.data_plane.signal_event_binding(
                    event_signal,
                )
            )
            if event_output != event_producer.output.name:
                raise RuntimeError("event route exposes another Dataset output")

        def prepare_owned(current):
            if current != request:
                raise RuntimeError("hosted request changed after request freeze")
            if event_signal is None:
                command = prepare(current, None)
            else:
                assert event_generation is not None
                assert event_producer is not None
                if not event_producer.running:
                    raise RuntimeError(
                        "selected Dataset producer stopped before prepare"
                    )
                (
                    generation,
                    source,
                    output_name,
                ) = host.data_plane.signal_event_binding(
                    event_signal,
                    expected_generation=event_generation,
                )
                if (
                    generation != event_generation
                    or source is not event_source
                    or output_name != event_output
                ):
                    raise RuntimeError("event-associated Dataset route changed")
                command = prepare(current, event_source)
                host.data_plane.bind_generation_source(
                    node,
                    source_name=event_signal,
                    expected_generation=event_generation,
                )
            return command

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
            prepare=prepare_owned,
            qualify_output=lambda name: console_signal_key(instance_id, name),
            request_owner_wake=host.request_owner_wake,
        )
        def start_prepared(command):
            if start_prepared_owner is not None:
                from zlc_neutral_atom.runtime.live_output_host import (
                    start_with_live_output as host_live_output,
                )

                return host_live_output(
                    command,
                    node,
                    host.data_plane,
                    start=lambda current, live_host: start_prepared_owner(
                        current,
                        live_host,
                        node.command_context,
                    ),
                )
            starter = getattr(command, "start", None)
            if not callable(starter):
                raise TypeError("prepared command exposes no start()")
            return starter(lifecycle_owner=node.command_context)

        node.bind_starter(start_prepared)
        return node

    start_prepared_owner = start_prepared
    return ConsoleCapabilityAttachment(spec, create_node)


def processor_attachment(
    spec,
    *,
    bind_request: Callable[[object, BoundNodeInputs], object],
    prepare: Callable[[object], object],
    resolve_artifact_reference: Callable[[object], object] | None = None,
) -> ConsoleCapabilityAttachment:
    """Attach a source-driven Processor to the host's internal live lane."""

    if not callable(bind_request):
        raise TypeError("bind_request must be callable")
    if not callable(prepare):
        raise TypeError("prepare must be callable")
    def materialize_publication(result, source):
        outputs = getattr(result, "outputs", None)
        if not isinstance(outputs, Mapping):
            raise TypeError("Processor evaluation exposes no typed output mapping")
        return outputs

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
        initial_publication = host.current_publication(source_input)

        return HostedProcessor(
            definition_key=spec.key,
            request=request,
            instance_id=instance_id,
            dataset_output_declarations=tuple(
                output.declaration for output in spec.outputs_for(request)
            ),
            source_signal=source_input.selection.signal_key,
            initial_publication=initial_publication,
            prepare_application=lambda: prepare(request),
            materialize_publication=materialize_publication,
            qualify_output=lambda name: console_signal_key(instance_id, name),
            data_plane=host.data_plane,
            request_owner_wake=host.request_owner_wake,
        )

    return ConsoleCapabilityAttachment(spec, create_node)


__all__ = ["processor_attachment", "run_attachment"]
