"""Exceptional TaskConsole adapter for PulseScan's scan-table editor."""

from __future__ import annotations

from zlc_neutral_atom.logic_nodes.pulse_scan.application import PreparedExactScan
from zlc_neutral_atom.logic_nodes.pulse_scan.declaration import PULSE_SCAN_LOGIC_NODE
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import PulseScanBoundRequest
from zlc_neutral_atom.runtime.hosted_run import HostedRun
from zlc_workbench.task_console.capability import (
    ConsoleCapabilityAttachment,
    ConsoleSignalEventSourceProvider,
)
from zlc_workbench.task_console.declaration_projection import project_declaration_spec
from zlc_workbench.task_console.console_records import console_signal_key

from .task_console_form import pulse_scan_form


def pulse_scan_task_console_adapter(*, prepare, read_pulse_template):
    """Bind PulseScan without letting TaskConsole interpret its physical y."""

    base_spec = project_declaration_spec(PULSE_SCAN_LOGIC_NODE)
    inputs = PULSE_SCAN_LOGIC_NODE.input_specs
    form_spec = pulse_scan_form(base_spec.form.fields[0])
    input_fields = base_spec.input_fields

    def editor_factory(*, runtime, parent=None):
        from .task_console_parameter_form import PulseScanParameterForm

        return PulseScanParameterForm(
            form_spec,
            input_fields=input_fields,
            runtime=runtime,
            pulse_template_reader=read_pulse_template,
            parent=parent,
        )
    spec = project_declaration_spec(
        PULSE_SCAN_LOGIC_NODE,
        form=form_spec,
        editor_factory=editor_factory,
    )

    def create_node(host, current_spec, values, instance_id):
        if current_spec is not spec:
            raise RuntimeError("attachment received another ConsoleNodeSpec")
        resolved = host.bind_inputs(spec, values)
        program = spec.build_request(values)
        request = PULSE_SCAN_LOGIC_NODE.bind_request(program, resolved.bound)
        if not isinstance(request, PulseScanBoundRequest):
            raise TypeError("PulseScan owner returned another bound request type")
        source_input = resolved.only_dataset()
        source_node = source_input.producer.run_node
        if not isinstance(source_node, ConsoleSignalEventSourceProvider):
            raise ValueError("PulseScan source must be a running Logic node")

        def prepare_scan(current):
            if current != request:
                raise RuntimeError("PulseScan request changed after request freeze")
            if not source_input.producer.running:
                raise RuntimeError(
                    "start the selected signal producer before PulseScan"
                )
            return prepare(current, source_node.signal_event_source())

        node = HostedRun(
            definition_key=spec.key,
            request=request,
            instance_id=instance_id,
            dataset_output_declarations=tuple(
                output.declaration for output in spec.outputs_for(request)
            ),
            artifact_output_declarations=tuple(
                output.declaration for output in spec.artifact_outputs
            ),
            prepare=prepare_scan,
            qualify_output=lambda name: console_signal_key(instance_id, name),
            request_owner_wake=host.request_owner_wake,
        )

        def start_scan(command):
            if not isinstance(command, PreparedExactScan):
                raise TypeError("PulseScan preparer returned another command type")
            return command.start()

        node.bind_starter(start_scan)
        return node

    return ConsoleCapabilityAttachment(spec, create_node)


__all__ = ["pulse_scan_task_console_adapter"]
