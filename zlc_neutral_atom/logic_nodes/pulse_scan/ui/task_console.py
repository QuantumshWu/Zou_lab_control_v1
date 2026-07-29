"""Exceptional TaskConsole adapter for PulseScan's scan-table editor."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from zlc_neutral_atom.logic_nodes.pulse_scan.application import PreparedExactScan
from zlc_neutral_atom.logic_nodes.pulse_scan.declaration import PULSE_SCAN_LOGIC_NODE
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import PulseScanBoundRequest
from zlc_neutral_atom.runtime.hosted_run import HostedRun
from zlc_frontend.form import project_authoring_form

from .task_console_form import pulse_scan_form


def pulse_scan_task_console_adapter(
    *,
    prepare,
    read_pulse_template,
    build_request,
    pulses_root: Path,
    project_custom,
):
    """Bind PulseScan without letting TaskConsole interpret its physical y."""

    if not callable(project_custom):
        raise TypeError("project_custom must be callable")
    if not callable(build_request):
        raise TypeError("build_request must be callable")
    path_presentations = {
        value.field_key: (
            replace(value, base_dir=str(Path(pulses_root).resolve()))
            if value.base_dir == "pulses"
            else value
        )
        for value in PULSE_SCAN_LOGIC_NODE.path_presentations
    }
    base_form = project_authoring_form(
        PULSE_SCAN_LOGIC_NODE.authoring_schema,
        path_presentations=path_presentations,
    )
    form_spec = pulse_scan_form(base_form.fields[0])

    def editor_factory(*, runtime, input_fields, parent=None):
        from .task_console_parameter_form import PulseScanParameterForm

        return PulseScanParameterForm(
            form_spec,
            input_fields=input_fields,
            runtime=runtime,
            pulse_template_reader=read_pulse_template,
            parent=parent,
        )

    def create_node(host, current_spec, values, instance_id):
        resolved = host.bind_inputs(current_spec, values)
        program = build_request(values)
        request = PULSE_SCAN_LOGIC_NODE.bind_request(program, resolved.bound)
        if not isinstance(request, PulseScanBoundRequest):
            raise TypeError("PulseScan owner returned another bound request type")
        source_input = resolved.only_dataset()
        if not source_input.producer.running:
            raise ValueError("PulseScan source must be a running Logic node")
        signal_key = source_input.selection.signal_key
        frozen_source, frozen_output_name, frozen_transform = (
            host.data_plane.signal_event_binding(signal_key)
        )
        if frozen_output_name != request.signal.output.name:
            raise RuntimeError("PulseScan signal route exposes another output")
        if frozen_transform != request.signal.transform:
            raise RuntimeError("PulseScan signal route exposes another transform")

        def prepare_scan(current):
            if current != request:
                raise RuntimeError("PulseScan request changed after request freeze")
            if not source_input.producer.running:
                raise RuntimeError(
                    "start the selected signal producer before PulseScan"
                )
            source, output_name, transform = host.data_plane.signal_event_binding(
                signal_key,
            )
            if source is not frozen_source:
                raise RuntimeError("PulseScan signal route changed its event source")
            if output_name != request.signal.output.name:
                raise RuntimeError("PulseScan signal route changed its output")
            if transform != request.signal.transform:
                raise RuntimeError("PulseScan signal route changed its transform")
            return prepare(current, frozen_source)

        node = HostedRun(
            definition_key=current_spec.key,
            request=request,
            instance_id=instance_id,
            dataset_output_declarations=tuple(
                output.declaration for output in current_spec.outputs_for(request)
            ),
            artifact_output_declarations=tuple(
                output.declaration for output in current_spec.artifact_outputs
            ),
            prepare=prepare_scan,
            qualify_output=lambda name: host.qualify_output(instance_id, name),
            request_owner_wake=host.request_owner_wake,
        )

        def start_scan(command):
            if not isinstance(command, PreparedExactScan):
                raise TypeError("PulseScan preparer returned another command type")
            return command.start()

        node.bind_starter(start_scan)
        return node

    return project_custom(
        PULSE_SCAN_LOGIC_NODE,
        form=form_spec,
        editor_factory=editor_factory,
        create_node=create_node,
    )


__all__ = ["pulse_scan_task_console_adapter"]
