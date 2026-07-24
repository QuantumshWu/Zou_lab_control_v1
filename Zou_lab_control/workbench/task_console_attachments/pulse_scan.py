"""PulseScan Measurement presentation and lifecycle attachment."""

from __future__ import annotations

from zlc_neutral_atom.logic_nodes.pulse_scan import (
    PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS,
    PULSE_SCAN_MEASUREMENT_DEFINITION,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.application import PreparedExactScan
from zlc_neutral_atom.logic_nodes.pulse_scan.authoring import (
    build_pulse_scan_program,
    pulse_scan_authoring_schema,
    pulse_scan_input_specs,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.contracts import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import bind_scan_source
from zlc_workbench.form_projection import PathPresentation, project_authoring_form
from zlc_workbench.input_binding import project_input_fields
from zlc_workbench.task_console.capability import ConsoleCapabilityAttachment
from zlc_workbench.task_console.catalog_bridge import (
    ConsoleNodeSpec,
    ConsoleSignalDecl,
)
from zlc_workbench.task_console.run_bridge import ConsoleRunNode

from .pulse_scan_form import pulse_scan_form
from .pulse_scan_parameter_form import PulseScanParameterForm


def pulse_scan_attachment(*, prepare, read_pulse_template):
    """Bind PulseScan without letting TaskConsole interpret its physical y."""

    projected = project_authoring_form(
        pulse_scan_authoring_schema(),
        path_presentations={
            "pulse": PathPresentation(
                mode="file",
                file_filter="Pulse program (*.json);;All files (*)",
                base_dir="pulses",
            )
        },
    )
    inputs = pulse_scan_input_specs()
    form_spec = pulse_scan_form(projected.fields[0])
    input_fields = project_input_fields(inputs)

    def editor_factory(*, runtime, parent=None):
        return PulseScanParameterForm(
            form_spec,
            input_fields=input_fields,
            runtime=runtime,
            pulse_template_reader=read_pulse_template,
            parent=parent,
        )
    spec = ConsoleNodeSpec(
        definition=PULSE_SCAN_MEASUREMENT_DEFINITION,
        title="Pulse scan",
        description="Acquire one exact Dataset over a Pulse program scan table",
        form=form_spec,
        declared_outputs=(
            ConsoleSignalDecl(
                PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS[0],
                "scan",
                "Signal",
                "scan result",
            ),
        ),
        build_request=build_pulse_scan_program,
        input_specs=inputs,
        input_fields=input_fields,
        editor_factory=editor_factory,
    )

    def create_node(host, current_spec, values, instance_id, instance_label):
        if current_spec is not spec:
            raise RuntimeError("attachment received another ConsoleNodeSpec")
        resolved = host.bind_inputs(spec, values)
        program = spec.build_request(values)
        if not isinstance(
            program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("PulseScan owner returned another program type")
        source = bind_scan_source(resolved.bound.dataset(inputs[0]))
        serving_nodes = resolved.runtime_nodes()

        def prepare_scan(current):
            if current != program:
                raise RuntimeError("PulseScan program changed after request freeze")
            for node in serving_nodes:
                node.cancel("PulseScan is taking exact source ownership")
            for node in serving_nodes:
                wait = getattr(node, "wait_until_terminal", None)
                if callable(wait):
                    wait(reason="PulseScan is taking exact source ownership")
            return prepare(current, source)

        node = ConsoleRunNode(
            spec,
            values,
            instance_id=instance_id,
            instance_label=instance_label,
            prepare=prepare_scan,
            request_owner_wake=host.request_owner_wake,
            frozen_request=program,
        )

        def start_scan(command):
            if not isinstance(command, PreparedExactScan):
                raise TypeError("PulseScan preparer returned another command type")
            return command.start()

        node.bind_starter(start_scan)
        return node

    return ConsoleCapabilityAttachment(spec, create_node)


__all__ = ["pulse_scan_attachment"]
