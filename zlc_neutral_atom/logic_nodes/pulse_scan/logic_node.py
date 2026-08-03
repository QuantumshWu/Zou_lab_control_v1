"""The sole discovered descriptor and operation for PulseScan."""

from __future__ import annotations

from zlc_neutral_atom.logic_node import (
    ArtifactOutputSpec,
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
    UiContribution,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeExecutionContext
from zlc_neutral_atom.runtime.signal_source import SignalEventAssociationSource
from zlc_pulse import load_pulse_document
from zlc_storage import resolve_under

from .application import compile_pulse_scan
from .artifact import (
    load_scan_artifact,
    materialize_scan_data,
    project_scan_dataset,
)
from .authoring import (
    PULSE_SCAN_SOURCE_INPUT_SPEC,
    _build_pulse_scan_program,
    _freeze_pulse_scan_authoring,
    pulse_scan_authoring_schema,
)
from .contracts import PULSE_SCAN_MEASUREMENT_DEFINITION
from .final_output import (
    PULSE_SCAN_ARTIFACT_OUTPUT_DECLARATION,
    PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS,
    scan_final_outputs,
)
from .reference import SCAN_ARTIFACT_REF_SCHEMA, ScanArtifactRef
from .source_binding import PulseScanRequest, ScanSignalBinding


_SEQUENCER_FIELD = "sequencer_instance_id"
_PULSE_CAPABILITY = "pulse.execute"
_EDITOR_UI = UiContribution(
    "task_console_editor",
    "zlc_neutral_atom.logic_nodes.pulse_scan.ui.task_console_parameter_form",
    "task_console_editor",
)


def _bind_execute(
    authored: object,
    context: LogicNodeApplicationContext,
):
    if not isinstance(authored, dict):
        raise TypeError("PulseScan authored request must be a mapping")
    signal_name = context.input(PULSE_SCAN_SOURCE_INPUT_SPEC)
    if not isinstance(signal_name, str):
        raise TypeError("PulseScan y input must resolve to one signal name")
    _generation, source, output_name = context.signal_plane.signal_event_binding(
        signal_name
    )
    if not isinstance(source, SignalEventAssociationSource):
        raise TypeError("PulseScan y signal has no formal event association")
    program = _build_pulse_scan_program(
        authored,
        load_pulse=lambda value: load_pulse_document(
            resolve_under(context.pulses_root, value)
        ),
    )
    request = PulseScanRequest(
        program,
        ScanSignalBinding(signal_name, output_name),
        trigger_port=authored["trigger_port"],
    )
    pulse_port = context.device(_SEQUENCER_FIELD, _PULSE_CAPABILITY)
    project_root = context.project_root

    def execute(execution: LogicNodeExecutionContext):
        plan = compile_pulse_scan(
            request,
            source,
            pulse_port=pulse_port,
            project_root=project_root,
        ).with_lifecycle(owner=execution, preemptible=False)
        reference = execution.start_and_wait(lambda: context.start_run(plan))
        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("PulseScan Run returned another result type")
        materialized = materialize_scan_data(project_root, reference)
        execution.publish_final(scan_final_outputs(materialized))
        context.remember_artifact(SCAN_ARTIFACT_REF_SCHEMA, reference)
        return reference

    return execute


def _load(
    context: LogicNodeApplicationContext,
    reference: ScanArtifactRef | None = None,
):
    selected = reference
    if selected is None:
        selected = context.default_artifact(SCAN_ARTIFACT_REF_SCHEMA)
    if not isinstance(selected, ScanArtifactRef):
        raise ValueError("no PulseScan artifact is selected")
    return load_scan_artifact(context.project_root, selected)


def _materialize(
    context: LogicNodeApplicationContext,
    reference: ScanArtifactRef | None = None,
):
    return materialize_scan_data(context.project_root, _load(context, reference).ref)


def _project(
    context: LogicNodeApplicationContext,
    reference: ScanArtifactRef,
    *,
    materialize: bool = False,
    abort_check=None,
):
    return project_scan_dataset(
        context.project_root,
        reference,
        materialize=materialize,
        abort_check=abort_check,
    )


LOGIC_NODE = LogicNodeDescriptor(
    api_name="pulse_scan",
    definition=PULSE_SCAN_MEASUREMENT_DEFINITION,
    description="Acquire an associated y signal over one frozen pulse scan",
    authoring_schema=pulse_scan_authoring_schema(),
    input_specs=(PULSE_SCAN_SOURCE_INPUT_SPEC,),
    outputs=(
        DatasetOutputSpec(
            PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS[0],
            "scan",
            "Signal",
            "Lossless R x P x data-shape scan result",
        ),
    ),
    artifact_outputs=(
        ArtifactOutputSpec(
            PULSE_SCAN_ARTIFACT_OUTPUT_DECLARATION,
            "scan artifact",
            "Durable PulseScan record",
        ),
    ),
    build_request=_freeze_pulse_scan_authoring,
    bind_execute=_bind_execute,
    device_requirements=((_SEQUENCER_FIELD, (_PULSE_CAPABILITY,)),),
    ui_contributions=(_EDITOR_UI,),
    operations={"load": _load, "materialize": _materialize, "project": _project},
)


__all__ = ["LOGIC_NODE"]
