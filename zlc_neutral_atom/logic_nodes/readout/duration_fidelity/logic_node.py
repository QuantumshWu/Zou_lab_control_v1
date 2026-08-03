"""The sole discovered descriptor for source-neutral duration fidelity."""

from __future__ import annotations

from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.logic_node import (
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.application import compile_pulse_scan
from zlc_neutral_atom.logic_nodes.pulse_scan.artifact import materialize_scan_data
from zlc_neutral_atom.logic_nodes.pulse_scan.authoring import (
    PULSE_SCAN_SOURCE_INPUT_SPEC,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import ScanArtifactRef
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import (
    PulseScanRequest,
    ScanSignalBinding,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeExecutionContext
from zlc_neutral_atom.runtime.signal_source import SignalEventAssociationSource
from zlc_pulse import load_pulse_document
from zlc_storage import resolve_under

from .measurement import (
    READOUT_DURATION_FIDELITY_DEFINITION,
    READOUT_DURATION_FIDELITY_OUTPUT_DECLARATION,
    ReadoutDurationFidelityRequest,
    build_readout_duration_fidelity_request,
    build_readout_duration_program,
    readout_duration_fidelity_authoring_schema,
)


def _bind_execute(request: object, context: LogicNodeApplicationContext):
    if not isinstance(request, ReadoutDurationFidelityRequest):
        raise TypeError("duration request must be ReadoutDurationFidelityRequest")
    signal_name = context.input(PULSE_SCAN_SOURCE_INPUT_SPEC)
    if not isinstance(signal_name, str):
        raise TypeError("duration y input must resolve to one signal name")
    _generation, source, output_name = context.signal_plane.signal_event_binding(
        signal_name
    )
    if not isinstance(source, SignalEventAssociationSource):
        raise TypeError("duration y signal has no formal event association")
    document = load_pulse_document(resolve_under(context.pulses_root, request.pulse))
    scan_request = PulseScanRequest(
        build_readout_duration_program(request, document),
        ScanSignalBinding(signal_name, output_name),
    )
    pulse_port = context.device("sequencer_instance_id", "pulse.execute")
    project_root = context.project_root

    def execute(execution: LogicNodeExecutionContext):
        plan = compile_pulse_scan(
            scan_request,
            source,
            pulse_port=pulse_port,
            project_root=project_root,
        ).with_lifecycle(owner=execution, preemptible=False)
        reference = execution.start_and_wait(lambda: context.start_run(plan))
        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("duration scan returned another result type")
        snapshot = materialize_scan_data(project_root, reference).snapshot
        output = FinalDatasetOutput(
            READOUT_DURATION_FIDELITY_OUTPUT_DECLARATION,
            snapshot,
        )
        execution.publish_final({output.name: output})
        return reference

    return execute


LOGIC_NODE = LogicNodeDescriptor(
    api_name="readout_duration_fidelity",
    definition=READOUT_DURATION_FIDELITY_DEFINITION,
    description=(
        "Sample any associated y signal over detection duration; shots form R"
    ),
    authoring_schema=readout_duration_fidelity_authoring_schema(),
    input_specs=(PULSE_SCAN_SOURCE_INPUT_SPEC,),
    outputs=(
        DatasetOutputSpec(
            READOUT_DURATION_FIDELITY_OUTPUT_DECLARATION,
            "fidelity",
            "Signal",
            "Lossless per-shot y samples over detection duration",
        ),
    ),
    build_request=build_readout_duration_fidelity_request,
    bind_execute=_bind_execute,
    device_requirements=(("sequencer_instance_id", ("pulse.execute",)),),
)


__all__ = ["LOGIC_NODE"]
