"""Pulse Scan's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    UiContributionDescriptor,
)

from .api import PulseScanApi
from .application import prepare_exact_scan
from .authoring import _build_pulse_scan_program
from .declaration import PULSE_SCAN_LOGIC_NODE
from .reference import (
    SCAN_ARTIFACT_REF_SCHEMA,
    ScanArtifactRef,
    scan_artifact_ref_from_tree,
    scan_artifact_ref_to_tree,
)
from .source_binding import bind_pulse_scan_request


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> PulseScanApi:
    (
        output_root,
        resolve_sequencer_ref,
        load_pulse,
        pulse_port,
        start_run,
        wait_run,
    ) = facts
    scans_root = output_root / "scans"

    def prepare(request, source, sequencer_role):
        sequencer_ref = resolve_sequencer_ref(sequencer_role)
        return prepare_exact_scan(
            request,
            source,
            pulse_port=pulse_port(sequencer_ref),
            scans_root=scans_root,
            start_run=start_run,
        )

    return PulseScanApi(
        scans_root,
        load_pulse=load_pulse,
        prepare=prepare,
        wait_run=wait_run,
    )


def _prepare_hosted(api, request, event_source):
    if event_source is None:
        raise ValueError("PulseScan requires one exact event-associated source")
    return api.prepare_scan(request, event_source)


def _bind_hosted_request(api, authored, inputs):
    program = _build_pulse_scan_program(
        authored,
        load_pulse=api._load_pulse,
    )
    return bind_pulse_scan_request(program, inputs)


def _availability(catalog, _apparatus):
    if not any(
        item.domain == "sequencer" and "pulse.execute" in item.capabilities
        for item in catalog.values()
    ):
        return "PulseScan requires a pulse sequencer"
    return None


def _bind_artifact_capabilities(
    api: PulseScanApi,
) -> tuple[ArtifactCapability, ...]:
    return (
        ArtifactCapability(
            format_id=SCAN_ARTIFACT_REF_SCHEMA,
            source_label="scan",
            reference_type=ScanArtifactRef,
            project_dataset=api._project_dataset_source,
            reference_to_tree=scan_artifact_ref_to_tree,
            reference_from_tree=scan_artifact_ref_from_tree,
        ),
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="pulse_scan",
    declaration=PULSE_SCAN_LOGIC_NODE,
    api_requirements=(
        "output_root",
        "resolve_sequencer_ref",
        "load_pulse",
        "pulse_port",
        "start_run",
        "wait_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    bind_hosted_request=_bind_hosted_request,
    availability=_availability,
    ui_contributions=(
        UiContributionDescriptor(
            "task_console_editor",
            "zlc_neutral_atom.logic_nodes.pulse_scan.ui."
            "task_console_parameter_form",
            "task_console_editor",
        ),
    ),
    bind_artifact_capabilities=_bind_artifact_capabilities,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
