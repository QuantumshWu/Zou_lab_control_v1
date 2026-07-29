"""Pulse Scan's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    UiContributionDescriptor,
)

from .api import PulseScanApi
from .application import prepare_exact_scan
from .declaration import PULSE_SCAN_LOGIC_NODE
from .reference import (
    SCAN_ARTIFACT_REF_SCHEMA,
    ScanArtifactRef,
    scan_artifact_ref_from_tree,
    scan_artifact_ref_to_tree,
)
from .repository import ScanRepository


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> PulseScanApi:
    (
        repository_root,
        resolve_sequencer_ref,
        pulse_port,
        start_run,
    ) = facts
    repository = ScanRepository(repository_root / "scans")

    def prepare(request, source, sequencer_role):
        sequencer_ref = resolve_sequencer_ref(sequencer_role)
        return prepare_exact_scan(
            request,
            source,
            pulse_port=pulse_port(sequencer_ref),
            repository=repository,
            start_run=start_run,
        )

    return PulseScanApi(
        repository,
        prepare=prepare,
    )


def _close_api(api: PulseScanApi) -> tuple[Exception, ...]:
    return api.close()


def _prepare_hosted(api, request, event_source):
    if event_source is None:
        raise ValueError("PulseScan requires one exact event-associated source")
    return api.prepare_scan_source(request, event_source)


def _availability(catalog, _apparatus):
    sequencer = catalog.find("sequencer")
    if sequencer is None or sequencer.domain != "sequencer":
        return "PulseScan requires the installed Sequencer role"
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
            admit_dataset_content=True,
        ),
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="pulse_scan",
    declaration=PULSE_SCAN_LOGIC_NODE,
    api_requirements=(
        "repository_root",
        "resolve_sequencer_ref",
        "pulse_port",
        "start_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    availability=_availability,
    ui_contributions=(
        UiContributionDescriptor(
            "task_console_editor",
            "zlc_neutral_atom.logic_nodes.pulse_scan.ui."
            "task_console_parameter_form",
            "task_console_editor",
        ),
    ),
    close_api=_close_api,
    bind_artifact_capabilities=_bind_artifact_capabilities,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
