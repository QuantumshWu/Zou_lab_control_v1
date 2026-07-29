"""Pulse Scan's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_package import LogicNodePackage
from zlc_storage.paths import resolve_under

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


def _bind_api(host: object, _dependencies: tuple[object, ...]) -> PulseScanApi:
    operations = host._logic_node_operations()
    repository = ScanRepository(operations.repository_root / "scans")
    pulse_port = operations.pulse_port
    start_run = operations.start_run
    resolve_sequencer_ref = host.resolve_readout_sequencer_ref

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
        pulses_root=operations.pulses_root,
        prepare=prepare,
    )


def _close_api(api: PulseScanApi) -> tuple[Exception, ...]:
    return api.close()


def _bind_task_console(api: PulseScanApi, _catalog: object, projection):
    from zlc_pulse import describe_pulse_template

    from .authoring import build_pulse_scan_program
    from .ui.task_console import pulse_scan_task_console_adapter

    def resolve_values(values):
        normalized = dict(values)
        normalized["pulse"] = str(
            resolve_under(api._pulses_root, normalized["pulse"])
        )
        return normalized

    return pulse_scan_task_console_adapter(
        prepare=api.prepare_scan_source,
        read_pulse_template=lambda path: describe_pulse_template(
            resolve_under(api._pulses_root, path)
        ),
        build_request=lambda values: build_pulse_scan_program(
            resolve_values(values)
        ),
        pulses_root=api._pulses_root,
        project_custom=projection.custom,
    )


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
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=80,
    close_api=_close_api,
    bind_artifact_capabilities=_bind_artifact_capabilities,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
