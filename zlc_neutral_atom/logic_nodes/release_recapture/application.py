"""Shared application mechanics for the two release-recapture Measurements."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import BlockId
from zlc_data.codec import dataset_revision_ref_to_tree
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration, FinalDatasetOutput, final_dataset_join_digest
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import calibration_artifact_ref_to_tree
from zlc_neutral_atom.logic_nodes.release_recapture.pipeline import ReleaseRecapturePipelineSpec
from zlc_neutral_atom.logic_nodes.release_recapture.timing import TriggeredReleaseRecaptureResult, compile_triggered_release_recapture_pipeline
from zlc_neutral_atom.devices.rf import rf_table_terminal_to_tree
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.timing.lineage import pulse_capture_evidence_to_tree
from zlc_storage import canonical_digest


class PreparedReleaseRecapture:
    """One immutable, one-shot flat survival pipeline."""

    __slots__ = (
        "_declaration",
        "_final_owner",
        "_spec",
        "_start_run",
        "_started",
        "_lock",
    )

    def __init__(
        self,
        spec: ReleaseRecapturePipelineSpec,
        *,
        declaration: DatasetOutputDeclaration,
        final_owner: str,
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(spec, ReleaseRecapturePipelineSpec):
            raise TypeError("spec must be ReleaseRecapturePipelineSpec")
        if not isinstance(declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not isinstance(final_owner, str) or not final_owner:
            raise ValueError("final_owner must be non-empty")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._spec = spec
        self._declaration = declaration
        self._final_owner = final_owner
        self._start_run = start_run
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError(
                    "PreparedReleaseRecapture is one-shot"
                )
            self._started = True
        plan = compile_triggered_release_recapture_pipeline(self._spec)
        return self._start_run(
            plan.with_lifecycle(
                owner=self,
                preemptible=False,
            )
        )

    def final_dataset_outputs(
        self,
        result: TriggeredReleaseRecaptureResult,
    ) -> dict[str, FinalDatasetOutput]:
        return _final_release_recapture_output(
            result,
            declaration=self._declaration,
            owner=self._final_owner,
        )


def _final_release_recapture_output(
    result: TriggeredReleaseRecaptureResult,
    *,
    declaration: DatasetOutputDeclaration,
    owner: str,
) -> dict[str, FinalDatasetOutput]:
    if type(result) is not TriggeredReleaseRecaptureResult:
        raise TypeError("result must be TriggeredReleaseRecaptureResult")
    snapshot = result.survival
    pipeline = result.pipeline
    output = FinalDatasetOutput(
        declaration,
        snapshot,
        final_dataset_join_digest(
            owner=owner,
            declaration=declaration,
            source_identity={
                "run_id": pipeline.run_id,
                "dataset": dataset_revision_ref_to_tree(snapshot.ref),
                "chain": pipeline.chain_contract_digest,
                "pulse": pulse_capture_evidence_to_tree(
                    result.lineage.evidence()
                ),
                "rf": rf_table_terminal_to_tree(result.rf_terminal),
            },
            snapshot=snapshot,
        ),
    )
    return {output.name: output}


def prepare_release_recapture(
    *,
    name: str,
    owner: str,
    program_fingerprint: str,
    camera_binding,
    calibration: ResolvedCalibration,
    model_kind,
    per_site: bool,
    declaration: DatasetOutputDeclaration,
    final_owner: str,
    start_run: Callable[[RunPlan], RunHandle],
    rf_port=None,
    rf_table=None,
) -> PreparedReleaseRecapture:
    selected = calibration.artifact.select_model(model_kind)
    identity = canonical_digest(
        {
            "owner": owner,
            "program": program_fingerprint,
            "camera_schema": (
                camera_binding.capture.capture_contract
                .dataset_schema.fingerprint
            ),
            "calibration": calibration_artifact_ref_to_tree(
                calibration.reference
            ),
            "model_kind": selected.kind.value,
            "per_site": per_site,
        }
    )
    pipeline = ReleaseRecapturePipelineSpec(
        name,
        camera_binding,
        calibration,
        selected,
        per_site,
        StreamId(f"release-recapture-{identity}"),
        f"release-recapture-{identity}",
        BlockId(f"release-recapture-{identity[:20]}"),
        rf_port=rf_port,
        rf_table=rf_table,
    )
    return PreparedReleaseRecapture(
        pipeline,
        declaration=declaration,
        final_owner=final_owner,
        start_run=start_run,
    )


__all__ = ["PreparedReleaseRecapture", "prepare_release_recapture"]
