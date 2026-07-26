"""Shared application mechanics for the two release-recapture Measurements."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import BlockId, DatasetSchema, dataset_revision_ref_to_tree
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration, FinalDatasetOutput, final_dataset_join_digest
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import calibration_artifact_ref_to_tree
from zlc_neutral_atom.logic_nodes.release_recapture.pipeline import ReleaseRecapturePipelineSpec, release_recapture_output_schema
from zlc_neutral_atom.logic_nodes.release_recapture.timing import TriggeredReleaseRecaptureResult, TriggeredReleaseRecaptureSpec, compile_triggered_release_recapture_pipeline
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_storage import canonical_digest


class PreparedReleaseRecapture:
    """One immutable, one-shot flat survival pipeline."""

    __slots__ = ("_schema", "_spec", "_start_run", "_started", "_lock")

    def __init__(
        self,
        spec: TriggeredReleaseRecaptureSpec,
        *,
        output_schema: DatasetSchema,
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(spec, TriggeredReleaseRecaptureSpec):
            raise TypeError("spec must be TriggeredReleaseRecaptureSpec")
        if not isinstance(output_schema, DatasetSchema):
            raise TypeError("output_schema must be DatasetSchema")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._spec = spec
        self._schema = output_schema
        self._start_run = start_run
        self._started = False
        self._lock = threading.Lock()

    @property
    def output_schema(self) -> DatasetSchema:
        return self._schema

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError(
                    "PreparedReleaseRecapture is one-shot"
                )
            self._started = True
        return self._start_run(
            compile_triggered_release_recapture_pipeline(self._spec)
        )


def final_release_recapture_output(
    result: TriggeredReleaseRecaptureResult,
    *,
    declaration: DatasetOutputDeclaration,
    owner: str,
) -> dict[str, FinalDatasetOutput]:
    if type(result) is not TriggeredReleaseRecaptureResult:
        raise TypeError("result must be TriggeredReleaseRecaptureResult")
    snapshot = result.survival
    pipeline = result.release_recapture.pipeline
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
                camera_binding.measurement.capture_contract
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
        camera_binding.measurement,
        calibration,
        selected.kind,
        per_site,
        StreamId(f"release-recapture-{identity}"),
        f"release-recapture-{identity}",
        BlockId(f"release-recapture-{identity[:20]}"),
    )
    output_schema = release_recapture_output_schema(pipeline)
    triggered = TriggeredReleaseRecaptureSpec(
        pipeline,
        camera_binding.pulse_port,
        camera_binding.pulse_request,
        camera_binding.trigger_channel,
        camera_binding.cell_plan,
        rf_port=rf_port,
        rf_table=rf_table,
    )
    return PreparedReleaseRecapture(
        triggered,
        output_schema=output_schema,
        start_run=start_run,
    )


__all__ = ["PreparedReleaseRecapture", "final_release_recapture_output", "prepare_release_recapture"]
