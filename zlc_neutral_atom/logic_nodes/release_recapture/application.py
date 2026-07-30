"""Shared application mechanics for the two release-recapture Measurements."""

from __future__ import annotations

import threading
from typing import Callable
from uuid import uuid4

from zlc_data import BlockId
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration, FinalDatasetOutput
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.release_recapture.pipeline import ReleaseRecapturePipelineSpec
from zlc_neutral_atom.logic_nodes.release_recapture.timing import TriggeredReleaseRecaptureResult, compile_triggered_release_recapture_pipeline
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId


class PreparedReleaseRecapture:
    """One immutable, one-shot flat survival pipeline."""

    __slots__ = (
        "_declaration",
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
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(spec, ReleaseRecapturePipelineSpec):
            raise TypeError("spec must be ReleaseRecapturePipelineSpec")
        if not isinstance(declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._spec = spec
        self._declaration = declaration
        self._start_run = start_run
        self._started = False
        self._lock = threading.Lock()

    def start(
        self,
        *,
        lifecycle_owner: object | None = None,
    ) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError(
                    "PreparedReleaseRecapture is one-shot"
                )
            self._started = True
        plan = compile_triggered_release_recapture_pipeline(self._spec)
        return self._start_run(
            plan.with_lifecycle(
                owner=self if lifecycle_owner is None else lifecycle_owner,
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
        )


def _final_release_recapture_output(
    result: TriggeredReleaseRecaptureResult,
    *,
    declaration: DatasetOutputDeclaration,
) -> dict[str, FinalDatasetOutput]:
    if type(result) is not TriggeredReleaseRecaptureResult:
        raise TypeError("result must be TriggeredReleaseRecaptureResult")
    snapshot = result.survival
    pipeline = result.pipeline
    output = FinalDatasetOutput(
        declaration,
        snapshot,
    )
    return {output.name: output}


def prepare_release_recapture(
    *,
    name: str,
    camera_binding,
    calibration: ResolvedCalibration,
    model_kind,
    per_site: bool,
    declaration: DatasetOutputDeclaration,
    start_run: Callable[[RunPlan], RunHandle],
    rf_port=None,
    rf_table=None,
) -> PreparedReleaseRecapture:
    selected = calibration.artifact.select_model(model_kind)
    identity = uuid4().hex
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
        start_run=start_run,
    )


__all__ = [
    "PreparedReleaseRecapture",
    "prepare_release_recapture",
]
