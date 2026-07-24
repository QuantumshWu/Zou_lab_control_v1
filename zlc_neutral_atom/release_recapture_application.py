"""One-shot application seam for the release-recapture Measurement."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import BlockId, DatasetSchema, dataset_revision_ref_to_tree
from zlc_neutral_atom.dataset_output import (
    FinalDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.readout.calibration import ResolvedCalibration
from zlc_neutral_atom.readout.calibration_reference import (
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    BoundGreyMolassesDetuning,
    BoundTemperatureReleaseRecapture,
    GreyMolassesDetuningRequest,
    GREY_MOLASSES_DETUNING_OUTPUT_NAMES,
    TemperatureReleaseRecaptureRequest,
    TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_NAMES,
    bind_grey_molasses_detuning,
    bind_temperature_release_recapture,
)
from zlc_neutral_atom.rf import BoundRfTablePort
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.readout.release_recapture_pipeline import (
    ReleaseRecapturePipelineSpec,
    release_recapture_output_schema,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.timing.release_recapture import (
    TriggeredReleaseRecaptureResult,
    TriggeredReleaseRecaptureSpec,
    compile_triggered_release_recapture_pipeline,
)
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_storage import canonical_digest


class PreparedTemperatureReleaseRecapture:
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
                    "PreparedTemperatureReleaseRecapture is one-shot"
                )
            self._started = True
        return self._start_run(
            compile_triggered_release_recapture_pipeline(self._spec)
        )


def _final_output(
    result: TriggeredReleaseRecaptureResult,
    *,
    name: str,
    owner: str,
) -> dict[str, FinalDatasetOutput]:
    if type(result) is not TriggeredReleaseRecaptureResult:
        raise TypeError("result must be TriggeredReleaseRecaptureResult")
    snapshot = result.survival
    pipeline = result.release_recapture.pipeline
    output = FinalDatasetOutput(
        name,
        snapshot,
        final_dataset_join_digest(
            owner=owner,
            output_name=name,
            source_identity={
                "run_id": pipeline.run_id,
                "dataset": dataset_revision_ref_to_tree(snapshot.ref),
                "chain": pipeline.chain_contract_digest,
            },
            snapshot=snapshot,
        ),
    )
    return {output.name: output}


def temperature_final_outputs(
    result: TriggeredReleaseRecaptureResult,
) -> dict[str, FinalDatasetOutput]:
    """Publish the Temperature Measurement's exact survival curve."""

    return _final_output(
        result,
        name=TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_NAMES[0],
        owner="temperature-release-recapture",
    )


def grey_molasses_final_outputs(
    result: TriggeredReleaseRecaptureResult,
) -> dict[str, FinalDatasetOutput]:
    """Publish the Grey-molasses Measurement's exact recapture curve."""

    return _final_output(
        result,
        name=GREY_MOLASSES_DETUNING_OUTPUT_NAMES[0],
        owner="grey-molasses-detuning",
    )


def prepare_temperature_release_recapture(
    request: TemperatureReleaseRecaptureRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedTemperatureReleaseRecapture:
    if not isinstance(request, TemperatureReleaseRecaptureRequest):
        raise TypeError("request must be TemperatureReleaseRecaptureRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    bound = bind_temperature_release_recapture(
        request,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )
    if not isinstance(bound, BoundTemperatureReleaseRecapture):
        raise RuntimeError("temperature binding returned another domain value")
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound Measurement request")
    return _prepare_release_recapture(
        name=f"Temperature release-recapture {bound.program.document.name}",
        owner="zlc_neutral_atom.release-recapture",
        program_fingerprint=bound.program.fingerprint,
        camera_binding=bound.camera_binding,
        calibration=calibration,
        model_kind=bound.request.model_kind,
        per_site=bound.request.per_site,
        start_run=start_run,
    )


def prepare_grey_molasses_detuning(
    request: GreyMolassesDetuningRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    rf_port: BoundRfTablePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedTemperatureReleaseRecapture:
    if not isinstance(request, GreyMolassesDetuningRequest):
        raise TypeError("request must be GreyMolassesDetuningRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    bound = bind_grey_molasses_detuning(
        request,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
        rf_port=rf_port,
    )
    if not isinstance(bound, BoundGreyMolassesDetuning):
        raise RuntimeError("Grey-molasses binding returned another domain value")
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound Measurement request")
    return _prepare_release_recapture(
        name=f"Grey molasses detuning {bound.program.document.name}",
        owner="zlc_neutral_atom.grey-molasses-detuning",
        program_fingerprint=bound.program.fingerprint,
        camera_binding=bound.camera_binding,
        calibration=calibration,
        model_kind=bound.request.model_kind,
        per_site=bound.request.per_site,
        start_run=start_run,
        rf_port=bound.rf_port,
        rf_table=bound.rf_table,
    )


def _prepare_release_recapture(
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
) -> PreparedTemperatureReleaseRecapture:
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
    return PreparedTemperatureReleaseRecapture(
        triggered,
        output_schema=output_schema,
        start_run=start_run,
    )


__all__ = [
    "PreparedTemperatureReleaseRecapture",
    "grey_molasses_final_outputs",
    "prepare_grey_molasses_detuning",
    "prepare_temperature_release_recapture",
    "temperature_final_outputs",
]
