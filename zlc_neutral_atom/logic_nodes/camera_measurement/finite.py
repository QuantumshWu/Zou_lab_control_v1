"""Finite exact application branch of Camera Measurement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    READOUT_EVENT,
    REPEAT,
    Valid,
    ValidityMode,
)
from zlc_data.value import expand_component_validity
from zlc_neutral_atom.capture.artifact import (
    compile_capture_artifact_pipeline,
    load_capture_artifact,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.dataset_output import LiveDatasetOutput, single_live_dataset_output
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode
from zlc_neutral_atom.devices.camera.capture_port import (
    BoundCapturePort,
    configure_camera_exposure,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    DatasetCoverage,
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
)
from zlc_neutral_atom.capture.pipeline import (
    MinimalPipelineSpec,
)
from zlc_neutral_atom.capture.binding import (
    CameraCaptureBindingRequest,
    bind_camera_capture,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport, run_cleanup_steps
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_storage import positive_integer

from .definition import (
    CameraMeasurementRequest,
    _finite_camera_event_column,
    camera_measurement_final_outputs,
    project_camera_measurement_outputs,
)


_CAMERA_REPEAT_AXIS_ID = AxisId("camera.repeat")
_CAMERA_READOUT_EVENT_AXIS_ID = AxisId("camera.readout_event")


class _FiniteCameraLiveProjection:
    """Publish only complete finite Camera cycles from the exact builder."""

    __slots__ = (
        "_complete_cells",
        "_pending",
        "_request",
        "_source_identity",
        "_source_schema",
        "_validity",
        "_values",
    )

    def __init__(
        self,
        request: CameraMeasurementRequest,
        source_schema: DatasetSchema,
    ) -> None:
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if source_schema.repeat_axis.size != request.repeat:
            raise ValueError("finite Camera source repeat axis differs from request")
        _finite_camera_event_column(source_schema, request.frames_per_cycle)
        self._request = request
        self._source_schema = source_schema
        self._source_identity = None
        self._complete_cells = 0
        self._values = np.zeros(
            source_schema.physical_shape,
            dtype=source_schema.cell_schema.dtype,
        )
        validity_contract = source_schema.cell_schema.validity_contract
        if validity_contract.mode is ValidityMode.VALUE:
            validity_shape = source_schema.physical_shape[:2]
        else:
            validity_shape = (
                *source_schema.physical_shape[:2],
                *(
                    source_schema.cell_schema.axis(axis_id).size
                    for axis_id in validity_contract.component_axis_ids
                ),
            )
        self._validity = np.zeros(validity_shape, dtype=bool)
        self._pending: list[tuple[DatasetCellAddress, object, object]] = []

    def consume(
        self,
        delta: DatasetPreviewDelta,
    ) -> bool:
        """Apply exact delta cells while exposing only whole physical cycles."""

        if not isinstance(delta, DatasetPreviewDelta):
            raise TypeError("delta must be DatasetPreviewDelta")
        if delta.ref.schema_fingerprint != self._source_schema.fingerprint:
            raise ValueError("finite Camera delta has another source schema")
        cursor = DatasetRevision(self._complete_cells + len(self._pending))
        if delta.after != cursor:
            raise RuntimeError("finite Camera delta is not contiguous")
        total = self._request.repeat * self._request.frames_per_cycle
        if delta.coverage.total_cells != total:
            raise ValueError("finite Camera delta has another frozen cardinality")
        identity = (delta.ref.block_id, delta.ref.stream_generation)
        if self._source_identity is not None and identity != self._source_identity:
            raise RuntimeError("finite Camera delta changed source identity")
        event_count = self._request.frames_per_cycle

        contract = self._source_schema.cell_schema.validity_contract
        pending = list(self._pending)
        complete_cycles = []
        for offset, cell in enumerate(delta.cells):
            ordinal = cursor.value + offset
            expected_address = DatasetCellAddress(
                ordinal // event_count,
                ordinal % event_count,
            )
            if cell.address != expected_address:
                raise RuntimeError("finite Camera delta changed frozen cycle order")
            if cell.value.schema != self._source_schema.cell_schema:
                raise ValueError("finite Camera delta changed its frame schema")
            validity = cell.value.validity
            if contract.mode is ValidityMode.VALUE:
                if not isinstance(validity, (Valid, Invalid)):
                    raise ValueError("finite Camera value validity changed contract")
                validity_mask = isinstance(validity, Valid)
            else:
                validity_mask = expand_component_validity(
                    validity,
                    self._source_schema.cell_schema,
                )
            pending.append(
                (cell.address, cell.value.values, validity_mask)
            )
            if expected_address.point_ordinal == event_count - 1:
                if len(pending) != event_count:
                    raise RuntimeError("finite Camera cycle is not atomic")
                complete_cycles.append(tuple(pending))
                pending = []

        for cycle in complete_cycles:
            for address, values, validity_mask in cycle:
                cell = (address.repeat_index, address.point_ordinal)
                self._values[cell] = values
                self._validity[cell] = validity_mask

        self._pending = pending
        self._source_identity = identity
        if complete_cycles:
            self._complete_cells += len(complete_cycles) * event_count
        return bool(complete_cycles)

    def freeze_live_outputs(self) -> dict[str, LiveDatasetOutput]:
        """Project one immutable complete-cycle front into atomic siblings."""

        if self._complete_cells < self._request.frames_per_cycle:
            raise RuntimeError("finite Camera live projection has no complete cycle")
        if self._complete_cells % self._request.frames_per_cycle:
            raise RuntimeError("finite Camera live projection exposed a partial cycle")
        identity = self._source_identity
        if identity is None:
            raise RuntimeError("finite Camera live projection has no source identity")
        block_id, generation = identity
        front_revision = DatasetRevision(self._complete_cells)
        validity_contract = self._source_schema.cell_schema.validity_contract
        validity = (
            CellValidity(self._validity)
            if validity_contract.mode is ValidityMode.VALUE
            else DatasetComponentValidity(
                validity_contract.component_axis_ids,
                self._validity,
            )
        )
        source_ref = DatasetRevisionRef(
            block_id,
            generation,
            self._source_schema.fingerprint,
            front_revision,
        )
        source = OwnedSnapshot(
            source_ref,
            DataBlock(
                block_id,
                front_revision,
                self._values,
                validity,
                self._source_schema,
            ),
        )
        projected = project_camera_measurement_outputs(source, self._request)
        outputs: dict[str, LiveDatasetOutput] = {}
        event_count = self._request.frames_per_cycle
        written_cycles = self._complete_cells // event_count
        for declaration in self._request.output_declarations:
            output_name = declaration.name
            snapshot = projected[output_name]
            frozen = DatasetPreviewSnapshot(
                snapshot,
                DatasetCoverage(
                    written_cycles,
                    self._request.repeat,
                ),
                (None,) * self._request.repeat,
            )
            output = single_live_dataset_output(declaration, frozen)
            outputs[output.name] = output
        return outputs


@dataclass
class _ConfiguredFiniteCapture:
    exposure_session_id: str
    exposure_attempted: bool = False
    inner_plan: RunPlan | None = None
    inner_prepared: object | None = None


def _compile_exposure_configured_camera_artifact(
    request: CameraMeasurementRequest,
    pipeline: MinimalPipelineSpec,
    project_root: Path,
    *,
    exact_preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Configure, capture, and restore one Camera request in a flat Run."""

    exposure = request.exposure_seconds
    if exposure is None:
        raise ValueError("configured Camera artifact requires an exposure request")
    if not isinstance(pipeline, MinimalPipelineSpec):
        raise TypeError("pipeline must be MinimalPipelineSpec")
    port = pipeline.capture.capture_port
    state = _ConfiguredFiniteCapture(uuid.uuid4().hex)

    def preflight(context):
        try:
            state.exposure_attempted = True
            leased_port = configure_camera_exposure(
                context,
                port,
                state.exposure_session_id,
                exposure,
            )
            pipeline = bind_finite_camera_measurement(
                request,
                camera_port=leased_port,
            )
            inner = compile_capture_artifact_pipeline(
                pipeline,
                project_root,
                exact_preview=exact_preview,
                settle_exact_preview=False,
            )
            if (
                inner.resource_claims != (port.resource_claim,)
                or inner.bound_devices != (port.device,)
                or inner.interrupt_operations != port.interrupt_operations
            ):
                raise RuntimeError(
                    "configured Camera inner plan changed its device binding"
                )
            state.inner_plan = inner
            prepared = inner.preflight(context)
            state.inner_prepared = prepared
            return state
        except BaseException as error:
            notify_preview_failure(exact_preview, error)
            raise

    def execute(context, prepared: _ConfiguredFiniteCapture):
        if prepared is not state or state.inner_plan is None:
            raise RuntimeError("configured Camera preflight transaction differs")
        return state.inner_plan.execute(context, state.inner_prepared)

    def cleanup(
        context,
        prepared: _ConfiguredFiniteCapture | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        def cleanup_capture() -> CleanupReport:
            inner = state.inner_plan
            if inner is None:
                try:
                    report = port.verify_idle(context)
                except BaseException as error:
                    notify_preview_failure(exact_preview, primary or error)
                    raise
                if primary is not None or report.errors:
                    notify_preview_failure(
                        exact_preview,
                        primary or report.errors[0],
                    )
                return report
            return inner.cleanup(context, state.inner_prepared, primary)

        steps = [cleanup_capture]
        if state.exposure_attempted:
            steps.append(
                lambda: port.cleanup(context, state.exposure_session_id)
            )
        report = run_cleanup_steps(*steps)
        failure = primary or (report.errors[0] if report.errors else None)
        if failure is not None:
            notify_preview_failure(exact_preview, failure)
        elif exact_preview is not None:
            try:
                exact_preview.source_terminal()
            except BaseException as error:
                notify_preview_failure(exact_preview, error)
        return report

    def finalize(context, result):
        inner = state.inner_plan
        if inner is None:
            raise RuntimeError("configured Camera lost its finalization owner")
        return inner.finalize(context, result)

    def dispose_unfinalized(result) -> None:
        inner = state.inner_plan
        if inner is None or inner.dispose_unfinalized is None:
            return
        inner.dispose_unfinalized(result)

    return RunPlan(
        name=f"Camera {request.camera_instance_id}",
        resource_claims=(port.resource_claim,),
        bound_devices=(port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=port.interrupt_operations,
        dispose_unfinalized=dispose_unfinalized,
    )


def bind_finite_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    camera_port: BoundCapturePort,
) -> MinimalPipelineSpec:
    """Bind ``repeat=K`` Camera to K×E passive hardware-triggered frames."""

    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("request must be CameraMeasurementRequest")
    repeats = positive_integer(request.repeat, "repeat")
    events = positive_integer(request.frames_per_cycle, "frames_per_cycle")
    capability = camera_port.capability
    facts = capability.camera_capability_evidence.physical_facts
    repeat_axis = AxisSpec(
        _CAMERA_REPEAT_AXIS_ID,
        "repeat",
        REPEAT,
        repeats,
        tuple(range(repeats)),
    )
    event_axis = AxisSpec(
        _CAMERA_READOUT_EVENT_AXIS_ID,
        "readout event",
        READOUT_EVENT,
        events,
        tuple(range(events)),
    )
    schema = DatasetSchema(
        repeat_axis,
        PointTable(
            events,
            (
                PointColumn(
                    event_axis.axis_id,
                    event_axis.name,
                    event_axis.role,
                    PointColumn.NUMERIC,
                    event_axis.coordinates or (),
                ),
            ),
        ),
        None,
        capability.payload_contract.value_schema,
    )
    schedule = DatasetCellSchedule.from_cells(
        schema,
        (
            DatasetCellAddress(repeat_index, event_index)
            for repeat_index in range(repeats)
            for event_index in range(events)
        ),
    )
    camera_capture = bind_camera_capture(
        camera_port,
        CameraCaptureBindingRequest(
            request.camera_instance_id,
            schema,
            schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            tuple(facts.event_setting(index) for index in range(events)),
        ),
    )
    pipeline = MinimalPipelineSpec(
        f"Camera {request.camera_instance_id}",
        camera_capture,
        BlockId(f"camera-{uuid.uuid4().hex}"),
    )
    return pipeline


def compile_finite_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    pipeline: MinimalPipelineSpec,
    project_root: Path,
    exact_preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile the finite completion policy over one bound Camera pipeline."""

    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("request must be CameraMeasurementRequest")
    if not isinstance(pipeline, MinimalPipelineSpec):
        raise TypeError("pipeline must be MinimalPipelineSpec")
    root = Path(project_root).expanduser().resolve()
    return (
        compile_capture_artifact_pipeline(
            pipeline,
            root,
            exact_preview=exact_preview,
        )
        if request.exposure_seconds is None
        else _compile_exposure_configured_camera_artifact(
            request,
            pipeline,
            root,
            exact_preview=exact_preview,
        )
    )


def finite_camera_outputs(
    project_root: Path,
    reference: CaptureArtifactRef,
    request: CameraMeasurementRequest,
):
    """Materialize FINAL siblings from the record-last Capture artifact."""

    if not isinstance(reference, CaptureArtifactRef):
        raise TypeError("Camera FINAL result must be CaptureArtifactRef")
    source = load_capture_artifact(
        Path(project_root).expanduser().resolve(),
        reference,
        materialize=True,
    ).materialize_snapshot()
    return camera_measurement_final_outputs(source, request)


__all__ = [
    "bind_finite_camera_measurement",
    "compile_finite_camera_measurement",
    "finite_camera_outputs",
]
