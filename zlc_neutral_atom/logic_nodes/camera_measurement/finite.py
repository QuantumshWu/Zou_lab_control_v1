"""Finite exact application branch of Camera Measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import uuid

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, PointLayout, READOUT_EVENT, REPEAT
from zlc_neutral_atom.capture.artifact import (
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.dataset_output import LiveDatasetOutput, single_live_dataset_output
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode
from zlc_neutral_atom.capture.prepared import PreparedExactCapture
from zlc_neutral_atom.devices.camera.capture_port import (
    BoundCapturePort,
    configure_camera_exposure,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    DatasetPreviewSnapshot,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.capture.pipeline import (
    CapturePreviewPort,
    CapturePreviewSpec,
    MinimalPipelineSpec,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport, run_cleanup_steps
from zlc_neutral_atom.runtime.preview import notify_preview_failure
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_storage import positive_integer

from .binding import CameraCaptureBindingRequest, bind_camera_measurement
from .definition import (
    CameraMeasurementDescriptor,
    CameraMeasurementRequest,
    camera_measurement_final_outputs,
)


_CAMERA_REPEAT_AXIS_ID = AxisId("camera.repeat")
_CAMERA_READOUT_EVENT_AXIS_ID = AxisId("camera.readout_event")


class PreparedFiniteCameraMeasurement(PreparedExactCapture):
    """Passive finite form of the one public Camera Measurement."""

    def __init__(
        self,
        pipeline: MinimalPipelineSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: CameraMeasurementDescriptor,
        request: CameraMeasurementRequest,
    ) -> None:
        if not isinstance(pipeline, MinimalPipelineSpec):
            raise TypeError("pipeline must be MinimalPipelineSpec")
        if not isinstance(descriptor, CameraMeasurementDescriptor):
            raise TypeError("descriptor must be CameraMeasurementDescriptor")
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        self._request = request
        super().__init__(
            pipeline,
            repository,
            start_run,
            descriptor,
            one_shot_name="PreparedFiniteCameraMeasurement",
        )

    @property
    def descriptor(self) -> CameraMeasurementDescriptor:
        descriptor = self._descriptor
        assert isinstance(descriptor, CameraMeasurementDescriptor)
        return descriptor

    @property
    def live_preview_output_name(self) -> str | None:
        """Return the only honest single-frame Camera preview name, if any."""

        if self._request.frames_per_cycle != 1:
            return None
        return self._request.output_names[0]

    def live_dataset_outputs(
        self,
        frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
    ) -> dict[str, LiveDatasetOutput]:
        """Publish the finite preview under the request-owned Camera name."""

        if self.live_preview_output_name is None:
            raise RuntimeError(
                "a single-frame preview cannot identify a multi-frame Camera cycle"
            )
        output = single_live_dataset_output(
            self._request.output_declarations[0],
            frozen,
        )
        return {output.name: output}

    def final_dataset_outputs(self, reference: CaptureArtifactRef):
        """Materialize the request-owned Camera outputs from its FINAL ref."""

        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("Camera FINAL result must be CaptureArtifactRef")
        source = self._repository.materialize_final(reference)
        return camera_measurement_final_outputs(
            reference,
            source,
            self._request,
        )

    def start(self) -> RunHandle:
        if self._request.exposure_seconds is None:
            return super().start()
        self._claim_start()
        return self._start_run(
            _compile_exposure_configured_camera_artifact(
                self._request,
                self._pipeline.measurement.capture_port,
                self._repository,
            )
        )

    def start_with_preview(
        self,
        *,
        factory: Callable[[CapturePreviewSpec], CapturePreviewPort],
        source_ordinals: tuple[int, ...] | None = None,
    ) -> RunHandle:
        if self._request.exposure_seconds is None:
            return super().start_with_preview(
                factory=factory,
                source_ordinals=source_ordinals,
            )
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.preview_schema
        self._claim_start()
        preview_spec = CapturePreviewSpec(
            self._preview_block_id,
            self._preview_edge,
            source_ordinals,
        )
        preview = factory(preview_spec)
        plan = _compile_exposure_configured_camera_artifact(
            self._request,
            self._pipeline.measurement.capture_port,
            self._repository,
            preview=preview,
        )
        try:
            return self._start_run(plan)
        except BaseException as error:
            notify_preview_failure(preview, error)
            raise


@dataclass
class _ConfiguredFiniteCapture:
    exposure_session_id: str
    exposure_attempted: bool = False
    inner_plan: RunPlan | None = None
    inner_prepared: object | None = None


def _compile_exposure_configured_camera_artifact(
    request: CameraMeasurementRequest,
    port: BoundCapturePort,
    repository: CaptureRepository,
    *,
    preview: CapturePreviewPort | None = None,
) -> RunPlan:
    """Configure, capture, and restore one Camera request in a flat Run."""

    exposure = request.exposure_seconds
    if exposure is None:
        pipeline, _descriptor = bind_finite_camera_measurement(
            request,
            camera_port=port,
        )
        return compile_capture_artifact_pipeline(
            pipeline,
            repository,
            preview=preview,
        )
    state = _ConfiguredFiniteCapture(uuid.uuid4().hex)

    def preflight(context):
        state.exposure_attempted = True
        leased_port = configure_camera_exposure(
            context,
            port,
            state.exposure_session_id,
            exposure,
        )
        pipeline, _descriptor = bind_finite_camera_measurement(
            request,
            camera_port=leased_port,
        )
        inner = compile_capture_artifact_pipeline(
            pipeline,
            repository,
            preview=preview,
        )
        if (
            inner.resource_claims != (port.resource_claim,)
            or inner.bound_devices != (port.device,)
            or inner.interrupt_operations != port.interrupt_operations
            or not inner.requires_final_commit
        ):
            raise RuntimeError(
                "configured Camera inner plan changed its admitted authority"
            )
        state.inner_plan = inner
        prepared = inner.preflight(context)
        state.inner_prepared = prepared
        return state

    def execute(context, prepared: _ConfiguredFiniteCapture):
        if prepared is not state or state.inner_plan is None:
            raise RuntimeError("configured Camera preflight authority differs")
        return state.inner_plan.execute(context, state.inner_prepared)

    def cleanup(
        context,
        prepared: _ConfiguredFiniteCapture | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        def cleanup_capture() -> CleanupReport:
            inner = state.inner_plan
            if inner is None:
                return port.verify_idle(context)
            return inner.cleanup(context, state.inner_prepared, primary)

        steps = [cleanup_capture]
        if state.exposure_attempted:
            steps.append(
                lambda: port.cleanup(context, state.exposure_session_id)
            )
        return run_cleanup_steps(*steps)

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
        name=f"Camera {request.camera_ref.role}",
        resource_claims=(port.resource_claim,),
        bound_devices=(port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=port.interrupt_operations,
        requires_final_commit=True,
        dispose_unfinalized=dispose_unfinalized,
    )


def bind_finite_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    camera_port: BoundCapturePort,
) -> tuple[MinimalPipelineSpec, CameraMeasurementDescriptor]:
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
    point_layout = PointLayout.rect_c((events,))
    schema = DatasetSchema(
        repeat_axis,
        (event_axis,),
        point_layout,
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
    measurement = bind_camera_measurement(
        camera_port,
        CameraCaptureBindingRequest(
            request.camera_ref.role,
            repeat_axis,
            (event_axis,),
            point_layout,
            schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            tuple(facts.event_setting(index) for index in range(events)),
        ),
    )
    pipeline = MinimalPipelineSpec(
        f"Camera {request.camera_ref.role}",
        measurement,
        BlockId(f"camera-{schema.fingerprint[:20]}"),
    )
    descriptor = CameraMeasurementDescriptor(
        "Camera",
        request.camera_ref.role,
        schema,
        str(camera_port.resource_claim.key),
    )
    return pipeline, descriptor


def prepare_finite_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    camera_port: BoundCapturePort,
    repository: CaptureRepository,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedFiniteCameraMeasurement:
    """Prepare the finite branch of the one public Camera Measurement."""

    pipeline, descriptor = bind_finite_camera_measurement(
        request,
        camera_port=camera_port,
    )
    return PreparedFiniteCameraMeasurement(
        pipeline,
        repository,
        start_run,
        descriptor,
        request,
    )


__all__ = [
    "PreparedFiniteCameraMeasurement",
    "bind_finite_camera_measurement",
    "prepare_finite_camera_measurement",
]
