"""Finite exact application branch of Camera Measurement."""

from __future__ import annotations

from typing import Callable

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, PointLayout, READOUT_EVENT, REPEAT
from zlc_neutral_atom.logic_nodes.camera_capture.artifact import CaptureRepository
from zlc_neutral_atom.logic_nodes.camera_capture.reference import CaptureArtifactRef
from zlc_neutral_atom.dataset_output import LiveDatasetOutput, single_live_dataset_output
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode
from zlc_neutral_atom.logic_nodes.camera_capture.prepared import PreparedExactCapture
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    DatasetPreviewSnapshot,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.logic_nodes.camera_capture.pipeline import MinimalPipelineSpec
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
