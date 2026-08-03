"""The sole discovered descriptor and operation for Camera Measurement."""

from __future__ import annotations

from zlc_neutral_atom.logic_node import (
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeExecutionContext
from zlc_neutral_atom.runtime.preview import ExactDatasetPreviewSpec

from .definition import (
    CAMERA_MEASUREMENT_DEFINITION,
    CameraMeasurementRequest,
    build_camera_measurement_request,
    camera_measurement_authoring_schema,
)
from .finite import (
    _FiniteCameraLiveProjection,
    bind_finite_camera_measurement,
    compile_finite_camera_measurement,
    finite_camera_outputs,
)
from .monitor import open_live_camera_measurement


_CAMERA_FIELD = "camera_instance_id"
_CAPTURE_CAPABILITY = "camera.capture"
_MONITOR_CAPABILITY = "camera.monitor"
_ASSOCIATION_CAPABILITY = "camera.signal_association"


def _outputs_for(request: object) -> tuple[DatasetOutputSpec, ...]:
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("Camera outputs require CameraMeasurementRequest")
    return tuple(
        DatasetOutputSpec(
            declaration,
            declaration.name,
            "Counts",
            "One ordered frame from the same atomic camera cycle",
        )
        for declaration in request.output_declarations
    )


def _bind_execute(
    request: object,
    context: LogicNodeApplicationContext,
):
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("Camera request must be CameraMeasurementRequest")
    project_root = context.project_root

    if request.repeat == 0:
        monitor_port = context.device(_CAMERA_FIELD, _MONITOR_CAPABILITY)
        association = context.optional_device(
            _CAMERA_FIELD,
            _ASSOCIATION_CAPABILITY,
        )

        def execute(execution: LogicNodeExecutionContext):
            plan = open_live_camera_measurement(
                request,
                monitor_port=monitor_port,
                open_dataset=execution.open_live_dataset,
                association_authority=association,
            ).with_lifecycle(owner=execution, preemptible=True)
            return execution.start_and_wait(lambda: context.start_run(plan))

        return execute

    capture_port = context.device(_CAMERA_FIELD, _CAPTURE_CAPABILITY)
    pipeline = bind_finite_camera_measurement(
        request,
        camera_port=capture_port,
    )
    source_schema = pipeline.capture.capture_contract.dataset_schema

    def execute(execution: LogicNodeExecutionContext):
        preview = execution.open_exact_dataset(
            ExactDatasetPreviewSpec(source_schema.fingerprint),
            projection=_FiniteCameraLiveProjection(request, source_schema),
        )
        plan = compile_finite_camera_measurement(
            request,
            pipeline=pipeline,
            project_root=project_root,
            exact_preview=preview,
        ).with_lifecycle(owner=execution, preemptible=False)
        reference = execution.start_and_wait(lambda: context.start_run(plan))
        outputs = finite_camera_outputs(project_root, reference, request)
        execution.publish_final(outputs)
        return reference

    return execute


LOGIC_NODE = LogicNodeDescriptor(
    api_name="camera_measurement",
    definition=CAMERA_MEASUREMENT_DEFINITION,
    description="Acquire native Camera frames as live or finite siblings",
    authoring_schema=camera_measurement_authoring_schema(),
    input_specs=(),
    outputs=(),
    resolve_outputs=_outputs_for,
    build_request=build_camera_measurement_request,
    bind_execute=_bind_execute,
    device_requirements=(
        (_CAMERA_FIELD, (_CAPTURE_CAPABILITY, _MONITOR_CAPABILITY)),
    ),
    optional_device_capabilities=(
        (_CAMERA_FIELD, (_ASSOCIATION_CAPABILITY,)),
    ),
)


__all__ = ["LOGIC_NODE"]
