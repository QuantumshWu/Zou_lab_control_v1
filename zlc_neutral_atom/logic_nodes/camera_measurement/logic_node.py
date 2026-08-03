"""The sole discovered descriptor and operation for Camera Measurement."""

from __future__ import annotations

import numpy as np

from zlc_data import SPATIAL_X, SPATIAL_Y, OwnedSnapshot
from zlc_neutral_atom.logic_node import (
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
    SelectionParameterPatch,
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


def _selection_axis_bounds(axis, low: float, high: float) -> tuple[int, int] | None:
    """Convert an ordered camera coordinate interval to an integer ROI slice."""

    coordinates = np.asarray(axis.coordinates)
    if coordinates.ndim != 1 or coordinates.size != axis.size:
        return None
    if coordinates.size == 0 or np.any(np.diff(coordinates) <= 0):
        return None
    lo_value = min(float(low), float(high))
    hi_value = max(float(low), float(high))
    first = int(np.searchsorted(coordinates, lo_value, side="left"))
    stop = int(np.searchsorted(coordinates, hi_value, side="right"))
    first = max(0, min(first, int(axis.size)))
    stop = max(first, min(stop, int(axis.size)))
    if stop <= first:
        return None
    return first, stop - first


def camera_selection_parameter_patch(
    request: object,
    selection: object,
    source: object,
) -> SelectionParameterPatch | None:
    """Map a committed 2-D camera Area to an installation ROI draft.

    This function only describes a Device Manager draft.  It never touches a
    camera port, restarts acquisition, or changes the Camera Measurement
    request.  Non-camera/invalid selections simply have no patch.
    """

    if not isinstance(request, CameraMeasurementRequest):
        return None
    selector = getattr(selection, "selector", None)
    if selector is None:
        return None
    kind = getattr(getattr(selector, "kind", None), "value", None)
    if kind != "area":
        return None
    value = getattr(selector, "value", None)
    if (
        value is None
        or not all(hasattr(value, name) for name in ("x", "y"))
        or not isinstance(source, OwnedSnapshot)
    ):
        return None
    axes = tuple(source.block.schema.cell_schema.data_axes)
    x_axis = next((axis for axis in axes if axis.role == SPATIAL_X), None)
    y_axis = next((axis for axis in axes if axis.role == SPATIAL_Y), None)
    if x_axis is None or y_axis is None:
        return None
    x_bounds = _selection_axis_bounds(x_axis, value.x.low, value.x.high)
    y_bounds = _selection_axis_bounds(y_axis, value.y.low, value.y.high)
    if x_bounds is None or y_bounds is None:
        return None
    x, width = x_bounds
    y, height = y_bounds
    return SelectionParameterPatch(
        request.camera_instance_id,
        {
            "roi_x": x,
            "roi_y": y,
            "roi_width": width,
            "roi_height": height,
        },
        "Camera Area selection staged as device ROI",
    )


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
    selection_parameter_patch=camera_selection_parameter_patch,
)


__all__ = ["LOGIC_NODE"]
