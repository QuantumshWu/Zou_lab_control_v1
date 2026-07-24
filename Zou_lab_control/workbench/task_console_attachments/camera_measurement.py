"""Camera Measurement presentation and lifecycle attachment."""

from __future__ import annotations

import uuid

from functools import partial

from zlc_frontend.figure import DatasetId
from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CAMERA_MEASUREMENT_DEFINITION,
    CameraMeasurementRequest,
    PreparedFiniteCameraMeasurement,
    PreparedLiveCameraMeasurement,
    build_camera_measurement_request_from_authoring,
    camera_measurement_authoring_schema,
    camera_measurement_default_role,
    camera_measurement_roles,
)
from zlc_workbench.form_projection import (
    DynamicChoiceProjection,
    PresentedChoice,
    project_authoring_form,
)
from zlc_workbench.live_slot import LiveDatasetSlot
from zlc_workbench.task_console.catalog_bridge import ConsoleNodeSpec

from ._common import no_inputs, run_attachment


def _start_finite_preview(command, node, host):
    try:
        command.preview_schema
    except ValueError:
        return command.start()

    token = uuid.uuid4().hex
    attached = False

    def factory(preview_spec):
        nonlocal attached
        slot = LiveDatasetSlot(
            preview_spec,
            dataset_id=DatasetId(f"console-capture-{token}"),
            retain_on_terminal=True,
            output_owner=command,
        )
        try:
            host.data_plane.attach(node, slot)
        except BaseException:
            slot.close()
            raise
        attached = True
        slot.set_change_listener(lambda: host.data_plane.mark_changed(node))
        return slot

    try:
        return command.start_with_preview(factory=factory)
    except BaseException:
        if attached:
            host.data_plane.detach_live(node)
        raise


def _start_camera(command, node, host):
    if isinstance(command, PreparedLiveCameraMeasurement):
        dataset_id = DatasetId(
            f"console-{node.spec.key.stable_definition_id}-{id(node):x}"
        )
        attached = False

        def live_factory(view_spec):
            nonlocal attached
            slot = LiveDatasetSlot(
                view_spec,
                dataset_id=dataset_id,
                retain_on_terminal=True,
                output_owner=command,
            )
            try:
                host.data_plane.attach(node, slot)
            except BaseException:
                slot.close()
                raise
            attached = True
            slot.set_change_listener(
                lambda: host.data_plane.mark_changed(node)
            )
            return slot

        try:
            return command.start_with_view(factory=live_factory)
        except BaseException:
            if attached:
                host.data_plane.detach_live(node)
            raise

    if not isinstance(command, PreparedFiniteCameraMeasurement):
        raise TypeError("Camera preparer returned another command type")
    if command.live_preview_output_name is None:
        return command.start()
    return _start_finite_preview(command, node, host)


def camera_measurement_attachment(
    *,
    installed_camera_roles: tuple[str, ...],
    request_builder,
    prepare,
):
    """Bind the Camera owner; frame_i vocabulary stays request-owned."""

    roles = camera_measurement_roles(tuple(installed_camera_roles))

    def request_outputs(request):
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("Camera output owner received another request type")
        return request.output_declarations

    spec = ConsoleNodeSpec(
        definition=CAMERA_MEASUREMENT_DEFINITION,
        title="Camera",
        description="Acquire camera frames as a live or finite Measurement",
        form=project_authoring_form(
            camera_measurement_authoring_schema(),
            dynamic_choices={
                "camera_role": DynamicChoiceProjection(
                    choices=tuple(PresentedChoice(role, role) for role in roles),
                    default=camera_measurement_default_role(roles),
                    unavailable_reason=(
                        "Camera Measurement requires an installed camera role"
                        if not roles
                        else ""
                    ),
                )
            },
        ),
        declared_outputs=(),
        build_request=partial(
            build_camera_measurement_request_from_authoring,
            request_builder,
        ),
        request_output_declarations=request_outputs,
        request_output_axis_label="Counts",
        request_output_description=(
            "ordered camera readout event; repeat, point, and trailing data "
            "axes are preserved"
        ),
    )

    def prepare_camera(request):
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("Camera owner returned another request type")
        return prepare(request)

    return run_attachment(
        spec,
        bind_request=no_inputs,
        prepare=prepare_camera,
        start=_start_camera,
    )


__all__ = ["camera_measurement_attachment"]
