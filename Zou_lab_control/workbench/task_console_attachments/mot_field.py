"""MOT-field Task presentation and lifecycle attachment."""

from __future__ import annotations

from zlc_neutral_atom.logic_nodes.mot_field import (
    MOT_FIELD_FINAL_OUTPUT_DECLARATIONS,
    MOT_FIELD_LIVE_OUTPUT_DECLARATIONS,
    MOT_FIELD_TASK_DEFINITION,
    MotFieldTaskIntent,
    PreparedMotFieldTask,
    build_mot_field_intent_from_authoring,
    mot_field_authoring_schema,
    mot_field_camera_roles,
)
from zlc_workbench.form_projection import (
    DynamicChoiceProjection,
    PathPresentation,
    PresentedChoice,
    project_authoring_form,
)
from zlc_workbench.task_console.catalog_bridge import (
    ConsoleDefaultPanel,
    ConsoleNodeSpec,
    ConsoleSignalDecl,
)

from ._common import no_inputs, run_attachment


def mot_field_attachment(
    *,
    installed_camera_roles: tuple[str, ...],
    prepare,
):
    """Bind the MOT owner to the generic console host."""

    roles = mot_field_camera_roles(tuple(installed_camera_roles))
    spec = ConsoleNodeSpec(
        definition=MOT_FIELD_TASK_DEFINITION,
        title="Optimize MOT field",
        description=(
            "Sweep da_x/da_y/da_z in one autonomous hardware scan, measure "
            "MOT fluorescence, and report the refined optimum"
        ),
        form=project_authoring_form(
            mot_field_authoring_schema(),
            dynamic_choices={
                "camera_role": DynamicChoiceProjection(
                    tuple(PresentedChoice(role, role) for role in roles),
                    roles[0] if roles else None,
                    (
                        "MOT field requires the installation's external-trigger-"
                        "capable mot_camera role"
                        if not roles
                        else ""
                    ),
                )
            },
            path_presentations={
                "pulse": PathPresentation(
                    mode="file",
                    file_filter="Pulse program (*.json);;All files (*)",
                    base_dir="pulses",
                ),
                "folder": PathPresentation(mode="dir"),
            },
        ),
        declared_outputs=(
            ConsoleSignalDecl(
                MOT_FIELD_LIVE_OUTPUT_DECLARATIONS[0],
                "MOT intensity grid",
                "Counts",
                "provisional Bx/By/Bz intensity while the scan runs",
            ),
            ConsoleSignalDecl(
                MOT_FIELD_FINAL_OUTPUT_DECLARATIONS[0],
                "MOT field",
                "Counts",
                "FINAL optimum + 3-D intensity",
            ),
            ConsoleSignalDecl(
                MOT_FIELD_FINAL_OUTPUT_DECLARATIONS[1],
                "scan",
                "Signal",
                "exact source scan artifact",
            ),
        ),
        build_request=build_mot_field_intent_from_authoring,
        default_panels=(
            ConsoleDefaultPanel(
                MOT_FIELD_LIVE_OUTPUT_DECLARATIONS[0].name,
                "grid",
            ),
            ConsoleDefaultPanel(
                MOT_FIELD_FINAL_OUTPUT_DECLARATIONS[0].name,
                "grid",
            ),
        ),
    )

    def prepare_task(request):
        if not isinstance(request, MotFieldTaskIntent):
            raise TypeError("MOT owner returned another request type")
        return prepare(request)

    def start_task(command, node, host):
        if not isinstance(command, PreparedMotFieldTask):
            raise TypeError("MOT preparer returned another command type")
        live_output = command.live_output
        attached = False
        try:
            host.data_plane.attach(node, live_output)
            attached = True
            live_output.set_change_listener(
                lambda: host.data_plane.mark_changed(node)
            )
            return command.start()
        except BaseException:
            if attached:
                host.data_plane.detach_live(node)
            else:
                live_output.close()
            raise

    return run_attachment(
        spec,
        bind_request=no_inputs,
        prepare=prepare_task,
        start=start_task,
    )


__all__ = ["mot_field_attachment"]
