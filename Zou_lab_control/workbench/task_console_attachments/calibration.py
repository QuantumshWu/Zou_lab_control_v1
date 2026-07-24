"""Calibration Task presentation and lifecycle attachment."""

from __future__ import annotations

from zlc_frontend.figure import DatasetId
from zlc_neutral_atom.logic_nodes.calibration.projection import (
    CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
)
from zlc_neutral_atom.logic_nodes.calibration.sitemap import (
    SITEMAP_CALIBRATION_TASK_DEFINITION,
)
from zlc_neutral_atom.logic_nodes.calibration.task import (
    CALIBRATION_LIVE_OUTPUT_DECLARATIONS,
    DEFAULT_CALIBRATION_FOLDER,
    CalibrationTaskIntent,
    PreparedCalibrationTask,
    build_calibration_task_intent_from_authoring,
    calibration_task_authoring_schema,
    calibration_task_default_camera_role,
)
from zlc_workbench.form_projection import (
    DynamicChoiceProjection,
    PathPresentation,
    PresentedChoice,
    project_authoring_form,
)
from zlc_workbench.live_slot import LiveDatasetSlot
from zlc_workbench.logic_node_presentations.calibration import (
    materialize_calibration_final_presentations,
)
from zlc_workbench.task_console.catalog_bridge import (
    ConsoleDefaultPanel,
    ConsoleNodeSpec,
    ConsoleSignalDecl,
)

from ._common import no_inputs, run_attachment


class _CalibrationPreviewHost:
    """Host the Task-owned preview slot without interpreting its data."""

    __slots__ = ("_data_plane", "_node", "_slot")

    def __init__(self, node, data_plane) -> None:
        self._node = node
        self._data_plane = data_plane
        self._slot = None

    def open_calibration_preview(self, spec, *, output_owner):
        if self._slot is not None:
            raise RuntimeError("Calibration Task already owns a live preview")
        slot = LiveDatasetSlot(
            spec,
            dataset_id=DatasetId(f"console-calibration-{id(self._node):x}"),
            retain_on_terminal=True,
            output_owner=output_owner,
        )
        try:
            self._data_plane.attach(self._node, slot)
            slot.set_change_listener(
                lambda: self._data_plane.mark_changed(self._node)
            )
        except BaseException:
            slot.close()
            raise
        self._slot = slot
        return slot


def calibration_attachment(
    *,
    sitemap_camera_roles: tuple[str, ...],
    prepare,
):
    """Bind Calibration's owner declarations and application command."""

    roles = tuple(sitemap_camera_roles)
    spec = ConsoleNodeSpec(
        definition=SITEMAP_CALIBRATION_TASK_DEFINITION,
        title="Calibrate readout",
        description="Acquire reference/readout frames and commit a Calibration",
        form=project_authoring_form(
            calibration_task_authoring_schema(),
            dynamic_choices={
                "camera_role": DynamicChoiceProjection(
                    tuple(PresentedChoice(role, role) for role in roles),
                    calibration_task_default_camera_role(roles),
                    (
                        "Calibrate readout requires an installed camera role "
                        "with a site-map acquisition profile"
                        if not roles
                        else ""
                    ),
                )
            },
            path_presentations={
                "folder": PathPresentation(
                    mode="dir",
                    base_dir=DEFAULT_CALIBRATION_FOLDER,
                ),
                "pulse": PathPresentation(
                    mode="file",
                    file_filter="Pulse program (*.json);;All files (*)",
                    base_dir="zlc_neutral_atom/assets",
                ),
            },
        ),
        declared_outputs=(
            ConsoleSignalDecl(
                CALIBRATION_LIVE_OUTPUT_DECLARATIONS[0],
                "reference frame",
                "Counts",
                "exact capture frame while Calibration is running",
            ),
            ConsoleSignalDecl(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0],
                "calibration",
                "Calibration",
                "FINAL Calibration artifact",
            ),
            ConsoleSignalDecl(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[1],
                "site fidelity",
                "Readout fidelity",
                "held-out balanced fidelity for each canonical site",
            ),
            ConsoleSignalDecl(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[2],
                "site threshold",
                "Readout threshold",
                "trained per-site threshold",
            ),
            ConsoleSignalDecl(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[3],
                "site centres",
                "Site centre",
                "calibrated x/y centre for each canonical site",
            ),
            ConsoleSignalDecl(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[4],
                "aggregate fidelity",
                "Aggregate fidelity",
                "held-out balanced fidelity using per-site thresholds",
            ),
            ConsoleSignalDecl(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[5],
                "global fidelity",
                "Global fidelity",
                "held-out balanced fidelity using one shared threshold",
            ),
        ),
        build_request=build_calibration_task_intent_from_authoring,
        default_panels=(
            ConsoleDefaultPanel(
                CALIBRATION_LIVE_OUTPUT_DECLARATIONS[0].name,
                "2d",
            ),
            ConsoleDefaultPanel(
                CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0].name,
                "sites",
            ),
        ),
    )

    def prepare_task(request):
        if not isinstance(request, CalibrationTaskIntent):
            raise TypeError("Calibration owner returned another request type")
        return prepare(request)

    def start_task(command, node, host):
        if not isinstance(command, PreparedCalibrationTask):
            raise TypeError("Calibration preparer returned another command type")
        if not command.has_live_output:
            return command.start()
        return command.start(_CalibrationPreviewHost(node, host.data_plane))

    return run_attachment(
        spec,
        bind_request=no_inputs,
        prepare=prepare_task,
        start=start_task,
        materialize_final_presentations=materialize_calibration_final_presentations,
    )


__all__ = ["calibration_attachment"]
