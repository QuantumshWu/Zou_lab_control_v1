"""Headless declaration of the Calibration Task capability."""

from __future__ import annotations

from zlc_neutral_atom.authoring import AuthoringChoice
from zlc_neutral_atom.logic_node_declaration import (
    ArtifactOutputPresentation,
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
    TaskPreviewPlot,
)
from zlc_neutral_atom.node_input import bind_no_node_inputs
from zlc_plot.kinds import PlotKind

from .outputs import (
    CALIBRATION_ARTIFACT_OUTPUT_DECLARATION,
    CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
)
from .sitemap import SITEMAP_CALIBRATION_TASK_DEFINITION
from .task import (
    CALIBRATION_LIVE_OUTPUT_DECLARATIONS,
    build_calibration_task_intent_from_authoring,
    calibration_task_authoring_schema,
    calibration_task_default_camera_role,
)


_LIVE_REFERENCE = CALIBRATION_LIVE_OUTPUT_DECLARATIONS[0]
_FINAL = {
    declaration.name: declaration
    for declaration in CALIBRATION_FINAL_OUTPUT_DECLARATIONS
}


def _calibration_camera_choices(
    context: object,
) -> tuple[DynamicChoicePresentation, ...]:
    if not isinstance(context, tuple):
        raise TypeError("Calibration dynamic choice context must be a role tuple")
    roles = tuple(context)
    default = calibration_task_default_camera_role(roles)
    return (
        DynamicChoicePresentation(
            "camera_role",
            tuple(AuthoringChoice(role, role) for role in roles),
            default,
            (
                "Calibrate readout requires an installed camera role with a "
                "site-map acquisition profile"
                if not roles
                else ""
            ),
        ),
    )

CALIBRATION_LOGIC_NODE = LogicNodeDeclaration(
    definition=SITEMAP_CALIBRATION_TASK_DEFINITION,
    description="Acquire reference/readout frames and commit a Calibration",
    authoring_schema=calibration_task_authoring_schema(),
    input_specs=(),
    outputs=(
        OutputPresentation(
            _LIVE_REFERENCE,
            "reference frame",
            "Counts",
            "exact capture frame while Calibration is running",
        ),
        OutputPresentation(
            _FINAL["site_map"],
            "site map",
            "Counts",
            "reference-average image with calibrated site geometry",
        ),
        OutputPresentation(
            _FINAL["fidelity_site"],
            "site fidelity",
            "Readout fidelity",
            "held-out balanced fidelity for each canonical site",
        ),
        OutputPresentation(
            _FINAL["fidelity_threshold"],
            "site threshold",
            "Readout threshold",
            "trained per-site threshold",
        ),
        OutputPresentation(
            _FINAL["fidelity_centers"],
            "site centres",
            "Site centre",
            "calibrated x/y centre for each canonical site",
        ),
        OutputPresentation(
            _FINAL["readout_samples"],
            "readout samples",
            "Readout signal",
            "raw per-repeat/per-context samples for each canonical site",
        ),
        OutputPresentation(
            _FINAL["aggregate_fidelity"],
            "aggregate fidelity",
            "Aggregate fidelity",
            "held-out balanced fidelity using per-site thresholds",
        ),
        OutputPresentation(
            _FINAL["global_fidelity"],
            "global fidelity",
            "Global fidelity",
            "held-out balanced fidelity using one shared threshold",
        ),
    ),
    build_request=build_calibration_task_intent_from_authoring,
    bind_request=bind_no_node_inputs,
    artifact_outputs=(
        ArtifactOutputPresentation(
            CALIBRATION_ARTIFACT_OUTPUT_DECLARATION,
            "calibration",
            "FINAL Calibration artifact",
        ),
    ),
    task_previews=(
        TaskPreviewPlot(_LIVE_REFERENCE.name, PlotKind.IMAGE),
        TaskPreviewPlot(_FINAL["site_map"].name, PlotKind.IMAGE),
    ),
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
    resolve_dynamic_choices=_calibration_camera_choices,
)


__all__ = ["CALIBRATION_LOGIC_NODE"]
