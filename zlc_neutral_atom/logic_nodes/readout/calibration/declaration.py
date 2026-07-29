"""Headless declaration of the Calibration Task capability."""

from __future__ import annotations

from zlc_neutral_atom.authoring import AuthoringChoice
from zlc_neutral_atom.logic_node_declaration import (
    ArtifactOutputPresentation,
    DefaultOutputView,
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)
from zlc_neutral_atom.node_input import bind_no_node_inputs

from .projection import (
    CALIBRATION_ARTIFACT_OUTPUT_DECLARATION,
    CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
)
from .sitemap import SITEMAP_CALIBRATION_TASK_DEFINITION
from .task import (
    CALIBRATION_LIVE_OUTPUT_DECLARATIONS,
    DEFAULT_CALIBRATION_FOLDER,
    build_calibration_task_intent_from_authoring,
    calibration_task_authoring_schema,
    calibration_task_default_camera_role,
)


_LIVE_REFERENCE = CALIBRATION_LIVE_OUTPUT_DECLARATIONS[0]
_FINAL = CALIBRATION_FINAL_OUTPUT_DECLARATIONS


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
            _FINAL[0],
            "site map",
            "Counts",
            "reference-average image with calibrated site geometry",
        ),
        OutputPresentation(
            _FINAL[1],
            "site fidelity",
            "Readout fidelity",
            "held-out balanced fidelity for each canonical site",
        ),
        OutputPresentation(
            _FINAL[2],
            "site threshold",
            "Readout threshold",
            "trained per-site threshold",
        ),
        OutputPresentation(
            _FINAL[3],
            "site centres",
            "Site centre",
            "calibrated x/y centre for each canonical site",
        ),
        OutputPresentation(
            _FINAL[4],
            "aggregate fidelity",
            "Aggregate fidelity",
            "held-out balanced fidelity using per-site thresholds",
        ),
        OutputPresentation(
            _FINAL[5],
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
    default_views=(
        DefaultOutputView(_LIVE_REFERENCE.name, "2d"),
        DefaultOutputView(_FINAL[0].name, "sites"),
    ),
    path_presentations=(
        PathPresentationHint("folder", mode="dir", base_dir="output"),
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
    resolve_dynamic_choices=_calibration_camera_choices,
)


__all__ = ["CALIBRATION_LOGIC_NODE"]
