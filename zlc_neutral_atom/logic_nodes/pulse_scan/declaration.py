"""Headless declaration of PulseScan's ordinary capability facts."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_declaration import (
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)

from .authoring import (
    _freeze_pulse_scan_authoring,
    pulse_scan_authoring_schema,
    pulse_scan_input_specs,
)
from .contracts import PULSE_SCAN_MEASUREMENT_DEFINITION
from .final_output import PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS


PULSE_SCAN_LOGIC_NODE = LogicNodeDeclaration(
    definition=PULSE_SCAN_MEASUREMENT_DEFINITION,
    description="Acquire one exact Dataset over a Pulse program scan table",
    authoring_schema=pulse_scan_authoring_schema(),
    input_specs=pulse_scan_input_specs(),
    outputs=(
        OutputPresentation(
            PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS[0],
            "scan",
            "Signal",
            "scan result",
        ),
    ),
    build_request=_freeze_pulse_scan_authoring,
    bind_request=None,
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
)


__all__ = ["PULSE_SCAN_LOGIC_NODE"]
