"""Headless declaration of PulseScan's ordinary capability facts."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_declaration import (
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)

from .authoring import (
    build_pulse_scan_program,
    pulse_scan_authoring_schema,
    pulse_scan_input_specs,
)
from .contracts import PULSE_SCAN_MEASUREMENT_DEFINITION
from .final_output import PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS
from .source_binding import bind_pulse_scan_request


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
    build_request=build_pulse_scan_program,
    bind_request=bind_pulse_scan_request,
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
)


__all__ = ["PULSE_SCAN_LOGIC_NODE"]
