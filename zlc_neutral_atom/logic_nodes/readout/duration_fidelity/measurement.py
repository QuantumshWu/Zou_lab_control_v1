"""Readout-duration scan request and pulse-program specialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from zlc_neutral_atom.authoring import (
    MINIMUM_POSITIVE_FLOAT,
    AuthoringField,
    AuthoringSchema,
)
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.logic_nodes.readout.measurement_values import (
    duration_axis_for_document,
    linear_axis_from_range,
    numeric_axis,
    scale_authored_value,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
)
from zlc_pulse import FIELD_DURATION, TIME_UNIT_TO_NS, PulseDocument
from zlc_storage import canonical_text, positive_integer


DEFAULT_READOUT_DURATION_FIDELITY_PULSE_PATH = "probe_template.json"
DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE = (2.0, 20_000.0, 11)
DEFAULT_READOUT_DURATION_SHOTS = 60

READOUT_DURATION_FIDELITY_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.readout.duration_fidelity",
    "readout-duration-fidelity",
)
READOUT_DURATION_FIDELITY_DEFINITION = LogicNodeDefinition(
    READOUT_DURATION_FIDELITY_KEY,
    "Fidelity vs duration",
    "measurement",
)
READOUT_DURATION_FIDELITY_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "fidelity",
    "zlc_neutral_atom.readout-duration-fidelity.samples",
)

_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "sequencer_instance_id",
            "choice",
            "Sequencer",
            required=True,
            dynamic_choices=True,
        ),
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_READOUT_DURATION_FIDELITY_PULSE_PATH,
            required=True,
        ),
        AuthoringField(
            "duration",
            "axis_range",
            "Detection time",
            default=DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE,
            unit="us",
            minimum=MINIMUM_POSITIVE_FLOAT,
            required=True,
        ),
        AuthoringField(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_READOUT_DURATION_SHOTS,
            minimum=1,
            required=True,
            allow_blank=False,
        ),
    )
)


def readout_duration_fidelity_authoring_schema() -> AuthoringSchema:
    return _AUTHORING_SCHEMA


@dataclass(frozen=True, slots=True)
class ReadoutDurationFidelityRequest:
    """One source-neutral duration scan; R is shots and y keeps its full shape."""

    sequencer_instance_id: str
    pulse: str
    duration_seconds: tuple[float, ...]
    shots: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequencer_instance_id",
            canonical_text(self.sequencer_instance_id, "sequencer_instance_id"),
        )
        object.__setattr__(self, "pulse", canonical_text(self.pulse, "pulse"))
        object.__setattr__(
            self,
            "duration_seconds",
            numeric_axis(
                self.duration_seconds,
                "duration_seconds",
                positive=True,
            ),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))


def build_readout_duration_fidelity_request(
    values: Mapping[str, object],
) -> ReadoutDurationFidelityRequest:
    authored = _AUTHORING_SCHEMA.freeze(values)
    return ReadoutDurationFidelityRequest(
        sequencer_instance_id=authored["sequencer_instance_id"],
        pulse=authored["pulse"],
        duration_seconds=linear_axis_from_range(
            authored["duration"],
            "duration",
            scale=1e-6,
            positive=True,
        ),
        shots=authored["shots"],
    )


def build_readout_duration_program(
    request: ReadoutDurationFidelityRequest,
    document: PulseDocument,
) -> ApiSlotSegmentedProgram:
    """Bind only the duration axis; collection remains the shared PulseScan path."""

    if not isinstance(request, ReadoutDurationFidelityRequest):
        raise TypeError("request must be ReadoutDurationFidelityRequest")
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    durations = duration_axis_for_document(
        request.duration_seconds,
        "duration_seconds",
        document,
    )
    if document.scan_parameters or document.scan_table is not None:
        raise ValueError("duration template uses one API duration, not SCAN_SLOT")
    if document.repeat is not None:
        raise ValueError("duration template must describe one physical shot")
    if len(document.api_parameters) != 1:
        raise ValueError("duration template must declare exactly one API parameter")
    parameter = document.api_parameters[0]
    if parameter.field.kind != FIELD_DURATION:
        raise ValueError("duration API parameter must bind a period duration")
    execution_document = replace(document, scan_sweep_count=request.shots)
    scale = 1e9 / TIME_UNIT_TO_NS[parameter.unit]
    return ApiSlotSegmentedProgram(
        execution_document,
        ApiSegmentTable(
            (parameter.parameter_id,),
            tuple(
                (scale_authored_value(value, scale, "duration_seconds"),)
                for value in durations
            ),
        ),
        "The authored duration is an API-slot segmented hardware exception",
    )


__all__ = [
    "DEFAULT_READOUT_DURATION_FIDELITY_PULSE_PATH",
    "DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE",
    "DEFAULT_READOUT_DURATION_SHOTS",
    "READOUT_DURATION_FIDELITY_DEFINITION",
    "READOUT_DURATION_FIDELITY_KEY",
    "READOUT_DURATION_FIDELITY_OUTPUT_DECLARATION",
    "ReadoutDurationFidelityRequest",
    "build_readout_duration_fidelity_request",
    "build_readout_duration_program",
    "readout_duration_fidelity_authoring_schema",
]
