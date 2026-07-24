"""Owner declarations for PulseScan's ordinary path and exact Dataset input."""

from __future__ import annotations

from typing import Mapping

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CAMERA_FRAME_OUTPUT_CONTRACT_ID,
)
from zlc_data import (
    AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    projected_dataset_output_contract_id,
)
from zlc_neutral_atom.input_spec import (
    DatasetInputSpec,
)
from zlc_neutral_atom.logic_nodes.occupancy.processor import (
    OCCUPANCY_EXACT_SCAN_OUTPUT_DECLARATIONS,
)
from zlc_pulse import (
    commit_scan_table,
    evaluate_numeric_scan_program,
    freeze_scan_program,
    load_pulse_document,
)
from zlc_pulse.scan_template import SWEEP_API_SLOT, SWEEP_SCAN_SLOT

from .contracts import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)


DEFAULT_PROBE_PULSE_PATH = "pulses/probe_template.json"


_PULSE_SCAN_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_PROBE_PULSE_PATH,
            required=True,
            description=(
                "Current PulseDocument whose declared scan/API slots are edited below"
            ),
        ),
    )
)

_EXACT_SOURCE_CONTRACTS = (
    CAMERA_FRAME_OUTPUT_CONTRACT_ID,
    *(
        declaration.contract_id
        for declaration in OCCUPANCY_EXACT_SCAN_OUTPUT_DECLARATIONS
    ),
)
_PULSE_SCAN_INPUT_SPECS = (
    DatasetInputSpec(
        "y_signal",
        "Exact source (y)",
        _EXACT_SOURCE_CONTRACTS
        + tuple(
            projected_dataset_output_contract_id(
                contract_id,
                AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
            )
            for contract_id in _EXACT_SOURCE_CONTRACTS
        ),
        description=(
            "Exact Camera frame or Occupancy counts/occupied Dataset, optionally "
            "through one explicit authoritative Figure Area selection"
        ),
    ),
)


def pulse_scan_authoring_schema() -> AuthoringSchema:
    return _PULSE_SCAN_AUTHORING_SCHEMA


def pulse_scan_input_specs():
    return _PULSE_SCAN_INPUT_SPECS


def build_pulse_scan_program(
    values: Mapping[str, object],
) -> AutonomousScanSlotProgram | ApiSlotSegmentedProgram:
    """Commit one owner-authored PulseScan program with the Pulse evaluator."""

    if not isinstance(values, Mapping):
        raise TypeError("PulseScan values must be a mapping")
    unknown = set(values) - {"pulse", "pulse_slots"}
    if unknown:
        raise ValueError(
            f"PulseScan values contain unknown fields: {tuple(sorted(unknown))}"
        )
    authored = pulse_scan_authoring_schema().freeze(
        {"pulse": values["pulse"]} if "pulse" in values else {}
    )
    document = load_pulse_document(authored["pulse"])
    slots = dict(values.get("pulse_slots") or {})
    sweep_kind = str(slots.get("sweep_kind") or "")
    source = str(slots.get("program") or "")
    if sweep_kind == SWEEP_SCAN_SLOT:
        table, _normalization = freeze_scan_program(document, source)
        committed = commit_scan_table(
            document,
            table,
            recipe_source=source,
        )
        api_values = dict(slots.get("api") or {})
        api_order = tuple(
            parameter.parameter_id for parameter in committed.api_parameters
        )
        expected = set(api_order)
        missing = tuple(
            parameter_id
            for parameter_id in api_order
            if parameter_id not in api_values
        )
        extra = tuple(
            parameter_id for parameter_id in api_values if parameter_id not in expected
        )
        if missing or extra:
            raise ValueError(
                "SCAN_SLOT requires one fixed value for every API parameter; "
                f"missing={missing}, extra={extra}"
            )
        return AutonomousScanSlotProgram(
            committed,
            tuple(
                (parameter_id, api_values[parameter_id])
                for parameter_id in api_order
            ),
        )
    if sweep_kind == SWEEP_API_SLOT:
        columns = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        rows = evaluate_numeric_scan_program(source, width=len(columns))
        return ApiSlotSegmentedProgram(
            document,
            ApiSegmentTable(columns, rows),
            "Explicit API-slot sweep authored in TaskConsole",
        )
    raise ValueError("choose a SCAN_SLOT or API_SLOT sweep")


__all__ = [
    "DEFAULT_PROBE_PULSE_PATH",
    "build_pulse_scan_program",
    "pulse_scan_authoring_schema",
    "pulse_scan_input_specs",
]
