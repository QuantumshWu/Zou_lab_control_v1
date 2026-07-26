"""Owner declarations for PulseScan's ordinary path and exact Dataset input."""

from __future__ import annotations

from typing import Mapping

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.input_spec import (
    DatasetInputSpec,
)
from zlc_neutral_atom.pulse_catalog import PROBE_PULSE_PATH
from zlc_pulse import (
    commit_scan_table,
    evaluate_numeric_scan_program,
    freeze_scan_program,
    load_pulse_document,
)
from zlc_pulse.scan_template import SWEEP_API_SLOT, SWEEP_SCAN_SLOT
from zlc_storage.paths import resolve_under_project
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)


_PULSE_SCAN_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=PROBE_PULSE_PATH,
            required=True,
            description=(
                "Current PulseDocument whose declared scan/API slots are edited below"
            ),
        ),
    )
)

PULSE_SCAN_SOURCE_INPUT_SPEC = DatasetInputSpec(
    "y_signal",
    "Signal (y)",
    None,
    description=(
        "Any live Dataset signal published by another running Measurement, "
        "Processor, or association-preserving Area/Cross selector. Fit "
        "parameters and other snapshot-only values remain ordinary signals but "
        "are not offered here. PulseScan samples the next producer-associated "
        "value and never owns the producer or its device."
    ),
    requires_event_association=True,
)
_PULSE_SCAN_INPUT_SPECS = (PULSE_SCAN_SOURCE_INPUT_SPEC,)


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
    document = load_pulse_document(resolve_under_project(authored["pulse"]))
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
    "PULSE_SCAN_SOURCE_INPUT_SPEC",
    "build_pulse_scan_program",
    "pulse_scan_authoring_schema",
    "pulse_scan_input_specs",
]
