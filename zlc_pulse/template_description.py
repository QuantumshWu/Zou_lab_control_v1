"""Typed authoring description of one saved :class:`PulseDocument`.

The declaration order, bound field targets, native units, clock/range-derived
column specifications, and stored scan program are pulse-domain facts.  A form
may render this immutable description, but it must not rediscover those facts
from a document or a raw lane layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .document import PulseDocument, load_pulse_document, pulse_document_path
from .scan_columns import (
    ScanColumnSpec,
    api_column_specs,
    parameter_value_in_unit,
    scan_column_specs,
)


@dataclass(frozen=True, slots=True)
class PulseTemplateDescription:
    """Complete immutable input required by the shared Pulse-slots form."""

    api_rows: tuple[tuple[str, str, str, str, str, int | float], ...]
    scan_rows: tuple[tuple[str, str, str, str, str], ...]
    api_columns: tuple[ScanColumnSpec, ...]
    scan_columns: tuple[ScanColumnSpec, ...]
    program: str
    program_id: str


def _slot_target(document: PulseDocument, field) -> str:
    period_indices = {
        period.period_id: index for index, period in enumerate(document.periods)
    }
    if field.kind == "duration":
        return str(period_indices[field.period_id])
    if field.kind == "dac":
        return f"{field.port}@{period_indices[field.period_id]}"
    return str(field.port)


def describe_pulse_template(path: str | Path) -> PulseTemplateDescription:
    """Load one explicit template and return its pulse-owned form description."""

    text = str(path).strip()
    if not text:
        raise ValueError("pulse template path must be non-empty")
    source = pulse_document_path(text)
    document = load_pulse_document(source)
    api_rows = tuple(
        (
            parameter.parameter_id,
            parameter.parameter_id,
            parameter.field.kind,
            _slot_target(document, parameter.field),
            parameter.unit,
            parameter_value_in_unit(document, parameter),
        )
        for parameter in document.api_parameters
    )
    scan_rows = tuple(
        (
            parameter.parameter_id,
            parameter.field.kind,
            _slot_target(document, parameter.field),
            parameter.unit,
            parameter.label,
        )
        for parameter in document.scan_parameters
    )
    program = (
        ""
        if document.scan_recipe is None
        else str(document.scan_recipe.source)
    )
    if not program.strip() and document.scan_table is not None:
        program = (
            "scan_table = np.array("
            + repr([list(row) for row in document.scan_table.rows])
            + ", dtype=float)"
        )
    return PulseTemplateDescription(
        api_rows,
        scan_rows,
        api_column_specs(document),
        scan_column_specs(document),
        program,
        document.fingerprint,
    )


__all__ = ["PulseTemplateDescription", "describe_pulse_template"]
