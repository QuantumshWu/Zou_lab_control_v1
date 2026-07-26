"""Pulse-scan time rows are physical clock ticks before a Run exists."""

from __future__ import annotations

from fractions import Fraction

import pytest

from zlc_pulse.scan_template import scan_table_template
from zlc_neutral_atom.pulse_catalog import PROBE_PULSE_PATH
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
)
from zlc_pulse import (
    ScanColumnSpec,
    api_column_specs,
    evaluate_numeric_scan_program,
    load_pulse_document,
)


def _default_api_program_parts():
    document = load_pulse_document(PROBE_PULSE_PATH)
    specs = api_column_specs(document)
    source = scan_table_template("column_stack", specs)
    rows = evaluate_numeric_scan_program(source, width=len(specs))
    columns = tuple(
        parameter.parameter_id for parameter in document.api_parameters
    )
    return document, specs, ApiSegmentTable(columns, rows)


def test_default_api_scan_rows_are_exact_native_unit_ticks():
    document, specs, table = _default_api_program_parts()

    program = ApiSlotSegmentedProgram(
        document,
        table,
        "validate the formal default before Run submission",
    )

    assert len(program.resolved_point_documents) == len(table.rows)
    for row in table.rows:
        for value, spec in zip(row, specs, strict=True):
            if spec.is_dac:
                continue
            assert spec.quantum is not None
            ratio = Fraction(str(value)) / Fraction(str(spec.quantum))
            assert ratio.denominator == 1


def test_api_scan_rejects_an_off_grid_row_when_the_program_is_built():
    document, specs, table = _default_api_program_parts()
    rows = [list(row) for row in table.rows]
    duration_index = next(
        index for index, spec in enumerate(specs) if not spec.is_dac
    )
    quantum = specs[duration_index].quantum
    assert quantum is not None
    rows[0][duration_index] = quantum * 1.5

    with pytest.raises(ValueError, match="clock grid"):
        ApiSlotSegmentedProgram(
            document,
            ApiSegmentTable(
                table.columns,
                tuple(tuple(row) for row in rows),
            ),
            "this intent must never reach Run admission",
        )


def test_duration_column_cannot_exist_without_domain_owned_quantum():
    with pytest.raises(TypeError, match="native-unit quantum"):
        ScanColumnSpec("duration", 20, 200, unit="ns")
