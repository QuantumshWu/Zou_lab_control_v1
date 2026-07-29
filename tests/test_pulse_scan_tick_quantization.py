"""Pulse-scan time rows are physical clock ticks before a Run exists."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from zlc_pulse.scan_template import scan_table_template
from zlc_neutral_atom.logic_nodes.pulse_scan.authoring import (
    DEFAULT_PULSE_SCAN_PULSE_PATH,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
)
from zlc_pulse import (
    ApiParameter,
    FIELD_DELAY,
    OutputDelay,
    PulseFieldRef,
    ScanColumnSpec,
    api_column_specs,
    evaluate_numeric_scan_program,
    load_pulse_document,
)


ROOT = Path(__file__).resolve().parents[1]


def _default_api_program_parts():
    document = load_pulse_document(
        ROOT / "pulses" / DEFAULT_PULSE_SCAN_PULSE_PATH
    )
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


def test_api_delay_starter_is_signed_and_clock_aligned():
    document = load_pulse_document(
        ROOT / "pulses" / DEFAULT_PULSE_SCAN_PULSE_PATH
    )
    port = next(port for port in document.target.ports if port.kind == "digital")
    field = PulseFieldRef(FIELD_DELAY, None, port.key)
    document = replace(
        document,
        delays=(OutputDelay(port.key, -40, "ns"),),
        api_parameters=(ApiParameter("camera_delay", field, "ns"),),
    )

    spec = api_column_specs(document)[0]
    assert spec.is_delay
    assert spec.lo <= -40 < 0 < spec.hi
    assert spec.quantum is not None

    source = scan_table_template("column_stack", (spec,))
    rows = evaluate_numeric_scan_program(source, width=1)
    assert rows[0][0] < 0 < rows[-1][0]
    assert any(row[0] == 0 for row in rows)
    for (value,) in rows:
        ratio = Fraction(str(value)) / Fraction(str(spec.quantum))
        assert ratio.denominator == 1
