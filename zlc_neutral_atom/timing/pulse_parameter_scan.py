"""Frozen pulse-parameter scan programs shared by neutral-atom experiments.

This module owns the bridge between a :class:`zlc_pulse.PulseDocument` and the
named Dataset point coordinates that execute it.  It deliberately contains no
signal association, collector, artifact, or logic-node semantics: PulseScan,
temperature release/recapture, and readout-duration fidelity consume the same
frozen timing vocabulary without depending on one another's leaf packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Any, Mapping

from zlc_data import SCAN_POINT, AxisId, PointColumn, PointTable
from zlc_pulse import (
    PulseDocument,
    pulse_document_from_tree,
    pulse_document_to_tree,
    resolve_api_parameters,
    resolve_api_segment_document,
)
from zlc_storage import canonical_text, exact_mapping


PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA = (
    "zlc_neutral_atom.PulseParameterScanProgram"
)

_AUTONOMOUS_SCAN_SLOT_KIND = "AUTONOMOUS_SCAN_SLOT"
_API_SLOT_SEGMENTED_KIND = "API_SLOT_SEGMENTED_EXISTING"


def _scan_axis_id(parameter_id: str) -> AxisId:
    return AxisId(f"scan.parameter.{parameter_id}")


def _scan_sweep_count(
    document: PulseDocument,
    *,
    execution_name: str,
) -> int:
    count = document.scan_sweep_count
    if count < 1:
        raise ValueError(f"{execution_name} scan_sweep_count must be positive")
    return count


@dataclass(frozen=True)
class ApiSegmentTable:
    """Ordered, complete API rows; values are never rounded or inferred."""

    columns: tuple[str, ...]
    rows: tuple[tuple[int | float, ...], ...]

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("API segment table columns must be non-empty")
        for column in columns:
            canonical_text(column, "API segment table column")
        if len(set(columns)) != len(columns):
            raise ValueError("API segment table columns must be unique")
        rows = tuple(tuple(row) for row in self.rows)
        if not rows:
            raise ValueError("API segment table must contain at least one row")
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("every API segment row must match its named columns")
        for row in rows:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError("API segment values must be numeric")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("API segment values must be finite")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)


def _point_table_from_rows(
    *,
    columns: tuple[str, ...],
    rows: tuple[tuple[int | float, ...], ...],
    labels: dict[str, str],
    units: dict[str, str],
) -> PointTable:
    """Freeze authored pulse rows directly; correlated columns never Cartesianize."""

    if not rows or any(len(row) != len(columns) for row in rows):
        raise ValueError("scan rows must be a non-empty rectangular table")
    return PointTable(
        len(rows),
        tuple(
            PointColumn(
                _scan_axis_id(parameter_id),
                labels[parameter_id],
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(row[position] for row in rows),
                units[parameter_id],
            )
            for position, parameter_id in enumerate(columns)
        ),
    )


def _point_table_from_pulse_document(document: PulseDocument) -> PointTable:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    table = document.scan_table
    if table is None or not document.scan_parameters:
        raise ValueError("a scan point table requires declared parameters and rows")
    parameters = document.scan_parameter_by_id
    return _point_table_from_rows(
        columns=table.columns,
        rows=table.rows,
        labels={
            parameter_id: parameters[parameter_id].label or parameter_id
            for parameter_id in table.columns
        },
        units={
            parameter_id: parameters[parameter_id].unit
            for parameter_id in table.columns
        },
    )


def _point_table_from_api_segments(
    document: PulseDocument,
    table: ApiSegmentTable,
) -> PointTable:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(table, ApiSegmentTable):
        raise TypeError("table must be ApiSegmentTable")
    expected = tuple(parameter.parameter_id for parameter in document.api_parameters)
    if table.columns != expected:
        raise ValueError("API segment columns must match declared parameter order")
    parameters = document.api_parameter_by_id
    return _point_table_from_rows(
        columns=table.columns,
        rows=table.rows,
        labels={parameter_id: parameter_id for parameter_id in table.columns},
        units={
            parameter_id: parameters[parameter_id].unit
            for parameter_id in table.columns
        },
    )


@dataclass(frozen=True)
class AutonomousScanSlotProgram:
    """Editable SCAN_SLOT document plus frozen whole-run API constants."""

    document: PulseDocument
    api_values: tuple[tuple[str, int | float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        _point_table_from_pulse_document(self.document)
        _scan_sweep_count(
            self.document,
            execution_name="autonomous SCAN_SLOT",
        )
        api_values = tuple(self.api_values)
        if any(not isinstance(item, tuple) or len(item) != 2 for item in api_values):
            raise TypeError("api_values must contain (parameter_id, value) tuples")
        for parameter_id, _value in api_values:
            canonical_text(parameter_id, "API value parameter_id")
        expected = tuple(
            parameter.parameter_id for parameter in self.document.api_parameters
        )
        actual = tuple(parameter_id for parameter_id, _value in api_values)
        if actual != expected:
            raise ValueError(
                "autonomous SCAN_SLOT API values must exactly cover declared "
                "parameters in declaration order"
            )
        execution_document = resolve_api_parameters(
            self.document,
            dict(api_values),
        )
        if execution_document.api_parameters:
            raise RuntimeError("autonomous SCAN_SLOT retained unresolved APIs")
        object.__setattr__(self, "api_values", api_values)

    @classmethod
    def from_api_values(
        cls,
        document: PulseDocument,
        values: Mapping[str, int | float] | None = None,
    ) -> "AutonomousScanSlotProgram":
        """Freeze the document's complete whole-run API assignment."""

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        supplied = {} if values is None else dict(values)
        expected = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        if set(supplied) != set(expected):
            missing = tuple(key for key in expected if key not in supplied)
            extra = tuple(key for key in supplied if key not in set(expected))
            raise ValueError(
                "SCAN_SLOT requires explicit whole-run values for every API "
                f"parameter; missing={missing}, extra={extra}"
            )
        return cls(
            document,
            tuple((key, supplied[key]) for key in expected),
        )

    @property
    def execution_document(self) -> PulseDocument:
        return resolve_api_parameters(self.document, dict(self.api_values))

    @property
    def point_table(self) -> PointTable:
        return _point_table_from_pulse_document(self.document)

    @property
    def point_count(self) -> int:
        return self.point_table.row_count

    @property
    def sweep_count(self) -> int:
        return _scan_sweep_count(
            self.document,
            execution_name="autonomous SCAN_SLOT",
        )

@dataclass(frozen=True)
class ApiSlotSegmentedProgram:
    """An explicitly segmented API_SLOT sweep over finite static pulses."""

    document: PulseDocument
    table: ApiSegmentTable
    segmentation_rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(self.table, ApiSegmentTable):
            raise TypeError("table must be ApiSegmentTable")
        object.__setattr__(
            self,
            "segmentation_rationale",
            canonical_text(
                self.segmentation_rationale,
                "segmentation_rationale",
            ),
        )
        expected = tuple(
            parameter.parameter_id for parameter in self.document.api_parameters
        )
        if not expected:
            raise ValueError("API_SLOT segmented program requires API parameters")
        if self.table.columns != expected:
            raise ValueError(
                "API segment columns must exactly match declared API parameter order"
            )
        _scan_sweep_count(
            self.document,
            execution_name="API_SLOT segmented",
        )
        for row in self.table.rows:
            resolve_api_segment_document(
                self.document,
                dict(zip(self.table.columns, row)),
            )

    @cached_property
    def point_table(self) -> PointTable:
        return _point_table_from_api_segments(self.document, self.table)

    @property
    def point_count(self) -> int:
        return len(self.table.rows)

    @property
    def sweep_count(self) -> int:
        return _scan_sweep_count(
            self.document,
            execution_name="API_SLOT segmented",
        )

    @property
    def segment_count(self) -> int:
        return self.sweep_count * self.point_count

    @cached_property
    def resolved_point_documents(self) -> tuple[PulseDocument, ...]:
        """Return P single-fire documents resolved once in row order."""

        documents = tuple(
            resolve_api_segment_document(
                self.document,
                dict(zip(self.table.columns, row)),
            )
            for row in self.table.rows
        )
        if len(documents) != self.point_count:
            raise RuntimeError("API segment resolution changed the point count")
        return documents

PulseParameterScanProgram = AutonomousScanSlotProgram | ApiSlotSegmentedProgram


def _api_segment_table_to_tree(value: ApiSegmentTable) -> dict[str, Any]:
    if not isinstance(value, ApiSegmentTable):
        raise TypeError("value must be ApiSegmentTable")
    return {
        "columns": list(value.columns),
        "rows": [list(row) for row in value.rows],
    }


def _api_segment_table_from_tree(tree: Any) -> ApiSegmentTable:
    data = exact_mapping(
        tree,
        {"columns", "rows"},
        "ApiSegmentTable",
        discriminator=None,
    )
    if not isinstance(data["columns"], list) or not isinstance(data["rows"], list):
        raise TypeError("ApiSegmentTable columns and rows must be lists")
    if any(not isinstance(row, list) for row in data["rows"]):
        raise TypeError("ApiSegmentTable rows must contain lists")
    return ApiSegmentTable(
        tuple(data["columns"]),
        tuple(tuple(row) for row in data["rows"]),
    )


def pulse_parameter_scan_program_to_tree(
    value: PulseParameterScanProgram,
) -> dict[str, Any]:
    """Encode the sole current discriminated pulse-parameter program schema."""

    if isinstance(value, AutonomousScanSlotProgram):
        return {
            "schema": PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA,
            "kind": _AUTONOMOUS_SCAN_SLOT_KIND,
            "document": pulse_document_to_tree(value.document),
            "api_values": [
                [parameter_id, parameter_value]
                for parameter_id, parameter_value in value.api_values
            ],
        }
    if isinstance(value, ApiSlotSegmentedProgram):
        return {
            "schema": PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA,
            "kind": _API_SLOT_SEGMENTED_KIND,
            "document": pulse_document_to_tree(value.document),
            "table": _api_segment_table_to_tree(value.table),
            "segmentation_rationale": value.segmentation_rationale,
        }
    raise TypeError("value must be a PulseParameterScanProgram")


def pulse_parameter_scan_program_from_tree(tree: Any) -> PulseParameterScanProgram:
    """Decode only current pulse-parameter program variants and field sets."""

    if not isinstance(tree, dict):
        raise TypeError("PulseParameterScanProgram must be a mapping")
    if tree.get("schema") != PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA:
        raise ValueError("PulseParameterScanProgram schema differs")
    kind = tree.get("kind")
    if kind == _AUTONOMOUS_SCAN_SLOT_KIND:
        data = exact_mapping(
            tree,
            {"schema", "kind", "document", "api_values"},
            PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA,
        )
        raw_api_values = data["api_values"]
        if not isinstance(raw_api_values, list) or any(
            not isinstance(item, list) or len(item) != 2
            for item in raw_api_values
        ):
            raise TypeError("PulseParameterScanProgram api_values must be pair lists")
        value: PulseParameterScanProgram = AutonomousScanSlotProgram(
            pulse_document_from_tree(data["document"]),
            tuple((item[0], item[1]) for item in raw_api_values),
        )
    elif kind == _API_SLOT_SEGMENTED_KIND:
        data = exact_mapping(
            tree,
            {
                "schema",
                "kind",
                "document",
                "table",
                "segmentation_rationale",
            },
            PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA,
        )
        value = ApiSlotSegmentedProgram(
            pulse_document_from_tree(data["document"]),
            _api_segment_table_from_tree(data["table"]),
            data["segmentation_rationale"],
        )
    else:
        raise ValueError("PulseParameterScanProgram kind differs")
    if pulse_parameter_scan_program_to_tree(value) != tree:
        raise ValueError("PulseParameterScanProgram tree is typed but non-canonical")
    return value


__all__ = [
    "ApiSegmentTable",
    "ApiSlotSegmentedProgram",
    "AutonomousScanSlotProgram",
    "PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA",
    "PulseParameterScanProgram",
    "pulse_parameter_scan_program_from_tree",
    "pulse_parameter_scan_program_to_tree",
]
