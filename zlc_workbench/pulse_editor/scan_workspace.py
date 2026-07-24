"""Current PulseGUI scan-program workspace, independent of Qt.

The formal UI has two scan-table candidates: Python generated in the Scan tab
and a numeric array loaded beside the schedule.  This module keeps those
candidates typed and separate.  It never pads, truncates, or guesses an array
shape when the authored scan-parameter schema changes; explicit ParameterIds
are the only basis for canonical column ordering.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence

from zlc_pulse.scan_template import scan_table_template
from zlc_pulse import (
    FIELD_DAC,
    FIELD_DURATION,
    FrozenScanTable,
    PulseDocument,
    ScanNormalizationReport,
    ScanSlotSchema,
    coerce_numeric_scan_table,
    commit_scan_table,
    evaluate_numeric_scan_program,
    freeze_scan_table,
    load_pulse_document,
    parameter_value_in_unit,
    scan_column_specs,
    scan_slot_schema,
)


ScanCandidateSource = Literal["generated", "loaded"]


@dataclass(frozen=True)
class ScanWorkspaceCandidate:
    """One frozen candidate plus the schema under which it was produced."""

    table: FrozenScanTable
    schema: ScanSlotSchema
    recipe_source: str | None = None
    origin_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.table, FrozenScanTable):
            raise TypeError("scan candidate table must be FrozenScanTable")
        if not isinstance(self.schema, ScanSlotSchema):
            raise TypeError("scan candidate schema must be ScanSlotSchema")
        if self.recipe_source is not None:
            source = str(self.recipe_source)
            if not source.strip():
                raise ValueError("generated scan candidate source must be non-empty")
            object.__setattr__(self, "recipe_source", source)
        if self.origin_path is not None:
            object.__setattr__(self, "origin_path", Path(self.origin_path).resolve())


@dataclass(frozen=True)
class ScanCandidateSnapshot:
    table: FrozenScanTable
    compatible: bool
    origin_path: Path | None


@dataclass(frozen=True)
class ScanWorkspaceSnapshot:
    """Immutable state consumed by the existing Scan/Edit widgets."""

    source_text: str
    source_revision: int
    default_template: str
    source_dirty: bool
    selected_source: ScanCandidateSource
    generated: ScanCandidateSnapshot | None
    loaded: ScanCandidateSnapshot | None
    selected_table: FrozenScanTable | None
    selected_compatible: bool
    loaded_path: Path | None
    table_text: str
    slots_text: str
    busy_operation: str | None
    diagnostic: str


@dataclass(frozen=True)
class ScanCandidateResult:
    candidate: ScanWorkspaceCandidate
    normalization: ScanNormalizationReport


def default_scan_program(document: PulseDocument, kind: str = "column_stack") -> str:
    normalized = str(kind)
    if normalized not in ("column_stack", "grid"):
        raise ValueError("scan template kind must be column_stack or grid")
    columns = scan_column_specs(document)
    if not columns:
        return ""
    return scan_table_template(normalized, columns)


def execute_scan_program(document: PulseDocument, source: str) -> ScanCandidateResult:
    """Execute trusted local Python and strictly freeze its two-dimensional result."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not document.scan_parameters:
        raise ValueError("bind at least one field to a scan parameter first")
    rows = evaluate_numeric_scan_program(
        source,
        width=len(document.scan_parameters),
    )
    return _freeze_candidate(document, rows, recipe_source=str(source))


def load_scan_array(document: PulseDocument, path: str | Path) -> ScanCandidateResult:
    """Read a numeric matrix or one schema-checked current PulseDocument JSON."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not document.scan_parameters:
        raise ValueError("bind at least one field to a scan parameter first")
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    width = len(document.scan_parameters)
    if suffix == ".npy":
        import numpy as np

        value = np.load(resolved, allow_pickle=False)
        rows = coerce_numeric_scan_table(value, width=width, field=resolved.name)
    elif suffix == ".csv":
        with resolved.open("r", encoding="utf-8", newline="") as stream:
            parsed = [
                row
                for row in csv.reader(stream)
                if row and any(cell.strip() for cell in row)
            ]
        rows = _strict_text_rows(parsed, width=width, field=resolved.name)
    elif suffix == ".txt":
        parsed = [
            line.split()
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = _strict_text_rows(parsed, width=width, field=resolved.name)
    elif suffix == ".json":
        try:
            loaded_document = load_pulse_document(resolved)
        except Exception as error:
            raise ValueError(
                "JSON scan input must be a current PulseDocument; compiled or "
                "historical program artifacts are not supported"
            ) from error
        loaded_candidate = candidate_from_document(loaded_document)
        if loaded_candidate is None:
            raise ValueError(f"{resolved.name} contains no frozen scan table")
        current_schema = scan_slot_schema(document)
        if loaded_candidate.schema != current_schema:
            raise ValueError(
                "PulseDocument scan ParameterIds, fields, clock, or target differ "
                "from the current editor"
            )
        rows = loaded_candidate.table.rows
    else:
        raise ValueError("scan arrays must be .npy, .csv, .txt, or current PulseDocument .json")
    result = _freeze_candidate(document, rows)
    return ScanCandidateResult(
        replace(result.candidate, origin_path=resolved),
        result.normalization,
    )


def load_scan_program_source(path: str | Path) -> tuple[Path, str]:
    """Read the formal Scan-tab Python source, without accepting data artifacts."""

    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() not in (".py", ".txt"):
        raise ValueError("scan programs must be .py or .txt files")
    source = resolved.read_text(encoding="utf-8")
    if not source.strip():
        raise ValueError("scan program file is empty")
    compile(source, str(resolved), "exec")
    return resolved, source


def save_scan_array(table: FrozenScanTable, path: str | Path) -> Path:
    """Atomically export the exact selected frozen table."""

    if not isinstance(table, FrozenScanTable):
        raise TypeError("table must be FrozenScanTable")
    target = Path(path).expanduser().resolve()
    if not target.suffix:
        target = target.with_suffix(".npy")
    suffix = target.suffix.lower()
    if suffix not in (".npy", ".csv"):
        raise ValueError("scan array output must be .npy or .csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    array = np.asarray(table.rows, dtype=float)
    temporary_name: str | None = None
    try:
        if suffix == ".npy":
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, delete=False
            ) as stream:
                temporary_name = stream.name
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                np.savetxt(stream, array, delimiter="," if suffix == ".csv" else " ")
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target


def candidate_from_document(document: PulseDocument) -> ScanWorkspaceCandidate | None:
    """Seed the generated side from the document's authoritative frozen table."""

    if document.scan_table is None:
        return None
    ordered = tuple(parameter.parameter_id for parameter in document.scan_parameters)
    if document.scan_table.columns != ordered:
        by_column = {
            parameter_id: index
            for index, parameter_id in enumerate(document.scan_table.columns)
        }
        rows = tuple(
            tuple(row[by_column[parameter_id]] for parameter_id in ordered)
            for row in document.scan_table.rows
        )
        table = FrozenScanTable(ordered, rows)
    else:
        table = document.scan_table
    return ScanWorkspaceCandidate(
        table,
        scan_slot_schema(document),
        recipe_source=(
            None if document.scan_recipe is None else document.scan_recipe.source
        ),
    )


def commit_scan_candidate(
    document: PulseDocument,
    candidate: ScanWorkspaceCandidate,
    source: ScanCandidateSource,
) -> PulseDocument:
    """Commit one compatible candidate; never reconcile it to another schema."""

    normalized_source = _scan_source(source)
    current_schema = scan_slot_schema(document)
    if candidate.schema != current_schema:
        raise ValueError("scan candidate is stale because the parameter schema changed")
    if candidate.table.columns != current_schema.columns:
        raise ValueError("scan candidate columns are not in current ParameterId order")
    return commit_scan_table(
        document,
        candidate.table,
        recipe_source=(
            candidate.recipe_source
            if normalized_source == "generated"
            else None
        ),
    )


def candidate_snapshot(
    candidate: ScanWorkspaceCandidate | None,
    document: PulseDocument,
) -> ScanCandidateSnapshot | None:
    if candidate is None:
        return None
    return ScanCandidateSnapshot(
        candidate.table,
        candidate.schema == scan_slot_schema(document),
        candidate.origin_path,
    )


def format_scan_table(
    table: FrozenScanTable | None,
    *,
    column_names: Sequence[str] | None = None,
    stale: bool = False,
) -> str:
    """Format the existing read-only table surface without changing its geometry."""

    if table is None:
        return "(empty — Run code, or Load Array in the Delay/Scan panel)"
    if column_names is None:
        names = tuple(f"column_{index + 1}" for index in range(len(table.columns)))
    else:
        names = tuple(str(name) for name in column_names)
        if len(names) != len(table.columns) or any(not name.strip() for name in names):
            raise ValueError("scan table display names must match its non-empty columns")
    header = "   ".join(names)
    shown = [
        "   ".join(_format_number(value) for value in row)
        for row in table.rows[:40]
    ]
    footer = (
        f"\n... {len(table.rows)} points total"
        if len(table.rows) > 40
        else f"\n{len(table.rows)} point(s)"
    )
    body = header + "\n" + "\n".join(shown) + footer
    if stale:
        return (
            "STALE — scan parameters changed; Run code or reload the array.\n"
            + body
        )
    return body


def format_scan_slots(document: PulseDocument) -> str:
    """Describe the current semantic parameters for the formal Scan info panel."""

    if not document.scan_parameters and not document.api_parameters:
        return (
            "No scan or API slots bound yet. In the Edit tab, click the dot next "
            "to any duration or DAC value: 1st click = semantic scan parameter "
            "(orange; sN is only its compiler column), 2nd = API handle (aN, "
            "violet), 3rd = off. (A channel delay can be an API slot but is not "
            "scannable.)"
        )
    sections: list[str] = []
    if document.scan_parameters:
        lines = ["Columns of the scan table (one row = one scan point):"]
        display_specs = scan_column_specs(document)
        period_indices = {
            period.period_id: index for index, period in enumerate(document.periods)
        }
        for index, (parameter, display) in enumerate(
            zip(document.scan_parameters, display_specs, strict=True)
        ):
            nominal = parameter_value_in_unit(document, parameter)
            if parameter.field.kind == FIELD_DAC:
                allowed = (
                    f"signed integer {_format_number(display.lo)}.."
                    f"{_format_number(display.hi)} "
                    "(0 = 0 V)"
                )
            else:
                assert display.quantum is not None
                allowed = (
                    f"snapped to a whole {_format_number(display.quantum)} "
                    f"{display.unit} "
                    "tick (≥ 1 tick)"
                )
            field = parameter.field
            where = (
                f"Period {period_indices[field.period_id] + 1} duration"
                if field.kind == FIELD_DURATION
                else f"{field.port} (Period {period_indices[field.period_id] + 1})"
            )
            lines.append(
                f"  {display.name} (compiler s{index}): {where}  "
                f"[{parameter.unit}]  (nominal {_format_number(nominal)}) → {allowed}"
            )
        lines.extend(
            (
                "",
                "Every point is snapped automatically before it runs (durations "
                "→ whole ticks, DAC → integer codes in range), so this table is "
                "exactly what the hardware plays.",
            )
        )
        sections.append("\n".join(lines))
    if document.api_parameters:
        lines = [
            "API slots — set by name from a notebook/API or the Calibrate task:"
        ]
        period_indices = {
            period.period_id: index for index, period in enumerate(document.periods)
        }
        for parameter in document.api_parameters:
            field = parameter.field
            if field.kind == FIELD_DURATION:
                where = f"period {period_indices[field.period_id]} duration"
            elif field.kind == FIELD_DAC:
                where = f"{field.port} @ period {period_indices[field.period_id]}"
            else:
                where = f"{field.port} delay"
            lines.append(
                f"  {parameter.parameter_id}: {where}   e.g.  "
                f"pulse.{parameter.parameter_id} = <value>"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _freeze_candidate(
    document: PulseDocument,
    rows: tuple[tuple[int | float, ...], ...],
    *,
    recipe_source: str | None = None,
) -> ScanCandidateResult:
    columns = tuple(parameter.parameter_id for parameter in document.scan_parameters)
    table, report = freeze_scan_table(document, columns, rows)
    return ScanCandidateResult(
        ScanWorkspaceCandidate(
            table,
            scan_slot_schema(document),
            recipe_source=recipe_source,
        ),
        report,
    )


def _strict_text_rows(
    rows: Sequence[Sequence[str]],
    *,
    width: int,
    field: str,
) -> tuple[tuple[int | float, ...], ...]:
    if not rows:
        raise ValueError(f"{field} must contain at least one scan point")
    parsed: list[tuple[float, ...]] = []
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{field} row {index} has {len(row)} columns; current scan "
                f"parameters require {width}"
            )
        try:
            values = tuple(float(cell) for cell in row)
        except ValueError as error:
            raise TypeError(f"{field} row {index} contains non-numeric text") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{field} must contain only finite values")
        parsed.append(values)
    return tuple(parsed)


def _scan_source(value: str) -> ScanCandidateSource:
    normalized = str(value).strip().lower()
    if normalized not in ("generated", "loaded"):
        raise ValueError("scan source must be generated or loaded")
    return normalized  # type: ignore[return-value]


def _format_number(value: int | float) -> str:
    if not math.isfinite(float(value)):
        return str(value)
    return f"{float(value):.12g}".replace("e+0", "e").replace("e+", "e")


__all__ = [
    "ScanCandidateResult",
    "ScanCandidateSnapshot",
    "ScanCandidateSource",
    "ScanWorkspaceCandidate",
    "ScanWorkspaceSnapshot",
    "candidate_from_document",
    "candidate_snapshot",
    "commit_scan_candidate",
    "default_scan_program",
    "execute_scan_program",
    "format_scan_slots",
    "format_scan_table",
    "load_scan_array",
    "load_scan_program_source",
    "save_scan_array",
]
