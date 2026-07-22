"""Pure current-only transformations for executing a frozen pulse scan.

These helpers derive immutable execution documents.  They do not own a Run,
touch hardware, or reinterpret the authored pulse-timeline ``RepeatRegion``.
"""

from __future__ import annotations

from dataclasses import replace

from .authoring import replace_pulse_field
from .document import FrozenScanTable, PulseDocument


def materialize_scan_sweeps(
    document: PulseDocument,
    sweep_count: int,
) -> PulseDocument:
    """Return ``sweep_count`` copies of the frozen table in sweep-major order.

    The run-level sweep count is deliberately independent of the document's
    authored ``RepeatRegion``.  That region remains the per-point pulse-timeline
    loop and is preserved byte-for-byte by this transformation.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if isinstance(sweep_count, bool) or not isinstance(sweep_count, int):
        raise TypeError("sweep_count must be an integer")
    if sweep_count < 1:
        raise ValueError("sweep_count must be positive")
    table = document.scan_table
    if table is None:
        raise ValueError("scan sweep materialization requires a frozen scan table")
    if sweep_count == 1:
        return document
    return replace(
        document,
        scan_table=FrozenScanTable(
            table.columns,
            tuple(row for _sweep in range(sweep_count) for row in table.rows),
        ),
        # The recipe generated one sweep, not this derived execution table.
        scan_recipe=None,
    )


def resolve_scan_point(document: PulseDocument, point_index: int) -> PulseDocument:
    """Bake one frozen scan row into its typed physical fields.

    Table columns are joined to declarations by stable ``ParameterId``.  Column
    position is never assumed to match declaration order.  Scan declarations
    and provenance are removed only after every field has been replaced; API
    declarations and the authored ``RepeatRegion`` remain unchanged.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if isinstance(point_index, bool) or not isinstance(point_index, int):
        raise TypeError("point_index must be an integer")
    table = document.scan_table
    if table is None:
        raise ValueError("scan point resolution requires a frozen scan table")
    if not 0 <= point_index < len(table.rows):
        raise IndexError("scan point index is outside the frozen table")

    resolved = document
    row = table.rows[point_index]
    parameters = document.scan_parameter_by_id
    for parameter_id, value in zip(table.columns, row, strict=True):
        parameter = parameters[parameter_id]
        resolved = replace_pulse_field(
            resolved,
            parameter.field,
            value,
            unit=parameter.unit,
        )
    return replace(
        resolved,
        scan_parameters=(),
        scan_table=None,
        scan_recipe=None,
    )


__all__ = ["materialize_scan_sweeps", "resolve_scan_point"]
