"""Trusted-local Pulse scan-program evaluation and authoritative commit.

The Python editor is presentation, but the meaning of its ``scan_table``
result is pulse authoring: exact column order, clock/range normalization and
recipe provenance.  Keeping that transaction here gives PulseGUI,
TaskConsole and notebook code one implementation without one GUI importing
another GUI's workspace.
"""

from __future__ import annotations

from dataclasses import replace
import math

from zlc_storage import canonical_text, positive_integer

from .authoring import (
    ScanNormalizationReport,
    attach_scan_recipe,
    freeze_scan_table,
)
from .document import FrozenScanTable, PulseDocument

__all__ = [
    "coerce_numeric_scan_table",
    "commit_scan_table",
    "evaluate_numeric_scan_program",
    "freeze_scan_program",
]


def coerce_numeric_scan_table(
    value: object,
    *,
    width: int,
    field: str,
) -> tuple[tuple[int | float, ...], ...]:
    width = positive_integer(width, "scan table width")
    field = canonical_text(field, "scan table field")
    import numpy as np

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be one rectangular numeric matrix") from error
    if array.ndim == 1 and width == 1:
        array = array.reshape((-1, 1))
    elif array.ndim != 2:
        raise ValueError(f"{field} must be exactly two-dimensional")
    if array.shape[0] < 1:
        raise ValueError(f"{field} must contain at least one scan point")
    if array.shape[1] != width:
        raise ValueError(
            f"{field} has {array.shape[1]} columns; current parameters require {width}"
        )
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain only real numeric values")
    numeric = array.astype(float, copy=False)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError(f"{field} must contain only finite values")
    return tuple(tuple(row) for row in numeric.tolist())


def evaluate_numeric_scan_program(
    source: str,
    *,
    width: int,
) -> tuple[tuple[int | float, ...], ...]:
    """Evaluate the formal trusted-local ``scan_table`` Python contract."""

    width = positive_integer(width, "scan program width")
    text = str(source)
    if not text.strip():
        raise ValueError("scan program source must be non-empty")
    import numpy as np

    namespace = {
        "__name__": "__zlc_scan_program__",
        "np": np,
        "numpy": np,
        "math": math,
        "n_slots": width,
    }
    # This is an explicit trusted-local experiment-program surface, not a
    # security sandbox.  Normal Python/import semantics are intentional.
    exec(compile(text, "<Pulse scan program>", "exec"), namespace, namespace)  # noqa: S102
    if "scan_table" not in namespace:
        raise ValueError(
            "assign an N_points x N_parameters array to 'scan_table'"
        )
    return coerce_numeric_scan_table(
        namespace["scan_table"],
        width=width,
        field="scan_table",
    )


def freeze_scan_program(
    document: PulseDocument,
    source: str,
) -> tuple[FrozenScanTable, ScanNormalizationReport]:
    """Evaluate and normalize one program against the current PulseDocument."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    columns = tuple(
        parameter.parameter_id for parameter in document.scan_parameters
    )
    if not columns:
        raise ValueError("bind at least one field to a scan parameter first")
    rows = evaluate_numeric_scan_program(source, width=len(columns))
    return freeze_scan_table(document, columns, rows)


def commit_scan_table(
    document: PulseDocument,
    table: FrozenScanTable,
    *,
    recipe_source: str | None = None,
) -> PulseDocument:
    """Commit a current-schema table and optional reproducible source recipe."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(table, FrozenScanTable):
        raise TypeError("table must be FrozenScanTable")
    columns = tuple(
        parameter.parameter_id for parameter in document.scan_parameters
    )
    if table.columns != columns:
        raise ValueError(
            "scan table columns are not in the current ParameterId order"
        )
    verified, _report = freeze_scan_table(document, columns, table.rows)
    if verified.fingerprint != table.fingerprint:
        raise ValueError("scan table is not normalized for this PulseDocument")
    committed = replace(document, scan_table=table, scan_recipe=None)
    if recipe_source is None:
        return committed
    source = str(recipe_source)
    if not source.strip():
        raise ValueError("scan recipe source must be non-empty")
    generated_columns = {
        parameter_id: tuple(row[index] for row in table.rows)
        for index, parameter_id in enumerate(table.columns)
    }
    return attach_scan_recipe(
        committed,
        source=source,
        generated_columns=generated_columns,
    )
