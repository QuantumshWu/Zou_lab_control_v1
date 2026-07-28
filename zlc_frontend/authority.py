"""Headless text projection for explicit data-transform authority.

The values and codecs remain owned by :mod:`zlc_data`.  This module only
formats an already-authored transform for immutable UI summaries; it never
infers, edits, commits, or executes an operation.
"""

from __future__ import annotations

from zlc_data import (
    CoordinateRangeSelection,
    DataTransformSpec,
    HistogramSpec,
    IndexRangeSelection,
    IndexSelection,
    ReductionSpec,
    Selection,
)


def describe_authoritative_transform(
    spec: DataTransformSpec | None,
) -> str:
    """Describe every persisted authority operation without inferring axes."""

    if spec is None:
        return "None · no user-authored Select/Reduce"
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec or None")

    def source_text(source) -> str:
        return (
            source.kind.lower()
            if source.axis_id is None
            else f"{source.kind.lower()}:{source.axis_id.value}"
        )

    operations: list[str] = []
    for operation in spec.operations:
        if isinstance(operation, Selection):
            terms: list[str] = []
            for term in operation.terms:
                axis = term.axis_id.value
                if isinstance(term, IndexSelection):
                    terms.append(f"{axis}=index[{term.index}]")
                elif isinstance(term, IndexRangeSelection):
                    terms.append(f"{axis}=indices[{term.start}:{term.stop}]")
                elif isinstance(term, CoordinateRangeSelection):
                    frame = (
                        ""
                        if term.coordinate_frame is None
                        else f"@{term.coordinate_frame.value}"
                    )
                    terms.append(
                        f"{axis}=coordinates[{term.lower},{term.upper}]{frame}"
                    )
                else:  # pragma: no cover - Selection owns the closed term union.
                    raise TypeError("Selection contains an unsupported term")
            operations.append("select(" + ", ".join(terms) + ")")
        elif isinstance(operation, ReductionSpec):
            axes = ",".join(source_text(source) for source in operation.sources)
            minimum = (
                ""
                if operation.minimum_valid_count is None
                else f"/min={operation.minimum_valid_count}"
            )
            operations.append(
                f"reduce({axes})={operation.method.value}"
                f"/{operation.missing_policy.value}"
                f"/{operation.validity_policy.value}"
                f"{minimum}"
            )
        elif isinstance(operation, HistogramSpec):
            axes = ",".join(source_text(source) for source in operation.sources)
            operations.append(
                f"histogram({axes})→{operation.bin_axis_id.value}"
                f"[{len(operation.bin_edges) - 1} bins]"
            )
        else:  # pragma: no cover - DataTransformSpec owns the closed union.
            raise TypeError("DataTransformSpec contains an unsupported operation")
    return "AUTHORITATIVE · " + " → ".join(operations)


__all__ = ["describe_authoritative_transform"]
