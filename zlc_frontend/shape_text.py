"""Canonical, state-free presentation text shared by frontend surfaces.

This module owns only visible labels.  Dataset structure remains authoritative
in :mod:`zlc_data`; ``describe_dataset_shape`` consumes that schema without
inventing or inferring any dimensions.
"""

from __future__ import annotations

import re

from zlc_data.schema import DatasetSchema

__all__ = [
    "describe_dataset_shape",
    "indexed_unique_name",
]


def _format_dims(dims) -> str:
    """Spell one declared data-shape tuple with the canonical ``×`` glyph."""

    parts = tuple(str(int(n)) for n in (dims or ()))
    return "×".join(parts) if parts else "1"


def _contract_shape_label(repeat, point_count, data_shape) -> str:
    """Spell the physical ``R × P × (*data_shape)`` tensor contract.

    ``P`` is one storage dimension even when ``PointLayout`` maps its rows to
    several logical scan axes.  Logical topology is presented separately; it
    must never make an ``(R, P, 1)`` scalar look like a higher-rank ndarray.
    """

    p = int(point_count)
    if p <= 0:
        raise ValueError("point_count must be positive")
    ds = tuple(int(n) for n in (data_shape or ()))
    if not ds:
        raise ValueError("data_shape must include the declared scalar/data axis")
    return f"{int(repeat)} × {p} × ({_format_dims(ds)})"


def describe_dataset_shape(schema, value=None) -> str:
    """Return the exact physical ``R × P × (*data_shape)`` display text.

    ``P`` is ``PointLayout.storage_size`` even when several logical scan axes
    describe those rows.  A scalar therefore reads ``R × P × (1)`` through its
    declared singleton carrier; a two-dimensional image reads
    ``R × P × (H×W)``.  When a value is supplied, its full ndarray shape is
    checked against the same schema before any text is returned.  ``None`` is
    accepted only for a signal that has not published a schema or value yet.
    """

    if schema is None:
        if value is None:
            return "—"
        raise TypeError("a published signal value requires DatasetSchema")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("signal schema must be DatasetSchema")
    if value is not None:
        actual = tuple(int(size) for size in value.shape)
        expected = tuple(schema.physical_shape)
        if actual != expected:
            raise ValueError(
                "signal value shape does not match DatasetSchema: "
                f"{actual} != {expected}"
            )
    return _contract_shape_label(
        schema.repeat_axis.size,
        schema.point_layout.storage_size,
        schema.cell_schema.data_shape,
    )


_INDEX_SUFFIX_RE = re.compile(r"^(.*?)\s*#\d+$")


def indexed_unique_name(base: str, taken) -> str:
    """Return ``"<root> #N"`` using the smallest free positive index."""

    text = str(base or "panel").strip() or "panel"
    match = _INDEX_SUFFIX_RE.match(text)
    root = (match.group(1).strip() if match else text) or "panel"
    occupied = set(taken)
    index = 1
    while f"{root} #{index}" in occupied:
        index += 1
    return f"{root} #{index}"
