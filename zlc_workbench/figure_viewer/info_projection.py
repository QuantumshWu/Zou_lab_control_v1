"""Headless projection of a current Figure archive into operator-facing facts.

This module performs no I/O and owns no Qt objects.  It translates the typed
archive already accepted by the FigureViewer into the five stable Info surfaces;
the Qt pane only lays those values out.
"""

from __future__ import annotations

from collections.abc import Mapping
from pprint import pformat
from typing import TypeAlias

from zlc_frontend import FlowGraph, flow_graph_from_tree
from zlc_storage.paths import display_path


InfoRows: TypeAlias = tuple[tuple[str, object], ...]


FigureInfoProjection: TypeAlias = tuple[
    InfoRows,
    InfoRows,
    InfoRows,
    FlowGraph | None,
    str,
]


def _axis_text(axis) -> str:
    """Describe one declared axis without inferring anything from array rank."""

    unit = "" if axis.unit is None else f" [{axis.unit}]"
    return (
        f"{axis.name} ({axis.axis_id}; role={axis.role}; "
        f"size={axis.size}{unit})"
    )


def _binding_text(binding) -> str:
    """Describe one source-owned display binding and its visible operation."""

    source = binding.source
    source_text = source.kind.lower()
    if source.axis_id is not None:
        source_text = f"{source_text}:{source.axis_id}"
    operation = ""
    if binding.selector is not None:
        operation = f"({binding.selector})"
    elif binding.reduction is not None:
        operation = f"({binding.reduction.method.value.lower()})"
    return f"{source_text}={binding.role.value}{operation}"


def _view_text(view) -> str:
    bindings = ", ".join(
        _binding_text(binding) for binding in view.source_bindings
    )
    points = (
        "all point rows"
        if view.point_ordinals is None
        else f"point rows={view.point_ordinals}"
    )
    return (
        f"intent={view.intent.value}; "
        f"{bindings or 'no source bindings'}; {points}"
    )


def _point_column_text(column) -> str:
    """Describe one correlated PointTable column without axis-shaped fiction."""

    unit = "" if column.unit is None else f" [{column.unit}]"
    return (
        f"{column.name} ({column.coordinate_id}; role={column.role}; "
        f"kind={column.value_kind}{unit})"
    )


def _dataset_projection(figure) -> InfoRows:
    """Project typed source schemas into human-readable, array-free rows."""

    rows: list[tuple[str, object]] = []
    document = figure.document
    datasets = figure.datasets
    for descriptor in document.datasets:
        snapshot = datasets.resolve(descriptor.dataset_id)
        schema = snapshot.block.schema
        cell = schema.cell_schema
        prefix = descriptor.label
        topology = schema.grid_topology
        topology_text = "(none; ordered point sequence)"
        if topology is not None:
            dimensions = ", ".join(str(item) for item in topology.dimension_ids)
            topology_text = (
                f"dimensions=({dimensions}); shape={topology.logical_shape}; "
                f"mapped_rows={len(topology.row_to_cell)}"
            )
        rows.extend(
            (
                (f"{prefix} id", descriptor.dataset_id),
                (f"{prefix} revision", snapshot.ref),
                (f"{prefix} shape", schema.physical_shape),
                (f"{prefix} repeat", _axis_text(schema.repeat_axis)),
                (f"{prefix} point rows", schema.point_table.row_count),
                (
                    f"{prefix} point columns",
                    ", ".join(
                        _point_column_text(column)
                        for column in schema.point_table.columns
                    )
                    or "(none)",
                ),
                (
                    f"{prefix} grid topology",
                    topology_text,
                ),
                (
                    f"{prefix} data",
                    ", ".join(_axis_text(axis) for axis in cell.data_axes)
                    or "scalar",
                ),
                (f"{prefix} dtype", cell.dtype),
                (f"{prefix} unit", cell.value_unit or "(none)"),
                (
                    f"{prefix} validity",
                    cell.validity_contract.mode.value,
                ),
            )
        )
    return tuple(rows)


def _flow_graph(metadata: Mapping[str, object]) -> FlowGraph | None:
    """Decode the sole current top-level flow-graph metadata contract.

    The canonical archive decoder already rejects historical archive envelopes.
    This projection likewise has no nested ``provenance`` fallback and never
    passes an untyped mapping into the Qt renderer.
    """

    if "flow_graph" not in metadata:
        return None
    return flow_graph_from_tree(metadata["flow_graph"])


def _raw_projection(archive) -> str:
    """Show the complete typed descriptive record, excluding source array bytes."""

    value = archive.archive
    figure = value.figure
    datasets = []
    for descriptor in figure.document.datasets:
        snapshot = figure.datasets.resolve(descriptor.dataset_id)
        datasets.append(
            {
                "descriptor": descriptor,
                "reference": snapshot.ref,
                "schema": snapshot.block.schema,
                "validity": snapshot.block.validity,
            }
        )
    return pformat(
        {
            "path": str(archive.path),
            "payload_digest": value.payload_digest,
            "document": figure.document,
            "datasets": tuple(datasets),
            "fit_results": dict(figure.fit_results),
            "presentation": value.presentation,
            "metadata": dict(value.metadata),
        },
        sort_dicts=False,
        width=100,
    )


def project_figure_info(archive) -> FigureInfoProjection:
    """Project one fully decoded current archive without touching its array bytes."""

    value = archive.archive
    figure = value.figure
    document = figure.document
    plot_rows: list[tuple[str, object]] = [
        ("document", document.document_id),
        ("revision", document.revision),
        ("payload_digest", value.payload_digest),
    ]
    for layer in document.layers:
        descriptor = document.descriptor(layer.dataset_id)
        plot_rows.append(
            (
                f"layer {layer.layer_id}",
                f"{descriptor.label} ({descriptor.dataset_id}); "
                f"{_view_text(layer.view)}",
            )
        )
    if document.selections:
        plot_rows.append(("selections", len(document.selections)))
    plot_rows.append(("presentation", value.presentation))

    measurement_rows = list(_dataset_projection(figure))
    measurement_rows.append(("path", display_path(str(archive.path))))

    device_rows: list[tuple[str, object]] = []
    for key, item in value.metadata.items():
        if key != "flow_graph":
            device_rows.append((str(key), item))
    if not device_rows:
        device_rows.append(("metadata", "(none recorded)"))

    return (
        tuple(plot_rows),
        tuple(measurement_rows),
        tuple(device_rows),
        _flow_graph(value.metadata),
        _raw_projection(archive),
    )


__all__ = ["FigureInfoProjection", "InfoRows", "project_figure_info"]
