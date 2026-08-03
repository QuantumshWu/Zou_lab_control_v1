"""Array-free operator information for one current Figure archive."""

from __future__ import annotations

from collections.abc import Mapping
from pprint import pformat
from typing import TypeAlias

from zlc_frontend.flow_graph import FlowGraph, flow_graph_from_tree
from zlc_plot import plot_spec_to_tree
from zlc_workbench.data_figure.archive_io import LoadedFigureArchive


InfoRows: TypeAlias = tuple[tuple[str, object], ...]
FigureInfoProjection: TypeAlias = tuple[
    InfoRows,
    InfoRows,
    InfoRows,
    FlowGraph | None,
    str,
]


def _axis_text(axis) -> str:
    unit = "" if axis.unit is None else f" [{axis.unit}]"
    return (
        f"{axis.name} ({axis.axis_id}; role={axis.role}; "
        f"size={axis.size}{unit})"
    )


def _point_column_text(column) -> str:
    unit = "" if column.unit is None else f" [{column.unit}]"
    return (
        f"{column.name} ({column.coordinate_id}; role={column.role}; "
        f"kind={column.value_kind}{unit})"
    )


def _dataset_rows(loaded: LoadedFigureArchive) -> InfoRows:
    snapshot = loaded.archive.snapshot
    schema = snapshot.block.schema
    topology = schema.grid_topology
    topology_text = "(none; ordered point sequence)"
    if topology is not None:
        dimensions = ", ".join(str(item) for item in topology.dimension_ids)
        topology_text = (
            f"dimensions=({dimensions}); shape={topology.logical_shape}; "
            f"mapped_rows={len(topology.row_to_cell)}"
        )
    return (
        ("path", str(loaded.path)),
        ("revision", snapshot.ref),
        ("shape", schema.physical_shape),
        ("repeat", _axis_text(schema.repeat_axis)),
        ("point rows", schema.point_table.row_count),
        (
            "point columns",
            ", ".join(_point_column_text(column) for column in schema.point_table.columns)
            or "(none)",
        ),
        ("grid topology", topology_text),
        (
            "data",
            ", ".join(_axis_text(axis) for axis in schema.cell_schema.data_axes)
            or "scalar carrier",
        ),
        ("dtype", schema.cell_schema.dtype),
        ("unit", schema.cell_schema.value_unit or "(none)"),
        ("validity", type(snapshot.block.validity).__name__),
    )


def _flow_graph(metadata: Mapping[str, object]) -> FlowGraph | None:
    tree = metadata.get("flow_graph")
    return None if tree is None else flow_graph_from_tree(tree)


def project_figure_info(loaded: LoadedFigureArchive) -> FigureInfoProjection:
    """Project one decoded archive without traversing its ndarray values."""

    if not isinstance(loaded, LoadedFigureArchive):
        raise TypeError("archive must be LoadedFigureArchive")
    archive = loaded.archive
    kind = getattr(archive.spec, "kind", None)
    plot_rows: InfoRows = (
        ("kind", getattr(kind, "value", kind)),
        ("size", archive.size),
        ("spec", plot_spec_to_tree(archive.spec)),
        ("parameters", dict(archive.parameters)),
    )
    device_rows = tuple(
        (str(key), value)
        for key, value in archive.metadata.items()
        if key != "flow_graph"
    ) or (("metadata", "(none recorded)"),)
    raw = pformat(
        {
            "path": str(loaded.path),
            "snapshot_ref": archive.snapshot.ref,
            "schema": archive.snapshot.block.schema,
            "validity": archive.snapshot.block.validity,
            "plot_spec": plot_spec_to_tree(archive.spec),
            "size": archive.size,
            "parameters": dict(archive.parameters),
            "metadata": dict(archive.metadata),
        },
        sort_dicts=False,
        width=100,
    )
    return (
        plot_rows,
        _dataset_rows(loaded),
        device_rows,
        _flow_graph(archive.metadata),
        raw,
    )


__all__ = ["FigureInfoProjection", "InfoRows", "project_figure_info"]
