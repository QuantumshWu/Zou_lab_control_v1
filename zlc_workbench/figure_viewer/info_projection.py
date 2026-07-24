"""Headless projection of a current Figure archive into operator-facing facts.

This module performs no I/O and owns no Qt objects.  It translates the typed
archive already accepted by the FigureViewer into the five stable Info surfaces;
the Qt pane only lays those values out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pprint import pformat
from typing import TypeAlias

from zlc_storage.paths import display_path


InfoRows: TypeAlias = tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class FlowGraphNode:
    """One immutable node projected from current archive metadata."""

    node_id: str
    name: str
    role: str
    has_devices: bool = False


@dataclass(frozen=True, slots=True)
class FlowGraphEdge:
    """One immutable directed edge projected from current archive metadata."""

    source_id: str
    target_id: str
    signal: str = ""
    shape: tuple[int, ...] | None = None
    role: str = ""


@dataclass(frozen=True, slots=True)
class FlowGraph:
    """Current archive-to-Qt graph DTO with exact structural invariants."""

    nodes: tuple[FlowGraphNode, ...]
    edges: tuple[FlowGraphEdge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("flow graph requires at least one node")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("flow graph node ids must be unique")
        known = frozenset(node_ids)
        if any(
            edge.source_id not in known or edge.target_id not in known
            for edge in self.edges
        ):
            raise ValueError("flow graph edge endpoint is not a declared node")

        outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            outgoing[edge.source_id].append(edge.target_id)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("flow graph must be acyclic")
            if node_id in visited:
                return
            visiting.add(node_id)
            for target_id in outgoing[node_id]:
                visit(target_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in node_ids:
            visit(node_id)


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


def _view_text(view) -> str:
    bindings = ", ".join(
        f"{binding.axis_id}={binding.role.value}"
        for binding in view.axis_bindings
    )
    selections = len(view.display_selections)
    suffix = "" if selections == 0 else f"; selections={selections}"
    return f"intent={view.intent.value}; {bindings or 'no axis bindings'}{suffix}"


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
        rows.extend(
            (
                (f"{prefix} id", descriptor.dataset_id),
                (f"{prefix} revision", snapshot.ref),
                (f"{prefix} shape", schema.physical_shape),
                (f"{prefix} repeat", _axis_text(schema.repeat_axis)),
                (
                    f"{prefix} points",
                    ", ".join(_axis_text(axis) for axis in schema.point_axes)
                    or "(none)",
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


def _exact_fields(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    keys = frozenset(value)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra or any(not isinstance(key, str) for key in keys):
        raise ValueError(
            f"{label} fields do not match the current contract; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _flow_shape(value: object, label: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in value
    ):
        raise TypeError(
            f"{label} must be a tuple of non-negative integers or None"
        )
    return value


def _flow_graph(metadata: Mapping[str, object]) -> FlowGraph | None:
    """Decode the sole current top-level flow-graph metadata contract.

    The canonical archive decoder already rejects historical archive envelopes.
    This projection likewise has no nested ``provenance`` fallback and never
    passes an untyped mapping into the Qt renderer.
    """

    if "flow_graph" not in metadata:
        return None
    tree = _exact_fields(
        metadata["flow_graph"],
        required=frozenset({"nodes", "edges"}),
        label="figure metadata flow_graph",
    )
    node_values = tree["nodes"]
    edge_values = tree["edges"]
    if not isinstance(node_values, tuple):
        raise TypeError("figure metadata flow_graph nodes must be a tuple")
    if not isinstance(edge_values, tuple):
        raise TypeError("figure metadata flow_graph edges must be a tuple")

    nodes: list[FlowGraphNode] = []
    for index, value in enumerate(node_values):
        label = f"figure metadata flow_graph node {index}"
        item = _exact_fields(
            value,
            required=frozenset({"id", "name", "role"}),
            optional=frozenset({"has_devices"}),
            label=label,
        )
        has_devices = item.get("has_devices", False)
        if not isinstance(has_devices, bool):
            raise TypeError(f"{label} has_devices must be bool")
        nodes.append(
            FlowGraphNode(
                node_id=_required_text(item["id"], f"{label} id"),
                name=_required_text(item["name"], f"{label} name"),
                role=_required_text(item["role"], f"{label} role"),
                has_devices=has_devices,
            )
        )

    edges: list[FlowGraphEdge] = []
    for index, value in enumerate(edge_values):
        label = f"figure metadata flow_graph edge {index}"
        item = _exact_fields(
            value,
            required=frozenset({"from", "to"}),
            optional=frozenset({"signal", "shape", "role"}),
            label=label,
        )
        edges.append(
            FlowGraphEdge(
                source_id=_required_text(item["from"], f"{label} from"),
                target_id=_required_text(item["to"], f"{label} to"),
                signal=_optional_text(item.get("signal", ""), f"{label} signal"),
                shape=_flow_shape(item.get("shape"), f"{label} shape"),
                role=_optional_text(item.get("role", ""), f"{label} role"),
            )
        )
    return FlowGraph(nodes=tuple(nodes), edges=tuple(edges))


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
            "display": value.display,
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
    if value.display is not None:
        plot_rows.append(("display", value.display))

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
