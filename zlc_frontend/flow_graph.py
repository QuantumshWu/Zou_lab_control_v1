"""Headless production-flow values shared by Figure presentation surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class FlowGraphNode:
    """One named operation or device in a production-flow graph."""

    node_id: str
    name: str
    role: str
    has_devices: bool = False

    def __post_init__(self) -> None:
        canonical_text(self.node_id, "flow graph node id")
        canonical_text(self.name, "flow graph node name")
        canonical_text(self.role, "flow graph node role")
        if type(self.has_devices) is not bool:
            raise TypeError("flow graph node has_devices must be bool")


@dataclass(frozen=True, slots=True)
class FlowGraphEdge:
    """One directed Dataset/Artifact relationship between graph nodes."""

    source_id: str
    target_id: str
    signal: str = ""
    shape: tuple[int, ...] | None = None
    role: str = ""

    def __post_init__(self) -> None:
        canonical_text(self.source_id, "flow graph edge source id")
        canonical_text(self.target_id, "flow graph edge target id")
        canonical_text(self.signal, "flow graph edge signal", empty=True)
        canonical_text(self.role, "flow graph edge role", empty=True)
        shape = self.shape
        if shape is not None:
            if not isinstance(shape, tuple) or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise TypeError(
                    "flow graph edge shape must be a tuple of non-negative "
                    "integers or None"
                )


@dataclass(frozen=True, slots=True)
class FlowGraph:
    """One immutable acyclic production-flow graph."""

    nodes: tuple[FlowGraphNode, ...]
    edges: tuple[FlowGraphEdge, ...]

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if not nodes:
            raise ValueError("flow graph requires at least one node")
        if any(not isinstance(node, FlowGraphNode) for node in nodes):
            raise TypeError("flow graph nodes must contain FlowGraphNode values")
        if any(not isinstance(edge, FlowGraphEdge) for edge in edges):
            raise TypeError("flow graph edges must contain FlowGraphEdge values")
        node_ids = tuple(node.node_id for node in nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("flow graph node ids must be unique")
        known = frozenset(node_ids)
        if any(
            edge.source_id not in known or edge.target_id not in known
            for edge in edges
        ):
            raise ValueError("flow graph edge endpoint is not a declared node")

        outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in edges:
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
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)


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
    if any(not isinstance(key, str) for key in keys):
        raise TypeError(f"{label} keys must be strings")
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        raise ValueError(
            f"{label} fields do not match the current contract; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    return value


def flow_graph_from_tree(value: object) -> FlowGraph:
    """Decode the sole current descriptive tree into a validated graph."""

    tree = _exact_fields(
        value,
        required=frozenset({"nodes", "edges"}),
        label="flow graph",
    )
    node_values = tree["nodes"]
    edge_values = tree["edges"]
    if not isinstance(node_values, tuple):
        raise TypeError("flow graph nodes must be a tuple")
    if not isinstance(edge_values, tuple):
        raise TypeError("flow graph edges must be a tuple")

    nodes = []
    for index, value in enumerate(node_values):
        label = f"flow graph node {index}"
        item = _exact_fields(
            value,
            required=frozenset({"id", "name", "role"}),
            optional=frozenset({"has_devices"}),
            label=label,
        )
        nodes.append(
            FlowGraphNode(
                node_id=canonical_text(item["id"], f"{label} id"),
                name=canonical_text(item["name"], f"{label} name"),
                role=canonical_text(item["role"], f"{label} role"),
                has_devices=item.get("has_devices", False),
            )
        )

    edges = []
    for index, value in enumerate(edge_values):
        label = f"flow graph edge {index}"
        item = _exact_fields(
            value,
            required=frozenset({"from", "to"}),
            optional=frozenset({"signal", "shape", "role"}),
            label=label,
        )
        edges.append(
            FlowGraphEdge(
                source_id=canonical_text(item["from"], f"{label} from"),
                target_id=canonical_text(item["to"], f"{label} to"),
                signal=canonical_text(
                    item.get("signal", ""),
                    f"{label} signal",
                    empty=True,
                ),
                shape=item.get("shape"),
                role=canonical_text(
                    item.get("role", ""),
                    f"{label} role",
                    empty=True,
                ),
            )
        )
    return FlowGraph(tuple(nodes), tuple(edges))


__all__ = [
    "FlowGraph",
    "FlowGraphEdge",
    "FlowGraphNode",
    "flow_graph_from_tree",
]
