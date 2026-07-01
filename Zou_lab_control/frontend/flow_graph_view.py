"""A reusable, self-contained DAG renderer for a saved figure's PROVENANCE flow -- the "how was this
data produced" tree the :mod:`~.figure_viewer` shows on its **Flow** tab.

A figure's provenance is NOT a single chain: a site map consumes occupancy + centres + an underlay frame,
each of which flows up its OWN chain to a device.  So the graph BRANCHES UPWARD (several parents) and can
CONVERGE (one source feeding two processors that both feed the plot).  This widget takes the neutral
``{"nodes": [...], "edges": [...]}`` graph captured by
:func:`~...neutral_atom.operations.figure_capture.capture_flow_graph` and draws it as a node-link diagram:

* nodes are laid out in TOPOLOGICAL LAYERS -- the terminal ``plot`` at the bottom, each producing node one
  layer up from the node it feeds (longest-path layering, so an edge always points DOWN across >=1 layer),
  several nodes allowed per layer;
* within a layer, nodes are ordered by the barycentre of their downstream neighbours to keep the lines
  short and un-crossed;
* each node is a rounded Fluent box coloured by its ROLE (device / measurement / processor / plot / raw
  data) with a small role badge; a device-holding source is marked so the reader sees its snapshot is
  attached;
* each edge is a line labelled with the signal name + shape it carries.

Everything -- geometry, colours, fonts -- comes from the frontend's OWN tokens (``style.PALETTE``,
``qt_fluent`` colours + ``scaled_px``), never a per-call art knob, so it obeys the sealed-API contract and
scales with the display like every other Fluent control.  The widget is a plain ``QWidget`` that paints
itself and reports its natural size, so a caller drops it inside a ``FluentScrollArea`` (the Flow tab does)
and a large graph simply scrolls.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from .qt_fluent import (
    ACCENT,
    DIVIDER,
    FONT,
    GREEN,
    GREY,
    ORANGE,
    TEXT,
    fluent_font_size,
    scaled_px,
)


#: Role -> (fill, border) colours, all from the shared Fluent token set (never a fresh per-call colour):
#: a device source is the warm ORANGE (an apparatus), a measurement the lab ACCENT blue, a processor the
#: GREEN transform, the terminal plot the neutral TEXT-dark, and a bare raw-data leaf plain GREY.  A role
#: the capture ever adds that is not listed here falls back to the neutral plot style.
_ROLE_STYLE: Mapping[str, tuple[str, str]] = {
    "measurement": (ACCENT, "#4A7BA6"),
    "processor": (GREEN, "#4E8B77"),
    "task": (ORANGE, "#A9743F"),
    "plot": ("#5B5B5B", "#3A3A3A"),
    "raw data": (GREY, "#7C7C7C"),
}
_DEFAULT_STYLE = ("#5B5B5B", "#3A3A3A")


def _role_style(role: str) -> tuple[str, str]:
    return _ROLE_STYLE.get(str(role), _DEFAULT_STYLE)


class FlowGraphView(QtWidgets.QWidget):
    """Paints a provenance ``flow_graph`` as a layered, branching node-link diagram (see module docstring).

    ``set_graph({"nodes": [...], "edges": [...]})`` lays it out and repaints; ``set_graph(None)`` (or an
    empty / malformed graph) shows a muted placeholder line, so the Flow tab is never blank for an old npz
    that recorded no flow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: dict | None = None
        self._placeholder = "No data-flow was recorded for this figure."
        # Laid-out geometry, rebuilt on every set_graph: node id -> QRectF (box), plus the edge list.
        self._boxes: dict[str, QtCore.QRectF] = {}
        self._layout_nodes: dict[str, dict] = {}
        self._layout_edges: list[dict] = []
        self._content = QtCore.QSize(scaled_px(320, minimum=200), scaled_px(120, minimum=90))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(self._content)

    # ------------------------------------------------------------------ tokens
    def _node_w(self) -> int:
        return scaled_px(150, minimum=108)

    def _node_h(self) -> int:
        return scaled_px(46, minimum=34)

    def _gap_x(self) -> int:
        return scaled_px(46, minimum=30)

    def _gap_y(self) -> int:
        return scaled_px(72, minimum=50)

    def _margin(self) -> int:
        return scaled_px(18, minimum=12)

    def _label_font(self) -> QtGui.QFont:
        return QtGui.QFont(FONT, fluent_font_size())

    def _small_font(self) -> QtGui.QFont:
        f = QtGui.QFont(FONT, max(1, fluent_font_size() - 2))
        return f

    # ------------------------------------------------------------------ public
    def set_graph(self, graph: object) -> None:
        """Accept a ``{"nodes": [...], "edges": [...]}`` mapping (or ``None`` / anything malformed -> the
        placeholder) and lay it out.  Tolerant of a stored dict whose lists came back from an npz as
        object arrays -- it only reads ``id`` / ``name`` / ``role`` off each node and ``from`` / ``to`` /
        ``signal`` / ``shape`` off each edge."""
        self._graph = graph if self._valid(graph) else None
        self._relayout()
        self.update()

    @staticmethod
    def _valid(graph: object) -> bool:
        return (isinstance(graph, Mapping)
                and isinstance(graph.get("nodes"), Sequence)
                and len(graph.get("nodes")) > 0)

    # ------------------------------------------------------------------ layout
    def _relayout(self) -> None:
        self._boxes = {}
        self._layout_nodes = {}
        self._layout_edges = []
        if self._graph is None:
            self._content = QtCore.QSize(scaled_px(320, minimum=200), scaled_px(120, minimum=90))
            self.setMinimumSize(self._content)
            return

        nodes = {str(n.get("id")): dict(n) for n in self._graph.get("nodes", []) if isinstance(n, Mapping)}
        edges = [dict(e) for e in self._graph.get("edges", []) if isinstance(e, Mapping)]
        # Keep only edges whose endpoints both exist (a malformed npz never crashes the paint).
        edges = [e for e in edges if str(e.get("from")) in nodes and str(e.get("to")) in nodes]
        self._layout_nodes = nodes
        self._layout_edges = edges

        # DEPTH = longest path FROM a node down to a terminal (a node with no outgoing edge, i.e. the
        # plot).  A node feeding the plot is depth 0-from-terminal +1... -- we instead compute depth as
        # the longest distance UP from the terminal so the plot sits at layer 0 (bottom) and sources at
        # the top.  Equivalent: depth(n) = 0 if n has no outgoing edge, else 1 + max(depth(target)).
        out_targets: dict[str, list[str]] = {nid: [] for nid in nodes}
        for e in edges:
            out_targets[str(e.get("from"))].append(str(e.get("to")))

        depth: dict[str, int] = {}

        def _depth(nid: str, stack: frozenset) -> int:
            if nid in depth:
                return depth[nid]
            if nid in stack:                                 # cycle guard (should not happen in a DAG)
                return 0
            targets = out_targets.get(nid, [])
            d = 0 if not targets else 1 + max(_depth(t, stack | {nid}) for t in targets)
            depth[nid] = d
            return d

        for nid in nodes:
            _depth(nid, frozenset())
        max_depth = max(depth.values(), default=0)

        # Group nodes by layer (row): layer index counted from the TOP so sources are row 0 and the plot
        # is the last row -- matches reading the flow top (apparatus) -> bottom (figure).
        layers: dict[int, list[str]] = {}
        for nid, d in depth.items():
            layer = max_depth - d
            layers.setdefault(layer, []).append(nid)

        # Order within each layer by the barycentre of the layer BELOW (the nodes each feeds), so lines
        # stay short and crossings are reduced.  Seed the bottom layer (the plot) then sweep upward.
        order: dict[int, list[str]] = {}
        n_layers = max_depth + 1
        for layer in range(n_layers - 1, -1, -1):
            ids = layers.get(layer, [])
            if layer == n_layers - 1:
                order[layer] = sorted(ids)                   # bottom (plot) -- deterministic seed
                continue
            below = order.get(layer + 1, [])
            pos_below = {nid: i for i, nid in enumerate(below)}

            def _bary(nid: str) -> float:
                tgt = [pos_below[t] for t in out_targets.get(nid, []) if t in pos_below]
                return sum(tgt) / len(tgt) if tgt else 1e9   # no downstream in the next layer -> park right
            order[layer] = sorted(ids, key=lambda nid: (_bary(nid), nid))

        # Place the boxes: each layer is a horizontal row, centred so the whole graph is balanced.
        nw, nh = self._node_w(), self._node_h()
        gx, gy, m = self._gap_x(), self._gap_y(), self._margin()
        widest = max((len(order.get(l, [])) for l in range(n_layers)), default=1)
        content_w = m * 2 + widest * nw + max(0, widest - 1) * gx
        content_h = m * 2 + n_layers * nh + max(0, n_layers - 1) * gy
        for layer in range(n_layers):
            ids = order.get(layer, [])
            row_w = len(ids) * nw + max(0, len(ids) - 1) * gx
            x0 = (content_w - row_w) / 2.0
            y = m + layer * (nh + gy)
            for i, nid in enumerate(ids):
                x = x0 + i * (nw + gx)
                self._boxes[nid] = QtCore.QRectF(x, y, nw, nh)

        self._content = QtCore.QSize(max(scaled_px(320, minimum=200), int(content_w)),
                                     max(scaled_px(120, minimum=90), int(content_h)))
        self.setMinimumSize(self._content)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        return self._content

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        if self._graph is None or not self._boxes:
            painter.setPen(QtGui.QColor(GREY))
            painter.setFont(self._small_font())
            painter.drawText(self.rect().adjusted(scaled_px(14), 0, -scaled_px(14), 0),
                             int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap),
                             self._placeholder)
            painter.end()
            return
        self._paint_edges(painter)
        self._paint_nodes(painter)
        painter.end()

    def _paint_edges(self, painter: QtGui.QPainter) -> None:
        painter.setFont(self._small_font())
        fm = QtGui.QFontMetrics(self._small_font())
        # Edges sharing ONE downstream target fan into DISTINCT points on its top edge (so two parents do
        # not both plug into the exact centre); edges sharing ONE source likewise fan OUT of distinct
        # points on its bottom edge -- so the common case of several signals flowing from the SAME producer
        # into the SAME plot (occupied + centres + frame) reads as a spread fan, not one overlapping line.
        # Their labels are then STAGGERED along the line by index so parallel branch labels never smear.
        by_target: dict[str, list[dict]] = {}
        by_source: dict[str, list[dict]] = {}
        for e in self._layout_edges:
            by_target.setdefault(str(e.get("to")), []).append(e)
            by_source.setdefault(str(e.get("from")), []).append(e)

        def _fan_x(box: QtCore.QRectF, group: list, e: dict) -> float:
            k, n = group.index(e), max(1, len(group))
            return box.left() + box.width() * (0.2 + 0.6 * (k + 1) / (n + 1))

        for e in self._layout_edges:
            src = self._boxes.get(str(e.get("from")))
            dst = self._boxes.get(str(e.get("to")))
            if src is None or dst is None:
                continue
            tgroup = by_target.get(str(e.get("to")), [])
            sgroup = by_source.get(str(e.get("from")), [])
            k = tgroup.index(e)
            n = max(1, len(tgroup))
            # An edge flows DOWN: from a fanned point on the upstream box's bottom edge to a fanned point on
            # the downstream box's top edge.
            p1 = QtCore.QPointF(_fan_x(src, sgroup, e), src.bottom())
            p2 = QtCore.QPointF(_fan_x(dst, tgroup, e), dst.top())
            pen = QtGui.QPen(QtGui.QColor(GREY))
            pen.setWidthF(max(1.0, scaled_px(1, minimum=1)))
            painter.setPen(pen)
            painter.drawLine(p1, p2)
            self._draw_arrow_head(painter, p1, p2)
            # Edge label = signal name (+ shape) on a small white plate.  Position it at a per-edge fraction
            # down the line (staggered by index) so parallel branch labels sit at different heights.
            label = self._edge_label(e)
            if label:
                t = 0.30 + 0.40 * (k / max(1, n - 1)) if n > 1 else 0.5
                cx = p1.x() + (p2.x() - p1.x()) * t
                cy = p1.y() + (p2.y() - p1.y()) * t
                tw = fm.horizontalAdvance(label) + scaled_px(8, minimum=6)
                th = fm.height() + scaled_px(2, minimum=2)
                plate = QtCore.QRectF(cx - tw / 2.0, cy - th / 2.0, tw, th)
                painter.setBrush(QtGui.QColor("#FFFFFF"))
                painter.setPen(QtGui.QColor(DIVIDER))
                painter.drawRoundedRect(plate, scaled_px(3, minimum=2), scaled_px(3, minimum=2))
                painter.setPen(QtGui.QColor(TEXT))
                painter.drawText(plate, int(QtCore.Qt.AlignCenter), label)

    @staticmethod
    def _edge_label(e: Mapping) -> str:
        sig = str(e.get("signal") or "")
        shape = e.get("shape")
        if sig and shape is not None:
            try:
                dims = "×".join(str(int(n)) for n in shape)
            except Exception:
                dims = ""
            return f"{sig} ({dims})" if dims else sig
        return sig

    def _draw_arrow_head(self, painter: QtGui.QPainter, p1: QtCore.QPointF, p2: QtCore.QPointF) -> None:
        import math
        ang = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        size = scaled_px(6, minimum=4)
        tip = p2
        left = QtCore.QPointF(tip.x() - size * math.cos(ang - math.pi / 6),
                              tip.y() - size * math.sin(ang - math.pi / 6))
        right = QtCore.QPointF(tip.x() - size * math.cos(ang + math.pi / 6),
                               tip.y() - size * math.sin(ang + math.pi / 6))
        painter.setBrush(QtGui.QColor(GREY))
        painter.setPen(QtGui.QColor(GREY))
        painter.drawPolygon(QtGui.QPolygonF([tip, left, right]))

    def _paint_nodes(self, painter: QtGui.QPainter) -> None:
        radius = scaled_px(6, minimum=4)
        for nid, box in self._boxes.items():
            node = self._layout_nodes.get(nid, {})
            role = str(node.get("role") or "node")
            fill, border = _role_style(role)
            # A device-holding source gets a warm apparatus fill regardless of its measurement/task role,
            # so the reader spots where the hardware snapshot is attached.
            if node.get("has_devices"):
                fill, border = _ROLE_STYLE["task"]
            painter.setBrush(QtGui.QColor(fill))
            pen = QtGui.QPen(QtGui.QColor(border))
            pen.setWidthF(max(1.0, scaled_px(1, minimum=1)))
            painter.setPen(pen)
            painter.drawRoundedRect(box, radius, radius)

            # Node NAME (primary line) -- white on the coloured fill for contrast.
            name_font = self._label_font()
            painter.setPen(QtGui.QColor("#FFFFFF"))
            painter.setFont(name_font)
            name_rect = QtCore.QRectF(box.x() + scaled_px(6), box.y() + scaled_px(4),
                                      box.width() - scaled_px(12), box.height() * 0.5)
            painter.drawText(name_rect, int(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom),
                             self._elide(str(node.get("name") or "node"), name_rect.width(), name_font))
            # Role badge line beneath the name (the layer word), slightly muted.
            badge_font = self._small_font()
            painter.setFont(badge_font)
            painter.setPen(QtGui.QColor(255, 255, 255, 205))
            badge = role if not node.get("has_devices") else f"{role} · device"
            role_rect = QtCore.QRectF(box.x() + scaled_px(6), box.center().y(),
                                      box.width() - scaled_px(12), box.height() * 0.5 - scaled_px(3))
            painter.drawText(role_rect, int(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop),
                             self._elide(badge, role_rect.width(), badge_font))

    @staticmethod
    def _elide(text: str, width: float, font: QtGui.QFont) -> str:
        fm = QtGui.QFontMetrics(font)
        return fm.elidedText(str(text), QtCore.Qt.ElideRight, int(max(0, width)))


__all__ = ["FlowGraphView"]
