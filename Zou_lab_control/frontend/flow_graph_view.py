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
  attached, and the flow is traced all the way UP to the apparatus: a device-holding source expands into
  compact ``device`` LEAVES (camera / sequencer) in the TOP layer, one per held device;
* each edge is a curved (cubic-Bezier) line labelled with the signal name + shape it carries; the labels
  are placed by a GLOBAL collision pass so any two signal names in the graph are mutually non-overlapping
  (a plate that had to slide off its edge keeps a thin leader line back to the edge).

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
    YELLOW,
    fluent_font_size,
    scaled_px,
)


#: Role -> (fill, border) colours, all from the shared Fluent token set (never a fresh per-call colour):
#: the most-upstream DEVICE leaf (the apparatus -- camera / sequencer) is the warm YELLOW, a measurement
#: source the lab ACCENT blue, a processor the GREEN transform, the terminal plot the neutral TEXT-dark,
#: and a bare raw-data leaf plain GREY.  A device-HOLDING source (a measurement whose ``has_devices`` is
#: set) keeps the warm ORANGE apparatus badge (see ``_paint_nodes``).  A role the capture ever adds that
#: is not listed here falls back to the neutral plot style.
_ROLE_STYLE: Mapping[str, tuple[str, str]] = {
    "measurement": (ACCENT, "#4A7BA6"),
    "processor": (GREEN, "#4E8B77"),
    "task": (ORANGE, "#A9743F"),
    "device": (YELLOW, "#B39A2E"),
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

    def _device_w(self) -> int:
        """A DEVICE leaf is a smaller box than a producing node -- it is an apparatus, not a logic node, so
        it reads as a compact terminal source (narrower + shorter, same token family)."""
        return scaled_px(104, minimum=80)

    def _device_h(self) -> int:
        return scaled_px(34, minimum=26)

    @staticmethod
    def _is_device(node: dict) -> bool:
        return str(node.get("role")) == "device"

    def _box_size(self, node: dict) -> tuple[int, int]:
        """The (w, h) for a node's box -- the compact device size for a ``device`` leaf, the standard node
        size otherwise -- so a single layout routine sizes every box from its role (no per-kind branch at
        the call site)."""
        return (self._device_w(), self._device_h()) if self._is_device(node) else (self._node_w(), self._node_h())

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

        # Place the boxes: each layer is a horizontal row, centred so the whole graph is balanced.  Boxes
        # are sized PER NODE (a device leaf is a compact box), so a row's width is the sum of its own boxes
        # + gaps and its band height is the tallest box in it (a device-only row is shorter).
        gx, gy, m = self._gap_x(), self._gap_y(), self._margin()

        def _row_w(ids: list[str]) -> float:
            return sum(self._box_size(nodes[nid])[0] for nid in ids) + max(0, len(ids) - 1) * gx

        def _row_h(ids: list[str]) -> float:
            return max((self._box_size(nodes[nid])[1] for nid in ids), default=self._node_h())

        # Content width fits the widest ROW of boxes AND the widest edge LABEL (so a signal-name plate has
        # room to sit near mid-height without being clipped by the canvas edge -- labels are often wider than
        # a node box, e.g. ``frame_alpha (1×1×96×128)``).
        fm = QtGui.QFontMetrics(self._small_font())
        widest_row = max((_row_w(order.get(l, [])) for l in range(n_layers)), default=self._node_w())
        widest_label = max((fm.horizontalAdvance(self._edge_label(e)) + scaled_px(8, minimum=6)
                            for e in edges if self._edge_label(e)), default=0)
        content_w = m * 2 + max(widest_row, widest_label)
        band_h = [_row_h(order.get(l, [])) for l in range(n_layers)]
        content_h = m * 2 + sum(band_h) + max(0, n_layers - 1) * gy
        y = m
        for layer in range(n_layers):
            ids = order.get(layer, [])
            row_w = _row_w(ids)
            x = (content_w - row_w) / 2.0
            bh = band_h[layer]
            for nid in ids:
                bw, bnh = self._box_size(nodes[nid])
                # Centre each box vertically within the layer band (a shorter device box sits mid-band).
                self._boxes[nid] = QtCore.QRectF(x, y + (bh - bnh) / 2.0, bw, bnh)
                x += bw + gx
            y += bh + gy

        self._content = QtCore.QSize(max(scaled_px(320, minimum=200), int(content_w)),
                                     max(scaled_px(120, minimum=90), int(content_h)))
        self.setMinimumSize(self._content)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        return self._content

    def label_rects(self) -> list[QtCore.QRectF]:
        """The final, collision-resolved edge-label plates for the CURRENT layout (the exact rects the paint
        draws) -- so a caller / contract test can assert the signal names are mutually non-overlapping
        without reaching into the paint.  Empty when no graph is laid out."""
        if self._graph is None or not self._boxes:
            return []
        return [item["plate"] for item in self._place_labels(self._edge_endpoints())]

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

    def _edge_endpoints(self) -> list[tuple[dict, QtCore.QPointF, QtCore.QPointF]]:
        """Every drawable edge as ``(edge, p1, p2)`` -- the fanned start point on the upstream box's bottom
        edge and the fanned end point on the downstream box's top edge.  Edges sharing ONE downstream target
        fan into DISTINCT points on its top edge (two parents do not both plug into the exact centre); edges
        sharing ONE source fan OUT of distinct points on its bottom edge -- so several signals from the SAME
        producer into the SAME plot read as a spread fan, not one overlapping line.  Shared by the paint and
        the label-placement passes (one source of edge geometry)."""
        by_target: dict[str, list[dict]] = {}
        by_source: dict[str, list[dict]] = {}
        for e in self._layout_edges:
            by_target.setdefault(str(e.get("to")), []).append(e)
            by_source.setdefault(str(e.get("from")), []).append(e)

        def _fan_x(box: QtCore.QRectF, group: list, e: dict) -> float:
            k, n = group.index(e), max(1, len(group))
            return box.left() + box.width() * (0.2 + 0.6 * (k + 1) / (n + 1))

        out: list[tuple[dict, QtCore.QPointF, QtCore.QPointF]] = []
        for e in self._layout_edges:
            src = self._boxes.get(str(e.get("from")))
            dst = self._boxes.get(str(e.get("to")))
            if src is None or dst is None:
                continue
            p1 = QtCore.QPointF(_fan_x(src, by_source.get(str(e.get("from")), []), e), src.bottom())
            p2 = QtCore.QPointF(_fan_x(dst, by_target.get(str(e.get("to")), []), e), dst.top())
            out.append((e, p1, p2))
        return out

    @staticmethod
    def _edge_path(p1: QtCore.QPointF, p2: QtCore.QPointF) -> QtGui.QPainterPath:
        """A cubic Bezier from ``p1`` (down) to ``p2`` with VERTICAL control handles, so the edge leaves the
        source going straight down and enters the target going straight down -- a smooth S when the boxes are
        offset horizontally (the curve absorbs the offset so lines and their labels do not crowd a diagonal)
        and a straight drop when they line up."""
        path = QtGui.QPainterPath(p1)
        dy = (p2.y() - p1.y()) * 0.5
        c1 = QtCore.QPointF(p1.x(), p1.y() + dy)
        c2 = QtCore.QPointF(p2.x(), p2.y() - dy)
        path.cubicTo(c1, c2, p2)
        return path

    def _place_labels(self, endpoints: list) -> list[dict]:
        """GLOBALLY non-overlapping label plates.  Each edge's label starts at the curve midpoint; the
        plates are then placed one by one and each is nudged along the edge NORMAL (perpendicular, so it
        slides sideways off the line) in growing steps until its bounding box clears every plate ALREADY
        placed AND every node box -- so any two signal names in the whole graph are mutually exclusive
        (crossing lines or not) and a name never sits on top of a node.  Returns a list of
        ``{plate, label, anchor}`` (``anchor`` = the on-curve point, for a leader line when the plate had to
        move).  Longest labels are placed FIRST (hardest to fit -> best pick of free space)."""
        import math
        fm = QtGui.QFontMetrics(self._small_font())
        pad_w = scaled_px(8, minimum=6)
        pad_h = scaled_px(2, minimum=2)
        step = scaled_px(6, minimum=4)                       # normal-offset increment per collision retry
        gap = scaled_px(2, minimum=1)                        # min clear gap between two plates
        # Node boxes are OBSTACLES too (a name must not sit on a node); grow them by the gap so a plate rests
        # just clear of a box edge.
        node_obstacles = [b.adjusted(-gap, -gap, gap, gap) for b in self._boxes.values()]

        cand: list[dict] = []
        for e, p1, p2 in endpoints:
            label = self._edge_label(e)
            if not label:
                continue
            path = self._edge_path(p1, p2)
            anchor = path.pointAtPercent(0.5)
            tw = fm.horizontalAdvance(label) + pad_w
            th = fm.height() + pad_h
            # Edge unit normal (perpendicular to the p1->p2 chord) -- the direction a plate slides to escape.
            dx, dy = (p2.x() - p1.x()), (p2.y() - p1.y())
            length = math.hypot(dx, dy) or 1.0
            nx, ny = (-dy / length, dx / length)
            cand.append({"label": label, "anchor": anchor, "tw": tw, "th": th, "nx": nx, "ny": ny})

        cand.sort(key=lambda c: c["tw"], reverse=True)       # hardest (widest) first
        placed: list[dict] = []
        margin = self._margin()
        max_w = float(self._content.width())
        for c in cand:
            tw, th = c["tw"], c["th"]
            ax, ay = c["anchor"].x(), c["anchor"].y()
            nx, ny = c["nx"], c["ny"]
            chosen = None
            # Try the on-curve midpoint, then step alternately to +/- normal offsets until the plate is clear
            # of every already-placed plate AND every node box (and inside the canvas width).
            for i in range(0, 120):
                off = 0 if i == 0 else (((i + 1) // 2) * step) * (1 if i % 2 else -1)
                cx = ax + nx * off
                cy = ay + ny * off
                plate = QtCore.QRectF(cx - tw / 2.0, cy - th / 2.0, tw, th)
                if plate.left() < margin:
                    plate.moveLeft(margin)
                if plate.right() > max_w - margin:
                    plate.moveRight(max_w - margin)
                grown = plate.adjusted(-gap, -gap, gap, gap)
                clear_labels = not any(grown.intersects(p["plate"]) for p in placed)
                clear_nodes = not any(plate.intersects(nb) for nb in node_obstacles)
                if clear_labels and clear_nodes:
                    chosen = plate
                    break
            if chosen is None:                               # never fully clear -> take the last try anyway
                chosen = QtCore.QRectF(ax - tw / 2.0, ay - th / 2.0, tw, th)
            placed.append({"plate": chosen, "label": c["label"],
                           "anchor": QtCore.QPointF(ax, ay)})
        return placed

    def _paint_edges(self, painter: QtGui.QPainter) -> None:
        painter.setFont(self._small_font())
        endpoints = self._edge_endpoints()
        # 1) the curved edges + arrow heads.
        for e, p1, p2 in endpoints:
            path = self._edge_path(p1, p2)
            pen = QtGui.QPen(QtGui.QColor(GREY))
            pen.setWidthF(max(1.0, scaled_px(1, minimum=1)))
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(path)
            # Arrow head along the curve's final tangent (approach the target box from its last segment).
            near = path.pointAtPercent(0.92)
            self._draw_arrow_head(painter, near, p2)
        # 2) the globally non-overlapping labels, each on a small white plate; a plate that had to slide off
        #    its edge gets a thin leader line back to the on-curve anchor so the reader still sees which edge
        #    it names.
        leader = QtGui.QPen(QtGui.QColor(DIVIDER))
        leader.setWidthF(max(1.0, scaled_px(1, minimum=1)))
        for item in self._place_labels(endpoints):
            plate, label, anchor = item["plate"], item["label"], item["anchor"]
            if not plate.contains(anchor):
                painter.setPen(leader)
                painter.drawLine(anchor, plate.center())
            painter.setBrush(QtGui.QColor("#FFFFFF"))
            painter.setPen(QtGui.QColor(DIVIDER))
            painter.drawRoundedRect(plate, scaled_px(3, minimum=2), scaled_px(3, minimum=2))
            painter.setPen(QtGui.QColor(TEXT))
            painter.setFont(self._small_font())
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
            radius_n = scaled_px(4, minimum=3) if self._is_device(node) else radius
            painter.drawRoundedRect(box, radius_n, radius_n)

            # A compact DEVICE leaf has no room for a two-line name+badge; its name IS its role (camera /
            # sequencer), so draw a single centred name and skip the badge line.
            if self._is_device(node):
                dev_font = self._small_font()
                painter.setPen(QtGui.QColor("#FFFFFF"))
                painter.setFont(dev_font)
                dev_rect = QtCore.QRectF(box.x() + scaled_px(5), box.y(),
                                         box.width() - scaled_px(10), box.height())
                painter.drawText(dev_rect, int(QtCore.Qt.AlignCenter),
                                 self._elide(str(node.get("name") or "device"), dev_rect.width(), dev_font))
                continue

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
