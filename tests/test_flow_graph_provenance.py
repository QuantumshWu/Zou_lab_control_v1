"""Contracts for saved-figure flow provenance and its layered viewer.

The capture (``operations.figure_capture.capture_flow_graph``) is pure over the ``SignalHub`` +
``LogicNode`` contracts (no frontend / Qt / sim ground truth); it is folded into the save's
``info['provenance']['flow_graph']`` (an append-only key that never disturbs the flat ``provenance`` /
``signals`` structure).  The focused cases cover multi-input branching, processor
chain identity, save/load round-trip, raw-data fallback, device leaves, stable
layout, and the absent-graph placeholder.

Runs headless (``QT_QPA_PLATFORM=offscreen``); virtual sleeps are fast-forwarded.
"""

from __future__ import annotations

from conftest import raw_device_set

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")

import Zou_lab_control.neutral_atom as na  # noqa: E402
from Zou_lab_control.neutral_atom.core.signals import SignalHub  # noqa: E402
from Zou_lab_control.neutral_atom.operations.logic import (  # noqa: E402
    CameraMeasurement,
    LogicNode,
    Processor,
)
from Zou_lab_control.neutral_atom.operations.figure_capture import capture_flow_graph  # noqa: E402
from Zou_lab_control.frontend import load_figure  # noqa: E402
from conftest import fire_live_imaging  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _resolver(nodes):
    """A ``signal_name -> producing node`` resolver over a list of nodes -- the same shape the console's
    ``_node_for_signal`` provides, for a headless capture without a console."""
    def resolve(name):
        for n in nodes:
            try:
                if name in n.published_signals():
                    return n
            except Exception:
                continue
        return None
    return resolve


def _graph_by_role(graph):
    return {n["id"]: n for n in graph["nodes"]}


def _parents(graph, node_id):
    """The set of upstream node ids that have an edge INTO ``node_id`` (its parents)."""
    return {str(e["from"]) for e in graph["edges"] if str(e["to"]) == str(node_id)}


def _role_of(graph, node_id):
    return _graph_by_role(graph)[node_id]["role"]


class _ScalarSource(LogicNode):
    layer = "measurement"
    node_label = "scalar source"

    def shot(self):  # pragma: no cover - values are published atomically by the fixture
        raise NotImplementedError

    def _bare_published_signals(self):
        return frozenset({"value"})


class _FuseProcessor(Processor):
    provides = ("value",)

    def transform(self, inputs):
        values = [float(np.asarray(inputs[name]).reshape(-1)[0]) for name in self.consumes]
        return {"value": float(np.mean(values))}


def _branching_fixture():
    hub = SignalHub()
    left = _ScalarSource(hub, prefix="left_")
    right = _ScalarSource(hub, prefix="right_")
    fused = _FuseProcessor(
        hub, consumes=("left_value", "right_value"), prefix="fused_")
    shot = hub.next_source_shot()
    hub.publish({"left_value": 2.0, "right_value": 6.0}, provenance=shot)
    fused.step()
    assert hub.latest_provenance("fused_value") == shot
    return hub, left, right, fused


# --------------------------------------------------------------------------- 1: multi-input branching
def test_multi_input_processor_captures_a_branching_graph():
    hub, left, right, fused = _branching_fixture()
    resolve = _resolver([left, right, fused])
    graph = capture_flow_graph(hub, fused, ["fused_value"], resolve_node=resolve)
    by_id = _graph_by_role(graph)
    roles = sorted(node["role"] for node in graph["nodes"])
    assert roles == ["measurement", "measurement", "plot", "processor"]
    processor_id = next(node_id for node_id, node in by_id.items()
                        if node["role"] == "processor")
    assert len(_parents(graph, processor_id)) == 2
    incoming = [edge for edge in graph["edges"] if edge["to"] == processor_id]
    assert {edge["signal"] for edge in incoming} == {"value"}
    assert all(edge.get("shape") for edge in incoming)


# --------------------------------------------------------------------------- 2: processor chain kept
def test_processor_chain_keeps_each_intermediate_identity():
    hub, left, right, fused = _branching_fixture()
    scaled = _FuseProcessor(hub, consumes=("fused_value",), prefix="scaled_")
    # The downstream subscription starts after the first fused publication, so
    # publish one new coherent input transaction for it.
    shot = hub.next_source_shot()
    hub.publish({"left_value": 4.0, "right_value": 8.0}, provenance=shot)
    fused.step()
    scaled.step()

    resolve = _resolver([left, right, fused, scaled])
    graph = capture_flow_graph(hub, scaled, ["scaled_value"], resolve_node=resolve)
    processor_ids = [node["id"] for node in graph["nodes"] if node["role"] == "processor"]
    assert len(processor_ids) == 2
    plot_id = next(node["id"] for node in graph["nodes"] if node["role"] == "plot")
    terminal = next(node_id for node_id in processor_ids if node_id in _parents(graph, plot_id))
    intermediate = next(node_id for node_id in processor_ids if node_id != terminal)
    assert intermediate in _parents(graph, terminal)
    assert intermediate not in _parents(graph, plot_id)


# --------------------------------------------------------------------------- 3: raw-data degrade
def test_raw_data_figure_degrades_to_raw_leaf():
    """A figure whose data was handed in directly (no producing node) -> the graph degrades to a single
    ``raw data -> plot`` -- so the Flow tab ALWAYS has something to draw, even for a plain
    ``plot(arr).save()`` with NO inputs and NO node (#4: no early ``None`` return)."""
    # no inputs AND no node -- a bare plot(arr) -- still yields a raw-data leaf feeding the plot.
    graph = capture_flow_graph(None, None, [])
    assert graph is not None, "even a bare plot(arr) gets a raw-data -> plot tree"
    roles = sorted(n["role"] for n in graph["nodes"])
    assert roles == ["plot", "raw data"], roles
    assert _parents(graph, "__plot__") == {"__raw__"}

    # a bare plot with a wired input name but no producing node (resolve finds nothing) still yields a graph
    graph = capture_flow_graph(SignalHub(), None, ["mystery"], resolve_node=lambda name: None)
    roles = sorted(n["role"] for n in graph["nodes"])
    assert roles == ["plot", "raw data"], roles
    assert _parents(graph, "__plot__") == {"__raw__"}


def test_capture_rich_info_without_node_falls_back_to_raw_graph():
    """The ONE composition point every save routes through (``capture_rich_info``), given a source with
    NO producing node (an unbound / bare-array figure -- or no source at all), degrades exactly as the
    individual captures document: no ``signals`` block, and ``provenance`` reduced to the
    ``raw data -> plot`` flow graph (built by ``capture_flow_graph``'s own ``node=None`` path)."""
    from Zou_lab_control.neutral_atom.operations.figure_capture import (
        capture_rich_info, raw_data_flow_graph)

    for source in (None, {}, {"hub": SignalHub(), "node": None, "inputs": ["mystery"],
                              "resolve_node": lambda name: None, "session": None}):
        out = capture_rich_info(source)
        assert "signals" not in out, f"no producing node -> no signals block (source={source!r})"
        assert out["provenance"] == {"flow_graph": raw_data_flow_graph()}, \
            "provenance degrades to exactly the raw data -> plot fallback"


def test_bare_plot_save_folds_a_raw_flow_graph(tmp_path):
    """#4: a plain ``plot(arr).save()`` (no ``bind_source``) still writes ``provenance['flow_graph']`` with
    a ``raw data -> plot`` tree, so the reopened figure's Flow tab is NOT blank -- reproduced end-to-end
    through the real notebook save path (``DataFigure.save``), not just the capture in isolation."""
    from Zou_lab_control.frontend import plot as make_plot

    fig = make_plot(np.linspace(0, 1, 40), np.random.default_rng(1).normal(size=40))
    out = fig.to_data_figure()
    saved = out.save(str(tmp_path / "bare"))
    npz = saved["data"]
    info = load_figure(npz).info
    flow = info["provenance"]["flow_graph"]
    assert isinstance(flow, dict) and flow["nodes"] and flow["edges"]
    roles = sorted(n["role"] for n in flow["nodes"])
    assert roles == ["plot", "raw data"], roles
    assert {str(e["from"]) for e in flow["edges"] if str(e["to"]) == "__plot__"} == {"__raw__"}
    plt.close("all")


# --------------------------------------------------------------------------- 4: device source marked
def test_source_node_is_marked_as_holding_devices():
    """A camera measurement source node is flagged ``has_devices`` so the viewer can show its device
    snapshot is attached -- the snapshot itself stays in ``provenance['devices']``, not re-stored here."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer,
                                prefix="cam_", repeat=1)
        fire_live_imaging(exp)
        for _ in range(2):
            cam.step()
        cam.refresh_provenance()
        graph = capture_flow_graph(hub, cam, ["cam_frame_0"], resolve_node=_resolver([cam]))
        by_id = _graph_by_role(graph)
        cam_node = next(n for n in by_id.values() if n["role"] == "measurement")
        assert cam_node.get("has_devices") is True
        assert "camera" in cam_node.get("devices", []) and "sequencer" in cam_node.get("devices", [])
    finally:
        exp.close()
        plt.close("all")


def test_device_holding_source_expands_to_device_leaves():
    """#5a: a device-holding measurement is traced UP to the apparatus -- its held devices (camera +
    sequencer) become their OWN most-upstream ``device`` leaf nodes, each with an edge ``device ->
    measurement``, and (via the longest-path layering) they sit ABOVE the measurement.  One level only (no
    TTL/DAC sub-tree)."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer,
                                prefix="cam_", repeat=1)
        fire_live_imaging(exp)
        for _ in range(2):
            cam.step()
        cam.refresh_provenance()
        graph = capture_flow_graph(hub, cam, ["cam_frame_0"], resolve_node=_resolver([cam]))
        by_id = _graph_by_role(graph)
        # the two devices are their OWN leaf nodes, role=device, named for the apparatus
        devs = [n for n in graph["nodes"] if n["role"] == "device"]
        assert {n["name"] for n in devs} == {"camera", "sequencer"}, [n["name"] for n in devs]
        meas_id = next(nid for nid, n in by_id.items() if n["role"] == "measurement")
        # each device feeds the measurement (device -> measurement), and the measurement is their child
        dev_ids = {n["id"] for n in devs}
        assert _parents(graph, meas_id) == dev_ids, "both devices feed the measurement"
        # a device leaf is the MOST upstream: it has NO parent of its own (one level, no sub-tree)
        for did in dev_ids:
            assert _parents(graph, did) == set(), "a device leaf is a terminal source (no TTL/DAC below)"
    finally:
        exp.close()
        plt.close("all")


def test_flow_view_places_device_leaves_in_the_top_layer():
    """#5a (layout): the ``FlowGraphView`` drops the expanded ``device`` leaves in the TOP layer (smallest
    y), above the measurement that holds them -- the flow reads top (apparatus) -> bottom (figure)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.flow_graph_view import FlowGraphView

    ensure_qt_app()
    graph = {
        "nodes": [
            {"id": "cam0", "name": "camera", "role": "device"},
            {"id": "seq0", "name": "sequencer", "role": "device"},
            {"id": "m", "name": "camera A", "role": "measurement", "has_devices": True,
             "devices": ["camera", "sequencer"]},
            {"id": "__plot__", "name": "figure", "role": "plot"},
        ],
        "edges": [
            {"from": "cam0", "to": "m", "signal": "camera", "role": "device"},
            {"from": "seq0", "to": "m", "signal": "sequencer", "role": "device"},
            {"from": "m", "to": "__plot__", "signal": "frame_0", "shape": [1, 1, 96, 128], "role": "value"},
        ],
    }
    view = FlowGraphView()
    view.set_graph(graph)
    dev_tops = [view._boxes["cam0"].top(), view._boxes["seq0"].top()]
    assert dev_tops[0] == dev_tops[1], "the two device leaves share the top layer"
    assert dev_tops[0] < view._boxes["m"].top() < view._boxes["__plot__"].top(), \
        "device (apparatus, top) -> measurement -> plot (bottom)"
    # a device box is COMPACT: narrower than a standard producing node box
    assert view._boxes["cam0"].width() < view._boxes["m"].width()


def test_flow_view_labels_are_globally_non_overlapping():
    """#5b: on a BUSY multi-input graph (two device-expanded cameras -> two processors -> one plot, and
    several named signals fanning into that SAME plot) the ``FlowGraphView`` places its edge labels by an
    ITERATIVE ALL-PAIRS push-apart -- so ACROSS THE WHOLE GRAPH, checked by a full O(n²) sweep, (1) any two
    signal-name plates are mutually non-overlapping and (2) no plate sits on any node box.  The old greedy
    one-at-a-time-vs-already-placed pass did NOT guarantee this: a plate that dodged A could land where C
    then had to sit (a plate B avoided ending up under C), so ~13 labels near mid-height overlapped."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.flow_graph_view import FlowGraphView

    ensure_qt_app()
    # Two cameras, each expanded to its own camera + sequencer device leaves, feed two processors; both
    # processors AND a measurement then fan several named signals into the one plot -> ~13 labelled edges
    # crowd the layer gaps, exactly the case the greedy pass left overlapping.
    graph = {
        "nodes": [
            {"id": "cam0", "name": "camera", "role": "device"},
            {"id": "seq0", "name": "sequencer", "role": "device"},
            {"id": "cam1", "name": "camera", "role": "device"},
            {"id": "seq1", "name": "sequencer", "role": "device"},
            {"id": "mA", "name": "bright A", "role": "measurement", "has_devices": True,
             "devices": ["camera", "sequencer"]},
            {"id": "mB", "name": "bright B", "role": "measurement", "has_devices": True,
             "devices": ["camera", "sequencer"]},
            {"id": "occ", "name": "occupancy", "role": "processor"},
            {"id": "cal", "name": "calibrate", "role": "processor"},
            {"id": "__plot__", "name": "figure", "role": "plot"},
        ],
        "edges": [
            {"from": "cam0", "to": "mA", "signal": "camera", "role": "device"},
            {"from": "seq0", "to": "mA", "signal": "sequencer", "role": "device"},
            {"from": "cam1", "to": "mB", "signal": "camera", "role": "device"},
            {"from": "seq1", "to": "mB", "signal": "sequencer", "role": "device"},
            {"from": "mA", "to": "occ", "signal": "frame_0", "shape": [1, 1, 96, 128]},
            {"from": "mB", "to": "occ", "signal": "frame_1", "shape": [1, 1, 96, 128]},
            {"from": "mA", "to": "cal", "signal": "frame_0", "shape": [1, 1, 96, 128]},
            {"from": "mB", "to": "cal", "signal": "frame_1", "shape": [1, 1, 96, 128]},
            {"from": "occ", "to": "__plot__", "signal": "occupied_sites", "shape": [1, 1, 35], "role": "value"},
            {"from": "occ", "to": "__plot__", "signal": "survival_prob", "shape": [1, 1, 35], "role": "value"},
            {"from": "cal", "to": "__plot__", "signal": "centers_map", "shape": [35, 2], "role": "centers"},
            {"from": "cal", "to": "__plot__", "signal": "thresholds_map", "shape": [1, 1, 35], "role": "value"},
            {"from": "mA", "to": "__plot__", "signal": "underlay_frame", "shape": [1, 1, 96, 128],
             "role": "frame"},
        ],
    }
    view = FlowGraphView()
    view.set_graph(graph)
    rects = view.label_rects()
    assert len(rects) >= 13, f"one plate per labelled edge (got {len(rects)})"
    # (1) every PAIR of label plates is disjoint -- the core #5b guarantee, over the FULL O(n²) sweep.
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects[i].intersects(rects[j]), \
                f"label plates {i} and {j} overlap: {rects[i]} vs {rects[j]}"
    # (2) NO plate sits on ANY node box -- a signal name never lands on a device / measurement / processor /
    # plot box, checked against every laid-out node.
    boxes = list(view._boxes.values())
    assert len(boxes) == 9, f"every node laid out (got {len(boxes)})"
    for k, r in enumerate(rects):
        for nid, b in view._boxes.items():
            assert not r.intersects(b), f"label plate {k} sits on node {nid}: {r} vs {b}"
    # (3) nothing is clipped: every plate lies inside the reported content canvas (no-cutoff half of #5b).
    cw, ch = float(view.sizeHint().width()), float(view.sizeHint().height())
    for k, r in enumerate(rects):
        assert r.left() >= -0.5 and r.top() >= -0.5 and r.right() <= cw + 0.5 and r.bottom() <= ch + 0.5, \
            f"label plate {k} is clipped by the {cw:.0f}x{ch:.0f} canvas: {r}"


# --------------------------------------------------------------------------- end-to-end save/load
def test_branching_flow_graph_round_trips_through_figure_save(tmp_path):
    from Zou_lab_control.frontend import panel_plot

    hub, left, right, fused = _branching_fixture()
    resolve = _resolver([left, right, fused])
    plot = panel_plot(np.array([4.0]), kind="hist", size="2x4", bins=4, title="fused")
    try:
        plot.bind_source(
            hub, fused, inputs=["fused_value"], resolve_node=resolve, session=None)
        path = plot.save(str(tmp_path / "branching_flow"))["data"]
        graph = load_figure(path).info["provenance"]["flow_graph"]
        roles = sorted(node["role"] for node in graph["nodes"])
        assert roles == ["measurement", "measurement", "plot", "processor"]
        processor_id = next(node["id"] for node in graph["nodes"]
                            if node["role"] == "processor")
        assert len(_parents(graph, processor_id)) == 2
    finally:
        plt.close(plot.fig)




# --------------------------------------------------------------------------- 5: the Flow widget
def test_flow_view_builds_layered_layout_and_tolerates_absence():
    """The reusable ``FlowGraphView`` lays out a graph (every node placed, the plot in the LAST layer) and
    shows a muted placeholder (no boxes) for an absent / malformed graph -- never a crash."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.flow_graph_view import FlowGraphView

    ensure_qt_app()
    # a hand-built branching graph: two sources -> a processor -> the plot
    graph = {
        "nodes": [
            {"id": "s1", "name": "camera A", "role": "measurement", "has_devices": True,
             "devices": ["camera", "sequencer"]},
            {"id": "s2", "name": "camera B", "role": "measurement", "has_devices": True},
            {"id": "p", "name": "occupancy", "role": "processor"},
            {"id": "__plot__", "name": "figure", "role": "plot"},
        ],
        "edges": [
            {"from": "s1", "to": "p", "signal": "frame_0", "shape": [1, 1, 96, 128]},
            {"from": "s2", "to": "p", "signal": "frame_0", "shape": [1, 1, 96, 128]},
            {"from": "p", "to": "__plot__", "signal": "occupied", "shape": [1, 1, 35], "role": "value"},
        ],
    }
    view = FlowGraphView()
    view.set_graph(graph)
    assert set(view._boxes) == {"s1", "s2", "p", "__plot__"}, "every node is laid out"
    # the plot is the BOTTOM (last) layer -- its y is the largest of all boxes
    plot_y = view._boxes["__plot__"].top()
    assert all(plot_y >= b.top() for b in view._boxes.values()), "the plot sits at the bottom layer"
    # the two sources share the TOP layer (same y), above the processor
    assert view._boxes["s1"].top() == view._boxes["s2"].top()
    assert view._boxes["s1"].top() < view._boxes["p"].top() < plot_y
    # natural size grows to fit the content
    assert view.sizeHint().width() > 0 and view.sizeHint().height() > 0

    # absent graph -> no boxes, a placeholder, no crash
    view.set_graph(None)
    assert view._boxes == {}
    view.set_graph({"nodes": [], "edges": []})       # malformed / empty
    assert view._boxes == {}
