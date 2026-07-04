"""Contract: a SAVED figure records the SYSTEMATIC upstream DAG of how its data was produced -- not a
flat, single-chain record but a BRANCHING tree (a panel consumes several signals, each from a different
upstream source) -- and the FigureViewer's Flow tab can build a layered node-link diagram from it.

The capture (``operations.figure_capture.capture_flow_graph``) is pure over the ``SignalHub`` +
``LogicNode`` contracts (no frontend / Qt / sim ground truth); it is folded into the save's
``info['provenance']['flow_graph']`` (an append-only key that never disturbs the flat ``provenance`` /
``signals`` structure).  These pin:

1. a MULTI-INPUT figure (a processor fusing TWO camera frames) round-trips a graph whose processor node
   has >= 2 PARENTS (branching), with one edge per consumed signal;
2. a processor CHAIN keeps the intermediate processor's identity (the old flat hoist DROPPED it): the graph
   has measurement -> processor -> plot, not a flattened measurement -> plot;
3. a RAW-DATA figure (a bare array, no ``bind_source``) degrades to a single ``raw data -> plot`` graph;
4. a device-holding SOURCE node is marked ``has_devices`` (its snapshot lives in ``provenance['devices']``);
5. the reusable ``FlowGraphView`` builds a layered layout from a graph (node/edge counts, plot in the last
   layer) and shows a placeholder for an absent graph -- never a crash.

Runs headless (``QT_QPA_PLATFORM=offscreen``); virtual sleeps are fast-forwarded.
"""

from __future__ import annotations

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
    OccupancyProcessor,
)
from Zou_lab_control.neutral_atom.operations.signal_expr import SignalExpr, DEFAULT_SOURCE  # noqa: E402
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


# --------------------------------------------------------------------------- 1: multi-input branching
def test_multi_input_figure_captures_a_branching_graph():
    """A processor that CONSUMES TWO camera frames (from two different measurements) -> its node has >= 2
    PARENTS in the flow graph, and the graph carries one edge per consumed signal.  This is the branching
    the old flat provenance discarded: the tree forks upward, each fork a different upstream source."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam_a = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                  prefix="a_", repeat=1)
        cam_b = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                  prefix="b_", repeat=1)
        # A processor whose source expression FUSES the two cameras' frames -> it consumes BOTH signals.
        proc = OccupancyProcessor(
            hub, source_expr=SignalExpr(["a_frame_0", "b_frame_0"], "value = (signal[0] + signal[1]) / 2"),
            prefix="occ_")
        assert set(proc.consumes) == {"a_frame_0", "b_frame_0"}
        fire_live_imaging(exp)
        for _ in range(2):
            cam_a.step(); cam_b.step()
        for m in (cam_a, cam_b):
            m.refresh_provenance()

        resolve = _resolver([cam_a, cam_b, proc])
        graph = capture_flow_graph(hub, proc, ["occ_occupied"], resolve_node=resolve)
        assert graph is not None
        by_id = _graph_by_role(graph)
        # exactly one plot terminal, one processor, two measurements (device leaves are an ADDITIVE #5a
        # expansion of each measurement, so assert the PRODUCING roles here, devices asserted separately)
        roles = sorted(n["role"] for n in graph["nodes"] if n["role"] != "device")
        assert roles == ["measurement", "measurement", "plot", "processor"], roles
        # the processor node has TWO parents (the two cameras) -- the branch
        proc_id = next(nid for nid, n in by_id.items() if n["role"] == "processor")
        assert len(_parents(graph, proc_id)) == 2, "the fusing processor forks up to BOTH cameras"
        # one edge per consumed signal, each labelled with the bare signal name it carries
        into_proc = [e for e in graph["edges"] if str(e["to"]) == proc_id]
        assert {e.get("signal") for e in into_proc} == {"frame_0", "frame_0"} or \
            {e.get("signal") for e in into_proc} == {"frame_0"}
        assert all(e.get("shape") for e in into_proc), "each upstream edge carries the block shape"
    finally:
        exp.close()
        plt.close("all")


# --------------------------------------------------------------------------- 2: processor chain kept
def test_processor_chain_keeps_intermediate_identity():
    """A camera -> occupancy processor -> plot chain keeps the PROCESSOR node in the middle: the graph has
    measurement -> processor -> plot (three producing roles), NOT a flattened measurement -> plot.  This is
    exactly what the old flat provenance dropped (it hoisted the camera devices past the processor)."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                prefix="cam_", repeat=1)
        proc = OccupancyProcessor(hub, source_expr=SignalExpr(["cam_frame_0"], DEFAULT_SOURCE),
                                  prefix="occ_")
        fire_live_imaging(exp)
        for _ in range(2):
            cam.step()
        cam.refresh_provenance()

        resolve = _resolver([cam, proc])
        graph = capture_flow_graph(hub, proc, ["occ_occupied"], resolve_node=resolve)
        by_id = _graph_by_role(graph)
        plot_id = next(nid for nid, n in by_id.items() if n["role"] == "plot")
        proc_id = next(nid for nid, n in by_id.items() if n["role"] == "processor")
        meas_id = next(nid for nid, n in by_id.items() if n["role"] == "measurement")
        # the intermediate processor is BETWEEN the measurement and the plot
        assert proc_id in _parents(graph, plot_id), "the processor feeds the plot"
        assert meas_id in _parents(graph, proc_id), "the measurement feeds the processor (chain kept)"
        assert meas_id not in _parents(graph, plot_id), "the measurement does NOT jump straight to the plot"
    finally:
        exp.close()
        plt.close("all")


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
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
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
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
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
def test_flow_graph_round_trips_through_the_console_save(tmp_path):
    """The console Save folds the flow graph into ``info['provenance']['flow_graph']``; reopened with
    ``load_figure`` it carries the branching processor node (>= 2 parents) -- the full multi-input DAG
    survives the npz round-trip alongside the flat provenance."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelConfig, PanelEditor, GAP)
    from Zou_lab_control.frontend import panel_plot

    ensure_qt_app()
    exp = na.connect("virtual")
    console = None
    try:
        hub = SignalHub()
        cam_a = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                  prefix="a_", repeat=1)
        cam_b = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                                  prefix="b_", repeat=1)
        proc = OccupancyProcessor(
            hub, source_expr=SignalExpr(["a_frame_0", "b_frame_0"], "value = (signal[0] + signal[1]) / 2"),
            prefix="occ_")
        fire_live_imaging(exp)
        for _ in range(2):
            cam_a.step(); cam_b.step()
        for m in (cam_a, cam_b):
            m.refresh_provenance()
        console = TaskConsole(hub=hub, state=default_console_state(), session=exp,
                              running_nodes=[cam_a, cam_b, proc], window_px=(900, 600))
        console._timer.stop()
        console.refresh_once = lambda: None
        card = console._new_panel_card(PanelConfig(kind="hist", title="occ", source="value = occ_occupied",
                                                   inputs=["occ_occupied"], row=GAP, col=GAP, size="2x4"))
        console._attach_card(card)
        card.plotter = panel_plot(np.random.default_rng(0).normal(0.5, 0.1, 200),
                                  kind="hist", size="2x4", bins=20, title="occ")
        editor = PanelEditor(card, console)
        editor.rebuild()
        editor.save_dir_edit.setText(str(tmp_path / "flow"))
        editor.save_autoname.setChecked(False)
        editor.save()
        npz = next(tmp_path.glob("flow*.npz"), None)
        assert npz is not None, f"the console Save must write an .npz (status: {editor.status.text()})"

        info = load_figure(npz).info
        flow = info["provenance"]["flow_graph"]
        assert isinstance(flow, dict) and flow["nodes"] and flow["edges"]
        # the PRODUCING roles survive the round-trip (device leaves are an additive #5a expansion)
        roles = sorted(n["role"] for n in flow["nodes"] if n["role"] != "device")
        assert roles == ["measurement", "measurement", "plot", "processor"], roles
        # the device leaves (camera + sequencer per measurement) survive the npz round-trip too
        assert any(n["role"] == "device" for n in flow["nodes"]), "device leaves round-trip through the npz"
        proc_id = next(n["id"] for n in flow["nodes"] if n["role"] == "processor")
        parents = {str(e["from"]) for e in flow["edges"] if str(e["to"]) == proc_id}
        assert len(parents) == 2, "the branching processor survives the npz round-trip with both parents"
    finally:
        if console is not None:
            console.shutdown()
        exp.close()
        plt.close("all")


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
