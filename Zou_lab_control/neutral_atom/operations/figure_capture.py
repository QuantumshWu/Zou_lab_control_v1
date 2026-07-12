"""Frontend-neutral capture of the RICH ``info`` a saved figure needs -- the ONE source of the
"save" logic, shared by the notebook (``na.save_figure``) and the GUI (``PanelEditor.save``).

A saved ``.npz`` carries more than ``data_x`` / ``data_y``: to re-BUILD every panel kind losslessly
(a site map needs its 2-D underlay frame + per-site centres, which a flat scatter dropped) and to
record "what the apparatus was doing when this data was taken", the save folds two blocks into
``info``:

  * ``info['signals']`` -- the canonical Hub tensors the panel consumes, keyed by bare name, each with
    its registered :class:`SignalSchema` metadata and a ROLE (value / x / centers / frame);
  * ``info['provenance']`` -- a FLAT device-state record of the SOURCE that produced the data (the
    ``devices`` snapshots + acquisition params always at the top level, whether the panel is wired
    straight to a measurement or to a processor of one).

Both were originally only reachable from the Qt ``PanelEditor``.  They live HERE now -- pure over the
:class:`~..core.signals.SignalHub` + :class:`~.logic.LogicNode` contracts, importing NO frontend / Qt --
so notebook and GUI produce byte-identical rich npz through ONE implementation (``na.save_figure`` and
``PanelEditor.save`` both call these).  Device state is read ONLY through the public ``.snapshot()``
contract (no simulation ground truth), so the module stays inside the sealed analysis layer
(``test_virtual_equals_real_contract`` / ``test_figure_capture_frontend_neutral``).

Resolving a signal name to its producing node is the caller's job (a ``resolve_node`` callable): the
hub stores only ``{name: value}`` and does not know which node published a name.  The GUI passes
``console._node_for_signal``; the notebook passes a resolver over its running nodes.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np

#: A resolver ``signal_name -> producing LogicNode (or None)``.  The hub does not track producers, so
#: the caller supplies this (the console's ``_node_for_signal``, or a notebook resolver over its nodes).
ResolveNode = Callable[[str], object]


def _schema_shapes(schema) -> tuple[tuple, tuple, str, str]:
    """Serialize the schema carried by one atomic Hub tensor snapshot.

    The Hub is the normalization boundary for both declared producer outputs and raw external values.
    In particular, an external auxiliary array is registered as one point whose full ndarray shape is
    ``data_shape``.  Reading the schema from the same :class:`SignalTensor` as the block keeps save and
    load faithful across schema epochs without a second registry read or any ndarray-rank guess.
    """
    return (tuple(schema.point_shape), tuple(schema.data_shape),
            str(schema.label or ""), str(schema.unit or ""))


def capture_figure_signals(hub, node, inputs) -> dict[str, dict]:
    """The RAW native hub blocks a panel consumes, keyed by bare name -- folded into a save's
    ``info['signals']`` so ``load_figure`` can REBUILD every panel kind losslessly.

    A site map is the reason this exists: its 2-D underlay FRAME (``frame_judged``) and per-site
    ``centers`` are NOT recoverable from the flat ``data_x`` / ``data_y`` a scatter saves, so without
    them a reloaded site map has nothing to draw.

    ``node`` is the ONE producing node of the panel's first input (resolved by the caller, e.g.
    ``console._node_for_signal(inputs[0])``); ``inputs`` is the panel's wired signal list.  Roles are
    read off that node:

      * ``value``   = the panel's own signal (``inputs[0]``);
      * ``x``       = its companion x-axis signal for a 1-D curve (``node.x_signal``);
      * ``centers`` = the node's site-centre output (``prefix + sitemap_centers_key``);
      * ``frame``   = the node's underlay-image output (``prefix + sitemap_image_key``).

    Each role stores one atomic ``hub.latest_tensor`` snapshot: its CURRENT canonical block
    (``(R,P,*data_shape)``) plus that same tensor's registered ``point_shape`` / ``data_shape`` /
    ``label`` / ``unit``.  A missing
    role (no such signal / not on the hub / no producing node) is simply SKIPPED -- a save never fails
    for a role it cannot resolve, and only the ``signals`` key is added (``data_x`` / ``data_y``
    unchanged).  Returns ``{}`` when there is no wired input or no producing node."""
    wired = [str(s) for s in (inputs or []) if s]
    if not wired or node is None:
        return {}
    prefix = str(getattr(node, "prefix", "") or "")
    centers_key = str(getattr(node, "sitemap_centers_key", "") or "")
    image_key = str(getattr(node, "sitemap_image_key", "") or "")
    # (role, full hub name) -- only the roles that resolve to a name; the rest never enter the loop.
    roles: list[tuple[str, str]] = [("value", wired[0])]
    x_full = str(getattr(node, "x_signal", "") or "")
    if x_full:
        roles.append(("x", x_full))
    if centers_key:
        roles.append(("centers", prefix + centers_key))
    if image_key:
        roles.append(("frame", prefix + image_key))

    signals: dict[str, dict] = {}
    for role, full in roles:
        if not full:
            continue
        try:
            tensor = hub.latest_tensor(full)                # one atomic data + schema snapshot
        except Exception:
            continue                                        # signal not on the hub right now -> skip role
        block = tensor.data
        bare = full[len(prefix):] if prefix and full.startswith(prefix) else full
        # One SIGNAL can fill two roles (a 2-D camera panel whose ``value`` IS the underlay ``frame``:
        # inputs[0] == sitemap_image_key).  ``roles`` lists ``value`` FIRST, so the first writer wins --
        # keep ``value`` (the panel's own data, what the reloaded plot draws) and skip the later
        # same-signal ``frame`` rather than clobbering it to role="frame".
        if bare in signals:
            continue
        ps, ds, label, unit = _schema_shapes(tensor.schema)
        signals[bare] = {
            "block": np.asarray(block),
            "points_shape": ps,
            "data_shape": ds,
            "label": label,
            "unit": unit,
            "role": role,
        }
    return signals


def _hoist_upstream_devices(node, resolve_node, *, visited: set) -> dict:
    """Walk a DERIVED node's consumed inputs to the nearest node that HOLDS devices and return that
    node's ``{devices, acquisition_parameters}`` (only the keys it has) to hoist to the top level -- so a
    processed panel's flat provenance carries the source apparatus state directly, not buried in a
    per-signal ``upstream`` sub-dict.  The walk is transitive (a processor of a processor chains up to the
    measurement) and visited-guarded (a set of ``id(node)`` -- no cycle / no double-visit); the FIRST
    device-holding node found wins (breadth-first over each level's consumed signals)."""
    consumes = list(getattr(node, "consumes", ()) or [])
    deferred: list = []
    for sig in consumes:
        src = resolve_node(sig)
        if src is None or id(src) in visited:
            continue
        visited.add(id(src))
        try:
            src_prov = dict(src.provenance_snapshot())
        except Exception:
            continue
        if src_prov.get("devices"):
            # This upstream node HOLDS the devices: hoist its device state + acquisition params.
            hoist = {"devices": src_prov["devices"]}
            if src_prov.get("acquisition_parameters"):
                hoist["acquisition_parameters"] = src_prov["acquisition_parameters"]
            return hoist
        deferred.append(src)                                # a processor-of-a-processor: chase next level
    for src in deferred:                                    # transitive: recurse into the deeper processors
        deeper = _hoist_upstream_devices(src, resolve_node, visited=visited)
        if deeper:
            return deeper
    return {}


#: The synthetic id/role of the terminal PLOT node in a flow graph -- the figure itself, which every
#: wired input flows INTO (it is not a LogicNode, so it has no producing node to resolve).
_PLOT_ID = "__plot__"
#: The synthetic id/role of a RAW-DATA leaf -- a figure whose data was handed in directly (a bare array,
#: no ``bind_source``), so there is no producing node and the flow degrades to ``raw data -> plot``.
_RAW_ID = "__raw__"
#: Prefix for a synthetic DEVICE leaf id (``<prefix><source_id>::<device>``) -- the apparatus a
#: device-holding source expands up into.  Scoped by the source id so two measurements each holding a
#: ``camera`` get DISTINCT camera leaves (no cross-source collision).
_DEVICE_ID_PREFIX = "__dev__"


def raw_data_flow_graph() -> dict:
    """The minimal flow graph a figure with NO producing node degrades to: a single ``raw data`` leaf into
    the ``plot`` terminal (``raw data -> plot``).  The SINGLE source of that fallback -- both
    :func:`capture_flow_graph` (a bare-array figure) and the viewer's load-time synthesis (an old npz that
    predates ``flow_graph``, or a capture that failed) build it from HERE, so the Flow tab NEVER shows a
    "no data-flow" placeholder: there is always at least this tree to draw."""
    return {
        "nodes": [
            {"id": _PLOT_ID, "name": "figure", "role": "plot"},
            {"id": _RAW_ID, "name": "raw data", "role": "raw data"},
        ],
        "edges": [{"from": _RAW_ID, "to": _PLOT_ID}],
    }


def _node_id(node) -> str:
    """A stable per-instance id for a producing node in the flow graph.  ``id()`` is unique within one
    capture (the whole graph is built in a single call), so two same-kind nodes (two occupancy judges)
    never collide, and a node reached along two different paths (a diamond: one source feeding two
    processors that both feed the plot) maps to the SAME graph node -- so convergence is preserved, not
    duplicated."""
    return f"n{id(node)}"


def _node_role(node) -> str:
    """The flow-graph ROLE of a producing node = its architecture LAYER (``measurement`` / ``processor``
    / ``task``), read straight off ``node.layer`` -- structure-driven, never a per-kind special-case.  A
    node without a layer (should not happen) degrades to ``node``."""
    return str(getattr(node, "layer", "") or "node")


def _bare_name(node, full: str) -> str:
    """A signal's bare name (its published name minus the producing node's prefix) for a readable edge
    label -- so an edge reads ``frame_0`` not ``cam_frame_0``.  Falls back to the full name when the
    prefix does not apply."""
    prefix = str(getattr(node, "prefix", "") or "")
    return full[len(prefix):] if prefix and full.startswith(prefix) else full


def _plot_input_roles(node, inputs) -> list[tuple[str, str]]:
    """The (role, full-signal-name) edges that flow straight INTO the plot -- the SAME role resolution
    ``capture_figure_signals`` uses (value / x / centers / frame), so the flow graph's terminal edges name
    exactly the signals the figure consumes.  ``value`` is always ``inputs[0]``; the companion x / the
    sitemap centres+underlay are read off the producing ``node`` (its ``x_signal`` / ``sitemap_*_key``),
    exactly as the signal capture does.  A role whose signal cannot be resolved is simply omitted."""
    wired = [str(s) for s in (inputs or []) if s]
    if not wired:
        return []
    roles: list[tuple[str, str]] = [("value", wired[0])]
    x_full = str(getattr(node, "x_signal", "") or "")
    if x_full:
        roles.append(("x", x_full))
    prefix = str(getattr(node, "prefix", "") or "")
    centers_key = str(getattr(node, "sitemap_centers_key", "") or "")
    image_key = str(getattr(node, "sitemap_image_key", "") or "")
    if centers_key:
        roles.append(("centers", prefix + centers_key))
    if image_key:
        roles.append(("frame", prefix + image_key))
    # De-dup a signal that fills two roles (a 2-D camera panel whose value IS the underlay frame): keep
    # the FIRST (value) exactly like capture_figure_signals, so the plot has one edge per distinct signal.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for role, full in roles:
        if not full or full in seen:
            continue
        seen.add(full)
        out.append((role, full))
    return out


def _signal_shape(hub, node, full: str) -> Optional[list]:
    """The shape of a signal's current hub block as a plain list (or ``None`` when it is not on the hub /
    the shape cannot be read) -- so an edge can carry ``shape`` without re-storing the whole array (the raw
    blocks already live in ``info['signals']``).  Read through the public ``hub.latest`` contract only."""
    if hub is None:
        return None
    try:
        block = hub.latest(full)
    except Exception:
        return None
    try:
        return [int(n) for n in np.shape(block)]
    except Exception:
        return None


def capture_flow_graph(hub, node, inputs, *, resolve_node: ResolveNode | None = None) -> dict:
    """The SYSTEMATIC upstream DAG of how a figure's data was produced -- ``{"nodes": [...], "edges": [...]}``
    -- so a reopened figure can DRAW "where did this data come from": raw data? a measurement signal? through
    which processor(s)?  and, crucially, the tree BRANCHES UPWARD (a site map consumes occupancy + centres +
    an underlay frame, each along its OWN chain), which the old flat provenance discarded.

    Built by RECURSIVELY expanding every input, driven purely by the node graph (never a per-kind
    special-case):

      * the terminal ``plot`` node is the figure itself; each wired input (value / x / centres / frame,
        resolved exactly as :func:`capture_figure_signals`) is an EDGE from its producing node INTO the
        plot, labelled with the signal name + its current shape;
      * each producing node is then expanded via its ``consumes`` list -- every consumed signal resolves
        (through ``resolve_node``) to a FURTHER upstream node and adds an edge ``upstream -> node``.  A node
        with several consumed inputs therefore has several PARENTS (branching); a source reached along two
        paths maps to ONE graph node (convergence).  The walk is ``id``-guarded so a cycle / a re-visit
        never loops or double-adds;
      * a SOURCE node that holds devices is marked ``has_devices`` -- the viewer shows that its device
        snapshot provenance is attached (the snapshots themselves stay in ``provenance['devices']``, not
        re-stored here).

    Roles come from each node's architecture ``layer`` (``measurement`` / ``processor`` / ``task``) plus the
    synthetic ``plot`` terminal, the ``device`` LEAVES a device-holding source expands into (one per held
    device -- ``camera`` / ``sequencer`` -- as the MOST-upstream nodes: an edge ``device -> source`` so the
    flow is traced all the way UP to the apparatus that produced the raw frames, and the existing
    longest-path layering drops them naturally in the TOP layer), and, when there is no producing node at
    all (a bare-array figure), a single ``raw data`` leaf -> ``plot``.  ALWAYS returns a graph (at minimum
    ``raw data -> plot`` -- even a plain ``plot(arr).save()`` with no inputs and no node), so the Flow tab
    always has something to draw.  ``resolve_node`` is the upstream resolver (``console._node_for_signal`` /
    a notebook resolver); without it the graph still captures the DIRECT plot inputs, just not the deeper
    chain."""
    wired = [str(s) for s in (inputs or []) if s]

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def _ensure_node(n) -> str:
        nid = _node_id(n)
        if nid not in nodes:
            entry: dict = {"id": nid, "name": str(getattr(n, "display_label", "") or "node"),
                           "role": _node_role(n)}
            # A source that HOLDS devices carries its snapshot provenance (stored flat in
            # provenance['devices']); mark it so the viewer can badge it, WITHOUT re-storing the snapshot.
            try:
                prov = dict(n.provenance_snapshot())
            except Exception:
                prov = {}
            if prov.get("devices"):
                entry["has_devices"] = True
                entry["devices"] = sorted(str(k) for k in prov["devices"])
            nodes[nid] = entry
            _expand_devices(nid, entry)                       # trace up to the apparatus (device leaves)
        return nid

    def _expand_devices(source_id: str, entry: dict) -> None:
        """Trace a device-holding SOURCE up to the apparatus: expand each held device (``camera`` /
        ``sequencer``) into its OWN most-upstream leaf node (``role="device"``) and add an edge
        ``device -> source``.  The device is the true origin of the raw frames the source published, so the
        edge points DOWN into the source and the existing longest-path layering places the device in the TOP
        layer -- one level (no TTL/DAC sub-tree).  The device's snapshot itself stays flat in
        ``provenance['devices']`` (not re-stored here); this only adds the structural leaf."""
        for dev in entry.get("devices", []):
            dev_id = f"{_DEVICE_ID_PREFIX}{source_id}::{dev}"
            if dev_id not in nodes:
                nodes[dev_id] = {"id": dev_id, "name": str(dev), "role": "device"}
            _add_edge(dev_id, source_id, signal=str(dev), role="device")

    def _add_edge(frm: str, to: str, *, signal: str = "", shape=None, role: str = "") -> None:
        edge: dict = {"from": frm, "to": to}
        if signal:
            edge["signal"] = signal
        if shape is not None:
            edge["shape"] = shape
        if role:
            edge["role"] = role
        edges.append(edge)

    def _expand(n, *, visited: set) -> None:
        """Walk a node's consumed inputs upstream, adding one node + edge per consumed signal (branching
        preserved: several consumes -> several parents)."""
        nid = _node_id(n)
        consumes = list(getattr(n, "consumes", ()) or [])
        for sig in consumes:
            sig = str(sig)
            src = resolve_node(sig) if resolve_node is not None else None
            if src is None:
                continue
            src_id = _ensure_node(src)
            _add_edge(src_id, nid, signal=_bare_name(src, sig), shape=_signal_shape(hub, src, sig))
            if id(src) in visited:
                continue                                    # convergence / cycle -> edge kept, no re-expand
            visited.add(id(src))
            _expand(src, visited=visited)

    # The terminal plot node.
    nodes[_PLOT_ID] = {"id": _PLOT_ID, "name": "figure", "role": "plot"}

    if node is None:
        # No producing node: the figure's data was handed in directly -> the raw-data->plot fallback (the
        # ONE source, so the console Save, a notebook plot(arr).save(), and the viewer's load-time synthesis
        # all draw the identical tree).
        return raw_data_flow_graph()

    # The DIRECT edges INTO the plot: each wired role's signal -> its producing node -> the plot.  The
    # producing node of the panel's own value is ``node`` (already resolved by the caller); the companion
    # roles (x / centres / frame) are published by that SAME node, so they share it.
    node_id = _ensure_node(node)
    for role, full in _plot_input_roles(node, inputs):
        _add_edge(node_id, _PLOT_ID, signal=_bare_name(node, full),
                  shape=_signal_shape(hub, node, full), role=role)
    # Expand every producing node upstream (branching + convergence preserved).
    _expand(node, visited={id(node)})
    return {"nodes": list(nodes.values()), "edges": edges}


def capture_figure_provenance(node, *, resolve_node: ResolveNode | None = None,
                              session=None) -> Optional[dict]:
    """The FLAT device-state record of the SOURCE that produced a panel's data -- folded into a save's
    ``info['provenance']`` so a reopened figure shows "what the apparatus was doing when this data was
    taken".

    The record is FLAT and UNIFORM regardless of what kind of node produced the data: the ``devices``
    that produced the plotted data (``camera`` + ``sequencer`` snapshots) and their
    ``acquisition_parameters`` are ALWAYS at the top level.

      * a MEASUREMENT node holds its own devices -> they are already top-level;
      * a DERIVED node (a processor: no device of its own, only a ``consumes`` list) is walked UPSTREAM
        via ``resolve_node`` -- each consumed signal -> its producing node -> ... transitively -- to the
        nearest node that DOES hold devices, and THAT node's ``devices`` + ``acquisition_parameters`` are
        HOISTED to the top level.  So a site-map panel wired to ``Judge occupancy`` and a 2-D panel wired
        straight to the camera both end up with ``provenance['devices']`` carrying camera + sequencer.

    The processor's OWN identity keys (``node`` / ``layer`` / ``consumes`` / ``calibration_fingerprint``)
    stay at the top level too.  ``node`` is the producing node (already resolved by the caller);
    ``resolve_node`` is only needed for the upstream walk (skip it and a processor's devices simply stay
    empty).  When no node produces the signal (a derived expression, a loaded static figure) it falls
    back to the whole-session immutable ``session.device_catalog`` if a session is given, else
    ``None`` -- so a save NEVER fails for lack of provenance.  Only public observation state is read
    (no adapter, SDK handle, command capability, or simulation ground truth)."""
    if node is not None:
        try:
            prov = dict(node.provenance_snapshot())
        except Exception:
            prov = None
        if prov is not None:
            # If this node holds no devices of its own (a processor), hoist the nearest device-holding
            # upstream node's devices + acquisition params to the top level, so the record is flat and
            # identical to a measurement's -- no nested ``upstream`` structure.
            if not prov.get("devices") and resolve_node is not None:
                hoisted = _hoist_upstream_devices(node, resolve_node, visited={id(node)})
                if hoisted:
                    prov.update(hoisted)                    # top-level devices + acquisition_parameters
            return prov
    catalog = getattr(session, "device_catalog", None)
    to_dict = getattr(catalog, "to_dict", None)
    if callable(to_dict):
        try:
            return {"devices": to_dict()}
        except Exception:
            return None
    return None


def capture_rich_info(source: Optional[Mapping]) -> dict:
    """The COMPLETE rich blocks a figure save folds into ``info`` -- ``signals`` + ``provenance``
    (the latter always carrying ``provenance['flow_graph']``) -- captured from the ONE unified
    figure-source mapping every save surface stamps (``BaseLivePlot.bind_source`` /
    ``DataFigure.bind_source`` / the grid's ``_GridData.bind_source``):

      * ``hub``          -- the :class:`SignalHub` the data was published to;
      * ``node``         -- the producing logic node of the figure's first input;
      * ``inputs``       -- the figure's wired signal names;
      * ``resolve_node`` -- the ``signal name -> producing node`` resolver for the upstream walk;
      * ``session``      -- the fallback device-snapshot source when no node produced the data.

    Missing keys (and ``source=None`` -- an unbound bare-array figure) degrade exactly as each
    capture documents: no ``signals`` block, no device ``provenance``, and the flow graph reduced to
    the ``raw data -> plot`` fallback (:func:`capture_flow_graph` builds it itself when ``node`` is
    ``None``), so even a plain ``plot(arr).save()`` gives the Flow tab a tree to draw.

    This is the SINGLE composition point: the flat ``DataFigure._rich_capture`` and the grid's
    ``_GridData._rich_capture`` both delegate here, so the three captures are never hand-wired twice
    (a second hand-rolled copy once mis-called :func:`capture_flow_graph` and -- behind a blanket
    ``except`` -- silently saved every grid npz with NO flow graph).  There is deliberately NO
    blanket ``except`` around the captures: each one soft-fails INTERNALLY by its own documented
    contract (an unresolvable role is skipped, provenance degrades to ``None``, the flow graph
    always returns at least ``raw data -> plot``), so an exception escaping is a CONTRACT DRIFT
    that must surface, not be swallowed into a silently poorer save."""
    src = dict(source) if source else {}
    hub = src.get("hub")
    node = src.get("node")
    inputs = src.get("inputs")
    resolve_node = src.get("resolve_node")

    out: dict = {}
    signals = capture_figure_signals(hub, node, inputs)
    if signals:
        out["signals"] = signals
    prov = capture_figure_provenance(node, resolve_node=resolve_node, session=src.get("session"))
    prov = dict(prov) if prov else {}
    prov["flow_graph"] = capture_flow_graph(hub, node, inputs, resolve_node=resolve_node)
    out["provenance"] = prov
    return out


__all__ = ["capture_figure_signals", "capture_figure_provenance", "capture_flow_graph",
           "capture_rich_info", "raw_data_flow_graph", "ResolveNode"]
