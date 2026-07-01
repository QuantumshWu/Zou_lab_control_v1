"""Frontend-neutral capture of the RICH ``info`` a saved figure needs -- the ONE source of the
"save" logic, shared by the notebook (``na.save_figure``) and the GUI (``PanelEditor.save``).

A saved ``.npz`` carries more than ``data_x`` / ``data_y``: to re-BUILD every panel kind losslessly
(a site map needs its 2-D underlay frame + per-site centres, which a flat scatter dropped) and to
record "what the apparatus was doing when this data was taken", the save folds two blocks into
``info``:

  * ``info['signals']`` -- the RAW native hub blocks the panel consumes, keyed by bare name, each with
    its :class:`SignalSpec` contract metadata and a ROLE (value / x / centers / frame);
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


def _spec_shapes(node, full: str) -> tuple[Optional[tuple], Optional[tuple], str, str]:
    """The ``(points_shape, data_shape, label, unit)`` a signal's :class:`SignalSpec` declares, read off
    its producing ``node`` -- so the reloader has the SAME contract metadata the live panel had.  Every
    lookup soft-fails (a node without ``signal_spec`` / a signal with no spec -> defaults), so a save is
    never blocked for a role whose metadata cannot be resolved."""
    spec = None
    try:
        spec = node.signal_spec(full)
    except Exception:
        spec = None
    ps = tuple(spec.points_shape) if (spec is not None and spec.points_shape is not None) else None
    ds = tuple(spec.data_shape) if (spec is not None and spec.data_shape is not None) else None
    label = str(getattr(spec, "label", "") or "") if spec is not None else ""
    unit = str(getattr(spec, "unit", "") or "") if spec is not None else ""
    return ps, ds, label, unit


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

    Each role stores the CURRENT native block (``hub.latest`` -- a copy of the raw, un-reshaped array)
    plus the :class:`SignalSpec`'s ``points_shape`` / ``data_shape`` / ``label`` / ``unit``.  A missing
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
            block = hub.latest(full)                        # raw native block (a copy), un-reshaped
        except Exception:
            continue                                        # signal not on the hub right now -> skip role
        bare = full[len(prefix):] if prefix and full.startswith(prefix) else full
        # One SIGNAL can fill two roles (a 2-D camera panel whose ``value`` IS the underlay ``frame``:
        # inputs[0] == sitemap_image_key).  ``roles`` lists ``value`` FIRST, so the first writer wins --
        # keep ``value`` (the panel's own data, what the reloaded plot draws) and skip the later
        # same-signal ``frame`` rather than clobbering it to role="frame".
        if bare in signals:
            continue
        ps, ds, label, unit = _spec_shapes(node, full)
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
    back to the whole-session device snapshot (``session.devices.snapshot()``) if a session is given, else
    ``None`` -- so a save NEVER fails for lack of provenance.  Only public ``.snapshot()`` state is read
    (no simulation ground truth)."""
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
    devices = getattr(session, "devices", None)
    snap = getattr(devices, "snapshot", None)
    if callable(snap):
        try:
            return snap()
        except Exception:
            return None
    return None


__all__ = ["capture_figure_signals", "capture_figure_provenance", "ResolveNode"]
