"""#2: the SignalHub must not accumulate stale signals run-after-run, and a new same-kind node must not
clobber an earlier (stopped, lingering) node's signals.

- REMOVING a logic node purges ITS signals from the hub (``remove_signals``); STOPPING keeps them (a
  finished scan stays plottable / a panel can be wired before the next run).
- ``_logic_node_prefix`` disambiguates a new node's keys against EVERY live hub signal (running AND a
  stopped node's lingering ones), so the second same-kind node takes a distinct prefix instead of
  overwriting the first on the hub.
"""

from __future__ import annotations

from conftest import fire_live_imaging, make_console, raw_device_set

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.signals import SignalHub  # noqa: E402


def test_remove_signals_purges_keeps_others_and_is_idempotent():
    hub = SignalHub()
    hub.publish({"occupied": np.zeros(35), "rate": 0.5, "keepme": 1.0})
    v0 = hub.version
    removed = hub.remove_signals(["occupied", "rate", "absent"])
    assert set(removed) == {"occupied", "rate"}            # only the present ones
    assert "keepme" in hub.names() and "occupied" not in hub.names()
    assert hub.version > v0                                 # consumers told to refresh
    assert hub.remove_signals(["occupied"]) == []          # idempotent: already gone, no-op
    assert "occupied" not in hub.signal_versions()         # version counter dropped too


def _console():
    import pytest
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = make_console(exp)
    return exp, console


def test_logic_node_prefix_disambiguates_against_a_lingering_hub_signal():
    """A stopped processor's output lingers; a second instance must disambiguate it."""
    from Zou_lab_control.frontend.task_console import LogicNodeConfig
    exp, console = _console()
    try:
        spec = next(item for item in console.processors if item.name == "Analysis")
        cfg = LogicNodeConfig(kind="processor", name=spec.name, title="Analysis #2")
        key = console._node_bare_keys(cfg)[0]
        # Simulate an earlier processor that stopped after publishing its base name.
        console.hub.publish({key: np.zeros(6)})
        assert list(console.running_nodes) == [] or all(not getattr(n, "running", False) for n in console.running_nodes)
        prefix = console._logic_node_prefix(cfg)
        assert prefix != "", "second processor must disambiguate a lingering output"
        assert (prefix + key) not in console.hub.signal_versions()
    finally:
        console.shutdown()
        exp.close()


def test_restart_reclaims_its_own_lingering_prefix_not_a_new_one():
    """#issue-1: restarting the SAME node must REUSE its (empty) prefix -- reclaiming its OWN lingering
    signals -- not take a fresh numbered prefix.  Otherwise after Stop->Start its signals are
    renamed and every panel bound to 'occupied' / the sitemap goes UNBOUND.  A DIFFERENT same-kind node
    still disambiguates (the H3y behaviour is preserved)."""
    from types import SimpleNamespace
    from Zou_lab_control.frontend.task_console import LogicNodeConfig
    exp, console = _console()
    try:
        spec = next(item for item in console.processors if item.name == "Analysis")
        cfg = LogicNodeConfig(kind="processor", name=spec.name, title="Analysis")
        keys = console._node_bare_keys(cfg)
        assert console._logic_node_prefix(cfg) == ""                 # first start: no collision, short names
        # It ran then stopped: its output lingers and its built node survives in _last_node (tagged
        # with instance_label == the node title, as _start_logic_node does).
        console.hub.publish({key: np.zeros(6) for key in keys})
        console._last_node[id(cfg)] = SimpleNamespace(
            instance_label="Analysis", published_signals=lambda: list(keys))
        # RESTART the SAME node -> must reclaim "" (its own lingering signals are not a collision)
        assert console._logic_node_prefix(cfg) == "", "restart must reuse its prefix, not rename -> unbound"
        # A different same-kind node still disambiguates against the lingering outputs.
        other = LogicNodeConfig(kind="processor", name=spec.name, title="Analysis #2")
        assert console._logic_node_prefix(other) != ""
    finally:
        console.shutdown()
        exp.close()


def test_declared_signal_name_equals_the_name_the_node_publishes():
    """#prebind: a Plot wired to a measurement's 'survival' BEFORE it starts must keep working once the
    node publishes. That holds iff the DECLARED picker name == the PUBLISHED hub name. A measurement
    node publishes under 'f{spec.key}_' (temperature_survival), so the declared key the picker offers
    (and a binding / a saved layout stores) must be that SAME prefixed name -- never the bare
    'survival', which never re-resolves once the producer starts under its real name."""
    from types import SimpleNamespace
    from Zou_lab_control.frontend.task_console import LogicNodeConfig, strip_node_prefix
    exp, console = _console()
    try:
        spec = next(s for s in exp.readout.measurement_specs() if getattr(s, "key", "") == "temperature")
        row = SimpleNamespace(node=LogicNodeConfig(kind="measurement", name=spec.name, title=spec.name))
        declared = console._declared_signal_keys(row)
        pfx = console._declared_node_prefix(row)
        # declared == the FULL names a ScannedMeasurementNode(prefix=f"{spec.key}_") publishes
        assert pfx == f"{spec.key}_"
        assert set(declared) == {f"{spec.key}_{spec.x_key}", f"{spec.key}_{spec.y_key}"}
        assert "temperature_survival" in declared, declared
        # the Logic-row legend still shows the SHORT name (prefix stripped), like the running path
        assert [strip_node_prefix(k, pfx) for k in declared] == [spec.x_key, spec.y_key]
        # A processor keeps its empty default prefix.
        proc = SimpleNamespace(node=LogicNodeConfig(kind="processor", name="Analysis", title="Analysis"))
        assert console._declared_node_prefix(proc) == ""
    finally:
        console.shutdown()
        exp.close()


def _add_node(console, match):
    kc = console.kind_combo
    for i in range(kc.count()):
        data = kc.itemData(i)
        if match(data):
            kc.setCurrentIndex(i); console._add_panel()
            return console.logic_nodes[-1]
    raise AssertionError(f"no kind matched in {[kc.itemData(i) for i in range(kc.count())]}")


def test_live_switch_that_drops_outputs_purges_orphans_without_a_rebuild():
    """The gap the eager purges miss (#5 unbound): a LIVE acquisition edit that SHRINKS a running node's
    output set -- ``frames_per_cycle`` 3 -> 1 on the camera -- happens with NO rebuild and NO remove, so
    neither _start_logic_node nor _remove_logic_node runs.  The systematic GC in _refresh_signal_info
    catches it: frame_1/frame_2 lose their producer (the camera now publishes only frame_0) -> they are
    purged from the hub so they never linger as '(unbound)' picker entries."""
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement
    exp, console = _console()
    try:
        hub = console.hub
        cam = CameraMeasurement(hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer,
                                frames_per_cycle=3, repeat=0)
        console.running_nodes = [cam]
        fire_live_imaging(exp)
        cam.step()                                              # frame_0/1/2 published
        assert {"frame_0", "frame_1", "frame_2"} <= set(hub.names())
        console._refresh_signal_info()                          # cache the fpc=3 provider signature

        cam.set_acquisition_parameters(frames_per_cycle=1)      # LIVE switch: now publishes only frame_0
        assert set(cam.published_signals()) == {"frame_0"}
        console._refresh_signal_info()                          # provider map shrank -> GC fires
        assert "frame_0" in hub.names()                         # the kept output stays
        assert "frame_1" not in hub.names() and "frame_2" not in hub.names()   # orphans purged
    finally:
        console.shutdown()
        exp.close()


def test_orphan_gc_purges_every_unowned_signal_with_no_exception():
    """The orphan GC removes EVERY hub signal no live producer publishes -- with no special-case
    exemption.  (A node error is surfaced via instance attrs + the banner, never as a hub signal, so
    there is no reserved 'node_error' channel to keep; this pins that the GC has no carve-out left.)"""
    import numpy as np
    exp, console = _console()
    try:
        console.hub.publish({"stray": 2.0, "occupied": np.zeros(6)})   # neither has a live producer
        console.running_nodes = []
        console._signal_info_sig = None                        # force the GC to run this call
        console._refresh_signal_info()
        assert console.hub.names() == []                       # both orphans purged, no exception
    finally:
        console.shutdown()
        exp.close()


def test_stop_then_remove_purges_a_lingering_nodes_signals():
    """The exact bug the human-flow run caught: STOP a node (its signals linger -> kept), then REMOVE it.
    Remove must purge those lingering signals even though the live node ref was already None'd at stop
    (the fix uses ``_last_node``, retained through stop, not ``_logic_nodes`` which is None'd)."""
    import time
    exp, console = _console()
    try:
        # imaging pulse on -> the trigger-driven camera streams frames
        fire_live_imaging(exp)
        cam = _add_node(console, lambda d: isinstance(d, tuple) and d and d[0] == "camera")
        console._start_logic_node(cam)
        deadline = time.monotonic() + 2.0
        while "frame_0" not in console.hub.names() and time.monotonic() < deadline:
            console.refresh_once()
            time.sleep(0.005)
        assert "frame_0" in console.hub.names(), "camera did not publish before the lifecycle deadline"
        node = console._logic_nodes.get(id(cam)) or console._last_node.get(id(cam))
        camera_signals = set(node.published_signals()) & set(console.hub.signal_versions())
        assert camera_signals, "camera node published nothing to the hub"
        console._stop_logic_node(cam)                                  # STOP -> signals LINGER
        assert camera_signals <= set(console.hub.signal_versions()), "STOP must keep the node's signals"
        console._remove_logic_node(cam)                                # REMOVE -> signals PURGED
        left = camera_signals & set(console.hub.signal_versions())
        assert not left, f"REMOVE after STOP must purge the lingering signals, still present: {sorted(left)}"
    finally:
        console.shutdown()
        exp.close()
