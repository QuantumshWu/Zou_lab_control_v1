"""#2: the SignalHub must not accumulate stale signals run-after-run, and a new same-kind node must not
clobber an earlier (stopped, lingering) node's signals.

- REMOVING a logic node purges ITS signals from the hub (``remove_signals``); STOPPING keeps them (a
  finished scan stays plottable / a panel can be wired before the next run).
- ``_logic_node_prefix`` disambiguates a new node's keys against EVERY live hub signal (running AND a
  stopped node's lingering ones), so the second same-kind node takes a distinct prefix instead of
  overwriting the first on the hub.
"""

from __future__ import annotations

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
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    return exp, console


def test_logic_node_prefix_disambiguates_against_a_lingering_hub_signal():
    """A stopped occupancy node's ``occupied`` lingers in the hub; adding a SECOND occupancy node must
    get a distinct (non-empty) prefix so it doesn't clobber the lingering signal -- even though NO node
    is currently running (the bug: prefix only checked running_nodes)."""
    from Zou_lab_control.frontend.task_console import LogicNodeConfig
    exp, console = _console()
    try:
        # simulate an earlier occupancy node that ran then STOPPED -> its 'occupied' lingers, no running node
        console.hub.publish({"occupied": np.zeros(6), "rate": 0.4})
        assert list(console.running_nodes) == [] or all(not getattr(n, "running", False) for n in console.running_nodes)
        cfg = LogicNodeConfig(kind="processor", name="Judge occupancy", title="Judge occupancy #2")
        prefix = console._logic_node_prefix(cfg)
        assert prefix != "", "second occupancy node must get a disambiguating prefix vs the lingering 'occupied'"
        assert (prefix + "occupied") not in console.hub.signal_versions()   # the prefixed name is free
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


def test_stop_then_remove_purges_a_lingering_nodes_signals():
    """The exact bug the human-flow run caught: STOP a node (its signals linger -> kept), then REMOVE it.
    Remove must purge those lingering signals even though the live node ref was already None'd at stop
    (the fix uses ``_last_node``, retained through stop, not ``_logic_nodes`` which is None'd)."""
    import time
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.timing import imaging_sequence
    exp, console = _console()
    try:
        # imaging pulse on -> the trigger-driven camera streams frames
        exp.devices.sequencer.prepare(imaging_sequence(exposure=exp.devices.camera.exposure, load=True).forever())
        exp.devices.sequencer.fire()
        cam = _add_node(console, lambda d: isinstance(d, tuple) and d and d[0] == "camera")
        console._start_logic_node(cam)
        for _ in range(8):
            console.refresh_once(); time.sleep(0.005)
        occ = _add_node(console, lambda d: isinstance(d, tuple) and d and "occupanc" in str(d[1]).lower())
        console._start_logic_node(occ)
        for _ in range(3):
            console.refresh_once(); time.sleep(0.005)
        node = console._logic_nodes.get(id(occ)) or console._last_node.get(id(occ))
        occ_sigs = {s for s in node.published_signals()} & set(console.hub.signal_versions())
        assert occ_sigs, "occupancy node published nothing to the hub"
        console._stop_logic_node(occ)                                  # STOP -> signals LINGER
        assert occ_sigs <= set(console.hub.signal_versions()), "STOP must keep the node's signals"
        console._remove_logic_node(occ)                               # REMOVE -> signals PURGED
        left = occ_sigs & set(console.hub.signal_versions())
        assert not left, f"REMOVE after STOP must purge the lingering signals, still present: {sorted(left)}"
    finally:
        console.shutdown()
        exp.close()
