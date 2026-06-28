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
