"""MECHANICAL guard for the decoupling core (#4): opening the console must NOT
auto-start ANY acquisition.  A freshly built console -- even with a fully calibrated
session and the whole measurement / processor / task catalog wired in -- has an
EMPTY hub, no running nodes and no plot cards.  Signals appear ONLY after the user
adds a logic node and Starts it; a plot is a blank view until then.

This pins the bug the user hit ("一个 measurement 都没加进来怎么就有一堆 signal"):
building a plot / opening the dashboard never silently spins up a measurement.

Offscreen Qt + virtual backend (the same contract path as real hardware).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def test_console_with_full_catalog_opens_idle_and_publishes_nothing():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    hub = SignalHub()
    console = TaskConsole(
        hub=hub, state=default_console_state(), session=exp,
        measurements=exp.readout.measurement_specs(),
        processors=exp.readout.processor_specs(),
        tasks=exp.readout.task_specs(), window_px=(1000, 700))
    console._timer.stop()
    try:
        # nothing acquires on its own: empty hub, nothing running, no panels, no logic rows
        assert hub.names() == []
        assert console.running_nodes == []
        assert console.cards == []
        assert console.logic_nodes == []
        # a few refresh ticks must NOT conjure a node or a signal
        for _ in range(3):
            console.refresh_once()
        assert hub.names() == []
        assert console.running_nodes == []
    finally:
        console.shutdown()
        exp.close()
