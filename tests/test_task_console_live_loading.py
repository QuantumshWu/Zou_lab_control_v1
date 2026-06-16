"""Contract: loading-rate monitoring is buildable FROM Add Panel (FRONTEND).

The design principle of task_console is free composition: from a blank console you
Add Panel -> build it yourself. The continuous loading-rate producer (LoadingFeed)
was previously only wirable in the notebook; now passing the session makes it an
Add-Panel producer ("Live: Loading readout") that streams rate / occupied / centers
/ frame to the hub, so you compose loading-rate / occupancy / site monitoring
entirely in the GUI. Building it must NOT block the GUI (lazy calibration on the
feed's own thread).

Offscreen Qt + virtual backend (same contract path as real hardware).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
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


def test_add_panel_live_loading_streams_rate_and_sites():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.feeds import LoadingFeed

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp)
    console._timer.stop()
    try:
        # the continuous producer is offered in Add Panel (a session was passed)
        kc = console.kind_combo
        idx = next(j for j in range(kc.count()) if kc.itemData(j) == ("live", "loading"))
        assert kc.itemText(idx) == "Live: Loading readout"

        kc.setCurrentIndex(idx)
        console._add_panel()

        # a LoadingFeed got built (lazily-calibrated: construction did NOT block) and
        # a "Loading rate" monitor reading value = rate landed
        feed = next(f for f in console.feeds if isinstance(f, LoadingFeed))
        assert not feed._calibrated                          # not calibrated on the GUI thread
        assert any(c.config.title == "Loading rate" and c.config.source == "value = rate"
                   for c in console.cards)

        # the feed streams: rate / occupied / centers / frame appear on the hub, so
        # loading-rate + occupancy + site monitoring are all composable from here
        deadline = time.monotonic() + 8.0
        while "rate" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.02)
        names = set(console.hub.names())
        assert {"rate", "occupied", "centers", "rate_sites", "frame"} <= names
        assert np.ndim(console.hub.latest("rate")) == 0      # a scalar loading rate
        assert np.asarray(console.hub.latest("centers")).shape == (12, 2)   # 3x4 detected sites
    finally:
        console.shutdown()
