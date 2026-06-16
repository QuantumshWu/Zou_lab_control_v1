"""Contract: loading-rate monitoring is buildable FROM Add Panel, as COMPOSED nodes.

The design principle of task_console is free composition: from a blank console you Add
Panel -> build it yourself.  "Data processing: Loading readout" builds the loading
readout as SEPARATE producer nodes -- a calibrate Task + a camera Measurement (frame)
+ a DetectProcessor (real per-frame detect) -- NOT one monolithic feed fabricating
every signal.  It is non-blocking (calibration runs on the task's own thread; the
detector no-ops until ready) and streams rate / occupied / centers / frame to the hub,
so loading-rate / occupancy / site monitoring is composed entirely in the GUI.

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


def test_add_panel_live_loading_builds_composed_nodes_and_streams():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.feeds import (
        CalibrateReadoutTask, CameraFrameFeed, DetectProcessor)

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp)
    console._timer.stop()
    try:
        # the loading readout is offered in Add Panel (a session was passed) as the
        # one-click COMPOSITE -- camera measurement + detect processor + calibrate task.
        kc = console.kind_combo
        idx = next(j for j in range(kc.count()) if kc.itemData(j) == ("live", "loading"))
        assert kc.itemText(idx) == "Readout: Loading (camera+detect+calibrate)"

        kc.setCurrentIndex(idx)
        console._add_panel()

        # the loading readout was built as THREE separate composed nodes -- a calibrate
        # task + a camera measurement + a detect processor -- not one monolithic feed.
        kinds = {type(f) for f in console.feeds}
        assert CalibrateReadoutTask in kinds
        assert CameraFrameFeed in kinds
        assert DetectProcessor in kinds
        # a "Loading rate" monitor reading value = rate landed
        assert any(c.config.title == "Loading rate" and c.config.source == "value = rate"
                   for c in console.cards)

        # the composed chain streams: camera -> frame -> REAL detect -> rate/occupied/
        # centers, so loading-rate + occupancy + site monitoring are all composable here
        deadline = time.monotonic() + 12.0
        while "occupied" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        names = set(console.hub.names())
        assert {"rate", "occupied", "centers", "rate_sites", "frame"} <= names
        assert np.ndim(console.hub.latest("rate")) == 0          # a scalar loading rate
        assert np.asarray(console.hub.latest("centers")).shape == (12, 2)   # 3x4 detected sites

        # a dedicated MID-RUN output panel shows the calibrate task's template frames
        # under the cal_ namespace (confocal-task style), distinct from the live frame
        assert any(c.config.source == "value = cal_frame" for c in console.cards)
        assert "cal_frame" in console.hub.names() and "cal_frame" != "frame"

        # the signal FLOW (who -> who -> who) is traceable: the DetectProcessor
        # consumes the camera's frame, so its producer chain is camera THEN detect
        # (the upstream camera measurement precedes the detect processor).
        detect = next(f for f in console.feeds if type(f) is DetectProcessor)
        camera = next(f for f in console.feeds if type(f) is CameraFrameFeed)
        chain = console._producer_chain(detect)
        assert chain[-1] == console._producer_label(detect)
        assert console._producer_label(camera) in chain
        assert chain.index(console._producer_label(camera)) < chain.index(console._producer_label(detect))
    finally:
        console.shutdown()
