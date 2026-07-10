"""Contract: when acquisition (now uncapped) outruns the display, the console raises a NON-INTRUSIVE
amber heads-up on its existing summary banner -- never a modal, and it never touches the run.

Two halves:
* na-side -- the SignalHub exposes the primitive a consumer needs (``history_len`` + the ``shot`` counter)
  so a display can PROVE its ring dropped shots: publishing more than the ring depth between two reads
  leaves only ``history_len`` in the ring.
* console-side -- ``_update_summary`` turns that overrun into an amber advisory on the persistent
  status strip (severity ``"warning"``) and clears it once the display catches up.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.signals import SignalHub


def test_hub_history_len_and_shot_delta_detect_ring_drop():
    """The detection primitive is consumer-side arithmetic on public hub state: shots published since the
    last read, minus the ring depth.  Publishing 20 into an 8-deep ring keeps only 8 -> a display that last
    read at shot 0 can compute that 12 shots rolled off before it could see them."""
    hub = SignalHub(history=8)
    assert hub.history_len == 8
    for i in range(20):
        hub.publish({"v": float(i)})
    assert hub.shot == 20
    assert len(hub.history("v")) == 8                   # the ring physically kept only the last 8
    assert (hub.shot - 0) - hub.history_len == 12       # exactly what the console's _note_display_drops computes


def test_hub_per_signal_history_policy_preserves_native_image_dtype():
    """Image streams are not scalar histories: a producer can declare a compact ring, and the hub stores
    the camera's native dtype instead of inflating each frame to float64."""
    import numpy as np

    hub = SignalHub(history=16)
    hub.configure_signal("img", history=3)
    for i in range(5):
        hub.publish({"img": np.full((4, 5), i, dtype=np.uint16)})

    assert hub.history_limit("img") == 3
    assert hub.history_len == 3                         # conservative minimum active ring
    hist = hub.history("img")
    assert hist.shape == (3, 1, 1, 4, 5)               # updates, R, P, H, W
    assert hist.dtype == np.uint16
    assert int(hist[-1, 0, 0, 0, 0]) == 4

    hub.remove_signals(["img"])
    assert hub.history_limit("img") == 16               # lifecycle cleanup removes the compact policy too


@pytest.fixture
def offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def test_console_warns_amber_when_display_falls_behind_then_clears(offscreen):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    hub = SignalHub(history=8)
    console = TaskConsole(hub=hub, state=default_console_state(), running_nodes=[])
    console._timer.stop()
    try:
        console._update_summary()                        # establish the baseline read (shot 0)
        assert "signals" in console.summary.text() and "shot 0" in console.summary.text()
        assert console.status_strip.text() == ""         # idle status is not a telemetry duplicate
        for i in range(40):                              # a burst FAR exceeding the 8-deep ring
            hub.publish({"v": float(i)})
        console._update_summary()                        # the display "reads" again -> detects the overrun
        assert console.status_strip.severity == "warning"
        assert "display behind" in console.status_strip.text()
        assert "acquisition unaffected" in console.status_strip.text()  # the run is never implicated
        assert "1 signals" in console.summary.text() and "shot 40" in console.summary.text()

        console._update_summary()                        # no new shots since -> display caught up -> clears
        assert console.status_strip.severity != "warning"
        assert console.status_strip.text() == ""
    finally:
        console.shutdown()
