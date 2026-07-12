"""DESIGN: under a producer node in the signal-picker NEST, the leaf is the SHORT signal name (the
node's prefix stripped: ``frame``, ``survival``, ``rate``) -- NEVER the full prefixed key and NEVER the
verbose SignalSpec axis label (e.g. ``camera image``).  ONE rule (``strip_node_prefix``) shared by the
Logic tab and the picker.  Regression: the picker used to prefer the axis label, so a camera frame showed
as ``camera image`` instead of ``frame``.
"""

from __future__ import annotations

from conftest import raw_device_set

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_strip_node_prefix_rule():
    from Zou_lab_control.frontend.task_console import strip_node_prefix
    assert strip_node_prefix("temperature_survival", "temperature_") == "survival"
    assert strip_node_prefix("judge_occupancy_rate", "judge_occupancy_") == "rate"
    assert strip_node_prefix("frame", "") == "frame"            # no prefix -> unchanged
    assert strip_node_prefix("frame_0", "") == "frame_0"        # camera per-trigger stays short+meaningful


def test_camera_nest_leaf_is_short_name_not_axis_label():
    """A running camera's picker nest shows ``frame`` / ``frame_0`` (short names), not ``camera image``
    (the SignalSpec axis label).  Driven through the real console short-name provider + tree builder."""
    import pytest
    pytest.importorskip("PyQt5")
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.timing import imaging_sequence
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.frontend.param_widgets import signal_tree_groups
    ensure_qt_app()

    exp = na.connect("virtual")
    console = None
    try:
        seq = imaging_sequence(exposure=raw_device_set(exp).camera.exposure, load=True, name="live").forever()
        raw_device_set(exp).sequencer.prepare(seq); raw_device_set(exp).sequencer.fire()   # On Pulse so the camera streams
        console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                              measurements=exp.readout.measurement_specs(), window_px=(900, 600))
        console._timer.stop()
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("camera", "live"))
        kc.setCurrentIndex(i); console._add_panel()
        console._start_logic_node(console.logic_nodes[-1])
        for _ in range(8):
            console.refresh_once(); time.sleep(0.01)

        short = console._signal_short_names()
        assert short.get("frame_0") == "frame_0"     # camera publishes per-event frame_i (no lumped "frame")
        groups = signal_tree_groups(sorted(short), console._signal_providers(),
                                    console._signal_formats(), short)
        leaves = [lbl for _prod, lvs in groups for (lbl, _bare, _full) in lvs]
        assert any(l.startswith("frame") for l in leaves), leaves
        assert not any("camera image" in l for l in leaves), leaves   # the verbose axis label must NOT leak
    finally:
        if console is not None:
            console.shutdown()
        exp.close()
