"""Contract: a signal's SHAPE is AUTO-EXTRACTED from its real published value, never a
hand-typed name->format map.

The Logic-tab row "publishes:" line and the Plot signal picker both show each signal's
shape.  Those shapes must be read off the actual array (``logic.describe_shape``) so they
can never drift from what a node really emits -- a hand-maintained ``{name: "per-site
(N,)"}`` dict silently lies the moment a node's output changes.  This pins:

  * ``describe_shape`` standardizes ANY value to a shape string (no name lookup);
  * the console's ``_signal_formats`` reflects the REAL hub array shapes, including a
    shape no hand-written table would have anticipated.

Offscreen Qt for the console half; the ``describe_shape`` half is pure.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def test_describe_shape_standardizes_straight_from_the_value():
    from Zou_lab_control.neutral_atom.operations.logic import describe_shape

    assert describe_shape(3.0) == "scalar"
    assert describe_shape(np.float64(1.0)) == "scalar"
    assert describe_shape(np.zeros(())) == "scalar"
    assert describe_shape(np.zeros(35)) == "(35,)"             # 1-D keeps the numpy comma
    assert describe_shape(np.zeros((35, 2))) == "(35, 2)"
    assert describe_shape(np.zeros((96, 128))) == "(96, 128)"
    assert describe_shape(np.zeros((7, 3))) == "(7, 3)"        # an arbitrary shape, no table
    assert describe_shape(None) == "—"                         # no value yet


def test_console_signal_formats_are_the_real_hub_array_shapes():
    pytest.importorskip("PyQt5")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import describe_shape
    ensure_qt_app()

    hub = SignalHub()
    # publish values of several shapes -- the formats must match these, not a name table
    published = {
        "occupied": np.zeros(35),
        "centers": np.zeros((35, 2)),
        "frame": np.zeros((96, 128)),
        "rate": 0.5,
        "odd_blob": np.zeros((7, 3)),     # a name no hand table would know about
    }
    hub.publish(published)
    console = TaskConsole(hub=hub, state=default_console_state())
    console._timer.stop()
    try:
        fmts = console._signal_formats()
        for name, value in published.items():
            assert fmts[name] == describe_shape(np.asarray(value)), name
        # the arbitrary-shape signal proves it is read from the VALUE, not a lookup
        assert fmts["odd_blob"] == "(7, 3)"
    finally:
        console.shutdown()
