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
    assert describe_shape(np.zeros(35)) == "(35)"              # ONE '×' spelling, never the numpy comma
    assert describe_shape(np.zeros((35, 2))) == "(35×2)"
    assert describe_shape(np.zeros((96, 128))) == "(96×128)"
    assert describe_shape(np.zeros((7, 3))) == "(7×3)"         # an arbitrary shape, no table
    assert describe_shape(None) == "—"                         # no value yet


def test_contract_shape_label_is_the_single_grammar():
    """The canonical ``R × P × (data)`` spelling lives in ONE place (contract_shape_label), and
    describe_shape's value-driven contract branch delegates to it -- so a value-present render and a
    schema-only render of the same logical shape can never diverge (issue #12)."""
    from Zou_lab_control.neutral_atom.operations.logic import contract_shape_label, describe_shape

    assert contract_shape_label(1, (1,), (1200, 1920)) == "1 × 1 × (1200×1920)"
    assert contract_shape_label(1, (5, 5, 5), (1,)) == "1 × 5×5×5 × (1)"
    assert contract_shape_label(5, (4, 5), (1,), (4, 5)) == "5 × 4×5 × (1)"
    # a real canonical block routes through the SAME label
    assert describe_shape(np.zeros((1, 1, 1200, 1920)), points_shape=(1,), data_shape=(1200, 1920)) \
        == contract_shape_label(1, (1,), (1200, 1920))


def test_task_and_hub_signals_of_same_shape_render_identically():
    """#12: a TASK output (calibration / mot-field) and a measurement/processor signal of the same
    logical shape render the IDENTICAL ``R × P × (data)`` string -- never the raw one-outer-paren
    ``(R×P×data)`` the task legend used to emit when it forgot the declared schema.  The value-less
    declared form (before the first publish) equals the value-present form."""
    pytest.importorskip("PyQt5")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub, SignalSchema
    ensure_qt_app()

    console = TaskConsole(hub=SignalHub(), state=default_console_state())
    console._timer.stop()
    try:
        # mot-field 3-D scan grid and cali single-shot frame, as their TaskOutput declares them
        grid = SignalSchema(point_shape=(5, 5, 5), data_shape=(1,), dtype=np.float64, repeat_capacity=1)
        frame = SignalSchema(point_shape=(1,), data_shape=(1200, 1920), dtype=np.float64, repeat_capacity=1)
        assert console._describe_from_schema(None, grid) == "1 × 5×5×5 × (1)"
        assert console._describe_from_schema(np.zeros((1, 125, 1)), grid) == "1 × 5×5×5 × (1)"
        assert console._describe_from_schema(None, frame) == "1 × 1 × (1200×1920)"
        assert console._describe_from_schema(np.zeros((1, 1, 1200, 1920)), frame) == "1 × 1 × (1200×1920)"
        # a value with no schema still falls back to the raw spelling (only path that keeps outer paren)
        assert console._describe_from_schema(np.zeros((1, 125, 1)), None) == "(1×125×1)"
    finally:
        console.shutdown()


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
        for name in published:
            schema = hub.schema(name)
            assert fmts[name] == describe_shape(
                hub.latest(name),
                points_shape=schema.point_shape,
                data_shape=schema.data_shape,
            ), name
        # An unregistered external matrix is deterministically one datum; rank
        # never invents repeat or point axes.
        assert fmts["odd_blob"] == "1 × 1 × (7×3)"
    finally:
        console.shutdown()
