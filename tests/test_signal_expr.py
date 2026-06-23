"""Contract tests for the ONE multi-slot signal + value-expression evaluator.

``SignalExpr`` is the single source for the slot-packing rule (scalar for one input, list for
many) and the ``value = ...`` contract, shared by plot panels, the OccupancyProcessor source,
and PulseScanNode's y.  These guard that single definition so a future "source" field can never
re-roll the rule.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations.signal_expr import (
    DEFAULT_SOURCE,
    SignalExpr,
    hub_namespace,
    seed_source_for_slots,
)


def test_one_slot_signal_is_the_scalar_value():
    ns = {"a": np.array([1.0, 2.0, 3.0])}
    out = SignalExpr(["a"], "value = signal").evaluate(ns)
    assert np.array_equal(np.asarray(out), ns["a"])      # signal IS the single picked value


def test_many_slots_signal_is_a_list_combined_by_index():
    ns = {"a": np.array([3.0, 4.0]), "b": np.array([1.0, 1.0])}
    out = SignalExpr(["a", "b"], "value = signal[0] - signal[1]").evaluate(ns)
    assert np.array_equal(np.asarray(out), np.array([2.0, 3.0]))


def test_co_names_includes_referenced_names_and_picked_slots():
    expr = SignalExpr(["occupied"], "value = np.mean(signal) + rate")
    names = expr.co_names()
    assert "occupied" in names and "rate" in names       # picked slot + directly-named signal
    assert "signal" in names                              # the pseudo identifier is folded in


def test_missing_value_assignment_raises():
    with pytest.raises(ValueError, match="value"):
        SignalExpr(["a"], "signal + 1").evaluate({"a": np.array([1.0])})


def test_resolve_hook_rewrites_name_before_lookup():
    ns = {"frame": np.array([1.0]), "frame_judged": np.array([9.0])}
    out = SignalExpr(["frame"], "value = signal").evaluate(ns, resolve=lambda n: "frame_judged")
    assert np.array_equal(np.asarray(out), np.array([9.0]))   # the coherence rewrite took effect


def test_from_value_round_trips_the_gui_dict():
    expr = SignalExpr.from_value({"inputs": ["x", "y"], "source": "value = signal[0]"})
    assert expr.inputs == ["x", "y"] and expr.source == "value = signal[0]"
    assert expr.as_value() == {"inputs": ["x", "y"], "source": "value = signal[0]"}


def test_seed_source_for_slots_canonical_and_keeps_custom():
    assert seed_source_for_slots(1) == DEFAULT_SOURCE
    assert seed_source_for_slots(2) == "value = signal[0] - signal[1]"
    # a real custom expression is preserved when adding/removing slots
    assert seed_source_for_slots(2, "value = signal[0] * 2") == "value = signal[0] * 2"


def test_hub_namespace_exposes_signals_and_helpers():
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    hub = SignalHub()
    hub.publish({"rate": 0.5})
    ns = hub_namespace(hub)
    assert ns["rate"] == pytest.approx(0.5)
    for key in ("history", "latest", "names", "shot", "np", "math"):
        assert key in ns


def test_occupancy_source_default_signal_expr_equals_single_frame_path():
    """The OccupancyProcessor source is now a signal_expr; its default ({"inputs": ["frame"],
    "source": "value = signal"}) consumes the single ``frame`` signal exactly as the old single
    ``source="frame"`` did -- so the migration is behaviour-preserving."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    try:
        exp.readout.sitemap(method="box", frames=4, display=False)
        exp.readout.thresholds(frames=20, display=False)
        cal = exp.readout.require(thresholds=True)
        # default signal_expr: consumes the bare 'frame' signal
        occ = OccupancyProcessor(hub := SignalHub(), calibration=cal,
                                 source_expr={"inputs": ["frame"], "source": "value = signal"},
                                 grid_shape=(3, 4))
        assert occ.consumes == ("frame",)                # inputs drive what it reacts to
        # Acquire a real virtual frame through the SAME firing + camera contract as hardware,
        # publish it, and judge it -- no simulation ground truth read.
        from tests.conftest import fire_live_imaging
        fire_live_imaging(exp)
        img = exp.devices.camera.acquire(1, sequencer=exp.devices.sequencer)[0]
        hub.publish({"frame": np.asarray(img, dtype=float)})
        out = occ.step()
        assert "occupied" in out and "rate" in out        # it judged the consumed frame
    finally:
        exp.close()
