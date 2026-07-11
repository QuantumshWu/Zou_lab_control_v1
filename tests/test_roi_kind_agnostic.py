"""Contract: ROI is generic over EVERY plot kind (issue #4).

* ``region_binding`` (the frontend's per-kind resolver) is the exact INVERSE of ``coerce_panel_value``:
  a full-frame ROI reduce equals the reduce of the whole array the panel shows -- for image, 1-D,
  distribution, site-map AND rolling-trace kinds.  This pins the two so they can never drift.
* ``kind_supports_roi`` offers ROI for every data-array kind and only excludes the pulse timeline and
  the facet grid -- the ONE source the Setting's Analysis combo reads.
* The Setting popup actually shows the "ROI" action for the acceptance kinds (a 1-D panel AND a
  distribution panel), proving the combo item the visual acceptance grabs.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.live import (
    PLOT_KINDS, ROI_KINDS, coerce_panel_value, kind_supports_roi, region_binding)
from Zou_lab_control.neutral_atom.core.selection import Selection, encode_region
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema, SignalTensor
from Zou_lab_control.neutral_atom.operations.processors.roi import RoiProcessor


def _roi_value(block, bound):
    """Run a mean RoiProcessor once over ``block`` with the bound Selection published as its region
    signal (the console's real drag path), and return its scalar."""
    hub = SignalHub()
    hub.register_signal("s", SignalSchema(
        point_shape=(block.shape[1],), data_shape=block.shape[2:], dtype=block.dtype,
        repeat_capacity=block.shape[0]))
    hub.publish({"s": block})
    values, metadata = encode_region(bound)
    rows = int(values.shape[0])
    rschema = SignalSchema(point_shape=(1,), data_shape=(rows, 2), dtype=np.float64,
                           repeat_capacity=1, label="region", metadata=metadata)
    hub.register_signal("s_region", rschema)
    hub.publish({"s_region": SignalTensor(values.reshape(1, 1, rows, 2), rschema)})
    node = RoiProcessor(hub, source_expr={"inputs": ["s"], "source": "value = signal"},
                        region="s_region", reduce="mean")
    hub.publish({"s": block})
    node.step()
    return float(hub.latest("roi_value")[0, 0, 0])


@pytest.mark.parametrize("kind", sorted(ROI_KINDS))
def test_region_binding_round_trips_coerce(kind):
    """A full-frame ROI reduce == the reduce of the whole array the panel shows: region_binding binds
    to the SAME block axes coerce_panel_value flattens, for every ROI-capable kind."""
    rng = np.random.default_rng(3)
    if kind == "2d":
        block = rng.normal(50, 5, size=(1, 1, 8, 10))
        structure = {"points_shape": [1], "data_shape": [8, 10]}
        selection = Selection.rectangle(0, 10, 0, 8)
        coords = {}
    elif kind == "1d":
        block = rng.normal(50, 5, size=(1, 12, 1))           # a 12-point scan curve
        structure = {"points_shape": [12], "data_shape": [1]}
        selection = Selection.x(0, 12)
        coords = {"x": np.arange(12, dtype=float)}
    elif kind == "hist":
        block = rng.normal(50, 5, size=(1, 1, 40))           # 40 samples
        structure = {"points_shape": [1], "data_shape": [40]}
        selection = Selection.value(float(block.min() - 1), float(block.max() + 1))
        coords = {}
    elif kind == "sites":
        block = rng.uniform(0, 1, size=(1, 1, 6))            # 6-site vector
        structure = {"points_shape": [1], "data_shape": [6]}
        cx = np.arange(6, dtype=float) * 10.0
        cy = np.zeros(6)
        selection = Selection.rectangle(-5, 100, -5, 5)
        coords = {"x": cx, "y": cy}
    else:  # monitor: a scalar per shot
        block = rng.normal(50, 5, size=(1, 1, 1))
        structure = {"points_shape": [1], "data_shape": [1]}
        selection = Selection.x(0, 2)
        coords = {}

    bound = region_binding(kind, selection, structure=structure, coordinates=coords)
    roi_value = _roi_value(block, bound)
    # coerce_panel_value takes the repeat-reduced (P,*data) slice (R=1 here).
    shown = coerce_panel_value(kind, block[0], structure=structure)
    assert abs(roi_value - float(np.nanmean(np.asarray(shown, dtype=float)))) < 1e-9


def test_kind_supports_roi_covers_every_panel_kind():
    """The ONE ROI-availability source: every data-array kind offers ROI; only the pulse timeline and
    the facet grid (which has no single frame) opt out -- checked over EVERY registered plot kind."""
    assert ROI_KINDS == {"2d", "sites", "1d", "monitor", "hist"}
    for pk in PLOT_KINDS:
        expected = pk.key in ROI_KINDS
        assert kind_supports_roi(pk.key) is expected, pk.key
    assert not kind_supports_roi("pulse") and not kind_supports_roi("grid")


def _console_with_signals():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state

    ensure_qt_app()
    hub = SignalHub()
    rng = np.random.default_rng(7)
    hub.publish({"vals": rng.normal(100.0, 5.0, 400)})
    console = TaskConsole(hub=hub, state=default_console_state(), window_px=(1100, 700))
    console._timer.stop()
    return console


def _add_card(console, kind, signal):
    kc = console.kind_combo
    idx = next(j for j in range(kc.count()) if kc.itemData(j) == kind)
    kc.setCurrentIndex(idx)
    console._add_panel()
    card = console.cards[-1]
    card.config.inputs = [signal]
    card.config.source = "value = signal"
    card._compiled_source = card.config.source
    console.refresh_once()
    return card


@pytest.mark.parametrize("kind", ["1d", "hist"])
def test_roi_action_offered_in_setting_for_supported_kinds(kind):
    """The Setting popup's Analysis combo carries the 'ROI' action for a 1-D panel AND a distribution
    panel -- the acceptance kinds the visual grab shows (no longer gated to 2-D images)."""
    console = _console_with_signals()
    try:
        card = _add_card(console, kind, "vals")
        card._build_settings()
        actions = [card.analysis_combo.itemData(i) for i in range(card.analysis_combo.count())]
        assert "roi" in actions, f"{kind} Setting must offer the ROI action; got {actions}"
    finally:
        console.shutdown()
