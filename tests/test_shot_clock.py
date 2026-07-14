"""Focused contracts for coherent display-shot selection.

Snapshot lookup must align a fast source with a lagging derived producer and stay
cheap with deep rings.  Faulted producers stop constraining the board, a lone
camera uses its newest provenance, and signals without lineage do not constrain it.
"""

from __future__ import annotations

from conftest import raw_device_set

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub, NO_LINEAGE
from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, Processor

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


def _virtual_experiment(grid=(3, 4)):
    return na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (40, 50)})


class _LaggingMean(Processor):
    """Neutral derived producer used to exercise provenance lag."""

    provides = ("derived",)

    def transform(self, inputs):
        return {"derived": float(np.mean(np.asarray(inputs["frame_0"])))}


# ----------------------------------------------------------------------------- hub level
def test_snapshot_at_aligns_a_fast_source_with_a_lagging_processor():
    exp = _virtual_experiment()
    hub = SignalHub()
    cam = CameraMeasurement(
        hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=0)
    derived = _LaggingMean(hub, consumes=("frame_0",))
    try:
        fire_live_imaging(exp)
        first = cam.step()
        derived.step()
        cam.step()
        cam.step()

        source_shot = hub.latest_provenance("frame_0")
        derived_shot = hub.latest_provenance("derived")
        assert source_shot > derived_shot
        snap = hub.snapshot_at(min(source_shot, derived_shot))
        assert np.array_equal(snap["frame_0"], first["frame_0"])
        assert not np.array_equal(snap["frame_0"], hub.latest("frame_0"))
        assert np.array_equal(snap["derived"], hub.latest("derived"))
    finally:
        exp.close()

def test_snapshot_at_is_cheap_with_deep_no_lineage_rings():
    """Perf guard (#shot-clock froze the GUI): snapshot_at runs EVERY display tick, so it must be cheap
    even with full (2048-deep) rings.  The freeze was an O(n^2) scan -- a free-running NO_LINEAGE signal
    never early-broke, walking its whole ring via deque INDEXING (O(n) per element).  With several such
    signals at 10 Hz that wedged PyQt.  The fix skips NO_LINEAGE signals and walks via reversed iterators
    (O(1)/step).  A generous bound (the regressed cost is ~1000x over it) keeps this non-flaky."""
    import time
    hub = SignalHub()
    for _ in range(2100):                              # overflow the 2048 history -> full rings
        sid = hub.next_source_shot()
        hub.publish({"frame_0": np.zeros((1, 1, 8, 8))}, provenance=sid)
        hub.publish({"rate": 0.5, "counts": np.zeros(12), "centers": np.zeros((12, 2)),
                     "thresholds": np.zeros(12)}, provenance=None)   # 4 NO_LINEAGE deep rings
    target = hub.latest_provenance("frame_0") - 1      # an OLD shot: the held-back-camera worst case
    t = time.perf_counter()
    for _ in range(50):
        hub.snapshot_at(target)
    per_call_ms = (time.perf_counter() - t) / 50 * 1000
    assert per_call_ms < 20.0, f"snapshot_at too slow ({per_call_ms:.2f} ms) -- the O(n^2) ring scan regressed"


# ----------------------------------------------------------------------------- console helpers
@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _console(exp):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    console = TaskConsole(
        hub=SignalHub(), state=default_console_state(), session=exp,
        measurements=exp.readout.measurement_specs(), processors=exp.readout.processor_specs(),
        tasks=exp.readout.task_specs(), window_px=(1200, 800))
    console._timer.stop()
    return console


def _add_2d(console, inputs):
    """Add a 2-D plot card bound to ``inputs`` (kind is irrelevant to the shot clock -- it reads
    ``config.source`` + ``config.inputs`` via _card_reads)."""
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == "2d")
    kc.setCurrentIndex(i)
    console._add_panel()
    card = console.cards[-1]
    card.config.source = "value = signal"
    card.config.inputs = list(inputs)
    return card


# ----------------------------------------------------------------------------- console coherence
def test_console_display_shot_holds_source_to_a_lagging_processor():
    exp = _virtual_experiment()
    console = _console(exp)
    hub = console.hub
    cam = CameraMeasurement(
        hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=0)
    derived = _LaggingMean(hub, consumes=("frame_0",))
    try:
        _add_2d(console, ["frame_0"])
        _add_2d(console, ["derived"])
        console.running_nodes = [cam, derived]
        fire_live_imaging(exp)
        first = cam.step()
        derived.step()
        cam.step()
        cam.step()

        target = console._display_shot()
        assert target == hub.latest_provenance("derived")
        assert target < hub.latest_provenance("frame_0")
        namespace = console._expression_namespace()
        assert np.array_equal(namespace["frame_0"], first["frame_0"])
        assert np.array_equal(namespace["derived"], hub.latest("derived"))
    finally:
        console.shutdown()
        exp.close()


def test_display_shot_excludes_a_faulted_producers_frozen_signal():
    exp = _virtual_experiment()
    console = _console(exp)
    hub = console.hub
    cam = CameraMeasurement(
        hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=0)
    derived = _LaggingMean(hub, consumes=("frame_0",))
    try:
        _add_2d(console, ["frame_0"])
        _add_2d(console, ["derived"])
        console.running_nodes = [cam, derived]
        fire_live_imaging(exp)
        cam.step()
        derived.step()
        cam.step()
        cam.step()

        assert console._display_shot() == hub.latest_provenance("derived")
        derived.consecutive_errors = 1
        derived.last_error = "RuntimeError: failed transform"
        assert console._display_shot() == hub.latest_provenance("frame_0")
    finally:
        console.shutdown()
        exp.close()


# ----------------------------------------------------------------------------- anti-oscillation
def test_lone_camera_panel_is_not_held_back():
    """A LONE camera panel (no slower co-displayed producer) is NOT held back: its display shot is the
    camera's OWN latest, so a repeat / repeat_mode accumulation stays fully live.  This is the property
    the whole-board min must NOT break -- the hold appears only when a slower producer is shown too."""
    exp = _virtual_experiment()
    console = _console(exp)
    hub = console.hub
    cam = CameraMeasurement(hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=0)
    try:
        _add_2d(console, ["frame_0"])
        console.running_nodes = [cam]
        fire_live_imaging(exp)
        cam.step(); cam.step(); cam.step()       # frames advance S1..S3 with nothing lagging

        assert console._display_shot() == hub.latest_provenance("frame_0")   # its OWN latest -> live
        ns = console._expression_namespace()
        assert np.array_equal(ns["frame_0"], hub.latest("frame_0"))          # newest block, not held back
    finally:
        console.shutdown()
        exp.close()


# ----------------------------------------------------------------------------- scalar-only / no lineage
def test_free_running_scalar_does_not_constrain_the_clock():
    """A NO_LINEAGE signal (a free-running loading rate) never constrains the display shot -- a board
    of only such signals has no coherent shot to hold (``_display_shot`` -> None == snapshot_latest)."""
    exp = _virtual_experiment()
    console = _console(exp)
    hub = console.hub
    try:
        _add_2d(console, ["rate"])
        # a node that publishes only a free-running scalar (no provenance)
        hub.publish({"rate": 0.5}, provenance=None)
        assert hub.latest_provenance("rate") == NO_LINEAGE

        from types import SimpleNamespace
        console.running_nodes = [SimpleNamespace(published_signals=lambda: frozenset({"rate"}))]
        assert console._display_shot() is None        # nothing with a lineage -> no hold
    finally:
        console.shutdown()
        exp.close()
