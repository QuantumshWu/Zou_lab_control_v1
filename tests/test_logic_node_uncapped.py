"""Contract: a running LogicNode's acquisition loop is DATA-paced, NEVER rate-capped.

The user's iron rule: a backend measurement must run at full hardware speed -- rate limiting
belongs to UI parameter application / display refresh, never to the acquisition loop.  So a shot
that publishes loops straight into the next (its pace comes from the blocking device read); only a
pass that publishes NOTHING (a reactive processor whose input has not advanced) idles briefly so it
does not hot-spin.

Pure na-side (no Qt, no hardware): tiny LogicNodes whose shot() we control.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations.logic import LogicNode, SignalSpec
from Zou_lab_control.neutral_atom.core.signals import SignalHub


class _FastNode(LogicNode):
    """A measurement whose every shot takes ~1 ms (a fast-but-real acquisition) and always publishes."""

    PER_SHOT_S = 0.001

    def shot(self):
        time.sleep(self.PER_SHOT_S)   # stands in for a blocking device read
        return {"v": 1.0}

    def _bare_published_signals(self):
        return frozenset({"v"})

    def _bare_output_specs(self):
        return (SignalSpec("v", "acquired value", points_shape=(1,), data_shape=(1,)),)


class _IdleNode(LogicNode):
    """A reactive node that never has work: shot() always returns {} (no publish, no counter bump)."""

    def shot(self):
        return {}


def test_acquisition_loop_is_not_rate_capped():
    """In WINDOW seconds a ~1 kHz acquisition must publish FAR more than a 5 Hz cap would allow.  The
    old design slept the remainder of 1/rate_hz each pass, so ~5 shots/s; the loop is now data-paced, so
    the ~1 ms shot alone sets the cadence -> hundreds of shots.  A generous floor keeps it non-flaky on a
    loaded CI box while still being impossible to pass under any <=50 Hz cap."""
    hub = SignalHub()
    node = _FastNode(hub)
    window = 0.5
    node.start()
    try:
        time.sleep(window)
    finally:
        node.stop()
    # ~1 ms/shot over 0.5 s could reach ~500; require >=50, which a 5 Hz (or even 50 Hz) cap can never hit
    # (5 Hz -> ~3 shots; 50 Hz -> ~25).  The floor proves the loop adds NO throttle of its own.
    assert hub.shot >= 50, f"acquisition loop looks throttled: only {hub.shot} shots in {window}s"


def test_idle_reactive_node_publishes_nothing_and_does_not_fault():
    """A pass that publishes {} is a clean no-op: the shot counter never advances and no error banner is
    raised -- the loop just idles ``_IDLE_POLL_S`` and re-checks (no hot-spin, no synthetic signal)."""
    hub = SignalHub()
    node = _IdleNode(hub)
    node.start()
    try:
        time.sleep(0.2)
    finally:
        node.stop()
    assert hub.shot == 0                              # nothing was ever published
    assert getattr(node, "last_error", None) is None  # idling is not an error


class _Reactor(LogicNode):
    """A reactive processor: republishes ``out`` whenever its input signal ``in`` advances, else no-op."""

    def __init__(self, hub):
        super().__init__(hub)
        self._seen = 0

    def shot(self):
        v = self.hub.signal_versions().get("in", 0)
        if v <= self._seen:
            return {}                    # input has not advanced -> idle no-op (waits on the hub event)
        self._seen = v
        return {"out": float(v)}

    def _bare_published_signals(self):
        return frozenset({"out"})

    def _bare_output_specs(self):
        return (SignalSpec("out", "reacted version", points_shape=(1,), data_shape=(1,)),)


def test_reactive_node_wakes_on_publish_not_after_the_idle_cap():
    """The idle wait is EVENT-DRIVEN: a reactive node reacts the instant its input is published, NOT after
    the _IDLE_POLL_S fallback.  This is what keeps a DECOUPLED pulse scan (whose y comes from a reactive
    processor) running at the producer's cadence instead of a per-point idle-poll floor.  10 sequential
    inputs, each reacted to before the next is sent: a poll-based loop would take ~10*_IDLE_POLL_S (~0.5 s);
    event-driven takes tens of ms."""
    hub = SignalHub()
    node = _Reactor(hub)
    node.start()
    try:
        time.sleep(0.02)                 # let it settle into the idle (event) wait
        t0 = time.monotonic()
        got = []
        for i in range(1, 11):
            hub.publish({"in": float(i)})
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                try:
                    if float(hub.latest("out").item()) == float(i):
                        break
                except KeyError:
                    pass
                time.sleep(0.001)
            got.append(float(hub.latest("out").item()))
        elapsed = time.monotonic() - t0
    finally:
        node.stop()
    assert got == [float(i) for i in range(1, 11)]        # every input was reacted to, in order
    assert elapsed < 0.20, f"reactions took {elapsed:.3f}s -- looks poll-throttled, not event-driven"
