"""Contract: Producer.acquisition_epoch marks the FIRST published frame
computed with a just-applied acquisition-parameter edit.

This is what lets the task-console Edit panel re-snapshot the fresh frame instead
of the stale pre-edit one: a running feed only QUEUES an edit and the loop applies
it BETWEEN shots, so the new-param frame appears a beat later -- the epoch advances
exactly when that frame is on the hub.  An idle feed applies + publishes
synchronously, so the epoch advances immediately.

Pure na-side (no Qt, no hardware): a tiny Producer whose shot() echoes its
one editable parameter.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations.feeds import Producer
from Zou_lab_control.neutral_atom.core.signals import SignalHub


class _EchoFeed(Producer):
    """Publishes its single editable parameter ``v`` each shot."""

    def __init__(self, hub):
        super().__init__(hub)
        self._v = 0.0

    def shot(self):
        return {"v": float(self._v)}

    def acquisition_parameters(self):
        return {"v": self._v}

    def set_acquisition_parameters(self, **values):
        if "v" in values:
            self._v = float(values["v"])


def test_idle_apply_bumps_epoch_and_publishes_synchronously():
    hub = SignalHub()
    feed = _EchoFeed(hub)
    assert feed.acquisition_epoch() == 0
    feed.apply_acquisition_parameters(v=5)        # idle: applied + published right here
    assert feed.acquisition_epoch() == 1
    assert float(hub.latest("v")) == 5.0


def test_running_apply_advances_epoch_only_when_new_param_frame_is_published():
    hub = SignalHub()
    feed = _EchoFeed(hub)
    feed.start(rate_hz=100.0)
    try:
        epoch0 = feed.acquisition_epoch()
        feed.apply_acquisition_parameters(v=9)    # running: QUEUED, applied between shots
        deadline = time.monotonic() + 2.0
        while feed.acquisition_epoch() == epoch0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert feed.acquisition_epoch() > epoch0  # advanced once the loop applied + published
        assert float(hub.latest("v")) == 9.0      # the epoch-advance frame reflects the new param
    finally:
        feed.stop()
