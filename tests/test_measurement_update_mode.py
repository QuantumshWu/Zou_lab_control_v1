"""Contract: a Measurement's ``update_mode`` accumulation is real behaviour.

roll/replace pass through (one publish per shot; the rolling-vs-overwrite VIEW is the
plot's job).  ``average`` publishes the cumulative running mean.  ``single`` publishes
once then self-stops.  ``repeat=N`` publishes only the mean of every N shots, with the
in-between ticks suppressed.  A bogus mode is rejected at construction.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.feeds import Measurement


class _EchoMeasurement(Measurement):
    """Echoes a fixed sequence of scalar values, one per shot (deterministic)."""

    def __init__(self, hub, seq, **kw):
        super().__init__(hub, **kw)
        self._seq = [float(v) for v in seq]
        self._i = 0

    def shot(self):
        value = self._seq[self._i % len(self._seq)]
        self._i += 1
        return {"v": value}


def test_roll_is_pass_through():
    hub = SignalHub()
    m = _EchoMeasurement(hub, [1, 2, 3], update_mode="roll")
    m.step(); m.step(); m.step()
    assert hub.latest("v") == 3.0
    assert hub.signal_versions()["v"] == 3        # one publish per shot


def test_average_is_cumulative_mean():
    hub = SignalHub()
    m = _EchoMeasurement(hub, [1, 2, 3], update_mode="average")
    m.step(); assert hub.latest("v") == 1.0
    m.step(); assert hub.latest("v") == 1.5
    m.step(); assert hub.latest("v") == 2.0


def test_single_publishes_once_then_stops():
    hub = SignalHub()
    m = _EchoMeasurement(hub, [5], update_mode="single")
    m.step()
    assert hub.latest("v") == 5.0
    assert m._stop.is_set()


def test_repeat_emits_mean_every_n_and_suppresses_between():
    hub = SignalHub()
    m = _EchoMeasurement(hub, [3, 6, 9], update_mode="repeat", repeats=3)
    assert m.step() == {}                         # accumulating
    assert m.step() == {}
    assert m.step() == {"v": 6.0}                 # mean of (3, 6, 9)
    assert hub.signal_versions()["v"] == 1        # only ONE publish for three shots


def test_bogus_update_mode_rejected():
    with pytest.raises(ValueError):
        _EchoMeasurement(SignalHub(), [1], update_mode="bogus")
