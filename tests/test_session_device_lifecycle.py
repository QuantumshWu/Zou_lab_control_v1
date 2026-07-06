"""Contract: the SESSION owns the device-consumer lifecycle seams (#A1, #device-change).

Long-lived device consumers -- the task console's running logic nodes, whose worker
threads block inside ``camera.acquire`` and hold camera / RPyC handles -- register on
the session so they are stopped BEFORE the hardware they use goes away.  There are TWO
seams, distinct on purpose:

* ``add_device_teardown_hook(hook)`` -- FULL teardown, run by ``exp.close()``: the whole
  console is going away, so every consumer stops (``hook()``, no args).
* ``add_device_change_hook(hook)`` -- FINE-GRAINED, run by ``load_config()``: only the
  devices actually being swapped are named (``hook(affected_ids)``), so a consumer stops
  only the work riding THOSE devices -- swapping the camera leaves a scan on the untouched
  sequencer running.

``load_config`` builds the NEW set FIRST: a bad config leaves the session -- including its
running consumers -- untouched.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na


def test_close_runs_teardown_hooks_before_devices_close():
    exp = na.connect("virtual")
    order: list[str] = []
    exp.add_device_teardown_hook(lambda: order.append("hook"))
    real_close = exp.devices.close
    exp.devices.close = lambda: (order.append("devices"), real_close())[1]
    exp.close()
    assert order == ["hook", "devices"]              # consumers stopped FIRST, then the hardware


def test_load_config_notifies_change_hook_after_build_but_before_swap():
    exp = na.connect("virtual")
    try:
        old_devices = exp.devices
        old_ids = {id(d) for d in old_devices.devices.values()}
        seen: list[tuple] = []
        # At notify time the session must still hold the OLD devices (consumers are stopped while
        # the hardware they use is intact), and the swap happens only afterwards.  The affected set
        # is EXACTLY the old instances (a full-config swap replaces every role).
        exp.add_device_change_hook(lambda affected: seen.append((exp.devices, frozenset(affected))))
        exp.load_config("virtual")
        assert len(seen) == 1
        seen_devices, seen_affected = seen[0]
        assert seen_devices is old_devices           # notified BEFORE the swap
        assert seen_affected == frozenset(old_ids)   # every old instance flagged (whole set swapped)
        assert exp.devices is not old_devices        # then the new set went in
    finally:
        exp.close()


def test_bad_config_leaves_consumers_untouched():
    exp = na.connect("virtual")
    try:
        calls: list[str] = []
        exp.add_device_change_hook(lambda affected: calls.append("hook"))
        with pytest.raises(Exception):
            exp.load_config({"camera": {"type": "no_such_device_type"}})
        assert calls == []                           # build-first: nothing was notified/stopped
    finally:
        exp.close()


def test_hook_registration_is_idempotent_and_survives_a_failing_hook():
    exp = na.connect("virtual")
    calls: list[str] = []

    def _boom():
        calls.append("boom")
        raise RuntimeError("broken consumer")

    exp.add_device_teardown_hook(_boom)
    exp.add_device_teardown_hook(_boom)              # duplicate registration is a no-op
    exp.add_device_teardown_hook(lambda: calls.append("after"))
    exp.close()                                      # a broken hook never blocks the close
    assert calls == ["boom", "after"]
