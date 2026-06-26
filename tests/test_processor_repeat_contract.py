"""Contract (#H3o, design panel): a Processor is a PURE TYPED TRANSFORM with NO user-facing mode.

The repeat axis has exactly TWO knobs in the whole pipeline -- the measurement's ``repeat``/``free_run``
(how many shots) and the plot's ``repeat_mode`` (how to collapse them for display).  A processor must
NOT add a third.  Its relationship to the repeat axis is a STATIC class fact, ``repeat_contract``:

  * ``"reduce"``  -- emits derived signals carrying NO repeat axis (a per-shot judgement / a statistic
    over a shot set).  There is nothing left for the plot to collapse, so it never collides with
    ``repeat_mode``.
  * ``"preserve"`` -- maps each repeat slice 1:1 and emits a >=3-D block (axis 0 = repeat), so the SAME
    plot ``reduce_repeat`` machinery collapses it.

This test mechanically forbids (a) an undeclared/invalid contract, (b) a 'reduce' processor leaking a
repeat axis, and (c) ``repeat_contract`` ever becoming a constructor arg (which would auto-render it as
a form field -- a user knob).  It encodes the panel's verdict that OccupancyProcessor stays per-shot.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na  # noqa: E402
from Zou_lab_control.neutral_atom.core.signals import SignalHub  # noqa: E402
from Zou_lab_control.neutral_atom.operations.logic import (  # noqa: E402
    CameraMeasurement, OccupancyProcessor, Processor,
)

from conftest import fire_live_imaging  # noqa: E402


def _all_processor_subclasses() -> list:
    seen, out = set(), []
    stack = list(Processor.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        out.append(cls)
        stack.extend(cls.__subclasses__())
    return out


def test_every_processor_declares_a_valid_repeat_contract():
    subs = _all_processor_subclasses()
    assert OccupancyProcessor in subs                      # the canonical reactive processor is enumerated
    for cls in subs:
        assert cls.repeat_contract in ("reduce", "preserve"), (
            f"{cls.__name__}.repeat_contract={cls.repeat_contract!r} must be 'reduce' or 'preserve'")


def test_repeat_contract_is_never_a_constructor_arg():
    """It is a STATIC class fact, not a runtime knob -- so it can never be auto-rendered as a form
    field (the params a console form shows come from the build/ctor signature)."""
    for cls in _all_processor_subclasses():
        params = inspect.signature(cls.__init__).parameters
        assert "repeat_contract" not in params, (
            f"{cls.__name__}.__init__ must not accept repeat_contract (it would become a user knob)")


def test_occupancy_is_reduce_and_publishes_no_repeat_axis():
    """A 'reduce' processor fed the camera's ``(repeat,1,H,W)`` block emits ONLY signals with no
    repeat axis (per-site vectors (N,), a single judged (H,W) frame, scalars).  This is why the
    per-site loading PROBABILITY is the processor's cumulative ``rate_sites`` -- not a plot knob on a
    (repeat, n_sites) block, which the plot's collapse machinery silently skips."""
    assert OccupancyProcessor.repeat_contract == "reduce"

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer, repeat=5, free_run=True)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame"], "source": "value = signal"},
                             method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)
    try:
        for _ in range(6):
            cam.step()
            det.step()
        assert np.asarray(hub.latest("frame")).ndim == 4   # the camera DOES publish a (repeat,1,H,W) block
        for key in det.published_signals():
            value = np.asarray(hub.latest(key))
            assert value.ndim < 3, f"reduce processor leaked a repeat axis on {key!r}: shape {value.shape}"
    finally:
        exp.close()
