"""Contract (#H3o, design panel): a Processor is a PURE TYPED TRANSFORM with NO user-facing mode.

The repeat axis has exactly TWO knobs in the whole pipeline -- the measurement's ``repeat``/``free_run``
(how many shots) and the plot's ``repeat_mode`` (how to collapse them for display).  A processor must
NOT add a third.  Its relationship to the repeat axis is a STATIC class fact, ``repeat_contract``:

  * ``"reduce"``  -- emits derived signals carrying NO repeat axis (a per-shot judgement / a statistic
    over a shot set).  There is nothing left for the plot to collapse, so it never collides with
    ``repeat_mode``.
  * ``"preserve"`` -- maps each repeat slice 1:1 and emits a >=3-D block (axis 0 = repeat), so the SAME
    plot ``reduce_repeat`` machinery collapses it.

This test mechanically forbids (a) an undeclared/invalid contract and (b) ``repeat_contract`` ever
becoming a constructor arg (which would auto-render it as a form field -- a user knob).  It also pins
OccupancyProcessor as the canonical 'preserve' processor: it judges every repeat slice so the repeat
axis flows THROUGH the node (#H3q), and ``repeat_mode=average`` over ``occupied`` is the per-site
loading probability -- one mechanism, no in-node accumulator.
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


def test_occupancy_is_preserve_and_the_repeat_axis_flows_through():
    """A 'preserve' processor fed the camera's ``(repeat,1,H,W)`` block judges EVERY slice and emits
    the per-shot signals as CLEAN ``(repeat, n_sites)`` blocks (#H3s-F3): a LEADING repeat axis, NO
    vestigial middle 1.  The repeat axis flows THROUGH the node, so ``repeat_mode=average`` over
    ``occupied`` (collapsed by the signal's declared ``core_ndim``) recovers every site (the per-site
    loading PROBABILITY) -- ONE mechanism, not an in-node accumulator.  Static geometry
    (centers/thresholds) and the scalar ``rate`` carry NO repeat axis."""
    from Zou_lab_control.frontend.live import reduce_repeat
    assert OccupancyProcessor.repeat_contract == "preserve"

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
        occupied = np.asarray(hub.latest("occupied"))
        n_sites = occupied.shape[-1]
        # the per-shot signals carry a LEADING repeat axis (axis 0 == camera's repeat) and NO middle 1:
        # the CLEAN (repeat, n_sites) contract block (#H3s-F3).
        for key in ("occupied", "counts"):
            blk = np.asarray(hub.latest(key))
            assert blk.ndim == 2 and blk.shape == (5, n_sites), f"{key} shape {blk.shape}"
        assert np.asarray(hub.latest("frame_judged")).ndim == 3    # clean (repeat, H, W) -- no middle 1
        # static / scalar outputs do NOT gain a repeat axis
        assert np.asarray(hub.latest("centers")).ndim == 2         # (N, 2)
        assert np.asarray(hub.latest("thresholds")).ndim == 1      # (N,)
        assert np.ndim(hub.latest("rate")) == 0                    # scalar loading fraction
        # repeat_mode=average over the (repeat, n_sites) block (core_ndim=1) -> per-site loading probability
        prob = reduce_repeat(occupied, "average", core_ndim=1)
        assert prob.shape == (n_sites,) and np.all((prob >= 0) & (prob <= 1))
    finally:
        exp.close()
