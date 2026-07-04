"""#issue-1 (sitemap/2d alignment): the calibration site centers must be in the SAME order as the
device occupancy index, so a STRUCTURED pattern (checkerboard) round-trips render -> detect exactly.

A pure permutation/transpose of the centers would still score ~1.0 on a RANDOM 50% pattern (the
comparison is symmetric) but COLLAPSES to chance on a spatially-structured pattern -- so the
checkerboard is the discriminating test that the sitemap and the 2D frame share one coordinate /
index convention.  This is a data-level invariant (the virtual frame goes through the SAME
``calibration.detect`` the real camera uses), so it MUST be a test.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")


def test_checkerboard_occupancy_round_trips_through_detect():
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual")
    try:
        trap = exp.devices.trap_array
        exp.devices.camera.exposure = 20e-3                 # high SNR -> readout error is negligible
        exp.readout.sitemap(method="box", frames=12, display=False)
        exp.readout.thresholds(frames=300, display=False)
        cal = exp.readout.current

        # a spatially-STRUCTURED pattern -- the discriminator for a center-order / transpose bug
        truth = np.zeros(trap.n_sites, dtype=bool)
        truth[::2] = True
        trap.set_occupancy(truth)
        frame = trap._render(truth, np.where(truth, 20e-3, 0.0), 20e-3).astype(float)

        pred = np.asarray(cal.detect(frame).occupied, dtype=bool).reshape(-1)
        acc = float((pred == truth).mean())
        assert acc >= 0.97, (
            f"checkerboard occupancy must round-trip (centers aligned to the occupancy index); "
            f"got acc={acc:.3f} -> sitemap centres and the 2D frame disagree on order/coords"
        )
        # and the inverse pattern too, so it is not an accidental all-True/all-False pass
        inv = ~truth
        trap.set_occupancy(inv)
        frame2 = trap._render(inv, np.where(inv, 20e-3, 0.0), 20e-3).astype(float)
        pred2 = np.asarray(cal.detect(frame2).occupied, dtype=bool).reshape(-1)
        assert float((pred2 == inv).mean()) >= 0.97
    finally:
        exp.close()
