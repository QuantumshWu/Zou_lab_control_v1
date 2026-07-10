"""#H3w-3: temperature survival must DROP (not stick at 1) after the GUI "Calibrate readout" TASK.

Root cause of the stuck-at-1 regression: a threshold is exposure-specific, but only the
``ReadoutSubsystem.thresholds`` API path recorded the exposure -- the GUI ``CalibrateReadoutTask``
did NOT, so temperature skipped its exposure-matching and imaged the survival frames at the template's
baked 20 ms while the thresholds were learnt at 5 ms.  Every empty site then crossed the too-low
threshold in BOTH frames -> survival stuck at ~1.

Fix: the exposure is recorded in the typed frame contract by the ONE function every threshold path routes through
(``calibrate_threshold_from_images``), so EVERY calibration -- API or GUI task -- carries it.  This
pins that the GUI-task calibration records the exposure and temperature then drops correctly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na  # noqa: E402
from Zou_lab_control.neutral_atom.core.signals import SignalHub  # noqa: E402


def test_gui_calibrate_task_stamps_exposure_and_temperature_drops():
    exp = na.connect("virtual", sitemap={"grid_shape": (5, 7), "image_shape": (64, 80)})
    try:
        hub = SignalHub()
        # the GUI "Calibrate readout" TASK (readout_exposure default 5 ms) -- the path that was broken
        task = exp.readout.calibrate_task(hub, grid_shape=(5, 7))
        task.shot()
        cal = exp.readout.current
        expo = cal.readout_exposure()
        assert expo is not None, "the GUI Calibrate task must record the calibration exposure"
        assert abs(float(expo) - float(task.readout_exposure)) < 1e-12   # the exposure it learnt at

        # temperature now self-matches that exposure -> survival DROPS (not stuck at 1)
        spec = {s.key: s for s in exp.readout.measurement_specs()}["temperature"]
        res = spec.build(t_off=(0.0, 2000.0, 6), shots=40).run(live=False, display=False)
        y = np.asarray(res.y)
        assert y[0] >= 0.8, f"survival should start high (atoms recaptured), got {y[0]:.3f}"
        assert y[-1] <= 0.1, f"survival should drop at large trap-off (atoms lost), got {y[-1]:.3f}"
        assert y[-1] < 0.5, "survival must NOT be stuck at ~1"
    finally:
        exp.close()
