"""#H3v-1: the "Fidelity vs duration" measurement is the COUPLED pulse-scan special case that is the
SIBLING of "Temperature" -- full parity:

- both LOAD a selectable pulse ``template`` (editable in the pulse GUI),
- both sweep ONE DURATION scan slot of that template (temperature: trap-off ``s0``; fidelity: the
  imaging template's image-period readout window),
- both reduce each point INLINE (temperature: survival from 2 frames; fidelity: Otsu fidelity over the
  point's frame set),
- both declare ``scan_tier="coupled"``.

They are MEASUREMENTS (a live, hub-published, fittable x-y curve), NOT Tasks (a Task publishes nothing
to the hub and yields an off-hub artifact via a single mid-run frame).  This pins the parity + that the
virtual fidelity rises with readout duration (short = indistinguishable ~0.5; long = higher SNR).
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

GRID = (5, 7)


def _calibrated_exp():
    exp = na.connect("virtual", sitemap={"grid_shape": GRID, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    return exp


def _spec(exp, key):
    return {sp.key: sp for sp in exp.readout.measurement_specs()}[key]


def test_fidelity_and_temperature_share_the_coupled_template_scan_shape():
    """Both specs are the coupled tier (scan_tier='coupled') and BOTH expose a selectable pulse
    ``template`` param + an ``axis_range`` sweep -- the parity #H3v-1 demanded for fidelity."""
    exp = _calibrated_exp()
    try:
        fid = _spec(exp, "readout")
        temp = _spec(exp, "temperature")
        for sp in (fid, temp):
            assert (sp.metadata or {}).get("scan_tier") == "coupled", f"{sp.name} must be coupled tier"
            keys = {p.key for p in sp.params}
            assert "template" in keys, f"{sp.name} must load a selectable pulse template"
            assert any(p.kind == "axis_range" for p in sp.params), f"{sp.name} must sweep an axis_range"
        # fidelity's swept axis is the readout DURATION; its y is single-shot fidelity
        assert fid.x_key == "detection_time" and fid.y_key == "fidelity"
    finally:
        exp.close()


def test_fidelity_template_scan_rises_with_readout_duration():
    """Building the fidelity spec from its imaging template + running it in virtual yields a finite
    fidelity curve in [0,1] that is HIGHER at a long readout than at a short one (more collected
    fluorescence -> the bright/dark counts separate -> higher single-shot fidelity).

    The sweep runs up to the ~20 ms qCMOS working point (the science camera's default exposure): at
    50 us a readout collects too few photons to tell bright from dark (~0.5), and at 20 ms the
    bright/dark counts fully separate (near the fidelity ceiling).  The long-end FLOOR is a real guard
    on the virtual model's SNR -- it is why the v16 photon-rate retune (``atom_rate`` 2150 -> 1100, so
    the model no longer over-separated) has to keep a 20 ms readout genuinely high-fidelity, not merely
    'climbing a little' by the old 5 ms (which the retune left marginal at ~0.55)."""
    exp = _calibrated_exp()
    try:
        fid = _spec(exp, "readout")
        scan = fid.build(duration=(50.0, 20000.0, 5), shots=24)
        res = scan.run(live=False, display=False)
        x, y = np.asarray(res.x), np.asarray(res.y)
        assert x.shape == y.shape and x.size == 5
        assert np.all(np.isfinite(y)) and np.all((y >= 0.0) & (y <= 1.0))
        assert y[-1] >= y[0] + 0.05, f"fidelity should climb with readout duration, got {np.round(y,3)}"
        assert y[0] <= 0.65, f"a 50 us readout should be near-indistinguishable (~0.5), got {y[0]:.3f}"
        assert y[-1] >= 0.8, f"a 20 ms readout (qCMOS working point) should be high-fidelity, got {y[-1]:.3f}"
    finally:
        exp.close()
