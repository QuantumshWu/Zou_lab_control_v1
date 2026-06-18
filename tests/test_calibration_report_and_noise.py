"""Contract: the virtual camera carries a real qCMOS NOISE floor, and a calibration
writes the rb87-style distribution + fidelity REPORT to a folder.

These pin two things the experimenter relies on:
  * a rendered frame is not noiseless -- it has the qCMOS read-noise + offset + Poisson
    shot noise (so dark/bright distributions have realistic WIDTH), matching the Rb87
    ``CameraConfig`` (offset 200, gain 0.107 e-/count, read 0.43 e- rms);
  * running the calibrate task leaves a folder of reviewable artifacts (per-site count
    distribution grid, global distribution, site map, an npz + summary.json with a
    finite per-site fidelity) -- the Rb87 readout outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.virtual import VirtualTrapArray


def test_qcmos_frame_has_realistic_read_noise_offset_and_shot_noise():
    ta = VirtualTrapArray(grid_shape=(5, 7), image_shape=(96, 128), seed=7)
    ta.set_occupancy(np.ones(ta.n_sites, dtype=bool))
    img = ta.render_image(exposure=5e-3, all_sites=False).astype(float)
    # a background patch away from the sites: offset ~200, std ~ read-noise in ADU.
    bg = img[80:96, 0:20]
    read_noise_counts = ta.read_noise_e / ta.conversion_e_per_count
    assert ta.offset_counts == pytest.approx(200.0)            # the Rb87 qCMOS bias
    assert abs(bg.mean() - ta.offset_counts) < 5.0             # background sits at the offset
    assert 0.5 * read_noise_counts < bg.std() < 2.0 * read_noise_counts  # ~read noise, NOT flat
    assert len(np.unique(bg.astype(int))) >= 8                 # genuinely noisy, not a constant
    # an atom is far above the noise floor (a real bright/dark separation exists).
    assert img.max() > ta.offset_counts + 10.0 * read_noise_counts


def test_calibrate_writes_distribution_and_fidelity_report(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt5")
    import Zou_lab_control.neutral_atom as na
    # Importing the frontend registers the viewer (the viewer-registry seam): the calibrate
    # task then renders its report PNGs through the FRONTEND plot types (site_histogram_grid
    # / hist / sites), not hand-rolled matplotlib -- the same path the running console uses.
    import Zou_lab_control.frontend  # noqa: F401
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5), "image_shape": (80, 100)}, seed=5)
    try:
        hub = SignalHub()
        folder = tmp_path / "cal_run"
        task = exp.readout.calibrate_task(
            hub, source="live", threshold_method="otsu",
            calibration_frames=10, threshold_frames=40,
            sitemap_exposure=0.05, readout_exposure=0.02, folder=str(folder))
        task.run_to_completion()

        # the report lands in a timestamped run sub-folder of `folder` (never overwriting)
        report_dir = Path(task.result["report_dir"])
        assert report_dir.exists()
        assert report_dir.parent == folder
        # the rb87-style artifacts all landed there: ONE per-site distribution grid PER
        # readout method (box / per-site PSF / uniform PSF -- the cali computes all three),
        # the pooled distribution + site map, the reloadable calibration + numeric bundle.
        for name in ("site_distribution_box.png", "site_distribution_psf.png",
                     "site_distribution_uniform_psf.png", "global_distribution.png",
                     "site_map.png", "calibration.npz", "calibration.json", "summary.json"):
            assert (report_dir / name).exists(), name
        # the report carries a FINITE per-site fidelity (the distributions separate)
        assert task.report["n_sites"] == exp.devices.trap_array.n_sites
        assert 0.5 <= task.report["mean_fidelity"] <= 1.0
        bundle = np.load(report_dir / "calibration.npz")
        assert bundle["counts"].shape[0] == 40                  # the readout frames
        assert np.all(np.isfinite(bundle["thresholds"]))

        # EVERY readout method (box / per-site PSF / uniform PSF) must read on the SAME
        # scale its thresholds were calibrated on: signals(method=m) and thresholds_for(m)
        # share the method's OWN background (a PSF method subtracts an annulus, box does
        # not).  Regression for the bug where signals(psf) used box's "none" background, so
        # the psf threshold landed BELOW every shot (all bright; the per-site figure's
        # threshold line + fidelity vanished off-axis).
        from Zou_lab_control.neutral_atom.operations.calibration_report import (
            per_site_counts, per_site_fidelity)
        cal = task.calibration
        frames = task._readout_samples
        assert set(cal.methods()) == {"box", "psf", "uniform_psf"}
        for m in cal.methods():
            counts = per_site_counts(cal, frames, method=m)
            thr = np.asarray(cal.thresholds_for(m), dtype=float)
            # the threshold lies INSIDE the per-site count range (separates the populations)
            assert np.all(thr > counts.min(axis=0) - 1e-6)
            assert np.all(thr < counts.max(axis=0) + 1e-6)
            # ... so the held-out per-site fidelity is finite for every site
            fid = per_site_fidelity(counts, thr)
            assert np.all(np.isfinite(fid))
            assert 0.5 <= float(np.mean(fid)) <= 1.0
    finally:
        exp.close()
