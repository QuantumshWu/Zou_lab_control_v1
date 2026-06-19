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

        # the report lands DIRECTLY in the user's explicit `folder` -- one place, no hidden
        # timestamped sub-folder (re-running overwrites; a different run uses a different folder).
        report_dir = Path(task.result["report_dir"])
        assert report_dir == folder
        assert report_dir.exists()
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


def test_reference_bracket_gives_distinct_per_method_fidelity():
    """The Rb87 readout-fidelity flow: a long-short-long reference bracket images the SAME
    atoms, so the two long frames vote ground-truth occupancy for the short readout (atom
    loss makes a shot ambiguous).  Each method's per-site threshold is then trained + scored
    HELD-OUT against those labels.  Because box / per-site PSF / uniform PSF weight the
    photons differently, their held-out fidelity at a fidelity-LIMITED (short) readout must
    DIFFER -- regression for the bug where all three reported a bitwise-identical fidelity
    (the affine-invariant self-consistent estimate, which cannot tell the methods apart)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt5")
    import Zou_lab_control.neutral_atom as na
    import Zou_lab_control.frontend  # noqa: F401  (registers the viewer)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.calibration_report import _held_out_by_method

    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5), "image_shape": (80, 100)}, seed=5)
    try:
        task = exp.readout.calibrate_task(
            SignalHub(), source="live", calibration_frames=6, threshold_frames=160,
            sitemap_exposure=0.03, readout_exposure=5e-4)        # short readout -> not saturated
        task.run_to_completion()
        # the bracket frames were kept, grouped (n_ref long frames + one short readout each)
        assert len(task._reference_groups) == len(task._readout_by_group) > 0
        assert len(task._reference_groups[0]) == task.REFERENCE_FRAMES_PER_BRACKET

        by_method = _held_out_by_method(task.calibration, task._reference_groups, task._readout_by_group)
        assert set(by_method) == {"box", "psf", "uniform_psf"}
        means = {m: float(np.nanmean(d["fidelity"])) for m, d in by_method.items()}
        # every method is a real held-out classification fidelity (a sane 0.5..1.0)
        assert all(0.5 <= v <= 1.0 for v in means.values())
        # and the three are NOT all the bitwise-identical number the old estimate forced:
        # at a fidelity-limited readout the methods genuinely separate the populations
        # differently, so at least two differ by a real margin.
        spread = max(means.values()) - min(means.values())
        assert spread > 1e-3, f"per-method fidelity must differ, got {means}"
    finally:
        exp.close()


def test_overlapping_readout_reports_sub_unity_fidelity_and_per_method_summary(tmp_path):
    """Regression for the report headlining a spurious ~100% fidelity on a CLEARLY overlapping
    distribution.  At a fidelity-LIMITED (very short) readout the dark/bright populations
    overlap, so the per-site fidelity drawn + summarised must be the two-Gaussian MODEL
    (overlap) fidelity -- which matches the plotted distribution and is < 1 -- NOT the small
    held-out test-split classification accuracy that quantises onto exactly 1.000.  And
    summary.json must report EACH readout method's fidelity (box / per-site PSF / uniform PSF),
    not one box aggregate."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt5")
    import json
    import Zou_lab_control.neutral_atom as na
    import Zou_lab_control.frontend  # noqa: F401  (registers the viewer)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.calibration import ALL_READOUT_METHODS
    from Zou_lab_control.neutral_atom.operations.calibration_report import write_calibration_report

    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5), "image_shape": (80, 100)}, seed=5)
    try:
        task = exp.readout.calibrate_task(
            SignalHub(), source="live", calibration_frames=6, threshold_frames=200,
            sitemap_exposure=0.03, readout_exposure=3e-4)        # very short -> populations OVERLAP
        task.run_to_completion()
        summary = write_calibration_report(
            str(tmp_path / "report"), calibration=task.calibration,
            readout_frames=task._readout_samples, reference_groups=task._reference_groups,
            readout_by_group=task._readout_by_group, threshold_method="otsu")

        data = json.loads((Path(summary["folder"]) / "summary.json").read_text(encoding="utf-8"))
        n_sites = int(data["n_sites"])
        # EVERY readout method's fidelity is in the summary (not just one box aggregate)
        assert set(data["methods"]) == set(ALL_READOUT_METHODS)
        for m, e in data["methods"].items():
            assert e["mean_fidelity"] is not None and 0.5 <= e["mean_fidelity"] <= 1.0
            assert len(e["per_site_fidelity"]) == n_sites
            assert e["held_out"] is True and e["held_out_accuracy_mean"] is not None   # out-of-sample check kept
        # the OVERLAPPING readout does NOT report a spurious ~100%: the box model fidelity is
        # clearly below 1, and the worst site is well under unity (a real overlap, as drawn).
        box = data["methods"]["box"]
        assert box["mean_fidelity"] < 0.97, box["mean_fidelity"]
        assert box["worst_site_fidelity"] < 0.95, box["worst_site_fidelity"]
        finite = [v for v in box["per_site_fidelity"] if v is not None]
        assert finite and not any(v > 0.999 for v in finite)   # nothing pinned at 100%
    finally:
        exp.close()
