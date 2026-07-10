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
    exposure = 5e-3
    img = ta.render_image(exposure=exposure, all_sites=False).astype(float)
    # A background patch away from the sites carries the qCMOS offset PLUS a realistic
    # stray-light floor: ``background_rate`` photons/s land on EVERY pixel (the real Rb87 v16
    # corner-median is ~16.5 counts above the 200 offset at 3 ms), so dark pixels are NOT
    # pinned at the offset -- they sit at offset + the stray-light counts and carry its shot
    # noise.  Everything here is derived from the sim params (single source), no magic number.
    bg = img[80:96, 0:20]
    read_noise_counts = ta.read_noise_e / ta.conversion_e_per_count
    floor_e = (ta.background_rate + ta.dark_current_e_per_s) * exposure        # stray-light e-/px
    floor_counts = floor_e / ta.conversion_e_per_count                         # the floor in ADU
    assert ta.offset_counts > 0                                                # a real qCMOS bias (single source: VirtualTrapArray.offset_counts)
    # background sits at offset + the stray-light floor (a real, non-zero scatter level)
    assert abs(bg.mean() - (ta.offset_counts + floor_counts)) < 6.0
    assert floor_counts > 2.0 * read_noise_counts                              # a real floor, not just read noise
    # std combines read noise + background shot noise -> strictly above pure read noise
    expected_std = np.sqrt(floor_e + ta.read_noise_e**2) / ta.conversion_e_per_count
    assert 0.5 * expected_std < bg.std() < 2.0 * expected_std
    assert len(np.unique(bg.astype(int))) >= 8                                 # genuinely noisy, not a constant
    # an atom is far above the noise floor (a real bright/dark separation exists).
    assert img.max() > ta.offset_counts + floor_counts + 10.0 * read_noise_counts


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
            threshold_frames=40,
            readout_exposure=0.02, folder=str(folder))
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
            SignalHub(), source="live", threshold_frames=160,
            readout_exposure=5e-4)        # short readout -> not saturated
        task.run_to_completion()
        # the bracket frames were kept, grouped (n_ref long frames + one short readout each)
        assert len(task._reference_groups) == len(task._readout_by_group) > 0
        assert len(task._reference_groups[0]) == 2   # long-short-long -> two long reference frames

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
            SignalHub(), source="live", threshold_frames=200,
            readout_exposure=3e-4)        # very short -> populations OVERLAP
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


# ------------------------------------------------- figure-rendering failure contract (fast)
# No calibration run: a real two-site TrapCalibration + a handful of noise frames is the
# whole input write_calibration_report needs, so these run in milliseconds.

def _tiny_report_inputs():
    from Zou_lab_control.neutral_atom.core.calibration import FrameContract, TrapCalibration

    rng = np.random.default_rng(3)
    cal = TrapCalibration(
        centers=[[3.0, 3.0], [11.0, 3.0]], thresholds=[5.0, 5.0],
        frame_contract=FrameContract(image_shape=(8, 16)))
    frames = [rng.normal(4.0, 1.0, size=(8, 16)) for _ in range(6)]
    return cal, frames


def test_report_registered_plotter_bug_raises_no_silent_numbers_only_report(tmp_path, monkeypatch):
    """A REGISTERED plotter whose rendering raises is a real bug (plot-contract drift) and
    must propagate -- regression for the blanket ``except Exception: return {}`` that wrote
    a numbers-only report with no hint the figures failed."""
    from Zou_lab_control.neutral_atom.operations import calibration_report as cr

    class _BuggyPlotter:
        def save_calibration_report(self, folder, **kwargs):
            raise TypeError("plot contract drift")

    monkeypatch.setattr(cr, "active_plotter", lambda: _BuggyPlotter())
    cal, frames = _tiny_report_inputs()
    with pytest.raises(TypeError, match="plot contract drift"):
        cr.write_calibration_report(tmp_path / "rep", calibration=cal, readout_frames=frames)
    assert not (tmp_path / "rep" / "summary.json").exists()   # no silent numbers-only report


def test_report_headless_fallback_still_writes_numbers(tmp_path, monkeypatch):
    """No viewer registered (or one without the report hook) is the DOCUMENTED headless
    fallback: the numeric bundle is complete, the PNGs simply skipped -- no error recorded."""
    from Zou_lab_control.neutral_atom.operations import calibration_report as cr

    cal, frames = _tiny_report_inputs()
    for plotter in (None, object()):                      # absent / missing the report hook
        monkeypatch.setattr(cr, "active_plotter", lambda p=plotter: p)
        folder = tmp_path / ("none" if plotter is None else "hookless")
        summary = cr.write_calibration_report(folder, calibration=cal, readout_frames=frames)
        assert (folder / "summary.json").exists() and (folder / "calibration.npz").exists()
        assert "figure_error" not in summary              # headless is a fallback, not a failure
        assert set(summary["files"]) == {"npz", "calibration"}   # numbers only, by design


def test_report_figure_write_oserror_tolerated_and_self_described(tmp_path, monkeypatch):
    """An ENVIRONMENTAL write failure (disk full / permission) must not lose the numbers,
    but the summary self-describes why the figures are missing (never a bare numbers-only
    report), and a warning surfaces at write time."""
    import json
    from Zou_lab_control.neutral_atom.operations import calibration_report as cr

    class _DiskFullPlotter:
        def save_calibration_report(self, folder, **kwargs):
            raise OSError("disk full")

    monkeypatch.setattr(cr, "active_plotter", lambda: _DiskFullPlotter())
    cal, frames = _tiny_report_inputs()
    folder = tmp_path / "rep"
    with pytest.warns(UserWarning, match="disk full"):
        summary = cr.write_calibration_report(folder, calibration=cal, readout_frames=frames)
    assert "disk full" in summary["figure_error"]
    on_disk = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert "disk full" in on_disk["figure_error"]         # the folder self-describes offline too
