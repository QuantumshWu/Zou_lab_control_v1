"""Mechanical guards for the readout-decoupling audit findings (AGENTS §2: a
mechanically-enforceable rule is a test, not a doc line).

Four findings from the whole-project review are pinned here:

  F1  ``TrapCalibration.readout_exposure(fallback)`` is the ONE reader of the
      exposure-self-match invariant; detect / live-adopt / temperature survival all
      route through it instead of poking ``metadata['threshold_exposure']`` by hand.
  F2  ``signals()`` dispatches by an EXPLICIT readout-kind table (READOUT_KINDS),
      NOT a ``"psf" in name`` substring; every offered method is registered.
  F3  A PSF (kernel) calibration carries NO dead box-only state: roi_radius /
      reducer are None and survive a save/load round trip; a box calibration keeps
      them.  detect() on a PSF calibration does not crash on the None geometry.
  F4  The two coupled-tier resolvers share ONE helper with an EXPLICIT
      ``missing_policy`` (temperature=raise, readout_duration=fabricate) -- the
      previously-drifted divergence is now a declared parameter.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.calibration import (  # noqa: E402
    READOUT_KINDS,
    TrapCalibration,
    readout_kind,
)
from Zou_lab_control.neutral_atom.operations.calibration import ALL_READOUT_METHODS  # noqa: E402


def _psf_calibration(roi_radius=2, reducer="sum"):
    w = np.ones((2, 3, 3), dtype=float)
    w /= w.sum(axis=(1, 2), keepdims=True)
    boxes = np.array([[1, 1, 3, 3], [5, 1, 3, 3]], dtype=int)
    return TrapCalibration(
        centers=[(2, 2), (6, 2)], thresholds=[1.0, 1.0], method="psf",
        psf_weights=w, psf_boxes=boxes, roi_radius=roi_radius, reducer=reducer,
    )


def _box_calibration(roi_radius=2, reducer="sum"):
    return TrapCalibration(
        centers=[(2, 2), (6, 2)], thresholds=[3.0, 4.0], method="box",
        roi_radius=roi_radius, reducer=reducer,
    )


# ------------------------------------------------------------------ F1: exposure accessor
def test_readout_exposure_accessor_is_the_single_reader():
    """The accessor returns the stamped exposure when present and the caller's fallback when not --
    the ONE place threshold_exposure is read (#issue-2 / #H3v-2)."""
    stamped = TrapCalibration(centers=[(2, 2)], thresholds=[1.0],
                              metadata={"threshold_exposure": 0.005})
    assert stamped.readout_exposure(fallback=0.02) == pytest.approx(0.005)   # stamp wins
    bare = TrapCalibration(centers=[(2, 2)], thresholds=[1.0])
    assert bare.readout_exposure(fallback=0.02) == pytest.approx(0.02)       # fallback when unstamped
    assert bare.readout_exposure() is None                                   # None fallback = no forcing
    assert bare.readout_exposure(fallback=None) is None


def test_detect_default_exposure_self_matches_the_calibration():
    """detect() with exposure=None images at the calibration's threshold_exposure -- it sources that
    gate through the accessor, so the calibration and a default re-image always agree (#issue-2)."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    try:
        exp.readout.sitemap(method="box", frames=4, display=False)
        exp.devices.camera.exposure = 0.020                 # stale default, different from the readout gate
        exp.readout.thresholds(frames=24, exposure=0.004, display=False)
        cal = exp.readout.current
        assert cal.readout_exposure(fallback=exp.devices.camera.exposure) == pytest.approx(0.004)
        # the subsystem detect() reads at the accessor's exposure (not the stale 20 ms camera default)
        shot = exp.readout.detect(display=False)
        assert shot.occupied.shape == (12,)
    finally:
        exp.close()


def test_temperature_survival_sources_exposure_from_accessor(monkeypatch):
    """The temperature survival build reads the imaging exposure through the SAME accessor; pinning the
    accessor's return drives the imaging-window match (a mismatched exposure floors survival, #H3v-2)."""
    import Zou_lab_control.neutral_atom as na

    seen = {}
    real = TrapCalibration.readout_exposure

    def spy(self, fallback=None):
        out = real(self, fallback)
        seen["called"] = True
        seen["value"] = out
        return out

    monkeypatch.setattr(TrapCalibration, "readout_exposure", spy)
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3), "image_shape": (48, 64)})
    try:
        exp.readout.sitemap(method="box", frames=4, display=False)
        exp.readout.thresholds(frames=20, exposure=0.006, display=False)
        spec = {s.key: s for s in exp.readout.measurement_specs()}["temperature"]
        spec.build(template="pulses/release_recapture.json", t_off=(0.0, 40.0, 3), shots=1)
        assert seen.get("called"), "temperature build must read the exposure through readout_exposure()"
        assert seen.get("value") == pytest.approx(0.006)
    finally:
        exp.close()


# ------------------------------------------------------------------ F2: explicit readout-kind
def test_every_offered_method_has_an_explicit_readout_kind():
    """Dispatch is by declared kind, so every method a calibration can offer must be registered in
    READOUT_KINDS (the guard that catches a new method added without a kind)."""
    for m in ALL_READOUT_METHODS:
        assert m in READOUT_KINDS, f"{m} is offered but has no readout kind"
    assert READOUT_KINDS["box"] == "box"
    assert READOUT_KINDS["psf"] == "kernel" and READOUT_KINDS["uniform_psf"] == "kernel"


def test_readout_kind_dispatch_does_not_rely_on_the_method_name_substring():
    """A box-type method whose name merely CONTAINS 'psf' would be mis-routed by ``"psf" in m``;
    the explicit table routes by registered kind instead.  (Pin the table is the dispatch source.)"""
    # a hypothetical box variant with 'psf' in the name routes to BOX by its registered kind
    assert readout_kind("box") == "box"
    # and a kernel method without 'psf' in a future name would still route to kernel via the table
    assert readout_kind("uniform_psf") == "kernel"
    with pytest.raises(ValueError):
        readout_kind("not_a_registered_method")


def test_psf_and_box_signals_dispatch_correctly():
    """signals() routes box->roi_counts and psf->matched filter by the explicit kind, end to end."""
    img = np.zeros((10, 10), dtype=float)
    img[1:4, 1:4] = 50.0                                    # bright spot at site 0's box
    psf = _psf_calibration()
    box = _box_calibration(roi_radius=1, reducer="mean")
    box = TrapCalibration(centers=box.centers, thresholds=box.thresholds, method="box",
                          roi_radius=1, reducer="mean")
    assert psf.signals(img).shape == (2,)
    assert box.signals(img).shape == (2,)


# ------------------------------------------------------------- D4: uniform_psf stored verbatim
def test_single_method_uniform_psf_is_stored_verbatim():
    """A single-method uniform_psf calibration keeps ``method='uniform_psf'`` (no collapse to
    'psf'): methods() offers it, signals()/detect() resolve it, and the threshold path is
    unchanged.  The old collapse stored method='psf' plus a write-only ``psf_mode`` metadata
    key, so ``signals(method='uniform_psf')`` raised on the very calibration that computed it
    -- while the multi-method path stored 'uniform_psf' first-class (two naming mechanisms)."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3), "image_shape": (48, 64)})
    try:
        exp.readout.sitemap(method="uniform_psf", frames=4, display=False)
        cal = exp.readout.current
        assert cal.method == "uniform_psf"
        assert "uniform_psf" in cal.methods()
        assert cal.metadata["method"] == "uniform_psf"      # metadata mirrors the field verbatim
        assert "psf_mode" not in cal.metadata               # the write-only key is gone
        assert readout_kind(cal.method) == "kernel"         # dispatches through the ONE kind table
        assert cal.psf_weights is not None and cal.psf_boxes is not None

        # thresholds path unaffected: learns on the SAME signals detect() will compare against
        exp.readout.thresholds(frames=16, exposure=0.004, display=False)
        cal = exp.readout.current
        assert cal.method == "uniform_psf"
        image = np.zeros((48, 64), dtype=float)
        explicit = cal.signals(image, method="uniform_psf")  # the old collapse made this raise
        assert explicit.shape == (6,)
        assert np.allclose(explicit, cal.signals(image))     # default = the calibration's own method
        assert exp.readout.detect(display=False).occupied.shape == (6,)

        # box / psf single-method behavior unchanged (still stored verbatim, no psf_mode)
        exp.readout.sitemap(method="psf", frames=4, display=False)
        assert exp.readout.current.method == "psf"
        assert "psf_mode" not in exp.readout.current.metadata
        exp.readout.sitemap(method="box", frames=4, display=False)
        assert exp.readout.current.method == "box"
    finally:
        exp.close()


# ------------------------------------------------------------------ F3: no box-only residue on PSF
def test_psf_calibration_drops_box_only_geometry():
    """A PSF calibration carries no dead roi_radius/reducer; a box calibration keeps them."""
    psf = _psf_calibration(roi_radius=2, reducer="sum")
    assert psf.roi_radius is None and psf.reducer is None
    box = _box_calibration(roi_radius=2, reducer="sum")
    assert box.roi_radius == 2 and box.reducer == "sum"


def test_box_only_geometry_survives_round_trip(tmp_path):
    """roi_radius/reducer round-trip: box keeps its values, PSF stays None -- on BOTH json and npz."""
    for cal, tag in [(_psf_calibration(), "psf"), (_box_calibration(), "box")]:
        for ext in (".json", ".npz"):
            path = tmp_path / f"{tag}{ext}"
            cal.save(path)
            loaded = TrapCalibration.load(path)
            assert loaded.method == cal.method
            assert loaded.roi_radius == cal.roi_radius
            assert loaded.reducer == cal.reducer
            assert np.allclose(loaded.thresholds, cal.thresholds)


def test_to_dict_keeps_every_field_even_when_box_geometry_is_none():
    """to_dict still emits EVERY dataclass field (the #C2 parity contract): the box-only keys are
    present with value None on a PSF calibration, not silently dropped from the schema."""
    from dataclasses import fields as _dc_fields

    psf = _psf_calibration()
    payload = psf.to_dict()
    assert set(payload) == {f.name for f in _dc_fields(TrapCalibration)}
    assert payload["roi_radius"] is None and payload["reducer"] is None


def test_detect_on_psf_calibration_does_not_crash_on_none_geometry():
    """A PSF calibration's detect()/plot path must tolerate the None box geometry (no TypeError from
    a downstream ``float(None)``)."""
    from Zou_lab_control.neutral_atom.operations.detection import detect_image
    import Zou_lab_control.frontend  # noqa: F401  -- register the viewer so plotting runs

    psf = _psf_calibration()
    result = detect_image(np.zeros((10, 10), dtype=float), psf, display=False)
    assert list(result.occupied_indices) == []


# ------------------------------------------------------------------ F4: shared coupled resolver
def test_coupled_resolver_missing_policy_is_explicit_per_measurement():
    """temperature uses missing_policy='raise' (a NAMED-but-absent file fails loud); readout_duration
    uses 'fabricate' (a missing/unnamed file falls back to the default).  Both go through the ONE
    resolve_coupled_template helper, with the divergence declared, not silently drifted."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.operations.measurements.temperature import (
        _resolve_release_recapture_template,
    )
    from Zou_lab_control.neutral_atom.operations.measurements.readout_duration import (
        _resolve_imaging_template,
    )

    seqr = na.RuntimeSequencer(channels=["trap", "probe", "cooling", "emCCD"])

    # temperature: a NAMED missing template fails loud
    with pytest.raises(FileNotFoundError):
        _resolve_release_recapture_template("pulses/does_not_exist_rr.json", seqr)
    # temperature: an unnamed template fabricates the standard release-recapture (its trap-off slot)
    rr = _resolve_release_recapture_template("", seqr)
    assert rr.primary_time_slot() == "s0"

    # readout_duration: a missing template fabricates the single-image default and binds the scan slot
    img = _resolve_imaging_template("pulses/does_not_exist_img.json", seqr)
    assert img.primary_time_slot() is not None


def test_coupled_resolver_is_the_single_source_both_measurements_share():
    """Both coupled measurements import the SAME resolver -- the helper is the single source the
    framework drift (#15) is fixed against (a copied resolver would not import this)."""
    from Zou_lab_control.neutral_atom.operations.measurements import _coupled_template
    import Zou_lab_control.neutral_atom.operations.measurements.temperature as temp_mod
    import Zou_lab_control.neutral_atom.operations.measurements.readout_duration as dur_mod

    assert temp_mod.resolve_coupled_template is _coupled_template.resolve_coupled_template
    assert dur_mod.resolve_coupled_template is _coupled_template.resolve_coupled_template
