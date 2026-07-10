"""Contract (#iron-law): every measurement/processor block is ``(R,P,*data_shape)`` --
the physical P axis is ALWAYS present (a no-scan acquisition = exactly ONE point), and a processor
PRESERVES it.  Occupancy judges ONE readout per shot, so ``occupied``/``counts`` are
``(repeat, 1, n_sites)`` and ``frame_judged`` is ``(repeat, 1, H, W)`` -- the SAME shape the camera
frame carries (a frame keeps one shape raw or judged).  The middle 1 is NEVER dropped.

Every signal carries a registered SignalSchema.  Static site geometry is also canonical:
centres are ``(1,1,N,2)`` and thresholds ``(1,1,N)``; site remains data, never P.

Real reproduction: a virtual exp + camera + OccupancyProcessor run for ``repeat`` shots, then fed
through the REAL sites-plot consumption path (``PanelCard._signal_then_repeat`` -> ``reduce_repeat`` ->
``_coerce``), not a synthesized array.
"""

from __future__ import annotations

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
    CameraMeasurement, OccupancyProcessor, describe_shape,
)

from conftest import fire_live_imaging  # noqa: E402


REPEAT = 5
GRID = (5, 7)
N_SITES = GRID[0] * GRID[1]


def _calibrated_occupancy(hub):
    """A calibrated virtual exp + a camera + an OccupancyProcessor, all wired through the REAL
    contract path (only the camera frames are simulated)."""
    exp = na.connect("virtual", sitemap={"grid_shape": GRID})
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                            repeat=REPEAT)                       # 0=∞; REPEAT = a REPEAT-deep block
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame_0"], "source": "value = signal"},
                             method="box", grid_shape=GRID)
    fire_live_imaging(exp)
    for _ in range(REPEAT + 2):
        cam.step(); det.step()
    return exp, cam, det


def test_occupancy_block_keeps_point_and_full_data_shape_axes():
    """`occupied` / `counts` are ``(R,P,*data_shape)`` with P=1 and data_shape=(n_sites,) (P is PRESENT,
    size 1 for a no-scan readout); `frame_judged` is ``(repeat, 1, H, W)`` -- the SAME shape the camera
    `frame_0` carries; static geometry uses a single repeat/point cell."""
    hub = SignalHub()
    exp, cam, _ = _calibrated_occupancy(hub)
    try:
        occupied = np.asarray(hub.latest("occupied"))
        counts = np.asarray(hub.latest("counts"))
        frame_judged = np.asarray(hub.latest("frame_judged"))
        frame_raw = np.asarray(hub.latest("frame_0"))
        assert occupied.shape == (REPEAT, 1, N_SITES)          # (R,P,*data_shape), P=1
        assert counts.shape == (REPEAT, 1, N_SITES)
        assert frame_judged.shape == frame_raw.shape           # judged frame == raw frame shape (r,1,H,W)
        assert frame_judged.ndim == 4 and frame_judged.shape[1] == 1
        assert np.asarray(hub.latest("centers")).shape == (1, 1, N_SITES, 2)
        assert np.asarray(hub.latest("thresholds")).shape == (1, 1, N_SITES)
        assert np.asarray(hub.latest("rate")).shape == (REPEAT, 1, 1)
        assert hub.schema("centers").point_shape == (1,)
        assert hub.schema("centers").data_shape == (N_SITES, 2)
    finally:
        exp.close()


def test_schema_validity_guides_repeat_average_to_per_site_loading_probability():
    """The declared R axis is reduced while P=1 and data_shape=(N,) remain intact."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data
    hub = SignalHub()
    exp, _, _ = _calibrated_occupancy(hub)
    try:
        tensor = hub.latest_tensor("occupied")
        occupied = np.asarray(tensor.data)
        reduced = reduce_repeat(occupied, "average", valid=tensor.valid)
        assert reduced.shape == (1, N_SITES)                   # R collapsed; P axis kept
        prob = np.squeeze(reduced)                             # the display squeeze -> (n_sites,)
        assert prob.shape == (N_SITES,)
        np.testing.assert_allclose(prob, np.nanmean(occupied, axis=0).reshape(-1))
        assert np.all((prob >= 0) & (prob <= 1))
        assert repeats_with_data(occupied, valid=tensor.valid) == REPEAT
        # An external already-reduced value has no R axis only when the caller
        # says so explicitly; ndarray rank is never used to infer that fact.
        passthrough = reduce_repeat(reduced, "average", has_repeat=False)
        np.testing.assert_array_equal(passthrough, reduced)
    finally:
        exp.close()


def test_describe_shape_renders_all_three_groups():
    """`describe_shape` renders each occupancy signal off ITS declared structure, ALWAYS as
    ``R × P × data_shape`` (the middle is never dropped): occupied ``5 × 1 × (35)``,
    frame_judged ``5 × 1 × (96×128)``, centers ``1 × 1 × (35×2)``."""
    hub = SignalHub()
    exp, _, det = _calibrated_occupancy(hub)
    try:
        specs = {s.name: s for s in det.output_specs()}
        occ_spec = specs["occupied"]
        frame_spec = specs["frame_judged"]
        occupied = hub.latest("occupied")
        frame_judged = hub.latest("frame_judged")
        H, W = np.asarray(frame_judged).shape[2:]
        assert describe_shape(occupied, points_shape=occ_spec.points_shape,
                              data_shape=occ_spec.data_shape) == f"{REPEAT} × 1 × ({N_SITES})"
        # #H3v-3: a no-scan signal's P shows the literal 1, NEVER a trap-grid -- even if a
        # grid_shape is passed it is ignored (the 35 sites are not "5×7"; that layout is the sitemap's
        # display concern, not data_shape).  grid_shape only reshapes a 2-D SCAN's points.
        assert describe_shape(occupied, points_shape=occ_spec.points_shape,
                              data_shape=occ_spec.data_shape,
                              grid_shape=GRID) == f"{REPEAT} × 1 × ({N_SITES})"
        assert describe_shape(frame_judged, points_shape=frame_spec.points_shape,
                              data_shape=frame_spec.data_shape) == f"{REPEAT} × 1 × ({H}×{W})"
        center_spec = specs["centers"]
        assert center_spec.points_shape == (1,)
        assert center_spec.data_shape == (N_SITES, 2)
        assert describe_shape(
            hub.latest("centers"),
            points_shape=center_spec.points_shape,
            data_shape=center_spec.data_shape,
        ) == f"1 × 1 × ({N_SITES}×2)"
    finally:
        exp.close()


def test_sites_panel_renders_the_averaged_per_site_map_from_occupancy():
    """Reproduce the REAL sitemap consumption: a `sites` PanelCard bound to the `occupied` signal
    (default `value = signal`) runs `_signal_then_repeat` through its registered schema -> `_coerce`
    and yields exactly the per-site loading-probability vector (n_sites values), which the sites plotter
    draws as the ring colours.  Not a mock -- the same path the live site map takes."""
    pytest.importorskip("PyQt5")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    from Zou_lab_control.frontend.live import reduce_repeat
    ensure_qt_app()

    hub = SignalHub()
    exp, _, det = _calibrated_occupancy(hub)
    try:
        # the console resolves a bound signal's producing-node structure + the site-map centres/underlay
        def structure_provider(name):
            if name == "occupied":
                return {"points_shape": (1,), "data_shape": (N_SITES,), "grid_shape": ()}
            return None

        def sites_inputs_provider(occ):
            return ("centers", "frame_judged") if occ == "occupied" else (None, None)

        card = PanelCard(PanelConfig(kind="sites", inputs=["occupied"], source="value = signal"),
                         structure_provider=structure_provider,
                         sites_inputs_provider=sites_inputs_provider)
        namespace = dict(hub.snapshot_latest())
        # the panel pipeline: per-slice eval -> structure-driven R collapse -> squeeze P
        reduced = card._signal_then_repeat(namespace)
        coerced = card._coerce(reduced)
        # it IS the per-site loading probability (mean over the REPEAT shots), one value per site
        tensor = hub.latest_tensor("occupied")
        expected = np.squeeze(reduce_repeat(tensor.data, "average", valid=tensor.valid))
        assert np.asarray(coerced).shape == (N_SITES,)
        np.testing.assert_allclose(np.asarray(coerced), expected)
        assert np.all((np.asarray(coerced) >= 0) & (np.asarray(coerced) <= 1))
        # the occupancy ROLE drives the repeat-mode tooltip (occupancy meaning, not generic)
        assert card._bound_is_occupancy()
        assert "LOADING PROBABILITY" in card._repeat_mode_tooltip()
    finally:
        if hasattr(card, "shutdown"):
            card.shutdown()
        exp.close()


def test_centers_align_with_the_judged_frame():
    """#H3u-3 GUARD: the published `centers` (x, y) land on the BRIGHT atoms of `frame_judged` -- the
    mean intensity AT the centers is much higher than the whole-frame mean, so the site-map rings sit ON
    the atoms (not offset/swapped).  The wrong (row, col) order would NOT beat the control."""
    from Zou_lab_control.frontend.live import reduce_repeat
    hub = SignalHub()
    exp, _, _ = _calibrated_occupancy(hub)
    try:
        centers = np.asarray(hub.latest("centers"), dtype=float)[0, 0]
        fj = np.squeeze(reduce_repeat(np.asarray(hub.latest("frame_judged"), dtype=float), "average"))
        h_img, w_img = fj.shape

        def at(xy):
            cx, cy = int(round(xy[0])), int(round(xy[1]))
            return float(fj[cy, cx]) if (0 <= cy < h_img and 0 <= cx < w_img) else np.nan

        i_xy = np.nanmean([at(c) for c in centers])                 # (x=col, y=row) -- the contract
        i_swap = np.nanmean([at((c[1], c[0])) for c in centers])    # the wrong (row, col) order
        control = float(np.nanmean(fj))
        assert i_xy > control * 1.2, f"centers must sit on the bright atoms: {i_xy} vs control {control}"
        assert i_xy > i_swap, "the (x, y) order beats the swapped (row, col) order -> not a coordinate swap"
    finally:
        exp.close()


def test_sites_underlay_obeys_the_panel_repeat_mode():
    """#H3u-3 GUARD: the site-map underlay (frame_judged) obeys the SAME `repeat_mode` Setting as the
    rings -- 'average' = the long-exposure mean over the kept frames, 'replace' = the latest frame -- so
    they differ (the hardcoded-'replace' bug had made repeat_mode a no-op for the underlay)."""
    pytest.importorskip("PyQt5")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    from Zou_lab_control.frontend.live import reduce_repeat
    ensure_qt_app()

    hub = SignalHub()
    exp, _, _ = _calibrated_occupancy(hub)
    try:
        def structure_provider(name):
            if name == "occupied":
                return {"points_shape": (1,), "data_shape": (N_SITES,), "grid_shape": GRID}
            return None

        def sites_inputs_provider(occ):
            return ("centers", "frame_judged") if occ == "occupied" else (None, None)

        ns = dict(hub.snapshot_latest())
        raw = np.asarray(hub.latest("frame_judged"), dtype=float)

        def underlay(mode):
            card = PanelCard(PanelConfig(kind="sites", inputs=["occupied"], source="value = signal",
                                         params={"repeat_mode": mode}),
                             structure_provider=structure_provider,
                             sites_inputs_provider=sites_inputs_provider)
            try:
                _, image = card._sites_aux(ns)
            finally:
                if hasattr(card, "shutdown"):
                    card.shutdown()
            return np.asarray(image, dtype=float)

        avg, rep = underlay("average"), underlay("replace")
        np.testing.assert_allclose(avg, np.squeeze(reduce_repeat(raw, "average")))   # long-exposure mean
        np.testing.assert_allclose(rep, np.squeeze(reduce_repeat(raw, "replace")))   # the latest frame
        assert not np.allclose(avg, rep)            # repeat_mode is NOT a no-op for the underlay
    finally:
        exp.close()
