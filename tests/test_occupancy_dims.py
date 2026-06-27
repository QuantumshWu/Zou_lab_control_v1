"""Contract (#H3s-F3/F4): the occupancy dimension contract is CLEAN and repeat-collapse is
STRUCTURE-driven, not an ndim guess.

The muddle this pins down: `OccupancyProcessor` used to publish `occupied`/`counts` as
`(repeat, 1, n_sites)` and `frame_judged` as `(repeat, 1, H, W)` -- a vestigial middle 1 that existed
ONLY so the plot's `reduce_repeat` (which collapsed axis 0 only when `ndim >= 3`) would still average
the repeat axis. The clean design drops the middle 1 (`occupied` -> `(repeat, n_sites)`,
`frame_judged` -> `(repeat, H, W)`) and makes `reduce_repeat` collapse axis 0 by the signal's DECLARED
`core_ndim` (= len(points_shape)+len(data_shape)), so a clean 2-D occupancy block IS still recognised as
a repeat block and averaged to the per-site loading probability.

Real reproduction: a virtual exp + camera + OccupancyProcessor run for `repeat` shots, then the occupied
signal is fed through the REAL sites-plot consumption path (`PanelCard._signal_then_repeat` ->
`reduce_repeat` -> `_coerce`), not a synthesized array.
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
                            repeat=REPEAT, free_run=True)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame"], "source": "value = signal"},
                             method="box", grid_shape=GRID)
    fire_live_imaging(exp)
    for _ in range(REPEAT + 2):
        cam.step(); det.step()
    return exp, cam, det


def test_occupancy_block_shapes_have_no_vestigial_middle_one():
    """`occupied` / `counts` are CLEAN `(repeat, n_sites)` (no middle 1); `frame_judged` is
    `(repeat, H, W)`; static geometry carries no repeat axis."""
    hub = SignalHub()
    exp, _, _ = _calibrated_occupancy(hub)
    try:
        occupied = np.asarray(hub.latest("occupied"))
        counts = np.asarray(hub.latest("counts"))
        frame_judged = np.asarray(hub.latest("frame_judged"))
        assert occupied.shape == (REPEAT, N_SITES)             # NO middle 1
        assert counts.shape == (REPEAT, N_SITES)
        assert frame_judged.ndim == 3 and frame_judged.shape[0] == REPEAT   # (repeat, H, W)
        # static / scalar outputs do NOT gain a repeat axis
        assert np.asarray(hub.latest("centers")).shape == (N_SITES, 2)
        assert np.asarray(hub.latest("thresholds")).shape == (N_SITES,)
        assert np.ndim(hub.latest("rate")) == 0
    finally:
        exp.close()


def test_reduce_repeat_core_ndim_averages_to_per_site_loading_probability():
    """`reduce_repeat((repeat, n_sites), 'average', core_ndim=1)` -> `(n_sites,)` = the mean over the
    N shots' 0/1 = the per-site LOADING PROBABILITY (structure-driven; a bare ndim heuristic would miss
    the ndim-2 block).  Pass-through when ndim == core_ndim."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data
    hub = SignalHub()
    exp, _, _ = _calibrated_occupancy(hub)
    try:
        occupied = np.asarray(hub.latest("occupied"))
        prob = reduce_repeat(occupied, "average", core_ndim=1)
        assert prob.shape == (N_SITES,)
        # it IS the mean over shots (nanmean over axis 0)
        np.testing.assert_allclose(prob, np.nanmean(occupied, axis=0))
        assert np.all((prob >= 0) & (prob <= 1))
        # how many shots have data -> the xN label
        assert repeats_with_data(occupied, core_ndim=1) == REPEAT
        # an already-reduced (n_sites,) value (ndim == core_ndim) passes through untouched
        passthrough = reduce_repeat(prob, "average", core_ndim=1)
        np.testing.assert_array_equal(passthrough, prob)
    finally:
        exp.close()


def test_reduce_repeat_legacy_ndim_fallback_is_byte_identical():
    """With `core_ndim=None` the EXISTING ndim>=3 rule is kept EXACTLY -- a camera `(repeat,1,H,W)`
    block still averages over axis 0, and a 2-D array is NOT treated as a repeat block (so undeclared
    camera/scan callers are unaffected)."""
    from Zou_lab_control.frontend.live import reduce_repeat
    rng = np.random.default_rng(0)
    cam_block = rng.random((4, 1, 6, 8))                       # camera (repeat, 1, H, W)
    out = reduce_repeat(cam_block, "average")                  # no core_ndim -> ndim>=3 path
    np.testing.assert_allclose(out, np.nanmean(cam_block, axis=0))
    assert out.shape == (1, 6, 8)
    # a 2-D array with NO core_ndim is NOT a repeat block -> passes through (legacy behaviour)
    flat = rng.random((4, 9))
    np.testing.assert_array_equal(reduce_repeat(flat, "average"), flat)


def test_describe_shape_renders_per_signal_structure():
    """`describe_shape` renders each occupancy signal off ITS declared structure: occupied as
    `5 × (35)`, frame_judged as `5 × (96×128)`, centers as the raw `(35, 2)` (static, no repeat)."""
    hub = SignalHub()
    exp, _, det = _calibrated_occupancy(hub)
    try:
        specs = {s.name: s for s in det.output_specs()}
        occ_spec = specs["occupied"]
        frame_spec = specs["frame_judged"]
        occupied = hub.latest("occupied")
        frame_judged = hub.latest("frame_judged")
        H, W = np.asarray(frame_judged).shape[1:]
        assert describe_shape(occupied, points_shape=occ_spec.points_shape,
                              data_shape=occ_spec.data_shape) == f"{REPEAT} × ({N_SITES})"
        assert describe_shape(frame_judged, points_shape=frame_spec.points_shape,
                              data_shape=frame_spec.data_shape) == f"{REPEAT} × ({H}×{W})"
        # centers declares NO structure -> raw shape, no repeat axis
        assert not specs["centers"].has_structure
        assert describe_shape(hub.latest("centers")) == f"({N_SITES}, 2)"
        # the per-signal core_ndim is the single source the plot collapses by
        assert occ_spec.core_ndim == 1 and frame_spec.core_ndim == 2
        assert specs["centers"].core_ndim is None
    finally:
        exp.close()


def test_sites_panel_renders_the_averaged_per_site_map_from_occupancy():
    """Reproduce the REAL sitemap consumption: a `sites` PanelCard bound to the `occupied` signal
    (default `value = signal`) runs `_signal_then_repeat` -> `reduce_repeat(core_ndim=1)` -> `_coerce`
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
                return {"points_shape": (), "data_shape": (N_SITES,), "grid_shape": (),
                        "core_ndim": 1, "per_signal": True}
            return None

        def sites_inputs_provider(occ):
            return ("centers", "frame_judged") if occ == "occupied" else (None, None)

        card = PanelCard(PanelConfig(kind="sites", inputs=["occupied"], source="value = signal"),
                         structure_provider=structure_provider,
                         sites_inputs_provider=sites_inputs_provider)
        namespace = dict(hub.snapshot_latest())
        # the panel pipeline: per-slice eval -> structure-driven repeat collapse
        reduced = card._signal_then_repeat(namespace)
        coerced = card._coerce(reduced)
        # it IS the per-site loading probability (mean over the REPEAT shots), one value per site
        expected = reduce_repeat(np.asarray(hub.latest("occupied")), "average", core_ndim=1)
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
