"""#H3x: AOD rearrangement DEVICE layer -- virtual actuator + real ramp builder.

Virtual: ``exp.devices.aod.apply_moves`` drags atoms in the SAME array the camera images, so a re-image
shows the assembled defect-free pattern (the lowest-layer physics fake).  Real: ``RampAOD.move_program``
builds a COMPILABLE DAC-bus ramp (src code -> ramp -> hold at dst code), so the hardware path won't error.
Both satisfy the SAME ``AODDevice.apply_moves`` contract (virtual==real, branch at the device).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.base import AODDevice  # noqa: E402
from Zou_lab_control.neutral_atom.devices.registry import load_devices  # noqa: E402
from Zou_lab_control.neutral_atom.devices.aod import RampAOD  # noqa: E402
from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer  # noqa: E402
from Zou_lab_control.neutral_atom.operations.rearrangement import (  # noqa: E402
    Move, target_sites, plan_rearrangement)

GRID = (5, 7)


def test_virtual_config_provides_an_aod_bound_to_the_imaged_array():
    dev = load_devices("virtual", open_devices=True)
    try:
        assert isinstance(dev.aod, AODDevice)
        # the AOD drags atoms in the SAME array the camera images (shared trap_array, virtual==real seam)
        assert dev.aod.trap_array is dev.trap_array
    finally:
        dev.close()


def test_virtual_aod_apply_moves_assembles_defect_free_array():
    dev = load_devices("virtual", open_devices=True)
    try:
        trap = dev.trap_array
        occ = np.zeros(trap.n_sites, dtype=bool)
        occ[[0, 6, 28, 34, 17]] = True                          # corners + centre
        trap.set_occupancy(occ)
        targets = target_sites(trap.grid_shape, n_target=5)
        plan = plan_rearrangement(trap.occupancy, trap.grid_shape, targets)
        dev.aod.apply_moves(plan.moves, survival=1.0)           # no loss -> exact assembly
        assert all(trap.occupancy[t] for t in targets), "virtual AOD must fill every target"
        assert trap.consume_pin(), "the rearranged array must be pinned for the next image"
    finally:
        dev.close()


def test_virtual_aod_per_move_loss_can_drop_atoms():
    dev = load_devices("virtual", overrides=None, open_devices=True)
    try:
        trap = dev.trap_array
        occ = np.zeros(trap.n_sites, dtype=bool)
        occ[[0, 6, 28, 34]] = True
        trap.set_occupancy(occ)
        plan = plan_rearrangement(trap.occupancy, trap.grid_shape, target_sites(trap.grid_shape, n_target=4))
        dev.aod.apply_moves(plan.moves, survival=0.0)           # every moved atom lost in transit
        moved_dst = {m.dst for m in plan.moves}
        assert not any(trap.occupancy[d] for d in moved_dst)
    finally:
        dev.close()


def test_ramp_aod_move_program_ramps_FROM_src_to_dst_and_compiles():
    """THREE periods: park at the SOURCE codes (so the ramp's carry-in is the source, not 0 V), ramp to
    the destination, settle holding the destination.  Regression: a 2-period (ramp, settle) program
    ramped from 0 V and dropped the atom on every move."""
    ramp = RampAOD(VirtualSequencer(), grid_shape=(5, 7))
    prog = ramp.move_program(6, 17)                            # site6 (col6,row0) -> site17 (col3,row2)
    prog.to_sequence(clock_hz=50e6)                            # compiles -> hardware won't error
    px = prog.analog_bus_plan("aod_x")
    py = prog.analog_bus_plan("aod_y")
    assert px[0]["mode"] == "edge" and px[0]["value"] == ramp.x_codes[6]   # PARK at source column code
    assert px[1]["mode"] == "ramp" and px[1]["value"] == ramp.x_codes[3]   # RAMP from src -> dst column
    assert px[2]["value"] == ramp.x_codes[3]                               # SETTLE holds dst
    assert py[0]["value"] == ramp.y_codes[0] and py[1]["mode"] == "ramp" and py[1]["value"] == ramp.y_codes[2]


def test_ramp_aod_validates_dac_codes_at_construction():
    """A site->code map exceeding the signed DAC range fails at CONSTRUCTION (config/connect time),
    not lazily on the first move that touches the out-of-range edge site mid-experiment."""
    import pytest
    with pytest.raises(ValueError, match="outside the signed"):
        RampAOD(VirtualSequencer(), grid_shape=(1, 14), x_code_step=40.0)  # col 13 -> 520 > 511


def test_rearrange_task_move_survival_is_optional_not_a_sentinel():
    """move_survival is a clean Optional[float] (NO -1 in-band sentinel): None = use the AOD device's own
    default (apply_moves with no survival kwarg); a value in [0,1] overrides it; out-of-range raises."""
    import pytest
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import RearrangeTask
    assert RearrangeTask(SignalHub(), camera=None, aod=None, move_survival=None).move_survival is None
    assert RearrangeTask(SignalHub(), camera=None, aod=None).move_survival is None      # default = device default
    assert RearrangeTask(SignalHub(), camera=None, aod=None, move_survival=0.9).move_survival == 0.9
    with pytest.raises(ValueError, match="must be in"):
        RearrangeTask(SignalHub(), camera=None, aod=None, move_survival=-1.0)            # no sentinel: -1 is illegal
    # acquisition_parameters round-trips None as None (NOT -1.0)
    assert RearrangeTask(SignalHub(), camera=None, aod=None).acquisition_parameters()["move_survival"] is None


def test_rearrange_task_explicit_targets_override_count_and_layout():
    """An explicit ``targets`` site list (GUI text field / list) assembles exactly those sites and is
    parsed from a comma/space string; blank falls back to target_count + layout."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import RearrangeTask
    assert RearrangeTask(SignalHub(), camera=None, aod=None, targets="0 1 2, 7 8").targets == (0, 1, 2, 7, 8)
    assert RearrangeTask(SignalHub(), camera=None, aod=None, targets=[3, 4]).targets == (3, 4)
    assert RearrangeTask(SignalHub(), camera=None, aod=None).targets == ()              # blank = central-block default


def test_ramp_aod_apply_moves_fires_one_program_per_move():
    seq = VirtualSequencer()
    ramp = RampAOD(seq, grid_shape=(5, 7))
    ramp.apply_moves([Move(0, 17), Move(6, 18), Move(34, 24)])
    fires = sum(1 for row in seq.history if row["action"] == "fire")
    assert fires == 3, "one ramp program fired per move (sequential single-tweezer)"


# -- end-to-end (the user's "virtual must SEE the defect-free array") --------------------
def _calibrated_virtual_session():
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=12, display=False)
    exp.readout.thresholds(frames=80, display=False)
    return exp


def test_rearrange_task_is_auto_discovered():
    exp = _calibrated_virtual_session()
    names = [spec.name for spec in exp.readout.task_specs()]
    assert "Rearrange array" in names, "the rearrange task must appear in the Add-Panel Task group"


def test_post_rearrange_image_does_not_reload(monkeypatch):
    """Real-correctness invariant: the INITIAL image loads a fresh array (load=True) but the
    post-rearrange image must NOT re-cool a new array on top of the assembled atoms (load=False).
    On the virtual camera the pin masks this; on real hardware a reload would destroy the array."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import RearrangeTask

    loads = []

    class _Det:
        occupied = np.zeros(GRID[0] * GRID[1], dtype=bool)

    class _Cal:
        def detect(self, frame):
            return _Det()

    class _AOD:
        def apply_moves(self, moves, *, survival=None):
            pass

    def frame_provider(exposure, load):
        loads.append(load)
        return np.zeros((8, 8), dtype=float)

    task = RearrangeTask(SignalHub(), camera=None, aod=_AOD(), grid_shape=GRID,
                         frame_provider=frame_provider, calibration_provider=_Cal)
    task.run_to_completion()
    assert loads == [True, False], f"expected [load, no-reload], got {loads}"


def test_end_to_end_virtual_rearrange_assembles_defect_free_array():
    """The whole flow on the virtual backend: calibrate -> load sparsely -> exp.rearrange.execute() ->
    the re-image SEES a filled defect-free target (fill fraction == 1 at perfect transfer).

    This checks the ASSEMBLY LOGIC, so it runs with the shot-to-shot brightness jitter OFF -> a clean,
    deterministic readout -> detection is perfect and the only thing under test is the plan+move+re-image
    chain.  The realistic (noisy) readout is exercised separately by
    ``test_virtual_readout_has_realistic_per_site_spread_and_psf_beats_box``."""
    exp = _calibrated_virtual_session()
    trap = exp.devices.trap_array
    trap.shot_brightness_jitter = 0.0                      # clean readout: isolate the assembly logic
    occ = np.zeros(trap.n_sites, dtype=bool)
    occ[[0, 6, 28, 34, 3, 21, 13]] = True                  # 7 scattered atoms
    trap.set_occupancy(occ)
    res = exp.rearrange.execute(target_count=6, move_survival=1.0)
    assert res["n_loaded"] >= 6 and res["fill_fraction"] >= 0.99, res
    assert res["n_filled"] == res["n_target"] == 6
    # occupancy_after must actually show atoms ON the target sites (the camera re-image sees it)
    occ_after = np.asarray(res["occupancy_after"], dtype=bool)
    assert int(occ_after.sum()) >= 6


def test_virtual_readout_has_realistic_per_site_spread_and_psf_beats_box():
    """#3: the virtual readout reproduces the REAL Rb87 per-site behaviour (no fake): each site has its
    OWN brightness + spot shape, so (a) the per-site BRIGHT signal is NON-uniform (a real spread, not a
    flat level), and (b) a per-site PSF matched filter beats a box -- exactly what was lost when the
    per-site spread "disappeared".  Reproduced through the SAME calibration+readout path real data uses."""
    import numpy as np
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualTrapArray
    from Zou_lab_control.neutral_atom.operations.calibration import calibrate_all_methods_from_images
    from Zou_lab_control.neutral_atom.core.bimodal import fit_bimodal
    trap = VirtualTrapArray(grid_shape=GRID)
    expo = 3e-3
    R = lambda o: trap._render(o, np.where(o, expo, 0.0), expo).astype(float)
    ref = [R(np.ones(trap.n_sites, bool)) for _ in range(16)]
    readout = [R(trap.rng.random(trap.n_sites) < 0.5) for _ in range(120)]
    cal = calibrate_all_methods_from_images(ref, readout, grid_shape=GRID, exposure=expo)

    def per_site(method):
        sig, lab = [], []
        for _ in range(120):
            o = trap.rng.random(trap.n_sites) < 0.5
            sig.append(cal.signals(R(o), method=method)); lab.append(o)
        sig = np.array(sig); lab = np.array(lab)
        fids = np.array([fit_bimodal(sig[:, i]).fidelity for i in range(trap.n_sites)])
        # per-site NET bright signal = mean(bright shots) - mean(dark shots): the real per-site
        # brightness with the constant offset/background removed (the quantity a per-site PSF / the
        # background-subtracted readout actually shows -- NOT the raw box value, which the offset floods).
        net = np.array([sig[lab[:, i], i].mean() - sig[~lab[:, i], i].mean() for i in range(trap.n_sites)])
        return fids[np.isfinite(fids)], net

    fid_box, net_box = per_site("box")
    fid_psf, _ = per_site("psf")
    # (a) per-site NET bright signal is genuinely non-uniform (NOT a flat level): each tweezer has its own
    #     brightness + shape -- this is the "counts per site 不均匀" the user said had disappeared.
    assert (net_box.max() - net_box.min()) / net_box.mean() > 0.20, "per-site bright signal must vary (non-flat)"
    # (b) per-site fidelity is a SPREAD, not a flat value, and lands in the realistic Rb87 band
    assert fid_box.std() > 0.01 and 0.80 < fid_box.mean() < 0.999, (fid_box.mean(), fid_box.std())
    # (c) a per-site PSF matched filter beats a box (the per-site-PSF advantage the regression killed)
    assert fid_psf.mean() >= fid_box.mean(), (fid_psf.mean(), fid_box.mean())
