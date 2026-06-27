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


def test_ramp_aod_move_program_ramps_src_to_dst_and_compiles():
    ramp = RampAOD(VirtualSequencer(), grid_shape=(5, 7))
    prog = ramp.move_program(0, 17)                             # corner (col0,row0) -> centre (col3,row2)
    prog.to_sequence(clock_hz=50e6)                             # compiles -> hardware won't error
    px = prog.analog_bus_plan("aod_x")
    py = prog.analog_bus_plan("aod_y")
    assert px[0]["mode"] == "ramp" and px[0]["value"] == ramp.x_codes[3]   # ramp X to dst column code
    assert py[0]["mode"] == "ramp" and py[0]["value"] == ramp.y_codes[2]   # ramp Y to dst row code
    assert px[1]["value"] == ramp.x_codes[3] and py[1]["value"] == ramp.y_codes[2]  # settle HOLDS dst


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
    the re-image SEES a filled defect-free target (fill fraction == 1 at perfect transfer)."""
    exp = _calibrated_virtual_session()
    trap = exp.devices.trap_array
    occ = np.zeros(trap.n_sites, dtype=bool)
    occ[[0, 6, 28, 34, 3, 21, 13]] = True                  # 7 scattered atoms
    trap.set_occupancy(occ)
    res = exp.rearrange.execute(target_count=6, move_survival=1.0)
    assert res["n_loaded"] >= 6 and res["fill_fraction"] >= 0.99, res
    assert res["n_filled"] == res["n_target"] == 6
    # occupancy_after must actually show atoms ON the target sites (the camera re-image sees it)
    occ_after = np.asarray(res["occupancy_after"], dtype=bool)
    assert int(occ_after.sum()) >= 6
