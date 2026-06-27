"""#H3x: AOD rearrangement PLANNER (pure analysis layer) -- defect-free array assembly.

Pins the assignment core ported from the lab reference (compress_algorithm.LSAP_compressed): center-sorted
Hungarian assignment + a collision-safe sequential move order, with an atomic occupancy-remap oracle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations.rearrangement import (  # noqa: E402
    Move, target_sites, plan_rearrangement, apply_moves_to_occupancy)

GRID = (5, 7)
N = GRID[0] * GRID[1]


def test_target_sites_center_is_compact_and_central():
    tg = target_sites(GRID, n_target=9, layout="center")
    assert len(tg) == 9 and len(set(tg)) == 9
    # the central site (row 2, col 3 -> index 17) must be in a 9-site centred target
    assert 2 * GRID[1] + 3 in tg
    # default = half the array
    assert len(target_sites(GRID)) == N // 2
    # "all" = every site
    assert set(target_sites(GRID, layout="all")) == set(range(N))


def test_feasible_load_assembles_a_defect_free_array():
    """A loading with enough atoms -> after applying the plan (no loss) EVERY target holds an atom."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        occ = rng.random(N) < 0.55
        tg = target_sites(GRID, n_target=10, layout="center")
        plan = plan_rearrangement(occ, GRID, tg)
        if not plan.feasible:
            continue
        final = apply_moves_to_occupancy(occ, plan.moves, survival=1.0)
        assert all(final[t] for t in tg), f"trial {trial}: target not fully filled"
        # an atom already on a target is NOT moved (efficiency)
        for m in plan.moves:
            assert m.src not in plan.already_filled


def test_moves_are_collision_safe_in_sequence():
    """Executing the moves IN ORDER (one moving tweezer) never drops an atom onto an occupied site:
    at each step the destination is empty in the running occupancy."""
    rng = np.random.default_rng(1)
    occ = rng.random(N) < 0.6
    plan = plan_rearrangement(occ, GRID, target_sites(GRID, n_target=12))
    state = np.asarray(occ, dtype=bool).copy()
    for m in plan.moves:
        assert state[m.src], "source must hold an atom when its move runs"
        assert not state[m.dst], "destination must be EMPTY when the move runs (collision-safe order)"
        state[m.src] = False
        state[m.dst] = True


def test_infeasible_load_reports_shortfall_and_fills_what_it_can():
    occ = np.zeros(N, dtype=bool)
    occ[[0, 1, 2]] = True                                       # only 3 atoms
    plan = plan_rearrangement(occ, GRID, target_sites(GRID, n_target=10))
    assert not plan.feasible and plan.shortfall == 7 and plan.n_loaded == 3


def test_per_move_loss_drops_atoms_in_transit():
    occ = np.zeros(N, dtype=bool)
    occ[[0, 6, 28, 34]] = True                                  # corners -> must all move inward
    plan = plan_rearrangement(occ, GRID, target_sites(GRID, n_target=4))
    assert len(plan.moves) >= 1
    # survival=0 -> every MOVED atom is lost (only never-moved already-on-target atoms could remain)
    final = apply_moves_to_occupancy(occ, plan.moves, survival=0.0, rng=np.random.default_rng(3))
    moved_dst = {m.dst for m in plan.moves}
    assert not any(final[d] for d in moved_dst), "survival=0 must lose every atom moved in transit"
