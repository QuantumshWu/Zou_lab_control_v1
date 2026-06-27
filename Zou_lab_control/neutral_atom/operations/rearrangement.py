"""AOD atom rearrangement -- defect-free array assembly PLANNING (pure analysis layer).

Given a per-site occupancy (which tweezers loaded an atom, from the readout) and a TARGET set of
sites (the defect-free pattern to fill), compute the ordered list of single-atom MOVES that drag loaded
atoms onto the targets, minimising total displacement.  This is the host-side plan; a device (real AOD
or the virtual trap array) EXECUTES it.

The algorithm mirrors the lab's reference ``compress_algorithm.LSAP_compressed``
(references/source_archives/PythonCamDemo): sort both source and target sites by distance from the array
centre (move central atoms first -- they are the most stable and least collision-prone), then solve the
assignment with the Hungarian method (``scipy.optimize.linear_sum_assignment``) over the Manhattan-
distance cost, keeping atoms already on a target in place.  The moves are then ordered so each move's
destination is EMPTY when it runs (a collision-safe SEQUENTIAL single-tweezer schedule).  The reference's
extra per-move waypoint path-collision optimisation (X-then-Y vs Y-then-X) is a real-hardware refinement
that does not change which atom goes where; it is noted as a follow-up -- the move ASSIGNMENT here is the
faithful core, and the virtual occupancy remap is atomic so it is exact without paths.

Dependency-free + backend-free (the virtual==real contract): this never imports a device or reads sim
ground truth -- it plans from the occupancy the readout already produced, and the SAME plan drives the
virtual trap array and the real AOD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.analysis import grid_shape_tuple


@dataclass(frozen=True)
class Move:
    """One single-atom move: drag the atom from source site ``src`` to destination ``dst`` (both are
    FLAT row-major site indices, ``site = row * n_cols + col``)."""

    src: int
    dst: int


@dataclass(frozen=True)
class RearrangePlan:
    """The result of :func:`plan_rearrangement`.

    ``moves``          the ordered, collision-safe single-atom moves to execute.
    ``targets``        the target (defect-free) site set, flat indices.
    ``loaded``         the loaded (occupied) source sites the plan saw.
    ``already_filled`` targets already holding an atom (left in place, not moved).
    ``feasible``       whether enough atoms loaded to fill EVERY target.
    ``shortfall``      targets that cannot be filled (max(0, n_target - n_loaded)).
    """

    moves: tuple[Move, ...]
    targets: tuple[int, ...]
    loaded: tuple[int, ...]
    already_filled: tuple[int, ...]
    n_target: int
    n_loaded: int
    feasible: bool
    shortfall: int


def _rc(site: int, n_cols: int) -> tuple[int, int]:
    return (int(site) // int(n_cols), int(site) % int(n_cols))


def target_sites(grid_shape, n_target: int | None = None, layout: str = "center") -> tuple[int, ...]:
    """The flat site indices of the defect-free TARGET pattern to fill.

    ``layout="center"`` (default): the ``n_target`` sites CLOSEST to the array centre -- the natural
    defect-free goal (a filled central block), and the order central atoms should be assembled in.
    ``n_target=None`` defaults to HALF the array (floor(n_sites/2)), which a ~50 %-loaded array can
    typically fill.  ``layout="all"`` targets every site (a fully-filled array; only feasible at high
    loading)."""
    ny, nx = grid_shape_tuple(grid_shape)
    n_sites = ny * nx
    if n_target is None:
        n_target = n_sites if layout == "all" else n_sites // 2
    n_target = int(max(0, min(int(n_target), n_sites)))
    if layout == "all" or n_target >= n_sites:
        return tuple(range(n_sites))
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    order = sorted(range(n_sites), key=lambda s: ((_rc(s, nx)[0] - cy) ** 2 + (_rc(s, nx)[1] - cx) ** 2, s))
    return tuple(sorted(order[:n_target]))


def plan_rearrangement(occupied, grid_shape, targets) -> RearrangePlan:
    """Plan the moves that fill ``targets`` from the loaded sites in ``occupied`` (a length-n_sites
    boolean, row-major).  Center-sorted Hungarian assignment over Manhattan distance + a collision-safe
    sequential ordering (see module docstring)."""
    ny, nx = grid_shape_tuple(grid_shape)
    n_sites = ny * nx
    occ = np.asarray(occupied, dtype=bool).reshape(-1)
    if occ.size != n_sites:
        raise ValueError(f"occupied must have {n_sites} entries (grid {ny}x{nx}), got {occ.size}.")
    target_set = sorted({int(t) for t in targets if 0 <= int(t) < n_sites})
    loaded = [int(s) for s in np.flatnonzero(occ)]
    n_target, n_loaded = len(target_set), len(loaded)

    already = sorted(set(loaded) & set(target_set))            # atoms already on a target -> stay put
    free_targets = [t for t in target_set if t not in set(already)]
    movable = [s for s in loaded if s not in set(already)]

    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    def _centre_key(s):
        r, c = _rc(s, nx)
        return ((r - cy) ** 2 + (c - cx) ** 2, s)
    # central-first: assemble the centre of the array first (reference LSAP_compressed ordering)
    movable.sort(key=_centre_key)
    free_targets.sort(key=_centre_key)

    moves: list[Move] = []
    if movable and free_targets:
        # Hungarian on the Manhattan-distance cost (sources x targets); excess sources stay unused.
        cost = np.empty((len(movable), len(free_targets)), dtype=float)
        for i, s in enumerate(movable):
            sr, sc = _rc(s, nx)
            for j, t in enumerate(free_targets):
                tr, tc = _rc(t, nx)
                cost[i, j] = abs(sr - tr) + abs(sc - tc)
        rows, cols = linear_sum_assignment(cost)
        assigned = [(movable[i], free_targets[j]) for i, j in zip(rows, cols)]
        # Collision-safe SEQUENTIAL order: only execute a move whose dst is currently empty, so a
        # single moving tweezer never drops an atom onto an occupied site.  Repeatedly drain ready
        # moves; a leftover cycle (dst still blocked by a pending src) is broken by forcing the
        # smallest-index move (rare for a centre-out defect-free fill, where targets are mostly empty).
        state = occ.copy()
        pending = list(assigned)
        while pending:
            ready = [m for m in pending if not state[m[1]]]
            if not ready:
                ready = [pending[0]]                          # break a cycle (forced move)
            for src, dst in ready:
                moves.append(Move(int(src), int(dst)))
                state[src] = False
                state[dst] = True
                pending.remove((src, dst))

    return RearrangePlan(
        moves=tuple(moves),
        targets=tuple(target_set),
        loaded=tuple(loaded),
        already_filled=tuple(already),
        n_target=n_target,
        n_loaded=n_loaded,
        feasible=n_loaded >= n_target,
        shortfall=max(0, n_target - n_loaded),
    )


def apply_moves_to_occupancy(occupied, moves, *, survival=1.0, rng=None):
    """Apply ``moves`` to a COPY of the boolean ``occupied`` array and return the new occupancy -- the
    pure model of executing a rearrangement: each move relocates its atom src->dst, lost in transit with
    probability ``1 - survival`` (then the atom is gone from BOTH sites).  Used by the virtual trap array
    (the lowest-layer physics fake) AND by tests/oracles, so the predicted defect-free array is one
    function everywhere.  Atomic + injective (the plan guarantees each dst is filled at most once)."""
    occ = np.asarray(occupied, dtype=bool).reshape(-1).copy()
    survival = float(survival)
    draw = (np.random.default_rng() if rng is None else rng)
    for m in moves:
        src, dst = int(m.src), int(m.dst)
        if not occ[src]:
            continue                                          # nothing to move (lost on a prior step)
        occ[src] = False
        if survival >= 1.0 or draw.random() < survival:
            occ[dst] = True                                   # atom arrives; else lost in transit
    return occ
