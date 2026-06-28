"""``grid_for_points`` is the SINGLE source of "may this signal's flat swept-points axis be reshaped into
a 2-D map".  Every consumer -- ``describe_shape`` (the legend), the console's ``_signal_structure`` (what a
panel may reshape) and ``_coerce`` (the reshape itself) -- must share it, so the rule cannot drift.

Root cause this guards (a7c031c -> made systematic): a node may carry a ``grid_shape`` that is NOT a
swept-points grid -- an OccupancyProcessor stores ``grid_shape=(5,7)`` as the physical TRAP
LAYOUT.  Reshape eligibility is the dimensional FACT ``points non-empty and prod(grid)==prod(points)``, so
that trap grid is never smeared onto the per-site ``occupied`` (points=()) and imshow'd as a (5x7) heatmap
-- and the SAME fact, not a per-node special case, decides it in the legend and the reshape too.
"""

from __future__ import annotations

import numpy as np

from Zou_lab_control.neutral_atom.operations.logic import grid_for_points, describe_shape


def test_grid_for_points_is_the_reshape_validity_precondition():
    # a 2-D scan: flat points (35,) the grid divides exactly -> the grid applies
    assert grid_for_points((5, 7), (35,)) == (5, 7)
    # per-site DATA (occupancy points=()) -> prod 1 != 35 -> never a grid (the trap layout is not a points grid)
    assert grid_for_points((5, 7), ()) == ()
    # a grid that does NOT divide the points -> not a valid reshape -> ()
    assert grid_for_points((5, 7), (30,)) == ()
    # no grid declared -> ()
    assert grid_for_points((), (35,)) == ()
    # robust to None / numpy ints
    assert grid_for_points(None, None) == ()
    assert grid_for_points((np.int64(5), np.int64(7)), (np.int64(35),)) == (5, 7)


def test_describe_shape_shares_the_same_rule_no_drift():
    # describe_shape must apply the IDENTICAL rule, not a weaker copy that trusts grid_shape blindly.
    # occupancy (repeat, n_sites) with points=() -> flat site count, NEVER a 5x7 grid string
    assert describe_shape(np.zeros((10, 35)), points_shape=(), data_shape=(35,), grid_shape=(5, 7)) == "10 × (35)"
    # a real 2-D scan whose grid divides the points -> the grid IS shown
    assert describe_shape(np.zeros((4, 35, 1)), points_shape=(35,), data_shape=(1,), grid_shape=(5, 7)) == "4 × 5×7 × (1)"
    # a grid that does NOT divide the points must fall back to the flat points -- the weaker pre-fix copy
    # would have wrongly printed "5×7" here; sharing grid_for_points makes it impossible.
    assert describe_shape(np.zeros((4, 30, 1)), points_shape=(30,), data_shape=(1,), grid_shape=(5, 7)) == "4 × 30 × (1)"


def test_console_signal_structure_reuses_the_same_helper():
    # The console must derive a panel's reshape grid from the SAME helper -- not re-implement the predicate.
    import inspect
    from Zou_lab_control.frontend import task_console
    src = inspect.getsource(task_console.TaskConsole._signal_structure)
    assert "grid_for_points" in src, "_signal_structure must route through the shared grid_for_points helper"
