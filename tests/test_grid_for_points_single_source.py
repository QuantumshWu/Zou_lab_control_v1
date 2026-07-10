"""Logical point geometry never reshapes a signal's trailing data dimensions."""

from __future__ import annotations

import numpy as np

from Zou_lab_control.neutral_atom.operations.logic import grid_for_points, describe_shape


def test_grid_for_points_is_the_reshape_validity_precondition():
    # a 2-D scan: flat points (35,) the grid divides exactly -> the grid applies
    assert grid_for_points((5, 7), (35,)) == (5, 7)
    # Per-site data has one acquisition point; a 5x7 trap layout belongs to data,
    # so it cannot become a point grid.
    assert grid_for_points((5, 7), (1,)) == ()
    # a grid that does NOT divide the points -> not a valid reshape -> ()
    assert grid_for_points((5, 7), (30,)) == ()
    # no grid declared -> ()
    assert grid_for_points((), (35,)) == ()
    # robust to None / numpy ints
    assert grid_for_points(None, None) == ()
    assert grid_for_points((np.int64(5), np.int64(7)), (np.int64(35),)) == (5, 7)


def test_describe_shape_shares_the_same_rule_no_drift():
    # describe_shape must apply the IDENTICAL rule, not a weaker copy that trusts grid_shape blindly.
    # occupancy (R,1,n_sites): the physical P axis is NEVER a 5x7 grid (the grid does not
    # divide one point) -> flat site count, the literal P=1 kept (#iron-law, #H3v-3)
    assert describe_shape(np.zeros((10, 1, 35)), points_shape=(1,), data_shape=(35,), grid_shape=(5, 7)) == "10 × 1 × (35)"
    # a real 2-D scan whose grid divides the points -> the grid IS shown
    assert describe_shape(np.zeros((4, 35, 1)), points_shape=(35,), data_shape=(1,), grid_shape=(5, 7)) == "4 × 5×7 × (1)"
    # a grid that does NOT divide the points must fall back to the flat points -- the weaker pre-fix copy
    # would have wrongly printed "5×7" here; sharing grid_for_points makes it impossible.
    assert describe_shape(np.zeros((4, 30, 1)), points_shape=(30,), data_shape=(1,), grid_shape=(5, 7)) == "4 × 30 × (1)"


def test_console_signal_structure_reads_the_hub_schema_directly():
    import inspect
    from Zou_lab_control.frontend import task_console
    src = inspect.getsource(task_console.TaskConsole._signal_structure)
    assert "self.hub.schema" in src
    assert "signal_spec" not in src and "output_specs" not in src
