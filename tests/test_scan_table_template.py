"""#H3t-1: the scan-table template must AUTO-SCALE to any slot count.

The ONE template generator (`scan_table_template`, shared by the pulse GUI Scan tab + the task-console
Pulse-scan form) used to special-case only 1/2 slots; the 3+-slot `column_stack` branch put inline
`# s{j}` comments INSIDE a single-line list literal, so the first `#` swallowed the rest of the list and
the generated code was a hard SyntaxError.  A 3-or-more-slot pulse therefore got a BROKEN template, not
the per-slot template the operator asked for.  Pin that every slot count, both kinds, produces VALID
Python that builds an (N_points x n_slots) array.
"""

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.timing.pulse_table import scan_table_template


def _exec_template(src: str) -> dict:
    ns: dict = {}
    exec(compile(src, "<scan_table_template>", "exec"), {"np": np}, ns)  # noqa: S102 (operator-authored starter)
    return ns


@pytest.mark.parametrize("kind", ["column_stack", "grid"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_template_is_valid_python_and_has_n_columns(kind, n):
    """Every (kind, n_slots) emits runnable Python whose `scan_table` is (N_points x n) -- one column
    per bound slot.  This is the regression the 3+-slot SyntaxError broke."""
    ns = _exec_template(scan_table_template(kind, n))
    table = np.asarray(ns["scan_table"])
    assert table.ndim == 2, f"{kind} n={n}: scan_table must be 2-D, got {table.shape}"
    assert table.shape[1] == n, f"{kind} n={n}: must have one column per slot, got {table.shape[1]}"
    assert table.shape[0] >= 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_grid_carries_a_matching_scan_shape(n):
    """The grid template is a real N-D grid: it declares a `scan_shape` whose product is the point
    count (so a 2-D grid can render as a scan map), with one axis length per slot."""
    ns = _exec_template(scan_table_template("grid", n))
    table = np.asarray(ns["scan_table"])
    shape = tuple(int(s) for s in ns["scan_shape"])
    assert len(shape) == n, f"grid n={n}: scan_shape must have one axis per slot"
    assert int(np.prod(shape)) == table.shape[0], "grid point count must equal prod(scan_shape)"


def test_column_stack_three_slots_is_not_a_syntax_error():
    """The exact failure mode that motivated this: the old 3-slot column_stack template raised
    SyntaxError ('[' was never closed) because of an inline comment inside a one-line list."""
    src = scan_table_template("column_stack", 3)
    compile(src, "<scan_table_template>", "exec")        # must NOT raise
    assert src.count("scan_table") >= 1
    # comments stay on their own lines -- never an inline '#' inside the column_stack([...]) literal
    stack_line = next(ln for ln in src.splitlines() if "np.column_stack([" in ln)
    assert "#" not in stack_line
