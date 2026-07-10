"""The scan-table template must (1) AUTO-SCALE to any slot count and (2) seed each column with a
KIND-APPROPRIATE sweep -- a DAC column over its signed INTEGER code range, a duration column over a
ns range -- so a DAC slot is never given a duration's ns sweep (where every point clamps to +511).
Plus the duration FLOOR: ``set_period_duration`` can never store a 0 / sub-tick duration (which the
driver rejects as a degenerate zero-length period -- the user's "unexpected error").

The ONE template generator (`scan_table_template`, shared by the pulse GUI Scan tab + the task-console
Pulse-scan form) used to special-case only 1/2 slots and put inline `# s{j}` comments INSIDE a
single-line list literal, so the 3+-slot template was a hard SyntaxError; that regression is also
pinned here.
"""
from Zou_lab_control.neutral_atom.ports import PortCatalog

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.timing.pulse_table import (
    PulsePeriod, PulseTableState, scan_column_spec, scan_table_template)


def _exec_template(src: str) -> dict:
    ns: dict = {}
    exec(compile(src, "<scan_table_template>", "exec"), {"np": np}, ns)  # noqa: S102 (operator-authored starter)
    return ns


def _duration_specs(n: int) -> list:
    return [scan_column_spec(f"s{i}", "duration", nominal=200.0, unit="ns") for i in range(n)]


@pytest.mark.parametrize("kind", ["column_stack", "grid"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_template_is_valid_python_and_has_n_columns(kind, n):
    """Every (kind, n_slots) emits runnable Python whose `scan_table` is (N_points x n) -- one column
    per bound slot.  This is the regression the 3+-slot SyntaxError broke."""
    ns = _exec_template(scan_table_template(kind, _duration_specs(n)))
    table = np.asarray(ns["scan_table"])
    assert table.ndim == 2, f"{kind} n={n}: scan_table must be 2-D, got {table.shape}"
    assert table.shape[1] == n, f"{kind} n={n}: must have one column per slot, got {table.shape[1]}"
    assert table.shape[0] >= 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_grid_carries_a_matching_scan_shape(n):
    """The grid template is a real N-D grid: it declares a `scan_shape` whose product is the point
    count (so a 2-D grid can render as a scan map), with one axis length per slot."""
    ns = _exec_template(scan_table_template("grid", _duration_specs(n)))
    table = np.asarray(ns["scan_table"])
    shape = tuple(int(s) for s in ns["scan_shape"])
    assert len(shape) == n, f"grid n={n}: scan_shape must have one axis per slot"
    assert int(np.prod(shape)) == table.shape[0], "grid point count must equal prod(scan_shape)"


def test_column_stack_three_slots_is_not_a_syntax_error():
    """The exact failure mode that motivated the auto-scale: the old 3-slot column_stack template
    raised SyntaxError because of an inline comment inside a one-line list."""
    src = scan_table_template("column_stack", _duration_specs(3))
    compile(src, "<scan_table_template>", "exec")        # must NOT raise
    assert src.count("scan_table") >= 1
    # comments stay on their own lines -- never an inline '#' inside the column_stack([...]) literal
    stack_line = next(ln for ln in src.splitlines() if "np.column_stack([" in ln)
    assert "#" not in stack_line


def test_dac_column_sweeps_signed_codes_not_a_duration_ns_range():
    """The fix: a DAC slot's column is an INTEGER sweep over its signed code range (-512..511 for a
    10-bit bus, 0 = 0 V), NOT the ns range a duration gets -- so a +-512 DAC is never fed a duration's
    range (where every point would collapse to the +511 clamp)."""
    specs = [scan_column_spec("s0", "duration", nominal=200.0, unit="ns"),
             scan_column_spec("s1", "dac", unit="code")]                       # default 10-bit -> +-512
    src = scan_table_template("column_stack", specs)
    table = np.asarray(_exec_template(src)["scan_table"])
    dac_col, dur_col = table[:, 1], table[:, 0]
    assert dac_col.min() >= -512 and dac_col.max() <= 511                      # within the signed range
    assert np.allclose(dac_col, np.round(dac_col))                            # integer codes
    assert dac_col.max() > 200                                                # spans the range, not a clamp-to-511 collapse
    assert dur_col.max() > 511                                                # the duration column is a DIFFERENT (ns) range
    dac_line = next(ln for ln in src.splitlines() if ln.startswith("s1"))
    assert ".round().astype(int)" in dac_line                                # DAC line rounds to int
    assert ".round().astype(int)" not in next(ln for ln in src.splitlines() if ln.startswith("s0"))


def test_set_period_duration_floors_a_zero_or_subtick_to_one_tick():
    """A 0 / sub-tick LITERAL duration can never be stored -- it is floored to exactly one tick (the
    api-sweep path sets durations point-by-point with NO table snap, so this is the structural guard
    against a 0-length period reaching the driver).  A negative literal still raises."""
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["ch00", "t"], channel_labels={"ch00": "da[0]"}), visible_ports=["ch00", "t"], time_step_ns=20.0,
                         periods=[PulsePeriod(200, (0, 1), unit="ns")])
    st.set_period_duration(0, 0.0)
    assert float(st.periods[0].duration) == 20.0                              # 0 -> 1 tick (20 ns)
    st.set_period_duration(0, 5.0)                                            # sub-tick -> 1 tick
    assert float(st.periods[0].duration) == 20.0
    st.set_period_duration(0, 1000.0)                                         # a >= 1-tick value is untouched
    assert float(st.periods[0].duration) == 1000.0
    with pytest.raises(ValueError):
        st.set_period_duration(0, -40.0)                                      # a negative literal still raises
