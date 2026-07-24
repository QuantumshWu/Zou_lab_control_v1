"""The starter scan-table program: one column model, shared by every scan editor.

This is DESCRIPTION, not domain.  ``ScanColumnSpec`` is five scalars and
``scan_table_template`` turns a list of them into Python source text - no port
catalog, no streamer geometry, no state.  It lived in ``pulse_table`` only
because that is where scan slots live, which meant the pulse GUI's Scan tab and
the task console's Pulse-scan form both had to reach into a three-thousand line
timing module to seed an editor with text.

Its sibling ``scan_column_spec`` did NOT come along, and the split is the point.
That one derives a per-kind default sweep from the bus signed range and the clock
tick, so it needs the pulse geometry and stays with it; the render layer receives
the finished columns instead of building them.  Relocate what is pure; invert
what is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from typing import Sequence

__all__ = ["ScanColumnSpec", "scan_table_template"]


@dataclass(frozen=True)
class ScanColumnSpec:
    """One column of a starter scan/api template: the column variable name + a KIND-APPROPRIATE
    default sweep range.  A ``dac`` column sweeps INTEGER codes over the bus's signed range
    (``0`` = 0 V); a time column sweeps its unit (ns by default), ``>= 1`` tick.  This is why a
    DAC slot no longer inherits a duration's ns range -- the bug the operator hit, where a +-512
    DAC column was seeded with a ``20 .. 200000`` ns sweep (every point then clamped to +511)."""

    name: str
    lo: float
    hi: float
    is_dac: bool = False
    unit: str = "ns"
    label: str = ""
    quantum: float | None = None

    def __post_init__(self) -> None:
        for field in ("lo", "hi"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"ScanColumnSpec.{field} must be finite numeric")
        if float(self.hi) < float(self.lo):
            raise ValueError("ScanColumnSpec.hi must not be below lo")
        if self.is_dac:
            if self.quantum is not None:
                raise ValueError("a DAC scan column cannot carry a time quantum")
            return
        quantum = self.quantum
        if (
            isinstance(quantum, bool)
            or not isinstance(quantum, (int, float))
            or not math.isfinite(float(quantum))
            or float(quantum) <= 0.0
        ):
            raise ValueError(
                "a duration scan column requires its positive native-unit quantum"
            )
        object.__setattr__(self, "quantum", float(quantum))


def _duration_tick_bounds(spec: ScanColumnSpec) -> tuple[int, int]:
    """Return exact integer bounds from domain-authored physical values."""

    assert spec.quantum is not None
    quantum = Fraction(str(spec.quantum))
    ratios = tuple(Fraction(str(value)) / quantum for value in (spec.lo, spec.hi))
    if any(ratio.denominator != 1 for ratio in ratios):
        raise ValueError(
            f"duration column {spec.name!r} bounds are not multiples of its quantum"
        )
    lo_ticks, hi_ticks = (ratio.numerator for ratio in ratios)
    if lo_ticks < 1:
        raise ValueError(f"duration column {spec.name!r} must start at one tick")
    return lo_ticks, hi_ticks


def _duration_decimal_places(spec: ScanColumnSpec) -> int:
    """Digits needed to recover canonical decimal tick values after NumPy math."""

    assert spec.quantum is not None
    exponent = Decimal(str(spec.quantum)).normalize().as_tuple().exponent
    return max(0, -int(exponent))


def scan_table_template(kind: str, columns: Sequence[ScanColumnSpec]) -> str:
    """Starter Python for a scan-table program -- the ONE scan model, shared by the pulse GUI
    Scan tab and the task-console Pulse-scan form.

    The program builds an ``(N_points x n_cols)`` array and assigns it to ``scan_table``: one ROW
    per scan point, one COLUMN per bound slot.  The whole table is one object, so the slots advance
    together (lockstep); correlations (anti-correlated, grid, loaded array) are just different ways
    of building that one array.  Each column is seeded by its slot's KIND (``columns`` carries the
    per-slot range): a DAC column sweeps integer codes over its signed range, a duration column
    sweeps ns ticks bracketing the nominal -- so a DAC slot is NOT given a duration's ns range.

    * ``column_stack`` (default): one independent column per slot.
    * ``grid``: every combination (outer product) of per-axis arrays.
    """

    cols = list(columns)
    if not cols:
        raise ValueError(
            "scan_table_template requires at least one domain-owned column"
        )
    n = len(cols)

    def sweep(spec: ScanColumnSpec, size) -> str:
        if spec.is_dac:
            return f"np.linspace({spec.lo:g}, {spec.hi:g}, {size}).round().astype(int)"
        lo_ticks, hi_ticks = _duration_tick_bounds(spec)
        decimals = _duration_decimal_places(spec)
        ticks = (
            f"np.linspace({lo_ticks}, {hi_ticks}, {size})"
            ".round().astype(np.int64)"
        )
        return f"np.round(({ticks}) * {spec.quantum!r}, {decimals})"

    def note(spec: ScanColumnSpec) -> str:
        subject = spec.name
        if str(spec.label).strip():
            subject = f"{subject} ({str(spec.label).strip()})"
        return (
            f"{subject}: DAC code [{spec.lo:g}..{spec.hi:g}], 0 = 0 V"
            if spec.is_dac
            else (
                f"{subject}: duration [{spec.unit}], "
                f"tick = {spec.quantum:g} {spec.unit}"
            )
        )

    if str(kind) == "grid":
        # A real N-D grid: ONE axis per slot, every combination (outer product).  Each axis is seeded
        # in its own unit/range; scan_shape lets the grid show as a scan map.  Modest default sizes.
        sizes = [5, 4, 3] + [2] * max(0, n - 3)
        lines = ["import numpy as np", "",
                 f"# Grid scan over {n} slot(s) {cols[0].name}..{cols[-1].name}: every combination of the per-slot axes."]
        for j, spec in enumerate(cols):
            lines.append(f"a{j} = {sweep(spec, sizes[j])}        # axis for {note(spec)}")
        mesh = ", ".join(f"A{j}" for j in range(n))
        axes = ", ".join(f"a{j}" for j in range(n))
        ravel = ", ".join(f"A{j}.ravel()" for j in range(n))
        shape = ", ".join(f"len(a{j})" for j in range(n))
        shape_expr = f"({shape},)" if n == 1 else f"({shape})"   # always a tuple, even for n == 1
        lines.append(f"{mesh}, = np.meshgrid({axes}, indexing=\"ij\")")
        lines.append(f"scan_table = np.column_stack([{ravel}])")
        lines.append(f"scan_shape = {shape_expr}        # {n}-D grid -> a scan map")
        return "\n".join(lines) + "\n"
    # column_stack: one independent column per slot, the columns advancing together (lockstep).
    lines = ["import numpy as np", "",
             f"# {n} bound slot(s) {cols[0].name}..{cols[-1].name}: build an (N_points x {n}) array -- one row",
             "# per scan point, one column per slot (each in its OWN unit: ns for a duration, integer",
             "# code for a DAC).  Edit each column; the columns advance together (lockstep).",
             "N = 21        # number of scan points"]
    for spec in cols:
        lines.append(f"{spec.name} = {sweep(spec, 'N')}        # {note(spec)}")
    slots = ", ".join(spec.name for spec in cols)
    lines.append(f"scan_table = np.column_stack([{slots}])")
    return "\n".join(lines) + "\n"
