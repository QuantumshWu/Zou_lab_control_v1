"""Human-editable starter source for the two pulse scan programs.

Pulse/domain owners supply structurally compatible column descriptions.  This
module writes human-editable Python beside the compiler-owned column physics;
it has no GUI dependency.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Protocol, Sequence

__all__ = [
    "SWEEP_API_SLOT",
    "SWEEP_SCAN_SLOT",
    "ScanTemplateColumn",
    "scan_table_template",
]


# These strings are the frontend form's two choices.  The composition root
# turns the chosen presentation value into one of the two typed domain program
# classes; the strings are not a third domain execution model.
SWEEP_SCAN_SLOT = "scan_slot"
SWEEP_API_SLOT = "api_slot"


class ScanTemplateColumn(Protocol):
    """Structural input required to render one starter-program column."""

    name: str
    lo: float
    hi: float
    is_dac: bool
    unit: str
    label: str
    quantum: float | None
    is_delay: bool


def _time_tick_bounds(spec: ScanTemplateColumn) -> tuple[int, int]:
    """Return exact integer bounds from domain-authored physical values."""

    if spec.quantum is None:
        raise ValueError(
            f"time column {spec.name!r} has no native-unit quantum"
        )
    quantum = Fraction(str(spec.quantum))
    ratios = tuple(Fraction(str(value)) / quantum for value in (spec.lo, spec.hi))
    if any(ratio.denominator != 1 for ratio in ratios):
        raise ValueError(
            f"time column {spec.name!r} bounds are not multiples of its quantum"
        )
    lo_ticks, hi_ticks = (ratio.numerator for ratio in ratios)
    if not spec.is_delay and lo_ticks < 1:
        raise ValueError(f"duration column {spec.name!r} must start at one tick")
    return lo_ticks, hi_ticks


def _time_decimal_places(spec: ScanTemplateColumn) -> int:
    """Digits needed to recover canonical decimal tick values after NumPy math."""

    if spec.quantum is None:
        raise ValueError(
            f"time column {spec.name!r} has no native-unit quantum"
        )
    exponent = Decimal(str(spec.quantum)).normalize().as_tuple().exponent
    return max(0, -int(exponent))


def scan_table_template(
    kind: str,
    columns: Sequence[ScanTemplateColumn],
) -> str:
    """Render the shared human-editable ``scan_table`` starter program.

    ``column_stack`` advances all columns in lockstep.  ``grid`` constructs the
    Cartesian product and records ``scan_shape``.  Physical units, ranges, and
    clock quantum come exclusively from the supplied pulse-owned descriptors.
    """

    cols = list(columns)
    if not cols:
        raise ValueError(
            "scan_table_template requires at least one domain-owned column"
        )
    n = len(cols)

    def sweep(spec: ScanTemplateColumn, size: object) -> str:
        if spec.is_dac:
            return f"np.linspace({spec.lo:g}, {spec.hi:g}, {size}).round().astype(int)"
        lo_ticks, hi_ticks = _time_tick_bounds(spec)
        decimals = _time_decimal_places(spec)
        ticks = (
            f"np.linspace({lo_ticks}, {hi_ticks}, {size})"
            ".round().astype(np.int64)"
        )
        return f"np.round(({ticks}) * {spec.quantum!r}, {decimals})"

    def note(spec: ScanTemplateColumn) -> str:
        subject = spec.name
        if str(spec.label).strip():
            subject = f"{subject} ({str(spec.label).strip()})"
        return (
            f"{subject}: DAC code [{spec.lo:g}..{spec.hi:g}], 0 = 0 V"
            if spec.is_dac
            else (
                f"{subject}: {'delay' if spec.is_delay else 'duration'} [{spec.unit}], "
                f"tick = {spec.quantum:g} {spec.unit}"
            )
        )

    if str(kind) == "grid":
        sizes = [5, 4, 3] + [2] * max(0, n - 3)
        lines = [
            "import numpy as np",
            "",
            f"# Grid scan over {n} slot(s) {cols[0].name}..{cols[-1].name}: every combination of the per-slot axes.",
        ]
        for index, spec in enumerate(cols):
            lines.append(
                f"a{index} = {sweep(spec, sizes[index])}        # axis for {note(spec)}"
            )
        mesh = ", ".join(f"A{index}" for index in range(n))
        axes = ", ".join(f"a{index}" for index in range(n))
        ravel = ", ".join(f"A{index}.ravel()" for index in range(n))
        shape = ", ".join(f"len(a{index})" for index in range(n))
        shape_expr = f"({shape},)" if n == 1 else f"({shape})"
        lines.append(f'{mesh}, = np.meshgrid({axes}, indexing="ij")')
        lines.append(f"scan_table = np.column_stack([{ravel}])")
        lines.append(
            f"scan_shape = {shape_expr}        # {n}-D grid -> a scan map"
        )
        return "\n".join(lines) + "\n"

    lines = [
        "import numpy as np",
        "",
        f"# {n} bound slot(s) {cols[0].name}..{cols[-1].name}: build an (N_points x {n}) array -- one row",
        "# per scan point, one column per slot (each in its OWN unit: time for a duration/delay, integer",
        "# code for a DAC).  Edit each column; the columns advance together (lockstep).",
        "N = 21        # number of scan points",
    ]
    for spec in cols:
        lines.append(f"{spec.name} = {sweep(spec, 'N')}        # {note(spec)}")
    slots = ", ".join(spec.name for spec in cols)
    lines.append(f"scan_table = np.column_stack([{slots}])")
    return "\n".join(lines) + "\n"
