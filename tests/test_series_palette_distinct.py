"""Contract: the 1D / monitor multi-curve series palette is DISTINGUISHABLE.

Multiple curves in one plot cycle through ``style.PALETTE['series']`` (`live.py` does
`line.set_color(DEFAULT_COLORS[i % len])`).  The old palette was `['grey','skyblue','tab:blue',
'tab:orange']` -- only 4 colours, two of them near-identical blues -- so 5+ curves wrapped fast and
adjacent curves were "一堆颜色分不清".  This guard pins the fix: enough colours, all unique, and
pairwise perceptually separated (redmean distance), so the palette stays distinguishable.  A design
rule that CAN be a test MUST be one.
"""

import itertools
import math

import pytest
from matplotlib.colors import to_rgb

from Zou_lab_control.frontend.style import PALETTE

#: redmean (a perceptual-ish RGB metric).  The Okabe-Ito series palette's closest pair is ~133;
#: 100 leaves margin while still rejecting an accidental near-duplicate / typo'd colour.
_MIN_REDMEAN = 100.0


def _redmean(a, b) -> float:
    r1, g1, b1 = (x * 255 for x in to_rgb(a))
    r2, g2, b2 = (x * 255 for x in to_rgb(b))
    rm = (r1 + r2) / 2.0
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return math.sqrt((2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db)


def test_series_palette_is_distinguishable():
    series = list(PALETTE["series"])
    # 1) enough distinct colours before the cycle wraps (the old palette had only 4)
    assert len(series) >= 8, f"series palette has only {len(series)} colours -- too few, wraps fast"
    # 2) every colour is a valid, parseable colour and they are all unique
    rgbs = [to_rgb(c) for c in series]          # raises on an invalid colour spec
    assert len(set(series)) == len(series), "series palette has a duplicate colour"
    # 3) every pair is perceptually separated -- no two near-identical colours (the skyblue/tab:blue
    #    clash that motivated this).  The CLOSEST pair sets the floor.
    closest = min(_redmean(a, b) for a, b in itertools.combinations(series, 2))
    assert closest >= _MIN_REDMEAN, (
        f"two series colours are too similar (closest redmean {closest:.0f} < {_MIN_REDMEAN:.0f}) -- "
        "adjacent curves would be indistinguishable")
