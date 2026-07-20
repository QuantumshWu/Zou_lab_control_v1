"""The repeat-display vocabularies: how a plot kind may collapse its repeat axis.

Pure string tuples with no imports at all.  They sank out of the Matplotlib figure
module because they are a SPELLING, not a rendering: the console builds its
repeat-mode combo from them, the measurement layer validates against them, and the
figure layer applies them - three surfaces that must agree on the same words.

This is the first layer of the plot-kind vocabulary split.  The rest of ``PlotKind``
cannot follow yet: it carries ``cls``, a live renderer class, and the table is built
in TWO steps (a literal plus a later append for the ``grid`` kind), so separating the
vocabulary from the renderer binding is its own slice.
"""

from __future__ import annotations

__all__ = ["BASE_REPEAT_MODES", "HIST_REPEAT_MODES", "IMAGE_REPEAT_MODES",
           "ROLLING_REPEAT_MODES", "TRACE_REPEAT_MODES"]


# Repeat-display vocabularies (single source).  The BASE verbs (average/add/replace) are GENERIC --
# ``reduce_repeat`` collapses the repeat axis the same way for ANY plot kind -- so every kind offers
# them.  Only two specialisations: ``create`` (one line / one sub-distribution per repeat) is for the
# 1-D families incl. the distribution, but NOT 2d/sites (an image has no per-repeat-line meaning);
# ``pool`` (bin EVERY repeat's samples into ONE histogram) is the distribution's own extra mode.
BASE_REPEAT_MODES: tuple[str, ...] = ("average", "add", "replace")
TRACE_REPEAT_MODES: tuple[str, ...] = BASE_REPEAT_MODES + ("create",)                 # 1-D vector: base + per-repeat lines (NO roll)
ROLLING_REPEAT_MODES: tuple[str, ...] = BASE_REPEAT_MODES + ("roll", "create")        # rolling trace ONLY adds 'roll' (a rolling buffer)
IMAGE_REPEAT_MODES: tuple[str, ...] = BASE_REPEAT_MODES                               # a frame: mean/sum/latest, no create/roll
HIST_REPEAT_MODES: tuple[str, ...] = ("pool",) + BASE_REPEAT_MODES + ("create",)      # pool (default) + base + create (one overlaid histogram per repeat)
