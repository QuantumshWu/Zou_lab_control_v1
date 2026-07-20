"""The repeat-display vocabularies are one definition, reachable without the renderer.

Three surfaces must spell these identically: the console builds its repeat-mode combo
from them, the measurement layer validates against them, and the figure layer applies
them.  They are pure string tuples with no rendering in them at all, yet they lived
inside the Matplotlib figure module, so anything that merely wanted to KNOW the words
had to import the whole render stack.

This is the FIRST layer of the plot-kind vocabulary split, not the whole of it.  The
rest of ``PlotKind`` cannot follow yet for a reason worth writing down: it carries
``cls``, a live renderer class, AND the table is assembled in TWO steps - a literal
plus a later ``PLOT_KINDS = PLOT_KINDS + (...)`` append for the ``grid`` kind - so
separating vocabulary from renderer there is its own slice with its own golden.

Every test that covers this table - test_plot_kind_table.py, test_frontend_plot_contract.py,
test_scan_repeat.py - sits outside ``tests/migration_active_tests.txt`` and is frozen:
not run, not modified, not evidence.  This file is the oracle.
"""

from __future__ import annotations

import ast
import pathlib

import zlc_data.repeat_modes as canonical

#: Captured from the legacy figure module immediately before the move.
GOLDEN = {
    "BASE_REPEAT_MODES": ("average", "add", "replace"),
    "TRACE_REPEAT_MODES": ("average", "add", "replace", "create"),
    "ROLLING_REPEAT_MODES": ("average", "add", "replace", "roll", "create"),
    "IMAGE_REPEAT_MODES": ("average", "add", "replace"),
    "HIST_REPEAT_MODES": ("pool", "average", "add", "replace", "create"),
}


def test_every_vocabulary_is_reproduced_exactly():
    """Order matters: the FIRST entry is the kind's default repeat mode."""

    for name, expected in GOLDEN.items():
        assert getattr(canonical, name) == expected, name


def test_the_render_layer_reads_back_the_same_objects():
    """Not copies - the same tuples, so the two can never disagree."""

    from zlc_frontend.live_plot import live

    assert live.IMAGE_REPEAT_MODES is canonical.IMAGE_REPEAT_MODES
    assert live.HIST_REPEAT_MODES is canonical.HIST_REPEAT_MODES
    assert live.TRACE_REPEAT_MODES is canonical.TRACE_REPEAT_MODES
    assert live.ROLLING_REPEAT_MODES is canonical.ROLLING_REPEAT_MODES


def test_the_specialisations_extend_the_base_rather_than_restating_it():
    """The design fact that keeps the four in step: three shared verbs, two extras.

    A vocabulary that spelled its own base would be free to drift from the others -
    the exact failure the single source exists to prevent.
    """

    base = GOLDEN["BASE_REPEAT_MODES"]
    assert canonical.IMAGE_REPEAT_MODES == base                       # a frame: no per-repeat lines
    assert canonical.TRACE_REPEAT_MODES == base + ("create",)          # 1-D adds one line per repeat
    assert canonical.ROLLING_REPEAT_MODES == base + ("roll", "create")  # rolling ALSO gets a buffer
    assert canonical.HIST_REPEAT_MODES == ("pool",) + base + ("create",)  # pool is the default, first

    # "roll" is meaningless on anything but a rolling trace; that is what stops a
    # histogram silently ignoring a control it was offered.
    assert "roll" not in canonical.IMAGE_REPEAT_MODES
    assert "roll" not in canonical.HIST_REPEAT_MODES
    assert "roll" not in canonical.TRACE_REPEAT_MODES


def test_the_vocabulary_module_imports_nothing():
    """The whole reason it could sink, asserted rather than assumed."""

    tree = ast.parse(pathlib.Path(canonical.__file__).read_text(encoding="utf-8"))
    imports = [n for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom))
               and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    assert not imports, f"repeat_modes.py grew imports: {[ast.dump(n) for n in imports]}"


def test_the_shell_reads_the_vocabulary_from_its_new_home():
    """Structural, so this file keeps no legacy-tree dependency of its own.

    A shell still importing it from the figure module would work today - the re-export
    is the same object - which is exactly why only the import graph can catch it.
    """

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "Zou_lab_control" / "frontend" / "task_console.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    sources = {node.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom)
               and any(alias.name.endswith("REPEAT_MODES") for alias in node.names)}
    assert sources == {"zlc_data.repeat_modes"}, sources
