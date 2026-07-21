"""What is left in the two shells, measured rather than remembered -- and no dead privates.

Slices (m) through (w) moved a lot out of ``task_console``, and the natural next question --
"what else can go?" -- was being answered from a list written several slices ago.  This file
answers it from the source each time it runs, and freezes the answer as a ratchet.

Two measurements, both by AST because substrings lie about code:

* how many top-level defs in ``task_console`` touch the renderer or Qt at all.  A def is
  render-bound if it names anything imported from Matplotlib, PyQt5, ``zlc_frontend``'s
  render/Qt packages, or the legacy render modules -- transitively, and including class
  bases.  ``from .live import X`` puts ``"live"`` in ``node.module`` with the dot in
  ``node.level``: the first cut of this analysis forgot that and read every relative import
  as absolute, which silently reported three render-bound defs as clean.
* whether any PRIVATE top-level def is referenced nowhere.  A move that takes a function AND
  its callers but leaves the definition behind is invisible: nothing breaks, the shell just
  keeps a body nobody runs.  That is exactly what S5-shell(p) did to ``_is_number``.

The reference count is deliberately scoped to the DEFINING module plus by-name imports of it.
A repo-wide count by name cannot tell two same-named functions apart -- and a botched move is
precisely the situation that leaves two of them, so the naive count reports the corpse as
alive.  That confound is the reason this test looks the way it does.
"""

from __future__ import annotations

#: C41 -- this oracle guards a legacy artifact and is deleted in the same commit that
#: deletes it (swept by test_design_charter).
DIES_WITH = ('Zou_lab_control/frontend/task_console.py', 'Zou_lab_control/frontend/pulse_gui.py')

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

SHELLS = ["Zou_lab_control/frontend/task_console.py",
          "Zou_lab_control/frontend/pulse_gui.py"]

#: Measured on the current tree.  These are EXACT, not ceilings.
#:
#: They were ceilings (``<=``) for one slice, and that was a mistake worth recording: when
#: S5-shell(y) moved the panel-param catalog out, the real numbers fell 39/15 -> 34/10 while
#: the constants stayed at 39/15 and every assertion still passed.  A ratchet that only
#: notices growth stops ratcheting the moment you forget to lower it -- it had silently
#: acquired five defs of slack.  Equality means a slice that empties the shell MUST come here
#: and write the new number down, which is the whole point of keeping a count at all.
#:
#: 2 is the FLOOR, and it is worth saying why rather than waiting for someone to try.  The two
#: survivors are ``_GridFocus`` and ``_StopAttempt``.  This census calls them render-free because
#: it reads IMPORTS -- and neither imports a renderer.  But ``_GridFocus`` HOLDS a grid plotter,
#: its detached canvas and the mpl callback ids to disconnect, and ``_StopAttempt`` HOLDS a
#: running daemon thread.  What a record holds is as much a placement fact as what its module
#: imports, and this criterion cannot see it; moving either into ``zlc_data`` would put a live
#: Qt/Matplotlib object behind a rule that says the package holds serialisable values (L303).
#: So: the shell's movable surface is exhausted, and what is left is not small -- it is bound.
RENDER_FREE_TOP_LEVEL_DEFS = 1
#: 25 after the test cull exposed _min_board_width: its ONLY referencer was a frozen test,
#: so deleting the archive turned a delegation stub into a corpse this very guard then caught.
TOTAL_TOP_LEVEL_DEFS = 6

RENDER_PACKAGES = ("PyQt5", "matplotlib", "zlc_frontend.render_style", "zlc_frontend.qt_widgets",
                   "zlc_frontend.live_plot", "zlc_frontend.qt_canvas", "zlc_frontend.render")
RENDER_RELATIVE = (".live", ".canvas", ".data_figure", ".qt_canvas", ".render_loop",
                   ".figure_viewer", ".selectors", ".ticks")


def _module_of(node: ast.ImportFrom) -> str:
    """``from .live import X`` -> ``".live"``.  The dot lives in ``node.level``, not in
    ``node.module``; treating ``node.module`` as the whole name makes every relative import
    read as an absolute one."""
    return ("." * node.level) + (node.module or "")


def _is_render(module: str) -> bool:
    if not module:
        return False
    return any(module == m or module.startswith(m + ".")
               for m in RENDER_RELATIVE + RENDER_PACKAGES)


def _root_names(node) -> set[str]:
    """Every identifier the node reads, by AST -- ``QtWidgets.QLabel`` contributes
    ``QtWidgets``.  Word-level, so ``config`` never matches ``fig``."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            root = sub
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                out.add(root.id)
    return out


def _census(relative: str):
    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    origin = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                origin[alias.asname or alias.name] = _module_of(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                origin[alias.asname or alias.name.split(".")[0]] = alias.name

    tops = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    tainted = {name for name, module in origin.items() if _is_render(module)}
    used = {n.name: _root_names(n) for n in tops}
    bases = {n.name: {b.id for b in getattr(n, "bases", []) if isinstance(b, ast.Name)}
             | {b.attr for b in getattr(n, "bases", []) if isinstance(b, ast.Attribute)}
             for n in tops}
    changed = True
    while changed:                      # a def that reaches a render name is render-bound too
        changed = False
        for n in tops:
            if n.name not in tainted and (used[n.name] | bases[n.name]) & tainted:
                tainted.add(n.name)
                changed = True
    return tree, tops, tainted, used


def test_the_recorded_counts_are_what_the_shell_actually_holds():
    """Exact on purpose -- see the note on the constants.  A slice that moves defs out is
    supposed to come back here and write the smaller number down; being told to is the point."""

    tree, tops, tainted, _ = _census(SHELLS[0])
    free = sorted(n.name for n in tops if n.name not in tainted)
    assert len(tops) == TOTAL_TOP_LEVEL_DEFS, (
        f"task_console holds {len(tops)} top-level defs, recorded {TOTAL_TOP_LEVEL_DEFS}. "
        f"If a slice just emptied it, lower the constant; if it GREW, the shell is being "
        f"filled instead of emptied and that is the bug.")
    assert len(free) == RENDER_FREE_TOP_LEVEL_DEFS, (
        f"{len(free)} render-free defs, recorded {RENDER_FREE_TOP_LEVEL_DEFS}: {free}")


def test_the_relative_import_of_the_render_module_is_seen_as_render():
    """The bug that made the first census wrong, pinned so it cannot come back.

    ``from .live import PLOT_KIND_BY_KEY`` must taint its importer.  Read carelessly, the
    module reads as ``"live"``, matches nothing, and three render-bound defs pass as clean."""

    assert _is_render(".live") and _is_render(".canvas")
    node = ast.parse("from .live import X").body[0]
    assert _module_of(node) == ".live"
    assert _is_render(_module_of(node))


@pytest.mark.parametrize("relative", SHELLS)
def test_no_private_top_level_def_is_referenced_nowhere(relative):
    """A move that takes a function AND its callers but forgets the body leaves this.

    Scoped to the defining module (plus a by-name import of it) on purpose: counting the name
    across the repository cannot separate two same-named functions, and a botched move is
    exactly what leaves two."""

    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    privates = [n for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name.startswith("_")]

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            referenced.add(node.value)          # getattr("_x") / __all__

    module = relative[:-3].replace("/", ".")
    for other in REPO.rglob("*.py"):
        if "__pycache__" in other.parts or other == REPO / relative:
            continue
        try:
            tree_other = ast.parse(other.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree_other):
            if isinstance(node, ast.ImportFrom) and node.module and module.endswith(node.module):
                referenced.update(alias.name for alias in node.names)

    dead = [f"{n.name} (line {n.lineno})" for n in privates if n.name not in referenced]
    assert not dead, (
        f"{relative} keeps private defs nothing reaches -- a move that took the callers but "
        f"left the body:\n" + "\n".join(f"  {item}" for item in dead))
