"""The panel-size vocabulary is one definition, reachable without the render stack.

``PANEL_SIZES`` and ``panel_size_cells`` are a spelling three surfaces must agree on -
the console lays cards out by it, the pulse GUI sizes its preview by it, and the figure
layer turns it into a data region.  They lived inside the Matplotlib figure module, so
anything that merely wanted to KNOW the presets had to import the whole render stack;
that is what kept the console's board geometry from moving into the target package at
all, since the placement axiom forbids a Qt-only module importing Matplotlib.

The rendered SIZE of a panel genuinely depends on figure margins and stays where it is.
Only the vocabulary and its parser sank.

The two tests that cover panel geometry - ``test_size_driven_geometry.py`` and
``test_panel_hug_layout.py`` - sit outside ``tests/migration_active_tests.txt``, so they
are frozen: not run, not modified, and not evidence for this branch.  This file is the
oracle.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).

import ast
import importlib
import pathlib

import pytest

import zlc_data.panel_size as canonical




def test_every_preset_parses_and_anything_else_is_refused():
    """The parser is the presets' only reader, so it must agree with them exactly."""

    for size in canonical.PANEL_SIZES:
        rows, cols = canonical.panel_size_cells(size)
        assert (rows, cols) == tuple(int(part) for part in size.split("x"))
        # The GUI hands this whatever a combo box or a saved workspace produced.
        assert canonical.panel_size_cells(f"  {size.upper()} ") == (rows, cols)

    with pytest.raises(ValueError):
        canonical.panel_size_cells("9x9")


def test_the_vocabulary_module_imports_nothing():
    """The reason it could sink at all, asserted rather than assumed.

    One import here - of the figure layer, of Qt, of anything - and the module stops
    being reachable from the places that need it, which is exactly the trap it was in.
    """

    path = pathlib.Path(canonical.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    assert not imports, f"panel_size.py grew imports: {[ast.dump(n) for n in imports]}"


