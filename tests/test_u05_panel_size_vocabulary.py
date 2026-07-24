"""The frontend panel-size vocabulary has one renderer-free owner."""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

import zlc_frontend.panel_size as canonical




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


