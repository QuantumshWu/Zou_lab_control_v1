"""The headless plot-kind vocabulary contains only facts current readers consume."""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY, PLOT_KIND_SPECS

#: key -> (label, render_family, panel, input_format), in menu order.
GOLDEN = {
    "2d": ("2D image", "2D", True, "value must be a 2D array / camera frame (H×W)"),
    "sites": ("Site map", "auto", False,
              "value must be a typed per-site (N,) dataset with its site coordinates"),
    "1d": ("1D vector", "1D", True, "value must be a 1D vector (N,) or per-site array"),
    "monitor": ("Rolling trace", "1D", True, "value must be a scalar per shot (rolling trace)"),
    "hist": ("Distribution", "1D", True, "value must be a 1D sample vector"),
    "pulse": ("Pulse sequence", "1D", False, ""),
    "grid": ("Site grid", "1D", False, ""),
}

REPO = pathlib.Path(__file__).resolve().parents[1]


def _module_tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_the_vocabulary_is_reproduced_exactly_and_in_menu_order():
    assert [spec.key for spec in PLOT_KIND_SPECS] == list(GOLDEN)
    for spec in PLOT_KIND_SPECS:
        actual = (spec.label, spec.render_family, spec.panel, spec.input_format)
        assert actual == GOLDEN[spec.key], spec.key


def test_the_lookup_is_the_same_objects_as_the_table():
    assert list(PLOT_KIND_SPEC_BY_KEY) == [spec.key for spec in PLOT_KIND_SPECS]
    assert all(PLOT_KIND_SPEC_BY_KEY[spec.key] is spec for spec in PLOT_KIND_SPECS)



def test_the_vocabulary_module_reaches_for_no_renderer_and_no_toolkit():
    """The whole reason it could sink, asserted rather than assumed."""

    import zlc_data.plot_kind as vocabulary

    modules = _imported_modules(_module_tree(pathlib.Path(vocabulary.__file__)))
    roots = {name.split(".")[0] for name in modules} - {"__future__"}
    assert roots <= {"dataclasses", "zlc_data"}, roots


@pytest.mark.parametrize("relative, expected", [
    # The console used to name these directly.  Its per-kind PANEL_* tables moved into
    # zlc_data.console_records with PanelConfig, so the derivation -- and with it this
    # import -- now sits one layer down; the console reads the PANEL vocabulary instead.
    ("zlc_data/console_records.py", {"PLOT_KIND_SPECS"}),
    # The figure viewer was pinned here too, for the helper that turned a stored kind key
    # into its display label.  Opening a stored figure is not connected on the current data
    # plane, so nothing calls that helper and it went with the rest of the load path; a
    # reader that reads nothing cannot be pinned to a single source.  The row comes back
    # with the Info column's kind label, and it is this list that has to grow again then.
])
def test_each_vocabulary_reader_names_the_new_home(relative, expected):
    """Structural, so this file keeps no legacy-tree dependency of its own.

    A reader still going through the render table would work today -- the row delegates to
    the same spec -- which is precisely why only the import graph can catch it."""

    tree = _module_tree(REPO / relative)
    imported: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported.setdefault(alias.name, set()).add(node.module)
    for name in expected:
        assert imported.get(name) == {"zlc_data.plot_kind"}, (name, imported.get(name))


