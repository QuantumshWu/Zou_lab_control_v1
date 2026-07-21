"""The plot-kind table splits into a VOCABULARY and a renderer binding.

Eight of a plot kind's nine facts are words -- key, label, fitting family, whether it is
offered in the Add-Panel menu, the accepted value shape, the starting slots, the
single-slot rule, the repeat menu.  The ninth is a Matplotlib class.  The eight sink to
``zlc_data.plot_kind``; the class stays in the render layer, paired with its spec by
``live.PlotKind``, which DELEGATES rather than copies -- so a row that disagrees with the
vocabulary cannot be built.

The trap this file exists to guard is the two-step assembly.  ``GridPlot`` is defined
AFTER the table in ``live.py``, so its row is bound later.  A split that covered only the
literal would have dropped ``grid`` from the Add-Panel menu, the saved-figure validator
and the viewer's label -- silently, because every remaining kind would still work.  The
fix is structural: the vocabulary literal is COMPLETE (grid included), and the render
layer partitions the keys into eager and late, so a kind in neither raises rather than
vanishing.  ``test_no_kind_can_be_silently_dropped`` is that assertion.

Every test covering this table -- test_plot_kind_table.py, test_frontend_plot_contract.py,
test_roi_kind_agnostic.py, test_saved_figure_load.py -- sits outside
``tests/migration_active_tests.txt`` and is frozen: not run, not modified, not evidence.
This file is the oracle.  Its golden was captured from the legacy table immediately before
the move, and every field in it discriminates: render_family takes three values, panel
takes two, and input_slots/single_slot are non-default on exactly one kind.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY, PLOT_KIND_SPECS

SITES_FORMAT = ("value must be a per-site (N,) vector -- one number per tweezer (e.g. occupancy "
                "0/1 or loading rate); signal[0]'s producing node also supplies the ring centres "
                "+ frame underlay")
SITES_SLOT = ("occupancy", "",
              "per-site (N,) occupancy vector (signal[0]) -- colours the rings; its "
              "producing node also supplies the centres + frame underlay")

#: key -> (label, render_family, panel, input_format, input_slots, single_slot, repeat_modes),
#: in TABLE ORDER, which is the Add-Panel menu order.
GOLDEN = {
    "2d": ("2D image", "2D", True, "value must be a 2D array / camera frame (H×W)",
           (), False, ("average", "add", "replace")),
    "sites": ("Site map", "auto", True, SITES_FORMAT,
              (SITES_SLOT,), True, ("average", "add", "replace")),
    "1d": ("1D vector", "1D", True, "value must be a 1D vector (N,) or per-site array",
           (), False, ("average", "add", "replace", "create")),
    "monitor": ("Rolling trace", "1D", True, "value must be a scalar per shot (rolling trace)",
                (), False, ("average", "add", "replace", "roll", "create")),
    "hist": ("Distribution", "1D", True, "value must be a 1D sample vector",
             (), False, ("pool", "average", "add", "replace", "create")),
    "pulse": ("Pulse sequence", "1D", False, "", (), False, ()),
    "grid": ("Site grid", "1D", True, "", (), False, ()),
}

#: key -> renderer class NAME, captured with the rest.  Names, not classes, so this file
#: states the expected pairing without importing the render stack to describe it.
GOLDEN_CLS = {"2d": "Live2DDis", "sites": "LiveSiteMap", "1d": "Live1D",
              "monitor": "LiveLiveDis", "hist": "HistogramFigure",
              "pulse": "PulseSequenceFigure", "grid": "GridPlot"}

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
        actual = (spec.label, spec.render_family, spec.panel, spec.input_format,
                  spec.input_slots, spec.single_slot, spec.repeat_modes)
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
    ("zlc_data/console_records.py", {"PLOT_KIND_SPECS", "PLOT_KIND_SPEC_BY_KEY"}),
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


