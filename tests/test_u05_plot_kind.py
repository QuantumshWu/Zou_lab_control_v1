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


def test_no_kind_can_be_silently_dropped():
    """The two-step assembly, made mechanical.

    ``GridPlot`` is defined after the table, so ``grid`` is bound in a second step.  That
    is a fact about class-definition ORDER in one file, and the vocabulary must not know
    about it: the literal carries all seven kinds.  What the render layer must guarantee
    is that its two steps between them cover every spec -- a spec in neither the eager map
    nor the late list would produce a table missing a kind, and nothing else would fail.
    """

    from zlc_frontend.live_plot import live

    eager = set(live._CLS_BY_KEY)
    late = set(live._LATE_BOUND_KEYS)
    vocabulary = {spec.key for spec in PLOT_KIND_SPECS}
    assert eager.isdisjoint(late), eager & late
    assert eager | late == vocabulary, vocabulary - (eager | late)
    assert [pk.key for pk in live.PLOT_KINDS] == [spec.key for spec in PLOT_KIND_SPECS]


def test_each_row_pairs_the_shared_spec_object_with_its_renderer():
    """Delegation, not duplication: the row holds the SAME spec, so it cannot disagree."""

    from zlc_frontend.live_plot import live

    for pk in live.PLOT_KINDS:
        assert pk.spec is PLOT_KIND_SPEC_BY_KEY[pk.key]
        assert pk.cls.__name__ == GOLDEN_CLS[pk.key]
        # Read through the row exactly as every existing caller does.
        assert (pk.label, pk.render_family, pk.panel, pk.input_format,
                pk.input_slots, pk.single_slot, pk.repeat_modes) == GOLDEN[pk.key]
        # ... and the mutable-looking ones are the spec's own objects, not copies.
        assert pk.input_slots is pk.spec.input_slots
        assert pk.repeat_modes is pk.spec.repeat_modes


def test_the_row_stores_only_the_spec_and_the_class():
    """A vocabulary field copied back into the row would pass every test above."""

    from zlc_frontend.live_plot.live import PlotKind

    assert set(PlotKind.__dataclass_fields__) == {"spec", "cls"}


def test_binding_the_late_kinds_again_changes_nothing():
    """It runs at import; a second call (re-import, a test) must not duplicate a row."""

    from zlc_frontend.live_plot import live

    before = live.PLOT_KINDS
    live._bind_late_plot_kinds()
    assert live.PLOT_KINDS is before
    assert [pk.key for pk in live.PLOT_KINDS].count("grid") == 1


def test_a_plotter_still_resolves_back_to_its_kind():
    """The class half of the pairing, exercised rather than merely stored."""

    from zlc_frontend.live_plot import live

    for pk in live.PLOT_KINDS:
        assert live.kind_for_plotter(pk.cls.__new__(pk.cls)) == pk.key
    assert live.kind_for_plotter(object()) is None


def test_every_offered_repeat_verb_is_one_the_reducer_implements():
    """The kinds' menus and the reducer's verb set are TWO facts, deliberately not merged.

    ``live.REPEAT_MODES`` is what the reduction code can actually do; each kind's
    ``repeat_modes`` is what its menu offers.  They coincide today for the rolling trace,
    which is exactly why merging them would be wrong -- one is a capability, the other a
    per-kind choice.  The relation that must hold is containment: a menu may never offer a
    verb the reducer would ignore.
    """

    from zlc_frontend.live_plot import live

    offered = {mode for spec in PLOT_KIND_SPECS for mode in spec.repeat_modes}
    assert offered <= live._REDUCE_REPEAT_MODES, offered - live._REDUCE_REPEAT_MODES


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
    ("zlc_frontend/live_plot/plot_figure.py", {"PLOT_KIND_SPECS", "PLOT_KIND_SPEC_BY_KEY"}),
    # The viewer body moved whole into the workbench app package; the pin follows the code.
    ("zlc_workbench/figure_viewer/plot_bridge_figure_viewer.py", {"PLOT_KIND_SPEC_BY_KEY"}),
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


def test_the_console_no_longer_pulls_the_render_table_in_to_read_words():
    """The payoff: building the Add-Panel menu stopped being a rendering question.

    The console still reaches for ``PLOT_KIND_BY_KEY`` inside two functions that genuinely
    need a CLASS; what must not come back is the module-level import that made merely
    importing the console drag the table along."""

    tree = _module_tree(REPO / "Zou_lab_control" / "frontend" / "task_console.py")
    top_level = {alias.name for node in tree.body
                 if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "PLOT_KINDS" not in top_level
    assert "PLOT_KIND_BY_KEY" not in top_level
