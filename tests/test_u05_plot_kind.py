"""The frontend plot-kind vocabulary contains only current presentation facts."""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_frontend.plot_kind import PLOT_KIND_SPEC_BY_KEY, PLOT_KIND_SPECS

#: key -> (label, panel), in menu order.  Human help text is asserted by
#: semantics below rather than frozen as prose.
GOLDEN = {
    "2d": ("2D image", True),
    "sites": ("Site map", True),
    "1d": ("1D vector", True),
    "monitor": ("Rolling trace", True),
    "hist": ("Distribution", True),
    "pulse": ("Pulse sequence", False),
    "grid": ("Site grid", True),
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


def test_the_vocabulary_preserves_menu_identity_and_order():
    assert [spec.key for spec in PLOT_KIND_SPECS] == list(GOLDEN)
    for spec in PLOT_KIND_SPECS:
        actual = (spec.label, spec.panel)
        assert actual == GOLDEN[spec.key], spec.key


def test_site_map_help_names_the_generic_typed_contract_not_domain_producers():
    help_text = PLOT_KIND_SPEC_BY_KEY["sites"].input_format

    assert "SiteMapPresentation" in help_text
    assert "joined" in help_text
    assert "calibration" not in help_text.lower()
    assert "occupancy" not in help_text.lower()


def test_only_the_non_panel_pulse_kind_may_omit_input_help():
    assert {
        spec.key for spec in PLOT_KIND_SPECS if not spec.input_format
    } == {"pulse"}


def test_the_lookup_is_the_same_objects_as_the_table():
    assert list(PLOT_KIND_SPEC_BY_KEY) == [spec.key for spec in PLOT_KIND_SPECS]
    assert all(PLOT_KIND_SPEC_BY_KEY[spec.key] is spec for spec in PLOT_KIND_SPECS)
    with pytest.raises(TypeError):
        PLOT_KIND_SPEC_BY_KEY["foreign"] = PLOT_KIND_SPECS[0]



def test_the_vocabulary_module_reaches_for_no_renderer_and_no_toolkit():
    """The whole reason it could sink, asserted rather than assumed."""

    import zlc_frontend.plot_kind as vocabulary

    modules = _imported_modules(_module_tree(pathlib.Path(vocabulary.__file__)))
    roots = {name.split(".")[0] for name in modules} - {"__future__"}
    assert roots <= {"dataclasses", "types", "typing"}, roots


