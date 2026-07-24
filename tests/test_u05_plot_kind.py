"""The frontend plot-kind vocabulary contains only current presentation facts."""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_frontend.plot_kind import PLOT_KIND_SPEC_BY_KEY, PLOT_KIND_SPECS

#: key -> (label, panel, input_format), in menu order.
GOLDEN = {
    "2d": ("2D image", True, "value must be a 2D array / camera frame (H×W)"),
    "sites": ("Site map", True,
              "value must carry a typed calibration site map, or an exact single-cell occupancy view with its same-shot frame and admitted calibration geometry"),
    "1d": ("1D vector", True, "value must be a 1D vector (N,) or per-site array"),
    "monitor": ("Rolling trace", True, "value must be a scalar per shot (rolling trace)"),
    "hist": ("Distribution", True, "value must be a 1D sample vector"),
    "pulse": ("Pulse sequence", False, ""),
    "grid": ("Site grid", True,
             "value must admit an explicit named-axis CURVE, HISTOGRAM, or IMAGE facet view"),
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
        actual = (spec.label, spec.panel, spec.input_format)
        assert actual == GOLDEN[spec.key], spec.key


def test_the_lookup_is_the_same_objects_as_the_table():
    assert list(PLOT_KIND_SPEC_BY_KEY) == [spec.key for spec in PLOT_KIND_SPECS]
    assert all(PLOT_KIND_SPEC_BY_KEY[spec.key] is spec for spec in PLOT_KIND_SPECS)



def test_the_vocabulary_module_reaches_for_no_renderer_and_no_toolkit():
    """The whole reason it could sink, asserted rather than assumed."""

    import zlc_frontend.plot_kind as vocabulary

    modules = _imported_modules(_module_tree(pathlib.Path(vocabulary.__file__)))
    roots = {name.split(".")[0] for name in modules} - {"__future__"}
    assert roots <= {"dataclasses"}, roots


