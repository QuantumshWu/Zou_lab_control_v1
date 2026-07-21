"""The panel-param catalog leaves the shell, and the histogram-fit vocabulary stops being copied.

Two things happen here and the second is the reason the first was worth doing.

The catalog -- which knobs a panel kind carries -- and its four resolvers moved to
``zlc_frontend.panel_params``.  It could not sink further: it reads the DEFAULT colormaps from
``zlc_frontend.render_style``, and L303 admits into ``zlc_data`` only "领域中立、无头、可序列化的
数据语义和值上的纯算法".  Copying a colour decision down to satisfy a layer rule would trade a
legal import (L327: ``zlc_frontend -> zlc_storage``, and the frontend owns its own palette) for a
second source of the same fact.  So the record sits one layer up, and the guard below is that the
shell holds no copy of it.

Moving it exposed the real defect.  The histogram-fit verbs existed FOUR times across THREE
layers: twice as a set literal inside ``zlc_data.curve_fitting`` (the solver validating its own
argument), once as the ``choices`` tuple in this catalog, and the DEFAULT they open on lived
across the layer boundary in the render module as ``DEFAULT_HIST_FIT``.  Adding a fifth fit mode
meant editing three files in three packages, and forgetting one of them fails differently in each:
the solver rejects what the chooser offers, or the chooser hides what the solver supports.  Verbs
and default now belong to the solver's module and every surface cites them -- so the sweep at the
bottom, which fails if any module writes the triple as a literal again, is the mechanical part.

``test_frontend_smoke.py`` and ``test_panel_view_logic_split.py`` cover these helpers and sit
outside ``tests/migration_active_tests.txt``, hence frozen: the cases here were captured from the
shell immediately before the move and diffed against it afterwards (904 lines, identical).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_data.curve_fitting import DEFAULT_HISTOGRAM_FIT, HISTOGRAM_FIT_MODES
from zlc_frontend.panel_params import (
    CMAPS,
    GRID_TITLE_PARAMS,
    PANEL_PARAMS,
    panel_display_decls,
    panel_param_default,
    resolved_cmap,
    resolved_param,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_shell_takes_every_catalog_name_from_the_new_home():
    """Structural: a shell that kept its own copy would pass every behaviour test below."""

    tree = ast.parse((REPO / "Zou_lab_control" / "frontend" / "task_console.py")
                     .read_text(encoding="utf-8"))
    sources: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                sources.setdefault(alias.name, set()).add(node.module)
    for name in ("CMAPS", "GRID_TITLE_PARAMS", "PANEL_PARAMS", "panel_display_decls",
                 "panel_param_default", "resolved_cmap", "resolved_param"):
        assert sources.get(name) == {"zlc_frontend.panel_params"}, (name, sources.get(name))

    from Zou_lab_control.frontend import task_console as shell

    assert shell.PANEL_PARAMS is PANEL_PARAMS
    assert shell._resolved_param is resolved_param
    assert shell._resolved_cmap is resolved_cmap
    assert shell._panel_param_default is panel_param_default
    assert shell._panel_display_decls is panel_display_decls


def test_the_catalog_module_pulls_in_no_toolkit():
    """It is read by save/load and by the Setting UI alike, so Qt must not ride along."""

    tree = ast.parse((REPO / "zlc_frontend" / "panel_params.py").read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0] if node.level == 0 else "")
    assert "PyQt5" not in roots, roots
    assert roots <= {"", "__future__", "typing", "zlc_data"}, roots


# --- the vocabulary, said once -------------------------------------------------------------

def test_the_fit_chooser_offers_exactly_what_the_solver_accepts():
    """The four-copy defect, stated as one identity rather than four equal-looking literals."""

    fit = next(decl for decl in PANEL_PARAMS["hist"] if decl.key == "fit")
    assert tuple(fit.choices) == HISTOGRAM_FIT_MODES
    assert fit.default == DEFAULT_HISTOGRAM_FIT
    assert DEFAULT_HISTOGRAM_FIT in HISTOGRAM_FIT_MODES


def test_the_solver_rejects_a_verb_that_is_not_in_the_catalog_and_accepts_every_one_that_is():
    """Both directions, so the tie is real and not just a shared spelling."""

    import numpy as np

    from zlc_data.curve_fitting import HistogramFitResult, fit_histogram

    edges, counts = np.arange(6.0), np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    for mode in HISTOGRAM_FIT_MODES:
        assert isinstance(fit_histogram(edges, counts, mode=mode), HistogramFitResult)
    with pytest.raises(ValueError):
        fit_histogram(edges, counts, mode="triple")
    with pytest.raises(ValueError):
        HistogramFitResult.invalid("triple", "nope")


def test_the_histogram_figure_opens_on_the_declared_default():
    """The consequence an operator sees: the Setting UI's default IS what the figure draws."""

    import inspect

    from zlc_frontend.live_plot.live import HistogramFigure

    assert (inspect.signature(HistogramFigure.__init__).parameters["fit"].default
            == DEFAULT_HISTOGRAM_FIT)


def test_no_module_restates_the_fit_verbs_as_a_literal():
    """The sweep, so the next surface that needs the verbs cites them instead of retyping them.

    Anchored on the triple appearing together in one expression -- a lone ``"double"`` is an
    ordinary comparison, three verbs side by side is a copy of the vocabulary."""

    offenders = []
    for package in ("zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench",
                    "Zou_lab_control"):
        for path in (REPO / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                    continue
                words = tuple(item.value for item in node.elts
                              if isinstance(item, ast.Constant) and isinstance(item.value, str))
                if set(HISTOGRAM_FIT_MODES) <= set(words):
                    offenders.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno}")
    assert offenders == ["zlc_data/curve_fitting.py:70"], offenders


# --- the resolvers, unchanged --------------------------------------------------------------

def test_a_declared_default_is_read_from_the_declaration_never_retyped():
    for kind, decls in PANEL_PARAMS.items():
        for decl in decls:
            assert panel_param_default(kind, decl.key) == decl.default
            assert resolved_param(kind, {}, decl.key) == decl.default
    assert panel_param_default("nope", "cmap") is None
    assert panel_param_default("hist", "missing") is None


#: (kind, key, falsy stored value).  Only pairs whose DECLARED default is not itself falsy are
#: listed: ``hist.ylog`` defaults to ``False``, so storing ``False`` there cannot tell a
#: presence check from a truthiness check -- including it would look like coverage and prove
#: nothing.  These three would each come back WRONG under ``params.get(key) or default``.
@pytest.mark.parametrize("kind, key, stored", [
    ("hist", "bins", 0), ("monitor", "show_dist", False), ("monitor", "length", 0),
])
def test_a_stored_false_or_zero_is_a_value_not_an_absence(kind, key, stored):
    """Presence is ``key in params``: a truthiness test would silently restore the default."""

    assert resolved_param(kind, {key: stored}, key) == stored
    assert panel_param_default(kind, key), "this case only discriminates for a truthy default"
    assert resolved_param(kind, {key: stored}, key) != panel_param_default(kind, key)


@pytest.mark.parametrize("params, expected", [
    ({}, "inferno"), (None, "inferno"), ({"cmap": "viridis"}, "viridis"),
    ({"cmap": "  magma  "}, "magma"), ({"cmap": ""}, "inferno"), ({"cmap": "   "}, "inferno"),
    ({"cmap": None}, "inferno"),
])
def test_an_image_kind_always_resolves_to_a_real_colormap_name(params, expected):
    assert resolved_cmap("2d", params) == expected


@pytest.mark.parametrize("kind", ["1d", "hist", "monitor", "grid", "nope"])
def test_a_kind_that_draws_no_image_resolves_to_the_empty_string(kind):
    """So a caller can STORE ``''`` for "no colormap" without it meaning "use the default"."""

    assert resolved_cmap(kind, {}) == ""
    assert resolved_cmap(kind, {"cmap": "viridis"}) == "viridis"   # an explicit pick still wins


def test_only_a_grid_folds_in_the_per_cell_title_knob():
    grid = [decl.key for decl in panel_display_decls("grid", "hist")]
    plain = [decl.key for decl in panel_display_decls("hist", "hist")]
    assert grid == plain + [decl.key for decl in GRID_TITLE_PARAMS]
    assert "title_template" not in plain


def test_every_declared_colormap_default_is_one_the_chooser_offers():
    for kind, decls in PANEL_PARAMS.items():
        for decl in decls:
            if decl.key == "cmap":
                assert decl.default in CMAPS, (kind, decl.default)
                assert tuple(decl.choices) == CMAPS
