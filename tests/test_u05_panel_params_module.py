"""Current TaskConsole panel parameters name only renderer-consumed facts."""

from __future__ import annotations

import ast
from pathlib import Path

from zlc_frontend.panel_params import (
    CMAPS,
    PANEL_PARAMS,
    panel_param_decls,
    resolved_panel_param,
)


REPO = Path(__file__).resolve().parents[1]


def test_the_catalog_module_pulls_in_no_toolkit() -> None:
    tree = ast.parse(
        (REPO / "zlc_frontend" / "panel_params.py").read_text(encoding="utf-8")
    )
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0] if node.level == 0 else "")
    assert "PyQt5" not in roots
    assert roots <= {"", "__future__", "typing", "zlc_data"}


def test_only_current_renderer_keys_are_declared() -> None:
    assert tuple(decl.key for decl in panel_param_decls("2d")) == ("colormap",)
    assert tuple(decl.key for decl in panel_param_decls("hist")) == ("bins",)
    assert panel_param_decls("monitor") == ()
    assert set(PANEL_PARAMS) == {"2d", "hist"}


def test_every_colormap_declaration_uses_the_current_closed_vocabulary() -> None:
    declaration = panel_param_decls("2d")[0]
    assert declaration.default in CMAPS
    assert tuple(declaration.choices) == CMAPS


def test_render_resolution_uses_the_same_declared_default_as_the_form() -> None:
    assert resolved_panel_param("2d", {}, "colormap") == "inferno"
    assert resolved_panel_param("hist", {}, "bins") == 60
    assert resolved_panel_param("hist", {"bins": 0}, "bins") == 0
