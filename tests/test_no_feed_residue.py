"""Mechanical guard that the removed task-console ``Feed`` abstraction stays gone.

The old layer name is an architectural identifier, not a forbidden English verb.
UART parsers legitimately ``feed`` byte streams and prose may describe one node feeding
another; neither recreates the deleted layer.  This guard therefore inspects executable
identifiers and exact string tokens only in the task-console graph/logic boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
_ARCHITECTURE_FILES = (
    REPO_ROOT / "Zou_lab_control" / "frontend" / "task_console.py",
    REPO_ROOT / "Zou_lab_control" / "frontend" / "live.py",
    REPO_ROOT / "Zou_lab_control" / "frontend" / "flow_graph_view.py",
    REPO_ROOT / "Zou_lab_control" / "neutral_atom" / "operations" / "logic.py",
    REPO_ROOT / "Zou_lab_control" / "neutral_atom" / "operations" / "figure_capture.py",
)


def _architectural_identifiers(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A layer/registry key is an exact token.  Docstrings and user prose are not.
            if node.value.strip().casefold() == "feed":
                yield node.value, node.lineno


def test_removed_feed_layer_has_no_architectural_identifier():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {name}"
        for path in _ARCHITECTURE_FILES
        for name, line in _architectural_identifiers(path)
        if str(name).casefold() == "feed"
    ]
    assert offenders == [], (
        "the removed task-console Feed layer reappeared as an executable identifier:\n  "
        + "\n  ".join(offenders)
    )
