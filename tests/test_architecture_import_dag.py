"""Mechanical import-boundary ratchet for the target bounded contexts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "zlc_data": ("Zou_lab_control", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_storage": ("Zou_lab_control", "zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_pulse": ("Zou_lab_control", "zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_workbench"),
    "zlc_frontend": ("Zou_lab_control", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_neutral_atom": ("Zou_lab_control", "zlc_frontend", "zlc_workbench"),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            result.add(node.module)
    return result


@pytest.mark.parametrize("package,forbidden", FORBIDDEN.items())
def test_target_package_has_no_reverse_imports(package, forbidden):
    root = ROOT / package
    if not root.exists():
        pytest.skip(f"{package} has not entered its migration slice")
    violations = []
    for path in root.rglob("*.py"):
        for imported in _imports(path):
            if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
            if package == "zlc_data" and imported.startswith("zlc_storage"):
                if imported != "zlc_storage.canonical" and not imported.startswith(
                    "zlc_storage.canonical."
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)} imports storage I/O boundary {imported}"
                    )
    assert not violations, "reverse package dependencies:\n" + "\n".join(violations)
