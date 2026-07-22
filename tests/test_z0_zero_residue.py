"""Zero legacy residue -- the end state, asserted as arithmetic over the whole tree.

The one-shot purge (directive 2026-07-21) deleted the legacy backend trees, the
legacy render pipeline and every migration bridge outright.  What remains is the
original UI skeletons on the current zlc_* stack.  This file keeps that true:
no migration bridges or allowlists -- every count below is zero and stays zero.

Every assertion is a property computed over ``git ls-files`` (plus untracked,
non-ignored files) -- never a hard-coded list of names, because a name list
cannot catch the file nobody remembered.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

#: Module prefixes that stopped existing at the purge.  An import of any of these,
#: absolute or relative, eager or lazy, is residue.
DEAD_PREFIXES = (
    "Zou_lab_control.frontend",
    "Zou_lab_control.neutral_atom",
    "Zou_lab_control._clock",
    "Zou_lab_control._paths",
    "Zou_lab_control._readout_math",
    "Zou_lab_control._streamer_geometry",
    "Zou_lab_control._viewer_registry",
    "zlc_frontend.live_plot",
    "zlc_frontend.qt_widgets.render_loop",
    "zlc_workbench.task_console.plot_bridge_canvas",
    "zlc_workbench.legacy",
)

#: Paths whose very existence is residue (trees and single files the purge removed).
DEAD_PATHS = (
    "Zou_lab_control/frontend",
    "Zou_lab_control/neutral_atom",
    "Zou_lab_control/_clock.py",
    "Zou_lab_control/_paths.py",
    "Zou_lab_control/_readout_math.py",
    "Zou_lab_control/_streamer_geometry.py",
    "Zou_lab_control/_viewer_registry.py",
    "zlc_frontend/live_plot",
    "zlc_frontend/qt_widgets/render_loop.py",
    "zlc_workbench/task_console/plot_bridge_canvas.py",
    "zlc_workbench/legacy.py",
)

#: Migration-bridge classes.  All died with the purge; a re-definition anywhere is
#: a bridge being rebuilt, which the end state forbids.
DEAD_BRIDGE_CLASSES = ("CatalogRouter", "SerializedLegacyAggBridge",
                       "LegacyPanelHost", "LegacyRuntimeFence")


def _tracked(pattern: str = "") -> tuple[str, ...]:
    """Tracked files PLUS untracked-but-not-ignored ones, so a brand-new file is
    visible to the guard before it is staged."""

    args = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    if pattern:
        args.append(pattern)
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return tuple(sorted({line for line in out.stdout.split() if line}))


def _is_dead(dotted: str) -> bool:
    return any(dotted == p or dotted.startswith(p + ".") for p in DEAD_PREFIXES)


def _resolved_imports(relative: str):
    """Every absolute dotted target this file imports, with relative imports
    resolved against the file's package -- a ``from .render_loop import X`` inside
    ``zlc_frontend/qt_widgets/`` must count as ``zlc_frontend.qt_widgets.render_loop``."""

    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    parts = relative[:-3].split("/")
    package = parts[:-1] if not relative.endswith("__init__.py") else parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                target = node.module or ""
            else:
                if node.level > len(package):
                    continue
                anchor = package[: len(package) - node.level + 1]
                target = ".".join([*anchor, *([node.module] if node.module else [])])
            if target:
                yield target, node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno


def test_the_dead_paths_stay_dead():
    present = [p for p in DEAD_PATHS if (ROOT / p).exists()]
    assert not present, f"purged paths have reappeared: {present}"
    ghosts = [p for p in _tracked() if p.startswith(tuple(t + "/" for t in DEAD_PATHS))]
    assert not ghosts, f"files exist under purged trees: {ghosts[:20]}"


def test_no_python_file_imports_a_dead_module():
    offenders = sorted(
        f"{path}:{lineno} -> {target}"
        for path in _tracked("*.py")
        for target, lineno in _resolved_imports(path)
        if _is_dead(target)
    )
    assert not offenders, (
        "these files import purged legacy modules:\n"
        + "\n".join(f"  {item}" for item in offenders[:40])
    )


def test_no_migration_bridge_class_is_defined():
    definitions = sorted(
        f"{path}:{node.lineno} {node.name}"
        for path in _tracked("*.py")
        for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path))
        if isinstance(node, ast.ClassDef) and node.name in DEAD_BRIDGE_CLASSES
    )
    assert not definitions, f"migration bridges are being rebuilt: {definitions}"


def test_no_forwarding_shim_remains():
    shims = [
        p for p in _tracked("*.py")
        if not p.startswith("tests/")
        and "MOVED to" in (ROOT / p).read_text(encoding="utf-8")
    ]
    assert not shims, f"forwarding shims remain: {shims}"


def test_the_manifest_is_the_suite():
    """Anti-cheat: a test file off the manifest is a mistake, not an archive --
    while files sit outside it, 'the suite is green' means 'the collected part
    is green'."""

    manifest = {
        line.strip().replace("\\", "/")
        for line in (ROOT / "tests" / "migration_active_tests.txt")
        .read_text(encoding="utf-8").split()
        if line.strip()
    }
    all_tests = {p for p in _tracked() if fnmatch.fnmatch(p, "tests/test_*.py")}
    off = sorted(all_tests - manifest)
    assert not off, f"tests off the manifest: {off}"
    missing = sorted(p for p in manifest if not (ROOT / p).exists())
    assert not missing, f"manifest names tests that do not exist: {missing}"
