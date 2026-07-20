"""Every first-party import must point at a module that exists.

A migration renames and deletes packages for a living, and Python only notices a
broken import when the line actually runs.  This codebase imports lazily almost
everywhere - to keep Qt, SciPy and the pulse compiler off the import path - so a
target can rot for a long time with nobody finding out.  It did: the legacy
session root has been raising ``ModuleNotFoundError`` on
``zlc_workbench.legacy_neutral_atom``, a module that does not exist, and
``Zou_lab_control.neutral_atom.session.connect('virtual')`` has been dead.
Nothing reported it, because the only tests covering that path are outside the
active manifest and are therefore never collected.

The three populations are asserted separately, because "how bad is it" differs by
an order of magnitude between them and one blended number would hide that:

* PRODUCTION code - sharp.  Exactly the two ledgered lines, nothing else.
* ACTIVE tests (the authorised evidence surface) - sharp, and ZERO.
* FROZEN tests - a declining budget.  Rule 8 freezes the 176 unlisted test files
  and Z0 deletes them; repairing their imports is explicitly out of scope, so the
  only honest thing is to COUNT the rot and forbid it from growing.  When those
  files are deleted the budget goes to zero on its own.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

#: Import roots this repository owns.  Anything else is a third-party dependency
#: and none of this guard's business.
FIRST_PARTY = ("Zou_lab_control", "zlc_data", "zlc_storage", "zlc_pulse",
               "zlc_neutral_atom", "zlc_frontend", "zlc_workbench", "fpga")

#: Broken imports in PRODUCTION code, with why they are not repaired here.
#:
#: Both are S0.5 "temporary composition bridge" imports whose target module was
#: never created in ``zlc_workbench``.  They are NOT resurrected on a guess about
#: what the bridge provided: the legacy session root is a 完成态不存在 item - the
#: design document allows exactly one headless session root,
#: ``Zou_lab_control.notebook.connect`` - so these lines die with their files.
#: ``_gui.py`` goes with the pulse editor window (W2), ``session.py`` with the
#: legacy tree at Z0.  Until then the breakage is at least VISIBLE, which is the
#: entire point: it was not, and that is how it survived.
KNOWN_DEAD_PRODUCTION = {
    (Path("Zou_lab_control/neutral_atom/session.py"),
     "zlc_workbench.legacy_neutral_atom"),
    (Path("Zou_lab_control/neutral_atom/_gui.py"),
     "zlc_workbench.pulse_control"),
}

#: Broken imports inside FROZEN (non-manifest) test files.  Overwhelmingly
#: ``Zou_lab_control.frontend.qt_fluent`` and ``.style``, which moved into
#: ``zlc_frontend`` in earlier slices; the frozen tests still name the old paths.
#: MAY ONLY DECREASE.  Recorded in the design doc's S4 ledger alongside the other
#: residue budgets.
FROZEN_TEST_BUDGET = 165


def _tracked_python_files():
    listing = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return tuple(sorted(Path(line) for line in listing.stdout.split() if line))


def _module_exists(dotted: str) -> bool:
    """Whether ``dotted`` resolves to a module or package file in this repo."""

    base = ROOT.joinpath(*dotted.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _absolute_target(relative_path: Path, module: str | None, level: int) -> str | None:
    """The absolute dotted name an import refers to, or ``None`` if not ours.

    ``level`` is the number of leading dots: ``from ..core import x`` inside
    ``a/b/c.py`` resolves against ``a``.
    """

    if level == 0:
        target = module or ""
    else:
        package = list(relative_path.parts[:-1])
        if level > len(package):
            return None
        anchor = package[: len(package) - level + 1]
        target = ".".join([*anchor, *([module] if module else [])])
    return target if target.startswith(FIRST_PARTY) else None


def _import_targets(relative_path: Path):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"),
                     filename=str(relative_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _absolute_target(relative_path, node.module, node.level)
            if target:
                yield target, node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FIRST_PARTY):
                    yield alias.name, node.lineno


def _unresolved():
    """Every (path, target, line) whose module part does not exist on disk.

    There is deliberately no "the parent package exists, close enough" fallback:
    an import's module part must itself resolve.  ``from pkg import name`` puts
    ``pkg`` here and the interpreter checks ``name``; excusing a missing
    ``pkg.sub`` because ``pkg`` exists is precisely what let
    ``zlc_workbench.legacy_neutral_atom`` pass unnoticed.
    """

    for relative in _tracked_python_files():
        for target, lineno in _import_targets(relative):
            if not _module_exists(target):
                yield relative, target, lineno


def _manifest_paths():
    text = (ROOT / "tests" / "migration_active_tests.txt").read_text(encoding="utf-8")
    return {line.strip().replace("\\", "/") for line in text.split() if line.strip()}


def _partition():
    manifest = _manifest_paths()
    production, active, frozen = set(), set(), set()
    lines = {}
    for relative, target, lineno in _unresolved():
        posix = str(relative).replace("\\", "/")
        lines[(relative, target)] = lineno
        if not posix.startswith("tests/"):
            production.add((relative, target))
        elif posix in manifest:
            active.add((relative, target))
        else:
            frozen.add((relative, target))
    return production, active, frozen, lines


def test_production_code_has_no_unledgered_broken_import():
    production, _active, _frozen, lines = _partition()

    grew = production - KNOWN_DEAD_PRODUCTION
    assert not grew, (
        "production code imports modules that do not exist:\n"
        + "\n".join(f"  {p}:{lines[(p, t)]} -> {t}" for p, t in sorted(grew))
        + "\nA lazy import only fails when it runs, which is how a path stays "
          "broken unnoticed. Fix the target or delete the caller."
    )

    repaired = KNOWN_DEAD_PRODUCTION - production
    assert not repaired, (
        f"{sorted(f'{p} -> {t}' for p, t in repaired)} resolve again; delete "
        "those rows from KNOWN_DEAD_PRODUCTION in the same commit."
    )


def test_the_active_evidence_surface_has_no_broken_import():
    """Zero, with no ledger: a guard that cannot be collected proves nothing."""

    _production, active, _frozen, lines = _partition()
    assert not active, (
        "these WHITELISTED tests import modules that do not exist, so they are "
        "evidence in name only:\n"
        + "\n".join(f"  {p}:{lines[(p, t)]} -> {t}" for p, t in sorted(active))
    )


def test_the_frozen_test_rot_only_shrinks():
    """The debt the manifest gate hides, as a number that may only go down.

    Frozen tests are not repaired (rule 8 freezes them, Z0 deletes them) - but
    leaving the count unmeasured is how ``test_legacy_runtime_fence.py`` came to
    be dead-on-import with nobody aware.
    """

    _production, _active, frozen, _lines = _partition()
    assert len(frozen) <= FROZEN_TEST_BUDGET, (
        f"frozen-test broken imports grew to {len(frozen)}, over the "
        f"{FROZEN_TEST_BUDGET} budget. Frozen files are not edited - a rise means "
        "live code moved out from under them, so move the budget only by "
        "DELETING frozen files, never by editing them."
    )
