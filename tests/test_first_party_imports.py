"""Every current first-party import must point at a module that exists."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

#: Import roots this repository owns.  Anything else is a third-party dependency
#: and none of this guard's business.
FIRST_PARTY = ("Zou_lab_control", "zlc_data", "zlc_storage", "zlc_pulse",
               "zlc_neutral_atom", "zlc_frontend", "zlc_workbench", "fpga")


def _tracked_python_files():
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return tuple(
        sorted(
            path
            for line in listing.stdout.splitlines()
            if line and (path := Path(line)).is_file()
        )
    )


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


def test_every_first_party_import_resolves():
    """Zero, with no migration exception list.

    There is deliberately no "the parent package exists, close enough" fallback:
    an import's module part must itself resolve.  ``from pkg import name`` puts
    ``pkg`` here and the interpreter checks ``name``; excusing a missing
    ``pkg.sub`` because ``pkg`` exists is precisely how a dead composition-bridge
    import once survived unnoticed.
    """

    broken = [
        (relative, target, lineno)
        for relative in _tracked_python_files()
        for target, lineno in _import_targets(relative)
        if not _module_exists(target)
    ]
    assert not broken, (
        "these files import modules that do not exist:\n"
        + "\n".join(f"  {p}:{n} -> {t}" for p, t, n in sorted(broken))
        + "\nA lazy import only fails when it runs, which is how a path stays "
          "broken unnoticed. Fix the target or delete the caller."
    )
