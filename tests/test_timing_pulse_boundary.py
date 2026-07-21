"""The two pulse representations in ``zlc_neutral_atom/timing`` are a NAMED boundary.

Adjudicated 2026-07-21 (ledger row "timing 双源裁决"; the transitional death
condition was superseded the same day -- both sides are permanent).  ``pulse.py``
is the target pipeline's execution-session protocol; ``pulse_table.py``/
``sequence_model.py``/``runtime_compiler.py`` are the authoring model + production
compiler of the machine-verified pipeline (C22).  Their zero cross-references are
the boundary itself: an import in either direction would silently marry the two
pipelines before the authoring bridge exists, and a THIRD pulse representation
slipping into the package would recreate the very dual source the adjudication
resolved.  Hence three mechanical clamps:

1. the two sides never import each other (either direction, lazy included);
2. the package's file list is an EXACT roster - a new module must come here and be
   placed on one side of the boundary (or outside it) by name;
3. the adjudication text itself (the missing authoring bridge) stays in the
   package docstring, so the boundary cannot outlive its rationale unnoticed.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
TIMING = REPO / "zlc_neutral_atom" / "timing"

#: The target pipeline's side of the boundary (consumes the zlc_pulse IR).
TARGET_SIDE = {"pulse.py", "capture.py", "capture_plan.py", "lineage.py",
               "occupancy.py", "segmented.py", "_coordination.py"}
#: The production pipeline's side: authoring model + the machine-verified compiler.
PRODUCTION_SIDE = {"pulse_table.py", "sequence_model.py", "runtime_compiler.py",
                   "ports.py", "serialization.py", "clock.py", "streamer_geometry.py"}


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):        # ast.walk covers lazy imports inside defs
        if isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def test_the_package_roster_is_exact():
    actual = {p.name for p in TIMING.glob("*.py")}
    expected = TARGET_SIDE | PRODUCTION_SIDE | {"__init__.py"}
    assert actual == expected, (
        f"unrostered: {sorted(actual - expected)}; missing: {sorted(expected - actual)}. "
        "A new module in zlc_neutral_atom/timing must be placed on one side of the "
        "pulse boundary BY NAME here (or live outside the package)."
    )


def test_neither_pulse_side_imports_the_other():
    stems = {"target": {f[:-3] for f in TARGET_SIDE},
             "production": {f[:-3] for f in PRODUCTION_SIDE}}
    violations = []
    for side, other in (("target", "production"), ("production", "target")):
        for name in sorted(stems[side]):
            for module in _imported_modules(TIMING / f"{name}.py"):
                tail = module.lstrip(".").rsplit(".", 1)[-1]
                if tail in stems[other] and (module.startswith(".")
                                             or "timing" in module):
                    violations.append(f"{side}/{name}.py imports {module}")
    assert not violations, (
        "the two pulse pipelines married before the authoring bridge exists:\n"
        + "\n".join(violations)
    )


def test_the_adjudication_travels_with_the_package():
    doc = ast.get_docstring(ast.parse((TIMING / "__init__.py").read_text(encoding="utf-8")))
    for needle in ("pulse.py", "pulse_table.py", "runtime_compiler.py", "bridge", "C22"):
        assert needle in doc, f"the boundary lost its rationale: {needle!r} missing"
