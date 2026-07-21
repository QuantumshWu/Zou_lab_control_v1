"""The two pulse representations in ``zlc_neutral_atom/timing`` are a NAMED boundary.

``pulse.py`` is the target pipeline's execution-session protocol.  ``pulse_table.py``
/ ``sequence_model.py`` / ``runtime_compiler.py`` are a LEGACY authoring model plus
the production compiler that still emits the machine-verified wire bytes (C22).

Legacy, not permanent: the design keeps exactly one authoring contract,
``schema="zlc_pulse.PulseDocument"`` (SYSTEM_ARCHITECTURE_DESIGN_zh §15.1), and
prescribes retirement by migrating each remaining consumer to that document in its
own dependency-closed slice.  An earlier version of this docstring claimed "both
sides are permanent" and cited a ledger row that does not exist; under C2 a ledger
row could not have made law anyway, and the design says the opposite.

The zero cross-references are the boundary itself, and a converter is NOT the way
across: the design forbids a ``PulseDocument <-> PulseTableState`` converter by
name, because bridging duplicates entrenches them.  Hence four mechanical clamps:

1. the two sides never import each other (either direction, lazy included);
2. the package's file list is an EXACT roster - a new module must come here and be
   placed on one side of the boundary (or outside it) by name;
3. the retirement condition stays in the package docstring, so the boundary cannot
   outlive its rationale unnoticed;
4. no module anywhere may import BOTH the legacy authoring model and the pulse
   document/authoring API -- which is precisely what a converter must do.
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
#: ``board_config.py`` joins ``ports.py``/``clock.py``/``streamer_geometry.py``: all four read
#: ONE hardware fact (pin map / port topology / tick rate / capacity) that the authoring model
#: and the production compiler both quantise against.  It is not a third pulse representation.
PRODUCTION_SIDE = {"pulse_table.py", "sequence_model.py", "runtime_compiler.py",
                   "ports.py", "serialization.py", "clock.py", "streamer_geometry.py",
                   "board_config.py"}


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):        # ast.walk covers lazy imports inside defs
        if isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


#: The legacy authoring model, and the one sanctioned document API.  A converter is
#: definitionally a module that reaches for both.
_LEGACY_AUTHORING = "zlc_neutral_atom.timing.pulse_table"
_DOCUMENT_API = ("zlc_pulse.document", "zlc_pulse.authoring")
#: Production packages; a test may legitimately import both to compare them.
_PRODUCTION_ROOTS = ("zlc_neutral_atom", "zlc_pulse", "zlc_workbench", "zlc_frontend",
                     "zlc_data", "zlc_storage", "Zou_lab_control")


def test_no_module_bridges_the_legacy_model_to_the_document_api():
    """A ``PulseDocument <-> PulseTableState`` converter is forbidden by the design.

    Prose cannot enforce that, and the prohibition is easy to breach with good
    intentions -- the bridge deleted in this commit was written to "connect a missing
    seam".  The mechanical signature of a converter is reaching for both sides at once,
    so that is what is refused here.  The way across the boundary is migrating a
    consumer onto the document, after which its legacy reader dies with it (C25).
    """

    offenders = []
    for root in _PRODUCTION_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            imported = _imported_modules(path)
            reaches_legacy = any(name.startswith(_LEGACY_AUTHORING) for name in imported)
            reaches_document = any(
                name.startswith(api) for name in imported for api in _DOCUMENT_API)
            if reaches_legacy and reaches_document:
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "these modules import BOTH the legacy authoring model and the document API, "
        "which is what a converter does -- the design forbids one by name:\n"
        + "\n".join(sorted(offenders))
    )


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
