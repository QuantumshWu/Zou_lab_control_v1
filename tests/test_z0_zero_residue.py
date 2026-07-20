"""Zero residue as a NUMBER that may only fall, not a paragraph anyone can believe.

The design document's Z0 section is a prose checklist of things that "must be 0",
with no mechanical counterpart, and an audit found it unverifiable: it contained a
ghost entry, a clause referring to a section that only forward-references itself,
and no way to tell how far from zero anything actually was.  This file replaces
belief with arithmetic.

Every assertion is a property computed over ``git ls-files`` - never a hard-coded
list of names, because a name list cannot catch the file nobody remembered.  Each
one prints the offending paths so a failure is actionable without re-deriving it.

Two shapes, chosen per assertion:

* SHARP - the property already holds, so it is frozen as a ratchet.  Z4 is the
  important one: not one of the six target packages imports the legacy tree
  today, and this is what stops that from silently changing.
* BUDGET - the property does not hold yet.  The count is recorded, may only
  DECREASE, and reaching zero is what "migrated" means.  A budget is not a
  tolerance: it is a debt with a number on it.

Deliberately NOT here yet, so nothing pretends to be checked that is not:
Z10 (every UX-ledger row closed or approved) needs a parser for the design
document's table, and Z12 (every entry point resolves outside the rebuilt
workbench) needs to import the product entries at test time.  Both land with the
window work that makes them meaningful.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

#: The legacy subpackages Z0 deletes.  ``Zou_lab_control`` itself is NOT legacy -
#: ``Zou_lab_control.notebook`` and ``.workbench`` are the user's entry points and
#: survive - so the prefixes are named individually rather than by their root.
LEGACY_PREFIXES = (
    "Zou_lab_control.frontend",
    "Zou_lab_control.neutral_atom",
    "Zou_lab_control._paths",
    "Zou_lab_control._readout_math",
    "Zou_lab_control._viewer_registry",
    "Zou_lab_control._clock",
    "Zou_lab_control._streamer_geometry",
)

LEGACY_TREES = ("Zou_lab_control/frontend/", "Zou_lab_control/neutral_atom/")

#: The six target packages.  ``fpga`` is deliberately excluded: it is a separate
#: published package with its own row in Z2, not one of the six.
TARGET_PACKAGES = ("zlc_data/", "zlc_storage/", "zlc_pulse/", "zlc_neutral_atom/",
                   "zlc_frontend/", "zlc_workbench/")

# --------------------------------------------------------------------- budgets
# Every number below was MEASURED, not estimated, and is mirrored in
# the design doc's S4 progress ledger.  They may only be lowered.

Z1_LEGACY_FILES = 122          # tracked files under the two legacy trees
Z2A_PRODUCTION_IMPORTERS = {   # non-test code OUTSIDE legacy that still imports it
    Path("docs/task_console_design/build.py"),
    Path("figure_viewer.py"),                        # published double-click launcher
    Path("fpga/pulse_streamer/sim/_gen_replay_t.py"),  # fpga -> legacy, and zlc_pulse -> fpga
}
#: Whitelisted tests that reach into the legacy tree, NAMED rather than counted.
#: A count would let one more slip in unnoticed; naming each makes every addition
#: a decision.  All seven are legitimate today and all die with what they guard:
#: the migration ledgers must build the legacy windows in order to compare
#: against them, which is the whole point of a parity oracle.
Z2B_ACTIVE_TEST_IMPORTERS = {
    Path("tests/test_calibration_sitemap_inputs.py"),   # legacy calibration inputs
    Path("tests/test_public_hardware_boundary.py"),     # audits the legacy surface itself
    Path("tests/test_u04_signal_picker_owner.py"),      # picker ownership across the move
    Path("tests/test_u05_declaration_not_class.py"),    # pins predicates ON legacy console/domain nodes
    Path("tests/test_u05_render_island.py"),            # proves the legacy console reaches the loop only via the island
    Path("tests/test_u05_raster_front.py"),             # pins the legacy canvas' worker/GUI hand-off to the documented raster
    Path("tests/test_u06_shell_domain_ports.py"),       # proves the legacy root wires the port
    Path("tests/test_zlc_frontend_form.py"),            # form parity across the move
}
Z2C_FROZEN_TEST_IMPORTERS = 168
Z3_FORWARDING_SHIMS = 18
Z7_REBUILT_WORKBENCH_FILES = 12
Z8_TESTS_OFF_THE_MANIFEST = 176

#: The S0.5 bridges that were actually BUILT.  The design document names four
#: symbols; two of them (``LegacyPanelHost``, ``LegacyRuntimeFence``) have zero
#: class definitions in the repository and never did - see the design doc
#: section 3 F4.  Asserting on the two real ones keeps Z0 from hunting ghosts.
Z6_BRIDGES = ("CatalogRouter", "SerializedLegacyAggBridge")
Z6_BRIDGE_DEFINITIONS = 2


def _tracked(pattern: str = "") -> tuple[str, ...]:
    """Tracked files PLUS untracked-but-not-ignored ones.

    ``git ls-files`` alone lists only the index, so a brand new file is invisible
    until it is staged - and a guard that cannot see the file you just wrote
    gives false comfort exactly when you are most likely to be adding residue.
    Found by mutation: creating a new forwarding shim did not move the Z3 budget.
    """

    args = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    if pattern:
        args.append(pattern)
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return tuple(sorted({line for line in out.stdout.split() if line}))


def _imported_roots(relative: str) -> set[str]:
    """Every absolute module this file imports, including lazy in-function ones."""

    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _legacy_importers() -> set[str]:
    return {
        path for path in _tracked("*.py")
        if any(name.startswith(LEGACY_PREFIXES) for name in _imported_roots(path))
    }


def _fail(label: str, offenders, budget: int, remedy: str):
    offenders = sorted(offenders)
    assert len(offenders) <= budget, (
        f"{label}: {len(offenders)} exceeds the recorded budget of {budget}.\n"
        + "\n".join(f"  {item}" for item in offenders[:40])
        + (f"\n  ... and {len(offenders) - 40} more" if len(offenders) > 40 else "")
        + f"\n{remedy}"
    )


# ------------------------------------------------------------------ the ratchet


def test_z4_no_target_package_imports_the_legacy_tree():
    """SHARP. The one property already true, and the one worth never losing.

    A single lazy import from any of the six into the legacy tree would make the
    new architecture undeletable, and it would not show up until that line ran.
    """

    offenders = sorted(
        f"{path} -> {name}"
        for path in _tracked("*.py") if path.startswith(TARGET_PACKAGES)
        for name in _imported_roots(path)
        if name == "Zou_lab_control" or name.startswith("Zou_lab_control.")
    )
    assert not offenders, (
        "a target package reached into the legacy tree:\n"
        + "\n".join(f"  {item}" for item in offenders)
        + "\nThe six packages must be deletable-independent of Zou_lab_control. "
          "If a live domain object is genuinely needed, add a port in "
          "zlc_frontend/domain_ports.py and inject it from a composition root."
    )


def test_z1_the_legacy_trees_shrink():
    files = [p for p in _tracked() if p.startswith(LEGACY_TREES)]
    _fail("Z1 legacy tracked files", files, Z1_LEGACY_FILES,
          "Files leave these trees by being salvaged into a target package "
          "(design doc L4648, shell salvage), never by being copied.")


def test_z2a_production_code_outside_legacy_has_a_named_ledger():
    """Sharp on identity, not just count: three files, each with a known fate.

    build.py is a documentation builder, figure_viewer.py is the published
    double-click launcher (window W3), and _gen_replay_t.py is the fpga->legacy
    edge that matters because eleven modules in zlc_pulse/zlc_neutral_atom import
    fpga - so the new pulse stack depends transitively on the legacy tree.
    """

    found = {
        Path(path) for path in _legacy_importers()
        if not path.startswith(LEGACY_TREES) and not path.startswith("tests/")
    }
    grew = found - Z2A_PRODUCTION_IMPORTERS
    assert not grew, (
        "new production code outside the legacy trees now imports them:\n"
        + "\n".join(f"  {item}" for item in sorted(grew))
    )
    gone = Z2A_PRODUCTION_IMPORTERS - found
    assert not gone, (
        f"{sorted(str(p) for p in gone)} no longer import legacy; delete those "
        "rows from Z2A_PRODUCTION_IMPORTERS in the same commit."
    )


def test_z2bc_test_dependence_on_legacy_shrinks():
    manifest = {
        line.strip().replace("\\", "/")
        for line in (ROOT / "tests" / "migration_active_tests.txt")
        .read_text(encoding="utf-8").split()
        if line.strip()
    }
    importers = {p for p in _legacy_importers() if p.startswith("tests/")}
    active = {p for p in importers if p in manifest}
    frozen = importers - active

    named = {Path(path) for path in active}
    grew = named - Z2B_ACTIVE_TEST_IMPORTERS
    assert not grew, (
        "a whitelisted test started reaching into the legacy tree:\n"
        + "\n".join(f"  {item}" for item in sorted(grew))
        + "\nIf it genuinely must build a legacy window to compare against, add it "
          "to Z2B_ACTIVE_TEST_IMPORTERS with the reason - a NAMED dependency is a "
          "decision, a counted one is a leak."
    )
    gone = Z2B_ACTIVE_TEST_IMPORTERS - named
    assert not gone, (
        f"{sorted(str(p) for p in gone)} no longer reach legacy; delete those rows "
        "from Z2B_ACTIVE_TEST_IMPORTERS in the same commit."
    )
    _fail("Z2c frozen tests importing legacy", frozen, Z2C_FROZEN_TEST_IMPORTERS,
          "Frozen files are never edited; this budget falls only by DELETING them.")


def test_z3_forwarding_shims_shrink():
    shims = [
        p for p in _tracked("*.py")
        if not p.startswith("tests/") and "MOVED to" in (ROOT / p).read_text(encoding="utf-8")
    ]
    _fail("Z3 forwarding shims", shims, Z3_FORWARDING_SHIMS,
          "A shim exists only while both paths must work. Z0 deletes every one.")


def test_z6_only_the_bridges_that_exist_are_tracked():
    definitions = sorted(
        f"{path}:{node.lineno} {node.name}"
        for path in _tracked("*.py")
        for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path))
        if isinstance(node, ast.ClassDef) and node.name in Z6_BRIDGES
    )
    _fail("Z6 migration bridges still defined", definitions, Z6_BRIDGE_DEFINITIONS,
          "The design document requires all migration bridges to be 0 at Z0.")

    ghosts = sorted(
        name for name in ("LegacyPanelHost", "LegacyRuntimeFence")
        for path in _tracked("*.py")
        for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path))
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    assert not ghosts, (
        f"{ghosts} now exist as classes. They never did - the design document "
        "names them but nobody wrote them (design doc, S4 ledger). "
        "If one is genuinely being built, move it into Z6_BRIDGES."
    )


def test_z7_the_rebuilt_workbench_shrinks():
    files = [p for p in _tracked("*.py") if p.startswith("Zou_lab_control/workbench/")]
    _fail("Z7 rebuilt workbench modules", files, Z7_REBUILT_WORKBENCH_FILES,
          "Each falls when its window's UX is restored onto the section 12.5 "
          "board and the entry point is repointed - the two happen in ONE commit.")


def test_z8_the_evidence_surface_grows_to_cover_every_test():
    manifest = {
        line.strip().replace("\\", "/")
        for line in (ROOT / "tests" / "migration_active_tests.txt")
        .read_text(encoding="utf-8").split()
        if line.strip()
    }
    all_tests = {p for p in _tracked() if fnmatch.fnmatch(p, "tests/test_*.py")}
    _fail("Z8 tests off the manifest", all_tests - manifest, Z8_TESTS_OFF_THE_MANIFEST,
          "This is the anti-cheat clause: while files sit outside the manifest, "
          "'the suite is green' means 'the collected part is green'. The budget "
          "falls by deleting a frozen file or by making it pass and listing it - "
          "never by moving a red test OUT.")


