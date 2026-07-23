"""The charter is law only while a machine holds it to its own words.

Every clause here exists because prose alone already failed once in this repository: the sole
authority grew a 900-line changelog, the executor started quoting its own old ledger rows as if
they were the document, and a false premise ("the shell must enter zlc_frontend") survived 25
rounds without anyone re-reading the source.  The charter (docs/DESIGN_CHARTER_zh.md) is the
structural fix; this file is what keeps the fix from rotting the same way.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTER = ROOT / "docs" / "DESIGN_CHARTER_zh.md"
LEDGER = ROOT / "docs" / "MIGRATION_LEDGER_zh.md"
DESIGN = ROOT / "docs" / "SYSTEM_ARCHITECTURE_DESIGN_zh.md"
QT_WIDGETS = ROOT / "zlc_frontend" / "qt_widgets"

#: C21 -- the line ratchet.  EXACT recorded sizes for the grandfathered monoliths (they may only
#: FALL: lower the number in the same commit that shrinks the file); everything else, and every
#: new file, obeys the hard cap.  Equalities on purpose (C6): a ceiling that only notices growth
#: stops ratcheting the moment someone forgets to lower it.
QT_WIDGETS_LINE_CAP = 600
#: The 2026-07-22 UI locality cut established the current baseline after moving
#: the shared Fluent owner and keyed form reconciliation out of application
#: windows.  These files may only shrink after that cut; new widget modules
#: still obey the cap.
GRANDFATHERED = {
    "board.py": 4515,
    "fluent.py": 3820,
    "form.py": 735,
    "param_widgets.py": 769,
}


def test_the_charter_stays_short_enough_to_actually_read_every_round():
    """The whole root-cause theory is that a 4700-line authority cannot be re-read per round and
    therefore gets imagined instead.  300 lines is the promise the charter makes about itself."""

    lines = CHARTER.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 300, f"charter grew to {len(lines)} lines -- it is becoming the old doc"


def test_the_law_numbers_are_unique_and_the_hierarchy_is_declared():
    text = CHARTER.read_text(encoding="utf-8")
    numbers = re.findall(r"\*\*C(\d+)", text)
    assert numbers, "no numbered laws found"
    assert len(numbers) == len(set(numbers)), sorted(
        n for n in set(numbers) if numbers.count(n) > 1)
    for name in ("MIGRATION_LEDGER_zh.md", "SYSTEM_ARCHITECTURE_DESIGN_zh.md"):
        assert name in text, f"the charter must declare where {name} sits in the hierarchy"


def test_the_design_docs_section_22_stays_a_frozen_pointer():
    """The ledger was moved OUT because law drowned in it.  If rows start accreting in the design
    doc again, the disease is back regardless of what the charter says."""

    text = DESIGN.read_text(encoding="utf-8")
    start = text.index("## 22.")
    end = text.index("## 23.")
    body = text[start:end]
    assert "MIGRATION_LEDGER_zh.md" in body and "DESIGN_CHARTER_zh.md" in body
    assert len(body.splitlines()) < 15, "section 22 is growing again -- rows belong in the ledger"
    assert "| S5-shell" not in body


def test_new_ledger_rows_obey_the_cap_and_cite_the_law():
    """C2, mechanically.  Applies only to the capped section -- the historical rows above it are
    the disease being isolated, not a standard to meet."""

    text = LEDGER.read_text(encoding="utf-8")
    marker = "## 新台账"
    assert marker in text, "the capped section is gone"
    section = text[text.index(marker):]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("|")
        and "---" not in line
        and not re.match(r"\|\s*(日期|优先级)\s*\|", line)
    ]
    assert rows, "the capped section has no rows yet the pivot was recorded there"
    offenders = []
    for row in rows:
        if len(row) > 700:
            offenders.append(f"row too long ({len(row)} chars): {row[:60]}...")
        if not re.search(r"C\d+", row):
            offenders.append(f"row cites no charter law: {row[:60]}...")
    assert not offenders, "\n".join(offenders)


def test_the_qt_widgets_line_ratchet_holds_exactly():
    """C21.  Grandfathered files match their recorded size exactly (shrink the file -> lower the
    number, same commit); everything else stays under the cap.  This is the mechanical form of
    'no more 10k-line GUI files'."""

    offenders = []
    for path in sorted(QT_WIDGETS.glob("*.py")):
        n = len(path.read_text(encoding="utf-8").splitlines())
        recorded = GRANDFATHERED.get(path.name)
        if recorded is not None:
            if n != recorded:
                offenders.append(f"{path.name}: {n} lines, recorded {recorded} "
                                 f"({'update the record downward' if n < recorded else 'GREW'})")
        elif n > QT_WIDGETS_LINE_CAP:
            offenders.append(f"{path.name}: {n} lines exceeds the {QT_WIDGETS_LINE_CAP} cap")
    assert not offenders, "\n".join(offenders)


def test_every_scaffolding_test_dies_with_an_artifact_that_still_exists():
    """C41.  A test declaring ``DIES_WITH = <path>`` guards a legacy artifact and must be deleted
    in the same commit that deletes the artifact.  This sweep is the enforcement: a declaring
    test whose artifact is GONE is residue pretending to be coverage."""

    def paths_of(value):
        if isinstance(value, ast.Constant):
            return [str(value.value)]
        if isinstance(value, (ast.Tuple, ast.List)):
            return [str(e.value) for e in value.elts if isinstance(e, ast.Constant)]
        return []

    stale, declared = [], 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        test_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "DIES_WITH" in names:
                declared += 1
                for rel in paths_of(node.value):
                    if not (ROOT / rel).exists():
                        stale.append(f"{path.name} dies with {rel}, which is gone")
            elif "DIES_WITH_PARTIAL" in names and isinstance(node.value, ast.Dict):
                declared += 1
                for key, value in zip(node.value.keys, node.value.values):
                    test = str(key.value)
                    if test not in test_names:
                        stale.append(f"{path.name}: DIES_WITH_PARTIAL names unknown test {test!r}")
                    for rel in paths_of(value):
                        if not (ROOT / rel).exists():
                            stale.append(f"{path.name}::{test} dies with {rel}, which is gone "
                                         f"-- delete the test and its map entry")
    assert not stale, "\n".join(stale)
    # No minimum count: the one-shot purge (2026-07-21) deleted the legacy artifacts and
    # their DIES_WITH oracles together, so a low count is the healthy end state -- what
    # matters is only that any REMAINING declaration still points at a live artifact.


def test_the_manifest_is_the_suite():
    """C41 -- there is no third state.  Every test file is on the manifest and every manifest
    entry exists (Z8=0 in test_z0_zero_residue asserts the same from the other side)."""

    manifest = {line.strip().replace("tests/", "") for line in
                (ROOT / "tests" / "migration_active_tests.txt").read_text(encoding="utf-8")
                .replace("\r", "").splitlines() if line.strip()}
    files = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    assert manifest == files, (
        f"manifest-only: {sorted(manifest - files)}; file-only: {sorted(files - manifest)}")


def test_the_deletion_ledger_covers_every_legacy_file():
    """C24/C25, completed.  The census the ledger existed for is finished: the one-shot
    purge (2026-07-21) deleted both legacy trees outright, so the exhaustiveness claim
    collapses to its end state -- ZERO tracked legacy files, forever."""

    import subprocess

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    legacy = sorted(p for p in tracked
                    if p.startswith(("Zou_lab_control/frontend/", "Zou_lab_control/neutral_atom/")))
    assert not legacy, f"legacy trees have tracked files again: {legacy[:10]}"


def test_no_file_marries_qt_to_matplotlib_outside_the_sanctioned_zones():
    """C14 -- the render end state, enforced from today.  Qt sees pixels; matplotlib lives in
    the headless render leaf; the ONLY places allowed to touch both are the legacy frontend
    tree (which Z0 deletes wholesale) and the workbench plot_bridge transitional zone (which
    the post-migration worker-raster rework empties).  Measured at adoption: exactly five
    dual importers, all inside the legacy tree -- so a sixth anywhere else is a new marriage,
    not an inherited one."""

    import ast as _ast

    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        # tools/ carries dev probes (e.g. the window-fingerprint harness); they are not product
        # code, and a probe that builds real windows inevitably touches both toolkits.
        if ("__pycache__" in rel or rel.startswith(("tests/", "tools/", "fpga/", "docs/",
                                                    "_output/", "results/", "mot_field/"))):
            continue
        if "/plot_bridge" in rel:
            continue
        try:
            tree = _ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        roots = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, _ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        if "PyQt5" in roots and "matplotlib" in roots:
            offenders.append(rel)
    listing = "".join(f"\n  {item}" for item in offenders)
    assert not offenders, "new Qt+matplotlib marriages outside the sanctioned zones:" + listing
