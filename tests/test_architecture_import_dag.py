"""Mechanical import-boundary ratchet for the target bounded contexts."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "zlc_data": ("Zou_lab_control", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_storage": ("Zou_lab_control", "zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_pulse": ("Zou_lab_control", "zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_workbench"),
    "zlc_frontend": ("Zou_lab_control", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_neutral_atom": ("Zou_lab_control", "zlc_frontend", "zlc_workbench"),
}

CANONICAL_HELPER_NAMES = frozenset(
    {
        "_canonical_nonempty_text",
        "_canonical_text",
        "_exact_map",
        "_exact_tree",
        "_finite_real",
        "_integer",
        "_nonempty_text",
        "_nonnegative_int",
        "_positive_finite",
        "_positive_float",
        "_positive_int",
        "_positive_seconds",
        "_positive_timeout",
        "_require_digest",
        "_sha256",
        "_text",
    }
)

# These helpers normalize human-entered labels.  They intentionally strip text
# and therefore are not the strict persisted-value invariant owned by storage.
TEXT_NORMALIZER_ALLOWLIST = frozenset(
    {
        Path("zlc_frontend/render.py"),
        Path("zlc_workbench/legacy.py"),
        Path("zlc_workbench/workspace.py"),
    }
)


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


def test_zlc_data_import_does_not_initialize_storage_backends():
    code = """
import sys
import zlc_data

forbidden = (
    'zlc_storage.content_store',
    'zlc_storage.durability',
    'zlc_storage.framed_journal',
    'zlc_storage.repository_lease',
)
loaded = tuple(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit('storage backends loaded through zlc_data: ' + ', '.join(loaded))
if 'zlc_storage.canonical' not in sys.modules:
    raise SystemExit('zlc_data did not load its only permitted storage primitive module')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_canonical_primitive_validators_have_one_owner():
    violations = []
    for package in FORBIDDEN:
        for path in (ROOT / package).rglob("*.py"):
            relative = path.relative_to(ROOT)
            if relative == Path("zlc_storage/canonical.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name not in CANONICAL_HELPER_NAMES:
                    continue
                if node.name == "_text" and relative in TEXT_NORMALIZER_ALLOWLIST:
                    continue
                violations.append(f"{relative}:{node.lineno} defines {node.name}")
    assert not violations, (
        "canonical primitive validators belong to zlc_storage.canonical; "
        "domain modules must import and alias them:\n" + "\n".join(violations)
    )


def test_artifact_finalizers_do_not_replay_published_payload_digests():
    violations = []
    for relative in (
        Path("zlc_neutral_atom/artifacts/capture.py"),
        Path("zlc_neutral_atom/readout/occupancy_pipeline.py"),
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "digest_components"
            ):
                violations.append(f"{relative}:{node.lineno}")
    assert not violations, (
        "payload content digest belongs to stream publication; artifact/finalizer "
        "code must consume EventRef provenance instead of re-reading arrays:\n"
        + "\n".join(violations)
    )


def test_calibration_numeric_versions_are_not_admission_authority():
    forbidden = (
        "CalibrationNumericBackend",
        "numeric-backend-digest",
        "numeric_backend_digest",
        "backend_digest",
        "frozen_numeric_backend",
        "_sanitized_build_identity",
        "_numeric_backend",
    )
    violations = []
    for path in (ROOT / "zlc_neutral_atom" / "readout").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)} contains {token}")
    assert not violations, (
        "numeric package versions are passive CalibrationArtifact notes; they "
        "must not regain a digest, replay, model, manifest, or admission role:\n"
        + "\n".join(violations)
    )

    analysis_path = ROOT / "zlc_neutral_atom" / "readout" / "analysis.py"
    tree = ast.parse(
        analysis_path.read_text(encoding="utf-8"),
        filename=str(analysis_path),
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_numeric_lineage_notes"
    ]
    assert len(calls) == 1, (
        "numeric version notes must be sampled once, only when constructing "
        "the top-level CalibrationArtifact"
    )


def test_current_format_names_do_not_encode_edit_counters():
    pattern = re.compile(
        r"(?:\.v|/v)\d+"
        r"|[\"']schema[\"']\s*:\s*[\"'](?:v\d+|[^\"']*[-_]v\d+)[\"']"
        r"|application/vnd\.zlc\.[^\"']*-v\d+"
        r"|ZLC-CANONICAL-\d+"
    )
    hardware_protocol_allowlist = {
        Path("zlc_pulse/target.py"): ("zlc_pulse.PulseTargetABI/v1",),
        Path("zlc_storage/canonical.py"): ("ZLC-CANONICAL-1",),
        Path("tests/test_zlc_storage_canonical.py"): ("ZLC-CANONICAL-1",),
        Path("docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md"): (
            "zlc_pulse.PulseTargetABI/v1",
            "ZLC-CANONICAL-1",
        ),
    }
    ratchet_path = Path(__file__).resolve().relative_to(ROOT)
    roots = (
        "zlc_data",
        "zlc_storage",
        "zlc_pulse",
        "zlc_neutral_atom",
        "zlc_frontend",
        "zlc_workbench",
        "tests",
        "docs",
        "pulses",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots, ".gitignore"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    violations = []
    for item in tracked.split("\0"):
        if not item:
            continue
        relative = Path(item)
        path = ROOT / relative
        if relative == ratchet_path:
            continue
        if path.name != ".gitignore" and path.suffix not in {
            ".py",
            ".md",
            ".json",
            ".toml",
        }:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            candidate = line
            for allowed in hardware_protocol_allowlist.get(relative, ()):
                candidate = candidate.replace(allowed, "")
            if pattern.search(candidate):
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    assert not violations, (
        "current-only format and framing names are plain identities; encoded "
        "edit counters are forbidden:\n"
        + "\n".join(violations)
    )
