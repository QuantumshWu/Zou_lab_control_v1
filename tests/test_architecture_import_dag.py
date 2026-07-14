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
        "_finite_time",
        "_finite_timestamp",
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
        "_validate_timeout",
    }
)

ALLOWED_STRIP_CONTEXTS = frozenset(
    {
        (Path("zlc_pulse/document.py"), "ScanRecipeProvenance.__post_init__"),
        (Path("zlc_pulse/transport/axi.py"), "VivadoAxiRegisterTransport._parse_read"),
        (Path("zlc_neutral_atom/readout/analysis.py"), "_bounded_numeric_version"),
        (Path("zlc_workbench/camera_capture.py"), "_camera_dtype"),
        (Path("zlc_workbench/legacy_neutral_atom.py"), "_qcmos_identity_probe.probe"),
        (Path("zlc_workbench/legacy_neutral_atom.py"), "_pylon_live_identity"),
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


def test_zlc_pulse_has_no_historical_target_importer():
    forbidden = (
        "pulse_target_from_legacy",
        "Zou_lab_control.neutral_atom.PortCatalog",
        "channel_labels",
        "analog_buses",
        "clk_channels",
    )
    violations = []
    for path in (ROOT / "zlc_pulse").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)} contains {token}")
    assert not violations, (
        "zlc_pulse owns only the current PulseTarget contract; installed legacy "
        "device topology may be projected only at the composition boundary:\n"
        + "\n".join(violations)
    )


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
                violations.append(f"{relative}:{node.lineno} defines {node.name}")
    assert not violations, (
        "canonical primitive validators belong to zlc_storage.canonical; "
        "domain modules must import and alias them:\n" + "\n".join(violations)
    )


def test_text_normalization_is_confined_to_named_input_adapters():
    violations = []

    class StripVisitor(ast.NodeVisitor):
        def __init__(self, relative: Path) -> None:
            self.relative = relative
            self.context: list[str] = []

        def visit_ClassDef(self, node):
            self.context.append(node.name)
            self.generic_visit(node)
            self.context.pop()

        def visit_FunctionDef(self, node):
            self.context.append(node.name)
            self.generic_visit(node)
            self.context.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "strip":
                location = (self.relative, ".".join(self.context))
                if location not in ALLOWED_STRIP_CONTEXTS:
                    violations.append(
                        f"{self.relative}:{node.lineno} strips text in {location[1]}"
                    )
            self.generic_visit(node)

    for package in FORBIDDEN:
        for path in (ROOT / package).rglob("*.py"):
            relative = path.relative_to(ROOT)
            if relative == Path("zlc_storage/canonical.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            StripVisitor(relative).visit(tree)
    assert not violations, (
        "machine identities must reject non-canonical text; .strip() is reserved "
        "for explicitly named human/external input adapters:\n" + "\n".join(violations)
    )


def test_pulse_ir_digest_and_affine_formulas_have_one_owner():
    violations = []
    for relative in (
        Path("zlc_pulse/fpga.py"),
        Path("zlc_pulse/artifact.py"),
    ):
        tree = ast.parse(
            (ROOT / relative).read_text(encoding="utf-8"),
            filename=str(relative),
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "canonical_digest":
                continue
            argument = node.args[0]
            if (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id == "target_ir_to_tree"
            ):
                violations.append(f"{relative}:{node.lineno} recomputes TargetIR fingerprint")

    forbidden_helpers = {
        "_apply_affine",
        "_effective",
        "_effective_tick",
        "_exact_integer",
        "_exact_positive_ticks",
    }
    for source in (ROOT / "zlc_pulse").rglob("*.py"):
        relative = source.relative_to(ROOT)
        tree = ast.parse(
            source.read_text(encoding="utf-8"),
            filename=str(relative),
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in forbidden_helpers
            ):
                violations.append(f"{relative}:{node.lineno} defines duplicate {node.name}")
    assert not violations, (
        "TargetIR.fingerprint, ir.evaluate_affine_tick, document._integral_code, "
        "and document._exact_ticks are the sole pulse formula owners:\n"
        + "\n".join(violations)
    )


def test_readout_artifacts_do_not_restore_edit_counter_metadata():
    forbidden = {
        "algorithm_id",
        "algorithm_version",
        "default_model_policy",
        "gate_passed",
        "model_id",
        "model_version",
        "policy_version",
        "quality_gate_version",
        "required_model_kinds",
        "CalibrationCapability",
        "CalibrationStage",
        "DefaultModelPolicy",
        "OccupancyModelSelection",
    }
    violations = []
    for source in (ROOT / "zlc_neutral_atom" / "readout").rglob("*.py"):
        relative = source.relative_to(ROOT)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            candidate = None
            if isinstance(node, ast.Name):
                candidate = node.id
            elif isinstance(node, ast.Attribute):
                candidate = node.attr
            elif isinstance(node, ast.arg):
                candidate = node.arg
            elif isinstance(node, ast.keyword):
                candidate = node.arg
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                candidate = node.name
            if candidate in forbidden:
                violations.append(f"{relative}:{node.lineno} restores {candidate}")
    assert not violations, (
        "readout identity is CalibrationArtifactRef + ReadoutModelKind; descriptive "
        "gate ids and content fingerprints replace edit-counter metadata:\n"
        + "\n".join(violations)
    )


def test_notebook_facade_has_no_implicit_current_calibration_state():
    source = (ROOT / "Zou_lab_control/notebook/facade.py").read_text(
        encoding="utf-8"
    )
    assert "current_calibration" not in source, (
        "calibration-dependent notebook requests must receive an explicit typed ref; "
        "do not restore a session current/revision map"
    )


def test_readout_package_root_does_not_eagerly_aggregate_leaf_owners():
    path = ROOT / "zlc_neutral_atom" / "readout" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not imports, (
        "readout callers import the contracts/model/analysis/repository owner leaf; "
        "the package root must not make a light contract import initialize SciPy"
    )


def test_headless_notebook_import_does_not_load_frontend_renderer():
    code = (
        "import sys; import Zou_lab_control.notebook; "
        "assert 'zlc_frontend.render' not in sys.modules; "
        "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
        "or name == 'PyQt5' or name.startswith('PyQt5.') "
        "or name == 'PySide6' or name.startswith('PySide6.') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_content_ref_codec_and_cas_address_have_one_storage_owner():
    violations = []
    package_roots = (
        "fpga",
        "Zou_lab_control",
        "zlc_data",
        "zlc_frontend",
        "zlc_neutral_atom",
        "zlc_pulse",
        "zlc_workbench",
    )
    for package_root in package_roots:
        for source in (ROOT / package_root).rglob("*.py"):
            relative = source.relative_to(ROOT)
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    literal_keys = {
                        key.value
                        for key in node.keys
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                    }
                    if len(node.keys) == 2 and literal_keys == {"digest", "size"}:
                        violations.append(
                            f"{relative}:{node.lineno} duplicates the ContentRef tree"
                        )
                if isinstance(node, ast.Set):
                    literal_fields = {
                        item.value
                        for item in node.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    }
                    if len(node.elts) == 2 and literal_fields == {"digest", "size"}:
                        violations.append(
                            f"{relative}:{node.lineno} duplicates the ContentRef field set"
                        )
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name
                    in {
                        "content_ref_to_tree",
                        "content_ref_from_tree",
                        "_content_ref_to_tree",
                        "_content_ref_from_tree",
                    }
                ):
                    violations.append(
                        f"{relative}:{node.lineno} redefines the storage ContentRef codec"
                    )
                if not isinstance(node, ast.Call) or len(node.args) < 2:
                    continue
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                if function_name != "ContentRef":
                    continue
                digest_call, size_call = node.args[:2]
                digest_function_names = {
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else call.func.attr
                    if isinstance(call.func, ast.Attribute)
                    else ""
                    for call in ast.walk(digest_call)
                    if isinstance(call, ast.Call)
                }
                size_uses_len = any(
                    isinstance(call.func, ast.Name) and call.func.id == "len"
                    for call in ast.walk(size_call)
                    if isinstance(call, ast.Call)
                )
                if (
                    any("sha256" in name.lower() for name in digest_function_names)
                    and size_uses_len
                ):
                    violations.append(
                        f"{relative}:{node.lineno} mints a CAS address outside storage"
                    )

    for relative in (
        Path("zlc_neutral_atom/artifacts/capture.py"),
        Path("zlc_neutral_atom/readout/calibration_repository.py"),
    ):
        tree = ast.parse(
            (ROOT / relative).read_text(encoding="utf-8"),
            filename=str(relative),
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "target_ref"
            ):
                violations.append(
                    f"{relative}:{node.lineno} slices typed target_ref grammar"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "startswith"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "target_ref"
            ):
                violations.append(
                    f"{relative}:{node.lineno} reparses typed target_ref grammar"
                )

    assert not violations, (
        "ContentRef tree/address belong to zlc_storage and each typed Ref owns its "
        "target_ref grammar:\n" + "\n".join(violations)
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
