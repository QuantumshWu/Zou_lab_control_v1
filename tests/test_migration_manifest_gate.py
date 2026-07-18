from __future__ import annotations

from pathlib import Path

import pytest

import conftest as migration_gate


def _write_manifest(root: Path, text: str) -> Path:
    tests = root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    manifest = tests / "migration_active_tests.txt"
    manifest.write_text(text, encoding="utf-8")
    return manifest


def test_tracked_manifest_is_canonical_unique_and_contains_gate() -> None:
    active = migration_gate.load_active_test_paths()

    assert active
    assert len(active) == len(set(active))
    assert Path(__file__).resolve() in active
    assert all(path.is_file() for path in active)
    assert all(path.parent == migration_gate.REPO_ROOT / "tests" for path in active)


@pytest.mark.parametrize(
    "manifest_text, expected",
    (
        (
            "tests/test_allowed.py\ntests/test_allowed.py\n",
            "duplicate active test path",
        ),
        ("../test_escape.py\n", "invalid active test path"),
        ("tests/helper.py\n", "invalid active test path"),
        ("tests/nested/test_allowed.py\n", "invalid active test path"),
        ("tests/test_missing.py\n", "active test does not exist"),
    ),
)
def test_manifest_rejects_invalid_authority_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_text: str,
    expected: str,
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_allowed.py").write_text(
        "def test_allowed():\n    assert True\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, manifest_text)
    monkeypatch.setattr(migration_gate, "REPO_ROOT", tmp_path)

    with pytest.raises(pytest.UsageError, match=expected):
        migration_gate.load_active_test_paths(manifest)


def test_configure_rejects_manifest_external_test_invocation() -> None:
    class Config:
        args = ["tests/test_analysis_lifecycle.py"]

    with pytest.raises(pytest.UsageError, match="unauthorized path"):
        migration_gate.pytest_configure(Config())


def test_ignore_collect_discards_unrequested_historical_modules() -> None:
    active = frozenset(migration_gate.load_active_test_paths())

    class Config:
        _zlc_active_test_paths = active

    assert (
        migration_gate.pytest_ignore_collect(
            migration_gate.REPO_ROOT / "tests" / "test_analysis_lifecycle.py",
            Config(),
        )
        is True
    )
    assert (
        migration_gate.pytest_ignore_collect(Path(__file__).resolve(), Config())
        is False
    )


def test_full_manifest_collection_rejects_a_silent_zero_item_file() -> None:
    own_path = Path(__file__).resolve()
    missing_path = own_path.with_name("test_manifest_entry_with_no_items.py")

    class Config:
        _zlc_active_test_paths = frozenset((own_path, missing_path))
        _zlc_requested_test_paths = _zlc_active_test_paths

    class Item:
        path = own_path

    with pytest.raises(pytest.UsageError, match="collected no tests"):
        migration_gate.pytest_collection_modifyitems(Config(), [Item()])
