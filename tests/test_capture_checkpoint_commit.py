"""Checkpoint authority is absent from the current final-only capture design."""

from __future__ import annotations

import inspect

import pytest

import zlc_neutral_atom.runtime.commit as commit_module
from zlc_neutral_atom.artifacts import CaptureRepository
from zlc_neutral_atom.runtime.commit import CommitIntent, CommitTarget, FinalCommit
from zlc_neutral_atom.runtime.run import PostSafetyContext


def _target() -> CommitTarget:
    return CommitTarget(
        "capture-repository",
        "capture",
        "zlc_neutral_atom.CaptureArtifact",
        "capture/" + "a" * 64,
        "a" * 64,
    )


def test_commit_module_exports_only_final_authority() -> None:
    assert commit_module.__all__ == [
        "CommitIntent",
        "CommitTarget",
        "FinalCommit",
        "PublishVisibilityUnknown",
        "PublishedManifest",
    ]
    assert not hasattr(commit_module, "CheckpointCommit")
    assert not hasattr(commit_module, "CommitKind")
    assert "kind" not in CommitIntent.__dataclass_fields__


def test_capture_repository_has_no_checkpoint_surface(tmp_path) -> None:
    repository = CaptureRepository(tmp_path / "captures")
    try:
        assert not hasattr(repository, "checkpoint_commit")
        assert not hasattr(repository, "commit_checkpoint")
        assert not hasattr(repository, "load_checkpoint")
    finally:
        repository.close()


def test_post_safety_context_accepts_only_final_commit_authority() -> None:
    assert hasattr(PostSafetyContext, "commit_final")
    assert not hasattr(PostSafetyContext, "commit_checkpoint")
    signature = inspect.signature(PostSafetyContext.commit_final)
    assert tuple(signature.parameters) == ("self", "operation")
    assert "FinalCommit" in str(signature.parameters["operation"].annotation)


def test_final_commit_cannot_be_forged_from_a_target() -> None:
    with pytest.raises((TypeError, PermissionError)):
        FinalCommit(_target())  # type: ignore[call-arg]
    with pytest.raises((TypeError, PermissionError)):
        FinalCommit(object())  # type: ignore[call-arg]


def test_commit_intent_rejects_legacy_kind_field() -> None:
    values = {
        "commit_id": "capture-final-run-" + "a" * 64,
        "run_id": "run",
        "target": _target(),
        "created_at": 1.0,
    }
    intent = CommitIntent(**values)
    assert intent.target == _target()
    with pytest.raises(TypeError, match="unexpected keyword"):
        CommitIntent(**values, kind="CHECKPOINT")  # type: ignore[call-arg]
