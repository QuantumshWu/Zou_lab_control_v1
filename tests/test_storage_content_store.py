"""Immutable blob and manifest publication contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from zlc_storage import (
    ContentAddressedStore,
    ContentCorruptionError,
    ContentSizeLimitError,
    sha256_digest,
)


def test_content_store_publishes_blobs_and_manifests_idempotently(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    blob = store.put_blob(b"frame-bytes")
    assert store.put_blob(b"frame-bytes") == blob
    assert store.read_blob(blob) == b"frame-bytes"

    payload = b"canonical-owner-manifest"
    digest = sha256_digest(payload)
    published = store.publish_manifest("capture", payload, expected_digest=digest)
    assert published.content.digest == digest
    assert store.publish_manifest("capture", payload) == published
    assert store.has_manifest("capture", digest)
    assert store.read_manifest("capture", digest) == payload


def test_manifest_is_the_only_visibility_point(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    store.put_blob(b"orphaned-before-commit")
    missing = "0" * 64
    assert not store.has_manifest("capture", missing)
    with pytest.raises(FileNotFoundError):
        store.read_manifest("capture", missing)


def test_manifest_size_admission_happens_before_file_bytes_are_read(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    published = store.publish_manifest("capture", b"large-enough-manifest")
    with pytest.raises(ContentSizeLimitError, match="exceeds limit"):
        store.read_manifest("capture", published.content.digest, max_bytes=1)


def test_blob_size_admission_uses_physical_file_not_untrusted_reference_size(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"larger-than-budget")
    with pytest.raises(ContentSizeLimitError, match="exceeds limit"):
        store.read_blob(reference, max_bytes=1)


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_manifest_detects_growth_or_truncation_after_fstat(
    tmp_path,
    monkeypatch,
    size_delta,
):
    store = ContentAddressedStore(tmp_path / "repository")
    published = store.publish_manifest("capture", b"stable-manifest-bytes")

    import zlc_storage.content_store as content_store

    real_fstat = content_store.os.fstat

    def shifted_fstat(descriptor):
        result = real_fstat(descriptor)
        return SimpleNamespace(st_size=result.st_size + size_delta)

    monkeypatch.setattr(content_store.os, "fstat", shifted_fstat)
    with pytest.raises(ContentCorruptionError, match="immutable reference"):
        store.read_manifest("capture", published.content.digest)


def test_reads_reject_corrupt_immutable_content(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"original")
    path = (
        store.root
        / "blobs"
        / "sha256"
        / reference.digest[:2]
        / f"{reference.digest[2:]}.blob"
    )
    path.write_bytes(b"corrupt")
    with pytest.raises(ContentCorruptionError):
        store.read_blob(reference)


@pytest.mark.parametrize("namespace", ["../escape", "Capture", "", "two/parts"])
def test_manifest_namespace_cannot_escape_repository(tmp_path, namespace):
    store = ContentAddressedStore(tmp_path / "repository")
    with pytest.raises(ValueError):
        store.publish_manifest(namespace, b"payload")
