"""Immutable blob and manifest publication contracts."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from zlc_storage import (
    ContentAddressedStore,
    ContentCorruptionError,
    ContentRef,
    ContentSizeLimitError,
    ContentStoreAuthority,
    DirectoryDurabilityError,
    content_ref_from_tree,
    content_ref_to_tree,
    sha256_digest,
)


def test_content_ref_owner_codec_has_one_schema_free_current_shape():
    reference = ContentRef("a" * 64, 23)
    golden = {"digest": "a" * 64, "size": 23}

    assert content_ref_to_tree(reference) == golden
    assert "schema" not in golden and "version" not in golden
    assert content_ref_from_tree(golden) == reference
    assert content_ref_to_tree(content_ref_from_tree(golden)) == golden


@pytest.mark.parametrize(
    "tree",
    [
        {"digest": "a" * 64},
        {"digest": "a" * 64, "size": 1, "schema": "legacy"},
        {"digest": "not-a-digest", "size": 1},
        {"digest": "a" * 64, "size": True},
        {"digest": "a" * 64, "size": -1},
    ],
)
def test_content_ref_owner_codec_rejects_non_current_or_invalid_trees(tree):
    with pytest.raises((TypeError, ValueError)):
        content_ref_from_tree(tree)


@pytest.mark.parametrize("size", [True, -1])
def test_content_ref_uses_canonical_nonnegative_integer_contract(size):
    with pytest.raises((TypeError, ValueError)):
        ContentRef("a" * 64, size)


def test_content_store_publishes_blobs_and_manifests_idempotently(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    assert not (store.root / "tmp").exists()
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


@pytest.mark.parametrize("content_kind", ["blob", "manifest"])
def test_visible_target_is_not_acknowledged_until_retry_reconfirms_durability(
    tmp_path,
    monkeypatch,
    content_kind,
):
    import zlc_storage.content_store as content_store

    store = ContentAddressedStore(tmp_path / "repository")
    payload = f"lost-{content_kind}-directory-flush".encode()
    digest = sha256_digest(payload)
    target = (
        store._blob_path(digest)
        if content_kind == "blob"
        else store._manifest_path("capture", digest)
    )
    content_store.durability.durable_mkdir(target.parent)
    real_flush = content_store.durability.flush_directory
    failed = False

    def fail_first_target_flush(directory):
        nonlocal failed
        if directory == target.parent and not failed:
            failed = True
            raise DirectoryDurabilityError("lost target-directory acknowledgement")
        return real_flush(directory)

    monkeypatch.setattr(
        content_store.durability,
        "flush_directory",
        fail_first_target_flush,
    )
    publish = (
        (lambda: store.put_blob(payload))
        if content_kind == "blob"
        else (lambda: store.publish_manifest("capture", payload))
    )

    with pytest.raises(DirectoryDurabilityError, match="acknowledgement"):
        publish()
    assert target.is_file()

    # The exact-target retry must not equate visibility with durability: it
    # verifies/fsyncs the existing file and repeats the parent flush.
    published = publish()
    if content_kind == "blob":
        assert published.digest == digest
    else:
        assert published.content.digest == digest
    assert failed


def test_content_store_construction_fails_if_directory_flush_is_unavailable(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.content_store as content_store

    def fail_flush(_directory):
        raise DirectoryDurabilityError("directory backend unavailable")

    monkeypatch.setattr(content_store.durability, "flush_directory", fail_flush)
    with pytest.raises(DirectoryDurabilityError, match="unavailable"):
        ContentAddressedStore(tmp_path / "repository")


def test_manifest_is_the_only_visibility_point(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    store.put_blob(b"orphaned-before-commit")
    missing = "0" * 64
    assert not store.has_manifest("capture", missing)
    with pytest.raises(FileNotFoundError):
        store.read_manifest("capture", missing)
    with pytest.raises(FileNotFoundError):
        store.confirm_manifest_durable("capture", missing)
    assert not store._manifest_path("capture", missing).exists()


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


@pytest.mark.parametrize("max_bytes", [True, -1])
def test_read_limit_uses_canonical_nonnegative_integer_contract(tmp_path, max_bytes):
    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"bounded")
    with pytest.raises((TypeError, ValueError)):
        store.read_blob(reference, max_bytes=max_bytes)


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


def test_content_store_authority_is_final_process_local_and_integrity_checked(
    tmp_path,
):
    store = ContentAddressedStore(tmp_path / "repository")
    authority = store.authority()
    assert isinstance(authority, ContentStoreAuthority)
    assert authority is store.authority()
    reference = authority.put_blob(b"authority-bound")
    assert authority.read_blob(reference) == b"authority-bound"

    with pytest.raises(TypeError, match="final"):
        class _StoreSubclass(ContentAddressedStore):
            pass

    with pytest.raises(TypeError, match="final"):
        class _AuthoritySubclass(ContentStoreAuthority):
            pass

    with pytest.raises(AttributeError, match="immutable"):
        store.root = tmp_path / "redirected"
    with pytest.raises(AttributeError, match="immutable"):
        store.read_blob = lambda _reference: b"forged"
    with pytest.raises(AttributeError, match="immutable"):
        authority._root = tmp_path / "redirected"
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(authority)

    original_root = store.root
    object.__setattr__(store, "root", (tmp_path / "redirected").resolve())
    try:
        with pytest.raises(RuntimeError, match="authority changed"):
            authority.read_blob(reference)
    finally:
        object.__setattr__(store, "root", original_root)
    assert authority.read_blob(reference) == b"authority-bound"
