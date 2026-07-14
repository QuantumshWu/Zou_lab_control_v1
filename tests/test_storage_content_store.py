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


def test_put_blob_borrows_a_mutable_buffer_without_calling_bytes(tmp_path):
    class NoCopyBuffer(bytearray):
        def __bytes__(self):
            raise AssertionError("large blob buffer was duplicated")

    store = ContentAddressedStore(tmp_path / "repository")
    payload = NoCopyBuffer(b"bounded-frame-chunk")
    reference = store.put_blob(payload)

    assert payload == b"bounded-frame-chunk"
    assert store.read_blob(reference) == b"bounded-frame-chunk"


def test_identify_blob_uses_the_storage_owner_without_publishing(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    payload = bytearray(b"identified-before-publication")

    expected = store.identify_blob(payload)

    assert not store._blob_path_for_digest(expected.digest).exists()
    assert store.put_blob(payload) == expected


def test_mutated_borrowed_blob_cannot_poison_a_digest_target(
    tmp_path,
    monkeypatch,
):
    store = ContentAddressedStore(tmp_path / "repository")
    payload = bytearray(b"stable-before-publication")
    expected = store.identify_blob(payload)
    original_publish = ContentAddressedStore._publish_bytes

    def raced_publish(owner, target, data, reference):
        payload[0] ^= 0x01
        return original_publish(owner, target, data, reference)

    monkeypatch.setattr(
        ContentAddressedStore,
        "_publish_bytes",
        raced_publish,
    )
    with pytest.raises(ContentCorruptionError):
        store.put_blob(payload)
    assert not store._blob_path_for_digest(expected.digest).exists()


def test_failed_post_publish_verification_restores_absence_for_retry(
    tmp_path,
    monkeypatch,
):
    store = ContentAddressedStore(tmp_path / "repository")
    payload = b"verified-or-absent"
    expected = store.identify_blob(payload)
    target = store._blob_path_for_digest(expected.digest)
    original_verify = ContentAddressedStore._verify_open_handle

    def fail_verify(*_args, **_kwargs):
        raise ContentCorruptionError("injected post-publish verification failure")

    monkeypatch.setattr(
        ContentAddressedStore,
        "_verify_open_handle",
        fail_verify,
    )
    with pytest.raises(ContentCorruptionError, match="injected"):
        store.put_blob(payload)
    assert not target.exists()

    monkeypatch.setattr(
        ContentAddressedStore,
        "_verify_open_handle",
        original_verify,
    )
    assert store.put_blob(payload) == expected
    assert store.read_blob(expected) == payload


def test_typed_blob_reference_is_trusted_after_construction(tmp_path, monkeypatch):
    import zlc_storage.content_store as content_store

    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"typed-reference")

    def unexpected_text_validation(*_args, **_kwargs):
        raise AssertionError("ContentRef digest was revalidated")

    monkeypatch.setattr(content_store, "_sha256", unexpected_text_validation)
    assert store.read_blob(reference) == b"typed-reference"


@pytest.mark.parametrize(
    "operation",
    ("read_manifest", "has_manifest", "confirm_manifest_durable"),
)
def test_manifest_identity_is_validated_once_per_raw_api_call(
    tmp_path,
    monkeypatch,
    operation,
):
    import zlc_storage.content_store as content_store

    store = ContentAddressedStore(tmp_path / "repository")
    payload = b"manifest-identity"
    digest = store.publish_manifest("capture", payload).content.digest
    namespace_calls = 0
    digest_calls = 0
    real_namespace = content_store._canonical_namespace
    real_digest = content_store._sha256

    def counted_namespace(value):
        nonlocal namespace_calls
        namespace_calls += 1
        return real_namespace(value)

    def counted_digest(value, field):
        nonlocal digest_calls
        digest_calls += 1
        return real_digest(value, field)

    monkeypatch.setattr(content_store, "_canonical_namespace", counted_namespace)
    monkeypatch.setattr(content_store, "_sha256", counted_digest)

    result = getattr(store, operation)("capture", digest)
    if operation == "has_manifest":
        assert result is True
    else:
        assert result == payload
    assert namespace_calls == 1
    assert digest_calls == 1


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


def test_verify_blob_streams_without_materializing_payload(tmp_path, monkeypatch):
    import zlc_storage.content_store as content_store

    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"streamed-verification")
    real_open = content_store.Path.open
    read_sizes = []

    class TrackedStream:
        def __init__(self, stream):
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def fileno(self):
            return self._stream.fileno()

        def seek(self, offset, whence=0):
            return self._stream.seek(offset, whence)

        def read(self, size=-1):
            read_sizes.append(size)
            return self._stream.read(size)

    def tracked_open(path, *args, **kwargs):
        return TrackedStream(real_open(path, *args, **kwargs))

    monkeypatch.setattr(content_store, "_VERIFY_CHUNK_SIZE", 4)
    monkeypatch.setattr(content_store.Path, "open", tracked_open)

    assert store.verify_blob(reference) is None
    assert read_sizes
    assert -1 not in read_sizes
    assert max(read_sizes) <= 4


def test_verify_blob_applies_physical_size_limit_before_reading_bytes(
    tmp_path,
    monkeypatch,
):
    import zlc_storage.content_store as content_store

    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"larger-than-budget")
    real_open = content_store.Path.open
    read_called = False

    class NoReadStream:
        def __init__(self, stream):
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def fileno(self):
            return self._stream.fileno()

        def read(self, _size=-1):
            nonlocal read_called
            read_called = True
            raise AssertionError("size admission must happen before reading bytes")

    def no_read_open(path, *args, **kwargs):
        return NoReadStream(real_open(path, *args, **kwargs))

    monkeypatch.setattr(content_store.Path, "open", no_read_open)

    with pytest.raises(ContentSizeLimitError, match="exceeds limit"):
        store.verify_blob(reference, max_bytes=1)
    assert read_called is False


def test_verify_blob_rejects_size_or_digest_corruption(tmp_path):
    store = ContentAddressedStore(tmp_path / "repository")
    payload = b"original"
    reference = store.put_blob(payload)

    wrong_size = ContentRef(reference.digest, reference.size + 1)
    with pytest.raises(ContentCorruptionError, match="immutable reference"):
        store.verify_blob(wrong_size)

    store._blob_path(reference.digest).write_bytes(b"tampered")
    with pytest.raises(ContentCorruptionError, match="immutable reference"):
        store.verify_blob(reference)


@pytest.mark.parametrize("max_bytes", [True, -1])
@pytest.mark.parametrize("operation", ["read_blob", "verify_blob"])
def test_read_limit_uses_canonical_nonnegative_integer_contract(
    tmp_path,
    max_bytes,
    operation,
):
    store = ContentAddressedStore(tmp_path / "repository")
    reference = store.put_blob(b"bounded")
    with pytest.raises((TypeError, ValueError)):
        getattr(store, operation)(reference, max_bytes=max_bytes)


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
    assert authority.verify_blob(reference) is None

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
