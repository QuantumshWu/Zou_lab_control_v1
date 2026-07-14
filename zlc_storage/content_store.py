"""Content-addressed immutable blobs and atomically published manifests.

The store deliberately knows nothing about domain artifact schemas.  Owners
encode their own manifests, stage immutable blobs, and publish one manifest as
the sole visibility point.  A blob left behind by a process crash is not an
artifact; readers only follow a verified manifest.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .canonical import (
    exact_mapping,
    nonnegative_integer,
    sha256_digest,
    sha256_text as _sha256,
)
from . import durability


_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_VERIFY_CHUNK_SIZE = 1024 * 1024


def _canonical_namespace(value: object) -> str:
    if not isinstance(value, str) or _NAMESPACE.fullmatch(value) is None:
        raise ValueError(
            "manifest namespace must match [a-z0-9][a-z0-9._-]*"
        )
    return value

def _payload(value: object, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} must be bytes-like")
    return bytes(value)


def _blob_payload(value: object, field: str) -> memoryview:
    """Borrow one synchronous blob buffer without duplicating large payloads."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} must be bytes-like")
    view = memoryview(value)
    if not view.c_contiguous:
        raise ValueError(f"{field} must be C-contiguous")
    return view.cast("B")


@dataclass(frozen=True)
class ContentRef:
    digest: str
    size: int

    def __post_init__(self) -> None:
        _sha256(self.digest, "content digest")
        object.__setattr__(
            self,
            "size",
            nonnegative_integer(self.size, "content size"),
        )


def content_ref_to_tree(value: ContentRef) -> dict[str, object]:
    """Project the storage owner's immutable blob identity to primitives."""

    if not isinstance(value, ContentRef):
        raise TypeError("value must be ContentRef")
    return {"digest": value.digest, "size": value.size}


def content_ref_from_tree(tree: object) -> ContentRef:
    """Decode the one current, schema-free ContentRef primitive shape."""

    data = exact_mapping(
        tree,
        {"digest", "size"},
        "ContentRef",
        discriminator=None,
    )
    return ContentRef(data["digest"], data["size"])


@dataclass(frozen=True)
class StoredManifest:
    namespace: str
    content: ContentRef

    def __post_init__(self) -> None:
        _canonical_namespace(self.namespace)
        if not isinstance(self.content, ContentRef):
            raise TypeError("manifest content must be ContentRef")


class ContentCorruptionError(RuntimeError):
    """Stored bytes do not match the immutable reference naming them."""


class ContentSizeLimitError(RuntimeError):
    """Stored content exceeds a caller's pre-read byte admission limit."""


_CONTENT_STORE_AUTHORITY_TOKEN = object()


class ContentStoreAuthority:
    """Process-local proof of one immutable content-store filesystem binding.

    Domain repositories keep this capability instead of copying the storage
    owner's private path/lock layout.  Every operation rechecks the exact store
    instance and its resolved directories before touching the filesystem.
    """

    __slots__ = (
        "_store",
        "_root",
        "_blobs",
        "_manifests",
        "_lock",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("ContentStoreAuthority is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        store: "ContentAddressedStore",
        root: Path,
        blobs: Path,
        manifests: Path,
        lock: threading.RLock,
    ) -> None:
        if token is not _CONTENT_STORE_AUTHORITY_TOKEN:
            raise PermissionError(
                "ContentStoreAuthority can only be minted by ContentAddressedStore"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_blobs", blobs)
        object.__setattr__(self, "_manifests", manifests)
        object.__setattr__(self, "_lock", lock)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ContentStoreAuthority is immutable")

    def __reduce__(self):
        raise TypeError("ContentStoreAuthority is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("ContentStoreAuthority is process-local and cannot be serialized")

    @property
    def root(self) -> Path:
        self._require_integrity()
        return self._root

    def put_blob(self, payload: bytes | bytearray | memoryview) -> ContentRef:
        self._require_integrity()
        return ContentAddressedStore._put_blob(self._store, payload)

    def identify_blob(self, payload: bytes | bytearray | memoryview) -> ContentRef:
        """Return the canonical CAS identity without publishing the payload."""

        self._require_integrity()
        return ContentAddressedStore._identify_blob(payload)

    def read_blob(
        self,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        self._require_integrity()
        return ContentAddressedStore._read_blob(
            self._store,
            reference,
            max_bytes=max_bytes,
        )

    def verify_blob(
        self,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> None:
        """Verify an existing blob without materializing its payload."""

        self._require_integrity()
        ContentAddressedStore._verify_blob(
            self._store,
            reference,
            max_bytes=max_bytes,
        )

    def publish_manifest(
        self,
        namespace: str,
        payload: bytes,
        *,
        expected_digest: str | None = None,
    ) -> StoredManifest:
        self._require_integrity()
        return ContentAddressedStore._publish_manifest(
            self._store,
            namespace,
            payload,
            expected_digest=expected_digest,
        )

    def read_manifest(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        self._require_integrity()
        return ContentAddressedStore._read_manifest(
            self._store,
            namespace,
            digest,
            max_bytes=max_bytes,
        )

    def confirm_manifest_durable(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        """Verify/fsync an existing manifest and its parent; never create it."""

        self._require_integrity()
        return ContentAddressedStore._confirm_manifest_durable(
            self._store,
            namespace,
            digest,
            max_bytes=max_bytes,
        )

    def has_manifest(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bool:
        self._require_integrity()
        return ContentAddressedStore._has_manifest(
            self._store,
            namespace,
            digest,
            max_bytes=max_bytes,
        )

    def _require_integrity(self) -> None:
        store = self._store
        if type(store) is not ContentAddressedStore:
            raise RuntimeError("content-store authority has the wrong implementation")
        if (
            store._authority is not self
            or store.root != self._root
            or store._blobs != self._blobs
            or store._manifests != self._manifests
            or store._lock is not self._lock
        ):
            raise RuntimeError("content-store filesystem authority changed")


class ContentAddressedStore:
    """Filesystem store whose manifest file is the artifact visibility point."""

    __slots__ = (
        "root",
        "_blobs",
        "_manifests",
        "_lock",
        "_authority",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("ContentAddressedStore is final and cannot be subclassed")

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        object.__setattr__(self, "_blobs", self.root / "blobs" / "sha256")
        object.__setattr__(self, "_manifests", self.root / "manifests")
        object.__setattr__(self, "_lock", threading.RLock())
        durability.durable_mkdir(self.root)
        # Construction is a durability preflight even when the hierarchy was
        # created by an earlier process.
        durability.flush_directory(self.root)
        for directory in (self._blobs, self._manifests):
            durability.durable_mkdir(directory)
        object.__setattr__(
            self,
            "_authority",
            ContentStoreAuthority(
                _CONTENT_STORE_AUTHORITY_TOKEN,
                store=self,
                root=self.root,
                blobs=self._blobs,
                manifests=self._manifests,
                lock=self._lock,
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ContentAddressedStore authority is immutable")
        object.__setattr__(self, _name, _value)

    def authority(self) -> ContentStoreAuthority:
        return self._authority

    def put_blob(self, payload: bytes | bytearray | memoryview) -> ContentRef:
        return self.authority().put_blob(payload)

    def identify_blob(self, payload: bytes | bytearray | memoryview) -> ContentRef:
        return self.authority().identify_blob(payload)

    @staticmethod
    def _identify_blob(payload: bytes | bytearray | memoryview) -> ContentRef:
        data = _blob_payload(payload, "blob payload")
        return ContentRef(sha256_digest(data), len(data))

    def _put_blob(self, payload: bytes | bytearray | memoryview) -> ContentRef:
        data = _blob_payload(payload, "blob payload")
        reference = ContentAddressedStore._identify_blob(data)
        ContentAddressedStore._publish_bytes(
            self,
            self._blob_path_for_digest(reference.digest),
            data,
            reference,
        )
        return reference

    def read_blob(
        self,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        return self.authority().read_blob(reference, max_bytes=max_bytes)

    def _read_blob(
        self,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        if not isinstance(reference, ContentRef):
            raise TypeError("read_blob requires ContentRef")
        return ContentAddressedStore._read_verified(
            self._blob_path_for_digest(reference.digest),
            reference,
            max_bytes=max_bytes,
        )

    def verify_blob(
        self,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> None:
        """Verify an existing blob without materializing its payload."""

        self.authority().verify_blob(reference, max_bytes=max_bytes)

    def _verify_blob(
        self,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> None:
        if not isinstance(reference, ContentRef):
            raise TypeError("verify_blob requires ContentRef")
        ContentAddressedStore._verify_open_handle(
            self._blob_path_for_digest(reference.digest),
            reference,
            max_bytes=max_bytes,
        )

    def publish_manifest(
        self,
        namespace: str,
        payload: bytes,
        *,
        expected_digest: str | None = None,
    ) -> StoredManifest:
        return self.authority().publish_manifest(
            namespace,
            payload,
            expected_digest=expected_digest,
        )

    def _publish_manifest(
        self,
        namespace: str,
        payload: bytes,
        *,
        expected_digest: str | None = None,
    ) -> StoredManifest:
        data = _payload(payload, "manifest payload")
        digest = sha256_digest(data)
        if expected_digest is not None and digest != _sha256(
            expected_digest, "expected manifest digest"
        ):
            raise ValueError("manifest payload differs from expected digest")
        reference = ContentRef(digest, len(data))
        manifest = StoredManifest(namespace, reference)
        ContentAddressedStore._publish_bytes(
            self,
            self._manifest_path_for_identity(manifest.namespace, digest),
            data,
            reference,
        )
        return manifest

    def read_manifest(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        return self.authority().read_manifest(
            namespace,
            digest,
            max_bytes=max_bytes,
        )

    def confirm_manifest_durable(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        return self.authority().confirm_manifest_durable(
            namespace,
            digest,
            max_bytes=max_bytes,
        )

    def _read_manifest(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        path = self._manifest_path_for_identity(namespace, digest)
        return ContentAddressedStore._read_verified_digest(
            path,
            digest,
            max_bytes=max_bytes,
        )

    def _confirm_manifest_durable(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        path = self._manifest_path_for_identity(namespace, digest)
        return ContentAddressedStore._confirm_existing_durable(
            self,
            path,
            expected_digest=digest,
            expected_size=None,
            max_bytes=max_bytes,
        )

    def has_manifest(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bool:
        return self.authority().has_manifest(
            namespace,
            digest,
            max_bytes=max_bytes,
        )

    def _has_manifest(
        self,
        namespace: str,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bool:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        path = self._manifest_path_for_identity(namespace, digest)
        if not path.is_file():
            return False
        # Existence alone is not recovery evidence.  A corrupt visible manifest
        # must fail loudly so startup reconciliation cannot call it uncommitted.
        ContentAddressedStore._read_verified_digest(
            path,
            digest,
            max_bytes=max_bytes,
        )
        return True

    def _blob_path(self, digest: str) -> Path:
        digest = _sha256(digest, "blob digest")
        return self._blob_path_for_digest(digest)

    def _blob_path_for_digest(self, digest: str) -> Path:
        return self._blobs / digest[:2] / f"{digest[2:]}.blob"

    def _manifest_path(self, namespace: str, digest: str) -> Path:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        return self._manifest_path_for_identity(namespace, digest)

    def _manifest_path_for_identity(self, namespace: str, digest: str) -> Path:
        return self._manifests / namespace / f"{digest}.manifest"

    def _publish_bytes(
        self,
        target: Path,
        data: bytes | bytearray | memoryview,
        reference: ContentRef,
    ) -> None:
        with self._lock:
            durability.durable_mkdir(target.parent)
            if target.exists():
                # Visibility is not durability.  This path is also the retry
                # barrier after a prior replace became visible but its parent
                # directory flush acknowledgement failed.  Verify and fsync
                # the exact same open file, then persist its directory entry.
                ContentAddressedStore._confirm_existing_reference_durable(
                    self,
                    target,
                    reference,
                )
                return
            temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("x+b") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                    # Mutable borrowed buffers may change after their identity
                    # was computed.  Verify the staged file before replace so
                    # a race can fail this call but can never make mismatched
                    # bytes visible under a digest-derived target name.
                    ContentAddressedStore._verify_stream(
                        stream,
                        reference,
                        max_bytes=None,
                    )
                # The destination name is derived from the payload digest.  A
                # concurrent writer can only publish the same verified bytes.
                os.replace(temporary, target)
                durability.flush_directory(target.parent)
                try:
                    ContentAddressedStore._verify_open_handle(
                        target,
                        reference,
                        max_bytes=None,
                    )
                except BaseException as verification_error:
                    # This call created the target while holding the store
                    # lock.  If its post-publish verification fails, remove
                    # that target and durably restore absence so a correct
                    # retry is never blocked by poisoned digest storage.
                    try:
                        target.unlink()
                        durability.flush_directory(target.parent)
                    except BaseException as cleanup_error:
                        verification_error.add_note(
                            "failed to durably remove the unverified CAS target: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    raise
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _confirm_existing_reference_durable(
        self,
        target: Path,
        reference: ContentRef,
    ) -> None:
        """Verify/fsync an existing immutable target without reading it into RAM."""

        with self._lock:
            with target.open("r+b") as stream:
                ContentAddressedStore._verify_stream(
                    stream,
                    reference,
                    max_bytes=None,
                )
                stream.flush()
                os.fsync(stream.fileno())
            durability.flush_directory(target.parent)

    def _confirm_existing_durable(
        self,
        target: Path,
        *,
        expected_digest: str,
        expected_size: int | None,
        max_bytes: int | None,
    ) -> bytes:
        """Durability barrier for an existing immutable target; never publish."""

        with self._lock:
            with target.open("r+b") as stream:
                data = ContentAddressedStore._read_verified_stream(
                    stream,
                    expected_digest=expected_digest,
                    expected_size=expected_size,
                    max_bytes=max_bytes,
                )
                stream.flush()
                os.fsync(stream.fileno())
            durability.flush_directory(target.parent)
            return data

    @classmethod
    def _read_verified(
        cls,
        path: Path,
        reference: ContentRef,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        return cls._read_open_handle(
            path,
            expected_digest=reference.digest,
            expected_size=reference.size,
            max_bytes=max_bytes,
        )

    @classmethod
    def _read_verified_digest(
        cls,
        path: Path,
        digest: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        return cls._read_open_handle(
            path,
            expected_digest=digest,
            expected_size=None,
            max_bytes=max_bytes,
        )

    @staticmethod
    def _read_open_handle(
        path: Path,
        *,
        expected_digest: str,
        expected_size: int | None,
        max_bytes: int | None,
    ) -> bytes:
        with path.open("rb") as stream:
            return ContentAddressedStore._read_verified_stream(
                stream,
                expected_digest=expected_digest,
                expected_size=expected_size,
                max_bytes=max_bytes,
            )

    @staticmethod
    def _read_verified_stream(
        stream,
        *,
        expected_digest: str,
        expected_size: int | None,
        max_bytes: int | None,
    ) -> bytes:
        if max_bytes is not None:
            max_bytes = nonnegative_integer(max_bytes, "max_bytes")
        stream.seek(0)
        actual_size = os.fstat(stream.fileno()).st_size
        if max_bytes is not None and actual_size > max_bytes:
            raise ContentSizeLimitError(
                f"stored content size {actual_size} exceeds limit {max_bytes}"
            )
        if expected_size is not None and actual_size != expected_size:
            raise ContentCorruptionError(
                f"stored content does not match immutable reference {expected_digest}"
            )
        # Read through the same handle that was fstat'ed.  The extra byte
        # detects in-place growth after fstat without permitting an
        # unbounded read from a concurrently replaced/corrupt file.
        data = stream.read(actual_size + 1)
        if len(data) != actual_size or sha256_digest(data) != expected_digest:
            raise ContentCorruptionError(
                f"stored content does not match immutable reference {expected_digest}"
            )
        return data

    @staticmethod
    def _verify_open_handle(
        path: Path,
        reference: ContentRef,
        *,
        max_bytes: int | None,
    ) -> None:
        if max_bytes is not None:
            max_bytes = nonnegative_integer(max_bytes, "max_bytes")
        with path.open("rb") as stream:
            ContentAddressedStore._verify_stream(
                stream,
                reference,
                max_bytes=max_bytes,
            )

    @staticmethod
    def _verify_stream(stream, reference: ContentRef, *, max_bytes: int | None) -> None:
        if max_bytes is not None:
            max_bytes = nonnegative_integer(max_bytes, "max_bytes")
        actual_size = os.fstat(stream.fileno()).st_size
        if max_bytes is not None and actual_size > max_bytes:
            raise ContentSizeLimitError(
                f"stored content size {actual_size} exceeds limit {max_bytes}"
            )
        if actual_size != reference.size:
            raise ContentCorruptionError(
                "stored content does not match immutable reference "
                f"{reference.digest}"
            )

        stream.seek(0)
        digest = hashlib.sha256()
        remaining = actual_size
        while remaining:
            chunk = stream.read(min(_VERIFY_CHUNK_SIZE, remaining))
            if not chunk:
                raise ContentCorruptionError(
                    "stored content does not match immutable reference "
                    f"{reference.digest}"
                )
            digest.update(chunk)
            remaining -= len(chunk)

        # The extra byte detects growth after fstat while keeping the read
        # bounded to the admitted physical size plus one byte.
        if stream.read(1) or digest.hexdigest() != reference.digest:
            raise ContentCorruptionError(
                "stored content does not match immutable reference "
                f"{reference.digest}"
            )

__all__ = [
    "ContentAddressedStore",
    "ContentStoreAuthority",
    "ContentCorruptionError",
    "ContentSizeLimitError",
    "ContentRef",
    "StoredManifest",
    "content_ref_from_tree",
    "content_ref_to_tree",
]
