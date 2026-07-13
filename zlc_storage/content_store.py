"""Content-addressed immutable blobs and atomically published manifests.

The store deliberately knows nothing about domain artifact schemas.  Owners
encode their own manifests, stage immutable blobs, and publish one manifest as
the sole visibility point.  A blob left behind by a process crash is not an
artifact; readers only follow a verified manifest.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .canonical import sha256_digest
from . import durability


_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


def _canonical_namespace(value: object) -> str:
    if not isinstance(value, str) or _NAMESPACE.fullmatch(value) is None:
        raise ValueError(
            "manifest namespace must match [a-z0-9][a-z0-9._-]*"
        )
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _payload(value: object, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} must be bytes-like")
    return bytes(value)


@dataclass(frozen=True)
class ContentRef:
    digest: str
    size: int

    def __post_init__(self) -> None:
        _sha256(self.digest, "content digest")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("content size must be a non-negative integer")


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

    def put_blob(self, payload: bytes) -> ContentRef:
        self._require_integrity()
        return ContentAddressedStore._put_blob(self._store, payload)

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
        self._authority._require_integrity()
        return self._authority

    def put_blob(self, payload: bytes) -> ContentRef:
        return self.authority().put_blob(payload)

    def _put_blob(self, payload: bytes) -> ContentRef:
        data = _payload(payload, "blob payload")
        reference = ContentRef(sha256_digest(data), len(data))
        ContentAddressedStore._publish_bytes(
            self,
            ContentAddressedStore._blob_path(self, reference.digest),
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
            ContentAddressedStore._blob_path(self, reference.digest),
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
        namespace = _canonical_namespace(namespace)
        data = _payload(payload, "manifest payload")
        digest = sha256_digest(data)
        if expected_digest is not None and digest != _sha256(
            expected_digest, "expected manifest digest"
        ):
            raise ValueError("manifest payload differs from expected digest")
        reference = ContentRef(digest, len(data))
        ContentAddressedStore._publish_bytes(
            self,
            ContentAddressedStore._manifest_path(self, namespace, digest),
            data,
            reference,
        )
        return StoredManifest(namespace, reference)

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
        path = ContentAddressedStore._manifest_path(self, namespace, digest)
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
        path = ContentAddressedStore._manifest_path(self, namespace, digest)
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
        path = ContentAddressedStore._manifest_path(self, namespace, digest)
        if not path.is_file():
            return False
        # Existence alone is not recovery evidence.  A corrupt visible manifest
        # must fail loudly so startup reconciliation cannot call it uncommitted.
        ContentAddressedStore._read_manifest(
            self,
            namespace,
            digest,
            max_bytes=max_bytes,
        )
        return True

    def _blob_path(self, digest: str) -> Path:
        digest = _sha256(digest, "blob digest")
        return self._blobs / digest[:2] / f"{digest[2:]}.blob"

    def _manifest_path(self, namespace: str, digest: str) -> Path:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        return self._manifests / namespace / f"{digest}.manifest"

    def _publish_bytes(self, target: Path, data: bytes, reference: ContentRef) -> None:
        with self._lock:
            durability.durable_mkdir(target.parent)
            if target.exists():
                # Visibility is not durability.  This path is also the retry
                # barrier after a prior replace became visible but its parent
                # directory flush acknowledgement failed.  Verify and fsync
                # the exact same open file, then persist its directory entry.
                existing = ContentAddressedStore._confirm_existing_durable(
                    self,
                    target,
                    expected_digest=reference.digest,
                    expected_size=reference.size,
                    max_bytes=None,
                )
                if existing != data:
                    raise ContentCorruptionError(
                        f"content collision at immutable path {target}"
                    )
                return
            temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                # The destination name is derived from the payload digest.  A
                # concurrent writer can only publish the same verified bytes.
                os.replace(temporary, target)
                durability.flush_directory(target.parent)
                ContentAddressedStore._read_verified(target, reference)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

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
        if max_bytes is not None and (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise ValueError("max_bytes must be a non-negative integer or None")
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

__all__ = [
    "ContentAddressedStore",
    "ContentStoreAuthority",
    "ContentCorruptionError",
    "ContentSizeLimitError",
    "ContentRef",
    "StoredManifest",
]
