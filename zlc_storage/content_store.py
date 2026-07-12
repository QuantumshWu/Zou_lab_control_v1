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


class ContentAddressedStore:
    """Filesystem store whose manifest file is the artifact visibility point."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._blobs = self.root / "blobs" / "sha256"
        self._manifests = self.root / "manifests"
        self._temporary = self.root / "tmp"
        self._lock = threading.RLock()
        for directory in (self._blobs, self._manifests, self._temporary):
            directory.mkdir(parents=True, exist_ok=True)

    def put_blob(self, payload: bytes) -> ContentRef:
        data = _payload(payload, "blob payload")
        reference = ContentRef(sha256_digest(data), len(data))
        self._publish_bytes(self._blob_path(reference.digest), data, reference)
        return reference

    def read_blob(self, reference: ContentRef) -> bytes:
        if not isinstance(reference, ContentRef):
            raise TypeError("read_blob requires ContentRef")
        return self._read_verified(self._blob_path(reference.digest), reference)

    def publish_manifest(
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
        self._publish_bytes(
            self._manifest_path(namespace, digest),
            data,
            reference,
        )
        return StoredManifest(namespace, reference)

    def read_manifest(self, namespace: str, digest: str) -> bytes:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        path = self._manifest_path(namespace, digest)
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            raise
        return self._read_verified(path, ContentRef(digest, size))

    def has_manifest(self, namespace: str, digest: str) -> bool:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        path = self._manifest_path(namespace, digest)
        if not path.is_file():
            return False
        # Existence alone is not recovery evidence.  A corrupt visible manifest
        # must fail loudly so startup reconciliation cannot call it uncommitted.
        self.read_manifest(namespace, digest)
        return True

    def _blob_path(self, digest: str) -> Path:
        digest = _sha256(digest, "blob digest")
        return self._blobs / digest[:2] / f"{digest[2:]}.blob"

    def _manifest_path(self, namespace: str, digest: str) -> Path:
        namespace = _canonical_namespace(namespace)
        digest = _sha256(digest, "manifest digest")
        return self._manifests / namespace / f"{digest}.manifest"

    def _publish_bytes(self, target: Path, data: bytes, reference: ContentRef) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if target.exists():
                existing = self._read_verified(target, reference)
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
                self._fsync_directory(target.parent)
                self._read_verified(target, reference)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _read_verified(path: Path, reference: ContentRef) -> bytes:
        data = path.read_bytes()
        if len(data) != reference.size or sha256_digest(data) != reference.digest:
            raise ContentCorruptionError(
                f"stored content does not match immutable reference {reference.digest}"
            )
        return data

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        # POSIX requires syncing the directory entry for durable rename.  Windows
        # does not provide a portable directory fsync through os.open; file data
        # is already flushed and os.replace remains the atomic visibility point.
        flags = getattr(os, "O_DIRECTORY", None)
        if flags is None:
            return
        descriptor = os.open(directory, os.O_RDONLY | flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "ContentAddressedStore",
    "ContentCorruptionError",
    "ContentRef",
    "StoredManifest",
]
