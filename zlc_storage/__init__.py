"""Crash-safe storage primitives shared by ZLC bounded contexts.

Domain artifact schemas and typed references deliberately do not live here.
"""

from .canonical import (
    CANONICAL_MEDIA_TYPE,
    CanonicalEncodingError,
    canonical_digest,
    decode,
    encode,
    sha256_digest,
)
from .framed_journal import FramedJournal, JournalCorruptionError
from .content_store import (
    ContentAddressedStore,
    ContentCorruptionError,
    ContentRef,
    StoredManifest,
)

__all__ = [
    "CANONICAL_MEDIA_TYPE",
    "CanonicalEncodingError",
    "ContentAddressedStore",
    "ContentCorruptionError",
    "ContentRef",
    "canonical_digest",
    "FramedJournal",
    "JournalCorruptionError",
    "StoredManifest",
    "decode",
    "encode",
    "sha256_digest",
]
