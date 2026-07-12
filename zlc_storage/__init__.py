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

__all__ = [
    "CANONICAL_MEDIA_TYPE",
    "CanonicalEncodingError",
    "canonical_digest",
    "FramedJournal",
    "JournalCorruptionError",
    "decode",
    "encode",
    "sha256_digest",
]
