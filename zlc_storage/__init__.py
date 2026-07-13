"""Crash-safe storage primitives shared by ZLC bounded contexts.

Domain artifact schemas and typed references deliberately do not live here.
"""

from .canonical import (
    CANONICAL_MEDIA_TYPE,
    CanonicalArrayEvent,
    CanonicalDecodeLimits,
    CanonicalEncodingError,
    CanonicalListEvent,
    CanonicalStructureAdmission,
    CanonicalStructureEvent,
    DEFAULT_CANONICAL_DECODE_LIMITS,
    canonical_text,
    canonical_digest,
    decode,
    encode,
    exact_mapping,
    finite_real,
    integer,
    nonnegative_integer,
    nonnegative_real,
    positive_integer,
    positive_real,
    sha256_digest,
    sha256_text,
)
from .framed_journal import FramedJournal, JournalCorruptionError
from .durability import (
    DirectoryDurabilityError,
    durable_mkdir,
    flush_directory,
)
from .repository_lease import (
    RepositoryRootBusy,
    RepositoryRootLease,
    RepositoryRootLeaseBorrow,
)
from .content_store import (
    ContentAddressedStore,
    ContentStoreAuthority,
    ContentCorruptionError,
    ContentSizeLimitError,
    ContentRef,
    StoredManifest,
)

__all__ = [
    "CANONICAL_MEDIA_TYPE",
    "CanonicalEncodingError",
    "CanonicalArrayEvent",
    "CanonicalDecodeLimits",
    "CanonicalListEvent",
    "CanonicalStructureAdmission",
    "CanonicalStructureEvent",
    "DEFAULT_CANONICAL_DECODE_LIMITS",
    "DirectoryDurabilityError",
    "ContentAddressedStore",
    "ContentStoreAuthority",
    "ContentCorruptionError",
    "ContentSizeLimitError",
    "ContentRef",
    "canonical_text",
    "canonical_digest",
    "FramedJournal",
    "JournalCorruptionError",
    "RepositoryRootBusy",
    "RepositoryRootLease",
    "RepositoryRootLeaseBorrow",
    "StoredManifest",
    "decode",
    "durable_mkdir",
    "encode",
    "exact_mapping",
    "finite_real",
    "flush_directory",
    "integer",
    "nonnegative_integer",
    "nonnegative_real",
    "positive_integer",
    "positive_real",
    "sha256_digest",
    "sha256_text",
]
