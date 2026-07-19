"""Crash-safe storage primitives shared by ZLC bounded contexts.

Domain artifact schemas and typed references deliberately do not live here.
Repository backends are exposed lazily so importing ``zlc_storage.canonical``
does not initialize filesystem, journal, or lease machinery.
"""

from importlib import import_module

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
    normalized_text,
    positive_integer,
    positive_real,
    sha256_digest,
    sha256_text,
)

_LAZY_EXPORTS = {
    "FramedJournal": ("framed_journal", "FramedJournal"),
    "JournalCorruptionError": ("framed_journal", "JournalCorruptionError"),
    "DirectoryDurabilityError": ("durability", "DirectoryDurabilityError"),
    "durable_makedirs": ("durability", "durable_makedirs"),
    "durable_mkdir": ("durability", "durable_mkdir"),
    "flush_directory": ("durability", "flush_directory"),
    "RepositoryRootBusy": ("repository_lease", "RepositoryRootBusy"),
    "RepositoryRootLease": ("repository_lease", "RepositoryRootLease"),
    "RepositoryRootLeaseBorrow": ("repository_lease", "RepositoryRootLeaseBorrow"),
    "ContentAddressedStore": ("content_store", "ContentAddressedStore"),
    "ContentStoreAuthority": ("content_store", "ContentStoreAuthority"),
    "ContentCorruptionError": ("content_store", "ContentCorruptionError"),
    "ContentSizeLimitError": ("content_store", "ContentSizeLimitError"),
    "ContentRef": ("content_store", "ContentRef"),
    "StoredManifest": ("content_store", "StoredManifest"),
    "content_ref_from_tree": ("content_store", "content_ref_from_tree"),
    "content_ref_to_tree": ("content_store", "content_ref_to_tree"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
    "content_ref_from_tree",
    "content_ref_to_tree",
    "canonical_text",
    "canonical_digest",
    "FramedJournal",
    "JournalCorruptionError",
    "RepositoryRootBusy",
    "RepositoryRootLease",
    "RepositoryRootLeaseBorrow",
    "StoredManifest",
    "decode",
    "durable_makedirs",
    "durable_mkdir",
    "encode",
    "exact_mapping",
    "finite_real",
    "flush_directory",
    "integer",
    "nonnegative_integer",
    "nonnegative_real",
    "normalized_text",
    "positive_integer",
    "positive_real",
    "sha256_digest",
    "sha256_text",
]
