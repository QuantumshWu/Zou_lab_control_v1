"""Crash-safe storage primitives shared by ZLC bounded contexts.

Domain artifact schemas and typed references deliberately do not live here.
Filesystem operations are exposed lazily so importing ``zlc_storage.canonical``
does not initialize the durability backend.
"""

from importlib import import_module

from .canonical import (
    CanonicalArrayEvent,
    CanonicalListEvent,
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
    "atomic_write_bytes": ("durability", "atomic_write_bytes"),
    "atomic_write_file": ("durability", "atomic_write_file"),
    "atomic_write_text": ("durability", "atomic_write_text"),
    "DirectoryDurabilityError": ("durability", "DirectoryDurabilityError"),
    "durable_makedirs": ("durability", "durable_makedirs"),
    "durable_mkdir": ("durability", "durable_mkdir"),
    "flush_directory": ("durability", "flush_directory"),
    "resolve_under": ("paths", "resolve_under"),
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
    "CanonicalArrayEvent",
    "CanonicalListEvent",
    "DirectoryDurabilityError",
    "atomic_write_bytes",
    "atomic_write_file",
    "atomic_write_text",
    "canonical_text",
    "canonical_digest",
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
    "resolve_under",
    "sha256_digest",
    "sha256_text",
]
