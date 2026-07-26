"""Backend-neutral canonical storage for installation configuration documents.

Concrete values, field semantics, codecs, topology, and composition live in
their fixed ``devices/<backend>/package.py`` leaf.  This module only envelopes
one leaf-owned value and provides atomic persistence.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_storage import (
    flush_directory,
    sha256_digest,
    sha256_text,
)
from zlc_storage.file_lock import (
    acquire_file_lock,
    open_durable_lock_file,
    release_file_lock,
)


INSTALLATION_CONFIG_FORMAT = "zlc_neutral_atom.InstallationConfig"


@dataclass(frozen=True, slots=True)
class InstallationConfigDocument:
    """One immutable, canonicalizable current installation request."""

    config: object

    def __post_init__(self) -> None:
        from zlc_neutral_atom.installation_package import (
            installation_package_for_config,
        )

        installation_package_for_config(self.config).require_config(self.config)

    @classmethod
    def from_parameters(
        cls,
        backend: str,
        values: Mapping[str, object],
    ) -> "InstallationConfigDocument":
        """Build one leaf-validated document from an ordinary editor draft."""

        from zlc_neutral_atom.installation_package import installation_package

        package = installation_package(backend)
        frozen = package.authoring_schema(None).freeze(values)
        return cls(package.config_from_parameters(frozen))

    @property
    def backend(self) -> str:
        from zlc_neutral_atom.installation_package import (
            installation_package_for_config,
        )

        return installation_package_for_config(self.config).backend

    @property
    def parameters(self) -> dict[str, object]:
        from zlc_neutral_atom.installation_package import (
            installation_package_for_config,
        )

        return installation_package_for_config(self.config).parameters(self.config)

    @property
    def content_digest(self) -> str:
        return sha256_digest(self.to_bytes())

    def to_dict(self) -> dict[str, object]:
        return {
            "format": INSTALLATION_CONFIG_FORMAT,
            "backend": self.backend,
            "parameters": self.parameters,
        }

    def to_bytes(self) -> bytes:
        """Return deterministic UTF-8 JSON used for file content and identity."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: object) -> "InstallationConfigDocument":
        if not isinstance(value, dict) or set(value) != {
            "format",
            "backend",
            "parameters",
        }:
            raise ValueError(
                "installation config must contain exactly "
                "['backend', 'format', 'parameters']"
            )
        if value["format"] != INSTALLATION_CONFIG_FORMAT:
            raise ValueError(
                f"unsupported installation config format {value['format']!r}"
            )
        parameters = value["parameters"]
        if not isinstance(parameters, dict):
            raise TypeError("installation config parameters must be a mapping")
        backend = value["backend"]
        return cls.from_parameters(backend, parameters)

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview) -> "InstallationConfigDocument":
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("installation config payload must be bytes-like")
        try:
            text = bytes(payload).decode("utf-8")
            value = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("installation config is not valid UTF-8 JSON") from exc
        return cls.from_dict(value)


class InstallationConfigConflict(RuntimeError):
    """The file changed since the caller loaded its editing baseline."""

    def __init__(self, expected_digest: str, actual_digest: str | None) -> None:
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest
        actual = "missing" if actual_digest is None else actual_digest
        super().__init__(
            "installation config changed since it was loaded: "
            f"expected {expected_digest}, found {actual}"
        )


def load_installation_config(
    path: str | os.PathLike[str],
) -> InstallationConfigDocument:
    target = _config_path(path)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read installation config {target}: {exc}") from exc
    try:
        return InstallationConfigDocument.from_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid installation config {target}: {exc}") from exc


def save_installation_config(
    path: str | os.PathLike[str],
    document: InstallationConfigDocument,
    *,
    expected_digest: str | None = None,
) -> str:
    """Atomically save one document, optionally comparing its editing baseline.

    The adjacent permanent lock file linearizes the compare and replace across
    processes.  ``expected_digest`` names the canonical document previously
    loaded by the editor; a semantically unchanged reformat therefore does not
    create a false conflict.
    """

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    expected = (
        None
        if expected_digest is None
        else sha256_text(expected_digest, "expected_digest")
    )
    target = _config_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"installation config parent does not exist: {target.parent}"
        )
    lock_path = target.with_name(f".{target.name}.lock")
    lock_stream = open_durable_lock_file(lock_path)
    temporary: Path | None = None
    acquired = False
    try:
        acquire_file_lock(lock_stream, blocking=True)
        acquired = True
        if expected is not None:
            if not target.exists():
                raise InstallationConfigConflict(expected, None)
            current = load_installation_config(target)
            if current.content_digest != expected:
                raise InstallationConfigConflict(
                    expected,
                    current.content_digest,
                )
        payload = document.to_bytes()
        temporary = target.with_name(
            f".{target.name}.{uuid.uuid4().hex}.tmp"
        )
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        flush_directory(target.parent)
        return sha256_digest(payload)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if acquired:
            release_file_lock(lock_stream)
        lock_stream.close()


def default_installation_authoring_schema(backend: str) -> AuthoringSchema:
    """Return the backend owner's declared defaults and field semantics."""

    from zlc_neutral_atom.installation_package import installation_package

    return installation_package(backend).authoring_schema(None)


def installation_authoring_schema(document: InstallationConfigDocument):
    """Return the leaf schema populated with this document's exact values."""

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    from zlc_neutral_atom.installation_package import (
        installation_package_for_config,
    )

    return installation_package_for_config(document.config).authoring_schema(
        document.config
    )


def supported_installation_backends() -> tuple[str, ...]:
    """Project the deterministic backend names declared by built-in leaves."""

    from zlc_neutral_atom.installation_package import discover_installation_packages

    return tuple(package.backend for package in discover_installation_packages())


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _config_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("installation config path must be path-like")
    return Path(path).expanduser().resolve()


__all__ = [
    "INSTALLATION_CONFIG_FORMAT",
    "InstallationConfigConflict",
    "InstallationConfigDocument",
    "default_installation_authoring_schema",
    "installation_authoring_schema",
    "load_installation_config",
    "save_installation_config",
    "supported_installation_backends",
]
