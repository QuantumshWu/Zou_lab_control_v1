"""Current-only installation configuration documents.

This module owns the small, closed configuration vocabulary that the current
composition root can actually execute.  Documents describe how to establish a
new installation; they never contain live devices, adapter class names, runtime
references, or in-process replacement instructions.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from zlc_storage import (
    canonical_text,
    flush_directory,
    integer,
    positive_real,
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
class VirtualInstallationConfig:
    """The deterministic in-process installation currently used for simulation."""

    seed: int | None = 7

    def __post_init__(self) -> None:
        seed = integer(
            self.seed,
            "virtual seed",
            optional=True,
            nonnegative=True,
        )
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class RemotePulseInstallationConfig:
    """The current sequencer-only installation served by the pulse RPC server."""

    host: str
    port: int = 18861
    transport_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", canonical_text(self.host, "remote host"))
        port = integer(self.port, "remote port", minimum=1)
        assert port is not None
        if port > 65535:
            raise ValueError("remote port must be at most 65535")
        object.__setattr__(self, "port", port)
        object.__setattr__(
            self,
            "transport_timeout_seconds",
            positive_real(
                self.transport_timeout_seconds,
                "transport_timeout_seconds",
            ),
        )


InstallationConfig: TypeAlias = (
    VirtualInstallationConfig | RemotePulseInstallationConfig
)


@dataclass(frozen=True, slots=True)
class InstallationConfigDocument:
    """One immutable, canonicalizable current installation request."""

    config: InstallationConfig

    def __post_init__(self) -> None:
        if not isinstance(
            self.config,
            (VirtualInstallationConfig, RemotePulseInstallationConfig),
        ):
            raise TypeError("config must be a current installation config")

    @classmethod
    def virtual(cls, *, seed: int | None = 7) -> "InstallationConfigDocument":
        return cls(VirtualInstallationConfig(seed))

    @classmethod
    def remote_pulse(
        cls,
        *,
        host: str,
        port: int = 18861,
        transport_timeout_seconds: float = 120.0,
    ) -> "InstallationConfigDocument":
        return cls(
            RemotePulseInstallationConfig(
                host,
                port,
                transport_timeout_seconds,
            )
        )

    @property
    def backend(self) -> str:
        if isinstance(self.config, VirtualInstallationConfig):
            return "virtual"
        return "remote_pulse"

    @property
    def content_digest(self) -> str:
        return sha256_digest(self.to_bytes())

    def to_dict(self) -> dict[str, object]:
        config = self.config
        if isinstance(config, VirtualInstallationConfig):
            parameters: dict[str, object] = {"seed": config.seed}
        else:
            parameters = {
                "host": config.host,
                "port": config.port,
                "transport_timeout_seconds": config.transport_timeout_seconds,
            }
        return {
            "format": INSTALLATION_CONFIG_FORMAT,
            "backend": self.backend,
            "parameters": parameters,
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
        if backend == "virtual":
            _require_exact_fields(parameters, {"seed"}, "virtual parameters")
            return cls.virtual(seed=parameters["seed"])
        if backend == "remote_pulse":
            _require_exact_fields(
                parameters,
                {"host", "port", "transport_timeout_seconds"},
                "remote_pulse parameters",
            )
            return cls.remote_pulse(
                host=parameters["host"],
                port=parameters["port"],
                transport_timeout_seconds=parameters[
                    "transport_timeout_seconds"
                ],
            )
        raise ValueError(f"unsupported installation backend {backend!r}")

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


def _require_exact_fields(
    value: dict[str, object],
    fields: set[str],
    name: str,
) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")


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
    "InstallationConfig",
    "InstallationConfigConflict",
    "InstallationConfigDocument",
    "RemotePulseInstallationConfig",
    "VirtualInstallationConfig",
    "load_installation_config",
    "save_installation_config",
]
