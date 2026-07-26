"""Leaf-owned configuration for the sequencer-only remote installation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_storage import canonical_text, integer, positive_real


DEFAULT_REMOTE_PORT = 18861
DEFAULT_TRANSPORT_TIMEOUT_SECONDS = 120.0
_MIN_POSITIVE_FLOAT = float.fromhex("0x0.0000000000001p-1022")


@dataclass(frozen=True, slots=True)
class RemotePulseInstallationConfig:
    host: str
    port: int = DEFAULT_REMOTE_PORT
    transport_timeout_seconds: float = DEFAULT_TRANSPORT_TIMEOUT_SECONDS

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


def remote_pulse_authoring_schema(config: object | None) -> AuthoringSchema:
    if config is not None and not isinstance(config, RemotePulseInstallationConfig):
        raise TypeError("config must be RemotePulseInstallationConfig or None")
    host = "" if config is None else config.host
    port = DEFAULT_REMOTE_PORT if config is None else config.port
    timeout = (
        DEFAULT_TRANSPORT_TIMEOUT_SECONDS
        if config is None
        else config.transport_timeout_seconds
    )
    return AuthoringSchema(
        (
            AuthoringField(
                key="host",
                kind="text",
                label="Host",
                default=host,
                required=True,
                description="Pulse execution server host name or address.",
            ),
            AuthoringField(
                key="port",
                kind="int",
                label="Port",
                default=port,
                required=True,
                minimum=1,
                maximum=65535,
                description="TCP port exposed by the pulse execution server.",
            ),
            AuthoringField(
                key="transport_timeout_seconds",
                kind="float",
                label="Transport timeout",
                default=timeout,
                required=True,
                unit="s",
                minimum=_MIN_POSITIVE_FLOAT,
                description="Maximum duration of one pulse RPC transport call.",
            ),
        )
    )


def remote_pulse_config_from_parameters(
    values: Mapping[str, object],
) -> RemotePulseInstallationConfig:
    frozen = remote_pulse_authoring_schema(None).freeze(values)
    return RemotePulseInstallationConfig(
        host=frozen["host"],
        port=frozen["port"],
        transport_timeout_seconds=frozen["transport_timeout_seconds"],
    )


def remote_pulse_config_to_parameters(config: object) -> dict[str, object]:
    if not isinstance(config, RemotePulseInstallationConfig):
        raise TypeError("config must be RemotePulseInstallationConfig")
    return {
        "host": config.host,
        "port": config.port,
        "transport_timeout_seconds": config.transport_timeout_seconds,
    }


__all__ = [
    "DEFAULT_REMOTE_PORT",
    "DEFAULT_TRANSPORT_TIMEOUT_SECONDS",
    "RemotePulseInstallationConfig",
    "remote_pulse_authoring_schema",
    "remote_pulse_config_from_parameters",
    "remote_pulse_config_to_parameters",
]
