"""Leaf-owned configuration for the virtual installation product."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_storage import integer


DEFAULT_VIRTUAL_SEED = 7


@dataclass(frozen=True, slots=True)
class VirtualInstallationConfig:
    seed: int | None = DEFAULT_VIRTUAL_SEED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seed",
            integer(self.seed, "virtual seed", optional=True, nonnegative=True),
        )


def virtual_authoring_schema(config: object | None) -> AuthoringSchema:
    if config is not None and not isinstance(config, VirtualInstallationConfig):
        raise TypeError("config must be VirtualInstallationConfig or None")
    seed = DEFAULT_VIRTUAL_SEED if config is None else config.seed
    return AuthoringSchema(
        (
            AuthoringField(
                key="seed",
                kind="int",
                label="Random seed",
                default=seed,
                required=False,
                minimum=0,
                description=(
                    "Optional non-negative seed for deterministic virtual hardware; "
                    "blank selects non-deterministic initialization."
                ),
                allow_blank=True,
            ),
        )
    )


def virtual_config_from_parameters(
    values: Mapping[str, object],
) -> VirtualInstallationConfig:
    frozen = virtual_authoring_schema(None).freeze(values)
    return VirtualInstallationConfig(seed=frozen["seed"])


def virtual_config_to_parameters(config: object) -> dict[str, object]:
    if not isinstance(config, VirtualInstallationConfig):
        raise TypeError("config must be VirtualInstallationConfig")
    return {"seed": config.seed}


__all__ = [
    "DEFAULT_VIRTUAL_SEED",
    "VirtualInstallationConfig",
    "virtual_authoring_schema",
    "virtual_config_from_parameters",
    "virtual_config_to_parameters",
]
