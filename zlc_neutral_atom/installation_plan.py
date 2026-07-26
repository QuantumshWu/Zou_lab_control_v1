"""Backend-neutral public topology value used by installation leaves."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class InstallationDevicePlan:
    role: str
    domain: str
    adapter_kind: str
    summary: str
    configuration_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("role", "domain", "adapter_kind", "summary"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )
        keys = tuple(
            canonical_text(key, "installation configuration key")
            for key in self.configuration_keys
        )
        if len(keys) != len(set(keys)):
            raise ValueError("installation configuration keys must be unique")
        object.__setattr__(self, "configuration_keys", keys)


def installation_device_plan(backend: str) -> tuple[InstallationDevicePlan, ...]:
    """Project the exact public topology from its owning backend leaf."""

    from zlc_neutral_atom.installation_package import installation_package

    return installation_package(backend).device_plan


__all__ = ["InstallationDevicePlan", "installation_device_plan"]
