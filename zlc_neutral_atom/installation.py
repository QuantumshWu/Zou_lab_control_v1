"""Capability-free observations of one composed installation runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zlc_storage import canonical_text, nonnegative_integer


@dataclass(frozen=True, slots=True)
class DeviceRef:
    """Opaque public identity pinned to one non-reusable runtime instance."""

    installation_id: str
    runtime_instance_id: str
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "installation_id",
            canonical_text(self.installation_id, "installation_id"),
        )
        object.__setattr__(
            self,
            "runtime_instance_id",
            canonical_text(self.runtime_instance_id, "runtime_instance_id"),
        )
        object.__setattr__(self, "role", canonical_text(self.role, "role"))

    def to_dict(self) -> dict[str, str]:
        return {
            "installation_id": self.installation_id,
            "runtime_instance_id": self.runtime_instance_id,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Immutable observation; it contains no adapter, callback, or drive verb."""

    ref: DeviceRef
    domain: str
    adapter_kind: str
    resource_key: str
    availability: str = "available"
    health: str = "healthy"

    def __post_init__(self) -> None:
        if not isinstance(self.ref, DeviceRef):
            raise TypeError("ref must be DeviceRef")
        for field in (
            "domain",
            "adapter_kind",
            "resource_key",
            "availability",
            "health",
        ):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )

    @property
    def role(self) -> str:
        return self.ref.role

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref.to_dict(),
            "domain": self.domain,
            "adapter_kind": self.adapter_kind,
            "resource_key": self.resource_key,
            "availability": self.availability,
            "health": self.health,
        }


class DeviceCatalogView(Mapping[str, DeviceInfo]):
    """Frozen role catalog for one running installation instance."""

    __slots__ = (
        "_installation_id",
        "_runtime_instance_id",
        "_revision",
        "_items",
    )

    def __init__(
        self,
        installation_id: str,
        runtime_instance_id: str,
        revision: int,
        items: tuple[DeviceInfo, ...],
    ) -> None:
        self._installation_id = canonical_text(
            installation_id,
            "installation_id",
        )
        self._runtime_instance_id = canonical_text(
            runtime_instance_id,
            "runtime_instance_id",
        )
        self._revision = nonnegative_integer(revision, "revision")
        by_role: dict[str, DeviceInfo] = {}
        for item in tuple(items):
            if not isinstance(item, DeviceInfo):
                raise TypeError("items must contain DeviceInfo values")
            if item.role in by_role:
                raise ValueError(f"duplicate device role {item.role!r}")
            if (
                item.ref.installation_id != self._installation_id
                or item.ref.runtime_instance_id != self._runtime_instance_id
            ):
                raise ValueError("DeviceInfo belongs to another runtime instance")
            by_role[item.role] = item
        self._items = MappingProxyType(dict(sorted(by_role.items())))

    def __getitem__(self, role: str) -> DeviceInfo:
        return self._items[str(role)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def installation_id(self) -> str:
        return self._installation_id

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    @property
    def revision(self) -> int:
        return self._revision

    def find(self, role: str) -> DeviceInfo | None:
        return self._items.get(str(role))

    def require(self, role: str) -> DeviceInfo:
        normalized = str(role)
        try:
            return self._items[normalized]
        except KeyError as exc:
            raise KeyError(f"device role {normalized!r} is not configured") from exc

    def roles(self, domain: str | None = None) -> tuple[str, ...]:
        if domain is None:
            return tuple(self._items)
        normalized = str(domain)
        return tuple(
            role
            for role, info in self._items.items()
            if info.domain == normalized
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "installation_id": self._installation_id,
            "runtime_instance_id": self._runtime_instance_id,
            "revision": self._revision,
            "devices": [item.to_dict() for item in self._items.values()],
        }


__all__ = [
    "DeviceCatalogView",
    "DeviceInfo",
    "DeviceRef",
]
