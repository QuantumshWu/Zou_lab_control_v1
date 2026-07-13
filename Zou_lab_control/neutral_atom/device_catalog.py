"""Immutable public observations of one installation generation.

The installation runtime owns adapters, SDK handles, connections, and drive verbs.  A
``DeviceCatalogView`` is deliberately only canonical data: notebooks and GUIs can list
roles and display health without gaining a path back to the hardware object graph.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from zlc_storage import canonical_text as _text
from zlc_storage import positive_integer as _positive_int

class InstallationAvailability(str, Enum):
    AVAILABLE = "available"
    SWAPPING = "swapping"
    RECOVERY_REQUIRED = "recovery_required"
    UNAVAILABLE = "unavailable"


class DeviceHealth(str, Enum):
    HEALTHY = "healthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DeviceRef:
    """Generation-bound public identity; never an executable capability."""

    installation_id: str
    installation_generation: int
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "installation_id", _text(self.installation_id, "installation id")
        )
        object.__setattr__(
            self,
            "installation_generation",
            _positive_int(self.installation_generation, "installation generation"),
        )
        object.__setattr__(self, "role", _text(self.role, "device role"))

    def to_dict(self) -> dict[str, object]:
        return {
            "installation_id": self.installation_id,
            "installation_generation": self.installation_generation,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Immutable identity and observation metadata for one configured role."""

    ref: DeviceRef
    domain: str
    adapter_kind: str
    resource_key: str
    availability: InstallationAvailability = InstallationAvailability.AVAILABLE
    health: DeviceHealth = DeviceHealth.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.ref, DeviceRef):
            raise TypeError("device info ref must be DeviceRef")
        object.__setattr__(self, "domain", _text(self.domain, "device domain"))
        object.__setattr__(
            self, "adapter_kind", _text(self.adapter_kind, "device adapter kind")
        )
        key = _text(self.resource_key, "device resource key")
        if key != f"device/{self.ref.role}":
            raise ValueError("device resource key must be derived from its role")
        object.__setattr__(self, "resource_key", key)
        if not isinstance(self.availability, InstallationAvailability):
            raise TypeError("device availability must be InstallationAvailability")
        if not isinstance(self.health, DeviceHealth):
            raise TypeError("device health must be DeviceHealth")
        if (
            self.availability is not InstallationAvailability.AVAILABLE
            and self.health is not DeviceHealth.UNAVAILABLE
        ):
            raise ValueError("a non-available device must have unavailable health")

    @property
    def role(self) -> str:
        return self.ref.role

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref.to_dict(),
            "domain": self.domain,
            "adapter_kind": self.adapter_kind,
            "resource_key": self.resource_key,
            "availability": self.availability.value,
            "health": self.health.value,
        }


class DeviceCatalogView(Mapping[str, DeviceInfo]):
    """Immutable role-to-observation mapping with no hardware capability."""

    __slots__ = (
        "_installation_id",
        "_installation_generation",
        "_installation_state_revision",
        "_revision",
        "_availability",
        "_items",
        "_recovery_status_ref",
    )

    def __init__(
        self,
        installation_id: str,
        installation_generation: int,
        installation_state_revision: int,
        revision: int,
        items: tuple[DeviceInfo, ...],
        *,
        availability: InstallationAvailability = InstallationAvailability.AVAILABLE,
        recovery_status_ref: str | None = None,
    ) -> None:
        self._installation_id = _text(installation_id, "installation id")
        self._installation_generation = _positive_int(
            installation_generation, "installation generation"
        )
        self._installation_state_revision = _positive_int(
            installation_state_revision, "installation state revision"
        )
        self._revision = _positive_int(revision, "catalog revision")
        if not isinstance(availability, InstallationAvailability):
            raise TypeError("catalog availability must be InstallationAvailability")
        self._availability = availability

        by_role: dict[str, DeviceInfo] = {}
        for item in items:
            if not isinstance(item, DeviceInfo):
                raise TypeError("device catalog items must be DeviceInfo")
            if item.role in by_role:
                raise ValueError(f"duplicate device role {item.role!r}")
            if (
                item.ref.installation_id != self._installation_id
                or item.ref.installation_generation
                != self._installation_generation
            ):
                raise ValueError("device info ref belongs to another catalog generation")
            if item.availability is not availability:
                raise ValueError("device availability must match its catalog")
            by_role[item.role] = item
        self._items = MappingProxyType(dict(sorted(by_role.items())))

        if recovery_status_ref is None:
            self._recovery_status_ref = None
        else:
            self._recovery_status_ref = _text(
                recovery_status_ref, "recovery status ref"
            )
        if (
            availability is InstallationAvailability.AVAILABLE
            and self._recovery_status_ref is not None
        ):
            raise ValueError("an available catalog cannot have recovery status")
        if (
            availability is InstallationAvailability.RECOVERY_REQUIRED
            and self._recovery_status_ref is None
        ):
            raise ValueError("recovery-required catalog needs a recovery status ref")

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
    def installation_generation(self) -> int:
        return self._installation_generation

    @property
    def installation_state_revision(self) -> int:
        return self._installation_state_revision

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def availability(self) -> InstallationAvailability:
        return self._availability

    @property
    def recovery_status_ref(self) -> str | None:
        return self._recovery_status_ref

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
            role for role, info in self._items.items() if info.domain == normalized
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "installation_id": self._installation_id,
            "installation_generation": self._installation_generation,
            "installation_state_revision": self._installation_state_revision,
            "revision": self._revision,
            "availability": self._availability.value,
            "devices": [item.to_dict() for item in self._items.values()],
            "recovery_status_ref": self._recovery_status_ref,
        }


def unavailable_catalog(
    previous: DeviceCatalogView,
    *,
    installation_generation: int,
    installation_state_revision: int,
    revision: int,
    availability: InstallationAvailability,
    recovery_status_ref: str | None = None,
) -> DeviceCatalogView:
    """Project known roles into a capability-free unavailable generation."""

    if not isinstance(previous, DeviceCatalogView):
        raise TypeError("previous must be DeviceCatalogView")
    if availability is InstallationAvailability.AVAILABLE:
        raise ValueError("unavailable_catalog cannot publish AVAILABLE")
    items = tuple(
        DeviceInfo(
            DeviceRef(
                previous.installation_id,
                installation_generation,
                item.role,
            ),
            item.domain,
            item.adapter_kind,
            item.resource_key,
            availability,
            DeviceHealth.UNAVAILABLE,
        )
        for item in previous.values()
    )
    return DeviceCatalogView(
        previous.installation_id,
        installation_generation,
        installation_state_revision,
        revision,
        items,
        availability=availability,
        recovery_status_ref=recovery_status_ref,
    )


def _catalog_from_device_set(
    device_set,
    *,
    installation_id: str,
    installation_generation: int,
    installation_state_revision: int,
    revision: int,
) -> DeviceCatalogView:
    """Composition helper; raw ``DeviceSet`` never becomes public."""

    from .devices.registry import device_domains

    domains = tuple(device_domains())
    items = []
    for role, device in sorted(dict(device_set.devices).items()):
        domain = next(
            (item.key for item in domains if isinstance(device, item.base_type)),
            "device",
        )
        items.append(
            DeviceInfo(
                ref=DeviceRef(
                    installation_id,
                    installation_generation,
                    str(role),
                ),
                domain=domain,
                adapter_kind=f"{type(device).__module__}.{type(device).__qualname__}",
                resource_key=f"device/{role}",
            )
        )
    return DeviceCatalogView(
        installation_id,
        installation_generation,
        installation_state_revision,
        revision,
        tuple(items),
    )


__all__ = [
    "DeviceCatalogView",
    "DeviceHealth",
    "DeviceInfo",
    "DeviceRef",
    "InstallationAvailability",
]
