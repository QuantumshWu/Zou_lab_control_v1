"""Immutable observations of one connected installation runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zlc_storage import canonical_text

@dataclass(frozen=True, slots=True)
class ReadoutInstallationBinding:
    """Physical readout wiring, without experiment geometry.

    A camera connection can prove its endpoint identity and the sequencer lane
    that is wired to its external trigger.  It cannot know the atom-lattice
    grid or site centers: those are calibration intent/results and belong to
    the readout logic node.  Keeping only this binding here prevents the
    Device Manager's hardware form from becoming a calibration database.
    """

    camera_instance_id: str
    sequencer_instance_id: str
    trigger_channel: str

    def __post_init__(self) -> None:
        for field in ("camera_instance_id", "sequencer_instance_id", "trigger_channel"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )


@dataclass(frozen=True, slots=True)
class DeviceRef:
    """One stable instance pinned to exactly one runtime generation."""

    runtime_instance_id: str
    instance_id: str
    type_id: str
    role: str

    def __post_init__(self) -> None:
        for field in ("runtime_instance_id", "instance_id", "type_id", "role"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "runtime_instance_id": self.runtime_instance_id,
            "instance_id": self.instance_id,
            "type_id": self.type_id,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Immutable catalog fact; it contains no adapter or operation."""

    ref: DeviceRef
    domain: str
    capabilities: tuple[str, ...]
    resource_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, DeviceRef):
            raise TypeError("ref must be DeviceRef")
        for field in ("domain", "resource_key"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )
        capabilities = tuple(
            canonical_text(value, "device capability") for value in self.capabilities
        )
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("device capabilities must be unique")
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def instance_id(self) -> str:
        return self.ref.instance_id

    @property
    def type_id(self) -> str:
        return self.ref.type_id

    @property
    def role(self) -> str:
        return self.ref.role

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref.to_dict(),
            "domain": self.domain,
            "capabilities": list(self.capabilities),
            "resource_key": self.resource_key,
        }


class DeviceCatalogView(Mapping[str, DeviceInfo]):
    """Document-ordered instance catalog for one runtime generation."""

    __slots__ = ("_runtime_instance_id", "_items", "_roles")

    def __init__(
        self,
        runtime_instance_id: str,
        items: tuple[DeviceInfo, ...],
    ) -> None:
        self._runtime_instance_id = canonical_text(
            runtime_instance_id,
            "runtime_instance_id",
        )
        by_id: dict[str, DeviceInfo] = {}
        by_role: dict[str, DeviceInfo] = {}
        for item in tuple(items):
            if not isinstance(item, DeviceInfo):
                raise TypeError("items must contain DeviceInfo values")
            if item.instance_id in by_id:
                raise ValueError(f"duplicate device instance {item.instance_id!r}")
            if item.role in by_role:
                raise ValueError(f"duplicate device role {item.role!r}")
            if item.ref.runtime_instance_id != self._runtime_instance_id:
                raise ValueError("DeviceInfo belongs to another runtime generation")
            by_id[item.instance_id] = item
            by_role[item.role] = item
        self._items = MappingProxyType(by_id)
        self._roles = MappingProxyType(by_role)

    def __getitem__(self, instance_id: str) -> DeviceInfo:
        return self._items[str(instance_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    def find(self, instance_id: str) -> DeviceInfo | None:
        return self._items.get(str(instance_id))

    def require(self, instance_id: str) -> DeviceInfo:
        normalized = str(instance_id)
        try:
            return self._items[normalized]
        except KeyError as exc:
            raise KeyError(f"device instance {normalized!r} is not configured") from exc

    def require_role(self, role: str) -> DeviceInfo:
        normalized = str(role)
        try:
            return self._roles[normalized]
        except KeyError as exc:
            raise KeyError(f"device role {normalized!r} is not configured") from exc

    def roles(self, domain: str | None = None) -> tuple[str, ...]:
        if domain is None:
            return tuple(item.role for item in self._items.values())
        normalized = str(domain)
        return tuple(
            item.role for item in self._items.values() if item.domain == normalized
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_instance_id": self._runtime_instance_id,
            "devices": [item.to_dict() for item in self._items.values()],
        }


__all__ = [
    "DeviceCatalogView",
    "DeviceInfo",
    "DeviceRef",
    "ReadoutInstallationBinding",
]
