"""Capability-free observations of one composed installation runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from zlc_storage import canonical_text, nonnegative_integer


def _positive_pair(value: object, field: str) -> tuple[int, int]:
    try:
        pair = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field} must be a two-integer tuple") from exc
    if len(pair) != 2:
        raise ValueError(f"{field} must contain Y and X")
    normalized: list[int] = []
    for index, item in enumerate(pair):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"{field}[{index}] must be a positive integer")
        normalized.append(item)
    return normalized[0], normalized[1]


@dataclass(frozen=True, slots=True)
class ReadoutApparatusFacts:
    """Installed cross-device readout wiring and site geometry.

    This is capability-free physical configuration.  It deliberately does not
    contain a Calibration request, pulse recipe, Port, callback, or mutable
    metadata bag.  A Logic-node owner may combine it with the exact bound
    Camera and Sequencer Ports at the application composition root.
    """

    camera_role: str
    sequencer_role: str
    frame_shape_yx: tuple[int, int]
    grid_shape_yx: tuple[int, int]
    site_centers_xy: tuple[tuple[float, float], ...]
    trigger_channel: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_role",
            canonical_text(self.camera_role, "camera_role"),
        )
        object.__setattr__(
            self,
            "sequencer_role",
            canonical_text(self.sequencer_role, "sequencer_role"),
        )
        frame_shape = _positive_pair(self.frame_shape_yx, "frame_shape_yx")
        grid_shape = _positive_pair(self.grid_shape_yx, "grid_shape_yx")
        try:
            raw_centers = tuple(self.site_centers_xy)
        except TypeError as exc:
            raise TypeError("site_centers_xy must be an iterable of X,Y pairs") from exc
        if len(raw_centers) != grid_shape[0] * grid_shape[1]:
            raise ValueError(
                "site_centers_xy must contain one center per installed grid site"
            )
        height, width = frame_shape
        centers: list[tuple[float, float]] = []
        for index, value in enumerate(raw_centers):
            try:
                pair = tuple(value)
            except TypeError as exc:
                raise TypeError(
                    f"site_centers_xy[{index}] must be an X,Y pair"
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    f"site_centers_xy[{index}] must contain X and Y"
                )
            x, y = float(pair[0]), float(pair[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("site centers must be finite")
            if not 0.0 <= x < width or not 0.0 <= y < height:
                raise ValueError("site centers must lie inside the installed frame")
            centers.append((x, y))
        if len(set(centers)) != len(centers):
            raise ValueError("installed site centers must be unique")
        object.__setattr__(self, "frame_shape_yx", frame_shape)
        object.__setattr__(self, "grid_shape_yx", grid_shape)
        object.__setattr__(self, "site_centers_xy", tuple(centers))
        object.__setattr__(
            self,
            "trigger_channel",
            canonical_text(self.trigger_channel, "trigger_channel"),
        )


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
    "ReadoutApparatusFacts",
]
