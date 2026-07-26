"""Low-level physical asset facts for one installation.

The experiment config chooses a logical role.  It never invents a physical
identity, resource key, or adapter matcher: those facts live in this
machine-level document and are content-addressed as one immutable revision.
This owner is deliberately below both installation dispatch and concrete
device attachments so device composition never depends back on dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    ResourceKey,
)
from zlc_storage import canonical_digest, canonical_text as _canonical_text


ASSET_MAP_FORMAT = "zlc_neutral_atom.InstallationAssetMap"


def adapter_kind(device: object) -> str:
    cls = type(device)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class InstallationAsset:
    """One exact role-to-physical-asset binding owned by an installation."""

    asset_id: str
    role: str
    resource_key: ResourceKey
    adapter_kind: str
    evidence_kind: DeviceIdentityEvidenceKind
    expected_identity: str

    def __post_init__(self) -> None:
        _canonical_text(self.asset_id, "asset_id")
        _canonical_text(self.role, "role")
        if not isinstance(self.resource_key, ResourceKey):
            raise TypeError("resource_key must be ResourceKey")
        if self.resource_key.segments[0] != "device":
            raise ValueError("installation asset ResourceKeys must live below device/")
        _canonical_text(self.adapter_kind, "adapter_kind")
        if not isinstance(self.evidence_kind, DeviceIdentityEvidenceKind):
            raise TypeError("evidence_kind must be DeviceIdentityEvidenceKind")
        _canonical_text(self.expected_identity, "expected_identity")

    def canonical_value(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "role": self.role,
            "resource_key": str(self.resource_key),
            "adapter_kind": self.adapter_kind,
            "evidence_kind": self.evidence_kind.value,
            "expected_identity": self.expected_identity,
        }


class InstallationAssetMap:
    """Validated immutable AssetMap whose revision is its content digest."""

    def __init__(self, assets: tuple[InstallationAsset, ...]) -> None:
        normalized = tuple(sorted(assets, key=lambda asset: asset.role))
        by_role: dict[str, InstallationAsset] = {}
        asset_ids: set[str] = set()
        keys: set[ResourceKey] = set()
        for asset in normalized:
            if not isinstance(asset, InstallationAsset):
                raise TypeError("assets must contain InstallationAsset values")
            if asset.role in by_role:
                raise ValueError(f"duplicate AssetMap role {asset.role!r}")
            if asset.asset_id in asset_ids:
                raise ValueError(f"duplicate AssetMap asset_id {asset.asset_id!r}")
            if asset.resource_key in keys:
                raise ValueError(f"duplicate AssetMap ResourceKey {asset.resource_key}")
            by_role[asset.role] = asset
            asset_ids.add(asset.asset_id)
            keys.add(asset.resource_key)
        canonical = {
            "format": ASSET_MAP_FORMAT,
            "assets": [asset.canonical_value() for asset in normalized],
        }
        self._assets = normalized
        self._by_role = MappingProxyType(by_role)
        self._revision = canonical_digest(canonical)

    @property
    def assets(self) -> tuple[InstallationAsset, ...]:
        return self._assets

    @property
    def revision(self) -> str:
        return self._revision

    def require(self, role: str, device: object) -> InstallationAsset:
        role = _canonical_text(role, "role")
        asset = self._by_role.get(role)
        if asset is None:
            raise RuntimeError(f"installation AssetMap has no asset for role {role!r}")
        actual_kind = adapter_kind(device)
        if actual_kind != asset.adapter_kind:
            raise RuntimeError(
                f"AssetMap role {role!r} requires adapter {asset.adapter_kind!r}, "
                f"got {actual_kind!r}"
            )
        return asset

    def to_dict(self) -> dict[str, object]:
        return {
            "format": ASSET_MAP_FORMAT,
            "assets": [asset.canonical_value() for asset in self._assets],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "InstallationAssetMap":
        if not isinstance(value, Mapping):
            raise TypeError("installation AssetMap must be a mapping")
        if set(value) != {"format", "assets"}:
            raise ValueError("installation AssetMap must contain only format and assets")
        if value["format"] != ASSET_MAP_FORMAT:
            raise ValueError(
                f"unsupported installation AssetMap format {value['format']!r}"
            )
        rows = value["assets"]
        if not isinstance(rows, list):
            raise TypeError("installation AssetMap assets must be a list")
        assets = []
        required = {
            "asset_id",
            "role",
            "resource_key",
            "adapter_kind",
            "evidence_kind",
            "expected_identity",
        }
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping) or set(raw) != required:
                raise ValueError(
                    f"AssetMap assets[{index}] must contain exactly {sorted(required)}"
                )
            try:
                evidence_kind = DeviceIdentityEvidenceKind(str(raw["evidence_kind"]))
            except ValueError as exc:
                raise ValueError(
                    f"AssetMap assets[{index}] has unknown evidence_kind"
                ) from exc
            assets.append(
                InstallationAsset(
                    asset_id=str(raw["asset_id"]),
                    role=str(raw["role"]),
                    resource_key=ResourceKey.parse(str(raw["resource_key"])),
                    adapter_kind=str(raw["adapter_kind"]),
                    evidence_kind=evidence_kind,
                    expected_identity=str(raw["expected_identity"]),
                )
            )
        return cls(tuple(assets))

    @classmethod
    def load(cls, path: str | Path) -> "InstallationAssetMap":
        path = Path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"real hardware requires an installation AssetMap at {path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read installation AssetMap {path}: {exc}") from exc
        return cls.from_mapping(value)

    @classmethod
    def ephemeral(
        cls,
        devices: Mapping[str, object],
    ) -> "InstallationAssetMap":
        """Build the deterministic in-memory map used only by virtual/test rigs."""

        assets = []
        for role, device in sorted(devices.items()):
            role = _canonical_text(role, "role")
            assets.append(
                InstallationAsset(
                    asset_id=f"virtual-{role}",
                    role=role,
                    resource_key=ResourceKey.parse(f"device/{role}"),
                    adapter_kind=adapter_kind(device),
                    evidence_kind=(
                        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
                    ),
                    expected_identity=f"installation-endpoint:virtual:{role}",
                )
            )
        return cls(tuple(assets))


__all__ = [
    "ASSET_MAP_FORMAT",
    "InstallationAsset",
    "InstallationAssetMap",
    "adapter_kind",
]
