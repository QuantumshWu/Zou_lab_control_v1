"""Leaf-owned path identity for durable neutral-atom occupancy results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from zlc_storage import canonical_text


OCCUPANCY_ARTIFACT_NAMESPACE = "occupancy"


@dataclass(frozen=True, order=True, slots=True)
class OccupancyArtifactRef:
    """Path to ``occupancy.json`` relative to the Occupancy output root."""

    record_path: str

    def __post_init__(self) -> None:
        value = canonical_text(self.record_path, "record_path")
        if "\\" in value:
            raise ValueError("record_path must use POSIX separators")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("record_path must stay beneath the Occupancy output root")
        if path.name != "occupancy.json" or len(path.parts) != 2:
            raise ValueError("record_path must be '<run-name>/occupancy.json'")
        object.__setattr__(self, "record_path", path.as_posix())

    @property
    def target_ref(self) -> str:
        return f"{OCCUPANCY_ARTIFACT_NAMESPACE}/{self.record_path}"


__all__ = [
    "OCCUPANCY_ARTIFACT_NAMESPACE",
    "OccupancyArtifactRef",
]
