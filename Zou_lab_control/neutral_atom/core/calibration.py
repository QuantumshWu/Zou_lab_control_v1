"""Calibration record for the lightweight neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import centers_array, detect_atoms, grid_shape_tuple, threshold_array


@dataclass(frozen=True)
class TrapCalibration:
    centers: np.ndarray
    thresholds: np.ndarray | float
    grid_shape: tuple[int, int] | None = None
    roi_radius: int = 1
    reducer: str = "mean"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centers = centers_array(self.centers)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "thresholds", threshold_array(self.thresholds, len(centers)))
        if self.grid_shape is not None:
            shape = grid_shape_tuple(self.grid_shape)
            if int(np.prod(shape)) != len(centers):
                raise ValueError("grid_shape product must match number of centers.")
            object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "roi_radius", nonnegative_int(self.roi_radius, "roi_radius"))
        object.__setattr__(self, "reducer", str(self.reducer))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_sites(self) -> int:
        return len(self.centers)

    def detect(self, image):
        return detect_atoms(image, self.centers, self.thresholds, radius=self.roi_radius, reducer=self.reducer)

    def with_thresholds(self, thresholds, **metadata) -> "TrapCalibration":
        return TrapCalibration(
            self.centers,
            thresholds,
            grid_shape=self.grid_shape,
            roi_radius=self.roi_radius,
            reducer=self.reducer,
            metadata={**self.metadata, **metadata},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "thresholds": self.thresholds.tolist(),
            "grid_shape": None if self.grid_shape is None else list(self.grid_shape),
            "roi_radius": self.roi_radius,
            "reducer": self.reducer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrapCalibration":
        return cls(
            payload["centers"],
            payload["thresholds"],
            grid_shape=None if payload.get("grid_shape") is None else tuple(payload["grid_shape"]),
            roi_radius=payload.get("roi_radius", 1),
            reducer=payload.get("reducer", "mean"),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npz":
            np.savez(
                path,
                centers=self.centers,
                thresholds=self.thresholds,
                grid_shape=np.asarray([] if self.grid_shape is None else self.grid_shape),
                roi_radius=np.asarray(self.roi_radius),
                reducer=np.asarray(self.reducer),
                metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
            )
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrapCalibration":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=False)
            grid = data["grid_shape"]
            metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data.files else {}
            return cls(
                data["centers"],
                data["thresholds"],
                grid_shape=None if grid.size == 0 else tuple(int(v) for v in grid),
                roi_radius=int(data["roi_radius"].item()),
                reducer=str(data["reducer"].item()),
                metadata=metadata,
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def nonnegative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    numeric = float(value)
    if not np.isfinite(numeric) or int(numeric) != numeric or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


__all__ = ["TrapCalibration"]
