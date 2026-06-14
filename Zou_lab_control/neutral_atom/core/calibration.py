"""Calibration record for the lightweight neutral-atom session.

A :class:`TrapCalibration` is the single readout contract: it owns the site
centers, the per-site thresholds, and HOW a raw image becomes one scalar per
site.  ``method='box'`` (the default) reduces a square ROI like the original
readout; ``method='psf'`` carries a per-site PSF weight and does matched-filter
extraction (the Rb87 qCMOS readout).  ``signals(image)`` is the single dispatch
point; ``detect(image)`` thresholds it.  Everything downstream (results,
operations, the readout subsystem) is unchanged across methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import AtomDetection, centers_array, grid_shape_tuple, roi_counts, threshold_array
from .psf import psf_signals

SUPPORTED_METHODS = ("box", "psf")


@dataclass(frozen=True)
class TrapCalibration:
    centers: np.ndarray
    thresholds: np.ndarray | float
    grid_shape: tuple[int, int] | None = None
    roi_radius: int = 1
    reducer: str = "mean"
    method: str = "box"
    psf_weights: np.ndarray | None = None   # (N, h, w) normalized, method='psf'
    psf_boxes: np.ndarray | None = None      # (N, 4) int (x0, y0, w, h), method='psf'
    background: str = "none"                 # extraction background: 'none' | 'annulus'
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
        method = str(self.method).lower()
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "background", str(self.background).lower())
        if method == "psf":
            if self.psf_weights is None or self.psf_boxes is None:
                raise ValueError("method='psf' requires psf_weights and psf_boxes.")
            weights = np.ascontiguousarray(self.psf_weights, dtype=float)
            boxes = np.ascontiguousarray(self.psf_boxes, dtype=int)
            if weights.ndim != 3 or len(weights) != len(centers):
                raise ValueError("psf_weights must be (N_sites, h, w).")
            if boxes.shape != (len(centers), 4):
                raise ValueError("psf_boxes must be (N_sites, 4).")
            object.__setattr__(self, "psf_weights", weights)
            object.__setattr__(self, "psf_boxes", boxes)
        else:
            object.__setattr__(self, "psf_weights", None if self.psf_weights is None else np.asarray(self.psf_weights, dtype=float))
            object.__setattr__(self, "psf_boxes", None if self.psf_boxes is None else np.asarray(self.psf_boxes, dtype=int))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_sites(self) -> int:
        return len(self.centers)

    def signals(self, image) -> np.ndarray:
        """One scalar per site for ``image`` using this calibration's method."""

        if self.method == "psf":
            return psf_signals(image, self.psf_weights, self.psf_boxes, background=self.background or "annulus")
        return roi_counts(image, self.centers, radius=self.roi_radius, reducer=self.reducer)

    def detect(self, image) -> AtomDetection:
        """Classify occupancy by thresholding the per-site signals (atoms are bright)."""

        counts = np.asarray(self.signals(image), dtype=float)
        thresholds = self.thresholds
        occupied = counts > thresholds
        return AtomDetection(
            counts=counts,
            occupied=occupied,
            occupied_indices=np.flatnonzero(occupied).astype(int).tolist(),
            thresholds=thresholds,
        )

    def with_thresholds(self, thresholds, **metadata) -> "TrapCalibration":
        return TrapCalibration(
            self.centers,
            thresholds,
            grid_shape=self.grid_shape,
            roi_radius=self.roi_radius,
            reducer=self.reducer,
            method=self.method,
            psf_weights=self.psf_weights,
            psf_boxes=self.psf_boxes,
            background=self.background,
            metadata={**self.metadata, **metadata},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "thresholds": self.thresholds.tolist(),
            "grid_shape": None if self.grid_shape is None else list(self.grid_shape),
            "roi_radius": self.roi_radius,
            "reducer": self.reducer,
            "method": self.method,
            "psf_weights": None if self.psf_weights is None else self.psf_weights.tolist(),
            "psf_boxes": None if self.psf_boxes is None else self.psf_boxes.tolist(),
            "background": self.background,
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
            method=payload.get("method", "box"),
            psf_weights=payload.get("psf_weights"),
            psf_boxes=payload.get("psf_boxes"),
            background=payload.get("background", "none"),
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
                method=np.asarray(self.method),
                background=np.asarray(self.background),
                psf_weights=np.asarray([] if self.psf_weights is None else self.psf_weights),
                psf_boxes=np.asarray([] if self.psf_boxes is None else self.psf_boxes),
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
            psf_weights = data["psf_weights"] if "psf_weights" in data.files else None
            psf_boxes = data["psf_boxes"] if "psf_boxes" in data.files else None
            return cls(
                data["centers"],
                data["thresholds"],
                grid_shape=None if grid.size == 0 else tuple(int(v) for v in grid),
                roi_radius=int(data["roi_radius"].item()),
                reducer=str(data["reducer"].item()),
                method=str(data["method"].item()) if "method" in data.files else "box",
                psf_weights=None if psf_weights is None or psf_weights.size == 0 else psf_weights,
                psf_boxes=None if psf_boxes is None or psf_boxes.size == 0 else psf_boxes,
                background=str(data["background"].item()) if "background" in data.files else "none",
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


__all__ = ["TrapCalibration", "SUPPORTED_METHODS"]
