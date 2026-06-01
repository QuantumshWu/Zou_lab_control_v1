"""Result object contracts for neutral-atom notebooks and subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import AtomDetection, FidelityEstimate, estimate_threshold_fidelity
from .calibration import TrapCalibration
from .utils import html_summary, site_index
from ..timing import PulseSequence
from ..timing.verilog import VerilogBuild
from ..views.plots import plot_detection_image, plot_site_values, plot_threshold_hist


class ResultObject:
    """Common surface for notebook result objects that may own a frontend plot."""

    plot: Any = None

    @property
    def data_figure(self):
        """Frontend post-processing handle, created lazily from ``plot`` when possible."""

        plot = getattr(self, "plot", None)
        if plot is None:
            return None
        data_figure = getattr(plot, "data_figure", None)
        if data_figure is None and hasattr(plot, "to_data_figure"):
            data_figure = plot.to_data_figure()
        return data_figure

    @property
    def fig(self):
        plot = getattr(self, "plot", None)
        return None if plot is None else getattr(plot, "fig", None)

    @property
    def ax(self):
        plot = getattr(self, "plot", None)
        return None if plot is None else getattr(plot, "ax", None)


class MeasurementTaskResult(ResultObject):
    """Shared contract for live/async experiment tasks."""

    measurement: Any = None

    def stop(self):
        """Stop the live measurement and its attached frontend plot."""

        measurement = getattr(self, "measurement", None)
        if measurement is not None and hasattr(measurement, "stop"):
            measurement.stop()
            self.plot = getattr(measurement, "plot", getattr(self, "plot", None))
            return self

        plot = getattr(self, "plot", None)
        if plot is not None and hasattr(plot, "stop"):
            plot.stop()
        return self

    @property
    def points_done(self) -> int:
        measurement = getattr(self, "measurement", None)
        if measurement is not None and hasattr(measurement, "points_done"):
            return int(measurement.points_done)
        data_y = getattr(self, "data_y", None)
        if data_y is None:
            return 0
        arr = np.asarray(data_y)
        if arr.ndim == 1:
            return int(np.count_nonzero(np.isfinite(arr)))
        return int(np.count_nonzero(np.isfinite(arr[:, 0])))

    @property
    def running(self) -> bool:
        measurement = getattr(self, "measurement", None)
        return bool(measurement is not None and getattr(measurement, "running", False))

    @property
    def measurement_done(self) -> bool:
        measurement = getattr(self, "measurement", None)
        return bool(measurement is not None and getattr(measurement, "done", False))


@dataclass
class CaptureResult(ResultObject):
    """Raw frame capture plus the plot shown in the notebook."""

    images: list[np.ndarray]
    sequence: PulseSequence
    plot: Any = None

    @property
    def image(self) -> np.ndarray:
        return self.images[-1]

    def summary(self) -> dict[str, Any]:
        return {
            "frames": len(self.images),
            "image_shape": list(self.image.shape),
            "sequence": self.sequence.name,
        }

    def _repr_html_(self) -> str:
        return html_summary("CaptureResult", self.summary())


@dataclass
class SitemapResult(ResultObject):
    """Site-center calibration result."""

    calibration: TrapCalibration
    average_image: np.ndarray
    images: list[np.ndarray]
    plot: Any = None

    @property
    def centers(self) -> np.ndarray:
        return self.calibration.centers

    def save(self, path: str | Path) -> Path:
        return self.calibration.save(path)

    def summary(self) -> dict[str, Any]:
        return {
            "n_sites": self.calibration.n_sites,
            "grid_shape": None if self.calibration.grid_shape is None else list(self.calibration.grid_shape),
            "roi_radius": self.calibration.roi_radius,
        }

    def _repr_html_(self) -> str:
        return html_summary("SitemapResult", self.summary())


@dataclass
class ThresholdResult(ResultObject):
    """Threshold calibration data for all sites."""

    calibration: TrapCalibration
    counts: np.ndarray
    thresholds: np.ndarray
    selected_site: int
    plot: Any = None
    fidelity: FidelityEstimate | None = None

    def plot_site(self, site: int | None = None, *, display: bool = True):
        site = self.selected_site if site is None else site_index(site, self.counts.shape[1])
        fidelity = estimate_threshold_fidelity(self.counts[:, site], self.thresholds[site])
        plot = plot_threshold_hist(
            self.counts[:, site],
            threshold=self.thresholds[site],
            labels=(f"Site {site} ROI counts", "Shots", "Population"),
            display=display,
        )
        self.selected_site = site
        self.plot = plot
        self.fidelity = fidelity
        return plot

    def save(self, path: str | Path) -> Path:
        return self.calibration.save(path)

    def summary(self) -> dict[str, Any]:
        return {
            "shots": int(self.counts.shape[0]),
            "sites": int(self.counts.shape[1]),
            "selected_site": int(self.selected_site),
            "selected_threshold": float(self.thresholds[self.selected_site]),
            "selected_fidelity": None if self.fidelity is None else float(self.fidelity.fidelity),
        }

    def _repr_html_(self) -> str:
        return html_summary("ThresholdResult", self.summary())


@dataclass
class DetectionResult(ResultObject):
    """One classified detection shot."""

    image: np.ndarray
    detection: AtomDetection
    calibration: TrapCalibration
    sequence: PulseSequence
    plot: Any = None

    @property
    def counts(self) -> np.ndarray:
        return self.detection.counts

    @property
    def occupied(self) -> np.ndarray:
        return self.detection.occupied

    @property
    def occupied_indices(self) -> list[int]:
        return self.detection.occupied_indices

    def plot_counts(self, *, display: bool = True):
        self.plot = plot_site_values(
            self.calibration.centers,
            self.counts,
            labels=("Camera x (px)", "Camera y (px)", "ROI counts"),
            display=display,
        )
        return self.plot

    def plot_occupancy(self, *, display: bool = True):
        self.plot = plot_detection_image(
            self.image,
            self.calibration.centers,
            self.occupied,
            roi_radius=self.calibration.roi_radius,
            labels=("Camera x (px)", "Camera y (px)", "Counts"),
            display=display,
        )
        return self.plot

    def summary(self) -> dict[str, Any]:
        return {"loaded_atoms": len(self.occupied_indices), "occupied_indices": list(self.occupied_indices)}

    def _repr_html_(self) -> str:
        return html_summary("DetectionResult", self.summary())


@dataclass
class DetectionTimeScanResult(MeasurementTaskResult):
    """Live or completed detection-time scan."""

    times: np.ndarray
    data_y: np.ndarray
    thresholds: list[float] = field(default_factory=list)
    model_fidelities: list[float] = field(default_factory=list)
    reference_exposure: float | None = None
    reference_threshold: float | None = None
    reference_fidelity: float | None = None
    reference_counts: np.ndarray | None = None
    measurement: Any = None
    plot: Any = None

    @property
    def fidelities(self) -> np.ndarray:
        return self.data_y[:, 0]

    @property
    def finished(self) -> bool:
        return bool(np.all(np.isfinite(self.fidelities)))

    def summary(self) -> dict[str, Any]:
        finite = np.isfinite(self.fidelities)
        best = None
        if np.any(finite):
            idx = int(np.nanargmax(self.fidelities))
            best = {"time": float(self.times[idx]), "fidelity": float(self.fidelities[idx])}
        return {
            "points": len(self.times),
            "points_done": self.points_done,
            "running": self.running,
            "finished": self.finished,
            "best": best,
        }

    def _repr_html_(self) -> str:
        return html_summary("DetectionTimeScanResult", self.summary())


@dataclass
class PreflightReport:
    """Notebook-readable safety summary before a real shot."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    sequence_table: list[dict[str, object]]
    device_snapshot: dict[str, Any]
    verilog: VerilogBuild | None = None

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("preflight failed: " + "; ".join(self.errors))

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "sequence": self.sequence_table,
            "devices": self.device_snapshot,
            "verilog": None
            if self.verilog is None
            else {
                "module_name": self.verilog.module_name,
                "clock_hz": self.verilog.clock_hz,
                "channels": self.verilog.channels,
                "ticks": self.verilog.ticks,
                "masks": self.verilog.masks,
                "sha256": self.verilog.source_sha256,
            },
        }


__all__ = [
    "CaptureResult",
    "DetectionResult",
    "DetectionTimeScanResult",
    "MeasurementTaskResult",
    "PreflightReport",
    "ResultObject",
    "SitemapResult",
    "ThresholdResult",
]
