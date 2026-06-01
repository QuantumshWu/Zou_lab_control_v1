"""Camera-readout calibration, detection, and fidelity subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..core.calibration import TrapCalibration
from ..core.results import DetectionResult, DetectionTimeScanResult, SitemapResult, ThresholdResult
from ..operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from .base import ExperimentSubsystem

if TYPE_CHECKING:  # pragma: no cover
    from ..session import NeutralAtomSession


class ReadoutSubsystem(ExperimentSubsystem):
    """All actions that depend on camera readout calibration.

    This is intentionally one subsystem: site-map calibration, threshold
    calibration, atom detection, and readout-fidelity scans all share the same
    ``TrapCalibration`` and should evolve together.
    """

    _session: "NeutralAtomSession"

    @property
    def current(self) -> TrapCalibration | None:
        return self._session._calibration

    def require(self, *, thresholds: bool = True) -> TrapCalibration:
        return self._session.require_calibration(require_thresholds=thresholds)

    def sitemap(self, **kwargs) -> SitemapResult:
        return self._session._calibrate_sitemap(**kwargs)

    def sitemap_from_images(self, images, *, grid_shape: Sequence[int] | None = None, **kwargs) -> SitemapResult:
        result = calibrate_sitemap_from_images(images, grid_shape=self._session._grid_shape(grid_shape), **kwargs)
        self._session._calibration = result.calibration
        self._session.history.append(result)
        return result

    def thresholds(self, **kwargs) -> ThresholdResult:
        return self._session._calibrate_threshold(**kwargs)

    def thresholds_from_images(self, images, *, calibration: TrapCalibration | None = None, **kwargs) -> ThresholdResult:
        calibration = self._session.require_calibration(require_thresholds=False) if calibration is None else calibration
        result = calibrate_threshold_from_images(images, calibration, **kwargs)
        self._session._calibration = result.calibration
        self._session.history.append(result)
        return result

    def detect(self, **kwargs) -> DetectionResult:
        return self._session._detect(**kwargs)

    def from_image(self, image, *, calibration: TrapCalibration | None = None, **kwargs) -> DetectionResult:
        calibration = self._session.require_calibration(require_thresholds=True) if calibration is None else calibration
        result = detect_image(image, calibration, sequence=self._session.sequence, **kwargs)
        self._session.history.append(result)
        return result

    def detection_time(self, times: Sequence[float] | None = None, **kwargs) -> DetectionTimeScanResult:
        if times is None:
            times = self._session.defaults.get(
                "detection_times",
                np.array([2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]),
            )
        return self._session._scan_detection_time(times, **kwargs)

    def load(self, path: str | Path) -> TrapCalibration:
        return self._session.load_calibration(path)

    def save(self, path: str | Path) -> Path:
        return self._session.save_calibration(path)

    def clear(self) -> None:
        self._session._calibration = None


__all__ = ["ReadoutSubsystem"]
