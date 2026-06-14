"""Camera-readout calibration, detection, and fidelity subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from Zou_lab_control._viewer_registry import active_plotter

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold, positive_int, roi_counts
from ..core.calibration import TrapCalibration
from ..core.results import DetectionResult, DetectionTimeScanResult, SitemapResult, ThresholdResult
from ..core.utils import site_index
from ..operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from ..views.plots import plot_detection_scan
from .base import ExperimentSubsystem

if TYPE_CHECKING:  # pragma: no cover
    from ..session import NeutralAtomSession


class ReadoutSubsystem(ExperimentSubsystem):
    """All actions that depend on camera readout calibration.

    This subsystem OWNS the readout orchestration: site-map calibration,
    threshold calibration, atom detection, and readout-fidelity scans.  They all
    share the session's ``TrapCalibration`` (held on the session as the single
    calibration state) and the session's imaging-sequence/device helpers, but the
    acquire -> analyze -> store logic lives here, not in the session facade.
    """

    _session: "NeutralAtomSession"

    @property
    def current(self) -> TrapCalibration | None:
        return self._session._calibration

    def require(self, *, thresholds: bool = True) -> TrapCalibration:
        return self._session.require_calibration(require_thresholds=thresholds)

    # ------------------------------------------------------------- calibration
    def sitemap(
        self,
        *,
        frames: int = 20,
        grid_shape: Sequence[int] | None = None,
        ordering: str = "row-major",
        roi_radius: int | None = None,
        reducer: str = "mean",
        display: bool = True,
    ) -> SitemapResult:
        """Calibrate site centers from freshly acquired all-sites frames."""

        s = self._session
        grid_shape = s._grid_shape(grid_shape)
        roi_radius = int(s.defaults.get("roi_radius", 1) if roi_radius is None else roi_radius)
        exposure = s.defaults.get("sitemap_exposure", s._camera_exposure())
        sequence = s._imaging_sequence(exposure=exposure, load=True, name="sitemap")
        images = s.devices.camera.acquire(
            positive_int(frames, "frames"), sequence=sequence, sequencer=getattr(s.devices, "sequencer", None)
        )
        result = calibrate_sitemap_from_images(
            images, grid_shape=grid_shape, ordering=ordering, roi_radius=roi_radius, reducer=reducer, display=display
        )
        s._calibration = result.calibration
        s.history.append(result)
        return result

    def sitemap_from_images(self, images, *, grid_shape: Sequence[int] | None = None, **kwargs) -> SitemapResult:
        s = self._session
        result = calibrate_sitemap_from_images(images, grid_shape=s._grid_shape(grid_shape), **kwargs)
        s._calibration = result.calibration
        s.history.append(result)
        return result

    def thresholds(self, *, frames: int = 100, site: int = 0, exposure: float | None = None, display: bool = True) -> ThresholdResult:
        """Calibrate per-site thresholds from freshly acquired frames."""

        s = self._session
        calibration = s.require_calibration(require_thresholds=False)
        sequence = s._imaging_sequence(exposure=s._camera_exposure() if exposure is None else exposure, load=True, name="threshold")
        images = s.devices.camera.acquire(
            positive_int(frames, "frames"), sequence=sequence, sequencer=getattr(s.devices, "sequencer", None)
        )
        result = calibrate_threshold_from_images(images, calibration, site=site, display=display)
        s._calibration = result.calibration
        s.history.append(result)
        return result

    def thresholds_from_images(self, images, *, calibration: TrapCalibration | None = None, **kwargs) -> ThresholdResult:
        s = self._session
        calibration = s.require_calibration(require_thresholds=False) if calibration is None else calibration
        result = calibrate_threshold_from_images(images, calibration, **kwargs)
        s._calibration = result.calibration
        s.history.append(result)
        return result

    # ------------------------------------------------------------- detection
    def detect(self, *, exposure: float | None = None, display: bool = True, what: str = "occupancy") -> DetectionResult:
        """Acquire one shot and classify occupancy/counts against the calibration."""

        s = self._session
        calibration = s.require_calibration(require_thresholds=True)
        sequence = s._imaging_sequence(exposure=s._camera_exposure() if exposure is None else exposure, load=True, name="detect")
        images = s.devices.camera.acquire(1, sequence=sequence, sequencer=getattr(s.devices, "sequencer", None))
        result = detect_image(images[-1], calibration, sequence=sequence, display=display, what=what)
        s.history.append(result)
        return result

    def from_image(self, image, *, calibration: TrapCalibration | None = None, **kwargs) -> DetectionResult:
        s = self._session
        calibration = s.require_calibration(require_thresholds=True) if calibration is None else calibration
        result = detect_image(image, calibration, sequence=s.sequence, **kwargs)
        s.history.append(result)
        return result

    def detection_time(self, times: Sequence[float] | None = None, **kwargs) -> DetectionTimeScanResult:
        if times is None:
            times = self._session.defaults.get(
                "detection_times",
                np.array([2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]),
            )
        return self._scan_detection_time(times, **kwargs)

    def _scan_detection_time(
        self,
        times: Sequence[float],
        *,
        shots: int = 60,
        site: int | None = None,
        reference_exposure: float | None = None,
        reference_shots: int = 30,
        live: bool = True,
        update_time: float = 0.05,
        display: bool = True,
        pulse: Any | None = None,
    ) -> DetectionTimeScanResult:
        s = self._session
        calibration = s.require_calibration(require_thresholds=False)
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("times must contain positive finite detection times.")
        shots = positive_int(shots, "shots")
        reference_shots = positive_int(reference_shots, "reference_shots")
        data_y = np.full((len(times), 1), np.nan, dtype=float)
        reference_exposure = float(max(np.nanmax(times) * 3.0, s._camera_exposure()) if reference_exposure is None else reference_exposure)
        if not np.isfinite(reference_exposure) or reference_exposure <= 0:
            raise ValueError("reference_exposure must be positive and finite.")

        if pulse is None:
            reference_sequence = s._imaging_sequence(exposure=reference_exposure, load=True, name="reference_threshold")
            reference_sequencer = getattr(s.devices, "sequencer", None)
        else:
            reference_x_ns = float(reference_exposure) * 1e9
            configure = getattr(s.devices.camera, "configure", None)
            if callable(configure):
                configure(exposure=float(reference_exposure))
            frame_sequence = getattr(pulse, "frame_sequence", None)
            if not callable(frame_sequence):
                raise TypeError("pulse must be a PulseController returned by exp.timing.bind_pulse(...) or na.bind_pulse(...).")
            reference_sequence = frame_sequence(reference_shots, time_ns=reference_x_ns)
            reference_sequencer = getattr(pulse, "sequencer", getattr(s.devices, "sequencer", None))
        reference_images = s.devices.camera.acquire(
            reference_shots,
            sequence=reference_sequence,
            sequencer=reference_sequencer,
        )
        reference_counts = np.vstack(
            [roi_counts(image, calibration.centers, radius=calibration.roi_radius, reducer=calibration.reducer) for image in reference_images]
        )
        if site is None:
            reference_values = reference_counts.reshape(-1)
        else:
            site_idx_ref = site_index(site, reference_counts.shape[1])
            reference_values = reference_counts[:, site_idx_ref]
        reference_threshold = otsu_threshold(reference_values)
        reference_fidelity = estimate_threshold_fidelity(reference_values, reference_threshold)
        result = DetectionTimeScanResult(
            times=times,
            data_y=data_y,
            reference_exposure=reference_exposure,
            reference_threshold=float(reference_threshold),
            reference_fidelity=None if not np.isfinite(reference_fidelity.fidelity) else float(reference_fidelity.fidelity),
            reference_counts=reference_counts,
        )

        def measure(time_s: float, index: int | None = None) -> float:
            if pulse is None:
                sequence = s._imaging_sequence(exposure=float(time_s), load=True, name="detect_time_scan")
                sequencer = getattr(s.devices, "sequencer", None)
            else:
                x_ns = float(time_s) * 1e9
                configure = getattr(s.devices.camera, "configure", None)
                if callable(configure):
                    configure(exposure=float(time_s))
                frame_sequence = getattr(pulse, "frame_sequence", None)
                if not callable(frame_sequence):
                    raise TypeError("pulse must be a PulseController returned by exp.timing.bind_pulse(...) or na.bind_pulse(...).")
                sequence = frame_sequence(shots, time_ns=x_ns)
                sequencer = getattr(pulse, "sequencer", getattr(s.devices, "sequencer", None))
            images = s.devices.camera.acquire(shots, sequence=sequence, sequencer=sequencer)
            counts = np.vstack(
                [roi_counts(image, calibration.centers, radius=calibration.roi_radius, reducer=calibration.reducer) for image in images]
            )
            if site is None:
                values = counts.reshape(-1)
            else:
                site_idx = site_index(site, counts.shape[1])
                values = counts[:, site_idx]
            threshold = otsu_threshold(values)
            model = estimate_threshold_fidelity(values, threshold)
            fidelity = float(model.fidelity)
            if not np.isfinite(fidelity):
                fidelity = 0.5
            result.thresholds.append(float(threshold))
            result.model_fidelities.append(fidelity)
            return fidelity

        plotter = active_plotter()
        if live and plotter is not None:
            result.measurement = plotter.run(
                times.reshape(-1, 1),
                measure,
                data_y=data_y,
                labels=("Detection time (s)", "Fidelity", "Fidelity"),
                update_time=update_time,
                display=display,
                stop_hint="Live measurement started. Call scan.stop() to stop measurement and plot.",
            )
            result.plot = result.measurement.plot
        else:
            # No viewer registered (headless / frontend not imported): run the
            # scan synchronously and still return a complete result.
            for index, time_s in enumerate(times):
                data_y[index, 0] = measure(float(time_s), index)
            result.plot = plot_detection_scan(times, data_y[:, 0], display=display)
        s.history.append(result)
        return result

    # ------------------------------------------------------------- persistence
    def load(self, path: str | Path) -> TrapCalibration:
        return self._session.load_calibration(path)

    def save(self, path: str | Path) -> Path:
        return self._session.save_calibration(path)

    def clear(self) -> None:
        self._session._calibration = None


__all__ = ["ReadoutSubsystem"]
