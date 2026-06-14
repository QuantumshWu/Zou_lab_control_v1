"""Camera-readout calibration, detection, and fidelity subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from Zou_lab_control._viewer_registry import active_plotter

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold, positive_int
from ..core.calibration import TrapCalibration
from ..core.results import DetectionResult, DetectionTimeScanResult, SitemapResult, ThresholdResult
from ..core.utils import json_ready, site_index
from ..operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from ..operations.fidelity import FidelityReport, characterize_readout
from ..operations.imageio import index_run
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
        method: str = "box",
        psf_half_width: int = 3,
        display: bool = True,
    ) -> SitemapResult:
        """Calibrate site centers from freshly acquired all-sites frames.

        ``method='psf'`` additionally fits a per-site PSF weight from the
        all-sites average, so subsequent thresholds/detection use matched-filter
        (Rb87) readout instead of the square-ROI box reducer.
        """

        s = self._session
        grid_shape = s._grid_shape(grid_shape)
        roi_radius = int(s.defaults.get("roi_radius", 1) if roi_radius is None else roi_radius)
        exposure = s.defaults.get("sitemap_exposure", s._camera_exposure())
        sequence = s._imaging_sequence(exposure=exposure, load=True, name="sitemap")
        images = s.devices.camera.acquire(
            positive_int(frames, "frames"), sequence=sequence, sequencer=getattr(s.devices, "sequencer", None)
        )
        result = calibrate_sitemap_from_images(
            images, grid_shape=grid_shape, ordering=ordering, roi_radius=roi_radius, reducer=reducer,
            method=method, psf_half_width=psf_half_width, display=display,
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

    def thresholds(self, *, frames: int = 100, site: int = 0, exposure: float | None = None, method: str = "otsu", display: bool = True) -> ThresholdResult:
        """Calibrate per-site thresholds from freshly acquired frames.

        ``method='bimodal'`` fits dark/bright Gaussian cores per site (Rb87);
        ``'otsu'`` (default) is the single-split threshold.
        """

        s = self._session
        calibration = s.require_calibration(require_thresholds=False)
        sequence = s._imaging_sequence(exposure=s._camera_exposure() if exposure is None else exposure, load=True, name="threshold")
        images = s.devices.camera.acquire(
            positive_int(frames, "frames"), sequence=sequence, sequencer=getattr(s.devices, "sequencer", None)
        )
        result = calibrate_threshold_from_images(images, calibration, site=site, method=method, display=display)
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

    # --------------------------------------------- file-based (real-data) workflow
    def sitemap_from_dir(
        self,
        data_dir: str | Path,
        prefix: str = "img",
        *,
        shots_per_group: int = 4,
        short_shot: int = 3,
        ref_shots: Sequence[int] = (1, 2, 4),
        max_groups: int | None = None,
        grid_shape: Sequence[int] | None = None,
        ordering: str = "row-major",
        roi_radius: int | None = None,
        reducer: str = "mean",
        method: str = "psf",
        psf_half_width: int = 3,
        display: bool = False,
    ) -> SitemapResult:
        """Calibrate the site map from raw frames SAVED IN A FOLDER (the real-data
        workflow): index ``PREFIX<n>`` frames, average the reference frames into an
        all-sites template, DETECT site centers from it, and (``method='psf'``) fit
        per-site PSF weights.  Identical on real hardware -- only who wrote the
        folder differs (see ``na.write_virtual_run`` for the virtual writer)."""

        s = self._session
        grid_shape = s._grid_shape(grid_shape)
        roi_radius = int(s.defaults.get("roi_radius", 1) if roi_radius is None else roi_radius)
        run = index_run(data_dir, prefix, shots_per_group=shots_per_group, short_shot=short_shot,
                        ref_shots=ref_shots, max_groups=max_groups)
        template_frames = list(run.template_frames())
        result = calibrate_sitemap_from_images(
            template_frames, grid_shape=grid_shape, ordering=ordering, roi_radius=roi_radius,
            reducer=reducer, method=method, psf_half_width=psf_half_width, display=display,
        )
        s._calibration = result.calibration
        s.history.append(result)
        return result

    def characterize_from_dir(
        self,
        data_dir: str | Path,
        prefix: str = "img",
        *,
        shots_per_group: int = 4,
        short_shot: int = 3,
        ref_shots: Sequence[int] = (1, 2, 4),
        max_groups: int | None = None,
        train_fraction: float = 0.9,
        seed: int = 0,
        store_thresholds: bool = True,
        results_dir: str | Path | None = None,
        save: bool = True,
    ) -> FidelityReport:
        """Per-site readout fidelity from raw frames SAVED IN A FOLDER (the real-data
        Rb87 flow, replacing the in-memory acquire loop).

        Indexes ``PREFIX<n>`` frames into ``shots_per_group``-frame groups
        (``short_shot`` = the readout being characterized, ``ref_shots`` = the
        high-SNR frames that vote ground truth), extracts per-site signals through
        the current calibration's method (box/PSF), runs the per-site threshold +
        held-out fidelity characterization, writes results to ``results_dir``
        (default ``<data_dir>_results``), and optionally stores the trained
        thresholds back into the calibration.  This is the SAME code a real run
        uses -- only the frames' author (real camera vs ``na.write_virtual_run``)
        differs."""

        s = self._session
        cal = s.require_calibration(require_thresholds=False)
        run = index_run(data_dir, prefix, shots_per_group=shots_per_group, short_shot=short_shot,
                        ref_shots=ref_shots, max_groups=max_groups)
        n_groups, n_ref = run.n_groups, len(run.ref_shots)
        short = np.empty((n_groups, cal.n_sites), dtype=float)
        ref = np.empty((n_groups, n_ref, cal.n_sites), dtype=float)
        for g, frame in enumerate(run.short_frames()):
            short[g] = cal.signals(frame)
        ref_iter = run.reference_frames()
        for g in range(n_groups):
            for r in range(n_ref):
                ref[g, r] = cal.signals(next(ref_iter))

        report = characterize_readout(short, ref, train_fraction=train_fraction, seed=seed)
        if store_thresholds:
            thresholds = report.thresholds.copy()
            bad = ~np.isfinite(thresholds)
            if np.any(bad):
                fallback = report.global_threshold
                if not np.isfinite(fallback):
                    finite = thresholds[~bad]
                    fallback = float(np.median(finite)) if finite.size else 0.0
                thresholds[bad] = fallback
            s._calibration = cal.with_thresholds(
                thresholds, stage="characterized", thresholds_calibrated=True, threshold_method="per_site_reference"
            )
        if save:
            out_dir = Path(results_dir) if results_dir is not None else Path(str(Path(data_dir).expanduser()) + "_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_dir / "characterize_signals.npz",
                short_signals=short, reference_signals=ref,
                thresholds=np.asarray(report.thresholds, dtype=float),
                site_fidelities=np.asarray(report.site_fidelities, dtype=float),
            )
            meta = {
                **report.summary(),
                "data_dir": str(Path(data_dir).expanduser()),
                "prefix": str(prefix), "n_groups": n_groups, "shots_per_group": run.shots_per_group,
                "short_shot": run.short_shot, "ref_shots": list(run.ref_shots),
                "ablation": report.ablation,
            }
            (out_dir / "characterize_summary.json").write_text(
                json.dumps(json_ready(meta), indent=2), encoding="utf-8")
            try:
                report.results_dir = str(out_dir)
            except Exception:
                pass
        s.history.append(report)
        return report

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
        # Extract through the calibration's own method (box or PSF), so a
        # detection-time scan on a PSF calibration uses PSF signals -- the same
        # quantity detect() compares.
        reference_counts = np.vstack([calibration.signals(image) for image in reference_images])
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
            counts = np.vstack([calibration.signals(image) for image in images])
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
