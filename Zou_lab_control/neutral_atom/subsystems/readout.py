"""Camera-readout calibration, detection, and fidelity subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..core.analysis import positive_int
from ..core.calibration import TrapCalibration
from ..core.results import DetectionResult, DetectionTimeScanResult, SitemapResult, ThresholdResult
from ..core.utils import json_ready
from ..operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from ..operations.fidelity import FidelityReport, characterize_readout
from ..operations.imageio import DEFAULT_REF_SHOTS, DEFAULT_SHORT_SHOT, DEFAULT_SHOTS_PER_GROUP, index_run
from ..operations.measurement import (
    MeasurementSpec,
    NFramePlan,
    OtsuFidelityReducer,
    ParamDecl,
    ScanAxis,
    ScanResult,
    ScannedMeasurement,
    otsu_fidelity_from_frames,
)
from ..operations.temperature import ReleaseRecapturePlan, SurvivalReducer
from ..timing import PulseTableState
from ..devices.base import arm_then_fire
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
        seq_dev = getattr(s.devices, "sequencer", None)
        images = s.devices.camera.acquire(
            positive_int(frames, "frames"), sequence=sequence,
            on_armed=arm_then_fire(seq_dev, sequence),
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
        expo = s._camera_exposure() if exposure is None else float(exposure)
        sequence = s._imaging_sequence(exposure=expo, load=True, name="threshold")
        seq_dev = getattr(s.devices, "sequencer", None)
        images = s.devices.camera.acquire(
            positive_int(frames, "frames"), sequence=sequence,
            on_armed=arm_then_fire(seq_dev, sequence),
        )
        # The exposure these thresholds are learnt at is stamped INSIDE calibrate_threshold_from_images
        # (the single source every threshold path routes through, #H3w-3) so a later readout -- e.g. the
        # release-recapture survival frames -- self-matches it and never floors/sticks (#H3v-2).
        result = calibrate_threshold_from_images(
            images, calibration, site=site, method=method, exposure=expo, display=display)
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
        """Acquire one shot and classify occupancy/counts against the calibration.

        The default readout exposure is the calibration's ``threshold_exposure`` -- the gate time the
        thresholds were LEARNT at -- NOT the camera's current exposure.  The per-site thresholds (bright
        counts) are only valid at the exposure they were measured at: imaging at a different exposure
        (e.g. a 2 ms frame against 20 ms thresholds) classifies every atom dark -> ~50 % occupancy while
        the calibration still claims 99 %.  Single source: this exposure, the calibration, and a re-image
        always agree (#issue-2)."""

        s = self._session
        calibration = s.require_calibration(require_thresholds=True)
        if exposure is None:
            # The authoritative reader of the calibration's threshold_exposure (the gate the
            # thresholds were learnt at); fall back to the camera's current exposure when unstamped.
            exposure = calibration.readout_exposure(fallback=s._camera_exposure())
        sequence = s._imaging_sequence(exposure=float(exposure), load=True, name="detect")
        seq_dev = getattr(s.devices, "sequencer", None)
        images = s.devices.camera.acquire(
            1, sequence=sequence, on_armed=arm_then_fire(seq_dev, sequence))
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
        shots_per_group: int = DEFAULT_SHOTS_PER_GROUP,
        short_shot: int = DEFAULT_SHORT_SHOT,
        ref_shots: Sequence[int] = DEFAULT_REF_SHOTS,
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
        shots_per_group: int = DEFAULT_SHOTS_PER_GROUP,
        short_shot: int = DEFAULT_SHORT_SHOT,
        ref_shots: Sequence[int] = DEFAULT_REF_SHOTS,
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

    def readout_duration_fidelity(self, times: Sequence[float] | None = None, **kwargs) -> DetectionTimeScanResult:
        """Semantic alias of :meth:`detection_time`.

        A detection-time scan IS a readout-duration-vs-fidelity measurement (it
        sweeps the imaging/probe duration and reports single-shot fidelity).  This
        name reads clearly next to :meth:`temperature` for callers/GUI that pick a
        measurement by intent."""

        return self.detection_time(times, **kwargs)

    def build_detection_scan(
        self,
        times: Sequence[float],
        *,
        shots: int = 60,
        site: int | None = None,
        pulse: Any | None = None,
    ) -> ScannedMeasurement:
        """Build (but do not run) the readout-duration -> fidelity scan.

        The exposure axis (``times`` seconds) x an ``NFramePlan`` x an
        ``OtsuFidelityReducer`` over the one scan engine; ``pulse is None`` drives
        the session imaging path (the ``_SessionImagingController`` adapter), a
        bound ``PulseController`` drives its exposure slot and configures the real
        camera per point.  Single source of truth for the detection-time scan:
        both :meth:`_scan_detection_time` (live result wrapper) and the
        Readout-duration measurement spec / logic node call this same builder."""

        s = self._session
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("times must contain positive finite detection times.")
        calibration = s.require_calibration(require_thresholds=False)
        controller = self._scan_pulse(pulse, name="detect_time_scan")
        reducer = OtsuFidelityReducer(site=site)
        plan: Any = NFramePlan(n_frames=positive_int(shots, "shots"))
        if pulse is not None:
            # Configure the real camera exposure per point, matching the prior path.
            plan = _ExposureConfiguringPlan(plan, s.devices.camera)
        return self._build_slot_scan(
            controller, calibration, slot="exposure", values=times, label="Detection time",
            plan=plan, reducer=reducer, shots_per_point=1,
        )

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
        reference_exposure = float(max(np.nanmax(times) * 3.0, s._camera_exposure()) if reference_exposure is None else reference_exposure)
        if not np.isfinite(reference_exposure) or reference_exposure <= 0:
            raise ValueError("reference_exposure must be positive and finite.")

        # The detection-time scan IS a ScannedMeasurement over the exposure axis
        # with an N-frame plan and an otsu-fidelity reducer; build the controller
        # (the session imaging path when no GUI pulse is bound) and reuse the one
        # engine.  The reference-threshold acquisition stays here -- it is part of
        # this specific result object, not of the generic scan loop.
        reference_controller = self._scan_pulse(pulse, name="reference_threshold")
        configure = getattr(s.devices.camera, "configure", None)
        if pulse is not None and callable(configure):
            configure(exposure=float(reference_exposure))
        reference_sequence = reference_controller.frame_sequence(reference_shots, time_ns=float(reference_exposure) * 1e9)
        ref_seqr = reference_controller.sequencer
        reference_images = s.devices.camera.acquire(
            reference_shots, sequence=reference_sequence,
            on_armed=arm_then_fire(ref_seqr, reference_sequence),
        )
        # Extract through the calibration's own method (box or PSF) and Otsu-split + score via the
        # ONE shared pipeline -- so the held-out reference uses the SAME signals/threshold/fidelity
        # the live OtsuFidelityReducer does, the same quantity detect() compares (#C3).
        reference_counts, reference_threshold, reference_fidelity = otsu_fidelity_from_frames(
            reference_images, calibration, site)

        scan = self.build_detection_scan(times, shots=shots, site=site, pulse=pulse)
        reducer = scan.reducer
        inner = scan.run(
            live=live, update_time=update_time, display=display,
            stop_hint="Live measurement started. Call scan.stop() to stop measurement and plot.",
        )
        # Share the reducer's list objects (not copies): a live scan fills them on
        # the worker thread, so the returned result must see those same growing
        # lists -- exactly as the prior inline `result.thresholds.append(...)` did.
        result = DetectionTimeScanResult(
            times=times,
            data_y=inner.data_y,
            thresholds=reducer.thresholds,
            model_fidelities=reducer.fidelities,
            reference_exposure=reference_exposure,
            reference_threshold=float(reference_threshold),
            reference_fidelity=None if not np.isfinite(reference_fidelity.fidelity) else float(reference_fidelity.fidelity),
            reference_counts=reference_counts,
            measurement=inner.measurement,
            plot=inner.plot,
        )
        s.history.append(result)
        return result

    def _scan_pulse(self, pulse: Any | None, *, name: str):
        """Return the PulseController-shaped object the scan engine drives.

        A GUI/bound ``PulseController`` is used directly; with ``pulse is None``
        the scan falls back to the session's imaging sequence wrapped in a tiny
        adapter that exposes the SAME ``frame_sequence``/``sequencer`` contract,
        so the engine has a single code path."""

        if pulse is not None:
            if not callable(getattr(pulse, "frame_sequence", None)):
                raise TypeError("pulse must be a PulseController returned by exp.timing.bind_pulse(...) or na.bind_pulse(...).")
            return pulse
        return _SessionImagingController(self._session, name=name)

    def _build_slot_scan(
        self, controller: Any, calibration: TrapCalibration, *,
        slot: str, values: Sequence[float], label: str,
        plan: Any, reducer: Any, shots_per_point: int, unit: str = "s", kind: str = "duration",
    ) -> ScannedMeasurement:
        """Assemble a one-slot pulse-slot sweep on the shared scan engine.

        This is the SPINE every coupled-reduce measurement shares -- "sweep a NAMED
        pulse slot, reduce each point": a :class:`ScanAxis` over ``slot`` x a
        ``ShotPlan`` x a ``PointReducer``, run by the one :class:`ScannedMeasurement`.
        :meth:`build_temperature_scan` (slot ``s0`` trap-off -> survival) and
        :meth:`build_detection_scan` (slot ``exposure`` -> Otsu fidelity) are the SAME
        concept and differ ONLY in ``(slot, plan, reducer)`` -- so they call THIS instead
        of re-spelling the engine wiring (the duplication the audit flagged).

        The GUI ``pulse_scan`` measurement is the DECOUPLED sibling of this family: it
        sweeps named pulse slots too, but instead of an inline reducer it publishes the
        raw frames and reads y from a SEPARATE processor's signal (see the scan-taxonomy
        note in MAINTAINER_NOTES).  Reducers stay inline here only because survival
        (a two-frame pair) and per-frame fidelity need the multi-frame PLAN structure a
        single-frame processor cannot see."""

        s = self._session
        axis = ScanAxis(
            slot=slot, values=np.asarray(values, dtype=float).reshape(-1),
            label=label, unit=unit, kind=kind,
        )
        sequencer = getattr(controller, "sequencer", getattr(s.devices, "sequencer", None))
        return ScannedMeasurement(
            controller, s.devices.camera, sequencer, calibration,
            axis, plan, reducer, shots_per_point=positive_int(shots_per_point, "shots_per_point"),
        )

    def build_temperature_scan(
        self,
        t_off_s: Sequence[float],
        *,
        pulse: Any,
        shots: int = 1,
        per_site: bool = False,
    ) -> ScannedMeasurement:
        """Build (but do not run) the release-recapture survival scan.

        The trap-off axis (``t_off_s`` seconds, bound slot ``s0``) x a
        ``ReleaseRecapturePlan`` (two frames per point) x a ``SurvivalReducer``
        over the one scan engine.  Single source of truth for release-recapture:
        both :meth:`temperature` (which runs it) and the Temperature measurement
        spec / logic node call this same builder, so the API one-liner and the GUI
        Start button drive identical acquisition.  Validates the bound pulse and
        requires a thresholded calibration (survival is a binary occupancy
        comparison)."""

        s = self._session
        t_off = np.asarray(t_off_s, dtype=float).reshape(-1)
        if t_off.size == 0 or not np.all(np.isfinite(t_off)) or np.any(t_off < 0):
            raise ValueError("t_off_s must contain non-negative finite trap-off times.")
        calibration = s.require_calibration(require_thresholds=True)
        if not callable(getattr(pulse, "frame_sequence", None)):
            raise TypeError("pulse must be a PulseController returned by exp.timing.bind_pulse(...) or na.bind_pulse(...).")
        if isinstance(getattr(pulse, "pulse", None), PulseTableState) and pulse.pulse.primary_time_slot() is None:
            raise ValueError(
                "release-recapture pulse has no duration scan slot for t_off; bind the trap-off "
                "period via the GUI scan dot or build it with operations.temperature.build_release_recapture_pulse(...)."
            )
        reducer = SurvivalReducer(per_site=per_site)
        if per_site:
            reducer.bind_calibration(calibration)
        return self._build_slot_scan(
            pulse, calibration, slot="s0", values=t_off, label="Trap-off time",
            plan=ReleaseRecapturePlan(), reducer=reducer, shots_per_point=positive_int(shots, "shots"),
        )

    def temperature(
        self,
        t_off_s: Sequence[float] | None = None,
        *,
        pulse: Any,
        shots: int = 1,
        per_site: bool = False,
        live: bool = True,
        update_time: float = 0.05,
        display: bool = True,
    ) -> ScanResult:
        """Release-recapture thermometry: sweep trap-off time, report survival.

        Drives the bound release-recapture ``pulse`` (its trap-off period bound to
        the primary time slot) over ``t_off_s`` (seconds), acquiring two frames per
        point and reducing them to survival via ``calibration.detect``.  Requires a
        thresholded calibration (survival is a binary occupancy comparison).
        Returns a :class:`ScanResult`; post-process with
        :func:`operations.temperature.fit_temperature` to recover the temperature."""

        s = self._session
        if t_off_s is None:
            t_off_s = s.defaults.get(
                "release_recapture_t_off",
                np.array([0.0, 5e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4]),
            )
        scan = self.build_temperature_scan(t_off_s, pulse=pulse, shots=shots, per_site=per_site)
        result = scan.run(
            live=live, update_time=update_time, display=display,
            stop_hint="Live release-recapture started. Call scan.stop() to stop measurement and plot.",
        )
        s.history.append(result)
        return result

    # --------------------------------------------------- declarative spec catalog
    def measurement_specs(self) -> list[MeasurementSpec]:
        """Auto-discovered catalog of the measurements a GUI can drive.

        The catalog is NOT a hardcoded list -- it is assembled by the open
        measurement registry: every ``@measurement`` factory in
        ``operations/measurements/`` (the built-ins live there) plus anything a
        notebook registered via ``register_measurement(...)``.  Drop a new module
        into that package and it shows up here (and in the console's Add Panel)
        automatically.  Each factory receives THIS subsystem, so its
        ``build(**param_values)`` closure captures the session and routes through
        the SAME builders the API uses (:meth:`build_temperature_scan` /
        :meth:`build_detection_scan`) -- GUI and notebook one-liner cannot drift.
        Each :class:`MeasurementSpec` declares its parameters ONCE (the single
        source of truth a GUI form and the API default both derive from)."""

        from ..operations.measurement_registry import discovered_specs

        return discovered_specs(self)

    def processor_specs(self) -> list:
        """Auto-discovered catalog of one-shot DATA-PROCESSING actions a GUI can run.

        The discrete sibling of :meth:`measurement_specs`: every ``@processor``
        factory in ``operations/processors/`` (the built-in per-site fidelity
        characterization lives there) plus anything a notebook registered via
        ``register_processor(...)``.  Each factory receives THIS subsystem, so its
        ``run(ctx)`` DRIVES the subsystem's own analysis (e.g.
        :meth:`characterize_from_dir`) rather than re-implementing it, and returns a
        ``{signal_name: value}`` dict the host publishes to the SignalHub.  Drop a
        new module into that package and it appears here (and in the console's Add
        Panel "Data processing" group) automatically."""

        from ..operations.processor_registry import discovered_processor_specs

        return discovered_processor_specs(self)

    def task_specs(self) -> list:
        """Auto-discovered catalog of orchestration TASKS a GUI can run.

        The orchestration tier of :meth:`measurement_specs` / :meth:`processor_specs`:
        every ``@task`` factory in ``operations/tasks/`` (the built-in readout
        calibration lives there) plus anything a notebook registered via
        ``register_task(...)``.  Each factory receives THIS subsystem, so its
        ``build(hub)`` closure captures the session (camera / sequencer) and returns
        an UNRUN :class:`~..operations.logic.Task` that streams mid-run output to a
        dedicated panel and derives an artifact.  Drop a new module into that package
        and it appears here (and in the console's Add-Panel "Task" group)
        automatically."""

        from ..operations.task_registry import discovered_task_specs

        return discovered_task_specs(self)

    def camera_measurement(self, hub, *, prefix: str = "", frames_per_cycle: int = 1, repeat: int = 0):
        """Build a CONTINUOUS camera Measurement over the session camera (a
        :class:`~..operations.logic.CameraMeasurement` publishing raw ``frame``s).

        This is the standalone live-image logic node -- one of the primitives a
        notebook or the task console composes the loading readout from by wiring
        it alongside a :class:`~..operations.logic.OccupancyProcessor` and a
        :class:`~..operations.logic.CalibrateReadoutTask`.  Its editable
        parameters ARE the camera's (exposure / region), applied live; only the
        camera differs on real hardware (virtual == real).

        ``repeat`` is the acquisition knob (0 = ∞, the live monitor; K = keep & average a K-deep frame
        block then STOP) -- a FIRST-CLASS build parameter so EVERY entry point (notebook + GUI) applies
        it the SAME way (a plot then collapses the repeat axis via repeat_mode = average / add / ...).
        It is passed straight to :class:`CameraMeasurement` (whose ``set_repeat`` is the single source);
        a notebook calling ``readout.camera_spec().build(hub, repeat=50)`` gets a 50-deep ring exactly
        like the console's Repeat=50 form does -- they must never diverge (virtual == real == notebook)."""

        from ..operations.logic import CameraMeasurement

        s = self._session
        return CameraMeasurement(hub, s.devices.camera, sequencer=getattr(s.devices, "sequencer", None),
                                 frames_per_cycle=frames_per_cycle, prefix=prefix, repeat=repeat)

    def camera_spec(self) -> MeasurementSpec:
        """The camera live stream as a DECLARATIVE spec, so a GUI auto-form renders
        its editable parameters and the API builds it from the SAME declaration
        (single source of truth -- the parameters live on the logic side; the UI
        adapts to them, never the other way round).

        ``frames_per_cycle`` is a build parameter; ``exposure`` and ``region`` (the
        camera ROI) are applied LIVE to the camera after the node is built (they are
        acquisition parameters the running stream picks up).  These are the camera's
        OWN settings, so the form is identical for a real camera or a ``VirtualCamera``
        -- both honour ``camera.configure(exposure=, roi=)`` (only the data source
        differs).  ``build(hub, **values)`` returns the same
        :class:`~..operations.logic.CameraMeasurement` as :meth:`camera_measurement`
        -- so notebook (``readout.camera_spec().build(hub, exposure=, region=)``) and
        GUI share one path."""

        s = self._session
        exposure = float(s._camera_exposure())

        def _parse_region(text):
            """``"x0, x1, y0, y1"`` (sensor-pixel ENDPOINTS) -> a 4-list, or None for
            the full sensor (blank).  The endpoints<->device-ROI conversion lives in
            CameraMeasurement.set_acquisition_parameters (one source of truth)."""
            t = str(text).strip()
            if not t or t.lower() == "none":
                return None
            parts = [p for p in t.replace(";", ",").split(",") if p.strip() != ""]
            if len(parts) != 4:
                raise ValueError("region must be 'x0, x1, y0, y1' (4 numbers) or blank for full sensor.")
            return [float(p) for p in parts]

        def _build(hub, *, frames_per_cycle: int = 1, exposure: float = exposure,
                   region: str = "", repeat: int = 0, **_ignored):
            node = self.camera_measurement(hub, frames_per_cycle=int(frames_per_cycle),
                                           repeat=max(0, int(repeat)))
            apply: dict[str, object] = {}
            if exposure is not None:
                apply["exposure"] = float(exposure)
            # Always push ``region`` (None for blank) so a build matches the Edit
            # path: blank -> region=None -> roi=None -> FULL sensor, never a stale ROI
            # left on the camera by a previous node.  Endpoints<->ROI conversion is in
            # CameraMeasurement.set_acquisition_parameters (one source of truth).
            apply["region"] = _parse_region(region)
            if apply and hasattr(node, "apply_acquisition_parameters"):
                node.apply_acquisition_parameters(**apply)
            return node

        return MeasurementSpec(
            name="Camera (live frames)",
            params=(
                ParamDecl(key="frames_per_cycle", label="frames / cycle", kind="int",
                          default=1, lo=1, hi=10000,
                          tooltip="emCCD events (camera triggers) per cycle.  Each event i publishes its "
                                  "OWN signal frame_i = a (repeat,1,H,W) block, so you can apply repeat_mode "
                                  "to a chosen event (e.g. average just frame_1 of a 3-event cycle)."),
                ParamDecl(key="exposure", label="exposure", kind="float",
                          default=exposure, unit="s", lo=0.0, hi=100.0,
                          tooltip="Camera exposure time (applied live to the camera)."),
                ParamDecl(key="region", label="region x0,x1,y0,y1", kind="text", default="",
                          tooltip="Camera ROI as sensor-pixel endpoints x0,x1,y0,y1 (blank = full "
                                  "sensor).  Or drag a box on a frame plot to set it -- both go to "
                                  "camera.configure(roi=), identical for virtual / real."),
            ),
            result_labels=("shot", "frame_0"),
            x_key="frame_0", y_key="frame_0",
            build=_build,
        )

    def calibrate_task(self, hub, *, prefix: str = "cal_", **params):
        """Build the calibrate-readout TASK over the session camera (the sitemap +
        per-site threshold calibration as a first-class one-shot
        :class:`~..operations.logic.CalibrateReadoutTask`).

        Run it from the console (Add Panel -> "Task: Calibrate readout"): mid-run it
        streams its template frames + a progress fraction to a dedicated panel
        (``<prefix>frame`` / ``<prefix>progress``, confocal-task style) and produces a
        ``TrapCalibration`` (+ optional npz artifact) that an OccupancyProcessor then
        consumes.  Same calibration primitives the notebook/real readout uses --
        only the camera frames differ."""

        from ..operations.logic import CalibrateReadoutTask

        s = self._session
        params["grid_shape"] = s._grid_shape(params.get("grid_shape"))

        def _adopt(calibration):
            # The finished calibration BECOMES the session calibration (``readout.current``),
            # so a decoupled live OccupancyProcessor begins judging sites the moment this
            # task completes -- no path to type, the cali->occupancy connection.
            s._calibration = calibration
            # Pin the LIVE readout exposure to the gate the thresholds were learnt at
            # (``threshold_exposure``): the live OccupancyProcessor judges the RUNNING pulse's frames,
            # and a per-site threshold is exposure-specific -- calibrating at the short readout gate but
            # imaging live at the camera default classifies every atom dark (~50 % occupancy while the
            # report says 99 %).  Setting the camera exposure to the calibrated gate makes the default
            # live readout self-match (#issue-2 live path); a frame still imaged at a different gate is
            # the operator's explicit choice.  threshold_exposure is stamped by calibrate_threshold_from_images.
            # fallback=None: an UNstamped calibration does not force a re-pin (leave the camera as-is).
            expo = calibration.readout_exposure(fallback=None)
            if expo:
                try:
                    s.devices.camera.exposure = float(expo)
                except Exception:
                    pass

        return CalibrateReadoutTask(hub, s.devices.camera,
                                    sequencer=getattr(s.devices, "sequencer", None),
                                    calibration_sink=_adopt, prefix=prefix, **params)

    # ------------------------------------------------------------- persistence
    def load(self, path: str | Path) -> TrapCalibration:
        return self._session.load_calibration(path)

    def save(self, path: str | Path) -> Path:
        return self._session.save_calibration(path)

    def clear(self) -> None:
        self._session._calibration = None


class _SessionImagingController:
    """PulseController-shaped adapter over the session's imaging sequence.

    Lets the detection-time scan run through the SAME ``ScannedMeasurement``
    engine whether or not a GUI pulse is bound.  ``frame_sequence`` returns the
    session's single-trigger imaging sequence at the requested exposure (the
    camera repeats it per frame); ``set_time``/``set_slot`` are no-ops because the
    exposure is carried by the sequence itself, not a scan slot."""

    def __init__(self, session, *, name: str):
        self._session = session
        self._name = str(name)
        self.sequencer = getattr(session.devices, "sequencer", None)
        self.pulse = None

    def set_time(self, value_ns: float) -> "_SessionImagingController":
        return self

    def set_slot(self, key, value: float) -> "_SessionImagingController":
        return self

    def frame_sequence(self, frames: int, *, time_ns: float | None = None, slots=None, trigger_channels=None):
        exposure = self._session._camera_exposure() if time_ns is None else float(time_ns) * 1e-9
        return self._session._imaging_sequence(exposure=exposure, load=True, name=self._name)


class _ExposureConfiguringPlan:
    """Wrap an N-frame plan to set the real camera exposure before each acquire.

    The legacy detection-time scan called ``camera.configure(exposure=t)`` ahead
    of every bound-pulse acquire so a real camera's integration window tracks the
    scanned probe duration (the virtual camera derives exposure from the sequence
    and ignores it -- harmless).  ``sequence_for`` runs immediately before
    ``acquire`` in the engine, so configuring here keeps the prior ordering."""

    def __init__(self, inner: NFramePlan, camera):
        self.inner = inner
        self._camera = camera

    @property
    def n_frames(self) -> int:
        return self.inner.n_frames

    def sequence_for(self, pulse, axis: ScanAxis, value: float):
        configure = getattr(self._camera, "configure", None)
        if callable(configure):
            configure(exposure=float(value))
        return self.inner.sequence_for(pulse, axis, value)


__all__ = ["ReadoutSubsystem"]
