"""Jupyter-first neutral-atom experiment session.

This file is the public shape of the lightweight first milestone.  The lower
layers still exist as device/timing/analysis/verilog boundaries; the notebook
user mostly talks to ``NeutralAtomSession`` and result objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

from .core.analysis import (
    estimate_threshold_fidelity,
    grid_shape_tuple,
    otsu_threshold,
    positive_int,
    roi_counts,
)
from .core.calibration import TrapCalibration
from .core.results import (
    CaptureResult,
    DetectionResult,
    DetectionTimeScanResult,
    MeasurementTaskResult,
    PreflightReport,
    ResultObject,
    SitemapResult,
    ThresholdResult,
)
from .core.utils import html_summary, json_ready, site_index
from .devices import CameraDevice, DeviceSet, SequencerDevice, load_devices
from .devices.virtual import VirtualTrapArray, virtual_config
from .operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from .timing import PulseSequence, imaging_sequence
from .timing.verilog import generate_verilog, write_verilog_bundle
from .views.plots import plot_detection_scan
from .subsystems import ExperimentSubsystem, ReadoutSubsystem, TimingSubsystem


class NeutralAtomSession:
    """Notebook-facing experiment session.

    The session is deliberately stateful: it owns the currently connected
    devices, the current pulse sequence, the current calibration, and recent
    results.  This keeps notebook cells short while preserving a clean path to
    real hardware and future GUI frontends.
    """

    def __init__(self, devices: DeviceSet, *, name: str = "neutral_atom", defaults: dict[str, Any] | None = None):
        self.devices = devices
        self.name = str(name)
        self.defaults = dict(defaults or {})
        self.sequence = self._imaging_sequence(exposure=self._camera_exposure(), load=True)
        self._calibration: TrapCalibration | None = None
        self.history: list[Any] = []
        self.devices.camera.bind_experiment(self)
        self._readout_subsystem = ReadoutSubsystem(self)
        self._timing_subsystem = TimingSubsystem(self)

    @property
    def camera(self) -> CameraDevice:
        return self.devices.camera

    @property
    def sequencer(self) -> SequencerDevice:
        return self.devices.sequencer

    @property
    def readout(self) -> ReadoutSubsystem:
        if not hasattr(self, "_readout_subsystem"):
            self._readout_subsystem = ReadoutSubsystem(self)
        return self._readout_subsystem

    @property
    def calibration_data(self) -> TrapCalibration | None:
        return self._calibration

    @property
    def timing(self) -> TimingSubsystem:
        if not hasattr(self, "_timing_subsystem"):
            self._timing_subsystem = TimingSubsystem(self)
        return self._timing_subsystem

    def _configure_imaging(self, *, exposure: float | None = None, load: bool = True, trigger_width: float = 20e-6, pre_trigger: float = 100e-6) -> PulseSequence:
        if exposure is not None and hasattr(self.devices.camera, "configure"):
            self.devices.camera.configure(exposure=exposure)
        self.sequence = self._imaging_sequence(
            exposure=self._camera_exposure() if exposure is None else exposure,
            trigger_width=trigger_width,
            pre_trigger=pre_trigger,
            load=load,
        )
        return self.sequence

    def _imaging_sequence(self, **kwargs) -> PulseSequence:
        return imaging_sequence(**kwargs, **self._imaging_channel_kwargs())

    def _imaging_channel_kwargs(self) -> dict[str, str]:
        sequencer = getattr(self.devices, "sequencer", None)
        channels = list(getattr(sequencer, "channels", ()))
        trigger_channels = list(getattr(sequencer, "trigger_channels", ()))
        if all(channel in channels for channel in ("ch00", "ch03", "ch09")) and trigger_channels and trigger_channels[0] in channels:
            return {
                "trap_channel": "ch09",
                "cooling_channel": "ch00",
                "probe_channel": "ch03",
                "trigger_channel": trigger_channels[0],
            }
        if all(channel in channels for channel in ("trap", "cooling", "probe", "emCCD")):
            return {
                "trap_channel": "trap",
                "cooling_channel": "cooling",
                "probe_channel": "probe",
                "trigger_channel": "emCCD",
            }
        return {}

    def _calibrate_sitemap(
        self,
        *,
        frames: int = 20,
        grid_shape: Sequence[int] | None = None,
        ordering: str = "row-major",
        roi_radius: int | None = None,
        reducer: str = "mean",
        display: bool = True,
    ) -> SitemapResult:
        grid_shape = self._grid_shape(grid_shape)
        roi_radius = int(self.defaults.get("roi_radius", 1) if roi_radius is None else roi_radius)
        exposure = self.defaults.get("sitemap_exposure", self._camera_exposure())
        sequence = self._imaging_sequence(exposure=exposure, load=True, name="sitemap")
        images = self.devices.camera.acquire(positive_int(frames, "frames"), sequence=sequence, sequencer=getattr(self.devices, "sequencer", None))
        result = calibrate_sitemap_from_images(
            images,
            grid_shape=grid_shape,
            ordering=ordering,
            roi_radius=roi_radius,
            reducer=reducer,
            display=display,
        )
        self._calibration = result.calibration
        self.history.append(result)
        return result

    def _calibrate_threshold(
        self,
        *,
        frames: int = 100,
        site: int = 0,
        exposure: float | None = None,
        display: bool = True,
    ) -> ThresholdResult:
        calibration = self.require_calibration(require_thresholds=False)
        sequence = self._imaging_sequence(exposure=self._camera_exposure() if exposure is None else exposure, load=True, name="threshold")
        images = self.devices.camera.acquire(positive_int(frames, "frames"), sequence=sequence, sequencer=getattr(self.devices, "sequencer", None))
        result = calibrate_threshold_from_images(images, calibration, site=site, display=display)
        self._calibration = result.calibration
        self.history.append(result)
        return result

    def _detect(self, *, exposure: float | None = None, display: bool = True, what: str = "occupancy") -> DetectionResult:
        calibration = self.require_calibration(require_thresholds=True)
        sequence = self._imaging_sequence(exposure=self._camera_exposure() if exposure is None else exposure, load=True, name="detect")
        images = self.devices.camera.acquire(1, sequence=sequence, sequencer=getattr(self.devices, "sequencer", None))
        result = detect_image(images[-1], calibration, sequence=sequence, display=display, what=what)
        self.history.append(result)
        return result

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
        calibration = self.require_calibration(require_thresholds=False)
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("times must contain positive finite detection times.")
        shots = positive_int(shots, "shots")
        reference_shots = positive_int(reference_shots, "reference_shots")
        data_y = np.full((len(times), 1), np.nan, dtype=float)
        reference_exposure = float(max(np.nanmax(times) * 3.0, self._camera_exposure()) if reference_exposure is None else reference_exposure)
        if not np.isfinite(reference_exposure) or reference_exposure <= 0:
            raise ValueError("reference_exposure must be positive and finite.")

        if pulse is None:
            reference_sequence = self._imaging_sequence(exposure=reference_exposure, load=True, name="reference_threshold")
            reference_sequencer = getattr(self.devices, "sequencer", None)
        else:
            reference_x_ns = float(reference_exposure) * 1e9
            configure = getattr(self.devices.camera, "configure", None)
            if callable(configure):
                configure(exposure=float(reference_exposure))
            frame_sequence = getattr(pulse, "frame_sequence", None)
            if not callable(frame_sequence):
                raise TypeError("pulse must be a PulseController returned by exp.timing.bind_pulse(...) or na.bind_pulse(...).")
            reference_sequence = frame_sequence(reference_shots, time_ns=reference_x_ns)
            reference_sequencer = getattr(pulse, "sequencer", getattr(self.devices, "sequencer", None))
        reference_images = self.devices.camera.acquire(
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
                sequence = self._imaging_sequence(exposure=float(time_s), load=True, name="detect_time_scan")
                sequencer = getattr(self.devices, "sequencer", None)
            else:
                x_ns = float(time_s) * 1e9
                configure = getattr(self.devices.camera, "configure", None)
                if callable(configure):
                    configure(exposure=float(time_s))
                frame_sequence = getattr(pulse, "frame_sequence", None)
                if not callable(frame_sequence):
                    raise TypeError("pulse must be a PulseController returned by exp.timing.bind_pulse(...) or na.bind_pulse(...).")
                sequence = frame_sequence(shots, time_ns=x_ns)
                sequencer = getattr(pulse, "sequencer", getattr(self.devices, "sequencer", None))
            images = self.devices.camera.acquire(shots, sequence=sequence, sequencer=sequencer)
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

        if live:
            from Zou_lab_control import frontend as zf

            result.measurement = zf.run(
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
            for index, time_s in enumerate(times):
                data_y[index, 0] = measure(float(time_s), index)
            result.plot = plot_detection_scan(times, data_y[:, 0], display=display)
        self.history.append(result)
        return result

    def _preflight(self, *, sequence: PulseSequence | None = None, verilog: bool = True) -> PreflightReport:
        sequence = sequence or self.sequence
        errors: list[str] = []
        warnings: list[str] = []
        sequencer = getattr(self.devices, "sequencer", None)
        clock = getattr(sequencer, "clock_hz", 50_000_000.0)
        channels = getattr(sequencer, "channels", None)
        pulse_report = sequence.validate(clock_hz=clock, channels=channels)
        errors.extend(pulse_report.errors)
        warnings.extend(pulse_report.warnings)
        build = None
        if verilog and channels is not None:
            try:
                build = generate_verilog(sequence, channels=channels, clock_hz=clock, module_name="preflight_sequence")
            except Exception as exc:
                errors.append(str(exc))
        return PreflightReport(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            sequence_table=sequence.table(),
            device_snapshot=self.devices.snapshot(),
            verilog=build,
        )

    def _write_verilog(self, output_dir: str | Path = "generated_sequences", *, sequence: PulseSequence | None = None) -> Path:
        sequence = sequence or self.sequence
        sequencer = getattr(self.devices, "sequencer", None)
        channels = getattr(sequencer, "channels", sequence.channels)
        clock = getattr(sequencer, "clock_hz", 50_000_000.0)
        pin_map = getattr(sequencer, "pin_map", None)
        build = generate_verilog(sequence, channels=channels, clock_hz=clock, module_name="neutral_atom_sequence")
        files = write_verilog_bundle(build, output_dir, pin_map=pin_map)
        return files.verilog_path

    def load_calibration(self, path: str | Path) -> TrapCalibration:
        self._calibration = TrapCalibration.load(path)
        return self._calibration

    def save_calibration(self, path: str | Path) -> Path:
        return self.require_calibration(require_thresholds=False).save(path)

    def require_calibration(self, *, require_thresholds: bool = True) -> TrapCalibration:
        if self._calibration is None:
            raise RuntimeError("No calibration is loaded. Run exp.readout.sitemap() first.")
        if require_thresholds and not self._calibration.metadata.get("thresholds_calibrated", False):
            raise RuntimeError("No threshold calibration is loaded. Run exp.readout.thresholds() first.")
        return self._calibration

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "devices": self.devices.snapshot(),
            "sequence": self.sequence.table(),
            "calibration": None if self._calibration is None else self._calibration.to_dict(),
            "history_length": len(self.history),
        }

    def _repr_html_(self) -> str:
        calibration = "none" if self._calibration is None else f"{self._calibration.n_sites} sites"
        devices = ", ".join(sorted(self.devices.devices))
        return html_summary(
            "NeutralAtomSession",
            {
                "name": self.name,
                "devices": devices,
                "sequence": self.sequence.name,
                "calibration": calibration,
                "history": len(self.history),
            },
        )

    def save_status(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_ready(self.status()), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def close(self) -> None:
        self.devices.close()

    def _camera_exposure(self) -> float:
        return float(getattr(self.devices.camera, "exposure", getattr(getattr(self.devices.camera, "config", None), "exposure", 20e-3)))

    def _grid_shape(self, grid_shape: Sequence[int] | None) -> tuple[int, int]:
        if grid_shape is not None:
            return grid_shape_tuple(grid_shape)
        trap_array = getattr(self.devices, "trap_array", None)
        if trap_array is not None and hasattr(trap_array, "grid_shape"):
            return grid_shape_tuple(trap_array.grid_shape)
        raise ValueError("grid_shape is required when the device config has no trap_array.")

def connect(
    config: str | Path | dict[str, Any] = "virtual",
    *,
    name: str = "neutral_atom",
    trap_array: dict[str, Any] | None = None,
    sitemap: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    sequencer: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    open_devices: bool = False,
    **virtual_params,
) -> NeutralAtomSession:
    """Load devices and return a notebook-facing neutral-atom session."""

    default_values = dict(defaults or {})
    device_config = config
    device_overrides: dict[str, dict[str, Any]] = {}
    if _is_virtual_config(config):
        device_config, inferred_defaults = _virtual_config_with_overrides(
            trap_array=trap_array,
            sitemap=sitemap,
            camera=camera,
            sequencer=sequencer,
            params=virtual_params,
        )
        default_values.update(inferred_defaults)
    else:
        if sitemap or virtual_params:
            raise ValueError("sitemap and virtual shortcut parameters are only supported with config='virtual'.")
        for device_name, params in (("trap_array", trap_array), ("camera", camera), ("sequencer", sequencer)):
            if params:
                device_overrides[device_name] = dict(params)
    return NeutralAtomSession(
        load_devices(device_config, overrides=device_overrides or None, open_devices=open_devices),
        name=name,
        defaults=default_values,
    )


def _is_virtual_config(config) -> bool:
    return isinstance(config, str) and config.lower() == "virtual"


def _virtual_config_with_overrides(
    *,
    trap_array: dict[str, Any] | None = None,
    sitemap: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    sequencer: dict[str, Any] | None = None,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = virtual_config()
    trap_params = dict(trap_array or {})
    sitemap_params = dict(sitemap or {})
    camera_params = dict(camera or {})
    sequencer_params = dict(sequencer or {})
    defaults: dict[str, Any] = {}
    trap_fields = set(VirtualTrapArray.__dataclass_fields__)
    aliases = {
        "bright_count_rate": "atom_rate",
        "atom_bright_rate": "atom_rate",
        "background_count_rate": "background_rate",
        "load_probability": "loading_probability",
    }
    for key, value in sitemap_params.items():
        target = aliases.get(key, key)
        if target in trap_fields:
            trap_params[target] = value
        elif key in {"roi_radius", "sitemap_exposure", "detection_times"}:
            defaults[key] = value
        else:
            raise TypeError(f"unknown sitemap configuration parameter {key!r}.")
    for key, value in dict(params or {}).items():
        if key == "loss_rate":
            loss_rate = float(value)
            if not np.isfinite(loss_rate) or loss_rate <= 0:
                raise ValueError("loss_rate must be positive and finite.")
            trap_params["detection_lifetime"] = 1.0 / loss_rate
        elif key in {"sitemap_exposure", "detection_times", "roi_radius"}:
            defaults[key] = value
        else:
            target = aliases.get(key, key)
            if target in trap_fields:
                trap_params[target] = value
            elif key in {"exposure", "timeout"}:
                camera_params[key] = value
            elif key in {"clock_hz", "sleep_scale", "channels"}:
                sequencer_params[key] = value
            else:
                raise TypeError(f"unknown virtual configuration parameter {key!r}.")
    cfg["trap_array"].setdefault("params", {}).update(trap_params)
    cfg["camera"].setdefault("params", {}).update(camera_params)
    cfg["sequencer"].setdefault("params", {}).update(sequencer_params)
    return cfg, defaults


__all__ = [
    "CaptureResult",
    "DetectionResult",
    "DetectionTimeScanResult",
    "ExperimentSubsystem",
    "MeasurementTaskResult",
    "NeutralAtomSession",
    "PreflightReport",
    "ReadoutSubsystem",
    "ResultObject",
    "SitemapResult",
    "ThresholdResult",
    "TimingSubsystem",
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "connect",
    "detect_image",
]
