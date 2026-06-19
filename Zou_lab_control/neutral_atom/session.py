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

from .core.analysis import grid_shape_tuple
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
from .core.utils import html_summary, json_ready
from .devices import CameraDevice, DeviceSet, SequencerDevice, load_devices, resolve_connect_config
from .operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from .timing import PulseSequence, imaging_channel_kwargs, imaging_sequence
from .timing.verilog import generate_verilog, write_verilog_bundle
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

    # ---- GUI launchers (confocal-style ``exp.task_console()`` / ``exp.pulse_gui()``) --------
    # The windows live in the frontend; these are thin sugar reached LAZILY through the
    # GUI-action module ``_gui`` (which imports the frontend only when a window is opened), so
    # the analysis path (connect / sitemap / thresholds / detect) never pulls the frontend and
    # virtual==real stays headless.  ``session.py`` references only ``_gui`` -- never the
    # frontend itself -- keeping the one-directional neutral_atom -> frontend seal.
    def task_console(self, *, task: str | None = None, **kwargs):
        """Open the Task console GUI bound to this session.

        Sugar over ``frontend.show_task_console``: fills the hub + the auto-discovered
        measurement / processor / task catalogs from this session.  ``task`` loads a saved
        layout (``tasks/<name>.json``)."""
        from ._gui import open_task_console
        return open_task_console(self, task=task, **kwargs)

    def pulse_gui(self, *, state=None, **kwargs):
        """Open the pulse-sequence editor GUI bound to this session, so a measurement can read
        the edited program back.  To run the editor WITHOUT a session (it picks its own server
        connection, needing no experiment) call ``Zou_lab_control.frontend.show_pulse_gui()``
        directly."""
        from ._gui import open_pulse_gui
        return open_pulse_gui(self, state=state, **kwargs)

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
        # Single source of truth lives in timing.imaging_channel_kwargs so the
        # session and the loading readout map channels identically (see M4 / logic nodes).
        return imaging_channel_kwargs(getattr(self.devices, "sequencer", None))

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
    # The device layer (registry) owns backend dispatch + any backend-specific
    # config shortcuts; the session never imports a concrete backend or reads its
    # internal fields, so virtual <-> real is a one-line `config` change.
    device_config, device_overrides, inferred_defaults = resolve_connect_config(
        config,
        trap_array=trap_array,
        sitemap=sitemap,
        camera=camera,
        sequencer=sequencer,
        params=virtual_params,
    )
    default_values.update(inferred_defaults)
    return NeutralAtomSession(
        load_devices(device_config, overrides=device_overrides, open_devices=open_devices),
        name=name,
        defaults=default_values,
    )


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
