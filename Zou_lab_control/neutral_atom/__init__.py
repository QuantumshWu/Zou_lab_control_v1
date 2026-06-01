"""Lightweight neutral-atom control session for Jupyter.

First milestone:
    connect devices, capture camera images, calibrate site map, calibrate
    thresholds, detect occupancy, and scan detection time/fidelity.

The public notebook entry point is ``connect``.  Lower-level device/timing/
Verilog helpers remain available so the session can grow toward real
hardware and GUI integration without changing the notebook shape.
"""

from .core.analysis import (
    AtomDetection,
    FidelityEstimate,
    detect_atoms,
    estimate_threshold_fidelity,
    estimate_thresholds,
    find_site_centers,
    otsu_threshold,
    roi_counts,
    sort_centers_grid,
)
from .core.calibration import TrapCalibration
from .devices import QCMOSCamera, QCMOSConfig
from .devices import (
    BaseDevice,
    CameraDevice,
    DeviceSet,
    ManualSequencer,
    RemoteSequencer,
    RuntimeSequenceProgram,
    RuntimeSequencer,
    SequencerDevice,
    SequencerService,
    TrapArrayDevice,
    apply_device_overrides,
    available_device_configs,
    compile_runtime_program,
    device_config_dir,
    load_devices,
    serve_runtime_sequencer,
    validate_device_contract,
)
from .views import image_to_points, plot_detection_image, plot_detection_scan, plot_image, plot_site_values, plot_threshold_hist
from .devices import VerilogSequencer
from .timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    Pulse,
    PulseSequence,
    count_trigger_pulses,
    exposure_from_sequence,
    imaging_sequence,
    plot_sequence,
    sequence_for_frame_count,
)
from .timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle
from .devices import DEFAULT_CHANNELS, VirtualCamera, VirtualSequencer, VirtualTrapArray, virtual_config
from .session import (
    CaptureResult,
    DetectionResult,
    DetectionTimeScanResult,
    ExperimentSubsystem,
    MeasurementTaskResult,
    NeutralAtomSession,
    PreflightReport,
    ReadoutSubsystem,
    ResultObject,
    SitemapResult,
    ThresholdResult,
    TimingSubsystem,
    calibrate_sitemap_from_images,
    calibrate_threshold_from_images,
    connect,
    detect_image,
)


def run_sequencer_server(*args, **kwargs):
    """Start the FPGA/Vivado-computer sequencer server."""

    from .devices.sequencer_server import run_server

    return run_server(*args, **kwargs)


def __getattr__(name: str):
    if name == "CommandSequencerBackend":
        from .devices.sequencer_server import CommandSequencerBackend

        return CommandSequencerBackend
    raise AttributeError(name)

try:
    from .notes import build_neutral_atom_hardware_manual, build_neutral_atom_manual
except Exception:  # pragma: no cover - notes import should not block experiments
    build_neutral_atom_hardware_manual = None
    build_neutral_atom_manual = None


__all__ = [
    "AtomDetection",
    "BaseDevice",
    "CaptureResult",
    "CameraDevice",
    "CommandSequencerBackend",
    "DEFAULT_CHANNELS",
    "DetectionResult",
    "DetectionTimeScanResult",
    "DeviceSet",
    "ExperimentSubsystem",
    "FidelityEstimate",
    "ManualSequencer",
    "MeasurementTaskResult",
    "NeutralAtomSession",
    "PreflightReport",
    "Pulse",
    "PulseSequence",
    "QCMOSCamera",
    "QCMOSConfig",
    "ReadoutSubsystem",
    "RemoteSequencer",
    "ResultObject",
    "RuntimeSequenceProgram",
    "RuntimeSequencer",
    "SequencerDevice",
    "SequencerService",
    "SitemapResult",
    "ThresholdResult",
    "TimingSubsystem",
    "TrapArrayDevice",
    "TrapCalibration",
    "VerilogBuild",
    "VerilogFiles",
    "VerilogSequencer",
    "VirtualCamera",
    "VirtualSequencer",
    "VirtualTrapArray",
    "apply_device_overrides",
    "available_device_configs",
    "build_neutral_atom_manual",
    "build_neutral_atom_hardware_manual",
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "compile_runtime_program",
    "connect",
    "count_trigger_pulses",
    "detect_image",
    "detect_atoms",
    "device_config_dir",
    "estimate_threshold_fidelity",
    "estimate_thresholds",
    "exposure_from_sequence",
    "find_site_centers",
    "generate_verilog",
    "image_to_points",
    "imaging_sequence",
    "load_devices",
    "otsu_threshold",
    "plot_detection_image",
    "plot_detection_scan",
    "plot_image",
    "plot_sequence",
    "plot_site_values",
    "plot_threshold_hist",
    "roi_counts",
    "run_sequencer_server",
    "serve_runtime_sequencer",
    "sequence_for_frame_count",
    "sort_centers_grid",
    "virtual_config",
    "validate_device_contract",
    "write_verilog_bundle",
]
