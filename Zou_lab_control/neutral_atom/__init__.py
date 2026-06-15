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
from .core.bimodal import BimodalFit, fit_bimodal, fit_bimodal_per_site, gaussian_fidelity
from .core.calibration import TrapCalibration
from .core.psf import SitePSF, fit_site_psfs, psf_signals
from .devices import DEFAULT_DCAM_MODULE, QCMOSCamera, QCMOSConfig
from .devices import (
    BaseDevice,
    CameraDevice,
    DeviceSet,
    ManualSequencer,
    PulseController,
    RemoteSequencer,
    RuntimeSequenceProgram,
    RuntimeSequencer,
    SequencerDevice,
    SequencerService,
    TrapArrayDevice,
    apply_device_overrides,
    available_device_configs,
    bind_pulse,
    compile_pulse_table_scan_runtime_program,
    compile_pulse_table_runtime_program,
    compile_runtime_program,
    compile_runtime_program_for_payload,
    device_class_registry,
    device_config_dir,
    finite_frame_sequence,
    infer_xdc_channel_pins,
    infer_xdc_trigger_channels,
    load_devices,
    register_device_class,
    serve_runtime_sequencer,
    validate_device_contract,
)
from .views import image_to_points, plot_detection_image, plot_detection_scan, plot_image, plot_site_values, plot_threshold_hist
from .devices import VerilogSequencer
from .timing import (
    ANALOG_BUS_MODES,
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    Pulse,
    PulsePeriod,
    PulseSequence,
    PulseTableState,
    count_trigger_pulses,
    default_pulse_name,
    exposure_from_sequence,
    imaging_sequence,
    infer_bus_channels,
    plot_sequence,
    positive_time_step_ns,
    quantized_time_ns,
    quantized_time_steps,
    sequence_for_frame_count,
)
from .timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle
from .devices import DEFAULT_CHANNELS, VirtualCamera, VirtualSequencer, VirtualTrapArray, virtual_config, write_virtual_run
from .operations import (
    CameraFrameFeed,
    ExperimentFeed,
    LoadingFeed,
    MeasurementSpec,
    NFramePlan,
    OtsuFidelityReducer,
    ParamDecl,
    ReleaseRecapturePlan,
    RunIndex,
    ScanAxis,
    ScanResult,
    ScannedMeasurement,
    ScannedMeasurementFeed,
    SurvivalReducer,
    TemperatureFit,
    axis_range_tuple,
    build_release_recapture_pulse,
    fit_temperature,
    frame_files,
    index_run,
    load_frame,
    measurement,
    register_measurement,
    registered_measurements,
    release_recapture_survival,
    save_frame,
    unregister_measurement,
)
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


_PULSE_STREAMER_EXPORTS = {
    "DEFAULT_FPGA_CHANNEL_COUNT",
    "DEFAULT_MAX_SCAN_POINTS",
    "DEFAULT_SCAN_COEFF_FRAC_BITS",
    "DEFAULT_SCAN_COEFF_WIDTH",
    "hardware_channel_names",
    "infer_xdc_channel_count",
    "infer_xdc_channel_labels",
    "infer_xdc_channel_pins",
    "infer_xdc_channels",
    "infer_xdc_trigger_channels",
    "validate_pulse_streamer_program",
    "capacity_estimate_text",
}


def __getattr__(name: str):
    if name == "CommandSequencerBackend":
        from .devices.sequencer_server import CommandSequencerBackend

        return CommandSequencerBackend
    if name in _PULSE_STREAMER_EXPORTS:
        from .devices import fpga_pulse_streamer

        return getattr(fpga_pulse_streamer, name)
    raise AttributeError(name)

try:
    from .notes import build_fpga_manual, build_main_manual
except Exception:  # pragma: no cover - notes import should not block experiments
    build_fpga_manual = None
    build_main_manual = None


__all__ = [
    "AtomDetection",
    "BaseDevice",
    "BimodalFit",
    "CaptureResult",
    "CameraDevice",
    "CommandSequencerBackend",
    "DEFAULT_CHANNELS",
    "DEFAULT_DCAM_MODULE",
    "DEFAULT_FPGA_CHANNEL_COUNT",
    "DEFAULT_MAX_SCAN_POINTS",
    "DEFAULT_SCAN_COEFF_FRAC_BITS",
    "DEFAULT_SCAN_COEFF_WIDTH",
    "DetectionResult",
    "DetectionTimeScanResult",
    "DeviceSet",
    "CameraFrameFeed",
    "ExperimentFeed",
    "ExperimentSubsystem",
    "FidelityEstimate",
    "LoadingFeed",
    "MeasurementSpec",
    "ManualSequencer",
    "MeasurementTaskResult",
    "NeutralAtomSession",
    "ANALOG_BUS_MODES",
    "PreflightReport",
    "Pulse",
    "PulseController",
    "PulsePeriod",
    "PulseSequence",
    "PulseTableState",
    "QCMOSCamera",
    "QCMOSConfig",
    "ReadoutSubsystem",
    "RemoteSequencer",
    "ResultObject",
    "RuntimeSequenceProgram",
    "RuntimeSequencer",
    "SequencerDevice",
    "SequencerService",
    "SitePSF",
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
    "bind_pulse",
    "build_fpga_manual",
    "build_main_manual",
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "capacity_estimate_text",
    "compile_runtime_program",
    "compile_pulse_table_runtime_program",
    "compile_pulse_table_scan_runtime_program",
    "compile_runtime_program_for_payload",
    "connect",
    "count_trigger_pulses",
    "default_pulse_name",
    "detect_image",
    "detect_atoms",
    "device_class_registry",
    "device_config_dir",
    "estimate_threshold_fidelity",
    "estimate_thresholds",
    "exposure_from_sequence",
    "find_site_centers",
    "finite_frame_sequence",
    "fit_bimodal",
    "fit_bimodal_per_site",
    "fit_site_psfs",
    "gaussian_fidelity",
    "generate_verilog",
    "hardware_channel_names",
    "image_to_points",
    "imaging_sequence",
    "infer_bus_channels",
    "infer_xdc_channel_count",
    "infer_xdc_channel_labels",
    "infer_xdc_channels",
    "load_devices",
    "otsu_threshold",
    "plot_detection_image",
    "plot_detection_scan",
    "plot_image",
    "plot_sequence",
    "plot_site_values",
    "plot_threshold_hist",
    "positive_time_step_ns",
    "psf_signals",
    "quantized_time_ns",
    "quantized_time_steps",
    "roi_counts",
    "register_device_class",
    "run_sequencer_server",
    "serve_runtime_sequencer",
    "sequence_for_frame_count",
    "sort_centers_grid",
    "virtual_config",
    "write_virtual_run",
    "index_run",
    "load_frame",
    "save_frame",
    "frame_files",
    "RunIndex",
    "build_release_recapture_pulse",
    "fit_temperature",
    "release_recapture_survival",
    "axis_range_tuple",
    "measurement",
    "register_measurement",
    "registered_measurements",
    "unregister_measurement",
    "NFramePlan",
    "OtsuFidelityReducer",
    "ParamDecl",
    "ReleaseRecapturePlan",
    "ScanAxis",
    "ScannedMeasurement",
    "ScannedMeasurementFeed",
    "ScanResult",
    "SurvivalReducer",
    "TemperatureFit",
    "validate_pulse_streamer_program",
    "validate_device_contract",
    "write_verilog_bundle",
]
