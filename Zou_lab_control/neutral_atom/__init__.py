"""Neutral-atom control session for Jupyter.

It connects devices (virtual or the real qCMOS + FPGA pulse streamer), captures
camera images, calibrates the site map + per-site thresholds, detects occupancy,
and runs swept measurements (detection time / fidelity / release-recapture).

The public notebook entry point is ``connect``.  The lower-level device / timing /
Verilog helpers are also exported, so a notebook can drive the hardware directly and
the GUIs (pulse editor, task console) build on the same session.
"""

from .core.analysis import (
    AtomDetection,
    FidelityEstimate,
    estimate_threshold_fidelity,
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
    Pulse,
    PulsePeriod,
    PulseSequence,
    PulseTableState,
    default_pulse_name,
    exposure_from_sequence,
    imaging_sequence,
    infer_bus_channels,
    plot_sequence,
    positive_time_step_ns,
    quantized_time_ns,
    quantized_time_steps,
)
# The camera capture-trigger helpers live in the CAMERA layer now (the sequencer/timing know
# nothing about which channel gates a camera); re-exported here so notebooks keep `na.*` access.
from .devices.camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS, count_trigger_pulses
from .timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle
from .devices import DEFAULT_CHANNELS, VirtualCamera, VirtualSequencer, VirtualTrapArray, virtual_config, write_virtual_run
from .operations import (
    CameraMeasurement,
    MeasurementSpec,
    NFramePlan,
    OtsuFidelityReducer,
    ParamDecl,
    LogicNode,
    ProcessorSpec,
    ReleaseRecapturePlan,
    RunIndex,
    ScanAxis,
    ScanResult,
    ScannedMeasurement,
    ScannedMeasurementNode,
    SurvivalReducer,
    TaskSpec,
    TemperatureFit,
    axis_range_tuple,
    build_release_recapture_pulse,
    fit_temperature,
    frame_files,
    index_run,
    load_frame,
    measurement,
    processor,
    register_measurement,
    register_processor,
    register_task,
    registered_measurements,
    registered_processors,
    registered_tasks,
    release_recapture_survival,
    save_frame,
    task,
    unregister_measurement,
    unregister_processor,
    unregister_task,
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
    if name == "load_figure":
        # Reopen a saved figure npz (``na.load_figure('scan.npz')``) -- reached LAZILY through the
        # GUI-action module so the frontend is not pulled onto the analysis import path.
        from ._gui import load_figure

        return load_figure
    if name == "figure_viewer":
        # Open the saved-figure viewer window (``na.figure_viewer('scan.npz')``) -- reached LAZILY
        # through the GUI-action module so the frontend stays off the analysis import path.
        from ._gui import open_figure_viewer

        return open_figure_viewer
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
    "CameraMeasurement",
    "LogicNode",
    "ExperimentSubsystem",
    "FidelityEstimate",
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
    "device_class_registry",
    "device_config_dir",
    "estimate_threshold_fidelity",
    "exposure_from_sequence",
    "figure_viewer",
    "find_site_centers",
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
    "load_figure",
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
    "processor",
    "register_processor",
    "registered_processors",
    "unregister_processor",
    "task",
    "register_task",
    "registered_tasks",
    "unregister_task",
    "ProcessorSpec",
    "TaskSpec",
    "NFramePlan",
    "OtsuFidelityReducer",
    "ParamDecl",
    "ReleaseRecapturePlan",
    "ScanAxis",
    "ScannedMeasurement",
    "ScannedMeasurementNode",
    "ScanResult",
    "SurvivalReducer",
    "TemperatureFit",
    "validate_pulse_streamer_program",
    "validate_device_contract",
    "write_verilog_bundle",
]
