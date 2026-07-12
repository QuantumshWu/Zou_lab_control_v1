"""Neutral-atom control session for Jupyter.

It connects devices (virtual or the real qCMOS + FPGA pulse streamer), captures
camera images, calibrates the site map + per-site thresholds, detects occupancy,
and runs swept measurements (detection time / fidelity / release-recapture).

The public notebook entry point is ``connect``.  Raw adapters and drive verbs are
deliberately absent from this umbrella; adapter authors use ``adapter_sdk``, tests use
``testing``, and ordinary experiments act through the installation-owned session.
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
from .core.calibration import FrameContract, TrapCalibration
from .core.params import ParamDecl
from .core.psf import SitePSF, fit_site_psfs, psf_signals
from .ports import PortCatalog, PortSpec
from .devices.sequencer import (
    RuntimeSequenceProgram,
    compile_pulse_table_runtime_program,
    compile_pulse_table_scan_runtime_program,
    compile_runtime_program,
    compile_runtime_program_for_payload,
)
from .views import image_to_points, plot_detection_image, plot_detection_scan, plot_image, plot_site_values, plot_threshold_hist
from .timing import (
    ANALOG_BUS_MODES,
    Pulse,
    PulsePeriod,
    PulseSequence,
    PulseTableState,
    default_pulse_name,
    exposure_from_sequence,
    imaging_sequence,
    plot_sequence,
    positive_time_step_ns,
    quantized_time_ns,
    quantized_time_steps,
)
# The camera capture-trigger helpers live in the CAMERA layer now (the sequencer/timing know
# nothing about which channel gates a camera); re-exported here so notebooks keep `na.*` access.
from .devices.camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS, count_trigger_pulses
from .timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle
from .operations import (
    CameraMeasurement,
    MeasurementSpec,
    NFramePlan,
    OtsuFidelityReducer,
    LogicNode,
    ProcessorSpec,
    ReleaseRecapturePlan,
    RunIndex,
    RunManifest,
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
    index_manifest,
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
from . import simulation
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
    if name == "device_manager":
        # Open the device manager WITHOUT a session (``na.device_manager()``): the device-INIT
        # entry -- edit/create a config, press "Init devices" to connect it.  Reached LAZILY
        # through the GUI-action module so the frontend stays off the analysis import path.
        from ._gui import device_manager

        return device_manager
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
    "ANALOG_BUS_MODES",
    "AtomDetection",
    "BimodalFit",
    "CameraMeasurement",
    "CaptureResult",
    "DEFAULT_FPGA_CHANNEL_COUNT",
    "DEFAULT_MAX_SCAN_POINTS",
    "DEFAULT_SCAN_COEFF_FRAC_BITS",
    "DEFAULT_SCAN_COEFF_WIDTH",
    "DetectionResult",
    "DetectionTimeScanResult",
    "ExperimentSubsystem",
    "FidelityEstimate",
    "FrameContract",
    "LogicNode",
    "MeasurementSpec",
    "MeasurementTaskResult",
    "NFramePlan",
    "NeutralAtomSession",
    "OtsuFidelityReducer",
    "ParamDecl",
    "PortCatalog",
    "PortSpec",
    "PreflightReport",
    "ProcessorSpec",
    "Pulse",
    "PulsePeriod",
    "PulseSequence",
    "PulseTableState",
    "ReadoutSubsystem",
    "ReleaseRecapturePlan",
    "ResultObject",
    "RuntimeSequenceProgram",
    "RunIndex",
    "RunManifest",
    "ScanAxis",
    "ScanResult",
    "ScannedMeasurement",
    "ScannedMeasurementNode",
    "SitePSF",
    "SitemapResult",
    "SurvivalReducer",
    "TaskSpec",
    "TemperatureFit",
    "ThresholdResult",
    "TimingSubsystem",
    "TrapCalibration",
    "VerilogBuild",
    "VerilogFiles",
    "axis_range_tuple",
    "build_fpga_manual",
    "build_main_manual",
    "build_release_recapture_pulse",
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "capacity_estimate_text",
    "connect",
    "compile_pulse_table_runtime_program",
    "compile_pulse_table_scan_runtime_program",
    "compile_runtime_program",
    "compile_runtime_program_for_payload",
    "count_trigger_pulses",
    "default_pulse_name",
    "detect_image",
    "device_manager",
    "estimate_threshold_fidelity",
    "exposure_from_sequence",
    "figure_viewer",
    "find_site_centers",
    "fit_bimodal",
    "fit_bimodal_per_site",
    "fit_site_psfs",
    "fit_temperature",
    "frame_files",
    "gaussian_fidelity",
    "generate_verilog",
    "hardware_channel_names",
    "image_to_points",
    "imaging_sequence",
    "index_manifest",
    "index_run",
    "infer_xdc_channel_count",
    "infer_xdc_channel_labels",
    "infer_xdc_channels",
    "load_figure",
    "load_frame",
    "measurement",
    "otsu_threshold",
    "plot_detection_image",
    "plot_detection_scan",
    "plot_image",
    "plot_sequence",
    "plot_site_values",
    "plot_threshold_hist",
    "positive_time_step_ns",
    "processor",
    "psf_signals",
    "quantized_time_ns",
    "quantized_time_steps",
    "register_measurement",
    "register_processor",
    "register_task",
    "registered_measurements",
    "registered_processors",
    "registered_tasks",
    "release_recapture_survival",
    "roi_counts",
    "save_frame",
    "simulation",
    "sort_centers_grid",
    "task",
    "unregister_measurement",
    "unregister_processor",
    "unregister_task",
    "validate_pulse_streamer_program",
    "write_verilog_bundle",
]
