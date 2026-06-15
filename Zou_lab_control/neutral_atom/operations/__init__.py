"""Standalone image operations that do not require a connected experiment."""

from .calibration import calibrate_sitemap_from_images, calibrate_threshold_from_images
from .detection import detect_image
from .feeds import CameraFrameFeed, ExperimentFeed, LoadingFeed, ScannedMeasurementFeed
from .fidelity import (
    FidelityReport,
    ReferenceLabels,
    SiteFidelity,
    TrainTestSplit,
    characterize_readout,
    reference_labels,
    train_test_split,
)
from .imageio import RunIndex, frame_files, index_run, load_frame, save_frame
from .measurement import (
    MeasurementSpec,
    NFramePlan,
    OtsuFidelityReducer,
    ParamDecl,
    PointReducer,
    ScanAxis,
    ScanResult,
    ScannedMeasurement,
    ShotPlan,
    axis_range_tuple,
)
from .measurement_registry import (
    discovered_specs,
    measurement,
    register_measurement,
    registered_measurements,
    unregister_measurement,
)
from .temperature import (
    ReleaseRecapturePlan,
    SurvivalReducer,
    TemperatureFit,
    build_release_recapture_pulse,
    fit_temperature,
    release_recapture_survival,
)

__all__ = [
    "axis_range_tuple",
    "build_release_recapture_pulse",
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "characterize_readout",
    "CameraFrameFeed",
    "detect_image",
    "discovered_specs",
    "ExperimentFeed",
    "FidelityReport",
    "fit_temperature",
    "measurement",
    "register_measurement",
    "registered_measurements",
    "unregister_measurement",
    "frame_files",
    "index_run",
    "load_frame",
    "LoadingFeed",
    "MeasurementSpec",
    "NFramePlan",
    "OtsuFidelityReducer",
    "ParamDecl",
    "PointReducer",
    "ReferenceLabels",
    "reference_labels",
    "release_recapture_survival",
    "ReleaseRecapturePlan",
    "RunIndex",
    "save_frame",
    "ScanAxis",
    "ScannedMeasurement",
    "ScannedMeasurementFeed",
    "ScanResult",
    "ShotPlan",
    "SiteFidelity",
    "SurvivalReducer",
    "TemperatureFit",
    "train_test_split",
    "TrainTestSplit",
]
