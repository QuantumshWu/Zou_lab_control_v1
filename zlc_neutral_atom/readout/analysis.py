"""Leakage-free, bounded, headless readout-calibration analysis.

The only durable input is a raw CaptureArtifact-shaped value plus an explicit
CalibrationCaptureLayout.  A bracket partition is frozen before any learned
quantity.  Training brackets propose site geometry, reference thresholds, PSF
kernels, and short-readout thresholds; an independent reference-evidence
partition tests the frozen statistical valleys; held-out reference frames then
produce explicitly unsupervised pseudo-labels used only to score frozen
short-readout models.  No reference observation is called a physical oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from typing import Protocol

import numpy as np
import scipy
from scipy.ndimage import (
    gaussian_filter,
    label,
    maximum_filter,
    minimum_filter,
)
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.stats import beta as beta_distribution, binomtest

from zlc_data import (
    INVALID,
    VALID,
    SITE,
    AxisId,
    AxisSpec,
    ComponentValidity,
    DataBlock,
    DatasetSchema,
    expand_dataset_validity,
)
from zlc_storage import canonical_digest

from .calibration import (
    BackgroundMode,
    BoxReadoutModel,
    BoxReducer,
    CalibrationArtifact,
    CalibrationParameter,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    CalibrationSourceBinding,
    CalibrationStage,
    DEFAULT_CALIBRATION_RESOURCE_POLICY,
    DefaultModelPolicy,
    PerSitePsfReadoutModel,
    ReadoutFeatureSpec,
    ReadoutModel,
    ReadoutModelHeader,
    ReadoutModelKind,
    ReadoutModelQuality,
    SiteMap,
    UniformPsfReadoutModel,
    _derive_calibration_source_binding_with_resolved_brackets,
    _extract_readout_features_arrays,
    _is_bytes_backed_read_only,
    _readout_background_from_arrays,
    validate_calibration_artifact_resources,
)
from .contracts import CalibrationCaptureBracket, CalibrationCaptureLayout, FrameContract


CALIBRATION_ANALYSIS_ALGORITHM_ID = "zlc-readout-calibration-analysis"
CALIBRATION_ANALYSIS_ALGORITHM_VERSION = "7"
# The partition is an algorithm decision, not a post-capture tuning knob.  A
# public seed would let callers repeatedly repartition one observed capture
# until a nominal familywise gate happened to pass.
_PARTITION_SEED = 3817
_MAX_AFFINE_ASSIGNMENT_ITERATIONS = 8
_ASSIGNMENT_WORK_FACTOR = _MAX_AFFINE_ASSIGNMENT_ITERATIONS + 2
_ASSIGNMENT_SCRATCH_BYTES_PER_PAIR = 64
# Conservative phase peak for normalized smoothing, local-extrema labels,
# four vectorized 8-neighbour edge families, their concatenated COO/CSR
# materialization, the sparse spanning forest, sort order, and the reused
# union-find arrays.  The prominence and catchment Kruskal passes are
# sequential and reuse those arrays; they are not concurrent phase peaks.
_DETECTOR_WORKING_BYTES_PER_PIXEL = 384
_TOPOGRAPHIC_EDGE_COUNT_PER_PIXEL = 4
_DEFAULT_MODEL_POLICY_ID = "analysis-request"
_DEFAULT_MODEL_POLICY_VERSION = "4"
_MODEL_VERSION = "5"
_MODEL_ID_BY_KIND = {
    ReadoutModelKind.BOX: "box-v5",
    ReadoutModelKind.PER_SITE_PSF: "per-site-psf-v5",
    ReadoutModelKind.UNIFORM_PSF: "uniform-psf-v5",
}
_QUALITY_GATE_ID = (
    "precommitted-frozen-bracket-adverse-missingness-exact-binomial-"
    "iut-artifact-model-site-holm"
)
_QUALITY_GATE_VERSION = "2"
_REFERENCE_VALLEY_GATE_ID = (
    "independent-stationary-complete-three-bin-exact-binomial-iut-holm"
)
_REFERENCE_VALLEY_GATE_VERSION = "3"
_REFERENCE_AMBIGUITY_GATE_ID = "one-level-nested-valley-screen-holm"
_REFERENCE_AMBIGUITY_GATE_VERSION = "1"
_REFERENCE_EVIDENCE_ASSUMPTION = (
    "INDEPENDENT_STATIONARY_BRACKETS_COMPLETE_REFERENCE_FEATURES"
)


class CalibrationAnalysisError(ValueError):
    """A validly encoded source cannot produce an admitted calibration."""


class GridOrder(str, Enum):
    ROW_MAJOR = "ROW_MAJOR"
    COLUMN_MAJOR = "COLUMN_MAJOR"


class UsableSiteAcceptance(str, Enum):
    ALL = "ALL"
    MINIMUM_FRACTION = "MINIMUM_FRACTION"


class ReferenceLabelSource(str, Enum):
    """The physical authority available for calibration labels.

    The current protocol observes the same unknown state twice; it can admit a
    reproducible dominant binary valley under the declared screens, but it
    cannot turn that statistical evidence into a known physical preparation
    or prove that exactly two physical populations exist.  Its exact-binomial
    claim requires independently prepared, gate-stationary brackets as
    statistical units.  Invalid or non-finite reference features are never
    dropped: they are persisted separately as adverse evidence, and any such
    training/evidence observation prevents that site from acquiring
    reference-label authority.
    """

    UNSUPERVISED_REFERENCE_VALLEY = "UNSUPERVISED_REFERENCE_VALLEY"


class ReferenceClassOrientation(str, Enum):
    """Explicit physical interpretation supplied by the experiment protocol.

    Valley analysis can distinguish a lower and an upper statistical class;
    it cannot infer which class physically means an occupied atom.  Requiring
    this closed value prevents a convenience default from silently upgrading
    unsupervised evidence into an occupancy claim.
    """

    ABOVE_IS_OCCUPIED = "ABOVE_IS_OCCUPIED"
    BELOW_IS_OCCUPIED = "BELOW_IS_OCCUPIED"


class CalibrationBracketSamplingAssumption(str, Enum):
    """Protocol authority required by both finite-sample exact gates.

    Independence and stationarity are properties of how experimental brackets
    are prepared; neither can be proven by re-examining one captured value
    sequence.  For every pooled exact gate, the corresponding Bernoulli
    indicator must have one common, predeclared generating probability across
    the frozen bracket population.  Context-dependent level shifts may not be
    pooled under this assertion.  Exchangeability alone is intentionally not
    offered because correlated blocks invalidate the binomial and
    Clopper--Pearson guarantees.
    """

    INDEPENDENT_STATIONARY_BRACKETS = "INDEPENDENT_STATIONARY_BRACKETS"


class CalibrationAnalysisPlanningAssumption(str, Enum):
    """Protocol authority required for nominal post-selection guarantees.

    The analysis request must be frozen before inspecting the source capture.
    Repeatedly changing partitions, geometry, models, or quality gates on one
    observed capture until a calibration passes invalidates the declared
    familywise error rate.  Software persists and replays this assertion but
    cannot infer a caller's pre-inspection intent from the captured values.
    """

    PRECOMMITTED_BEFORE_SOURCE_INSPECTION = (
        "PRECOMMITTED_BEFORE_SOURCE_INSPECTION"
    )


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _same_typed_scalar(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _typed_parameter_maps_equal(
    actual: dict[str, object],
    expected: dict[str, object],
) -> bool:
    return set(actual) == set(expected) and all(
        _same_typed_scalar(actual[name], value)
        for name, value in expected.items()
    )


@dataclass(frozen=True)
class BoxAnalysisConfig:
    half_width: int = 2
    reducer: BoxReducer = BoxReducer.SUM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "half_width",
            _nonnegative_integer(self.half_width, "box half_width"),
        )
        if not isinstance(self.reducer, BoxReducer):
            raise TypeError("box reducer must be BoxReducer")


@dataclass(frozen=True)
class PsfAnalysisConfig:
    half_width: int = 3
    background: BackgroundMode = BackgroundMode.ANNULUS_MEDIAN
    background_padding: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "half_width",
            _nonnegative_integer(self.half_width, "PSF half_width"),
        )
        if not isinstance(self.background, BackgroundMode):
            raise TypeError("PSF background must be BackgroundMode")
        padding = _nonnegative_integer(
            self.background_padding,
            "PSF background_padding",
        )
        if self.background is BackgroundMode.ANNULUS_MEDIAN and padding == 0:
            raise ValueError("ANNULUS_MEDIAN requires positive background padding")
        if self.background is BackgroundMode.NONE and padding != 0:
            raise ValueError("NONE background requires canonical zero padding")
        object.__setattr__(self, "background_padding", padding)


@dataclass(frozen=True)
class SiteDetectionPolicy:
    smoothing_sigma_pixels: float = 0.75
    minimum_prominence_fraction: float = 0.10
    minimum_peak_separation_pixels: float = 2.0
    minimum_half_prominence_basin_area_pixels: int = 4
    reject_touching_half_prominence_basins: bool = True
    maximum_lattice_rms_residual_pixels: float = 0.75
    minimum_lattice_step_pixels: float = 2.0
    minimum_band_separation_pixels: float = 0.5
    minimum_affine_sin_angle: float = 0.20
    maximum_affine_condition_number: float = 10.0
    minimum_assignment_cost_gap_pixels_squared: float = 4.0

    def __post_init__(self) -> None:
        sigma = _finite_real(self.smoothing_sigma_pixels, "smoothing_sigma_pixels")
        if sigma < 0.0:
            raise ValueError("smoothing_sigma_pixels must be non-negative")
        object.__setattr__(self, "smoothing_sigma_pixels", sigma)
        prominence = _finite_real(
            self.minimum_prominence_fraction,
            "minimum_prominence_fraction",
        )
        if not 0.0 < prominence <= 1.0:
            raise ValueError("minimum_prominence_fraction must lie in (0, 1]")
        object.__setattr__(self, "minimum_prominence_fraction", prominence)
        basin_area = _positive_integer(
            self.minimum_half_prominence_basin_area_pixels,
            "minimum_half_prominence_basin_area_pixels",
        )
        object.__setattr__(
            self,
            "minimum_half_prominence_basin_area_pixels",
            basin_area,
        )
        if not isinstance(self.reject_touching_half_prominence_basins, bool):
            raise TypeError("reject_touching_half_prominence_basins must be bool")
        for name in (
            "minimum_peak_separation_pixels",
            "maximum_lattice_rms_residual_pixels",
            "minimum_lattice_step_pixels",
            "minimum_band_separation_pixels",
            "minimum_affine_sin_angle",
            "maximum_affine_condition_number",
            "minimum_assignment_cost_gap_pixels_squared",
        ):
            value = _finite_real(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.minimum_affine_sin_angle > 1.0:
            raise ValueError("minimum_affine_sin_angle must not exceed one")
        if self.maximum_affine_condition_number < 1.0:
            raise ValueError("maximum_affine_condition_number must be at least one")


@dataclass(frozen=True)
class CalibrationAnalysisResourcePolicy:
    artifact_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY
    max_source_cells: int = 2_000_000
    max_brackets: int = 200_000
    max_reference_frames: int = 1_000_000
    max_image_pixels: int = 20_000_000
    max_signal_evaluations: int = 100_000_000
    max_modality_test_work_units: int = 20_000_000_000
    max_reference_valley_diagnostics: int = 100_000
    max_sampled_pixel_operations: int = 5_000_000_000
    max_working_bytes: int = 2_000_000_000
    max_lattice_sites: int = 2_048
    max_detector_graph_work_units: int = 2_000_000_000
    max_dense_assignment_work_units: int = 20_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_policy, CalibrationResourcePolicy):
            raise TypeError("artifact_policy must be CalibrationResourcePolicy")
        for name in (
            "max_source_cells",
            "max_brackets",
            "max_reference_frames",
            "max_image_pixels",
            "max_signal_evaluations",
            "max_modality_test_work_units",
            "max_reference_valley_diagnostics",
            "max_sampled_pixel_operations",
            "max_working_bytes",
            "max_lattice_sites",
            "max_detector_graph_work_units",
            "max_dense_assignment_work_units",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))


@dataclass(frozen=True)
class CalibrationWorkPlan:
    source_cell_count: int
    bracket_upper_bound: int
    train_bracket_upper_bound: int
    reference_evidence_bracket_upper_bound: int
    reference_frame_upper_bound: int
    image_pixel_count: int
    full_frame_read_count: int
    feature_pixel_operations: int
    signal_evaluations: int
    modality_test_work_units: int
    reference_valley_diagnostic_count: int
    diagnostics_encoding_upper_bound_bytes: int
    planned_kernel_elements: int
    maximum_model_sampled_pixels: int
    total_model_sampled_pixels: int
    artifact_metadata_encoding_upper_bound_bytes: int
    artifact_encoding_upper_bound_bytes: int
    layout_working_bytes: int
    detector_working_bytes: int
    assignment_scratch_bytes: int
    feature_working_bytes: int
    psf_working_bytes: int
    artifact_array_bytes: int
    canonical_encoding_scratch_bytes: int
    working_peak_bytes: int
    detector_graph_work_units: int
    dense_assignment_work_units: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _nonnegative_integer(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.source_cell_count == 0 or self.image_pixel_count == 0:
            raise ValueError("calibration work plan requires non-empty source geometry")
        from .analysis_codec import (
            calibration_analysis_diagnostics_encoding_working_upper_bound,
        )
        from .calibration_codec import (
            calibration_artifact_encoding_working_upper_bound,
        )

        required_canonical_scratch = max(
            calibration_analysis_diagnostics_encoding_working_upper_bound(
                self.diagnostics_encoding_upper_bound_bytes
            ),
            calibration_artifact_encoding_working_upper_bound(
                self.artifact_array_bytes,
                self.artifact_metadata_encoding_upper_bound_bytes,
            ),
        )
        if self.canonical_encoding_scratch_bytes < required_canonical_scratch:
            raise ValueError(
                "canonical scratch is lower than an owner encoding working bound"
            )
        if (
            self.canonical_encoding_scratch_bytes
            < self.diagnostics_encoding_upper_bound_bytes
        ):
            raise ValueError(
                "canonical scratch is lower than diagnostics encoding bound"
            )
        if self.total_model_sampled_pixels < self.maximum_model_sampled_pixels:
            raise ValueError(
                "total model sampled pixels are lower than the per-model maximum"
            )
        if (
            self.canonical_encoding_scratch_bytes
            < self.artifact_encoding_upper_bound_bytes
        ):
            raise ValueError(
                "canonical scratch is lower than artifact encoding bound"
            )
        if self.working_peak_bytes < max(
            self.layout_working_bytes,
            self.detector_working_bytes + self.assignment_scratch_bytes,
            self.assignment_scratch_bytes,
            self.artifact_array_bytes + self.feature_working_bytes,
            self.artifact_array_bytes
            + self.psf_working_bytes
            + self.feature_working_bytes,
            self.artifact_array_bytes + self.canonical_encoding_scratch_bytes,
        ):
            raise ValueError("working_peak_bytes is lower than a declared phase peak")

    @property
    def fingerprint(self) -> str:
        from .analysis_codec import calibration_work_plan_to_tree

        return canonical_digest(calibration_work_plan_to_tree(self))


@dataclass(frozen=True)
class CalibrationAnalysisRequest:
    layout: CalibrationCaptureLayout
    grid_shape_yx: tuple[int, int]
    reference_label_source: ReferenceLabelSource
    reference_class_orientation: ReferenceClassOrientation
    bracket_sampling_assumption: CalibrationBracketSamplingAssumption
    analysis_planning_assumption: CalibrationAnalysisPlanningAssumption
    grid_order: GridOrder = GridOrder.ROW_MAJOR
    box: BoxAnalysisConfig = BoxAnalysisConfig()
    model_kinds: tuple[ReadoutModelKind, ...] = (ReadoutModelKind.BOX,)
    default_model_kind: ReadoutModelKind | None = ReadoutModelKind.BOX
    psf: PsfAnalysisConfig | None = None
    detection: SiteDetectionPolicy = SiteDetectionPolicy()
    train_fraction: float = 0.35
    reference_evidence_fraction: float = 0.35
    minimum_train_samples_per_class: int = 4
    minimum_test_samples_per_class: int = 4
    minimum_reference_cluster_separation_rss: float = 2.0
    reference_valley_familywise_error_rate: float = 0.01
    held_out_confidence_level: float = 0.95
    minimum_held_out_class_accuracy_lower_bound: float = 0.60
    usable_site_acceptance: UsableSiteAcceptance = UsableSiteAcceptance.ALL
    minimum_usable_site_fraction: float = 1.0
    resource_policy: CalibrationAnalysisResourcePolicy = CalibrationAnalysisResourcePolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.layout, CalibrationCaptureLayout):
            raise TypeError("layout must be CalibrationCaptureLayout")
        shape = tuple(self.grid_shape_yx)
        if len(shape) != 2:
            raise ValueError("grid_shape_yx must have two entries")
        shape = tuple(_positive_integer(value, "grid_shape_yx entry") for value in shape)
        object.__setattr__(self, "grid_shape_yx", shape)
        if not isinstance(self.reference_label_source, ReferenceLabelSource):
            raise TypeError("reference_label_source must be ReferenceLabelSource")
        if not isinstance(self.reference_class_orientation, ReferenceClassOrientation):
            raise TypeError(
                "reference_class_orientation must be ReferenceClassOrientation"
            )
        if not isinstance(
            self.bracket_sampling_assumption,
            CalibrationBracketSamplingAssumption,
        ):
            raise TypeError(
                "bracket_sampling_assumption must be "
                "CalibrationBracketSamplingAssumption"
            )
        if not isinstance(
            self.analysis_planning_assumption,
            CalibrationAnalysisPlanningAssumption,
        ):
            raise TypeError(
                "analysis_planning_assumption must be "
                "CalibrationAnalysisPlanningAssumption"
            )
        if not isinstance(self.grid_order, GridOrder):
            raise TypeError("grid_order must be GridOrder")
        if not isinstance(self.box, BoxAnalysisConfig):
            raise TypeError("box must be BoxAnalysisConfig")
        kinds = tuple(self.model_kinds)
        if not kinds or any(not isinstance(kind, ReadoutModelKind) for kind in kinds):
            raise ValueError("model_kinds must contain closed ReadoutModelKind values")
        if len(set(kinds)) != len(kinds):
            raise ValueError("model_kinds must be unique")
        kinds = tuple(sorted(kinds, key=lambda kind: kind.value))
        object.__setattr__(self, "model_kinds", kinds)
        if self.default_model_kind is not None:
            if not isinstance(self.default_model_kind, ReadoutModelKind):
                raise TypeError("default_model_kind must be ReadoutModelKind or None")
            if self.default_model_kind not in kinds:
                raise ValueError("default_model_kind must be one of model_kinds")
        needs_psf = any(kind is not ReadoutModelKind.BOX for kind in kinds)
        if needs_psf and not isinstance(self.psf, PsfAnalysisConfig):
            raise ValueError("requested PSF models require a PsfAnalysisConfig")
        if not needs_psf and self.psf is not None:
            raise ValueError("BOX-only request requires canonical absent PSF config")
        if not isinstance(self.detection, SiteDetectionPolicy):
            raise TypeError("detection must be SiteDetectionPolicy")
        fraction = _finite_real(self.train_fraction, "train_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between zero and one")
        object.__setattr__(self, "train_fraction", fraction)
        evidence_fraction = _finite_real(
            self.reference_evidence_fraction,
            "reference_evidence_fraction",
        )
        if not 0.0 < evidence_fraction < 1.0:
            raise ValueError(
                "reference_evidence_fraction must lie strictly between zero and one"
            )
        if fraction + evidence_fraction >= 1.0:
            raise ValueError(
                "train_fraction plus reference_evidence_fraction must be below one"
            )
        object.__setattr__(
            self,
            "reference_evidence_fraction",
            evidence_fraction,
        )
        for name in (
            "minimum_train_samples_per_class",
            "minimum_test_samples_per_class",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name),
            )
        reference_separation = _finite_real(
            self.minimum_reference_cluster_separation_rss,
            "minimum_reference_cluster_separation_rss",
        )
        if reference_separation <= 0.0:
            raise ValueError(
                "minimum_reference_cluster_separation_rss must be positive"
            )
        object.__setattr__(
            self,
            "minimum_reference_cluster_separation_rss",
            reference_separation,
        )
        familywise_error_rate = _finite_real(
            self.reference_valley_familywise_error_rate,
            "reference_valley_familywise_error_rate",
        )
        if not 0.0 < familywise_error_rate < 1.0:
            raise ValueError(
                "reference_valley_familywise_error_rate must lie inside (0, 1)"
            )
        object.__setattr__(
            self,
            "reference_valley_familywise_error_rate",
            familywise_error_rate,
        )
        confidence = _finite_real(
            self.held_out_confidence_level,
            "held_out_confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError("held_out_confidence_level must lie strictly inside (0, 1)")
        object.__setattr__(self, "held_out_confidence_level", confidence)
        lower_bound = _finite_real(
            self.minimum_held_out_class_accuracy_lower_bound,
            "minimum_held_out_class_accuracy_lower_bound",
        )
        if not 0.0 <= lower_bound <= 1.0:
            raise ValueError(
                "minimum_held_out_class_accuracy_lower_bound must lie in [0, 1]"
            )
        object.__setattr__(
            self,
            "minimum_held_out_class_accuracy_lower_bound",
            lower_bound,
        )
        if not isinstance(self.usable_site_acceptance, UsableSiteAcceptance):
            raise TypeError("usable_site_acceptance must be UsableSiteAcceptance")
        usable_fraction = _finite_real(
            self.minimum_usable_site_fraction,
            "minimum_usable_site_fraction",
        )
        if not 0.0 < usable_fraction <= 1.0:
            raise ValueError("minimum_usable_site_fraction must lie in (0, 1]")
        if (
            self.usable_site_acceptance is UsableSiteAcceptance.ALL
            and usable_fraction != 1.0
        ):
            raise ValueError("ALL usable-site acceptance requires canonical fraction 1.0")
        object.__setattr__(self, "minimum_usable_site_fraction", usable_fraction)
        if not isinstance(self.resource_policy, CalibrationAnalysisResourcePolicy):
            raise TypeError("resource_policy must be CalibrationAnalysisResourcePolicy")
        if self.site_count > self.resource_policy.artifact_policy.max_sites:
            raise CalibrationResourceExceeded("requested grid exceeds site resource policy")
        if self.site_count > self.resource_policy.max_lattice_sites:
            raise CalibrationResourceExceeded("requested grid exceeds lattice site budget")
        if len(kinds) > self.resource_policy.artifact_policy.max_models:
            raise CalibrationResourceExceeded("requested models exceed resource policy")

    @property
    def site_count(self) -> int:
        return math.prod(self.grid_shape_yx)

    @property
    def fingerprint(self) -> str:
        from .analysis_codec import calibration_analysis_request_to_tree

        return canonical_digest(calibration_analysis_request_to_tree(self))


@dataclass(frozen=True)
class SiteDetectionDiagnostic:
    candidate_count: int
    minimum_peak_to_saddle_prominence: float
    minimum_half_prominence_basin_area_pixels: int
    lattice_rms_residual_pixels: float
    minimum_band_separation_pixels: float | None
    affine_sin_angle: float | None
    affine_condition_number: float | None
    assignment_cost_gap_pixels_squared: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_count",
            _positive_integer(self.candidate_count, "candidate_count"),
        )
        object.__setattr__(
            self,
            "minimum_half_prominence_basin_area_pixels",
            _positive_integer(
                self.minimum_half_prominence_basin_area_pixels,
                "minimum_half_prominence_basin_area_pixels",
            ),
        )
        for name in (
            "minimum_peak_to_saddle_prominence",
            "lattice_rms_residual_pixels",
        ):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_band_separation_pixels",
            "affine_sin_angle",
            "affine_condition_number",
            "assignment_cost_gap_pixels_squared",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = _finite_real(value, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.affine_sin_angle is not None and self.affine_sin_angle > 1.0:
            raise ValueError("affine_sin_angle must not exceed one")
        if (
            self.affine_condition_number is not None
            and self.affine_condition_number < 1.0
        ):
            raise ValueError("affine_condition_number must be at least one")


@dataclass(frozen=True)
class ModelAnalysisDiagnostic:
    kind: ReadoutModelKind
    usable_site_count: int
    rejected_site_count: int
    minimum_fidelity: float
    mean_fidelity: float
    minimum_class_accuracy_lower_bound: float
    mean_class_accuracy_lower_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind")
        for name in ("usable_site_count", "rejected_site_count"):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name),
            )
        if self.usable_site_count == 0:
            raise ValueError("model diagnostic requires at least one usable site")
        for minimum_name, mean_name in (
            ("minimum_fidelity", "mean_fidelity"),
            (
                "minimum_class_accuracy_lower_bound",
                "mean_class_accuracy_lower_bound",
            ),
        ):
            minimum = _finite_real(getattr(self, minimum_name), minimum_name)
            mean = _finite_real(getattr(self, mean_name), mean_name)
            if not 0.0 <= minimum <= mean <= 1.0:
                raise ValueError(f"{minimum_name}/{mean_name} must satisfy 0 <= min <= mean <= 1")
            object.__setattr__(self, minimum_name, minimum)
            object.__setattr__(self, mean_name, mean)


@dataclass(frozen=True)
class ReferenceValleyEvidence:
    """Finite-sample exact evidence for one frozen three-bin valley."""

    sample_count: int
    left_count: int
    middle_count: int
    right_count: int
    outside_count: int
    invalid_count: int

    def __post_init__(self) -> None:
        for name in (
            "sample_count",
            "left_count",
            "middle_count",
            "right_count",
            "outside_count",
            "invalid_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name),
            )
        if (
            self.left_count
            + self.middle_count
            + self.right_count
            + self.outside_count
            + self.invalid_count
            != self.sample_count
        ):
            raise ValueError("reference-valley bin counts must cover all evidence")


    def _outer_vs_adverse_pvalue(self, outer_count: int) -> float:
        # Every valid evidence sample participates.  Values outside the three
        # frozen bins are adverse evidence, not silently renormalized away.
        adverse_count = (
            self.middle_count + self.outside_count + self.invalid_count
        )
        pair_count = outer_count + adverse_count
        if pair_count == 0:
            return 1.0
        return float(
            binomtest(
                outer_count,
                pair_count,
                0.5,
                alternative="greater",
            ).pvalue
        )

    @property
    def left_vs_adverse_pvalue(self) -> float:
        return self._outer_vs_adverse_pvalue(self.left_count)

    @property
    def right_vs_adverse_pvalue(self) -> float:
        return self._outer_vs_adverse_pvalue(self.right_count)

    @property
    def valley_pvalue(self) -> float:
        return max(
            self.left_vs_adverse_pvalue,
            self.right_vs_adverse_pvalue,
        )


@dataclass(frozen=True)
class ReferenceValleyDiagnostic:
    """Auditable evidence for one reference-event/site label proposal.

    ``site_accepted`` means only that the independently proposed statistical
    valley passed the declared familywise gates.  It does not assert that the
    two populations are physically prepared empty/occupied states.
    """

    reference_index: int
    site_index: int
    proposal_threshold: float | None
    proposal_lower_sample_count: int
    proposal_upper_sample_count: int
    cluster_separation_rss: float | None
    evidence: ReferenceValleyEvidence
    lower_cluster_evidence: ReferenceValleyEvidence | None
    upper_cluster_evidence: ReferenceValleyEvidence | None
    site_accepted: bool

    def __post_init__(self) -> None:
        for name in (
            "reference_index",
            "site_index",
            "proposal_lower_sample_count",
            "proposal_upper_sample_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name),
            )
        threshold = self.proposal_threshold
        separation = self.cluster_separation_rss
        if threshold is None:
            if (
                self.proposal_lower_sample_count != 0
                or self.proposal_upper_sample_count != 0
                or separation is not None
            ):
                raise ValueError("missing proposal must not carry proposal evidence")
        else:
            threshold = _finite_real(threshold, "proposal_threshold")
            if (
                self.proposal_lower_sample_count == 0
                or self.proposal_upper_sample_count == 0
            ):
                raise ValueError("proposal requires samples in both classes")
            if (
                separation is None
                or isinstance(separation, bool)
                or not isinstance(separation, Real)
            ):
                raise ValueError(
                    "present proposal requires cluster_separation_rss"
                )
            separation = float(separation)
            if math.isnan(separation) or separation <= 0.0:
                raise ValueError("cluster_separation_rss must be positive")
        object.__setattr__(self, "proposal_threshold", threshold)
        object.__setattr__(self, "cluster_separation_rss", separation)
        if not isinstance(self.evidence, ReferenceValleyEvidence):
            raise TypeError("evidence must be ReferenceValleyEvidence")
        for name in ("lower_cluster_evidence", "upper_cluster_evidence"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ReferenceValleyEvidence):
                raise TypeError(f"{name} must be ReferenceValleyEvidence or None")
        if not isinstance(self.site_accepted, bool):
            raise TypeError("site_accepted must be bool")


@dataclass(frozen=True)
class CalibrationAnalysisDiagnostics:
    bracket_count: int
    train_bracket_count: int
    reference_evidence_bracket_count: int
    test_bracket_count: int
    partition_digest: str
    reference_frame_count: int
    valid_training_reference_pixel_fraction: float
    consensus_dark_counts: tuple[int, ...]
    consensus_bright_counts: tuple[int, ...]
    reference_valleys: tuple[ReferenceValleyDiagnostic, ...]
    detection: SiteDetectionDiagnostic
    models: tuple[ModelAnalysisDiagnostic, ...]

    def __post_init__(self) -> None:
        for name in (
            "bracket_count",
            "train_bracket_count",
            "reference_evidence_bracket_count",
            "test_bracket_count",
            "reference_frame_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name),
            )
        if self.bracket_count == 0:
            raise ValueError("bracket_count must be positive")
        if self.reference_frame_count == 0:
            raise ValueError("reference_frame_count must be positive")
        if (
            self.train_bracket_count == 0
            or self.reference_evidence_bracket_count == 0
            or self.test_bracket_count == 0
        ):
            raise ValueError(
                "diagnostics require non-empty train, reference-evidence, and test partitions"
            )
        if (
            self.train_bracket_count
            + self.reference_evidence_bracket_count
            + self.test_bracket_count
            != self.bracket_count
        ):
            raise ValueError("diagnostic partition counts must sum to bracket_count")
        if (
            not isinstance(self.partition_digest, str)
            or len(self.partition_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.partition_digest)
        ):
            raise ValueError("partition_digest must be a lowercase SHA-256 digest")
        fraction = _finite_real(
            self.valid_training_reference_pixel_fraction,
            "valid_training_reference_pixel_fraction",
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("valid training reference fraction must lie in [0, 1]")
        object.__setattr__(
            self,
            "valid_training_reference_pixel_fraction",
            fraction,
        )
        dark = tuple(
            _nonnegative_integer(value, "consensus_dark_counts entry")
            for value in self.consensus_dark_counts
        )
        bright = tuple(
            _nonnegative_integer(value, "consensus_bright_counts entry")
            for value in self.consensus_bright_counts
        )
        if not dark or len(dark) != len(bright):
            raise ValueError("consensus count vectors must be non-empty and equal length")
        valleys = tuple(self.reference_valleys)
        if self.reference_frame_count % self.bracket_count != 0:
            raise ValueError("reference-frame count is not divisible by bracket count")
        reference_count = self.reference_frame_count // self.bracket_count
        expected_valley_count = reference_count * len(dark)
        if len(valleys) != expected_valley_count or any(
            not isinstance(item, ReferenceValleyDiagnostic) for item in valleys
        ):
            raise ValueError(
                "reference_valleys must contain one diagnostic per reference and site"
            )
        for reference in range(reference_count):
            for site in range(len(dark)):
                item = valleys[reference * len(dark) + site]
                if item.reference_index != reference or item.site_index != site:
                    raise ValueError("reference-valley diagnostic order is not canonical")
                if (
                    item.proposal_lower_sample_count
                    + item.proposal_upper_sample_count
                    > self.train_bracket_count
                ):
                    raise ValueError("reference proposal exceeds training partition")
                if item.evidence.sample_count != self.reference_evidence_bracket_count:
                    raise ValueError(
                        "reference evidence must cover every scheduled evidence bracket"
                    )
                for nested in (
                    item.lower_cluster_evidence,
                    item.upper_cluster_evidence,
                ):
                    if (
                        nested is not None
                        and nested.sample_count
                        > self.reference_evidence_bracket_count
                    ):
                        raise ValueError("nested evidence exceeds evidence partition")
                if item.proposal_threshold is None and (
                    item.evidence.left_count != 0
                    or item.evidence.middle_count != 0
                    or item.evidence.right_count != 0
                    or (
                        item.evidence.outside_count
                        + item.evidence.invalid_count
                        != item.evidence.sample_count
                    )
                    or item.evidence.valley_pvalue != 1.0
                    or item.lower_cluster_evidence is not None
                    or item.upper_cluster_evidence is not None
                    or item.site_accepted
                ):
                    raise ValueError("missing proposal cannot carry valley authority")
        for site in range(len(dark)):
            site_items = valleys[site::len(dark)]
            accepted = site_items[0].site_accepted
            if any(
                item.site_accepted is not accepted
                for item in site_items
            ):
                raise ValueError("reference-valley site decisions are inconsistent")
            if dark[site] + bright[site] > self.bracket_count:
                raise ValueError(
                    "consensus class counts exceed the bracket population"
                )
        if not isinstance(self.detection, SiteDetectionDiagnostic):
            raise TypeError("detection must be SiteDetectionDiagnostic")
        if self.detection.candidate_count != len(dark):
            raise ValueError("site-detection diagnostic count differs from site vectors")
        models = tuple(self.models)
        if not models or any(not isinstance(item, ModelAnalysisDiagnostic) for item in models):
            raise ValueError("models must contain ModelAnalysisDiagnostic values")
        if len({item.kind for item in models}) != len(models):
            raise ValueError("diagnostic model kinds must be unique")
        if tuple(item.kind for item in models) != tuple(
            sorted((item.kind for item in models), key=lambda kind: kind.value)
        ):
            raise ValueError("diagnostic model kinds must use canonical order")
        if any(
            item.usable_site_count + item.rejected_site_count != len(dark)
            for item in models
        ):
            raise ValueError("diagnostic model site counts differ from site vectors")
        object.__setattr__(self, "consensus_dark_counts", dark)
        object.__setattr__(self, "consensus_bright_counts", bright)
        object.__setattr__(self, "reference_valleys", valleys)
        object.__setattr__(self, "models", models)


@dataclass(frozen=True)
class CalibrationAnalysisResult:
    artifact: CalibrationArtifact
    diagnostics: CalibrationAnalysisDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, CalibrationArtifact):
            raise TypeError("artifact must be CalibrationArtifact")
        if not isinstance(self.diagnostics, CalibrationAnalysisDiagnostics):
            raise TypeError("diagnostics must be CalibrationAnalysisDiagnostics")
        if (
            self.artifact.algorithm_id != CALIBRATION_ANALYSIS_ALGORITHM_ID
            or self.artifact.algorithm_version != CALIBRATION_ANALYSIS_ALGORITHM_VERSION
        ):
            raise ValueError("analysis result artifact names another algorithm")
        site_count = self.artifact.site_map.site_axis.size
        if len(self.diagnostics.consensus_dark_counts) != site_count:
            raise ValueError("diagnostic consensus vectors differ from artifact site count")
        if self.diagnostics.bracket_count != self.artifact.source_binding.bracket_count:
            raise ValueError("diagnostic bracket count differs from source binding")
        expected_reference_frames = self.diagnostics.bracket_count * len(
            self.artifact.source_binding.layout.reference_event_indices
        )
        if self.diagnostics.reference_frame_count != expected_reference_frames:
            raise ValueError("diagnostic reference-frame count differs from source layout")
        if self.diagnostics.detection.candidate_count != site_count:
            raise ValueError("site-detection diagnostic differs from artifact site count")
        model_kinds = tuple(model.kind for model in self.artifact.models)
        diagnostic_kinds = tuple(item.kind for item in self.diagnostics.models)
        if diagnostic_kinds != model_kinds:
            raise ValueError("diagnostic model order differs from artifact")

        artifact_parameters = {
            item.name: item.value for item in self.artifact.parameters
        }
        if artifact_parameters.get("bracket-partition-digest") != (
            self.diagnostics.partition_digest
        ):
            raise ValueError("artifact partition lineage differs from diagnostics")
        for name in (
            "analysis-request-fingerprint",
            "analysis-work-plan-fingerprint",
            "numeric-backend-digest",
        ):
            value = artifact_parameters.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"artifact omits canonical {name}")
        expected_analysis_parameters = {
            "analysis-planning-assumption": (
                CalibrationAnalysisPlanningAssumption
                .PRECOMMITTED_BEFORE_SOURCE_INSPECTION.value
            ),
            "held-out-family-scope": "ARTIFACT_MODEL_SITE",
            "held-out-family-model-count": len(self.artifact.models),
            "held-out-family-hypothesis-count": (
                len(self.artifact.models) * site_count
            ),
        }
        for name, expected in expected_analysis_parameters.items():
            if not _same_typed_scalar(artifact_parameters.get(name), expected):
                raise ValueError(f"artifact analysis parameter {name} differs")
        expected_reference_parameters = {
            "bracket-sampling-assumption": (
                CalibrationBracketSamplingAssumption.INDEPENDENT_STATIONARY_BRACKETS.value
            ),
            "reference-label-source": (
                ReferenceLabelSource.UNSUPERVISED_REFERENCE_VALLEY.value
            ),
            "reference-valley-gate-id": (
                _REFERENCE_VALLEY_GATE_ID
            ),
            "reference-valley-gate-version": _REFERENCE_VALLEY_GATE_VERSION,
            "reference-ambiguity-gate-id": _REFERENCE_AMBIGUITY_GATE_ID,
            "reference-ambiguity-gate-version": _REFERENCE_AMBIGUITY_GATE_VERSION,
            "reference-statistical-unit": "BRACKET",
            "reference-evidence-assumption": (
                _REFERENCE_EVIDENCE_ASSUMPTION
            ),
            "train-bracket-count": self.diagnostics.train_bracket_count,
            "reference-evidence-bracket-count": (
                self.diagnostics.reference_evidence_bracket_count
            ),
            "test-bracket-count": self.diagnostics.test_bracket_count,
        }
        for name, expected in expected_reference_parameters.items():
            if not _same_typed_scalar(artifact_parameters.get(name), expected):
                raise ValueError(f"artifact reference-label parameter {name} differs")
        familywise_error_rate = artifact_parameters.get(
            "reference-valley-familywise-error-rate"
        )
        if (
            isinstance(familywise_error_rate, bool)
            or not isinstance(familywise_error_rate, Real)
            or not math.isfinite(float(familywise_error_rate))
            or not 0.0 < float(familywise_error_rate) < 1.0
        ):
            raise ValueError("artifact omits reference-valley familywise error rate")
        familywise_error_rate = float(familywise_error_rate)
        separation_gate = artifact_parameters.get(
            "minimum-reference-cluster-separation-rss"
        )
        if (
            isinstance(separation_gate, bool)
            or not isinstance(separation_gate, Real)
            or not math.isfinite(float(separation_gate))
            or float(separation_gate) <= 0.0
        ):
            raise ValueError("artifact omits positive reference separation gate")
        separation_gate = float(separation_gate)
        minimum_reference_samples = artifact_parameters.get(
            "minimum-reference-proposal-samples-per-class"
        )
        if (
            isinstance(minimum_reference_samples, bool)
            or not isinstance(minimum_reference_samples, Integral)
            or minimum_reference_samples <= 0
        ):
            raise ValueError("artifact omits reference proposal sample gate")
        orientation = artifact_parameters.get("reference-class-orientation")
        if orientation not in {"ABOVE_IS_OCCUPIED", "BELOW_IS_OCCUPIED"}:
            raise ValueError("artifact omits reference class orientation")
        for item in self.diagnostics.reference_valleys:
            if item.proposal_threshold is None:
                continue
            if (
                item.proposal_lower_sample_count < minimum_reference_samples
                or item.proposal_upper_sample_count < minimum_reference_samples
            ):
                raise ValueError("reference proposal violates minimum sample gate")
            if (
                item.cluster_separation_rss is not None
                and item.cluster_separation_rss < separation_gate
            ):
                raise ValueError("reference proposal violates separation gate")

        reference_count = len(
            self.artifact.source_binding.layout.reference_event_indices
        )
        valley_matrix = np.asarray(
            [
                item.evidence.valley_pvalue
                for item in self.diagnostics.reference_valleys
            ],
            dtype=np.float64,
        ).reshape(reference_count, site_count)
        expected_primary = _holm_rejections(
            np.max(valley_matrix, axis=0),
            familywise_error_rate,
        )
        nested_matrix = np.asarray(
            [
                1.0 if value is None else value
                for item in self.diagnostics.reference_valleys
                for evidence in (
                    item.lower_cluster_evidence,
                    item.upper_cluster_evidence,
                )
                for value in (
                    None if evidence is None else evidence.valley_pvalue,
                )
            ],
            dtype=np.float64,
        ).reshape(reference_count, site_count, 2)
        expected_ambiguous = np.any(
            _holm_rejections(
                nested_matrix.reshape(-1),
                familywise_error_rate,
            ).reshape(nested_matrix.shape),
            axis=(0, 2),
        )
        complete_reference_sites = np.ones(site_count, dtype=bool)
        for site in range(site_count):
            for reference in range(reference_count):
                item = self.diagnostics.reference_valleys[
                    reference * site_count + site
                ]
                if (
                    item.proposal_lower_sample_count
                    + item.proposal_upper_sample_count
                    != self.diagnostics.train_bracket_count
                    or item.evidence.invalid_count != 0
                    or any(
                        nested is not None and nested.invalid_count != 0
                        for nested in (
                            item.lower_cluster_evidence,
                            item.upper_cluster_evidence,
                        )
                    )
                ):
                    complete_reference_sites[site] = False
                    break
        expected_reference_sites = (
            expected_primary
            & ~expected_ambiguous
            & complete_reference_sites
        )
        reported_reference_sites = np.asarray(
            [
                self.diagnostics.reference_valleys[site].site_accepted
                for site in range(site_count)
            ],
            dtype=bool,
        )
        if not np.array_equal(expected_reference_sites, reported_reference_sites):
            raise ValueError("reference-valley decision differs from persisted evidence")
        declared_gate_policy: tuple[int, int, float, float, str, float] | None = None
        held_out_records: list[_HeldOutReplayRecord] = []
        for model, diagnostic in zip(
            self.artifact.models,
            self.diagnostics.models,
            strict=True,
        ):
            model_parameters = {
                item.name: item.value for item in model.header.parameters
            }
            if model_parameters.get("bracket-partition-digest") != (
                self.diagnostics.partition_digest
            ):
                raise ValueError("model partition lineage differs from diagnostics")
            for name in (
                "analysis-request-fingerprint",
                "analysis-work-plan-fingerprint",
                "numeric-backend-digest",
                "bracket-sampling-assumption",
                "analysis-planning-assumption",
                "held-out-family-scope",
                "held-out-family-model-count",
                "held-out-family-hypothesis-count",
            ):
                if not _same_typed_scalar(
                    model_parameters.get(name),
                    artifact_parameters.get(name),
                ):
                    raise ValueError(f"model {name} differs from artifact")
            if (
                model_parameters.get("train-bracket-count")
                != self.diagnostics.train_bracket_count
                or model_parameters.get("reference-evidence-bracket-count")
                != self.diagnostics.reference_evidence_bracket_count
                or model_parameters.get("test-bracket-count")
                != self.diagnostics.test_bracket_count
            ):
                raise ValueError("model partition counts differ from diagnostics")
            quality = model.header.quality
            if (
                quality.quality_gate_id != _QUALITY_GATE_ID
                or quality.quality_gate_version != _QUALITY_GATE_VERSION
                or not quality.gate_passed
            ):
                raise ValueError("model quality gate differs from analysis contract")
            usable = quality.usable_sites.mask
            if any(
                int(dark_count) + int(bright_count)
                > self.diagnostics.train_bracket_count
                for dark_count, bright_count in zip(
                    quality.dark_training_sample_counts,
                    quality.bright_training_sample_counts,
                    strict=True,
                )
            ):
                raise ValueError("model training evidence exceeds train partition")
            evidence_sites = quality.held_out_validity.mask
            for site in np.flatnonzero(evidence_sites):
                dark_unknown = int(
                    quality.held_out_dark_total_counts[site]
                    - quality.held_out_dark_labeled_counts[site]
                )
                bright_unknown = int(
                    quality.held_out_bright_total_counts[site]
                    - quality.held_out_bright_labeled_counts[site]
                )
                if dark_unknown != bright_unknown or (
                    int(quality.held_out_dark_labeled_counts[site])
                    + int(quality.held_out_bright_labeled_counts[site])
                    + dark_unknown
                    != self.diagnostics.test_bracket_count
                ):
                    raise ValueError(
                        "model held-out evidence does not cover the test partition"
                    )
            if np.any(usable & ~expected_reference_sites):
                raise ValueError("model uses a site rejected by reference-label evidence")

            def positive_integer_parameter(name: str) -> int:
                value = model_parameters.get(name)
                if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                    raise ValueError(f"model omits positive {name}")
                return int(value)

            minimum_train = positive_integer_parameter(
                "minimum-train-samples-per-class"
            )
            minimum_test = positive_integer_parameter(
                "minimum-test-samples-per-class"
            )
            confidence = model_parameters.get("held-out-confidence-level")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, Real)
                or not math.isfinite(float(confidence))
                or not 0.0 < float(confidence) < 1.0
            ):
                raise ValueError("model omits held-out confidence level")
            confidence = float(confidence)
            lower_gate = model_parameters.get(
                "minimum-held-out-class-accuracy-lower-bound"
            )
            if (
                isinstance(lower_gate, bool)
                or not isinstance(lower_gate, Real)
                or not math.isfinite(float(lower_gate))
                or not 0.0 <= float(lower_gate) <= 1.0
            ):
                raise ValueError("model omits held-out class lower-bound gate")
            lower_gate = float(lower_gate)
            evidence = quality.held_out_validity.mask
            expected_dark_lower = np.zeros(site_count, dtype=np.float64)
            expected_bright_lower = np.zeros(site_count, dtype=np.float64)
            for site in np.flatnonzero(evidence):
                expected_dark_lower[site] = _one_sided_clopper_pearson_lower_bound(
                    int(quality.held_out_dark_success_counts[site]),
                    int(quality.held_out_dark_total_counts[site]),
                    confidence,
                )
                expected_bright_lower[site] = _one_sided_clopper_pearson_lower_bound(
                    int(quality.held_out_bright_success_counts[site]),
                    int(quality.held_out_bright_total_counts[site]),
                    confidence,
                )
            if not np.allclose(
                quality.held_out_dark_accuracy_lower_bounds,
                expected_dark_lower,
                rtol=1e-12,
                atol=1e-15,
            ) or not np.allclose(
                quality.held_out_bright_accuracy_lower_bounds,
                expected_bright_lower,
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "model Clopper-Pearson evidence differs from success/total counts"
                )
            site_pvalues = np.ones(site_count, dtype=np.float64)
            for site in np.flatnonzero(evidence):
                site_pvalues[site] = max(
                    _one_sided_binomial_superiority_pvalue(
                        int(quality.held_out_dark_success_counts[site]),
                        int(quality.held_out_dark_total_counts[site]),
                        lower_gate,
                    ),
                    _one_sided_binomial_superiority_pvalue(
                        int(quality.held_out_bright_success_counts[site]),
                        int(quality.held_out_bright_total_counts[site]),
                        lower_gate,
                    ),
                )
            candidate_sites = (
                expected_reference_sites
                & evidence
                & (
                    quality.dark_training_sample_counts
                    >= minimum_train
                )
                & (
                    quality.bright_training_sample_counts
                    >= minimum_train
                )
                & (
                    quality.held_out_dark_labeled_counts
                    >= minimum_test
                )
                & (
                    quality.held_out_bright_labeled_counts
                    >= minimum_test
                )
                & (
                    quality.held_out_dark_accuracy_lower_bounds
                    >= lower_gate - 1e-12
                )
                & (
                    quality.held_out_bright_accuracy_lower_bounds
                    >= lower_gate - 1e-12
                )
            )
            acceptance = model_parameters.get("usable-site-acceptance")
            fraction = model_parameters.get("minimum-usable-site-fraction")
            if acceptance == UsableSiteAcceptance.ALL.value:
                required_usable = site_count
                if fraction != 1.0:
                    raise ValueError("ALL acceptance requires canonical fraction one")
                fraction = 1.0
            elif acceptance == UsableSiteAcceptance.MINIMUM_FRACTION.value:
                if (
                    isinstance(fraction, bool)
                    or not isinstance(fraction, Real)
                    or not 0.0 < float(fraction) <= 1.0
                ):
                    raise ValueError("model omits usable-site fraction")
                fraction = float(fraction)
                required_usable = math.ceil(fraction * site_count)
            else:
                raise ValueError("model omits usable-site acceptance policy")
            gate_policy = (
                minimum_train,
                minimum_test,
                confidence,
                lower_gate,
                acceptance,
                fraction,
            )
            if declared_gate_policy is None:
                declared_gate_policy = gate_policy
            elif gate_policy != declared_gate_policy:
                raise ValueError("models declare inconsistent calibration gate policies")
            held_out_records.append(
                _HeldOutReplayRecord(
                    diagnostic=diagnostic,
                    quality=quality,
                    usable=usable,
                    candidate_sites=candidate_sites,
                    site_pvalues=site_pvalues,
                    minimum_train=minimum_train,
                    minimum_test=minimum_test,
                    lower_gate=lower_gate,
                    required_usable=required_usable,
                )
            )

        if declared_gate_policy is None or not held_out_records:
            raise ValueError("calibration result omits held-out model evidence")
        confidence = declared_gate_policy[2]
        joint_certified = _artifact_wide_held_out_rejections(
            np.stack([item.site_pvalues for item in held_out_records], axis=0),
            1.0 - confidence,
        )
        for record, model_certified in zip(
            held_out_records,
            joint_certified,
            strict=True,
        ):
            quality = record.quality
            usable = record.usable
            expected_usable = record.candidate_sites & model_certified
            if not np.array_equal(usable, expected_usable):
                raise ValueError(
                    "model usable sites differ from replayed artifact-wide "
                    "familywise evidence"
                )
            usable_count = int(np.count_nonzero(usable))
            diagnostic = record.diagnostic
            if diagnostic.usable_site_count != usable_count or (
                diagnostic.rejected_site_count != site_count - usable_count
            ):
                raise ValueError("model site-count diagnostic differs from quality")
            if np.any(
                quality.dark_training_sample_counts[usable] < record.minimum_train
            ) or np.any(
                quality.bright_training_sample_counts[usable] < record.minimum_train
            ):
                raise ValueError("usable site violates minimum training evidence")
            if np.any(
                quality.held_out_dark_labeled_counts[usable] < record.minimum_test
            ) or np.any(
                quality.held_out_bright_labeled_counts[usable] < record.minimum_test
            ):
                raise ValueError("usable site violates minimum held-out evidence")
            if np.any(
                quality.held_out_dark_accuracy_lower_bounds[usable]
                < record.lower_gate - 1e-12
            ) or np.any(
                quality.held_out_bright_accuracy_lower_bounds[usable]
                < record.lower_gate - 1e-12
            ):
                raise ValueError("usable site violates class confidence gate")
            if usable_count < record.required_usable:
                raise ValueError("model quality violates usable-site acceptance")
            fidelity = quality.held_out_fidelity[usable]
            class_lower = np.minimum(
                quality.held_out_dark_accuracy_lower_bounds[usable],
                quality.held_out_bright_accuracy_lower_bounds[usable],
            )
            expected = (
                float(np.min(fidelity)),
                float(np.mean(fidelity)),
                float(np.min(class_lower)),
                float(np.mean(class_lower)),
            )
            observed = (
                diagnostic.minimum_fidelity,
                diagnostic.mean_fidelity,
                diagnostic.minimum_class_accuracy_lower_bound,
                diagnostic.mean_class_accuracy_lower_bound,
            )
            if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
                raise ValueError("model evidence diagnostic differs from quality")


@dataclass(frozen=True)
class _BracketPartition:
    train_indices: tuple[int, ...]
    reference_evidence_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class _CalibrationWorkPreparation:
    plan: CalibrationWorkPlan
    brackets: tuple[CalibrationCaptureBracket, ...]
    source_binding: CalibrationSourceBinding
    frame_contract: FrameContract
    partition: _BracketPartition
    numpy_version: str
    scipy_version: str
    backend_digest: str


@dataclass(frozen=True)
class _ReferenceThresholdProposal:
    threshold: float
    left_center: float
    center_spacing: float
    lower_sample_count: int
    upper_sample_count: int
    cluster_separation_rss: float


@dataclass(frozen=True)
class _LatticeResult:
    coordinates_xy: np.ndarray
    diagnostic: SiteDetectionDiagnostic


@dataclass(frozen=True)
class _PeakCandidates:
    coordinates_xy: np.ndarray
    prominences: np.ndarray
    half_prominence_basin_areas: np.ndarray


@dataclass(frozen=True)
class _TrainingResult:
    thresholds: np.ndarray
    occupied_above: np.ndarray
    usable: np.ndarray
    candidate_sites: np.ndarray
    site_pvalues: np.ndarray
    dark_training_counts: np.ndarray
    bright_training_counts: np.ndarray
    held_out_dark_success_counts: np.ndarray
    held_out_dark_total_counts: np.ndarray
    held_out_dark_labeled_counts: np.ndarray
    held_out_bright_success_counts: np.ndarray
    held_out_bright_total_counts: np.ndarray
    held_out_bright_labeled_counts: np.ndarray
    held_out_dark_lower_bounds: np.ndarray
    held_out_bright_lower_bounds: np.ndarray
    held_out_validity: np.ndarray
    held_out_fidelity: np.ndarray


@dataclass(frozen=True)
class _HeldOutReplayRecord:
    diagnostic: ModelAnalysisDiagnostic
    quality: ReadoutModelQuality
    usable: np.ndarray
    candidate_sites: np.ndarray
    site_pvalues: np.ndarray
    minimum_train: int
    minimum_test: int
    lower_gate: float
    required_usable: int


class _RawCapture(Protocol):
    block: DataBlock
    source_cell_schedule: tuple[object, ...]


def _sanitized_build_identity(module: object) -> dict[str, object]:
    try:
        raw = module.show_config(mode="dicts")  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return {"available": False}
    if not isinstance(raw, dict):
        return {"available": False}
    compilers = raw.get("Compilers", {})
    dependencies = raw.get("Build Dependencies", {})
    machine = raw.get("Machine Information", {})
    simd = raw.get("SIMD Extensions", {})

    def selected(mapping: object, names: tuple[str, ...]) -> dict[str, object]:
        if not isinstance(mapping, dict):
            return {}
        return {name: mapping[name] for name in names if name in mapping}

    return {
        "available": True,
        "compilers": {
            name: selected(value, ("name", "version"))
            for name, value in sorted(
                compilers.items() if isinstance(compilers, dict) else ()
            )
        },
        "blas": selected(
            dependencies.get("blas", {}) if isinstance(dependencies, dict) else {},
            ("name", "version", "openblas configuration", "has ilp64"),
        ),
        "lapack": selected(
            dependencies.get("lapack", {}) if isinstance(dependencies, dict) else {},
            ("name", "version", "openblas configuration", "has ilp64"),
        ),
        "machine": selected(
            machine.get("host", {}) if isinstance(machine, dict) else {},
            ("cpu", "family", "endian", "system"),
        ),
        "simd": selected(simd, ("baseline", "found")),
    }


def _numeric_backend() -> tuple[str, str, str]:
    numpy_version = str(np.__version__)
    scipy_version = str(scipy.__version__)
    digest = canonical_digest(
        {
            "schema": "zlc_neutral_atom.CalibrationNumericBackend/v2",
            "numpy": numpy_version,
            "scipy": scipy_version,
            "numpy_build": _sanitized_build_identity(np),
            "scipy_build": _sanitized_build_identity(scipy),
        }
    )
    return numpy_version, scipy_version, digest


def _validate_partition_capacity(
    train_count: int,
    evidence_count: int,
    test_count: int,
    request: CalibrationAnalysisRequest,
    *,
    label: str,
) -> None:
    if train_count < 1 or evidence_count < 1 or test_count < 1:
        raise CalibrationResourceExceeded(
            f"{label} cannot populate train, reference-evidence, and test partitions"
        )
    if train_count < 2 * request.minimum_train_samples_per_class:
        raise CalibrationResourceExceeded(
            f"{label} cannot satisfy minimum train samples per class"
        )
    if test_count < 2 * request.minimum_test_samples_per_class:
        raise CalibrationResourceExceeded(
            f"{label} cannot satisfy minimum test samples per class"
        )
    minimum_outer_evidence = math.ceil(
        math.log2(request.site_count / request.reference_valley_familywise_error_rate)
    )
    if evidence_count // 2 < minimum_outer_evidence:
        raise CalibrationResourceExceeded(
            f"{label} cannot possibly pass the declared familywise "
            "exact-binomial gate"
        )
    minimum_accuracy = request.minimum_held_out_class_accuracy_lower_bound
    smallest_balanced_class = test_count // 2
    held_out_alpha = 1.0 - request.held_out_confidence_level
    held_out_family_size = request.site_count * len(request.model_kinds)
    if minimum_accuracy == 1.0 or (
        minimum_accuracy > 0.0
        and smallest_balanced_class * math.log(minimum_accuracy)
        > math.log(held_out_alpha / held_out_family_size)
    ):
        raise CalibrationResourceExceeded(
            f"{label} cannot possibly pass the held-out familywise "
            "exact-binomial gate"
        )


def _planned_model_resources(
    request: CalibrationAnalysisRequest,
) -> tuple[int, int, int]:
    """Return per-model samples, total samples, and persisted kernel elements."""

    box_area = (2 * request.box.half_width + 1) ** 2
    maximum_sampled = 0
    total_sampled = 0
    kernel_elements = 0
    for kind in request.model_kinds:
        if kind is ReadoutModelKind.BOX:
            sample_area = box_area
        else:
            assert request.psf is not None
            sample_extent = 2 * request.psf.half_width + 1
            if request.psf.background is BackgroundMode.ANNULUS_MEDIAN:
                sample_extent += 2 * request.psf.background_padding
            sample_area = sample_extent**2
            kernel_extent = 2 * request.psf.half_width + 1
            kernel_elements += (
                request.site_count * kernel_extent**2
                if kind is ReadoutModelKind.PER_SITE_PSF
                else kernel_extent**2
            )
        sampled = request.site_count * sample_area
        maximum_sampled = max(maximum_sampled, sampled)
        total_sampled += sampled
    return maximum_sampled, total_sampled, kernel_elements


def _prepare_calibration_work(
    capture: _RawCapture,
    request: CalibrationAnalysisRequest,
    *,
    frozen_numeric_backend: tuple[str, str, str] | None = None,
) -> _CalibrationWorkPreparation:
    """Freeze source-specific work, lineage metadata, and resource bounds.

    A persistent replay supplies the backend identity frozen in the artifact.
    Resource-plan reconstruction must not depend on the package versions of
    the process performing admission: those versions are lineage of the
    original analysis, not a compatibility policy for reading its evidence.
    """

    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    try:
        block = capture.block
    except AttributeError as exc:
        raise TypeError("capture must expose a DataBlock") from exc
    if not isinstance(block, DataBlock):
        raise TypeError("capture.block must be DataBlock")
    schema = block.schema
    policy = request.resource_policy
    source_cells = schema.repeat_axis.size * schema.point_layout.storage_size
    if source_cells > policy.max_source_cells:
        raise CalibrationResourceExceeded("source cells exceed analysis budget")
    if len(schema.cell_schema.data_shape) != 2:
        raise CalibrationAnalysisError("calibration source cell must be exactly named Y,X")
    image_pixels = math.prod(schema.cell_schema.data_shape)
    if image_pixels > policy.max_image_pixels:
        raise CalibrationResourceExceeded("frame pixels exceed analysis budget")
    # Every bracket consumes one row for every selected reference plus one
    # readout row.  This remains cheap and safe for sparse arbitrary layouts.
    selected_rows_per_bracket = len(request.layout.reference_event_indices) + 1
    bracket_upper = source_cells // selected_rows_per_bracket
    if bracket_upper > policy.max_brackets:
        raise CalibrationResourceExceeded("bracket upper bound exceeds analysis budget")
    reference_count = len(request.layout.reference_event_indices)
    reference_upper = bracket_upper * reference_count
    if reference_upper > policy.max_reference_frames:
        raise CalibrationResourceExceeded("reference-frame upper bound exceeds budget")
    if bracket_upper >= 3:
        train_upper = int(math.floor(bracket_upper * request.train_fraction))
        evidence_upper = int(
            math.floor(bracket_upper * request.reference_evidence_fraction)
        )
    else:
        train_upper = bracket_upper
        evidence_upper = 0
    test_upper = bracket_upper - train_upper - evidence_upper
    _validate_partition_capacity(
        train_upper,
        evidence_upper,
        test_upper,
        request,
        label="bracket upper bound",
    )
    modality_work = request.site_count * reference_count * (
        2 * train_upper * max(1, train_upper.bit_length())
        + 3 * evidence_upper
    )
    modality_work += request.site_count * max(1, request.site_count.bit_length())
    nested_test_count = 2 * request.site_count * reference_count
    modality_work += nested_test_count * max(1, nested_test_count.bit_length())
    model_statistical_work = len(request.model_kinds) * request.site_count * (
        2 * train_upper * max(1, train_upper.bit_length())
        + test_upper
        + 256
    )
    held_out_hypothesis_count = len(request.model_kinds) * request.site_count
    model_statistical_work += held_out_hypothesis_count * max(
        1,
        held_out_hypothesis_count.bit_length(),
    )
    modality_work += model_statistical_work
    if modality_work > policy.max_modality_test_work_units:
        raise CalibrationResourceExceeded("reference-modality work exceeds budget")
    reference_valley_diagnostic_count = request.site_count * reference_count
    from .analysis_codec import (
        MAX_ANALYSIS_DIAGNOSTICS_BYTES,
        MAX_DIAGNOSTIC_VECTOR_ENTRIES,
        calibration_analysis_diagnostics_encoding_upper_bound,
        calibration_analysis_diagnostics_encoding_working_upper_bound,
    )
    from .calibration_codec import (
        calibration_artifact_metadata_encoding_upper_bound,
        calibration_artifact_encoding_upper_bound,
        calibration_artifact_encoding_working_upper_bound,
    )

    if (
        reference_valley_diagnostic_count
        > min(
            policy.max_reference_valley_diagnostics,
            MAX_DIAGNOSTIC_VECTOR_ENTRIES,
        )
    ):
        raise CalibrationResourceExceeded(
            "reference-valley diagnostics exceed analysis budget"
        )
    diagnostics_encoding_upper_bound = (
        calibration_analysis_diagnostics_encoding_upper_bound(
            site_count=request.site_count,
            reference_count=reference_count,
            bracket_upper_bound=bracket_upper,
            train_bracket_upper_bound=train_upper,
            reference_evidence_bracket_upper_bound=evidence_upper,
            model_count=len(request.model_kinds),
        )
    )
    if diagnostics_encoding_upper_bound > min(
        MAX_ANALYSIS_DIAGNOSTICS_BYTES,
        policy.artifact_policy.max_artifact_blob_bytes,
    ):
        raise CalibrationResourceExceeded(
            "calibration diagnostics canonical encoding exceeds persistence budget"
        )
    signal_evaluations = request.site_count * (
        train_upper * reference_count
        + evidence_upper * reference_count
        + bracket_upper * reference_count
        + bracket_upper * len(request.model_kinds)
    )
    if signal_evaluations > policy.max_signal_evaluations:
        raise CalibrationResourceExceeded("signal-evaluation upper bound exceeds budget")
    box_area = (2 * request.box.half_width + 1) ** 2
    full_frame_reads = (
        train_upper * reference_count
        + train_upper * reference_count
        + evidence_upper * reference_count
        + bracket_upper * reference_count
        + bracket_upper * len(request.model_kinds)
    )
    sampled = train_upper * reference_count * image_pixels
    sampled += (
        train_upper + evidence_upper + bracket_upper
    ) * reference_count * request.site_count * box_area
    psf_sample_area = 0
    if request.psf is not None:
        extent = 2 * request.psf.half_width + 1
        if request.psf.background is BackgroundMode.ANNULUS_MEDIAN:
            extent += 2 * request.psf.background_padding
        psf_sample_area = extent**2
        sampled += request.site_count * psf_sample_area
    (
        max_model_sampled,
        total_model_sampled,
        planned_kernel_elements,
    ) = _planned_model_resources(request)
    for kind in request.model_kinds:
        if kind is ReadoutModelKind.BOX:
            area = box_area
        else:
            assert request.psf is not None
            extent = 2 * request.psf.half_width + 1
            if request.psf.background is BackgroundMode.ANNULUS_MEDIAN:
                extent += 2 * request.psf.background_padding
            area = extent**2
        model_sampled = request.site_count * area
        sampled += bracket_upper * model_sampled
    if sampled > policy.max_sampled_pixel_operations:
        raise CalibrationResourceExceeded("sampled-pixel upper bound exceeds budget")
    artifact_policy = policy.artifact_policy
    if max_model_sampled > artifact_policy.max_sampled_pixels_per_model:
        raise CalibrationResourceExceeded(
            "planned model sampled pixels exceed artifact resource policy"
        )
    if total_model_sampled > artifact_policy.max_total_sampled_pixels_all_models:
        raise CalibrationResourceExceeded(
            "planned total sampled pixels exceed artifact resource policy"
        )
    psf_extent = 0 if request.psf is None else 2 * request.psf.half_width + 1
    psf_elements = psf_extent**2
    if planned_kernel_elements > artifact_policy.max_kernel_elements:
        raise CalibrationResourceExceeded(
            "planned calibration kernels exceed artifact resource policy"
        )
    dense_work = _ASSIGNMENT_WORK_FACTOR * request.site_count**3
    # E <= 4V for the undirected 8-neighbour graph.  This E log(V) envelope
    # covers sparse MST construction plus the N log(N) forest-edge ordering;
    # both subsequent Kruskal passes are linear and share that order.
    detector_graph_work = (
        _TOPOGRAPHIC_EDGE_COUNT_PER_PIXEL
        * image_pixels
        * max(1, image_pixels.bit_length())
    )
    if detector_graph_work > policy.max_detector_graph_work_units:
        raise CalibrationResourceExceeded("topographic detector work exceeds budget")
    if dense_work > policy.max_dense_assignment_work_units:
        raise CalibrationResourceExceeded("dense lattice-assignment work exceeds budget")
    assignment_scratch = (
        _ASSIGNMENT_SCRATCH_BYTES_PER_PAIR * request.site_count**2
    )
    detector_bytes = _DETECTOR_WORKING_BYTES_PER_PIXEL * image_pixels
    layout_bytes = source_cells + bracket_upper * 768
    if max(
        layout_bytes,
        detector_bytes + assignment_scratch,
    ) > policy.max_working_bytes:
        raise CalibrationResourceExceeded("working-byte lower bound exceeds budget")

    # Only after every schema/count/algorithmic bound passes may the planner
    # resolve the bounded layout and measure source-owned metadata.  A schema-
    # only plan cannot truthfully bound arbitrary canonical camera/source text.
    brackets = request.layout.brackets(schema)
    if len(brackets) > bracket_upper:
        raise CalibrationAnalysisError(
            "resolved brackets exceed frozen work-plan bound"
        )
    source_binding, frame_contract = (
        _derive_calibration_source_binding_with_resolved_brackets(
            capture,
            request.layout,
            resolved_brackets=brackets,
        )
    )
    partition = _freeze_partition(brackets, request)
    _validate_partition_capacity(
        len(partition.train_indices),
        len(partition.reference_evidence_indices),
        len(partition.test_indices),
        request,
        label="resolved bracket partition",
    )
    if frozen_numeric_backend is None:
        numpy_version, scipy_version, backend_digest = _numeric_backend()
    else:
        if (
            type(frozen_numeric_backend) is not tuple
            or len(frozen_numeric_backend) != 3
        ):
            raise TypeError(
                "frozen_numeric_backend must be a three-text tuple"
            )
        numpy_version, scipy_version, backend_digest = frozen_numeric_backend
        for name, value in (
            ("numpy_version", numpy_version),
            ("scipy_version", scipy_version),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if (
            type(backend_digest) is not str
            or len(backend_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in backend_digest
            )
        ):
            raise ValueError(
                "backend_digest must be a lowercase SHA-256 digest"
            )
    placeholder_digest = "f" * 64
    artifact_parameters = tuple(
        CalibrationParameter(name, value)
        for name, value in _artifact_parameter_values(
            request,
            partition_digest=partition.digest,
            train_count=len(partition.train_indices),
            reference_evidence_count=len(partition.reference_evidence_indices),
            test_count=len(partition.test_indices),
            work_plan_digest=placeholder_digest,
            backend_digest=backend_digest,
            numpy_version=numpy_version,
            scipy_version=scipy_version,
        ).items()
    )
    model_parameters = tuple(
        CalibrationParameter(name, value)
        for name, value in _model_parameter_values(
            request,
            partition_digest=partition.digest,
            train_count=len(partition.train_indices),
            reference_evidence_count=len(partition.reference_evidence_indices),
            test_count=len(partition.test_indices),
            backend_digest=backend_digest,
            work_plan_digest=placeholder_digest,
        ).items()
    )
    artifact_metadata_encoding_upper_bound = (
        calibration_artifact_metadata_encoding_upper_bound(
            source_binding=source_binding,
            frame_contract=frame_contract,
            artifact_parameters=artifact_parameters,
            model_parameters=tuple(
                model_parameters for _kind in request.model_kinds
            ),
            model_kinds=request.model_kinds,
            default_model_policy=DefaultModelPolicy(
                _DEFAULT_MODEL_POLICY_ID,
                _DEFAULT_MODEL_POLICY_VERSION,
                default_kind=request.default_model_kind,
            ),
            algorithm_id=CALIBRATION_ANALYSIS_ALGORITHM_ID,
            algorithm_version=CALIBRATION_ANALYSIS_ALGORITHM_VERSION,
        )
    )
    artifact_encoding_upper_bound = calibration_artifact_encoding_upper_bound(
        site_count=request.site_count,
        model_count=len(request.model_kinds),
        kernel_elements=planned_kernel_elements,
        metadata_encoding_upper_bound_bytes=(
            artifact_metadata_encoding_upper_bound
        ),
    )
    if artifact_encoding_upper_bound > artifact_policy.max_artifact_blob_bytes:
        raise CalibrationResourceExceeded(
            "calibration artifact canonical encoding exceeds persistence budget"
        )
    reference_feature_bytes = (
        (train_upper + evidence_upper)
        * reference_count
        * request.site_count
        * 9
    )
    reference_gate_state_bytes = (
        reference_count * request.site_count * 1536
    )
    reference_sort_scratch_bytes = max(train_upper, evidence_upper) * 128
    feature_bytes = max(
        reference_feature_bytes
        + reference_gate_state_bytes
        + reference_sort_scratch_bytes,
        bracket_upper * request.site_count * 20,
    )
    psf_bytes = 4 * planned_kernel_elements * 8 + 8 * psf_elements * 4
    artifact_array_bytes = (
        request.site_count * 24
        + len(request.model_kinds) * request.site_count * 160
        + reference_valley_diagnostic_count * 1024
        + planned_kernel_elements * 8
        + 64 * 1024
    )
    canonical_scratch = max(
        calibration_artifact_encoding_working_upper_bound(
            artifact_array_bytes,
            artifact_metadata_encoding_upper_bound,
        ),
        calibration_analysis_diagnostics_encoding_working_upper_bound(
            diagnostics_encoding_upper_bound
        ),
    )
    working_peak = max(
        layout_bytes,
        detector_bytes + assignment_scratch,
        artifact_array_bytes + feature_bytes,
        artifact_array_bytes + psf_bytes + feature_bytes,
        artifact_array_bytes + canonical_scratch,
    )
    plan = CalibrationWorkPlan(
        source_cell_count=source_cells,
        bracket_upper_bound=bracket_upper,
        train_bracket_upper_bound=train_upper,
        reference_evidence_bracket_upper_bound=evidence_upper,
        reference_frame_upper_bound=reference_upper,
        image_pixel_count=image_pixels,
        full_frame_read_count=full_frame_reads,
        feature_pixel_operations=sampled,
        signal_evaluations=signal_evaluations,
        modality_test_work_units=modality_work,
        reference_valley_diagnostic_count=reference_valley_diagnostic_count,
        diagnostics_encoding_upper_bound_bytes=diagnostics_encoding_upper_bound,
        planned_kernel_elements=planned_kernel_elements,
        maximum_model_sampled_pixels=max_model_sampled,
        total_model_sampled_pixels=total_model_sampled,
        artifact_metadata_encoding_upper_bound_bytes=(
            artifact_metadata_encoding_upper_bound
        ),
        artifact_encoding_upper_bound_bytes=artifact_encoding_upper_bound,
        layout_working_bytes=layout_bytes,
        detector_working_bytes=detector_bytes,
        assignment_scratch_bytes=assignment_scratch,
        feature_working_bytes=feature_bytes,
        psf_working_bytes=psf_bytes,
        artifact_array_bytes=artifact_array_bytes,
        canonical_encoding_scratch_bytes=canonical_scratch,
        working_peak_bytes=working_peak,
        detector_graph_work_units=detector_graph_work,
        dense_assignment_work_units=dense_work,
    )
    if plan.working_peak_bytes > policy.max_working_bytes:
        raise CalibrationResourceExceeded("working-byte upper bound exceeds budget")
    return _CalibrationWorkPreparation(
        plan=plan,
        brackets=brackets,
        source_binding=source_binding,
        frame_contract=frame_contract,
        partition=partition,
        numpy_version=numpy_version,
        scipy_version=scipy_version,
        backend_digest=backend_digest,
    )


def build_calibration_work_plan(
    capture: _RawCapture,
    request: CalibrationAnalysisRequest,
) -> CalibrationWorkPlan:
    """Derive the complete source-specific pre-allocation work contract."""

    return _prepare_calibration_work(capture, request).plan


def _validate_source_schedule(capture: _RawCapture) -> None:
    """Validate exact coverage with a bounded bitset, never a second giant set."""

    try:
        schedule = capture.source_cell_schedule
    except AttributeError as exc:
        raise TypeError("capture must expose source_cell_schedule") from exc
    if not isinstance(schedule, tuple):
        raise TypeError("source_cell_schedule must be an already bounded tuple")
    from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

    schema = capture.block.schema
    repeats = schema.repeat_axis.size
    points = schema.point_layout.storage_size
    total = repeats * points
    if len(schedule) != total:
        raise CalibrationAnalysisError("source_cell_schedule cardinality differs from dataset")
    seen = np.zeros(total, dtype=bool)
    for address in schedule:
        if not isinstance(address, DatasetCellAddress):
            raise TypeError("source_cell_schedule must contain DatasetCellAddress values")
        if address.repeat_index >= repeats or address.point_storage_index >= points:
            raise CalibrationAnalysisError("source_cell_schedule address is out of bounds")
        flat = address.repeat_index * points + address.point_storage_index
        if seen[flat]:
            raise CalibrationAnalysisError("source_cell_schedule contains a duplicate address")
        seen[flat] = True
    if not np.all(seen):
        raise CalibrationAnalysisError("source_cell_schedule omits a raw dataset cell")


class _FrameAccessor:
    """Borrow one immutable DataBlock frame at a time without a Value snapshot."""

    def __init__(self, block: DataBlock) -> None:
        if not isinstance(block, DataBlock):
            raise TypeError("block must be DataBlock")
        if len(block.schema.cell_schema.data_shape) != 2:
            raise CalibrationAnalysisError("calibration frames must be exactly named Y,X")
        if (
            block.values.flags.writeable
            or not block.values.flags.c_contiguous
            or not _is_bytes_backed_read_only(block.values)
        ):
            raise CalibrationAnalysisError(
                "analysis requires an immutable bytes-backed contiguous DataBlock"
            )
        self.block = block
        self._validity = expand_dataset_validity(block.validity, block.schema)
        self._repeat_axis_id = block.schema.repeat_axis.axis_id

    def arrays(
        self,
        bracket: CalibrationCaptureBracket,
        storage_row: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        context = dict(bracket.context_key)
        if self._repeat_axis_id not in context:
            raise CalibrationAnalysisError("calibration bracket omits repeat AxisId")
        repeat = context[self._repeat_axis_id]
        values = self.block.values[repeat, storage_row]
        mask = np.asarray(self._validity[repeat, storage_row])
        if (
            values.flags.writeable
            or not values.flags.c_contiguous
            or not np.shares_memory(values, self.block.values)
        ):
            raise CalibrationAnalysisError("DataBlock frame borrow lost immutable ownership")
        if mask.dtype != np.dtype(bool) or mask.shape != values.shape:
            raise CalibrationAnalysisError("dataset validity cannot align to borrowed frame")
        return values, mask


def _context_tree(context: tuple[tuple[AxisId, int], ...]) -> list[list[object]]:
    return [[axis_id.value, index] for axis_id, index in context]


def _freeze_partition(
    brackets: tuple[CalibrationCaptureBracket, ...],
    request: CalibrationAnalysisRequest,
) -> _BracketPartition:
    if len(brackets) < 3:
        raise CalibrationAnalysisError(
            "calibration requires distinct train, reference-evidence, and test brackets"
        )

    def key(index: int) -> bytes:
        context = ";".join(
            f"{axis_id.value}={value}" for axis_id, value in brackets[index].context_key
        )
        return hashlib.sha256(f"{_PARTITION_SEED}|{context}".encode("utf-8")).digest()

    ordered = tuple(sorted(range(len(brackets)), key=key))
    train_count = int(math.floor(len(ordered) * request.train_fraction))
    evidence_count = int(
        math.floor(len(ordered) * request.reference_evidence_fraction)
    )
    if train_count < 1 or evidence_count < 1:
        raise CalibrationAnalysisError(
            "partition policy leaves train or reference-evidence empty"
        )
    test_start = train_count + evidence_count
    if test_start >= len(ordered):
        raise CalibrationAnalysisError("partition policy leaves held-out test empty")
    train = tuple(sorted(ordered[:train_count]))
    reference_evidence = tuple(sorted(ordered[train_count:test_start]))
    test = tuple(sorted(ordered[test_start:]))
    digest = canonical_digest(
        {
            "schema": "zlc_neutral_atom.CalibrationBracketPartition/v3",
            "source_schema": request.layout.readout_event_axis_id.value,
            "partition_policy": "ALGORITHM_FIXED_HASH_ORDER_V1",
            "train": [_context_tree(brackets[index].context_key) for index in train],
            "reference_evidence": [
                _context_tree(brackets[index].context_key)
                for index in reference_evidence
            ],
            "test": [_context_tree(brackets[index].context_key) for index in test],
        }
    )
    return _BracketPartition(train, reference_evidence, test, digest)


def _training_reference_template(
    accessor: _FrameAccessor,
    brackets: tuple[CalibrationCaptureBracket, ...],
    train_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, float]:
    shape = accessor.block.schema.cell_schema.data_shape
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint64)
    for bracket_index in train_indices:
        bracket = brackets[bracket_index]
        for _event, row in bracket.reference_point_storage_rows:
            image, pixel_validity = accessor.arrays(bracket, row)
            valid = pixel_validity & np.isfinite(image)
            total[valid] += image[valid]
            count[valid] += 1
    valid = count > 0
    if not np.any(valid):
        raise CalibrationAnalysisError("training reference frames contain no valid pixels")
    average = np.zeros(shape, dtype=np.float64)
    average[valid] = total[valid] / count[valid]
    return average, valid, float(np.mean(valid))


def _topographic_prominences(
    smooth: np.ndarray,
    validity: np.ndarray,
    seed_indices: np.ndarray,
    peak_heights: np.ndarray,
    peak_component_floors: np.ndarray,
    validity_component_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 8-neighbour maximin saddles from a maximum-spanning forest."""

    if smooth.ndim != 2 or validity.shape != smooth.shape:
        raise ValueError("topographic image and validity must share one 2D shape")
    if validity.dtype != np.dtype(bool) or not np.any(validity):
        raise ValueError("topographic validity must contain bool valid pixels")
    seeds = np.asarray(seed_indices)
    if seeds.dtype.kind not in "iu" or seeds.ndim != 1 or len(seeds) < 2:
        raise ValueError("topographic seeds must be one integral candidate vector")
    seeds = seeds.astype(np.int64, copy=False)
    heights = np.asarray(peak_heights, dtype=np.float64)
    floors = np.asarray(peak_component_floors, dtype=np.float64)
    if heights.shape != seeds.shape or floors.shape != seeds.shape:
        raise ValueError("topographic peak metadata must align with seeds")
    if (
        np.any(seeds[1:] < 0)
        or np.any(seeds[1:] >= smooth.size)
        or len(np.unique(seeds[1:])) != len(seeds) - 1
    ):
        raise ValueError("topographic peak seeds must be unique in-bounds pixels")
    flat_smooth = smooth.ravel()
    flat_validity = validity.ravel()
    if np.any(~flat_validity[seeds[1:]]):
        raise ValueError("topographic peak seed lies outside validity")
    if np.any(~np.isfinite(heights[1:])) or np.any(~np.isfinite(floors[1:])):
        raise ValueError("topographic peak heights and floors must be finite")
    if not np.array_equal(flat_smooth[seeds[1:]], heights[1:]):
        raise ValueError("topographic peak height differs from its seed pixel")
    component_count = _positive_integer(
        validity_component_count,
        "validity_component_count",
    )

    pixel_ids = np.arange(smooth.size, dtype=np.int64).reshape(smooth.shape)
    edge_rows: list[np.ndarray] = []
    edge_columns: list[np.ndarray] = []
    edge_costs: list[np.ndarray] = []
    high = float(np.max(smooth[validity]))
    low = float(np.min(smooth[validity]))
    span = high - low
    offset = float(np.spacing(span))
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dx >= 0:
            left_x = slice(0, smooth.shape[1] - dx or None)
            right_x = slice(dx, None)
        else:
            left_x = slice(-dx, None)
            right_x = slice(0, dx)
        top_y = slice(0, smooth.shape[0] - dy or None)
        bottom_y = slice(dy, None)
        pair_valid = (
            validity[top_y, left_x]
            & validity[bottom_y, right_x]
        )
        if not np.any(pair_valid):
            continue
        edge_rows.append(pixel_ids[top_y, left_x][pair_valid])
        edge_columns.append(pixel_ids[bottom_y, right_x][pair_valid])
        saddle = np.minimum(
            smooth[top_y, left_x][pair_valid],
            smooth[bottom_y, right_x][pair_valid],
        )
        edge_costs.append((high - saddle) + offset)

    if edge_rows:
        rows = np.concatenate(edge_rows)
        columns = np.concatenate(edge_columns)
        costs = np.concatenate(edge_costs)
        del edge_rows, edge_columns, edge_costs
        graph = coo_matrix(
            (costs, (rows, columns)),
            shape=(smooth.size, smooth.size),
        ).tocsr()
        del rows, columns, costs
        tree = minimum_spanning_tree(graph, overwrite=True).tocoo(copy=False)
        del graph
    else:
        tree = coo_matrix((smooth.size, smooth.size), dtype=np.float64)
    expected_tree_edges = int(np.count_nonzero(validity)) - component_count
    if tree.nnz != expected_tree_edges:
        raise CalibrationAnalysisError(
            "topographic maximum-spanning forest lost valid connectivity"
        )
    tree_rows = np.asarray(tree.row, dtype=np.int64)
    tree_columns = np.asarray(tree.col, dtype=np.int64)
    tree_saddles = np.minimum(
        flat_smooth[tree_rows],
        flat_smooth[tree_columns],
    )
    del tree
    edge_order = np.argsort(-tree_saddles, kind="stable")

    parent = np.arange(smooth.size, dtype=np.int64)
    rank = np.zeros(smooth.size, dtype=np.uint8)
    owner = np.zeros(smooth.size, dtype=np.int32)
    owner[seeds[1:]] = np.arange(1, len(seeds), dtype=np.int32)
    prominences = np.full(len(seeds), np.nan, dtype=np.float64)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    for edge_index in edge_order:
        left_root = find(int(tree_rows[edge_index]))
        right_root = find(int(tree_columns[edge_index]))
        if left_root == right_root:
            continue
        left_owner = int(owner[left_root])
        right_owner = int(owner[right_root])
        merged_owner = left_owner or right_owner
        if left_owner and right_owner:
            if (
                heights[left_owner] > heights[right_owner]
                or (
                    heights[left_owner] == heights[right_owner]
                    and left_owner < right_owner
                )
            ):
                winner, loser = left_owner, right_owner
            else:
                winner, loser = right_owner, left_owner
            saddle = float(tree_saddles[edge_index])
            prominence = float(heights[loser] - saddle)
            tolerance = 128.0 * np.finfo(np.float64).eps * max(
                1.0,
                abs(float(heights[loser])),
                abs(saddle),
            )
            if prominence < -tolerance:
                raise CalibrationAnalysisError(
                    "topographic saddle lies above its regional maximum"
                )
            prominences[loser] = 0.0 if prominence <= tolerance else prominence
            merged_owner = winner
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        owner[left_root] = merged_owner

    for peak in range(1, len(seeds)):
        if np.isnan(prominences[peak]):
            root = find(int(seeds[peak]))
            if int(owner[root]) != peak:
                raise CalibrationAnalysisError(
                    "topographic peak lost without a recorded saddle"
                )
            prominences[peak] = max(0.0, float(heights[peak] - floors[peak]))
    if np.any(~np.isfinite(prominences[1:])):
        raise CalibrationAnalysisError("topographic prominence graph is incomplete")

    # A second Kruskal pass cuts, rather than merges, every edge that would
    # join two peak-owned components.  This produces disjoint maximin
    # catchments from the same physical forest used for the saddle evidence;
    # no IFT label-order or synthetic background marker can split a saddle.
    parent = np.arange(smooth.size, dtype=np.int64)
    rank.fill(0)
    owner.fill(0)
    owner[seeds[1:]] = np.arange(1, len(seeds), dtype=np.int32)
    for edge_index in edge_order:
        left_root = find(int(tree_rows[edge_index]))
        right_root = find(int(tree_columns[edge_index]))
        if left_root == right_root:
            continue
        left_owner = int(owner[left_root])
        right_owner = int(owner[right_root])
        if left_owner and right_owner and left_owner != right_owner:
            continue
        merged_owner = left_owner or right_owner
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        owner[left_root] = merged_owner

    while True:
        compressed = parent[parent]
        if np.array_equal(compressed, parent):
            break
        parent = compressed
    basins_flat = owner[parent]
    basins_flat[~flat_validity] = 0
    if not np.array_equal(basins_flat[seeds[1:]], np.arange(1, len(seeds))):
        raise CalibrationAnalysisError("topographic catchment lost a peak seed")
    return prominences, basins_flat.reshape(smooth.shape)


def _collapse_prominent_maxima(
    average: np.ndarray,
    validity: np.ndarray,
    policy: SiteDetectionPolicy,
) -> _PeakCandidates:
    weights = gaussian_filter(validity.astype(np.float64), policy.smoothing_sigma_pixels)
    numerator = gaussian_filter(
        np.where(validity, average, 0.0),
        policy.smoothing_sigma_pixels,
    )
    smooth = np.zeros_like(numerator)
    usable = weights > 1e-12
    smooth[usable] = numerator[usable] / weights[usable]
    finite = smooth[validity]
    low, high = float(np.min(finite)), float(np.max(finite))
    span = high - low
    if not math.isfinite(span) or span <= max(1e-12, abs(high) * 1e-12):
        raise CalibrationAnalysisError("training reference template has no site contrast")
    validity_components, validity_component_count = label(
        validity,
        structure=np.ones((3, 3), dtype=int),
    )
    flat_validity_components = validity_components.ravel()
    valid_indices = np.flatnonzero(flat_validity_components)
    valid_component_labels = flat_validity_components[valid_indices]
    component_floors = np.full(
        validity_component_count + 1,
        np.inf,
        dtype=np.float64,
    )
    np.minimum.at(
        component_floors,
        valid_component_labels,
        smooth.ravel()[valid_indices],
    )
    local_maximum = maximum_filter(
        np.where(validity, smooth, -np.inf),
        size=3,
        mode="constant",
        cval=-np.inf,
    )
    local_minimum = minimum_filter(
        np.where(validity, smooth, np.inf),
        size=3,
        mode="constant",
        cval=np.inf,
    )
    maxima = validity & (smooth == local_maximum) & (smooth > local_minimum)
    components, component_count = label(maxima, structure=np.ones((3, 3), dtype=int))
    if component_count == 0:
        raise CalibrationAnalysisError("training reference template has no regional maximum")
    flat_components = components.ravel()
    member_indices = np.flatnonzero(flat_components)
    member_labels = flat_components[member_indices]
    peak_heights = np.full(component_count + 1, -np.inf, dtype=np.float64)
    np.maximum.at(peak_heights, member_labels, smooth.ravel()[member_indices])
    seed_indices = np.full(component_count + 1, smooth.size, dtype=np.int64)
    np.minimum.at(seed_indices, member_labels, member_indices)

    required_prominence = policy.minimum_prominence_fraction * span
    peak_component_ids_by_label = flat_validity_components[seed_indices[1:]]
    candidate_component_floors = component_floors[peak_component_ids_by_label]
    # No topographic prominence can exceed peak_height minus the floor of its
    # own 8-connected valid component.  Never let an isolated valid island
    # borrow a lower floor from an unrelated component.
    capable_labels = np.flatnonzero(
        peak_heights[1:] - candidate_component_floors >= required_prominence
    ) + 1
    if capable_labels.size == 0:
        return _PeakCandidates(
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    peak_component_ids = peak_component_ids_by_label[capable_labels - 1]
    peak_component_floors = np.concatenate(
        ([-np.inf], component_floors[peak_component_ids])
    )
    peak_heights = np.concatenate(([-np.inf], peak_heights[capable_labels]))
    seed_indices = np.concatenate(([smooth.size], seed_indices[capable_labels]))
    component_count = int(capable_labels.size)

    # Prominence and catchment ownership must come from the same maximin
    # topology.  In particular, do not seed per-component floors as negative
    # watershed markers: a synthetic floor marker is a barrier that can split
    # a positive basin and manufacture a much deeper saddle.  A component
    # floor is evidence only for the one surviving maximum in that connected
    # component.  Components with no prominence-capable maximum remain basin
    # zero and therefore cannot contribute an admitted area or centroid.
    prominences, basins = _topographic_prominences(
        smooth,
        validity,
        seed_indices,
        peak_heights,
        peak_component_floors,
        validity_component_count,
    )
    thresholds = peak_heights - 0.5 * prominences
    thresholds[0] = np.inf
    basin_thresholds = thresholds[np.maximum(basins, 0)]
    half_mask = validity & (basins > 0) & (smooth >= basin_thresholds)
    areas = np.bincount(
        basins[half_mask],
        minlength=component_count + 1,
    ).astype(np.int64, copy=False)
    touching = np.zeros(component_count + 1, dtype=bool)
    if policy.reject_touching_half_prominence_basins:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if dx >= 0:
                first_x = slice(0, basins.shape[1] - dx or None)
                second_x = slice(dx, None)
            else:
                first_x = slice(-dx, None)
                second_x = slice(0, dx)
            first_y = slice(0, basins.shape[0] - dy or None)
            second_y = slice(dy, None)
            first_labels = basins[first_y, first_x]
            second_labels = basins[second_y, second_x]
            boundary = (
                half_mask[first_y, first_x]
                & half_mask[second_y, second_x]
                & (first_labels != second_labels)
            )
            if np.any(boundary):
                touching[first_labels[boundary]] = True
                touching[second_labels[boundary]] = True
    qualified = (
        (prominences >= required_prominence)
        & (areas >= policy.minimum_half_prominence_basin_area_pixels)
        & ~touching
    )
    qualified[0] = False
    labels_flat = basins.ravel()
    half_flat = half_mask.ravel()
    half_labels = labels_flat[half_flat]
    half_thresholds = thresholds[half_labels]
    weights_for_centroid = np.maximum(
        smooth.ravel()[half_flat] - half_thresholds,
        np.finfo(np.float64).eps,
    )
    yy, xx = np.indices(smooth.shape, dtype=np.float64)
    weight_sums = np.bincount(
        half_labels,
        weights=weights_for_centroid,
        minlength=component_count + 1,
    )
    x_sums = np.bincount(
        half_labels,
        weights=weights_for_centroid * xx.ravel()[half_flat],
        minlength=component_count + 1,
    )
    y_sums = np.bincount(
        half_labels,
        weights=weights_for_centroid * yy.ravel()[half_flat],
        minlength=component_count + 1,
    )
    admitted_labels = np.flatnonzero(qualified)
    points = np.column_stack(
        (
            x_sums[admitted_labels] / weight_sums[admitted_labels],
            y_sums[admitted_labels] / weight_sums[admitted_labels],
        )
    )
    admitted_prominences = prominences[admitted_labels]
    admitted_areas = areas[admitted_labels]
    order = np.lexsort((points[:, 0], points[:, 1], -admitted_prominences))
    points = points[order]
    admitted_prominences = admitted_prominences[order]
    admitted_areas = admitted_areas[order]
    if len(points) > 1:
        nearest = cKDTree(points).query(points, k=2)[0][:, 1]
        if float(np.min(nearest)) < policy.minimum_peak_separation_pixels:
            raise CalibrationAnalysisError("prominent site candidates violate minimum separation")
    return _PeakCandidates(points, admitted_prominences, admitted_areas)


def _fit_affine(ideal: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(len(ideal)), ideal[:, 0], ideal[:, 1]))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design,
        observed,
        rcond=None,
    )
    predicted = design @ coefficients
    return coefficients, predicted


def _second_best_assignment_cost_delta(
    cost: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[float, float | None]:
    source = np.asarray(cost)
    if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
        raise TypeError("assignment cost must have a real numeric dtype")
    matrix = np.asarray(source, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment cost must be one non-empty square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("assignment cost must contain finite non-negative values")
    count = matrix.shape[0]

    def permutation(values: np.ndarray, name: str) -> np.ndarray:
        source_values = np.asarray(values)
        if source_values.ndim != 1 or source_values.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},)")
        if source_values.dtype.kind not in "iu" or source_values.dtype.kind == "b":
            raise TypeError(f"{name} must have an integral dtype")
        normalized = source_values.astype(np.int64, copy=False)
        if not np.array_equal(np.sort(normalized), np.arange(count, dtype=np.int64)):
            raise ValueError(f"{name} must be a permutation of assignment indices")
        return normalized

    row_order = permutation(rows, "rows")
    columns = permutation(cols, "cols")
    assigned = np.empty(count, dtype=np.int64)
    assigned[row_order] = columns
    base_edges = matrix[np.arange(count), assigned]
    best = math.fsum(float(value) for value in base_edges)
    if count == 1:
        return best, None
    weights = matrix[:, assigned] - base_edges[:, None]
    np.fill_diagonal(weights, np.inf)
    distances = weights.copy()
    np.fill_diagonal(distances, 0.0)
    for pivot in range(count):
        via = distances[:, pivot, None] + distances[None, pivot, :]
        np.minimum(distances, via, out=distances)
    finite_weights = weights[np.isfinite(weights)]
    scale = max(1.0, float(np.max(matrix)))
    weight_scale = max(1.0, float(np.max(np.abs(finite_weights))))
    tolerance = 128.0 * np.finfo(np.float64).eps * (
        scale + count * weight_scale
    )
    if float(np.min(np.diag(distances))) < -tolerance:
        raise CalibrationAnalysisError(
            "assignment optimum has a negative alternating cycle"
        )
    closed_walks = weights + distances.T
    delta_raw = float(np.min(closed_walks))
    if not math.isfinite(delta_raw):
        raise CalibrationAnalysisError("assignment has no finite distinct alternative")
    if delta_raw < -tolerance:
        raise CalibrationAnalysisError(
            "assignment second-best delta is materially negative"
        )
    delta = 0.0 if delta_raw <= tolerance else delta_raw
    return best, delta


def _band_separation(
    assigned: np.ndarray,
    rows: int,
    columns: int,
) -> float | None:
    grid = assigned.reshape(rows, columns, 2)
    separations: list[float] = []
    for row in range(rows - 1):
        upper = grid[row, :, 1]
        lower = grid[row + 1, :, 1]
        separations.append(float(np.min(lower) - np.max(upper)))
    for column in range(columns - 1):
        left = grid[:, column, 0]
        right = grid[:, column + 1, 0]
        separations.append(float(np.min(right) - np.max(left)))
    return min(separations) if separations else None


def _assign_unique_affine_lattice(
    peaks: _PeakCandidates,
    request: CalibrationAnalysisRequest,
) -> _LatticeResult:
    candidates_xy = peaks.coordinates_xy
    rows, columns = request.grid_shape_yx
    expected = rows * columns
    if candidates_xy.shape != (expected, 2):
        raise CalibrationAnalysisError(
            f"prominence produced {len(candidates_xy)} candidates; grid requires {expected}"
        )
    # Initial row bands are defined in the declared ROI-local X/Y frame.  This
    # fixes reflections/90-degree symmetries before the affine refinement.
    by_y = np.lexsort((candidates_xy[:, 0], candidates_xy[:, 1]))
    initial_indices: list[int] = []
    for row in range(rows):
        band = by_y[row * columns : (row + 1) * columns]
        initial_indices.extend(
            int(index) for index in band[np.argsort(candidates_xy[band, 0], kind="stable")]
        )
    ideal = np.asarray(
        [(column, row) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    assignment_cols = np.asarray(initial_indices, dtype=np.int64)
    assigned = candidates_xy[assignment_cols]
    coefficients, predicted = _fit_affine(ideal, assigned)
    final_cost: np.ndarray | None = None
    assignment_rows = np.arange(expected, dtype=np.int64)
    for _iteration in range(_MAX_AFFINE_ASSIGNMENT_ITERATIONS):
        differences = predicted[:, None, :] - candidates_xy[None, :, :]
        cost = np.einsum("ijk,ijk->ij", differences, differences)
        next_rows, next_cols = linear_sum_assignment(cost)
        next_rows = next_rows.astype(np.int64, copy=False)
        next_cols = next_cols.astype(np.int64, copy=False)
        if not np.array_equal(next_rows, assignment_rows):
            raise CalibrationAnalysisError("lattice assignment omitted an ideal grid cell")
        if np.array_equal(next_cols, assignment_cols):
            final_cost = cost
            break
        assignment_cols = next_cols
        assigned = candidates_xy[assignment_cols]
        coefficients, predicted = _fit_affine(ideal, assigned)
    if final_cost is None:
        raise CalibrationAnalysisError("affine lattice assignment did not converge")
    assigned = candidates_xy[assignment_cols]
    coefficients, predicted = _fit_affine(ideal, assigned)
    verification_differences = predicted[:, None, :] - candidates_xy[None, :, :]
    verification_cost = np.einsum(
        "ijk,ijk->ij",
        verification_differences,
        verification_differences,
    )
    verification_rows, verification_cols = linear_sum_assignment(verification_cost)
    if not np.array_equal(verification_rows, assignment_rows) or not np.array_equal(
        verification_cols,
        assignment_cols,
    ):
        raise CalibrationAnalysisError("affine lattice assignment changed after final refit")
    final_cost = verification_cost
    residual = float(np.sqrt(np.mean(np.sum((assigned - predicted) ** 2, axis=1))))
    policy = request.detection
    if residual > policy.maximum_lattice_rms_residual_pixels:
        raise CalibrationAnalysisError("site candidates exceed affine lattice residual gate")
    column_step = coefficients[1]
    row_step = coefficients[2]
    active_steps = []
    if columns > 1:
        active_steps.append(float(np.linalg.norm(column_step)))
    if rows > 1:
        active_steps.append(float(np.linalg.norm(row_step)))
    if active_steps and min(active_steps) < policy.minimum_lattice_step_pixels:
        raise CalibrationAnalysisError("affine lattice step is below the declared minimum")
    # Declared row/column identity is top-to-bottom and left-to-right in the
    # ROI-local frame.  A reflected or axis-swapped solution is not equivalent.
    if (columns > 1 and column_step[0] <= 0.0) or (
        rows > 1 and row_step[1] <= 0.0
    ):
        raise CalibrationAnalysisError(
            "affine lattice orientation is inconsistent with grid order"
        )
    sin_angle: float | None = None
    condition_number: float | None = None
    if columns > 1 and rows > 1:
        column_norm = float(np.linalg.norm(column_step))
        row_norm = float(np.linalg.norm(row_step))
        basis = np.column_stack((column_step, row_step))
        sin_angle = min(
            1.0,
            abs(float(np.linalg.det(basis))) / (column_norm * row_norm),
        )
        condition_number = float(np.linalg.cond(basis))
        if (
            not math.isfinite(sin_angle)
            or sin_angle < policy.minimum_affine_sin_angle
        ):
            raise CalibrationAnalysisError("affine lattice basis is nearly collinear")
        if (
            not math.isfinite(condition_number)
            or condition_number > policy.maximum_affine_condition_number
        ):
            raise CalibrationAnalysisError("affine lattice basis is ill-conditioned")
    separation = _band_separation(assigned, rows, columns)
    if separation is not None and separation < policy.minimum_band_separation_pixels:
        raise CalibrationAnalysisError("row/column lattice bands are not uniquely separated")
    _best_cost, assignment_gap = _second_best_assignment_cost_delta(
        final_cost,
        assignment_rows,
        assignment_cols,
    )
    if (
        assignment_gap is not None
        and assignment_gap < policy.minimum_assignment_cost_gap_pixels_squared
    ):
        raise CalibrationAnalysisError("affine lattice has an ambiguous second-best assignment")
    row_major = candidates_xy[assignment_cols]
    if request.grid_order is GridOrder.ROW_MAJOR:
        ordered = row_major
    else:
        indices = [row * columns + column for column in range(columns) for row in range(rows)]
        ordered = row_major[indices]
    immutable = np.frombuffer(
        np.ascontiguousarray(ordered, dtype="<f8").tobytes(),
        dtype="<f8",
    ).reshape(expected, 2)
    immutable.setflags(write=False)
    return _LatticeResult(
        immutable,
        SiteDetectionDiagnostic(
            expected,
            float(np.min(peaks.prominences)),
            int(np.min(peaks.half_prominence_basin_areas)),
            residual,
            separation,
            sin_angle,
            condition_number,
            assignment_gap,
        ),
    )


def _detect_sites(
    average: np.ndarray,
    validity: np.ndarray,
    request: CalibrationAnalysisRequest,
) -> _LatticeResult:
    candidates = _collapse_prominent_maxima(average, validity, request.detection)
    return _assign_unique_affine_lattice(candidates, request)


def _boxes_for_centers(
    centers_xy: np.ndarray,
    *,
    half_width: int,
    image_shape_yx: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape_yx
    extent = 2 * half_width + 1
    boxes = np.empty((len(centers_xy), 4), dtype="<i8")
    for index, (x, y) in enumerate(centers_xy):
        center_x, center_y = int(round(float(x))), int(round(float(y)))
        x0, y0 = center_x - half_width, center_y - half_width
        if x0 < 0 or y0 < 0 or x0 + extent > width or y0 + extent > height:
            raise CalibrationAnalysisError(
                f"site {index} is too close to the frame edge for width {extent}"
            )
        boxes[index] = (x0, y0, extent, extent)
    return boxes


def _psf_kernels(
    average: np.ndarray,
    average_validity: np.ndarray,
    boxes: np.ndarray,
    config: PsfAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extent = 2 * config.half_width + 1
    kernels = np.zeros((len(boxes), extent, extent), dtype=np.float64)
    geometry_valid = np.ones(len(boxes), dtype=bool)
    for site, box in enumerate(boxes):
        x0, y0, width, height = (int(value) for value in box)
        cut = average[y0 : y0 + height, x0 : x0 + width]
        cut_valid = average_validity[y0 : y0 + height, x0 : x0 + width]
        if cut.shape != (extent, extent) or not np.all(cut_valid):
            geometry_valid[site] = False
            continue
        background = _readout_background_from_arrays(
            average,
            average_validity,
            box,
            mode=config.background,
            padding=config.background_padding,
        )
        if background is None:
            geometry_valid[site] = False
            continue
        positive = gaussian_filter(np.maximum(cut - background, 0.0), 0.35)
        total = float(np.sum(positive, dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            geometry_valid[site] = False
            continue
        kernels[site] = positive / total
    impulse = np.zeros((extent, extent), dtype=np.float64)
    impulse[0, 0] = 1.0
    kernels[~geometry_valid] = impulse
    if not np.any(geometry_valid):
        raise CalibrationAnalysisError("no site has a valid training-only PSF template")
    uniform = np.mean(kernels[geometry_valid], axis=0, dtype=np.float64)
    uniform = np.maximum(uniform, 0.0)
    uniform /= np.sum(uniform, dtype=np.float64)
    return kernels, uniform, geometry_valid


def _feature_spec(
    kind: ReadoutModelKind,
    *,
    site_axis_id: AxisId,
    boxes: np.ndarray,
    geometry_validity: np.ndarray,
    request: CalibrationAnalysisRequest,
    per_site_kernels: np.ndarray | None = None,
    uniform_kernel: np.ndarray | None = None,
) -> ReadoutFeatureSpec:
    validity = ComponentValidity((site_axis_id,), geometry_validity)
    if kind is ReadoutModelKind.BOX:
        return ReadoutFeatureSpec(
            kind,
            site_axis_id,
            boxes,
            validity,
            box_reducer=request.box.reducer,
        )
    assert request.psf is not None
    return ReadoutFeatureSpec(
        kind,
        site_axis_id,
        boxes,
        validity,
        per_site_kernels=(
            per_site_kernels if kind is ReadoutModelKind.PER_SITE_PSF else None
        ),
        uniform_kernel=(
            uniform_kernel if kind is ReadoutModelKind.UNIFORM_PSF else None
        ),
        background=request.psf.background,
        background_padding=request.psf.background_padding,
    )


def _reference_threshold_proposal(
    values: np.ndarray,
    minimum_samples_per_class: int,
    minimum_separation_rss: float,
) -> _ReferenceThresholdProposal | None:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    ordered = ordered[np.isfinite(ordered)]
    minimum_samples_per_class = _positive_integer(
        minimum_samples_per_class,
        "minimum_samples_per_class",
    )
    minimum_separation_rss = _finite_real(
        minimum_separation_rss,
        "minimum_separation_rss",
    )
    if minimum_separation_rss <= 0.0:
        raise ValueError("minimum_separation_rss must be positive")
    if (
        ordered.size < 2 * minimum_samples_per_class
        or ordered[0] == ordered[-1]
    ):
        return None
    normalized = (ordered - ordered[0]) / (ordered[-1] - ordered[0])
    cumulative = np.cumsum(normalized, dtype=np.float64)
    left_count = np.arange(1, ordered.size, dtype=np.float64)
    right_count = ordered.size - left_count
    distinct = ordered[:-1] < ordered[1:]
    score = left_count * right_count * (
        (cumulative[-1] - cumulative[:-1]) / right_count
        - cumulative[:-1] / left_count
    ) ** 2
    score[~distinct] = -np.inf
    index = int(np.argmax(score))
    if not np.isfinite(score[index]):
        return None
    left, right = ordered[: index + 1], ordered[index + 1 :]
    if (
        len(left) < minimum_samples_per_class
        or len(right) < minimum_samples_per_class
    ):
        return None
    left_median, right_median = float(np.median(left)), float(np.median(right))
    left_scatter = 1.4826 * float(np.median(np.abs(left - left_median)))
    right_scatter = 1.4826 * float(np.median(np.abs(right - right_median)))
    combined_scatter = math.hypot(left_scatter, right_scatter)
    separation = (
        math.inf
        if combined_scatter == 0.0
        else (right_median - left_median) / combined_scatter
    )
    if separation < minimum_separation_rss:
        return None
    center_spacing = 0.5 * (right_median - left_median)
    threshold = float(
        ordered[index] + 0.5 * (ordered[index + 1] - ordered[index])
    )
    if (
        not math.isfinite(center_spacing)
        or center_spacing <= 0.0
        or not math.isfinite(threshold)
    ):
        return None
    return _ReferenceThresholdProposal(
        threshold,
        left_median,
        center_spacing,
        len(left),
        len(right),
        separation,
    )


def _reference_valley_evidence(
    values: np.ndarray,
    proposal: _ReferenceThresholdProposal,
    validity: np.ndarray | None = None,
) -> ReferenceValleyEvidence:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1:
        raise ValueError("reference evidence must be one-dimensional")
    if validity is None:
        valid = np.ones(observed.shape, dtype=bool)
    else:
        valid = np.asarray(validity)
        if valid.dtype != np.dtype(bool) or valid.shape != observed.shape:
            raise ValueError("reference evidence validity must align to values")
    finite_valid = valid & np.isfinite(observed)
    selected = observed[finite_valid]
    normalized = (selected - proposal.left_center) / proposal.center_spacing
    # One boundary vector and a single (a, b] convention prevent overlaps,
    # gaps, and double-counting when an observation lands on a boundary.
    boundaries = np.asarray((-0.5, 0.5, 1.5, 2.5), dtype=np.float64)
    counts = tuple(
        int(np.count_nonzero((normalized > lower) & (normalized <= upper)))
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    left_count, middle_count, right_count = counts
    outside_count = int(selected.size) - sum(counts)
    invalid_count = int(observed.size - selected.size)

    return ReferenceValleyEvidence(
        int(observed.size),
        left_count,
        middle_count,
        right_count,
        outside_count,
        invalid_count,
    )


def _holm_rejections(pvalues: np.ndarray, familywise_error_rate: float) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm pvalues must be a non-empty vector")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm pvalues must be finite probabilities")
    alpha = _finite_real(familywise_error_rate, "familywise_error_rate")
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_error_rate must lie inside (0, 1)")
    indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((indices, values))
    rejected = np.zeros(values.size, dtype=bool)
    for rank, index in enumerate(order):
        if values[index] > alpha / (values.size - rank):
            break
        rejected[index] = True
    return rejected


def _artifact_wide_held_out_rejections(
    pvalues: np.ndarray,
    familywise_error_rate: float,
) -> np.ndarray:
    """Certify the single persisted model-by-site hypothesis family."""

    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(
            "artifact-wide held-out pvalues must be a non-empty model-by-site matrix"
        )
    return _holm_rejections(
        values.reshape(-1),
        familywise_error_rate,
    ).reshape(values.shape)


def _reference_feature_values(
    accessor: _FrameAccessor,
    brackets: tuple[CalibrationCaptureBracket, ...],
    indices: tuple[int, ...],
    spec: ReadoutFeatureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    reference_count = len(brackets[0].reference_point_storage_rows)
    site_count = spec.boxes_xywh.shape[0]
    values = np.zeros((len(indices), reference_count, site_count))
    validity = np.zeros(values.shape, dtype=bool)
    for position, bracket_index in enumerate(indices):
        bracket = brackets[bracket_index]
        for reference_position, (_event, row) in enumerate(
            bracket.reference_point_storage_rows
        ):
            image, pixel_validity = accessor.arrays(bracket, row)
            features = _extract_readout_features_arrays(
                spec,
                image,
                pixel_validity,
            )
            values[position, reference_position] = features.values
            validity[position, reference_position] = features.validity.mask
    return values, validity


def _learn_reference_thresholds(
    accessor: _FrameAccessor,
    brackets: tuple[CalibrationCaptureBracket, ...],
    partition: _BracketPartition,
    spec: ReadoutFeatureSpec,
    minimum_samples_per_class: int,
    minimum_separation_rss: float,
    familywise_error_rate: float,
) -> tuple[np.ndarray, np.ndarray, tuple[ReferenceValleyDiagnostic, ...]]:
    reference_count = len(brackets[0].reference_point_storage_rows)
    site_count = spec.boxes_xywh.shape[0]
    train_values, train_validity = _reference_feature_values(
        accessor,
        brackets,
        partition.train_indices,
        spec,
    )
    evidence_values, evidence_validity = _reference_feature_values(
        accessor,
        brackets,
        partition.reference_evidence_indices,
        spec,
    )
    complete_reference_sites = np.all(
        train_validity & np.isfinite(train_values),
        axis=(0, 1),
    ) & np.all(
        evidence_validity & np.isfinite(evidence_values),
        axis=(0, 1),
    )
    thresholds = np.zeros((reference_count, site_count), dtype=np.float64)
    threshold_validity = np.zeros((reference_count, site_count), dtype=bool)
    proposals: list[list[_ReferenceThresholdProposal | None]] = [
        [None] * site_count for _reference in range(reference_count)
    ]
    evidences: list[list[ReferenceValleyEvidence]] = [
        [
            ReferenceValleyEvidence(0, 0, 0, 0, 0, 0)
            for _site in range(site_count)
        ]
        for _reference in range(reference_count)
    ]
    pvalues = np.ones((reference_count, site_count), dtype=np.float64)
    for reference in range(reference_count):
        for site in range(site_count):
            selected = train_values[:, reference, site][
                train_validity[:, reference, site]
            ]
            proposal = _reference_threshold_proposal(
                selected,
                minimum_samples_per_class,
                minimum_separation_rss,
            )
            proposals[reference][site] = proposal
            evidence = evidence_values[:, reference, site]
            evidence_mask = evidence_validity[:, reference, site]
            if proposal is None:
                finite_valid_count = int(
                    np.count_nonzero(evidence_mask & np.isfinite(evidence))
                )
                evidences[reference][site] = ReferenceValleyEvidence(
                    int(evidence.size),
                    0,
                    0,
                    0,
                    finite_valid_count,
                    int(evidence.size) - finite_valid_count,
                )
                continue
            valley = _reference_valley_evidence(
                evidence,
                proposal,
                evidence_mask,
            )
            evidences[reference][site] = valley
            pvalues[reference, site] = valley.valley_pvalue
    # A site may supply a frozen binary pseudo-label only when every reference
    # event independently shows the dominant valley.  max(p_ref) is the IUT
    # p-value under arbitrary reference correlation; Holm then controls the
    # family of site decisions without assuming site independence.
    site_pvalues = np.max(pvalues, axis=0)
    primary_sites = _holm_rejections(
        site_pvalues,
        familywise_error_rate,
    )
    # A third physical population can also manufacture a very strong binary
    # Otsu split.  For sites that passed the primary gate, independently freeze
    # one nested proposal inside each outer training cluster and test it on the
    # same untouched evidence partition.  These tests only remove sites, never
    # create authority; Holm therefore makes the availability loss controlled
    # without weakening the primary familywise false-admission guarantee.
    nested_pvalues = np.ones(
        (reference_count, site_count, 2),
        dtype=np.float64,
    )
    nested_evidences: list[list[list[ReferenceValleyEvidence | None]]] = [
        [[None, None] for _site in range(site_count)]
        for _reference in range(reference_count)
    ]
    for reference in range(reference_count):
        for site in np.flatnonzero(primary_sites):
            site = int(site)
            proposal = proposals[reference][site]
            if proposal is None:
                continue
            training = train_values[:, reference, site][
                train_validity[:, reference, site]
            ]
            evidence = evidence_values[:, reference, site]
            evidence_valid = (
                evidence_validity[:, reference, site]
                & np.isfinite(evidence)
            )
            evidence_invalid = ~evidence_valid
            for cluster, (training_mask, evidence_member) in enumerate(
                (
                    (
                        training <= proposal.threshold,
                        evidence_valid & (evidence <= proposal.threshold),
                    ),
                    (
                        training > proposal.threshold,
                        evidence_valid & (evidence > proposal.threshold),
                    ),
                )
            ):
                nested = _reference_threshold_proposal(
                    training[training_mask],
                    minimum_samples_per_class,
                    minimum_separation_rss,
                )
                if nested is None:
                    continue
                # An invalid evidence bracket has unknown parent-cluster
                # membership.  It therefore contributes adverse evidence to
                # both nested screens; double accounting here can only make
                # this reject-only ambiguity screen more conservative.
                nested_members = evidence_member | evidence_invalid
                nested_evidence = _reference_valley_evidence(
                    evidence[nested_members],
                    nested,
                    evidence_validity[:, reference, site][nested_members],
                )
                nested_evidences[reference][site][cluster] = nested_evidence
                nested_pvalues[reference, site, cluster] = (
                    nested_evidence.valley_pvalue
                )
    nested_rejections = _holm_rejections(
        nested_pvalues.reshape(-1),
        familywise_error_rate,
    ).reshape(nested_pvalues.shape)
    ambiguous_sites = np.any(nested_rejections, axis=(0, 2))
    accepted_sites = (
        primary_sites & ~ambiguous_sites & complete_reference_sites
    )
    for reference in range(reference_count):
        for site in np.flatnonzero(accepted_sites):
            proposal = proposals[reference][int(site)]
            if proposal is None:
                continue
            thresholds[reference, site] = proposal.threshold
            threshold_validity[reference, site] = True
    diagnostics: list[ReferenceValleyDiagnostic] = []
    for reference in range(reference_count):
        for site in range(site_count):
            proposal = proposals[reference][site]
            evidence = evidences[reference][site]
            diagnostics.append(
                ReferenceValleyDiagnostic(
                    reference,
                    site,
                    None if proposal is None else proposal.threshold,
                    0 if proposal is None else proposal.lower_sample_count,
                    0 if proposal is None else proposal.upper_sample_count,
                    (
                        None
                        if proposal is None
                        else proposal.cluster_separation_rss
                    ),
                    evidence,
                    nested_evidences[reference][site][0],
                    nested_evidences[reference][site][1],
                    bool(accepted_sites[site]),
                )
            )
    return thresholds, threshold_validity, tuple(diagnostics)


def _reference_labels(
    accessor: _FrameAccessor,
    brackets: tuple[CalibrationCaptureBracket, ...],
    spec: ReadoutFeatureSpec,
    thresholds: np.ndarray,
    threshold_validity: np.ndarray,
    request: CalibrationAnalysisRequest,
) -> tuple[np.ndarray, np.ndarray]:
    site_count = spec.boxes_xywh.shape[0]
    labels = np.zeros((len(brackets), site_count), dtype=bool)
    validity = np.zeros(labels.shape, dtype=bool)
    for bracket_index, bracket in enumerate(brackets):
        decisions = np.zeros((len(bracket.reference_point_storage_rows), site_count), dtype=bool)
        valid = threshold_validity.copy()
        for reference, (_event, row) in enumerate(bracket.reference_point_storage_rows):
            image, pixel_validity = accessor.arrays(bracket, row)
            features = _extract_readout_features_arrays(
                spec,
                image,
                pixel_validity,
            )
            valid[reference] &= features.validity.mask
            decisions[reference] = (
                features.values > thresholds[reference]
                if request.reference_class_orientation
                is ReferenceClassOrientation.ABOVE_IS_OCCUPIED
                else features.values < thresholds[reference]
            )
        all_valid = np.all(valid, axis=0)
        consensus = np.all(decisions == decisions[:1], axis=0)
        selected = all_valid & consensus
        validity[bracket_index] = selected
        labels[bracket_index, selected] = decisions[0, selected]
    return labels, validity


def _extract_short_features(
    accessor: _FrameAccessor,
    brackets: tuple[CalibrationCaptureBracket, ...],
    spec: ReadoutFeatureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    site_count = spec.boxes_xywh.shape[0]
    values = np.zeros((len(brackets), site_count), dtype=np.float64)
    validity = np.zeros(values.shape, dtype=bool)
    for index, bracket in enumerate(brackets):
        image, pixel_validity = accessor.arrays(
            bracket,
            bracket.readout_point_storage_row,
        )
        features = _extract_readout_features_arrays(
            spec,
            image,
            pixel_validity,
        )
        values[index] = features.values
        validity[index] = features.validity.mask
    return values, validity


def _train_and_score_thresholds(
    *,
    values: np.ndarray,
    feature_validity: np.ndarray,
    labels: np.ndarray,
    label_validity: np.ndarray,
    partition: _BracketPartition,
    geometry_validity: np.ndarray,
    request: CalibrationAnalysisRequest,
) -> _TrainingResult:
    site_count = values.shape[1]
    thresholds = np.zeros(site_count, dtype=np.float64)
    occupied_above = np.zeros(site_count, dtype=bool)
    usable = np.zeros(site_count, dtype=bool)
    candidate = np.zeros(site_count, dtype=bool)
    site_pvalues = np.ones(site_count, dtype=np.float64)
    dark_counts = np.zeros(site_count, dtype="<u8")
    bright_counts = np.zeros(site_count, dtype="<u8")
    dark_success_counts = np.zeros(site_count, dtype="<u8")
    dark_total_counts = np.zeros(site_count, dtype="<u8")
    dark_labeled_counts = np.zeros(site_count, dtype="<u8")
    bright_success_counts = np.zeros(site_count, dtype="<u8")
    bright_total_counts = np.zeros(site_count, dtype="<u8")
    bright_labeled_counts = np.zeros(site_count, dtype="<u8")
    dark_lower_bounds = np.zeros(site_count, dtype=np.float64)
    bright_lower_bounds = np.zeros(site_count, dtype=np.float64)
    held_out_validity = np.zeros(site_count, dtype=bool)
    fidelity = np.zeros(site_count, dtype=np.float64)
    train = np.asarray(partition.train_indices, dtype=int)
    test = np.asarray(partition.test_indices, dtype=int)
    for site in range(site_count):
        if not geometry_validity[site]:
            continue
        train_feature_valid = (
            feature_validity[train, site]
            & np.isfinite(values[train, site])
        )
        train_valid = label_validity[train, site] & train_feature_valid
        train_dark = train[train_valid & ~labels[train, site]]
        train_bright = train[train_valid & labels[train, site]]
        dark_counts[site] = len(train_dark)
        bright_counts[site] = len(train_bright)
        if (
            len(train_dark) < request.minimum_train_samples_per_class
            or len(train_bright) < request.minimum_train_samples_per_class
        ):
            continue
        dark_median = float(np.median(values[train_dark, site]))
        bright_median = float(np.median(values[train_bright, site]))
        if not math.isfinite(dark_median + bright_median) or dark_median == bright_median:
            continue
        above = bright_median > dark_median
        threshold = 0.5 * (dark_median + bright_median)
        test_label_valid = label_validity[test, site]
        test_feature_valid = (
            feature_validity[test, site]
            & np.isfinite(values[test, site])
        )
        known_dark_mask = test_label_valid & ~labels[test, site]
        known_bright_mask = test_label_valid & labels[test, site]
        unknown_count = int(np.count_nonzero(~test_label_valid))
        dark_labeled = int(np.count_nonzero(known_dark_mask))
        bright_labeled = int(np.count_nonzero(known_bright_mask))
        if dark_labeled == 0 or bright_labeled == 0:
            continue
        dark_labeled_counts[site] = dark_labeled
        bright_labeled_counts[site] = bright_labeled
        dark_total = dark_labeled + unknown_count
        bright_total = bright_labeled + unknown_count
        dark_evaluable = known_dark_mask & test_feature_valid
        bright_evaluable = known_bright_mask & test_feature_valid
        if above:
            dark_correct = dark_evaluable & (values[test, site] <= threshold)
            bright_correct = bright_evaluable & (values[test, site] > threshold)
        else:
            dark_correct = dark_evaluable & (values[test, site] >= threshold)
            bright_correct = bright_evaluable & (values[test, site] < threshold)
        dark_success = int(np.count_nonzero(dark_correct))
        bright_success = int(np.count_nonzero(bright_correct))
        # Known-label invalid/non-finite features are class failures.  An
        # invalid or disagreeing reference label is unknown and therefore a
        # failure in both class denominators.  Selective missingness can only
        # reduce admission evidence; it can never hide a misclassification.
        held_out = 0.5 * (
            dark_success / dark_total + bright_success / bright_total
        )
        if not math.isfinite(held_out):
            continue
        dark_lower = _one_sided_clopper_pearson_lower_bound(
            dark_success,
            dark_total,
            request.held_out_confidence_level,
        )
        bright_lower = _one_sided_clopper_pearson_lower_bound(
            bright_success,
            bright_total,
            request.held_out_confidence_level,
        )
        dark_success_counts[site] = dark_success
        dark_total_counts[site] = dark_total
        bright_success_counts[site] = bright_success
        bright_total_counts[site] = bright_total
        dark_lower_bounds[site] = dark_lower
        bright_lower_bounds[site] = bright_lower
        held_out_validity[site] = True
        fidelity[site] = held_out
        site_pvalues[site] = max(
            _one_sided_binomial_superiority_pvalue(
                dark_success,
                dark_total,
                request.minimum_held_out_class_accuracy_lower_bound,
            ),
            _one_sided_binomial_superiority_pvalue(
                bright_success,
                bright_total,
                request.minimum_held_out_class_accuracy_lower_bound,
            ),
        )
        if (
            dark_labeled < request.minimum_test_samples_per_class
            or bright_labeled < request.minimum_test_samples_per_class
            or dark_lower < request.minimum_held_out_class_accuracy_lower_bound
            or bright_lower < request.minimum_held_out_class_accuracy_lower_bound
        ):
            continue
        candidate[site] = True
        thresholds[site] = threshold
        occupied_above[site] = above
    return _TrainingResult(
        thresholds=thresholds,
        occupied_above=occupied_above,
        usable=usable,
        candidate_sites=candidate,
        site_pvalues=site_pvalues,
        dark_training_counts=dark_counts,
        bright_training_counts=bright_counts,
        held_out_dark_success_counts=dark_success_counts,
        held_out_dark_total_counts=dark_total_counts,
        held_out_dark_labeled_counts=dark_labeled_counts,
        held_out_bright_success_counts=bright_success_counts,
        held_out_bright_total_counts=bright_total_counts,
        held_out_bright_labeled_counts=bright_labeled_counts,
        held_out_dark_lower_bounds=dark_lower_bounds,
        held_out_bright_lower_bounds=bright_lower_bounds,
        held_out_validity=held_out_validity,
        held_out_fidelity=fidelity,
    )


def _certify_artifact_wide_training_results(
    results: tuple[_TrainingResult, ...],
    *,
    familywise_error_rate: float,
) -> tuple[_TrainingResult, ...]:
    if not results:
        raise ValueError("artifact-wide certification requires model evidence")
    site_count = results[0].site_pvalues.size
    if any(
        item.site_pvalues.shape != (site_count,)
        or item.candidate_sites.shape != (site_count,)
        for item in results
    ):
        raise ValueError("model evidence has inconsistent site geometry")
    certified = _artifact_wide_held_out_rejections(
        np.stack([item.site_pvalues for item in results], axis=0),
        familywise_error_rate,
    )
    admitted: list[_TrainingResult] = []
    for item, model_certified in zip(results, certified, strict=True):
        usable = item.candidate_sites & model_certified
        thresholds = item.thresholds.copy()
        occupied_above = item.occupied_above.copy()
        thresholds[~usable] = 0.0
        occupied_above[~usable] = False
        admitted.append(
            replace(
                item,
                thresholds=thresholds,
                occupied_above=occupied_above,
                usable=usable,
            )
        )
    return tuple(admitted)


def _one_sided_clopper_pearson_lower_bound(
    successes: int,
    total: int,
    confidence_level: float,
) -> float:
    """Exact one-sided binomial lower confidence bound."""

    successes = _nonnegative_integer(successes, "successes")
    total = _positive_integer(total, "total")
    if successes > total:
        raise ValueError("successes cannot exceed total")
    confidence = _finite_real(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly inside (0, 1)")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    result = float(
        beta_distribution.ppf(
            alpha,
            successes,
            total - successes + 1,
        )
    )
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CalibrationAnalysisError("binomial confidence bound is not finite")
    return 0.0 if result == 0.0 else result


def _one_sided_binomial_superiority_pvalue(
    successes: int,
    total: int,
    null_probability: float,
) -> float:
    """Exact p-value for H0: success probability <= declared minimum."""

    successes = _nonnegative_integer(successes, "successes")
    total = _positive_integer(total, "total")
    if successes > total:
        raise ValueError("successes cannot exceed total")
    probability = _finite_real(null_probability, "null_probability")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("null_probability must lie in [0, 1]")
    return float(
        binomtest(
            successes,
            total,
            probability,
            alternative="greater",
        ).pvalue
    )


def _model_parameter_values(
    request: CalibrationAnalysisRequest,
    *,
    partition_digest: str,
    train_count: int,
    reference_evidence_count: int,
    test_count: int,
    backend_digest: str,
    work_plan_digest: str,
) -> dict[str, bool | int | float | str]:
    return {
        "analysis-request-fingerprint": request.fingerprint,
        "bracket-partition-digest": partition_digest,
        "numeric-backend-digest": backend_digest,
        "analysis-work-plan-fingerprint": work_plan_digest,
        "bracket-sampling-assumption": request.bracket_sampling_assumption.value,
        "analysis-planning-assumption": (
            request.analysis_planning_assumption.value
        ),
        "held-out-family-scope": "ARTIFACT_MODEL_SITE",
        "held-out-family-model-count": len(request.model_kinds),
        "held-out-family-hypothesis-count": (
            len(request.model_kinds) * request.site_count
        ),
        "train-bracket-count": train_count,
        "reference-evidence-bracket-count": reference_evidence_count,
        "test-bracket-count": test_count,
        "minimum-train-samples-per-class": (
            request.minimum_train_samples_per_class
        ),
        "minimum-test-samples-per-class": request.minimum_test_samples_per_class,
        "held-out-confidence-level": request.held_out_confidence_level,
        "minimum-held-out-class-accuracy-lower-bound": (
            request.minimum_held_out_class_accuracy_lower_bound
        ),
        "usable-site-acceptance": request.usable_site_acceptance.value,
        "minimum-usable-site-fraction": request.minimum_usable_site_fraction,
    }


def _artifact_parameter_values(
    request: CalibrationAnalysisRequest,
    *,
    partition_digest: str,
    train_count: int,
    reference_evidence_count: int,
    test_count: int,
    work_plan_digest: str,
    backend_digest: str,
    numpy_version: str,
    scipy_version: str,
) -> dict[str, bool | int | float | str]:
    return {
        "analysis-request-fingerprint": request.fingerprint,
        "bracket-partition-digest": partition_digest,
        "analysis-work-plan-fingerprint": work_plan_digest,
        "grid-rows": request.grid_shape_yx[0],
        "grid-columns": request.grid_shape_yx[1],
        "grid-order": request.grid_order.value,
        "numeric-backend-digest": backend_digest,
        "numpy-version": numpy_version,
        "scipy-version": scipy_version,
        "strict-reference-consensus": True,
        "bracket-sampling-assumption": request.bracket_sampling_assumption.value,
        "analysis-planning-assumption": (
            request.analysis_planning_assumption.value
        ),
        "held-out-family-scope": "ARTIFACT_MODEL_SITE",
        "held-out-family-model-count": len(request.model_kinds),
        "held-out-family-hypothesis-count": (
            len(request.model_kinds) * request.site_count
        ),
        "reference-label-source": request.reference_label_source.value,
        "reference-class-orientation": request.reference_class_orientation.value,
        "reference-valley-gate-id": _REFERENCE_VALLEY_GATE_ID,
        "reference-valley-gate-version": _REFERENCE_VALLEY_GATE_VERSION,
        "reference-statistical-unit": "BRACKET",
        "reference-evidence-assumption": _REFERENCE_EVIDENCE_ASSUMPTION,
        "reference-ambiguity-gate-id": _REFERENCE_AMBIGUITY_GATE_ID,
        "reference-ambiguity-gate-version": _REFERENCE_AMBIGUITY_GATE_VERSION,
        "train-bracket-count": train_count,
        "reference-evidence-bracket-count": reference_evidence_count,
        "test-bracket-count": test_count,
        "minimum-reference-cluster-separation-rss": (
            request.minimum_reference_cluster_separation_rss
        ),
        "minimum-reference-proposal-samples-per-class": (
            request.minimum_train_samples_per_class
        ),
        "reference-valley-familywise-error-rate": (
            request.reference_valley_familywise_error_rate
        ),
    }


def _model_header(
    kind: ReadoutModelKind,
    *,
    training: _TrainingResult,
    frame_contract: FrameContract,
    site_map: SiteMap,
    request: CalibrationAnalysisRequest,
    partition: _BracketPartition,
    backend_digest: str,
    work_plan_digest: str,
) -> ReadoutModelHeader:
    usable_count = int(np.count_nonzero(training.usable))
    if request.usable_site_acceptance is UsableSiteAcceptance.ALL:
        required_usable = request.site_count
    else:
        required_usable = math.ceil(
            request.minimum_usable_site_fraction * request.site_count
        )
    if usable_count < required_usable:
        raise CalibrationAnalysisError(
            f"{kind.value} admitted {usable_count}/{request.site_count} sites; "
            f"quality policy requires at least {required_usable}"
        )
    axis_id = site_map.site_axis.axis_id
    quality = ReadoutModelQuality(
        axis_id,
        ComponentValidity((axis_id,), training.usable),
        training.dark_training_counts,
        training.bright_training_counts,
        training.held_out_dark_success_counts,
        training.held_out_dark_total_counts,
        training.held_out_dark_labeled_counts,
        training.held_out_bright_success_counts,
        training.held_out_bright_total_counts,
        training.held_out_bright_labeled_counts,
        training.held_out_dark_lower_bounds,
        training.held_out_bright_lower_bounds,
        training.held_out_fidelity,
        ComponentValidity((axis_id,), training.held_out_validity),
        _QUALITY_GATE_ID,
        _QUALITY_GATE_VERSION,
        True,
    )
    model_id = _MODEL_ID_BY_KIND[kind]
    return ReadoutModelHeader(
        model_id,
        _MODEL_VERSION,
        frame_contract.fingerprint,
        site_map.fingerprint,
        axis_id,
        training.thresholds,
        training.occupied_above,
        quality,
        tuple(
            CalibrationParameter(name, value)
            for name, value in _model_parameter_values(
                request,
                partition_digest=partition.digest,
                train_count=len(partition.train_indices),
                reference_evidence_count=len(
                    partition.reference_evidence_indices
                ),
                test_count=len(partition.test_indices),
                backend_digest=backend_digest,
                work_plan_digest=work_plan_digest,
            ).items()
        ),
    )


def _canonical_boxes(boxes: np.ndarray, usable: np.ndarray, *, psf: bool) -> np.ndarray:
    result = np.asarray(boxes, dtype="<i8").copy()
    if psf:
        width, height = int(result[0, 2]), int(result[0, 3])
        result[~usable] = (0, 0, width, height)
    else:
        result[~usable] = (0, 0, 1, 1)
    return result


def validate_calibration_analysis_contract(
    result: CalibrationAnalysisResult,
    request: CalibrationAnalysisRequest,
    work_plan: CalibrationWorkPlan,
    *,
    source_brackets: tuple[CalibrationCaptureBracket, ...] | None = None,
) -> CalibrationAnalysisResult:
    """Replay every request-owned declaration at a repository trust boundary."""

    if not isinstance(result, CalibrationAnalysisResult):
        raise TypeError("result must be CalibrationAnalysisResult")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if not isinstance(work_plan, CalibrationWorkPlan):
        raise TypeError("work_plan must be CalibrationWorkPlan")
    artifact = result.artifact
    diagnostics = result.diagnostics
    if artifact.source_binding.layout != request.layout:
        raise ValueError("artifact capture layout differs from analysis request")
    site_count = request.site_count
    reference_count = len(request.layout.reference_event_indices)
    if artifact.site_map.site_axis.size != site_count:
        raise ValueError("artifact site count differs from requested grid")
    if diagnostics.bracket_count != artifact.source_binding.bracket_count:
        raise ValueError("diagnostic bracket count differs from source binding")

    train_count = int(math.floor(diagnostics.bracket_count * request.train_fraction))
    evidence_count = int(
        math.floor(
            diagnostics.bracket_count * request.reference_evidence_fraction
        )
    )
    test_count = diagnostics.bracket_count - train_count - evidence_count
    _validate_partition_capacity(
        train_count,
        evidence_count,
        test_count,
        request,
        label="persisted bracket population",
    )
    if (
        diagnostics.train_bracket_count != train_count
        or diagnostics.reference_evidence_bracket_count != evidence_count
        or diagnostics.test_bracket_count != test_count
    ):
        raise ValueError("diagnostic partition counts differ from request policy")
    if source_brackets is not None:
        validate_calibration_partition_against_source(
            diagnostics,
            request,
            source_brackets,
        )

    if work_plan.reference_valley_diagnostic_count != site_count * reference_count:
        raise ValueError("work plan reference diagnostic count differs from request")
    from .analysis_codec import (
        calibration_analysis_diagnostics_encoding_upper_bound,
    )

    expected_diagnostics_bound = (
        calibration_analysis_diagnostics_encoding_upper_bound(
            site_count=site_count,
            reference_count=reference_count,
            bracket_upper_bound=work_plan.bracket_upper_bound,
            train_bracket_upper_bound=work_plan.train_bracket_upper_bound,
            reference_evidence_bracket_upper_bound=(
                work_plan.reference_evidence_bracket_upper_bound
            ),
            model_count=len(request.model_kinds),
        )
    )
    if work_plan.diagnostics_encoding_upper_bound_bytes != (
        expected_diagnostics_bound
    ):
        raise ValueError("work plan diagnostics encoding bound is not canonical")
    (
        expected_maximum_sampled,
        expected_total_sampled,
        expected_kernel_elements,
    ) = _planned_model_resources(request)
    if work_plan.maximum_model_sampled_pixels != expected_maximum_sampled:
        raise ValueError("work plan per-model sampled pixels are not canonical")
    if work_plan.total_model_sampled_pixels != expected_total_sampled:
        raise ValueError("work plan total sampled pixels are not canonical")
    if work_plan.planned_kernel_elements != expected_kernel_elements:
        raise ValueError("work plan kernel elements are not canonical")
    artifact_values = {item.name: item.value for item in artifact.parameters}
    backend_digest = artifact_values.get("numeric-backend-digest")
    numpy_version = artifact_values.get("numpy-version")
    scipy_version = artifact_values.get("scipy-version")
    if not isinstance(backend_digest, str):
        raise ValueError("artifact omits numeric backend digest")
    if not isinstance(numpy_version, str) or not isinstance(scipy_version, str):
        raise ValueError("artifact omits numeric backend versions")
    expected_artifact_values = _artifact_parameter_values(
        request,
        partition_digest=diagnostics.partition_digest,
        train_count=train_count,
        reference_evidence_count=evidence_count,
        test_count=test_count,
        work_plan_digest=work_plan.fingerprint,
        backend_digest=backend_digest,
        numpy_version=numpy_version,
        scipy_version=scipy_version,
    )
    if not _typed_parameter_maps_equal(artifact_values, expected_artifact_values):
        raise ValueError("artifact parameters differ from frozen analysis request")

    expected_policy = DefaultModelPolicy(
        _DEFAULT_MODEL_POLICY_ID,
        _DEFAULT_MODEL_POLICY_VERSION,
        default_kind=request.default_model_kind,
    )
    if artifact.default_model_policy != expected_policy:
        raise ValueError("artifact default model policy differs from request")
    if artifact.required_model_kinds != request.model_kinds:
        raise ValueError("artifact required model kinds differ from request")
    if tuple(model.kind for model in artifact.models) != request.model_kinds:
        raise ValueError("artifact model kinds differ from request")

    box_geometry = _boxes_for_centers(
        artifact.site_map.coordinates_xy,
        half_width=request.box.half_width,
        image_shape_yx=artifact.frame_contract.frame_schema.data_shape,
    )
    psf_geometry = None
    if request.psf is not None:
        psf_geometry = _boxes_for_centers(
            artifact.site_map.coordinates_xy,
            half_width=request.psf.half_width,
            image_shape_yx=artifact.frame_contract.frame_schema.data_shape,
        )
    expected_model_values = _model_parameter_values(
        request,
        partition_digest=diagnostics.partition_digest,
        train_count=train_count,
        reference_evidence_count=evidence_count,
        test_count=test_count,
        backend_digest=backend_digest,
        work_plan_digest=work_plan.fingerprint,
    )
    from .calibration_codec import (
        calibration_artifact_encoding_upper_bound,
        calibration_artifact_metadata_encoding_upper_bound,
    )

    expected_metadata_bound = calibration_artifact_metadata_encoding_upper_bound(
        source_binding=artifact.source_binding,
        frame_contract=artifact.frame_contract,
        artifact_parameters=tuple(
            CalibrationParameter(name, value)
            for name, value in expected_artifact_values.items()
        ),
        model_parameters=tuple(
            tuple(
                CalibrationParameter(name, value)
                for name, value in expected_model_values.items()
            )
            for _kind in request.model_kinds
        ),
        model_kinds=request.model_kinds,
        default_model_policy=expected_policy,
        algorithm_id=CALIBRATION_ANALYSIS_ALGORITHM_ID,
        algorithm_version=CALIBRATION_ANALYSIS_ALGORITHM_VERSION,
    )
    if work_plan.artifact_metadata_encoding_upper_bound_bytes != (
        expected_metadata_bound
    ):
        raise ValueError("work plan artifact metadata bound is not canonical")
    expected_artifact_bound = calibration_artifact_encoding_upper_bound(
        site_count=site_count,
        model_count=len(request.model_kinds),
        kernel_elements=expected_kernel_elements,
        metadata_encoding_upper_bound_bytes=expected_metadata_bound,
    )
    if work_plan.artifact_encoding_upper_bound_bytes != expected_artifact_bound:
        raise ValueError("work plan artifact encoding bound is not canonical")
    for model in artifact.models:
        if (
            model.header.model_id != _MODEL_ID_BY_KIND[model.kind]
            or model.header.model_version != _MODEL_VERSION
        ):
            raise ValueError("model identity differs from current analysis contract")
        model_values = {
            item.name: item.value for item in model.header.parameters
        }
        if not _typed_parameter_maps_equal(model_values, expected_model_values):
            raise ValueError("model parameters differ from frozen analysis request")
        usable = model.header.quality.usable_sites.mask
        if isinstance(model, BoxReadoutModel):
            if model.reducer is not request.box.reducer:
                raise ValueError("BOX reducer differs from analysis request")
            expected_boxes = _canonical_boxes(box_geometry, usable, psf=False)
        else:
            if request.psf is None or psf_geometry is None:
                raise ValueError("PSF model exists without requested PSF configuration")
            if (
                model.background is not request.psf.background
                or model.background_padding != request.psf.background_padding
            ):
                raise ValueError("PSF background differs from analysis request")
            expected_boxes = _canonical_boxes(psf_geometry, usable, psf=True)
            extent = 2 * request.psf.half_width + 1
            if isinstance(model, PerSitePsfReadoutModel):
                if model.kernels.shape[1:] != (extent, extent):
                    raise ValueError("per-site PSF kernel shape differs from request")
            elif isinstance(model, UniformPsfReadoutModel):
                if model.kernel.shape != (extent, extent):
                    raise ValueError("uniform PSF kernel shape differs from request")
            else:  # pragma: no cover - CalibrationArtifact already closes the union
                raise TypeError("artifact contains an unknown readout model")
        if not np.array_equal(model.boxes_xywh, expected_boxes):
            raise ValueError("model extraction geometry differs from analysis request")
    return result


def validate_calibration_partition_against_source(
    diagnostics: CalibrationAnalysisDiagnostics,
    request: CalibrationAnalysisRequest,
    source_brackets: tuple[CalibrationCaptureBracket, ...],
) -> None:
    """Rejoin persisted partition lineage to one resolved source capture.

    Scientific replay belongs to :func:`validate_calibration_analysis_contract`.
    Admission has already performed that replay while loading persistent
    evidence, so its source join needs only this bounded partition witness.
    """

    if not isinstance(diagnostics, CalibrationAnalysisDiagnostics):
        raise TypeError("diagnostics must be CalibrationAnalysisDiagnostics")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if not isinstance(source_brackets, tuple) or any(
        not isinstance(item, CalibrationCaptureBracket) for item in source_brackets
    ):
        raise TypeError("source_brackets must be a tuple of CalibrationCaptureBracket")
    if len(source_brackets) != diagnostics.bracket_count:
        raise ValueError("source bracket population differs from diagnostics")
    if _freeze_partition(source_brackets, request).digest != diagnostics.partition_digest:
        raise ValueError("persisted partition differs from source brackets")


def _build_model(
    kind: ReadoutModelKind,
    *,
    feature_spec: ReadoutFeatureSpec,
    training: _TrainingResult,
    frame_contract: FrameContract,
    site_map: SiteMap,
    request: CalibrationAnalysisRequest,
    partition: _BracketPartition,
    backend_digest: str,
    work_plan_digest: str,
) -> tuple[ReadoutModel, ModelAnalysisDiagnostic]:
    header = _model_header(
        kind,
        training=training,
        frame_contract=frame_contract,
        site_map=site_map,
        request=request,
        partition=partition,
        backend_digest=backend_digest,
        work_plan_digest=work_plan_digest,
    )
    boxes = _canonical_boxes(
        feature_spec.boxes_xywh,
        training.usable,
        psf=kind is not ReadoutModelKind.BOX,
    )
    if kind is ReadoutModelKind.BOX:
        model: ReadoutModel = BoxReadoutModel(header, boxes, request.box.reducer)
    elif kind is ReadoutModelKind.PER_SITE_PSF:
        assert request.psf is not None and feature_spec.per_site_kernels is not None
        kernels = feature_spec.per_site_kernels.copy()
        impulse = np.zeros(kernels.shape[1:], dtype="<f8")
        impulse[0, 0] = 1.0
        kernels[~training.usable] = impulse
        model = PerSitePsfReadoutModel(
            header,
            boxes,
            kernels,
            request.psf.background,
            request.psf.background_padding,
        )
    else:
        assert request.psf is not None and feature_spec.uniform_kernel is not None
        model = UniformPsfReadoutModel(
            header,
            boxes,
            feature_spec.uniform_kernel,
            request.psf.background,
            request.psf.background_padding,
        )
    admitted = training.held_out_fidelity[training.usable]
    admitted_lower = np.minimum(
        training.held_out_dark_lower_bounds[training.usable],
        training.held_out_bright_lower_bounds[training.usable],
    )
    return model, ModelAnalysisDiagnostic(
        kind,
        int(np.count_nonzero(training.usable)),
        int(request.site_count - np.count_nonzero(training.usable)),
        float(np.min(admitted)),
        float(np.mean(admitted)),
        float(np.min(admitted_lower)),
        float(np.mean(admitted_lower)),
    )


def analyze_calibration(
    capture: _RawCapture,
    request: CalibrationAnalysisRequest,
) -> CalibrationAnalysisResult:
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    try:
        block = capture.block
    except AttributeError as exc:
        raise TypeError("capture must be a resolved raw CaptureArtifact") from exc
    if not isinstance(block, DataBlock):
        raise TypeError("capture.block must be DataBlock")

    # Preparation performs every cheap schema/count bound before resolving the
    # bounded layout or measuring source-owned metadata.
    preparation = _prepare_calibration_work(capture, request)
    work_plan = preparation.plan
    brackets = preparation.brackets
    _validate_source_schedule(capture)
    source_binding = preparation.source_binding
    frame_contract = preparation.frame_contract
    partition = preparation.partition
    accessor = _FrameAccessor(block)
    average, average_validity, valid_fraction = _training_reference_template(
        accessor,
        brackets,
        partition.train_indices,
    )
    lattice = _detect_sites(average, average_validity, request)
    centers = lattice.coordinates_xy
    site_axis = AxisSpec(
        AxisId("readout-site"),
        "readout site",
        SITE,
        request.site_count,
        coordinates=tuple(range(request.site_count)),
    )
    numpy_version = preparation.numpy_version
    scipy_version = preparation.scipy_version
    backend_digest = preparation.backend_digest
    template_hasher = hashlib.sha256()
    template_hasher.update(np.ascontiguousarray(average, dtype="<f8").tobytes())
    template_hasher.update(
        np.ascontiguousarray(average_validity, dtype=np.uint8).tobytes()
    )
    detection_lineage = canonical_digest(
        {
            "schema": "zlc_neutral_atom.SiteDetectionLineage/v3",
            "source_binding": source_binding.bracket_witness_digest,
            "partition": partition.digest,
            "request": request.fingerprint,
            "template_digest": template_hasher.hexdigest(),
            "numeric_backend": backend_digest,
            "work_plan": work_plan.fingerprint,
            "algorithm": CALIBRATION_ANALYSIS_ALGORITHM_ID,
            "version": CALIBRATION_ANALYSIS_ALGORITHM_VERSION,
        }
    )
    site_map = SiteMap(
        site_axis,
        centers,
        frame_contract.coordinate_frame,
        ComponentValidity((site_axis.axis_id,), np.ones(request.site_count, dtype=bool)),
        detection_lineage,
    )
    box_boxes = _boxes_for_centers(
        centers,
        half_width=request.box.half_width,
        image_shape_yx=frame_contract.frame_schema.data_shape,
    )
    all_sites = np.ones(request.site_count, dtype=bool)
    reference_spec = _feature_spec(
        ReadoutModelKind.BOX,
        site_axis_id=site_axis.axis_id,
        boxes=box_boxes,
        geometry_validity=all_sites,
        request=request,
    )
    (
        reference_thresholds,
        reference_threshold_validity,
        reference_valley_diagnostics,
    ) = _learn_reference_thresholds(
        accessor,
        brackets,
        partition,
        reference_spec,
        request.minimum_train_samples_per_class,
        request.minimum_reference_cluster_separation_rss,
        request.reference_valley_familywise_error_rate,
    )
    labels, label_validity = _reference_labels(
        accessor,
        brackets,
        reference_spec,
        reference_thresholds,
        reference_threshold_validity,
        request,
    )

    psf_boxes: np.ndarray | None = None
    per_site_kernels: np.ndarray | None = None
    uniform_kernel: np.ndarray | None = None
    psf_geometry: np.ndarray | None = None
    if request.psf is not None:
        psf_boxes = _boxes_for_centers(
            centers,
            half_width=request.psf.half_width,
            image_shape_yx=frame_contract.frame_schema.data_shape,
        )
        per_site_kernels, uniform_kernel, psf_geometry = _psf_kernels(
            average,
            average_validity,
            psf_boxes,
            request.psf,
        )

    prepared_models: list[
        tuple[ReadoutModelKind, ReadoutFeatureSpec, _TrainingResult]
    ] = []
    for kind in request.model_kinds:
        if kind is ReadoutModelKind.BOX:
            spec = reference_spec
        else:
            assert (
                psf_boxes is not None
                and per_site_kernels is not None
                and uniform_kernel is not None
                and psf_geometry is not None
            )
            spec = _feature_spec(
                kind,
                site_axis_id=site_axis.axis_id,
                boxes=psf_boxes,
                geometry_validity=psf_geometry,
                request=request,
                per_site_kernels=per_site_kernels,
                uniform_kernel=uniform_kernel,
            )
        values, validity = _extract_short_features(accessor, brackets, spec)
        training = _train_and_score_thresholds(
            values=values,
            feature_validity=validity,
            labels=labels,
            label_validity=label_validity,
            partition=partition,
            geometry_validity=spec.site_validity.mask,
            request=request,
        )
        prepared_models.append((kind, spec, training))

    admitted_training = _certify_artifact_wide_training_results(
        tuple(item[2] for item in prepared_models),
        familywise_error_rate=1.0 - request.held_out_confidence_level,
    )
    models: list[ReadoutModel] = []
    diagnostics: list[ModelAnalysisDiagnostic] = []
    for (kind, spec, _evidence), training in zip(
        prepared_models,
        admitted_training,
        strict=True,
    ):
        model, diagnostic = _build_model(
            kind,
            feature_spec=spec,
            training=training,
            frame_contract=frame_contract,
            site_map=site_map,
            request=request,
            partition=partition,
            backend_digest=backend_digest,
            work_plan_digest=work_plan.fingerprint,
        )
        models.append(model)
        diagnostics.append(diagnostic)

    artifact = CalibrationArtifact(
        source_binding,
        frame_contract,
        site_map,
        tuple(models),
        CalibrationStage.COMPLETE,
        request.model_kinds,
        DefaultModelPolicy(
            _DEFAULT_MODEL_POLICY_ID,
            _DEFAULT_MODEL_POLICY_VERSION,
            default_kind=request.default_model_kind,
        ),
        CALIBRATION_ANALYSIS_ALGORITHM_ID,
        CALIBRATION_ANALYSIS_ALGORITHM_VERSION,
        tuple(
            CalibrationParameter(name, value)
            for name, value in _artifact_parameter_values(
                request,
                partition_digest=partition.digest,
                train_count=len(partition.train_indices),
                reference_evidence_count=len(
                    partition.reference_evidence_indices
                ),
                test_count=len(partition.test_indices),
                work_plan_digest=work_plan.fingerprint,
                backend_digest=backend_digest,
                numpy_version=numpy_version,
                scipy_version=scipy_version,
            ).items()
        ),
    )
    validate_calibration_artifact_resources(
        artifact,
        request.resource_policy.artifact_policy,
    )
    dark_counts = tuple(
        int(np.count_nonzero(label_validity[:, site] & ~labels[:, site]))
        for site in range(request.site_count)
    )
    bright_counts = tuple(
        int(np.count_nonzero(label_validity[:, site] & labels[:, site]))
        for site in range(request.site_count)
    )
    result_diagnostics = CalibrationAnalysisDiagnostics(
        len(brackets),
        len(partition.train_indices),
        len(partition.reference_evidence_indices),
        len(partition.test_indices),
        partition.digest,
        len(brackets) * len(request.layout.reference_event_indices),
        valid_fraction,
        dark_counts,
        bright_counts,
        reference_valley_diagnostics,
        lattice.diagnostic,
        tuple(diagnostics),
    )
    return CalibrationAnalysisResult(artifact, result_diagnostics)


__all__ = [
    "BoxAnalysisConfig",
    "CalibrationAnalysisDiagnostics",
    "CalibrationAnalysisError",
    "CalibrationAnalysisRequest",
    "CalibrationAnalysisPlanningAssumption",
    "CalibrationAnalysisResourcePolicy",
    "CalibrationAnalysisResult",
    "CalibrationBracketSamplingAssumption",
    "CALIBRATION_ANALYSIS_ALGORITHM_ID",
    "CALIBRATION_ANALYSIS_ALGORITHM_VERSION",
    "CalibrationWorkPlan",
    "GridOrder",
    "ModelAnalysisDiagnostic",
    "PsfAnalysisConfig",
    "ReferenceClassOrientation",
    "ReferenceLabelSource",
    "ReferenceValleyDiagnostic",
    "ReferenceValleyEvidence",
    "SiteDetectionDiagnostic",
    "SiteDetectionPolicy",
    "UsableSiteAcceptance",
    "analyze_calibration",
    "build_calibration_work_plan",
    "validate_calibration_analysis_contract",
    "validate_calibration_partition_against_source",
]
