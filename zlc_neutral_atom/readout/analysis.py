"""Calibration analysis derived from the validated ``main`` readout physics.

The module has two layers only: pure image/statistics functions, and one adapter
that reads a raw CaptureArtifact through ``CalibrationCaptureLayout``.  Training
and runtime call the same feature extractor from :mod:`.calibration`; display
diagnostics never become runtime authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import scipy
from scipy import ndimage
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import erf

from zlc_data import (
    SITE,
    AxisId,
    AxisSpec,
    ComponentValidity,
    DatasetSchema,
    ValidityMode,
    expand_value_validity,
)
from zlc_neutral_atom.artifacts.capture_frames import CaptureFrameSource
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress
from zlc_storage import (
    finite_real as _finite_float,
    nonnegative_integer as _nonnegative_integer,
    positive_integer as _positive_integer,
)

from .calibration import (
    BoxFeature,
    CalibrationAnalysisRequest,
    CalibrationArtifact,
    CalibrationSourceBinding,
    GridOrder,
    PerSitePsfFeature,
    ReadoutFeature,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    UniformPsfFeature,
    _annulus_background,
    _extract_readout_arrays,
    _immutable_array,
    _ResolvedCalibrationSource,
    _resolve_calibration_source,
)
from .contracts import (
    FrameContract,
    _CalibrationCaptureJoin,
)
from .physical_context import ReadoutPhysicalContext


class CalibrationAnalysisError(ValueError):
    pass


_ADMITTED_ANALYSIS_TOKEN = object()


def _validate_site_center_admission(
    centers_xy: np.ndarray,
    request: CalibrationAnalysisRequest,
) -> None:
    """Admit exact-main detector output against independent spatial intent.

    The detector's returned coordinates are never snapped, reordered, or
    replaced here.  A request without spatial intent remains valid for a pure
    preview computation; only authority minting requires the paired fields.
    """

    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    expected = request.expected_centers_xy
    if expected is None:
        return
    observed = np.asarray(centers_xy)
    if observed.shape != expected.shape:
        raise CalibrationAnalysisError(
            "detector centers do not match the declared ordered site geometry"
        )
    limit = request.maximum_site_residual_px
    assert limit is not None  # paired by CalibrationAnalysisRequest
    for site, (actual, intended) in enumerate(zip(observed, expected, strict=True)):
        if not math.isfinite(float(actual[0])) or not math.isfinite(float(actual[1])):
            raise CalibrationAnalysisError(
                f"detected site {site} contains non-finite coordinates"
            )
        residual = math.hypot(
            float(actual[0]) - float(intended[0]),
            float(actual[1]) - float(intended[1]),
        )
        if residual > limit:
            raise CalibrationAnalysisError(
                f"detected site {site} differs from expected_centers_xy by "
                f"{residual:.6g} px; maximum_site_residual_px is {limit:.6g}"
            )


def estimate_calibration_analysis_peak_bytes(
    schema: DatasetSchema,
    request: CalibrationAnalysisRequest,
    *,
    source_read_scratch_bytes: int = 0,
) -> int:
    """Conservatively admit the observed calibration allocation pattern.

    A 2304x2304 qCMOS profile peaked at 146.82 MiB (about 29 bytes per
    pixel).  The 72-byte pixel allowance below covers the accumulator/count,
    validity, detector-filter temporaries, immutable report copy, and the
    worst dense local-maximum coordinate workspace.  Compact site/statistics
    arrays and the capture owner's bounded read scratch are added separately.
    This is deliberately a pure deterministic bound, not an OS-memory probe or
    a second resource scheduler.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    read_scratch = _nonnegative_integer(
        source_read_scratch_bytes,
        "source_read_scratch_bytes",
    )
    if len(schema.cell_schema.data_shape) != 2:
        raise CalibrationAnalysisError("calibration source cells must be 2D frames")
    group_count, join_build_peak, join_retained = (
        request.layout._memory_upper_bounds(schema)
    )
    reference_shots = len(request.layout.reference_event_indices)
    pixels = math.prod(schema.cell_schema.data_shape)
    sites = request.site_count
    model_count = len(request.model_kinds)
    rows, columns = request.grid_shape_yx
    reference_samples = group_count * reference_shots * sites
    short_samples = group_count * sites
    psf_extent = 2 * request.psf_half_width + 1

    frame_working_set = pixels * 72
    reference_working_set = reference_samples * 24
    # The one-pass extractor holds all model signal/validity arrays while each
    # immutable report copies those arrays and adds predictions/statistical
    # masks.  32 bytes per model/group/site covers both representations and
    # their concurrently live boolean workspaces.
    short_working_set = short_samples * 32 * model_count
    histogram_working_set = (
        model_count * sites * (request.histogram_bins + 1) * 8
    )
    # Each retained ablation point owns a site-sized boolean mask plus Python
    # objects for the ndarray, frozen result, scalar fields, and tuple slot.
    # 1024 bytes is deliberately conservative for that object graph; the mask
    # payload remains explicit so larger site arrays scale correctly.
    ablation_working_set = (
        model_count * (request.max_drop + 1) * (sites + 1024)
    )
    psf_working_set = sites * psf_extent * psf_extent * 24
    site_admission_working_set = (
        0
        if request.expected_centers_xy is None
        else request.expected_centers_xy.nbytes
    )
    # The inherited robust lattice fit retains one Python scalar per unordered
    # pair on each grid axis while np.median materializes its numeric workspace.
    # A 1x1000 profile exposed the otherwise-hidden quadratic peak; 64 bytes per
    # pair covers the measured CPython list/scalar/median overlap.
    lattice_slope_working_set = 64 * (
        math.comb(rows, 2) + math.comb(columns, 2)
    )
    # The report retains one BimodalFit plus per-model SiteFidelity/diagnostic
    # object graphs for each site.  Their Python containers dominate the small
    # ndarray payloads for slender grids and therefore need an explicit bound.
    site_object_working_set = sites * (1 + model_count) * 1024
    # CalibrationReport retains one named context tuple per group.  AxisId
    # values are shared, while tuple/int/container overhead is conservatively
    # bounded per logical axis rather than inferred from data shape.
    group_contexts_working_set = group_count * (
        256 + 128 * len(schema.point_axes)
    )
    analysis_peak = int(
        read_scratch
        + join_retained
        + group_contexts_working_set
        + frame_working_set
        + reference_working_set
        + short_working_set
        + histogram_working_set
        + ablation_working_set
        + psf_working_set
        + site_admission_working_set
        + lattice_slope_working_set
        + site_object_working_set
    )
    # Publication deliberately performs a fresh source admission.  At that
    # boundary the report remains live while the compact join is rebuilt, so
    # its retained arrays/object graphs must overlap the join-build peak even
    # though scientific scratch has already been released.
    retained_result = int(
        group_contexts_working_set
        + pixels * 16
        + reference_samples * 8
        + short_samples * (5 + 10 * model_count)
        + histogram_working_set
        + ablation_working_set
        + psf_working_set
        + site_admission_working_set
        + site_object_working_set
    )
    finalize_peak = join_build_peak + retained_result
    return max(join_build_peak, analysis_peak, finalize_peak)


@dataclass(frozen=True)
class BimodalFit:
    threshold: float
    fidelity: float
    dark_mean: float
    dark_sigma: float
    bright_mean: float
    bright_sigma: float
    bright_fraction: float
    dark_fidelity: float
    bright_fidelity: float
    bright_above: bool
    ok: bool


@dataclass(frozen=True, eq=False)
class ReferenceLabels:
    occupied: np.ndarray
    dark: np.ndarray
    valid: np.ndarray
    fits: tuple[BimodalFit, ...]
    n_reference_shots: int

    def __post_init__(self) -> None:
        occupied = _immutable_array(
            self.occupied,
            dtype="bool",
            field_name="occupied",
        )
        if occupied.ndim != 2:
            raise ValueError("reference labels must have shape (groups, sites)")
        dark = _immutable_array(
            self.dark,
            dtype="bool",
            shape=occupied.shape,
            field_name="dark",
        )
        valid = _immutable_array(
            self.valid,
            dtype="bool",
            shape=occupied.shape,
            field_name="valid",
        )
        fits = tuple(self.fits)
        if len(fits) != occupied.shape[1] or any(
            not isinstance(fit, BimodalFit) for fit in fits
        ):
            raise ValueError("fits must contain one BimodalFit per site")
        shots = _positive_integer(self.n_reference_shots, "n_reference_shots")
        if np.any(occupied & dark) or np.any((occupied | dark) & ~valid):
            raise ValueError("reference labels are internally inconsistent")
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "dark", dark)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "fits", fits)
        object.__setattr__(self, "n_reference_shots", shots)

    @property
    def n_groups(self) -> int:
        return int(self.occupied.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.occupied.shape[1])


@dataclass(frozen=True, eq=False)
class TrainTestSplit:
    train: np.ndarray
    test: np.ndarray
    seed: int
    train_fraction: float

    def __post_init__(self) -> None:
        train = _immutable_array(self.train, dtype="bool", field_name="train")
        if train.ndim != 2:
            raise ValueError("train must have shape (groups, sites)")
        test = _immutable_array(
            self.test,
            dtype="bool",
            shape=train.shape,
            field_name="test",
        )
        if np.any(train & test):
            raise ValueError("train and test masks must be disjoint")
        seed = _nonnegative_integer(self.seed, "seed")
        fraction = _finite_float(self.train_fraction, "train_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "train_fraction", fraction)


@dataclass(frozen=True)
class SiteFidelity:
    site: int
    threshold: float
    bright_above: bool
    fidelity: float
    fidelity_dark: float
    fidelity_bright: float
    model_fidelity: float
    dark_mean: float
    dark_sigma: float
    bright_mean: float
    bright_sigma: float
    n_test: int
    n_train_dark: int
    n_train_bright: int


@dataclass(frozen=True, eq=False)
class AblationPoint:
    drop_worst_k: int
    excluded_sites: np.ndarray
    fidelity: float
    errors: int
    n_valid: int

    def __post_init__(self) -> None:
        excluded = _immutable_array(
            self.excluded_sites,
            dtype="bool",
            field_name="excluded_sites",
        )
        if excluded.ndim != 1:
            raise ValueError("excluded_sites must have shape (sites,)")
        object.__setattr__(
            self,
            "drop_worst_k",
            _nonnegative_integer(self.drop_worst_k, "drop_worst_k"),
        )
        object.__setattr__(self, "excluded_sites", excluded)
        object.__setattr__(self, "errors", _nonnegative_integer(self.errors, "errors"))
        object.__setattr__(self, "n_valid", _nonnegative_integer(self.n_valid, "n_valid"))


@dataclass(frozen=True, eq=False)
class ModelCalibrationReport:
    kind: ReadoutModelKind
    quick_thresholds: np.ndarray
    short_signals: np.ndarray
    short_validity: np.ndarray
    bin_edges: np.ndarray
    predictions: np.ndarray
    site_fidelity: tuple[SiteFidelity, ...]
    aggregate_fidelity: float
    global_threshold: float
    global_bright_above: bool
    global_fidelity: float
    ablation: tuple[AblationPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind")
        short = _immutable_array(
            self.short_signals,
            dtype="<f8",
            field_name="short_signals",
        )
        if short.ndim != 2:
            raise ValueError("short_signals must have shape (groups, sites)")
        quick = _immutable_array(
            self.quick_thresholds,
            dtype="<f8",
            shape=(short.shape[1],),
            field_name="quick_thresholds",
        )
        short_validity = _immutable_array(
            self.short_validity,
            dtype="bool",
            shape=short.shape,
            field_name="short_validity",
        )
        edges = _immutable_array(self.bin_edges, dtype="<f8", field_name="bin_edges")
        if edges.ndim != 1 or edges.size < 3 or not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges must be a strictly increasing 1D array")
        predictions = _immutable_array(
            self.predictions,
            dtype="bool",
            shape=short.shape,
            field_name="predictions",
        )
        site_fidelity = tuple(self.site_fidelity)
        if len(site_fidelity) != short.shape[1] or any(
            not isinstance(item, SiteFidelity) for item in site_fidelity
        ):
            raise ValueError("site_fidelity must contain one item per site")
        if tuple(item.site for item in site_fidelity) != tuple(range(short.shape[1])):
            raise ValueError("site_fidelity must follow the canonical site order")
        if type(self.global_bright_above) is not bool:
            raise TypeError("global_bright_above must be bool")
        ablation = tuple(self.ablation)
        if any(not isinstance(item, AblationPoint) for item in ablation):
            raise TypeError("ablation must contain AblationPoint values")
        if any(item.excluded_sites.shape != (short.shape[1],) for item in ablation):
            raise ValueError("ablation masks must follow the report site axis")
        object.__setattr__(self, "quick_thresholds", quick)
        object.__setattr__(self, "short_signals", short)
        object.__setattr__(self, "short_validity", short_validity)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "site_fidelity", site_fidelity)
        object.__setattr__(self, "ablation", ablation)

    @property
    def thresholds(self) -> np.ndarray:
        values = np.asarray([item.threshold for item in self.site_fidelity], dtype="<f8")
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class PsfFitDiagnostic:
    site: int
    center_xy: tuple[float, float]
    sigma_xy: tuple[float, float]
    fit_ok: bool


@dataclass(frozen=True, eq=False)
class CalibrationReport:
    request: CalibrationAnalysisRequest
    software_lineage: tuple[tuple[str, str], ...]
    group_contexts: tuple[tuple[tuple[AxisId, int], ...], ...]
    reference_average: np.ndarray
    reference_average_validity: np.ndarray
    reference_box_signals: np.ndarray
    labels: ReferenceLabels
    split: TrainTestSplit
    psf_fits: tuple[PsfFitDiagnostic, ...]
    models: tuple[ModelCalibrationReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CalibrationAnalysisRequest):
            raise TypeError("request must be CalibrationAnalysisRequest")
        lineage = tuple(tuple(item) for item in self.software_lineage)
        if any(
            len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in lineage
        ):
            raise ValueError("software_lineage must contain non-empty text pairs")
        if len({name for name, _version in lineage}) != len(lineage):
            raise ValueError("software_lineage names must be unique")
        lineage = tuple(sorted(lineage))
        if not isinstance(self.group_contexts, tuple) or any(
            not isinstance(context, tuple) for context in self.group_contexts
        ):
            raise TypeError("group_contexts must be nested tuples")
        contexts = self.group_contexts
        if (
            not contexts
            or len(contexts) != self.labels.n_groups
            or len(set(contexts)) != len(contexts)
        ):
            raise ValueError("group_contexts must uniquely identify every calibration group")
        context_axis_ids = tuple(axis_id for axis_id, _index in contexts[0])
        for context in contexts:
            if any(
                not isinstance(item, tuple) or len(item) != 2
                for item in context
            ):
                raise TypeError("group context entries must be (AxisId, int) tuples")
            if any(
                not isinstance(axis_id, AxisId) or type(index) is not int or index < 0
                for axis_id, index in context
            ):
                raise TypeError("group context entries must use AxisId and non-negative int")
            if len({axis_id for axis_id, _index in context}) != len(context):
                raise ValueError("group context axes must be unique")
            if tuple(axis_id for axis_id, _index in context) != context_axis_ids:
                raise ValueError(
                    "every calibration group context must use the same ordered AxisIds"
                )
        average = _immutable_array(
            self.reference_average,
            dtype="<f8",
            field_name="reference_average",
        )
        if average.ndim != 2:
            raise ValueError("reference_average must be a 2D image")
        if not np.all(np.isfinite(average)):
            raise ValueError("reference_average must be finite")
        average_validity = _immutable_array(
            self.reference_average_validity,
            dtype="bool",
            shape=average.shape,
            field_name="reference_average_validity",
        )
        signals = _immutable_array(
            self.reference_box_signals,
            dtype="<f8",
            field_name="reference_box_signals",
        )
        expected = (
            self.labels.n_groups,
            self.labels.n_reference_shots,
            self.labels.n_sites,
        )
        if signals.shape != expected:
            raise ValueError(
                f"reference_box_signals must have shape {expected}, got {signals.shape}"
            )
        psf_fits = tuple(self.psf_fits)
        expected_psf_fits = (
            self.labels.n_sites
            if any(kind is not ReadoutModelKind.BOX for kind in self.request.model_kinds)
            else 0
        )
        if len(psf_fits) != expected_psf_fits or any(
            not isinstance(item, PsfFitDiagnostic) for item in psf_fits
        ):
            raise ValueError("psf_fits has the wrong size for the requested models")
        if tuple(item.site for item in psf_fits) != tuple(range(expected_psf_fits)):
            raise ValueError("psf_fits must follow the canonical site order")
        models = tuple(self.models)
        if not models or any(not isinstance(item, ModelCalibrationReport) for item in models):
            raise TypeError("models must contain ModelCalibrationReport values")
        if tuple(item.kind for item in models) != self.request.model_kinds:
            raise ValueError("report models must match the request model set and order")
        if self.labels.n_sites != self.request.site_count:
            raise ValueError("report labels and request contain different site counts")
        if any(item.short_signals.shape != self.labels.valid.shape for item in models):
            raise ValueError("model reports and reference labels have different shapes")
        if self.split.train.shape != self.labels.valid.shape:
            raise ValueError("split and reference labels have different shapes")
        object.__setattr__(self, "software_lineage", lineage)
        object.__setattr__(self, "group_contexts", contexts)
        object.__setattr__(self, "reference_average", average)
        object.__setattr__(self, "reference_average_validity", average_validity)
        object.__setattr__(self, "reference_box_signals", signals)
        object.__setattr__(self, "psf_fits", psf_fits)
        object.__setattr__(self, "models", models)

    def model(self, kind: ReadoutModelKind) -> ModelCalibrationReport:
        for report in self.models:
            if report.kind is kind:
                return report
        raise KeyError(kind)


def _main_reference_thresholds_available(
    reports: Sequence[ModelCalibrationReport],
) -> bool:
    """Mirror main's all-method gate for committing labelled thresholds."""

    ordered = tuple(reports)
    return bool(ordered) and all(
        any(math.isfinite(item.model_fidelity) for item in report.site_fidelity)
        for report in ordered
    )


def _runtime_model_values(
    feature: ReadoutFeature,
    report: ModelCalibrationReport,
    request: CalibrationAnalysisRequest,
    *,
    use_reference_thresholds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive runtime thresholds and usable sites from diagnostic evidence.

    ``main`` first calibrated every method with a quick Otsu threshold.  It
    replaced those thresholds with bracket-labelled thresholds only when every
    requested method produced held-out model evidence; a missing per-site
    labelled threshold then fell back to that site's quick threshold.  Keep
    that physical rule here while applying the migration's explicit safety
    corrections only to sites that actually use a labelled fit.
    """

    formal = report.thresholds
    quick = report.quick_thresholds
    has_labelled_threshold = np.isfinite(formal)
    if use_reference_thresholds:
        thresholds = np.where(has_labelled_threshold, formal, quick)
    else:
        thresholds = np.array(quick, copy=True)

    labelled_quality = np.asarray(
        [
            item.bright_above
            and math.isfinite(item.threshold)
            and math.isfinite(item.fidelity)
            and item.fidelity > request.minimum_site_fidelity
            for item in report.site_fidelity
        ],
        dtype=bool,
    )
    usable = (
        feature.valid_sites.mask
        & np.isfinite(thresholds)
        # Threshold selection follows main's all-method commit gate, but an
        # unrelated method's failure must never erase this site's explicit
        # reversed-polarity or chance-level labelled evidence.  Only a site
        # with no finite labelled threshold gets the unconditional quick
        # fallback.
        & (~has_labelled_threshold | labelled_quality)
    )
    return thresholds, usable


def _validate_calibration_binding(
    artifact: CalibrationArtifact,
    report: CalibrationReport,
) -> None:
    """Validate the pure artifact/report relationship without minting authority."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    if not isinstance(report, CalibrationReport):
        raise TypeError("report must be CalibrationReport")
    request = report.request
    if artifact.source_binding.layout != request.layout:
        raise ValueError("artifact and report name different capture layouts")
    if artifact.site_map.grid_shape_yx != request.grid_shape_yx:
        raise ValueError("artifact and report name different site grids")
    if artifact.site_map.ordering is not request.ordering:
        raise ValueError("artifact and report name different site ordering")
    _validate_site_center_admission(
        artifact.site_map.coordinates_xy,
        request,
    )
    if report.reference_average.shape != artifact.frame_contract.frame_schema.data_shape:
        raise ValueError("report reference image differs from the artifact FrameContract")
    if report.labels.n_sites != artifact.site_map.site_axis.size:
        raise ValueError("artifact and report contain different site counts")
    artifact_kinds = tuple(model.kind for model in artifact.models)
    if artifact_kinds != request.model_kinds:
        raise ValueError("artifact and report contain different model sets")
    if artifact.default_model_kind is not request.default_model_kind:
        raise ValueError("artifact and report name different default models")
    expected_box_boxes = _boxes_for_centers(
        artifact.site_map.coordinates_xy,
        report.reference_average.shape,
        request.box_radius,
    )
    expected_psf_boxes = np.asarray(
        [
            _crop_psf_box(
                report.reference_average.shape,
                float(x),
                float(y),
                request.psf_half_width,
            )
            for x, y in artifact.site_map.coordinates_xy
        ],
        dtype="<i8",
    )
    report_by_kind = {item.kind: item for item in report.models}
    use_reference_thresholds = _main_reference_thresholds_available(report.models)
    for model in artifact.models:
        model_report = report_by_kind[model.kind]
        expected_thresholds, expected_usable = _runtime_model_values(
            model.feature,
            model_report,
            request,
            use_reference_thresholds=use_reference_thresholds,
        )
        if not np.array_equal(
            model.thresholds,
            expected_thresholds,
            equal_nan=True,
        ):
            raise ValueError("artifact thresholds differ from the calibration report")
        if not np.array_equal(model.usable_sites.mask, expected_usable):
            raise ValueError("artifact usable sites differ from report evidence")
        feature = model.feature
        if isinstance(feature, BoxFeature):
            if feature.reducer is not request.box_reducer or not np.array_equal(
                feature.boxes_xywh,
                expected_box_boxes,
            ):
                raise ValueError("BOX feature differs from the calibration request")
        elif isinstance(feature, (PerSitePsfFeature, UniformPsfFeature)):
            expected_extent = 2 * request.psf_half_width + 1
            kernel_shape = (
                feature.kernels.shape[-2:]
                if isinstance(feature, PerSitePsfFeature)
                else feature.kernel.shape
            )
            if (
                feature.background is not request.psf_background
                or feature.background_padding != request.psf_background_padding
                or kernel_shape != (expected_extent, expected_extent)
                or not np.array_equal(feature.boxes_xywh, expected_psf_boxes)
            ):
                raise ValueError("PSF feature differs from the calibration request")


@dataclass(frozen=True)
class CalibrationComputation:
    """Pure, validated calibration output with no durable commit authority."""

    artifact: CalibrationArtifact
    report: CalibrationReport

    def __post_init__(self) -> None:
        _validate_calibration_binding(self.artifact, self.report)


@dataclass(frozen=True, init=False)
class CalibrationAnalysisResult:
    """Authority-bearing result produced only from an AdmittedCapture."""

    artifact: CalibrationArtifact
    report: CalibrationReport
    _source_admission: object = field(repr=False, compare=False)

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CalibrationAnalysisResult is final and cannot be subclassed")

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError(
            "CalibrationAnalysisResult is returned by analyze_calibration"
        )

    @classmethod
    def _from_admitted_analysis(
        cls,
        token: object,
        computation: CalibrationComputation,
        source: object,
    ) -> "CalibrationAnalysisResult":
        from zlc_neutral_atom.artifacts.capture import AdmittedCapture

        if token is not _ADMITTED_ANALYSIS_TOKEN:
            raise PermissionError(
                "admitted calibration results are minted by analyze_calibration"
            )
        if not isinstance(computation, CalibrationComputation):
            raise TypeError("computation must be CalibrationComputation")
        if type(source) is not AdmittedCapture:
            raise TypeError("source must be an exact AdmittedCapture")
        if computation.artifact.source_binding.source_capture_ref != source.reference:
            raise ValueError("calibration result names another admitted capture")
        result = object.__new__(cls)
        object.__setattr__(result, "artifact", computation.artifact)
        object.__setattr__(result, "report", computation.report)
        object.__setattr__(result, "_source_admission", source)
        return result

    def _require_source_admission(self, source: object) -> None:
        """Require the exact capture repository/journal authority used in analysis."""

        from zlc_neutral_atom.artifacts.capture import AdmittedCapture

        if type(source) is not AdmittedCapture:
            raise TypeError("source must be an exact AdmittedCapture")
        if not self._source_admission._matches_admission(source):
            raise PermissionError(
                "calibration result belongs to another source admission"
            )

def _gaussian_2d(coords, offset, amplitude, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return (
        offset
        + amplitude
        * np.exp(
            -0.5
            * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
        )
    ).ravel()


def _fit_gaussian_spot_2d(
    data: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    x0: float,
    y0: float,
    offset0: float,
    amplitude: float,
    sigma0: float = 0.9,
) -> tuple[float, float, float, float, bool]:
    amplitude = float(amplitude)
    try:
        initial = [
            float(offset0),
            max(amplitude, 1e-6),
            float(x0),
            float(y0),
            sigma0,
            sigma0,
        ]
        lower = [
            float(np.nanmin(data)) - abs(amplitude) - 1,
            0.0,
            float(xx.min()) - 0.5,
            float(yy.min()) - 0.5,
            0.2,
            0.2,
        ]
        upper = [
            float(np.nanmax(data)) + abs(amplitude) + 1,
            max(amplitude * 5, 1.0),
            float(xx.max()) + 0.5,
            float(yy.max()) + 0.5,
            4.0,
            4.0,
        ]
        fitted, _ = curve_fit(
            _gaussian_2d,
            (xx.ravel(), yy.ravel()),
            data.ravel(),
            p0=initial,
            bounds=(lower, upper),
            maxfev=5000,
        )
        _offset, _amplitude, x_fit, y_fit, sigma_x, sigma_y = fitted
        return (
            float(x_fit),
            float(y_fit),
            float(abs(sigma_x)),
            float(abs(sigma_y)),
            True,
        )
    except Exception:
        values = np.clip(data - np.nanpercentile(data, 20), 0, None)
        total = float(np.sum(values))
        if total <= 0:
            return float(x0), float(y0), float(sigma0), float(sigma0), False
        return (
            float(np.sum(xx * values) / total),
            float(np.sum(yy * values) / total),
            float(sigma0),
            float(sigma0),
            False,
        )


def _refine_center_subpixel(
    image: np.ndarray,
    x: float,
    y: float,
    half: int = 2,
) -> tuple[float, float]:
    height, width = image.shape
    x_int, y_int = int(round(x)), int(round(y))
    x0, x1 = max(0, x_int - half), min(width, x_int + half + 1)
    y0, y1 = max(0, y_int - half), min(height, y_int + half + 1)
    cut = image[y0:y1, x0:x1]
    if cut.size < 9 or not np.isfinite(cut).any():
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    background = float(np.nanmedian(cut))
    amplitude = float(np.nanmax(cut) - background)
    x_fit, y_fit, _sx, _sy, _ok = _fit_gaussian_spot_2d(
        cut,
        yy,
        xx,
        x0=float(x),
        y0=float(y),
        offset0=background,
        amplitude=amplitude,
    )
    return x_fit, y_fit


def _sort_centers_grid(
    centers: np.ndarray,
    grid_shape_yx: tuple[int, int],
    ordering: GridOrder,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    rows, columns = grid_shape_yx
    if centers.shape != (rows * columns, 2):
        raise ValueError("center count does not match grid_shape_yx")
    if ordering in (GridOrder.ROW_MAJOR, GridOrder.SERPENTINE):
        output = []
        for row_index, chunk in enumerate(
            np.array_split(centers[np.argsort(centers[:, 1])], rows)
        ):
            row = chunk[np.argsort(chunk[:, 0])]
            if ordering is GridOrder.SERPENTINE and row_index % 2:
                row = row[::-1]
            output.append(row)
        return np.vstack(output)
    output = []
    for column_index, chunk in enumerate(
        np.array_split(centers[np.argsort(centers[:, 0])], columns)
    ):
        column = chunk[np.argsort(chunk[:, 1])]
        if ordering is GridOrder.COLUMN_SERPENTINE and column_index % 2:
            column = column[::-1]
        output.append(column)
    return np.vstack(output)


def _robust_axis_lattice(anchors: np.ndarray, count: int) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=float)
    if count <= 1:
        return anchors.copy()
    indices = np.arange(count, dtype=float)
    slopes = [
        (anchors[j] - anchors[i]) / (j - i)
        for i in range(count)
        for j in range(i + 1, count)
    ]
    pitch = float(np.median(slopes)) if slopes else 0.0
    origin = float(np.median(anchors - pitch * indices))
    return origin + pitch * indices


def _regularize_grid(
    row_major_centers: np.ndarray,
    grid_shape_yx: tuple[int, int],
    image_shape_yx: tuple[int, int],
) -> np.ndarray:
    rows, columns = grid_shape_yx
    grid = np.asarray(row_major_centers, dtype=float).reshape(rows, columns, 2)
    height, width = image_shape_yx
    row_y = _robust_axis_lattice(np.median(grid[:, :, 1], axis=1), rows)
    column_x = _robust_axis_lattice(np.median(grid[:, :, 0], axis=0), columns)
    pitches = []
    if rows > 1:
        pitches.append(abs(float(row_y[1] - row_y[0])))
    if columns > 1:
        pitches.append(abs(float(column_x[1] - column_x[0])))
    pitches = [value for value in pitches if math.isfinite(value) and value > 0]
    pitch = float(min(pitches)) if pitches else float(min(height, width))
    lattice_x = np.broadcast_to(column_x[None, :], (rows, columns))
    lattice_y = np.broadcast_to(row_y[:, None], (rows, columns))
    off_node = (
        np.hypot(grid[:, :, 0] - lattice_x, grid[:, :, 1] - lattice_y)
        > 0.5 * pitch
    )
    grid[off_node, 0] = lattice_x[off_node]
    grid[off_node, 1] = lattice_y[off_node]
    margin = max(1.0, 0.25 * pitch)
    grid[:, :, 0] = np.clip(
        grid[:, :, 0],
        margin,
        max(margin, width - 1 - margin),
    )
    grid[:, :, 1] = np.clip(
        grid[:, :, 1],
        margin,
        max(margin, height - 1 - margin),
    )
    return grid.reshape(rows * columns, 2)


def find_site_centers(
    image: np.ndarray,
    grid_shape_yx: tuple[int, int],
    *,
    min_distance: int | None = None,
    threshold_rel: float = 0.35,
    ordering: GridOrder = GridOrder.ROW_MAJOR,
    refine_half: int = 2,
) -> np.ndarray:
    """Main-authority Gaussian/local-maximum detector with lattice repair."""

    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or 0 in image.shape or not np.isfinite(image).any():
        raise ValueError("image must be a non-empty finite 2D array")
    rows, columns = grid_shape_yx
    needed = rows * columns
    if min_distance is None:
        min_distance = max(3, int(min(image.shape) / max(rows, columns, 1) / 2))
    min_distance = _positive_integer(min_distance, "min_distance")
    smooth = ndimage.gaussian_filter(image, sigma=1.0)
    cutoff = float(
        np.nanmin(smooth)
        + threshold_rel * (np.nanmax(smooth) - np.nanmin(smooth))
    )
    local_max = ndimage.maximum_filter(smooth, size=min_distance)
    is_peak = smooth == local_max
    candidates_yx = np.argwhere(is_peak & (smooth >= cutoff))
    if len(candidates_yx) < needed:
        peaks_yx = np.argwhere(is_peak)
        if len(peaks_yx) < needed:
            raise CalibrationAnalysisError(
                f"only {len(peaks_yx)} local maxima for a {rows}x{columns} grid"
            )
        weights = smooth[peaks_yx[:, 0], peaks_yx[:, 1]]
        candidates_yx = peaks_yx[np.argsort(weights)[::-1][:needed]]
    weights = smooth[candidates_yx[:, 0], candidates_yx[:, 1]]
    selected = candidates_yx[np.argsort(weights)[::-1]][:needed]
    centers = np.asarray(
        [
            _refine_center_subpixel(
                image,
                float(column),
                float(row),
                half=refine_half,
            )
            for row, column in selected
        ],
        dtype=float,
    )
    row_major = _sort_centers_grid(centers, grid_shape_yx, GridOrder.ROW_MAJOR)
    repaired = _regularize_grid(row_major, grid_shape_yx, image.shape)
    return _sort_centers_grid(repaired, grid_shape_yx, ordering)


def _boxes_for_centers(
    centers_xy: np.ndarray,
    image_shape_yx: tuple[int, int],
    radius: int,
) -> np.ndarray:
    height, width = image_shape_yx
    boxes = []
    for x, y in centers_xy:
        center_x, center_y = int(round(float(x))), int(round(float(y)))
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise CalibrationAnalysisError("detected center is outside the image")
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return np.asarray(boxes, dtype="<i8")


def _crop_psf_box(
    image_shape_yx: tuple[int, int],
    x: float,
    y: float,
    half_width: int,
) -> tuple[int, int, int, int]:
    height, width = image_shape_yx
    x_int, y_int = int(round(float(x))), int(round(float(y)))
    x0, x1 = max(0, x_int - half_width), min(width, x_int + half_width + 1)
    y0, y1 = max(0, y_int - half_width), min(height, y_int + half_width + 1)
    return x0, y0, x1 - x0, y1 - y0


def _gaussian_psf(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    gaussian = np.exp(
        -0.5
        * (
            ((xx - x0) / max(sigma_x, 1e-6)) ** 2
            + ((yy - y0) / max(sigma_y, 1e-6)) ** 2
        )
    )
    gaussian = np.clip(gaussian, 0, None)
    total = float(np.sum(gaussian))
    return (
        gaussian / total
        if total > 0
        else np.ones(shape, dtype=float) / float(np.prod(shape))
    )


def _fit_psf_features(
    reference_average: np.ndarray,
    average_validity: np.ndarray,
    centers_xy: np.ndarray,
    site_axis: AxisSpec,
    request: CalibrationAnalysisRequest,
) -> tuple[PerSitePsfFeature, UniformPsfFeature, tuple[PsfFitDiagnostic, ...]]:
    box_extent = 2 * request.psf_half_width + 1
    boxes = []
    kernels = []
    geometry_valid = np.ones(site_axis.size, dtype=bool)
    diagnostics = []
    for site, (x, y) in enumerate(centers_xy):
        box = _crop_psf_box(
            reference_average.shape,
            float(x),
            float(y),
            request.psf_half_width,
        )
        box_x, box_y, width, height = box
        if (width, height) != (box_extent, box_extent):
            raise CalibrationAnalysisError(
                f"site {site} is too close to the image edge for a "
                f"{box_extent}x{box_extent} PSF"
            )
        boxes.append(box)
        cut = reference_average[box_y : box_y + height, box_x : box_x + width]
        cut_valid = average_validity[box_y : box_y + height, box_x : box_x + width]
        if not np.all(cut_valid & np.isfinite(cut)):
            geometry_valid[site] = False
            kernel = np.ones((height, width), dtype=float) / float(height * width)
            kernels.append(kernel)
            diagnostics.append(
                PsfFitDiagnostic(site, (float(x), float(y)), (0.9, 0.9), False)
            )
            continue
        background = _annulus_background(
            reference_average,
            average_validity,
            box,
            request.psf_background_padding,
        )
        subtracted = cut - background
        yy, xx = np.mgrid[box_y : box_y + height, box_x : box_x + width]
        amplitude = float(np.nanmax(subtracted)) if np.isfinite(subtracted).any() else 0.0
        x_fit, y_fit, sigma_x, sigma_y, fit_ok = _fit_gaussian_spot_2d(
            subtracted,
            yy,
            xx,
            x0=float(x),
            y0=float(y),
            offset0=0.0,
            amplitude=amplitude,
        )
        positive = np.clip(subtracted, 0, None)
        positive = ndimage.gaussian_filter(positive, 0.35)
        total = float(np.sum(positive))
        kernel = (
            positive / total
            if total > 0
            else _gaussian_psf(
                (height, width),
                x_fit - box_x,
                y_fit - box_y,
                sigma_x,
                sigma_y,
            )
        )
        kernels.append(np.ascontiguousarray(kernel, dtype=float))
        diagnostics.append(
            PsfFitDiagnostic(
                site,
                (x_fit, y_fit),
                (sigma_x, sigma_y),
                fit_ok,
            )
        )
    boxes_array = np.asarray(boxes, dtype="<i8")
    kernels_array = np.stack(kernels, axis=0)
    validity = ComponentValidity((site_axis.axis_id,), geometry_valid)
    per_site = PerSitePsfFeature(
        site_axis,
        boxes_array,
        kernels_array,
        request.psf_background,
        request.psf_background_padding,
        validity,
    )
    shared = (
        np.mean(kernels_array[geometry_valid], axis=0)
        if np.any(geometry_valid)
        else np.ones(kernels_array.shape[1:], dtype=float)
    )
    shared_total = float(np.sum(shared))
    shared = (
        shared / shared_total
        if shared_total > 0
        else np.ones_like(shared) / float(shared.size)
    )
    uniform = UniformPsfFeature(
        site_axis,
        boxes_array,
        shared,
        request.psf_background,
        request.psf_background_padding,
        validity,
    )
    return per_site, uniform, tuple(diagnostics)


def otsu_threshold(values: object, *, bins: int = 96) -> float:
    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return float("nan")
    if float(np.min(samples)) == float(np.max(samples)):
        return float(samples[0])
    histogram, edges = np.histogram(samples, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = histogram.astype(float)
    probability = weights / weights.sum()
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * centers)
    denominator = omega * (1.0 - omega)
    score = np.full_like(centers, -np.inf, dtype=float)
    valid = denominator > 0
    score[valid] = (
        (mu[-1] * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    )
    best = float(np.max(score[valid]))
    plateau = np.flatnonzero(
        valid & (score >= best - 1e-9 * (abs(best) + 1.0))
    )
    return float(np.mean(centers[plateau]))


def _normal_cdf(x, mean: float, sigma: float):
    sigma = max(abs(float(sigma)), 1e-12)
    result = 0.5 * (
        1.0
        + erf((np.asarray(x, dtype=float) - mean) / (sigma * math.sqrt(2.0)))
    )
    return float(result) if result.ndim == 0 else result


def _exact_otsu_threshold(values: np.ndarray, min_fraction: float = 0.02) -> float:
    samples = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    count = int(samples.size)
    if count < 4:
        return float("nan")
    minimum = max(2, int(math.ceil(min_fraction * count)))
    if count < 2 * minimum + 1:
        minimum = max(1, count // 4)
    positions = np.arange(1, count, dtype=float)
    cumulative = np.cumsum(samples)
    total = float(cumulative[-1])
    left_count = positions
    right_count = float(count) - positions
    valid = (
        (left_count >= minimum)
        & (right_count >= minimum)
        & (samples[:-1] < samples[1:])
    )
    if not np.any(valid):
        return float(np.median(samples))
    left_mean = cumulative[:-1] / left_count
    right_mean = (total - cumulative[:-1]) / right_count
    score = left_count * right_count * (right_mean - left_mean) ** 2
    score[~valid] = -np.inf
    index = int(np.argmax(score))
    if not np.isfinite(score[index]):
        return float(np.median(samples))
    return float(0.5 * (samples[index] + samples[index + 1]))


def _one_sided_core_stats(
    samples: np.ndarray,
    side: str,
    sigma_floor: float,
) -> tuple[float, float, bool]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 4:
        return float("nan"), float("nan"), False
    q16, q50, q84 = np.percentile(
        samples,
        [15.865525393145708, 50.0, 84.1344746068543],
    )
    sigma = float(q50 - q16) if side == "low" else float(q84 - q50)
    alternative = float(0.5 * (q84 - q16))
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = alternative
    if not math.isfinite(sigma) or sigma <= 0:
        median = float(np.median(samples))
        sigma = 1.482602218505602 * float(np.median(np.abs(samples - median)))
    return float(q50), max(float(sigma), sigma_floor, 1e-12), True


def _optimal_gaussian_threshold(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
) -> tuple[float, bool]:
    bright_above = bool(bright_mean >= dark_mean)
    lower, upper = min(dark_mean, bright_mean), max(dark_mean, bright_mean)
    if (
        not np.isfinite([lower, upper, dark_sigma, bright_sigma]).all()
        or upper <= lower
    ):
        return float(0.5 * (dark_mean + bright_mean)), bright_above

    def error(threshold: float) -> float:
        if bright_above:
            dark_error = 1.0 - float(_normal_cdf(threshold, dark_mean, dark_sigma))
            bright_error = float(_normal_cdf(threshold, bright_mean, bright_sigma))
        else:
            dark_error = float(_normal_cdf(threshold, dark_mean, dark_sigma))
            bright_error = 1.0 - float(
                _normal_cdf(threshold, bright_mean, bright_sigma)
            )
        return 0.5 * (dark_error + bright_error)

    result = minimize_scalar(error, bounds=(lower, upper), method="bounded")
    threshold = result.x if result.success else 0.5 * (lower + upper)
    return float(threshold), bright_above


def _gaussian_fidelity(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    threshold: float,
    bright_above: bool,
) -> tuple[float, float, float]:
    if not np.isfinite(
        [dark_mean, dark_sigma, bright_mean, bright_sigma, threshold]
    ).all():
        return float("nan"), float("nan"), float("nan")
    if bright_above:
        dark = float(_normal_cdf(threshold, dark_mean, dark_sigma))
        bright = 1.0 - float(_normal_cdf(threshold, bright_mean, bright_sigma))
    else:
        dark = 1.0 - float(_normal_cdf(threshold, dark_mean, dark_sigma))
        bright = float(_normal_cdf(threshold, bright_mean, bright_sigma))
    return dark, bright, 0.5 * (dark + bright)


def fit_bimodal(values: object, *, min_component_fraction: float = 0.01) -> BimodalFit:
    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    split = _exact_otsu_threshold(samples)
    if samples.size < 8 or not math.isfinite(split):
        return BimodalFit(
            split,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            True,
            False,
        )
    low = samples[samples <= split]
    high = samples[samples > split]
    minimum = max(4, int(math.ceil(min_component_fraction * samples.size)))
    if low.size < minimum or high.size < minimum:
        return BimodalFit(
            split,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float(high.size / samples.size),
            float("nan"),
            float("nan"),
            True,
            False,
        )
    full_sigma = float(np.std(samples)) if samples.size > 1 else 1.0
    floor = max(1e-6 * full_sigma, 1e-12)
    dark_mean, dark_sigma, dark_ok = _one_sided_core_stats(low, "low", floor)
    bright_mean, bright_sigma, bright_ok = _one_sided_core_stats(high, "high", floor)
    bright_fraction = float(high.size / samples.size)
    if (
        not (dark_ok and bright_ok)
        or not np.isfinite(
            [dark_mean, dark_sigma, bright_mean, bright_sigma]
        ).all()
        or bright_mean <= dark_mean
    ):
        return BimodalFit(
            split,
            float("nan"),
            dark_mean,
            dark_sigma,
            bright_mean,
            bright_sigma,
            bright_fraction,
            float("nan"),
            float("nan"),
            True,
            False,
        )
    threshold, bright_above = _optimal_gaussian_threshold(
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
    )
    dark_fidelity, bright_fidelity, fidelity = _gaussian_fidelity(
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        threshold,
        bright_above,
    )
    separation = (bright_mean - dark_mean) / max(dark_sigma + bright_sigma, 1e-12)
    return BimodalFit(
        threshold,
        fidelity,
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        bright_fraction,
        dark_fidelity,
        bright_fidelity,
        bright_above,
        bool(separation > 0.5),
    )


def reference_labels(
    reference_signals: np.ndarray,
    reference_validity: np.ndarray | None = None,
) -> ReferenceLabels:
    """Strict all-bright/all-dark consensus, with invalid data never called dark."""

    signals = np.asarray(reference_signals, dtype=float)
    if signals.ndim != 3:
        raise ValueError(
            "reference_signals must have shape (groups, reference_shots, sites)"
        )
    valid_input = (
        np.ones(signals.shape, dtype=bool)
        if reference_validity is None
        else np.asarray(reference_validity, dtype=bool)
    )
    if valid_input.shape != signals.shape:
        raise ValueError("reference_validity shape differs from reference_signals")
    valid_input = valid_input & np.isfinite(signals)
    groups, shots, sites = signals.shape
    bright = np.zeros((groups, shots, sites), dtype=bool)
    fit_ok = np.zeros(sites, dtype=bool)
    fits = []
    for site in range(sites):
        fit = fit_bimodal(signals[:, :, site][valid_input[:, :, site]])
        fits.append(fit)
        # Fluorescence physics is fixed: an occupied atom is brighter.  A
        # reversed fit is diagnostic evidence of a bad site, never a runtime mode.
        if fit.ok and fit.bright_above and math.isfinite(fit.threshold):
            bright[:, :, site] = signals[:, :, site] > fit.threshold
            fit_ok[site] = True
    every_reference_valid = np.all(valid_input, axis=1)
    all_bright = np.all(bright, axis=1)
    all_dark = np.all(~bright, axis=1)
    valid = (
        every_reference_valid
        & (all_bright | all_dark)
        & fit_ok[np.newaxis, :]
    )
    return ReferenceLabels(
        all_bright & valid,
        all_dark & valid,
        valid,
        tuple(fits),
        shots,
    )


def train_test_split(
    labels: ReferenceLabels,
    *,
    train_fraction: float = 0.9,
    seed: int = 0,
) -> TrainTestSplit:
    fraction = float(train_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    train = np.zeros_like(labels.valid)
    test = np.zeros_like(labels.valid)
    for site in range(labels.n_sites):
        for state in (False, True):
            indices = np.where(
                labels.valid[:, site] & (labels.occupied[:, site] == state)
            )[0]
            if not indices.size:
                continue
            permutation = rng.permutation(indices)
            train_count = int(round(fraction * indices.size))
            train_count = (
                min(max(train_count, 1), indices.size - 1)
                if indices.size >= 2
                else 1
            )
            train[permutation[:train_count], site] = True
            if train_count < indices.size:
                test[permutation[train_count:], site] = True
    return TrainTestSplit(train, test, int(seed), fraction)


def _common_bin_edges(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int,
) -> np.ndarray:
    samples = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        return np.linspace(-1.0, 1.0, bins + 1)
    lower, upper = np.quantile(samples, (0.001, 0.999))
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        lower, upper = float(np.min(samples)), float(np.max(samples))
    span = max(float(upper - lower), 1.0)
    return np.linspace(lower - 0.04 * span, upper + 0.04 * span, bins + 1)


def _empirical_threshold(
    dark: np.ndarray,
    bright: np.ndarray,
    edges: np.ndarray,
    *,
    bright_above: bool,
    tie_target: float,
) -> float:
    dark = np.asarray(dark, dtype=float)
    bright = np.asarray(bright, dtype=float)
    dark = dark[np.isfinite(dark)]
    bright = bright[np.isfinite(bright)]
    if not dark.size or not bright.size:
        return float("nan")
    dark_histogram, _ = np.histogram(dark, bins=edges)
    bright_histogram, _ = np.histogram(bright, bins=edges)
    dark_cumulative = np.concatenate([[0], np.cumsum(dark_histogram)])
    bright_cumulative = np.concatenate([[0], np.cumsum(bright_histogram)])
    if bright_above:
        dark_fidelity = dark_cumulative / dark.size
        bright_fidelity = (bright.size - bright_cumulative) / bright.size
    else:
        dark_fidelity = (dark.size - dark_cumulative) / dark.size
        bright_fidelity = bright_cumulative / bright.size
    fidelity = 0.5 * (dark_fidelity + bright_fidelity)
    best_value = float(np.nanmax(fidelity))
    best = np.flatnonzero(np.isclose(fidelity, best_value, rtol=0, atol=1e-12))
    index = (
        int(best[np.argmin(np.abs(edges[best] - tie_target))])
        if best.size > 1
        else int(best[0])
    )
    return float(edges[index])


def _fit_site_threshold(
    dark: np.ndarray,
    bright: np.ndarray,
    edges: np.ndarray,
) -> dict[str, float | bool | int]:
    dark = np.asarray(dark, dtype=float)
    bright = np.asarray(bright, dtype=float)
    dark = dark[np.isfinite(dark)]
    bright = bright[np.isfinite(bright)]
    if dark.size < 2 or bright.size < 2:
        return {
            "threshold": float("nan"),
            "bright_above": True,
            "dark_mean": float("nan"),
            "dark_sigma": float("nan"),
            "bright_mean": float("nan"),
            "bright_sigma": float("nan"),
            "model_fidelity": float("nan"),
            "n_train_dark": int(dark.size),
            "n_train_bright": int(bright.size),
        }
    dark_mean, bright_mean = float(np.mean(dark)), float(np.mean(bright))
    dark_sigma = max(float(np.std(dark, ddof=1)), 1e-12)
    bright_sigma = max(float(np.std(bright, ddof=1)), 1e-12)
    gaussian_threshold, bright_above = _optimal_gaussian_threshold(
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
    )
    threshold = _empirical_threshold(
        dark,
        bright,
        edges,
        bright_above=bright_above,
        tie_target=gaussian_threshold,
    )
    if not math.isfinite(threshold):
        threshold = gaussian_threshold
    _dark_fidelity, _bright_fidelity, model_fidelity = _gaussian_fidelity(
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        threshold,
        bright_above,
    )
    return {
        "threshold": threshold,
        "bright_above": bright_above,
        "dark_mean": dark_mean,
        "dark_sigma": dark_sigma,
        "bright_mean": bright_mean,
        "bright_sigma": bright_sigma,
        "model_fidelity": model_fidelity,
        "n_train_dark": int(dark.size),
        "n_train_bright": int(bright.size),
    }


def _confusion(
    prediction: np.ndarray,
    occupied: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=bool)[np.asarray(valid, dtype=bool)]
    occupied = np.asarray(occupied, dtype=bool)[np.asarray(valid, dtype=bool)]
    true_positive = int(np.sum(prediction & occupied))
    true_negative = int(np.sum(~prediction & ~occupied))
    false_positive = int(np.sum(prediction & ~occupied))
    false_negative = int(np.sum(~prediction & occupied))
    bright_count = int(np.sum(occupied))
    dark_count = int(np.sum(~occupied))
    dark_fidelity = true_negative / dark_count if dark_count else float("nan")
    bright_fidelity = true_positive / bright_count if bright_count else float("nan")
    fidelity = (
        0.5 * (dark_fidelity + bright_fidelity)
        if np.isfinite([dark_fidelity, bright_fidelity]).all()
        else float("nan")
    )
    return {
        "fidelity": fidelity,
        "dark_fidelity": dark_fidelity,
        "bright_fidelity": bright_fidelity,
        "n_valid": int(occupied.size),
        "errors": false_positive + false_negative,
    }


def characterize_readout(
    kind: ReadoutModelKind,
    short_signals: np.ndarray,
    short_validity: np.ndarray,
    labels: ReferenceLabels,
    split: TrainTestSplit,
    *,
    bins: int,
    max_drop: int,
) -> ModelCalibrationReport:
    short = np.asarray(short_signals, dtype=float)
    short_validity = np.asarray(short_validity, dtype=bool) & np.isfinite(short)
    if short.shape != labels.occupied.shape or short_validity.shape != short.shape:
        raise ValueError("short signals/validity must match reference label shape")
    combined_validity = labels.valid & short_validity
    edges = _common_bin_edges(short, combined_validity, bins=bins)
    quick = np.asarray(
        [
            otsu_threshold(short[:, site][short_validity[:, site]])
            for site in range(labels.n_sites)
        ],
        dtype=float,
    )
    metrics = []
    prediction = np.zeros_like(labels.occupied, dtype=bool)
    for site in range(labels.n_sites):
        finite = short_validity[:, site]
        train_mask = split.train[:, site] & labels.valid[:, site] & finite
        test_mask = split.test[:, site] & labels.valid[:, site] & finite
        fit = _fit_site_threshold(
            short[train_mask & ~labels.occupied[:, site], site],
            short[train_mask & labels.occupied[:, site], site],
            edges,
        )
        threshold = float(fit["threshold"])
        bright_above = bool(fit["bright_above"])
        if math.isfinite(threshold):
            prediction[:, site] = (
                short[:, site] > threshold
                if bright_above
                else short[:, site] < threshold
            )
        confusion = _confusion(
            prediction[:, site],
            labels.occupied[:, site],
            test_mask,
        )
        metrics.append(
            SiteFidelity(
                site,
                threshold,
                bright_above,
                float(confusion["fidelity"]),
                float(confusion["dark_fidelity"]),
                float(confusion["bright_fidelity"]),
                float(fit["model_fidelity"]),
                float(fit["dark_mean"]),
                float(fit["dark_sigma"]),
                float(fit["bright_mean"]),
                float(fit["bright_sigma"]),
                int(confusion["n_valid"]),
                int(fit["n_train_dark"]),
                int(fit["n_train_bright"]),
            )
        )
    aggregate = _confusion(
        prediction,
        labels.occupied,
        split.test & combined_validity,
    )["fidelity"]
    train_all = split.train & combined_validity
    global_fit = _fit_site_threshold(
        short[train_all & ~labels.occupied],
        short[train_all & labels.occupied],
        edges,
    )
    global_threshold = float(global_fit["threshold"])
    global_bright_above = bool(global_fit["bright_above"])
    if math.isfinite(global_threshold):
        global_prediction = (
            short > global_threshold
            if bool(global_fit["bright_above"])
            else short < global_threshold
        )
    else:
        global_prediction = np.zeros_like(labels.occupied, dtype=bool)
    # Label validity gates calibration statistics, not whether the same frame
    # can be classified at runtime.  Canonicalize only invalid measured
    # signals; every valid signal must reproduce the runtime decision.
    prediction[~short_validity] = False
    global_fidelity = _confusion(
        global_prediction,
        labels.occupied,
        split.test & combined_validity,
    )["fidelity"]
    order = list(
        np.argsort(
            [
                item.fidelity if math.isfinite(item.fidelity) else -np.inf
                for item in metrics
            ]
        )
    )
    ablation = []
    for drop_count in range(max_drop + 1):
        excluded = np.zeros(labels.n_sites, dtype=bool)
        for index in order[: min(drop_count, len(order))]:
            excluded[int(index)] = True
        confusion = _confusion(
            prediction,
            labels.occupied,
            split.test & combined_validity & ~excluded[np.newaxis, :],
        )
        ablation.append(
            AblationPoint(
                drop_count,
                excluded,
                float(confusion["fidelity"]),
                int(confusion["errors"]),
                int(confusion["n_valid"]),
            )
        )
    return ModelCalibrationReport(
        kind,
        quick,
        short,
        short_validity,
        edges,
        prediction,
        tuple(metrics),
        float(aggregate),
        global_threshold,
        global_bright_above,
        float(global_fidelity),
        tuple(ablation),
    )


def _normalize_frame_stack_validity(
    validity: np.ndarray,
    frame_contract: FrameContract,
) -> np.ndarray:
    """Conservatively express pixel observations in the declared frame contract."""

    schema = frame_contract.frame_schema
    mask = np.asarray(validity, dtype=bool)
    data_rank = len(schema.data_axes)
    if data_rank != 2 or mask.shape[-data_rank:] != schema.data_shape:
        raise ValueError("frame validity does not match the FrameContract")
    leading_rank = mask.ndim - data_rank
    contract = schema.validity_contract
    if contract.mode is ValidityMode.VALUE:
        reduced = np.all(
            mask,
            axis=tuple(range(leading_rank, mask.ndim)),
            keepdims=True,
        )
        return np.broadcast_to(reduced, mask.shape)
    declared = set(contract.component_axis_ids)
    omitted = tuple(
        leading_rank + index
        for index, axis in enumerate(schema.data_axes)
        if axis.axis_id not in declared
    )
    reduced = np.all(mask, axis=omitted, keepdims=True) if omitted else mask
    return np.broadcast_to(reduced, mask.shape)


def _prepare_frame(
    values: np.ndarray,
    validity: np.ndarray,
    frame_contract: FrameContract,
) -> tuple[np.ndarray, np.ndarray]:
    frame = np.asarray(values)
    mask = np.asarray(validity, dtype=bool)
    expected = frame_contract.frame_schema.data_shape
    if frame.shape != expected or mask.shape != expected:
        raise ValueError("frame and validity do not match the FrameContract")
    if np.issubdtype(frame.dtype, np.inexact):
        mask = mask & np.isfinite(frame)
    return frame, _normalize_frame_stack_validity(mask, frame_contract)


def _extract_source_stack(
    feature: ReadoutFeature,
    leading_shape: tuple[int, ...],
    frame_sequence: Callable[
        [],
        Iterator[tuple[np.ndarray, np.ndarray]],
    ],
    frame_contract: FrameContract,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty((*leading_shape, feature.site_axis.size), dtype=float)
    output_validity = np.empty(values.shape, dtype=bool)
    frames = _exact_frame_sequence(
        frame_sequence,
        math.prod(leading_shape),
        "feature frame sequence",
    )
    for index, raw_frame in zip(np.ndindex(leading_shape), frames, strict=True):
        frame, validity = _prepare_frame(*raw_frame, frame_contract)
        extracted, extracted_validity = _extract_readout_arrays(
            feature,
            frame,
            validity,
        )
        values[index] = extracted
        output_validity[index] = extracted_validity
    return values, output_validity


def _extract_source_feature_stacks(
    features: Sequence[ReadoutFeature],
    leading_shape: tuple[int, ...],
    frame_sequence: Callable[
        [],
        Iterator[tuple[np.ndarray, np.ndarray]],
    ],
    frame_contract: FrameContract,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract every model feature while traversing the frame source once."""

    ordered = tuple(features)
    if not ordered:
        raise ValueError("features must contain at least one ReadoutFeature")
    site_axis = ordered[0].site_axis
    site_count = site_axis.size
    if any(
        feature.site_axis.axis_id != site_axis.axis_id
        or feature.site_axis.size != site_count
        for feature in ordered
    ):
        raise ValueError("all readout features must use the same site axis")
    values = np.empty(
        (len(ordered), *leading_shape, site_count),
        dtype=float,
    )
    output_validity = np.empty(values.shape, dtype=bool)
    frames = _exact_frame_sequence(
        frame_sequence,
        math.prod(leading_shape),
        "feature frame sequence",
    )
    for index, raw_frame in zip(np.ndindex(leading_shape), frames, strict=True):
        frame, validity = _prepare_frame(*raw_frame, frame_contract)
        for model_index, feature in enumerate(ordered):
            extracted, extracted_validity = _extract_readout_arrays(
                feature,
                frame,
                validity,
            )
            output_index = (model_index, *index)
            values[output_index] = extracted
            output_validity[output_index] = extracted_validity
    return values, output_validity


def _exact_frame_sequence(
    factory: Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]],
    expected_count: int,
    field_name: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    iterator = iter(factory())
    for _index in range(expected_count):
        try:
            yield next(iterator)
        except StopIteration as exc:
            raise CalibrationAnalysisError(
                f"{field_name} ended before {expected_count} frames"
            ) from exc
    sentinel = object()
    if next(iterator, sentinel) is not sentinel:
        raise CalibrationAnalysisError(
            f"{field_name} contains more than {expected_count} frames"
        )


def _calibrate_readout_source(
    *,
    group_count: int,
    reference_shot_count: int,
    reference_frames: Callable[
        [],
        Iterator[tuple[np.ndarray, np.ndarray]],
    ],
    short_frames: Callable[
        [],
        Iterator[tuple[np.ndarray, np.ndarray]],
    ],
    group_contexts: tuple[tuple[tuple[AxisId, int], ...], ...],
    source_binding: CalibrationSourceBinding,
    frame_contract: FrameContract,
    readout_physical_context: ReadoutPhysicalContext,
    request: CalibrationAnalysisRequest,
) -> CalibrationComputation:
    if group_count <= 0:
        raise ValueError("calibration requires at least one frame group")
    if reference_shot_count != len(request.layout.reference_event_indices):
        raise ValueError("reference shot count differs from the capture layout")
    if len(group_contexts) != group_count:
        raise ValueError("calibration group contexts must be complete")
    image_shape = frame_contract.frame_schema.data_shape
    total = np.zeros(image_shape, dtype=float)
    count_dtype = np.min_scalar_type(group_count * reference_shot_count)
    count = np.zeros(image_shape, dtype=count_dtype)
    for raw_frame in _exact_frame_sequence(
        reference_frames,
        group_count * reference_shot_count,
        "reference frame sequence",
    ):
        frame, valid = _prepare_frame(*raw_frame, frame_contract)
        np.add(total, frame, out=total, where=valid, casting="unsafe")
        np.add(count, valid, out=count, casting="unsafe")
    average_validity = count > 0
    if not np.any(average_validity):
        raise CalibrationAnalysisError("reference frames contain no valid pixels")
    # Reuse the float64 accumulator for the average.  Calibration needs memory
    # proportional to one frame, not another frame-sized temporary per shot.
    average = total
    np.divide(total, count, out=average, where=average_validity)
    if np.all(average_validity):
        detector_image = average
    else:
        detector_image = np.array(average, copy=True)
        detector_image[~average_validity] = float(
            np.median(average[average_validity])
        )
    centers = find_site_centers(
        detector_image,
        request.grid_shape_yx,
        min_distance=request.detector_min_distance,
        threshold_rel=request.detector_threshold_rel,
        ordering=request.ordering,
        refine_half=request.detector_refine_half,
    )
    site_axis = AxisSpec(
        AxisId("readout-site"),
        "readout site",
        SITE,
        request.site_count,
        coordinates=tuple(range(request.site_count)),
    )
    all_sites = ComponentValidity(
        (site_axis.axis_id,),
        np.ones(site_axis.size, dtype=bool),
    )
    site_map = SiteMap(
        site_axis,
        centers,
        request.grid_shape_yx,
        request.ordering,
        frame_contract.coordinate_frame,
        all_sites,
    )
    box_feature = BoxFeature(
        site_axis,
        _boxes_for_centers(centers, average.shape, request.box_radius),
        request.box_reducer,
        all_sites,
    )
    reference_box_signals, reference_box_validity = _extract_source_stack(
        box_feature,
        (group_count, reference_shot_count),
        reference_frames,
        frame_contract,
    )
    labels = reference_labels(reference_box_signals, reference_box_validity)
    split = train_test_split(
        labels,
        train_fraction=request.train_fraction,
        seed=request.split_seed,
    )
    features: dict[ReadoutModelKind, ReadoutFeature] = {
        ReadoutModelKind.BOX: box_feature,
    }
    psf_fits: tuple[PsfFitDiagnostic, ...] = ()
    if any(kind is not ReadoutModelKind.BOX for kind in request.model_kinds):
        per_site_psf, uniform_psf, psf_fits = _fit_psf_features(
            average,
            average_validity,
            centers,
            site_axis,
            request,
        )
        features[ReadoutModelKind.PER_SITE_PSF] = per_site_psf
        features[ReadoutModelKind.UNIFORM_PSF] = uniform_psf
    ordered_features = tuple(features[kind] for kind in request.model_kinds)
    short_signal_stacks, short_validity_stacks = _extract_source_feature_stacks(
        ordered_features,
        (group_count,),
        short_frames,
        frame_contract,
    )
    reports = []
    for model_index, (kind, feature) in enumerate(
        zip(request.model_kinds, ordered_features, strict=True)
    ):
        signals = short_signal_stacks[model_index]
        signal_validity = short_validity_stacks[model_index]
        report = characterize_readout(
            kind,
            signals,
            signal_validity,
            labels,
            split,
            bins=request.histogram_bins,
            max_drop=request.max_drop,
        )
        reports.append(report)

    use_reference_thresholds = _main_reference_thresholds_available(reports)
    models = []
    for feature, report in zip(ordered_features, reports, strict=True):
        thresholds, usable = _runtime_model_values(
            feature,
            report,
            request,
            use_reference_thresholds=use_reference_thresholds,
        )
        models.append(
            ReadoutModel(
                feature,
                thresholds,
                ComponentValidity((site_axis.axis_id,), usable),
            )
        )
    artifact = CalibrationArtifact(
        source_binding,
        frame_contract,
        readout_physical_context,
        site_map,
        tuple(models),
        request.default_model_kind,
    )
    report = CalibrationReport(
        request,
        (("numpy", np.__version__), ("scipy", scipy.__version__)),
        group_contexts,
        average,
        average_validity,
        reference_box_signals,
        labels,
        split,
        psf_fits,
        tuple(reports),
    )
    return CalibrationComputation(artifact, report)


def _calibrate_readout_frames(
    reference_frames: np.ndarray,
    short_frames: np.ndarray,
    *,
    source_binding: CalibrationSourceBinding,
    frame_contract: FrameContract,
    readout_physical_context: ReadoutPhysicalContext,
    request: CalibrationAnalysisRequest,
    reference_validity: np.ndarray | None = None,
    short_validity: np.ndarray | None = None,
) -> CalibrationComputation:
    """Compute a non-submittable array oracle from ``(G,K,H,W)`` and ``(G,H,W)``.

    This path is for deterministic physics tests and offline diagnostics.  A
    durable calibration must be recomputed from an ``AdmittedCapture`` through
    :func:`analyze_calibration`; caller-supplied lineage never becomes commit
    authority.
    """

    if not isinstance(source_binding, CalibrationSourceBinding):
        raise TypeError("source_binding must be CalibrationSourceBinding")
    if not isinstance(frame_contract, FrameContract):
        raise TypeError("frame_contract must be FrameContract")
    if not isinstance(readout_physical_context, ReadoutPhysicalContext):
        raise TypeError(
            "readout_physical_context must be ReadoutPhysicalContext"
        )
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if source_binding.layout != request.layout:
        raise ValueError("source binding and request use different capture layouts")
    references = np.asarray(reference_frames)
    short = np.asarray(short_frames)
    if references.ndim != 4 or short.ndim != 3:
        raise ValueError(
            "reference_frames must be (groups, shots, y, x) and short_frames (groups, y, x)"
        )
    if references.shape[0] != short.shape[0] or references.shape[-2:] != short.shape[-2:]:
        raise ValueError("reference and short frames must share groups and image shape")
    if references.shape[-2:] != frame_contract.frame_schema.data_shape:
        raise ValueError("calibration images differ from the FrameContract")
    reference_valid = (
        np.broadcast_to(True, references.shape)
        if reference_validity is None
        else np.asarray(reference_validity, dtype=bool)
    )
    short_valid = (
        np.broadcast_to(True, short.shape)
        if short_validity is None
        else np.asarray(short_validity, dtype=bool)
    )
    if reference_valid.shape != references.shape or short_valid.shape != short.shape:
        raise ValueError("calibration validity masks have the wrong shape")
    group_axis = AxisId("calibration-group")
    contexts = tuple(
        ((group_axis, group),)
        for group in range(references.shape[0])
    )
    def reference_sequence() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for group, shot in np.ndindex(references.shape[:2]):
            yield references[group, shot], reference_valid[group, shot]

    def short_sequence() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for group in range(short.shape[0]):
            yield short[group], short_valid[group]

    return _calibrate_readout_source(
        group_count=references.shape[0],
        reference_shot_count=references.shape[1],
        reference_frames=reference_sequence,
        short_frames=short_sequence,
        group_contexts=contexts,
        source_binding=source_binding,
        frame_contract=frame_contract,
        readout_physical_context=readout_physical_context,
        request=request,
    )


def _capture_frame_source(
    source: CaptureFrameSource,
    join: _CalibrationCaptureJoin,
) -> tuple[
    int,
    Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]],
    Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]],
    tuple[tuple[tuple[AxisId, int], ...], ...],
]:
    if not isinstance(source, CaptureFrameSource):
        raise TypeError("source must be CaptureFrameSource")
    if not isinstance(join, _CalibrationCaptureJoin):
        raise TypeError("join must be _CalibrationCaptureJoin")
    if len(source.schema.cell_schema.data_shape) != 2:
        raise CalibrationAnalysisError("calibration source cells must be 2D frames")

    def frame_sequence(
        cell_sequence: Callable[[], Iterator[DatasetCellAddress]],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for expected, (cell, sample) in zip(
            cell_sequence(),
            source.iter_cells(cell_sequence()),
            strict=True,
        ):
            if cell != expected:
                raise CalibrationAnalysisError(
                    "capture frame source changed the requested cell order"
                )
            yield (
                sample.image.values,
                expand_value_validity(
                    sample.image.validity,
                    source.schema.cell_schema,
                ),
            )

    def reference_cells() -> Iterator[DatasetCellAddress]:
        for repeat, reference_rows, _readout_row in join.rows():
            for row in reference_rows:
                yield DatasetCellAddress(repeat, row)

    def short_cells() -> Iterator[DatasetCellAddress]:
        for repeat, _reference_rows, readout_row in join.rows():
            yield DatasetCellAddress(
                repeat,
                readout_row,
            )

    def reference_sequence() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return frame_sequence(reference_cells)

    def short_sequence() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return frame_sequence(short_cells)

    contexts = tuple(join.contexts())
    return join.group_count, reference_sequence, short_sequence, contexts


def _compute_calibration_resolved(
    capture: object,
    request: CalibrationAnalysisRequest,
    resolved: _ResolvedCalibrationSource,
) -> CalibrationComputation:
    """Run the science core from facts bound by one source resolution."""

    try:
        source = capture.frame_source  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("capture must be a resolved raw CaptureArtifact") from exc
    if not isinstance(source, CaptureFrameSource):
        raise TypeError("capture.frame_source must be CaptureFrameSource")
    group_count, reference_frames, short_frames, contexts = _capture_frame_source(
        source,
        resolved.join,
    )
    return _calibrate_readout_source(
        group_count=group_count,
        reference_shot_count=len(request.layout.reference_event_indices),
        reference_frames=reference_frames,
        short_frames=short_frames,
        group_contexts=contexts,
        source_binding=resolved.source_binding,
        frame_contract=resolved.frame_contract,
        readout_physical_context=resolved.readout_physical_context,
        request=request,
    )


def compute_calibration(
    capture: object,
    request: CalibrationAnalysisRequest,
) -> CalibrationComputation:
    """Compute a non-submittable result from one resolved raw CaptureArtifact."""

    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    from zlc_neutral_atom.artifacts.capture import CaptureArtifact

    if not isinstance(capture, CaptureArtifact):
        raise TypeError("capture must be a resolved raw CaptureArtifact")
    resolved = _resolve_calibration_source(capture, request.layout)
    return _compute_calibration_resolved(capture, request, resolved)


def _analyze_calibration_resolved(
    source: object,
    request: CalibrationAnalysisRequest,
    resolved: _ResolvedCalibrationSource,
) -> CalibrationAnalysisResult:
    from zlc_neutral_atom.artifacts.capture import AdmittedCapture

    if type(source) is not AdmittedCapture:
        raise TypeError("calibration analysis requires an exact AdmittedCapture")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if request.expected_centers_xy is None:
        raise CalibrationAnalysisError(
            "authoritative calibration requires expected_centers_xy and "
            "maximum_site_residual_px"
        )
    if not isinstance(resolved, _ResolvedCalibrationSource):
        raise TypeError("resolved must be _ResolvedCalibrationSource")
    computation = _compute_calibration_resolved(source.artifact, request, resolved)
    return CalibrationAnalysisResult._from_admitted_analysis(
        _ADMITTED_ANALYSIS_TOKEN,
        computation,
        source,
    )


def analyze_calibration(
    source: object,
    request: CalibrationAnalysisRequest,
) -> CalibrationAnalysisResult:
    """Analyze one repository-admitted capture into a submittable result."""

    from zlc_neutral_atom.artifacts.capture import AdmittedCapture

    if type(source) is not AdmittedCapture:
        raise TypeError("calibration analysis requires an exact AdmittedCapture")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    resolved = _resolve_calibration_source(source.artifact, request.layout)
    return _analyze_calibration_resolved(source, request, resolved)


__all__ = [
    "AblationPoint",
    "BimodalFit",
    "CalibrationAnalysisError",
    "CalibrationAnalysisResult",
    "CalibrationComputation",
    "CalibrationReport",
    "ModelCalibrationReport",
    "PsfFitDiagnostic",
    "ReferenceLabels",
    "SiteFidelity",
    "TrainTestSplit",
    "analyze_calibration",
    "characterize_readout",
    "compute_calibration",
    "estimate_calibration_analysis_peak_bytes",
    "find_site_centers",
    "fit_bimodal",
    "otsu_threshold",
    "reference_labels",
    "train_test_split",
]
