"""Typed Dataset projections of one admitted FINAL calibration.

Calibration owns the reference image, SITE diagnostics, validity and durable
lineage.  This module exposes only physical calibration values; the optional UI
adapter maps them into frontend's canonical Figure values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from zlc_data import (
    COMPONENT,
    REPEAT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    CoordinateFrameId,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_data._arrays import immutable_array
from zlc_data.value import expand_dataset_validity
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration
from zlc_storage import canonical_digest, canonical_text

from .calibration import (
    PerSitePsfFeature,
    UniformPsfFeature,
    site_grid_positions_yx,
)
from .reference import (
    CALIBRATION_ARTIFACT_REF_FORMAT,
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)

if TYPE_CHECKING:
    from .analysis import CalibrationComputation

__all__ = [
    "CALIBRATION_FINAL_OUTPUT_DECLARATIONS",
    "CALIBRATION_ARTIFACT_OUTPUT_DECLARATION",
    "CALIBRATION_DIAGNOSTIC_OUTPUT_DECLARATIONS",
    "CalibrationModelReportProjection",
    "CalibrationReportProjection",
    "CalibrationSiteMapContext",
    "calibration_final_outputs",
    "calibration_site_map_context",
    "materialize_calibration_report_datasets",
    "materialize_calibration_diagnostics",
    "materialize_calibration_reference_snapshot",
    "project_calibration_report",
]


CALIBRATION_DIAGNOSTIC_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "fidelity_site", "zlc_neutral_atom.calibration.site-fidelity"
    ),
    DatasetOutputDeclaration(
        "fidelity_threshold", "zlc_neutral_atom.calibration.site-threshold"
    ),
    DatasetOutputDeclaration(
        "fidelity_centers", "zlc_neutral_atom.calibration.site-centres"
    ),
    DatasetOutputDeclaration(
        "aggregate_fidelity", "zlc_neutral_atom.calibration.aggregate-fidelity"
    ),
    DatasetOutputDeclaration(
        "global_fidelity", "zlc_neutral_atom.calibration.global-fidelity"
    ),
)
CALIBRATION_ARTIFACT_OUTPUT_DECLARATION = ArtifactOutputDeclaration(
    "calibration",
    CALIBRATION_ARTIFACT_REF_FORMAT,
)
CALIBRATION_FINAL_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "site_map",
        "zlc_neutral_atom.calibration.site-map",
    ),
    *CALIBRATION_DIAGNOSTIC_OUTPUT_DECLARATIONS,
)
@dataclass(frozen=True, eq=False, slots=True)
class CalibrationSiteMapContext:
    """Calibration-owned geometry required to compose a SiteMap presentation.

    This carries physical calibration facts only.  It deliberately contains no
    frontend view, renderer, colour, label, or widget state.
    """

    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    site_validity: np.ndarray
    calibration_identity: str
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        centers = immutable_array(
            self.centers_xy,
            dtype=np.float64,
            shape=(self.site_axis.size, 2),
        )
        validity = immutable_array(
            self.site_validity,
            dtype=np.bool_,
            shape=(self.site_axis.size,),
        )
        if not np.isfinite(centers).all():
            raise ValueError("calibration site centers must be finite")
        identity = canonical_text(
            self.calibration_identity,
            "calibration_identity",
        )
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "site_validity", validity)
        object.__setattr__(self, "calibration_identity", identity)


@dataclass(frozen=True, eq=False, slots=True)
class CalibrationModelReportProjection:
    """Readout-owned, site-aligned facts needed by report presentation."""

    label: str
    is_default: bool
    signals: np.ndarray
    threshold_centered_signals: np.ndarray
    signal_validity: np.ndarray
    bin_edges: np.ndarray
    quick_thresholds: np.ndarray
    formal_thresholds: np.ndarray
    runtime_thresholds: np.ndarray
    runtime_threshold_sources: tuple[str, ...]
    feature_validity: np.ndarray
    runtime_usable: np.ndarray
    bright_above: np.ndarray
    model_fidelity: np.ndarray
    heldout_fidelity: np.ndarray
    dark_fidelity: np.ndarray
    bright_fidelity: np.ndarray
    dark_mean: np.ndarray
    dark_sigma: np.ndarray
    bright_mean: np.ndarray
    bright_sigma: np.ndarray
    n_test: np.ndarray
    n_train_dark: np.ndarray
    n_train_bright: np.ndarray
    runtime_model_fidelity_mean: float
    aggregate_fidelity: float
    global_threshold: float
    global_bright_above: bool
    global_fidelity: float
    ablation_drop_worst_k: np.ndarray
    ablation_excluded_sites: np.ndarray
    ablation_fidelity: np.ndarray
    ablation_errors: np.ndarray
    ablation_n_valid: np.ndarray
    __hash__ = None


@dataclass(frozen=True, eq=False, slots=True)
class CalibrationReportProjection:
    """Complete physical report projection; contains no renderer/UI state."""

    frame_schema: ValueSchema
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    reference_average: np.ndarray
    reference_average_validity: np.ndarray
    actual_centers_xy: np.ndarray
    expected_centers_xy: np.ndarray | None
    site_validity: np.ndarray
    default_boxes_xywh: np.ndarray
    grid_shape_yx: tuple[int, int]
    site_grid_positions_yx: tuple[tuple[int, int], ...]
    site_labels: tuple[str, ...]
    occupied_labels: np.ndarray
    dark_labels: np.ndarray
    label_validity: np.ndarray
    models: tuple[CalibrationModelReportProjection, ...]
    psf_kernels: np.ndarray | None
    psf_mode: str | None
    psf_fit_ok: np.ndarray | None
    psf_sigma_xy: np.ndarray | None
    calibration_ref: CalibrationArtifactRef
    source_capture_identity: str
    binding: str
    camera_identity: str
    roi_shape_yx: tuple[int, int]
    exposure_seconds: float
    group_count: int
    software_lineage: tuple[tuple[str, str], ...]
    __hash__ = None

    @property
    def calibration_identity(self) -> str:
        return self.calibration_ref.target_ref


def project_calibration_report(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> CalibrationReportProjection:
    """Project an admitted artifact/report pair without presentation inference."""

    from .analysis import calibration_runtime_threshold_sources

    _require_inputs(computation, reference)
    artifact = computation.artifact
    report = computation.report
    threshold_sources = calibration_runtime_threshold_sources(report)
    models: list[CalibrationModelReportProjection] = []
    psf_kernels = None
    psf_mode = None
    uniform_kernel = None
    for artifact_model, model_report, model_threshold_sources in zip(
        artifact.models,
        report.models,
        threshold_sources,
        strict=True,
    ):
        if artifact_model.kind is not model_report.kind:
            raise ValueError("calibration artifact/report model order differs")
        if isinstance(artifact_model.feature, PerSitePsfFeature):
            psf_kernels = artifact_model.feature.kernels
            psf_mode = "per-site"
        elif isinstance(artifact_model.feature, UniformPsfFeature):
            uniform_kernel = artifact_model.feature.kernel
        site_fidelity = model_report.site_fidelity
        site_count = artifact.site_map.site_axis.size
        model_fidelity = immutable_array(
            [item.model_fidelity for item in site_fidelity],
            dtype=np.float64,
            shape=(artifact.site_map.site_axis.size,),
        )
        runtime_usable = artifact_model.usable_sites.mask
        usable_model_fidelity = model_fidelity[
            runtime_usable & np.isfinite(model_fidelity)
        ]
        models.append(
            CalibrationModelReportProjection(
                label=artifact_model.kind.value,
                is_default=(
                    artifact_model.kind is artifact.default_model_kind
                ),
                signals=model_report.short_signals,
                threshold_centered_signals=immutable_array(
                    (
                        np.asarray(model_report.short_signals, dtype=np.float64)
                        - np.asarray(artifact_model.thresholds, dtype=np.float64)[
                            None, :
                        ]
                    ),
                    dtype=np.float64,
                    shape=model_report.short_signals.shape,
                ),
                signal_validity=model_report.short_validity,
                bin_edges=model_report.bin_edges,
                quick_thresholds=model_report.quick_thresholds,
                formal_thresholds=model_report.thresholds,
                runtime_thresholds=artifact_model.thresholds,
                runtime_threshold_sources=model_threshold_sources,
                feature_validity=artifact_model.feature.valid_sites.mask,
                runtime_usable=runtime_usable,
                bright_above=immutable_array(
                    [item.bright_above for item in site_fidelity],
                    dtype=np.bool_,
                    shape=(artifact.site_map.site_axis.size,),
                ),
                model_fidelity=model_fidelity,
                heldout_fidelity=immutable_array(
                    [item.fidelity for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                dark_fidelity=immutable_array(
                    [item.fidelity_dark for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                bright_fidelity=immutable_array(
                    [item.fidelity_bright for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                dark_mean=immutable_array(
                    [item.dark_mean for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                dark_sigma=immutable_array(
                    [item.dark_sigma for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                bright_mean=immutable_array(
                    [item.bright_mean for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                bright_sigma=immutable_array(
                    [item.bright_sigma for item in site_fidelity],
                    dtype=np.float64,
                    shape=(site_count,),
                ),
                n_test=immutable_array(
                    [item.n_test for item in site_fidelity],
                    dtype=np.int64,
                    shape=(site_count,),
                ),
                n_train_dark=immutable_array(
                    [item.n_train_dark for item in site_fidelity],
                    dtype=np.int64,
                    shape=(site_count,),
                ),
                n_train_bright=immutable_array(
                    [item.n_train_bright for item in site_fidelity],
                    dtype=np.int64,
                    shape=(site_count,),
                ),
                runtime_model_fidelity_mean=(
                    float(np.mean(usable_model_fidelity))
                    if usable_model_fidelity.size
                    else float("nan")
                ),
                aggregate_fidelity=float(model_report.aggregate_fidelity),
                global_threshold=float(model_report.global_threshold),
                global_bright_above=bool(model_report.global_bright_above),
                global_fidelity=float(model_report.global_fidelity),
                ablation_drop_worst_k=immutable_array(
                    [item.drop_worst_k for item in model_report.ablation],
                    dtype=np.int64,
                    shape=(len(model_report.ablation),),
                ),
                ablation_excluded_sites=immutable_array(
                    (
                        np.stack(
                            [item.excluded_sites for item in model_report.ablation]
                        )
                        if model_report.ablation
                        else np.empty((0, site_count), dtype=np.bool_)
                    ),
                    dtype=np.bool_,
                    shape=(len(model_report.ablation), site_count),
                ),
                ablation_fidelity=immutable_array(
                    [item.fidelity for item in model_report.ablation],
                    dtype=np.float64,
                    shape=(len(model_report.ablation),),
                ),
                ablation_errors=immutable_array(
                    [item.errors for item in model_report.ablation],
                    dtype=np.int64,
                    shape=(len(model_report.ablation),),
                ),
                ablation_n_valid=immutable_array(
                    [item.n_valid for item in model_report.ablation],
                    dtype=np.int64,
                    shape=(len(model_report.ablation),),
                ),
            )
        )
    if psf_kernels is None and uniform_kernel is not None:
        psf_kernels = np.broadcast_to(
            uniform_kernel,
            (artifact.site_map.site_axis.size, *uniform_kernel.shape),
        )
        psf_mode = "uniform"
    psf_fit_ok = psf_sigma = None
    if psf_kernels is not None:
        if len(report.psf_fits) != artifact.site_map.site_axis.size:
            raise ValueError("empirical PSF kernels lack aligned fit diagnostics")
        psf_fit_ok = immutable_array(
            [item.fit_ok for item in report.psf_fits],
            dtype=np.bool_,
            shape=(artifact.site_map.site_axis.size,),
        )
        psf_sigma = immutable_array(
            [item.sigma_xy for item in report.psf_fits],
            dtype=np.float64,
            shape=(artifact.site_map.site_axis.size, 2),
        )
    frame = artifact.frame_contract
    return CalibrationReportProjection(
        frame_schema=frame.frame_schema,
        site_axis=artifact.site_map.site_axis,
        coordinate_frame=artifact.site_map.coordinate_frame,
        reference_average=report.reference_average,
        reference_average_validity=report.reference_average_validity,
        actual_centers_xy=artifact.site_map.coordinates_xy,
        expected_centers_xy=report.request.expected_centers_xy,
        site_validity=artifact.site_map.validity.mask,
        default_boxes_xywh=artifact.select_model().feature.boxes_xywh,
        grid_shape_yx=artifact.site_map.grid_shape_yx,
        site_grid_positions_yx=site_grid_positions_yx(
            artifact.site_map.grid_shape_yx,
            artifact.site_map.ordering,
        ),
        site_labels=tuple(
            str(value) for value in artifact.site_map.site_axis.coordinates
        ),
        occupied_labels=report.labels.occupied,
        dark_labels=report.labels.dark,
        label_validity=report.labels.valid,
        models=tuple(models),
        psf_kernels=psf_kernels,
        psf_mode=psf_mode,
        psf_fit_ok=psf_fit_ok,
        psf_sigma_xy=psf_sigma,
        calibration_ref=reference,
        source_capture_identity=(
            artifact.source_binding.source_capture_ref.target_ref
        ),
        binding=frame.binding.value,
        camera_identity=frame.camera_identity,
        roi_shape_yx=frame.roi_shape_yx,
        exposure_seconds=frame.exposure_seconds,
        group_count=len(report.group_contexts),
        software_lineage=report.software_lineage,
    )


def _require_inputs(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> None:
    from .analysis import CalibrationComputation

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("computation must be CalibrationComputation")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")


def _report_snapshot(
    view: CalibrationReportProjection,
    key: str,
    schema: DatasetSchema,
    values: np.ndarray,
    validity,
) -> OwnedSnapshot:
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.calibration-report-dataset",
            "calibration_identity": view.calibration_identity,
            "key": key,
            "schema_fingerprint": schema.fingerprint,
        }
    )
    block = DataBlock(
        BlockId(f"calibration-report-{key}-{identity[:20]}"),
        DatasetRevision(0),
        np.asarray(values).reshape(schema.physical_shape),
        validity,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"calibration-report-{identity}")),
        block,
    )


def _calibration_reference_snapshot(
    frame_schema: ValueSchema,
    reference_average: np.ndarray,
    reference_average_validity: np.ndarray,
    reference: CalibrationArtifactRef,
) -> OwnedSnapshot:
    """Single canonical SiteMap carrier shared by FINAL and report surfaces."""

    if not isinstance(frame_schema, ValueSchema):
        raise TypeError("frame_schema must be ValueSchema")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    if len(frame_schema.data_axes) != 2:
        raise ValueError("calibration reference requires two declared frame axes")
    if np.asarray(reference_average).shape != frame_schema.data_shape:
        raise ValueError("calibration reference average differs from FrameContract")
    if np.asarray(reference_average_validity).shape != frame_schema.data_shape:
        raise ValueError("calibration reference validity differs from FrameContract")
    axes = frame_schema.data_axes
    schema = DatasetSchema(
        AxisSpec(
            AxisId("calibration.repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        PointTable(1),
        None,
        ValueSchema(
            axes,
            ValidityContract.components(*(axis.axis_id for axis in axes)),
            np.dtype("<f8"),
            frame_schema.value_unit,
        ),
    )
    identity = reference.manifest_digest
    block = DataBlock(
        BlockId(f"calibration-reference-{identity[:20]}"),
        DatasetRevision(0),
        np.asarray(reference_average, dtype="<f8").reshape(schema.physical_shape),
        DatasetComponentValidity(
            tuple(axis.axis_id for axis in axes),
            np.asarray(reference_average_validity, dtype=np.bool_).reshape(
                schema.physical_shape
            ),
        ),
        schema,
    )
    generation = StreamGenerationId(f"calibration-reference-{identity}")
    return OwnedSnapshot(block.ref(generation), block)


def _report_repeat_axis(key: str, size: int) -> AxisSpec:
    return AxisSpec(
        AxisId(f"calibration-report.{key}.repeat"),
        "sample",
        REPEAT,
        size,
        tuple(range(size)),
    )


def _report_population_axis(key: str) -> AxisSpec:
    return AxisSpec(
        AxisId(f"calibration-report.{key}.population"),
        "population",
        COMPONENT,
        2,
        ("dark", "bright"),
    )


def _point_table_for_axis(axis: AxisSpec) -> PointTable:
    """Project one declared report coordinate onto the physical point rows."""

    if not isinstance(axis, AxisSpec):
        raise TypeError("report point coordinate must be an AxisSpec")
    values = tuple(range(axis.size)) if axis.coordinates is None else axis.coordinates
    value_kind = (
        PointColumn.TEXT
        if all(value is None or isinstance(value, str) for value in values)
        else PointColumn.NUMERIC
    )
    return PointTable(
        axis.size,
        (
            PointColumn(
                axis.axis_id,
                axis.name,
                axis.role,
                value_kind,
                values,
                axis.unit,
                axis.coordinate_frame,
            ),
        ),
    )


def _report_population_values(
    view: CalibrationReportProjection,
    model: CalibrationModelReportProjection,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype="<f8")
    expected = (view.group_count, len(view.site_labels))
    if source.shape != expected:
        raise ValueError(
            f"calibration model samples have shape {source.shape}, expected {expected}"
        )
    common = (
        np.asarray(model.signal_validity, dtype=np.bool_)
        & np.asarray(view.label_validity, dtype=np.bool_)
    )
    population = np.stack(
        (
            np.asarray(view.dark_labels, dtype=np.bool_),
            np.asarray(view.occupied_labels, dtype=np.bool_),
        ),
        axis=-1,
    )
    return (
        np.repeat(source[..., None], 2, axis=-1),
        common[..., None] & population & np.isfinite(source)[..., None],
    )


def materialize_calibration_report_datasets(
    view: CalibrationReportProjection,
) -> dict[str, OwnedSnapshot]:
    """Declare every report plot as an ordinary typed Dataset.

    This is the sole physical projection for report pages.  Frontends choose
    their existing IMAGE/CURVE/HISTOGRAM views over these snapshots; they do
    not subtract thresholds, classify populations, infer grid positions, or
    reshape SITE-valued authority themselves.
    """

    if not isinstance(view, CalibrationReportProjection):
        raise TypeError("view must be CalibrationReportProjection")
    output: dict[str, OwnedSnapshot] = {}

    output["reference"] = _calibration_reference_snapshot(
        view.frame_schema,
        view.reference_average,
        view.reference_average_validity,
        view.calibration_ref,
    )

    model_axis = AxisSpec(
        AxisId("calibration-report.fidelity.model"),
        "readout model",
        COMPONENT,
        len(view.models),
        tuple(model.label for model in view.models),
    )
    # The calibration SITE coordinate may be a human label.  The interactive
    # curve contract needs a numeric, monotonic x coordinate, so this report
    # projection declares the positional index explicitly instead of changing
    # or guessing the authoritative SiteMap coordinate vocabulary.
    site_index_axis = AxisSpec(
        AxisId("calibration-report.fidelity.site-index"),
        "canonical site index",
        SITE,
        view.site_axis.size,
        tuple(range(view.site_axis.size)),
    )
    fidelity_values = np.stack(
        [np.asarray(model.model_fidelity, dtype="<f8") for model in view.models],
        axis=-1,
    )
    fidelity_validity = np.stack(
        [
            np.asarray(model.runtime_usable, dtype=np.bool_)
            & np.isfinite(np.asarray(model.model_fidelity, dtype=np.float64))
            for model in view.models
        ],
        axis=-1,
    )
    fidelity_schema = DatasetSchema(
        _report_repeat_axis("fidelity", 1),
        _point_table_for_axis(site_index_axis),
        None,
        ValueSchema(
            (model_axis,),
            ValidityContract.components(model_axis.axis_id),
            np.dtype("<f8"),
            "fidelity",
        ),
    )
    output["fidelity"] = _report_snapshot(
        view,
        "fidelity",
        fidelity_schema,
        fidelity_values.reshape(fidelity_schema.physical_shape),
        DatasetComponentValidity(
            (model_axis.axis_id,),
            fidelity_validity.reshape(fidelity_schema.physical_shape),
        ),
    )

    for model in view.models:
        histogram_key = f"hist-{model.label}"
        population_axis = _report_population_axis(histogram_key)
        histogram_values, histogram_validity = _report_population_values(
            view,
            model,
            model.threshold_centered_signals,
        )
        histogram_schema = DatasetSchema(
            _report_repeat_axis(histogram_key, view.group_count),
            _point_table_for_axis(view.site_axis),
            None,
            ValueSchema(
                (population_axis,),
                ValidityContract.components(population_axis.axis_id),
                np.dtype("<f8"),
                view.frame_schema.value_unit,
            ),
        )
        output[histogram_key] = _report_snapshot(
            view,
            histogram_key,
            histogram_schema,
            histogram_values,
            DatasetComponentValidity(
                (population_axis.axis_id,),
                histogram_validity.reshape(histogram_schema.physical_shape),
            ),
        )

    if view.psf_kernels is not None:
        kernels = np.asarray(view.psf_kernels, dtype="<f8")
        if kernels.ndim != 3 or kernels.shape[0] != len(view.site_labels):
            raise ValueError("PSF report kernels must have shape (site, y, x)")
        key = "psf-kernels"
        psf_frame = CoordinateFrameId(
            f"calibration-psf-{canonical_digest({'calibration': view.calibration_identity})}"
        )
        y_axis = AxisSpec(
            AxisId("calibration-report.psf.y"),
            "PSF y",
            SPATIAL_Y,
            kernels.shape[1],
            tuple(range(kernels.shape[1])),
            "pixel",
            psf_frame,
        )
        x_axis = AxisSpec(
            AxisId("calibration-report.psf.x"),
            "PSF x",
            SPATIAL_X,
            kernels.shape[2],
            tuple(range(kernels.shape[2])),
            "pixel",
            psf_frame,
        )
        psf_schema = DatasetSchema(
            _report_repeat_axis(key, 1),
            _point_table_for_axis(view.site_axis),
            None,
            ValueSchema(
                (y_axis, x_axis),
                ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
                np.dtype("<f8"),
                "normalized weight",
            ),
        )
        pixel_validity = np.broadcast_to(
            np.asarray(view.site_validity, dtype=np.bool_)[:, None, None],
            kernels.shape,
        )
        output[key] = _report_snapshot(
            view,
            key,
            psf_schema,
            kernels.reshape(psf_schema.physical_shape),
            DatasetComponentValidity(
                (y_axis.axis_id, x_axis.axis_id),
                pixel_validity.reshape(psf_schema.physical_shape),
            ),
        )
    return output


def materialize_calibration_reference_snapshot(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> OwnedSnapshot:
    """Materialize the report's reference-average image as one typed Dataset."""

    _require_inputs(computation, reference)
    artifact = computation.artifact
    report = computation.report
    return _calibration_reference_snapshot(
        artifact.frame_contract.frame_schema,
        report.reference_average,
        report.reference_average_validity,
        reference,
    )


def _diagnostic_snapshot(
    reference: CalibrationArtifactRef,
    output_name: str,
    values,
    data_axes: tuple[AxisSpec, ...],
    *,
    value_unit: str | None,
    validity_axis_ids: tuple[AxisId, ...] = (),
    validity_mask,
) -> OwnedSnapshot:
    array = np.asarray(values, dtype="<f8")
    axes = tuple(data_axes)
    expected = tuple(axis.size for axis in axes)
    if array.shape != expected:
        raise ValueError(
            f"{output_name} values have shape {array.shape}, expected {expected}"
        )
    repeat_axis = AxisSpec(
        AxisId("calibration.repeat"),
        "repeat",
        REPEAT,
        1,
        (0,),
    )
    if axes:
        axis_ids = tuple(validity_axis_ids)
        if not axis_ids:
            raise ValueError(
                f"{output_name} must declare the axes its validity follows"
            )
        validity_contract = ValidityContract.components(*axis_ids)
        value_schema = ValueSchema(
            axes,
            validity_contract,
            np.dtype("<f8"),
            value_unit,
        )
    else:
        axis_ids = ()
        value_schema = ValueSchema.scalar(np.dtype("<f8"), value_unit)
    schema = DatasetSchema(
        repeat_axis,
        PointTable(1),
        None,
        value_schema,
    )
    if axis_ids:
        axis_by_id = {axis.axis_id: axis for axis in axes}
        mask_shape = tuple(axis_by_id[axis_id].size for axis_id in axis_ids)
        mask = np.asarray(validity_mask, dtype=np.bool_)
        if mask.shape != mask_shape:
            raise ValueError(
                f"{output_name} validity has shape {mask.shape}, "
                f"expected {mask_shape}"
            )
        validity = DatasetComponentValidity(
            axis_ids,
            mask.reshape((1, 1, *mask_shape)),
        )
    else:
        mask = np.asarray(validity_mask, dtype=np.bool_)
        if mask.shape != ():
            raise ValueError(f"{output_name} scalar validity must be scalar")
        validity = CellValidity(mask.reshape((1, 1)))

    physical = array.reshape(schema.physical_shape)
    expanded_validity = np.asarray(
        expand_dataset_validity(validity, schema),
        dtype=np.bool_,
    )
    canonical = np.zeros(schema.physical_shape, dtype="<f8")
    np.copyto(canonical, physical, where=expanded_validity)
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.logic_nodes.readout.calibration.diagnostic",
            "calibration": calibration_artifact_ref_to_tree(reference),
            "output_name": output_name,
        }
    )
    block = DataBlock(
        BlockId(f"calibration-{output_name.replace('_', '-')}-{identity[:20]}"),
        DatasetRevision(0),
        canonical,
        validity,
        schema,
    )
    generation = StreamGenerationId(f"calibration-diagnostic-{identity}")
    return OwnedSnapshot(block.ref(generation), block)


def materialize_calibration_diagnostics(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> dict[str, OwnedSnapshot]:
    """Materialize the complete ordered report-owned diagnostic vocabulary."""

    _require_inputs(computation, reference)
    artifact = computation.artifact
    report = computation.report
    model = artifact.select_model()
    model_report = report.model(model.kind)
    site_axis = artifact.site_map.site_axis
    site_map_valid = np.asarray(artifact.site_map.validity.mask, dtype=np.bool_)
    feature_valid = (
        site_map_valid
        & np.asarray(model.feature.valid_sites.mask, dtype=np.bool_)
    )
    fidelity_evidence = np.asarray(
        [item.n_test > 0 for item in model_report.site_fidelity],
        dtype=np.bool_,
    )
    threshold_evidence = np.asarray(
        [
            item.n_train_dark > 0 and item.n_train_bright > 0
            for item in model_report.site_fidelity
        ],
        dtype=np.bool_,
    )
    site_fidelity = np.asarray(
        [item.fidelity for item in model_report.site_fidelity],
        dtype="<f8",
    )
    thresholds = np.asarray(model_report.thresholds, dtype="<f8")
    centers = np.asarray(artifact.site_map.coordinates_xy, dtype="<f8")
    if site_fidelity.shape != (site_axis.size,):
        raise ValueError("calibration fidelity does not follow its SITE axis")
    if thresholds.shape != (site_axis.size,):
        raise ValueError("calibration thresholds do not follow its SITE axis")
    if centers.shape != (site_axis.size, 2):
        raise ValueError("calibration centres must have shape (sites, 2)")

    output: dict[str, OwnedSnapshot] = {}
    output["fidelity_site"] = _diagnostic_snapshot(
            reference,
            "fidelity_site",
            site_fidelity,
            (site_axis,),
            value_unit="fidelity",
            validity_axis_ids=(site_axis.axis_id,),
            validity_mask=(
                feature_valid & fidelity_evidence & np.isfinite(site_fidelity)
            ),
        )
    output["fidelity_threshold"] = _diagnostic_snapshot(
            reference,
            "fidelity_threshold",
            thresholds,
            (site_axis,),
            value_unit=artifact.frame_contract.frame_schema.value_unit,
            validity_axis_ids=(site_axis.axis_id,),
            validity_mask=(
                feature_valid & threshold_evidence & np.isfinite(thresholds)
            ),
        )
    coordinate_axis = AxisSpec(
        AxisId(f"{site_axis.axis_id.value}.coordinate"),
        "coordinate",
        COMPONENT,
        2,
        ("x", "y"),
    )
    output["fidelity_centers"] = _diagnostic_snapshot(
            reference,
            "fidelity_centers",
            centers,
            (site_axis, coordinate_axis),
            value_unit="px",
            validity_axis_ids=(site_axis.axis_id,),
            validity_mask=site_map_valid & np.all(np.isfinite(centers), axis=1),
        )
    for output_name, value in (
        ("aggregate_fidelity", model_report.aggregate_fidelity),
        ("global_fidelity", model_report.global_fidelity),
    ):
        numeric = float(value)
        output[output_name] = _diagnostic_snapshot(
            reference,
            output_name,
            np.asarray(numeric, dtype="<f8"),
            (),
            value_unit="fidelity",
            validity_mask=np.asarray(np.isfinite(numeric)),
        )
    return output


def calibration_final_outputs(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> dict[str, FinalDatasetOutput]:
    """Publish the exact declared outputs of one admitted calibration.

    Numeric diagnostics and the reference-average SiteMap carrier are ordinary
    FINAL Datasets.  The ``calibration`` Artifact output is declared separately
    and resolves to the Run's exact ``CalibrationArtifactRef``; it is never
    disguised as image data.  SiteMap geometry is available through
    :func:`calibration_site_map_context`; numeric outputs never carry an open
    metadata bag.
    """

    _require_inputs(computation, reference)
    source_identity = calibration_artifact_ref_to_tree(reference)
    snapshots = {
        CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0].name: (
            materialize_calibration_reference_snapshot(computation, reference)
        ),
        **materialize_calibration_diagnostics(computation, reference),
    }
    expected = tuple(
        declaration.name for declaration in CALIBRATION_FINAL_OUTPUT_DECLARATIONS
    )
    if tuple(snapshots) != expected:
        raise RuntimeError("calibration output materializer changed its public order")
    outputs: dict[str, FinalDatasetOutput] = {}
    for declaration, snapshot in zip(
        CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
        snapshots.values(),
        strict=True,
    ):
        outputs[declaration.name] = FinalDatasetOutput(
            declaration,
            snapshot,
            final_dataset_join_digest(
                owner="zlc_neutral_atom.logic_nodes.readout.calibration.final-output",
                declaration=declaration,
                source_identity=source_identity,
                snapshot=snapshot,
            ),
        )
    return outputs


def calibration_site_map_context(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> CalibrationSiteMapContext:
    """Return the closed physical context for the calibration SiteMap view."""

    _require_inputs(computation, reference)
    site_map = computation.artifact.site_map
    return CalibrationSiteMapContext(
        site_map.site_axis,
        site_map.coordinate_frame,
        site_map.coordinates_xy,
        site_map.validity.mask,
        reference.target_ref,
    )
