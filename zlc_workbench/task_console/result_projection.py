"""Project successful domain results onto TaskConsole's immutable data plane.

This is a presentation adapter, not another result or artifact authority.  It
exposes already-FINAL dataset snapshots and freezes one reactive Camera front
with its derived occupancy presentation under outputs the catalog actually
declared.  Artifact admission/materialisation remains delegated to the owning
notebook/domain facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from zlc_data import (
    COMPONENT,
    MONITOR_HISTORY,
    READOUT_EVENT,
    REPEAT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    dataset_revision_ref_to_tree,
    expand_dataset_validity,
    IndexSelection,
    OwnedSnapshot,
    PointLayout,
    Selection,
    selection_to_tree,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.mot_field import MotFieldResult
from zlc_neutral_atom.readout.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout_duration_application import (
    ReadoutDurationFidelityResult,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.scan import ScanArtifactRef
from zlc_neutral_atom.timing.occupancy import (
    TriggeredOccupancyPipelineResult,
)
from zlc_neutral_atom.timing.release_recapture import (
    TriggeredReleaseRecaptureResult,
)
from zlc_storage import canonical_digest

if TYPE_CHECKING:
    from zlc_frontend.site_map_render import OccupancyCellView

__all__ = [
    "ProjectedFinalSignal",
    "project_final_signals",
    "project_reactive_occupancy",
]


@dataclass(frozen=True, slots=True)
class ProjectedFinalSignal:
    """One already-owned FINAL snapshot plus its durable source digest."""

    snapshot: OwnedSnapshot
    join_digest: str
    presentation: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.join_digest, str) or not self.join_digest:
            raise ValueError("join_digest must be a non-empty string")


def _evaluated_axis(axis: AxisSpec):
    """Project one complete declared axis without guessing from array shape."""

    from zlc_frontend.figure import EvaluatedAxis

    indices = tuple(range(axis.size))
    return EvaluatedAxis(
        axis.axis_id,
        axis.name,
        axis.role,
        axis.unit,
        indices,
        tuple(axis.coordinate_at(index) for index in indices),
        axis.coordinate_frame,
    )


def _occupancy_rate_snapshot(occupied: OwnedSnapshot) -> OwnedSnapshot:
    """Reduce only the declared SITE axis into one validity-aware cell scalar."""

    if not isinstance(occupied, OwnedSnapshot):
        raise TypeError("occupied must be an OwnedSnapshot")
    schema = occupied.block.schema
    axes = schema.cell_schema.data_axes
    if len(axes) != 1 or axes[0].role != SITE:
        raise ValueError("occupancy rate requires exactly one declared SITE axis")
    validity = np.asarray(
        expand_dataset_validity(occupied.block.validity, schema),
        dtype=np.bool_,
    )
    values = np.asarray(occupied.block.values, dtype=np.bool_)
    denominator = np.count_nonzero(validity, axis=2)
    numerator = np.count_nonzero(values & validity, axis=2)
    cell_validity = denominator > 0
    rate_values = np.zeros(cell_validity.shape, dtype="<f8")
    np.divide(
        numerator,
        denominator,
        out=rate_values,
        where=cell_validity,
    )
    rate_schema = DatasetSchema(
        schema.repeat_axis,
        schema.point_axes,
        schema.point_layout,
        ValueSchema(
            (),
            ValidityContract.value(),
            np.dtype("<f8"),
            "occupation",
        ),
    )
    block = DataBlock(
        BlockId("occupancy-rate"),
        occupied.block.revision,
        rate_values,
        CellValidity(cell_validity),
        rate_schema,
    )
    return OwnedSnapshot(block.ref(occupied.ref.stream_generation), block)


def _current_camera_cell_selection(
    schema: DatasetSchema,
    coverage: MonitorCoverage,
) -> tuple[int, tuple[int, ...], Selection]:
    """Resolve Main's current ``frame_0`` from a typed current-frame view."""

    if not isinstance(coverage, MonitorCoverage):
        raise TypeError(
            "current Camera selection requires MonitorCoverage; a formal "
            "dataset does not identify its current cell"
        )
    if coverage.written_cells == 0:
        raise ValueError("the current-frame Camera view has no committed cell")
    if schema.repeat_axis.size != 1:
        raise ValueError(
            "a current-frame Camera view requires one storage repeat"
        )
    history_axes = tuple(
        axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY
    )
    if len(history_axes) != 1:
        raise ValueError(
            "a current-frame Camera view requires one MONITOR_HISTORY axis"
        )
    event_axes = tuple(
        axis for axis in schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(event_axes) > 1:
        raise ValueError("a Camera presentation has multiple READOUT_EVENT axes")
    allowed = {MONITOR_HISTORY, READOUT_EVENT}
    if any(axis.role not in allowed for axis in schema.point_axes):
        raise ValueError("a Camera preview contains an unsupported point-axis role")

    logical = []
    terms = [IndexSelection(schema.repeat_axis.axis_id, 0)]
    for axis in schema.point_axes:
        # MonitorDataset materializes newest history at index zero; Main's
        # default Judge-occupancy source is frame_0.  Both choices are declared
        # role semantics, not rank/singleton inference.
        index = 0
        logical.append(index)
        terms.append(IndexSelection(axis.axis_id, index))
    logical_point = tuple(logical)
    return (
        schema.point_layout.storage_index(logical_point),
        logical_point,
        Selection(tuple(terms)),
    )


def _reactive_occupancy_cell_view(
    source: OwnedSnapshot,
    occupied: OwnedSnapshot,
    calibration: ResolvedCalibration,
    *,
    coverage: MonitorCoverage,
    run_id: str,
    epoch_id: str,
) -> "OccupancyCellView":
    """Compose the exact current Camera cell and its derived occupancy state."""

    from zlc_frontend.figure import (
        DatasetId,
        EvaluatedImage,
        EvaluatedInput,
    )
    from zlc_frontend.image_view import ImageViewportTransform
    from zlc_frontend.site_map import site_ring_radius
    from zlc_frontend.site_map_render import OccupancyCellView

    if not isinstance(source, OwnedSnapshot) or not isinstance(
        occupied, OwnedSnapshot
    ):
        raise TypeError("source and occupied must be OwnedSnapshot values")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    if source.ref.revision != occupied.ref.revision:
        raise ValueError("occupancy revision differs from its Camera source")
    source_schema = source.block.schema
    occupied_schema = occupied.block.schema
    if (
        source_schema.repeat_axis != occupied_schema.repeat_axis
        or source_schema.point_axes != occupied_schema.point_axes
        or source_schema.point_layout != occupied_schema.point_layout
    ):
        raise ValueError("occupancy outer axes differ from its Camera source")

    point_index, logical_point, selection = _current_camera_cell_selection(
        source_schema,
        coverage,
    )
    frame_axes = source_schema.cell_schema.data_axes
    x_positions = tuple(
        index for index, axis in enumerate(frame_axes) if axis.role == SPATIAL_X
    )
    y_positions = tuple(
        index for index, axis in enumerate(frame_axes) if axis.role == SPATIAL_Y
    )
    if (
        len(frame_axes) != 2
        or len(x_positions) != 1
        or len(y_positions) != 1
    ):
        raise ValueError(
            "physical occupancy presentation requires exactly one SPATIAL_X "
            "and SPATIAL_Y frame axis"
        )
    x_position, y_position = x_positions[0], y_positions[0]
    x_axis, y_axis = frame_axes[x_position], frame_axes[y_position]
    site_axes = occupied_schema.cell_schema.data_axes
    if len(site_axes) != 1 or site_axes[0].role != SITE:
        raise ValueError("occupancy presentation requires one declared SITE axis")
    site_axis = site_axes[0]
    site_map = calibration.artifact.site_map
    if site_axis != site_map.site_axis:
        raise ValueError("occupancy SITE axis differs from its calibration")
    if (
        x_axis.coordinate_frame != site_map.coordinate_frame
        or y_axis.coordinate_frame != site_map.coordinate_frame
    ):
        raise ValueError(
            "Camera spatial axes and calibration centers use different "
            "coordinate frames"
        )

    frame_validity = expand_dataset_validity(
        source.block.validity,
        source_schema,
    )[0, point_index]
    frame_values = source.block.values[0, point_index]
    order_yx = (y_position, x_position)
    background = EvaluatedImage(
        _evaluated_axis(x_axis),
        _evaluated_axis(y_axis),
        np.transpose(frame_values, order_yx),
        np.transpose(frame_validity, order_yx),
        source_schema.cell_schema.value_unit,
    )
    occupied_values = np.asarray(
        occupied.block.values[0, point_index],
        dtype=np.bool_,
    )
    site_validity = np.asarray(
        expand_dataset_validity(
            occupied.block.validity,
            occupied_schema,
        )[0, point_index],
        dtype=np.bool_,
    )
    if np.any(site_validity & ~site_map.validity.mask):
        raise ValueError("occupancy marks a calibration-invalid site as valid")

    reference = calibration.reference
    identity = canonical_digest(
        {
            "owner": "zlc-workbench.task-console.reactive-occupancy-cell",
            "source": dataset_revision_ref_to_tree(source.ref),
            "occupied": dataset_revision_ref_to_tree(occupied.ref),
            "calibration": calibration_artifact_ref_to_tree(reference),
            "selection": selection_to_tree(selection),
        }
    )
    background_input = EvaluatedInput(
        DatasetId(f"reactive-occupancy-frame-{identity}"),
        source.ref,
    )
    occupancy_input = EvaluatedInput(
        DatasetId(f"reactive-occupancy-sites-{identity}"),
        occupied.ref,
    )
    return OccupancyCellView(
        background=background,
        background_input=background_input,
        occupancy_input=occupancy_input,
        home_viewport=ImageViewportTransform((y_axis, x_axis)),
        site_axis=site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        site_radius=site_ring_radius(site_map.coordinates_xy),
        occupied=occupied_values,
        site_validity=site_validity,
        calibration_identity=reference.target_ref,
        cell_identity=identity,
        cell_selection=selection,
        run_id=run_id,
        provenance_epoch_id=epoch_id,
        summary=(
            f"Camera run={run_id} | calibration={reference.target_ref} | "
            f"revision={source.ref.revision.value} | logical point={logical_point}"
        ),
    )


def project_reactive_occupancy(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    *,
    coverage: MonitorCoverage,
    model_kind: ReadoutModelKind | None = None,
    run_id: str,
    epoch_id: str,
) -> tuple[OwnedSnapshot, OwnedSnapshot, OwnedSnapshot, "OccupancyCellView"]:
    """Classify one immutable Camera front and freeze all visible outputs together."""

    from zlc_neutral_atom.readout.occupancy import apply_occupancy_snapshot

    counts, occupied = apply_occupancy_snapshot(
        source,
        calibration,
        model_kind=model_kind,
    )
    rate = _occupancy_rate_snapshot(occupied)
    presentation = _reactive_occupancy_cell_view(
        source,
        occupied,
        calibration,
        coverage=coverage,
        run_id=run_id,
        epoch_id=epoch_id,
    )
    return counts, occupied, rate, presentation


def _calibration_site_map_projection(
    computation,
    reference: CalibrationArtifactRef,
) -> ProjectedFinalSignal:
    """Project one calibration artifact without inventing occupancy state."""

    from zlc_frontend.figure import (
        DatasetId,
        EvaluatedImage,
        EvaluatedInput,
    )
    from zlc_frontend.image_view import ImageViewportTransform
    from zlc_frontend.site_map import site_ring_radius
    from zlc_frontend.site_map_render import CalibrationSiteMapView
    from zlc_neutral_atom.readout.analysis import CalibrationComputation

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("calibration loader must return CalibrationComputation")
    artifact = computation.artifact
    report = computation.report
    frame_schema = artifact.frame_contract.frame_schema
    axes = frame_schema.data_axes
    x_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == SPATIAL_X
    )
    y_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == SPATIAL_Y
    )
    if len(x_positions) != 1 or len(y_positions) != 1:
        raise ValueError(
            "calibration reference image requires exactly one declared "
            "SPATIAL_X and SPATIAL_Y axis"
        )
    x_position, y_position = x_positions[0], y_positions[0]
    x_axis, y_axis = axes[x_position], axes[y_position]
    if report.reference_average.shape != frame_schema.data_shape:
        raise ValueError("calibration reference average differs from FrameContract")

    value_schema = ValueSchema(
        axes,
        ValidityContract.components(*(axis.axis_id for axis in axes)),
        np.dtype("<f8"),
        frame_schema.value_unit,
    )
    schema = DatasetSchema(
        AxisSpec(
            AxisId("calibration.repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        (),
        PointLayout.rect_c(()),
        value_schema,
    )
    identity = reference.manifest_digest
    block = DataBlock(
        BlockId(f"calibration-site-map-{identity[:20]}"),
        DatasetRevision(0),
        np.asarray(report.reference_average, dtype="<f8").reshape(
            (1, 1, *frame_schema.data_shape)
        ),
        ComponentValidity(
            tuple(axis.axis_id for axis in axes),
            np.asarray(
                report.reference_average_validity,
                dtype=np.bool_,
            ).reshape((1, 1, *frame_schema.data_shape)),
        ),
        schema,
    )
    generation = StreamGenerationId(f"calibration-site-map-{identity}")
    snapshot = OwnedSnapshot(block.ref(generation), block)

    background = EvaluatedImage(
        _evaluated_axis(x_axis),
        _evaluated_axis(y_axis),
        np.transpose(report.reference_average, (y_position, x_position)),
        np.transpose(
            report.reference_average_validity,
            (y_position, x_position),
        ),
        frame_schema.value_unit,
    )
    background_input = EvaluatedInput(
        DatasetId(f"calibration-reference-{identity}"),
        snapshot.ref,
    )
    calibration_input = EvaluatedInput(
        DatasetId(f"calibration-sites-{identity}"),
        snapshot.ref,
    )
    site_map = artifact.site_map
    view = CalibrationSiteMapView(
        background=background,
        background_input=background_input,
        calibration_input=calibration_input,
        home_viewport=ImageViewportTransform((y_axis, x_axis)),
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        site_radius=site_ring_radius(site_map.coordinates_xy),
        site_validity=site_map.validity.mask,
        calibration_identity=reference.target_ref,
        run_id=f"calibration-{identity}",
        provenance_epoch_id=generation.value,
        summary=(
            f"{reference.target_ref} | reference average | "
            f"valid sites={int(np.count_nonzero(site_map.validity.mask))}/"
            f"{site_map.site_axis.size}"
        ),
    )
    return ProjectedFinalSignal(snapshot, identity, view)


def _calibration_diagnostic_snapshot(
    reference: CalibrationArtifactRef,
    output_name: str,
    values,
    data_axes: tuple[AxisSpec, ...],
    *,
    value_unit: str | None,
    validity_axis_ids: tuple[AxisId, ...] = (),
    validity_mask,
) -> OwnedSnapshot:
    """Materialise one immutable diagnostic from an already-FINAL report."""

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
    else:
        axis_ids = ()
        validity_contract = ValidityContract.value()
    schema = DatasetSchema(
        repeat_axis,
        (),
        PointLayout.rect_c(()),
        ValueSchema(axes, validity_contract, np.dtype("<f8"), value_unit),
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
        validity = ComponentValidity(
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
            "owner": "zlc-workbench.task-console.calibration-diagnostic.v1",
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


def _calibration_diagnostic_projections(
    computation,
    reference: CalibrationArtifactRef,
    names: set[str],
) -> dict[str, ProjectedFinalSignal]:
    """Expose report-owned diagnostics without fitting or mutating calibration."""

    from zlc_neutral_atom.readout.analysis import CalibrationComputation

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("calibration loader must return CalibrationComputation")
    artifact = computation.artifact
    report = computation.report
    model = artifact.select_model()
    model_report = report.model(model.kind)
    site_axis = artifact.site_map.site_axis
    model_valid = np.asarray(model.usable_sites.mask, dtype=np.bool_)
    site_map_valid = np.asarray(artifact.site_map.validity.mask, dtype=np.bool_)
    site_fidelity = np.asarray(
        [item.fidelity for item in model_report.site_fidelity],
        dtype="<f8",
    )
    thresholds = np.asarray(model_report.thresholds, dtype="<f8")
    centers = np.asarray(artifact.site_map.coordinates_xy, dtype="<f8")
    if site_fidelity.shape != (site_axis.size,):
        raise ValueError("calibration fidelity does not follow its SITE axis")
    if thresholds.shape != (site_axis.size,):
        raise ValueError("calibration thresholds do not follow their SITE axis")
    if centers.shape != (site_axis.size, 2):
        raise ValueError("calibration centres must have shape (sites, 2)")

    output: dict[str, ProjectedFinalSignal] = {}
    join_digest = reference.manifest_digest
    if "fidelity_site" in names:
        output["fidelity_site"] = ProjectedFinalSignal(
            _calibration_diagnostic_snapshot(
                reference,
                "fidelity_site",
                site_fidelity,
                (site_axis,),
                value_unit="fidelity",
                validity_axis_ids=(site_axis.axis_id,),
                validity_mask=model_valid & np.isfinite(site_fidelity),
            ),
            join_digest,
        )
    if "fidelity_threshold" in names:
        output["fidelity_threshold"] = ProjectedFinalSignal(
            _calibration_diagnostic_snapshot(
                reference,
                "fidelity_threshold",
                thresholds,
                (site_axis,),
                value_unit=artifact.frame_contract.frame_schema.value_unit,
                validity_axis_ids=(site_axis.axis_id,),
                validity_mask=model_valid & np.isfinite(thresholds),
            ),
            join_digest,
        )
    if "fidelity_centers" in names:
        coordinate_axis = AxisSpec(
            AxisId(f"{site_axis.axis_id.value}.coordinate"),
            "coordinate",
            COMPONENT,
            2,
            ("x", "y"),
        )
        output["fidelity_centers"] = ProjectedFinalSignal(
            _calibration_diagnostic_snapshot(
                reference,
                "fidelity_centers",
                centers,
                (site_axis, coordinate_axis),
                value_unit="px",
                validity_axis_ids=(site_axis.axis_id,),
                validity_mask=(
                    site_map_valid & np.all(np.isfinite(centers), axis=1)
                ),
            ),
            join_digest,
        )
    for output_name, value in (
        ("aggregate_fidelity", model_report.aggregate_fidelity),
        ("global_fidelity", model_report.global_fidelity),
    ):
        if output_name not in names:
            continue
        numeric = float(value)
        output[output_name] = ProjectedFinalSignal(
            _calibration_diagnostic_snapshot(
                reference,
                output_name,
                np.asarray(numeric, dtype="<f8"),
                (),
                value_unit="fidelity",
                validity_mask=np.asarray(np.isfinite(numeric)),
            ),
            join_digest,
        )
    return output


def _occupancy_summary_site_map_view(experiment, result):
    """Join exact singleton site state to its labelled calibration background."""

    from zlc_frontend.figure import DatasetId, EvaluatedInput
    from zlc_frontend.site_map_render import (
        CalibrationSiteMapView,
        OccupancySummarySiteMapView,
    )

    if type(result) is not TriggeredOccupancyPipelineResult:
        raise TypeError("result must be TriggeredOccupancyPipelineResult")
    occupancy = result.occupancy
    snapshot = occupancy.dataset.occupied
    schema = snapshot.block.schema
    if schema.repeat_axis.size != 1 or schema.point_layout.storage_size != 1:
        return None
    data_axes = schema.cell_schema.data_axes
    if len(data_axes) != 1 or data_axes[0].role != SITE:
        raise ValueError("occupancy summary requires one declared SITE data axis")
    site_axis = data_axes[0]
    calibration_ref = occupancy.calibration_reference
    calibration_projection = _calibration_site_map_projection(
        experiment.readout.load_calibration_computation(calibration_ref),
        calibration_ref,
    )
    calibration = calibration_projection.presentation
    if not isinstance(calibration, CalibrationSiteMapView):
        raise TypeError("calibration projection omitted its typed SiteMap view")
    if calibration.site_axis != site_axis:
        raise ValueError("occupancy SITE axis differs from its calibration")

    occupied = np.asarray(snapshot.block.values[0, 0, :], dtype=np.bool_)
    validity = np.asarray(
        expand_dataset_validity(snapshot.block.validity, schema)[0, 0, :],
        dtype=np.bool_,
    )
    identity = canonical_digest(
        {
            "owner": "zlc-workbench.task-console.occupancy-summary-site-map",
            "occupancy": dataset_revision_ref_to_tree(snapshot.ref),
            "calibration": calibration_ref.target_ref,
        }
    )
    state_input = EvaluatedInput(
        DatasetId(f"occupancy-summary-sites-{identity}"),
        snapshot.ref,
    )
    return OccupancySummarySiteMapView(
        background=calibration.background,
        background_input=calibration.background_input,
        calibration_input=calibration.calibration_input,
        home_viewport=calibration.home_viewport,
        site_axis=site_axis,
        coordinate_frame=calibration.coordinate_frame,
        centers_xy=calibration.centers_xy,
        site_radius=calibration.site_radius,
        site_validity=validity,
        calibration_identity=calibration_ref.target_ref,
        run_id=occupancy.pipeline.run_id,
        provenance_epoch_id=snapshot.ref.stream_generation.value,
        summary=(
            f"occupancy run={occupancy.pipeline.run_id} | "
            f"calibration={calibration_ref.target_ref} | "
            "background=calibration reference average (not the same-shot frame)"
        ),
        occupancy_input=state_input,
        occupied=occupied,
        occupancy_identity=identity,
    )


def _mot_intensity_snapshot(result: MotFieldResult) -> OwnedSnapshot:
    """Express the typed 3-D coil grid in canonical DatasetSchema storage.

    ``MotFieldResult.intensity`` is in logical ``(x, y, z)`` order.  The loop
    below uses PointLayout's explicit mapping; it never flattens an anonymous
    ndarray and therefore cannot lose axis meaning.
    """

    axes = tuple(result.point_axes)
    layout = PointLayout.rect_c(tuple(axis.size for axis in axes))
    physical = np.empty((1, layout.storage_size), dtype="<f8")
    for storage_index in range(layout.storage_size):
        physical[0, storage_index] = result.intensity[
            layout.multi_index(storage_index)
        ]
    identity = canonical_digest(
        {
            "owner": "zlc_workbench.task-console.mot-field-view",
            "repository_id": result.scan_ref.repository_id,
            "scan_manifest": result.scan_ref.manifest_digest,
        }
    )
    schema = DatasetSchema(
        AxisSpec(
            AxisId("mot-field.repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        axes,
        layout,
        ValueSchema(
            (),
            ValidityContract.value(),
            np.dtype("<f8"),
            "counts",
        ),
    )
    block = DataBlock(
        BlockId(f"mot-field-intensity-{identity[:20]}"),
        DatasetRevision(0),
        physical,
        VALID,
        schema,
    )
    generation = StreamGenerationId(
        f"mot-field-result-{identity}"
    )
    return OwnedSnapshot(block.ref(generation), block)


def _declared(node) -> set[str]:
    return {
        str(output.name)
        for output in tuple(
            getattr(getattr(node, "spec", None), "declared_outputs", ()) or ()
        )
    }


def project_final_signals(experiment, node, result) -> dict[str, ProjectedFinalSignal]:
    """Return only data-bearing outputs truthfully present in ``result``.

    Artifact-only results are exposed only when a truthful typed projection
    exists.  Calibration remains one artifact/report authority: its reference
    image, site geometry, and read-only diagnostics are projections of that
    FINAL value, never a second analysis or mutable session calibration.
    """

    names = _declared(node)
    projected: dict[str, ProjectedFinalSignal] = {}

    if isinstance(result, CalibrationArtifactRef):
        computation = experiment.readout.load_calibration_computation(result)
        if "calibration" in names:
            projected["calibration"] = _calibration_site_map_projection(
                computation,
                result,
            )
        projected.update(
            _calibration_diagnostic_projections(computation, result, names)
        )
        return projected

    if isinstance(result, CaptureArtifactRef) and "frame" in names:
        projected["frame"] = ProjectedFinalSignal(
            experiment.readout.materialize_capture(result),
            result.manifest_digest,
        )
        return projected

    if isinstance(result, ScanArtifactRef) and "scan" in names:
        materialized = experiment.readout.materialize_scan(result)
        projected["scan"] = ProjectedFinalSignal(
            materialized.snapshot,
            result.manifest_digest,
        )
        return projected

    if type(result) is TriggeredOccupancyPipelineResult:
        dataset = result.occupancy.dataset
        identity = canonical_digest(
            {
                "owner": "zlc-workbench.task-console.finite-occupancy-view",
                "run_id": result.occupancy.pipeline.run_id,
                "counts": dataset_revision_ref_to_tree(dataset.counts.ref),
                "occupied": dataset_revision_ref_to_tree(dataset.occupied.ref),
            }
        )
        if "counts" in names:
            projected["counts"] = ProjectedFinalSignal(
                dataset.counts,
                identity,
            )
        if "occupied" in names:
            projected["occupied"] = ProjectedFinalSignal(
                dataset.occupied,
                identity,
                _occupancy_summary_site_map_view(experiment, result),
            )
        return projected

    if type(result) is TriggeredReleaseRecaptureResult:
        pipeline = result.release_recapture.pipeline
        # Temperature names this physical quantity survival; the grey-molasses
        # scan names the same exact reduction recapture rate, as Main does.
        output_name = (
            "survival"
            if "survival" in names
            else "recapture"
            if "recapture" in names
            else None
        )
        if output_name is not None:
            projected[output_name] = ProjectedFinalSignal(
                result.survival,
                canonical_digest(
                    {
                        "owner": (
                            "zlc-workbench.task-console."
                            "release-recapture-view"
                        ),
                        "run_id": pipeline.run_id,
                        "dataset": dataset_revision_ref_to_tree(
                            result.survival.ref
                        ),
                        "chain": pipeline.chain_contract_digest,
                    }
                ),
            )
        return projected

    if isinstance(result, ReadoutDurationFidelityResult):
        if "fidelity" in names:
            projected["fidelity"] = ProjectedFinalSignal(
                result.snapshot,
                result.identity,
            )
        return projected

    if isinstance(result, OccupancyArtifactRef):
        resolved = experiment.readout.load_occupancy(result)
        artifact = resolved.artifact
        occupied_schema = artifact.occupied_snapshot.block.schema
        presentation = (
            experiment.readout.occupancy_cell_view(result)
            if (
                occupied_schema.repeat_axis.size == 1
                and occupied_schema.point_layout.storage_size == 1
            )
            else None
        )
        if "counts" in names:
            projected["counts"] = ProjectedFinalSignal(
                artifact.counts_snapshot,
                result.manifest_digest,
            )
        if "occupied" in names:
            projected["occupied"] = ProjectedFinalSignal(
                artifact.occupied_snapshot,
                result.manifest_digest,
                presentation,
            )
        return projected

    if isinstance(result, MotFieldResult):
        if "mot_field" in names:
            projected["mot_field"] = ProjectedFinalSignal(
                _mot_intensity_snapshot(result),
                result.scan_ref.manifest_digest,
            )
        if "scan" in names:
            materialized = experiment.readout.materialize_scan(result.scan_ref)
            projected["scan"] = ProjectedFinalSignal(
                materialized.snapshot,
                result.scan_ref.manifest_digest,
            )
        return projected

    return projected
