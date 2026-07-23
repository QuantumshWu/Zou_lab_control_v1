"""Project successful domain results onto TaskConsole's immutable data plane.

This is a presentation adapter, not another result or artifact authority.  It
only exposes an already-FINAL dataset snapshot under outputs the catalog
actually declared.  Artifact admission/materialisation remains delegated to
the owning notebook/domain facade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zlc_data import (
    REPEAT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    dataset_revision_ref_to_tree,
    expand_dataset_validity,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.mot_field import MotFieldResult
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.scan import ScanArtifactRef
from zlc_neutral_atom.timing.occupancy import (
    TriggeredOccupancyPipelineResult,
)
from zlc_neutral_atom.timing.release_recapture import (
    TriggeredReleaseRecaptureResult,
)
from zlc_storage import canonical_digest

__all__ = ["ProjectedFinalSignal", "project_final_signals"]


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


def _calibration_site_map_projection(
    computation,
    reference: CalibrationArtifactRef,
) -> ProjectedFinalSignal:
    """Project one calibration artifact without inventing occupancy state."""

    from zlc_frontend.figure import (
        DatasetId,
        EvaluatedAxis,
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

    def evaluated_axis(axis: AxisSpec) -> EvaluatedAxis:
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

    background = EvaluatedImage(
        evaluated_axis(x_axis),
        evaluated_axis(y_axis),
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

    Artifact-only results are exposed only when a truthful typed presentation
    exists.  In particular calibration owns a reference image plus site
    geometry; it is never retyped as an occupancy array.
    """

    names = _declared(node)
    projected: dict[str, ProjectedFinalSignal] = {}

    if isinstance(result, CalibrationArtifactRef) and "calibration" in names:
        projected["calibration"] = _calibration_site_map_projection(
            experiment.readout.load_calibration_computation(result),
            result,
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
        if "survival" in names:
            projected["survival"] = ProjectedFinalSignal(
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
