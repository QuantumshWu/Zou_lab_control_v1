"""Authoritative FINAL Dataset outputs of one loaded calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zlc_data import (
    COMPONENT,
    REPEAT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
)
from zlc_storage import canonical_text

from .reference import (
    CALIBRATION_ARTIFACT_REF_FORMAT,
    CalibrationArtifactRef,
)

if TYPE_CHECKING:
    from .analysis import CalibrationComputation, CalibrationReport


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
        "readout_samples", "zlc_neutral_atom.calibration.readout-samples"
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


def _require_inputs(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> None:
    from .analysis import CalibrationComputation

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("computation must be CalibrationComputation")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")


def _calibration_reference_snapshot(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> OwnedSnapshot:
    artifact = computation.artifact
    report = computation.report
    frame_schema = artifact.frame_contract.frame_schema
    if len(frame_schema.data_axes) != 2:
        raise ValueError("calibration reference requires two declared frame axes")
    if report.reference_average.shape != frame_schema.data_shape:
        raise ValueError("calibration reference average differs from FrameContract")
    if report.reference_average_validity.shape != frame_schema.data_shape:
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
    identity = reference.record_path
    block = DataBlock(
        BlockId(f"calibration-reference:{identity}"),
        DatasetRevision(0),
        np.asarray(report.reference_average, dtype="<f8").reshape(
            schema.physical_shape
        ),
        DatasetComponentValidity(
            tuple(axis.axis_id for axis in axes),
            np.asarray(
                report.reference_average_validity,
                dtype=np.bool_,
            ).reshape(schema.physical_shape),
        ),
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"calibration-reference:{identity}")),
        block,
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
        value_schema = ValueSchema(
            axes,
            ValidityContract.components(*axis_ids),
            np.dtype("<f8"),
            value_unit,
        )
    else:
        axis_ids = ()
        value_schema = ValueSchema.scalar(np.dtype("<f8"), value_unit)
    schema = DatasetSchema(repeat_axis, PointTable(1), None, value_schema)
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

    identity = canonical_text(
        f"{reference.target_ref}:{output_name}",
        "calibration diagnostic Dataset identity",
    )
    block = DataBlock(
        BlockId(f"calibration-diagnostic:{identity}"),
        DatasetRevision(0),
        array.reshape(schema.physical_shape),
        validity,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"calibration-diagnostic:{identity}")),
        block,
    )


def _context_shape(
    report: CalibrationReport,
) -> tuple[AxisId, int, int]:
    """Return the source repeat identity and stable non-event point count."""

    contexts = tuple(report.group_contexts)
    if not contexts or any(not context for context in contexts):
        raise ValueError("calibration report has no complete group contexts")
    repeat_axis_id = contexts[0][0][0]
    by_repeat: dict[int, list[tuple[tuple[AxisId, int], ...]]] = {}
    for context in contexts:
        axis_id, repeat_index = context[0]
        if axis_id != repeat_axis_id:
            raise ValueError("calibration group repeat identity changed")
        by_repeat.setdefault(int(repeat_index), []).append(tuple(context[1:]))
    repeat_count = len(by_repeat)
    if tuple(by_repeat) != tuple(range(repeat_count)):
        raise ValueError("calibration group repeats are not canonical ordinals")
    point_contexts = tuple(by_repeat[0])
    if not point_contexts:
        raise ValueError("calibration report has no non-event point contexts")
    if any(tuple(rows) != point_contexts for rows in by_repeat.values()):
        raise ValueError("calibration point contexts differ between repeats")
    return repeat_axis_id, repeat_count, len(point_contexts)


def _readout_samples_snapshot(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> OwnedSnapshot:
    artifact = computation.artifact
    report = computation.report
    model = artifact.select_model()
    model_report = report.model(model.kind)
    repeat_axis_id, repeat_count, point_count = _context_shape(report)
    site_axis = artifact.site_map.site_axis
    expected = (repeat_count * point_count, site_axis.size)
    if model_report.short_signals.shape != expected:
        raise ValueError(
            "calibration readout samples do not match repeat/point/site contexts"
        )
    # ``group_contexts`` retains stable axis identities and enumerated context
    # order, not the source PointColumn coordinate scalars.  Bare authored rows
    # preserve P without inventing physical coordinates from those indices.
    schema = DatasetSchema(
        AxisSpec(
            repeat_axis_id,
            "repeat",
            REPEAT,
            repeat_count,
            tuple(range(repeat_count)),
        ),
        PointTable(point_count),
        None,
        ValueSchema(
            (site_axis,),
            ValidityContract.components(site_axis.axis_id),
            np.dtype("<f8"),
            artifact.frame_contract.frame_schema.value_unit,
        ),
    )
    values = np.asarray(model_report.short_signals, dtype="<f8").reshape(
        schema.physical_shape
    )
    validity_mask = (
        np.asarray(model_report.short_validity, dtype=np.bool_)
        & np.isfinite(values.reshape(expected))
    ).reshape(schema.physical_shape)
    identity = canonical_text(
        f"{reference.target_ref}:readout_samples",
        "calibration readout samples Dataset identity",
    )
    block = DataBlock(
        BlockId(f"calibration-readout-samples:{identity}"),
        DatasetRevision(0),
        values,
        DatasetComponentValidity((site_axis.axis_id,), validity_mask),
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"calibration-readout-samples:{identity}")),
        block,
    )


def _calibration_diagnostic_snapshots(
    computation: CalibrationComputation,
    reference: CalibrationArtifactRef,
) -> dict[str, OwnedSnapshot]:
    artifact = computation.artifact
    report = computation.report
    model = artifact.select_model()
    model_report = report.model(model.kind)
    site_axis = artifact.site_map.site_axis
    site_map_valid = np.asarray(artifact.site_map.validity.mask, dtype=np.bool_)
    runtime_usable = (
        site_map_valid
        & np.asarray(model.usable_sites.mask, dtype=np.bool_)
    )
    fidelity_evidence = np.asarray(
        [item.n_test > 0 for item in model_report.site_fidelity],
        dtype=np.bool_,
    )
    site_fidelity = np.asarray(
        [item.fidelity for item in model_report.site_fidelity],
        dtype="<f8",
    )
    thresholds = np.asarray(model.thresholds, dtype="<f8")
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
            runtime_usable & fidelity_evidence & np.isfinite(site_fidelity)
        ),
    )
    output["fidelity_threshold"] = _diagnostic_snapshot(
        reference,
        "fidelity_threshold",
        thresholds,
        (site_axis,),
        value_unit=artifact.frame_contract.frame_schema.value_unit,
        validity_axis_ids=(site_axis.axis_id,),
        validity_mask=runtime_usable & np.isfinite(thresholds),
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
    output["readout_samples"] = _readout_samples_snapshot(computation, reference)
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
    """Publish the exact declared data outputs of one loaded calibration."""

    _require_inputs(computation, reference)
    snapshots = {
        CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0].name: (
            _calibration_reference_snapshot(computation, reference)
        ),
        **_calibration_diagnostic_snapshots(computation, reference),
    }
    expected = tuple(
        declaration.name for declaration in CALIBRATION_FINAL_OUTPUT_DECLARATIONS
    )
    if tuple(snapshots) != expected:
        raise RuntimeError("calibration output materializer changed its public order")
    return {
        declaration.name: FinalDatasetOutput(declaration, snapshot)
        for declaration, snapshot in zip(
            CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
            snapshots.values(),
            strict=True,
        )
    }


__all__ = [
    "CALIBRATION_ARTIFACT_OUTPUT_DECLARATION",
    "CALIBRATION_DIAGNOSTIC_OUTPUT_DECLARATIONS",
    "CALIBRATION_FINAL_OUTPUT_DECLARATIONS",
    "calibration_final_outputs",
]
