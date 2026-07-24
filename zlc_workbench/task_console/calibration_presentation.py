"""The sole calibration-domain to frontend SiteMap composition seam."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_frontend.site_map_render import (
    SiteMapView,
    build_calibration_site_map_view,
)
from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.readout.calibration_projection import (
    CALIBRATION_FINAL_OUTPUT_NAMES,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.calibration_task import PreparedCalibrationTask


class CalibrationFinalPresentationAdapter:
    """Render a prepared task's typed FINAL context without redoing physics."""

    __slots__ = ()

    def materialize_final_presentations(
        self,
        command: PreparedCalibrationTask,
        result: CalibrationArtifactRef,
        outputs: Mapping[str, FinalDatasetOutput],
    ) -> dict[str, SiteMapView]:
        if not isinstance(command, PreparedCalibrationTask):
            raise TypeError("command must be PreparedCalibrationTask")
        if not isinstance(result, CalibrationArtifactRef):
            raise TypeError("result must be CalibrationArtifactRef")
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping")

        context = command.site_map_context(result)
        calibration_name = CALIBRATION_FINAL_OUTPUT_NAMES[0]
        output = outputs.get(calibration_name)
        if (
            not isinstance(output, FinalDatasetOutput)
            or output.name != calibration_name
        ):
            raise ValueError("calibration SiteMap requires the calibration output")
        snapshot = output.snapshot
        valid_count = int(context.site_validity.sum())
        view = build_calibration_site_map_view(
            snapshot,
            site_axis=context.site_axis,
            coordinate_frame=context.coordinate_frame,
            centers_xy=context.centers_xy,
            site_validity=context.site_validity,
            calibration_identity=context.calibration_identity,
            run_id=f"calibration-{output.join_digest}",
            provenance_epoch_id=snapshot.ref.stream_generation.value,
            summary=(
                f"{context.calibration_identity} | reference average | "
                f"valid sites={valid_count}/{context.site_axis.size}"
            ),
        )
        return {calibration_name: view}


__all__ = ["CalibrationFinalPresentationAdapter"]
