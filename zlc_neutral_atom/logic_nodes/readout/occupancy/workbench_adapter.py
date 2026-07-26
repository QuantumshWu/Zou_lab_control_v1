"""Narrow Workbench admission seam for Occupancy's Calibration input."""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)


def resolve_occupancy_calibration_input(
    binding,
    *,
    resolve_final_or_saved: Callable[..., object],
    load_saved_calibration: Callable[[object], object],
) -> CalibrationArtifactRef:
    """Resolve Occupancy's saved-or-FINAL Calibration reference exactly once."""

    if not callable(resolve_final_or_saved) or not callable(load_saved_calibration):
        raise TypeError("Occupancy artifact resolvers must be callable")

    def extract_reference(loaded: object) -> CalibrationArtifactRef:
        if type(loaded) is not ResolvedCalibration:
            raise TypeError("saved Calibration loader returned another value type")
        return loaded.reference

    reference = resolve_final_or_saved(
        binding,
        load_saved=load_saved_calibration,
        extract_reference=extract_reference,
    )
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("Occupancy Calibration input resolved another artifact type")
    return reference


__all__ = ["resolve_occupancy_calibration_input"]
