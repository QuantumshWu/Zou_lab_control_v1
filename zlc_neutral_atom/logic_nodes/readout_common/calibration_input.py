"""The one saved-or-Task calibration input shared by readout Measurements."""

from __future__ import annotations

from zlc_neutral_atom.input_spec import ArtifactInputSpec
from zlc_neutral_atom.logic_nodes.calibration.reference import (
    CALIBRATION_ARTIFACT_REF_FORMAT,
    CalibrationArtifactRef,
)
from zlc_neutral_atom.node_input import BoundNodeInputs


CALIBRATION_INPUT_SPEC = ArtifactInputSpec(
    "calibration",
    "Calibration",
    CALIBRATION_ARTIFACT_REF_FORMAT,
    description="FINAL calibration output of a successful Calibrate readout Task",
)


def calibration_input_specs() -> tuple[ArtifactInputSpec, ...]:
    return (CALIBRATION_INPUT_SPEC,)


def calibration_reference(inputs: BoundNodeInputs) -> CalibrationArtifactRef:
    """Resolve the declared calibration input without any Workbench semantics."""

    if not isinstance(inputs, BoundNodeInputs):
        raise TypeError("inputs must be BoundNodeInputs")
    reference = inputs.artifact(CALIBRATION_INPUT_SPEC).reference
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("Calibration input resolved another artifact reference")
    if len(inputs.values) != 1:
        raise ValueError("readout Measurement received undeclared inputs")
    return reference


__all__ = [
    "CALIBRATION_INPUT_SPEC",
    "calibration_input_specs",
    "calibration_reference",
]
