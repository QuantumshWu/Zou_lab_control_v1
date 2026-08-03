"""The one saved-or-Task calibration input shared by readout Measurements."""

from __future__ import annotations

from zlc_neutral_atom.input_spec import ArtifactInputSpec
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CALIBRATION_ARTIFACT_REF_FORMAT,
)


CALIBRATION_INPUT_SPEC = ArtifactInputSpec(
    "calibration",
    "Calibration",
    CALIBRATION_ARTIFACT_REF_FORMAT,
    description="FINAL calibration output of a successful Calibrate readout Task",
    allow_saved_reference=True,
)

__all__ = ["CALIBRATION_INPUT_SPEC"]
