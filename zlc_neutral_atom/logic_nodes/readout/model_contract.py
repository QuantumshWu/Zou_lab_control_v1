"""Shared readout-model identity and authoring contract.

Calibration constructs and trains concrete models.  Other readout-family
capabilities only need a stable model identity and the one declared authoring
choice that resolves either an explicit identity or the calibration default.
Keeping that vocabulary here prevents those capabilities from depending on
Calibration's model implementation.
"""

from __future__ import annotations

from enum import Enum

from zlc_neutral_atom.authoring import (
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
)


class ReadoutModelKind(str, Enum):
    BOX = "box"
    PER_SITE_PSF = "psf"
    UNIFORM_PSF = "uniform_psf"


_CALIBRATION_DEFAULT_MODEL_CHOICE = "calibration_default"
_READOUT_MODEL_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "model_kind",
            "choice",
            "Readout method",
            default=_CALIBRATION_DEFAULT_MODEL_CHOICE,
            required=True,
            choices=(
                AuthoringChoice(
                    _CALIBRATION_DEFAULT_MODEL_CHOICE,
                    "Calibration default",
                ),
                AuthoringChoice(ReadoutModelKind.BOX.value, "box"),
                AuthoringChoice(ReadoutModelKind.PER_SITE_PSF.value, "per-site PSF"),
                AuthoringChoice(ReadoutModelKind.UNIFORM_PSF.value, "uniform PSF"),
            ),
            description=(
                "Use the calibration's default model or explicitly select one "
                "model already stored in that calibration"
            ),
        ),
    )
)


def readout_model_authoring_schema() -> AuthoringSchema:
    """Return the readout family's one visible model-selection declaration."""

    return _READOUT_MODEL_AUTHORING_SCHEMA


def readout_model_kind_from_authoring(value: object) -> ReadoutModelKind | None:
    """Resolve the visible default-reference token without choosing a model."""

    if value == _CALIBRATION_DEFAULT_MODEL_CHOICE:
        return None
    try:
        return ReadoutModelKind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown readout model choice {value!r}") from error


__all__ = [
    "ReadoutModelKind",
    "readout_model_authoring_schema",
    "readout_model_kind_from_authoring",
]
