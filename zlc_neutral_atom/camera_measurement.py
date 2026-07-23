"""Public intent and descriptor for the one Camera Measurement."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import DatasetSchema
from zlc_neutral_atom.installation import DeviceRef
from zlc_storage import canonical_text, positive_integer


@dataclass(frozen=True)
class CameraMeasurementRequest:
    """Read raw camera cycles: ``repeat=0`` live, ``repeat=K`` finite.

    The request owns only the selected camera.  Trigger timing belongs to
    independently running hardware; Camera Measurement never acquires pulse or
    sequencer authority.
    """

    camera_ref: DeviceRef
    repeat: int = 0
    history_capacity: int = 8
    frames_per_cycle: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < 0:
            raise ValueError("repeat must be non-negative")
        object.__setattr__(
            self,
            "history_capacity",
            positive_integer(self.history_capacity, "history_capacity"),
        )
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )


@dataclass(frozen=True)
class CameraMeasurementDescriptor:
    name: str
    camera_role: str
    output_schema: DatasetSchema
    resource_claim: str

    def __post_init__(self) -> None:
        canonical_text(self.name, "camera measurement name")
        canonical_text(self.camera_role, "camera role")
        if not isinstance(self.output_schema, DatasetSchema):
            raise TypeError("output_schema must be DatasetSchema")
        canonical_text(self.resource_claim, "resource_claim")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.output_schema.physical_shape

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_schema.fingerprint


__all__ = ["CameraMeasurementDescriptor", "CameraMeasurementRequest"]
