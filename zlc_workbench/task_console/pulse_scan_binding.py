"""TaskConsole binding intent for a source-neutral Pulse scan.

The catalog can freeze the pulse program and the selected console signal, but
it cannot resolve another row's runtime/domain request.  That resolution
belongs to the TaskConsole composition root.  Keeping this value deliberately
small prevents the form layer from smuggling a camera into Pulse scan again.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import DataTransformSpec
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.camera_measurement import camera_frame_output_index
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.readout.occupancy import OCCUPANCY_STREAM_PROCESSOR_KEY
from zlc_neutral_atom.scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_storage import canonical_text

from .occupancy_binding import ConsoleProducerBinding


PULSE_SCAN_CAMERA_FRAME_SOURCE = "camera-frame"
PULSE_SCAN_OCCUPANCY_SOURCE = "occupancy"


def classify_pulse_scan_producer(
    definition_key: DefinitionKey,
    output_name: str,
) -> str | None:
    """Return the exact Pulse Scan source family, or ``None`` if rejected."""

    if definition_key == CAMERA_MEASUREMENT_KEY:
        try:
            camera_frame_output_index(output_name)
        except (TypeError, ValueError):
            pass
        else:
            return PULSE_SCAN_CAMERA_FRAME_SOURCE
    if (
        definition_key == OCCUPANCY_STREAM_PROCESSOR_KEY
        and output_name in ("counts", "occupied")
    ):
        return PULSE_SCAN_OCCUPANCY_SOURCE
    return None


@dataclass(frozen=True, slots=True)
class PulseScanBindingIntent:
    """One frozen pulse program and one explicitly selected y signal."""

    program: AutonomousScanSlotProgram | ApiSlotSegmentedProgram
    y_signal: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a current PulseScanProgram")
        canonical_text(self.y_signal, "Pulse scan y_signal")


@dataclass(frozen=True, slots=True)
class PulseScanSourceBinding:
    """The physical producer plus any Figure-owned authoritative selection."""

    source_kind: str
    producer: ConsoleProducerBinding
    transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.producer, ConsoleProducerBinding):
            raise TypeError("producer must be a ConsoleProducerBinding")
        expected = classify_pulse_scan_producer(
            self.producer.definition_key,
            self.producer.output_name,
        )
        if expected is None or self.source_kind != expected:
            raise ValueError(
                "source_kind must match an exact Camera frame or Occupancy "
                "counts/occupied producer"
            )
        if self.transform_spec is not None:
            if not isinstance(self.transform_spec, DataTransformSpec):
                raise TypeError("transform_spec must be DataTransformSpec or None")
            if not self.transform_spec.operations:
                raise ValueError("an empty transform_spec must be None")


__all__ = [
    "PULSE_SCAN_CAMERA_FRAME_SOURCE",
    "PULSE_SCAN_OCCUPANCY_SOURCE",
    "PulseScanBindingIntent",
    "PulseScanSourceBinding",
    "classify_pulse_scan_producer",
]
