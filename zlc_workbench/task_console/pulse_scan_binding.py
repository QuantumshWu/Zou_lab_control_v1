"""TaskConsole binding intent for a source-neutral Pulse scan.

The catalog can freeze the pulse program and the selected console signal, but
it cannot resolve another row's runtime/domain request.  That resolution
belongs to the TaskConsole composition root.  Keeping this value deliberately
small prevents the form layer from smuggling a camera into Pulse scan again.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import DataTransformSpec
from zlc_neutral_atom.scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_storage import canonical_text

from .occupancy_binding import ConsoleProducerBinding


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

    producer: ConsoleProducerBinding
    transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.producer, ConsoleProducerBinding):
            raise TypeError("producer must be a ConsoleProducerBinding")
        if self.transform_spec is not None:
            if not isinstance(self.transform_spec, DataTransformSpec):
                raise TypeError("transform_spec must be DataTransformSpec or None")
            if not self.transform_spec.operations:
                raise ValueError("an empty transform_spec must be None")


__all__ = ["PulseScanBindingIntent", "PulseScanSourceBinding"]
