"""Source-neutral signal selection owned by PulseScan."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseParameterScanProgram,
)
from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class ScanSignalBinding:
    """The active signal route and producer-owned event output sampled as y."""

    signal_name: str
    output_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_name",
            canonical_text(self.signal_name, "PulseScan signal name"),
        )
        object.__setattr__(
            self,
            "output_name",
            canonical_text(self.output_name, "PulseScan event output name"),
        )


@dataclass(frozen=True, slots=True)
class PulseScanRequest:
    """One pulse program and one already-running arbitrary y signal."""

    program: PulseParameterScanProgram
    signal: ScanSignalBinding
    trigger_port: str = "emCCD"

    def __post_init__(self) -> None:
        if not isinstance(
            self.program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a current PulseScan program")
        if not isinstance(self.signal, ScanSignalBinding):
            raise TypeError("signal must be ScanSignalBinding")
        object.__setattr__(
            self,
            "trigger_port",
            canonical_text(self.trigger_port, "PulseScan trigger_port"),
        )


__all__ = ["PulseScanRequest", "ScanSignalBinding"]
