"""Timing/preflight/verilog subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.results import PreflightReport
from ..timing import PulseSequence
from .base import ExperimentSubsystem

if TYPE_CHECKING:  # pragma: no cover
    from ..session import NeutralAtomSession


class TimingSubsystem(ExperimentSubsystem):
    """Timing and FPGA-facing actions for the current experiment sequence."""

    _session: "NeutralAtomSession"

    def configure_imaging(self, **kwargs) -> PulseSequence:
        return self._session._configure_imaging(**kwargs)

    def plot_sequence(self, *, sequence: PulseSequence | None = None, display: bool = True):
        from ..timing import plot_sequence

        return plot_sequence(sequence or self._session.sequence, display=display)

    def preflight(self, **kwargs) -> PreflightReport:
        return self._session._preflight(**kwargs)

    def write_verilog(self, output_dir: str | Path = "generated_sequences", **kwargs) -> Path:
        return self._session._write_verilog(output_dir, **kwargs)


__all__ = ["TimingSubsystem"]
