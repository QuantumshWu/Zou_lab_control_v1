"""Timing/preflight/verilog subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.results import PreflightReport
from ..timing import PulseSequence, PulseTableState
from .base import ExperimentSubsystem

if TYPE_CHECKING:  # pragma: no cover
    from ..devices import PulseController
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

    def bind_pulse(self, pulse: PulseSequence | PulseTableState | str | Path | None = None) -> "PulseController":
        """Bind a pulse payload or GUI pulse JSON to the session sequencer."""

        from ..devices import bind_pulse

        sequencer = getattr(self._session.devices, "sequencer", None)
        if sequencer is None:
            raise RuntimeError("This session has no sequencer device to bind a pulse to.")
        return bind_pulse(sequencer, self._load_pulse_payload(pulse))

    def write_verilog(self, output_dir: str | Path = "generated_sequences", **kwargs) -> Path:
        return self._session._write_verilog(output_dir, **kwargs)

    def _load_pulse_payload(self, pulse: PulseSequence | PulseTableState | str | Path | None) -> PulseSequence | PulseTableState:
        if pulse is None:
            return self._session.sequence
        if isinstance(pulse, (PulseSequence, PulseTableState)):
            return pulse
        path = Path(pulse)
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema", "")
        if schema == PulseTableState.schema:
            return PulseTableState.from_dict(payload)
        if schema == "Zou_lab_control.neutral_atom.PulseSequence":
            return PulseSequence.from_dict(payload)
        raise ValueError(f"unsupported pulse JSON schema in {path}: {schema!r}")


__all__ = ["TimingSubsystem"]
