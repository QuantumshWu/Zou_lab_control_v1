"""Timing/preflight/verilog subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from Zou_lab_control._clock import DEFAULT_CLOCK_HZ
from Zou_lab_control._paths import GENERATED_SEQUENCES_DIR
from ..core.results import PreflightReport
from ..timing import PulseSequence, PulseTableState, load_pulse_payload
from ..timing.verilog import generate_verilog, write_verilog_bundle
from .base import ExperimentSubsystem

if TYPE_CHECKING:  # pragma: no cover
    from ..devices import CameraDevice, PulseController
    from ..session import NeutralAtomSession


class TimingSubsystem(ExperimentSubsystem):
    """Timing and FPGA-facing actions for the current experiment sequence.

    This subsystem OWNS its orchestration (the ``subsystems/base`` contract: a
    larger organic capability, never a forwarding shell over the session):
    imaging-sequence configuration, sequence preflight, pulse binding, and
    Verilog export all live here, sharing the session's current ``sequence``
    plus its device set.  None of it can live on a single device, because every
    action orchestrates ACROSS devices while the devices themselves stay
    session-blind (a camera is a pure grabber, a sequencer a pure streamer):
    ``configure_imaging`` writes a camera's exposure AND rebuilds the imaging
    sequence around the readout camera's trigger line; ``preflight`` validates
    the sequence against the sequencer's clock/channels and snapshots EVERY
    device into the report; ``write_verilog`` compiles the sequence with the
    sequencer's clock and pin map.
    """

    _session: "NeutralAtomSession"

    def configure_imaging(self, *, exposure: float | None = None, camera: "CameraDevice | None" = None,
                          load: bool = True, trigger_width: float = 20e-6, pre_trigger: float = 100e-6) -> PulseSequence:
        """Set the imaging exposure and rebuild the session's imaging sequence.

        The ONE configure-imaging path (``capture`` routes its ``exposure=`` through
        here too): ``camera`` picks which sensor the exposure is written to (None =
        the conventional readout camera); the rebuilt sequence always gates the
        READOUT camera's trigger line and becomes ``session.sequence``.
        """

        s = self._session
        cam = getattr(s.devices, "camera", None) if camera is None else camera
        if exposure is not None and hasattr(cam, "configure"):
            cam.configure(exposure=exposure)
        s.sequence = s.build_imaging_sequence(
            exposure=s.camera_exposure() if exposure is None else exposure,
            trigger_width=trigger_width,
            pre_trigger=pre_trigger,
            load=load,
        )
        return s.sequence

    def plot_sequence(self, *, sequence: PulseSequence | None = None, display: bool = True):
        from ..timing import plot_sequence

        return plot_sequence(sequence or self._session.sequence, display=display)

    def preflight(self, *, sequence: PulseSequence | None = None, verilog: bool = True) -> PreflightReport:
        """Validate a sequence against the connected sequencer and snapshot all devices."""

        s = self._session
        sequence = sequence or s.sequence
        errors: list[str] = []
        warnings: list[str] = []
        sequencer = getattr(s.devices, "sequencer", None)
        clock = getattr(sequencer, "clock_hz", DEFAULT_CLOCK_HZ)
        channels = getattr(sequencer, "channels", None)
        pulse_report = sequence.validate(clock_hz=clock, channels=channels)
        errors.extend(pulse_report.errors)
        warnings.extend(pulse_report.warnings)
        build = None
        if verilog and channels is not None:
            try:
                build = generate_verilog(sequence, channels=channels, clock_hz=clock, module_name="preflight_sequence")
            except Exception as exc:
                errors.append(str(exc))
        return PreflightReport(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            sequence_table=sequence.table(),
            device_snapshot=s.devices.snapshot(),
            verilog=build,
        )

    def bind_pulse(self, pulse: PulseSequence | PulseTableState | str | Path | None = None) -> "PulseController":
        """Bind a pulse payload or GUI pulse JSON to the session sequencer."""

        from ..devices import bind_pulse

        s = self._session
        sequencer = getattr(s.devices, "sequencer", None)
        if sequencer is None:
            raise RuntimeError("This session has no sequencer device to bind a pulse to.")
        payload = s.sequence if pulse is None else load_pulse_payload(pulse)
        return bind_pulse(sequencer, payload)

    def write_verilog(self, output_dir: str | Path = GENERATED_SEQUENCES_DIR, *,
                      sequence: PulseSequence | None = None) -> Path:
        """Compile a sequence with the sequencer's clock/pin map and write the HDL bundle."""

        s = self._session
        sequence = sequence or s.sequence
        sequencer = getattr(s.devices, "sequencer", None)
        channels = getattr(sequencer, "channels", sequence.channels)
        clock = getattr(sequencer, "clock_hz", DEFAULT_CLOCK_HZ)
        pin_map = getattr(sequencer, "pin_map", None)
        build = generate_verilog(sequence, channels=channels, clock_hz=clock, module_name="neutral_atom_sequence")
        files = write_verilog_bundle(build, output_dir, pin_map=pin_map)
        return files.verilog_path


__all__ = ["TimingSubsystem"]
