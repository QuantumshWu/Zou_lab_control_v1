"""Vivado/VIO adapter for the existing address-switch imaging bitstream.

This module is intentionally a device-layer bridge, not an experiment
workflow.  It translates a prepared ``RuntimeSequenceProgram`` into the small
set of VIO parameters that the current fixed address-switch state machine
understands: the probe pulse width and the number of cycles/shots.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import subprocess
import time
from typing import Mapping

from .sequencer import RuntimeSequenceProgram
from ..timing import PulseSequence, positive_float


MAX_29BIT = (1 << 29) - 1


def legacy_imaging_parameters(
    program: RuntimeSequenceProgram,
    *,
    clock_hz: float | None = None,
    probe_channel: str = "probe",
    pulse_param: str = "pulse_lasting",
    cycle_param: str = "cycle_counts",
) -> dict[str, int]:
    """Return legacy VIO parameters for an imaging/readout program."""

    clock = positive_float(program.clock_hz if clock_hz is None else clock_hz, "clock_hz")
    pulse_ticks = _uniform_probe_width_ticks(program, clock_hz=clock, probe_channel=probe_channel)
    cycle_count = int(program.trigger_count)
    _validate_29bit(pulse_param, pulse_ticks)
    _validate_29bit(cycle_param, cycle_count)
    if cycle_count <= 0:
        raise ValueError("program.trigger_count must be positive for the legacy address-switch backend.")
    return {pulse_param: pulse_ticks, cycle_param: cycle_count}


def write_vivado_vio_tcl(
    path: str | Path,
    assignments: Mapping[str, int],
    *,
    project: str | None = None,
    bitstream: str | None = None,
    probes: str | None = None,
    vio_filter: str = 'CELL_NAME=~"*vio*"',
) -> Path:
    """Write a Vivado batch Tcl file that assigns named VIO probes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    project_line = "set project [env_or ZLC_VIVADO_PROJECT \"\"]" if project is None else f"set project {{{project}}}"
    bitstream_line = "set bitstream [env_or ZLC_VIVADO_BIT \"\"]" if bitstream is None else f"set bitstream {{{bitstream}}}"
    probes_line = "set probes [env_or ZLC_VIVADO_LTX \"\"]" if probes is None else f"set probes {{{probes}}}"
    lines = [
        "proc env_or {name default} {",
        "    if {[info exists ::env($name)]} { return $::env($name) }",
        "    return $default",
        "}",
        project_line,
        bitstream_line,
        probes_line,
        f"set vio_filter {{{vio_filter}}}",
        "set program_on_run [env_or ZLC_VIVADO_PROGRAM_ON_RUN \"0\"]",
        "if {$project ne \"\" && [file exists $project]} { open_project $project }",
        "open_hw_manager",
        "connect_hw_server -allow_non_jtag",
        "open_hw_target",
        "set device [lindex [get_hw_devices] 0]",
        "if {$device eq \"\"} { error \"No Vivado hardware device found.\" }",
        "if {$program_on_run ne \"0\" && $bitstream ne \"\" && [file exists $bitstream]} {",
        "    set_property PROGRAM.FILE $bitstream $device",
        "    if {$probes ne \"\" && [file exists $probes]} {",
        "        set_property PROBES.FILE $probes $device",
        "        set_property FULL_PROBES.FILE $probes $device",
        "    }",
        "    program_hw_devices $device",
        "    refresh_hw_device $device",
        "} elseif {$probes ne \"\" && [file exists $probes]} {",
        "    set_property PROBES.FILE $probes $device",
        "    set_property FULL_PROBES.FILE $probes $device",
        "    refresh_hw_device $device",
        "}",
        "set available_vios [get_hw_vios -of_objects $device]",
        "set vio [lindex [get_hw_vios -of_objects $device -filter $vio_filter] 0]",
        "if {$vio eq \"\"} {",
        "    puts \"Available VIO cores:\"",
        "    foreach candidate $available_vios {",
        "        puts \"  NAME=[get_property NAME $candidate] CELL_NAME=[get_property CELL_NAME $candidate]\"",
        "    }",
        "    error \"No VIO core matched filter '$vio_filter'.\"",
        "}",
        "proc set_vio_probe {vio name value} {",
        "    set probe [lindex [get_hw_probes $name -of_objects $vio] 0]",
        "    if {$probe eq \"\"} { error \"VIO probe '$name' was not found.\" }",
        "    set_property OUTPUT_VALUE_RADIX UNSIGNED $probe",
        "    set_property OUTPUT_VALUE $value $probe",
        "    commit_hw_vio $probe",
        "    puts \"ZLC legacy address-switch VIO: $name=$value\"",
        "}",
    ]
    for name, value in assignments.items():
        lines.append(f"set_vio_probe $vio {{{name}}} {{{int(value)}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_assignments(action: str, program: RuntimeSequenceProgram | None, *, clock_hz: float | None = None) -> dict[str, int]:
    """Build VIO assignments for prepare/fire/safe_state actions."""

    start_param = os.environ.get("ZLC_LEGACY_START_PARAM", "config_ready")
    debug_param = os.environ.get("ZLC_LEGACY_DEBUG_PARAM", "debug")
    pulse_param = os.environ.get("ZLC_LEGACY_PULSE_PARAM", "pulse_lasting")
    cycle_param = os.environ.get("ZLC_LEGACY_CYCLE_PARAM", "cycle_counts")
    probe_channel = os.environ.get("ZLC_LEGACY_PROBE_CHANNEL", "probe")
    if action == "prepare":
        if program is None:
            raise ValueError("prepare requires a RuntimeSequenceProgram.")
        _require_single_trigger_confirmation()
        assignments = {start_param: 0, debug_param: 0}
        assignments.update(_env_default_assignments())
        assignments.update(
            legacy_imaging_parameters(
                program,
                clock_hz=clock_hz,
                probe_channel=probe_channel,
                pulse_param=pulse_param,
                cycle_param=cycle_param,
            )
        )
        return assignments
    if action == "fire":
        return {start_param: 1}
    if action == "wait_done":
        return {start_param: 0}
    if action in {"safe_state", "abort"}:
        return {start_param: 0, debug_param: 0}
    raise ValueError(f"unknown legacy address-switch action {action!r}.")


def run_action(
    action: str,
    *,
    program_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    vivado: str = "vivado",
    dry_run: bool = False,
    timeout: float | None = None,
    clock_hz: float | None = None,
) -> Path | None:
    """Run one legacy address-switch action from a sequencer-server command."""

    state = Path(state_dir or os.environ.get("ZLC_STATE_DIR", "."))
    state.mkdir(parents=True, exist_ok=True)
    program = None
    if action in {"prepare", "wait_done"}:
        program_file = Path(program_path or os.environ["ZLC_SEQUENCE_PROGRAM"])
        program = RuntimeSequenceProgram.from_dict(json.loads(program_file.read_text(encoding="utf-8")))
    if action == "wait_done" and program is not None and _env_flag("ZLC_LEGACY_WAIT_FOR_DURATION", default=True):
        fire_time_path = state / "legacy_address_switch_fire_time.txt"
        fire_time = None
        if fire_time_path.exists():
            try:
                fire_time = float(fire_time_path.read_text(encoding="utf-8").strip())
            except ValueError:
                fire_time = None
        if fire_time is None:
            delay = float(program.duration)
        else:
            delay = fire_time + float(program.duration) - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    assignments = build_assignments(action, program, clock_hz=clock_hz)
    tcl_path = write_vivado_vio_tcl(
        state / f"legacy_address_switch_{action}.tcl",
        assignments,
        project=os.environ.get("ZLC_VIVADO_PROJECT"),
        bitstream=os.environ.get("ZLC_VIVADO_BIT"),
        probes=os.environ.get("ZLC_VIVADO_LTX"),
        vio_filter=os.environ.get("ZLC_VIO_FILTER", 'CELL_NAME=~"*vio*"'),
    )
    if dry_run:
        return tcl_path
    log_path = state / f"legacy_address_switch_{action}.log"
    try:
        result = subprocess.run(
            [vivado, "-mode", "batch", "-source", str(tcl_path)],
            cwd=state,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        message = (
            f"Vivado executable was not found: {vivado!r}.\n"
            "Set ZLC_VIVADO_BIN to the full Vivado executable path, for example "
            r"C:\Xilinx\Vivado\2023.2\bin\vivado.bat, or add Vivado bin to PATH."
        )
        log_path.write_text(message, encoding="utf-8", errors="replace")
        raise RuntimeError(f"legacy address-switch {action} could not start Vivado. See {log_path}.") from exc
    log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"legacy address-switch {action} failed with code {result.returncode}. See {log_path}.")
    if action == "fire":
        (state / "legacy_address_switch_fire_time.txt").write_text(str(time.monotonic()), encoding="utf-8")
    return tcl_path


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare/fire the legacy address-switch VIO backend from a RuntimeSequenceProgram.")
    parser.add_argument("action", choices=["prepare", "fire", "wait_done", "safe_state", "abort"], nargs="?", default=None)
    parser.add_argument("--program", default=None, help="RuntimeSequenceProgram JSON path. Defaults to ZLC_SEQUENCE_PROGRAM.")
    parser.add_argument("--state-dir", default=None, help="State/log directory. Defaults to ZLC_STATE_DIR.")
    parser.add_argument("--vivado", default=os.environ.get("ZLC_VIVADO_BIN", "vivado"))
    parser.add_argument("--clock-hz", type=float, default=None, help="Override the program clock when converting seconds to legacy ticks.")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only write the generated Tcl file.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    action = args.action or os.environ.get("ZLC_SEQUENCER_ACTION", "prepare")
    run_action(
        action,
        program_path=args.program,
        state_dir=args.state_dir,
        vivado=args.vivado,
        dry_run=args.dry_run,
        timeout=args.timeout,
        clock_hz=args.clock_hz,
    )
    return 0


def _uniform_probe_width_ticks(program: RuntimeSequenceProgram, *, clock_hz: float, probe_channel: str) -> int:
    if program.source_sequence is not None:
        sequence = PulseSequence.from_dict(program.source_sequence)
        widths = {
            int(round(pulse.duration * clock_hz))
            for pulse in sequence.effective_pulses()
            if pulse.channel == probe_channel and pulse.value
        }
    else:
        widths = set(_probe_widths_from_edges(program, probe_channel=probe_channel))
    if not widths:
        raise ValueError(f"program has no active {probe_channel!r} pulse.")
    if len(widths) != 1:
        raise ValueError(f"program has non-uniform {probe_channel!r} pulse widths: {sorted(widths)}")
    return int(next(iter(widths)))


def _probe_widths_from_edges(program: RuntimeSequenceProgram, *, probe_channel: str) -> list[int]:
    try:
        bit = program.channels.index(probe_channel)
    except ValueError as exc:
        raise ValueError(f"program channels do not contain {probe_channel!r}.") from exc
    widths: list[int] = []
    for tick, next_tick, mask in zip(program.ticks, program.ticks[1:], program.masks):
        if int(mask) & (1 << bit):
            widths.append(int(next_tick) - int(tick))
    return widths


def _validate_29bit(name: str, value: int) -> None:
    if int(value) < 0 or int(value) > MAX_29BIT:
        raise ValueError(f"{name}={value} does not fit the legacy 29-bit VIO field.")


def _env_default_assignments() -> dict[str, int]:
    raw = os.environ.get("ZLC_LEGACY_VIO_DEFAULTS", "").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("ZLC_LEGACY_VIO_DEFAULTS must be a JSON object.")
    return {str(key): int(value) for key, value in payload.items()}


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _require_single_trigger_confirmation() -> None:
    if not _env_flag("ZLC_LEGACY_REQUIRE_SINGLE_TRIGGER_CONFIRMATION", default=True):
        return
    if _env_flag("ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED", default=False):
        return
    raise RuntimeError(
        "Legacy address-switch prepare is blocked until the qCMOS trigger contract is confirmed. "
        "The original address_switch run branch can emit two emCCD/readout pulses per cycle and does "
        "not explicitly drive the trig output. Use an oscilloscope to confirm that the real qCMOS "
        "trigger input sees exactly one positive edge per cycle, then set "
        "ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED=1. If it sees two edges, patch the Verilog or use "
        "a real pulse-table backend before running readout/fidelity scans."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "build_assignments",
    "legacy_imaging_parameters",
    "main",
    "run_action",
    "write_vivado_vio_tcl",
]
