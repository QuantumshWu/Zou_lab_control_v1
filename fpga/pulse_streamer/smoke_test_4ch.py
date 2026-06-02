"""Upload and fire a simple 4-channel pulse-streamer smoke test.

Run this on the FPGA/Vivado computer after building and programming the 4ch
bitstream. It does not require the qCMOS camera. Use an oscilloscope on the
trap/cooling/probe/qcm_trigger pins to verify the output timing.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_neutral_atom():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import Zou_lab_control.neutral_atom as na

    return na


def build_smoke_program(clock_hz: float = 100_000_000.0):
    na = _load_neutral_atom()
    sequence = (
        na.PulseSequence(name="fpga_4ch_smoke")
        .pulse("trap", 0.0, 10e-6)
        .pulse("cooling", 0.0, 3e-6)
        .pulse("probe", 2e-6, 4e-6)
        .pulse("qcm_trigger", 2e-6, 1e-6)
    )
    return na.compile_runtime_program(
        sequence,
        channels=["trap", "cooling", "probe", "qcm_trigger"],
        clock_hz=clock_hz,
        trigger_channels=["qcm_trigger"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run a 4-channel ZLC pulse-streamer smoke test.")
    parser.add_argument("--state-dir", default=os.environ.get("ZLC_STATE_DIR", r"D:\zlc_sequencer_state"))
    parser.add_argument("--clock-hz", type=float, default=100_000_000.0)
    parser.add_argument("--vivado", default=os.environ.get("ZLC_PS_VIVADO_BIN", os.environ.get("ZLC_VIVADO_BIN", "vivado")))
    parser.add_argument("--write-only", action="store_true", help="Write the smoke program JSON without touching Vivado.")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args(argv)

    na = _load_neutral_atom()
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    program = build_smoke_program(clock_hz=args.clock_hz)
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=4)
    program_path = state_dir / "fpga_4ch_smoke_program.json"
    program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")

    print("ZLC 4ch smoke program")
    print(f"  program: {program_path}")
    print(f"  sequence_id: {program.sequence_id}")
    print(f"  channels: {program.channels}")
    print(f"  ticks: {program.ticks}")
    print(f"  masks: {program.masks}")
    print("  expected outputs at 100 MHz:")
    print("    trap high 0-10 us")
    print("    cooling high 0-3 us")
    print("    probe high 2-6 us")
    print("    qcm_trigger high 2-3 us")

    if args.write_only:
        return 0

    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import run_action

    run_action("prepare", program_path=program_path, state_dir=state_dir, vivado=args.vivado)
    run_action("fire", state_dir=state_dir, vivado=args.vivado)
    run_action("wait_done", state_dir=state_dir, vivado=args.vivado, timeout=args.timeout)
    run_action("safe_state", state_dir=state_dir, vivado=args.vivado)
    print("ZLC 4ch smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
