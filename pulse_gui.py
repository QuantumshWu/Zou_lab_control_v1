"""Standalone launcher for the Zou_lab_control pulse GUI.

This entrypoint is intentionally independent from experiment configs.  It can
open an offline/local RuntimeSequencer editor, or connect directly to a running
FPGA sequencer server through RemoteSequencer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _default_channels(count: int) -> list[str]:
    count = int(count)
    if count <= 0:
        raise argparse.ArgumentTypeError("channel count must be positive.")
    return [f"ch{i:02d}" for i in range(count)]


def _resolve_trigger_channels(args, channels: Sequence[str]) -> list[str]:
    if args.trigger_channels:
        return [str(channel) for channel in args.trigger_channels]
    for candidate in ("qcm_trigger", "camera_trigger", "trig", "ch03"):
        if candidate in channels:
            return [candidate]
    return [channels[min(3, len(channels) - 1)]]


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the standalone Zou_lab_control pulse GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Hardware channel names in FPGA bit order. Overrides --channel-count.",
    )
    parser.add_argument(
        "--channel-count",
        type=int,
        default=40,
        help="Build default hardware channels ch00... in FPGA bit order when --channels is not given.",
    )
    parser.add_argument("--clock-hz", type=_positive_float, default=100_000_000.0, help="Sequencer clock in Hz.")
    parser.add_argument("--trigger-channels", nargs="+", help="Hardware channel names counted as camera triggers.")
    parser.add_argument("--scale", type=_positive_float, default=0.82, help="GUI scale. Use 1.0 for full size.")
    parser.add_argument("--window-ratio", type=_positive_float, default=0.90, help="Fixed GUI window size as a fraction of the screen.")
    parser.add_argument("--state", type=Path, help="Load a PulseTableState JSON file.")
    parser.add_argument("--remote-host", help="Connect to an already running FPGA sequencer server.")
    parser.add_argument("--remote-port", type=int, default=18861, help="Remote sequencer server port.")
    parser.add_argument(
        "--no-sequencer",
        action="store_true",
        help="Open only the editor without prepare/fire/wait/safe backend calls.",
    )
    return parser


def _resolve_channels(args, state) -> list[str]:
    if args.channels:
        return [str(channel) for channel in args.channels]
    if state is not None:
        return list(state.channels)
    return _default_channels(args.channel_count)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from PyQt5 import QtWidgets

    import Zou_lab_control.frontend as zf
    import Zou_lab_control.neutral_atom as na

    state = na.PulseTableState.load(args.state) if args.state else None
    channels = _resolve_channels(args, state)
    trigger_channels = _resolve_trigger_channels(args, channels)
    if state is None and not args.channels:
        state = na.PulseTableState(
            channels=channels,
            time_step_ns=1_000_000_000.0 / args.clock_hz,
            visible_channels=channels[: min(4, len(channels))],
        )

    sequencer = None
    if not args.no_sequencer:
        if args.remote_host:
            sequencer = na.RemoteSequencer(
                host=args.remote_host,
                port=args.remote_port,
                channels=channels,
                clock_hz=args.clock_hz,
                trigger_channels=trigger_channels,
            )
        else:
            sequencer = na.RuntimeSequencer(
                channels=channels,
                clock_hz=args.clock_hz,
                trigger_channels=trigger_channels,
            )

    zf.show_pulse_gui(
        state=state,
        channels=channels,
        sequencer=sequencer,
        scale=args.scale,
        window_ratio=args.window_ratio,
    )
    QtWidgets.QApplication.instance().exec_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
