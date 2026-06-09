"""Standalone launcher for the Zou_lab_control pulse GUI.

This entrypoint is intentionally independent from experiment configs.  It can
open an offline/local RuntimeSequencer editor, or connect directly to a running
FPGA sequencer server through RemoteSequencer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


DEFAULT_PULSE_GUI_FALLBACK_CHANNELS = 62
# Default board pin map: the in-repo platform-config copy (see fpga/board_config/README.md).
DEFAULT_PULSE_GUI_XDC = Path("fpga") / "board_config" / "board.xdc"


def _default_channels(count: int) -> list[str]:
    count = int(count)
    if count <= 0:
        raise argparse.ArgumentTypeError("channel count must be positive.")
    return [f"ch{i:02d}" for i in range(count)]


def _resolve_trigger_channels(args, channels: Sequence[str], channel_labels: Mapping[str, str] | None = None) -> list[str]:
    if args.trigger_channels:
        return [str(channel) for channel in args.trigger_channels]
    labels = {str(channel): str(label).strip().lower() for channel, label in dict(channel_labels or {}).items()}
    for channel in channels:
        if labels.get(str(channel)) == "emccd":
            return [str(channel)]
    if "emCCD" in channels:
        return ["emCCD"]
    raise ValueError("No default emCCD trigger channel was found. Provide --trigger-channels explicitly after confirming the camera trigger output.")


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return value


def _optional_positive_int_env(name: str) -> int | None:
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    value = int(text)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
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
        default=None,
        help="Build default hardware channels ch00... in FPGA bit order. Defaults to the XDC channel count.",
    )
    parser.add_argument(
        "--xdc",
        type=Path,
        default=Path(os.environ.get("ZLC_PS_XDC", DEFAULT_PULSE_GUI_XDC)),
        help="Pulse-streamer XDC used to infer the default full channel count.",
    )
    parser.add_argument(
        "--max-channel-count",
        type=int,
        default=_optional_positive_int_env("ZLC_PS_MAX_CHANNEL_COUNT"),
        help="Maximum channel count accepted from --xdc. Omit for no GUI-side limit.",
    )
    parser.add_argument("--clock-hz", type=_positive_float, default=50_000_000.0, help="Sequencer clock in Hz.")
    parser.add_argument("--trigger-channels", nargs="+", help="Hardware channel names counted as camera triggers.")
    parser.add_argument(
        "--scale",
        type=_positive_float,
        default=None,
        help="GUI scale. Omit for automatic screen/DPI fitting; use 1.0 for full size.",
    )
    parser.add_argument("--window-ratio", type=_positive_float, default=0.90, help="GUI window size as a fraction of the available screen.")
    parser.add_argument("--state", type=Path, help="Load a PulseTableState JSON file.")
    parser.add_argument(
        "--remote-host",
        default=os.environ.get("ZLC_PS_REMOTE_HOST", "127.0.0.1"),
        help=(
            "Connect to an already running FPGA sequencer server. "
            "The default tries localhost and opens offline if no server is listening; "
            "an explicit --remote-host is treated as required."
        ),
    )
    parser.add_argument("--remote-port", type=int, default=18861, help="Remote sequencer server port.")
    parser.add_argument(
        "--no-sequencer",
        action="store_true",
        help="Open only the editor without On Pulse or Stop Pulse backend calls.",
    )
    return parser


def _remote_host_was_requested(argv: Sequence[str]) -> bool:
    """Return true when the user explicitly asked for a hardware server."""

    if os.environ.get("ZLC_PS_REMOTE_HOST"):
        return True
    for item in argv:
        text = str(item)
        if text == "--remote-host" or text.startswith("--remote-host="):
            return True
    return False


def _resolve_channels(args, state) -> list[str]:
    if args.channels:
        return [str(channel) for channel in args.channels]
    if args.channel_count is not None:
        return _default_channels(args.channel_count)
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import infer_xdc_channel_count

    count = infer_xdc_channel_count(
        args.xdc,
        default=DEFAULT_PULSE_GUI_FALLBACK_CHANNELS,
        max_count=args.max_channel_count,
    )
    return _default_channels(count)


def _resolve_channel_labels(args, channels: Sequence[str], state) -> dict[str, str]:
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import infer_xdc_channel_labels

    channels = [str(channel) for channel in channels]
    labels = {
        channel: label
        for channel, label in infer_xdc_channel_labels(
            args.xdc,
            default=len(channels) or DEFAULT_PULSE_GUI_FALLBACK_CHANNELS,
            max_count=args.max_channel_count,
        ).items()
        if channel in channels and label and label != channel
    }
    if state is not None:
        labels.update({str(channel): str(label) for channel, label in state.channel_labels.items() if channel in channels and label})
    return labels


def _resolve_channel_pins(args, channels: Sequence[str]) -> dict[str, str]:
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import infer_xdc_channel_pins

    channels = [str(channel) for channel in channels]
    return {
        channel: pin
        for channel, pin in infer_xdc_channel_pins(
            args.xdc,
            default=len(channels) or DEFAULT_PULSE_GUI_FALLBACK_CHANNELS,
            max_count=args.max_channel_count,
        ).items()
        if channel in channels and pin
    }


def _connect_remote_or_offline(args, state, na, *, explicit_remote: bool):
    seed_channels = _resolve_channels(args, state)
    seed_labels = _resolve_channel_labels(args, seed_channels, state)
    seed_trigger_channels = _resolve_trigger_channels(args, seed_channels, seed_labels)
    try:
        sequencer = na.RemoteSequencer(
            host=args.remote_host,
            port=args.remote_port,
            channels=seed_channels,
            clock_hz=args.clock_hz,
            trigger_channels=seed_trigger_channels,
            connect_on_init=True,
        )
    except Exception as exc:
        if explicit_remote:
            raise
        notice = (
            f"Could not connect to local sequencer server at {args.remote_host}:{args.remote_port}; "
            "opened offline editor. Start fpga\\run_server.bat for hardware control, "
            "or pass --remote-host for a required remote connection."
        )
        print(f"ZLC pulse GUI: {notice}\n{type(exc).__name__}: {exc}")
        return None, seed_channels, seed_trigger_channels, notice
    return sequencer, list(sequencer.channels), list(sequencer.trigger_channels), None


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv_list)
    explicit_remote = _remote_host_was_requested(argv_list)

    from PyQt5 import QtCore, QtWidgets

    import Zou_lab_control.frontend as zf
    import Zou_lab_control.neutral_atom as na

    state = na.PulseTableState.load(args.state) if args.state else None
    sequencer = None
    startup_notice = None
    if not args.no_sequencer:
        if args.remote_host:
            sequencer, channels, trigger_channels, startup_notice = _connect_remote_or_offline(
                args,
                state,
                na,
                explicit_remote=explicit_remote,
            )
        else:
            channels = _resolve_channels(args, state)
            channel_labels = _resolve_channel_labels(args, channels, state)
            trigger_channels = _resolve_trigger_channels(args, channels, channel_labels)
            sequencer = na.RuntimeSequencer(
                channels=channels,
                clock_hz=args.clock_hz,
                trigger_channels=trigger_channels,
            )
    else:
        channels = _resolve_channels(args, state)
        channel_labels = _resolve_channel_labels(args, channels, state)
        trigger_channels = _resolve_trigger_channels(args, channels, channel_labels)

    if state is not None and list(state.channels) != list(channels):
        if all(channel in channels for channel in state.channels):
            state = state.aligned_to_channels(channels)
        else:
            parser = _build_parser()
            parser.error(
                "loaded pulse state channels do not match the sequencer channels: "
                f"state={list(state.channels)!r}, sequencer={list(channels)!r}. "
                "Load a matching pulse JSON or use hardware channel names with display labels."
            )
    channel_labels = locals().get("channel_labels") or _resolve_channel_labels(args, channels, state)
    channel_pins = _resolve_channel_pins(args, channels)
    if state is not None and channel_labels:
        for channel, label in channel_labels.items():
            if channel in state.channels and channel not in state.channel_labels and label != channel:
                state.channel_labels[channel] = label
        state.validate()

    if state is None and not args.channels:
        state = na.PulseTableState(
            channels=channels,
            time_step_ns=1_000_000_000.0 / float(getattr(sequencer, "clock_hz", args.clock_hz)),
            visible_channels=channels[: min(4, len(channels))],
            channel_labels=channel_labels,
        )

    editor = zf.show_pulse_gui(
        state=state,
        channels=channels,
        sequencer=sequencer,
        channel_labels=channel_labels,
        channel_pins=channel_pins,
        scale=args.scale,
        window_ratio=args.window_ratio,
    )
    if startup_notice:
        if hasattr(editor, "summary"):
            editor.summary.setText(startup_notice)
        if hasattr(editor, "preview_status"):
            editor.preview_status.setText(startup_notice)
    app = QtWidgets.QApplication.instance()
    auto_close_ms = os.environ.get("ZLC_PULSE_GUI_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(max(0, int(auto_close_ms)), app.quit)
    app.exec_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
