"""Standalone composition launcher for the pulse GUI.

Virtual and remote execution both use a complete installation authority.  The editor
itself only receives a target descriptor and command port; ``--no-sequencer`` is the
explicit offline authoring mode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


DEFAULT_PULSE_GUI_FALLBACK_CHANNELS = 62
# Default board pin map: the in-repo platform-config copy (see fpga/board_config/README.md).
DEFAULT_PULSE_GUI_XDC = Path("fpga") / "board_config" / "board.xdc"


def _default_channels(count: int) -> list[str]:
    count = int(count)
    if count <= 0:
        raise argparse.ArgumentTypeError("channel count must be positive.")
    return [f"ch{i:02d}" for i in range(count)]


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
    parser.add_argument(
        "--clock-hz",
        type=_positive_float,
        default=50_000_000.0,
        help="Offline authoring clock in Hz. Managed targets use installation readback.",
    )
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
            "Connect to an already running FPGA sequencer server when explicitly "
            "selected. Without --remote-host the launcher uses the virtual installation."
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


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv_list)
    explicit_remote = _remote_host_was_requested(argv_list)

    # Silence the harmless Windows Qt font noise ("Unable to open default EUDC font:
    # EUDC.TTE") -- it is just Qt probing the end-user-defined-characters font.
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore, QtWidgets

    import Zou_lab_control.frontend as zf
    import Zou_lab_control.neutral_atom as na

    state = na.PulseTableState.load(args.state) if args.state else None
    session = None
    command_port = None
    target_descriptor = None
    if args.no_sequencer:
        channels = _resolve_channels(args, state)
        channel_labels = _resolve_channel_labels(args, channels, state)
        port_catalog = na.PortCatalog.from_channels(
            channels, channel_labels=channel_labels)
    else:
        if explicit_remote:
            session = na.connect(
                "remote_template",
                sequencer={"host": args.remote_host, "port": args.remote_port},
                open_devices=True,
            )
        else:
            session = na.connect("virtual")
        from zlc_workbench.pulse_control import managed_pulse_command_port

        raw_sequencer = session._device_set.sequencer
        command_port = managed_pulse_command_port(
            session, session._require_runtime_services(), raw_sequencer
        )
        target_descriptor = command_port.target
        port_catalog = target_descriptor.port_catalog

    if state is not None and state.port_catalog.fingerprint != port_catalog.fingerprint:
        try:
            state = state.aligned_to_catalog(port_catalog)
        except ValueError as exc:
            _build_parser().error(str(exc))
    if state is not None and target_descriptor is not None:
        state = state.snapped(time_step_ns=target_descriptor.time_step_ns)
    channels = list(port_catalog.raw_lanes)
    channel_pins = _resolve_channel_pins(args, channels)

    if state is None:
        visible_ports = [
            port.key for port in port_catalog.ports if port.kind != "clock"
        ][:4]
        state = na.PulseTableState(
            port_catalog=port_catalog,
            time_step_ns=(
                target_descriptor.time_step_ns
                if target_descriptor is not None
                else 1_000_000_000.0 / float(args.clock_hz)
            ),
            visible_ports=visible_ports,
        )

    editor = zf.show_pulse_gui(
        state=state,
        target_descriptor=target_descriptor,
        command_port=command_port,
        channel_pins=channel_pins,
        scale=args.scale,
        window_ratio=args.window_ratio,
    )
    app = QtWidgets.QApplication.instance()
    auto_close_ms = os.environ.get("ZLC_PULSE_GUI_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(max(0, int(auto_close_ms)), app.quit)
    try:
        app.exec_()
    finally:
        if session is not None:
            session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
