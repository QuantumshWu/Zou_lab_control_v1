"""Compatibility wrapper for the device-layer sequencer server.

New code should use ``Zou_lab_control.neutral_atom.devices.sequencer_server``.
The old module path is kept so existing lab commands fail less abruptly.
"""

from .devices.sequencer_server import CommandSequencerBackend, build_arg_parser, main, run_server


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["CommandSequencerBackend", "build_arg_parser", "main", "run_server"]
