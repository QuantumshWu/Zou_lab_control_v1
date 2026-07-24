"""Neutral timing projection of the deployed FPGA tick grid."""

from __future__ import annotations

from fpga.pulse_streamer.host.image import default_clock_hz as _default_clock_hz


def default_time_step_ns() -> float:
    """The tick grid in ns -- the SAME fact as the clock, expressed the other way.

    All timing is authored in ns and played in ticks, so both projections are needed, and
    both are derived HERE from the one configured rate.  A call site that spells its own
    tick default (``time_step_ns = 1.0``) silently pins a 1 GHz board: it does not fail,
    it just quantises to a grid the hardware does not have, and the GUI reports a clock
    nobody configured.
    """

    return 1_000_000_000.0 / _default_clock_hz()

__all__ = ["default_time_step_ns"]
