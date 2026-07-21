"""The ONE source for the runtime/compile clock default -- the FPGA tick rate.

The authoritative value lives in the single board-config file
(``fpga/board_config/streamer_config.json`` -> ``clock_hz``), read by
:func:`fpga.pulse_streamer.host.image.default_clock_hz`.  Every default that
once spelled the tick rate as a bare numeric literal -- the timing compilers
(:mod:`neutral_atom.timing`), the runtime sequencer + AXI session
(:mod:`neutral_atom.devices`), the session-facade fallback, the virtual
sequencer, and the pulse GUI -- imports :func:`default_clock_hz` (or
:data:`DEFAULT_CLOCK_HZ`) from HERE instead, so changing the config moves them
all together and the compile clock can never silently diverge from the AXI
runtime clock.

This module is the same dependency-free seam pattern as ``_paths`` /
``_readout_math`` / ``_viewer_registry``: the timing layer (which must NOT import
``devices`` or ``fpga`` directly) imports it with an absolute
``from Zou_lab_control._clock import default_clock_hz``, so the deep-relative
import trap (a swallowed ``ModuleNotFoundError``) cannot bite.  It is a thin
re-export of the config reader -- it adds no second definition of the value.

Keep this module free of any ``Zou_lab_control`` import.
"""

from __future__ import annotations

from fpga.pulse_streamer.host.image import default_clock_hz

# The static fallback (config-read failure / no file).  This mirrors
# host.image.DEFAULT_CLOCK_HZ, which is itself the documented fallback baked into
# the config reader; we re-expose it under one neutral name so a call site that
# needs a *literal* default (e.g. a dataclass field default that must be a constant
# expression, not a function call) has a single import to reach for.
DEFAULT_CLOCK_HZ: float = default_clock_hz()


def default_time_step_ns() -> float:
    """The tick grid in ns -- the SAME fact as the clock, expressed the other way.

    All timing is authored in ns and played in ticks, so both projections are needed, and
    both are derived HERE from the one configured rate.  A call site that spells its own
    tick default (``time_step_ns = 1.0``) silently pins a 1 GHz board: it does not fail,
    it just quantises to a grid the hardware does not have, and the GUI reports a clock
    nobody configured.
    """

    return 1_000_000_000.0 / default_clock_hz()


#: The literal form, for defaults that must be a constant expression.
DEFAULT_TIME_STEP_NS: float = default_time_step_ns()

__all__ = ["default_clock_hz", "DEFAULT_CLOCK_HZ",
           "default_time_step_ns", "DEFAULT_TIME_STEP_NS"]
