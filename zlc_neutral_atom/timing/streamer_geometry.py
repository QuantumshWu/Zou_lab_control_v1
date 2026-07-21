"""The ONE source for the compile-affecting streamer geometry SCALARS the timing/sequencer layer
needs -- the affine fixed-point fraction and the scan slot-operand narrow width.

The authoritative values live in the single board-config file
(``fpga/board_config/streamer_config.json`` -> ``params.coeff_frac_bits`` /
``params.slot_mul_width``), read by :mod:`fpga.pulse_streamer.host.image`.  Both are
bitstream-affecting (``coeff_frac_bits`` is a StreamerParams field folded into the geometry
fingerprint; ``slot_mul_width`` is an RTL localparam), so the host-side SCAN COMPILER
(:mod:`neutral_atom.timing.pulse_table`, :mod:`neutral_atom.devices.sequencer`) and the cycle
mirror (:mod:`fpga.pulse_streamer.host.engine_model`) MUST use exactly these values or the emitted
ticks / narrowed values disagree with the synthesized bitstream.  They import from HERE instead of
spelling a bare literal 8 / 25, so editing the config moves them all together.

This module is the same dependency-free seam pattern as ``_clock`` / ``_paths`` /
``_readout_math`` / ``_viewer_registry``: the timing layer (which must NOT import ``devices`` or
``fpga`` transitively-cyclically) imports it with an absolute
``from Zou_lab_control._streamer_geometry import default_coeff_frac_bits``, so the deep-relative
import trap (a swallowed ``ModuleNotFoundError``) cannot bite.  It is a thin re-export of the config
reader -- it adds no second definition of the value.

Keep this module free of any ``Zou_lab_control`` import.
"""

from __future__ import annotations

from fpga.pulse_streamer.host.image import default_coeff_frac_bits, default_slot_mul_width

# Static constants for call sites that need a *literal* default (a dataclass / function-signature
# default must be a constant expression, not a function call).  These mirror the shipped-config
# values via the reader; a config-read failure falls back to the same documented defaults the
# StreamerParams airbag uses.
DEFAULT_COEFF_FRAC_BITS: int = default_coeff_frac_bits()
DEFAULT_SLOT_MUL_WIDTH: int = default_slot_mul_width()

__all__ = [
    "default_coeff_frac_bits", "default_slot_mul_width",
    "DEFAULT_COEFF_FRAC_BITS", "DEFAULT_SLOT_MUL_WIDTH",
]


# ---- Channel/bus geometry the pulse editor derives its rows from ------------------
# Same config single source (fpga/board_config/streamer_config.json); the except arm
# mirrors the shipped-config literals exactly as the StreamerParams airbag does.
try:  # pragma: no cover - exercised whenever the fpga package is importable (the norm)
    from fpga.pulse_streamer.host.image import load_streamer_config as _load_streamer_config

    _CFG_PARAMS = _load_streamer_config()["params"]
    DEFAULT_FPGA_CHANNEL_COUNT = int(_CFG_PARAMS.channel_count)
    DEFAULT_BUS_COUNT = int(_CFG_PARAMS.bus_count)
    DEFAULT_BUS_WIDTH = int(_CFG_PARAMS.bus_width)
    EVT_FIFO_DEPTH = int(getattr(_CFG_PARAMS, "evt_fifo_depth", 64))
    BUS_EVT_FIFO_DEPTH = int(getattr(_CFG_PARAMS, "bus_evt_fifo_depth", 32))
except Exception:  # pragma: no cover - fpga package not importable
    DEFAULT_FPGA_CHANNEL_COUNT = 62
    DEFAULT_BUS_COUNT = 4
    DEFAULT_BUS_WIDTH = 10
    EVT_FIFO_DEPTH = 64
    BUS_EVT_FIFO_DEPTH = 32


def hardware_channel_names(count: int = DEFAULT_FPGA_CHANNEL_COUNT) -> list[str]:
    """Return hardware channel names in FPGA bit order."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive.")
    return [f"ch{index:02d}" for index in range(count)]


def delay_eligible_channel_count(channel_count: int, bus_count: int = DEFAULT_BUS_COUNT,
                                 bus_width: int = DEFAULT_BUS_WIDTH) -> int:
    """Number of leading channels that can carry a TTL output delay.

    A channel is delay-eligible iff its engine bit drives a real TTL pin -- i.e. it is
    NOT a bus-member bit (``bus_count*bus_width`` of them, pins driven by ``bus_out``)
    and NOT a per-bus ``da_clk`` pin (``bus_count`` of them).  The board lays the real
    TTL outputs out FIRST, so the eligible set is the leading
    ``channel_count - bus_count*(bus_width+1)`` indices -- matching the RTL's compacted
    (identity) event-FIFO map."""
    return max(0, int(channel_count) - int(bus_count) * (int(bus_width) + 1))


__all__ += [
    "DEFAULT_FPGA_CHANNEL_COUNT", "DEFAULT_BUS_COUNT", "DEFAULT_BUS_WIDTH",
    "EVT_FIFO_DEPTH", "BUS_EVT_FIFO_DEPTH",
    "hardware_channel_names", "delay_eligible_channel_count",
]
