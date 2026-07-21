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
