"""BRAM program image for the JTAG-to-AXI *loader* path of the affine edge-table
pulse streamer.

Architecture
------------
The validated, seamless affine edge-table engine (``fpga/pulse_streamer/
zlc_pulse_streamer.v``) is kept byte-for-byte.  Instead of programming its LUTRAM
tables through VIO probes, an on-chip loader FSM (``zlc_axi_program_loader.v``)
copies a *program image* out of the AXI-addressable BRAM into the engine's
``prog_*`` / ``scan_prog_*`` / ``bus_prog_*`` write ports, then releases reset and
pulses ``start``.  Because the engine is unchanged, repeat-to-repeat and
scan-point-to-scan-point playback stay single-cycle seamless and the affine scan
stays tick-exact -- the loader only has to deliver the *same data* the VIO upload
path used to deliver.

This module is the single source of truth for that image layout.  It packs a
:class:`RuntimeSequenceProgram` into 32-bit BRAM words at fixed word offsets, and
provides a loader-walk model (:func:`unpack_program`) that reconstructs the
program from the image exactly as the RTL loader walks it.  The round-trip
``unpack(pack(program)) == program`` is the contract the RTL loader must honour.

Word layout (word = 32 bits; multi-word fields are little-endian, low word first;
signed fields are two's complement):

  CTRL region   (offset 0, ``CTRL_WORDS`` words) -- scalars + counts + loop +
                runtime command/status mailbox.
  EDGE region   (``edge_base``) -- ``MAX_EDGES`` rows x ``EDGE_WORDS`` words:
                [tick, coeff_lo, coeff_hi, mask_lo, mask_hi].
  SCAN region   (``scan_base``) -- ``MAX_SCAN_POINTS`` rows x ``SCAN_WORDS`` words:
                one 32-bit signed slot value per word.
  BUS region    (``bus_base``) -- ``BUS_COUNT * MAX_BUS_SEGMENTS`` rows x
                ``BUS_WORDS`` words, grouped bus-major then addr-minor:
                [start_tick, stop_tick, start_coeff_lo, start_coeff_hi,
                 stop_coeff_lo, stop_coeff_hi, flags].

Only the *used* rows of each region are emitted by :func:`pack_program`
(returned as a sparse ``{word_offset: value}`` map), so the host uploads a small
image; the region bases stay at their maximum positions so the loader always
knows where each region starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "EdgeTableImageParams",
    "CtrlWords",
    "pack_program",
    "unpack_program",
    "image_word_span",
    "FpgaPartProfile",
    "FPGA_PARTS",
    "SolvedCapacity",
    "solve_capacity",
    "part_profile",
]


# --- CTRL word indices (offsets within the CTRL region) ----------------------
class CtrlWords:
    MAGIC = 0            # identifies a valid image (host writes, loader checks)
    COMMAND = 1          # host -> loader: bit0=load, bit1=fire, bit2=reset, bit3=safe
    STATUS = 2           # loader -> host: bit0=loaded, bit1=running, bit2=done, bit3=error
    PROG_COUNT = 3       # number of active edges
    SCAN_COUNT = 4       # number of scan points
    SCAN_ENABLE = 5      # bit0
    REPEAT_FOREVER = 6   # bit0
    LOOP_START_ADDR = 7
    LOOP_COUNT = 8
    LOOP_END_TICK = 9
    LOOP_END_COEFF_LO = 10
    LOOP_END_COEFF_HI = 11
    BUS_COUNTS = 12      # per-bus segment counts packed (bus j at [j*7 +: 7])
    SLOT_COUNT = 13      # number of scan slots actually used (<= num_slots)


CTRL_WORDS = 32  # reserve a round CTRL block; EDGE region starts here

IMAGE_MAGIC = 0x5A4C4531  # "ZLE1"

# Command / status bit masks (shared with the RTL loader).
CMD_LOAD = 1 << 0
CMD_FIRE = 1 << 1
CMD_RESET = 1 << 2
CMD_SAFE = 1 << 3

STATUS_LOADED = 1 << 0
STATUS_RUNNING = 1 << 1
STATUS_DONE = 1 << 2
STATUS_ERROR = 1 << 3


@dataclass(frozen=True)
class EdgeTableImageParams:
    """Geometry of the edge-table program image (matches the RTL parameters)."""

    channel_count: int = 62
    num_slots: int = 4
    coeff_width: int = 16
    tick_width: int = 32
    coeff_frac_bits: int = 8
    max_edges: int = 1024
    max_scan_points: int = 1024
    bus_count: int = 4
    bus_width: int = 10
    bus_seg_addr_width: int = 6  # MAX_BUS_SEGMENTS = 1 << this
    bus_sel_width: int = 3

    # --- derived word geometry ------------------------------------------------
    @property
    def coeff_bits(self) -> int:
        return self.num_slots * self.coeff_width

    @property
    def coeff_words(self) -> int:
        return _ceil_words(self.coeff_bits)

    @property
    def mask_words(self) -> int:
        return _ceil_words(self.channel_count)

    @property
    def edge_words(self) -> int:
        # tick(1) + coeff_words + mask_words
        return 1 + self.coeff_words + self.mask_words

    @property
    def scan_words(self) -> int:
        # one 32-bit signed slot value per word
        return self.num_slots

    @property
    def max_bus_segments(self) -> int:
        return 1 << self.bus_seg_addr_width

    @property
    def bus_rows(self) -> int:
        return self.bus_count * self.max_bus_segments

    @property
    def bus_words(self) -> int:
        # start_tick(1)+stop_tick(1)+start_coeff(coeff_words)+stop_coeff(coeff_words)+flags(1)
        return 2 + 2 * self.coeff_words + 1

    @property
    def edge_base(self) -> int:
        return CTRL_WORDS

    @property
    def scan_base(self) -> int:
        return self.edge_base + self.max_edges * self.edge_words

    @property
    def bus_base(self) -> int:
        return self.scan_base + self.max_scan_points * self.scan_words

    @property
    def total_words(self) -> int:
        return self.bus_base + self.bus_rows * self.bus_words


def _ceil_words(bits: int) -> int:
    return (int(bits) + 31) // 32


def _to_unsigned(value: int, width: int) -> int:
    """Two's-complement encode a (possibly signed) integer into ``width`` bits."""
    mask = (1 << width) - 1
    return int(value) & mask


def _from_unsigned(value: int, width: int) -> int:
    """Decode a two's-complement ``width``-bit field to a signed int."""
    value &= (1 << width) - 1
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _pack_field_words(value: int, total_bits: int) -> list[int]:
    """Split an unsigned ``total_bits`` integer into little-endian 32-bit words."""
    value &= (1 << total_bits) - 1
    return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(_ceil_words(total_bits))]


def _unpack_field_words(words: list[int], total_bits: int) -> int:
    value = 0
    for i, word in enumerate(words):
        value |= (int(word) & 0xFFFFFFFF) << (32 * i)
    return value & ((1 << total_bits) - 1)


def _pack_slot_coeffs(coeffs, params: EdgeTableImageParams) -> int:
    """Pack ``num_slots`` signed coeffs little-endian (slot j at j*coeff_width)."""
    coeffs = list(coeffs or [])
    acc = 0
    for j in range(params.num_slots):
        c = coeffs[j] if j < len(coeffs) else 0
        acc |= _to_unsigned(c, params.coeff_width) << (j * params.coeff_width)
    return acc


def _unpack_slot_coeffs(value: int, params: EdgeTableImageParams) -> list[int]:
    out = []
    for j in range(params.num_slots):
        field = (value >> (j * params.coeff_width)) & ((1 << params.coeff_width) - 1)
        out.append(_from_unsigned(field, params.coeff_width))
    return out


def _bus_mode_value(mode) -> int:
    mode = str(mode).strip().lower()
    if mode == "edge":
        return 1
    if mode == "ramp":
        return 2
    raise ValueError(f"unsupported bus segment mode {mode!r}.")


def _bus_mode_name(value: int) -> str:
    if value == 1:
        return "edge"
    if value == 2:
        return "ramp"
    raise ValueError(f"unsupported bus segment mode code {value}.")


def image_word_span(params: EdgeTableImageParams | None = None) -> int:
    """Total addressable word span of the image (for BRAM depth sizing)."""
    return (params or EdgeTableImageParams()).total_words


def pack_program(program, params: EdgeTableImageParams | None = None) -> dict[int, int]:
    """Pack a ``RuntimeSequenceProgram`` into a sparse ``{word_offset: value}`` image.

    Only used rows are emitted.  CTRL ``COMMAND``/``STATUS`` are runtime mailbox
    words and are NOT written here (the host drives them at fire time).
    """

    params = params or EdgeTableImageParams()
    ticks = [int(t) for t in program.ticks]
    masks = [int(m) for m in program.masks]
    if len(ticks) != len(masks):
        raise ValueError("program ticks and masks must have equal length.")
    n_edges = len(ticks)
    if n_edges > params.max_edges:
        raise ValueError(f"program has {n_edges} edges > max_edges {params.max_edges}.")

    slot_count = int(getattr(program, "slot_count", 0) or 0)
    if slot_count > params.num_slots:
        raise ValueError(f"program uses {slot_count} slots > num_slots {params.num_slots}.")
    frac_bits = int(getattr(program, "scan_coeff_frac_bits", params.coeff_frac_bits))
    if frac_bits != params.coeff_frac_bits:
        raise ValueError(
            f"program coeff_frac_bits {frac_bits} != image {params.coeff_frac_bits}."
        )

    tick_slot_coeffs = list(
        getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in ticks]
    )
    if len(tick_slot_coeffs) != n_edges:
        raise ValueError("tick_slot_coeffs rows must match the edge count.")

    scan_points = [list(p) for p in (getattr(program, "scan_points", None) or [])]
    n_points = len(scan_points)
    if n_points > params.max_scan_points:
        raise ValueError(
            f"program has {n_points} scan points > max_scan_points {params.max_scan_points}."
        )
    for p_idx, point in enumerate(scan_points):
        if len(point) != slot_count:
            raise ValueError(f"scan point {p_idx} must carry {slot_count} slot values.")

    bus_segments = list(getattr(program, "bus_segments", None) or [])

    words: dict[int, int] = {}

    # --- CTRL scalars ---------------------------------------------------------
    words[CtrlWords.MAGIC] = IMAGE_MAGIC
    words[CtrlWords.PROG_COUNT] = n_edges
    words[CtrlWords.SCAN_COUNT] = n_points
    words[CtrlWords.SCAN_ENABLE] = 1 if n_points > 0 else 0
    words[CtrlWords.REPEAT_FOREVER] = 1 if bool(getattr(program, "repeat_forever", False)) else 0
    words[CtrlWords.LOOP_START_ADDR] = int(getattr(program, "loop_start_index", 0))
    loop_count = int(getattr(program, "loop_count", 1) or 1)
    words[CtrlWords.LOOP_COUNT] = loop_count
    words[CtrlWords.LOOP_END_TICK] = _to_unsigned(int(getattr(program, "loop_end_tick", 0)), params.tick_width)
    loop_end_coeffs = _pack_slot_coeffs(getattr(program, "loop_end_slot_coeffs", None), params)
    lo_words = _pack_field_words(loop_end_coeffs, params.coeff_bits)
    words[CtrlWords.LOOP_END_COEFF_LO] = lo_words[0]
    words[CtrlWords.LOOP_END_COEFF_HI] = lo_words[1] if len(lo_words) > 1 else 0
    words[CtrlWords.SLOT_COUNT] = slot_count

    # --- EDGE region ----------------------------------------------------------
    base = params.edge_base
    for i in range(n_edges):
        row = base + i * params.edge_words
        words[row + 0] = _to_unsigned(ticks[i], params.tick_width)
        coeff_val = _pack_slot_coeffs(tick_slot_coeffs[i], params)
        cw = _pack_field_words(coeff_val, params.coeff_bits)
        for w in range(params.coeff_words):
            words[row + 1 + w] = cw[w] if w < len(cw) else 0
        mw = _pack_field_words(masks[i] & ((1 << params.channel_count) - 1), params.channel_count)
        for w in range(params.mask_words):
            words[row + 1 + params.coeff_words + w] = mw[w] if w < len(mw) else 0

    # --- SCAN region ----------------------------------------------------------
    base = params.scan_base
    for p in range(n_points):
        row = base + p * params.scan_words
        for j in range(params.num_slots):
            val = scan_points[p][j] if j < slot_count else 0
            words[row + j] = _to_unsigned(val, params.tick_width)

    # --- BUS region (bus-major, addr-minor) -----------------------------------
    per_bus: list[list[object]] = [[] for _ in range(params.bus_count)]
    for seg in bus_segments:
        b = int(getattr(seg, "bus_index", 0))
        if b < 0 or b >= params.bus_count:
            raise ValueError(f"bus segment bus_index {b} out of range [0,{params.bus_count}).")
        per_bus[b].append(seg)
    bus_counts_packed = 0
    cnt_width = params.bus_seg_addr_width + 1
    base = params.bus_base
    for b in range(params.bus_count):
        segs = per_bus[b]
        if len(segs) > params.max_bus_segments:
            raise ValueError(f"bus {b} has {len(segs)} segments > max {params.max_bus_segments}.")
        bus_counts_packed |= (len(segs) & ((1 << cnt_width) - 1)) << (b * cnt_width)
        for addr, seg in enumerate(segs):
            row = base + (b * params.max_bus_segments + addr) * params.bus_words
            words[row + 0] = _to_unsigned(int(getattr(seg, "start_tick", 0)), params.tick_width)
            words[row + 1] = _to_unsigned(int(getattr(seg, "stop_tick", 0)), params.tick_width)
            sc = _pack_field_words(_pack_slot_coeffs(getattr(seg, "start_tick_coeffs", None), params), params.coeff_bits)
            ec = _pack_field_words(_pack_slot_coeffs(getattr(seg, "stop_tick_coeffs", None), params), params.coeff_bits)
            for w in range(params.coeff_words):
                words[row + 2 + w] = sc[w] if w < len(sc) else 0
                words[row + 2 + params.coeff_words + w] = ec[w] if w < len(ec) else 0
            flags = 0
            flags |= (int(getattr(seg, "start_value", 0)) & ((1 << params.bus_width) - 1)) << 0
            flags |= (int(getattr(seg, "stop_value", 0)) & ((1 << params.bus_width) - 1)) << params.bus_width
            flags |= (_bus_mode_value(getattr(seg, "mode", "edge")) & 0x3) << (2 * params.bus_width)
            flags |= (int(getattr(seg, "value_select", 0)) & ((1 << params.bus_sel_width) - 1)) << (2 * params.bus_width + 2)
            flags |= (b & 0x3) << (2 * params.bus_width + 2 + params.bus_sel_width)
            words[row + 2 + 2 * params.coeff_words] = flags
    words[CtrlWords.BUS_COUNTS] = bus_counts_packed

    return words


def unpack_program(words: Mapping[int, int], params: EdgeTableImageParams | None = None) -> dict:
    """Reconstruct program fields from an image, exactly as the RTL loader walks it.

    ``words`` may be a sparse map (missing entries read as 0), matching a BRAM
    whose unused rows are never uploaded.
    """

    params = params or EdgeTableImageParams()

    def w(offset: int) -> int:
        return int(words.get(offset, 0)) & 0xFFFFFFFF

    if w(CtrlWords.MAGIC) != IMAGE_MAGIC:
        raise ValueError("image MAGIC mismatch -- not a valid edge-table program image.")

    n_edges = w(CtrlWords.PROG_COUNT)
    n_points = w(CtrlWords.SCAN_COUNT)
    slot_count = w(CtrlWords.SLOT_COUNT)

    ticks: list[int] = []
    masks: list[int] = []
    tick_slot_coeffs: list[list[int]] = []
    base = params.edge_base
    for i in range(n_edges):
        row = base + i * params.edge_words
        ticks.append(w(row + 0))  # ticks are non-negative in practice
        coeff_words = [w(row + 1 + k) for k in range(params.coeff_words)]
        coeff_val = _unpack_field_words(coeff_words, params.coeff_bits)
        tick_slot_coeffs.append(_unpack_slot_coeffs(coeff_val, params))
        mask_words = [w(row + 1 + params.coeff_words + k) for k in range(params.mask_words)]
        masks.append(_unpack_field_words(mask_words, params.channel_count))

    scan_points: list[list[int]] = []
    base = params.scan_base
    for p in range(n_points):
        row = base + p * params.scan_words
        point = [_from_unsigned(w(row + j), params.tick_width) for j in range(slot_count)]
        scan_points.append(point)

    cnt_width = params.bus_seg_addr_width + 1
    bus_counts_packed = w(CtrlWords.BUS_COUNTS)
    bus_segments: list[dict] = []
    base = params.bus_base
    for b in range(params.bus_count):
        count = (bus_counts_packed >> (b * cnt_width)) & ((1 << cnt_width) - 1)
        for addr in range(count):
            row = base + (b * params.max_bus_segments + addr) * params.bus_words
            start_tick = w(row + 0)
            stop_tick = w(row + 1)
            sc_words = [w(row + 2 + k) for k in range(params.coeff_words)]
            ec_words = [w(row + 2 + params.coeff_words + k) for k in range(params.coeff_words)]
            start_coeffs = _unpack_slot_coeffs(_unpack_field_words(sc_words, params.coeff_bits), params)
            stop_coeffs = _unpack_slot_coeffs(_unpack_field_words(ec_words, params.coeff_bits), params)
            flags = w(row + 2 + 2 * params.coeff_words)
            start_value = (flags >> 0) & ((1 << params.bus_width) - 1)
            stop_value = (flags >> params.bus_width) & ((1 << params.bus_width) - 1)
            mode = _bus_mode_name((flags >> (2 * params.bus_width)) & 0x3)
            value_select = (flags >> (2 * params.bus_width + 2)) & ((1 << params.bus_sel_width) - 1)
            bus_segments.append(
                {
                    "bus_index": b,
                    "start_tick": start_tick,
                    "stop_tick": stop_tick,
                    "start_value": start_value,
                    "stop_value": stop_value,
                    "mode": mode,
                    "value_select": value_select,
                    "start_tick_coeffs": start_coeffs,
                    "stop_tick_coeffs": stop_coeffs,
                }
            )

    loop_end_coeff_val = _unpack_field_words(
        [w(CtrlWords.LOOP_END_COEFF_LO), w(CtrlWords.LOOP_END_COEFF_HI)], params.coeff_bits
    )
    return {
        "ticks": ticks,
        "masks": masks,
        "tick_slot_coeffs": tick_slot_coeffs,
        "scan_points": scan_points,
        "slot_count": slot_count,
        "repeat_forever": bool(w(CtrlWords.REPEAT_FOREVER) & 1),
        "scan_enable": bool(w(CtrlWords.SCAN_ENABLE) & 1),
        "loop_start_index": w(CtrlWords.LOOP_START_ADDR),
        "loop_count": w(CtrlWords.LOOP_COUNT),
        "loop_end_tick": w(CtrlWords.LOOP_END_TICK),
        "loop_end_slot_coeffs": _unpack_slot_coeffs(loop_end_coeff_val, params),
        "bus_segments": bus_segments,
    }


# ---------------------------------------------------------------------------
# Capacity solver: derive (max_edges, max_scan_points, address widths, BRAM
# depth) from the FPGA part's resource budget + the XDC channel count, so a
# different XDC or FPGA does NOT require hand-editing the RTL/tcl.  This is the
# single source of truth the RTL localparams + create-project tcl + host flow
# from.
#
# Architecture D (validated engine + BRAM tables): the edge table (tick/coeff/
# mask) and the scan-point table (PING + PONG double-buffer) live in block RAM;
# the analog-bus segment tables stay in LUTRAM so the per-tick combinatorial
# bus/ramp engine is untouched.  So the BRAM budget counts edge + scan; the bus
# tables count against the distributed-RAM (LUTRAM) budget.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FpgaPartProfile:
    """Resource counts for an FPGA part (AMD DS180, 7-series overview)."""

    name: str
    ramb36: int               # number of 36 Kb block RAMs
    lut: int                  # total 6-input LUTs
    ff: int                   # total flip-flops
    dsp: int                  # DSP48 slices
    distributed_ram_kib: int  # vendor-rated max distributed RAM (LUTRAM)


# Keys are the bare part stems; package/speed suffixes (e.g. "...fgg484-2") are
# matched by prefix in :func:`part_profile`.
FPGA_PARTS: dict[str, FpgaPartProfile] = {
    "xc7a35t": FpgaPartProfile("xc7a35t", 50, 20800, 41600, 90, 400),
    "xc7a50t": FpgaPartProfile("xc7a50t", 75, 32600, 65200, 120, 600),
    "xc7a75t": FpgaPartProfile("xc7a75t", 105, 47200, 94400, 180, 892),
    "xc7a100t": FpgaPartProfile("xc7a100t", 135, 63400, 126800, 240, 1188),
    "xc7a200t": FpgaPartProfile("xc7a200t", 365, 134600, 269200, 740, 2888),
}


def part_profile(part: "str | FpgaPartProfile") -> FpgaPartProfile:
    if isinstance(part, FpgaPartProfile):
        return part
    key = str(part).strip().lower()
    for name in sorted(FPGA_PARTS, key=len, reverse=True):
        if key.startswith(name):
            return FPGA_PARTS[name]
    raise KeyError(f"unknown FPGA part {part!r}; add it to FPGA_PARTS.")


def _ramb36_for(width_bits: int, depth: int) -> int:
    """RAMB36 blocks for a width x depth memory (36-bit x 1024 per block)."""
    if depth <= 0 or width_bits <= 0:
        return 0
    return math.ceil(width_bits / 36) * math.ceil(depth / 1024)


def _pow2_at_least(value: int) -> int:
    n = 1
    while n < value:
        n <<= 1
    return n


def _addr_width(depth: int) -> int:
    """Address bits to index ``depth`` rows (depth treated as a power of two)."""
    return max(1, _pow2_at_least(max(1, depth)).bit_length() - 1)


@dataclass(frozen=True)
class SolvedCapacity:
    """Result of :func:`solve_capacity` — the geometry every consumer derives from."""

    part: str
    params: EdgeTableImageParams        # max_edges / max_scan_points solved
    edge_addr_width: int
    scan_addr_width: int
    pong_depth: int                     # streaming double-buffer window (0 = none)
    image_bram_depth: int               # AXI program-image span rounded to pow2 words
    ramb36_used: int                    # edge + scan + ctrl block RAMs
    ramb36_budget: int                  # floor(target_pct * part.ramb36)
    lutram_luts: int                    # bus tables in distributed RAM (LUTRAM)
    resource_report: dict               # per-resource used / budget / pct / ok

    def all_within_budget(self) -> bool:
        return all(r["ok"] for r in self.resource_report.values())


def solve_capacity(
    part: "str | FpgaPartProfile",
    *,
    channel_count: int = 62,
    num_slots: int = 4,
    coeff_width: int = 16,
    tick_width: int = 32,
    coeff_frac_bits: int = 8,
    bus_count: int = 4,
    bus_width: int = 10,
    bus_seg_addr_width: int = 6,
    bus_sel_width: int = 3,
    target_pct: float = 75.0,
    max_edges_cap: int = 2048,
    max_points_cap: int = 4096,
    pong_depth: int = 512,
    engine_logic_luts: int = 4000,
    engine_ff: int = 4000,
    engine_dsp: int = 8,
) -> SolvedCapacity:
    """Solve (max_edges, max_scan_points, addr widths, BRAM depth) for a part.

    Edges are capped at ``max_edges_cap``, then scan points are maximised under the
    remaining BRAM budget; a ``pong_depth`` streaming window keeps total points
    unbounded via host ping-pong refills.  Every resource (RAMB36/LUT/FF/DSP) is
    checked against ``target_pct``.  Architecture D: edge+scan in BRAM, bus LUTRAM.
    """

    prof = part_profile(part)
    pct = max(1.0, min(100.0, float(target_pct)))
    ramb36_budget = int(prof.ramb36 * pct / 100.0)

    coeff_bits = num_slots * coeff_width
    scan_bits = num_slots * tick_width
    max_bus_segments = 1 << bus_seg_addr_width
    bus_rows = bus_count * max_bus_segments
    bus_row_bits = 2 * tick_width + 2 * coeff_bits + (2 * bus_width + 2 + bus_sel_width)
    lutram_luts = math.ceil(bus_row_bits * bus_rows / 64)

    def edge_ramb(n: int) -> int:
        return (_ramb36_for(tick_width, n)
                + _ramb36_for(coeff_bits, n)
                + _ramb36_for(channel_count, n))

    def scan_ramb(points: int, pong: int) -> int:
        return _ramb36_for(scan_bits, points) + _ramb36_for(scan_bits, pong)

    ctrl_ramb = 1
    candidate_edges = [e for e in (max_edges_cap, 1024, 512, 256) if e <= max_edges_cap] or [256]
    chosen_edges = candidate_edges[-1]
    chosen_points = 0
    chosen_pong = pong_depth
    for edges in candidate_edges:
        edges_pow2 = _pow2_at_least(edges)
        rem = ramb36_budget - ctrl_ramb - edge_ramb(edges_pow2) - _ramb36_for(scan_bits, pong_depth)
        best_points = 0
        for cand in (4096, 2048, 1024, 512, 256):
            if cand > max_points_cap:
                continue
            if _ramb36_for(scan_bits, _pow2_at_least(cand)) <= rem:
                best_points = cand
                break
        if best_points >= 256:
            chosen_edges = edges
            chosen_points = best_points
            break
    if chosen_points == 0:
        chosen_pong = 0
        chosen_edges = candidate_edges[-1]
        rem = ramb36_budget - ctrl_ramb - edge_ramb(_pow2_at_least(chosen_edges))
        chosen_points = 256 if _ramb36_for(scan_bits, 256) <= rem else 0

    max_edges = _pow2_at_least(chosen_edges)
    max_points = _pow2_at_least(chosen_points) if chosen_points else 0

    params = EdgeTableImageParams(
        channel_count=channel_count,
        num_slots=num_slots,
        coeff_width=coeff_width,
        tick_width=tick_width,
        coeff_frac_bits=coeff_frac_bits,
        max_edges=max_edges,
        max_scan_points=max(1, max_points),
        bus_count=bus_count,
        bus_width=bus_width,
        bus_seg_addr_width=bus_seg_addr_width,
        bus_sel_width=bus_sel_width,
    )

    ramb36_used = ctrl_ramb + edge_ramb(max_edges) + scan_ramb(max_points, chosen_pong)
    image_bram_depth = _pow2_at_least(params.total_words)

    def _res(used: int, total: int) -> dict:
        budget = int(total * pct / 100.0)
        return {"used": int(used), "budget": budget, "total": int(total),
                "pct": round(100.0 * used / total, 1) if total else 0.0,
                "ok": used <= budget}

    resource_report = {
        "ramb36": _res(ramb36_used, prof.ramb36),
        "lut": _res(engine_logic_luts + lutram_luts, prof.lut),
        "ff": _res(engine_ff, prof.ff),
        "dsp": _res(engine_dsp, prof.dsp),
        "distributed_ram_luts": _res(lutram_luts, prof.distributed_ram_kib * 1024 // 64),
    }

    return SolvedCapacity(
        part=prof.name,
        params=params,
        edge_addr_width=_addr_width(max_edges),
        scan_addr_width=_addr_width(max_points) if max_points else 1,
        pong_depth=chosen_pong,
        image_bram_depth=image_bram_depth,
        ramb36_used=ramb36_used,
        ramb36_budget=ramb36_budget,
        lutram_luts=lutram_luts,
        resource_report=resource_report,
    )


# ===========================================================================
# Architecture-D image (for zlc_pulse_streamer_d_top.v): edge/scan tables live
# in their own asymmetric BRAMs (the engine reads a whole edge / scan point per
# port-B access), so the AXI write layout differs from the loader image:
#   - CTRL regfile region (scalars at the offsets below) at word 0.
#   - EDGE region: 8 words/edge (256-bit port B; low 158 bits = {mask,coeffs,tick}).
#   - SCAN region: NUM_SLOTS words/point.
#   - BUS region: 7 words/segment (bus-major), copied into the engine's bus LUTRAM
#     by the top's mini-loader.
# Region word bases match the top's localparams; geometry comes from an
# EdgeTableImageParams (with the solved max_edges/max_scan_points), so a part/XDC
# swap re-derives it.
# ===========================================================================

D_CTRL_WORDS = 64
D_EDGE_PORTB_WORDS = 8     # 256-bit edge row / 32


class DCtrl:
    COMMAND = 1
    STATUS = 2
    PROG_COUNT = 3
    SCAN_COUNT = 4
    SCAN_ENABLE = 5
    REPEAT_FOREVER = 6
    LOOP_START = 7
    LOOP_COUNT = 8
    LOOP_END_TICK = 9
    LOOP_END_LO = 10
    LOOP_END_HI = 11
    BUS_COUNTS = 12


D_CMD_LOAD = 1 << 0
D_CMD_FIRE = 1 << 1
D_CMD_RESET = 1 << 2
D_CMD_SAFE = 1 << 3
D_STATUS_LOADED = 1 << 0
D_STATUS_RUNNING = 1 << 1
D_STATUS_DONE = 1 << 2


def d_region_bases(params: EdgeTableImageParams) -> dict:
    edge_base = D_CTRL_WORDS
    scan_words = params.num_slots
    scan_base = edge_base + params.max_edges * D_EDGE_PORTB_WORDS
    bus_base = scan_base + params.max_scan_points * scan_words
    bus_words = params.bus_words
    total = bus_base + params.bus_rows * bus_words
    return {"ctrl": 0, "edge": edge_base, "scan": scan_base, "bus": bus_base,
            "scan_words": scan_words, "bus_words": bus_words, "total": total}


def pack_program_d(program, params: EdgeTableImageParams | None = None) -> dict[int, int]:
    """Pack a RuntimeSequenceProgram into the Architecture-D AXI write image
    (sparse ``{word_offset: value}``).  COMMAND/STATUS are runtime mailbox words."""

    params = params or EdgeTableImageParams()
    bases = d_region_bases(params)
    ticks = [int(t) for t in program.ticks]
    masks = [int(m) for m in program.masks]
    n_edges = len(ticks)
    if n_edges > params.max_edges:
        raise ValueError(f"{n_edges} edges > max_edges {params.max_edges}.")
    slot_count = int(getattr(program, "slot_count", 0) or 0)
    tick_slot_coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in ticks])
    scan_points = [list(p) for p in (getattr(program, "scan_points", None) or [])]
    if len(scan_points) > params.max_scan_points:
        raise ValueError(f"{len(scan_points)} scan points > max_scan_points {params.max_scan_points}.")
    bus_segments = list(getattr(program, "bus_segments", None) or [])

    words: dict[int, int] = {}
    words[DCtrl.PROG_COUNT] = n_edges
    words[DCtrl.SCAN_COUNT] = len(scan_points)
    words[DCtrl.SCAN_ENABLE] = 1 if scan_points else 0
    words[DCtrl.REPEAT_FOREVER] = 1 if bool(getattr(program, "repeat_forever", False)) else 0
    words[DCtrl.LOOP_START] = int(getattr(program, "loop_start_index", 0))
    words[DCtrl.LOOP_COUNT] = int(getattr(program, "loop_count", 1) or 1)
    words[DCtrl.LOOP_END_TICK] = _to_unsigned(int(getattr(program, "loop_end_tick", 0)), params.tick_width)
    lo = _pack_field_words(_pack_slot_coeffs(getattr(program, "loop_end_slot_coeffs", None), params), params.coeff_bits)
    words[DCtrl.LOOP_END_LO] = lo[0]
    words[DCtrl.LOOP_END_HI] = lo[1] if len(lo) > 1 else 0

    base = bases["edge"]
    for i in range(n_edges):
        row = base + i * D_EDGE_PORTB_WORDS
        words[row + 0] = _to_unsigned(ticks[i], params.tick_width)
        cv = _pack_field_words(_pack_slot_coeffs(tick_slot_coeffs[i], params), params.coeff_bits)
        words[row + 1] = cv[0]
        words[row + 2] = cv[1] if len(cv) > 1 else 0
        mv = _pack_field_words(masks[i] & ((1 << params.channel_count) - 1), params.channel_count)
        words[row + 3] = mv[0]
        words[row + 4] = mv[1] if len(mv) > 1 else 0

    base = bases["scan"]
    sw = bases["scan_words"]
    for p in range(len(scan_points)):
        row = base + p * sw
        for j in range(params.num_slots):
            val = scan_points[p][j] if j < slot_count else 0
            words[row + j] = _to_unsigned(val, params.tick_width)

    per_bus: list[list[object]] = [[] for _ in range(params.bus_count)]
    for seg in bus_segments:
        per_bus[int(getattr(seg, "bus_index", 0))].append(seg)
    cnt_width = params.bus_seg_addr_width + 1
    bus_counts_packed = 0
    base = bases["bus"]
    bw = bases["bus_words"]
    for b in range(params.bus_count):
        segs = per_bus[b]
        bus_counts_packed |= (len(segs) & ((1 << cnt_width) - 1)) << (b * cnt_width)
        for addr, seg in enumerate(segs):
            row = base + (b * params.max_bus_segments + addr) * bw
            words[row + 0] = _to_unsigned(int(getattr(seg, "start_tick", 0)), params.tick_width)
            words[row + 1] = _to_unsigned(int(getattr(seg, "stop_tick", 0)), params.tick_width)
            sc = _pack_field_words(_pack_slot_coeffs(getattr(seg, "start_tick_coeffs", None), params), params.coeff_bits)
            ec = _pack_field_words(_pack_slot_coeffs(getattr(seg, "stop_tick_coeffs", None), params), params.coeff_bits)
            words[row + 2] = sc[0]
            words[row + 3] = sc[1] if len(sc) > 1 else 0
            words[row + 4] = ec[0]
            words[row + 5] = ec[1] if len(ec) > 1 else 0
            flags = 0
            flags |= (int(getattr(seg, "start_value", 0)) & ((1 << params.bus_width) - 1)) << 0
            flags |= (int(getattr(seg, "stop_value", 0)) & ((1 << params.bus_width) - 1)) << params.bus_width
            flags |= (_bus_mode_value(getattr(seg, "mode", "edge")) & 0x3) << (2 * params.bus_width)
            flags |= (int(getattr(seg, "value_select", 0)) & ((1 << params.bus_sel_width) - 1)) << (2 * params.bus_width + 2)
            words[row + 6] = flags
    words[DCtrl.BUS_COUNTS] = bus_counts_packed
    return words


def unpack_program_d(words: Mapping[int, int], params: EdgeTableImageParams | None = None) -> dict:
    """Reconstruct program fields from a D image (the host->BRAM contract)."""

    params = params or EdgeTableImageParams()
    bases = d_region_bases(params)

    def w(o):
        return int(words.get(o, 0)) & 0xFFFFFFFF

    n_edges = w(DCtrl.PROG_COUNT)
    n_points = w(DCtrl.SCAN_COUNT)
    ticks, masks, coeffs = [], [], []
    base = bases["edge"]
    for i in range(n_edges):
        row = base + i * D_EDGE_PORTB_WORDS
        ticks.append(w(row + 0))
        coeffs.append(_unpack_slot_coeffs(_unpack_field_words([w(row + 1), w(row + 2)], params.coeff_bits), params))
        masks.append(_unpack_field_words([w(row + 3), w(row + 4)], params.channel_count))
    scan_points = []
    base = bases["scan"]
    sw = bases["scan_words"]
    for p in range(n_points):
        row = base + p * sw
        scan_points.append([_from_unsigned(w(row + j), params.tick_width) for j in range(params.num_slots)])
    cnt_width = params.bus_seg_addr_width + 1
    bus_counts_packed = w(DCtrl.BUS_COUNTS)
    bus_segments = []
    base = bases["bus"]
    bw = bases["bus_words"]
    for b in range(params.bus_count):
        count = (bus_counts_packed >> (b * cnt_width)) & ((1 << cnt_width) - 1)
        for addr in range(count):
            row = base + (b * params.max_bus_segments + addr) * bw
            flags = w(row + 6)
            bus_segments.append({
                "bus_index": b,
                "start_tick": w(row + 0), "stop_tick": w(row + 1),
                "start_tick_coeffs": _unpack_slot_coeffs(_unpack_field_words([w(row + 2), w(row + 3)], params.coeff_bits), params),
                "stop_tick_coeffs": _unpack_slot_coeffs(_unpack_field_words([w(row + 4), w(row + 5)], params.coeff_bits), params),
                "start_value": flags & ((1 << params.bus_width) - 1),
                "stop_value": (flags >> params.bus_width) & ((1 << params.bus_width) - 1),
                "mode": _bus_mode_name((flags >> (2 * params.bus_width)) & 0x3),
                "value_select": (flags >> (2 * params.bus_width + 2)) & ((1 << params.bus_sel_width) - 1),
            })
    return {
        "ticks": ticks, "masks": masks, "tick_slot_coeffs": coeffs,
        "scan_points": scan_points,
        "repeat_forever": bool(w(DCtrl.REPEAT_FOREVER) & 1),
        "loop_start_index": w(DCtrl.LOOP_START), "loop_count": w(DCtrl.LOOP_COUNT),
        "loop_end_tick": w(DCtrl.LOOP_END_TICK),
        "loop_end_slot_coeffs": _unpack_slot_coeffs(_unpack_field_words([w(DCtrl.LOOP_END_LO), w(DCtrl.LOOP_END_HI)], params.coeff_bits), params),
        "bus_segments": bus_segments,
    }
