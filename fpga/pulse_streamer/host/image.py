"""BRAM program image + capacity solver for the FINAL edge-table streamer
(``fpga/pulse_streamer/zlc_edge_streamer.v`` + ``zlc_pulse_streamer_top.v``).

Single source of truth for the host<->FPGA AXI write contract AND the geometry
the RTL localparams + create-project tcl derive from.  A different XDC (channel
count) or FPGA part re-derives everything via :func:`solve_capacity`.

Memory map (32-bit AXI words; the top decodes one axi_bram_ctrl port to regions):
  CTRL   regfile: scalars + COMMAND/STATUS mailbox + streaming cursor/bank_ready
  TICK   edge base ticks      (1 word/edge,  engine reads 32b)
  COEFF  edge slot coeffs     (2 words/edge, engine reads 64b)
  MASK   edge 62-bit masks    (2 words/edge, engine reads 64b: low 62 used)
  SCAN   2-bank ping-pong window: 2 * bank_size points * num_slots words
  BUS    bus-segment image    (7 words/seg, copied into engine LUTRAM by the top)

The edge fields are separate BRAMs read in PARALLEL (one whole edge per access,
no width padding) so max_edges is large; the scan window is small (2 banks) and
the host streams the rest -> UNBOUNDED scan points.  Bus tables stay LUTRAM.

The per-channel / per-bus OUTPUT delay is EVENT-SCHEDULED (queued toggles against a
free-running counter, 32-bit delay bound) for BOTH the TTL channels and the DAC buses; both
live as one 32-bit word per signal in the R_DELAY register region -- no dense delay CTRL words.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields as _dataclass_fields
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "StreamerParams", "CtrlWords", "FpgaPartProfile", "FPGA_PARTS", "part_profile",
    "SolvedCapacity", "solve_capacity", "estimate_resources",
    "pack_program", "unpack_program", "scan_bank_words", "region_bases", "check_rtl_assumptions",
    "CMD_LOAD", "CMD_FIRE", "CMD_RESET", "CMD_SAFE",
    "STATUS_LOADED", "STATUS_RUNNING", "STATUS_DONE", "STATUS_ERROR", "STATUS_UNDERFLOW",
    "IMAGE_MAGIC",
    "DEFAULT_CONFIG_PATH", "load_streamer_config", "params_from_config", "default_params",
    "default_part", "default_target_pct", "default_clock_hz",
    "check_config_capacity", "format_capacity_report",
]

IMAGE_MAGIC = 0x5A4C4532   # "ZLE2"

CMD_LOAD = 1 << 0
CMD_FIRE = 1 << 1
CMD_RESET = 1 << 2
CMD_SAFE = 1 << 3

STATUS_LOADED = 1 << 0
STATUS_RUNNING = 1 << 1
STATUS_DONE = 1 << 2
STATUS_ERROR = 1 << 3
STATUS_UNDERFLOW = 1 << 4


class CtrlWords:
    MAGIC = 0
    COMMAND = 1            # host -> top: LOAD/FIRE/RESET/SAFE (rising-edge)
    STATUS = 2            # top -> host: LOADED/RUNNING/DONE/ERROR/UNDERFLOW
    PROG_COUNT = 3        # number of edges
    SCAN_COUNT = 4        # TOTAL scan points N (may exceed the resident window)
    SCAN_ENABLE = 5
    REPEAT_FOREVER = 6
    LOOP_START = 7
    LOOP_COUNT = 8
    LOOP_END_TICK = 9
    LOOP_END_LO = 10
    LOOP_END_HI = 11
    BUS_COUNTS = 12
    BANK_SIZE = 13        # scan points per ping-pong bank
    SLOT_COUNT = 14
    CURSOR = 15           # top -> host: scan points consumed (for streaming refill)
    BANK_READY = 16       # host -> top: bit b = bank b is loaded/ready
    BANK0_CHUNK = 17      # host -> top: sweep-chunk index currently resident in bank 0
    BANK1_CHUNK = 18      # host -> top: sweep-chunk index currently resident in bank 1
    REPEAT_FROM_LOOP_START = 19  # repeat_forever rewinds to LOOP_START; with no channel
    #                              delay this is the whole frame (bracket loop start)
    # PER-CHANNEL CLK MASK -- channels wired to the FPGA clk (a DAC latch strobe).  One bit
    # per channel; the top muxes the strobe onto those pins (out_final[n]=clk_en[n]?~clk:out[n],
    # the INVERTED clk so the DAC latches at its data-eye centre -- see top "DAC LATCH PHASE")
    # and the engine's bit for them is forced 0 so it never fights the clk.  Sits right after the
    # command words (ceil(62/32)=2 words, 20..21) -- there are NO dense delay-tick CTRL words any
    # more (TTL+DAC delays are 32-bit-per-signal in the R_DELAY region).  Locked to
    # zlc_pulse_streamer_top.v by test_final_top_regions_match_image.
    CLK_ENABLE = 20


CTRL_WORDS = 64


@dataclass(frozen=True)
class StreamerParams:
    channel_count: int = 62
    num_slots: int = 4
    coeff_width: int = 16
    tick_width: int = 32
    coeff_frac_bits: int = 8
    max_edges: int = 4096
    bank_size: int = 512          # scan points per bank (2 banks resident)
    bus_count: int = 4
    bus_width: int = 10
    bus_seg_addr_width: int = 6
    bus_sel_width: int = 3
    # EVENT SCHEDULER (TTL channels AND DAC buses): a delay is queued TOGGLES against a free-running
    # global counter, so a TTL delay is bounded only by its 32-bit field
    # (ttl_delay_max_ticks ~ 42.9 s at 20 ns) -- e.g. millisecond emCCD delays.  The
    # constraint moves from delay LENGTH to toggles IN FLIGHT: at most evt_fifo_depth
    # toggles of one channel may fall inside any window of that channel's delay length
    # (validated at compile/pack time; sparse experiment triggers are far below this).
    ttl_delay_max_ticks: int = (1 << 31) - 1
    evt_fifo_depth: int = 128   # keep == streamer_config.json evt_fifo_depth (the synthesized depth)
    # per-DA-bit delay FIFO depth (bus_count*bus_width of them, shallower than TTL to fit LUT).
    bus_evt_fifo_depth: int = 64
    # DELAY register region: one 32b word per delay-eligible signal -- channel_count TTL channels
    # then bus_count per-bus DAC delays (both 32b now: DAC is event-scheduled per bit, not the old
    # 12b ring, so TTL and DAC ranges match).  128 words reserved (>= channel_count + bus_count).
    delay_region_words: int = 128

    @property
    def channel_bit_width(self) -> int:
        return _addr_width(max(2, self.channel_count))   # bits to index an output channel

    @property
    def bus_index_width(self) -> int:
        return _addr_width(max(2, self.bus_count))       # bits to index a DAC bus

    @property
    def clk_enable_words(self) -> int:
        # per-channel clk mask: 1 bit per channel, in 32b words
        return _ceil(self.channel_count, 32)

    @property
    def ctrl_scratch_base(self) -> int:
        """First CTRL word ABOVE every defined/used word -- command words (0..19) then
        CLK_ENABLE (20..21).  ``axi_self_test`` may scribble ONLY at/above this index
        (hardware regression: a stale hard-coded scratch once landed inside the CLK_ENABLE
        words, clk-enabling random channels at server bring-up -- pins ran at 50 MHz)."""
        base = int(CtrlWords.CLK_ENABLE) + self.clk_enable_words
        if base + 2 > CTRL_WORDS:
            raise ValueError(
                f"CTRL register file has no scratch room: defined words reach {base} "
                f"but the file holds only {CTRL_WORDS} words; grow CTRL_WORDS / the RTL "
                "ctrl_reg file in lock-step."
            )
        return base

    @property
    def coeff_bits(self) -> int:
        return self.num_slots * self.coeff_width

    @property
    def slot_bits(self) -> int:
        return self.num_slots * self.tick_width

    @property
    def coeff_words(self) -> int:
        return _ceil(self.coeff_bits, 32)

    @property
    def mask_words(self) -> int:
        return _ceil(self.channel_count, 32)

    @property
    def scan_words(self) -> int:
        return self.num_slots          # one 32-bit slot value per word

    @property
    def max_bus_segments(self) -> int:
        return 1 << self.bus_seg_addr_width

    @property
    def bus_rows(self) -> int:
        return self.bus_count * self.max_bus_segments

    @property
    def bus_words(self) -> int:
        return 2 + 2 * self.coeff_words + 1

    @property
    def edge_addr_width(self) -> int:
        return _addr_width(self.max_edges)

    @property
    def scan_addr_width(self) -> int:
        # addresses 2 banks of bank_size points
        return _addr_width(2 * self.bank_size)


def _ceil(a: int, b: int) -> int:
    return (int(a) + b - 1) // b


def _pow2_at_least(v: int) -> int:
    n = 1
    while n < v:
        n <<= 1
    return n


def _addr_width(depth: int) -> int:
    return max(1, _pow2_at_least(max(1, depth)).bit_length() - 1)


def region_bases(p: StreamerParams) -> dict:
    """Word-address bases of each AXI write region (the host<->top contract).

    TTL channel delays live in their own DELAY register region (one 32-bit word per
    channel, delay_region_words reserved) -- the event scheduler's delays outgrew the
    dense 12-bit CTRL fields, whose words (20..43) are now reserved/unused.  Bus delays
    (ring-capped) and the clk mask stay in CTRL."""
    ctrl = 0
    tick = CTRL_WORDS
    coeff = tick + p.max_edges * 1
    mask = coeff + p.max_edges * p.coeff_words
    scan = mask + p.max_edges * p.mask_words
    bus = scan + 2 * p.bank_size * p.scan_words
    delay = bus + p.bus_rows * p.bus_words
    total = delay + p.delay_region_words
    return {"ctrl": ctrl, "tick": tick, "coeff": coeff, "mask": mask,
            "scan": scan, "bus": bus, "delay": delay, "total": total}


# --------------------------------------------------------------------------- bits
def _to_unsigned(value: int, width: int) -> int:
    return int(value) & ((1 << width) - 1)


def _from_unsigned(value: int, width: int) -> int:
    value &= (1 << width) - 1
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _field_words(value: int, total_bits: int) -> list[int]:
    value &= (1 << total_bits) - 1
    return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(_ceil(total_bits, 32))]


def _unfield(words: Sequence[int], total_bits: int) -> int:
    v = 0
    for i, w in enumerate(words):
        v |= (int(w) & 0xFFFFFFFF) << (32 * i)
    return v & ((1 << total_bits) - 1)


def _pack_coeffs(coeffs, p: StreamerParams) -> int:
    coeffs = list(coeffs or [])
    acc = 0
    for j in range(p.num_slots):
        c = coeffs[j] if j < len(coeffs) else 0
        acc |= _to_unsigned(c, p.coeff_width) << (j * p.coeff_width)
    return acc


def _unpack_coeffs(value: int, p: StreamerParams) -> list[int]:
    return [_from_unsigned((value >> (j * p.coeff_width)) & ((1 << p.coeff_width) - 1), p.coeff_width)
            for j in range(p.num_slots)]


def check_rtl_assumptions(p: StreamerParams) -> None:
    """Reject a geometry the SHIPPED RTL cannot realise (it would synthesize but
    silently corrupt data).  These mirror comment guards in the .v sources:

    * the top's bus-loader assembles the coeff words as exactly TWO 32b caps, so
      ``num_slots * coeff_width`` must be 64 (zlc_pulse_streamer_top.v L_EMIT);
    * the bus flags word packs ``2*bus_width + 2 + 2*bus_sel_width`` bits into ONE
      32b cap (same place);
    * ``scan_addr_of`` concatenates ``{bank_bit, offset}``, so ``bank_size`` must be
      a power of two (zlc_edge_streamer.v scan_addr_of);
    * ``MAX_EDGES = 1 << EDGE_ADDR_WIDTH`` and the tick BRAM port is 32b wide.

    Change the RTL first, then relax the matching check here."""
    if p.num_slots * p.coeff_width != 64:
        raise ValueError(
            f"num_slots*coeff_width must be 64 for the shipped RTL (got {p.num_slots}*{p.coeff_width}="
            f"{p.num_slots * p.coeff_width}); the top's 2-word coeff assembly would truncate. "
            "Fix zlc_pulse_streamer_top.v L_EMIT before changing this geometry.")
    flags_bits = 2 * p.bus_width + 2 + 2 * p.bus_sel_width
    if flags_bits > 32:
        raise ValueError(
            f"bus flags word needs {flags_bits} bits (> 32) at bus_width={p.bus_width}, "
            f"bus_sel_width={p.bus_sel_width}; the top packs it into ONE 32b cap word.")
    if p.bank_size <= 0 or (p.bank_size & (p.bank_size - 1)) != 0:
        raise ValueError(
            f"bank_size must be a power of two (got {p.bank_size}); scan_addr_of concatenates "
            "{bank_bit, offset} and would alias the two banks otherwise.")
    if p.max_edges <= 0 or (p.max_edges & (p.max_edges - 1)) != 0:
        raise ValueError(f"max_edges must be a power of two (got {p.max_edges}); MAX_EDGES = 1 << EDGE_ADDR_WIDTH.")
    if p.tick_width != 32:
        raise ValueError(f"tick_width must be 32 (got {p.tick_width}); the tick BRAM port and CTRL words are 32b.")
    if p.channel_count + p.bus_count > p.delay_region_words:
        raise ValueError(
            f"channel_count {p.channel_count} + bus_count {p.bus_count} exceeds the DELAY register "
            f"region ({p.delay_region_words} words; one 32b delay word per channel, then per bus).")


def _bus_mode_value(mode) -> int:
    m = str(mode).strip().lower()
    return {"edge": 1, "ramp": 2}.get(m, 0) or _raise_mode(m)


def _raise_mode(m):
    raise ValueError(f"unsupported bus segment mode {m!r}.")


def _bus_mode_name(v: int) -> str:
    return {1: "edge", 2: "ramp"}.get(int(v)) or _raise_mode(v)


# --------------------------------------------------------------------------- pack
def scan_bank_words(program, p: StreamerParams, chunk_index: int,
                    target_bank: int | None = None) -> dict[int, int]:
    """Words to (re)load scan chunk ``chunk_index`` into a ping-pong bank.

    Chunk c = scan_points[c*bank_size:(c+1)*bank_size].  By default it lands in bank
    c%2 (the initial preload).  For the CONTINUOUS CYCLIC re-sweep the host streams
    chunks 0,1,..,K-1,0,1,.. into the ALTERNATING bank by MONOTONIC position, so it
    passes ``target_bank = mono % 2`` (which need not equal c%2 across a wrap) -- this
    matches the engine's scan_bank_base parity so the wrap is seamless.  Returns a
    sparse ``{word_offset: value}`` for just that bank.  Empty if the chunk is out of range."""
    bases = region_bases(p)
    points = [list(pt) for pt in (getattr(program, "scan_points", None) or [])]
    slot_count = int(getattr(program, "slot_count", 0) or 0)
    first = chunk_index * p.bank_size
    if first >= len(points):
        return {}
    bank = (chunk_index % 2) if target_bank is None else (int(target_bank) & 1)
    base = bases["scan"] + bank * p.bank_size * p.scan_words
    words: dict[int, int] = {}
    for off in range(p.bank_size):
        idx = first + off
        if idx >= len(points):
            break
        row = base + off * p.scan_words
        for j in range(p.num_slots):
            val = points[idx][j] if j < slot_count else 0
            words[row + j] = _to_unsigned(val, p.tick_width)
    return words


def pack_program(program, params: StreamerParams | None = None) -> dict[int, int]:
    """Pack a RuntimeSequenceProgram into the FINAL AXI write image (sparse).

    Edges -> TICK/COEFF/MASK regions; the first TWO scan chunks -> the two banks
    (the rest are streamed via :func:`scan_bank_words`); bus -> BUS region; scalars
    -> CTRL.  COMMAND/STATUS/CURSOR/BANK_READY are runtime mailbox words."""
    p = params or StreamerParams()
    check_rtl_assumptions(p)   # hard gate: never pack for a geometry the shipped RTL corrupts
    bases = region_bases(p)
    ticks = [int(t) for t in program.ticks]
    masks = [int(m) for m in program.masks]
    n_edges = len(ticks)
    if n_edges > p.max_edges:
        raise ValueError(f"{n_edges} edges > max_edges {p.max_edges}.")
    slot_count = int(getattr(program, "slot_count", 0) or 0)
    coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in ticks])
    points = [list(pt) for pt in (getattr(program, "scan_points", None) or [])]
    bus_segments = list(getattr(program, "bus_segments", None) or [])

    w: dict[int, int] = {}
    w[CtrlWords.MAGIC] = IMAGE_MAGIC
    w[CtrlWords.PROG_COUNT] = n_edges
    w[CtrlWords.SCAN_COUNT] = len(points)
    w[CtrlWords.SCAN_ENABLE] = 1 if points else 0
    w[CtrlWords.REPEAT_FOREVER] = 1 if bool(getattr(program, "repeat_forever", False)) else 0
    # An additive-delay program (repeat_from_index > 0, never with a finite bracket)
    # rewinds repeat_forever to its STEADY frame: point loop_start_addr there and flag
    # it.  A bracket (repeat_from_index == 0) keeps loop_start = the bracket start and
    # rewinds repeat_forever from edge 0.
    repeat_from_index = int(getattr(program, "repeat_from_index", 0) or 0)
    w[CtrlWords.LOOP_START] = repeat_from_index if repeat_from_index > 0 else int(getattr(program, "loop_start_index", 0))
    w[CtrlWords.REPEAT_FROM_LOOP_START] = 1 if repeat_from_index > 0 else 0
    w[CtrlWords.LOOP_COUNT] = int(getattr(program, "loop_count", 1) or 1)
    w[CtrlWords.LOOP_END_TICK] = _to_unsigned(int(getattr(program, "loop_end_tick", 0)), p.tick_width)
    le = _field_words(_pack_coeffs(getattr(program, "loop_end_slot_coeffs", None), p), p.coeff_bits)
    w[CtrlWords.LOOP_END_LO] = le[0]
    w[CtrlWords.LOOP_END_HI] = le[1] if len(le) > 1 else 0
    w[CtrlWords.BANK_SIZE] = p.bank_size
    w[CtrlWords.SLOT_COUNT] = slot_count

    # edge fields
    for i in range(n_edges):
        w[bases["tick"] + i] = _to_unsigned(ticks[i], p.tick_width)
        cw = _field_words(_pack_coeffs(coeffs[i], p), p.coeff_bits)
        for k in range(p.coeff_words):
            w[bases["coeff"] + i * p.coeff_words + k] = cw[k] if k < len(cw) else 0
        mw = _field_words(masks[i] & ((1 << p.channel_count) - 1), p.channel_count)
        for k in range(p.mask_words):
            w[bases["mask"] + i * p.mask_words + k] = mw[k] if k < len(mw) else 0

    # first two scan chunks -> banks 0 and 1; record which chunk each bank holds so
    # the engine's bank_chunk handshake accepts them (host updates these while
    # streaming/re-sweeping).
    for chunk in (0, 1):
        w.update(scan_bank_words(program, p, chunk))
    w[CtrlWords.BANK0_CHUNK] = 0
    w[CtrlWords.BANK1_CHUNK] = 1

    # bus segments (bus-major)
    per_bus: list[list[object]] = [[] for _ in range(p.bus_count)]
    for seg in bus_segments:
        per_bus[int(getattr(seg, "bus_index", 0))].append(seg)
    cnt_w = p.bus_seg_addr_width + 1
    bus_counts = 0
    for b in range(p.bus_count):
        segs = per_bus[b]
        bus_counts |= (len(segs) & ((1 << cnt_w) - 1)) << (b * cnt_w)
        for addr, seg in enumerate(segs):
            row = bases["bus"] + (b * p.max_bus_segments + addr) * p.bus_words
            w[row + 0] = _to_unsigned(int(getattr(seg, "start_tick", 0)), p.tick_width)
            w[row + 1] = _to_unsigned(int(getattr(seg, "stop_tick", 0)), p.tick_width)
            sc = _field_words(_pack_coeffs(getattr(seg, "start_tick_coeffs", None), p), p.coeff_bits)
            ec = _field_words(_pack_coeffs(getattr(seg, "stop_tick_coeffs", None), p), p.coeff_bits)
            for k in range(p.coeff_words):
                w[row + 2 + k] = sc[k] if k < len(sc) else 0
                w[row + 2 + p.coeff_words + k] = ec[k] if k < len(ec) else 0
            flags = 0
            flags |= (int(getattr(seg, "start_value", 0)) & ((1 << p.bus_width) - 1)) << 0
            flags |= (int(getattr(seg, "stop_value", 0)) & ((1 << p.bus_width) - 1)) << p.bus_width
            flags |= (_bus_mode_value(getattr(seg, "mode", "edge")) & 0x3) << (2 * p.bus_width)
            flags |= (int(getattr(seg, "value_select", 0)) & ((1 << p.bus_sel_width) - 1)) << (2 * p.bus_width + 2)
            # stop-endpoint select (bits above start select) lets a ramp scan BOTH
            # value endpoints; edge/hold segments default it to value_select.
            _stop_sel = int(getattr(seg, "stop_value_select", getattr(seg, "value_select", 0)))
            flags |= (_stop_sel & ((1 << p.bus_sel_width) - 1)) << (2 * p.bus_width + 2 + p.bus_sel_width)
            w[row + 2 + 2 * p.coeff_words] = flags
    w[CtrlWords.BUS_COUNTS] = bus_counts

    # PER-CHANNEL TTL OUTPUT DELAY -- the EVENT SCHEDULER.  One 32-bit word per channel in
    # the DELAY register region (0 = passthrough).  A delay is bounded by ttl_delay_max_ticks
    # (32-bit field, ~42.9 s), NOT by the bus-ring depth; pack always writes ALL channel
    # words so stale delays from a previous program can never linger.
    channel_delays = [int(d) for d in (getattr(program, "channel_delays", None) or [])]
    for ch, d in enumerate(channel_delays):
        if ch >= p.channel_count:
            if d:
                raise ValueError(f"channel-delay bit {ch} is outside channel_count {p.channel_count}.")
            continue
        if d < 0 or d > p.ttl_delay_max_ticks:
            raise ValueError(
                f"channel bit {ch} delay {d} ticks is outside [0, {p.ttl_delay_max_ticks}] "
                f"(~{p.ttl_delay_max_ticks * 20e-9:.1f} s at 20 ns/tick).")
    for ch in range(p.channel_count):
        d = channel_delays[ch] if ch < len(channel_delays) else 0
        w[bases["delay"] + ch] = _to_unsigned(d, 32)

    # PER-BUS DAC DELAY -- each DA bit is now its OWN event-scheduler channel (the bus's 10 bits
    # share one 32-bit delay), so a bus delay is 32-bit like TTL and rides the SAME R_DELAY region,
    # one 32b word per bus right after the channels (words channel_count .. channel_count+bus_count-1).
    for bd in (getattr(program, "bus_delays", None) or []):
        if isinstance(bd, Mapping):
            b, d = int(bd.get("bus_index", 0)), int(bd.get("delay", 0))
        else:
            b, d = int(getattr(bd, "bus_index", 0)), int(getattr(bd, "delay", 0))
        if b < 0 or b >= p.bus_count:
            raise ValueError(f"bus delay bus_index {b} is outside bus_count {p.bus_count}.")
        if d < 0 or d > p.ttl_delay_max_ticks:
            raise ValueError(
                f"bus {b} delay {d} ticks is outside [0, {p.ttl_delay_max_ticks}] "
                f"(~{p.ttl_delay_max_ticks * 20e-9:.1f} s at 20 ns/tick).")
        w[bases["delay"] + p.channel_count + b] = _to_unsigned(d, 32)

    # PER-CHANNEL CLK MASK -- 1 bit per channel (bit b = channel b's pin driven by clk).
    # The compiler already forced these bits to 0 in the edge masks; the top muxes clk on.
    clk_enable = int(getattr(program, "clk_enable", 0))
    for i in range((p.channel_count + 31) // 32):
        w[CtrlWords.CLK_ENABLE + i] = (clk_enable >> (32 * i)) & 0xFFFFFFFF
    return w


def unpack_program(words: Mapping[int, int], params: StreamerParams | None = None) -> dict:
    """Reconstruct program fields from a packed image (host<->FPGA contract check).
    Reads only the first two scan chunks (what's resident); streamed chunks are
    validated separately via :func:`scan_bank_words`."""
    p = params or StreamerParams()
    bases = region_bases(p)

    def g(o):
        return int(words.get(o, 0)) & 0xFFFFFFFF

    n_edges = g(CtrlWords.PROG_COUNT)
    n_points = g(CtrlWords.SCAN_COUNT)
    slot_count = g(CtrlWords.SLOT_COUNT)
    ticks, masks, coeffs = [], [], []
    for i in range(n_edges):
        ticks.append(g(bases["tick"] + i))
        coeffs.append(_unpack_coeffs(_unfield([g(bases["coeff"] + i * p.coeff_words + k) for k in range(p.coeff_words)], p.coeff_bits), p))
        masks.append(_unfield([g(bases["mask"] + i * p.mask_words + k) for k in range(p.mask_words)], p.channel_count))
    scan_points = []
    resident = min(n_points, 2 * p.bank_size)
    for idx in range(resident):
        bank = (idx // p.bank_size) % 2
        off = idx % p.bank_size
        row = bases["scan"] + bank * p.bank_size * p.scan_words + off * p.scan_words
        scan_points.append([_from_unsigned(g(row + j), p.tick_width) for j in range(slot_count)])
    cnt_w = p.bus_seg_addr_width + 1
    bus_counts = g(CtrlWords.BUS_COUNTS)
    bus_segments = []
    for b in range(p.bus_count):
        count = (bus_counts >> (b * cnt_w)) & ((1 << cnt_w) - 1)
        for addr in range(count):
            row = bases["bus"] + (b * p.max_bus_segments + addr) * p.bus_words
            flags = g(row + 2 + 2 * p.coeff_words)
            bus_segments.append({
                "bus_index": b, "start_tick": g(row + 0), "stop_tick": g(row + 1),
                "start_tick_coeffs": _unpack_coeffs(_unfield([g(row + 2 + k) for k in range(p.coeff_words)], p.coeff_bits), p),
                "stop_tick_coeffs": _unpack_coeffs(_unfield([g(row + 2 + p.coeff_words + k) for k in range(p.coeff_words)], p.coeff_bits), p),
                "start_value": flags & ((1 << p.bus_width) - 1),
                "stop_value": (flags >> p.bus_width) & ((1 << p.bus_width) - 1),
                "mode": _bus_mode_name((flags >> (2 * p.bus_width)) & 0x3),
                "value_select": (flags >> (2 * p.bus_width + 2)) & ((1 << p.bus_sel_width) - 1),
                "stop_value_select": (flags >> (2 * p.bus_width + 2 + p.bus_sel_width)) & ((1 << p.bus_sel_width) - 1),
            })
    # PER-SIGNAL OUTPUT DELAY -- one 32b R_DELAY word per channel, then one per bus (both
    # event-scheduled, 32b), exactly as zlc_pulse_streamer_top.v slices R_DELAY.
    channel_delays = [int(g(bases["delay"] + ch)) for ch in range(p.channel_count)]
    bus_delays = [{"bus_index": b, "delay": int(g(bases["delay"] + p.channel_count + b))}
                  for b in range(p.bus_count)
                  if int(g(bases["delay"] + p.channel_count + b)) != 0]
    clk_enable = 0
    for i in range((p.channel_count + 31) // 32):
        clk_enable |= (g(CtrlWords.CLK_ENABLE + i) & 0xFFFFFFFF) << (32 * i)
    clk_enable &= (1 << p.channel_count) - 1
    return {
        "ticks": ticks, "masks": masks, "tick_slot_coeffs": coeffs,
        "channel_delays": channel_delays,
        "clk_enable": clk_enable,
        "scan_points_resident": scan_points, "scan_count": n_points, "slot_count": slot_count,
        "repeat_forever": bool(g(CtrlWords.REPEAT_FOREVER) & 1),
        # LOOP_START is the additive-delay steady-frame anchor when the flag is set,
        # else the finite-bracket start.
        "loop_start_index": 0 if (g(CtrlWords.REPEAT_FROM_LOOP_START) & 1) else g(CtrlWords.LOOP_START),
        "repeat_from_index": g(CtrlWords.LOOP_START) if (g(CtrlWords.REPEAT_FROM_LOOP_START) & 1) else 0,
        "loop_count": g(CtrlWords.LOOP_COUNT),
        "loop_end_tick": g(CtrlWords.LOOP_END_TICK),
        "loop_end_slot_coeffs": _unpack_coeffs(_unfield([g(CtrlWords.LOOP_END_LO), g(CtrlWords.LOOP_END_HI)], p.coeff_bits), p),
        "bus_segments": bus_segments, "bank_size": g(CtrlWords.BANK_SIZE),
        "bus_delays": bus_delays,
    }


# --------------------------------------------------------------------- capacity
@dataclass(frozen=True)
class FpgaPartProfile:
    name: str
    ramb36: int
    lut: int
    ff: int
    dsp: int
    distributed_ram_kib: int


FPGA_PARTS: dict[str, FpgaPartProfile] = {
    "xc7a35t": FpgaPartProfile("xc7a35t", 50, 20800, 41600, 90, 400),
    "xc7a50t": FpgaPartProfile("xc7a50t", 75, 32600, 65200, 120, 600),
    "xc7a75t": FpgaPartProfile("xc7a75t", 105, 47200, 94400, 180, 892),
    "xc7a100t": FpgaPartProfile("xc7a100t", 135, 63400, 126800, 240, 1188),
    "xc7a200t": FpgaPartProfile("xc7a200t", 365, 134600, 269200, 740, 2888),
}


def part_profile(part) -> FpgaPartProfile:
    if isinstance(part, FpgaPartProfile):
        return part
    key = str(part).strip().lower()
    for name in sorted(FPGA_PARTS, key=len, reverse=True):
        if key.startswith(name):
            return FPGA_PARTS[name]
    raise KeyError(f"unknown FPGA part {part!r}; add it to FPGA_PARTS.")


@dataclass(frozen=True)
class SolvedCapacity:
    part: str
    params: StreamerParams
    ramb36_used: int
    ramb36_budget: int
    resource_report: dict

    def all_within_budget(self) -> bool:
        return all(r["ok"] for r in self.resource_report.values())


def _edge_ramb(max_edges: int, p: StreamerParams) -> int:
    # 3 parallel edge BRAMs: tick 32b, coeff coeff_bits, mask channel_count
    return (_ceil(p.tick_width, 36) * _ceil(max_edges, 1024)
            + _ceil(p.coeff_bits, 36) * _ceil(max_edges, 1024)
            + _ceil(p.channel_count, 36) * _ceil(max_edges, 1024))


def _scan_ramb(bank_size: int, p: StreamerParams) -> int:
    return _ceil(p.slot_bits, 36) * _ceil(2 * bank_size, 1024)


def estimate_resources(params: StreamerParams, *, part, target_pct: float = 90.0,
                       slot_mul_width: int = 25, engine_logic_luts: int = 14405,
                       engine_ff: int = 9000, engine_dsp: int | None = None) -> dict:
    """Resource usage of a CONCRETE ``StreamerParams`` vs a part, per axis.

    This is the single accounting model shared by :func:`solve_capacity` (which
    searches for the largest ``max_edges`` that fits) and the config-check CLI
    (which reports whether the configured geometry fits as-is).  Returns
    ``{"ramb36"|"lut"|"ff"|"dsp": {"used","budget","total","pct","ok"}}``.

    LUT is CALIBRATED to a REAL Vivado 2019.1 SYNTH+PLACE of the current 35T build
    (2026-06-11): the placer needed 21933 slice LUTs at evt_fifo_depth=256 /
    bus_evt_fifo_depth=64 (the build that OVERFLOWED 20800).  ``engine_logic_luts``
    (=14405) is the fixed, non-depth-scaled remainder (control logic + edge/scan/DSP
    glue) once the bus-segment LUTRAM and the two event-FIFO terms (ttl_sched + dac_evt,
    which DO scale with evt_fifo_depth / bus_evt_fifo_depth) are subtracted -- so the
    model now reproduces 21933 at (256/64) and predicts other depths honestly (the old
    8000 base under-read the real placement by ~40%).  The 40 DA bits cost ~2.2x per
    depth-tick vs the 18 TTL channels, so DEEPENING DA is far pricier than deepening TTL.
    FF/DSP/RAMB36 are estimates; edge fields are parallel BRAMs and the event FIFOs are
    distributed RAM (LUTs in SLICEM, no RAMB36)."""
    prof = part_profile(part)
    pct = max(1.0, min(100.0, float(target_pct)))
    ramb36_used = (_edge_ramb(params.max_edges, params) + _scan_ramb(params.bank_size, params)
                   + _ceil(params.bus_rows * params.bus_words, 1024) + 1)
    # per bus-segment row: start+stop tick (2*tick_width), start+stop tick coeffs
    # (2*coeff_bits), start+stop value (2*bus_width), mode (2), start+stop value_select
    # (2*bus_sel_width -- a ramp can scan both endpoints).
    bus_lutram = _ceil((2 * params.tick_width + 2 * params.coeff_bits + 2 * params.bus_width
                        + 2 + 2 * params.bus_sel_width) * params.bus_rows, 64)
    # TTL EVENT SCHEDULER: an EVT_DEPTH x 49b LUTRAM event FIFO (~ceil(EVT_DEPTH*49/64)
    # RAM LUTs), a 48b equality comparator (~14) and push/pop control (~6) per channel.
    # The FIFOs are COMPACTED to the channels that can carry a delay -- only channels
    # whose engine bit drives a pin, i.e. NOT the bus-member bits (their pin is driven by
    # bus_out, their `out` bit is always 0).  At deep EVT_DEPTH this is what keeps the
    # event RAM inside the 400 Kb distributed-RAM budget (every channel would not fit).
    evt_depth = max(1, int(getattr(params, "evt_fifo_depth", 256)))
    bus_evt_depth = max(1, int(getattr(params, "bus_evt_fifo_depth", 64)))
    # Delay-eligible channels = real TTL outputs: not bus-member bits (bus_count*bus_width,
    # pin driven by bus_out) and not the per-bus dedicated clk pins (bus_count, da_clk*).
    num_delay_ch = max(0, params.channel_count - params.bus_count * (params.bus_width + 1))
    # Each slot's FIFO is a SIMPLE-DUAL-PORT distributed RAM (sync write @wr, async read @rd
    # at an INDEPENDENT address), instantiated once per slot in the g_evtfifo generate loop.
    # It MUST be RAM, not a flat 3D reg array: a 3D array with per-slot independent pointers
    # does NOT infer as distributed RAM (Vivado falls back to registers -> 226k FF at depth
    # 256, which does not fit).  7-series packs SDP LUTRAM at ~0.7-1.0 LUT per 64x1 cell, so
    # ceil(EVT_DEPTH*49/64) LUTs per slot plus ~20 LUTs of pointer/comparator control is an
    # honest, slightly-conservative estimate.
    ttl_sched_luts = num_delay_ch * (20 + _ceil(evt_depth * 49, 64))
    # DAC delay is event-scheduled PER DA BIT (bus_count*bus_width 1-bit channels), each its own
    # BUS_EVT_DEPTH-deep 49b FIFO exactly like a TTL channel (the bus's bits share one delay), so
    # TTL and DAC delay use the same 32-bit range and the same mechanism -- a negative-delay global
    # shift G reaches the buses with no range mismatch.
    dac_evt_luts = (params.bus_count * params.bus_width) * (20 + _ceil(bus_evt_depth * 49, 64))
    delay_lutram = ttl_sched_luts + dac_evt_luts
    # DSP: engine affine-MAC call sites (2 evals/bus + 5 main) x num_slots products,
    # each coeff(<=18b) x slot(slot_mul_width); slot operand <=25b fits ONE DSP48E1.
    if engine_dsp is None:
        mac_instances = 2 * params.bus_count + 5
        dsp_per_mult = 1 if slot_mul_width <= 25 else 2
        engine_dsp = mac_instances * params.num_slots * dsp_per_mult

    def res(used, total):
        b = int(total * pct / 100.0)
        return {"used": int(used), "budget": b, "total": int(total),
                "pct": round(100.0 * used / total, 1) if total else 0.0, "ok": used <= b}

    return {
        "ramb36": res(ramb36_used, prof.ramb36),
        "lut": res(engine_logic_luts + bus_lutram + delay_lutram, prof.lut),
        "ff": res(engine_ff, prof.ff),
        "dsp": res(engine_dsp, prof.dsp),
    }


def solve_capacity(part, *, channel_count: int = 62, num_slots: int = 4, coeff_width: int = 16,
                   tick_width: int = 32, coeff_frac_bits: int = 8, bus_count: int = 4,
                   bus_width: int = 10, bus_seg_addr_width: int = 6, bus_sel_width: int = 3,
                   slot_mul_width: int = 25,
                   target_pct: float = 90.0, bank_size: int = 512,
                   max_edges_cap: int = 16384,
                   engine_logic_luts: int = 14405, engine_ff: int = 9000, engine_dsp: int | None = None) -> SolvedCapacity:
    """Maximise max_edges under <=target_pct of the part's RAMB36 (edges are the
    bounded resource; scan points are UNBOUNDED via streaming, so only the 2-bank
    window costs BRAM).  Edge fields are parallel BRAMs (no width padding).

    LUT/FF/DSP estimates are CALIBRATED to a real Vivado 2019.1 place+route of the
    35T build (zlc_pulse_streamer_top): 7376 slice LUTs (35%), 8059 FF (19%), 52
    DSP (58%), 40 RAMB36 (80%).  The defaults below sit just above those with margin
    so the contract test catches a regression that would push any axis past 90%."""
    prof = part_profile(part)
    pct = max(1.0, min(100.0, float(target_pct)))
    budget = int(prof.ramb36 * pct / 100.0)
    base = StreamerParams(channel_count=channel_count, num_slots=num_slots, coeff_width=coeff_width,
                          tick_width=tick_width, coeff_frac_bits=coeff_frac_bits, max_edges=256,
                          bank_size=bank_size, bus_count=bus_count, bus_width=bus_width,
                          bus_seg_addr_width=bus_seg_addr_width, bus_sel_width=bus_sel_width)
    # bus image is a small 32b BRAM (bus_rows*bus_words words); bus tables themselves
    # live in engine LUTRAM (counted under distributed RAM / LUT, not RAMB36).  The OUTPUT delay
    # event scheduler is per-channel / per-DA-bit distributed-RAM event FIFOs (NO BRAM image --
    # ram_style="distributed"), so it costs LUTs, not RAMB36.
    bus_img_ram = _ceil(base.bus_rows * base.bus_words, 1024)
    scan_ram = _scan_ramb(bank_size, base)
    ctrl_ram = 1
    fixed = scan_ram + bus_img_ram + ctrl_ram
    # largest pow2 max_edges whose edge BRAM fits the remaining budget
    max_edges = 256
    for cand in (16384, 8192, 4096, 2048, 1024, 512, 256):
        if cand > max_edges_cap:
            continue
        if _edge_ramb(cand, base) + fixed <= budget:
            max_edges = cand
            break
    # spend leftover RAMB36 on a BIGGER resident scan window (fewer host refills /
    # lower underflow risk); scan points stay unbounded via streaming regardless.
    chosen_bank = bank_size
    for cand in (8192, 4096, 2048, 1024, bank_size):
        if cand < bank_size:
            continue
        if _edge_ramb(max_edges, base) + _scan_ramb(cand, base) + bus_img_ram + ctrl_ram <= budget:
            chosen_bank = cand
            break
    bank_size = chosen_bank
    params = StreamerParams(channel_count=channel_count, num_slots=num_slots, coeff_width=coeff_width,
                            tick_width=tick_width, coeff_frac_bits=coeff_frac_bits, max_edges=max_edges,
                            bank_size=bank_size, bus_count=bus_count, bus_width=bus_width,
                            bus_seg_addr_width=bus_seg_addr_width, bus_sel_width=bus_sel_width)
    # Single accounting model (shared with the config-check CLI).  The LITERAL delay line
    # is distributed RAM (LUTs, no RAMB36); DSP is the engine affine-MAC sites.
    report = estimate_resources(params, part=prof, target_pct=pct, slot_mul_width=slot_mul_width,
                                engine_logic_luts=engine_logic_luts, engine_ff=engine_ff,
                                engine_dsp=engine_dsp)
    ramb36_used = report["ramb36"]["used"]
    return SolvedCapacity(part=prof.name, params=params, ramb36_used=ramb36_used,
                          ramb36_budget=budget, resource_report=report)


# --------------------------------------------------------------- config file
# Single user-editable source of truth for the reconfigurable, compile-affecting
# specifics (geometry + part + clock).  The host runtime defaults, the program
# validator, and the resource estimator all read this -- edit the JSON, never the
# scattered DEFAULT_* literals.  See fpga/board_config/streamer_config.json.
DEFAULT_CONFIG_FILENAME = "streamer_config.json"
DEFAULT_FPGA_PART = "xc7a35tfgg484-2"
DEFAULT_TARGET_PCT = 90.0
DEFAULT_CLOCK_HZ = 50_000_000.0
DEFAULT_SLOT_MUL_WIDTH = 25

# StreamerParams constructor field names (so config["params"] can carry extra keys
# like slot_mul_width without breaking the dataclass).
_PARAM_FIELD_NAMES = tuple(f.name for f in _dataclass_fields(StreamerParams))


def _config_search_paths() -> list[Path]:
    rel = Path("fpga") / "board_config" / DEFAULT_CONFIG_FILENAME
    paths: list[Path] = []
    env = os.environ.get("ZLC_PS_CONFIG")
    if env and env.strip():
        paths.append(Path(env))
    paths.append(Path.cwd() / rel)
    # image.py is fpga/pulse_streamer/host/image.py -> parents[2] == fpga/.
    paths.append(Path(__file__).resolve().parents[2] / "board_config" / DEFAULT_CONFIG_FILENAME)
    return paths


def _default_config_path() -> Path:
    """The canonical config path (parents[2]==fpga/), used for messages/round-trips."""
    return Path(__file__).resolve().parents[2] / "board_config" / DEFAULT_CONFIG_FILENAME


DEFAULT_CONFIG_PATH = _default_config_path()


def params_from_config(params_map: Mapping | None) -> StreamerParams:
    """Build a :class:`StreamerParams` from a config ``params`` mapping.

    Only known dataclass fields are forwarded; extra keys (``slot_mul_width``,
    underscore comment keys) are ignored, so the JSON can hold estimator-only knobs
    alongside the geometry."""
    kwargs = {k: v for k, v in dict(params_map or {}).items() if k in _PARAM_FIELD_NAMES}
    return StreamerParams(**kwargs)


def load_streamer_config(path: str | Path | None = None) -> dict:
    """Load the single streamer config file.

    Returns a normalized dict: ``{"params": StreamerParams, "fpga_part", "clock_hz",
    "target_pct", "slot_mul_width", "source": Path|None, "warnings": [...]}``.  Missing
    file or unreadable JSON falls back to built-in defaults (so offline/GUI workflows
    never crash) and records a warning -- the estimator CLI surfaces these."""
    warnings: list[str] = []
    raw: dict = {}
    source: Path | None = None
    candidates = [Path(path)] if path is not None and str(path).strip() else _config_search_paths()
    for candidate in candidates:
        try:
            if candidate.exists():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                source = candidate
                break
        except (OSError, ValueError) as exc:
            warnings.append(f"could not read config {candidate}: {exc}")
    if source is None:
        warnings.append("no streamer_config.json found; using built-in defaults.")
    if not isinstance(raw, dict):
        warnings.append("config root is not an object; using built-in defaults.")
        raw = {}
    params_map = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    try:
        params = params_from_config(params_map)
    except (TypeError, ValueError) as exc:
        warnings.append(f"invalid params in config ({exc}); using built-in defaults.")
        params = StreamerParams()
    slot_mul = params_map.get("slot_mul_width", DEFAULT_SLOT_MUL_WIDTH)
    try:
        slot_mul = int(slot_mul)
    except (TypeError, ValueError):
        slot_mul = DEFAULT_SLOT_MUL_WIDTH
    # Surface (don't fail) RTL-assumption violations at load time -- estimation should
    # still answer, but pack_program will hard-reject the same geometry before upload.
    try:
        check_rtl_assumptions(params)
    except ValueError as exc:
        warnings.append(f"geometry violates a shipped-RTL assumption: {exc}")
    return {
        "params": params,
        "fpga_part": str(raw.get("fpga_part", DEFAULT_FPGA_PART)),
        "clock_hz": float(raw.get("clock_hz", DEFAULT_CLOCK_HZ)),
        "target_pct": float(raw.get("target_pct", DEFAULT_TARGET_PCT)),
        "slot_mul_width": slot_mul,
        "source": source,
        "warnings": warnings,
    }


def default_params(path: str | Path | None = None) -> StreamerParams:
    """The configured runtime geometry (config-driven, defaults if the file is absent)."""
    return load_streamer_config(path)["params"]


def default_part(path: str | Path | None = None) -> str:
    return load_streamer_config(path)["fpga_part"]


def default_target_pct(path: str | Path | None = None) -> float:
    return load_streamer_config(path)["target_pct"]


def default_clock_hz(path: str | Path | None = None) -> float:
    return load_streamer_config(path)["clock_hz"]


def check_config_capacity(path: str | Path | None = None) -> dict:
    """Estimate whether the configured part has enough resources for the configured
    geometry.  Returns ``{config, params, part, target_pct, report, ok, warnings}``."""
    cfg = load_streamer_config(path)
    params = cfg["params"]
    report = estimate_resources(params, part=cfg["fpga_part"], target_pct=cfg["target_pct"],
                                slot_mul_width=cfg["slot_mul_width"])
    return {
        "config": cfg,
        "params": params,
        "part": part_profile(cfg["fpga_part"]).name,
        "part_string": cfg["fpga_part"],
        "target_pct": cfg["target_pct"],
        "report": report,
        "ok": all(axis["ok"] for axis in report.values()),
        "warnings": cfg["warnings"],
    }


def format_capacity_report(result: dict) -> str:
    """Human-readable pass/fail table for :func:`check_config_capacity`."""
    cfg = result["config"]
    p: StreamerParams = result["params"]
    report = result["report"]
    src = cfg["source"]
    lines = [
        "ZLC pulse-streamer resource estimate",
        f"  config:     {src if src else '(built-in defaults -- no streamer_config.json found)'}",
        f"  part:       {result['part_string']}  (profile {result['part']})",
        f"  target:     {result['target_pct']:g}% of each resource",
        f"  geometry:   channels={p.channel_count} edges={p.max_edges} bank_size={p.bank_size} "
        f"slots={p.num_slots} buses={p.bus_count}x{p.bus_width}b "
        f"evt_fifo={p.evt_fifo_depth} bus_evt_fifo={p.bus_evt_fifo_depth}",
        "",
        f"  {'resource':<8} {'used':>8} {'budget':>8} {'total':>8}  {'%use':>6}  verdict",
    ]
    label = {"ramb36": "RAMB36", "lut": "LUT", "ff": "FF", "dsp": "DSP"}
    for key in ("lut", "ff", "dsp", "ramb36"):
        a = report[key]
        verdict = "OK" if a["ok"] else "OVER BUDGET"
        lines.append(f"  {label[key]:<8} {a['used']:>8} {a['budget']:>8} {a['total']:>8}  "
                     f"{a['pct']:>5.1f}%  {verdict}")
    lines.append("")
    if result["ok"]:
        lines.append(f"  RESULT: the {result['part_string']} HAS enough resources for this configuration "
                     f"(every axis within {result['target_pct']:g}%).")
    else:
        over = [label[k] for k in ("lut", "ff", "dsp", "ramb36") if not report[k]["ok"]]
        lines.append(f"  RESULT: INSUFFICIENT -- {', '.join(over)} exceed {result['target_pct']:g}% on "
                     f"{result['part_string']}.  Reduce the geometry in {DEFAULT_CONFIG_FILENAME} "
                     f"or choose a larger part (see FPGA_PARTS).")
    for w in result.get("warnings", []):
        lines.append(f"  note: {w}")
    lines.append("")
    lines.append("  final evidence: Vivado report_utilization after synthesis; this is a design-budget estimate.")
    return "\n".join(lines)


def vivado_generics(params: "StreamerParams") -> "list[str]":
    """Verilog top-module parameter overrides that make the SYNTHESIZED bitstream match
    streamer_config.json.  Only the freely-resizable DEPTHS are emitted: the pin-map-coupled
    geometry (channel_count, num_slots/coeff_width with COEFF_BITS==64, bus_count/bus_width)
    is locked to the hand-written board pinout + DELAY_CH_MAP in the top, so changing it needs
    an RTL + XDC edit, not just a generic -- a JSON-vs-.v-default contract test guards those."""
    return [
        f"EDGE_ADDR_WIDTH={params.edge_addr_width}",
        f"BANK_SIZE={params.bank_size}",
        f"EVT_FIFO_DEPTH={params.evt_fifo_depth}",
        f"BUS_EVT_FIFO_DEPTH={params.bus_evt_fifo_depth}",
    ]


def emit_geom_tcl(params: "StreamerParams") -> str:
    """A tiny Tcl snippet that create_project.tcl sources so the BRAM-IP sizing variables AND
    the top-module generics come from streamer_config.json (one source of truth).  When the
    env var pointing here is unset, create_project.tcl falls back to its in-file literals, so
    the current build is byte-identical -- this only takes effect for a config that differs."""
    generics = " ".join(vivado_generics(params))
    return (
        "# AUTO-GENERATED from streamer_config.json by image.emit_geom_tcl -- do not edit.\n"
        "# Sets the BRAM-IP sizing vars + the top-module -generic overrides for synth.\n"
        f"set zlc_edge_addr_width {params.edge_addr_width}\n"
        f"set zlc_bank_size {params.bank_size}\n"
        f"set zlc_evt_fifo_depth {params.evt_fifo_depth}\n"
        f"set zlc_top_generics {{{generics}}}\n"
    )


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m fpga.pulse_streamer.host.image",
        description="Estimate whether the configured FPGA part has enough resources for the "
                    "configured pulse-streamer geometry (reads fpga/board_config/streamer_config.json).",
    )
    parser.add_argument("--config", default=None, help="Path to streamer_config.json (default: auto-detect).")
    parser.add_argument("--part", default=None, help="Override fpga_part for this report only.")
    parser.add_argument("--emit-geom-tcl", default=None, metavar="PATH",
                        help="Write the Vivado geometry/generics Tcl (derived from the config) to "
                             "PATH and exit -- create_project.tcl sources it so the bitstream geometry "
                             "(EVT_FIFO_DEPTH, BUS_EVT_FIFO_DEPTH, EDGE_ADDR_WIDTH, BANK_SIZE) follows the config.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.emit_geom_tcl:
        import pathlib
        params = default_params(args.config)
        pathlib.Path(args.emit_geom_tcl).write_text(emit_geom_tcl(params), encoding="utf-8")
        print(f"wrote geometry tcl -> {args.emit_geom_tcl}")
        return 0
    result = check_config_capacity(args.config)
    if args.part:
        # Re-estimate against an override part without editing the file.
        cfg = result["config"]
        report = estimate_resources(cfg["params"], part=args.part, target_pct=cfg["target_pct"],
                                    slot_mul_width=cfg["slot_mul_width"])
        result = {**result, "part": part_profile(args.part).name, "part_string": args.part,
                  "report": report, "ok": all(a["ok"] for a in report.values())}
    print(format_capacity_report(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
