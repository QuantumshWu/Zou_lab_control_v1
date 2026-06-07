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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "StreamerParams", "CtrlWords", "FpgaPartProfile", "FPGA_PARTS", "part_profile",
    "SolvedCapacity", "solve_capacity",
    "pack_program", "unpack_program", "scan_bank_words", "region_bases",
    "CMD_LOAD", "CMD_FIRE", "CMD_RESET", "CMD_SAFE",
    "STATUS_LOADED", "STATUS_RUNNING", "STATUS_DONE", "STATUS_ERROR", "STATUS_UNDERFLOW",
    "IMAGE_MAGIC",
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
    """Word-address bases of each AXI write region (the host<->top contract)."""
    ctrl = 0
    tick = CTRL_WORDS
    coeff = tick + p.max_edges * 1
    mask = coeff + p.max_edges * p.coeff_words
    scan = mask + p.max_edges * p.mask_words
    bus = scan + 2 * p.bank_size * p.scan_words
    total = bus + p.bus_rows * p.bus_words
    return {"ctrl": ctrl, "tick": tick, "coeff": coeff, "mask": mask,
            "scan": scan, "bus": bus, "total": total}


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


def _bus_mode_value(mode) -> int:
    m = str(mode).strip().lower()
    return {"edge": 1, "ramp": 2}.get(m, 0) or _raise_mode(m)


def _raise_mode(m):
    raise ValueError(f"unsupported bus segment mode {m!r}.")


def _bus_mode_name(v: int) -> str:
    return {1: "edge", 2: "ramp"}.get(int(v)) or _raise_mode(v)


# --------------------------------------------------------------------------- pack
def scan_bank_words(program, p: StreamerParams, chunk_index: int) -> dict[int, int]:
    """Words to (re)load scan chunk ``chunk_index`` into its ping-pong bank.

    Chunk c = scan_points[c*bank_size:(c+1)*bank_size] lives in bank c%2.  Returns
    a sparse ``{word_offset: value}`` for just that bank, used by the host to stream
    chunks beyond the initial two.  Empty if the chunk is out of range."""
    bases = region_bases(p)
    points = [list(pt) for pt in (getattr(program, "scan_points", None) or [])]
    slot_count = int(getattr(program, "slot_count", 0) or 0)
    first = chunk_index * p.bank_size
    if first >= len(points):
        return {}
    bank = chunk_index % 2
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
    w[CtrlWords.LOOP_START] = int(getattr(program, "loop_start_index", 0))
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

    # first two scan chunks -> banks 0 and 1
    for chunk in (0, 1):
        w.update(scan_bank_words(program, p, chunk))

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
    return {
        "ticks": ticks, "masks": masks, "tick_slot_coeffs": coeffs,
        "scan_points_resident": scan_points, "scan_count": n_points, "slot_count": slot_count,
        "repeat_forever": bool(g(CtrlWords.REPEAT_FOREVER) & 1),
        "loop_start_index": g(CtrlWords.LOOP_START), "loop_count": g(CtrlWords.LOOP_COUNT),
        "loop_end_tick": g(CtrlWords.LOOP_END_TICK),
        "loop_end_slot_coeffs": _unpack_coeffs(_unfield([g(CtrlWords.LOOP_END_LO), g(CtrlWords.LOOP_END_HI)], p.coeff_bits), p),
        "bus_segments": bus_segments, "bank_size": g(CtrlWords.BANK_SIZE),
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


def solve_capacity(part, *, channel_count: int = 62, num_slots: int = 4, coeff_width: int = 16,
                   tick_width: int = 32, coeff_frac_bits: int = 8, bus_count: int = 4,
                   bus_width: int = 10, bus_seg_addr_width: int = 6, bus_sel_width: int = 3,
                   target_pct: float = 90.0, bank_size: int = 512,
                   max_edges_cap: int = 16384,
                   engine_logic_luts: int = 4500, engine_ff: int = 5000, engine_dsp: int = 8) -> SolvedCapacity:
    """Maximise max_edges under <=target_pct of the part's RAMB36 (edges are the
    bounded resource; scan points are UNBOUNDED via streaming, so only the 2-bank
    window costs BRAM).  Edge fields are parallel BRAMs (no width padding)."""
    prof = part_profile(part)
    pct = max(1.0, min(100.0, float(target_pct)))
    budget = int(prof.ramb36 * pct / 100.0)
    base = StreamerParams(channel_count=channel_count, num_slots=num_slots, coeff_width=coeff_width,
                          tick_width=tick_width, coeff_frac_bits=coeff_frac_bits, max_edges=256,
                          bank_size=bank_size, bus_count=bus_count, bus_width=bus_width,
                          bus_seg_addr_width=bus_seg_addr_width, bus_sel_width=bus_sel_width)
    # bus image is a small 32b BRAM (bus_rows*bus_words words); bus tables themselves
    # live in engine LUTRAM (counted under distributed RAM / LUT, not RAMB36).
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
    scan_ram = _scan_ramb(bank_size, base)
    params = StreamerParams(channel_count=channel_count, num_slots=num_slots, coeff_width=coeff_width,
                            tick_width=tick_width, coeff_frac_bits=coeff_frac_bits, max_edges=max_edges,
                            bank_size=bank_size, bus_count=bus_count, bus_width=bus_width,
                            bus_seg_addr_width=bus_seg_addr_width, bus_sel_width=bus_sel_width)
    ramb36_used = _edge_ramb(max_edges, params) + scan_ram + bus_img_ram + ctrl_ram
    bus_lutram = _ceil((2 * tick_width + 2 * params.coeff_bits + 2 * bus_width + 2 + bus_sel_width) * params.bus_rows, 64)

    def res(used, total):
        b = int(total * pct / 100.0)
        return {"used": int(used), "budget": b, "total": int(total),
                "pct": round(100.0 * used / total, 1) if total else 0.0, "ok": used <= b}

    report = {
        "ramb36": res(ramb36_used, prof.ramb36),
        "lut": res(engine_logic_luts + bus_lutram, prof.lut),
        "ff": res(engine_ff, prof.ff),
        "dsp": res(engine_dsp, prof.dsp),
    }
    return SolvedCapacity(part=prof.name, params=params, ramb36_used=ramb36_used,
                          ramb36_budget=budget, resource_report=report)
