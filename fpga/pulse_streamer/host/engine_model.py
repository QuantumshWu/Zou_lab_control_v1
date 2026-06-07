"""Cycle-accurate behavioural models of the FINAL affine edge-table engine
(``fpga/pulse_streamer/zlc_edge_streamer.v``), used to prove tick-exactness +
gaplessness BEFORE hardware (no Verilog simulator in this repo).

Three models, all walking the SAME engine FSM:

* :func:`reference_play` -- the *combinatorial* ground truth: every cycle it reads
  the current edge in-line and fires same-cycle on ``time_count == effective_tick``
  (this is the behaviour the design must reproduce; min edge spacing = 1 tick).

* :func:`prefetch_play` -- the BRAM engine's EDGE path: edge tables in block RAM
  (synchronous ``read_latency``-cycle read), hidden by a depth-(latency+1)
  continuous prefetch FIFO + first/second-edge (and loop-start/+1) shadows, so the
  four gapless reload sites reseed instantly and back-to-back **1-tick** edges
  still fire one per cycle.  Proven == reference for latency 1 AND 2.

* :func:`streaming_scan_play` -- the SCAN path: the scan-point table is a 2-bank
  ping-pong window of ``bank_size`` points; the host refills the idle bank behind
  the engine cursor so the total number of scan points is UNBOUNDED.  Proven ==
  reference over the full N-point sweep when the host keeps up, and STALL (hold,
  never a wrong point) on a late refill.

The RTL combines the edge FIFO and the scan ping-pong; each is verified here
independently and against the same ``reference_play`` ground truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "EngineProgram", "effective_tick", "reference_play", "prefetch_play",
    "streaming_scan_play", "rtl_mirror_play", "bus_play", "min_edge_spacing",
    "PrefetchStall", "ScanUnderflow",
]


class PrefetchStall(RuntimeError):
    """Edge FIFO underran (a well-sized FIFO + shadows must never hit this)."""


class ScanUnderflow(RuntimeError):
    """The host did not refill the next scan bank before the engine reached it."""


# Affine-MAC slot operand width -- MUST match zlc_edge_streamer.v SLOT_MUL_WIDTH.
# The per-slot scan value is multiplied by a 16-bit coeff using a single DSP48E1
# (25x18), so the slot operand is the low SLOT_MUL_WIDTH bits taken as signed.
# This bounds the raw scan VALUE to +/-2^24 ticks (~+/-335 ms @ 20 ns); the coeff
# still scales it, so the resulting tick offset spans the full 32-bit range.
SLOT_MUL_WIDTH = 25


def _narrow_slot(value: int) -> int:
    """Low SLOT_MUL_WIDTH bits of ``value`` as a signed int (mirrors the RTL's
    ``$signed(slots[.. +: SLOT_MUL_WIDTH])``)."""
    mask = (1 << SLOT_MUL_WIDTH) - 1
    v = int(value) & mask
    if v & (1 << (SLOT_MUL_WIDTH - 1)):
        v -= 1 << SLOT_MUL_WIDTH
    return v


def effective_tick(base_tick: int, coeffs: Sequence[int], slots: Sequence[int], frac_bits: int) -> int:
    """base + (sum coeff_j*slot_j) >>> frac (arithmetic shift; matches the RTL MAC
    and the host compiler).  The slot operand is narrowed to SLOT_MUL_WIDTH signed
    bits exactly as the RTL does.  Python ``>>`` on a negative int is an arithmetic
    (floor) shift, identical to Verilog ``>>>`` on the signed accumulator."""
    total = 0
    for c, s in zip(coeffs, slots):
        total += int(c) * _narrow_slot(s)
    return int(base_tick) + (total >> int(frac_bits))


# ---- disjoint-bit delay lanes (a scanned-delay channel pulled out of the global table)
# A lane is a 1-bit affine sub-player on its own output bit, advanced by its OWN frame
# tick ``llt`` (reset at every boundary) and OR'd into the digital output -- structurally
# the per-bus DAC engine specialised to 1 bit.  Shared by every engine model so the
# combinatorial reference, the FIFO model, and the exact RTL mirror all play lanes
# identically.
def _make_lanes(p: "EngineProgram") -> list[dict]:
    return [{"cb": cb, "ti": ti, "co": co, "va": va, "idx": 0, "bit": 0}
            for (cb, ti, co, va) in (p.delay_lanes or [])]


def _lane_bits(lanes: list[dict], llt: int, slots: Sequence[int], frac_bits: int) -> int:
    bits = 0
    for L in lanes:
        while L["idx"] < len(L["ti"]) and effective_tick(L["ti"][L["idx"]], L["co"][L["idx"]], slots, frac_bits) <= llt:
            L["bit"] = L["va"][L["idx"]]; L["idx"] += 1
        bits |= L["bit"] << L["cb"]
    return bits


def _lane_reset(lanes: list[dict]) -> None:
    for L in lanes:
        L["idx"] = 0; L["bit"] = 0


@dataclass
class EngineProgram:
    ticks: list[int]
    masks: list[int]
    tick_slot_coeffs: list[list[int]]
    scan_points: list[list[int]]
    slot_count: int
    frac_bits: int
    loop_start_index: int
    loop_end_tick: int
    loop_end_slot_coeffs: list[int]
    loop_count: int
    repeat_forever: bool
    repeat_from_index: int = 0
    # each lane: (channel_bit, [base ticks], [[coeffs]], [values 0/1]) -- a 1-bit affine
    # sub-player on a disjoint output bit (a scanned-delay channel pulled out of the
    # global sorted table).
    delay_lanes: list | None = None

    @classmethod
    def from_program(cls, program) -> "EngineProgram":
        slot_count = int(getattr(program, "slot_count", 0) or 0)
        ticks = [int(t) for t in program.ticks]
        coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in ticks])
        coeffs = [list(r) + [0] * (slot_count - len(r)) for r in coeffs]
        return cls(
            ticks=ticks,
            masks=[int(m) for m in program.masks],
            tick_slot_coeffs=coeffs,
            scan_points=[list(p) for p in (getattr(program, "scan_points", None) or [])],
            slot_count=slot_count,
            frac_bits=int(getattr(program, "scan_coeff_frac_bits", 8)),
            loop_start_index=int(getattr(program, "loop_start_index", 0)),
            loop_end_tick=int(getattr(program, "loop_end_tick", 0)),
            loop_end_slot_coeffs=list(getattr(program, "loop_end_slot_coeffs", None) or [0] * slot_count),
            loop_count=max(1, int(getattr(program, "loop_count", 1) or 1)),
            repeat_forever=bool(getattr(program, "repeat_forever", False)),
            repeat_from_index=int(getattr(program, "repeat_from_index", 0) or 0),
            delay_lanes=[
                (int(lane.channel_bit), [int(t) for t in lane.ticks],
                 [list(c) for c in lane.coeffs], [int(v) for v in lane.values])
                for lane in (getattr(program, "delay_lanes", None) or [])
            ] or None,
        )


def _zero(p: EngineProgram) -> list[int]:
    return [0] * p.slot_count


def _first_values(p: EngineProgram) -> list[int]:
    return list(p.scan_points[0]) if p.scan_points else _zero(p)


def min_edge_spacing(program) -> int:
    """Smallest gap (ticks) between consecutive effective edge ticks over all scan
    points (edge 0 exempt).  The FINAL FIFO engine handles 1-tick spacing, so this
    is informational; a value < 1 would indicate a non-monotonic program bug."""
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    if len(p.ticks) < 2:
        return 1 << 30
    points = p.scan_points or [_zero(p)]
    worst = 1 << 30
    for slots in points:
        prev = effective_tick(p.ticks[0], p.tick_slot_coeffs[0], slots, p.frac_bits)
        for i in range(1, len(p.ticks)):
            e = effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)
            worst = min(worst, e - prev)
            prev = e
    return worst


# ----------------------------------------------------------------------------
# reference: combinatorial engine (the ground truth the RTL must reproduce)
# ----------------------------------------------------------------------------
def reference_play(program, n_ticks: int) -> list[int]:
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n = len(p.ticks)
    scan_en = bool(p.scan_points)
    scan_count = len(p.scan_points)

    def eff(i, slots):
        return effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_le(slots):
        return effective_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    slot = _first_values(p)
    final = 0 if n == 0 else eff(n - 1, slot)
    loop_end = eff_le(slot)
    loops = p.loop_count
    spi = 0
    running = n != 0
    if running and eff(0, slot) == 0:
        sm, tc, ei = p.masks[0], 1, 1
    else:
        sm, tc, ei = 0, 0, 0

    # Disjoint-bit DELAY LANES: a scanned-delay channel that reorders edges is pulled
    # out of the global sorted table onto its own bit, played by a 1-bit affine
    # sub-player advanced by its OWN frame-tick `llt` and reseeded at every boundary, so
    # a reordering delay never disturbs the global edge stream.  out = main | OR(lanes).
    lanes = _make_lanes(p)

    def lane_reset():
        _lane_reset(lanes)

    llt = 0
    out = []
    for _ in range(n_ticks):
        out.append(sm | _lane_bits(lanes, llt, slot, p.frac_bits))
        llt += 1
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]
            tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1
            loops -= 1
            llt = eff(p.loop_start_index, slot); lane_reset()
        elif tc >= final:
            if scan_en and spi + 1 < scan_count:
                slot = list(p.scan_points[spi + 1]); spi += 1
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
                llt = 0; lane_reset()
            elif p.repeat_forever:
                slot = _first_values(p); spi = 0
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                ri = p.repeat_from_index
                if ri > 0:
                    # rewind to the steady-state frame start (additive-delay preamble
                    # plays once); the engine seeds masks[ri] at its tick + 1.
                    sm, tc, ei = p.masks[ri], eff(ri, slot) + 1, ri + 1
                    llt = eff(ri, slot)
                else:
                    sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
                    llt = 0
                lane_reset()
            else:
                running = False; sm = 0
        else:
            if ei < n and tc == eff(ei, slot):
                sm = p.masks[ei]; ei += 1
            tc += 1
    return out


# ----------------------------------------------------------------------------
# edge FIFO prefetch (BRAM edge tables, 1-tick seamless)
# ----------------------------------------------------------------------------
def prefetch_play(program, n_ticks: int, *, read_latency: int = 2, fifo_depth: int = 3) -> list[int]:
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n = len(p.ticks)
    scan_en = bool(p.scan_points)
    scan_count = len(p.scan_points)
    if fifo_depth < read_latency + 1:
        fifo_depth = read_latency + 1

    def eff(i, slots):
        return effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_le(slots):
        return effective_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    fifo: deque[int] = deque()
    inflight: list[tuple[int, int]] = []
    fetch_idx = 0
    cycle = 0

    def reseed(target):
        nonlocal fetch_idx
        fifo.clear(); inflight.clear()
        if target < n:
            fifo.append(target)
        if target + 1 < n:
            fifo.append(target + 1)
        fetch_idx = target + 2
        issue()

    def issue():
        nonlocal fetch_idx
        resident = len(fifo) + len(inflight)
        while resident < fifo_depth and fetch_idx < n:
            inflight.append((fetch_idx, cycle + read_latency)); fetch_idx += 1; resident += 1

    def land():
        ready = sorted([it for it in inflight if it[1] <= cycle], key=lambda t: t[0])
        for it in ready:
            inflight.remove(it); fifo.append(it[0])

    slot = _first_values(p)
    final = 0 if n == 0 else eff(n - 1, slot)
    loop_end = eff_le(slot)
    loops = p.loop_count
    spi = 0
    running = n != 0
    reseed(0)
    if running and eff(0, slot) == 0:
        sm, tc, ei = p.masks[0], 1, 1
        if fifo and fifo[0] == 0:
            fifo.popleft()
        issue()
    else:
        sm, tc, ei = 0, 0, 0

    lanes = _make_lanes(p)
    llt = 0
    out = []
    for _ in range(n_ticks):
        cycle += 1; land(); out.append(sm | _lane_bits(lanes, llt, slot, p.frac_bits)); llt += 1
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]
            tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1
            loops -= 1
            reseed(p.loop_start_index)
            if fifo and fifo[0] == p.loop_start_index:
                fifo.popleft()
            issue()
            llt = eff(p.loop_start_index, slot); _lane_reset(lanes)
        elif tc >= final:
            if scan_en and spi + 1 < scan_count:
                slot = list(p.scan_points[spi + 1]); spi += 1
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                reseed(0)
                if eff(0, slot) == 0:
                    sm, tc, ei = p.masks[0], 1, 1
                    if fifo and fifo[0] == 0:
                        fifo.popleft()
                    issue()
                else:
                    sm, tc, ei = 0, 0, 0
                llt = 0; _lane_reset(lanes)
            elif p.repeat_forever:
                slot = _first_values(p); spi = 0
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                ri = p.repeat_from_index
                if ri > 0:
                    # additive-delay: rewind to the steady frame, not edge 0
                    sm, tc, ei = p.masks[ri], eff(ri, slot) + 1, ri + 1
                    reseed(ri)
                    if fifo and fifo[0] == ri:
                        fifo.popleft()
                    issue()
                    llt = eff(ri, slot)
                else:
                    reseed(0)
                    if eff(0, slot) == 0:
                        sm, tc, ei = p.masks[0], 1, 1
                        if fifo and fifo[0] == 0:
                            fifo.popleft()
                        issue()
                    else:
                        sm, tc, ei = 0, 0, 0
                    llt = 0
                _lane_reset(lanes)
            else:
                running = False; sm = 0
        else:
            if ei < n:
                if not fifo or fifo[0] != ei:
                    raise PrefetchStall(f"edge FIFO underrun: need edge {ei} at tick {tc}, head={fifo[0] if fifo else None}")
                if tc == eff(ei, slot):
                    sm = p.masks[ei]; fifo.popleft(); ei += 1; issue()
            tc += 1
    return out


# ----------------------------------------------------------------------------
# scan ping-pong streaming (unbounded scan points)
# ----------------------------------------------------------------------------
def streaming_scan_play(program, n_ticks: int, *, bank_size: int, refill_delay: int = 0,
                        raise_on_underflow: bool = False):
    """Play the engine with the scan-point table held in a 2-bank ping-pong window
    of ``bank_size`` points each.  Banks hold CHUNKS of the full sweep: chunk c =
    points[c*bank_size:(c+1)*bank_size] sits in bank c%2.  The host pre-loads
    chunks 0 and 1; when the engine crosses from chunk c into c+1 it frees the bank
    holding chunk c and the host refills it with chunk c+2 ``refill_delay`` ticks
    later (modelling JTAG-AXI write latency).  Returns (out, stalled, points_played).

    With ``refill_delay`` small enough the output equals :func:`reference_play` over
    the full N-point sweep (gapless, unbounded points).  A late refill makes the
    engine STALL (hold the current state, never emit a wrong point); set
    ``raise_on_underflow`` to turn that stall into a :class:`ScanUnderflow`."""
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n = len(p.ticks)
    points = p.scan_points
    N = len(points)
    if N == 0 or bank_size <= 0:
        return reference_play(program, n_ticks), False, 0

    def eff(i, slots):
        return effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_le(slots):
        return effective_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    n_chunks = (N + bank_size - 1) // bank_size
    bank_chunk = [-1, -1]     # which chunk each bank currently holds (-1 = none)
    bank_ready = [False, False]
    pending: list[tuple[int, int, int]] = []   # (bank, chunk, ready_cycle)

    def load(b, chunk):
        bank_chunk[b] = chunk
        bank_ready[b] = True

    def preload():
        bank_chunk[0] = bank_chunk[1] = -1
        bank_ready[0] = bank_ready[1] = False
        pending.clear()
        load(0, 0)
        if n_chunks > 1:
            load(1, 1)

    def point(idx):
        chunk = idx // bank_size
        b = chunk % 2
        if bank_chunk[b] == chunk and bank_ready[b]:
            return points[idx]
        return None           # this chunk isn't resident yet -> underflow/stall

    preload()
    slot = list(points[0])
    final = eff(n - 1, slot)
    loop_end = eff_le(slot)
    loops = p.loop_count
    spi = 0
    running = n != 0
    stalled = False
    cycle = 0
    sm, tc, ei = (p.masks[0], 1, 1) if (running and eff(0, slot) == 0) else (0, 0, 0)

    out = []
    for _ in range(n_ticks):
        cycle += 1
        for item in [it for it in pending if it[2] <= cycle]:
            pending.remove(item)
            load(item[0], item[1])
        out.append(sm)
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]; tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1; loops -= 1
        elif tc >= final:
            nxt_idx = spi + 1
            if nxt_idx < N:
                nxt = point(nxt_idx)
                if nxt is None:
                    if raise_on_underflow:
                        raise ScanUnderflow(f"scan chunk {nxt_idx // bank_size} not ready at tick {tc}")
                    stalled = True            # hold; re-check next tick
                else:
                    old_chunk = spi // bank_size
                    new_chunk = nxt_idx // bank_size
                    if new_chunk != old_chunk:
                        free_bank = old_chunk % 2
                        bank_ready[free_bank] = False
                        bank_chunk[free_bank] = -1
                        refill_chunk = old_chunk + 2
                        if refill_chunk < n_chunks:
                            pending.append((free_bank, refill_chunk, cycle + max(0, refill_delay)))
                    slot = list(nxt); spi = nxt_idx
                    final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                    sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
            elif p.repeat_forever:
                preload()
                slot = list(points[0]); spi = 0
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
            else:
                running = False; sm = 0
        else:
            if ei < n and tc == eff(ei, slot):
                sm = p.masks[ei]; ei += 1
            tc += 1
    return out, stalled, spi + 1


def rtl_mirror_play(program, n_ticks: int, *, rd_lat: int = 2, fifo_depth: int = 3) -> list[int]:
    """Re-implements the EXACT register transfers of zlc_edge_streamer.v's edge
    prefetch (arm[] shift-down FIFO + nv count + pend in-flight shift + the
    issue-occupancy condition + the four boundary reseeds), so a divergence from
    :func:`reference_play` flags a bug in THAT RTL realization (not just the
    abstract algorithm).  Resident scan (no streaming) -- the bank addressing is
    proven separately by :func:`streaming_scan_play`."""
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n = len(p.ticks)
    scan_en = bool(p.scan_points)
    scan_count = len(p.scan_points)

    def eff(i, slots):
        return effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_le(slots):
        return effective_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    arm: list[int] = []                          # arm[0] = head; len(arm) = nv
    pend: list = [None] * rd_lat                  # pend[rd_lat-1] lands this cycle
    fetch_idx = 0

    def reseed_from(start_idx):
        # Seed FIFO_DEPTH(=RD_LAT+1) resident shadows beginning at the first
        # not-yet-output edge.  That runway is exactly enough that the first
        # PREFETCHED edge (issued when the head fires) lands + registers into arm
        # in time for back-to-back 1-tick edges.  occupancy == #shadows <= depth,
        # so no read is issued at the boundary (no overflow); reads start when a
        # slot frees.
        nonlocal fetch_idx, pend
        arm.clear()
        for k in range(fifo_depth):
            if start_idx + k < n:
                arm.append(start_idx + k)
        fetch_idx = start_idx + fifo_depth
        pend = [None] * rd_lat

    def boundary_to(start_at_zero_mask):
        # Common start/scan-advance/repeat seed: output edge0 directly iff it
        # fires at tick 0, else let the FIFO fire it.
        if eff(0, slot) == 0:
            reseed_from(1)
            return p.masks[0], 1, 1
        reseed_from(0)
        return 0, 0, 0

    slot = _first_values(p)
    final = 0 if n == 0 else eff(n - 1, slot)
    loop_end = eff_le(slot)
    loops = p.loop_count
    spi = 0
    running = n != 0
    if running:
        sm, tc, ei = boundary_to(True)
    else:
        sm, tc, ei = 0, 0, 0

    lanes = _make_lanes(p)
    llt = 0
    out = []
    for _ in range(n_ticks):
        out.append(sm | _lane_bits(lanes, llt, slot, p.frac_bits)); llt += 1
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]
            tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1
            loops -= 1
            reseed_from(p.loop_start_index + 1)   # loop_start output directly above
            llt = eff(p.loop_start_index, slot); _lane_reset(lanes)
            continue
        if tc >= final:
            if scan_en and spi + 1 < scan_count:
                slot = list(p.scan_points[spi + 1]); spi += 1
            elif p.repeat_forever and p.repeat_from_index > 0 and not scan_en:
                # additive-delay repeat: rewind to the steady frame (loop_start
                # shadows), NOT edge 0 -- mirrors the RTL repeat_from_loop_start branch.
                ri = p.repeat_from_index
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                sm = p.masks[ri]; tc = eff(ri, slot) + 1; ei = ri + 1
                reseed_from(ri + 1)
                llt = eff(ri, slot); _lane_reset(lanes)
                continue
            elif p.repeat_forever:
                slot = _first_values(p); spi = 0
            else:
                running = False; sm = 0; continue
            final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
            sm, tc, ei = boundary_to(True)
            llt = 0; _lane_reset(lanes)
            continue
        # ---- normal cycle: exact RTL FIFO transfers ----
        landed_idx = pend[rd_lat - 1]
        fire_arm = (ei < n) and (len(arm) != 0) and (tc == eff(arm[0], slot))
        if fire_arm:
            sm = p.masks[arm[0]]
            ei += 1
            arm.pop(0)               # shift down
        if landed_idx is not None:
            arm.append(landed_idx)   # land at tail (register: visible next cycle)
        tc += 1
        # issue a read iff resident + still-in-flight is below depth
        inflight_after = sum(1 for x in pend[0:rd_lat - 1] if x is not None)
        occupancy = len(arm) + inflight_after
        issue = (occupancy < fifo_depth) and (fetch_idx < n)
        new_pend = [fetch_idx if issue else None] + pend[0:rd_lat - 1]
        if issue:
            fetch_idx += 1
        pend = new_pend
    return out


def bus_play(program, bus_index: int, n_ticks: int, scan_point: int = 0, *,
             bus_width: int = 10, frac_bits: int | None = None) -> list[int]:
    """Cycle-accurate mirror of zlc_edge_streamer.v's per-bus DAC engine
    (zlc_bus_start_table + zlc_bus_step + zlc_bus_apply_segment): the interpolating
    ramp, the DUAL start/stop value_select (a ramp can scan BOTH endpoints), and the
    affine segment ticks.  Returns bus_out at each tick for one scan point -- the
    bus-path counterpart of reference_play (which covers only the digital edges)."""

    frac = int(getattr(program, "scan_coeff_frac_bits", 8)) if frac_bits is None else frac_bits
    pts = list(getattr(program, "scan_points", None) or [])
    point = list(pts[scan_point]) if pts else []
    mask = (1 << bus_width) - 1
    segs = [s for s in (getattr(program, "bus_segments", None) or []) if int(s.bus_index) == bus_index]

    def eff(base, coeffs):
        c = [int(x) for x in (coeffs or [])]
        return effective_tick(int(base), c, point, frac) if (c and point) else int(base)

    segs.sort(key=lambda s: eff(s.start_tick, s.start_tick_coeffs))

    def endpoints(s):
        vs = (point[int(s.value_select) - 1] & mask) if int(getattr(s, "value_select", 0)) else (int(s.start_value) & mask)
        sss = int(getattr(s, "stop_value_select", getattr(s, "value_select", 0)))
        ve = (point[sss - 1] & mask) if sss else (int(s.stop_value) & mask)
        return vs, ve

    st = {"idx": 0, "value": 0, "ramp": False, "rstart": 0, "rstop": 0,
          "target": 0, "denom": 0, "accum": 0, "up": True, "delta": 0}

    def apply(s):
        vs, ve = endpoints(s)
        ts = eff(s.start_tick, s.start_tick_coeffs)
        te = eff(s.stop_tick, s.stop_tick_coeffs)
        if str(s.mode).lower() == "ramp" and te > ts:
            st.update(value=vs, ramp=True, rstart=ts, rstop=te, target=ve, denom=te - ts,
                      accum=0, up=ve >= vs, delta=(ve - vs) if ve >= vs else (vs - ve))
        else:
            st.update(value=ve, ramp=False, accum=0)

    if segs and eff(segs[0].start_tick, segs[0].start_tick_coeffs) == 0:
        apply(segs[0]); st["idx"] = 1

    out = []
    for t in range(n_ticks):
        out.append(st["value"])           # registered bus_out value at this tick
        if st["ramp"]:
            if t >= st["rstop"]:
                st["value"] = st["target"]; st["ramp"] = False; st["accum"] = 0
                if st["idx"] < len(segs) and eff(segs[st["idx"]].start_tick, segs[st["idx"]].start_tick_coeffs) <= t:
                    apply(segs[st["idx"]]); st["idx"] += 1
            elif t > st["rstart"] and st["denom"]:
                st["accum"] += st["delta"]
                if st["accum"] >= st["denom"]:
                    st["accum"] -= st["denom"]
                    if st["up"] and st["value"] < st["target"]:
                        st["value"] += 1
                    elif (not st["up"]) and st["value"] > st["target"]:
                        st["value"] -= 1
        elif st["idx"] < len(segs) and t >= eff(segs[st["idx"]].start_tick, segs[st["idx"]].start_tick_coeffs):
            apply(segs[st["idx"]]); st["idx"] += 1
    return out
