"""Cycle-accurate behavioural models of the FINAL affine edge-table engine
(``fpga/pulse_streamer/zlc_edge_streamer.v``), used to prove tick-exactness +
gaplessness BEFORE hardware (no Verilog simulator in this repo).

Three models, all walking the SAME engine FSM:

* :func:`reference_play` -- the *combinatorial* ground truth: every cycle it reads
  the current edge in-line and fires same-cycle on ``time_count == effective_tick``
  (this is the behaviour the design must reproduce; min edge spacing = 1 tick).

* :func:`prefetch_play` -- the BRAM engine's EDGE path: edge tables in block RAM
  (synchronous ``read_latency``-cycle read behind a REGISTERED address, so the true
  issue->data-valid latency is read_latency+1), hidden by a depth-(latency+2)
  continuous prefetch FIFO + edge/loop-start shadows, so the four gapless reload
  sites reseed instantly and back-to-back **1-tick** edges still fire one per cycle.
  Proven == reference for latency 1 AND 2.

* :func:`streaming_scan_play` -- the SCAN path: the scan-point table is a 2-bank
  ping-pong window of ``bank_size`` points; the host refills the idle bank behind
  the engine cursor so the total number of scan points is UNBOUNDED.  Proven ==
  reference over the full N-point sweep when the host keeps up, and STALL (hold,
  never a wrong point) on a late refill.

The RTL combines the edge FIFO and the scan ping-pong; each is verified here
independently and against the same ``reference_play`` ground truth.

The per-channel / per-bus OUTPUT delay is a LITERAL delay line (a plain circular buffer):
:func:`delay_line_reference` / :func:`bus_delay_line_reference` are the exact stream-shift
ground truth (out[t]=in[t-d], 0 before fire), and :func:`rtl_delay_line_mirror` /
:func:`rtl_bus_delay_line_mirror` are the cycle-exact register mirrors of the RTL circular
buffer (bounded to :data:`DELAY_DEPTH`; a d past it raises :class:`DelayDepthExceeded`).
``reference_play`` / ``prefetch_play`` / ``rtl_mirror_play`` apply the channel delay line as a
post-play shift via :func:`_apply_channel_delays`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "EngineProgram", "effective_tick", "reference_play", "prefetch_play",
    "streaming_scan_play", "rtl_mirror_play", "bus_play", "min_edge_spacing",
    "PrefetchStall", "ScanUnderflow", "DelayDepthExceeded", "DELAY_DEPTH",
    "delay_line_reference", "bus_delay_line_reference",
    "rtl_delay_line_mirror", "rtl_bus_delay_line_mirror",
    "bus_value_at",
]

# Per-channel / per-bus OUTPUT delay-line depth (in ticks) -- MUST match the RTL
# DELAY_DEPTH localparam and host.image / fpga_pulse_streamer DEFAULT_DELAY_DEPTH.
# A bounded cap: 2048 ticks * 20 ns = ~40 us (covers +/-15 us after the negative-delay
# global shift G, which can push an effective delay to ~30 us).
DELAY_DEPTH = 2048            # DAC bus ring depth (TTL no longer uses it)
TTL_DELAY_MAX_TICKS = (1 << 31) - 1
EVT_FIFO_DEPTH = 16


class DelayDepthExceeded(ValueError):
    """An effective channel/bus delay exceeds the bounded delay-line depth."""


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


# ----------------------------------------------------------------------------
# PHYSICAL CHANNEL DELAY -- a literal OUTPUT delay line (the final, correct model)
# ----------------------------------------------------------------------------
# A channel delay is NOT baked into the edge ticks; it is a per-channel delay on the engine
# OUTPUT:  output_delayed[t] = output_undelayed[t - d], zero before fire.  The TTL hardware is a
# per-channel variable-tap SHIFT REGISTER (the SRL primitive, depth DELAY_DEPTH, 1 bit): each
# tick shift the channel's UNDELAYED output bit in at [0]; the delayed bit is the tap at index
# d-1 (the value pushed d ticks ago).  d=0 is exact passthrough; d>0 reads a tap whose startup
# gate holds its FIRE-time 0 until t>=d (silent startup, for free).  Bounded to DELAY_DEPTH ticks
# (~40 us at 20 ns/tick).  (The DAC bus delay is still a 10-bit-wide circular buffer.)
#
#   * delay_line_reference     -- the EXACT stream-shift ground truth.
#   * rtl_delay_line_mirror    -- the cycle-exact RTL shift-register register mirror.
def delay_line_reference(undelayed: Sequence[int], channel_delays) -> list[int]:
    """Exact physical delay line: every channel bit delayed by its own ``d``, 0 before fire.
    Non-delayed bits pass through untouched (a delay never disturbs another channel)."""
    cds = {int(b): int(d) for b, d in dict(channel_delays).items() if int(d) != 0}
    delayed_mask = 0
    for b in cds:
        delayed_mask |= 1 << b
    out = []
    for t in range(len(undelayed)):
        m = int(undelayed[t]) & ~delayed_mask          # non-delayed channels: passthrough
        for bit, d in cds.items():
            s = t - d
            if s >= 0 and (int(undelayed[s]) >> bit) & 1:
                m |= 1 << bit
        out.append(m)
    return out


def bus_delay_line_reference(undelayed_bus: Sequence[int], delay: int,
                             *, safe_value: int = 512) -> list[int]:
    """Exact per-bus delay line: a 10-bit DAC value stream delayed by ``d`` (one delay
    shared by all 10 bits), holding ``safe_value`` before t == d.  The hardware default is
    the SAFE mid-scale code 512 (= BUS_SAFE_VALUE = true 0 V on the offset-binary driver).
    The hardware is a per-bus 10-bit-wide circular buffer, depth DELAY_DEPTH: push the
    undelayed value each tick, read d ticks ago.  d=0 is exact passthrough."""
    d = int(delay)
    out = []
    for t in range(len(undelayed_bus)):
        s = t - d
        out.append(int(undelayed_bus[s]) if s >= 0 else int(safe_value))
    return out


# ----------------------------------------------------------------------------
# CYCLE-EXACT REGISTER MIRRORS of the literal delay-line hardware (no Verilog sim)
# ----------------------------------------------------------------------------
# These reproduce the EXACT registers of zlc_edge_streamer.v's delay line so a divergence
# from delay_line_reference / bus_delay_line_reference flags an RTL bug.  The two paths differ:
#
#   TTL (rtl_delay_line_mirror): a per-channel variable-tap SHIFT REGISTER (the SRL primitive,
#   DELAY_DEPTH+1 bits, zero at FIRE).  Each RUNNING tick read the gated tap THEN shift in:
#       delayed_bit[ch] = (del_fill >= d_ch) ? sr[ch][d_ch - 1] : 0   # value pushed d_ch ago
#       sr[ch] = {sr[ch][DELAY_DEPTH-1:0], undelayed_bit[ch]}         # shift newest in at [0]
#   The pre-shift tap sr[k] == the bit pushed k+1 ticks ago, so sr[d-1] == d ticks ago; del_fill
#   (running tick index) gates it to 0 until t>=d.  d==0 is bypassed (passthrough).
#
#   DAC (rtl_bus_delay_line_mirror): a 10-bit-wide circular buffer of DELAY_DEPTH+1 slots (zero
#   at FIRE; wptr=0).  Each RUNNING tick:
#       ring[wptr] = undelayed_value                                 # push this tick's value
#       delayed_value = ring[(wptr - d) mod (DELAY_DEPTH+1)]         # value pushed d ago
#       wptr = (wptr + 1) mod (DELAY_DEPTH+1)
#   d=0 reads the slot just written this tick -> exact passthrough; d>0 before the buffer has
#   filled (t<d) reads a still-zero slot -> silent until t>=d.
#
#   Both are out[t]=in[t-d], 0 before fire, byte-for-byte == the *_reference functions.


def _validate_delay_depth(d: int, depth: int, what: str) -> int:
    d = int(d)
    if d < 0 or d > depth:
        raise DelayDepthExceeded(
            f"{what} delay {d} ticks exceeds the delay-line depth DELAY_DEPTH={depth} "
            f"(~{depth * 20 / 1000:.0f}us); reduce the delay.")
    return d


def rtl_delay_line_mirror(undelayed: Sequence[int], channel_delays,
                          *, depth: int = TTL_DELAY_MAX_TICKS,
                          evt_depth: int = EVT_FIFO_DEPTH) -> list[int]:
    """Cycle-exact register mirror of the RTL per-channel TTL delay -- now the EVENT
    SCHEDULER (``evt_mem``/``evt_out``/``g_time``), NOT the old per-tick shift register.

    Faithfully modelling the RTL:

      * during cycle t the undelayed bit differs from ``prev_undelayed`` (the toggle AT
        t) -> push ``(t + d - 1, new_level)`` into that channel's ``evt_depth``-deep FIFO
        (``d >= 2``; ``d == 1`` is served by the prev register; ``d == 0`` bypasses);
      * during cycle u == t + d - 1 the head time equals ``g_time`` -> the level registers
        into ``evt_out`` (visible at t + d)  ==>  out[t] = in[t-d], 0 before the first
        scheduled toggle -- byte-identical to :func:`delay_line_reference`;
      * a push into a FULL FIFO is DROPPED (the RTL guard) -- the host validator
        prevents ever getting there (toggle-density window check).

    Equals :func:`delay_line_reference` for every d in [0, depth]; a d > depth raises
    :class:`DelayDepthExceeded` (the 32-bit field cap)."""
    cds = {int(b): _validate_delay_depth(d, depth, f"channel bit {b}")
           for b, d in dict(channel_delays).items() if int(d) != 0}
    delayed_mask = 0
    for b in cds:
        delayed_mask |= 1 << b
    queues = {b: [] for b in cds}              # per-channel [(scheduled_time, level)]
    evt_out = {b: 0 for b in cds}
    prev = 0
    out = []
    for t in range(len(undelayed)):
        cur = int(undelayed[t])
        # OUTPUT for cycle t (registers updated at the END of the cycle, like the RTL)
        m = cur & ~delayed_mask
        for b, d in cds.items():
            level = (prev >> b) & 1 if d == 1 else evt_out[b]
            m |= (level & 1) << b
        out.append(m)
        # end-of-cycle register updates: pops (head == t), then pushes (toggle at t)
        for b, d in cds.items():
            q = queues[b]
            if q and q[0][0] == t:
                evt_out[b] = q.pop(0)[1]
            if d >= 2 and (((cur ^ prev) >> b) & 1):
                if len(q) < evt_depth:          # overflow guard: drop (validator prevents)
                    q.append((t + d - 1, (cur >> b) & 1))
        prev = cur
    return out

def rtl_bus_delay_line_mirror(undelayed_bus: Sequence[int], delay: int,
                              *, depth: int = DELAY_DEPTH, safe_value: int = 512) -> list[int]:
    """Cycle-exact circular-buffer register mirror of the RTL per-bus (10-bit) delay line.

    One ``depth``-slot ring of bus VALUES (the SAFE mid-scale code 512 = BUS_SAFE_VALUE =
    0 V at FIRE); push the undelayed value each tick, read ``d`` writes ago.  Equals
    :func:`bus_delay_line_reference`; a d > depth raises :class:`DelayDepthExceeded`."""
    d = _validate_delay_depth(delay, depth, "bus")
    slots = depth + 1                                  # see rtl_delay_line_mirror (d==depth fits)
    ring = [int(safe_value)] * slots
    wptr = 0
    out = []
    for t in range(len(undelayed_bus)):
        ring[wptr] = int(undelayed_bus[t])             # push this tick's undelayed value
        out.append(ring[(wptr - d) % slots])           # value pushed d ticks ago
        wptr = (wptr + 1) % slots
    return out


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
    # PHYSICAL CHANNEL DELAY: per-output-bit delay in ticks, applied to the engine OUTPUT
    # (a delay line) AFTER the undelayed play -- never baked into the edges.
    channel_delays: list[int] | None = None

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
            channel_delays=[int(v) for v in (getattr(program, "channel_delays", None) or [])] or None,
        )


def _apply_channel_delays(out: list[int], p: "EngineProgram") -> list[int]:
    """Apply the per-channel OUTPUT delay (the literal delay line) to a finished play.
    No-op when no channel is delayed.  This is the EXACT ``delay_line_reference`` (the
    ground truth); the RTL realises the same with a per-channel circular buffer (depth
    DELAY_DEPTH), proven equal by ``rtl_delay_line_mirror``."""
    if not p.channel_delays or not any(p.channel_delays):
        return out
    cds = {b: int(d) for b, d in enumerate(p.channel_delays) if int(d)}
    return delay_line_reference(out, cds)


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

    out = []
    for _ in range(n_ticks):
        out.append(sm)
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]
            tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1
            loops -= 1
        elif tc >= final:
            if scan_en and spi + 1 < scan_count:
                slot = list(p.scan_points[spi + 1]); spi += 1
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
            elif p.repeat_forever:
                slot = _first_values(p); spi = 0
                final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                ri = p.repeat_from_index
                if ri > 0:
                    # rewind to the steady-state frame start (additive-delay preamble
                    # plays once); the engine seeds masks[ri] at its tick + 1.
                    sm, tc, ei = p.masks[ri], eff(ri, slot) + 1, ri + 1
                else:
                    sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
            else:
                running = False; sm = 0
        else:
            if ei < n and tc == eff(ei, slot):
                sm = p.masks[ei]; ei += 1
            tc += 1
    return _apply_channel_delays(out, p)


# ----------------------------------------------------------------------------
# edge FIFO prefetch (BRAM edge tables, 1-tick seamless)
# ----------------------------------------------------------------------------
def prefetch_play(program, n_ticks: int, *, read_latency: int = 2, fifo_depth: int = 4) -> list[int]:
    # fifo_depth = read_latency + 2 (NOT +1): the read pipeline is read_latency+1 deep because
    # edge_raddr is a registered address (an issued read reaches the BRAM the NEXT cycle, then
    # the BRAM adds read_latency).  Sustaining 1-tick playback needs a resident head plus one
    # in-flight slot per pipeline stage = (read_latency+1) + 1.  See zlc_edge_streamer.v.
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n = len(p.ticks)
    scan_en = bool(p.scan_points)
    scan_count = len(p.scan_points)
    if fifo_depth < read_latency + 2:
        fifo_depth = read_latency + 2

    def eff(i, slots):
        return effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_le(slots):
        return effective_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    fifo: deque[int] = deque()
    inflight: list[tuple[int, int]] = []
    fetch_idx = 0
    cycle = 0

    def reseed(target):
        # Seed FIFO_DEPTH resident shadows (the RTL latches FIFO_DEPTH edge shadows at every
        # boundary).  With the issue->data-valid latency now read_latency+1 (the registered
        # edge_raddr), 2 resident + in-flight refill would underrun 1-tick playback; the full
        # FIFO_DEPTH resident gives the prefetch enough runway to refill behind each fire.
        nonlocal fetch_idx
        fifo.clear(); inflight.clear()
        for k in range(fifo_depth):
            if target + k < n:
                fifo.append(target + k)
        fetch_idx = target + fifo_depth
        issue()

    def issue():
        nonlocal fetch_idx
        resident = len(fifo) + len(inflight)
        while resident < fifo_depth and fetch_idx < n:
            # +1 = the registered edge_raddr stage: issue->data-valid is read_latency+1 cycles.
            inflight.append((fetch_idx, cycle + read_latency + 1)); fetch_idx += 1; resident += 1

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

    out = []
    for _ in range(n_ticks):
        cycle += 1; land(); out.append(sm)
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
                else:
                    reseed(0)
                    if eff(0, slot) == 0:
                        sm, tc, ei = p.masks[0], 1, 1
                        if fifo and fifo[0] == 0:
                            fifo.popleft()
                        issue()
                    else:
                        sm, tc, ei = 0, 0, 0
            else:
                running = False; sm = 0
        else:
            if ei < n:
                if not fifo or fifo[0] != ei:
                    raise PrefetchStall(f"edge FIFO underrun: need edge {ei} at tick {tc}, head={fifo[0] if fifo else None}")
                if tc == eff(ei, slot):
                    sm = p.masks[ei]; fifo.popleft(); ei += 1; issue()
            tc += 1
    return _apply_channel_delays(out, p)


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
    bank_chunk = [-1, -1]     # which DATA chunk each bank currently holds (-1 = none)
    bank_ready = [False, False]
    pending: list[tuple[int, int, int]] = []   # (bank, chunk, ready_cycle)

    # CONTINUOUS CYCLIC PING-PONG (the seamless-wrap design).  The bank a chunk lives in is
    # NOT chunk%2 but the parity of the MONOTONIC chunk count -- so the sweep WRAP (chunk
    # K-1 -> chunk 0) is just another chunk boundary, fed one-ahead like every other, with
    # NO special reload and NO stall.  bank = (chunk%2) ^ scan_bank_base, where scan_bank_base
    # toggles by (n_chunks & 1) at each wrap (0 for K even -- chunk%2 already alternates across
    # the wrap; 1 for K odd -- chunk K-1 and chunk 0 would otherwise collide in the same bank).
    # For a RESIDENT scan (n_chunks <= 2, fits in the 2 banks, never streamed) base stays 0 and
    # this is byte-identical to the old chunk%2 mapping -- the proven, seamless small-scan path.
    streaming = n_chunks > 2
    wrap_toggle = (n_chunks & 1) if streaming else 0

    def load(b, chunk):
        bank_chunk[b] = chunk
        bank_ready[b] = True

    def preload():
        # monotonic chunks 0 and 1 -> banks 0 and 1 (base starts at 0); for n_chunks==1 the
        # second bank mirrors chunk 0 (1 % n_chunks) so a resident wrap still finds it.
        bank_chunk[0] = bank_chunk[1] = -1
        bank_ready[0] = bank_ready[1] = False
        pending.clear()
        load(0, 0)
        if n_chunks > 1:
            load(1, 1)

    preload()
    slot = list(points[0])
    final = eff(n - 1, slot)
    loop_end = eff_le(slot)
    loops = p.loop_count
    spi = 0
    base = 0                  # scan_bank_base: parity offset, toggles by wrap_toggle each wrap
    running = n != 0
    stalled = False
    cycle = 0
    sm, tc, ei = (p.masks[0], 1, 1) if (running and eff(0, slot) == 0) else (0, 0, 0)

    def bank_of_chunk(chunk, b):
        return (chunk % 2) ^ b

    def host_refill():
        # one-ahead cyclic refill (streaming only): keep the bank that will hold the NEXT
        # monotonic chunk loaded with that chunk's data.  The "next" chunk after the current
        # (cur_chunk, base) is (cur_chunk+1) within the sweep or 0 at the wrap, and its base is
        # base (within sweep) or base^wrap_toggle (across the wrap).
        if not streaming:
            return
        cur_chunk = spi // bank_size
        if cur_chunk + 1 < n_chunks:
            nxt_chunk, nxt_base = cur_chunk + 1, base
        else:
            nxt_chunk, nxt_base = 0, base ^ wrap_toggle
        nb = bank_of_chunk(nxt_chunk, nxt_base)
        if (bank_ready[nb] and bank_chunk[nb] == nxt_chunk) or \
           any(it[0] == nb and it[1] == nxt_chunk for it in pending):
            return
        bank_ready[nb] = False; bank_chunk[nb] = -1
        pending.append((nb, nxt_chunk, cycle + max(0, refill_delay)))

    out = []
    for _ in range(n_ticks):
        cycle += 1
        for item in [it for it in pending if it[2] <= cycle]:
            pending.remove(item)
            load(item[0], item[1])
        host_refill()
        out.append(sm)
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]; tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1; loops -= 1
        elif tc >= final:
            last = spi + 1 >= N
            if last and not p.repeat_forever:
                running = False; sm = 0
            else:
                nxt_idx = 0 if last else spi + 1
                cur_chunk = spi // bank_size
                new_chunk = nxt_idx // bank_size
                new_base = (base ^ wrap_toggle) if last else base
                crossing = last or (new_chunk != cur_chunk)
                nb = bank_of_chunk(new_chunk, new_base)
                if crossing and not (bank_ready[nb] and bank_chunk[nb] == new_chunk):
                    if raise_on_underflow:
                        raise ScanUnderflow(f"scan chunk {new_chunk} not ready at tick {tc}")
                    stalled = True            # hold; re-check next tick (the gap, if host late)
                else:
                    if crossing:
                        base = new_base
                    slot = list(points[nxt_idx]); spi = nxt_idx
                    final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
                    sm, tc, ei = (p.masks[0], 1, 1) if eff(0, slot) == 0 else (0, 0, 0)
        else:
            if ei < n and tc == eff(ei, slot):
                sm = p.masks[ei]; ei += 1
            tc += 1
    return out, stalled, spi + 1


def rtl_mirror_play(program, n_ticks: int, *, rd_lat: int = 2, fifo_depth: int = 4) -> list[int]:
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
    # PIPE = rd_lat + 1: the issue->data-valid latency INCLUDING the registered edge_raddr
    # (an issued read reaches the BRAM address port the next cycle, then the BRAM adds rd_lat).
    # The earlier pend depth of rd_lat fired `landed` one cycle early and dropped a streamed
    # edge (the emCCD 40 ms bug); it must be PIPE so a read lands rd_lat+1 cycles after issue.
    pipe = rd_lat + 1
    pend: list = [None] * pipe                    # pend[pipe-1] lands this cycle
    fetch_idx = 0

    def reseed_from(start_idx):
        # Seed FIFO_DEPTH(=RD_LAT+2) resident shadows beginning at the first
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
        pend = [None] * pipe

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

    out = []
    for _ in range(n_ticks):
        out.append(sm)
        if not running:
            continue
        if p.loop_count > 1 and loops > 1 and tc >= loop_end:
            sm = p.masks[p.loop_start_index]
            tc = eff(p.loop_start_index, slot) + 1
            ei = p.loop_start_index + 1
            loops -= 1
            reseed_from(p.loop_start_index + 1)   # loop_start output directly above
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
                continue
            elif p.repeat_forever:
                slot = _first_values(p); spi = 0
            else:
                running = False; sm = 0; continue
            final = eff(n - 1, slot); loop_end = eff_le(slot); loops = p.loop_count
            sm, tc, ei = boundary_to(True)
            continue
        # ---- normal cycle: exact RTL FIFO transfers ----
        landed_idx = pend[pipe - 1]
        # >= mirrors the RTL do_fire backstop: a head edge whose effective tick was
        # passed fires late rather than freezing the frame.  Ticks strictly increase
        # per scan point, so on a valid program this is identical to ==.
        fire_arm = (ei < n) and (len(arm) != 0) and (tc >= eff(arm[0], slot))
        if fire_arm:
            sm = p.masks[arm[0]]
            ei += 1
            arm.pop(0)               # shift down
        if landed_idx is not None:
            arm.append(landed_idx)   # land at tail (register: visible next cycle)
        tc += 1
        # issue a read iff resident + still-in-flight is below depth (popcount over ALL
        # pipe stages -- not just pend[0] -- so every in-flight read has a landing slot)
        inflight_after = sum(1 for x in pend[0:pipe - 1] if x is not None)
        occupancy = len(arm) + inflight_after
        issue = (occupancy < fifo_depth) and (fetch_idx < n)
        new_pend = [fetch_idx if issue else None] + pend[0:pipe - 1]
        if issue:
            fetch_idx += 1
        pend = new_pend
    return _apply_channel_delays(out, p)


def rtl_mirror_play_stale_seed(program, n_ticks: int, prior_count: int, *,
                               rd_lat: int = 2, fifo_depth: int = 4) -> list[int]:
    """Model the PRE-FIX hardware bug: at FIRE the seed read a STALE ``active_count``
    (the previous program's edge count, ``prior_count``) because ``active_count <=
    prog_count`` is a non-blocking write that had not committed in the seed cycle.

    The first frame is seeded with ``min(fifo_depth, prior_count[-1])`` valid shadows
    instead of the real count, so resident shadows beyond ``prior_count`` are dropped
    and the frame's tail edges never fire.  After ``final`` the engine reseeds with the
    (now-committed) real count, so only the FIRST frame is corrupted.  This exists ONLY
    to prove the fix: with ``prior_count >= len(ticks)`` it must equal
    :func:`rtl_mirror_play`; with a smaller ``prior_count`` it must drop edges.
    """
    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n = len(p.ticks)

    def eff(i, slots):
        return effective_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    arm: list[int] = []
    pipe = rd_lat + 1                 # issue->data-valid latency incl. the registered edge_raddr
    pend: list = [None] * pipe
    fetch_idx = 0
    first_frame = True

    def reseed_from(start_idx, cnt):
        nonlocal fetch_idx, pend
        arm.clear()
        # the buggy seed admits at most (cnt - start_idx) shadows, clamped to depth
        avail = max(0, cnt - start_idx)
        for k in range(min(fifo_depth, avail)):
            if start_idx + k < n:
                arm.append(start_idx + k)
        fetch_idx = start_idx + fifo_depth
        pend = [None] * pipe

    def boundary_to(cnt):
        if cnt != 0 and eff(0, slot) == 0:
            reseed_from(1, cnt)
            return p.masks[0], 1, 1
        reseed_from(0, cnt)
        return 0, 0, 0

    slot = _first_values(p)
    final = 0 if n == 0 else eff(n - 1, slot)
    running = n != 0
    sm, tc, ei = (boundary_to(prior_count) if running else (0, 0, 0))

    out = []
    for _ in range(n_ticks):
        out.append(sm)
        if not running:
            continue
        if tc >= final:
            if p.repeat_forever:
                slot = _first_values(p)
                final = eff(n - 1, slot)
                first_frame = False
                sm, tc, ei = boundary_to(n)   # subsequent frames: count is committed
                continue
            running = False; sm = 0; continue
        landed_idx = pend[pipe - 1]
        fire_arm = (ei < n) and (len(arm) != 0) and (tc >= eff(arm[0], slot))
        if fire_arm:
            sm = p.masks[arm[0]]; ei += 1; arm.pop(0)
        if landed_idx is not None:
            arm.append(landed_idx)
        tc += 1
        inflight_after = sum(1 for x in pend[0:pipe - 1] if x is not None)
        issue = (len(arm) + inflight_after < fifo_depth) and (fetch_idx < n)
        pend = [fetch_idx if issue else None] + pend[0:pipe - 1]
        if issue:
            fetch_idx += 1
    return _apply_channel_delays(out, p)


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

    # The bus rests at the SAFE mid-scale code (BUS_SAFE_VALUE in the RTL): the DAC
    # driver is offset-binary, so mid-code = true 0 V.  An unwritten bus therefore
    # idles at 0 V, and a tick-0 ramp carries IN from 0 V, exactly like the hardware.
    safe = 1 << (bus_width - 1)
    st = {"idx": 0, "value": safe, "ramp": False, "rstart": 0, "rstop": 0,
          "target": 0, "denom": 0, "accum": 0, "up": True, "step": 0, "rem": 0}

    def apply(s):
        vs, ve = endpoints(s)
        ts = eff(s.start_tick, s.start_tick_coeffs)
        te = eff(s.stop_tick, s.stop_tick_coeffs)
        if str(s.mode).lower() == "ramp" and te > ts:
            # Bresenham split (mirrors zlc_bus_apply_segment): per-tick base step =
            # delta//span with the remainder feeding the carry accumulator, so a STEEP
            # ramp moves multiple LSBs per tick and tracks floor(k*delta/span) exactly.
            span = te - ts
            delta = (ve - vs) if ve >= vs else (vs - ve)
            if span < delta:
                step, rem = divmod(delta, span)
            else:
                step, rem = 0, delta
            st.update(value=vs, ramp=True, rstart=ts, rstop=te, target=ve, denom=span,
                      accum=0, up=ve >= vs, step=step, rem=rem)
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
                st["accum"] += st["rem"]
                inc = st["step"]
                if st["accum"] >= st["denom"]:
                    st["accum"] -= st["denom"]
                    inc += 1
                if inc:
                    if st["up"]:
                        st["value"] = min(st["target"], st["value"] + inc)
                    else:
                        st["value"] = max(st["target"], st["value"] - inc)
        elif st["idx"] < len(segs) and t >= eff(segs[st["idx"]].start_tick, segs[st["idx"]].start_tick_coeffs):
            apply(segs[st["idx"]]); st["idx"] += 1
    return out


# ----------------------------------------------------------------------------
# UNDELAYED DAC-BUS VALUE evaluator (sampled, then fed to the bus delay line)
# ----------------------------------------------------------------------------
# A DELAYED DAC bus value is the bus's UNDELAYED value stream delayed by d (a per-bus
# 10-bit circular buffer -- see bus_delay_line_reference / rtl_bus_delay_line_mirror).
# To build that undelayed stream tick-by-tick (so the delay line can shift it), the
# engine reads the active bus segment combinationally at each tick:
# :func:`bus_value_at` is the COMBINATIONAL evaluator (the RTL ``zlc_bus_value_at``
# function); it is byte-identical to :func:`bus_play` when evaluated at the running
# ``time_count`` (proven in the test-suite), i.e. it is the undelayed bus stream
# re-derived combinationally.  Feed ``[bus_value_at(.., t) for t]`` into the delay line.


def bus_value_at(program, bus_index: int, phase: int, scan_point: int = 0, *,
                 bus_width: int = 10, frac_bits: int | None = None) -> int:
    """Combinational evaluation of bus ``bus_index``'s value at frame phase ``phase``
    for one scan point -- the RTL ``zlc_bus_value_at`` function.

    Walks the bus's segments in effective-tick order, holds the segment applied most
    recently at or before ``phase`` (the RTL registers the apply, so a segment whose
    effective start is ``ts`` first shows at ``ts+1``; a segment at ``ts == 0`` is
    pre-applied and shows at phase 0), resolves its value (literal or value_select
    slot read for either endpoint) and, for a RAMP active over ``[ts, te)``, returns
    the closed-form accumulator staircase value at ``phase`` (exactly the value the
    interpolating :func:`bus_play` FSM holds at that tick).  Sampling this at the
    running ``time_count`` reproduces :func:`bus_play` tick-for-tick (no FSM needed)."""

    frac = int(getattr(program, "scan_coeff_frac_bits", 8)) if frac_bits is None else frac_bits
    pts = list(getattr(program, "scan_points", None) or [])
    point = list(pts[scan_point]) if pts else []
    mask = (1 << bus_width) - 1
    segs = [s for s in (getattr(program, "bus_segments", None) or []) if int(s.bus_index) == bus_index]

    def eff(base, coeffs):
        c = [int(x) for x in (coeffs or [])]
        return effective_tick(int(base), c, point, frac) if (c and point) else int(base)

    def endval(sel, lit):
        return (int(point[sel - 1]) & mask) if sel else (int(lit) & mask)

    chosen = None
    for s in sorted(segs, key=lambda s: eff(s.start_tick, s.start_tick_coeffs)):
        ts = eff(s.start_tick, s.start_tick_coeffs)
        if ts < phase or ts == 0:        # registered apply (ts shows at ts+1); seg@0 pre-applied
            chosen = s
        else:
            break
    if chosen is None:
        return 1 << (bus_width - 1)   # rest = BUS_SAFE_VALUE (mid code = true 0 V)
    ts = eff(chosen.start_tick, chosen.start_tick_coeffs)
    te = eff(chosen.stop_tick, chosen.stop_tick_coeffs)
    vstart = endval(int(getattr(chosen, "value_select", 0)), chosen.start_value)
    stop_sel = int(getattr(chosen, "stop_value_select", getattr(chosen, "value_select", 0)))
    vstop = endval(stop_sel, chosen.stop_value)
    if str(chosen.mode).lower() == "ramp" and te > ts:
        if phase <= ts:
            return vstart
        if phase > te:
            return vstop
        denom = te - ts
        delta = abs(vstop - vstart)
        k = (phase - 1) - ts             # accumulator-increment ticks elapsed (registered)
        # Unified Bresenham closed form, ANY slope: after k stepping ticks the engine
        # has moved floor(k*delta/denom) codes (steep ramps move >1 LSB per tick).
        moves = 0 if (delta == 0 or denom == 0) else (k * delta) // denom
        if moves > delta:
            moves = delta
        return (vstart + moves) if vstop >= vstart else (vstart - moves)
    return vstop
