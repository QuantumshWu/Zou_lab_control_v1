"""Cycle-accurate behavioural models of the FINAL affine edge-table engine
(``fpga/pulse_streamer/zlc_edge_streamer.v``), used to prove tick-exactness +
gaplessness BEFORE hardware (the second pre-hardware layer is the xsim
testbenches in ``fpga/pulse_streamer/sim/`` running the real RTL).

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
  ping-pong window of ``bank_size`` points; the sole host observer refills the
  idle bank behind the engine cursor.  This model verifies the active refill
  handshake against reference and proves STALL (hold, never a wrong point) on a
  late refill.  Production treats any observed UNDERFLOW as a failed run; the
  FPGA remains the autonomous owner of point timing.

The RTL combines the edge FIFO and the scan ping-pong; each is verified here
independently and against the same ``reference_play`` ground truth.

The OUTPUT delay differs by signal kind.  A TTL channel is a per-bit EVENT SCHEDULER:
:func:`delay_line_reference` is the exact stream-shift ground truth (out[t]=in[t-d], 0 before
fire) and :func:`rtl_delay_line_mirror` is its cycle-exact register mirror.  A DAC bus is an
INSTRUCTION-LEVEL SEGMENT delay: :func:`bus_delay_line_reference` is the same out[t]=in[t-d]
ground truth (safe = BUS_SAFE_VALUE before fire), and :func:`rtl_bus_segment_delay_mirror`
mirrors the RTL that captures each RESOLVED segment (:func:`bus_undelayed_and_log`) and re-plays
it d ticks later -- so a delayed dense ramp costs ONE descriptor, not ~span events.  Both use the
32-bit delay field (a d past it raises :class:`DelayTooLargeError`).  ``reference_play`` /
``prefetch_play`` / ``rtl_mirror_play`` apply the channel delay line as a post-play shift via
:func:`_apply_channel_delays`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "EngineProgram", "effective_tick", "reference_play", "prefetch_play",
    "streaming_scan_play", "rtl_mirror_play", "bus_play",
    "PrefetchStall", "ScanUnderflow", "DelayTooLargeError",
    "delay_line_reference", "bus_delay_line_reference",
    "rtl_delay_line_mirror", "rtl_bus_segment_delay_mirror",
    "bus_undelayed_and_log", "bus_value_at",
]

# The delay cap (32-bit field), the FIFO depths, AND the affine-MAC slot-operand width all have
# ONE SOURCE = fpga/board_config/streamer_config.json (the same file the RTL header + validator
# read), so this cycle mirror never drifts from the synthesized bitstream.  The except-branch
# fallbacks mirror the shipped config for the offline (import-failure) path.
try:
    from fpga.pulse_streamer.host.image import load_streamer_config as _load_cfg
    _CFG_RAW = _load_cfg()
    _CFG = _CFG_RAW.get("params")                                  # a StreamerParams
    #: OUTPUT delay cap (in ticks): the 32-bit delay field, SAME for TTL channels and DAC buses;
    #: a delay past it raises DelayTooLargeError (working limit is toggles IN FLIGHT <= the depths).
    TTL_DELAY_MAX_TICKS = int(_CFG.ttl_delay_max_ticks)
    EVT_FIFO_DEPTH = int(_CFG.evt_fifo_depth)                      # per-channel TTL event FIFO depth
    BUS_EVT_FIFO_DEPTH = int(_CFG.bus_evt_fifo_depth)              # per-bus segment FIFO depth
    #: Affine-MAC slot operand width -- MUST match zlc_edge_streamer.v SLOT_MUL_WIDTH (config
    #: localparam).  The per-slot scan value is multiplied by a 16-bit coeff in one DSP48E1 (25x18),
    #: so the slot operand is the low SLOT_MUL_WIDTH bits as signed (bounds the raw scan value; the
    #: coeff still scales it to the full 32-bit tick range).
    SLOT_MUL_WIDTH = int(_CFG_RAW.get("slot_mul_width", 25))
except Exception:
    TTL_DELAY_MAX_TICKS = (1 << 31) - 1
    EVT_FIFO_DEPTH = 64
    BUS_EVT_FIFO_DEPTH = 32
    SLOT_MUL_WIDTH = 25


class DelayTooLargeError(ValueError):
    """An effective channel/bus delay exceeds the 32-bit delay field."""


class PrefetchStall(RuntimeError):
    """Edge FIFO underran (a well-sized FIFO + shadows must never hit this)."""


class ScanUnderflow(RuntimeError):
    """The host did not refill the next scan bank before the engine reached it."""


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
    for c, s in zip(coeffs, slots, strict=True):
        total += int(c) * _narrow_slot(s)
    return int(base_tick) + (total >> int(frac_bits))


# ----------------------------------------------------------------------------
# PHYSICAL CHANNEL DELAY -- a literal OUTPUT delay line (the final, correct model)
# ----------------------------------------------------------------------------
# A channel delay is NOT baked into the edge ticks; it is a per-channel delay on the engine
# OUTPUT:  output_delayed[t] = output_undelayed[t - d], zero before fire.  The hardware is the
# per-signal EVENT SCHEDULER (see zlc_edge_streamer): each toggle is queued at {t + d - 1} and
# pops d ticks later, so storage scales with toggles IN FLIGHT (not with d) and the delay range is
# the full 32-bit field.  d=0 is exact passthrough; before the first scheduled toggle the gated
# output holds its FIRE-time 0 (silent startup, for free).  These reference functions express the
# ideal out[t]=in[t-d] semantics; the cycle-exact event-scheduler register mirrors are below.
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
    The hardware queues resolved segment descriptors once per bus and replays the same ramp
    stepper after the shared delay, so out[t] = in[t-d].  d=0 is exact passthrough."""
    d = int(delay)
    out = []
    for t in range(len(undelayed_bus)):
        s = t - d
        out.append(int(undelayed_bus[s]) if s >= 0 else int(safe_value))
    return out


# ----------------------------------------------------------------------------
# CYCLE-EXACT REGISTER MIRRORS of the literal delay-line hardware (no Verilog sim)
# ----------------------------------------------------------------------------
# These reproduce the EXACT registers of zlc_edge_streamer.v's EVENT SCHEDULER so a divergence
# from delay_line_reference / bus_delay_line_reference flags an RTL bug.  TTL and DAC use the SAME
# mechanism (per-signal event FIFO against the free-running g_time):
#
#   TTL (rtl_delay_line_mirror): per channel, on a toggle at t push {t + d - 1, new_level} into an
#   EVT_FIFO_DEPTH-deep FIFO; when g_time reaches the head time pop it into the output register
#   (visible at t + d).  d==1 is served by the prev-undelayed register; d==0 bypasses.
#
#   DAC (rtl_bus_segment_delay_mirror): INSTRUCTION-LEVEL, one per-bus FIFO (g_busseg) of RESOLVED
#   segment descriptors, NOT per-bit value events.  On each apply push {emit = g_time + d, ramp
#   descriptor}; a delayed re-player pops it at emit and RE-RUNS the ramp stepper d ticks later, so
#   the whole DAC value shifts coherently.  d==1 is the prev register; before the first emit the
#   bus holds BUS_SAFE_VALUE.  Storage scales with SEGMENTS in flight (<= the FIFO depth), so a
#   delayed dense/long ramp is ONE descriptor -- density/swing no longer overflow.
#
#   TTL storage scales with value-change events in flight per bit; DAC with segments in flight.
#   Both are out[t]=in[t-d] before any overflow, == the *_reference functions (TTL safe 0, DAC safe
#   = BUS_SAFE_VALUE), with an in-flight-count capacity bound instead of a delay-length cap.


def _check_delay_cap(d: int, cap: int, what: str) -> int:
    d = int(d)
    if d < 0 or d > cap:
        raise DelayTooLargeError(
            f"{what} delay {d} ticks exceeds the 32-bit delay field cap {cap} "
            f"(~{cap * 20e-9:.1f}s); reduce the delay.")
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
    :class:`DelayTooLargeError` (the 32-bit field cap)."""
    cds = {int(b): _check_delay_cap(d, depth, f"channel bit {b}")
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
        # The RTL derives both guards from the pre-clock FIFO count.  A full
        # queue therefore rejects a push even when its head pops on this same
        # edge; the host capacity validator rejects that closed-window
        # boundary before deployment, but this literal mirror must still match
        # the frozen registers for an out-of-contract input sequence.
        for b, d in cds.items():
            q = queues[b]
            push_has_space = len(q) < evt_depth
            if q and q[0][0] == t:
                evt_out[b] = q.pop(0)[1]
            if d >= 2 and (((cur ^ prev) >> b) & 1):
                if push_has_space:              # overflow guard: drop (validator prevents)
                    q.append((t + d - 1, (cur >> b) & 1))
        prev = cur
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
    ground truth); the RTL realises the same with a per-channel event scheduler, proven
    equal by ``rtl_delay_line_mirror``."""
    if not p.channel_delays or not any(p.channel_delays):
        return out
    cds = {b: int(d) for b, d in enumerate(p.channel_delays) if int(d)}
    return delay_line_reference(out, cds)


def _zero(p: EngineProgram) -> list[int]:
    return [0] * p.slot_count


def _first_values(p: EngineProgram) -> list[int]:
    return list(p.scan_points[0]) if p.scan_points else _zero(p)


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
# scan ping-pong refill model (current sole-host-observer contract)
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
    the full N-point sweep.  A late refill makes the
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
             bus_width: int = 10, frac_bits: int | None = None,
             seed_value: int | None = None, apply_log: list | None = None,
             carry_out: list | None = None) -> list[int]:
    """Cycle-accurate mirror of zlc_edge_streamer.v's per-bus DAC engine
    (zlc_bus_start_table + zlc_bus_step + zlc_bus_apply_segment): the interpolating
    ramp and the affine segment ticks.  Returns bus_out at each tick for one scan point
    -- the bus-path counterpart of reference_play (which covers only the digital edges).

    A ramp ALWAYS ramps from the engine's CURRENT value register (#ramp-carry), carried in
    from the previous period.  The frame enters at idle 0 V on the FIRST fire, or at
    ``seed_value`` -- the value held at the end of the previous frame / scan-point -- when the
    caller chains frames to model the cross-wrap carry (a looping [ramp V, hold V] then
    converges to FLAT V; a scanned ramp staircases from the prior scan point)."""

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

    # The bus rests at the SAFE mid-scale code (BUS_SAFE_VALUE in the RTL): the DAC driver is
    # offset-binary, so mid-code = true 0 V.  The frame enters at idle 0 V on the first fire, or
    # at ``seed_value`` (the previous frame / scan-point's end value) when chaining the carry.
    safe = 1 << (bus_width - 1)
    st = {"idx": 0, "value": safe if seed_value is None else (int(seed_value) & mask),
          "ramp": False, "rstart": 0, "rstop": 0,
          "target": 0, "denom": 0, "accum": 0, "up": True, "step": 0, "rem": 0}

    def apply(s, t_apply):
        vs, ve = endpoints(s)         # vs = scanned/baked START endpoint; ve = TARGET (stop) value
        ts = eff(s.start_tick, s.start_tick_coeffs)
        te = eff(s.stop_tick, s.stop_tick_coeffs)
        if str(s.mode).lower() == "ramp" and te > ts:
            # Bresenham split (mirrors zlc_bus_apply_segment): per-tick base step =
            # delta//span with the remainder feeding the carry accumulator, so a STEEP
            # ramp moves multiple LSBs per tick and tracks floor(k*delta/span) exactly.
            # The ramp START: a SCANNED-start ramp (value_select set) reads its slot; otherwise the
            # ramp carries from the engine's CURRENT register value -- the default (#ramp-carry).
            vstart = vs if int(getattr(s, "value_select", 0)) else st["value"]
            span = te - ts
            delta = (ve - vstart) if ve >= vstart else (vstart - ve)
            if span < delta:
                step, rem = divmod(delta, span)
            else:
                step, rem = 0, delta
            st.update(value=vstart, ramp=True, rstart=ts, rstop=te, target=ve, denom=span,
                      accum=0, up=ve >= vstart, step=step, rem=rem)
        else:
            st.update(value=ve, ramp=False, accum=0)
        if apply_log is not None:
            # The RESOLVED segment the SEGMENT-DELAY re-player will re-run: a snapshot of the
            # engine state right after apply (carried/scanned start value + ramp step/rem/denom/
            # target/up), tagged with the tick it applied at.  Delaying by d = replaying THIS same
            # descriptor d ticks later (rstart/rstop shifted +d), so the delayed staircase is
            # byte-identical to the undelayed one shifted -- the density-free instruction-level delay.
            apply_log.append((int(t_apply), dict(st)))

    if segs and eff(segs[0].start_tick, segs[0].start_tick_coeffs) == 0:
        # seg@tick0 is PRE-applied before the loop, so its value is visible AT frame tick 0 (not
        # tick+1 like a mid-frame apply); log it as applied one tick earlier so the delayed re-player
        # shows it at emit+1 == frame-tick-0 + d.
        apply(segs[0], -1); st["idx"] = 1

    out = []
    for t in range(n_ticks):
        out.append(st["value"])           # registered bus_out value at this tick
        if st["ramp"]:
            if t >= st["rstop"]:
                st["value"] = st["target"]; st["ramp"] = False; st["accum"] = 0
                if st["idx"] < len(segs) and eff(segs[st["idx"]].start_tick, segs[st["idx"]].start_tick_coeffs) <= t:
                    apply(segs[st["idx"]], t); st["idx"] += 1
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
            apply(segs[st["idx"]], t); st["idx"] += 1
    if carry_out is not None:
        # The engine's VALUE REGISTER after the last tick's transition -- the #ramp-carry value the
        # NEXT frame's ramp starts from (NOT out[-1], which is the pre-transition output sample).
        carry_out.append(st["value"])
    return out


def _segment_replay_step(st: dict) -> None:
    """Advance ONE tick of a resolved-segment descriptor's staircase, in place -- the exact
    :func:`bus_play` stepping (Bresenham carry accumulator, saturate at target at rstop), shared by
    the live engine and the SEGMENT-DELAY re-player so the two can never drift.  ``st`` is a live/
    delayed engine snapshot (value/ramp/rstart/rstop/target/denom/accum/up/step/rem) and ``st['t']``
    is the current absolute tick."""
    t = st["t"]
    if st["ramp"]:
        if t >= st["rstop"]:
            st["value"] = st["target"]; st["ramp"] = False; st["accum"] = 0
        elif t > st["rstart"] and st["denom"]:
            st["accum"] += st["rem"]
            inc = st["step"]
            if st["accum"] >= st["denom"]:
                st["accum"] -= st["denom"]; inc += 1
            if inc:
                st["value"] = (min(st["target"], st["value"] + inc) if st["up"]
                               else max(st["target"], st["value"] - inc))


def bus_undelayed_and_log(program, bus_index: int, n_ticks: int, scan_point: int = 0, *,
                          bus_width: int = 10, frac_bits: int | None = None,
                          scan_point_seq: Sequence[int] | None = None):
    """The undelayed bus value stream over ``n_ticks`` PLUS the resolved-segment apply log, chaining
    frames with the #ramp-carry for a ``repeat_forever`` program so a delay window can span many
    frames (the RTL re-applies the segment table at every wrap; g_time free-runs across it).  Returns
    ``(stream, log)`` where ``log`` = ``[(apply_tick, resolved_state_snapshot), ...]``.

    ``scan_point_seq`` (one scan-point index per frame, cycled) drives a repeat_forever SCAN so the
    resolved segment values differ frame-to-frame -- the delay log captures the ALREADY-RESOLVED
    values the live engine emitted, so replaying them delayed stays scan-correct across the sweep."""
    repeat = bool(getattr(program, "repeat_forever", False))
    frame = int(getattr(program, "loop_end_tick", 0) or 0) or int(program.ticks[-1])
    if (not repeat or frame <= 0 or n_ticks <= frame) and not scan_point_seq:
        log: list = []
        stream = bus_play(program, bus_index, n_ticks, scan_point, bus_width=bus_width,
                          frac_bits=frac_bits, apply_log=log)
        # FINITE done tail: at done (== final_tick == frame) the engine snaps the undelayed bus to
        # BUS_SAFE_VALUE (zlc_bus_clear_runtime), visible the next tick.  A DELAYED bus must therefore
        # DRAIN to SAFE d ticks later (out[t]=in[t-d]), NOT hold its last programmed value forever.
        # Model that snap: force the undelayed stream to SAFE past done, and log a SAFE-hold EDGE apply
        # at done so the segment-delay re-player captures + drains it -- mirrors zlc_bus_capture_safe_hold
        # in the RTL, so bus_delay_line_reference (via this stream) and rtl_bus_segment_delay_mirror
        # (via this log) both drain in lock-step instead of both holding a stale value.
        if not repeat and frame > 0 and n_ticks > frame:
            safe = 1 << (bus_width - 1)
            for t in range(frame + 1, n_ticks):
                stream[t] = safe
            log.append((frame, {"value": safe, "ramp": False, "rstart": frame, "rstop": frame,
                                 "target": safe, "denom": 0, "accum": 0, "up": True, "step": 0, "rem": 0}))
        return stream, log
    stream: list = []
    log = []
    seed = None
    origin = 0
    fk = 0
    while len(stream) < n_ticks:
        flen = min(frame, n_ticks - len(stream))
        flog: list = []
        carry: list = []
        pt = scan_point_seq[fk % len(scan_point_seq)] if scan_point_seq else scan_point
        fstream = bus_play(program, bus_index, flen, pt, bus_width=bus_width,
                           frac_bits=frac_bits, seed_value=seed, apply_log=flog, carry_out=carry)
        fk += 1
        stream.extend(fstream)
        frame_end = origin + flen
        for (ta, snap) in flog:                       # offset this frame's apply ticks to absolute time
            s2 = dict(snap)
            s2["rstart"] = int(snap["rstart"]) + origin
            s2["rstop"] = int(snap["rstop"]) + origin
            # The live engine plays only ticks [origin, frame_end); at the wrap it re-inits (stops
            # any active ramp, HOLDS the carried value).  Record the frame boundary so the delayed
            # re-player FREEZES this descriptor there instead of ramping to its own rstop into the
            # next frame's idle window (a ramp with rstop == loop_end_tick is truncated, never snaps
            # to target).
            s2["frame_end"] = frame_end
            log.append((int(ta) + origin, s2))
        # The carry into the next frame is the engine's VALUE REGISTER after the frame's last tick
        # (bus_play's carry_out), NOT the last OUTPUT sample (fstream[-1] drops a final-tick apply) and
        # NOT bus_value_at (which recomputes from SAFE, ignoring this cross-frame #ramp-carry seed).
        seed = carry[0] if carry else seed
        origin += flen
    return stream[:n_ticks], log


def rtl_bus_segment_delay_mirror(program, bus_index: int, delay: int, n_ticks: int,
                                 scan_point: int = 0, *, bus_width: int = 10,
                                 seg_depth: int | None = None, frac_bits: int | None = None,
                                 scan_point_seq: Sequence[int] | None = None,
                                 return_occupancy: bool = False):
    """Cycle-exact mirror of the NEW per-bus SEGMENT-DESCRIPTOR delay (the instruction-level DAC
    delay implemented by the per-bus segment-descriptor FIFO and re-player).

    Instead of buffering one event per per-tick value-CHANGE (a ramp = ~span events), the delay
    captures each RESOLVED segment the live engine applies -- ``{emit = apply_tick + d, descriptor}``
    -- into a shallow per-BUS FIFO, and a delayed re-player pops it at ``emit`` and RE-RUNS the ramp
    (:func:`_segment_replay_step`), holding ``BUS_SAFE_VALUE`` until the first emit (first-frame
    correct) and continuing the last d ticks past the program end (the done-tail).  Storage is
    therefore O(segments in flight) = segments/frame x ceil(d/frame), INDEPENDENT of ramp DENSITY --
    a dense ``0->1012 over 500 ticks`` ramp is ONE descriptor, not ~500 events.

    Returns the delayed value stream, byte-identical to :func:`bus_delay_line_reference` of the
    undelayed :func:`bus_undelayed_and_log` stream whenever no more than ``seg_depth`` descriptors
    are ever in flight; a push into a full FIFO is DROPPED (the RTL overflow guard, which the host
    validator prevents).  With ``return_occupancy`` also returns the peak in-flight descriptor count
    (the depth the program actually needs)."""
    d = int(delay)
    undelayed, log = bus_undelayed_and_log(program, bus_index, n_ticks, scan_point,
                                           bus_width=bus_width, frac_bits=frac_bits,
                                           scan_point_seq=scan_point_seq)
    if d == 0:
        return (list(undelayed), 0) if return_occupancy else list(undelayed)

    safe = 1 << (bus_width - 1)
    cap = (1 << 62) if seg_depth is None else int(seg_depth)

    # Push each captured apply at emit = apply_tick + d; a full FIFO drops (overflow guard).  A
    # descriptor sits in flight from its apply_tick until the re-player consumes it at emit (d ticks),
    # so peak occupancy = the most applies inside any d-window (segments/frame x ceil(d/frame)).
    emits: dict[int, list] = {}
    peak = 0
    live = 0
    window: deque = deque()             # emit ticks of descriptors still in flight
    for (t_apply, snap) in log:
        # Like g_busseg, push/pop eligibility is computed from the pre-edge
        # count.  Preserve that detail even though the validator rejects a
        # full+simultaneous-pop schedule on the formal path.
        push_has_space = live < cap
        while window and window[0] <= t_apply:        # retire descriptors already emitted
            window.popleft(); live -= 1
        if not push_has_space:
            continue                    # OVERFLOW: drop this segment (diverges from reference)
        window.append(t_apply + d); live += 1
        peak = max(peak, live)
        shifted = dict(snap)
        shifted["rstart"] = int(snap["rstart"]) + d
        shifted["rstop"] = int(snap["rstop"]) + d
        # Frame boundary (shifted by d): past it the delayed ramp FREEZES (holds), matching the live
        # engine's per-frame truncation.  A finite / single-frame descriptor has no boundary -> never
        # freeze (a huge sentinel).
        shifted["frame_end"] = (int(snap["frame_end"]) + d) if "frame_end" in snap else (1 << 62)
        emits.setdefault(t_apply + d, []).append(shifted)

    player = {"value": safe, "ramp": False, "rstart": 0, "rstop": 0, "target": 0,
              "denom": 0, "accum": 0, "up": True, "step": 0, "rem": 0, "started": False,
              "t": 0, "frame_end": 1 << 62}
    out = []
    for t in range(n_ticks):
        # Output the REGISTERED value first, then transition -- exactly the live engine's
        # output-before-apply/step order, so a descriptor emitted at ``emit`` becomes visible at
        # ``emit+1`` (== the undelayed apply showing at apply_tick+1, shifted by d).
        out.append(player["value"] if player["started"] else safe)
        player["t"] = t
        pending = emits.get(t)
        if pending:                                   # apply descriptor(s) scheduled at this tick
            for shifted in pending:
                for k in ("value", "ramp", "rstart", "rstop", "target", "denom", "accum", "up", "step", "rem", "frame_end"):
                    player[k] = shifted[k]
                player["started"] = True
        elif player["started"] and t < player["frame_end"]:  # advance the staircase, FROZEN at the wrap
            _segment_replay_step(player)
    return (out, peak) if return_occupancy else out


# ----------------------------------------------------------------------------
# UNDELAYED DAC-BUS VALUE evaluator (closed-form combinational cross-check)
# ----------------------------------------------------------------------------
# A DELAYED DAC bus value is the bus's UNDELAYED value stream delayed by d.  The INSTRUCTION-LEVEL
# segment delay builds that undelayed stream from the interpolating FSM (:func:`bus_undelayed_and_log`
# = :func:`bus_play` + the resolved-segment apply log), and re-plays each captured segment d ticks
# later (:func:`rtl_bus_segment_delay_mirror`).  :func:`bus_value_at` is an INDEPENDENT closed-form
# evaluator of the same undelayed value: it reads the active bus segment combinationally at a phase
# and returns the accumulator staircase value there, byte-identical to :func:`bus_play` sampled at
# the running ``time_count`` (proven in the test-suite).  It is a verification cross-check of the
# per-tick bus value, not the delay's data path.


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

    safe = 1 << (bus_width - 1)
    def seg_stop(s):
        return endval(int(getattr(s, "stop_value_select", getattr(s, "value_select", 0))), s.stop_value)
    # Track the value CARRIED INTO the chosen segment: idle (mid code) at the frame start, else the
    # previous (superseded) segment's stop.  A ramp starts from THIS, not a baked endpoint (#ramp-carry).
    chosen = None
    vstart = safe
    for s in sorted(segs, key=lambda s: eff(s.start_tick, s.start_tick_coeffs)):
        ts = eff(s.start_tick, s.start_tick_coeffs)
        if ts < phase or ts == 0:        # registered apply (ts shows at ts+1); seg@0 pre-applied
            if chosen is not None:
                vstart = seg_stop(chosen)
            chosen = s
        else:
            break
    if chosen is None:
        return safe                   # rest = BUS_SAFE_VALUE (mid code = true 0 V)
    ts = eff(chosen.start_tick, chosen.start_tick_coeffs)
    te = eff(chosen.stop_tick, chosen.stop_tick_coeffs)
    vstop = seg_stop(chosen)
    if int(getattr(chosen, "value_select", 0)):        # a SCANNED-start ramp overrides the carry
        vstart = endval(int(chosen.value_select), chosen.start_value)
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
