"""Cycle-accurate behavioural models of the affine edge-table engine's DIGITAL
playback, used to prove the Architecture-D BRAM+prefetch redesign is tick-exact
and gapless BEFORE hardware (there is no Verilog simulator in this repo).

Two independent implementations of the SAME engine FSM
(``fpga/pulse_streamer/zlc_pulse_streamer.v`` lines 440-526):

* :func:`reference_play` — the *combinatorial* engine that is validated today:
  every cycle it reads ``tick_mem/coeff_mem/mask_mem[edge_index]`` in-line and
  fires same-cycle on ``time_count == effective_tick`` (min edge spacing = 1 tick).

* :func:`prefetch_play` — the new Architecture-D engine: the edge table lives in
  block RAM with a synchronous read of ``read_latency`` cycles, so the engine
  keeps a small prefetch FIFO of upcoming edges plus *first/second-edge shadows*
  (and loop-start / loop-start+1 shadows) latched at program time, so the four
  gapless reload sites (start, loop-rewind, scan-advance, repeat-forever) reseed
  the FIFO instantly and back-to-back 1-tick edges still fire one per cycle.

If ``prefetch_play(...) == reference_play(...)`` for every program + scan sweep
(including 1-tick-spaced edges right at the boundaries, finite loops, repeat,
and multi-point scans), the BRAM/prefetch retiming is proven to change nothing
observable — the proof the RTL must then mirror.

Only the DIGITAL ``out`` (the 62-bit channel mask) is modelled.  In Architecture
D the analog-bus tables stay in LUTRAM and the bus engine is byte-for-byte
unchanged, so its output is identical by construction and is not re-modelled here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from .fpga_pulse_streamer import _apply_scan_tick

__all__ = [
    "EngineProgram", "reference_play", "prefetch_play", "prefetch_d1_play",
    "min_edge_spacing", "PrefetchStall",
]


class PrefetchStall(RuntimeError):
    """Raised when the prefetch FIFO underruns (edge needed before its read lands).

    A well-sized FIFO (depth >= read_latency + 1) + shadows must never hit this;
    the equivalence test treats a stall as a failure (it would be a hardware gap).
    """


@dataclass
class EngineProgram:
    """Minimal view of a RuntimeSequenceProgram for the digital playback model."""

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

    @classmethod
    def from_program(cls, program) -> "EngineProgram":
        slot_count = int(getattr(program, "slot_count", 0) or 0)
        ticks = [int(t) for t in program.ticks]
        coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in ticks])
        coeffs = [list(row) + [0] * (slot_count - len(row)) for row in coeffs]
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
        )


def _zero_slots(p: EngineProgram) -> list[int]:
    return [0] * p.slot_count


def _first_values(p: EngineProgram) -> list[int]:
    return list(p.scan_points[0]) if p.scan_points else _zero_slots(p)


def reference_play(program, n_ticks: int) -> list[int]:
    """Combinatorial reference: faithful mirror of zlc_pulse_streamer.v digital FSM."""

    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n_edges = len(p.ticks)
    scan_enable = bool(p.scan_points)
    scan_count = len(p.scan_points)

    def eff(i: int, slots: Sequence[int]) -> int:
        return _apply_scan_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_loop_end(slots: Sequence[int]) -> int:
        return _apply_scan_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    slot_active = _first_values(p)
    final_tick = 0 if n_edges == 0 else eff(n_edges - 1, slot_active)
    loop_end_active = eff_loop_end(slot_active)
    loops_remaining = p.loop_count
    scan_point_index = 0
    running = n_edges != 0
    if running and eff(0, slot_active) == 0:
        state_mask = p.masks[0]
        time_count = 1
        edge_index = 1
    else:
        state_mask = 0
        time_count = 0
        edge_index = 0

    out: list[int] = []
    for _ in range(n_ticks):
        out.append(state_mask)
        if not running:
            continue
        if p.loop_count > 1 and loops_remaining > 1 and time_count >= loop_end_active:
            state_mask = p.masks[p.loop_start_index]
            time_count = eff(p.loop_start_index, slot_active) + 1
            edge_index = p.loop_start_index + 1
            loops_remaining -= 1
        elif time_count >= final_tick:
            if scan_enable and scan_point_index + 1 < scan_count:
                nxt = list(p.scan_points[scan_point_index + 1])
                slot_active = nxt
                final_tick = eff(n_edges - 1, nxt)
                loop_end_active = eff_loop_end(nxt)
                if eff(0, nxt) == 0:
                    state_mask, time_count, edge_index = p.masks[0], 1, 1
                else:
                    state_mask, time_count, edge_index = 0, 0, 0
                loops_remaining = p.loop_count
                scan_point_index += 1
            elif p.repeat_forever:
                slot_active = _first_values(p)
                final_tick = eff(n_edges - 1, slot_active)
                loop_end_active = eff_loop_end(slot_active)
                if eff(0, slot_active) == 0:
                    state_mask, time_count, edge_index = p.masks[0], 1, 1
                else:
                    state_mask, time_count, edge_index = 0, 0, 0
                loops_remaining = p.loop_count
                scan_point_index = 0
            else:
                running = False
                state_mask = 0
        else:
            if edge_index < n_edges and time_count == eff(edge_index, slot_active):
                state_mask = p.masks[edge_index]
                edge_index += 1
            time_count += 1
    return out


def min_edge_spacing(program) -> int:
    """Smallest gap (in ticks) between consecutive effective edge ticks over ALL
    scan points -- the quantity the host must keep >= the engine's settle so the
    depth-1 prefetch read always lands before the edge fires.  Edge 0 is exempt
    (it comes from the first-edge shadow, no read).  Returns a large number if
    there are < 2 edges."""

    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    if len(p.ticks) < 2:
        return 1 << 30
    points = p.scan_points or [_zero_slots(p)]
    worst = 1 << 30
    for slots in points:
        prev = _apply_scan_tick(p.ticks[0], p.tick_slot_coeffs[0], slots, p.frac_bits)
        for i in range(1, len(p.ticks)):
            e = _apply_scan_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)
            worst = min(worst, e - prev)
            prev = e
    return worst


def prefetch_d1_play(program, n_ticks: int, *, settle: int = 2) -> list[int]:
    """Architecture-D *depth-1* engine: ``cur`` holds the current edge (edge 0 from
    the first-edge shadow, every other edge from a BRAM read that becomes valid
    ``settle`` cycles after edge_index advances).  Same combinatorial-style fire as
    the reference (``cur`` is a register, not a live BRAM read).  Raises
    :class:`PrefetchStall` if an edge is needed before its read lands (i.e. the host
    did not keep min edge spacing >= settle+1) -- which would be a hardware gap.

    For programs whose min edge spacing >= settle+1 the output is byte-identical to
    :func:`reference_play`; that equivalence + the stall guard is the proof of the
    depth-1 design used by zlc_pulse_streamer_d.v.
    """

    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n_edges = len(p.ticks)
    scan_enable = bool(p.scan_points)
    scan_count = len(p.scan_points)

    def eff(i: int, slots: Sequence[int]) -> int:
        return _apply_scan_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_loop_end(slots: Sequence[int]) -> int:
        return _apply_scan_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    # cur readiness: edge 0 is always ready (shadow); edge k>0 ready `settle`
    # cycles after edge_index was set to k.
    cur_ready_in = 0   # cycles until cur (for the current edge_index) is valid

    def arm_cur(idx: int) -> int:
        # returns the cur_ready_in for the new edge_index
        return 0 if idx == 0 else settle

    slot_active = _first_values(p)
    final_tick = 0 if n_edges == 0 else eff(n_edges - 1, slot_active)
    loop_end_active = eff_loop_end(slot_active)
    loops_remaining = p.loop_count
    scan_point_index = 0
    running = n_edges != 0
    if running and eff(0, slot_active) == 0:
        state_mask = p.masks[0]
        time_count = 1
        edge_index = 1
        cur_ready_in = arm_cur(1)
    else:
        state_mask = 0
        time_count = 0
        edge_index = 0
        cur_ready_in = arm_cur(0)

    out: list[int] = []
    for _ in range(n_ticks):
        if cur_ready_in > 0:
            cur_ready_in -= 1
        out.append(state_mask)
        if not running:
            continue
        if p.loop_count > 1 and loops_remaining > 1 and time_count >= loop_end_active:
            state_mask = p.masks[p.loop_start_index]
            time_count = eff(p.loop_start_index, slot_active) + 1
            edge_index = p.loop_start_index + 1
            loops_remaining -= 1
            cur_ready_in = arm_cur(edge_index)
        elif time_count >= final_tick:
            if scan_enable and scan_point_index + 1 < scan_count:
                nxt = list(p.scan_points[scan_point_index + 1])
                slot_active = nxt
                final_tick = eff(n_edges - 1, nxt)
                loop_end_active = eff_loop_end(nxt)
                if eff(0, nxt) == 0:
                    state_mask, time_count, edge_index = p.masks[0], 1, 1
                else:
                    state_mask, time_count, edge_index = 0, 0, 0
                loops_remaining = p.loop_count
                scan_point_index += 1
                cur_ready_in = arm_cur(edge_index)
            elif p.repeat_forever:
                slot_active = _first_values(p)
                final_tick = eff(n_edges - 1, slot_active)
                loop_end_active = eff_loop_end(slot_active)
                if eff(0, slot_active) == 0:
                    state_mask, time_count, edge_index = p.masks[0], 1, 1
                else:
                    state_mask, time_count, edge_index = 0, 0, 0
                loops_remaining = p.loop_count
                scan_point_index = 0
                cur_ready_in = arm_cur(edge_index)
            else:
                running = False
                state_mask = 0
        else:
            if edge_index < n_edges:
                if time_count == eff(edge_index, slot_active):
                    if cur_ready_in > 0:
                        raise PrefetchStall(
                            f"depth-1 prefetch underrun: edge {edge_index} needed at "
                            f"tick {time_count} but read not ready ({cur_ready_in} left); "
                            f"min edge spacing < settle+1 ({settle + 1})."
                        )
                    state_mask = p.masks[edge_index]
                    edge_index += 1
                    cur_ready_in = arm_cur(edge_index)
            time_count += 1
    return out


def prefetch_play(program, n_ticks: int, *, read_latency: int = 2, fifo_depth: int = 3) -> list[int]:
    """Architecture-D engine: edge table in BRAM (``read_latency``-cycle synchronous
    read) + a prefetch FIFO of depth ``fifo_depth`` + first/second/loop-start
    shadows.  Returns the per-tick ``out`` stream; raises :class:`PrefetchStall`
    if the FIFO underruns (which would be a hardware gap).

    The FIFO stores edge INDICES whose BRAM row (base tick / coeffs / mask) is
    resident; the effective tick is recomputed every cycle from the CURRENT
    ``slot_active`` (exactly as the RTL MAC does), so a scan-point change correctly
    re-derives effective ticks from the stored bases.
    """

    p = program if isinstance(program, EngineProgram) else EngineProgram.from_program(program)
    n_edges = len(p.ticks)
    scan_enable = bool(p.scan_points)
    scan_count = len(p.scan_points)
    if fifo_depth < read_latency + 1:
        # Not enough lookahead to hide the read latency for back-to-back edges.
        fifo_depth = read_latency + 1

    def eff(i: int, slots: Sequence[int]) -> int:
        return _apply_scan_tick(p.ticks[i], p.tick_slot_coeffs[i], slots, p.frac_bits)

    def eff_loop_end(slots: Sequence[int]) -> int:
        return _apply_scan_tick(p.loop_end_tick, p.loop_end_slot_coeffs, slots, p.frac_bits)

    # Prefetch state.
    fifo: deque[int] = deque()             # resident edge indices, in order
    inflight: list[tuple[int, int]] = []   # (edge_index, lands_at_cycle)
    fetch_idx = 0                          # next edge index to issue a read for
    cycle = 0

    def reseed(target: int) -> None:
        """Reseed the FIFO at a jump target using the (instant) shadows for
        edges target and target+1, then begin fetching target+2..."""
        nonlocal fetch_idx
        fifo.clear()
        inflight.clear()
        if target < n_edges:
            fifo.append(target)            # first-edge shadow (target)
        if target + 1 < n_edges:
            fifo.append(target + 1)        # second-edge shadow (target+1)
        fetch_idx = target + 2
        _issue_reads()

    def _issue_reads() -> None:
        """Issue BRAM reads to top the FIFO+inflight up to fifo_depth."""
        nonlocal fetch_idx
        resident = len(fifo) + len(inflight)
        while resident < fifo_depth and fetch_idx < n_edges:
            inflight.append((fetch_idx, cycle + read_latency))
            fetch_idx += 1
            resident += 1

    def _land_reads() -> None:
        """Move reads whose latency elapsed into the FIFO (in index order)."""
        ready = [item for item in inflight if item[1] <= cycle]
        ready.sort(key=lambda it: it[0])
        for item in ready:
            inflight.remove(item)
            fifo.append(item[0])

    # --- init (start), mirroring the RTL ---
    slot_active = _first_values(p)
    final_tick = 0 if n_edges == 0 else eff(n_edges - 1, slot_active)
    loop_end_active = eff_loop_end(slot_active)
    loops_remaining = p.loop_count
    scan_point_index = 0
    running = n_edges != 0
    reseed(0)
    # edge 0 either fires at tick 0 (eff==0) or waits.
    if running and eff(0, slot_active) == 0:
        state_mask = p.masks[0]
        time_count = 1
        edge_index = 1
        if fifo and fifo[0] == 0:
            fifo.popleft()                 # edge 0 consumed at seed
        _issue_reads()
    else:
        state_mask = 0
        time_count = 0
        edge_index = 0

    out: list[int] = []
    for _ in range(n_ticks):
        cycle += 1
        _land_reads()
        out.append(state_mask)
        if not running:
            continue
        if p.loop_count > 1 and loops_remaining > 1 and time_count >= loop_end_active:
            state_mask = p.masks[p.loop_start_index]
            time_count = eff(p.loop_start_index, slot_active) + 1
            edge_index = p.loop_start_index + 1
            loops_remaining -= 1
            reseed(p.loop_start_index)
            # edges loop_start and loop_start+1 are seeded; loop_start already
            # output as state_mask, so drop it from the FIFO head.
            if fifo and fifo[0] == p.loop_start_index:
                fifo.popleft()
            _issue_reads()
        elif time_count >= final_tick:
            if scan_enable and scan_point_index + 1 < scan_count:
                nxt = list(p.scan_points[scan_point_index + 1])
                slot_active = nxt
                final_tick = eff(n_edges - 1, nxt)
                loop_end_active = eff_loop_end(nxt)
                loops_remaining = p.loop_count
                scan_point_index += 1
                reseed(0)
                if eff(0, nxt) == 0:
                    state_mask, time_count, edge_index = p.masks[0], 1, 1
                    if fifo and fifo[0] == 0:
                        fifo.popleft()
                    _issue_reads()
                else:
                    state_mask, time_count, edge_index = 0, 0, 0
            elif p.repeat_forever:
                slot_active = _first_values(p)
                final_tick = eff(n_edges - 1, slot_active)
                loop_end_active = eff_loop_end(slot_active)
                loops_remaining = p.loop_count
                scan_point_index = 0
                reseed(0)
                if eff(0, slot_active) == 0:
                    state_mask, time_count, edge_index = p.masks[0], 1, 1
                    if fifo and fifo[0] == 0:
                        fifo.popleft()
                    _issue_reads()
                else:
                    state_mask, time_count, edge_index = 0, 0, 0
            else:
                running = False
                state_mask = 0
        else:
            if edge_index < n_edges:
                if not fifo or fifo[0] != edge_index:
                    raise PrefetchStall(
                        f"FIFO underrun: need edge {edge_index} at tick {time_count} "
                        f"but FIFO head is {fifo[0] if fifo else None} "
                        f"(latency={read_latency}, depth={fifo_depth})."
                    )
                if time_count == eff(edge_index, slot_active):
                    state_mask = p.masks[edge_index]
                    fifo.popleft()
                    edge_index += 1
                    _issue_reads()
            time_count += 1
    return out
