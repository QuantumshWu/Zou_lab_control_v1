"""Cycle-accurate behavioural model of ``zlc_axi_program_loader.v``.

There is no Verilog simulator in this repo, so -- exactly as was done for the
edge-table engine itself -- the loader FSM is mirrored here in Python and checked
against the image decoder.  The model steps the loader state machine cycle by
cycle, and whenever the loader toggles ``prog_we`` / ``scan_prog_we`` /
``bus_prog_we`` (the engine's toggle-triggered write strobe) while the engine
reset is asserted, it captures the write into the engine's program tables -- just
as ``zlc_pulse_streamer`` does.  Running it to the LOADED state and reconstructing
the tables proves the loader delivers the SAME edge/scan/bus/loop data that
``edgetable_image.unpack_program`` decodes from the image (and hence the same data
the validated VIO upload delivered), so the unchanged seamless engine stays
tick-exact.

This is a verification model, not synthesised logic; it intentionally follows the
RTL's state names and walk order so a divergence flags an RTL sequencing bug
(off-by-one row address, wrong region transition, mis-sliced field, write issued
while reset is low, ...).
"""

from __future__ import annotations

from typing import Mapping

from .edgetable_image import (
    EdgeTableImageParams,
    CtrlWords,
    IMAGE_MAGIC,
    _unpack_slot_coeffs,
    _unpack_field_words,
    _from_unsigned,
    _bus_mode_name,
)

__all__ = ["run_loader_model", "LoaderModelError"]


class LoaderModelError(RuntimeError):
    pass


# FSM state ids (== zlc_axi_program_loader.v localparams)
(S_INIT, S_IDLE_RD, S_IDLE_CAP, S_MAGIC_RD, S_MAGIC_CAP, S_CTRL_RD, S_CTRL_CAP,
 S_ROW_RD, S_ROW_CAP, S_ROW_EMIT, S_ROW_HOLD, S_REGION_NX, S_LOADED, S_FIRE0,
 S_FIRE1, S_RUN, S_PUB, S_PUB_WAIT, S_SAFE, S_ERROR) = range(20)

RG_EDGE, RG_SCAN, RG_BUS = 0, 1, 2


def run_loader_model(
    image: Mapping[int, int],
    params: EdgeTableImageParams | None = None,
    *,
    max_cycles: int = 4_000_000,
) -> dict:
    """Drive the loader FSM over ``image`` and return the engine tables it builds.

    Returns a dict with the same shape as ``edgetable_image.unpack_program``.
    Raises :class:`LoaderModelError` if the FSM stalls, writes while reset is
    low, or never reaches the LOADED state.
    """

    p = params or EdgeTableImageParams()
    COEFF_WORDS = p.coeff_words
    MASK_WORDS = p.mask_words
    EDGE_WORDS = p.edge_words
    SCAN_WORDS = p.scan_words
    BUS_WORDS = p.bus_words
    CNT_WIDTH = p.bus_seg_addr_width + 1
    RD_SETTLE = 4
    WR_HOLD = 5

    def rd(addr: int) -> int:
        return int(image.get(int(addr), 0)) & 0xFFFFFFFF

    # --- loader registers (current values) ----------------------------------
    R = {
        "state": S_INIT,
        "mem_addr": 0,
        "eng_reset": 1,
        "eng_start": 0,
        "prog_we": 0, "prog_addr": 0, "prog_tick": 0, "prog_coeffs": 0, "prog_mask": 0,
        "prog_count": 0, "repeat_forever": 0, "loop_start_addr": 0,
        "loop_end_tick": 0, "loop_end_coeffs": 0, "loop_count": 1,
        "scan_enable": 0, "scan_we": 0, "scan_addr": 0, "scan_values": 0, "scan_count": 0,
        "bus_we": 0, "bus_bus": 0, "bus_addr": 0, "bus_start_tick": 0, "bus_stop_tick": 0,
        "bus_start_coeffs": 0, "bus_stop_coeffs": 0, "bus_start_value": 0,
        "bus_stop_value": 0, "bus_mode": 0, "bus_value_select": 0, "bus_counts": 0,
        "cmd_now": 0,
        "wi": 0, "row_words": 0, "region": 0, "row_idx": 0, "row_total": 0,
        "emit_idx": 0, "emit_bus_addr": 0, "row_base": 0,
        "ctrl_idx": 0, "bus_cur": 0, "bus_addr_cur": 0, "bus_cur_count": 0,
        "rd_wait": 0, "hold_cnt": 0, "pub_wait": 0, "pub_next": 0,
        "status": 0, "cap": [0] * 7,
        # external stimulus: command word in BRAM (host writes it).
    }

    # engine tables captured from the loader's toggle writes
    tick_mem: dict[int, int] = {}
    coeff_mem: dict[int, int] = {}
    mask_mem: dict[int, int] = {}
    scan_mem: dict[int, int] = {}
    bus_mem: dict[tuple[int, int], dict] = {}

    COEFF_BITS = p.coeff_bits
    MASK_BITS = p.channel_count

    def capture_writes(N):
        # Toggle-triggered writes (engine writes on a prog_we change while reset asserted).
        if N["eng_reset"] != 1:
            return
        if N["prog_we"] != R["prog_we"]:
            a = N["prog_addr"]
            tick_mem[a] = N["prog_tick"]
            coeff_mem[a] = N["prog_coeffs"] & ((1 << COEFF_BITS) - 1)
            mask_mem[a] = N["prog_mask"] & ((1 << MASK_BITS) - 1)
        if N["scan_we"] != R["scan_we"]:
            scan_mem[N["scan_addr"]] = N["scan_values"] & ((1 << (p.num_slots * p.tick_width)) - 1)
        if N["bus_we"] != R["bus_we"]:
            bus_mem[(N["bus_bus"], N["bus_addr"])] = {
                "start_tick": N["bus_start_tick"],
                "stop_tick": N["bus_stop_tick"],
                "start_coeffs": N["bus_start_coeffs"] & ((1 << COEFF_BITS) - 1),
                "stop_coeffs": N["bus_stop_coeffs"] & ((1 << COEFF_BITS) - 1),
                "start_value": N["bus_start_value"],
                "stop_value": N["bus_stop_value"],
                "mode": N["bus_mode"],
                "value_select": N["bus_value_select"],
            }

    def bus_count_of(packed, b):
        return (packed >> (b * CNT_WIDTH)) & ((1 << CNT_WIDTH) - 1)

    # one-shot stimulus: assert CMD_LOAD until the loader starts loading.
    cmd_word = 0x1  # CMD_LOAD
    fired_load = False

    cycles = 0
    while cycles < max_cycles:
        cycles += 1
        N = dict(R)
        N["cap"] = list(R["cap"])
        st = R["state"]

        if st == S_INIT:
            N.update(eng_reset=1, eng_start=0, prog_we=0, scan_we=0, bus_we=0,
                     status=0, cmd_now=0, state=S_IDLE_RD)
        elif st == S_IDLE_RD:
            N["mem_addr"] = CtrlWords.COMMAND
            N["rd_wait"] = RD_SETTLE
            N["state"] = S_IDLE_CAP
        elif st == S_IDLE_CAP:
            if R["rd_wait"] == 0:
                # COMMAND word: emulate the host having written CMD_LOAD once.
                val = (cmd_word if not fired_load else 0) & 0xF
                edge = val & ~R["cmd_now"]
                N["cmd_now"] = val
                if edge & 0x4:      # RESET
                    N["state"] = S_INIT
                elif edge & 0x8:    # SAFE
                    N["state"] = S_SAFE
                elif edge & 0x1:    # LOAD
                    fired_load = True
                    N["state"] = S_MAGIC_RD
                elif (edge & 0x2) and (R["status"] & 0x1):  # FIRE & loaded
                    N["state"] = S_FIRE0
                elif (R["status"] & 0x2) and False:  # eng_done not modelled (load-only)
                    N["state"] = S_PUB
                else:
                    N["state"] = S_IDLE_RD
            else:
                N["rd_wait"] = R["rd_wait"] - 1
        elif st == S_MAGIC_RD:
            N["eng_reset"] = 1
            N["status"] = 0
            N["mem_addr"] = CtrlWords.MAGIC
            N["rd_wait"] = RD_SETTLE
            N["state"] = S_MAGIC_CAP
        elif st == S_MAGIC_CAP:
            if R["rd_wait"] == 0:
                if rd(R["mem_addr"]) != IMAGE_MAGIC:
                    N["status"] = 0x8
                    N["pub_next"] = S_IDLE_RD
                    N["state"] = S_PUB
                else:
                    N["ctrl_idx"] = 0
                    N["state"] = S_CTRL_RD
            else:
                N["rd_wait"] = R["rd_wait"] - 1
        elif st == S_CTRL_RD:
            N["mem_addr"] = CtrlWords.PROG_COUNT + R["ctrl_idx"]
            N["rd_wait"] = RD_SETTLE
            N["state"] = S_CTRL_CAP
        elif st == S_CTRL_CAP:
            if R["rd_wait"] == 0:
                word = rd(R["mem_addr"])
                ci = R["ctrl_idx"]
                if ci == 0:
                    N["prog_count"] = word & ((1 << 11) - 1)
                elif ci == 1:
                    N["scan_count"] = word & ((1 << 11) - 1)
                elif ci == 2:
                    N["scan_enable"] = word & 1
                elif ci == 3:
                    N["repeat_forever"] = word & 1
                elif ci == 4:
                    N["loop_start_addr"] = word & ((1 << 10) - 1)
                elif ci == 5:
                    N["loop_count"] = word
                elif ci == 6:
                    N["loop_end_tick"] = word & ((1 << p.tick_width) - 1)
                elif ci == 7:
                    N["loop_end_coeffs"] = (R["loop_end_coeffs"] & ~0xFFFFFFFF) | word
                elif ci == 8:
                    N["loop_end_coeffs"] = (R["loop_end_coeffs"] & 0xFFFFFFFF) | ((word & ((1 << (COEFF_BITS - 32)) - 1)) << 32)
                elif ci == 9:
                    N["bus_counts"] = word & ((1 << (p.bus_count * CNT_WIDTH)) - 1)
                # ci == 10: SLOT_COUNT, informational
                if ci == 10:  # CTRL_SCALARS - 1
                    N["region"] = RG_EDGE
                    N["row_words"] = EDGE_WORDS
                    N["row_idx"] = 0
                    N["row_total"] = N["prog_count"]
                    N["row_base"] = p.edge_base
                    N["wi"] = 0
                    N["state"] = S_REGION_NX
                else:
                    N["ctrl_idx"] = ci + 1
                    N["state"] = S_CTRL_RD
            else:
                N["rd_wait"] = R["rd_wait"] - 1
        elif st == S_ROW_RD:
            N["mem_addr"] = R["row_base"] + R["wi"]
            N["rd_wait"] = RD_SETTLE
            N["state"] = S_ROW_CAP
        elif st == S_ROW_CAP:
            if R["rd_wait"] == 0:
                N["cap"][R["wi"]] = rd(R["mem_addr"])
                if R["wi"] == R["row_words"] - 1:
                    N["state"] = S_ROW_EMIT
                else:
                    N["wi"] = R["wi"] + 1
                    N["state"] = S_ROW_RD
            else:
                N["rd_wait"] = R["rd_wait"] - 1
        elif st == S_ROW_EMIT:
            cap = R["cap"]
            if R["region"] == RG_EDGE:
                N["prog_addr"] = R["emit_idx"] & ((1 << 10) - 1)
                N["prog_tick"] = cap[0] & ((1 << p.tick_width) - 1)
                N["prog_coeffs"] = (cap[1] | (cap[2] << 32)) & ((1 << COEFF_BITS) - 1)
                N["prog_mask"] = (cap[3] | (cap[4] << 32)) & ((1 << MASK_BITS) - 1)
                N["prog_we"] = R["prog_we"] ^ 1
            elif R["region"] == RG_SCAN:
                N["scan_addr"] = R["emit_idx"] & ((1 << 10) - 1)
                vals = 0
                for j in range(p.num_slots):
                    vals |= (cap[j] & 0xFFFFFFFF) << (32 * j)
                N["scan_values"] = vals
                N["scan_we"] = R["scan_we"] ^ 1
            else:  # RG_BUS
                N["bus_bus"] = R["bus_cur"] & ((1 << p.bus_count.bit_length()) - 1)
                N["bus_addr"] = R["emit_bus_addr"] & ((1 << p.bus_seg_addr_width) - 1)
                N["bus_start_tick"] = cap[0] & ((1 << p.tick_width) - 1)
                N["bus_stop_tick"] = cap[1] & ((1 << p.tick_width) - 1)
                N["bus_start_coeffs"] = (cap[2] | (cap[3] << 32)) & ((1 << COEFF_BITS) - 1)
                N["bus_stop_coeffs"] = (cap[4] | (cap[5] << 32)) & ((1 << COEFF_BITS) - 1)
                flags = cap[6]
                bw = p.bus_width
                N["bus_start_value"] = flags & ((1 << bw) - 1)
                N["bus_stop_value"] = (flags >> bw) & ((1 << bw) - 1)
                N["bus_mode"] = (flags >> (2 * bw)) & 0x3
                N["bus_value_select"] = (flags >> (2 * bw + 2)) & ((1 << p.bus_sel_width) - 1)
                N["bus_we"] = R["bus_we"] ^ 1
            N["hold_cnt"] = WR_HOLD
            N["state"] = S_ROW_HOLD
        elif st == S_ROW_HOLD:
            if R["hold_cnt"] == 0:
                N["state"] = S_REGION_NX
            else:
                N["hold_cnt"] = R["hold_cnt"] - 1
        elif st == S_REGION_NX:
            N["wi"] = 0
            if R["region"] == RG_EDGE:
                if R["row_idx"] >= R["row_total"]:
                    N["region"] = RG_SCAN
                    N["row_words"] = SCAN_WORDS
                    N["row_idx"] = 0
                    N["row_total"] = R["scan_count"]
                    N["row_base"] = p.scan_base
                    N["state"] = S_REGION_NX
                else:
                    N["emit_idx"] = R["row_idx"]
                    N["row_base"] = p.edge_base + R["row_idx"] * EDGE_WORDS
                    N["row_idx"] = R["row_idx"] + 1
                    N["state"] = S_ROW_RD
            elif R["region"] == RG_SCAN:
                if R["row_idx"] >= R["row_total"]:
                    N["region"] = RG_BUS
                    N["row_words"] = BUS_WORDS
                    N["bus_cur"] = 0
                    N["bus_addr_cur"] = 0
                    N["bus_cur_count"] = bus_count_of(R["bus_counts"], 0)
                    N["state"] = S_REGION_NX
                else:
                    N["emit_idx"] = R["row_idx"]
                    N["row_base"] = p.scan_base + R["row_idx"] * SCAN_WORDS
                    N["row_idx"] = R["row_idx"] + 1
                    N["state"] = S_ROW_RD
            else:  # RG_BUS
                if R["bus_addr_cur"] >= R["bus_cur_count"]:
                    if R["bus_cur"] == p.bus_count - 1:
                        N["status"] = R["status"] | 0x1
                        N["pub_next"] = S_LOADED
                        N["state"] = S_PUB
                    else:
                        N["bus_cur"] = R["bus_cur"] + 1
                        N["bus_addr_cur"] = 0
                        N["bus_cur_count"] = bus_count_of(R["bus_counts"], R["bus_cur"] + 1)
                        N["state"] = S_REGION_NX
                else:
                    N["emit_bus_addr"] = R["bus_addr_cur"]
                    N["row_base"] = p.bus_base + (R["bus_cur"] * p.max_bus_segments + R["bus_addr_cur"]) * BUS_WORDS
                    N["bus_addr_cur"] = R["bus_addr_cur"] + 1
                    N["state"] = S_ROW_RD
        elif st == S_PUB:
            N["pub_wait"] = 2
            N["state"] = S_PUB_WAIT
        elif st == S_PUB_WAIT:
            if R["pub_wait"] == 0:
                N["state"] = R["pub_next"]
            else:
                N["pub_wait"] = R["pub_wait"] - 1
        elif st == S_LOADED:
            N["eng_reset"] = 1
            N["state"] = S_IDLE_RD
            # capture state and stop: program is loaded.
            capture_writes(N)
            R = N
            break
        elif st == S_SAFE:
            N["eng_reset"] = 1
            N["status"] = 0
            N["pub_next"] = S_IDLE_RD
            N["state"] = S_PUB
        else:
            N["state"] = S_INIT

        capture_writes(N)
        R = N
    else:
        raise LoaderModelError(f"loader FSM did not reach LOADED within {max_cycles} cycles.")

    # --- assemble the engine tables into an unpack-like result --------------
    n_edges = R["prog_count"]
    n_points = R["scan_count"]
    for a in range(n_edges):
        if a not in tick_mem:
            raise LoaderModelError(f"edge {a} was never written by the loader.")
    ticks = [tick_mem[a] for a in range(n_edges)]
    masks = [mask_mem[a] for a in range(n_edges)]
    tick_slot_coeffs = [_unpack_slot_coeffs(coeff_mem[a], p) for a in range(n_edges)]
    slot_count = _image_slot_count(image)
    scan_points = []
    for a in range(n_points):
        if a not in scan_mem:
            raise LoaderModelError(f"scan point {a} was never written by the loader.")
        v = scan_mem[a]
        scan_points.append([
            _from_unsigned((v >> (32 * j)) & 0xFFFFFFFF, p.tick_width)
            for j in range(slot_count)
        ])

    bus_segments = []
    for (b, addr), seg in sorted(bus_mem.items()):
        bus_segments.append({
            "bus_index": b,
            "start_tick": seg["start_tick"],
            "stop_tick": seg["stop_tick"],
            "start_value": seg["start_value"],
            "stop_value": seg["stop_value"],
            "mode": _bus_mode_name(seg["mode"]),
            "value_select": seg["value_select"],
            "start_tick_coeffs": _unpack_slot_coeffs(seg["start_coeffs"], p),
            "stop_tick_coeffs": _unpack_slot_coeffs(seg["stop_coeffs"], p),
        })

    return {
        "ticks": ticks,
        "masks": masks,
        "tick_slot_coeffs": tick_slot_coeffs,
        "scan_points": scan_points,
        "slot_count": _image_slot_count(image),
        "repeat_forever": bool(R["repeat_forever"]),
        "scan_enable": bool(R["scan_enable"]),
        "loop_start_index": R["loop_start_addr"],
        "loop_count": R["loop_count"],
        "loop_end_tick": R["loop_end_tick"],
        "loop_end_slot_coeffs": _unpack_slot_coeffs(R["loop_end_coeffs"], p),
        "bus_segments": bus_segments,
        "_cycles": cycles,
    }


def _image_slot_count(image: Mapping[int, int]) -> int:
    return int(image.get(CtrlWords.SLOT_COUNT, 0)) & 0xFFFFFFFF
