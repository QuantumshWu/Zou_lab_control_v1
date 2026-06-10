# Real-IP xsim verification (optional)

The Python contract tests (`tests/`) are the repo's primary, always-run verification.
These Verilog testbenches are an **optional, on-demand** cross-check that drives the REAL
`zlc_edge_streamer` engine with the **actual synthesised Xilinx block-RAM IP netlists** in
Vivado `xsim`. They are not run in CI (they need a prior Vivado build to exist) but they are
the strongest possible end-to-end proof when chasing a real-hardware discrepancy.

They were written to settle the "emCCD 2nd pulse = 40 ms" hardware report. The conclusions:

- **`tb_bram_lat.v`** — measures the port-B read latency of the real `blk_mem_gen_edge_tick`
  (symmetric 32/32) vs `blk_mem_gen_edge_mask` (asymmetric 32-write/64-read) IPs.
  Result: **both = 2 cycles, ALIGNED.** Each port B is symmetric within itself, so there is
  **no tick/mask read-latency skew.** (This disproved the 2a2c0d1/e92a78a "skew" theory.)

- **`tb_real_engine.v`** — drives the real `zlc_edge_streamer` with the real tick & mask IP
  BRAMs, preloaded with the user's exact uploaded edge table, then FIREs. Result: **two clean
  20 ms emCCD pulses** — the current RTL plays the table correctly end-to-end. (The genuine
  40 ms root cause was the stale-`active_count` FIRE seed, fixed in commit `8ff451c`; a
  pre-`8ff451c` engine in this same harness drops/corrupts the emCCD edges.)

- **`tb_edge_streamer.v`** — parameterised behavioural-latency harness used to explore
  hypothetical tick/mask skews. Confirms aligned latency 2/3 plays correctly and that an
  artificial skew, or an absolute latency exceeding `RD_LAT+1`, is what would break it — i.e.
  exactly the conditions the real IPs do **not** create.

## Running them

```sh
# 1) build once so the IP sim netlists exist under fpga/build/ps/.../ip/
fpga\build_and_program.bat --build-only

# 2) from this directory, with Vivado on PATH (xvlog/xelab/xsim):
VIV=/c/Xilinx/Vivado/2019.1/bin
IPT=../../build/ps/ps.srcs/sources_1/ip/blk_mem_gen_edge_tick
IPM=../../build/ps/ps.srcs/sources_1/ip/blk_mem_gen_edge_mask
"$VIV/xvlog" ../zlc_edge_streamer.v \
  "$IPT/sim/blk_mem_gen_edge_tick.v" "$IPT/simulation/blk_mem_gen_v8_4.v" \
  "$IPM/sim/blk_mem_gen_edge_mask.v" "$IPM/simulation/blk_mem_gen_v8_4.v" \
  tb_real_engine.v
"$VIV/xelab" work.tb_real_engine -s sreal
"$VIV/xsim" sreal -runall
```
