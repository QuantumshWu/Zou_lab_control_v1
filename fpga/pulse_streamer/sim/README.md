# Current xsim checks

These optional Vivado `xsim` benches exercise the checked-in frozen RTL. They
are hardware-development evidence, not part of normal experiment startup and
not permission to rebuild or program a board.

There is currently no tracked runner that turns every bench's textual result
into a process-level pass/fail result.  Several benches print `**FAIL**` or
`TB RESULT: FAIL` and then call `$finish`; a zero simulator exit code alone is
therefore not evidence of success.  Until an approved runner checks the exact
PASS marker and rejects any FAIL marker, retain the complete transcript and
review it explicitly.  The normal Python test suite does not run Vivado.

The maintained benches are self-contained except where explicitly noted:

- `tb_1tick.v`, `tb_edge_streamer.v`, `tb_gapsweep.v`, and `tb_loop.v` cover
  edge prefetch, one-tick playback, gaps, and repeat boundaries.
- `tb_scan_wrap.v` covers resident two-bank scan wrap behavior.
- `tb_delay_sched.v`, `tb_delay_compact.v`, `tb_evt_depth.v`, and
  `tb_bus_delay.v` cover the current 32-bit TTL event and DAC segment delay
  schedulers.
- `tb_ramp_scan.v`, `tb_da_clk_phase.v`, and `tb_da_ttl_align.v` cover DAC
  ramps, latch timing, and TTL/DAC alignment.
- `tb_real_engine.v` uses the generated edge BRAM simulation models.
- `tb_t_ff.v` is the retained full-top consecutive-FIRE regression. Its
  committed `replay_t.vh` and `replay_t_frame.vh` are a literal current-layout
  host image; the bench checks that repeated runs produce identical first
  frames through the CTRL decoder, bus loader, engine, and output mapping.

The former captured-session replay (`tb_full_top.v`/`replay_image.vh`) was
removed because it encoded an obsolete register map and had no current
generator. A stale historical trace is not valid deployment evidence.

## Running a bench

Build artifacts for Xilinx IP simulation models are required. A build is only
performed inside the separately approved evidence-driven hardware workflow.
With those artifacts already available, invoke `xvlog`, `xelab`, and `xsim`
from this directory, for example:

```sh
VIV=/c/Xilinx/Vivado/2019.1/bin
IPT=../../build/ps/ps.srcs/sources_1/ip/blk_mem_gen_edge_tick
IPM=../../build/ps/ps.srcs/sources_1/ip/blk_mem_gen_edge_mask
"$VIV/xvlog" ../zlc_edge_streamer.v \
  "$IPT/sim/blk_mem_gen_edge_tick.v" \
  "$IPT/simulation/blk_mem_gen_v8_4.v" \
  "$IPM/sim/blk_mem_gen_edge_mask.v" \
  "$IPM/simulation/blk_mem_gen_v8_4.v" \
  tb_real_engine.v
"$VIV/xelab" work.tb_real_engine -s sreal
"$VIV/xsim" sreal -runall
```

This particular `tb_real_engine.v` example prints diagnostic pulse widths and
a final `DONE` note but has no self-checking PASS marker.  Its transcript must
be inspected against the stated expected widths; it is not a stand-alone pass
oracle.  For self-checking benches, absence of the exact success marker or any
`FAIL` text is failure even when `xsim` itself exits successfully.

Every bench must use the checked-in `zlc_geometry.vh` geometry. A bench that
needs a different geometry is a hardware change proposal, not a valid oracle
for the frozen deployment.
