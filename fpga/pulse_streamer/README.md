# ZLC FPGA Pulse Streamer

Vivado Tcl and HDL sources for the neutral-atom runtime pulse streamer. The
user-facing Windows entry points live one directory up (`fpga\build_and_program.bat`,
`fpga\run_server.bat`).

This is a short subsystem pointer. For the full design tutorial (RTL walk, the
N-slot affine scan engine, the analog-bus DAC engine, the host compiler ->
upload flow, resource budgets, and the v3 roadmap) see the **FPGA manual** in
`docs/fpga_manual/`.

## Files

- `zlc_pulse_streamer.v`: runtime edge-table pulse-streamer core. Parameterized
  by `NUM_SLOTS` (affine scan slots), `CHANNEL_COUNT`, `MAX_EDGES`,
  `MAX_SCAN_POINTS`, `COEFF_FRAC_BITS`, and the analog-bus parameters.
- `zlc_pulse_streamer_top_address_switch.v`: address-switch top wrapper and VIO
  contract. `NUM_SLOTS=4`. Exposes the original XDC ports and maps them to the
  core's `out[0..61]` mask bits plus four 10-bit DAC buses.
- `create_project_address_switch.tcl`: create project + VIO IP, synth,
  implement, write bitstream.
- `program_fpga_address_switch.tcl`: program with the generated `.bit`/`.ltx`.
- `check_address_switch_synth.tcl`: no-output-pin synthesis self-check.
- `diagnose_hw_target.tcl`: non-destructive hardware-target diagnostic.

## Contract Summary

Target FPGA is the Artix-7 35T `xc7a35tfgg484-2`. The default XDC is
`references\source_archives\address_switch\...\addre.xdc` (62 outputs,
`ch00..ch61`; `emCCD=ch11/M13`, `trig=ch06/R17`). The bitstream is fixed; every
`On Pulse` uploads a fresh runtime table through the already-built VIO probes.
One edge row means "at this absolute FPGA tick, set all outputs to this mask".

Scans use named slots: each edge row stores a base tick plus `NUM_SLOTS`
fixed-point coefficients, and the FPGA computes
`effective_tick = base + (sum_j coeff_j * slot_j) >> COEFF_FRAC_BITS` while
iterating the streamed scan-point table. Analog buses upload through a separate
segment table (`bus_id, start_tick, stop_tick, start_value, stop_value, mode`)
so a ramp costs one segment, not hundreds of TTL edge rows.

Default profile: `CHANNEL_COUNT=62`, `NUM_SLOTS=4`, `MAX_EDGES=1024`,
`MAX_SCAN_POINTS=1024`, `TICK_WIDTH=32`, `COEFF_WIDTH=16`, `COEFF_FRAC_BITS=8`,
`CLOCK_HZ=50 MHz`, `RESOURCE_TARGET=70%`. Vivado `report_utilization` is the
final resource authority.

## VIO Probe Contract (NUM_SLOTS=4)

```text
probe_out0  zlc_reset               width 1
probe_out1  zlc_start               width 1
probe_out2  zlc_prog_we             width 1
probe_out3  zlc_prog_addr           width 10
probe_out4  zlc_prog_tick           width 32
probe_out5  zlc_prog_mask           width 62
probe_out6  zlc_prog_count          width 11
probe_out7  zlc_repeat_forever      width 1
probe_out8  zlc_loop_start_addr     width 10
probe_out9  zlc_loop_end_tick       width 32
probe_out10 zlc_loop_count          width 32
probe_out11 zlc_prog_tick_coeffs    width 64   (NUM_SLOTS*COEFF_WIDTH = 4*16)
probe_out12 zlc_scan_enable         width 1
probe_out13 zlc_scan_prog_we        width 1
probe_out14 zlc_scan_prog_addr      width 10
probe_out15 zlc_scan_prog_values    width 128  (NUM_SLOTS*TICK_WIDTH = 4*32)
probe_out16 zlc_scan_count          width 11
probe_out17 zlc_loop_end_coeffs     width 64   (NUM_SLOTS*COEFF_WIDTH = 4*16)
probe_out18 zlc_bus_prog_we         width 1
probe_out19 zlc_bus_prog_bus        width 2
probe_out20 zlc_bus_prog_addr       width 6
probe_out21 zlc_bus_prog_start_tick width 32
probe_out22 zlc_bus_prog_stop_tick  width 32
probe_out23 zlc_bus_prog_start_value width 10
probe_out24 zlc_bus_prog_stop_value width 10
probe_out25 zlc_bus_prog_mode       width 2
probe_out26 zlc_bus_counts          width 28
probe_in0   zlc_running             width 1
probe_in1   zlc_done                width 1
```

Keep this table in sync with the `vio_0` instantiation in
`zlc_pulse_streamer_top_address_switch.v`. A Python contract test asserts the
generated VIO widths match the Python generator.
