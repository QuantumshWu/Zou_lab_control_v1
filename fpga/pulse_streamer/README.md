# ZLC FPGA Pulse Streamer

This directory contains the Vivado Tcl and HDL sources for the neutral-atom
runtime pulse streamer. The user-facing Windows entry points live one directory
up:

```powershell
.\fpga\build_and_program.bat
.\fpga\run_server.bat
```

The hardware profile is now 40-channel by default and by contract. The pulse
GUI may show only `ch00..ch03`, or any other subset, but the backend always
compiles against the server's full `ch00..ch39` hardware list. Hidden or missing
channels are uploaded as zero bits in the 40-bit `prog_mask`.

## Files

- `zlc_pulse_streamer.v`: runtime edge-table pulse-streamer core.
- `zlc_pulse_streamer_top_40ch.v`: 40-output top wrapper and VIO contract.
- `zlc_pulse_streamer_40ch.xdc.template`: copy this to
  `zlc_pulse_streamer_40ch.xdc` and fill all real package pins before building a
  board-ready bitstream.
- `create_project_40ch.tcl`: create the Vivado project, create VIO IP,
  synthesize, implement, and write the 40ch bitstream.
- `program_fpga_40ch.tcl`: program the FPGA with the generated `.bit` and
  `.ltx` probes.
- `check_40ch_synth.tcl`: no-XDC 40ch synthesis self-check. It verifies the
  40-bit VIO/top-level HDL path without producing a board-ready bitstream.
- `diagnose_hw_target.tcl`: non-destructive Vivado hardware-target diagnostic.

## Batch Entry Points

From the repository root:

```powershell
.\fpga\build_and_program.bat --help
.\fpga\run_server.bat --help
```

`fpga\build_and_program.bat` supports:

```powershell
.\fpga\build_and_program.bat              # build and program 40ch
.\fpga\build_and_program.bat --build-only # build 40ch only
.\fpga\build_and_program.bat --program-only
.\fpga\build_and_program.bat --check      # no-XDC 40ch synth self-check
.\fpga\build_and_program.bat --diagnose   # list Vivado hw targets/devices
```

The real build requires a completed 40ch XDC:

```powershell
copy .\fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc.template `
     .\fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc

# edit every <PIN_CHxx>, voltage standard, and any board-specific constraints
.\fpga\build_and_program.bat
```

You may also keep the board XDC elsewhere:

```powershell
$env:ZLC_PS_40CH_XDC = "D:\fpga_pin_maps\zlc_pulse_streamer_40ch_my_board.xdc"
.\fpga\build_and_program.bat
```

The script refuses to build if the XDC is missing or still contains
`<PIN_CHxx>` placeholders.

## Vivado Discovery

The two batch files search for Vivado in this order:

1. `ZLC_PS_VIVADO_BIN`, then `ZLC_VIVADO_BIN`.
2. `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`.
3. `vivado.bat` or `vivado` on `PATH`.

Vivado 2019 debug cores are path-length sensitive. `fpga\build_and_program.bat`
uses a temporary `subst` short drive when possible; a short checkout such as
`D:\ZLC` is still recommended on the FPGA computer. If your Vivado install is
elsewhere:

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

## Server

After the FPGA has been programmed:

```powershell
.\fpga\run_server.bat
```

The server defaults are:

```text
backend:  vivado-session
host:     0.0.0.0
port:     18861
clock:    100 MHz
channels: ch00 ... ch39
trigger:  ch03
max rows: 128
```

The default persistent `vivado-session` backend opens the hardware target and
loads the `.ltx` probes once, then reuses the same Vivado Tcl process for
`prepare/fire/wait_done/safe_state`.

## Runtime Contract

Python uploads an edge table:

```text
ticks: absolute FPGA clock ticks
masks: 40-bit output mask after each tick
```

During `prepare`, the backend holds reset, writes each row through VIO, writes
repeat metadata, clears `prog_we`, and releases reset. During `fire`, it toggles
`zlc_start` once; after that transition reaches the FPGA, timing is owned by the
FPGA clock.

The VIO probe contract is:

```text
probe_out0  zlc_reset           width 1
probe_out1  zlc_start           width 1
probe_out2  zlc_prog_we         width 1
probe_out3  zlc_prog_addr       width 7
probe_out4  zlc_prog_tick       width 32
probe_out5  zlc_prog_mask       width 40
probe_out6  zlc_prog_count      width 8
probe_out7  zlc_repeat_forever  width 1
probe_out8  zlc_loop_start_addr width 7
probe_out9  zlc_loop_end_tick   width 32
probe_out10 zlc_loop_count      width 32
probe_in0   zlc_running         width 1
probe_in1   zlc_done            width 1
```

Display labels such as `trap`, `cooling`, `probe`, and `qcm_trigger` are GUI
labels unless they are also actual server channel names. For the 40ch server,
the hardware names and mask bit order are `ch00..ch39`.
