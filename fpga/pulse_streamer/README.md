# ZLC FPGA Pulse Streamer

This directory contains a Vivado-ready pulse-streamer entry point for the
neutral-atom sequencer backend.

The first-light build is `4ch`: it drives the historical pins for
`trap/cooling/probe/qcm_trigger` from the old `address_switch` XDC. The 40
channel top is included, but its XDC must be completed with the real board
connector map before building a bitstream.

## Files

- `zlc_pulse_streamer.v`: synthesizable runtime edge-table pulse-streamer core.
- `zlc_pulse_streamer_top_4ch.v`: first-light top wrapper with 4 outputs and VIO.
- `zlc_pulse_streamer_4ch.xdc`: known clock, LED, and 4-output constraints.
- `create_project_4ch.tcl`: create Vivado project, create VIO IP, synthesize,
  implement, and write bitstream.
- `program_fpga_4ch.tcl`: program the FPGA with the generated bitstream.
- `build_4ch_bitstream.bat`: Windows one-click batch wrapper for build.
- `program_4ch_fpga.bat`: Windows one-click batch wrapper for programming.
- `simulate_4ch_core.bat` / `tb_zlc_pulse_streamer_4ch.v`: board-free xsim
  simulation of upload, start, output timing, done, and safe-off behavior.
- `zlc_pulse_streamer_top_40ch.v`: 40-output top wrapper and VIO contract.
- `zlc_pulse_streamer_40ch.xdc.template`: pin-map template for the 40-output build.
- `build_40ch_bitstream.bat`: build wrapper for the completed 40-output XDC.
- `program_40ch_fpga.bat`: programming wrapper for the 40-output bitstream.
- `check_40ch_synth.bat`: no-XDC 40-channel synthesis self-check. It verifies
  the 40-bit VIO/prog_mask/top-level HDL path but does not produce a
  board-ready bitstream.
- `start_server_4ch.bat` / `start_server_40ch.bat`: convenience launchers for
  the FPGA sequencer server.
- `vivado_env.bat` / `vivado_run_tcl.bat`: shared Vivado/Python discovery and
  short-path batch helpers.
- `smoke_test_4ch.py` / `smoke_test_4ch_upload.bat`: upload and fire a known
  4-channel pulse table for oscilloscope verification.

## Vivado Version And Path Discovery

The batch files do not require a hard-coded Vivado version. They search in this
order:

1. User override `ZLC_PS_VIVADO_BIN`.
2. Installed `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`.
3. `vivado.bat` or `vivado` on `PATH`.

This supports the local test computer with Vivado 2019.1 and the historical
`address_switch` project, which was generated with Vivado 2019.2 under a short
path such as `D:\time_sequence\address_switch`. If the FPGA computer has a
different version, set the path explicitly before running any batch file:

```bat
set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
```

Vivado 2019 debug cores can fail when the project path is too deep. The shared
batch helper automatically uses a temporary short drive letter through `subst`
when possible. A real short checkout such as `D:\ZLC` or `D:\time_sequence\ZLC`
is still recommended on the Verilog/FPGA computer.

Check the detected tools without building:

```powershell
.\fpga\pulse_streamer\vivado_env.bat
.\fpga\pulse_streamer\build_4ch_bitstream.bat --help
.\fpga\pulse_streamer\build_40ch_bitstream.bat --help
```

## Build And Program The 4-Channel First-Light Bitstream

Run this on the Verilog/FPGA computer:

```powershell
cd D:\GitHub\Zou_lab_control_v1
.\fpga\pulse_streamer\simulate_4ch_core.bat
.\fpga\pulse_streamer\build_4ch_bitstream.bat
```

The generated project is:

```text
fpga/pulse_streamer/build/zlc_pulse_streamer_4ch
```

The generated bitstream/probes are normally:

```text
fpga/pulse_streamer/build/zlc_pulse_streamer_4ch/zlc_pulse_streamer_4ch.runs/impl_1/zlc_pulse_streamer_top_4ch.bit
fpga/pulse_streamer/build/zlc_pulse_streamer_4ch/zlc_pulse_streamer_4ch.runs/impl_1/zlc_pulse_streamer_top_4ch.ltx
```

Then program the FPGA:

```powershell
.\fpga\pulse_streamer\program_4ch_fpga.bat
```

The build Tcl checks that synthesis and implementation runs completed and that
both `.bit` and `.ltx` were generated. The program Tcl also requires the `.ltx`
probe file, because the server controls this bitstream through VIO. The batch
wrappers return Vivado/Python's exit code to the shell.

In the normal bring-up flow, run this programming batch explicitly before the
smoke test and before starting the server. You can skip it only when the same
bitstream/probes were already programmed manually in Vivado Hardware Manager, or
when `ZLC_PS_VIVADO_PROGRAM_ON_RUN="1"` is intentionally used so the backend
programs the board during its first hardware action.

After programming, start the sequencer server with `ZLC_PS_VIVADO_BIT` and
`ZLC_PS_VIVADO_LTX` pointing to the two generated files above. You may set
`ZLC_PS_VIVADO_PROGRAM_ON_RUN="0"` because the board has already been programmed.

Before connecting the qCMOS camera, run the 4-channel smoke test:

```powershell
.\fpga\pulse_streamer\smoke_test_4ch_upload.bat
```

The smoke-test batch file defaults `ZLC_PS_VIVADO_PROJECT`,
`ZLC_PS_VIVADO_BIT`, and `ZLC_PS_VIVADO_LTX` to the build paths above, so it can
reload the `.ltx` probes in a fresh Vivado batch session. Override those
environment variables only if the bitstream/probe files live somewhere else.

Expected oscilloscope timing at 100 MHz:

```text
trap         high 0-10 us
cooling      high 0-3 us
probe        high 2-6 us
qcm_trigger  high 2-3 us
```

## VIO Probe Contract

Both top wrappers instantiate a Vivado VIO IP named `vio_0`:

```text
probe_out0 zlc_reset      width 1
probe_out1 zlc_start      width 1
probe_out2 zlc_prog_we    width 1
probe_out3 zlc_prog_addr  width 10
probe_out4 zlc_prog_tick  width 32
probe_out5 zlc_prog_mask  width CHANNEL_COUNT
probe_out6 zlc_prog_count width 11
probe_in0  zlc_running    width 1
probe_in1  zlc_done       width 1
```

`MAX_EDGES=1024`, so `prog_addr` is 10 bits and `prog_count` is 11 bits.
`TICK_WIDTH=32`.

The first-light core marks the edge table arrays as distributed RAM. For the
default 1024-edge design this is intentionally simple and keeps the async
edge-table read path predictable. For much larger tables, use a BRAM-oriented
synchronous pipeline instead of scaling this VIO/LUTRAM transport indefinitely.

The Python backend looks probes up by the semantic net names above first. If
the generated `.ltx` exposes Vivado's native VIO port names instead, it falls
back to these aliases:

```text
zlc_reset      -> probe_out0
zlc_start      -> probe_out1
zlc_prog_we    -> probe_out2
zlc_prog_addr  -> probe_out3
zlc_prog_tick  -> probe_out4
zlc_prog_mask  -> probe_out5
zlc_prog_count -> probe_out6
zlc_running    -> probe_in0
zlc_done       -> probe_in1
```

## Output Mapping

4-channel first-light mapping:

```text
out[0] -> trap         -> PACKAGE_PIN M17
out[1] -> cooling      -> PACKAGE_PIN F15
out[2] -> probe        -> PACKAGE_PIN N15
out[3] -> qcm_trigger  -> PACKAGE_PIN R17
```

40-channel top mapping:

```text
out[0]  -> ch[0]
out[1]  -> ch[1]
...
out[39] -> ch[39]
```

Before using the 40-channel top, copy `zlc_pulse_streamer_40ch.xdc.template` to
`zlc_pulse_streamer_40ch.xdc` and replace every placeholder with the real,
verified package pin and connector meaning. The 40-channel build script stops
early if any `<PIN_CHxx>` placeholder remains.

Then build/program:

```powershell
.\fpga\pulse_streamer\check_40ch_synth.bat
.\fpga\pulse_streamer\build_40ch_bitstream.bat
.\fpga\pulse_streamer\program_40ch_fpga.bat
```

`check_40ch_synth.bat` is safe to run before the real 40-output XDC exists; it
uses no output-pin constraints and only verifies the 40-channel HDL/VIO logic
contract. `build_40ch_bitstream.bat` is intentionally stricter: it stops until
`zlc_pulse_streamer_40ch.xdc` exists and contains no `<PIN_CHxx>` placeholders.

## Sequencer Server Environment

Example after building the 4-channel bitstream:

```powershell
cd D:\GitHub\Zou_lab_control_v1
$env:PYTHONPATH = (Get-Location).Path

# Optional. Omit this if vivado_env.bat finds the right Vivado automatically.
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
$env:ZLC_PS_VIVADO_PROJECT = "$PWD\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.xpr"
$env:ZLC_PS_VIVADO_BIT = "$PWD\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.bit"
$env:ZLC_PS_VIVADO_LTX = "$PWD\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.ltx"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "0"
$env:ZLC_PS_VIO_FILTER = 'CELL_NAME=~"*vio*"'
$env:ZLC_PS_MAX_EDGES = "1024"
$env:ZLC_PS_TICK_WIDTH = "32"
$env:ZLC_PS_CHANNEL_COUNT = "4"

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --host 0.0.0.0 `
  --port 18861 `
  --channels trap cooling probe qcm_trigger `
  --trigger-channels qcm_trigger `
  --clock-hz 100000000 `
  --state-dir D:\zlc_sequencer_state `
  --prepare-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer prepare" `
  --fire-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer fire" `
  --wait-done-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer wait_done" `
  --safe-state-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer safe_state"
```

The same 4-channel server can be started with:

```powershell
.\fpga\pulse_streamer\start_server_4ch.bat
```

For the 40-channel server after the 40-channel bitstream is built and
programmed:

```powershell
.\fpga\pulse_streamer\start_server_40ch.bat
```

The 40-channel launcher uses channels `ch00 ... ch39`, `ch03` as the trigger
channel, a 100 MHz clock, and `ZLC_PS_CHANNEL_COUNT=40`. Override
`ZLC_PS_HOST`, `ZLC_PS_PORT`, `ZLC_PS_STATE_DIR`, or the Vivado project/bit/LTX
environment variables when the hardware setup differs.

## Pulse GUI Frontend

The root launcher is independent of experiment configs:

```powershell
.\pulse_gui.bat
```

To use the GUI on the FPGA computer after `start_server_40ch.bat` is running,
open a second terminal and connect to localhost:

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_40ch.json
```

To use the GUI on the control/qCMOS computer, replace `127.0.0.1` with the IP
printed by the server. The GUI only edits pulse state and calls the attached
`RemoteSequencer.prepare/fire/wait_done/set_safe_state`; the hardware upload
still happens through the server and `fpga_pulse_streamer` backend.
