# Zou_lab_control Neutral Atom Hardware Runbook

This is the short operational hardware runbook. The default FPGA path is a
40-channel runtime pulse-streamer bitstream. The control computer sends a
`PulseSequence` or GUI `PulseTableState`; the FPGA/Vivado computer compiles it
against the full `ch00..ch39` channel list, uploads `ticks/masks` through Vivado
VIO, and the FPGA clock executes the pulses.

The GUI is only a frontend. It may display four channels, ten channels, or all
forty. At upload time the sequencer server still produces a full-width 40-bit
program; hidden or unconfigured channels are off.

## 0. Install

On both computers:

```powershell
cd C:\path\to\Zou_lab_control_v1
install_requirements.bat
```

or:

```powershell
python -m pip install -e .
```

Vivado is separate from Python requirements and only needs to be installed on
the Verilog/FPGA computer.

## 1. Build And Program On The FPGA Computer

Use a short checkout path when possible, for example `D:\ZLC`.

```powershell
cd D:\ZLC
.\fpga\build_and_program.bat --help
.\fpga\build_and_program.bat --check
```

`--check` performs a no-XDC 40ch synthesis self-check. It is useful before the
board pin map is ready. It does not create a board-ready bitstream.

For a real build, copy and complete the 40ch XDC:

```powershell
copy .\fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc.template `
     .\fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc
```

Fill every `ch00` through `ch39` package pin, voltage standard, and any
board-level constraints. Then build and program:

```powershell
.\fpga\build_and_program.bat
```

If your XDC lives elsewhere:

```powershell
$env:ZLC_PS_40CH_XDC = "D:\fpga_pin_maps\zlc_pulse_streamer_40ch_my_board.xdc"
.\fpga\build_and_program.bat
```

The script refuses to build if the XDC is missing or still contains
`<PIN_CHxx>` placeholders. It also returns nonzero if synthesis,
implementation, `.bit/.ltx` generation, or programming fails.

Generated files normally live at:

```text
fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.xpr
fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit
fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx
```

Vivado discovery order:

1. `ZLC_PS_VIVADO_BIN`, then `ZLC_VIVADO_BIN`;
2. `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`;
3. `vivado.bat` or `vivado` on `PATH`.

Manual override:

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

Vivado GUI route:

1. Open Vivado 2019.x.
2. If the project does not exist yet, run `.\fpga\build_and_program.bat --build-only`.
3. Open
   `fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.xpr`.
4. Run Synthesis, Run Implementation, then Generate Bitstream.
5. Open Hardware Manager, Open Target, Auto Connect.
6. Select the FPGA device and Program Device.
7. Choose `zlc_pulse_streamer_top_40ch.bit` and
   `zlc_pulse_streamer_top_40ch.ltx`.

If Vivado sees a Digilent target but no FPGA device, run:

```powershell
.\fpga\build_and_program.bat --diagnose
```

Then check board power, JTAG/mode jumpers, power-source jumper, cable seating,
and Hardware Manager Auto Connect.

## 2. Start The 40ch Sequencer Server

After programming:

```powershell
cd D:\ZLC
.\fpga\run_server.bat
```

Defaults:

```text
host:     0.0.0.0
port:     18861
backend:  vivado-session
clock:    100 MHz
channels: ch00 ... ch39
trigger:  ch03
```

The persistent `vivado-session` backend opens Vivado Tcl once, loads the `.ltx`
probes once, and reuses that process for `prepare/fire/wait_done/safe_state`.
Set `ZLC_PS_SERVER_BACKEND=command` only for debugging the older
subprocess-per-action path.

## 3. Pulse GUI

Root GUI entry remains separate from FPGA scripts:

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_40ch.json
```

From a control computer, replace `127.0.0.1` with the FPGA computer IP printed
by the server.

`pulses\camera_imaging_40ch.json` stores forty hardware channels but only shows
the first four by default:

```text
ch00 label trap
ch01 label cooling
ch02 label probe
ch03 label qcm_trigger
```

This is still a 40ch pulse for the FPGA. The GUI visible-channel list only
controls editor clutter. When `On Pulse` is pressed, the current state is sent
to the sequencer, compiled against `ch00..ch39`, uploaded, and fired. Channels
not present in the state or not visible in the GUI are zero in the uploaded
40-bit masks.

The camera preset compiles at 100 MHz to:

```text
ticks: 0, 200000, 210000, 212000, 2210000, 2212000
masks: 3, 1, 13, 5, 1, 0
trigger_count: 1
repeat_forever: true
```

All high bits above `ch03` are zero.

## 4. Control/qCMOS Computer

Open:

```powershell
jupyter lab tutorials\neutral_atom_hardware_quickstart.ipynb
```

Connect to the FPGA computer:

```python
exp = na.connect(
    "remote_template",
    sequencer={"host": "192.168.0.20", "port": 18861},
    open_devices=True,
)
```

Then configure imaging, run preflight, capture raw camera frames, calibrate
sitemap/threshold, detect, and scan detection time from the notebook.

## Runtime Principle

The fixed FPGA bitstream stores an edge table:

```text
ticks: absolute FPGA clock ticks
masks: full 40-bit output masks
```

During `prepare`, Python writes the table and repeat metadata through VIO while
reset is asserted. During `fire`, Python toggles `zlc_start` once. After that,
`time_count` advances on the FPGA clock and updates outputs when
`time_count == tick_mem[edge_index]`.

Repeat is metadata, not expanded rows. A GUI table with `repeat infinity` is
uploaded once with `zlc_repeat_forever=1`. A finite repeat bracket uses
`loop_start_addr`, `loop_end_tick`, and `loop_count`. This is why a simple GUI
pulse using only four visible channels can still run cleanly on the 40ch
bitstream without changing Verilog.
