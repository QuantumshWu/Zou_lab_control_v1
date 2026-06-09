# fpga/board_config — board / platform configuration

Put the files that change **per board / per wiring / per machine** here. They are
intentionally separate from the RTL and the host code so you can reconfigure for a
different FPGA board or cabling without touching anything else.

This is the **default** location the toolchain reads from. (The old
`references/source_archives/address_switch/...` copy is deprecated and no longer read —
don't edit it.)

## What goes in here

### `board.xdc` — the board pin map (required for a real build)

The Xilinx constraints file that maps each logical output to a physical **package pin**
on your board, plus the input clock. The shipped default is the 62-output address-switch
board map. It defines, by port name:

- the **62 TTL outputs** (named: `trap`, `cooling`, `probe`, `repump`, `trig`,
  `emCCD`, the shutters, etc. — these names become the GUI/API channel labels),
- the **DAC clock(s)** (`da_clk[...]`) and any analog-bus pins,
- the input **`clk`** and the `GND`/unused pins.

Contract the build enforces (`fpga/build_and_program.bat`, `create_project.tcl`):

- it **must** define the `trig` output (`[get_ports trig]`),
- it must **not** contain unfilled `<PIN_CHxx>` placeholders,
- the host infers the **channel count + labels + pins** from it
  (`infer_xdc_channel_count` / `_labels` / `_pins`), so the GUI shows the right
  channels even with no hardware attached.

## How to configure a different board

Two options:

1. **Replace** `fpga/board_config/board.xdc` with your board's pin map (same port
   names → your package pins), or
2. **Point at it without copying** — set the `ZLC_PS_XDC` environment variable to the
   absolute path of your `.xdc`:

   - PowerShell: `$env:ZLC_PS_XDC = "C:\path\to\your_board.xdc"`
   - cmd: `set ZLC_PS_XDC=C:\path\to\your_board.xdc`

`ZLC_PS_XDC`, when set, overrides this folder everywhere.

## Who reads this folder (default search order: `ZLC_PS_XDC` env → this file)

- `fpga/pulse_streamer/create_project.tcl` (the Vivado build)
- `fpga/build_and_program.bat` (build + program)
- `fpga/run_server.bat` (channel-count inference for the server)
- `Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py` (`_resolve_xdc_path`)
- `pulse_gui.py` (standalone GUI default channel map)
