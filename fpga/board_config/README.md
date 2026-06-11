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

> **Analog (DAC) buses are auto-detected from the XDC, not hard-coded.** Any group of
> outputs whose names follow `base[0]`, `base[1]`, … `base[N]` (contiguous bits, ≥2 wide)
> is grouped into a logical DAC bus named `base` (e.g. `da_dipole[0..9]` → bus `da_dipole`).
> Rename or re-pin freely — the grouping is by the `[bit]` pattern, not the literal name.
> **Order matters when the XDC has no `ch[n]` constraints:** with name-only ports (the
> shipped board), the file's port ORDER defines each channel's FPGA bit index, so it must
> match `zlc_pulse_streamer_top.v`'s `out_final[]` assignments. Keep the order stable when
> editing, or switch to `set_property PACKAGE_PIN .. [get_ports {ch[N]}]` + `;# chNN <- name`
> comments (the order-independent form the inference also supports).

### `streamer_config.json` — the reconfigurable streamer geometry + part (single source)

The one place to edit the **compile-affecting** specifics. The Python host (program
validation, capacity estimate, AXI runtime defaults) reads this file, so a geometry or
part change is made **once** here instead of in scattered constants:

| field | meaning |
|---|---|
| `fpga_part` | Vivado part string (e.g. `xc7a35tfgg484-2`). Drives the capacity estimate **and**, via `build_and_program.bat`, the synthesis target (`create_project.tcl`'s `set part`). |
| `clock_hz` | sequencer clock (50 MHz → 20 ns tick); the ns↔tick conversion + delay-µs labels. |
| `target_pct` | resource-budget target for the estimate (e.g. 90). |
| `params.max_edges` / `bank_size` / `evt_fifo_depth` | edge-table depth, resident scan ping-pong bank, per-signal delay event-FIFO depth. |
| `params.channel_count` / `num_slots` / `bus_count` / `bus_width` / widths | the rest of the geometry. |

> **Important:** `params` must match the localparams the **bitstream** was built with
> (`zlc_pulse_streamer_top.v`). Editing the JSON does **not** re-synthesize — it only
> re-aligns host validation/estimation. To actually change the geometry, change BOTH the
> `.v` parameters and this file, then rebuild. `channel_count` is normally **inferred** from
> `board.xdc`; the value here is only the offline/fallback default.

**Double-click `estimate_resources.bat`** (repo root) after editing to print a LUT/FF/DSP/
BRAM pass-fail table for the configured part — it tells you whether the part has enough
resources before you spend a Vivado run.

## How to configure a different board

Two options:

1. **Replace** `fpga/board_config/board.xdc` with your board's pin map (same port
   names → your package pins), or
2. **Point at it without copying** — set the `ZLC_PS_XDC` environment variable to the
   absolute path of your `.xdc`:

   - PowerShell: `$env:ZLC_PS_XDC = "C:\path\to\your_board.xdc"`
   - cmd: `set ZLC_PS_XDC=C:\path\to\your_board.xdc`

`ZLC_PS_XDC`, when set, overrides this folder everywhere.

## Environment overrides (so a moved board / Vivado / part never hard-breaks)

All optional — set only what differs from the defaults:

| variable | overrides | used by |
|---|---|---|
| `ZLC_PS_XDC` | the board pin map path | build, server, GUI, host inference |
| `ZLC_PS_CONFIG` | the `streamer_config.json` path | host validation/estimate, `estimate_resources.bat` |
| `ZLC_PS_FPGA_PART` | the synthesis part (else read from `streamer_config.json`) | `create_project.tcl`, capacity estimate |
| `ZLC_PS_VIVADO_BIN` | the `vivado.bat` path | build + server |

Vivado is auto-found in `C:\Xilinx\Vivado\*` / `D:\Xilinx\Vivado\*` (any version, newest
wins) or on `PATH`; set `ZLC_PS_VIVADO_BIN` for a non-standard location. The synthesis
part follows `streamer_config.json`'s `fpga_part` (or `ZLC_PS_FPGA_PART`), so moving to a
different Artix-7 retargets the build without editing the `.tcl`.

## Who reads this folder

`board.xdc` (search order: `ZLC_PS_XDC` env → this file):

- `fpga/pulse_streamer/create_project.tcl` (the Vivado build)
- `fpga/build_and_program.bat` (build + program)
- `fpga/run_server.bat` (channel-count inference for the server)
- `Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py` (`_resolve_xdc_path`)
- `pulse_gui.py` (standalone GUI default channel map)

`streamer_config.json` (search order: `ZLC_PS_CONFIG` env → cwd → this file):

- `fpga/pulse_streamer/host/image.py` (`load_streamer_config`, capacity estimate CLI)
- `Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py` (validator `DEFAULT_*`)
- `Zou_lab_control/neutral_atom/devices/axi_session.py` (`DEFAULT_PARAMS`, clock)
- `estimate_resources.bat` (repo root, double-click capacity check)
- `fpga/build_and_program.bat` (synthesis `fpga_part`)
