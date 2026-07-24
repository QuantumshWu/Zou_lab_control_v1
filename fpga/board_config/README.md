# fpga/board_config — board / platform configuration

Put the files that change **per board / per wiring / per machine** here. They are
intentionally separate from the RTL and the host code so you can reconfigure for a
different FPGA board or cabling without touching anything else.

This is the sole in-repository default. `ZLC_PS_XDC` may explicitly select an
external board file for the FPGA build **and the pulse server**; it is never a
client-side Remote-GUI override.

## What goes in here

### `board.xdc` — the board pin map (required for a real build)

The Xilinx constraints file that maps each logical output to a physical **package pin**
on your board, plus the input clock. The shipped default is the 62-output address-switch
board map. It defines, by port name:

- the **18 logical TTL outputs** (`trap`, `cooling`, `probe`, `repump`, `trig`,
  `emCCD`, the shutters, etc. — projected into the deployed `PulseTarget`),
- four 10-bit **DAC data buses** and their four **DAC latch clocks**; together
  with the TTL lanes these form the streamer's 62 raw output lanes,
- the input **`clk`** and the `GND`/unused pins.

The checked-in `board.xdc` is currently a pin/electrical map only: it contains
no `create_clock` constraint and no external DAC timing constraints.  This
known gap does **not** authorize rebuilding the frozen bitstream.  If hardware
evidence later requires a rebuild, add the real board-clock and DAC interface
constraints first, then require routed timing signoff before qualification.

Contract the build enforces (`fpga/build_and_program.bat`, `create_project.tcl`):

- it **must** define the `trig` output (`[get_ports trig]`),
- it must **not** contain unfilled `<PIN_CHxx>` placeholders,
- the build infers the **channel count + labels + pins** from it, while
  `run_server.bat` validates the deployed `PulseTarget` against the same XDC and
  publishes one `PulseTargetManifest`; the Remote GUI displays those server-owned
  package pins without reading a client XDC.

> **Analog (DAC) buses are auto-detected from the XDC, not hard-coded.** Any group of
> outputs whose names follow `base[0]`, `base[1]`, … `base[N]` (contiguous bits, ≥2 wide)
> is grouped into a logical DAC bus named `base` (e.g. `da_dipole[0..9]` → bus `da_dipole`).
> Rename or re-pin freely — the grouping is by the `[bit]` pattern, not the literal name.
> **Order matters when the XDC has no `ch[n]` constraints:** with name-only ports (the
> shipped board), the file's port ORDER defines each channel's FPGA bit index, so it must
> match `zlc_pulse_streamer_top.v`'s `out_final[]` assignments. Keep the order stable when
> editing, or switch to `set_property PACKAGE_PIN .. [get_ports {ch[N]}]` + `;# chNN <- name`
> comments (the order-independent form the inference also supports).

### `streamer_config.json` — frozen deployment geometry manifest + build inputs

The host reads this file to validate the currently deployed geometry and to
estimate a future evidence-driven rebuild. It is not a runtime tuning surface:
editing it cannot change the programmed FPGA, and the server refuses any
geometry fingerprint mismatch before upload.

| field | meaning |
|---|---|
| `fpga_part` | Vivado part string (e.g. `xc7a35tfgg484-2`). Drives the capacity estimate **and**, via `build_and_program.bat`, the synthesis target (`create_project.tcl`'s `set part`). |
| `clock_hz` | frozen sequencer clock (50 MHz → 20 ns tick); any other value is rejected because it is not in the existing geometry fingerprint. |
| `target_pct` | resource-budget target for the estimate (e.g. 90). |
| `params.max_edges` / `bank_size` / `evt_fifo_depth` | edge-table depth, resident scan ping-pong bank, per-signal delay event-FIFO depth. |
| `params.channel_count` / `num_slots` / `bus_count` / `bus_width` / widths | the rest of the geometry. |

> **Important:** the checked-in values describe the approved frozen bitstream.
> `clock_hz=50_000_000` and `slot_mul_width=25` are hard RTL facts outside the
> current fingerprint and are therefore rejected if edited. Any hardware change
> requires separate evidence, review, a complete rebuild/qualification, and an
> updated deployment record; changing this JSON alone never grants authority.

**Double-click `estimate_resources.bat`** (repo root) to print the LUT/FF/DSP/
BRAM report for this manifest. It is an estimator, not permission to alter the
frozen deployment.

## How to configure a different board

Two options:

1. **Replace** `fpga/board_config/board.xdc` with your board's pin map (same port
   names → your package pins), or
2. **Point at it without copying** — set the `ZLC_PS_XDC` environment variable to the
   absolute path of your `.xdc`:

   - PowerShell: `$env:ZLC_PS_XDC = "C:\path\to\your_board.xdc"`
   - cmd: `set ZLC_PS_XDC=C:\path\to\your_board.xdc`

After changing boards, produce a canonical `PulseTarget` from that exact XDC and
start the server with `ZLC_PS_TARGET` and `ZLC_PS_XDC` pointing at the paired files.
The server rejects any lane/signal/DAC-bit mismatch before listening. The Remote GUI
deliberately does not read its local XDC: it accepts the manifest published by the
server because client and board may be on different machines. Package pins remain
deployment/bitstream evidence; pulse execution still addresses ordered raw lanes.

## Environment overrides (so a moved board / Vivado / part never hard-breaks)

All optional — set only what differs from the defaults:

| variable | overrides | used by |
|---|---|---|
| `ZLC_PS_XDC` | the board pin map path | FPGA build/inference and `fpga/run_server.bat` manifest publication |
| `ZLC_PS_TARGET` | canonical target paired with the deployed bitstream/XDC | `fpga/run_server.bat`; published to Remote clients in the server snapshot |
| `ZLC_PS_CONFIG` | the `streamer_config.json` path | host validation/estimate, `estimate_resources.bat` |
| `ZLC_PS_FPGA_PART` | the synthesis part (else read from `streamer_config.json`) | `create_project.tcl`, capacity estimate |
| `ZLC_PS_VIVADO_BIN` | the `vivado.bat` path | build + server |

Vivado is auto-found in `C:\Xilinx\Vivado\*` / `D:\Xilinx\Vivado\*` (any version, newest
wins) or on `PATH`; set `ZLC_PS_VIVADO_BIN` for a non-standard location. The synthesis
part follows `streamer_config.json`'s `fpga_part` (or `ZLC_PS_FPGA_PART`), so moving to a
different Artix-7 retargets the build without editing the `.tcl`.

## Who reads this folder

`board.xdc`:

- `fpga/pulse_streamer/create_project.tcl` (the Vivado build)
- `fpga/build_and_program.bat` (build + program)
- `fpga/run_server.bat` / `zlc_pulse.server_app` (server-side target validation and
  package-pin manifest publication)

`fpga/run_server.bat` loads `ZLC_PS_TARGET` (default:
`zlc_pulse/assets/deployed_target.json`) together with `ZLC_PS_XDC`, validates them as
one deployment, and publishes the resulting manifest to Remote clients. The
checked-in architecture test requires the default target to equal the default XDC's
lane/name/bus projection as a whole.

`streamer_config.json` (search order: `ZLC_PS_CONFIG` env → cwd → this file):

- `fpga/pulse_streamer/host/image.py` (`load_streamer_config`, capacity estimate CLI)
- `zlc_pulse.server_app` and `zlc_neutral_atom.bootstrap._installation`
  (frozen deployment geometry/clock validation)
- `zlc_neutral_atom.timing.clock` (20 ns authoring grid projection)
- `estimate_resources.bat` (repo root, double-click capacity check)
- `fpga/build_and_program.bat` (synthesis `fpga_part`)
