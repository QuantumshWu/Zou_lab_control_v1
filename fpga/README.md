# FPGA Submodule

`fpga/` is the standalone hardware side of the ZLC pulse-streamer path. It can
be copied to the Vivado computer with the Python package and run without an
experiment configuration.

For the full hardware tutorial (RTL, the N-slot affine scan engine, the VIO
probe contract, the analog-bus DAC engine, resource budgets, and the
build/program workflow), see the **FPGA manual** in `docs/fpga_manual/`. For
capacity invariants and the v3 roadmap, see `docs/MAINTAINER_NOTES.md`.

## Layout

- `build_and_program.bat`: build/check/diagnose/program the address-switch
  pulse-streamer bitstream.
- `run_server.bat`: start the RPyC sequencer server. It opens a persistent
  Vivado/VIO session before accepting GUI or notebook clients.
- `pulse_streamer/`: HDL, Vivado Tcl, and the pulse-streamer design pointer.
- Generated Vivado projects and server state default to `fpga\build`. The
  default project is `fpga\build\address_switch`; the default server state dir is
  `fpga\build\state_address_switch`.

The root-level `pulse_gui.bat` is the frontend entry point; the FPGA batch files
stay here so hardware setup is separate from the GUI.

## Runtime Chain

```text
Pulse GUI / notebook
  -> RemoteSequencer.prepare/fire/wait_done
  -> sequencer_server on the Vivado computer
  -> VivadoPulseStreamerSession (persistent Vivado Tcl + VIO probes)
  -> zlc_pulse_streamer_top_address_switch.bit on the FPGA
```

The server compiles against the full hardware channel list inferred from the
address-switch XDC (62 controllable outputs; missing/hidden GUI channels are
zero bits). The camera-imaging preset and default trigger inference use
`ch11/emCCD/M13`; `ch06/trig/R17` is a separate output.

## Normal Use

```powershell
.\fpga\build_and_program.bat --check    # no-board HDL/VIO width + capacity self-check
.\fpga\build_and_program.bat            # build + program (default address-switch XDC)
.\fpga\run_server.bat --check-config    # print resolved project/bit/ltx/xdc/channels/clock/capacity
.\fpga\run_server.bat                    # start the persistent server (host 0.0.0.0, port 18861)
```

Default clock is 50 MHz (20 ns tick); default capacity is 1024 edge rows and
1024 scan points. Configure with `ZLC_PS_XDC`, `ZLC_PS_MAX_EDGES`,
`ZLC_PS_MAX_SCAN_POINTS`, `ZLC_PS_RESOURCE_TARGET_PCT`, `ZLC_PS_VIVADO_BIN`.

## Path Rules

Vivado 2019 debug cores are path-length sensitive. Keep the checkout short
(`D:\ZLC`). The batch files print `ZLC build root` / `ZLC project dir`; those
printed paths are the source of truth for the generated `.xpr/.bit/.ltx`. Stale
environment variables pointing into the old `fpga\pulse_streamer\build` folder
are ignored in favour of `fpga\build`. `run_server.bat` uses the interpreter in
`.zlc_python_path` before falling back to PATH; override with
`ZLC_FPGA_SERVER_PYTHON`. Set `ZLC_NO_PAUSE=1` for automation.
