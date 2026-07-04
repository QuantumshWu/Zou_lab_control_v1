# FPGA Submodule

`fpga/` is the standalone hardware side of the ZLC pulse-streamer path. It can
be copied to the Vivado computer with the Python package and run without an
experiment configuration.

For the full hardware tutorial (the edge-table RTL, the 1-tick FIFO prefetch
pipeline, the 2-bank streaming scan window, the affine N-slot scan engine, the
analog-bus DAC engine, resource budgets, and the JTAG-to-AXI build/program
workflow), see the **FPGA manual** in `docs/fpga_manual/`. For capacity
invariants and architecture notes, see `docs/MAINTAINER_NOTES.md`.

## Layout

- `build_and_program.bat`: build/check/diagnose/program the pulse-streamer
  bitstream (`create_project.tcl` -> `program_fpga.tcl`).
- `run_server.bat`: start the RPyC sequencer server. It opens a persistent
  Vivado `hw_axi` (JTAG-to-AXI) session before accepting GUI or notebook clients.
- `pulse_streamer/`: HDL (`zlc_edge_streamer.v`, `zlc_pulse_streamer_top.v`),
  Vivado Tcl, and the `host/` Python image packer + behavioral engine model.
- Generated Vivado projects and server state default to `fpga\build`. The
  default project is `fpga\build\ps`; the short name `ps` (-> `ps.runs`) keeps
  Vivado's deep run/.Xil temp path under the Windows MAX_PATH limit while the
  build stays in-repo. The default server state dir is `fpga\build\state`.

The root-level `pulse_gui.bat` is the frontend entry point; the FPGA batch files
stay here so hardware setup is separate from the GUI.

## Runtime Chain

```text
Pulse GUI / notebook
  -> RemoteSequencer.prepare/fire/wait_done
  -> sequencer_server on the Vivado computer (jtag-axi backend)
  -> VivadoAxiStreamerSession (persistent Vivado hw_axi session)
  -> zlc_pulse_streamer_top.bit on the FPGA
```

The host packs the compiled program into a BRAM image and writes it over
JTAG-to-AXI (`axi_bram_ctrl`), then drives the CTRL register-file mailbox
(`COMMAND`/`STATUS` + the streaming-scan `BANK_READY`/`BANK*_CHUNK` handshake).
The server compiles against the full hardware channel list inferred from the
board XDC (62 controllable outputs; missing/hidden GUI channels are zero bits).
The camera-imaging preset and default trigger inference use `ch11/emCCD/M13`;
`ch06/trig/R17` is a separate output.

## Normal Use

```powershell
.\fpga\build_and_program.bat --check    # no-board HDL + capacity self-check
.\fpga\build_and_program.bat            # build + program (create_project.tcl -> program_fpga.tcl)
.\fpga\run_server.bat --check-config    # print resolved project/bit/ltx/xdc/channels/clock/capacity
.\fpga\run_server.bat                    # start the persistent server (host 0.0.0.0, port 18861)
```

Default clock is 50 MHz (20 ns tick); the minimal pulse width and resolution are
1 tick. On the 35T part the engine resolves to 4096 edge rows + a 2-bank scan
window of `bank_size` 2048 (4096 resident points) backed by UNBOUNDED host
streaming. Capacity is fixed in `fpga.pulse_streamer.host.image.solve_capacity`
(no per-build override). Configure the build with `ZLC_PS_XDC`,
`ZLC_PS_VIVADO_BIN`, and `ZLC_PS_CLOCK_HZ`.

## Path Rules

Vivado 2019 debug cores are path-length sensitive. Keep the checkout short
(`D:\ZLC`). The batch files print `ZLC build root` / `ZLC project dir`; those
printed paths are the source of truth for the generated
`impl_1\zlc_pulse_streamer_top.{bit,ltx}`. `run_server.bat` uses the interpreter
in `.zlc_python_path` before falling back to PATH; override with
`ZLC_FPGA_SERVER_PYTHON`. Set `ZLC_NO_PAUSE=1` for automation.
