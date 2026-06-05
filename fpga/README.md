# FPGA Submodule

`fpga/` is the standalone hardware side of the ZLC pulse-streamer path.  It can
be copied to the Vivado computer with the Python package and run without an
experiment configuration.

## Layout

- `build_and_program.bat`: user entry point for building and programming the
  address-switch pulse-streamer bitstream.
- `run_server.bat`: user entry point for the RPyC sequencer server.  It starts
  a persistent Vivado/VIO session before accepting GUI or notebook clients.
- `pulse_streamer/`: HDL, Vivado Tcl, and the pulse-streamer design note.
- Generated Vivado projects and server state default to `fpga\build`.  The
  default project directory is `fpga\build\address_switch`, and the default
  server state directory is `fpga\build\state_address_switch`.

The root-level `pulse_gui.bat` is the frontend entry point.  The FPGA batch
files stay here so hardware setup remains separate from the GUI.

## Runtime Chain

```text
Pulse GUI / notebook
  -> RemoteSequencer.prepare/fire/wait_done
  -> sequencer_server on the Vivado computer
  -> VivadoPulseStreamerSession
  -> Vivado Hardware Manager + VIO probes
  -> zlc_pulse_streamer_top_address_switch.bit on the FPGA
```

The GUI may show only a few logical rows.  The server still compiles against
the full hardware channel list inferred from
`references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc`.
The current default XDC has 62 controllable outputs.  Missing or hidden GUI
channels are uploaded as zero bits.

The checked-in address-switch XDC labels `ch11` as the physical `emCCD` output
and `ch06` as the physical `trig` output.  The camera-imaging preset and default
trigger inference use `ch11/emCCD/M13` for the qCMOS/external camera trigger.
`ch06/trig/R17` remains a separate output and should only be used when the pulse
JSON or lab wiring explicitly selects it.

## Path Rules

Vivado 2019 debug cores fail when the project path is too long.  The batch
files print `ZLC build root: ...` and `ZLC project dir: ...`; those printed
paths are the source of truth for the generated `.xpr/.bit/.ltx`.

If the repository path is too long and Vivado's debug path guard stops the
build, move the repo to a shorter folder such as `D:\ZLC`, or set:

```powershell
$env:ZLC_PS_PROJECT_DIR = "D:\ZLC\fpga\build\address_switch"
```

If a stale terminal environment points `ZLC_PS_PROJECT_DIR`,
`ZLC_PS_CHECK_PROJECT_DIR`, `ZLC_PS_VIVADO_BIT`, or `ZLC_PS_VIVADO_LTX` inside
the old `fpga\pulse_streamer\build` folder, the batch files ignore it and fall
back to `fpga\build`.  This prevents loading stale probes from an old build.

`install_requirements.bat` writes the installed Python path to the ignored
repository-local `.zlc_python_path`.  `run_server.bat` uses that interpreter
before falling back to PATH, so the server imports the same editable checkout
and dependencies as the GUI.  For a one-terminal override, set
`ZLC_FPGA_SERVER_PYTHON`.

## Normal Use

```powershell
.\fpga\build_and_program.bat
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

`build_and_program.bat --check` runs a no-output-pin synthesis self-check.  A
real build uses the original address-switch XDC by default.  For a different
board/cable map, set `ZLC_PS_XDC` to a completed board XDC.

`run_server.bat --check-config` prints the resolved project, `.bit`, `.ltx`,
full XDC-inferred channel list, default trigger channel, 50 MHz clock, and scan
capacity, then exits without opening a long-lived server.

Set `ZLC_NO_PAUSE=1` for automation.  Without it, the outer batch wrapper keeps
double-clicked windows readable after success or failure.

## Runtime Notes

The server starts Vivado once and keeps that session alive.  The slow
hardware-target setup therefore happens at server startup, not at the first GUI
`On Pulse`.

After the first successful prepare, the persistent session keeps a Python copy
of the uploaded edge table.  A later prepare with the same channel order and
clock rewrites only changed edge rows plus the shadow-critical rows
`0`, `loop_start_index`, and the final row.

With `repeat_forever=True`, the full uploaded table restarts forever.  A finite
repeat bracket inside that table is finite; after it finishes, any post-loop
rows run, then the full table starts again.  For camera acquisition or API
single shots, call the pulse controller with `repeat_forever=False`.
