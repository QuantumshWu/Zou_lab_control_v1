# FPGA Submodule

`fpga/` is the standalone hardware side of the ZLC pulse-streamer path.  It can
be copied to the Vivado computer with the Python package and run without an
experiment configuration.

## Layout

- `build_and_program.bat`: user entry point for building and programming the
  40-channel bitstream.
- `run_server.bat`: user entry point for the RPyC sequencer server.  It starts a
  persistent Vivado/VIO session before accepting GUI or notebook clients.
- `pulse_streamer/`: HDL, XDC, Vivado Tcl, and the pulse-streamer design note.
- Generated Vivado projects and server state default to `fpga\build`.  The
  batch files print `ZLC build root: ...` and `ZLC project dir: ...`; by
  default the bitstream project is `fpga\build\p40` and the server state is
  `fpga\build\state40`.

The root-level `pulse_gui.bat` is the frontend entry point.  The FPGA batch
files stay here so hardware setup remains separate from the GUI.

## Runtime Chain

```text
Pulse GUI / notebook
  -> RemoteSequencer.prepare/fire/wait_done
  -> sequencer_server on the Vivado computer
  -> VivadoPulseStreamerSession
  -> Vivado Hardware Manager + VIO probes
  -> zlc_pulse_streamer_top_40ch.bit on the FPGA
```

The GUI may show only a few channels, but the server compiles against the full hardware channel list.  The default full list is inferred from
`pulse_streamer/zlc_pulse_streamer_40ch.xdc` by reading `get_ports {ch[n]}`.
Missing GUI channels are uploaded as zero bits.

## Path Rules

Vivado 2019 debug cores fail when the project path is too long.  The build
script therefore uses the short project name `p40` under `fpga\build` and
checks the expected debug-core temporary path before starting the slow build:

```text
<printed by fpga\build_and_program.bat>
```

The lines `ZLC build root: ...` and `ZLC project dir: ...` in the terminal are
the source of truth for the generated `.xpr/.bit/.ltx`.  If the repository is
checked out under an unusually long path and Vivado's debug path guard stops the
build, move the repo to a shorter project folder such as `D:\ZLC`, or set:

```powershell
$env:ZLC_PS_PROJECT_DIR = "D:\ZLC\fpga\build\p40"
```

If a stale terminal environment points `ZLC_PS_PROJECT_DIR`,
`ZLC_PS_CHECK_PROJECT_DIR`, `ZLC_PS_VIVADO_BIT`, or `ZLC_PS_VIVADO_LTX` inside
the old `fpga\pulse_streamer\build` folder, the batch files ignore it and fall
back to `fpga\build`.  This is intentional; old `pulse_streamer\build` projects
were built with the wrong path/name contract and the server must not load their
stale probes file.  In particular, the server must not load a stale repo-local
`.ltx` from the old build folder.

`run_server.bat` checks this short project directory when locating the generated
`.bit` and `.ltx`.  If you programmed from a different Vivado project, set
`ZLC_PS_VIVADO_LTX` to that exact probes file before starting the server.

`install_requirements.bat` writes the installed Python path to the ignored
repository-local `.zlc_python_path`.  `run_server.bat` uses that interpreter
before falling back to PATH, so the server imports the same editable checkout
and dependencies as the GUI.  For a one-terminal override, set
`ZLC_FPGA_SERVER_PYTHON`; paths with spaces are supported.  For the legacy
command form, set `ZLC_PY_CMD`.

## Normal Use

```powershell
.\fpga\build_and_program.bat
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

On success, `build_and_program.bat` prints a completion message and waits for
the user to close the window.  Set `ZLC_NO_PAUSE=1` for automation.

`run_server.bat --check-config` prints the resolved project, `.bit`, `.ltx`,
channel list, and trigger channel, then exits without opening a long-lived
server.  Use it after a build to confirm the server will use the same 40ch
artifact path before starting the real server.

The outer batch wrapper also prints a completion line and waits before closing
after a successful config check or after the long-running server exits.  This
keeps double-clicked windows readable; set `ZLC_NO_PAUSE=1` for automation.

The server starts Vivado once and keeps that session alive.  The slow Vivado
hardware-target setup therefore happens at server startup rather than on the
first GUI `On Pulse`.

After the first successful prepare, the persistent session keeps a Python copy
of the uploaded edge table.  A later prepare with the same channel order only
rewrites edge rows whose `tick` or `mask` changed, plus the shadow-critical rows
`0`, `loop_start_index`, and the final row.  That makes `pulse.x` scans and
small exposure tweaks cheaper than a full table upload while preserving the FPGA
first-edge, loop-start, and final-tick shadows.  A prepare that changes only
loop metadata still rewrites the new loop-start row after staging the new
`loop_start_addr`, so the FPGA cannot reuse a stale loop-start shadow.

## Scope Notes

The FPGA repeats exactly the metadata it receives.  With `repeat_forever=True`,
the full uploaded table restarts forever.  A finite repeat bracket inside that
table is finite; after it finishes, any post-loop rows run, then the full table
starts again.  On an oscilloscope this can look like a periodic extra pulse if
period 0 contains load/cooling/probe states.  For a steady scope train, make the
whole table be the steady train, or use finite readout/camera shots.
