# Frontend Submodule

`Zou_lab_control.frontend` is the standalone user-interface layer.  It owns
plotting, live updates, Fluent PyQt widgets, pulse editing, and document
rendering helpers.  It does not own hardware actions.

## Layout

- `pulse_gui.py`: Confocal-style pulse editor.  It edits `PulseTableState` and
  calls a supplied `SequencerDevice`.
- `qt_fluent.py`: reusable Fluent/Confocal PyQt styling and widgets.
- `live.py`, `data_figure.py`, `canvas.py`: plotting and live-refresh layer.
- `content/`: manual and notebook source templates that ship with the package.
- `notes.py`: PDF/manual rendering helpers.

The repository root keeps `pulse_gui.py` and `pulse_gui.bat` as thin standalone
launchers because users need to open the pulse editor without loading an
experiment config.

`install_requirements.bat` installs `requirements.txt`, then installs this
repository with `pip install -e .` in the selected Python/Jupyter kernel.  A
plain `python -m pip install -e .` also installs the frontend GUI/Jupyter
runtime dependencies declared in `pyproject.toml`.  After a successful install,
the batch file records the selected interpreter in the ignored local file
`.zlc_python_path`; root `pulse_gui.bat` uses that interpreter before falling
back to PATH.  For a temporary override, set `ZLC_PULSE_GUI_PYTHON`.

## Pulse GUI Contract

The hardware channel names are FPGA bit names such as `ch00..ch39`.  Display
labels such as `trap` or `qcm_trigger` are labels only.  Hiding channels changes
only the GUI view; upload still uses the sequencer's full channel order and
zeros all missing bits.

The standalone launcher infers the default channel count from the FPGA XDC.  It
falls back to `DEFAULT_PULSE_GUI_MAX_CHANNELS = 40`.  A loaded JSON that only
contains a subset such as `ch00..ch03` is aligned to that full hardware list
before upload; missing channels stay off.  The hardware list can be overridden:

```powershell
.\pulse_gui.bat --channel-count 24
.\pulse_gui.bat --xdc D:\pin_maps\my_board.xdc
.\pulse_gui.bat --max-channel-count 40
```

By default the launcher tries a sequencer server on `127.0.0.1:18861`, which is
the normal mode on the FPGA/Vivado computer after `fpga\run_server.bat` is
started.  If that default local server is not listening, the GUI opens as an
offline editor instead of exiting; the summary/status line tells the user that
no hardware backend is attached.  On a control computer, pass the FPGA computer
address with `--remote-host`; an explicit `--remote-host` is treated as a
required hardware connection and reports an error if it cannot connect.  For
pure offline editing with no backend calls, pass `--no-sequencer`; in that mode
`On Pulse` validates the edited pulse locally and `Stop Pulse` has no hardware
backend to reset.

```powershell
.\pulse_gui.bat --remote-host 192.168.0.20 --state .\pulses\camera_imaging_40ch.json
.\pulse_gui.bat --no-sequencer --state .\pulses\camera_imaging_40ch.json
```

The GUI does not expose a separate whole-table repeat switch.  The visible
repeat control is the bracket marker in the period timeline.  With no internal
bracket, the preview still shows the whole table as `∞`, matching the default
pulse-streamer behavior.  A bracket gives a finite sub-loop count inside that
overall table.  Script and camera workflows can still request finite hardware
shots through the API when they need to wait for completion.

If a finite internal repeat bracket is used while the uploaded table is run in
the default repeating mode, the FPGA finishes that bracket count and then
continues with the rest of the table.  The editor can detect table-boundary
highs with
`PulseTableState.repeat_forever_boundary_active_channels()` and shows those
channels plus the full-table restart interval in the summary line, for example
`table restart high every 1.2 us: trap`.

## PDF And Notebook Content

Manual and tutorial source lives under `content/` so docs can be regenerated
from package data.  Prefer `render_tex_pdf(tex, output_pdf)` for one-shot PDF
generation from a TeX string or TeX file.  It compiles in a temporary directory
and copies out only the final PDF, so callers do not have to manage XeLaTeX
auxiliary files.  When the input is a TeX file, sibling assets are copied into
the temporary build tree, but stale same-name outputs such as `manual.pdf`,
`.aux`, `.log`, and `.build.log` are not copied.  A failed build removes stale
`output_pdf` and writes only `output_pdf.with_suffix(".build.log")` for
diagnosis, including the case where XeLaTeX is missing or the configured
executable path is wrong.  `render_notes_pdf(..., clean_compile=True)` uses the
same clean compiler; on success `NotesBuildResult.log_path` is `None`, while
the legacy/debug `compile_notes_pdf(...)` keeps an in-place build log beside the
TeX file.  `render_latex_pdf_clean(...)` remains as a compatibility alias.
