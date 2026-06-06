# Frontend Submodule

`Zou_lab_control.frontend` is the standalone user-interface layer: plotting,
live updates, Fluent PyQt widgets, the pulse editor, and PDF/manual rendering.
It does not own hardware actions.

For the full tutorial (widget library + layout primitives, the Edit/Preview/Scan
pulse GUI, the per-field scan-dot workflow, the plotting API, and the PDF
rendering API), see the **frontend manual** in `docs/frontend_manual/`. For
maintainer rules (Fluent visual grammar, pulse-GUI alignment contract, screenshot
QA), see `docs/MAINTAINER_NOTES.md`.

## Layout

- `pulse_gui.py`: Confocal-style pulse editor. Edits `PulseTableState` and calls
  a supplied `SequencerDevice`.
- `qt_fluent.py`: reusable Fluent/Confocal PyQt styling, widgets, and the layout
  primitives (`Metrics`, `ElidedLabel`, `FluentScanDot`, `mark_scan_field`,
  `FluentLabeledField`, `FluentFormGrid`, `measure_text_width`).
- `live.py`, `data_figure.py`, `canvas.py`: plotting and live-refresh layer.
- `content/`: manual and notebook source templates shipped with the package.
- `notes.py`: PDF/manual rendering helpers (`render_tex_pdf`,
  `render_notes_pdf`, `build_frontend_manual`).

The repository root keeps `pulse_gui.py` / `pulse_gui.bat` as thin launchers so
users can open the editor without an experiment config.

## Launcher

By default the launcher tries a sequencer server on `127.0.0.1:18861` (the
normal mode on the FPGA computer after `fpga\run_server.bat`). If that local
server is not listening, the GUI opens as an offline editor. On a control
computer pass `--remote-host`; for pure offline editing pass `--no-sequencer`.

```powershell
.\pulse_gui.bat --remote-host 192.168.0.20 --state .\pulses\camera_imaging_address_switch.json
.\pulse_gui.bat --no-sequencer --state .\pulses\camera_imaging_address_switch.json
.\pulse_gui.bat --xdc D:\pin_maps\my_board.xdc --channel-count 24
```

The launcher infers channel count, display labels, and package pins from the
FPGA XDC (fallback 62). A JSON with only a subset is aligned to the full
hardware list before upload; hidden channels are a view-only operation and are
zeroed in the uploaded masks.
