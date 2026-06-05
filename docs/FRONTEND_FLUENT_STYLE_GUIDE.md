# Frontend Fluent Style Guide

This guide is for maintainers and future coding agents. User manuals should stay tutorial-like; keep implementation warnings and review notes here.

## Source Of Truth

The PyQt visual source of truth is the historical Confocal GUI Fluent layer:

- `references/source_archives/Confocal_GUIv2_refactored_v6/Confocal_GUIv2_refactored/Confocal_GUIv2/gui/base.py`
- `references/source_archives/Confocal_GUIv2_refactored_v6/Confocal_GUIv2_refactored/Confocal_GUIv2/gui/_app_bootstrap.py`
- `references/source_archives/Confocal_GUIv2_refactored_v6/Confocal_GUIv2_refactored/Confocal_GUIv2/gui/gui_device.py`
- `references/source_archives/Confocal_GUIv2_refactored_v6/Confocal_GUIv2_refactored/Confocal_GUIv2/gui/gui_individual.py`
- `references/source_archives/Confocal_GUIv2_refactored_v6/Confocal_GUIv2_refactored/Confocal_GUIv2/gui/gui_task.py`
- `references/source_archives/Confocal_GUIv2_refactored_v6/Confocal_GUIv2_refactored/Confocal_GUIv2/device/base.py`

Before adding or redesigning a PyQt frontend, read those files first. For pulse-related work also read the full `PulseGUI`, `StateUIManager`, and `DragContainer` sections in `gui_device.py`. The reusable implementation in this repo lives in `Zou_lab_control/frontend/qt_fluent.py`; extend that module instead of creating one-off Qt styles in each GUI.

## Visual Rules

- Use Segoe UI, 12 pt text, background `#F3F3F3`, text `#323130`, accent `#77AADD`, radius 4 px, and soft card shadows.
- Use `FluentWindow` or `run_fluent_window` for desktop PyQt windows when a top-level shell is needed. The optional `qframelesswindow` package should be used when available.
- Use `FluentGroupBox` for white card panels with the grey title pill. Keep the original no-border, soft-shadow frame style; do not add hard outlines just to make panels more visible, and do not replace the pulse editor with plain tables.
- Use `FluentTabWidget` as a single tab leaf: the selected tab should visually connect to the page pane, and child sections such as channel names, delay/X, period cards, and channel view should remain independent `FluentGroupBox` or `FluentFrame` cards inside that leaf.
- When `FluentGroupBox` or `FluentFrame` cards live inside a tab pane or scroll viewport, give the layout a small shadow margin around the cards. If a card shadow disappears, fix the parent layout spacing or clipping first; do not change the shared Fluent CSS.
- Keep form controls on the Confocal rhythm: mostly 30 px high rows, narrow labels, compact spacing, and fixed-format controls where resizing would make timing/channel rows jump.
- Use `FluentComboBox`, `FluentSpinBox` / `FluentDoubleSpinBox`, and `FluentScrollArea` so popup lists, repeat counters, numeric controls, and scrollbars share the same Fluent selection, hover, and thin-scrollbar styling. Closed combo boxes should ignore mouse-wheel events so dataset scrolling does not accidentally change units or hidden-channel selections.
- Use `set_fluent_scale()` / `scaled_px()` for fixed PyQt geometry. Do not hard-code a second scaling system in a GUI module.
- Long button labels must fit at the active DPI. Prefer explicit line breaks inside fixed-size Confocal buttons over letting text overflow.

## Pulse GUI Layout Contract

The pulse GUI is a frontend for `PulseTableState` and a passed-in sequencer. It must not create a separate hardware-control layer.

The expected layout is:

1. Top status bar with status dot, file/run state label, and compact summary.
2. `Channel Names and Duration` panel for pulse name, total duration, and visible channel display names.
3. `Delay and Scan` panel for minimal step, active scan parameters, linked scan table file, per-channel delay, unit, scan-bind dot buttons, and `X` clear control. Duration, delay, and analog-bus value fields are numeric by default. Pressing the adjacent dot binds that field to a named scan parameter, replaces the edit text with the parameter name, turns the line edit orange/read-only, and lists it in the active scan parameter row. Pressing the same dot again unbinds the field and restores the previous value. The scan file is a named table, for example a text file headed `# vars: camera_exposure_ns(ns), trig_delay(ns)`. The GUI must keep symbolic expressions such as `100000-camera_exposure_ns` visible instead of expanding scan rows into many period columns. The Name panel's left column is always the hardware channel name; the right column is an optional display label. Do not auto-map generic hardware channels such as `ch00` to physical meanings. When the user edits a display label, the Delay row and period checkbox text may follow that label while saved/compiled pulses still use the hardware channel name. `X` clears all period states for that channel but does not hide it. Hiding is handled by the global `Hide Off` control, which hides channels with no period on and preserves stored label and delay values.
4. Horizontal period-card timeline, not a grid/table. Each card owns duration, unit selector, period position such as `Period 1/5`, and channel checkboxes. The unit selector is shown without a separate `Unit` label to keep period cards narrow.
5. Bottom Confocal-style large buttons for stop/off, on-pulse, wait, add/remove period, repeat bracket, save pulse, load, and temporary left-panel collapse. `On Pulse` prepares/uploads the current pulse state and then starts it; there is no separate sync button in the GUI.
6. A compact `Channel View` panel for adding hidden channels, hiding inactive channels, and showing all channels.
7. A Fluent `Preview` tab that reuses `frontend.plot(..., kind="pulse")`; it should redraw automatically when the tab is opened or the `Show off rows` toggle changes. Do not add a manual refresh button. It should default to active channels, offer a Confocal-style capsule toggle switch for inactive rows, place the one-line `Save Figure` button at the far right of the top controls row, and show repeat notation. Preview should plot the unexpanded period table. A bracket covering every period is a finite outer repeat. An internal bracket means the full sequence remains `repeat ∞` and the internal span is a finite nested repeat; draw both brackets with distinct colors, and make the outer bracket visibly longer in the y-axis projection than the inner bracket. Hardware prepare/compile sends the unexpanded table plus repeat metadata, not an expanded repeat sequence.

For address-switch hardware, the editor should infer the hardware channel order from the selected XDC, default to a small visible subset, preserve the full hardware channel order internally, and use scroll/compact mode when all channels are shown. Hiding a channel is only a view operation; adding it back must restore the original hardware-order position. Clearing a channel turns period states off while preserving stored label and delay values.

The channel-name column, delay column, and period checkbox columns must share one vertical scroll position. Period cards may scroll horizontally as a timeline, but period cards and side panels must not have independent vertical scrollbars. The horizontal timeline scrollbar should remain visible whenever horizontal overflow exists; do not require the user to scroll to the vertical bottom to reach it. Row y coordinates for the same channel should match exactly across name, delay, and period checkbox widgets.

Keep channel-name widths content-driven. The default channel label/edit widths need to fit names such as `cooling_pgc`, `probe_shutter`, and `da_dipole`; do not widen the left panels so much that normal users lose period-card visibility.

Pulse GUI windows are fixed-size by default, normally about 90% of the available screen. Resize handles should not change the period alignment contract.

### Pulse GUI Non-Negotiable Alignment Rules

These rules are here because small one-off widget edits have repeatedly broken the pulse editor on different Windows DPI settings.

- All top control rows in `Channel Names and Duration`, `Delay and Scan`, period cards, and repeat brackets use the same fixed row height and the same top-panel height. Do not add a new row, spacer, title, switch text, or local margin unless the matching panels still line up.
- Form-row labels such as `Step:`, `Params:`, and `File:` keep Confocal-style centered text. Their label widgets share one fixed x position, and their right-side controls share one fixed x position. Do not rely on button size or text length to make a row look aligned.
- The active scan parameter row is read-only. It lists bound parameter names compactly and should not be a second scan-value editor.
- The scan file row links to one named parameter table. Do not add an x/y toggle or a pair-array text box.
- The repeat-bracket end spinbox must align vertically with period-card duration edits. The start-bracket placeholder row must reserve the same height.
- The left raw hardware column shows package pins, such as `F15` or `M13`, when the XDC pin map is available. The hardware bit name `chNN` belongs in the tooltip and saved/API state, not in the raw display column.
- `Channel View` and `Control Buttons` are `FluentGroupBox` cards with real title padding. If a title appears clipped or detached, fix parent layout margins and groupbox height; do not remove shadows or replace the Fluent card style.
- The bottom `Control Buttons` area should be a tight fixed-height frame. Do not let it expand vertically just because the desktop window is tall; the dataset scroll area above it should receive the spare height so more channels are visible.
- Scan snapping must use shared Fluent input behavior: `qt_fluent.align_to_resolution()` and the line-edit `editingFinished` path. Do not hand-roll a separate numeric snap parser in `pulse_gui.py`.
- Do not normalize or rewrite the scan line edit while it has focus. Summary and preview refreshes may parse the current text, but they must not call `setText()` or move the cursor; snapping belongs to `editingFinished` or explicit commands such as save/prepare/add-column.

Add object-level checks whenever these rows are touched. At minimum, assert that the same channel's raw-label, delay edit, and first period checkbox have the same `mapTo(editor, QPoint(0, 0)).y()`, that the Step/Params/File labels share x/y geometry with their controls, that the scan parameter/file rows align with the corresponding left-panel rows, and that the repeat end spinbox aligns with the first period duration edit.

### Pulse Preview Rules

- Repeat labels are `×∞` for forever and `×N` for finite repeats. The literal string `inf` should not appear in the plot. Repeat-label font sizes must match, and the finite nested label should sit lower than the outer `×∞` label to avoid overlap.
- Preview y-axis labels are channel display labels only. Do not show a y-axis title such as `Pulse`.
- Variable regions for named parameters and expressions such as `100000-camera_exposure_ns` span the full plotted channel area, including when `Show off rows` is enabled. Annotation labels use the same y position and must not be clipped.
- The preview toggle text is `Show off rows`, not `Always-off`. The phrase should describe what the toggle displays.
- Analog bus preview rows are hollow stair-step lines. Do not draw filled digital blocks, duplicate top rails, or `0..1024` range text for these rows.

### Pulse Timing Architecture Notes

The current FPGA upload path is a digital edge-template table, a two-axis timing scan array, and a separate analog-bus segment table. The GUI/API names scan parameters, while the compact Artix-7 35T profile accepts at most five active named scan parameters per FPGA chunk, with at most two timing-axis tick columns per scan point. One template row stores `base_tick`, two fixed-point axis coefficients, and `mask`; the FPGA evaluates the effective tick for each scan point. Static DA value scan columns are packed separately into per-row bus values. Do not expand a scan table into many GUI columns. Large scan files should be split by the host/API into bounded FPGA chunks instead of expanding the GUI timeline.

Analog bus ramps use the analog-bus segment table instead of TTL mask stair-step expansion:

```text
bus_id, start_tick, stop_tick, start_value, stop_value, mode
```

The FPGA bus engine runs one accumulator per bus, using a DDA/Bresenham-style stepper so a ramp costs one bus segment instead of hundreds of digital edge rows. Digital edge rows still handle laser/shutter/camera TTL changes. Hardware scan arrays cannot currently combine with analog bus segments in the same upload; show that as a compile-time validation message. Partial-delay logic can follow the same separation: keep shared digital edge rows for simultaneous mask changes, and use per-channel or per-bus local delay metadata only when it prevents unnecessary global edge-row growth.

## QA Checklist

Run these checks before handing off PyQt frontend changes:

```powershell
pytest -q tests/test_frontend_smoke.py::test_pulse_gui_constructs_xdc_channel_editor
python -m py_compile Zou_lab_control\frontend\qt_fluent.py Zou_lab_control\frontend\pulse_gui.py
```

For visual changes, generate screenshots with both default visible channels and all XDC-inferred channels. In headless Qt, text may not render in screenshots; also run object-level checks that button text fits each button, `show_all_channels()` keeps the full channel list visible, and `On Pulse` prepares then fires the passed-in sequencer while `Stop Pulse` calls safe state.

When verifying screenshots on Windows, prefer the native Qt backend and never grab immediately after `show()` or a state-changing action. For pulse-editor screenshots, use the built-in wait path whenever possible:

```python
editor.show()
editor.grab_screenshot(path, settle_ms=1000)
```

If a widget does not provide a screenshot helper, run the event loop and wait before every capture, for example:

```python
editor.show()
app.processEvents()
QtTest.QTest.qWait(1000)
app.processEvents()
editor.grab().save(str(path))
```

Repeat the wait after `show_all_channels()`, tab switches, scroll changes, and preview rendering. Offscreen Qt screenshots can miss text completely, so screenshots are visual QA only; object-level tests must still assert text, geometry, and state. Capture at least:

- default visible channels;
- all XDC-inferred channels at the top of the shared vertical scroll;
- all XDC-inferred channels after scrolling to the middle.

Check that period cards are visible inside `FluentWindow`, not only when grabbing the inner editor widget. State rebuilds must activate the editor, dataset, drag container, and outer window layouts.
