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
3. `Delay and X` panel for `x_ns`, minimal step, per-channel delay, unit, and `X` clear control. The Name panel's left column is always the hardware channel name; the right column is an optional display label. Do not auto-map generic hardware channels such as `ch00` to physical meanings. When the user edits a display label, the Delay row and period checkbox text may follow that label while saved/compiled pulses still use the hardware channel name. `X` clears all period states for that channel but does not hide it. Hiding is handled by the global `Hide Off` control, which hides channels with no period on and preserves stored label and delay values.
4. Horizontal period-card timeline, not a grid/table. Each card owns duration, unit selector, period position such as `Period 1/5`, and channel checkboxes. The unit selector is shown without a separate `Unit` label to keep period cards narrow.
5. Bottom Confocal-style large buttons for stop/off, on-pulse, wait, add/remove period, repeat bracket, save pulse, load, and temporary left-panel collapse. `On Pulse` prepares/uploads the current pulse state and then starts it; there is no separate sync button in the GUI.
6. A compact `Channel View` panel for adding hidden channels, hiding inactive channels, and showing all channels.
7. A Fluent `Preview` tab that reuses `frontend.plot(..., kind="pulse")`; it should redraw automatically when the tab is opened or the always-off toggle changes. Do not add a manual refresh button. It should default to active channels, offer a Confocal-style capsule toggle switch for always-off channels, place the one-line `Save Figure` button at the far right of the top controls row, and show repeat notation. Preview should plot the unexpanded period table. A bracket covering every period is a finite outer repeat. An internal bracket means the full sequence remains `repeat ∞` and the internal span is a finite nested repeat; draw both brackets with distinct colors, and make the outer bracket visibly longer in the y-axis projection than the inner bracket. Hardware prepare/compile sends the unexpanded table plus repeat metadata, not an expanded repeat sequence.

For 40-channel hardware, the editor should default to a small visible subset, preserve the full hardware channel order internally, and use scroll/compact mode when all channels are shown. Hiding a channel is only a view operation; adding it back must restore the original hardware-order position. Clearing a channel turns period states off while preserving stored label and delay values.

The channel-name column, delay column, and period checkbox columns must share one vertical scroll position. Period cards may scroll horizontally as a timeline, but period cards and side panels must not have independent vertical scrollbars. The horizontal timeline scrollbar should remain visible whenever horizontal overflow exists; do not require the user to scroll to the vertical bottom to reach it. Row y coordinates for the same channel should match exactly across name, delay, and period checkbox widgets.

Keep channel-name widths content-driven. The default channel label/edit widths only need to fit names such as `qcm_trigger`; do not widen the left panels so much that normal users lose period-card visibility.

Pulse GUI windows are fixed-size by default, normally about 90% of the available screen. Resize handles should not change the period alignment contract.

## QA Checklist

Run these checks before handing off PyQt frontend changes:

```powershell
pytest -q tests/test_frontend_smoke.py::test_pulse_gui_constructs_40_channel_editor
python -m py_compile Zou_lab_control\frontend\qt_fluent.py Zou_lab_control\frontend\pulse_gui.py
```

For visual changes, generate screenshots with both default visible channels and all 40 channels. In headless Qt, text may not render in screenshots; also run object-level checks that button text fits each button, `show_all_channels()` keeps 40 channels visible, and `On Pulse` prepares then fires the passed-in sequencer while `Stop Pulse` calls safe state.

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
- 40 visible channels at the top of the shared vertical scroll;
- 40 visible channels after scrolling to the middle.

Check that period cards are visible inside `FluentWindow`, not only when grabbing the inner editor widget. State rebuilds must activate the editor, dataset, drag container, and outer window layouts.
