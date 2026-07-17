"""Developer GUI self-check helpers for the pulse editor.

PyQt lays out and paints asynchronously: grabbing a widget immediately after
building or mutating it can capture a half-laid-out frame.  These helpers pump
the event loop and wait so screenshots reflect the *final* layout, which makes
them reliable for visual self-checks (alignment, cutoff, overlap) in tests or
ad-hoc scripts.

Example::

    from Zou_lab_control.frontend import devtools as dt
    ed = dt.demo_editor()                       # representative editor
    dt.screenshot(ed, "edit.png")               # current (Edit) tab
    dt.screenshot_tab(ed, "preview", "prev.png")
"""

from __future__ import annotations

from pathlib import Path

_FONT_INSTALLED = False


def install_screenshot_font() -> str | None:
    """Register a glyph-complete TTF so offscreen screenshots render text.

    The Qt ``offscreen`` platform often ships without the UI font (``Segoe UI``),
    so ``drawText`` produces blank boxes.  We load matplotlib's bundled
    DejaVuSans and substitute it for the UI/monospace families the widgets ask
    for, which makes text appear in grabs.  No-op on real displays where the
    fonts already exist; safe to call repeatedly.
    """

    global _FONT_INSTALLED
    from PyQt5 import QtGui, QtWidgets

    if QtWidgets.QApplication.instance() is None or _FONT_INSTALLED:
        return None
    family = None
    try:
        import matplotlib

        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if ttf.exists():
            font_id = QtGui.QFontDatabase.addApplicationFont(str(ttf))
            families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
            family = families[0] if families else None
    except Exception:
        family = None
    if family is None:
        available = QtGui.QFontDatabase().families()
        family = available[0] if available else None
    if family:
        for requested in ("Segoe UI", "Helvetica", "Consolas", "Courier New", "monospace", "DejaVu Sans"):
            QtGui.QFont.insertSubstitution(requested, family)
    _FONT_INSTALLED = True
    return family


def settle(widget=None, ms: int = 500) -> None:
    """Pump the Qt event loop and wait so pending layout/paint completes."""

    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    install_screenshot_font()
    app.processEvents()
    if widget is not None:
        widget.repaint()
    app.processEvents()
    try:
        from PyQt5 import QtTest

        QtTest.QTest.qWait(int(ms))
    except Exception:  # pragma: no cover - QtTest not always present
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(int(ms), loop.quit)
        loop.exec_()
    app.processEvents()
    if widget is not None:
        widget.repaint()
        app.processEvents()


def screenshot(widget, path, *, settle_ms: int = 500) -> Path:
    """Settle, then grab ``widget`` to ``path`` (PNG), FLATTENED over white.  Returns the path.

    ``QWidget.grab`` of an OFFSCREEN widget can leave genuinely-transparent pixels, which then
    save as BLACK.  The Fluent tab bar is exactly this: its unselected tabs are transparent (the
    white rounded card is the tab widget's OWN ``paintEvent``, and the tab strip over it stays
    see-through).  Flattening the grab onto an opaque WHITE canvas (the card colour) makes a saved
    figure -- and the PDF it lands in -- show the tabs as the running GUI renders them, not black."""

    from PyQt5 import QtGui

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settle(widget, settle_ms)
    grabbed = widget.grab()
    canvas = QtGui.QPixmap(grabbed.size())
    # The grab's pixmap is in DEVICE pixels but carries a devicePixelRatio (N on a scaled /
    # high-DPI screen); the canvas must share it or drawPixmap(0,0) paints the grab at its
    # LOGICAL size into the top-left and leaves the rest white (cramming a DPR=1.5/2 capture
    # into a corner).  Match the dpr so the flatten is device-pixel-exact and full-resolution.
    canvas.setDevicePixelRatio(grabbed.devicePixelRatio())
    canvas.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(canvas)
    painter.drawPixmap(0, 0, grabbed)
    painter.end()
    canvas.save(str(path))
    return path


def screenshot_tab(editor, tab: str, path, *, settle_ms: int = 500) -> Path:
    """Switch a :class:`PulseSequenceEditor` to ``tab`` and screenshot it.

    ``tab`` is ``"edit"``, ``"preview"`` or ``"scan"``.
    """

    target = {"edit": editor.edit_tab, "preview": editor.preview_tab, "scan": editor.scan_tab}[tab]
    editor.tabs.setCurrentWidget(target)
    if tab == "preview":
        editor.refresh_preview()
    elif tab == "scan":
        editor._refresh_scan_tab()
    return screenshot(editor, path, settle_ms=settle_ms)


def demo_state(*, channels: int = 24):
    """Build a representative :class:`PulseTableState` that stresses layout.

    Mixes named digital channels, a 10-bit DAC bus (``da_dipole``), several
    periods, and realistic durations/delays so screenshots exercise the
    name/delay/period columns, the DAC rows, and the control/channel-view bars.
    """

    from Zou_lab_control.neutral_atom.ports import PortCatalog
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState

    ch = [f"ch{i:02d}" for i in range(max(channels, 32))]
    labels = {"ch00": "trap", "ch01": "cooling", "ch02": "probe", "ch03": "repump",
              "ch04": "pushout", "ch05": "grey_cooling", "ch11": "emCCD", "ch12": "microwave"}
    for i in range(10):
        labels[f"ch{18 + i:02d}"] = f"da_dipole[{i}]"
    visible = ["ch00", "ch01", "ch02", "ch03", "ch04", "ch05", "ch11", "ch12", "da_dipole"]

    def states(active):
        return tuple(1 if c in active else 0 for c in range(len(ch)))

    periods = [
        PulsePeriod(2_000_000, states({0, 1}), unit="ns"),
        PulsePeriod(100_000, states({0}), unit="ns"),
        PulsePeriod(20_000, states({0, 1}), unit="ns"),
        PulsePeriod(50_000, states({0, 2}), unit="ns"),
        PulsePeriod(20_000, states({0, 1}), unit="ns"),
    ]
    state = PulseTableState(
        port_catalog=PortCatalog.from_channels(ch, channel_labels=labels),
        visible_ports=visible,
        periods=periods,
        time_step_ns=20.0,
        name="demo_scan",
    )
    return state


def demo_editor(*, scale: float = 1.0, size=(1440, 880), bind_scans: bool = True):
    """Return a shown :class:`PulseSequenceEditor` with optional scan bindings."""

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from zlc_frontend.qt_widgets import ensure_qt_app

    ensure_qt_app()
    install_screenshot_font()  # before building so build-time text metrics match the render font
    ed = PulseSequenceEditor(state=demo_state(), scale=scale)
    if bind_scans:
        # Re-fetch cards after each bind: load_state() rebuilds the cards.
        ed._toggle_duration_scan(ed.drag_container.pulse_cards()[3])  # period-4 duration -> s0
        buses = list(ed.state.bus_channels().keys())
        if buses:
            ed._toggle_dac_scan(ed.drag_container.pulse_cards()[1], buses[0])  # da_dipole p2 -> s1
        ed.state.set_scan_table([[10000, 100], [20000, 300], [40000, 700]])
        ed.load_state(ed.state)
    if size is not None:
        ed.setFixedSize(int(size[0]), int(size[1]))
        ed._activate_layout_tree()
    ed.show()
    return ed



def _demo_board_state(*, board_width: int):
    """Representative camera dashboard for deterministic GUI screenshots.

    The fixture uses only the raw camera producer and display-only expressions;
    it does not fabricate a calibration or occupancy pipeline.  Site-map support
    remains a generic frontend capability and will be exercised by its current
    controller once that vertical slice is wired.
    """

    from Zou_lab_control.frontend.task_console import (
        PanelConfig,
        TaskConsoleState,
        _first_free_slot,
    )

    frame = ["frame_0"]
    scalar = "value = np.mean(signal)"
    curve = "value = np.mean(signal, axis=-2)"
    panels = [
        PanelConfig(
            kind="2d",
            title="Camera image",
            size="2x2",
            source="value = signal",
            inputs=frame,
        ),
        PanelConfig(
            kind="monitor",
            title="Mean intensity (dist)",
            size="1x2",
            source=scalar,
            inputs=frame,
            params={"length": 300, "show_dist": True},
        ),
        PanelConfig(
            kind="monitor",
            title="Mean intensity",
            size="1x2",
            source=scalar,
            inputs=frame,
            params={"length": 300, "show_dist": False},
        ),
        PanelConfig(
            kind="hist",
            title="Pixel distribution",
            size="1x2",
            source="value = signal",
            inputs=frame,
            params={"bins": 80},
        ),
        PanelConfig(
            kind="1d",
            title="Column profile",
            size="2x4",
            source=curve,
            inputs=frame,
        ),
    ]
    placed = []
    for panel in panels:
        panel.col, panel.row = _first_free_slot(panel, placed, int(board_width))
        placed.append(panel)
    return TaskConsoleState(name="demo_camera_views", panels=panels)



def demo_console(*, scale: float = 1.0, size=(1480, 980), grid=(5, 7), state=None,
                 shots: int = 8):
    """Return a deterministic virtual-camera dashboard.

    This developer fixture exercises the TaskConsole shell and display pipeline
    without standing up a second readout implementation.  One real
    :class:`CameraMeasurement` publishes ``frame_0``; panels derive their camera
    image, distribution, profile and scalar monitor as display-only expressions.
    The timer is stopped and the fixture is stepped synchronously for stable
    screenshots.
    """

    import Zou_lab_control.neutral_atom as na
    from zlc_frontend.qt_widgets import (
        WINDOW_SCREEN_FRACTION,
        ensure_qt_app,
        screen_fit_window_size,
        set_fluent_scale,
    )
    from Zou_lab_control.frontend.task_console import LogicNodeConfig, TaskConsole
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    install_screenshot_font()
    exp = na.connect("virtual", sitemap={"grid_shape": tuple(grid)})

    # The virtual camera observes the same independently running trigger source as
    # the hardware adapter; the measurement remains a passive reader.
    live_pulse = exp.build_imaging_sequence(name="live", load=True).forever()
    exp.timing.bind_pulse(live_pulse).on_pulse(repeat_forever=True, wait=False)

    if state is not None:
        layout = state
    else:
        set_fluent_scale(scale)
        board_width = int(size[0]) if size is not None else int(
            screen_fit_window_size(WINDOW_SCREEN_FRACTION).width()
        )
        layout = _demo_board_state(board_width=board_width)

    console = TaskConsole(
        hub=SignalHub(),
        state=layout,
        session=exp,
        measurements=exp.readout.measurement_specs(),
        processors=exp.readout.processor_specs(),
        tasks=exp.readout.task_specs(),
        scale=scale,
        window_px=size,
    )
    # This fixture creates and owns the virtual experiment.  Route teardown
    # through the console's existing resource-close seam so ``shutdown()`` does
    # not leave the session runtime thread alive after a test or screenshot.
    console._on_close = exp.close
    console._timer.stop()

    camera_row = console._add_logic_node(
        LogicNodeConfig(
            kind="camera",
            name="live",
            title=exp.readout.camera_spec().name,
        ),
        focus=False,
    )
    camera = console._build_logic_node(camera_row.node, dict(camera_row.node.values))
    console._logic_nodes[id(camera_row)] = camera
    console.running_nodes.append(camera)
    camera_row.set_state("running", status="running")

    for _ in range(max(1, int(shots))):
        camera.step()
    console.refresh_once()
    console.save_button.set_dirty(False)
    console.show()
    return console



def demo_console_measurements(*, scale: float = 1.0, size=(1480, 980), grid=(5, 7)):
    """Return a shown :class:`TaskConsole` wired with the readout measurement
    catalog: a measurement is added as a LOGIC NODE (Logic tab) from the header's
    Add Panel, and its OWN Edit tab carries the live parameter form + Start / Stop,
    fed by a virtual session.

    The catalog and node builder follow the same contract path as real hardware.
    Start a node from its Edit to stream signals to the hub, then add a Plot panel
    on the Monitor board pointed at them.
    """

    from zlc_frontend.qt_widgets import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, TaskConsoleState
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    import Zou_lab_control.neutral_atom as na

    ensure_qt_app()
    install_screenshot_font()
    exp = na.connect("virtual", sitemap={"grid_shape": tuple(grid)})
    measurements = exp.readout.measurement_specs()
    hub = SignalHub()
    # Start with an EMPTY board; a measurement is a Logic node, added below, whose
    # Edit (its params + Start/Stop) is the whole measurement flow.
    console = TaskConsole(hub=hub, state=TaskConsoleState(name="measurements"),
                          measurements=measurements, session=exp, scale=scale, window_px=size)
    console.show()
    # Add the first measurement as a Logic node + open its Edit tab, so the demo
    # opens showing the measurement form (its params + Start) on the Logic tab.
    if measurements:
        kc = console.kind_combo
        target = ("measurement", measurements[0].name)
        idx = next((i for i in range(kc.count()) if kc.itemData(i) == target), -1)
        if idx >= 0:
            kc.setCurrentIndex(idx)
            console._add_panel()
    return console


def capture_user_view(target: str, out_dir, *, scales=(1.0, 1.25, 1.5), size=(1480, 1000),
                      shots: int = 60, settle_ms: int = 900, timeout: float = 300.0):
    """Render ``target`` ("console", "editor" or "parity") AS THE USER SEES IT
    and save one full-window screenshot per Windows display scale.

    ``parity`` opens BOTH GUIs in one process with scale=None (each window's
    REAL automatic-scale path, sharing one screen) and fails if their fluent
    control sizes disagree -- the check that catches cross-GUI scaling drift
    the per-GUI captures cannot see (both demo helpers pin scale=1.0).

    This is THE visual-acceptance interface: every UI/plot change must pass it,
    not a DPR=1 screenshot.  Each scale runs in a SUBPROCESS with
    ``QT_SCALE_FACTOR`` set (Qt reads it once at QApplication creation; the
    offscreen platform honours it, reproducing scaled Windows screens), the
    window settles ``settle_ms`` before grabbing (fonts/canvases render late),
    and the saved PNGs are meant to be inspected as 1:1 pixel crops.

    Returns ``{scale: Path}``.  Also runnable from a shell::

        python -m Zou_lab_control.frontend.devtools --capture console --out shots/
    """

    import os
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[float, Path] = {}
    for scale in scales:
        out_path = out_dir / f"{target}_sf{scale}.png"
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["QT_SCALE_FACTOR"] = str(scale)
        command = [
            sys.executable, "-m", "Zou_lab_control.frontend.devtools",
            "--capture", str(target), "--out", str(out_path),
            "--size", f"{int(size[0])}x{int(size[1])}",
            "--shots", str(int(shots)), "--settle-ms", str(int(settle_ms)),
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"capture_user_view({target!r}, scale={scale}) failed:\n{result.stdout}\n{result.stderr}")
        results[float(scale)] = out_path
    return results


def _capture_main(argv=None) -> int:
    """CLI driver for capture_user_view's per-scale subprocesses."""

    import argparse

    parser = argparse.ArgumentParser(description="Render a frontend GUI offscreen and screenshot it.")
    parser.add_argument("--capture", required=True, choices=("console", "editor", "parity"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="1480x1000")
    parser.add_argument("--shots", type=int, default=60)
    parser.add_argument("--settle-ms", type=int, default=900)
    args = parser.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    from zlc_frontend.qt_widgets import FluentWindow

    if args.capture == "parity":
        # BOTH GUIs, REAL automatic scale, ONE (offscreen) screen: their fluent
        # controls must be pixel-identical or the capture fails.
        from PyQt5 import QtGui
        from zlc_frontend.qt_widgets import FluentLineEdit

        editor = demo_editor(scale=None, size=(width, height))
        console = demo_console(scale=None, shots=args.shots, size=(width, height))
        wins = [FluentWindow(widget=editor, title="PulseGUI@Zou lab", hide_on_close=False),
                FluentWindow(widget=console, title="TaskConsole@Zou lab", hide_on_close=False)]
        for win in wins:
            win.adjustSize(); win.setFixedSize(win.size()); win.show()
        settle(wins[0], args.settle_ms)
        console.refresh_once()
        settle(wins[1], args.settle_ms)
        heights = []
        for body in (editor, console):
            hs = sorted({w.height() for w in body.findChildren(FluentLineEdit) if w.isVisible()})
            heights.append(hs)
        # the STANDARD fluent row (the 30px basis) must match exactly; the pulse
        # editor also carries its intentional dense per-channel rows (26px basis
        # beyond 16 channels), so compare the standard = tallest control.
        grabs = [win.grab() for win in wins]
        gap = 12
        combined = QtGui.QPixmap(grabs[0].width() + gap + grabs[1].width(),
                                 max(grabs[0].height(), grabs[1].height()))
        combined.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(combined)
        painter.drawPixmap(0, 0, grabs[0])
        painter.drawPixmap(grabs[0].width() + gap, 0, grabs[1])
        painter.end()
        combined.save(str(args.out))
        print(f"parity lineedit heights: editor={heights[0]} console={heights[1]} -> {args.out}")
        if max(heights[0]) != max(heights[1]):
            print("PARITY FAILED: the standard fluent row differs between the two GUIs")
            return 2
        return 0

    if args.capture == "console":
        body = demo_console(shots=args.shots, size=(width, height))
        title = "TaskConsole@Zou lab"
    else:
        body = demo_editor(size=(width, height))
        title = "PulseGUI@Zou lab"
    window = FluentWindow(widget=body, title=title, hide_on_close=False)
    window.adjustSize()
    window.setFixedSize(window.size())
    window.show()
    settle(window, args.settle_ms)
    if args.capture == "console":
        body.refresh_once()
        settle(window, args.settle_ms)
    screenshot(window, args.out)
    print(f"captured {args.capture} -> {args.out}")
    return 0


def capture_gallery(out_dir, *, settle_ms: int = 550) -> dict[str, Path]:
    """Render the editor in several states for a visual self-check sweep."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ed = demo_editor()
    paths["edit"] = screenshot_tab(ed, "edit", out_dir / "edit.png", settle_ms=settle_ms)
    paths["scan"] = screenshot_tab(ed, "scan", out_dir / "scan.png", settle_ms=settle_ms)
    paths["preview_active"] = screenshot_tab(ed, "preview", out_dir / "preview_active.png", settle_ms=settle_ms)
    ed.preview_include_off.setChecked(True)
    ed.refresh_preview()
    paths["preview_all"] = screenshot(ed, out_dir / "preview_all.png", settle_ms=settle_ms)
    ed.preview_include_off.setChecked(False)
    ed.refresh_preview()
    paths["preview_back"] = screenshot(ed, out_dir / "preview_back.png", settle_ms=settle_ms)
    return paths


__all__ = ["settle", "screenshot", "screenshot_tab", "demo_state", "demo_editor", "demo_console",
           "demo_console_measurements", "capture_gallery", "capture_user_view"]


if __name__ == "__main__":  # pragma: no cover - exercised through capture_user_view subprocesses
    import sys as _sys
    _sys.exit(_capture_main())
