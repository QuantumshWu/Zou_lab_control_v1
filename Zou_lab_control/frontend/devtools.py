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
    """Settle, then grab ``widget`` to ``path`` (PNG).  Returns the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settle(widget, settle_ms)
    widget.grab().save(str(path))
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

    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState

    ch = [f"ch{i:02d}" for i in range(max(channels, 32))]
    labels = {"ch00": "trap", "ch01": "cooling", "ch02": "probe", "ch03": "repump",
              "ch04": "pushout", "ch05": "grey_cooling", "ch11": "emCCD", "ch12": "microwave"}
    for i in range(10):
        labels[f"ch{18 + i:02d}"] = f"da_dipole[{i}]"
    visible = ["ch00", "ch01", "ch02", "ch03", "ch04", "ch05", "ch11", "ch12"] + [f"ch{18 + i:02d}" for i in range(10)]

    def states(active):
        return tuple(1 if c in active else 0 for c in range(len(ch)))

    periods = [
        PulsePeriod(2_000_000, states({0, 1}), unit="ns"),
        PulsePeriod(100_000, states({0}), unit="ns"),
        PulsePeriod(20_000, states({0, 1}), unit="ns"),
        PulsePeriod(50_000, states({0, 2}), unit="ns"),
        PulsePeriod(20_000, states({0, 1}), unit="ns"),
    ]
    state = PulseTableState(channels=ch, visible_channels=visible, periods=periods,
                            channel_labels=labels, time_step_ns=20.0, name="demo_scan")
    return state


def demo_editor(*, scale: float = 1.0, size=(1440, 880), bind_scans: bool = True):
    """Return a shown :class:`PulseSequenceEditor` with optional scan bindings."""

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    install_screenshot_font()  # before building so build-time text metrics match the render font
    ed = PulseSequenceEditor(state=demo_state(), scale=scale)
    if bind_scans:
        # Re-fetch cards after each bind: load_state() rebuilds the cards.
        ed._toggle_duration_scan(ed.drag_container.pulse_cards()[3])  # period-4 duration -> s0
        ed._toggle_delay_scan("ch01")                                # cooling delay     -> s1
        buses = list(ed.state.bus_channels().keys())
        if buses:
            ed._toggle_dac_scan(ed.drag_container.pulse_cards()[1], buses[0])  # da_dipole p2 -> s2
        ed.state.set_scan_table([[10000, 0, 100], [20000, 200, 300], [40000, 400, 700]])
        ed.load_state(ed.state)
    if size is not None:
        ed.setFixedSize(int(size[0]), int(size[1]))
        ed._activate_layout_tree()
    ed.show()
    return ed


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


__all__ = ["settle", "screenshot", "screenshot_tab", "demo_state", "demo_editor", "capture_gallery"]
