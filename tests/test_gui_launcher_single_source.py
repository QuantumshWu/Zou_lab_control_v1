"""The top-level GUI launcher sequence is ONE helper (``qt_fluent.launch_fluent_window``).

Four ``show_*`` entry points (pulse editor / task console / figure viewer / device manager) used to
hand-copy the wrap -> size -> centre -> show -> retain sequence; the fourth copy (device manager)
had silently dropped ensure_qt_app + the shared fluent scale + centring + retention -- so
``exp.device_manager()`` as a session's FIRST GUI was a Qt fatal and its controls sized at the
silent 1.0 fallback.  These contracts pin:

  * every launcher module routes through ``launch_fluent_window`` and no longer hand-rolls any
    step of the sequence (source-level, so a fifth copy cannot regrow);
  * the parallel second launcher ``run_fluent_window`` (zero callers, diverging semantics) is gone;
  * functionally, ``show_device_manager`` (the drifted one) now runs the WHOLE shared sequence
    offscreen: QApplication first, auto scale resolved, explicit size, centred, shown, retained.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

_FRONTEND = Path(__file__).resolve().parent.parent / "Zou_lab_control" / "frontend"

_LAUNCHER_MODULES = ("pulse_gui.py", "task_console.py", "figure_viewer.py", "device_manager.py")


def test_all_four_launchers_route_through_the_one_helper():
    for name in _LAUNCHER_MODULES:
        src = (_FRONTEND / name).read_text(encoding="utf-8")
        assert "launch_fluent_window(" in src, f"{name} must launch via the ONE helper"
        for step in ("retain_window(", "center_window_on_primary_screen(",
                     "setFixedSize(window.size())", "FluentWindow(widget="):
            assert step not in src, (
                f"{name} hand-rolls the launcher step {step!r} -- the sequence lives ONLY in "
                "qt_fluent.launch_fluent_window")


def test_parallel_run_fluent_window_is_gone():
    src = (_FRONTEND / "qt_fluent.py").read_text(encoding="utf-8")
    assert "run_fluent_window" not in src, "run_fluent_window was the zero-caller parallel launcher"
    # its private FluentWindow construction path (widget_class/widget_kwargs) is cleaned with it
    assert "widget_class" not in src, "FluentWindow(widget_class=) existed only for run_fluent_window"


def test_show_device_manager_runs_the_shared_launcher_sequence(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets

    from Zou_lab_control.frontend import device_manager as dm
    from Zou_lab_control.frontend import qt_fluent as qf
    from Zou_lab_control.frontend.device_manager import show_device_manager

    # Pin that the launcher invokes the ONE shared scale rule (set_fluent_scale(None) = auto):
    # asserting on the resolved VALUE would be flaky (the auto rule is not a fixed point across
    # re-resolves), so record the call through the module's own name instead.
    scale_calls: list = []
    real_set_scale = qf.set_fluent_scale
    monkeypatch.setattr(dm, "set_fluent_scale",
                        lambda scale=None: (scale_calls.append(scale), real_set_scale(scale))[1])

    device_set = SimpleNamespace(devices={}, device_names=lambda base_type: [])
    window = show_device_manager(device_set)
    try:
        app = QtWidgets.QApplication.instance()
        assert app is not None, "ensure_qt_app must run BEFORE the panel ctor"
        assert scale_calls == [None], "the launcher must resolve the shared auto-scale rule"
        # shown + retained on the ONE app registry (pruned when the window dies)
        assert window.isVisible()
        assert window in getattr(app, "_zlc_retained_windows", []), \
            "the device-manager window must be retained like every other GUI"
        # the explicit-size path (its scrolling editor has no usable content hint); the size
        # is the module's ONE design constant, never re-typed here
        assert (window.width(), window.height()) == (
            qf.scaled_px(dm.WINDOW_SIZE[0]), qf.scaled_px(dm.WINDOW_SIZE[1]))
        # centred on the primary screen's available area, like every other GUI
        avail = app.primaryScreen().availableGeometry()
        centre = window.frameGeometry().center()
        assert abs(centre.x() - avail.center().x()) <= 1
        assert abs(centre.y() - avail.center().y()) <= 1
    finally:
        window.close()
        window.deleteLater()
