"""One reusable operator path for PulseGUI offscreen fast screenshots.

The fast path intentionally uses the same ``ensure_qt_app`` and
``open_pulse_editor`` composition as ``pulse_gui.py``.  It never constructs a
``QApplication``, forces a scale factor, resizes the product window, or selects
tabs by mutating widget state.  Every transition below is a real Qt input event.

Running this file directly selects the offscreen Qt backend before calling
``ensure_qt_app``.  It is the fast debug/acceptance path and must not open a
desktop window.  Final or disputed appearance uses the same formal product entry
through the separate desktop human-flow path.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import tempfile
import time

from PyQt5 import QtCore, QtTest, QtWidgets

from gui_user_flow import (
    capture_offscreen_window,
    click_tab,
    configure_offscreen_fast_path,
    require_offscreen_platform,
    until,
)


def choose_mode(body, mode: str) -> None:
    """Choose Offline/Virtual/Remote through the visible combo popup."""

    combo = body.schedule_view.conn_target_combo
    model_row = combo.findData(mode)
    if model_row < 0:
        raise ValueError(f"PulseGUI has no connection mode {mode!r}")
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    view = combo.view()
    application = QtWidgets.QApplication.instance()
    if application is None:
        raise RuntimeError("choose_mode requires the formal QApplication owner")
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    # Combo popups are separate Qt surfaces.  Drive the visible popup with the
    # same keyboard events an operator can use; no index or model state is
    # assigned directly.
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _index in range(model_row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    if combo.currentData() != mode:
        raise AssertionError(f"connection combo did not select {mode!r}")

    # The 40-ms owner snapshot must not overwrite a human's draft selection.
    deadline = time.monotonic() + 0.15
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    if combo.currentData() != mode:
        raise AssertionError("owner refresh overwrote the selected connection mode")


def exercise_offline_dac_round_trip(
    body,
    application: QtWidgets.QApplication,
    *,
    observe_draft: Callable[[], None] | None = None,
) -> str:
    """Add/apply/remove one DAC using only visible PulseGUI controls."""

    original_abi = body.current_document.target.abi_fingerprint
    click_tab(body, body.target_view)
    original_rows = {row.key: row for row in body.target_view._rows}
    QtTest.QTest.mouseClick(
        body.target_view.add_dac_button,
        QtCore.Qt.LeftButton,
    )
    added_key = body.target_view._rows[-1].key
    draft_added_row = body.target_view._rows[-1]
    assert all(
        next(row for row in body.target_view._rows if row.key == key) is widget
        for key, widget in original_rows.items()
    )
    scrollbar = body.target_view.cards_scroll.verticalScrollBar()

    def remove_button_is_visible() -> bool:
        row = next(
            (item for item in body.target_view._rows if item.key == added_key),
            None,
        )
        if row is None:
            return False
        viewport = body.target_view.cards_scroll.viewport()
        center = row.remove_button.mapTo(
            viewport,
            row.remove_button.rect().center(),
        )
        return viewport.rect().contains(center)

    until(application, remove_button_is_visible)
    if observe_draft is not None:
        observe_draft()

    QtTest.QTest.mouseClick(
        body.target_view.apply_button,
        QtCore.Qt.LeftButton,
    )
    until(application, lambda: added_key in body.current_document.target.by_key)
    assert next(
        row for row in body.target_view._rows if row.key == added_key
    ) is draft_added_row
    assert all(
        next(row for row in body.target_view._rows if row.key == key) is widget
        for key, widget in original_rows.items()
    )

    # Reach the bottom row through a real scrollbar key event before clicking
    # it; sending QTest input straight to an off-viewport child is not a human
    # flow.  Apply itself keeps every keyed row and the scroll position stable.
    QtTest.QTest.keyClick(scrollbar, QtCore.Qt.Key_End)
    until(application, remove_button_is_visible)
    added_row = next(row for row in body.target_view._rows if row.key == added_key)
    QtTest.QTest.mouseClick(added_row.remove_button, QtCore.Qt.LeftButton)
    QtTest.QTest.mouseClick(
        body.target_view.apply_button,
        QtCore.Qt.LeftButton,
    )
    until(
        application,
        lambda: (
            added_key not in body.current_document.target.by_key
            and body.current_document.target.abi_fingerprint == original_abi
        ),
    )
    assert all(
        next(row for row in body.target_view._rows if row.key == key) is widget
        for key, widget in original_rows.items()
    )
    return added_key


def capture_offscreen_pulse_gui_user_flow(
    output_directory: str | Path,
    *,
    repository: str | Path | None = None,
    settle_ms: int = 1000,
) -> tuple[Path, ...]:
    """Run the offscreen fast flow and return its whole-window PNG files."""

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    platform = require_offscreen_platform(application)

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    if repository is None:
        temporary = tempfile.TemporaryDirectory(prefix="zlc-pulse-user-view-")
        repository_path = Path(temporary.name)
    else:
        repository_path = Path(repository).expanduser().resolve()

    body = open_pulse_editor(repository=repository_path)
    wrapper = body.window()
    captures: list[dict[str, object]] = []
    try:
        captures.append(
            capture_offscreen_window(
                application,
                body,
                output / "01-offline-edit.png",
                settle_ms=settle_ms,
            )
        )

        click_tab(body, body.target_view)
        captures.append(
            capture_offscreen_window(
                application,
                body,
                output / "02-offline-target.png",
                settle_ms=settle_ms,
            )
        )

        exercise_offline_dac_round_trip(
            body,
            application,
            observe_draft=lambda: captures.append(
                capture_offscreen_window(
                    application,
                    body,
                    output / "03-offline-target-added-dac.png",
                    settle_ms=settle_ms,
                )
            ),
        )

        click_tab(body, body.schedule_view)
        choose_mode(body, "virtual")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: body._controller.snapshot().connection_state == "ready",
        )
        captures.append(
            capture_offscreen_window(
                application,
                body,
                output / "04-virtual-edit.png",
                settle_ms=settle_ms,
            )
        )

        click_tab(body, body.target_view)
        captures.append(
            capture_offscreen_window(
                application,
                body,
                output / "05-virtual-target-readonly.png",
                settle_ms=settle_ms,
            )
        )

        click_tab(body, body.schedule_view)
        choose_mode(body, "offline")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: (
                body._controller.snapshot().connection_state == "offline"
                and body._controller.snapshot().connection_mode == "offline"
            ),
        )
        captures.append(
            capture_offscreen_window(
                application,
                body,
                output / "06-offline-restored.png",
                settle_ms=settle_ms,
            )
        )
    finally:
        body.request_close(discard_unsaved=True)
        until(application, lambda: body._controller.snapshot().close_complete)
        until(application, lambda: wrapper is None or not wrapper.isVisible())
        if temporary is not None:
            temporary.cleanup()

    manifest = {
        "schema": "zlc.tests.PulseGuiOffscreenFastView",
        "qt_platform": platform,
        "flow": [
            "formal open_pulse_editor composition",
            "Offline Edit",
            "Target tab",
            "Add DAC",
            "Apply",
            "Remove DAC",
            "Apply",
            "Virtual combo",
            "Connect",
            "Virtual Target",
            "Offline combo",
            "Connect",
        ],
        "captures": captures,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tuple(Path(item["path"]) for item in captures)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture the formal PulseGUI through offscreen Qt input events."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--settle-ms", type=int, default=1000)
    arguments = parser.parse_args()
    paths = capture_offscreen_pulse_gui_user_flow(
        arguments.out,
        repository=arguments.repository,
        settle_ms=arguments.settle_ms,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
