"""Human-control contract for the formal Pulse GUI Preview tab.

The visible product is frozen.  These tests deliberately enter through the
same tabs, switches, combo popup, wheel gesture, and Save Figure button an
operator uses.  They do not reach through an obsolete editor object or invoke
controller actions as a substitute for GUI interaction.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

from gui_user_flow import close_pulse_editor
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_frontend.render import PulsePanelPayload
from zlc_pulse import load_pulse_document
from zlc_workbench.pulse_editor.session import project_pulse_preview


ROOT = Path(__file__).parents[1]
PULSE_PATH = ROOT / "pulses" / "imaging_template.json"


def _until(application, predicate, *, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _click_tab(body, page) -> None:
    index = body.tabs.indexOf(page)
    assert index >= 0
    bar = body.tabs.tabBar()
    QtTest.QTest.mouseClick(
        bar,
        QtCore.Qt.LeftButton,
        pos=bar.tabRect(index).center(),
    )


def _visible_pulse_payload(board) -> PulsePanelPayload | None:
    """Read the typed payload from the board's current public front."""

    frame = board.front_frame
    if frame is None or len(frame.panels) != 1:
        return None
    payload = frame.panels[0].display_payload
    return payload if isinstance(payload, PulsePanelPayload) else None


def _choose_combo(application, combo, text: str) -> None:
    index = combo.findText(text)
    assert index >= 0
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    _until(application, lambda: combo.view().isVisible())
    view = combo.view()
    # Keyboard navigation is a real operator path and is less dependent on the
    # platform popup container's translucent outer padding than a synthetic
    # coordinate inside that top-level native window.
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _position in range(index):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    _until(application, lambda: combo.currentText() == text)


def _send_wheel(application, board, *, delta: int = -120) -> None:
    centre = board.rect().center()
    event = QtGui.QWheelEvent(
        QtCore.QPointF(centre),
        QtCore.QPointF(board.mapToGlobal(centre)),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    QtWidgets.QApplication.sendEvent(board, event)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def _send_mouse_move_with_buttons(application, board, position, buttons) -> None:
    """Deliver one real Qt pointer-motion event while a button stays held."""

    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(position),
        QtCore.QPointF(board.mapToGlobal(position)),
        QtCore.Qt.NoButton,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(board, event)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)


@pytest.fixture
def preview_body(tmp_path):
    application = ensure_qt_app()
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    body = open_pulse_editor(
        path=PULSE_PATH,
        pulses_root=tmp_path / "pulses",
        output_root=tmp_path / "output",
    )
    _until(
        application,
        lambda: body.window() is not body and body.window().isVisible(),
    )
    _click_tab(body, body.preview_view)
    _until(
        application,
        lambda: body.tabs.currentWidget() is body.preview_view
        and body.preview_host.front_frame is not None,
    )
    _until(application, lambda: body.worker_idle)
    yield body
    close_pulse_editor(application, body)
    assert body.worker_idle


def test_preview_opens_as_the_full_formal_plot_with_display_labels(preview_body) -> None:
    body = preview_body
    frame = body.preview_host.front_frame
    assert frame is not None
    raster = frame.panels[0].raster
    payload = _visible_pulse_payload(body.preview_host.board)
    assert payload is not None

    assert raster.height > 100
    assert body.preview_host.height() > 100
    assert body.preview_view.preview_body.height() >= body.preview_host.height()
    assert re.fullmatch(
        r"\d+/\d+ plotted \((active|all) channels\)"
        r" \| repeat (?:∞(?: \+ P\d+-P\d+ x\d+)?|P\d+-P\d+ x\d+)",
        body.preview_view.preview_status.text(),
    )

    timeline = project_pulse_preview(load_pulse_document(PULSE_PATH))
    expected = tuple(
        (row.row_id, row.label) for row in timeline.rows if row.active
    )
    assert tuple(zip(payload.row_keys, payload.row_labels, strict=True)) == expected
    assert any(key != label for key, label in expected), (
        "the Preview axis regressed to raw lane keys instead of board labels"
    )


def test_preview_dpr_change_immediately_retires_the_old_front(preview_body) -> None:
    body = preview_body
    application = ensure_qt_app()
    previous = body.preview_host.front_frame
    assert previous is not None
    ratio = body._preview_surface_pixel_ratio + 0.5

    body._apply_preview_pixel_ratio(ratio)
    assert body.preview_host.front_frame is None
    assert not body.preview_host.isVisible()
    _until(
        application,
        lambda: body.worker_idle
        and body.preview_host.front_frame is not None
        and body.preview_host.isVisible(),
    )

    current = body.preview_host.front_frame
    assert current is not previous
    raster = current.panels[0].raster
    qround = lambda value: int(math.floor(value + 0.5))
    assert (raster.width, raster.height) == (
        qround(body.preview_host.width() * ratio),
        qround(body.preview_host.height() * ratio),
    )


def test_preview_keeps_the_placeholder_until_the_first_complete_front(
    monkeypatch,
    tmp_path,
) -> None:
    """The deferred renderer must never expose its empty black Qt board."""

    import threading

    import zlc_frontend.matplotlib_render as render_module
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    application = ensure_qt_app()
    entered = threading.Event()
    release = threading.Event()
    render = render_module.render_pulse_timeline_panel

    def blocked_render(*args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("preview render was not released")
        return render(*args, **kwargs)

    monkeypatch.setattr(
        render_module,
        "render_pulse_timeline_panel",
        blocked_render,
    )
    body = open_pulse_editor(
        path=PULSE_PATH,
        pulses_root=tmp_path / "pulses",
        output_root=tmp_path / "output",
    )
    try:
        _until(
            application,
            lambda: body.window() is not body and body.window().isVisible(),
        )
        _click_tab(body, body.preview_view)
        assert entered.wait(5.0)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert body.preview_view.preview_placeholder.isVisible()
        assert not body.preview_host.isVisible()

        release.set()
        _until(
            application,
            lambda: body.preview_host.front_frame is not None
            and body.preview_host.isVisible(),
        )
        assert not body.preview_view.preview_placeholder.isVisible()
    finally:
        release.set()
        close_pulse_editor(application, body)
        assert body.worker_idle


def test_show_off_rows_is_a_visible_switch_that_rebuilds_the_same_plot(preview_body) -> None:
    application = ensure_qt_app()
    body = preview_body
    switch = body.preview_view.preview_include_off
    assert not switch.isChecked()
    before_frame = body.preview_host.front_frame
    before = _visible_pulse_payload(body.preview_host.board)
    assert before_frame is not None and before is not None

    QtTest.QTest.mouseClick(switch, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: body.preview_host.front_frame is not before_frame
        and _visible_pulse_payload(body.preview_host.board) is not None
        and len(_visible_pulse_payload(body.preview_host.board).row_keys)
        > len(before.row_keys),
    )
    after_frame = body.preview_host.front_frame
    after = _visible_pulse_payload(body.preview_host.board)
    assert after_frame is not None and after is not None
    timeline = project_pulse_preview(load_pulse_document(PULSE_PATH))
    assert tuple(after.row_labels) == tuple(row.label for row in timeline.rows)
    digital = tuple(row for row in timeline.rows if row.port_kind == "digital")
    analog = tuple(row for row in timeline.rows if row.port_kind == "dac")
    assert tuple(after.row_keys[: len(digital)]) == tuple(
        row.row_id for row in digital
    )
    # Analog rows use renderer-local unique hit-test keys; their visible labels
    # remain the exact board names and their keys retain the typed row id.
    assert all(
        key.endswith(f":{row.row_id}")
        for key, row in zip(
            after.row_keys[len(digital) :], analog, strict=True
        )
    )
    assert after_frame.panels[0].raster.pixels != before_frame.panels[0].raster.pixels
    assert "(all channels)" in body.preview_view.preview_status.text()


def test_size_popup_is_operator_owned_but_reentering_preview_restores_auto(preview_body) -> None:
    application = ensure_qt_app()
    body = preview_body
    combo = body.preview_view.preview_size_combo
    assert combo.currentText() == "1x2"

    previous = body.preview_host.front_frame
    _choose_combo(application, combo, "8x8")
    _until(
        application,
        lambda: body.preview_host.front_frame is not previous
        and body.preview_view.preview_size_pinned,
    )
    large = body.preview_host.front_frame.panels[0].raster
    assert combo.currentText() == "8x8"

    _click_tab(body, body.schedule_view)
    _until(application, lambda: body.tabs.currentWidget() is body.schedule_view)
    _click_tab(body, body.preview_view)
    _until(
        application,
        lambda: body.tabs.currentWidget() is body.preview_view
        and not body.preview_view.preview_size_pinned
        and combo.currentText() == "1x2",
    )
    automatic = body.preview_host.front_frame.panels[0].raster
    assert automatic.width < large.width and automatic.height < large.height


def test_selectors_switch_alone_arms_the_preview_wheel(preview_body) -> None:
    application = ensure_qt_app()
    body = preview_body
    board = body.preview_host.board
    switch = body.preview_view.preview_selectors_switch
    before = _visible_pulse_payload(board)
    assert before is not None
    home = before.viewport.home_x_limits
    assert before.viewport.x_limits == home
    assert not switch.isChecked()

    _send_wheel(application, board)
    parked = _visible_pulse_payload(board)
    assert parked is not None and parked.viewport.x_limits == home

    QtTest.QTest.mouseClick(switch, QtCore.Qt.LeftButton)
    _until(application, switch.isChecked)
    _send_wheel(application, board)
    _until(
        application,
        lambda: _visible_pulse_payload(board) is not None
        and _visible_pulse_payload(board).viewport.x_limits != home,
    )
    after = _visible_pulse_payload(board)
    assert after is not None
    assert after.viewport.home_x_limits == home
    assert after.viewport.y_limits == before.viewport.y_limits


def test_middle_drag_repaints_continuously_before_mouse_release(preview_body) -> None:
    """The formal Preview follows a held middle-button drag, not its release."""

    application = ensure_qt_app()
    body = preview_body
    board = body.preview_host.board
    switch = body.preview_view.preview_selectors_switch
    QtTest.QTest.mouseClick(switch, QtCore.Qt.LeftButton)
    _until(application, switch.isChecked)

    initial = _visible_pulse_payload(board)
    assert initial is not None
    initial_revision = initial.viewport.display_revision
    layout_bounds = initial.viewport.plot_bounds
    authoring_before = (
        body._controller.current_document,
        body._controller.current_document_generation,
        body._controller.current_editor_revision,
        body._controller.dirty,
    )
    centre = board.rect().center()
    revisions_seen_while_held: set[int] = set()

    QtTest.QTest.mousePress(
        board,
        QtCore.Qt.MiddleButton,
        pos=centre,
    )
    try:
        deadline = time.monotonic() + 3.0
        step = 0
        while time.monotonic() < deadline and len(revisions_seen_while_held) < 2:
            # Keep authoring new view revisions while the button remains down.
            # The back-and-forth path stays safely inside the plot on every
            # supported formal window size.
            offset = int(board.width() * (0.18 + (step % 6) * 0.05))
            if (step // 7) % 2:
                offset = -offset
            position = QtCore.QPoint(
                max(1, min(board.width() - 2, centre.x() + offset)),
                centre.y(),
            )
            _send_mouse_move_with_buttons(
                application,
                board,
                position,
                QtCore.Qt.MiddleButton,
            )
            payload = _visible_pulse_payload(board)
            if (
                payload is not None
                and payload.viewport.display_revision > initial_revision
            ):
                assert payload.viewport.plot_bounds == pytest.approx(
                    layout_bounds,
                    abs=1e-12,
                )
                revisions_seen_while_held.add(
                    payload.viewport.display_revision
                )
            step += 1
            time.sleep(0.008)
        assert len(revisions_seen_while_held) >= 2, revisions_seen_while_held
    finally:
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=centre,
        )
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)

    # A Preview viewport is presentation-only: the continuously emitted view
    # candidates must never dirty or revise the authoritative pulse document.
    authoring_after = (
        body._controller.current_document,
        body._controller.current_document_generation,
        body._controller.current_editor_revision,
        body._controller.dirty,
    )
    assert authoring_after == authoring_before

    _until(application, lambda: body.worker_idle)
    settled = _visible_pulse_payload(board)
    assert settled is not None
    settled_revision = settled.viewport.display_revision
    _send_wheel(application, board)
    _until(
        application,
        lambda: _visible_pulse_payload(board) is not None
        and _visible_pulse_payload(board).viewport.display_revision
        > settled_revision,
    )


def test_failed_latest_drag_frame_releases_its_exact_pending_intent(
    preview_body,
    monkeypatch,
) -> None:
    """An admitted intermediate cannot orphan a newer failed view intent."""

    import zlc_frontend.matplotlib_render as rendering

    application = ensure_qt_app()
    body = preview_body
    board = body.preview_host.board
    switch = body.preview_view.preview_selectors_switch
    QtTest.QTest.mouseClick(switch, QtCore.Qt.LeftButton)
    _until(application, switch.isChecked)
    initial = _visible_pulse_payload(board)
    assert initial is not None
    base_revision = initial.viewport.display_revision
    first_revision = base_revision + 1
    failed_revision = base_revision + 2
    original_render = rendering.render_pulse_timeline_panel

    def controlled_render(*args, **kwargs):
        revision = int(kwargs["display_revision"])
        if revision == first_revision:
            # Keep the first candidate in flight until the second human move
            # becomes the exact latest request.
            time.sleep(0.12)
        if revision == failed_revision:
            # Leave enough owner turns for the successful intermediate raster
            # to be visibly presented before the latest request fails.
            time.sleep(0.12)
            raise RuntimeError("forced latest Preview render failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(
        rendering,
        "render_pulse_timeline_panel",
        controlled_render,
    )
    centre = board.rect().center()
    QtTest.QTest.mousePress(
        board,
        QtCore.Qt.MiddleButton,
        pos=centre,
    )
    try:
        _send_mouse_move_with_buttons(
            application,
            board,
            centre + QtCore.QPoint(70, 0),
            QtCore.Qt.MiddleButton,
        )
        _send_mouse_move_with_buttons(
            application,
            board,
            centre + QtCore.QPoint(120, 0),
            QtCore.Qt.MiddleButton,
        )
        _until(
            application,
            lambda: _visible_pulse_payload(board) is not None
            and _visible_pulse_payload(board).viewport.display_revision
            == first_revision,
        )
        _until(
            application,
            lambda: "forced latest Preview render failure"
            in body._controller.preview_update().preview_error,
        )
    finally:
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=centre,
        )
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)

    # Restore the real worker and prove the exact failed pending capability was
    # released: a fresh human zoom must be admitted and presented.
    monkeypatch.setattr(
        rendering,
        "render_pulse_timeline_panel",
        original_render,
    )
    _send_wheel(application, board)
    _until(
        application,
        lambda: _visible_pulse_payload(board) is not None
        and _visible_pulse_payload(board).viewport.display_revision
        > failed_revision,
    )


def test_failed_inflight_present_reissues_the_newer_human_drag_intent(
    preview_body,
    monkeypatch,
) -> None:
    """A Qt-present fault cannot discard the mailbox's later drag intent."""

    import threading

    import zlc_frontend.matplotlib_render as rendering

    application = ensure_qt_app()
    body = preview_body
    board = body.preview_host.board
    switch = body.preview_view.preview_selectors_switch
    QtTest.QTest.mouseClick(switch, QtCore.Qt.LeftButton)
    _until(application, switch.isChecked)
    initial = _visible_pulse_payload(board)
    assert initial is not None
    first_revision = initial.viewport.display_revision + 1
    latest_revision = first_revision + 1

    first_started = threading.Event()
    release_first = threading.Event()
    latest_started = threading.Event()
    release_latest = threading.Event()
    original_render = rendering.render_pulse_timeline_panel

    def controlled_render(*args, **kwargs):
        revision = int(kwargs["display_revision"])
        if revision == first_revision:
            first_started.set()
            assert release_first.wait(5.0)
        elif revision == latest_revision:
            latest_started.set()
            assert release_latest.wait(5.0)
        return original_render(*args, **kwargs)

    monkeypatch.setattr(
        rendering,
        "render_pulse_timeline_panel",
        controlled_render,
    )
    original_present = body.preview_host.present_panel
    failed_intermediate = False

    def controlled_present(raster, payload, **kwargs):
        nonlocal failed_intermediate
        revision = int(payload.viewport.display_revision)
        if revision == first_revision and not failed_intermediate:
            failed_intermediate = True
            raise RuntimeError("forced stale intermediate present failure")
        return original_present(raster, payload, **kwargs)

    monkeypatch.setattr(body.preview_host, "present_panel", controlled_present)

    centre = board.rect().center()
    QtTest.QTest.mousePress(
        board,
        QtCore.Qt.MiddleButton,
        pos=centre,
    )
    try:
        _send_mouse_move_with_buttons(
            application,
            board,
            centre + QtCore.QPoint(65, 0),
            QtCore.Qt.MiddleButton,
        )
        _until(application, first_started.is_set)
        _send_mouse_move_with_buttons(
            application,
            board,
            centre + QtCore.QPoint(115, 0),
            QtCore.Qt.MiddleButton,
        )
        binding = next(iter(board._numeric_bindings.values()))
        assert body._pending_preview_revision == first_revision
        assert binding.queued_viewport_limits is not None

        release_first.set()
        _until(application, latest_started.is_set)
        assert failed_intermediate
        assert body._pending_preview_revision == latest_revision
        assert body._pending_preview_origin is not None

        release_latest.set()
        _until(
            application,
            lambda: _visible_pulse_payload(board) is not None
            and _visible_pulse_payload(board).viewport.display_revision
            == latest_revision,
        )
    finally:
        release_first.set()
        release_latest.set()
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=centre,
        )
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_save_figure_button_exports_the_visible_preview(
    preview_body,
    tmp_path,
    monkeypatch,
) -> None:
    application = ensure_qt_app()
    body = preview_body
    target = tmp_path / "operator-preview.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "Pulse figure (*.png)"),
    )

    QtTest.QTest.mouseClick(
        body.preview_view.preview_save_figure_button,
        QtCore.Qt.LeftButton,
    )
    _until(
        application,
        lambda: target.exists()
        and target.stat().st_size > 0
        and body.worker_idle,
    )
    image = QtGui.QImage(str(target))
    assert not image.isNull()
    assert image.width() > body.preview_host.width()
    assert image.height() > body.preview_host.height()


@pytest.mark.parametrize("scale", ("1", "1.25", "1.5", "1.75", "2", "3"))
def test_preview_raster_tracks_the_real_screen_dpr_in_a_fresh_qt_process(
    scale,
    tmp_path,
) -> None:
    """Every supported screen scale gets Qt's exact physical pixel box."""

    source = r'''
import os
from pathlib import Path
import math
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

from PyQt5 import QtCore, QtTest
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_workbench.pulse_editor.app import open_pulse_editor

app = ensure_qt_app()
path = Path("pulses/imaging_template.json").resolve()
root = Path(os.environ["ZLC_TEST_WORKSPACE"]).resolve()
body = open_pulse_editor(
    path=path,
    pulses_root=root / "pulses",
    output_root=root / "output",
)
index = body.tabs.indexOf(body.preview_view)
bar = body.tabs.tabBar()
QtTest.QTest.mouseClick(
    bar,
    QtCore.Qt.LeftButton,
    pos=bar.tabRect(index).center(),
)
deadline = time.monotonic() + 12.0
ratio = float(body.devicePixelRatioF())
qround = lambda value: int(math.floor(value + 0.5))
while time.monotonic() < deadline:
    frame = body.preview_host.front_frame
    board = body.preview_host.board
    if (
        frame is not None
        and frame.panels[0].raster.width == qround(board.width() * ratio)
        and frame.panels[0].raster.height == qround(board.height() * ratio)
    ):
        break
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    time.sleep(0.005)
assert body.preview_host.front_frame is not None
assert abs(ratio - float(os.environ["EXPECTED_SCALE"])) < 0.01, ratio
raster = body.preview_host.front_frame.panels[0].raster
board = body.preview_host.board
assert raster.width == qround(board.width() * ratio), (
    raster.width, board.width(), ratio
)
assert raster.height == qround(board.height() * ratio), (
    raster.height, board.height(), ratio
)
physical_front = board.grab()
assert (raster.width, raster.height) == (
    physical_front.width(), physical_front.height()
), (
    (raster.width, raster.height),
    (physical_front.width(), physical_front.height()),
    ratio,
)
body.request_close(discard_unsaved=True)
deadline = time.monotonic() + 10.0
while not body.permanently_closed and time.monotonic() < deadline:
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    time.sleep(0.005)
assert body.permanently_closed and body.worker_idle
QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents(QtCore.QEventLoop.AllEvents, 20)
'''
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["QT_SCALE_FACTOR"] = scale
    environment["EXPECTED_SCALE"] = scale
    environment["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    environment["ZLC_TEST_WORKSPACE"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
