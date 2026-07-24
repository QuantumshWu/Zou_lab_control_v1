"""Frozen operator contract for the document-fed Pulse schedule view."""

from __future__ import annotations

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_pulse import (
    DEFAULT_PERIOD_DURATION,
    DEFAULT_REPEAT_COUNT,
    DEFAULT_SCAN_SWEEP_COUNT,
    DEFAULT_TIME_UNIT,
    MIN_REPEAT_COUNT,
    MIN_SCAN_SWEEP_COUNT,
    PORT_DAC,
    PORT_DIGITAL,
    TIME_UNIT_CHOICES,
    PulseDocument,
    PulsePeriod,
    PulsePortSpec,
    PulseTarget,
    PulseTargetPortDraft,
    RepeatRegion,
    build_pulse_target_manifest,
    new_period,
    pulse_target_port_width_spec,
)
from zlc_workbench.pulse_editor.scan_view import PulseScanView
from zlc_workbench.pulse_editor.schedule_view import PulseScheduleView
from zlc_workbench.pulse_editor.target_view import PulseTargetView
from zlc_workbench.pulse import project_pulse_preview


def _target(count: int) -> PulseTarget:
    lanes = tuple(f"ch{index}" for index in range(count))
    return PulseTarget(
        lanes,
        tuple(
            PulsePortSpec(
                lane,
                PORT_DIGITAL,
                (lane,),
                chr(ord("A") + index),
                None,
                1,
                "binary",
                0,
                None,
            )
            for index, lane in enumerate(lanes)
        ),
    )


def _interaction_document(*, period_count: int = 8) -> PulseDocument:
    target = _target(8)
    return PulseDocument(
        "focus contract",
        target,
        10.0,
        tuple(
            PulsePeriod(
                f"p{index + 1}",
                100,
                "ns",
                f"name {index + 1}",
                (0,) * len(target.raw_lanes),
            )
            for index in range(period_count)
        ),
        visible_ports=tuple(port.key for port in target.ports),
    )


def _process_events(application, count: int = 5) -> None:
    for _index in range(count):
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def _click_period(view: PulseScheduleView, index: int) -> None:
    card = view.period_cards()[index]
    position = view.drag_container.mapFromGlobal(
        card.mapToGlobal(card.rect().center())
    )
    QtTest.QTest.mouseClick(
        view.drag_container,
        QtCore.Qt.LeftButton,
        pos=position,
    )


def _click_gap(view: PulseScheduleView, position: int) -> None:
    item_position = view.drag_container._items_pos_of_period_gap(position)
    x = view.drag_container._indicator_x_for_items_pos(item_position)
    y = view.period_cards()[0].geometry().center().y()
    QtTest.QTest.mouseClick(
        view.drag_container,
        QtCore.Qt.LeftButton,
        pos=QtCore.QPoint(x, y),
    )


def test_summary_keeps_expanded_pulse_and_repeat_warnings() -> None:
    application = ensure_qt_app()
    target = _target(2)
    document = PulseDocument(
        "summary contract",
        target,
        10.0,
        (
            PulsePeriod("p1", 100, "ns", "one", (1, 1)),
            PulsePeriod("p2", 100, "ns", "two", (0, 0)),
            PulsePeriod("p3", 100, "ns", "three", (0, 0)),
        ),
        visible_ports=("ch0",),
        repeat=RepeatRegion("p2", "p3", 2),
    )
    view = PulseScheduleView(document)
    try:
        assert view.summary_text() == (
            "1/2 ports visible | 3 periods | step 10 ns | 500 ns | "
            "2 pulses | repeat ∞ + P2-P3 x2 | hidden active: B | "
            "table restart high every 500 ns: A, B"
        )
    finally:
        view.close()
        _process_events(application)


def test_offline_authoring_does_not_preflight_frozen_streamer_geometry() -> None:
    """A valid authoring target may exceed the currently deployed FPGA ABI.

    Edit and Preview must remain usable; Run/Deploy is the only boundary that
    may reject the frozen hardware geometry.  This reproduces the former
    Qt-slot crash after adding an Offline DAC port.
    """

    application = ensure_qt_app()
    lanes = tuple(f"authoring_{index}" for index in range(70))
    target = PulseTarget(
        lanes,
        tuple(
            PulsePortSpec(
                lane,
                PORT_DIGITAL,
                (lane,),
                f"Offline {index}",
                None,
                1,
                "binary",
                0,
                None,
            )
            for index, lane in enumerate(lanes)
        ),
    )
    document = PulseDocument(
        "offline authoring geometry",
        target,
        10.0,
        (
            PulsePeriod(
                "p1",
                100,
                "ns",
                "one logical pulse",
                (1,) + (0,) * (len(lanes) - 1),
            ),
        ),
        visible_ports=(lanes[0],),
    )
    view = PulseScheduleView(document)
    try:
        assert " | 1 pulses | " in view.summary_text()
        preview = project_pulse_preview(document)
        assert len(preview.rows) == 70
        assert preview.rows[0].label == "Offline 0"
    finally:
        view.close()
        _process_events(application)


def test_summary_repeat_wording_uses_the_same_three_state_policy() -> None:
    application = ensure_qt_app()
    base = _interaction_document(period_count=3)
    first = base.periods[0].period_id
    second = base.periods[1].period_id
    last = base.periods[-1].period_id
    cases = (
        (base, "repeat ∞"),
        (
            replace(base, repeat=RepeatRegion(second, last, 2)),
            "repeat ∞ + P2-P3 x2",
        ),
        (
            replace(base, repeat=RepeatRegion(first, last, 5)),
            "repeat P1-P3 x5",
        ),
    )
    for document, expected in cases:
        view = PulseScheduleView(document)
        try:
            assert f" | {expected}" in view.summary_text()
        finally:
            view.close()
            _process_events(application)


def test_document_refresh_preserves_valid_selection_focus_cursor_and_scroll() -> None:
    application = ensure_qt_app()
    document = _interaction_document()
    view = PulseScheduleView(document)
    view.resize(900, 500)
    view.show()
    try:
        _process_events(application)
        assert view.timeline_scroll.horizontalScrollBar().maximum() > 0
        assert view.timeline_scroll.verticalScrollBar().maximum() > 0

        _click_period(view, 4)
        editor = view.period_cards()[4].name_edit
        QtTest.QTest.mouseClick(editor, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key_End)
        QtTest.QTest.keyClicks(editor, "x")
        QtTest.QTest.keyClick(
            editor,
            QtCore.Qt.Key_Left,
            QtCore.Qt.ShiftModifier,
        )
        assert QtWidgets.QApplication.focusWidget() is editor
        assert editor.selectedText() == "x"

        horizontal = view.timeline_scroll.horizontalScrollBar()
        vertical = view.timeline_scroll.verticalScrollBar()
        horizontal.setValue(horizontal.maximum() // 2)
        vertical.setValue(vertical.maximum() // 2)
        expected_scroll = (horizontal.value(), vertical.value())

        periods = tuple(
            replace(period, name=editor.text())
            if period.period_id == "p5"
            else period
            for period in document.periods
        )
        document = replace(document, periods=periods)
        assert view.set_document(
            document,
            document_generation=0,
            revision=1,
        )
        _process_events(application)

        refreshed_editor = view.period_cards()[4].name_edit
        assert view._selected_period_id == "p5"
        assert view.drag_container._selected_period_id == "p5"
        assert QtWidgets.QApplication.focusWidget() is refreshed_editor
        assert refreshed_editor.selectedText() == "x"
        assert (horizontal.value(), vertical.value()) == expected_scroll

        _click_gap(view, 3)
        assert view._selected_gap == 3
        changed_again = replace(document, name="another semantic field")
        assert view.set_document(
            changed_again,
            document_generation=0,
            revision=2,
        )
        _process_events(application)
        assert view._selected_period_id is None
        assert view._selected_gap == 3
        assert view.drag_container._selected_gap == 3
    finally:
        view.close()
        _process_events(application)


def test_one_period_add_bracket_emits_the_frozen_feedback() -> None:
    application = ensure_qt_app()
    target = _target(1)
    document = PulseDocument(
        "one period",
        target,
        10.0,
        (PulsePeriod("p1", 100, "ns", "", (0,)),),
        visible_ports=("ch0",),
    )
    view = PulseScheduleView(document)
    feedback: list[str] = []
    repeats: list[tuple[object, object, int]] = []
    view.feedbackRequested.connect(feedback.append)
    view.repeatEdited.connect(
        lambda start, stop, count: repeats.append((start, stop, count))
    )
    view.show()
    try:
        _process_events(application)
        QtTest.QTest.mouseClick(view.bracket_button, QtCore.Qt.LeftButton)
        _process_events(application)
        assert feedback == ["Repeat needs at least two periods."]
        assert repeats == []
    finally:
        view.close()
        _process_events(application)


def test_leaf_editors_consume_owner_grid_defaults_and_unbounded_counts() -> None:
    application = ensure_qt_app()
    base = _interaction_document(period_count=2)
    document = replace(
        base,
        repeat=RepeatRegion("p1", "p2", DEFAULT_REPEAT_COUNT),
    )
    view = PulseScheduleView(document)
    scan = PulseScanView()
    try:
        card = view.period_cards()[0]
        added = new_period(document)
        assert (
            tuple(
                card.unit_combo.itemText(index)
                for index in range(card.unit_combo.count())
            ),
            card.duration_edit._res_step,
            (added.duration, added.unit),
        ) == (
            TIME_UNIT_CHOICES,
            document.time_step_ns,
            (DEFAULT_PERIOD_DURATION, DEFAULT_TIME_UNIT),
        )

        card.duration_edit.setText("13")
        card.duration_edit.editingFinished.emit()
        delay = view.channel_panel.delay_edits["ch0"]
        delay.setText("-13")
        delay.editingFinished.emit()
        assert (card.duration_edit.text(), delay.text()) == ("10", "-10")

        bracket = next(
            item.widget
            for item in view.drag_container.items
            if item.item_type == "bracket_end"
        )
        assert (
            bracket.repeat_spin.minimum(),
            bracket.repeat_spin.maximum() > 999,
            scan.scan_repeats_spin.minimum(),
            scan.scan_repeats_spin.value(),
            scan.scan_repeats_spin.maximum() > 999,
        ) == (
            MIN_REPEAT_COUNT,
            True,
            MIN_SCAN_SWEEP_COUNT,
            DEFAULT_SCAN_SWEEP_COUNT,
            True,
        )
    finally:
        view.close()
        scan.close()
        _process_events(application)


def test_target_width_uses_owner_default_without_a_gui_product_cap() -> None:
    application = ensure_qt_app()
    manifest = build_pulse_target_manifest(
        (PulseTargetPortDraft("ttl", PORT_DIGITAL, "TTL", ("TTL0",)),)
    )
    view = PulseTargetView(manifest, editable=True, mode="offline")
    try:
        view._add_dac()
        row = next(item for item in view._rows if item.kind == PORT_DAC)
        owner = pulse_target_port_width_spec(PORT_DAC)
        assert (
            owner.minimum,
            owner.default,
            owner.maximum,
            row.width.minimum(),
            row.width.value(),
            row.width.maximum() > 32,
        ) == (2, 10, None, 2, 10, True)

        row.width.setValue(33)
        assert (
            len(view._endpoint_values(row.endpoints.text())),
            view.draft_manifest().target.by_key[row.key].width,
        ) == (33, 33)
    finally:
        view.close()
        _process_events(application)
