"""Fluent Target-manifest page for the Pulse Workbench."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    ACCENT,
    GREY,
    ElidedLabel,
    FluentButton,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentSpinBox,
    signals_blocked,
)
from zlc_pulse import (
    PORT_DAC,
    PORT_DIGITAL,
    PulseTargetManifest,
    PulseTargetPortDraft,
    build_pulse_target_manifest,
    pulse_target_port_drafts,
    pulse_target_port_width_spec,
)

from ._layout import px, row_height


# QSpinBox stores a C++ int; this is a widget representation boundary, not a
# Pulse-target width limit.
_QT_SPINBOX_MAXIMUM = (1 << 31) - 1


@dataclass
class _TargetRowWidgets:
    key: str
    kind: str
    clock_key: str | None
    signal: FluentLineEdit
    endpoints: FluentLineEdit
    width: FluentSpinBox
    clock_endpoint: FluentLineEdit
    remove_button: FluentButton
    lane_order: tuple[int, ...]


class PulseTargetView(QtWidgets.QWidget):
    """One manifest projection; mutations remain draft-only until Apply."""

    applyRequested = QtCore.pyqtSignal(object)
    feedbackRequested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        manifest: PulseTargetManifest,
        *,
        editable: bool,
        mode: str,
        parent=None,
    ) -> None:
        if not isinstance(manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest")
        super().__init__(parent)
        self._manifest_fingerprint = ""
        self._editable = False
        self._rows: list[_TargetRowWidgets] = []
        self._build_ui()
        self.set_manifest(manifest, editable=editable, mode=mode)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(px(10), px(10), px(10), px(10))
        layout.setSpacing(px(8))

        header = FluentFrame()
        header.setFixedHeight(px(58, minimum=48))
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(px(12), px(8), px(12), px(8))
        header_layout.setSpacing(px(8))
        self.status_label = ElidedLabel("")
        header_layout.addWidget(self.status_label, 1)

        self.add_digital_button = FluentButton("Add Digital", color=ACCENT)
        self.add_dac_button = FluentButton("Add DAC", color=ACCENT)
        self.apply_button = FluentButton("Apply target", color=ACCENT)
        header_layout.addWidget(self.add_digital_button)
        header_layout.addWidget(self.add_dac_button)
        header_layout.addWidget(self.apply_button)
        layout.addWidget(header)

        self.cards_scroll = FluentScrollArea()
        self.cards_scroll.setObjectName("pulseTargetScroll")
        self.cards_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Reserve the Fluent scrollbar gutter so adding a port cannot move the
        # form horizontally when the first overflow appears.
        self.cards_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.cards_body = QtWidgets.QWidget()
        self.cards_body.setStyleSheet("background: transparent;")
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_body)
        self.cards_layout.setContentsMargins(px(4), px(4), px(4), px(8))
        self.cards_layout.setSpacing(px(8))

        self.digital_card = FluentGroupBox("Digital outputs")
        self.digital_layout = QtWidgets.QGridLayout(self.digital_card)
        self.digital_layout.setContentsMargins(px(12), px(10), px(12), px(12))
        self.digital_layout.setHorizontalSpacing(px(10, minimum=6))
        self.digital_layout.setVerticalSpacing(px(6, minimum=4))
        self.digital_layout.setColumnStretch(0, 1)
        self.digital_layout.setColumnStretch(1, 1)
        for column, text in enumerate(("signal name", "endpoint", "")):
            self.digital_layout.addWidget(self._header_label(text), 0, column)
        self.cards_layout.addWidget(self.digital_card)

        self.dac_card = FluentGroupBox("DAC outputs")
        self.dac_layout = QtWidgets.QGridLayout(self.dac_card)
        self.dac_layout.setContentsMargins(px(12), px(10), px(12), px(12))
        self.dac_layout.setHorizontalSpacing(px(10, minimum=6))
        self.dac_layout.setVerticalSpacing(px(6, minimum=4))
        self.dac_layout.setColumnStretch(0, 1)
        self.dac_layout.setColumnStretch(2, 2)
        self.dac_layout.setColumnStretch(3, 1)
        for column, text in enumerate(
            (
                "signal name",
                "width",
                "data endpoints (bit order)",
                "latch endpoint",
                "",
            )
        ):
            self.dac_layout.addWidget(self._header_label(text), 0, column)
        self.cards_layout.addWidget(self.dac_card)
        self.cards_layout.addStretch(1)
        self.cards_scroll.setWidget(self.cards_body)
        layout.addWidget(self.cards_scroll, 1)

        self.add_digital_button.clicked.connect(self._add_digital)
        self.add_dac_button.clicked.connect(self._add_dac)
        self.apply_button.clicked.connect(self._emit_apply)

    def set_manifest(
        self,
        manifest: PulseTargetManifest,
        *,
        editable: bool,
        mode: str,
    ) -> None:
        if not isinstance(manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest")
        editable = bool(editable)
        mode = str(mode).strip().lower()
        fingerprint = manifest.fingerprint
        self._editable = editable
        if fingerprint != self._manifest_fingerprint:
            self._reconcile_drafts(pulse_target_port_drafts(manifest))
            self._manifest_fingerprint = fingerprint
        self._set_editable(editable)
        if editable:
            self.status_label.setText(
                "Offline target draft · edit signal and endpoints, or add/remove "
                "complete Digital and DAC ports. Nothing changes until Apply target."
            )
        else:
            self.status_label.setText(
                f"{mode.title()} backend target · read-only endpoints published by "
                "the connected backend."
            )

    def _reconcile_drafts(
        self,
        drafts: tuple[PulseTargetPortDraft, ...],
    ) -> None:
        ordered = tuple(row for row in drafts if row.kind == PORT_DIGITAL) + tuple(
            row for row in drafts if row.kind == PORT_DAC
        )
        if len({draft.key for draft in ordered}) != len(ordered):
            raise ValueError("Pulse target draft keys must be unique")

        previous_rows = tuple(self._rows)
        existing = {row.key: row for row in previous_rows}
        reconciled: list[_TargetRowWidgets] = []
        for draft in ordered:
            row = existing.pop(draft.key, None)
            if row is None:
                row = self._make_row(draft)
            elif row.kind != draft.kind:
                self._destroy_row(row)
                row = self._make_row(draft)
            else:
                self._reconcile_row(row, draft)
            reconciled.append(row)

        for removed in existing.values():
            self._destroy_row(removed)

        structure_changed = len(previous_rows) != len(reconciled) or any(
            before is not after
            for before, after in zip(previous_rows, reconciled)
        )
        self._rows = reconciled
        if structure_changed:
            self._place_rows()

        digital_count = sum(row.kind == PORT_DIGITAL for row in self._rows)
        dac_count = sum(row.kind == PORT_DAC for row in self._rows)
        digital_title = f"Digital outputs · {digital_count}"
        dac_title = f"DAC outputs · {dac_count}"
        if self.digital_card.title() != digital_title:
            self.digital_card.setTitle(digital_title)
        if self.dac_card.title() != dac_title:
            self.dac_card.setTitle(dac_title)
        self._set_editable(self._editable)

    def _reconcile_row(
        self,
        row: _TargetRowWidgets,
        draft: PulseTargetPortDraft,
    ) -> None:
        if row.key != draft.key or row.kind != draft.kind:
            raise ValueError("Only an identical target-row identity can be reconciled")
        endpoint_text = ", ".join(draft.endpoints)
        clock_text = draft.clock_endpoint or ""
        with signals_blocked(
            row.signal,
            row.endpoints,
            row.width,
            row.clock_endpoint,
        ):
            self._set_line_text(row.signal, draft.signal)
            self._set_line_text(row.endpoints, endpoint_text)
            if row.width.value() != draft.width:
                row.width.setValue(draft.width)
            self._set_line_text(row.clock_endpoint, clock_text)
        row.endpoints.setToolTip(endpoint_text)
        row.clock_endpoint.setToolTip(clock_text)
        row.clock_key = draft.clock_key
        row.lane_order = draft.lane_order

    @staticmethod
    def _set_line_text(field: FluentLineEdit, text: str) -> None:
        """Update authoritative text without disturbing an unchanged editor."""

        text = str(text)
        if field.text() == text:
            return
        cursor = field.cursorPosition()
        selection_start = field.selectionStart()
        selection_length = len(field.selectedText())
        had_focus = field.hasFocus()
        field.setText(text)
        if not had_focus:
            field.setCursorPosition(0)
            return
        if selection_start >= 0 and selection_length:
            start = min(selection_start, len(text))
            field.setSelection(start, min(selection_length, len(text) - start))
        else:
            field.setCursorPosition(min(cursor, len(text)))

    @staticmethod
    def _placed_widgets(row: _TargetRowWidgets) -> tuple[QtWidgets.QWidget, ...]:
        if row.kind == PORT_DIGITAL:
            return row.signal, row.endpoints, row.remove_button
        return (
            row.signal,
            row.width,
            row.endpoints,
            row.clock_endpoint,
            row.remove_button,
        )

    def _destroy_row(self, row: _TargetRowWidgets) -> None:
        layout = self.digital_layout if row.kind == PORT_DIGITAL else self.dac_layout
        for widget in self._placed_widgets(row):
            layout.removeWidget(widget)
        for widget in (
            row.signal,
            row.endpoints,
            row.width,
            row.clock_endpoint,
            row.remove_button,
        ):
            widget.hide()
            widget.deleteLater()

    def _place_rows(self) -> None:
        """Move stable row widgets to their desired grid positions."""

        positions = {PORT_DIGITAL: 1, PORT_DAC: 1}
        desired: list[
            tuple[QtWidgets.QGridLayout, QtWidgets.QWidget, int, int]
        ] = []
        for row in self._rows:
            position = positions[row.kind]
            if row.kind == PORT_DIGITAL:
                desired.extend(
                    (
                        (self.digital_layout, row.signal, position, 0),
                        (self.digital_layout, row.endpoints, position, 1),
                        (self.digital_layout, row.remove_button, position, 2),
                    )
                )
            else:
                desired.extend(
                    (
                        (self.dac_layout, row.signal, position, 0),
                        (self.dac_layout, row.width, position, 1),
                        (self.dac_layout, row.endpoints, position, 2),
                        (self.dac_layout, row.clock_endpoint, position, 3),
                        (self.dac_layout, row.remove_button, position, 4),
                    )
                )
            positions[row.kind] += 1

        moved: list[tuple[QtWidgets.QGridLayout, QtWidgets.QWidget, int, int]] = []
        for layout, widget, row_index, column_index in desired:
            item_index = layout.indexOf(widget)
            if item_index >= 0:
                current_row, current_column, row_span, column_span = (
                    layout.getItemPosition(item_index)
                )
                if (
                    current_row == row_index
                    and current_column == column_index
                    and row_span == 1
                    and column_span == 1
                ):
                    continue
                layout.removeWidget(widget)
            moved.append((layout, widget, row_index, column_index))

        for layout, widget, row_index, column_index in moved:
            layout.addWidget(widget, row_index, column_index)

    @staticmethod
    def _header_label(text: str) -> FluentLabel:
        label = FluentLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        label.setFixedHeight(row_height())
        return label

    def _make_row(self, draft: PulseTargetPortDraft) -> _TargetRowWidgets:
        signal = FluentLineEdit(draft.signal)
        signal.setPlaceholderText("operator signal name")
        signal.setToolTip(
            "Operator-visible signal name. The stable internal target identity "
            "is intentionally not used as a channel label."
        )
        signal.setFixedHeight(row_height())

        endpoints = FluentLineEdit(", ".join(draft.endpoints))
        endpoints.setPlaceholderText(
            "one endpoint"
            if draft.kind == PORT_DIGITAL
            else "comma-separated endpoints in DAC bit order"
        )
        endpoints.setToolTip(", ".join(draft.endpoints))
        endpoints.setCursorPosition(0)
        endpoints.setFixedHeight(row_height())

        width = FluentSpinBox()
        width_spec = pulse_target_port_width_spec(draft.kind)
        width.setRange(
            width_spec.minimum,
            (
                _QT_SPINBOX_MAXIMUM
                if width_spec.maximum is None
                else width_spec.maximum
            ),
        )
        width.setValue(draft.width)
        width.setFixedSize(px(82, minimum=70), row_height())
        if draft.kind == PORT_DAC:
            width.valueChanged.connect(
                lambda value, key=draft.key, field=endpoints: self._resize_dac_endpoints(
                    key,
                    field,
                    int(value),
                )
            )

        clock_endpoint = FluentLineEdit(draft.clock_endpoint or "")
        clock_endpoint.setPlaceholderText("paired DAC latch-clock endpoint")
        clock_endpoint.setToolTip(draft.clock_endpoint or "")
        clock_endpoint.setCursorPosition(0)
        clock_endpoint.setFixedHeight(row_height())

        remove_button = FluentButton("Remove", color=GREY)
        remove_button.setFixedSize(px(86, minimum=74), row_height())
        remove_button.setToolTip(
            "Remove this channel from the draft. A DAC and its latch clock "
            "are removed together."
        )
        remove_button.clicked.connect(
            lambda _checked=False, key=draft.key: self._remove_key(key)
        )

        return _TargetRowWidgets(
            draft.key,
            draft.kind,
            draft.clock_key,
            signal,
            endpoints,
            width,
            clock_endpoint,
            remove_button,
            draft.lane_order,
        )

    def _set_editable(self, editable: bool) -> None:
        for row in self._rows:
            row.signal.setReadOnly(not editable)
            row.signal.setEnabled(editable)
            row.endpoints.setReadOnly(not editable)
            row.endpoints.setEnabled(editable)
            row.width.setReadOnly(not editable)
            row.width.setEnabled(editable)
            row.width.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.UpDownArrows
                if editable
                else QtWidgets.QAbstractSpinBox.NoButtons
            )
            row.clock_endpoint.setReadOnly(not editable)
            row.clock_endpoint.setEnabled(editable)
            row.remove_button.setVisible(True)
            row.remove_button.setEnabled(editable)
        for button in (
            self.add_digital_button,
            self.add_dac_button,
            self.apply_button,
        ):
            button.setVisible(True)
            button.setEnabled(editable)

    def _used_keys(self) -> set[str]:
        return {
            key
            for row in self._rows
            for key in (row.key, row.clock_key)
            if key is not None
        }

    def _allocate_key(self, stem: str) -> str:
        used = self._used_keys()
        index = 1
        while f"{stem}_{index}" in used:
            index += 1
        return f"{stem}_{index}"

    @staticmethod
    def _endpoint_values(text: str) -> tuple[str, ...]:
        return tuple(value.strip() for value in str(text).split(",") if value.strip())

    def _resize_dac_endpoints(
        self,
        key: str,
        field: FluentLineEdit,
        width: int,
    ) -> None:
        width = pulse_target_port_width_spec(PORT_DAC).normalize(int(width))
        endpoints = list(self._endpoint_values(field.text()))
        if len(endpoints) < width:
            endpoints.extend(
                f"endpoint:{key}[{bit}]" for bit in range(len(endpoints), width)
            )
        elif len(endpoints) > width:
            endpoints = endpoints[:width]
        field.setText(", ".join(endpoints))

    def _queue_reveal_row(self, key: str) -> None:
        """Reveal one newly inserted row after Qt publishes the new scroll range."""

        def reveal() -> None:
            row = next((item for item in self._rows if item.key == key), None)
            if row is not None:
                self.cards_scroll.ensureWidgetVisible(row.signal)

        QtCore.QTimer.singleShot(0, reveal)

    def _current_drafts(self) -> tuple[PulseTargetPortDraft, ...]:
        drafts = []
        for row in self._rows:
            endpoints = self._endpoint_values(row.endpoints.text())
            if len(endpoints) != row.width.value():
                raise ValueError(
                    f"{row.signal.text().strip() or row.key}: width is "
                    f"{row.width.value()} but {len(endpoints)} endpoints are listed"
                )
            drafts.append(
                PulseTargetPortDraft(
                    row.key,
                    row.kind,
                    row.signal.text().strip(),
                    endpoints,
                    row.clock_key,
                    (
                        row.clock_endpoint.text().strip()
                        if row.kind == PORT_DAC
                        else None
                    ),
                    (
                        row.lane_order
                        if len(row.lane_order) == len(endpoints)
                        else tuple(range(len(endpoints)))
                    ),
                )
            )
        return tuple(drafts)

    def _add_digital(self) -> None:
        try:
            current = self._current_drafts()
        except ValueError as error:
            self.feedbackRequested.emit(str(error))
            return
        key = self._allocate_key("digital")
        draft = PulseTargetPortDraft(
            key,
            PORT_DIGITAL,
            key,
            (f"endpoint:{key}",),
        )
        split = next(
            (index for index, row in enumerate(current) if row.kind == PORT_DAC),
            len(current),
        )
        self._reconcile_drafts((*current[:split], draft, *current[split:]))
        self._queue_reveal_row(key)

    def _add_dac(self) -> None:
        try:
            current = self._current_drafts()
        except ValueError as error:
            self.feedbackRequested.emit(str(error))
            return
        key = self._allocate_key("dac")
        clock_key = self._allocate_key(f"{key}_clock")
        width = pulse_target_port_width_spec(PORT_DAC).default
        draft = PulseTargetPortDraft(
            key,
            PORT_DAC,
            key,
            tuple(f"endpoint:{key}[{bit}]" for bit in range(width)),
            clock_key,
            f"endpoint:{clock_key}",
        )
        self._reconcile_drafts((*current, draft))
        self._queue_reveal_row(key)

    def _remove_key(self, key: str) -> None:
        try:
            current = self._current_drafts()
        except ValueError as error:
            self.feedbackRequested.emit(str(error))
            return
        drafts = tuple(row for row in current if row.key != key)
        self._reconcile_drafts(drafts)

    def draft_manifest(self) -> PulseTargetManifest:
        drafts = self._current_drafts()
        return build_pulse_target_manifest(drafts)

    def _emit_apply(self) -> None:
        try:
            manifest = self.draft_manifest()
        except (TypeError, ValueError) as error:
            self.feedbackRequested.emit(str(error))
            return
        self.applyRequested.emit(manifest)


__all__ = ["PulseTargetView"]
