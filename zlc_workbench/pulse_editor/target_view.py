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
)
from zlc_pulse import (
    PORT_DAC,
    PORT_DIGITAL,
    PulseTargetManifest,
    PulseTargetPortDraft,
    build_pulse_target_manifest,
    pulse_target_port_drafts,
)

from ._layout import px, row_height


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
        self.cards_layout.addWidget(self.digital_card)

        self.dac_card = FluentGroupBox("DAC outputs")
        self.dac_layout = QtWidgets.QGridLayout(self.dac_card)
        self.dac_layout.setContentsMargins(px(12), px(10), px(12), px(12))
        self.dac_layout.setHorizontalSpacing(px(10, minimum=6))
        self.dac_layout.setVerticalSpacing(px(6, minimum=4))
        self.dac_layout.setColumnStretch(0, 1)
        self.dac_layout.setColumnStretch(2, 2)
        self.dac_layout.setColumnStretch(3, 1)
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
        if fingerprint != self._manifest_fingerprint:
            self._populate(pulse_target_port_drafts(manifest))
            self._manifest_fingerprint = fingerprint
            QtCore.QTimer.singleShot(
                0,
                lambda: self.cards_scroll.verticalScrollBar().setValue(0),
            )
        self._editable = editable
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

    @staticmethod
    def _clear_grid(layout: QtWidgets.QGridLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _populate(self, drafts: tuple[PulseTargetPortDraft, ...]) -> None:
        self._clear_grid(self.digital_layout)
        self._clear_grid(self.dac_layout)
        self._rows.clear()
        ordered = tuple(row for row in drafts if row.kind == PORT_DIGITAL) + tuple(
            row for row in drafts if row.kind == PORT_DAC
        )
        digital_headers = ("signal name", "endpoint", "")
        dac_headers = (
            "signal name",
            "width",
            "data endpoints (bit order)",
            "latch endpoint",
            "",
        )
        for column, text in enumerate(digital_headers):
            self.digital_layout.addWidget(self._header_label(text), 0, column)
        for column, text in enumerate(dac_headers):
            self.dac_layout.addWidget(self._header_label(text), 0, column)
        positions = {PORT_DIGITAL: 1, PORT_DAC: 1}
        for draft in ordered:
            row = self._make_row(draft)
            self._rows.append(row)
            position = positions[draft.kind]
            if draft.kind == PORT_DIGITAL:
                self.digital_layout.addWidget(row.signal, position, 0)
                self.digital_layout.addWidget(row.endpoints, position, 1)
                self.digital_layout.addWidget(row.remove_button, position, 2)
            else:
                self.dac_layout.addWidget(row.signal, position, 0)
                self.dac_layout.addWidget(row.width, position, 1)
                self.dac_layout.addWidget(row.endpoints, position, 2)
                self.dac_layout.addWidget(row.clock_endpoint, position, 3)
                self.dac_layout.addWidget(row.remove_button, position, 4)
            positions[draft.kind] += 1
        self.digital_card.setTitle(f"Digital outputs · {positions[PORT_DIGITAL] - 1}")
        self.dac_card.setTitle(f"DAC outputs · {positions[PORT_DAC] - 1}")

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
        width.setRange(1 if draft.kind == PORT_DIGITAL else 2, 32)
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
            row.endpoints.setReadOnly(not editable)
            row.width.setReadOnly(not editable)
            row.width.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.UpDownArrows
                if editable
                else QtWidgets.QAbstractSpinBox.NoButtons
            )
            row.clock_endpoint.setReadOnly(not editable)
            row.remove_button.setVisible(editable)
        for button in (
            self.add_digital_button,
            self.add_dac_button,
            self.apply_button,
        ):
            button.setVisible(editable)
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
        endpoints = list(self._endpoint_values(field.text()))
        if len(endpoints) < width:
            endpoints.extend(
                f"endpoint:{key}[{bit}]" for bit in range(len(endpoints), width)
            )
        elif len(endpoints) > width:
            endpoints = endpoints[:width]
        field.setText(", ".join(endpoints))

    def _queue_reveal_row(self, key: str) -> None:
        """Reveal a newly rebuilt row after Qt publishes its final scroll range."""

        def reveal() -> None:
            row = next((item for item in self._rows if item.key == key), None)
            if row is not None:
                self.cards_scroll.ensureWidgetVisible(row.signal)

        # ``_populate`` replaces every grid item with deleteLater-owned widgets.
        # One owner turn polishes the new Fluent controls; the following layout
        # request publishes the final content height to QScrollArea.  Scrolling
        # in the first turn uses the old maximum and leaves the new bottom row
        # below the viewport, so reveal only after both ordered Qt phases.
        QtCore.QTimer.singleShot(
            0,
            lambda: QtCore.QTimer.singleShot(0, reveal),
        )

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
        self._populate((*current[:split], draft, *current[split:]))
        self._set_editable(True)
        self._queue_reveal_row(key)

    def _add_dac(self) -> None:
        try:
            current = self._current_drafts()
        except ValueError as error:
            self.feedbackRequested.emit(str(error))
            return
        key = self._allocate_key("dac")
        clock_key = self._allocate_key(f"{key}_clock")
        width = 10
        draft = PulseTargetPortDraft(
            key,
            PORT_DAC,
            key,
            tuple(f"endpoint:{key}[{bit}]" for bit in range(width)),
            clock_key,
            f"endpoint:{clock_key}",
        )
        self._populate((*current, draft))
        self._set_editable(True)
        self._queue_reveal_row(key)

    def _remove_key(self, key: str) -> None:
        try:
            current = self._current_drafts()
        except ValueError as error:
            self.feedbackRequested.emit(str(error))
            return
        drafts = tuple(row for row in current if row.key != key)
        self._populate(drafts)
        self._set_editable(True)

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
