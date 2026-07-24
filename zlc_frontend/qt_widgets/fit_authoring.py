"""Reusable Qt leaf for authoring one explicit named-axis Fit request.

The pane deliberately has one model selector and one arguments line.  Model
metadata and numeric validity remain owned by the typed Fit contract; this
widget owns only the user's reversible text draft.  Pressing Fit is the sole
promotion boundary from that draft to an authority-bearing ``FitSpec``.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_data import FitSpec

from ..fit_editor import (
    FitAuthoringDraft,
    FitAuthoringOption,
    fit_spec_from_arguments,
    reconcile_fit_authoring_draft,
)
from .fluent import (
    FluentButton,
    FluentComboBox,
    FluentLineEdit,
    FluentSettingRow,
)
from .style import GREEN, GREY


class FitAuthoringPane(QtWidgets.QWidget):
    """One shared model/arguments editor for Figure and embedded panels."""

    fitRequested = QtCore.pyqtSignal(int, object)
    fitRequestRejected = QtCore.pyqtSignal(str)
    clearRequested = QtCore.pyqtSignal()
    editorChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fit_options: dict[str, FitAuthoringOption] = {}
        self._argument_drafts: dict[str, str] = {}
        self._active_model_id: str | None = None
        self._editor_revision = 0
        self._installing = False
        self._busy_kind: str | None = None
        self._draft_ready = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.model_combo = FluentComboBox(self)
        self.model_combo.setObjectName("fitAuthoringModel")
        layout.addWidget(FluentSettingRow("model", self.model_combo))

        self.arguments_edit = FluentLineEdit("", self)
        self.arguments_edit.setObjectName("fitAuthoringArguments")
        self.arguments_edit.setPlaceholderText(
            "auto (for example: center=50, sigma_lower=0)"
        )
        layout.addWidget(FluentSettingRow("args", self.arguments_edit))

        self.fit_button = FluentButton("Fit", self, color=GREEN)
        self.fit_button.setObjectName("fitAuthoringFitButton")
        self.clear_button = FluentButton("Clear", self, color=GREY)
        self.clear_button.setObjectName("fitAuthoringClearButton")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.fit_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.arguments_edit.textEdited.connect(self._arguments_changed)
        self.fit_button.clicked.connect(self._request_fit)
        self.clear_button.clicked.connect(self._request_clear)
        self.set_busy("prepare", draft_ready=False)

    @property
    def editor_revision(self) -> int:
        return self._editor_revision

    @property
    def fit_models(self) -> tuple[str, ...]:
        return tuple(self._fit_options)

    @property
    def arguments_text(self) -> str:
        return self.arguments_edit.text()

    @property
    def draft_state(self) -> FitAuthoringDraft | None:
        """Return the complete reversible draft shown by this view."""

        if not self._fit_options:
            return None
        self._store_active_draft()
        selected = self._active_model_id
        if selected not in self._fit_options:
            raise RuntimeError("Fit editor has no selected prepared model")
        return FitAuthoringDraft(
            selected,
            tuple(
                (model_id, self._argument_drafts[model_id])
                for model_id in self._fit_options
            ),
        )

    @property
    def axis_summary_text(self) -> str:
        option = self._optional_current_option()
        return "" if option is None else option.axis_summary

    @property
    def authority_summary_text(self) -> str:
        option = self._optional_current_option()
        return "" if option is None else option.authority_summary

    def current_option(self) -> FitAuthoringOption:
        option = self._optional_current_option()
        if option is None:
            raise RuntimeError("Fit model control has no authoring option")
        return option

    def current_spec(self) -> FitSpec:
        return fit_spec_from_arguments(
            self.current_option(),
            self.arguments_edit.text(),
        )

    def install_options(
        self,
        options: tuple[FitAuthoringOption, ...],
        *,
        selected_model: str | None = None,
    ) -> None:
        prepared, models = self._validated_options(options, selected_model)
        self._replace_options(
            prepared,
            models,
            selected_model=selected_model,
            preserve_drafts=False,
        )

    def reconcile_options(
        self,
        options: tuple[FitAuthoringOption, ...],
        *,
        selected_model: str | None = None,
    ) -> None:
        """Reconcile model metadata while preserving each surviving text draft."""

        prepared, models = self._validated_options(options, selected_model)
        self._store_active_draft()
        self._replace_options(
            prepared,
            models,
            selected_model=selected_model,
            preserve_drafts=True,
        )

    def set_editor_draft(
        self,
        model_id: str,
        arguments: str,
        *,
        notify: bool = False,
    ) -> None:
        """Set one model's visible text draft without creating editor widgets."""

        if model_id not in self._fit_options:
            raise ValueError("model_id is not present in Fit options")
        if not isinstance(arguments, str):
            raise TypeError("Fit arguments must be text")
        self._store_active_draft()
        self._argument_drafts[model_id] = arguments
        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.setCurrentIndex(
                self.model_combo.findData(model_id)
            )
            self._show_model(model_id)
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False
        if notify:
            self._advance_editor_revision()

    def set_draft_state(
        self,
        draft: FitAuthoringDraft,
        *,
        notify: bool = False,
    ) -> None:
        """Present one owner-held draft without creating another authority."""

        if not self._fit_options:
            raise RuntimeError("Fit options must be installed before its draft")
        reconciled = reconcile_fit_authoring_draft(
            tuple(self._fit_options.values()),
            draft,
            selected_model=draft.selected_model_id,
        )
        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            self._argument_drafts = dict(reconciled.arguments_by_model)
            self.model_combo.setCurrentIndex(
                self.model_combo.findData(reconciled.selected_model_id)
            )
            self._show_model(reconciled.selected_model_id)
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False
        if notify:
            self._advance_editor_revision()

    def clear_options(self) -> None:
        """Synchronously clear metadata and drafts; there are no child forms."""

        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            self._fit_options.clear()
            self._argument_drafts.clear()
            self._active_model_id = None
            self.model_combo.clear()
            self.model_combo.setToolTip("")
            self.arguments_edit.clear()
            self.arguments_edit.setToolTip("")
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False
        self.set_busy(None, draft_ready=False)

    def set_busy(self, kind: str | None, *, draft_ready: bool) -> None:
        if kind not in (None, "prepare", "fit", "save", "render"):
            raise ValueError("unknown Fit authoring busy kind")
        self._busy_kind = kind
        self._draft_ready = bool(draft_ready)
        editor_ready = bool(self._fit_options)
        idle = kind is None
        self.model_combo.setEnabled(idle and editor_ready)
        self.arguments_edit.setEnabled(idle and editor_ready)
        self.fit_button.setEnabled(idle and editor_ready)
        self.clear_button.setEnabled(
            kind == "fit" or (idle and (editor_ready or self._draft_ready))
        )

    @staticmethod
    def _validated_options(
        options: tuple[FitAuthoringOption, ...],
        selected_model: str | None,
    ) -> tuple[tuple[FitAuthoringOption, ...], tuple[str, ...]]:
        prepared = tuple(options)
        if not prepared or any(
            not isinstance(option, FitAuthoringOption) for option in prepared
        ):
            raise ValueError("Fit authoring requires FitAuthoringOption values")
        schema_fingerprints = {
            option.spec.input_schema_fingerprint for option in prepared
        }
        models = tuple(option.spec.model_id for option in prepared)
        if len(schema_fingerprints) != 1 or len(models) != len(set(models)):
            raise ValueError(
                "Fit authoring options require one source schema and unique models"
            )
        if selected_model is not None and selected_model not in models:
            raise ValueError("selected_model is not present in Fit options")
        return prepared, models

    def _replace_options(
        self,
        options: tuple[FitAuthoringOption, ...],
        models: tuple[str, ...],
        *,
        selected_model: str | None,
        preserve_drafts: bool,
    ) -> None:
        previous = self.draft_state if preserve_drafts else None
        option_by_model = {option.spec.model_id: option for option in options}
        draft = reconcile_fit_authoring_draft(
            options,
            previous,
            selected_model=selected_model,
        )

        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            self._fit_options = option_by_model
            self._argument_drafts = dict(draft.arguments_by_model)
            self.model_combo.clear()
            for option in options:
                self.model_combo.addItem(
                    option.display_name,
                    option.spec.model_id,
                )
            self.model_combo.setCurrentIndex(
                self.model_combo.findData(draft.selected_model_id)
            )
            self._show_model(draft.selected_model_id)
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False
        self.set_busy(None, draft_ready=self._draft_ready)

    def _optional_current_option(self) -> FitAuthoringOption | None:
        model_id = self.model_combo.currentData()
        return self._fit_options.get(model_id)

    def _store_active_draft(self) -> None:
        model_id = self._active_model_id
        if model_id in self._fit_options:
            self._argument_drafts[model_id] = self.arguments_edit.text()

    def _show_model(self, model_id: str) -> None:
        option = self._fit_options[model_id]
        self._active_model_id = model_id
        self.arguments_edit.setText(self._argument_drafts[model_id])
        parameters = ", ".join(option.parameter_names)
        authority = f"{option.axis_summary}\n{option.authority_summary}"
        self.model_combo.setToolTip(authority)
        self.arguments_edit.setToolTip(
            "Empty uses automatic initialization. "
            "Use name=value to fix a parameter, or "
            "name_initial/name_lower/name_upper for solver constraints.\n"
            f"Parameters: {parameters}\n{authority}"
        )

    def _model_changed(self, _index: int) -> None:
        if self._installing or not self._fit_options:
            return
        model_id = self.model_combo.currentData()
        if model_id not in self._fit_options:
            return
        self._store_active_draft()
        self._show_model(model_id)
        self._advance_editor_revision()

    def _arguments_changed(self, text: str) -> None:
        if self._installing:
            return
        model_id = self._active_model_id
        if model_id in self._fit_options:
            self._argument_drafts[model_id] = text
        self._advance_editor_revision()

    def _advance_editor_revision(self) -> None:
        self._editor_revision += 1
        self.editorChanged.emit(self._editor_revision)

    def _request_clear(self) -> None:
        self.clearRequested.emit()

    def _request_fit(self) -> None:
        try:
            spec = self.current_spec()
        except (TypeError, ValueError, RuntimeError) as error:
            self.fitRequestRejected.emit(str(error) or type(error).__name__)
            return
        self.fitRequested.emit(self._editor_revision, spec)


__all__ = ["FitAuthoringPane"]
