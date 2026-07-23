"""Reusable Qt leaf for authoring one explicit named-axis Fit request.

This widget owns only editable presentation state.  It never sees an artifact
reference, repository, executor, ``FitExecution``, or save capability.  A host
installs already-bound model options and receives an exact ``FitSpec`` when the
user presses Fit; that button is the sole promotion boundary from the visible
editor draft to an authority request.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_data import FitSpec

from ..fit_editor import (
    FitAuthoringOption,
    fit_spec_from_form,
)
from .fluent import (
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentScrollArea,
    FluentSettingRow,
)
from .form import FluentParameterForm
from .style import GREEN, GREY, ORANGE


class FitAuthoringPane(QtWidgets.QWidget):
    """One shared model/constraint editor for Figure and direct Fit entry.

    ``editorChanged`` is emitted after the pane has advanced its monotonic
    editor revision.  Hosts use that revision to invalidate an old solver
    completion or draft overlay.  The pane deliberately does not trigger a
    solve when a field changes: a subsequent ordinary Fit click submits the
    newly visible request without an additional confirmation dialog.
    """

    fitRequested = QtCore.pyqtSignal(int, object)
    fitRequestRejected = QtCore.pyqtSignal(str)
    cancelRequested = QtCore.pyqtSignal()
    saveRequested = QtCore.pyqtSignal()
    clearRequested = QtCore.pyqtSignal()
    clearSelectionRequested = QtCore.pyqtSignal()
    editorChanged = QtCore.pyqtSignal(int)
    optionsReleased = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fit_options: dict[str, FitAuthoringOption] = {}
        self._constraint_forms: dict[str, FluentParameterForm] = {}
        self._constraint_stack: QtWidgets.QStackedWidget | None = None
        self._constraint_form: FluentParameterForm | None = None
        self._editor_revision = 0
        self._installing = False
        self._selection_present = False
        self._busy_kind: str | None = None
        self._draft_ready = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.model_combo = FluentComboBox(self)
        self.model_combo.setObjectName("fitAuthoringModel")
        layout.addWidget(FluentSettingRow("model", self.model_combo))

        self.axis_summary = FluentLabel("", self)
        self.axis_summary.setObjectName("fitAuthoringAxes")
        self.axis_summary.setWordWrap(True)
        layout.addWidget(self.axis_summary)

        self.authority_summary = FluentLabel("", self)
        self.authority_summary.setObjectName("fitAuthoringAuthority")
        self.authority_summary.setWordWrap(True)
        layout.addWidget(self.authority_summary)

        self.constraint_scroll = FluentScrollArea(self)
        self.constraint_scroll.setObjectName("fitAuthoringConstraintScroll")
        self.constraint_scroll.setWidgetResizable(True)
        self.constraint_scroll.setMaximumHeight(260)
        layout.addWidget(self.constraint_scroll)

        self.fit_button = FluentButton("Fit", self, color=GREEN)
        self.fit_button.setObjectName("fitAuthoringFitButton")
        self.cancel_button = FluentButton("Cancel fit", self, color=ORANGE)
        self.cancel_button.setObjectName("fitAuthoringCancelButton")
        self.save_button = FluentButton("Save Fit Result", self, color=GREEN)
        self.save_button.setObjectName("fitAuthoringSaveButton")
        self.clear_button = FluentButton("Clear", self, color=GREY)
        self.clear_button.setObjectName("fitAuthoringClearButton")
        self.full_range_button = FluentButton("Use full range", self, color=GREY)
        self.full_range_button.setObjectName("fitAuthoringFullRangeButton")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.fit_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.full_range_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.fit_button.clicked.connect(self._request_fit)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.save_button.clicked.connect(self.saveRequested)
        self.clear_button.clicked.connect(self.clearRequested)
        self.full_range_button.clicked.connect(self.clearSelectionRequested)
        self.set_busy("prepare", draft_ready=False)

    @property
    def editor_revision(self) -> int:
        return self._editor_revision

    @property
    def fit_models(self) -> tuple[str, ...]:
        return tuple(self._fit_options)

    @property
    def constraint_form(self) -> FluentParameterForm | None:
        return self._constraint_form

    def current_option(self) -> FitAuthoringOption:
        model = self.model_combo.currentData()
        try:
            return self._fit_options[model]
        except KeyError as exc:
            raise RuntimeError("fit model control has no authoring option") from exc

    def current_spec(self) -> FitSpec:
        form = self._constraint_form
        if form is None:
            raise RuntimeError("fit constraint form is not ready")
        return fit_spec_from_form(self.current_option(), form.read_all())

    def install_options(
        self,
        options: tuple[FitAuthoringOption, ...],
        *,
        selected_model: str | None = None,
    ) -> None:
        prepared = tuple(options)
        if not prepared or any(
            not isinstance(option, FitAuthoringOption) for option in prepared
        ):
            raise ValueError("fit authoring requires FitAuthoringOption values")
        schema_fingerprints = {
            option.spec.input_schema_fingerprint for option in prepared
        }
        models = tuple(option.spec.model_id for option in prepared)
        if len(schema_fingerprints) != 1 or len(models) != len(set(models)):
            raise ValueError(
                "fit authoring options require one source schema and unique models"
            )
        if selected_model is not None and selected_model not in models:
            raise ValueError("selected_model is not present in fit options")
        if self._constraint_stack is not None or self._constraint_forms:
            raise RuntimeError(
                "old Fit option widgets must be released before replacement"
            )

        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            self._fit_options = {
                option.spec.model_id: option for option in prepared
            }
            stack = QtWidgets.QStackedWidget(self)
            stack.setObjectName("fitAuthoringConstraintStack")
            forms = {}
            for option in prepared:
                form = FluentParameterForm(option.constraint_form, parent=stack)
                form.setObjectName(
                    f"fitAuthoringConstraints_{option.spec.model_id}"
                )
                form.changed.connect(self._editor_changed)
                stack.addWidget(form)
                forms[option.spec.model_id] = form
            self.constraint_scroll.setWidget(stack)
            self._constraint_stack = stack
            self._constraint_forms = forms
            self.model_combo.clear()
            for option in prepared:
                self.model_combo.addItem(
                    option.display_name,
                    option.spec.model_id,
                )
            index = (
                0
                if selected_model is None
                else self.model_combo.findData(selected_model)
            )
            self.model_combo.setCurrentIndex(index)
            self._rebuild_constraint_form()
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False

    def reconcile_options(
        self,
        options: tuple[FitAuthoringOption, ...],
        *,
        selected_model: str | None = None,
    ) -> None:
        """Keyed-diff fit models/forms into this stable pane.

        Live Figure hosts may retain the same widget while a source schema or
        authoritative Area selection changes.  Rebuilding the whole pane would
        disturb scroll/focus state and make Setting/Edit drift.  Models are
        therefore added/removed by id and each surviving constraint form uses
        :meth:`FluentParameterForm.reconcile` to keep compatible leaf widgets.
        """

        prepared = tuple(options)
        if not prepared or any(
            not isinstance(option, FitAuthoringOption) for option in prepared
        ):
            raise ValueError("fit authoring requires FitAuthoringOption values")
        schema_fingerprints = {
            option.spec.input_schema_fingerprint for option in prepared
        }
        models = tuple(option.spec.model_id for option in prepared)
        if len(schema_fingerprints) != 1 or len(models) != len(set(models)):
            raise ValueError(
                "fit authoring options require one source schema and unique models"
            )
        if selected_model is not None and selected_model not in models:
            raise ValueError("selected_model is not present in fit options")
        if self._constraint_stack is None:
            self.install_options(prepared, selected_model=selected_model)
            return

        current_model = self.model_combo.currentData()
        retained_values = {}
        for model_id, form in self._constraint_forms.items():
            try:
                retained_values[model_id] = form.read_all()
            except (TypeError, ValueError):
                # An incomplete local draft must not poison a source/schema
                # transition.  Its compatible widgets remain, but the new
                # owner-authored defaults are authoritative for population.
                pass

        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            stack = self._constraint_stack
            wanted = set(models)
            for model_id in tuple(self._constraint_forms):
                if model_id in wanted:
                    continue
                form = self._constraint_forms.pop(model_id)
                stack.removeWidget(form)
                form.deleteLater()

            forms: dict[str, FluentParameterForm] = {}
            for index, option in enumerate(prepared):
                model_id = option.spec.model_id
                defaults = {
                    field.key: field.default
                    for field in option.constraint_form.fields
                }
                form = self._constraint_forms.get(model_id)
                if form is None:
                    form = FluentParameterForm(
                        option.constraint_form,
                        values=defaults,
                        parent=stack,
                    )
                    form.setObjectName(
                        f"fitAuthoringConstraints_{model_id}"
                    )
                    form.changed.connect(self._editor_changed)
                else:
                    values = retained_values.get(model_id, defaults)
                    if set(values) != set(option.constraint_form.keys):
                        values = defaults
                    form.reconcile(option.constraint_form, values)
                    stack.removeWidget(form)
                stack.insertWidget(index, form)
                forms[model_id] = form

            self._constraint_forms = forms
            self._fit_options = {
                option.spec.model_id: option for option in prepared
            }
            self.model_combo.clear()
            for option in prepared:
                self.model_combo.addItem(
                    option.display_name,
                    option.spec.model_id,
                )
            wanted_model = (
                selected_model
                if selected_model is not None
                else current_model
                if current_model in self._fit_options
                else models[0]
            )
            self.model_combo.setCurrentIndex(
                self.model_combo.findData(wanted_model)
            )
            self._rebuild_constraint_form()
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False
        self.set_busy(None, draft_ready=self._draft_ready)

    def clear_options(self) -> bool:
        """Detach all option widgets and report whether DeferredDelete is pending."""

        self._installing = True
        self.model_combo.blockSignals(True)
        try:
            previous = self.constraint_scroll.takeWidget()
            if previous is not None:
                previous.destroyed.connect(self.optionsReleased.emit)
                previous.deleteLater()
            self._constraint_stack = None
            self._constraint_forms = {}
            self._constraint_form = None
            self._fit_options = {}
            self.model_combo.clear()
            self.axis_summary.setText("")
            self.authority_summary.setText("")
        finally:
            self.model_combo.blockSignals(False)
            self._installing = False
        self.set_busy(None, draft_ready=False)
        return previous is not None

    def set_busy(self, kind: str | None, *, draft_ready: bool) -> None:
        if kind not in (None, "prepare", "fit", "save", "render"):
            raise ValueError("unknown fit authoring busy kind")
        self._busy_kind = kind
        self._draft_ready = bool(draft_ready)
        busy = kind is not None
        editor_ready = bool(self._fit_options) and self._constraint_form is not None
        self.model_combo.setEnabled(not busy and editor_ready)
        if self._constraint_form is not None:
            self._constraint_form.setEnabled(not busy)
        self.fit_button.setEnabled(not busy and editor_ready)
        self.cancel_button.setEnabled(kind == "fit")
        self.save_button.setEnabled(not busy and bool(draft_ready))
        self.clear_button.setEnabled(not busy and editor_ready)
        self.full_range_button.setEnabled(
            not busy and editor_ready and self._selection_present
        )

    def set_selection_present(self, present: bool) -> None:
        self._selection_present = bool(present)
        self.set_busy(self._busy_kind, draft_ready=self._draft_ready)

    def _rebuild_constraint_form(self) -> None:
        option = self.current_option()
        stack = self._constraint_stack
        try:
            form = self._constraint_forms[option.spec.model_id]
        except KeyError as exc:
            raise RuntimeError("Fit constraint stack lacks the selected model") from exc
        if stack is None:
            raise RuntimeError("Fit constraint stack is not installed")
        stack.setCurrentWidget(form)
        self._constraint_form = form
        self.axis_summary.setText(option.axis_summary)
        self.authority_summary.setText(option.authority_summary)

    def _model_changed(self, _index: int) -> None:
        if self._installing or not self._fit_options:
            return
        self._rebuild_constraint_form()
        self._editor_changed()

    def _editor_changed(self, *_args) -> None:
        if self._installing:
            return
        self._editor_revision += 1
        self.editorChanged.emit(self._editor_revision)

    def _request_fit(self) -> None:
        try:
            spec = self.current_spec()
        except (TypeError, ValueError, RuntimeError) as error:
            self.fitRequestRejected.emit(str(error) or type(error).__name__)
            return
        self.fitRequested.emit(self._editor_revision, spec)


__all__ = ["FitAuthoringPane"]
