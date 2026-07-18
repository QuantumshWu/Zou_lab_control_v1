"""Typed Qt authoring for one committed-capture fit.

The window owns presentation draft state only.  ``zlc_data`` remains the sole
fit/model/constraint owner, the notebook composition closure resolves the
committed source, and a fit-specific capacity-one lane performs every solve and
render away from the Qt owner.  A draft result becomes a durable authority only
after the user presses Save; the saved overlay is then rebuilt from its
``CaptureFitResultArtifactRef`` rather than from the process-local execution.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import math
import threading
import time

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitSpec,
)
from zlc_frontend import (
    DataFigure,
    fit_axis_summary,
    fit_constraint_form,
    fit_spec_from_form,
)
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentParameterForm,
    FluentSettingRow,
    GREEN,
    GREY,
    ORANGE,
)
from zlc_neutral_atom.capture_fit_reference import CaptureFitResultArtifactRef
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_storage import positive_integer, positive_real
from zlc_workbench.fit import CaptureFitDraftAuthority, FitDraftResult

from ._figure import _render_figure
from ._frozen_raster import (
    FrozenRasterWindow,
    _error_summary,
    _load_bundle,
    _open_frozen_window,
)


_DEFAULT_FIT_GUI_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_DEFAULT_FIT_TIMEOUT_SECONDS = 30.0
_FIT_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-capture-fit",
)


def _fit_summary(draft: FitDraftResult) -> str:
    result = draft.result
    counts = Counter(status.value for status in result.statuses)
    status_text = ", ".join(
        f"{name.lower()}={count}" for name, count in sorted(counts.items())
    )
    quality = tuple(
        float(value)
        for status, value in zip(result.statuses, result.rmse, strict=True)
        if status is FitBatchStatus.CONVERGED and math.isfinite(float(value))
    )
    quality_text = (
        "no converged RMSE"
        if not quality
        else f"RMSE {min(quality):.4g}–{max(quality):.4g}"
    )
    return (
        f"{result.spec.model_id} · {len(result.statuses)} named batch cell(s) · "
        f"{status_text} · {quality_text} · draft is not saved"
    )


def _render_source(
    figure_factory,
    source,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    return _load_bundle(
        lambda token: _render_figure(
            lambda: figure_factory(
                source,
                memory_limit_bytes=memory_limit_bytes,
            ),
            memory_limit_bytes,
            token,
        ),
        memory_limit_bytes,
        cancelled,
    )


def _validate_fit_options(
    options: object,
    selected_model: str | None,
) -> tuple[BoundFit, ...]:
    prepared = tuple(options) if not isinstance(options, tuple) else options
    if not prepared or any(not isinstance(option, BoundFit) for option in prepared):
        raise ValueError("fit Workbench requires BoundFit options")
    schemas = {option.expected_schema.fingerprint for option in prepared}
    models = tuple(option.spec.model_id for option in prepared)
    if len(schemas) != 1 or len(models) != len(set(models)):
        raise ValueError("fit options require one source schema and unique models")
    if any(option.spec.committed_transform is not None for option in prepared):
        raise ValueError("fit Workbench only accepts raw untransformed options")
    if selected_model is not None and selected_model not in models:
        raise ValueError("selected_model is not present in fit options")
    return prepared


def _prepare_and_render(
    fit_preparer,
    figure_factory,
    source: CaptureArtifactRef,
    selected_model: str | None,
    memory_limit_bytes: int,
    cancelled: threading.Event,
):
    if cancelled.is_set():
        raise CancelledError()
    options = _validate_fit_options(fit_preparer(), selected_model)
    if cancelled.is_set():
        raise CancelledError()
    try:
        bundle = _render_source(
            figure_factory,
            source,
            memory_limit_bytes,
            cancelled,
        )
    except BaseException as error:
        if cancelled.is_set() or isinstance(error, CancelledError):
            raise CancelledError() from error
        return options, None, _error_summary(error)
    return options, bundle, None


def _fit_and_render(
    authority: CaptureFitDraftAuthority,
    draft_figure_factory,
    spec: FitSpec,
    memory_limit_bytes: int,
    deadline_monotonic: float,
    window_cancelled: threading.Event,
    analysis_cancelled: threading.Event,
):
    def cancelled() -> bool:
        return window_cancelled.is_set() or analysis_cancelled.is_set()

    if cancelled():
        raise CancelledError()
    if time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired while waiting for its worker lane")
    draft = authority.execute(spec, cancelled, deadline_monotonic)
    if cancelled():
        authority.discard(draft)
        raise CancelledError()
    try:
        bundle = _render_source(
            draft_figure_factory,
            draft.result,
            memory_limit_bytes,
            window_cancelled,
        )
    except BaseException as error:
        if cancelled() or isinstance(error, CancelledError):
            authority.discard(draft)
            raise CancelledError() from error
        return draft, None, _error_summary(error)
    if cancelled():
        authority.discard(draft)
        raise CancelledError()
    return draft, bundle, None


def _save_and_render(
    figure_factory,
    authority: CaptureFitDraftAuthority,
    draft: FitDraftResult,
    memory_limit_bytes: int,
    cancelled: threading.Event,
):
    if cancelled.is_set():
        raise CancelledError()
    reference = authority.save(draft)
    try:
        bundle = _render_source(
            figure_factory,
            reference,
            memory_limit_bytes,
            cancelled,
        )
    except BaseException as error:
        if cancelled.is_set() or isinstance(error, CancelledError):
            raise CancelledError() from error
        return reference, None, _error_summary(error)
    return reference, bundle, None


class CaptureFitWorkbenchWindow(FrozenRasterWindow):
    """One typed fit draft and explicit durable-save workflow."""

    def __init__(
        self,
        figure_factory,
        draft_figure_factory,
        fit_preparer,
        fit_executor,
        fit_saver,
        source: CaptureArtifactRef,
        *,
        selected_model: str | None,
        memory_limit_bytes: int,
        timeout_seconds: float,
    ) -> None:
        if not all(
            callable(item)
            for item in (
                figure_factory,
                draft_figure_factory,
                fit_preparer,
                fit_executor,
                fit_saver,
            )
        ):
            raise TypeError("fit Workbench factories must be callable")
        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if selected_model is not None and (
            not isinstance(selected_model, str) or not selected_model.strip()
        ):
            raise ValueError("selected_model must be non-empty text or None")

        self._figure_factory = figure_factory
        self._draft_figure_factory = draft_figure_factory
        self._fit_preparer = fit_preparer
        self._draft_authority = CaptureFitDraftAuthority(fit_executor, fit_saver)
        self._source = source
        self._selected_model = selected_model
        self._fit_options: dict[str, BoundFit] = {}
        self._fit_timeout_seconds = positive_real(
            timeout_seconds,
            "timeout_seconds",
        )
        self._job_kind: str | None = None
        self._job_revision: int | None = None
        self._editor_revision = 0
        self._close_deferred_during_save = False
        self._analysis_cancelled: threading.Event | None = None
        self._draft_result: FitDraftResult | None = None
        self._save_inflight: FitDraftResult | None = None
        self._saved_reference: CaptureFitResultArtifactRef | None = None
        self._constraint_form: FluentParameterForm | None = None

        super().__init__(
            None,
            window_title="Capture Fit",
            mode_text="COMMITTED CAPTURE FIT · DRAFT OVERLAY · EXPLICIT SAVE",
            loading_summary="Resolving immutable source…",
            object_prefix="captureFit",
            subject="fit",
            memory_limit_bytes=memory_limit_bytes,
            executor=_FIT_WORK_EXECUTOR,
        )

        editor = QtWidgets.QWidget(self)
        editor.setObjectName("captureFitEditor")
        editor_layout = QtWidgets.QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        self._model_combo = FluentComboBox(editor)
        self._model_combo.setObjectName("captureFitModel")
        editor_layout.addWidget(FluentSettingRow("model", self._model_combo))

        self._axis_summary = FluentLabel("", editor)
        self._axis_summary.setObjectName("captureFitAxes")
        self._axis_summary.setWordWrap(True)
        editor_layout.addWidget(self._axis_summary)
        self._constraint_scroll = QtWidgets.QScrollArea(editor)
        self._constraint_scroll.setObjectName("captureFitConstraintScroll")
        self._constraint_scroll.setWidgetResizable(True)
        self._constraint_scroll.setMaximumHeight(260)
        editor_layout.addWidget(self._constraint_scroll)

        self._fit_button = FluentButton("Fit", editor, color=GREEN)
        self._fit_button.setObjectName("captureFitButton")
        self._cancel_fit_button = FluentButton("Cancel fit", editor, color=ORANGE)
        self._cancel_fit_button.setObjectName("cancelCaptureFitButton")
        self._save_button = FluentButton("Save Fit Result", editor, color=GREEN)
        self._save_button.setObjectName("saveCaptureFitButton")
        self._clear_button = FluentButton("Clear", editor, color=GREY)
        self._clear_button.setObjectName("clearCaptureFitButton")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._fit_button)
        buttons.addWidget(self._cancel_fit_button)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._clear_button)
        buttons.addStretch(1)
        editor_layout.addLayout(buttons)
        self._layout.insertWidget(3, editor)

        self._model_combo.currentIndexChanged.connect(self._model_changed)
        self._fit_button.clicked.connect(self._start_fit)
        self._cancel_fit_button.clicked.connect(self._cancel_fit)
        self._save_button.clicked.connect(self._save_fit)
        self._clear_button.clicked.connect(self._clear_fit)
        self._set_busy("prepare")
        self._job_kind = "prepare"
        if not self._submit_future(
            _prepare_and_render,
            self._fit_preparer,
            self._figure_factory,
            self._source,
            self._selected_model,
            self._memory_limit_bytes,
            self._cancelled,
        ):
            self._job_kind = None
            self._set_busy(None)

    @property
    def draft_ready(self) -> bool:
        return self._draft_result is not None

    @property
    def saved_reference(self) -> CaptureFitResultArtifactRef | None:
        return self._saved_reference

    @property
    def fit_models(self) -> tuple[str, ...]:
        return tuple(self._fit_options)

    def _install_options(self, options: tuple[BoundFit, ...]) -> None:
        prepared = _validate_fit_options(options, self._selected_model)
        self._fit_options = {
            option.spec.model_id: option for option in prepared
        }
        self._model_combo.blockSignals(True)
        try:
            self._model_combo.clear()
            for option in prepared:
                self._model_combo.addItem(
                    option.model.display_name,
                    option.spec.model_id,
                )
            index = (
                0
                if self._selected_model is None
                else self._model_combo.findData(self._selected_model)
            )
            self._model_combo.setCurrentIndex(index)
        finally:
            self._model_combo.blockSignals(False)
        self._rebuild_constraint_form()

    def _current_bound(self) -> BoundFit:
        model = self._model_combo.currentData()
        try:
            return self._fit_options[model]
        except KeyError as exc:
            raise RuntimeError("fit model control has no BoundFit option") from exc

    def _rebuild_constraint_form(self) -> None:
        previous = self._constraint_scroll.takeWidget()
        if previous is not None:
            previous.deleteLater()
        bound = self._current_bound()
        form = FluentParameterForm(fit_constraint_form(bound), parent=self)
        form.setObjectName("captureFitConstraints")
        form.changed.connect(self._editor_changed)
        self._constraint_scroll.setWidget(form)
        self._constraint_form = form
        self._axis_summary.setText(fit_axis_summary(bound))

    def _model_changed(self, _index: int) -> None:
        self._rebuild_constraint_form()
        self._editor_changed()

    def _editor_changed(self, *_args) -> None:
        self._editor_revision += 1
        if self._draft_result is None and self._saved_reference is None:
            return
        draft, self._draft_result = self._draft_result, None
        if draft is not None:
            self._draft_authority.discard(draft)
        previous = self._saved_reference
        self._save_button.setEnabled(False)
        self._status.setText("FIT DRAFT CHANGED")
        self._summary.setText(
            "Visible overlay belongs to the previous specification; click Fit "
            "or Clear."
            + (
                " Saved artifact remains "
                f"{previous.repository_id}:{previous.manifest_digest}."
                if previous is not None
                else ""
            )
        )

    def _set_busy(self, kind: str | None) -> None:
        busy = kind is not None
        editor_ready = bool(self._fit_options) and self._constraint_form is not None
        self._model_combo.setEnabled(not busy and editor_ready)
        if self._constraint_form is not None:
            self._constraint_form.setEnabled(not busy)
        self._fit_button.setEnabled(not busy and editor_ready)
        self._cancel_fit_button.setEnabled(kind == "fit")
        self._save_button.setEnabled(
            not busy and self._draft_result is not None
        )
        self._clear_button.setEnabled(not busy and editor_ready)

    def _start_fit(self) -> None:
        if self._future is not None or self._closing:
            return
        form = self._constraint_form
        if form is None:
            return
        try:
            spec = fit_spec_from_form(self._current_bound(), form.read_all())
        except (TypeError, ValueError) as error:
            self._status.setText("FIT REQUEST INVALID")
            self._diagnostic.setText(_error_summary(error))
            return
        previous_draft, self._draft_result = self._draft_result, None
        if previous_draft is not None:
            self._draft_authority.discard(previous_draft)
        self._saved_reference = None
        self._analysis_cancelled = threading.Event()
        self._clear_bundle()
        self._status.setText("FITTING")
        self._summary.setText(fit_axis_summary(self._current_bound()))
        self._diagnostic.setText("")
        self._job_kind = "fit"
        self._job_revision = self._editor_revision
        deadline_monotonic = time.monotonic() + self._fit_timeout_seconds
        self._set_busy("fit")
        if not self._submit_future(
            _fit_and_render,
            self._draft_authority,
            self._draft_figure_factory,
            spec,
            self._memory_limit_bytes,
            deadline_monotonic,
            self._cancelled,
            self._analysis_cancelled,
        ):
            self._job_kind = None
            self._job_revision = None
            self._analysis_cancelled = None
            self._set_busy(None)

    def _cancel_fit(self) -> None:
        cancelled = self._analysis_cancelled
        if self._job_kind != "fit" or cancelled is None:
            return
        cancelled.set()
        self._cancel_fit_button.setEnabled(False)
        self._status.setText("CANCELLING FIT")

    def _save_fit(self) -> None:
        draft = self._draft_result
        if self._future is not None or self._closing or draft is None:
            return
        self._clear_bundle()
        self._status.setText("SAVING FIT")
        self._summary.setText("Publishing immutable fit artifact…")
        self._diagnostic.setText("")
        self._job_kind = "save"
        self._job_revision = self._editor_revision
        self._save_inflight = draft
        self._set_busy("save")
        if not self._submit_future(
            _save_and_render,
            self._figure_factory,
            self._draft_authority,
            draft,
            self._memory_limit_bytes,
            self._cancelled,
        ):
            self._job_kind = None
            self._job_revision = None
            self._save_inflight = None
            self._set_busy(None)

    def _clear_fit(self) -> None:
        if self._future is not None or self._closing:
            return
        draft, self._draft_result = self._draft_result, None
        if draft is not None:
            self._draft_authority.discard(draft)
        self._saved_reference = None
        self._clear_bundle()
        self._status.setText("CLEARING FIT")
        self._summary.setText("Reloading the committed source without an overlay…")
        self._diagnostic.setText("")
        self._job_kind = "clear"
        self._job_revision = self._editor_revision
        self._set_busy("clear")
        if not self._submit_future(
            _render_source,
            self._figure_factory,
            self._source,
            self._memory_limit_bytes,
            self._cancelled,
        ):
            self._job_kind = None
            self._job_revision = None
            self._set_busy(None)

    def _present_optional_bundle(
        self,
        bundle: EncodedRasterDocument | None,
        render_error: str | None,
    ) -> str | None:
        if bundle is not None and not self._present_bundle(bundle):
            return render_error or self._diagnostic.text() or "Qt presentation failed"
        return render_error

    def shutdown(self) -> None:
        if self._job_kind == "save" and self._future is not None:
            self._close_deferred_during_save = True
            self._status.setText("SAVE IN PROGRESS · CLOSE DEFERRED")
            self._diagnostic.setText(
                "Wait for the immutable artifact reference, then close again."
            )
            return
        self._draft_authority.close()
        super().shutdown()

    def _accept_finished_future(self, future: Future) -> None:
        kind, self._job_kind = self._job_kind, None
        job_revision, self._job_revision = self._job_revision, None
        analysis_cancelled, self._analysis_cancelled = self._analysis_cancelled, None
        save_inflight, self._save_inflight = self._save_inflight, None
        completed_draft: FitDraftResult | None = None
        if kind == "clear":
            super()._accept_finished_future(future)
            if not self._closing and self.raster_ready:
                self._status.setText("SOURCE READY")
                self._summary.setText(
                    "No fit authority is active · "
                    + fit_axis_summary(self._current_bound())
                )
            self._set_busy(None)
            return
        try:
            result = future.result()
            if kind == "prepare":
                options, bundle, render_error = result
                options = _validate_fit_options(options, self._selected_model)
                if bundle is not None and not isinstance(
                    bundle,
                    EncodedRasterDocument,
                ):
                    raise TypeError("prepare worker returned an invalid raster")
                if render_error is not None and not isinstance(render_error, str):
                    raise TypeError("prepare worker returned an invalid diagnostic")
            elif kind == "fit":
                completed_draft, bundle, render_error = result
                if not isinstance(completed_draft, FitDraftResult):
                    raise TypeError("fit worker returned an invalid draft result")
                if bundle is not None and not isinstance(
                    bundle,
                    EncodedRasterDocument,
                ):
                    raise TypeError("fit worker returned an invalid raster")
                if render_error is not None and not isinstance(render_error, str):
                    raise TypeError("fit worker returned an invalid diagnostic")
                if (
                    analysis_cancelled is not None
                    and analysis_cancelled.is_set()
                ):
                    raise CancelledError()
            elif kind == "save":
                reference, bundle, render_error = result
                if not isinstance(reference, CaptureFitResultArtifactRef):
                    raise TypeError("save worker returned an invalid reference")
                if bundle is not None and not isinstance(
                    bundle,
                    EncodedRasterDocument,
                ):
                    raise TypeError("save worker returned an invalid raster")
                if render_error is not None and not isinstance(render_error, str):
                    raise TypeError("save worker returned an invalid diagnostic")
            else:
                raise RuntimeError(f"unknown fit worker completion {kind!r}")
        except (CancelledError, FitCancelled):
            if completed_draft is not None:
                self._draft_authority.discard(completed_draft)
            if not self._closing:
                label = {
                    "prepare": "FIT PREPARATION CANCELLED",
                    "fit": "FIT CANCELLED",
                    "save": "FIT SAVE CANCELLED",
                }.get(kind, "FIT WORK CANCELLED")
                self._status.setText(label)
                self._summary.setText("No new fit result was accepted")
                self._diagnostic.setText("")
        except FitDeadlineExceeded as error:
            if not self._closing:
                self._status.setText("FIT DEADLINE EXCEEDED")
                self._diagnostic.setText(_error_summary(error))
        except BaseException as error:
            if completed_draft is not None:
                self._draft_authority.discard(completed_draft)
            if (
                kind == "save"
                and save_inflight is not None
                and job_revision != self._editor_revision
            ):
                self._draft_authority.discard(save_inflight)
            if not self._closing:
                label = {
                    "prepare": "FIT PREPARATION FAILED",
                    "fit": "FIT FAILED",
                    "save": "SAVE FAILED",
                }.get(kind, "FIT WORK FAILED")
                self._status.setText(label)
                self._diagnostic.setText(_error_summary(error))
        else:
            if not self._closing:
                try:
                    if kind == "prepare":
                        self._install_options(options)
                        render_error = self._present_optional_bundle(
                            bundle,
                            render_error,
                        )
                        self._status.setText(
                            "SOURCE READY"
                            if render_error is None
                            else "SOURCE READY · DISPLAY FAILED"
                        )
                        self._summary.setText(
                            "No fit authority is active · "
                            + fit_axis_summary(self._current_bound())
                        )
                        self._diagnostic.setText(
                            "" if render_error is None else render_error
                        )
                    elif kind == "fit":
                        if job_revision != self._editor_revision:
                            self._draft_result = None
                            self._draft_authority.discard(completed_draft)
                            self._status.setText("STALE FIT DISCARDED")
                            self._summary.setText(
                                "The editor changed after this fit request; click Fit "
                                "again or Clear."
                            )
                            self._diagnostic.setText("")
                        else:
                            self._draft_result = completed_draft
                            render_error = self._present_optional_bundle(
                                bundle,
                                render_error,
                            )
                            self._status.setText(
                                "DRAFT FIT READY"
                                if render_error is None
                                else "DRAFT FIT · DISPLAY FAILED"
                            )
                            self._summary.setText(_fit_summary(completed_draft))
                            self._diagnostic.setText(
                                "" if render_error is None else render_error
                            )
                    elif kind == "save":
                        self._saved_reference = reference
                        self._draft_result = None
                        editor_changed = job_revision != self._editor_revision
                        render_error = self._present_optional_bundle(
                            bundle,
                            render_error,
                        )
                        status = ["FIT SAVED"]
                        if render_error is not None:
                            status.append("DISPLAY FAILED")
                        if editor_changed:
                            status.append("EDITOR CHANGED")
                        if self._close_deferred_during_save:
                            status.append("CLOSE AGAIN")
                        self._status.setText(" · ".join(status))
                        self._summary.setText(
                            f"{reference.repository_id}:{reference.manifest_digest} · "
                            "overlay reopened from saved artifact"
                            + (
                                " · current editor differs from this saved request"
                                if editor_changed
                                else ""
                            )
                        )
                        self._diagnostic.setText(
                            "" if render_error is None else render_error
                        )
                except BaseException as error:
                    self._status.setText("FIT PRESENTATION FAILED")
                    self._diagnostic.setText(_error_summary(error))
        if kind == "save" and self._close_deferred_during_save:
            self._close_deferred_during_save = False
        self._set_busy(None)


def open_capture_fit_workbench(
    figure_factory,
    draft_figure_factory,
    fit_preparer,
    fit_executor,
    fit_saver,
    source: CaptureArtifactRef,
    *,
    selected_model: str | None = None,
    memory_limit_bytes: int = _DEFAULT_FIT_GUI_MEMORY_LIMIT_BYTES,
    timeout_seconds: float = _DEFAULT_FIT_TIMEOUT_SECONDS,
) -> CaptureFitWorkbenchWindow:
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    timeout = positive_real(timeout_seconds, "timeout_seconds")
    return _open_frozen_window(
        lambda: CaptureFitWorkbenchWindow(
            figure_factory,
            draft_figure_factory,
            fit_preparer,
            fit_executor,
            fit_saver,
            source,
            selected_model=selected_model,
            memory_limit_bytes=limit,
            timeout_seconds=timeout,
        )
    )


__all__ = [
    "CaptureFitWorkbenchWindow",
    "open_capture_fit_workbench",
]
