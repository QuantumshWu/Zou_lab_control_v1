"""Read-only Workbench for one committed calibration artifact/report pair."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    FluentParameterForm,
    FluentScrollArea,
    GREEN,
    GREY,
    ORANGE,
)
from zlc_neutral_atom.readout.calibration_application import (
    CalibrationArtifactRequest,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.runtime.run import RunCancelled, RunHandle
from zlc_workbench.calibration import (
    calibration_analysis_form,
    calibration_analysis_from_form,
    calibration_authority_summary,
)
from zlc_workbench.run_owner import QtRunOwnerMailbox

from zlc_workbench.frozen_raster import FrozenRasterWindow
from zlc_workbench.window_runtime import error_summary

from .jobs import (
    _prepare_calibration_editor,
    _render_calibration,
)


class CalibrationWorkbenchWindow(FrozenRasterWindow):
    """Edit one frozen calibration request and run an atomic FINAL commit."""

    def __init__(
        self,
        computation_loader,
        run_starter,
        *,
        request: CalibrationArtifactRequest | None,
        reference: CalibrationArtifactRef | None,
    ) -> None:
        if not callable(computation_loader) or not callable(run_starter):
            raise TypeError("calibration Workbench callables must be callable")
        if (request is None) == (reference is None):
            raise ValueError("provide exactly one calibration request or reference")
        if request is not None and not isinstance(
            request,
            CalibrationArtifactRequest,
        ):
            raise TypeError("request must be CalibrationArtifactRequest or None")
        if reference is not None and not isinstance(
            reference,
            CalibrationArtifactRef,
        ):
            raise TypeError("reference must be CalibrationArtifactRef or None")

        self._computation_loader = computation_loader
        self._run_starter = run_starter
        self._request: CalibrationArtifactRequest | None = None
        self._opened_reference = reference
        self._form: FluentParameterForm | None = None
        self._editor_revision = 0
        self._run_revision: int | None = None
        self._run_active = False
        self._cancel_requested = False
        self._close_requested = False
        self._receipt_close_again = False
        self._run_warnings: tuple[str, ...] = ()
        self._saved_reference: CalibrationArtifactRef | None = None
        self._raster_job_kind: str | None = None
        self._mailbox_closed = False

        super().__init__(
            None,
            window_title="Readout Calibration",
            mode_text="FORMAL CALIBRATION · ATOMIC FINAL COMMIT",
            loading_summary="Resolving immutable calibration authority…",
            object_prefix="calibrationEditor",
            subject="calibration",
        )
        self._run_owner = QtRunOwnerMailbox(
            self._wake.request_owner_wake,
            thread_name_prefix="zlc-calibration-workbench",
            max_workers=2,
        )

        editor = QtWidgets.QWidget(self)
        editor.setObjectName("calibrationEditor")
        editor_layout = QtWidgets.QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        self._authority = FluentLabel("", editor)
        self._authority.setObjectName("calibrationAuthority")
        self._authority.setWordWrap(True)
        self._authority.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        editor_layout.addWidget(self._authority)
        self._form_scroll = FluentScrollArea(editor)
        self._form_scroll.setObjectName("calibrationFormScroll")
        self._form_scroll.setWidgetResizable(True)
        self._form_scroll.setMaximumHeight(360)
        editor_layout.addWidget(self._form_scroll)
        self._calibrate_button = FluentButton(
            "Calibrate & Save",
            editor,
            color=GREEN,
        )
        self._calibrate_button.setObjectName("calibrateAndSaveButton")
        self._stop_button = FluentButton("Stop", editor, color=ORANGE)
        self._stop_button.setObjectName("stopCalibrationButton")
        self._reset_button = FluentButton("Reset request", editor, color=GREY)
        self._reset_button.setObjectName("resetCalibrationButton")
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._calibrate_button)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._reset_button)
        controls.addStretch(1)
        editor_layout.addLayout(controls)
        self._layout.insertWidget(3, editor)

        self._calibrate_button.clicked.connect(self._start_calibration)
        self._stop_button.clicked.connect(self._stop_calibration)
        self._reset_button.clicked.connect(self._reset_request)
        if request is not None:
            self._install_request(request, previous_reference=None)
            self._status.setText("CALIBRATION REQUEST READY")
            self._summary.setText(calibration_authority_summary(request))
            self._set_busy(None)
        else:
            assert reference is not None
            self._set_busy("prepare")
            self._raster_job_kind = "prepare"
            if not self._submit_future(
                _prepare_calibration_editor,
                self._computation_loader,
                reference,
                self._cancelled,
            ):
                self._raster_job_kind = None
                self._set_busy(None)

    @property
    def saved_reference(self) -> CalibrationArtifactRef | None:
        return self._saved_reference

    @property
    def editor_ready(self) -> bool:
        return self._request is not None and self._form is not None

    @property
    def worker_idle(self) -> bool:
        return (
            super().worker_idle
            and self._run_owner.worker_idle
            and not self._run_active
        )

    def _set_busy(self, kind: str | None) -> None:
        busy = kind is not None
        ready = self._request is not None and self._form is not None
        if self._form is not None:
            self._form.setEnabled(not busy and not self._close_requested)
        self._calibrate_button.setEnabled(not busy and ready)
        self._stop_button.setEnabled(kind in {"start", "run"})
        self._reset_button.setEnabled(not busy and ready)
        if not self._closing:
            self._close_button.setEnabled(not self._close_requested)

    def _install_request(
        self,
        request: CalibrationArtifactRequest,
        *,
        previous_reference: CalibrationArtifactRef | None,
    ) -> None:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("prepare worker returned an invalid calibration request")
        if previous_reference is not None and not isinstance(
            previous_reference,
            CalibrationArtifactRef,
        ):
            raise TypeError("previous_reference must be CalibrationArtifactRef or None")
        prior = self._form_scroll.takeWidget()
        if prior is not None:
            prior.deleteLater()
        form = FluentParameterForm(
            calibration_analysis_form(request.analysis),
            parent=self,
        )
        form.setObjectName("calibrationParameters")
        form.changed.connect(self._editor_changed)
        self._form_scroll.setWidget(form)
        self._form = form
        self._request = request
        self._saved_reference = previous_reference
        self._authority.setText(
            calibration_authority_summary(request, previous_reference)
        )

    def _editor_changed(self, *_args) -> None:
        self._editor_revision += 1
        if self._run_active:
            return
        if self._saved_reference is None:
            self._status.setText("CALIBRATION REQUEST CHANGED")
            self._summary.setText(
                "No artifact has been committed from the current editor state."
            )
        else:
            self._status.setText("FINAL CALIBRATION · EDITOR CHANGED")
            self._summary.setText(
                f"Visible report belongs to {self._saved_reference.target_ref}; "
                "the edited request will create a new artifact."
            )

    def _start_calibration(self) -> None:
        if not self.worker_idle or self._closing or self._close_requested:
            return
        request = self._request
        form = self._form
        if request is None or form is None:
            return
        try:
            analysis = calibration_analysis_from_form(
                request.analysis,
                form.read_all(),
            )
            frozen_request = CalibrationArtifactRequest(
                request.source_capture_ref,
                request.readout_binding,
                analysis,
            )
        except (TypeError, ValueError) as error:
            self._status.setText("CALIBRATION REQUEST INVALID")
            self._diagnostic.setText(error_summary(error))
            return
        self._run_revision = self._editor_revision
        self._run_active = True
        self._cancel_requested = False
        self._receipt_close_again = False
        self._run_warnings = ()
        generation = self._run_owner.begin_generation()
        self._status.setText("STARTING FORMAL CALIBRATION")
        self._summary.setText(
            "The frozen source and this exact editor revision are entering one "
            "formal Run; success commits immediately."
        )
        self._diagnostic.setText("")
        self._set_busy("start")
        try:
            self._run_owner.submit(
                "start",
                lambda: self._run_starter(frozen_request),
                generation=generation,
            )
        except BaseException as error:
            self._run_owner.mark_owner_reaped()
            self._run_active = False
            self._status.setText("CALIBRATION START FAILED")
            self._diagnostic.setText(error_summary(error))
            self._set_busy(None)

    def _stop_calibration(self) -> None:
        if not self._run_active:
            return
        self._cancel_requested = True
        handle = self._run_owner.handle
        if handle is not None:
            handle.cancel("calibration Workbench requested stop")
        self._status.setText("CANCELLING CALIBRATION")
        self._stop_button.setEnabled(False)

    def _reset_request(self) -> None:
        form = self._form
        if not self.worker_idle or form is None:
            return
        form.populate(form.spec.default_values())
        self._editor_changed()

    def _accept_start_completion(self, generation: int, future: Future) -> None:
        try:
            handle = future.result()
            if not isinstance(handle, RunHandle):
                raise TypeError("calibration starter returned a non-RunHandle")
        except BaseException as error:
            self._run_owner.mark_owner_reaped()
            self._run_active = False
            if self._close_requested:
                self._close_requested = False
                self._set_busy(None)
                super().shutdown()
            else:
                self._status.setText("CALIBRATION START FAILED")
                self._diagnostic.setText(error_summary(error))
                self._set_busy(None)
            return
        self._run_owner.set_handle(handle)
        if self._cancel_requested or self._close_requested:
            handle.cancel("calibration Workbench cancelled before Run attachment")
        if not self._run_owner.begin_terminal_job("terminal", handle.result):
            raise RuntimeError("calibration terminal owner was not admitted")
        self._status.setText(
            "CANCELLING CALIBRATION"
            if self._cancel_requested or self._close_requested
            else "CALIBRATING & COMMITTING"
        )
        self._set_busy("run")

    def _terminal_status(
        self,
        head: str = "CALIBRATION SAVED",
        *,
        display_failed: bool = False,
    ) -> str:
        labels = [head]
        if self._run_warnings:
            labels.append("RUN WARNING")
        if display_failed:
            labels.append("REPORT DISPLAY FAILED")
        if self._run_revision != self._editor_revision:
            labels.append("EDITOR CHANGED")
        if self._receipt_close_again:
            labels.append("CLOSE AGAIN")
        return " · ".join(labels)

    def _record_terminal_warnings(self):
        handle = self._run_owner.handle
        if handle is None:
            raise RuntimeError("calibration terminal completion has no RunHandle")
        snapshot = handle.snapshot()
        warnings = []
        if snapshot.commit_recovery_warning is not None:
            warnings.append(
                f"commit recovery warning: {snapshot.commit_recovery_warning}"
            )
        warnings.extend(f"cleanup warning: {item}" for item in snapshot.cleanup_errors)
        self._run_warnings = tuple(warnings)
        return snapshot

    def _diagnostic_with_run_warnings(self, extra: str | None = None) -> str:
        messages = list(self._run_warnings)
        if extra:
            messages.append(extra)
        return "\n".join(messages)

    def _accept_terminal_completion(self, generation: int, future: Future) -> None:
        try:
            result = future.result()
        except BaseException as error:
            self._run_owner.finish_terminal_job(generation, owner_reaped=True)
            self._run_active = False
            snapshot = self._record_terminal_warnings()
            if snapshot.final_committed:
                self._receipt_close_again = self._close_requested
                self._close_requested = False
                self._status.setText(
                    self._terminal_status("COMMIT RECEIPT UNAVAILABLE")
                )
                self._summary.setText(
                    "The Run reports FINAL commit, but its exact artifact reference "
                    "could not be accepted.  The window will not claim cancellation "
                    "or close automatically."
                )
                self._diagnostic.setText(
                    self._diagnostic_with_run_warnings(error_summary(error))
                )
                self._set_busy(None)
                return
            if self._close_requested:
                self._close_requested = False
                self._set_busy(None)
                super().shutdown()
            elif isinstance(error, RunCancelled):
                self._status.setText("CALIBRATION CANCELLED")
                self._summary.setText("No new calibration artifact was committed.")
                self._diagnostic.setText(error_summary(error))
                self._set_busy(None)
            else:
                self._status.setText("CALIBRATION FAILED")
                self._summary.setText("No new calibration artifact was committed.")
                self._diagnostic.setText(
                    self._diagnostic_with_run_warnings(error_summary(error))
                )
                self._set_busy(None)
            return

        self._run_owner.finish_terminal_job(generation, owner_reaped=True)
        self._run_active = False
        snapshot = self._record_terminal_warnings()
        if not isinstance(result, CalibrationArtifactRef):
            error = TypeError("calibration Run returned an invalid reference")
            if snapshot.final_committed:
                self._receipt_close_again = self._close_requested
                self._close_requested = False
                self._status.setText(self._terminal_status("COMMIT RECEIPT INVALID"))
                self._summary.setText(
                    "The Run reports FINAL commit, but returned a non-calibration "
                    "receipt.  No exact reference is being guessed or hidden."
                )
                self._diagnostic.setText(
                    self._diagnostic_with_run_warnings(error_summary(error))
                )
                self._set_busy(None)
                return
            if self._close_requested:
                self._close_requested = False
                self._set_busy(None)
                super().shutdown()
            else:
                self._status.setText("CALIBRATION FAILED")
                self._summary.setText("No calibration artifact was committed.")
                self._diagnostic.setText(error_summary(error))
                self._set_busy(None)
            return
        if not snapshot.final_committed:
            self._receipt_close_again = self._close_requested
            self._close_requested = False
            self._status.setText(self._terminal_status("COMMIT EVIDENCE MISSING"))
            self._summary.setText(
                "A typed calibration reference was returned without terminal "
                "final-commit evidence.  It is not being admitted as SAVED or FINAL."
            )
            self._diagnostic.setText(
                self._diagnostic_with_run_warnings(
                    "Run protocol violation: final_committed is false"
                )
            )
            self._set_busy(None)
            return
        reference = result
        self._saved_reference = reference
        self._receipt_close_again = self._close_requested
        self._close_requested = False
        self._status.setText(self._terminal_status())
        self._summary.setText(
            f"FINAL {reference.repository_id}:{reference.manifest_digest} · "
            "reopening the paired artifact/report from this exact reference."
        )
        self._diagnostic.setText(self._diagnostic_with_run_warnings())
        self._raster_job_kind = "report"
        self._set_busy("report")
        if not self._submit_future(
            _render_calibration,
            self._computation_loader,
            reference,
            self._cancelled,
        ):
            self._raster_job_kind = None
            self._status.setText(self._terminal_status(display_failed=True))
            self._set_busy(None)

    def _accept_finished_future(self, future: Future) -> None:
        kind, self._raster_job_kind = self._raster_job_kind, None
        if kind == "prepare":
            try:
                request, bundle, render_error = future.result()
                if bundle is not None and not isinstance(
                    bundle,
                    EncodedRasterDocument,
                ):
                    raise TypeError("prepare worker returned an invalid raster")
            except CancelledError:
                if not self._closing:
                    self._status.setText("CALIBRATION PREPARATION CANCELLED")
            except BaseException as error:
                if not self._closing:
                    self._status.setText("CALIBRATION PREPARATION FAILED")
                    self._diagnostic.setText(error_summary(error))
            else:
                if not self._closing:
                    try:
                        self._install_request(
                            request,
                            previous_reference=self._opened_reference,
                        )
                        displayed = bundle is not None and self._present_bundle(bundle)
                        self._status.setText(
                            "CALIBRATION READY"
                            if render_error is None and (bundle is None or displayed)
                            else "CALIBRATION READY · PRIOR REPORT DISPLAY FAILED"
                        )
                        self._summary.setText(
                            calibration_authority_summary(
                                request,
                                self._opened_reference,
                            )
                        )
                        if render_error is not None:
                            self._diagnostic.setText(render_error)
                        elif bundle is None or displayed:
                            self._diagnostic.setText("")
                    except BaseException as error:
                        self._status.setText("CALIBRATION PRESENTATION FAILED")
                        self._diagnostic.setText(error_summary(error))
            self._set_busy(None)
            return
        if kind != "report":
            super()._accept_finished_future(future)
            return
        try:
            bundle = future.result()
            if not isinstance(bundle, EncodedRasterDocument):
                raise TypeError("report worker returned an invalid raster")
        except CancelledError:
            if not self._closing:
                self._status.setText(self._terminal_status(display_failed=True))
                self._diagnostic.setText(
                    self._diagnostic_with_run_warnings(
                        "Report rendering was cancelled after commit."
                    )
                )
        except BaseException as error:
            if not self._closing:
                self._status.setText(self._terminal_status(display_failed=True))
                self._diagnostic.setText(
                    self._diagnostic_with_run_warnings(error_summary(error))
                )
        else:
            if not self._closing:
                displayed = self._present_bundle(bundle)
                self._status.setText(
                    self._terminal_status(display_failed=not displayed)
                )
                reference = self._saved_reference
                if reference is not None:
                    self._summary.setText(
                        f"FINAL {reference.repository_id}:{reference.manifest_digest}\n"
                        f"{bundle.summary}"
                    )
                if displayed:
                    self._diagnostic.setText(
                        self._diagnostic_with_run_warnings()
                    )
        self._set_busy(None)

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        super()._owner_cycle()
        for completion in self._run_owner.drain_completions():
            if completion.generation != self._run_owner.generation:
                continue
            if completion.kind == "start":
                self._accept_start_completion(
                    completion.generation,
                    completion.future,
                )
            elif completion.kind == "terminal":
                self._accept_terminal_completion(
                    completion.generation,
                    completion.future,
                )
            else:
                raise RuntimeError(
                    f"unknown calibration completion {completion.kind!r}"
                )
        self._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        if self._run_active or not self._run_owner.owner_reaped:
            self._close_requested = True
            self._cancel_requested = True
            handle = self._run_owner.handle
            if handle is not None:
                handle.cancel("calibration Workbench requested close")
            self._status.setText("CANCELLING CALIBRATION · CLOSE DEFERRED")
            self._diagnostic.setText(
                "If FINAL commit wins, the exact reference will be shown before "
                "this window may close."
            )
            self._set_busy("run")
            return
        super().shutdown()

    def _finish_close_if_ready(self) -> None:
        if (
            self._closing
            and (
                not self._run_owner.worker_idle
                or not self._run_owner.owner_reaped
            )
        ):
            return
        if self._closing and not self._mailbox_closed:
            self._run_owner.shutdown()
            self._mailbox_closed = True
        super()._finish_close_if_ready()


__all__ = [
    "CalibrationWorkbenchWindow",
    "open_calibration_report_workbench",
    "open_calibration_workbench",
]
