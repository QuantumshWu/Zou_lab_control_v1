"""MOT-field task preparation, lifecycle, live output and persistence.

A form boundary constructs :class:`MotFieldTaskIntent`; the composition root
passes its installation-bound semantic service to :func:`prepare_mot_field_task`.
The returned command is the complete application.  A desktop may attach its
typed live output, start/cancel/observe the command and route its named FINAL
datasets, but never assembles the scan, projection, analysis or materializer.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Mapping, Protocol

import numpy as np

from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.logic_node_declaration import (
    DefaultOutputView,
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)
from zlc_neutral_atom.pulse_catalog import MOT_FIELD_PULSE_PATH
from zlc_neutral_atom.node_input import bind_no_node_inputs
from zlc_neutral_atom.authoring import (
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
    MINIMUM_POSITIVE_FLOAT,
)
from .mot_field import (
    DEFAULT_MOT_FIELD_CAMERA_ROLE,
    DEFAULT_MOT_FIELD_CENTER_CODE,
    DEFAULT_MOT_FIELD_POINTS,
    DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
    DEFAULT_MOT_FIELD_SPAN_CODE,
    MINIMUM_MOT_FIELD_POINTS,
    MOT_FIELD_FINAL_OUTPUT_DECLARATIONS,
    MOT_FIELD_TASK_DEFINITION,
    MotFieldAcquisitionResult,
    MotFieldRequest,
    MotFieldResult,
    analyze_mot_scan,
    mot_field_final_outputs,
)
from .mot_field_live import MOT_FIELD_LIVE_OUTPUT_DECLARATIONS
from .mot_field_task_live import MotFieldTaskLiveOutput
from .application import (
    MotFieldAcquisitionHandle,
    PreparedMotFieldAcquisition,
)
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunCancelled,
    RunFailed,
    RunId,
    RunStartRejected,
    RunSnapshot,
    RunState,
)
from zlc_neutral_atom.runtime.resources import ResourceBusy
from zlc_storage import (
    canonical_text,
    finite_real,
    integer,
    normalized_text,
    positive_real,
)
from zlc_storage.paths import resolve_under_project


DEFAULT_MOT_FIELD_REPORT_FOLDER = "_output/mot_field"
DEFAULT_MOT_FIELD_PULSE_PATH = MOT_FIELD_PULSE_PATH


@dataclass(frozen=True)
class MotFieldTaskIntent:
    """Complete MOT-field application intent before hardware binding.

    ``None`` means frame-centred ROI.  Numeric zero is an ordinary explicit
    pixel coordinate; UI sentinel interpretation must already have disappeared
    before this typed value is constructed.
    """

    pulse: str
    center_x: float
    center_y: float
    center_z: float
    span: float
    points: int
    roi_cx: float | None
    roi_cy: float | None
    roi_radius: float
    folder: str
    camera_role: str

    def __post_init__(self) -> None:
        pulse = normalized_text(self.pulse, "pulse")
        center_x = finite_real(self.center_x, "center_x")
        center_y = finite_real(self.center_y, "center_y")
        center_z = finite_real(self.center_z, "center_z")
        span = finite_real(self.span, "span", minimum=0.0)
        points = integer(
            self.points,
            "points",
            minimum=MINIMUM_MOT_FIELD_POINTS,
        )
        assert points is not None
        roi_cx = (
            None
            if self.roi_cx is None
            else finite_real(self.roi_cx, "roi_cx", minimum=0.0)
        )
        roi_cy = (
            None
            if self.roi_cy is None
            else finite_real(self.roi_cy, "roi_cy", minimum=0.0)
        )
        roi_radius = positive_real(self.roi_radius, "roi_radius")
        folder = normalized_text(self.folder, "folder")
        camera_role = normalized_text(self.camera_role, "camera_role")
        if camera_role != DEFAULT_MOT_FIELD_CAMERA_ROLE:
            raise ValueError(
                "MOT field task requires the "
                f"{DEFAULT_MOT_FIELD_CAMERA_ROLE} role"
            )
        object.__setattr__(self, "pulse", pulse)
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", center_y)
        object.__setattr__(self, "center_z", center_z)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "roi_cx", roi_cx)
        object.__setattr__(self, "roi_cy", roi_cy)
        object.__setattr__(self, "roi_radius", roi_radius)
        object.__setattr__(self, "folder", folder)
        object.__setattr__(self, "camera_role", camera_role)


_MOT_FIELD_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_MOT_FIELD_PULSE_PATH,
            required=True,
            description=(
                "Autonomous SCAN_SLOT template declaring da_x, da_y and da_z"
            ),
        ),
        AuthoringField(
            "center_x",
            "float",
            "Bx centre",
            default=DEFAULT_MOT_FIELD_CENTER_CODE,
            unit="code",
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "center_y",
            "float",
            "By centre",
            default=DEFAULT_MOT_FIELD_CENTER_CODE,
            unit="code",
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "center_z",
            "float",
            "Bz centre",
            default=DEFAULT_MOT_FIELD_CENTER_CODE,
            unit="code",
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "span",
            "float",
            "Span (+/-)",
            default=DEFAULT_MOT_FIELD_SPAN_CODE,
            unit="code",
            minimum=0.0,
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "points",
            "int",
            "Points per axis",
            default=DEFAULT_MOT_FIELD_POINTS,
            minimum=MINIMUM_MOT_FIELD_POINTS,
            required=True,
            allow_blank=False,
            description="Total autonomous scan cells are points^3",
        ),
        AuthoringField(
            "roi_cx",
            "float",
            "ROI centre x",
            default=None,
            unit="px",
            minimum=0.0,
            required=False,
            allow_blank=True,
            description="Blank uses the frame centre; 0 is the left pixel coordinate",
        ),
        AuthoringField(
            "roi_cy",
            "float",
            "ROI centre y",
            default=None,
            unit="px",
            minimum=0.0,
            required=False,
            allow_blank=True,
            description="Blank uses the frame centre; 0 is the top pixel coordinate",
        ),
        AuthoringField(
            "roi_radius",
            "float",
            "ROI radius",
            default=DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
            unit="px",
            minimum=MINIMUM_POSITIVE_FLOAT,
            required=True,
            allow_blank=False,
            description="The 1x..2x annulus supplies the local background",
        ),
        AuthoringField(
            "folder",
            "path",
            "Report folder",
            default=DEFAULT_MOT_FIELD_REPORT_FOLDER,
            required=True,
            description=(
                "Raw intensity block, exact Bx/By/Bz axes, and refined optimum "
                "are written to mot_field_scan.npz"
            ),
        ),
        AuthoringField(
            "camera_role",
            "choice",
            "Camera role",
            required=True,
            dynamic_choices=True,
            description=(
                "External-trigger-capable camera physically observing the MOT"
            ),
        ),
    )
)


def mot_field_authoring_schema() -> AuthoringSchema:
    return _MOT_FIELD_AUTHORING_SCHEMA


def mot_field_camera_roles(installed_roles) -> tuple[str, ...]:
    roles = tuple(installed_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("MOT camera roles must be unique")
    for role in roles:
        canonical_text(role, "MOT camera role")
    return tuple(role for role in roles if role == DEFAULT_MOT_FIELD_CAMERA_ROLE)


def build_mot_field_intent_from_authoring(
    values: Mapping[str, object],
) -> MotFieldTaskIntent:
    authored = mot_field_authoring_schema().freeze(values)
    if authored["camera_role"] is None:
        raise RuntimeError(
            "MOT field requires the installation's external-trigger-capable "
            "mot_camera role"
        )
    for key in ("roi_cx", "roi_cy"):
        if authored[key] == "":
            authored[key] = None
    return MotFieldTaskIntent(**authored)  # type: ignore[arg-type]


def _mot_camera_choices(context: object) -> tuple[DynamicChoicePresentation, ...]:
    if not isinstance(context, tuple):
        raise TypeError("MOT dynamic choice context must be a role tuple")
    roles = mot_field_camera_roles(context)
    return (
        DynamicChoicePresentation(
            "camera_role",
            tuple(AuthoringChoice(role, role) for role in roles),
            roles[0] if roles else None,
            "MOT field requires the installed mot_camera role" if not roles else "",
        ),
    )


MOT_FIELD_LOGIC_NODE = LogicNodeDeclaration(
    definition=MOT_FIELD_TASK_DEFINITION,
    description=(
        "Sweep da_x/da_y/da_z in one autonomous hardware scan, measure "
        "MOT fluorescence, and report the refined optimum"
    ),
    authoring_schema=_MOT_FIELD_AUTHORING_SCHEMA,
    input_specs=(),
    outputs=(
        OutputPresentation(
            MOT_FIELD_LIVE_OUTPUT_DECLARATIONS[0],
            "MOT intensity grid",
            "Counts",
            "provisional Bx/By/Bz intensity while the scan runs",
        ),
        OutputPresentation(
            MOT_FIELD_FINAL_OUTPUT_DECLARATIONS[0],
            "MOT field",
            "Counts",
            "FINAL optimum and complete three-dimensional intensity grid",
        ),
        OutputPresentation(
            MOT_FIELD_FINAL_OUTPUT_DECLARATIONS[1],
            "scan",
            "Signal",
            "exact raw Camera source scan",
        ),
    ),
    build_request=build_mot_field_intent_from_authoring,
    bind_request=bind_no_node_inputs,
    default_views=(
        DefaultOutputView("grid", "grid"),
        DefaultOutputView("mot_field", "grid"),
    ),
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
        PathPresentationHint(
            "folder",
            mode="dir",
            base_dir=DEFAULT_MOT_FIELD_REPORT_FOLDER,
        ),
    ),
    resolve_dynamic_choices=_mot_camera_choices,
)


def write_mot_field_report(
    result: MotFieldResult,
    folder: str | Path,
) -> Path:
    """Atomically write the authoritative MOT analysis report.

    The exact acquisition result remains the source of Camera values and run
    lineage.  This derived report preserves the analyzed 3-D intensity block,
    all three physical DAC coordinate axes, and the refined optimum.
    """

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    directory = resolve_under_project(folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mot_field_scan.npz"
    axes = tuple(
        np.asarray(
            tuple(axis.coordinate_at(index) for index in range(axis.size))
        )
        for axis in result.point_axes
    )
    temporary = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                intensity=np.asarray(result.intensity),
                bx=axes[0],
                by=axes[1],
                bz=axes[2],
                best=np.asarray(result.best_field, dtype=np.float64),
                best_intensity=np.asarray(
                    result.best_intensity,
                    dtype=np.float64,
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class _AcquisitionEnded(Exception):
    def __init__(self, snapshot: RunSnapshot) -> None:
        self.snapshot = snapshot


class _CancelledAfterAcquisition(Exception):
    pass


class MotFieldTaskDependencies(Protocol):
    """Installation-bound semantic port required by the MOT application.

    This is deliberately not a bag of callbacks.  One composition service
    binds the neutral intent to devices and prepares the coupled exact
    Camera + Sequencer command.
    """

    def mot_field_request(
        self,
        pulse: str,
        *,
        center_x: float,
        center_y: float,
        center_z: float,
        span: float,
        points: int,
        roi_cx: float | None,
        roi_cy: float | None,
        roi_radius: float,
        camera_role: str,
    ) -> MotFieldRequest: ...

    def prepare_mot_field_acquisition(
        self,
        request: MotFieldRequest,
    ) -> PreparedMotFieldAcquisition: ...

def _require_dependencies(dependencies) -> MotFieldTaskDependencies:
    for name in (
        "mot_field_request",
        "prepare_mot_field_acquisition",
    ):
        if not callable(getattr(dependencies, name, None)):
            raise TypeError(
                "MOT task dependencies must expose the semantic method "
                f"{name}()"
            )
    return dependencies


class PreparedMotFieldTask:
    """One-shot, fully bound MOT-field application command.

    Preparation freezes the physical request, exact acquisition and live
    projection before any Run starts.  A frontend may attach ``live_output``
    and then call :meth:`start`; it never assembles scan or analysis stages.
    """

    __slots__ = (
        "_acquisition",
        "_handle",
        "_intent",
        "_live_output",
        "_lock",
        "_request",
        "_started",
    )

    def __init__(
        self,
        intent: MotFieldTaskIntent,
        request: MotFieldRequest,
        acquisition: PreparedMotFieldAcquisition,
        live_output: MotFieldTaskLiveOutput,
    ) -> None:
        if not isinstance(intent, MotFieldTaskIntent):
            raise TypeError("intent must be MotFieldTaskIntent")
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(acquisition, PreparedMotFieldAcquisition):
            raise TypeError("acquisition must be PreparedMotFieldAcquisition")
        if not isinstance(live_output, MotFieldTaskLiveOutput):
            raise TypeError("live_output must be MotFieldTaskLiveOutput")
        self._intent = intent
        self._request = request
        self._acquisition = acquisition
        self._live_output = live_output
        self._lock = threading.Lock()
        self._started = False
        self._handle: MotFieldTaskHandle | None = None

    @property
    def intent(self) -> MotFieldTaskIntent:
        return self._intent

    @property
    def request(self) -> MotFieldRequest:
        return self._request

    @property
    def live_output(self) -> MotFieldTaskLiveOutput:
        return self._live_output

    def start(self) -> "MotFieldTaskHandle":
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedMotFieldTask is one-shot")
            self._started = True
            try:
                handle = MotFieldTaskHandle(
                    self._request,
                    report_folder=self._intent.folder,
                    prepared_acquisition=self._acquisition,
                    live_output=self._live_output,
                )
            except BaseException as error:
                self._live_output.fail(f"{type(error).__name__}: {error}")
                raise
            self._handle = handle
            return handle

    def final_dataset_outputs(
        self,
        result: MotFieldResult,
    ) -> dict[str, FinalDatasetOutput]:
        """Materialize the two named FINAL outputs from this exact task Run."""

        with self._lock:
            handle = self._handle
        if handle is None:
            raise RuntimeError("MOT task has not started")
        return handle.final_dataset_outputs(result)

    def completion_summary(self, result: MotFieldResult) -> str:
        """Expose the exact report path committed by this successful task."""

        if not isinstance(result, MotFieldResult):
            raise TypeError("MOT completion result must be MotFieldResult")
        with self._lock:
            handle = self._handle
        if handle is None or handle.report_path is None:
            raise RuntimeError("MOT task has no committed report path")
        return f"done; report: {handle.report_path}"


def prepare_mot_field_task(
    intent: MotFieldTaskIntent,
    dependencies: MotFieldTaskDependencies,
) -> PreparedMotFieldTask:
    """Bind one complete MOT task without starting hardware execution."""

    if not isinstance(intent, MotFieldTaskIntent):
        raise TypeError("intent must be MotFieldTaskIntent")
    dependencies = _require_dependencies(dependencies)
    request = dependencies.mot_field_request(
        intent.pulse,
        center_x=intent.center_x,
        center_y=intent.center_y,
        center_z=intent.center_z,
        span=intent.span,
        points=intent.points,
        roi_cx=intent.roi_cx,
        roi_cy=intent.roi_cy,
        roi_radius=intent.roi_radius,
        camera_role=intent.camera_role,
    )
    if not isinstance(request, MotFieldRequest):
        raise TypeError("MOT dependency returned a non-MotFieldRequest")
    acquisition = dependencies.prepare_mot_field_acquisition(request)
    if not isinstance(acquisition, PreparedMotFieldAcquisition):
        raise TypeError("MOT dependency returned another acquisition type")
    live_output = MotFieldTaskLiveOutput(
        request,
        acquisition.source_schema,
    )
    try:
        return PreparedMotFieldTask(
            intent,
            request,
            acquisition,
            live_output,
        )
    except BaseException:
        live_output.close()
        raise


class MotFieldTaskHandle:
    """Run-like owner of one exact acquisition and MOT analysis/reporting."""

    def __init__(
        self,
        request: MotFieldRequest,
        *,
        report_folder: str | Path,
        prepared_acquisition: PreparedMotFieldAcquisition,
        live_output: MotFieldTaskLiveOutput,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(prepared_acquisition, PreparedMotFieldAcquisition):
            raise TypeError("prepared_acquisition must be PreparedMotFieldAcquisition")
        if not isinstance(live_output, MotFieldTaskLiveOutput):
            raise TypeError("live_output must be MotFieldTaskLiveOutput")
        self.run_id = RunId(f"mot-field-task-{uuid.uuid4().hex}")
        self._request = request
        self._report_folder = resolve_under_project(report_folder)
        self._prepared_acquisition = prepared_acquisition
        self._live_output = live_output
        self._condition = threading.Condition(threading.RLock())
        self._active: MotFieldAcquisitionHandle | None = None
        self._phase = "acquisition-starting"
        self._cancel_requested = False
        self._cancel_reason = "user requested stop"
        self._terminal: RunSnapshot | None = None
        self._acquisition_result: MotFieldAcquisitionResult | None = None
        self._result: MotFieldResult | None = None
        self._report_path: Path | None = None
        self._thread = threading.Thread(
            target=self._coordinate,
            name=f"zlc-mot-field-{self.run_id.value[-12:]}",
            daemon=False,
        )
        self._thread.start()

    @property
    def source_identity(self) -> str | None:
        with self._condition:
            source = self._acquisition_result
            return None if source is None else source.source_identity

    @property
    def report_path(self) -> Path | None:
        """Return the committed report path, or ``None`` before success."""

        with self._condition:
            return self._report_path

    def _checkpoint(self) -> None:
        with self._condition:
            if self._cancel_requested:
                raise _CancelledAfterAcquisition

    def _finish(
        self,
        state: RunState,
        phase: str,
        *,
        child: RunSnapshot | None = None,
        error: str | None = None,
        admission_rejection: ResourceBusy | None = None,
    ) -> None:
        with self._condition:
            self._terminal = RunSnapshot(
                self.run_id,
                state,
                phase,
                self._result is not None,
                None if child is None else child.commit_recovery_warning,
                error if error is not None else (
                    None if child is None else child.primary_error
                ),
                () if child is None else child.cleanup_errors,
                None if child is None else child.recovery_instruction,
                (
                    admission_rejection
                    if admission_rejection is not None
                    else None if child is None else child.admission_rejection
                ),
            )
            self._active = None
            self._condition.notify_all()

    def _coordinate(self) -> None:
        child: MotFieldAcquisitionHandle | None = None
        try:
            child = self._prepared_acquisition.start(self._live_output.preview_port)
            if not isinstance(child, MotFieldAcquisitionHandle):
                raise TypeError("MOT acquisition starter returned another handle type")
            with self._condition:
                self._active = child
                self._phase = "acquisition-running"
                cancelled = self._cancel_requested
                reason = self._cancel_reason
                self._condition.notify_all()
            if cancelled:
                child.cancel(reason)
            try:
                source = child.result()
            except (RunCancelled, RunFailed) as error:
                raise _AcquisitionEnded(error.snapshot) from None
            finally:
                with self._condition:
                    if self._active is child:
                        self._active = None
            if not isinstance(source, MotFieldAcquisitionResult):
                raise TypeError("MOT acquisition Run returned another result type")
            with self._condition:
                self._acquisition_result = source
                self._phase = "analyzing-final-acquisition"
            self._checkpoint()
            result = analyze_mot_scan(self._request, source)
            self._checkpoint()
            with self._condition:
                self._phase = "writing-report"
            report_path = write_mot_field_report(
                result,
                self._report_folder,
            )
            with self._condition:
                self._result = result
                self._report_path = report_path
            self._finish(
                RunState.SUCCEEDED,
                "mot-field-complete",
                child=child.snapshot(),
            )
        except _CancelledAfterAcquisition:
            self._finish(RunState.CANCELLED, "cancelled")
        except _AcquisitionEnded as ended:
            self._finish(
                ended.snapshot.state,
                "cancelled"
                if ended.snapshot.state is RunState.CANCELLED
                else "failed",
                child=ended.snapshot,
            )
        except RunStartRejected as error:
            self._finish(
                RunState.FAILED,
                "start-rejected",
                error=safe_error_summary(error),
                admission_rejection=error.outcome,
            )
        except BaseException as error:
            with self._condition:
                cancelled = self._cancel_requested
            self._finish(
                RunState.CANCELLED if cancelled else RunState.FAILED,
                "cancelled" if cancelled else "failed",
                child=None if child is None else child.snapshot(),
                error=None if cancelled else safe_error_summary(error),
            )

    def snapshot(self) -> RunSnapshot:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            child = self._active
            phase = self._phase
            cancelling = self._cancel_requested
        child_snapshot = None if child is None else child.snapshot()
        if child_snapshot is not None:
            phase = f"acquisition/{child_snapshot.phase}"
        return RunSnapshot(
            self.run_id,
            RunState.CANCELLING if cancelling else RunState.RUNNING,
            phase,
            False,
            None
            if child_snapshot is None
            else child_snapshot.commit_recovery_warning,
            None if child_snapshot is None else child_snapshot.primary_error,
            () if child_snapshot is None else child_snapshot.cleanup_errors,
            None if child_snapshot is None else child_snapshot.recovery_instruction,
            None if child_snapshot is None else child_snapshot.admission_rejection,
        )

    def cancel(self, reason: str = "user requested stop") -> CancelOutcome:
        text = canonical_text(reason, "cancellation reason")
        with self._condition:
            if self._terminal is not None:
                return CancelOutcome.ALREADY_TERMINAL
            if self._cancel_requested:
                return CancelOutcome.ALREADY_REQUESTED
            self._cancel_requested = True
            self._cancel_reason = text
            child = self._active
            self._condition.notify_all()
        return (
            CancelOutcome.REQUESTED
            if child is None
            else child.cancel(text)
        )

    def wait(self, timeout: float | None = None) -> RunSnapshot:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout < 0
        ):
            raise ValueError("wait timeout must be a non-negative real or None")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._terminal is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"MOT field task {self.run_id} is active")
                self._condition.wait(remaining)
            snapshot = self._terminal
        remaining = (
            None
            if deadline is None
            else max(0.0, deadline - time.monotonic())
        )
        self._thread.join(remaining)
        if self._thread.is_alive():
            raise TimeoutError(
                f"MOT field task {self.run_id} is terminal but not reaped"
            )
        return snapshot

    def result(self, timeout: float | None = None) -> MotFieldResult:
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED:
            assert self._result is not None
            return self._result
        if snapshot.state is RunState.CANCELLED:
            raise RunCancelled(snapshot)
        raise RunFailed(snapshot)

    def final_dataset_outputs(
        self,
        result: MotFieldResult,
    ) -> dict[str, FinalDatasetOutput]:
        """Publish FINAL result/source datasets from the completed acquisition."""

        if not isinstance(result, MotFieldResult):
            raise TypeError("result must be MotFieldResult")
        with self._condition:
            if self._terminal is None or self._terminal.state is not RunState.SUCCEEDED:
                raise RuntimeError("MOT final outputs require successful task terminal")
            if result is not self._result:
                raise ValueError("MOT result belongs to another task")
            acquisition = self._acquisition_result
        if acquisition is None:
            raise RuntimeError("MOT task lost its exact source acquisition")
        return mot_field_final_outputs(result, acquisition)


__all__ = [
    "DEFAULT_MOT_FIELD_PULSE_PATH",
    "DEFAULT_MOT_FIELD_REPORT_FOLDER",
    "MOT_FIELD_LOGIC_NODE",
    "MotFieldTaskDependencies",
    "MotFieldTaskHandle",
    "MotFieldTaskIntent",
    "PreparedMotFieldTask",
    "build_mot_field_intent_from_authoring",
    "mot_field_authoring_schema",
    "mot_field_camera_roles",
    "prepare_mot_field_task",
    "write_mot_field_report",
]
