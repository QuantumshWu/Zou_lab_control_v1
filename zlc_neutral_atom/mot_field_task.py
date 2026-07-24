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
from typing import Protocol

import numpy as np

from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.mot_field import (
    DEFAULT_MOT_FIELD_CAMERA_ROLE,
    MINIMUM_MOT_FIELD_POINTS,
    MotFieldRequest,
    MotFieldResult,
    analyze_mot_scan,
    mot_field_final_outputs,
)
from zlc_neutral_atom.mot_field_task_live import MotFieldTaskLiveOutput
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunCancelled,
    RunFailed,
    RunHandle,
    RunId,
    RunStartRejected,
    RunSnapshot,
    RunState,
)
from zlc_neutral_atom.runtime.resources import ResourceBusy
from zlc_neutral_atom.scan import ScanArtifactRef
from zlc_neutral_atom.scan.application import PreparedExactScan
from zlc_neutral_atom.scan.repository import MaterializedScanData
from zlc_storage import (
    canonical_text,
    finite_real,
    integer,
    normalized_text,
    positive_real,
)
from zlc_storage.paths import resolve_under_project


DEFAULT_MOT_FIELD_REPORT_FOLDER = "mot_field"


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


def write_mot_field_report(
    result: MotFieldResult,
    folder: str | Path,
) -> Path:
    """Atomically write the authoritative MOT analysis report.

    The exact scan repository remains the owner of camera frames and run
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


class _ScanEnded(Exception):
    def __init__(self, snapshot: RunSnapshot) -> None:
        self.snapshot = snapshot


class _CancelledAfterScan(Exception):
    pass


class MotFieldScanMaterializer(Protocol):
    """Typed canonical scan-admission port retained by a prepared MOT task."""

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData: ...


class MotFieldTaskDependencies(MotFieldScanMaterializer, Protocol):
    """Installation-bound semantic port required by the MOT application.

    This is deliberately not a bag of callbacks.  One composition service
    binds the neutral intent to devices, prepares the exact scan command, and
    admits the resulting canonical scan artifact.
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

    def prepare_mot_field_scan(
        self,
        request: MotFieldRequest,
    ) -> PreparedExactScan: ...

def _require_dependencies(dependencies) -> MotFieldTaskDependencies:
    for name in (
        "mot_field_request",
        "prepare_mot_field_scan",
        "materialize_scan",
    ):
        if not callable(getattr(dependencies, name, None)):
            raise TypeError(
                "MOT task dependencies must expose the semantic method "
                f"{name}()"
            )
    return dependencies


def _require_materializer(materializer) -> MotFieldScanMaterializer:
    if not callable(getattr(materializer, "materialize_scan", None)):
        raise TypeError(
            "MOT scan materializer must expose materialize_scan()"
        )
    return materializer


class PreparedMotFieldTask:
    """One-shot, fully bound MOT-field application command.

    Preparation freezes the physical request, exact-scan contract and live
    projection before any Run starts.  A frontend may attach ``live_output``
    and then call :meth:`start`; it never assembles scan or analysis stages.
    """

    __slots__ = (
        "_handle",
        "_intent",
        "_live_output",
        "_lock",
        "_materializer",
        "_request",
        "_scan",
        "_started",
    )

    def __init__(
        self,
        intent: MotFieldTaskIntent,
        request: MotFieldRequest,
        scan: PreparedExactScan,
        live_output: MotFieldTaskLiveOutput,
        materializer: MotFieldScanMaterializer,
    ) -> None:
        if not isinstance(intent, MotFieldTaskIntent):
            raise TypeError("intent must be MotFieldTaskIntent")
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(scan, PreparedExactScan):
            raise TypeError("scan must be PreparedExactScan")
        if not isinstance(live_output, MotFieldTaskLiveOutput):
            raise TypeError("live_output must be MotFieldTaskLiveOutput")
        self._intent = intent
        self._request = request
        self._scan = scan
        self._live_output = live_output
        self._materializer = _require_materializer(materializer)
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
                    prepared_scan=self._scan,
                    live_output=self._live_output,
                    materializer=self._materializer,
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
    scan = dependencies.prepare_mot_field_scan(request)
    if not isinstance(scan, PreparedExactScan):
        raise TypeError("MOT dependency returned a non-PreparedExactScan")
    live_output = MotFieldTaskLiveOutput(
        request,
        scan.source_schema,
        scan.output_contract,
    )
    try:
        return PreparedMotFieldTask(
            intent,
            request,
            scan,
            live_output,
            dependencies,
        )
    except BaseException:
        live_output.close()
        raise


class MotFieldTaskHandle:
    """Run-like owner of one exact scan followed by MOT analysis/reporting."""

    def __init__(
        self,
        request: MotFieldRequest,
        *,
        report_folder: str | Path,
        prepared_scan: PreparedExactScan,
        live_output: MotFieldTaskLiveOutput,
        materializer: MotFieldScanMaterializer,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(prepared_scan, PreparedExactScan):
            raise TypeError("prepared_scan must be PreparedExactScan")
        if not isinstance(live_output, MotFieldTaskLiveOutput):
            raise TypeError("live_output must be MotFieldTaskLiveOutput")
        self.run_id = RunId(f"mot-field-task-{uuid.uuid4().hex}")
        self._request = request
        self._report_folder = resolve_under_project(report_folder)
        self._prepared_scan = prepared_scan
        self._live_output = live_output
        self._materializer = _require_materializer(materializer)
        self._condition = threading.Condition(threading.RLock())
        self._active: RunHandle | None = None
        self._phase = "scan-starting"
        self._cancel_requested = False
        self._cancel_reason = "user requested stop"
        self._terminal: RunSnapshot | None = None
        self._scan_ref: ScanArtifactRef | None = None
        self._materialized_scan: MaterializedScanData | None = None
        self._result: MotFieldResult | None = None
        self._report_path: Path | None = None
        self._thread = threading.Thread(
            target=self._coordinate,
            name=f"zlc-mot-field-{self.run_id.value[-12:]}",
            daemon=False,
        )
        self._thread.start()

    @property
    def source_scan_ref(self) -> ScanArtifactRef | None:
        with self._condition:
            return self._scan_ref

    @property
    def report_path(self) -> Path | None:
        """Return the committed report path, or ``None`` before success."""

        with self._condition:
            return self._report_path

    def _checkpoint(self) -> None:
        with self._condition:
            if self._cancel_requested:
                raise _CancelledAfterScan

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
        child: RunHandle | None = None
        try:
            child = self._prepared_scan.start(self._live_output.preview_port)
            if not isinstance(child, RunHandle):
                raise TypeError("MOT scan starter returned a non-RunHandle")
            with self._condition:
                self._active = child
                self._phase = "scan-running"
                cancelled = self._cancel_requested
                reason = self._cancel_reason
                self._condition.notify_all()
            if cancelled:
                child.cancel(reason)
            try:
                source = child.result()
            except (RunCancelled, RunFailed) as error:
                raise _ScanEnded(error.snapshot) from None
            finally:
                with self._condition:
                    if self._active is child:
                        self._active = None
            if not isinstance(source, ScanArtifactRef):
                raise TypeError("MOT scan Run returned a non-ScanArtifactRef")
            with self._condition:
                self._scan_ref = source
                self._phase = "analyzing-final-scan"
            self._checkpoint()
            materialized = self._materializer.materialize_scan(source)
            if not isinstance(materialized, MaterializedScanData):
                raise TypeError("MOT materializer returned a non-MaterializedScanData")
            with self._condition:
                self._materialized_scan = materialized
            self._checkpoint()
            result = analyze_mot_scan(self._request, materialized)
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
        except _CancelledAfterScan:
            self._finish(RunState.CANCELLED, "cancelled")
        except _ScanEnded as ended:
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
            phase = f"scan/{child_snapshot.phase}"
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
        """Publish FINAL result/source datasets from the already admitted scan."""

        if not isinstance(result, MotFieldResult):
            raise TypeError("result must be MotFieldResult")
        with self._condition:
            if self._terminal is None or self._terminal.state is not RunState.SUCCEEDED:
                raise RuntimeError("MOT final outputs require successful task terminal")
            if result is not self._result:
                raise ValueError("MOT result belongs to another task")
            materialized = self._materialized_scan
        if materialized is None:
            raise RuntimeError("MOT task lost its admitted source scan")
        return mot_field_final_outputs(result, materialized)


__all__ = [
    "DEFAULT_MOT_FIELD_REPORT_FOLDER",
    "MotFieldScanMaterializer",
    "MotFieldTaskDependencies",
    "MotFieldTaskHandle",
    "MotFieldTaskIntent",
    "PreparedMotFieldTask",
    "prepare_mot_field_task",
    "write_mot_field_report",
]
