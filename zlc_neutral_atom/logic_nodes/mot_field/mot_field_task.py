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
import uuid
from typing import Callable, Mapping, Protocol

import numpy as np

from zlc_neutral_atom.capture.artifact import (
    CaptureRepository,
    PendingCaptureArtifact,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.capture.triggered import TriggeredPipelineResult
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
    mot_field_source_identity,
    mot_intensity_schema,
)
from .mot_field_live import MOT_FIELD_LIVE_OUTPUT_DECLARATIONS
from .mot_field_task_live import MotFieldTaskLiveOutput
from .application import (
    PreparedMotFieldAcquisition,
)
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunHandle,
    RunPlan,
)
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
    pixel coordinate.
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
    """Write the human-readable MOT analysis export after source FINAL.

    The CaptureArtifact CAS reference is the sole machine authority.  This
    replace-last ``.npz`` is a derived convenience report and never substitutes
    for repository admission, source lineage, or the frozen analysis request.
    """

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    directory = resolve_under_project(folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mot_field_scan.npz"
    axes = tuple(
        np.asarray(domain)
        for domain in result.grid_topology.coordinate_domains
    )
    temporary = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                source_identity=np.asarray(result.source_identity),
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


class MotFieldTaskDependencies(Protocol):
    """Installation-bound semantic port required by the MOT application."""

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


def _acquisition_from_pipeline(
    request: MotFieldRequest,
    pipeline: TriggeredPipelineResult,
) -> MotFieldAcquisitionResult:
    """Validate one exact in-process capture as MOT source data."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(pipeline, TriggeredPipelineResult):
        raise TypeError("MOT analysis requires TriggeredPipelineResult")
    if (
        request.trigger_channel is not None
        and pipeline.lineage.trigger_channel != request.trigger_channel
    ):
        raise ValueError("MOT trigger lineage differs from the explicit request")
    dataset = pipeline.capture.dataset
    mot_intensity_schema(request, dataset.block.schema)
    return MotFieldAcquisitionResult(
        dataset.snapshot,
        dataset.provenance,
        mot_field_source_identity(dataset.snapshot, dataset.provenance),
    )


@dataclass(frozen=True, slots=True)
class _FinalMotFieldTaskResult:
    """One analysis bound to the CaptureArtifact that made it FINAL."""

    reference: CaptureArtifactRef
    source: MotFieldAcquisitionResult
    analysis: MotFieldResult

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        if not isinstance(self.source, MotFieldAcquisitionResult):
            raise TypeError("source must be MotFieldAcquisitionResult")
        if not isinstance(self.analysis, MotFieldResult):
            raise TypeError("analysis must be MotFieldResult")
        if self.analysis.source_identity != self.source.source_identity:
            raise ValueError("MOT analysis belongs to another source acquisition")


class _MotFieldTaskResultOwner:
    """One-assignment bridge from pending immutable data to its FINAL ref."""

    __slots__ = ("_analysis", "_final", "_lock", "_source")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source: MotFieldAcquisitionResult | None = None
        self._analysis: MotFieldResult | None = None
        self._final: _FinalMotFieldTaskResult | None = None

    def publish_analysis(
        self,
        source: MotFieldAcquisitionResult,
        analysis: MotFieldResult,
    ) -> None:
        if not isinstance(source, MotFieldAcquisitionResult):
            raise TypeError("source must be MotFieldAcquisitionResult")
        if not isinstance(analysis, MotFieldResult):
            raise TypeError("analysis must be MotFieldResult")
        if analysis.source_identity != source.source_identity:
            raise ValueError("MOT analysis belongs to another source acquisition")
        with self._lock:
            if self._source is not None or self._analysis is not None:
                raise RuntimeError("MOT analysis was already published")
            self._source = source
            self._analysis = analysis

    def bind_final(
        self,
        reference: CaptureArtifactRef,
    ) -> _FinalMotFieldTaskResult:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        with self._lock:
            current = self._final
            if current is not None:
                if current.reference != reference:
                    raise ValueError("MOT task is already bound to another FINAL capture")
                return current
            source = self._source
            analysis = self._analysis
            if source is None or analysis is None:
                raise RuntimeError("MOT task has no completed immutable analysis")
            final = _FinalMotFieldTaskResult(reference, source, analysis)
            self._final = final
            return final


def _compile_mot_field_task_plan(
    request: MotFieldRequest,
    capture_plan: RunPlan,
    report_folder: str | Path,
    result_owner: _MotFieldTaskResultOwner,
) -> RunPlan:
    """Extend one capture plan with MOT validation, FINAL, and derived export.

    The wrapped capture plan remains the only hardware and commit lifecycle
    owner.  MOT analysis validates the pending exact immutable dataset once so
    invalid physics cannot be committed.  The capture CAS/codec owns persisted
    byte integrity; after publication the exact same analysis object is bound
    to the FINAL reference and reused by report and Dataset outputs.
    """

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(capture_plan, RunPlan):
        raise TypeError("capture_plan must be RunPlan")
    if not isinstance(result_owner, _MotFieldTaskResultOwner):
        raise TypeError("result_owner must be _MotFieldTaskResultOwner")
    report_folder = resolve_under_project(report_folder)

    base_preflight = capture_plan.preflight
    base_execute = capture_plan.execute
    base_cleanup = capture_plan.cleanup
    base_finalize = capture_plan.finalize
    base_dispose = capture_plan.dispose_unfinalized

    def finalize(
        context: PostSafetyContext,
        pending: PendingCaptureArtifact,
    ) -> CaptureArtifactRef:
        if not isinstance(pending, PendingCaptureArtifact):
            raise TypeError("MOT finalize requires PendingCaptureArtifact")
        try:
            pipeline = pending.pipeline_result
            if not isinstance(pipeline, TriggeredPipelineResult):
                raise TypeError("MOT capture plan lost its triggered result")
            source = _acquisition_from_pipeline(request, pipeline)
            analysis = analyze_mot_scan(request, source)
            result_owner.publish_analysis(source, analysis)
        except BaseException as error:
            if base_dispose is not None:
                try:
                    base_dispose(pending)
                except BaseException as dispose_error:
                    try:
                        error.add_note(
                            "pending capture disposal also failed: "
                            f"{type(dispose_error).__name__}: {dispose_error}"
                        )
                    except BaseException:
                        pass
            raise

        reference = base_finalize(context, pending)
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("capture plan returned another FINAL reference type")

        # Everything below this line is post-FINAL derived work.  The runtime
        # preserves a successful committed result even if this export fails,
        # recording the failure as a terminal diagnostic.
        final = result_owner.bind_final(reference)
        write_mot_field_report(final.analysis, report_folder)
        return reference

    return RunPlan(
        name="mot-field-task",
        resource_claims=capture_plan.resource_claims,
        bound_devices=capture_plan.bound_devices,
        preflight=base_preflight,
        execute=base_execute,
        cleanup=base_cleanup,
        finalize=finalize,
        interrupt_operations=capture_plan.interrupt_operations,
        timeout_seconds=capture_plan.timeout_seconds,
        requires_final_commit=True,
        dispose_unfinalized=base_dispose,
    )


class PreparedMotFieldTask:
    """One-shot MOT command backed by one ordinary repository RunPlan."""

    __slots__ = (
        "_intent",
        "_live_output",
        "_lock",
        "_handle",
        "_plan",
        "_request",
        "_result_owner",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        intent: MotFieldTaskIntent,
        request: MotFieldRequest,
        plan: RunPlan,
        start_run: Callable[[RunPlan], RunHandle],
        live_output: MotFieldTaskLiveOutput,
        result_owner: _MotFieldTaskResultOwner,
    ) -> None:
        if not isinstance(intent, MotFieldTaskIntent):
            raise TypeError("intent must be MotFieldTaskIntent")
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be RunPlan")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        if not isinstance(live_output, MotFieldTaskLiveOutput):
            raise TypeError("live_output must be MotFieldTaskLiveOutput")
        if not isinstance(result_owner, _MotFieldTaskResultOwner):
            raise TypeError("result_owner must be _MotFieldTaskResultOwner")
        self._intent = intent
        self._request = request
        self._plan = plan
        self._start_run = start_run
        self._live_output = live_output
        self._result_owner = result_owner
        self._lock = threading.Lock()
        self._started = False
        self._handle: RunHandle | None = None

    @property
    def intent(self) -> MotFieldTaskIntent:
        return self._intent

    @property
    def request(self) -> MotFieldRequest:
        return self._request

    @property
    def live_output(self) -> MotFieldTaskLiveOutput:
        return self._live_output

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedMotFieldTask is one-shot")
            self._started = True
        try:
            handle = self._start_run(self._plan)
            if not isinstance(handle, RunHandle):
                raise TypeError("MOT start_run returned another handle type")
            with self._lock:
                self._handle = handle
            return handle
        except BaseException as error:
            self._live_output.fail(f"{type(error).__name__}: {error}")
            raise

    def _require_own_success(
        self,
        reference: CaptureArtifactRef,
    ) -> _FinalMotFieldTaskResult:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("MOT FINAL result must be CaptureArtifactRef")
        with self._lock:
            handle = self._handle
        if handle is None:
            raise RuntimeError("MOT task has not started")
        successful = handle.result(timeout=0.0)
        if not isinstance(successful, CaptureArtifactRef):
            raise TypeError("MOT Run returned another FINAL result type")
        if successful != reference:
            raise ValueError("MOT result belongs to another prepared task")
        # Normal publication binds in finalize.  A durable commit recovered by
        # RunController deliberately does not re-enter finalize; the successful
        # exact Run result is then the authority that completes this binding.
        return self._result_owner.bind_final(reference)

    def final_dataset_outputs(
        self,
        reference: CaptureArtifactRef,
    ) -> dict[str, FinalDatasetOutput]:
        """Materialize the named FINAL outputs from the committed capture."""

        final = self._require_own_success(reference)
        return mot_field_final_outputs(final.analysis, final.source)

    def completion_summary(self, reference: CaptureArtifactRef) -> str:
        """Name the repository FINAL rather than a replace-last report file."""

        self._require_own_success(reference)
        return f"done; capture: {reference.target_ref}"


def prepare_mot_field_task(
    intent: MotFieldTaskIntent,
    dependencies: MotFieldTaskDependencies,
    *,
    capture_repository: CaptureRepository,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedMotFieldTask:
    """Bind one complete MOT task without starting hardware execution."""

    if not isinstance(intent, MotFieldTaskIntent):
        raise TypeError("intent must be MotFieldTaskIntent")
    dependencies = _require_dependencies(dependencies)
    if type(capture_repository) is not CaptureRepository:
        raise TypeError("capture_repository must be CaptureRepository")
    if not callable(start_run):
        raise TypeError("start_run must be callable")
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
    result_owner = _MotFieldTaskResultOwner()
    try:
        capture_plan = acquisition.compile_capture_plan(
            capture_repository,
            preview=live_output.preview_port,
        )
        plan = _compile_mot_field_task_plan(
            request,
            capture_plan,
            intent.folder,
            result_owner,
        )
        return PreparedMotFieldTask(
            intent,
            request,
            plan,
            start_run,
            live_output,
            result_owner,
        )
    except BaseException:
        live_output.close()
        raise


__all__ = [
    "DEFAULT_MOT_FIELD_PULSE_PATH",
    "DEFAULT_MOT_FIELD_REPORT_FOLDER",
    "MOT_FIELD_LOGIC_NODE",
    "MotFieldTaskDependencies",
    "MotFieldTaskIntent",
    "PreparedMotFieldTask",
    "build_mot_field_intent_from_authoring",
    "mot_field_authoring_schema",
    "mot_field_camera_roles",
    "prepare_mot_field_task",
    "write_mot_field_report",
]
