"""Calibration task application orchestration.

The ordinary Task always acquires one exact capture and calibrates that capture
as two linked flat Runs.  Recalibration of an existing CaptureArtifactRef is the
separate public Calibration API path, not a second Task mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import threading
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Mapping, Protocol

from zlc_neutral_atom.capture.artifact import CaptureRepository
from zlc_neutral_atom.authoring import (
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
    MINIMUM_POSITIVE_FLOAT,
)
from zlc_neutral_atom.capture.application import (
    CAPTURE_READOUT_EVENT_AXIS_ID,
    CaptureRequest,
    PreparedFiniteCapture,
)
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
)
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
    LiveDatasetOutputOwner,
    single_live_dataset_output,
)
from .calibration import (
    CalibrationAnalysisRequest,
    ResolvedCalibration,
    ThresholdMethod,
)
from .reference import CalibrationArtifactRef
from .repository import CalibrationRepository
from .task_output import (
    CalibrationTaskOutput,
    read_calibration_task_output,
    write_calibration_task_output,
)
from .sitemap import DEFAULT_CALIBRATION_PULSE_PATH, SitemapCalibrationRequest
from zlc_neutral_atom.runtime.dataset import (
    DatasetPreviewSnapshot,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.capture.pipeline import (
    CapturePreviewPort,
    CapturePreviewSpec,
)
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_storage import (
    canonical_text,
    integer,
    normalized_text,
    positive_real,
)
from zlc_pulse import PulseExecutionForm

if TYPE_CHECKING:
    from .analysis import CalibrationComputation
    from .projection import CalibrationSiteMapContext

CALIBRATION_THRESHOLD_METHODS = tuple(item.value for item in ThresholdMethod)
CALIBRATION_LIVE_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "frame",
        "zlc_neutral_atom.calibration-task.live-frame",
    ),
)
DEFAULT_CALIBRATION_FOLDER = "calibrations"
DEFAULT_CALIBRATION_THRESHOLD_METHOD = "otsu"
DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S = 0.020
DEFAULT_CALIBRATION_READOUT_EXPOSURE_S = 0.005
DEFAULT_CALIBRATION_THRESHOLD_FRAMES = 100
MINIMUM_CALIBRATION_THRESHOLD_FRAMES = 2
DEFAULT_CALIBRATION_ROI_RADIUS = 1
MINIMUM_CALIBRATION_ROI_RADIUS = 1
DEFAULT_CALIBRATION_CAMERA_ROLE = "camera"


def admit_calibration_task_output(
    path: str | Path,
    *,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
) -> ResolvedCalibration:
    """Admit a saved task pointer and its complete capture lineage."""

    if not isinstance(capture_repository, CaptureRepository):
        raise TypeError("capture_repository must be CaptureRepository")
    if not isinstance(calibration_repository, CalibrationRepository):
        raise TypeError("calibration_repository must be CalibrationRepository")
    output = read_calibration_task_output(path)
    resolved = calibration_repository.admit(
        output.calibration_ref,
        capture_repository,
    )
    if (
        resolved.artifact.source_binding.source_capture_ref
        != output.source_capture_ref
    ):
        raise ValueError("saved calibration pointer names another source capture")
    return resolved


def write_calibration_task_outputs(
    source: CaptureArtifactRef,
    calibration: CalibrationArtifactRef,
    *,
    folder: str | Path,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
    expected_camera_role: str | None = None,
    render_report: Callable | None = None,
) -> None:
    """Write the typed pointer and non-authoritative human result bundle.

    Capture and calibration repositories remain the only canonical artifact
    owners.  Existing-capture recalibration consumes the canonical Capture ref;
    the Task never creates a duplicate raw-frame authority.  Frontend rendering
    is an explicit composition callback; this module never imports a renderer.
    """

    from .projection import project_calibration_report
    from .result_bundle import write_calibration_result_bundle

    if not isinstance(source, CaptureArtifactRef):
        raise TypeError("source must be CaptureArtifactRef")
    if not isinstance(calibration, CalibrationArtifactRef):
        raise TypeError("calibration must be CalibrationArtifactRef")
    if not isinstance(capture_repository, CaptureRepository):
        raise TypeError("capture_repository must be CaptureRepository")
    if not isinstance(calibration_repository, CalibrationRepository):
        raise TypeError("calibration_repository must be CalibrationRepository")
    if not callable(render_report):
        raise TypeError("render_report must be callable")
    camera_role = (
        None
        if expected_camera_role is None
        else normalized_text(expected_camera_role, "expected_camera_role")
    )
    computation = calibration_repository.load_computation(calibration)
    calibration_artifact = computation.artifact
    if calibration_artifact.source_binding.source_capture_ref != source:
        raise ValueError("calibration task result belongs to another source capture")
    admitted = capture_repository.admit(source)
    if (
        camera_role is not None
        and admitted.artifact.camera_provenance.binding.value != camera_role
    ):
        raise ValueError(
            "calibration task source belongs to camera role "
            f"{admitted.artifact.camera_provenance.binding.value!r}, not "
            f"{camera_role!r}"
        )
    root = Path(folder).expanduser()
    if not root.is_absolute():
        raise ValueError("calibration output folder must be resolved by composition")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    pointer_path = root / "calibration_ref.json"
    pointer_temp = root / f".calibration_ref.{nonce}.tmp"
    report = root / "report"
    report_stage = root / f".report.{nonce}.tmp"
    report_backup = root / f".report.{nonce}.old"
    installed_report = False
    committed = False
    try:
        write_calibration_task_output(
            pointer_temp,
            CalibrationTaskOutput(calibration, source),
        )
        write_calibration_result_bundle(
            report_stage,
            project_calibration_report(computation, calibration),
            calibration,
            source,
            calibration_repository_root=calibration_repository.root,
            capture_repository_root=capture_repository.root,
            render_report=render_report,
        )
        if report.exists() and (
            report.is_symlink()
            or not report.is_dir()
            or report.resolve().parent != root
        ):
            raise ValueError(
                "calibration report output is not a task-owned directory"
            )

        if report.exists():
            os.replace(report, report_backup)
        os.replace(report_stage, report)
        installed_report = True

        # The typed pointer is the task output's final commit record.  The
        # report is staged and installed before the pointer changes.
        os.replace(pointer_temp, pointer_path)
        committed = True
    finally:
        if not committed:
            if installed_report and report.exists():
                shutil.rmtree(report)
            if report_backup.exists():
                os.replace(report_backup, report)
        if report_stage.exists():
            shutil.rmtree(report_stage)
        if committed and report_backup.exists():
            shutil.rmtree(report_backup)
        for temporary_path in (pointer_temp,):
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class CalibrationTaskIntent:
    """Complete capture-then-calibrate Task intent before service binding."""

    folder: str
    pulse: str
    threshold_method: str
    reference_exposure_s: float
    readout_exposure_s: float
    threshold_frames: int
    roi_radius: int
    camera_role: str

    def __post_init__(self) -> None:
        folder = normalized_text(self.folder, "folder")
        pulse = normalized_text(self.pulse, "pulse")
        threshold_method = normalized_text(
            self.threshold_method,
            "threshold_method",
        ).lower()
        if threshold_method not in CALIBRATION_THRESHOLD_METHODS:
            raise ValueError(
                "threshold_method must be one of "
                f"{CALIBRATION_THRESHOLD_METHODS}"
            )
        reference_exposure_s = positive_real(
            self.reference_exposure_s,
            "reference_exposure_s",
        )
        readout_exposure_s = positive_real(
            self.readout_exposure_s,
            "readout_exposure_s",
        )
        threshold_frames = integer(
            self.threshold_frames,
            "threshold_frames",
            minimum=MINIMUM_CALIBRATION_THRESHOLD_FRAMES,
        )
        assert threshold_frames is not None
        roi_radius = integer(
            self.roi_radius,
            "roi_radius",
            minimum=MINIMUM_CALIBRATION_ROI_RADIUS,
        )
        assert roi_radius is not None
        camera_role = normalized_text(self.camera_role, "camera_role")
        object.__setattr__(self, "folder", folder)
        object.__setattr__(self, "pulse", pulse)
        object.__setattr__(self, "threshold_method", threshold_method)
        object.__setattr__(
            self,
            "reference_exposure_s",
            reference_exposure_s,
        )
        object.__setattr__(self, "readout_exposure_s", readout_exposure_s)
        object.__setattr__(self, "threshold_frames", threshold_frames)
        object.__setattr__(self, "roi_radius", roi_radius)
        object.__setattr__(self, "camera_role", camera_role)


_CALIBRATION_TASK_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "folder",
            "path",
            "Folder",
            default=DEFAULT_CALIBRATION_FOLDER,
            required=True,
            description="Directory for the typed result pointer and human report.",
        ),
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_CALIBRATION_PULSE_PATH,
            required=True,
            description="Imaging pulse used for each long-short-long bracket.",
        ),
        AuthoringField(
            "threshold_method",
            "choice",
            "Threshold",
            default=DEFAULT_CALIBRATION_THRESHOLD_METHOD,
            required=True,
            choices=tuple(
                AuthoringChoice(value, value)
                for value in CALIBRATION_THRESHOLD_METHODS
            ),
            description="Per-site threshold estimator.",
        ),
        AuthoringField(
            "reference_exposure_s",
            "float",
            "Reference exposure (long)",
            default=DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S,
            required=True,
            unit="s",
            minimum=MINIMUM_POSITIVE_FLOAT,
            allow_blank=False,
            description=(
                "Positive long exposure for the two outer reference "
                "frames."
            ),
        ),
        AuthoringField(
            "readout_exposure_s",
            "float",
            "Readout exposure (short)",
            default=DEFAULT_CALIBRATION_READOUT_EXPOSURE_S,
            required=True,
            unit="s",
            minimum=MINIMUM_POSITIVE_FLOAT,
            allow_blank=False,
            description="Positive exposure for the middle readout frame.",
        ),
        AuthoringField(
            "threshold_frames",
            "int",
            "Reference brackets",
            default=DEFAULT_CALIBRATION_THRESHOLD_FRAMES,
            required=True,
            minimum=MINIMUM_CALIBRATION_THRESHOLD_FRAMES,
            allow_blank=False,
            description="Number of long-short-long calibration shots.",
        ),
        AuthoringField(
            "roi_radius",
            "int",
            "ROI radius",
            default=DEFAULT_CALIBRATION_ROI_RADIUS,
            required=True,
            unit="px",
            minimum=MINIMUM_CALIBRATION_ROI_RADIUS,
            allow_blank=False,
            description="Per-site square ROI half-width in pixels.",
        ),
        AuthoringField(
            "camera_role",
            "choice",
            "Camera",
            required=True,
            dynamic_choices=True,
            description="Camera used for live calibration acquisition.",
        ),
    )
)


def calibration_task_authoring_schema() -> AuthoringSchema:
    """Return the ordinary authoring declaration owned by this typed intent."""

    return _CALIBRATION_TASK_AUTHORING_SCHEMA


def calibration_task_default_camera_role(available_roles) -> str | None:
    """Choose the owner default from a frozen installation role snapshot."""

    roles = tuple(available_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("calibration camera roles must be unique")
    for role in roles:
        canonical_text(role, "calibration camera role")
    if DEFAULT_CALIBRATION_CAMERA_ROLE in roles:
        return DEFAULT_CALIBRATION_CAMERA_ROLE
    return roles[0] if roles else None


def build_calibration_task_intent_from_authoring(
    values: Mapping[str, object],
) -> CalibrationTaskIntent:
    authored = calibration_task_authoring_schema().freeze(values)
    if authored["camera_role"] is None:
        raise RuntimeError(
            "Calibrate readout requires an installed camera role with a "
            "site-map acquisition profile"
        )
    return CalibrationTaskIntent(**authored)  # type: ignore[arg-type]


class CalibrationTaskApplicationPort(Protocol):
    """Installation/runtime capabilities required by one calibration task.

    This is deliberately one use-case port, not a service locator.  Every method
    has a concrete calibration meaning and typed arguments/results.  A
    composition root supplies a dedicated adapter for this port; the command
    never accepts an Experiment/session/service locator, arbitrary callbacks, or
    a caller-interpreted physical event layout.
    """

    def sitemap_request(
        self,
        *,
        frames: int,
        camera_role: str,
        pulse: str,
        reference_exposure_s: float,
        readout_exposure_s: float,
        threshold_method: str,
        roi_radius: int,
    ) -> SitemapCalibrationRequest: ...

    def prepare_capture(
        self,
        request: CaptureRequest,
    ) -> PreparedFiniteCapture: ...

    def start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
        *,
        lifecycle_owner: object | None = None,
    ) -> RunHandle: ...

    def write_calibration_task_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str,
        expected_camera_role: str,
    ) -> None: ...

    def load_calibration_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationComputation: ...


class CalibrationTaskLiveOutputPort(Protocol):
    """Attach a task-owned live Dataset without interpreting its event roles."""

    def open_live_dataset(
        self,
        spec: CapturePreviewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
    ) -> CapturePreviewPort: ...


@dataclass(frozen=True, slots=True)
class _CalibrationTaskPlan:
    """Package-private result of binding one complete application intent."""

    intent: CalibrationTaskIntent
    analysis: CalibrationAnalysisRequest
    sequence: SitemapCalibrationRequest

    def __post_init__(self) -> None:
        if not isinstance(self.intent, CalibrationTaskIntent):
            raise TypeError("intent must be CalibrationTaskIntent")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        if not isinstance(self.sequence, SitemapCalibrationRequest):
            raise TypeError("calibration Task requires SitemapCalibrationRequest")
        if self.sequence.analysis != self.analysis:
            raise ValueError("calibration sequence differs from frozen analysis")


def _require_analysis_matches_intent(
    intent: CalibrationTaskIntent,
    analysis: CalibrationAnalysisRequest,
) -> None:
    if not isinstance(analysis, CalibrationAnalysisRequest):
        raise TypeError("calibration application returned an invalid analysis request")
    if analysis.layout.readout_event_axis_id != CAPTURE_READOUT_EVENT_AXIS_ID:
        raise ValueError("calibration analysis uses another capture event axis")
    if analysis.threshold_method is not ThresholdMethod(intent.threshold_method):
        raise ValueError("calibration analysis changed the requested threshold method")
    if analysis.box_radius != intent.roi_radius:
        raise ValueError("calibration analysis changed the requested ROI radius")
    if (
        analysis.expected_centers_xy is None
        or analysis.maximum_site_residual_px is None
    ):
        raise ValueError(
            "formal calibration requires installation-owned spatial admission intent"
        )


def _reference_preview_ordinals(
    intent: CalibrationTaskIntent,
    sequence: SitemapCalibrationRequest,
) -> tuple[int, ...]:
    """Select the first declared reference event once per complete bracket.

    The preview is presentation-only; the exact capture remains complete.  This
    is nevertheless a physical event-role decision, so it belongs here rather
    than in a Workbench window or live-slot factory.
    """

    capture = sequence.capture_request
    if capture.execution_form is not PulseExecutionForm.STATIC_ONCE:
        raise ValueError("live calibration must use one finite STATIC_ONCE pulse")
    if capture.camera_ref.role != intent.camera_role:
        raise ValueError("calibration capture changed the requested camera role")
    if capture.repeat_count != intent.threshold_frames:
        raise ValueError("calibration capture changed the requested bracket count")
    event_count = capture.readout_events_per_repeat
    if event_count is None:
        raise ValueError("calibration capture has no declared event count")
    layout = sequence.analysis.layout
    role_indices = (
        *layout.reference_event_indices,
        layout.readout_event_index,
    )
    if set(role_indices) != set(range(event_count)):
        raise ValueError(
            "calibration capture events differ from its reference/readout layout"
        )
    expected_grouping = tuple(
        (repeat, event)
        for repeat in range(intent.threshold_frames)
        for event in range(event_count)
    )
    if capture.within_point_grouping != expected_grouping:
        raise ValueError(
            "calibration capture grouping is not repeat-major complete brackets"
        )
    preview_event = layout.reference_event_indices[0]
    ordinals = tuple(
        ordinal
        for ordinal, (_repeat, event) in enumerate(expected_grouping)
        if event == preview_event
    )
    if len(ordinals) != intent.threshold_frames:
        raise RuntimeError("calibration reference preview lost a bracket")
    return ordinals


class PreparedCalibrationTask:
    """Closed one-shot calibration application command.

    Preparation freezes exact physical capture grouping, analysis intent and
    application ports.  The command sequences one capture Run and returns the
    second calibration RunHandle; it is not a third pseudo-Run.
    """

    __slots__ = (
        "_analysis_handle",
        "_capture",
        "_dependencies",
        "_lock",
        "_plan",
        "_preview_ordinals",
        "_post_final_warning",
        "_source_capture_ref",
        "_started",
        "_task_outputs_written",
    )

    def __init__(
        self,
        plan: _CalibrationTaskPlan,
        dependencies: CalibrationTaskApplicationPort,
        *,
        capture: PreparedFiniteCapture,
        preview_ordinals: tuple[int, ...],
    ) -> None:
        if not isinstance(plan, _CalibrationTaskPlan):
            raise TypeError("plan must be a prepared calibration task plan")
        if not isinstance(capture, PreparedFiniteCapture):
            raise TypeError("calibration Task requires a prepared capture")
        if not preview_ordinals:
            raise ValueError("calibration Task requires reference preview ordinals")
        self._plan = plan
        self._dependencies = dependencies
        self._capture = capture
        self._preview_ordinals = preview_ordinals
        self._lock = threading.Lock()
        self._started = False
        self._analysis_handle: RunHandle | None = None
        self._source_capture_ref: CaptureArtifactRef | None = None
        self._task_outputs_written = False
        self._post_final_warning: str | None = None

    @property
    def intent(self) -> CalibrationTaskIntent:
        return self._plan.intent

    def live_dataset_outputs(
        self,
        frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
    ) -> dict[str, LiveDatasetOutput]:
        output = single_live_dataset_output(
            CALIBRATION_LIVE_OUTPUT_DECLARATIONS[0],
            frozen,
        )
        return {output.name: output}

    def start(
        self,
        live_output: CalibrationTaskLiveOutputPort,
        *,
        command_context,
    ) -> RunHandle:
        if not callable(getattr(live_output, "open_live_dataset", None)):
            raise TypeError("Calibration Task requires a live-output port")
        cancel_requested = getattr(command_context, "cancel_requested", None)
        start_and_wait = getattr(command_context, "start_and_wait", None)
        if not callable(cancel_requested) or not callable(start_and_wait):
            raise TypeError("Calibration start requires a hosted command context")
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedCalibrationTask is one-shot")
            self._started = True
        source = start_and_wait(lambda: self._start_capture(live_output))
        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("capture Run returned a non-CaptureArtifactRef")
        with self._lock:
            self._source_capture_ref = source
        # The application/SignalPlane performs the final short admission check
        # against command_context.cancel_requested.  Returning this handle lets
        # HostedRun observe the second flat Run directly instead of inventing a
        # composite lifecycle.
        handle = self._start_calibration_analysis(source)
        with self._lock:
            self._analysis_handle = handle
        return handle

    def _start_capture(
        self,
        live_output: CalibrationTaskLiveOutputPort,
    ) -> RunHandle:
        capture = self._capture
        ordinals = self._preview_ordinals

        def attach(spec: CapturePreviewSpec) -> CapturePreviewPort:
            port = live_output.open_live_dataset(
                spec,
                output_owner=self,
            )
            if port.spec != spec:
                raise ValueError(
                    "calibration live-output port changed the frozen preview spec"
                )
            return port

        return capture.start_with_preview(
            factory=attach,
            source_ordinals=ordinals,
            lifecycle_owner=self,
        )

    def _start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
    ) -> RunHandle:
        handle = self._dependencies.start_calibration_analysis(
            source,
            self._plan.analysis,
            lifecycle_owner=self,
        )
        if not isinstance(handle, RunHandle):
            raise TypeError("calibration application port returned a non-RunHandle")
        return handle

    def _write_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
    ) -> None:
        intent = self._plan.intent
        self._dependencies.write_calibration_task_outputs(
            source,
            calibration,
            folder=intent.folder,
            expected_camera_role=intent.camera_role,
        )

    @property
    def source_capture_ref(self) -> CaptureArtifactRef | None:
        """Exact source retained once the capture/saved branch resolves."""

        with self._lock:
            return self._source_capture_ref

    def _require_own_success(
        self,
        result: CalibrationArtifactRef,
    ) -> CalibrationArtifactRef:
        if not isinstance(result, CalibrationArtifactRef):
            raise TypeError("result must be CalibrationArtifactRef")
        with self._lock:
            handle = self._analysis_handle
        if handle is None:
            raise RuntimeError("calibration Task has no analysis Run")
        successful = handle.result(timeout=0.0)
        if not isinstance(successful, CalibrationArtifactRef):
            raise TypeError("calibration Run returned another FINAL result type")
        if result != successful:
            raise ValueError("calibration result belongs to another prepared task")
        return result

    def final_dataset_outputs(
        self,
        result: CalibrationArtifactRef,
    ) -> dict[str, FinalDatasetOutput]:
        """Materialize this command's complete typed FINAL Dataset vocabulary."""

        reference = self._require_own_success(result)
        computation = self._dependencies.load_calibration_computation(reference)
        from .projection import calibration_final_outputs

        outputs = calibration_final_outputs(computation, reference)
        with self._lock:
            if self._task_outputs_written:
                raise RuntimeError("calibration task outputs were already finalized")
            self._task_outputs_written = True
            source = self._source_capture_ref
        if not isinstance(source, CaptureArtifactRef):
            raise RuntimeError("calibration task lost its exact source capture")
        try:
            self._write_outputs(source, reference)
        except BaseException as error:
            from zlc_neutral_atom.runtime._failure import safe_error_summary

            with self._lock:
                self._post_final_warning = safe_error_summary(error)
        return outputs

    def post_final_warning(self) -> str | None:
        """Operator-bundle failure after the calibration artifact committed."""

        with self._lock:
            return self._post_final_warning

    def site_map_context(
        self,
        result: CalibrationArtifactRef,
    ) -> CalibrationSiteMapContext:
        """Return this command's closed physical SiteMap presentation context."""

        reference = self._require_own_success(result)
        computation = self._dependencies.load_calibration_computation(reference)
        from .projection import calibration_site_map_context

        return calibration_site_map_context(computation, reference)

    def completion_summary(self, result: CalibrationArtifactRef) -> str:
        """Report the committed authority without claiming a report succeeded."""

        self._require_own_success(result)
        root = Path(self._plan.intent.folder)
        return f"calibration artifact committed; requested output: {root}"


def start_calibration_task_command(
    command: PreparedCalibrationTask,
    live_output_host,
    command_context,
):
    """Attach Calibration's live preview and start its two flat Runs."""

    if not isinstance(command, PreparedCalibrationTask):
        raise TypeError("Calibration preparer returned another command type")
    if not callable(getattr(command_context, "start_and_wait", None)):
        raise TypeError("Calibration start requires a hosted command context")
    if not callable(getattr(live_output_host, "open_live_dataset", None)):
        raise TypeError("Calibration start requires a live-output host")
    return command.start(
        live_output_host,
        command_context=command_context,
    )


def prepare_calibration_task(
    intent: CalibrationTaskIntent,
    dependencies: CalibrationTaskApplicationPort,
) -> PreparedCalibrationTask:
    """Bind one complete calibration intent into a closed one-shot command."""

    if not isinstance(intent, CalibrationTaskIntent):
        raise TypeError("intent must be CalibrationTaskIntent")
    sequence = dependencies.sitemap_request(
        frames=intent.threshold_frames,
        camera_role=intent.camera_role,
        pulse=intent.pulse,
        reference_exposure_s=intent.reference_exposure_s,
        readout_exposure_s=intent.readout_exposure_s,
        threshold_method=intent.threshold_method,
        roi_radius=intent.roi_radius,
    )
    if not isinstance(sequence, SitemapCalibrationRequest):
        raise TypeError(
            "calibration application port returned an invalid sitemap request"
        )
    _require_analysis_matches_intent(intent, sequence.analysis)
    preview_ordinals = _reference_preview_ordinals(intent, sequence)
    capture = dependencies.prepare_capture(sequence.capture_request)
    if not isinstance(capture, PreparedFiniteCapture):
        raise TypeError(
            "calibration application port returned an invalid prepared capture"
        )
    descriptor = capture.descriptor
    if descriptor.camera_role != intent.camera_role:
        raise ValueError("prepared calibration capture changed camera role")
    if descriptor.expected_frames != len(
        sequence.capture_request.within_point_grouping or ()
    ):
        raise ValueError("prepared calibration capture changed frame cardinality")
    plan = _CalibrationTaskPlan(
        intent,
        sequence.analysis,
        sequence,
    )
    return PreparedCalibrationTask(
        plan,
        dependencies,
        capture=capture,
        preview_ordinals=preview_ordinals,
    )


__all__ = [
    "CALIBRATION_LIVE_OUTPUT_DECLARATIONS",
    "CALIBRATION_THRESHOLD_METHODS",
    "DEFAULT_CALIBRATION_CAMERA_ROLE",
    "DEFAULT_CALIBRATION_FOLDER",
    "DEFAULT_CALIBRATION_PULSE_PATH",
    "DEFAULT_CALIBRATION_READOUT_EXPOSURE_S",
    "DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S",
    "DEFAULT_CALIBRATION_ROI_RADIUS",
    "DEFAULT_CALIBRATION_THRESHOLD_FRAMES",
    "DEFAULT_CALIBRATION_THRESHOLD_METHOD",
    "MINIMUM_CALIBRATION_ROI_RADIUS",
    "MINIMUM_CALIBRATION_THRESHOLD_FRAMES",
    "CalibrationTaskApplicationPort",
    "CalibrationTaskIntent",
    "CalibrationTaskLiveOutputPort",
    "PreparedCalibrationTask",
    "admit_calibration_task_output",
    "build_calibration_task_intent_from_authoring",
    "calibration_task_authoring_schema",
    "calibration_task_default_camera_role",
    "prepare_calibration_task",
    "write_calibration_task_outputs",
]
