"""Calibration task application orchestration.

This module owns the complete live-or-saved calibration application intent and
its run-like lifecycle.  Desktop frontends may construct the public intent and
bind the required application services, but do not own capture/calibration
sequencing, cancellation, or terminal-state semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Mapping, Protocol

import numpy as np

from zlc_data.value import expand_dataset_validity
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
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
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
from .sitemap import SitemapCalibrationRequest
from zlc_neutral_atom.pulse_catalog import CALIBRATION_PULSE_PATH
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.dataset import (
    DatasetPreviewSnapshot,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.capture.pipeline import (
    CapturePreviewPort,
    CapturePreviewSpec,
)
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

CALIBRATION_SOURCE_MODES = ("live", "saved frames")
CALIBRATION_THRESHOLD_METHODS = tuple(item.value for item in ThresholdMethod)
CALIBRATION_CAPTURE_EXPORT_SCHEMA = "zlc.calibration-task.capture-export"
CALIBRATION_FRAME_EXPORT_POLICIES = ("replace", "remove", "preserve")
CALIBRATION_LIVE_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "frame",
        "zlc_neutral_atom.calibration-task.live-frame",
    ),
)
DEFAULT_CALIBRATION_SOURCE_MODE = "live"
DEFAULT_CALIBRATION_FOLDER = "calibrations"
DEFAULT_CALIBRATION_SAVE_FRAMES = True
DEFAULT_CALIBRATION_PULSE_PATH = CALIBRATION_PULSE_PATH
DEFAULT_CALIBRATION_THRESHOLD_METHOD = "otsu"
DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S = 0.020
DEFAULT_CALIBRATION_READOUT_EXPOSURE_S = 0.005
DEFAULT_CALIBRATION_THRESHOLD_FRAMES = 100
MINIMUM_CALIBRATION_THRESHOLD_FRAMES = 2
DEFAULT_CALIBRATION_ROI_RADIUS = 1
MINIMUM_CALIBRATION_ROI_RADIUS = 1
DEFAULT_CALIBRATION_CAMERA_ROLE = "camera"


def admit_calibration_capture_export(
    source_path: str | Path,
    *,
    expected_camera_role: str,
    capture_repository: CaptureRepository,
) -> CaptureArtifactRef:
    """Validate a saved raw export against its canonical capture authority.

    The exported arrays are an operator copy, never an alternate artifact.
    Admission therefore succeeds only when the referenced capture still exists,
    belongs to the requested camera, and byte-for-byte represents that capture's
    values and validity.  Anonymous image folders cannot manufacture lineage.
    """

    camera_role = normalized_text(expected_camera_role, "expected_camera_role")
    if not isinstance(capture_repository, CaptureRepository):
        raise TypeError("capture_repository must be CaptureRepository")
    folder = Path(source_path).expanduser()
    if not folder.is_absolute():
        raise ValueError("saved calibration source must be resolved by composition")
    folder = folder.resolve()
    metadata_path = folder / "capture.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"saved calibration source has no {metadata_path.name}; "
            "plain frames have no camera/pulse lineage"
        )
    tree = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema",
        "capture_ref",
        "dataset_schema_fingerprint",
        "values_file",
        "validity_file",
    }
    if not isinstance(tree, dict) or set(tree) != expected_fields:
        raise ValueError("saved calibration capture metadata is not current")
    if tree["schema"] != CALIBRATION_CAPTURE_EXPORT_SCHEMA:
        raise ValueError("saved calibration capture metadata has unknown schema")
    if (
        tree["values_file"] != "values.npy"
        or tree["validity_file"] != "validity.npy"
    ):
        raise ValueError("saved calibration capture payload names are not canonical")
    reference = capture_artifact_ref_from_tree(tree["capture_ref"])
    values_path = folder / str(tree["values_file"])
    validity_path = folder / str(tree["validity_file"])
    if not values_path.is_file() or not validity_path.is_file():
        raise FileNotFoundError("saved calibration raw arrays are incomplete")

    admitted = capture_repository.admit(reference)
    actual_role = admitted.artifact.camera_provenance.binding.value
    if actual_role != camera_role:
        raise ValueError(
            "saved calibration capture belongs to camera role "
            f"{actual_role!r}, not {camera_role!r}"
        )
    block = admitted.materialize_snapshot().block
    if tree["dataset_schema_fingerprint"] != block.schema.fingerprint:
        raise ValueError("saved calibration schema differs from its capture authority")
    values = np.load(values_path, allow_pickle=False)
    validity = np.load(validity_path, allow_pickle=False)
    expected_validity = expand_dataset_validity(block.validity, block.schema)
    if (
        values.dtype != block.values.dtype
        or values.shape != block.values.shape
        or not np.array_equal(values, block.values, equal_nan=True)
        or validity.dtype != np.dtype(bool)
        or validity.shape != expected_validity.shape
        or not np.array_equal(validity, expected_validity)
    ):
        raise ValueError(
            "saved calibration raw arrays differ from the lineage-bearing capture"
        )
    return reference


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
    frame_export_policy: str,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
    expected_camera_role: str | None = None,
    render_report: Callable | None = None,
) -> None:
    """Write the complete human result bundle and optional raw capture export.

    Capture and calibration repositories remain the only canonical artifact
    owners.  The task folder contains a typed pointer, a non-authoritative human
    report projection, and an optional reproducible raw export admitted by
    :func:`admit_calibration_capture_export`.  Frontend rendering is an explicit
    composition callback; this module never imports a renderer.
    """

    from .projection import project_calibration_report
    from .result_bundle import write_calibration_result_bundle

    if not isinstance(source, CaptureArtifactRef):
        raise TypeError("source must be CaptureArtifactRef")
    if not isinstance(calibration, CalibrationArtifactRef):
        raise TypeError("calibration must be CalibrationArtifactRef")
    frame_policy = normalized_text(
        frame_export_policy,
        "frame_export_policy",
    ).lower()
    if frame_policy not in CALIBRATION_FRAME_EXPORT_POLICIES:
        raise ValueError(
            "frame_export_policy must be one of "
            f"{CALIBRATION_FRAME_EXPORT_POLICIES}"
        )
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
    frames = root / "frames"
    frames_stage = root / f".frames.{nonce}.tmp"
    frames_backup = root / f".frames.{nonce}.old"
    staged_directories: list[Path] = [report_stage]
    installed: list[tuple[Path, Path | None]] = []
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
        if frame_policy == "replace":
            frames_stage.mkdir()
            staged_directories.append(frames_stage)
            block = admitted.materialize_snapshot().block
            validity = np.asarray(
                expand_dataset_validity(block.validity, block.schema),
                dtype=bool,
            )
            np.save(frames_stage / "values.npy", block.values, allow_pickle=False)
            np.save(frames_stage / "validity.npy", validity, allow_pickle=False)
            (frames_stage / "capture.json").write_text(
                json.dumps(
                    {
                        "schema": CALIBRATION_CAPTURE_EXPORT_SCHEMA,
                        "capture_ref": capture_artifact_ref_to_tree(source),
                        "dataset_schema_fingerprint": block.schema.fingerprint,
                        "values_file": "values.npy",
                        "validity_file": "validity.npy",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        checked_targets = (
            (report, frames)
            if frame_policy != "preserve"
            else (report,)
        )
        for target in checked_targets:
            if target.exists() and (
                target.is_symlink()
                or not target.is_dir()
                or target.resolve().parent != root
            ):
                raise ValueError(
                    f"calibration {target.name} output is not a task-owned directory"
                )

        if report.exists():
            os.replace(report, report_backup)
            report_prior: Path | None = report_backup
        else:
            report_prior = None
        installed.append((report, report_prior))
        os.replace(report_stage, report)
        staged_directories.remove(report_stage)

        if frame_policy != "preserve":
            if frames.exists():
                os.replace(frames, frames_backup)
                frames_prior: Path | None = frames_backup
            else:
                frames_prior = None
            installed.append((frames, frames_prior))
            if frame_policy == "replace":
                os.replace(frames_stage, frames)
                staged_directories.remove(frames_stage)

        # The typed pointer is the whole task output's final commit record.  All
        # human/report/raw payloads are staged and installed before it changes.
        os.replace(pointer_temp, pointer_path)
        committed = True
    finally:
        if not committed:
            for target, prior in reversed(installed):
                if target.exists():
                    shutil.rmtree(target)
                if prior is not None and prior.exists():
                    os.replace(prior, target)
        for staged in staged_directories:
            if staged.exists():
                shutil.rmtree(staged)
        if committed:
            for backup in (report_backup, frames_backup):
                if backup.exists():
                    shutil.rmtree(backup)
        for temporary_path in (pointer_temp,):
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class CalibrationTaskIntent:
    """Complete calibration application intent before service binding.

    Live acquisition and saved-frame calibration are two branches of the same
    task.  Output placement and raw-frame retention belong to this application
    intent rather than to the numeric calibration request.
    """

    source_mode: str
    folder: str
    save_frames: bool
    pulse: str
    threshold_method: str
    reference_exposure_s: float
    readout_exposure_s: float
    threshold_frames: int
    roi_radius: int
    camera_role: str

    def __post_init__(self) -> None:
        source_mode = normalized_text(self.source_mode, "source_mode").lower()
        if source_mode not in CALIBRATION_SOURCE_MODES:
            raise ValueError(
                f"source_mode must be one of {CALIBRATION_SOURCE_MODES}"
            )
        folder = normalized_text(self.folder, "folder")
        if type(self.save_frames) is not bool:
            raise TypeError("save_frames must be bool")
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
        object.__setattr__(self, "source_mode", source_mode)
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
            "source_mode",
            "choice",
            "Source",
            default=DEFAULT_CALIBRATION_SOURCE_MODE,
            required=True,
            choices=tuple(
                AuthoringChoice(value, value) for value in CALIBRATION_SOURCE_MODES
            ),
            description="Acquire live frames now or calibrate saved raw frames.",
        ),
        AuthoringField(
            "folder",
            "path",
            "Folder",
            default=DEFAULT_CALIBRATION_FOLDER,
            required=True,
            description=(
                "The calibration directory: live writes the result and optional "
                "raw frames here; saved frames reads this directory's frames export."
            ),
        ),
        AuthoringField(
            "save_frames",
            "bool",
            "Save frames (live)",
            default=DEFAULT_CALIBRATION_SAVE_FRAMES,
            description=(
                "Keep raw live frames so the same acquisition can be recalibrated."
            ),
        ),
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_CALIBRATION_PULSE_PATH,
            required=True,
            description=(
                "Live only: imaging pulse used for each long-short-long bracket."
            ),
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
                "Live only: positive long exposure for the two outer reference "
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
            description="Live only: positive exposure for the middle readout frame.",
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

    def sitemap_analysis_request(
        self,
        *,
        camera_role: str,
        threshold_method: str,
        roi_radius: int,
    ) -> CalibrationAnalysisRequest: ...

    def prepare_capture(
        self,
        request: CaptureRequest,
    ) -> PreparedFiniteCapture: ...

    def admit_saved_calibration_capture(
        self,
        source_path: Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef: ...

    def start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> RunHandle: ...

    def write_calibration_task_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str,
        frame_export_policy: str,
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
    sequence: SitemapCalibrationRequest | None = None
    source_capture_ref: CaptureArtifactRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intent, CalibrationTaskIntent):
            raise TypeError("intent must be CalibrationTaskIntent")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        if self.intent.source_mode == "live":
            if not isinstance(self.sequence, SitemapCalibrationRequest):
                raise TypeError("live calibration requires SitemapCalibrationRequest")
            if self.sequence.analysis != self.analysis:
                raise ValueError("live calibration sequence differs from frozen analysis")
            if self.source_capture_ref is not None:
                raise ValueError("live calibration cannot preselect a source capture")
        else:
            if not isinstance(self.source_capture_ref, CaptureArtifactRef):
                raise TypeError("saved calibration requires CaptureArtifactRef")
            if self.sequence is not None:
                raise ValueError("saved calibration cannot contain a capture request")


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

    Preparation freezes the live/saved branch, exact physical capture grouping,
    analysis intent and repositories/application port.  Starting it
    requires no callback and exposes no event ordinal to the UI.  The same
    command is also the only result-to-Dataset/presentation context owner.
    """

    __slots__ = (
        "_capture",
        "_dependencies",
        "_lock",
        "_plan",
        "_preview_ordinals",
        "_started",
        "_successful_result",
    )

    def __init__(
        self,
        plan: _CalibrationTaskPlan,
        dependencies: CalibrationTaskApplicationPort,
        *,
        capture: PreparedFiniteCapture | None,
        preview_ordinals: tuple[int, ...] | None,
    ) -> None:
        if not isinstance(plan, _CalibrationTaskPlan):
            raise TypeError("plan must be a prepared calibration task plan")
        if plan.intent.source_mode == "live":
            if not isinstance(capture, PreparedFiniteCapture):
                raise TypeError("live calibration requires a prepared capture")
            if preview_ordinals is None:
                raise TypeError("live calibration requires reference preview ordinals")
        else:
            if capture is not None or preview_ordinals is not None:
                raise ValueError("saved calibration cannot own a live capture preview")
        self._plan = plan
        self._dependencies = dependencies
        self._capture = capture
        self._preview_ordinals = preview_ordinals
        self._lock = threading.Lock()
        self._started = False
        self._successful_result: CalibrationArtifactRef | None = None

    @property
    def intent(self) -> CalibrationTaskIntent:
        return self._plan.intent

    @property
    def has_live_output(self) -> bool:
        return self._capture is not None

    def live_dataset_outputs(
        self,
        frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
    ) -> dict[str, LiveDatasetOutput]:
        if not self.has_live_output:
            raise RuntimeError("saved calibration has no live capture output")
        output = single_live_dataset_output(
            CALIBRATION_LIVE_OUTPUT_DECLARATIONS[0],
            frozen,
        )
        return {output.name: output}

    def start(
        self,
        live_output: CalibrationTaskLiveOutputPort | None = None,
    ) -> CalibrationTaskHandle:
        if live_output is not None and not self.has_live_output:
            raise ValueError("saved calibration cannot attach a live output")
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedCalibrationTask is one-shot")
            self._started = True
        return CalibrationTaskHandle(self, live_output)

    def _start_capture(
        self,
        live_output: CalibrationTaskLiveOutputPort | None,
    ) -> RunHandle:
        capture = self._capture
        if capture is None:
            raise RuntimeError("saved calibration has no capture stage")
        if live_output is None:
            return capture.start()
        ordinals = self._preview_ordinals
        assert ordinals is not None

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
        )

    def _start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
    ) -> RunHandle:
        handle = self._dependencies.start_calibration_analysis(
            source,
            self._plan.analysis,
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
            frame_export_policy=(
                "preserve"
                if intent.source_mode == "saved frames"
                else ("replace" if intent.save_frames else "remove")
            ),
            expected_camera_role=intent.camera_role,
        )

    def _record_success(self, result: CalibrationArtifactRef) -> None:
        if not isinstance(result, CalibrationArtifactRef):
            raise TypeError("calibration result must be CalibrationArtifactRef")
        with self._lock:
            if self._successful_result is not None:
                raise RuntimeError("calibration task success was already recorded")
            self._successful_result = result

    def _require_own_success(
        self,
        result: CalibrationArtifactRef,
    ) -> CalibrationArtifactRef:
        if not isinstance(result, CalibrationArtifactRef):
            raise TypeError("result must be CalibrationArtifactRef")
        with self._lock:
            successful = self._successful_result
        if successful is None:
            raise RuntimeError("calibration task has no successful committed result")
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

        return calibration_final_outputs(computation, reference)

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
        """Expose the exact operator bundle written by this successful task."""

        self._require_own_success(result)
        root = Path(self._plan.intent.folder)
        return f"done; results: {root}; report: {root / 'report'}"


def prepare_calibration_task(
    intent: CalibrationTaskIntent,
    dependencies: CalibrationTaskApplicationPort,
) -> PreparedCalibrationTask:
    """Bind one complete calibration intent into a closed one-shot command."""

    if not isinstance(intent, CalibrationTaskIntent):
        raise TypeError("intent must be CalibrationTaskIntent")
    if intent.source_mode == "live":
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
            sequence=sequence,
        )
        return PreparedCalibrationTask(
            plan,
            dependencies,
            capture=capture,
            preview_ordinals=preview_ordinals,
        )

    source = dependencies.admit_saved_calibration_capture(
        Path(intent.folder) / "frames",
        expected_camera_role=intent.camera_role,
    )
    if not isinstance(source, CaptureArtifactRef):
        raise TypeError(
            "calibration application port returned an invalid saved capture"
        )
    analysis = dependencies.sitemap_analysis_request(
        camera_role=intent.camera_role,
        threshold_method=intent.threshold_method,
        roi_radius=intent.roi_radius,
    )
    _require_analysis_matches_intent(intent, analysis)
    plan = _CalibrationTaskPlan(
        intent,
        analysis,
        source_capture_ref=source,
    )
    return PreparedCalibrationTask(
        plan,
        dependencies,
        capture=None,
        preview_ordinals=None,
    )


class _CancelledBetweenStages(Exception):
    pass


class _ChildEnded(Exception):
    def __init__(self, stage: str, snapshot: RunSnapshot) -> None:
        self.stage = stage
        self.snapshot = snapshot


class CalibrationTaskHandle:
    """Run-like owner of analysis after an optional live capture.

    Live input commits its capture independently; saved input is an already
    admitted exact ``CaptureArtifactRef``.  Both branches enter the same
    analysis run and only that run can produce ``CalibrationArtifactRef``.
    """

    def __init__(
        self,
        prepared: PreparedCalibrationTask,
        live_output: CalibrationTaskLiveOutputPort | None,
    ) -> None:
        if not isinstance(prepared, PreparedCalibrationTask):
            raise TypeError("prepared must be PreparedCalibrationTask")
        self.run_id = RunId(f"calibration-task-{uuid.uuid4().hex}")
        self._prepared = prepared
        self._live_output = live_output
        self._condition = threading.Condition(threading.RLock())
        self._phase = "capture-starting"
        self._active: RunHandle | None = None
        self._stage: str | None = None
        self._cancel_requested = False
        self._cancel_reason = "user requested stop"
        self._machine_committed = False
        self._terminal: RunSnapshot | None = None
        self._result: CalibrationArtifactRef | None = None
        self._source: CaptureArtifactRef | None = None
        self._thread = threading.Thread(
            target=self._coordinate,
            name=f"zlc-calibration-task-{self.run_id.value[-12:]}",
            daemon=False,
        )
        self._thread.start()

    @property
    def source_capture_ref(self) -> CaptureArtifactRef | None:
        with self._condition:
            return self._source

    def _checkpoint(self) -> None:
        with self._condition:
            if self._cancel_requested:
                raise _CancelledBetweenStages

    def _run_child(self, stage: str, handle: RunHandle):
        if not isinstance(handle, RunHandle):
            raise TypeError(f"{stage} starter returned a non-RunHandle")
        with self._condition:
            self._active = handle
            self._stage = stage
            self._phase = f"{stage}-running"
            cancelled = self._cancel_requested
            reason = self._cancel_reason
            self._condition.notify_all()
        if cancelled:
            handle.cancel(reason)
        try:
            result = handle.result()
            if stage == "calibration":
                if not handle.snapshot().final_committed:
                    raise RuntimeError(
                        "calibration analysis succeeded without a FINAL commit"
                    )
                with self._condition:
                    self._machine_committed = True
                    self._condition.notify_all()
            return result
        except (RunCancelled, RunFailed) as error:
            raise _ChildEnded(stage, error.snapshot) from None
        finally:
            with self._condition:
                if self._active is handle:
                    self._active = None
                    self._stage = None

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
                None if child is None else child.commit_publication_warning,
                (
                    error
                    if error is not None
                    else None if child is None else child.primary_error
                ),
                () if child is None else child.cleanup_errors,
                (
                    admission_rejection
                    if admission_rejection is not None
                    else None if child is None else child.admission_rejection
                ),
            )
            self._active = None
            self._stage = None
            self._condition.notify_all()

    def _coordinate(self) -> None:
        try:
            plan = self._prepared._plan
            sequence = plan.sequence
            if sequence is None:
                source = plan.source_capture_ref
                assert isinstance(source, CaptureArtifactRef)
            else:
                source = self._run_child(
                    "capture",
                    self._prepared._start_capture(self._live_output),
                )
                if not isinstance(source, CaptureArtifactRef):
                    raise TypeError("capture Run returned a non-CaptureArtifactRef")
            with self._condition:
                self._source = source
                self._phase = "calibration-preparing"
            self._checkpoint()
            handle = self._prepared._start_calibration_analysis(source)
            result = self._run_child("calibration", handle)
            if not isinstance(result, CalibrationArtifactRef):
                raise TypeError(
                    "calibration Run returned a non-CalibrationArtifactRef"
                )
            self._result = result
            with self._condition:
                self._phase = "writing-task-outputs"
            self._prepared._write_outputs(source, result)
            self._prepared._record_success(result)
            self._finish(
                RunState.SUCCEEDED,
                "calibration-committed",
                child=handle.snapshot(),
            )
        except _CancelledBetweenStages:
            self._finish(RunState.CANCELLED, "cancelled")
        except _ChildEnded as ended:
            source_note = (
                None
                if (
                    ended.snapshot.state is not RunState.FAILED
                    or ended.stage != "calibration"
                    or self._source is None
                )
                else (
                    f"{ended.snapshot.primary_error or 'calibration Run failed'}; "
                    f"source capture remains {self._source!r}"
                )
            )
            self._finish(
                ended.snapshot.state,
                "cancelled"
                if ended.snapshot.state is RunState.CANCELLED
                else "failed",
                child=ended.snapshot,
                error=source_note,
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
                source = self._source
            failure = safe_error_summary(error)
            if source is not None:
                failure += f"; source capture remains {source!r}"
            if self._result is not None:
                failure += f"; calibration remains {self._result!r}"
            self._finish(
                RunState.CANCELLED if cancelled else RunState.FAILED,
                "cancelled" if cancelled else "failed",
                error=None if cancelled else failure,
            )

    def snapshot(self) -> RunSnapshot:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            active = self._active
            stage = self._stage
            phase = self._phase
            cancelling = self._cancel_requested
        if active is None:
            child = None
        else:
            child = active.snapshot()
            phase = f"{stage}/{child.phase}"
        return RunSnapshot(
            self.run_id,
            RunState.CANCELLING if cancelling else RunState.RUNNING,
            phase,
            bool(
                self._machine_committed
                or (
                    child is not None
                    and stage == "calibration"
                    and child.final_committed
                )
            ),
            None if child is None else child.commit_publication_warning,
            None if child is None else child.primary_error,
            () if child is None else child.cleanup_errors,
            None if child is None else child.admission_rejection,
        )

    def cancel(self, reason: str = "user requested stop") -> CancelOutcome:
        text = canonical_text(reason, "cancellation reason")
        with self._condition:
            if self._terminal is not None:
                return CancelOutcome.ALREADY_TERMINAL
            if self._machine_committed:
                return CancelOutcome.TOO_LATE_ALREADY_COMMITTED
            if self._cancel_requested:
                return CancelOutcome.ALREADY_REQUESTED
            active = self._active
            if active is None:
                self._cancel_requested = True
                self._cancel_reason = text
                self._condition.notify_all()
                return CancelOutcome.REQUESTED

        outcome = active.cancel(text)
        if outcome in (
            CancelOutcome.TOO_LATE_ALREADY_COMMITTED,
            CancelOutcome.TOO_LATE_FINALIZING,
            CancelOutcome.ALREADY_TERMINAL,
        ):
            return outcome
        with self._condition:
            if self._terminal is not None:
                return CancelOutcome.ALREADY_TERMINAL
            if self._machine_committed:
                return CancelOutcome.TOO_LATE_ALREADY_COMMITTED
            if self._cancel_requested:
                return CancelOutcome.ALREADY_REQUESTED
            self._cancel_requested = True
            self._cancel_reason = text
            self._condition.notify_all()
        return outcome

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
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"calibration task {self.run_id} is active")
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
                f"calibration task {self.run_id} is terminal but not reaped"
            )
        return snapshot

    def result(self, timeout: float | None = None) -> CalibrationArtifactRef:
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED:
            assert self._result is not None
            return self._result
        if snapshot.state is RunState.CANCELLED:
            raise RunCancelled(snapshot)
        raise RunFailed(snapshot)


__all__ = [
    "CALIBRATION_CAPTURE_EXPORT_SCHEMA",
    "CALIBRATION_FRAME_EXPORT_POLICIES",
    "CALIBRATION_LIVE_OUTPUT_DECLARATIONS",
    "CALIBRATION_SOURCE_MODES",
    "CALIBRATION_THRESHOLD_METHODS",
    "DEFAULT_CALIBRATION_CAMERA_ROLE",
    "DEFAULT_CALIBRATION_FOLDER",
    "DEFAULT_CALIBRATION_PULSE_PATH",
    "DEFAULT_CALIBRATION_READOUT_EXPOSURE_S",
    "DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S",
    "DEFAULT_CALIBRATION_ROI_RADIUS",
    "DEFAULT_CALIBRATION_SAVE_FRAMES",
    "DEFAULT_CALIBRATION_SOURCE_MODE",
    "DEFAULT_CALIBRATION_THRESHOLD_FRAMES",
    "DEFAULT_CALIBRATION_THRESHOLD_METHOD",
    "MINIMUM_CALIBRATION_ROI_RADIUS",
    "MINIMUM_CALIBRATION_THRESHOLD_FRAMES",
    "CalibrationTaskApplicationPort",
    "CalibrationTaskHandle",
    "CalibrationTaskIntent",
    "CalibrationTaskLiveOutputPort",
    "PreparedCalibrationTask",
    "admit_calibration_capture_export",
    "admit_calibration_task_output",
    "build_calibration_task_intent_from_authoring",
    "calibration_task_authoring_schema",
    "calibration_task_default_camera_role",
    "prepare_calibration_task",
    "write_calibration_task_outputs",
]
