"""MOT-field task preparation plus live and FINAL domain projection.

A form boundary constructs :class:`MotFieldTaskIntent`; the composition root
passes its installation-bound semantic service to :func:`prepare_mot_field_task`.
The returned command is the complete application.  A desktop may attach its
typed live output, start/cancel/observe the command and route its named FINAL
datasets, but never assembles the scan, projection, analysis or materializer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from zlc_data import DatasetSchema
from zlc_neutral_atom.capture.artifact import CaptureArtifact
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
    TaskPreviewPlot,
)
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
    _MOT_SCAN_COORDINATE_IDS,
    MotFieldAcquisitionResult,
    MotFieldRequest,
    MotFieldResult,
    analyze_mot_scan,
    mot_field_final_outputs,
    mot_intensity_schema,
)
from .mot_field_live import (
    MOT_FIELD_LIVE_OUTPUT_DECLARATIONS,
    MotFieldLiveProjection,
)
from .application import (
    PreparedMotFieldAcquisition,
)
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
)
from zlc_storage import (
    canonical_text,
    finite_real,
    integer,
    normalized_text,
    positive_real,
)
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import FacetGridPlot, ImagePlot


DEFAULT_MOT_FIELD_PULSE_PATH = "mot_field_template.json"

_MOT_FIELD_GRID_AXES = tuple(
    AxisRef.point_dimension(axis_id.value) for axis_id in _MOT_SCAN_COORDINATE_IDS
)
_MOT_FIELD_PREVIEW_PLOT = FacetGridPlot(
    _MOT_FIELD_GRID_AXES[2],
    ImagePlot(_MOT_FIELD_GRID_AXES[0], _MOT_FIELD_GRID_AXES[1]),
)


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
    task_previews=(
        TaskPreviewPlot("grid", _MOT_FIELD_PREVIEW_PLOT),
        TaskPreviewPlot("mot_field", _MOT_FIELD_PREVIEW_PLOT),
    ),
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
    resolve_dynamic_choices=_mot_camera_choices,
)


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


def _acquisition_from_capture(
    request: MotFieldRequest,
    artifact: CaptureArtifact,
) -> MotFieldAcquisitionResult:
    """Validate one visible direct-output Capture as exact MOT source data."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError("MOT analysis requires CaptureArtifact")
    evidence = artifact.pulse_evidence
    if evidence is None:
        raise ValueError("MOT analysis requires pulse-associated Capture evidence")
    if (
        request.trigger_channel is not None
        and evidence.trigger_channel != request.trigger_channel
    ):
        raise ValueError("MOT trigger lineage differs from the explicit request")
    snapshot = artifact.materialize_snapshot()
    mot_intensity_schema(request, snapshot.block.schema)
    return MotFieldAcquisitionResult(
        snapshot,
        artifact.provenance,
    )


class PreparedMotFieldTask:
    """Stateless MOT projection over one generic direct-output Capture.

    ``mot_field_result()`` exposes the refined optimum; named Dataset outputs
    expose the complete intensity grid and its raw source scan.
    """

    __slots__ = (
        "_acquisition",
        "_intent",
        "_output_schema",
        "_request",
        "_source_schema",
    )

    def __init__(
        self,
        intent: MotFieldTaskIntent,
        request: MotFieldRequest,
        acquisition: PreparedMotFieldAcquisition,
    ) -> None:
        if not isinstance(intent, MotFieldTaskIntent):
            raise TypeError("intent must be MotFieldTaskIntent")
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(acquisition, PreparedMotFieldAcquisition):
            raise TypeError("acquisition must be PreparedMotFieldAcquisition")
        if acquisition.request != request:
            raise ValueError("MOT acquisition belongs to another request")
        source_schema = acquisition.source_schema
        self._intent = intent
        self._request = request
        self._acquisition = acquisition
        self._source_schema = source_schema
        self._output_schema = mot_intensity_schema(request, source_schema)

    @property
    def intent(self) -> MotFieldTaskIntent:
        return self._intent

    @property
    def request(self) -> MotFieldRequest:
        return self._request

    @property
    def output_schema(self) -> DatasetSchema:
        """The generation-static scalar Bx/By/Bz schema shared by live and FINAL."""

        return self._output_schema

    @property
    def preview_spec(self) -> ExactDatasetPreviewSpec:
        return ExactDatasetPreviewSpec(self._source_schema.fingerprint)

    def live_projection(self) -> MotFieldLiveProjection:
        return MotFieldLiveProjection(
            self._request,
            self._source_schema,
            self._output_schema,
        )

    def start(
        self,
        preview: ExactDatasetPreviewPort | None = None,
    ) -> RunHandle:
        return self._acquisition.start(preview)

    def final_dataset_outputs(
        self,
        reference: CaptureArtifactRef,
    ) -> dict[str, FinalDatasetOutput]:
        """Statelessly analyze one visible Capture into named typed outputs."""

        source, analysis = self._analyze_final_capture(reference)
        return mot_field_final_outputs(
            analysis,
            source,
            self._output_schema,
        )

    def mot_field_result(
        self,
        reference: CaptureArtifactRef,
    ) -> MotFieldResult:
        """Return the refined field and intensity from one FINAL Capture."""

        _source, analysis = self._analyze_final_capture(reference)
        return analysis

    def _analyze_final_capture(
        self,
        reference: CaptureArtifactRef,
    ) -> tuple[MotFieldAcquisitionResult, MotFieldResult]:
        artifact = self._acquisition.load_capture(reference)
        source = _acquisition_from_capture(self._request, artifact)
        analysis = analyze_mot_scan(self._request, source)
        return source, analysis

    def completion_summary(self, reference: CaptureArtifactRef) -> str:
        """Name the generic direct-output Capture that made the task FINAL."""

        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("MOT FINAL result must be CaptureArtifactRef")
        return f"done; capture: {reference.target_ref}"


def start_mot_field_task_command(
    command: PreparedMotFieldTask,
    live_output_host,
    _command_context,
):
    """Attach MOT's declared live source before its one physical start."""

    if not isinstance(command, PreparedMotFieldTask):
        raise TypeError("MOT-field preparer returned another command type")
    open_exact_dataset = getattr(live_output_host, "open_exact_dataset", None)
    if not callable(open_exact_dataset):
        raise TypeError("MOT-field start requires an exact Dataset host")
    preview = open_exact_dataset(
        command.preview_spec,
        projection=command.live_projection(),
    )
    return command.start(preview)


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
    return PreparedMotFieldTask(
        intent,
        request,
        acquisition,
    )


__all__ = [
    "DEFAULT_MOT_FIELD_PULSE_PATH",
    "MOT_FIELD_LOGIC_NODE",
    "MotFieldTaskDependencies",
    "MotFieldTaskIntent",
    "PreparedMotFieldTask",
    "build_mot_field_intent_from_authoring",
    "mot_field_authoring_schema",
    "mot_field_camera_roles",
    "prepare_mot_field_task",
]
