"""Optimize the three MOT coil fields in one autonomous hardware scan."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_data import AxisId, AxisSpec, BlockId, REPEAT
from zlc_neutral_atom.authoring import (
    AuthoringField,
    AuthoringSchema,
    MINIMUM_POSITIVE_FLOAT,
)
from zlc_neutral_atom.capture.artifact import (
    compile_capture_artifact_pipeline,
    load_capture_artifact,
)
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
)
from zlc_neutral_atom.device_types import (
    CAPABILITY_MOT_FIELD_CAPTURE,
    CAPABILITY_PULSE_EXECUTE,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.logic_node import (
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
    TaskPreview,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeExecutionContext
from zlc_neutral_atom.runtime.preview import ExactDatasetPreviewSpec
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import FacetGridPlot, ImagePlot
from zlc_pulse import PORT_DIGITAL, PulseExecutionForm, load_pulse_document
from zlc_storage.paths import resolve_under

from .mot_field import (
    DEFAULT_MOT_FIELD_CENTER_CODE,
    DEFAULT_MOT_FIELD_POINTS,
    DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
    DEFAULT_MOT_FIELD_SPAN_CODE,
    MINIMUM_MOT_FIELD_POINTS,
    MotFieldAcquisitionResult,
    MotFieldRequest,
    MotFieldResult,
    _MOT_SCAN_COORDINATE_IDS,
    analyze_mot_scan,
    build_mot_scan_program,
    materialize_mot_field_snapshot,
    mot_intensity_schema,
)
from .mot_field_live import MotFieldLiveProjection


DEFAULT_MOT_FIELD_PULSE_PATH = "mot_field_template.json"


def _pulse_trigger_channel(document, label: str = "trig") -> str:
    matches = tuple(
        port
        for port in document.target.ports
        if port.kind == PORT_DIGITAL and port.label == label and len(port.lanes) == 1
    )
    if len(matches) != 1:
        raise ValueError(
            "MOT-field pulse must expose exactly one single-lane trig endpoint"
        )
    return matches[0].lanes[0]

_MOT_REPEAT_AXIS = AxisSpec(
    AxisId("mot-field.repeat"),
    "repeat",
    REPEAT,
    1,
    (0,),
)
_MOT_READOUT_EVENT_AXIS_ID = AxisId("mot-field.readout-event")
_GRID_OUTPUT = DatasetOutputDeclaration(
    "grid",
    "zlc_neutral_atom.mot-field.live-grid",
)
_FIELD_OUTPUT = DatasetOutputDeclaration(
    "mot_field",
    "zlc_neutral_atom.mot-field.result",
)
_SCAN_OUTPUT = DatasetOutputDeclaration(
    "scan",
    "zlc_neutral_atom.mot-field.source-scan",
)

_GRID_AXES = tuple(
    AxisRef.point_dimension(axis_id.value) for axis_id in _MOT_SCAN_COORDINATE_IDS
)
_GRID_PLOT = FacetGridPlot(
    _GRID_AXES[2],
    ImagePlot(_GRID_AXES[0], _GRID_AXES[1]),
)

_AUTHORING_SCHEMA = AuthoringSchema(
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
            description="Blank uses the frame centre",
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
            description="Blank uses the frame centre",
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
            "camera_instance_id",
            "choice",
            "MOT camera",
            required=True,
            dynamic_choices=True,
            description="Camera providing MOT-field triggered capture",
        ),
        AuthoringField(
            "sequencer_instance_id",
            "choice",
            "Pulse sequencer",
            required=True,
            dynamic_choices=True,
            description="Sequencer executing the autonomous scan",
        ),
    )
)


def _build_request(values: Mapping[str, object]) -> MotFieldRequest:
    authored = _AUTHORING_SCHEMA.freeze(values)
    if authored["camera_instance_id"] is None:
        raise ValueError("select a MOT-field camera")
    if authored["sequencer_instance_id"] is None:
        raise ValueError("select a pulse sequencer")
    return MotFieldRequest(**authored)  # type: ignore[arg-type]


def _bind_execute(
    request: object,
    context: LogicNodeApplicationContext,
):
    if not isinstance(request, MotFieldRequest):
        raise TypeError("MOT Field requires MotFieldRequest")
    camera_port = context.device("camera_instance_id")
    pulse_port = context.device("sequencer_instance_id")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("MOT camera capability returned no BoundCapturePort")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse capability returned no BoundPulsePort")

    document = load_pulse_document(resolve_under(context.pulses_root, request.pulse))
    program = build_mot_scan_program(
        document,
        center_x=request.center_x,
        center_y=request.center_y,
        center_z=request.center_z,
        span=request.span,
        points=request.points,
    )
    binding = bind_triggered_camera_acquisition(
        pulse_port,
        camera_port,
        camera_instance_id=request.camera_instance_id,
        pulse_document=program.document,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channel=_pulse_trigger_channel(program.document),
        layout=TriggeredCameraLayout(
            repeat_axis=_MOT_REPEAT_AXIS,
            readout_event_axis_id=_MOT_READOUT_EVENT_AXIS_ID,
            readout_events_per_repeat=1,
            scan_point_table=program.point_table,
            scan_grid_topology=program.grid_topology,
        ),
    )
    if binding.expected_frames != program.point_table.row_count:
        raise RuntimeError("MOT pulse trigger count differs from its frozen grid")
    source_schema = binding.capture.capture_contract.dataset_schema
    output_schema = mot_intensity_schema(program, source_schema)
    pipeline = MinimalPipelineSpec(
        f"Optimize MOT field {program.document.name}",
        binding.capture,
        BlockId(
            f"mot-field-source-{binding.compiled_artifact.fingerprint[:20]}"
        ),
    )
    capture = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    project_root = context.project_root

    def execute(execution_context: LogicNodeExecutionContext) -> MotFieldResult:
        projection = MotFieldLiveProjection(
            request,
            program,
            source_schema,
            output_schema,
            _GRID_OUTPUT,
        )
        preview = execution_context.open_exact_dataset(
            ExactDatasetPreviewSpec(source_schema.fingerprint),
            projection=projection,
        )
        plan = compile_capture_artifact_pipeline(
            capture,
            project_root,
            exact_preview=preview,
        ).with_lifecycle(execution_context, preemptible=False)
        reference = execution_context.start_and_wait(
            lambda: context.start_run(plan)
        )
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("MOT capture returned no CaptureArtifactRef")
        artifact = load_capture_artifact(
            project_root,
            reference,
            materialize=True,
        )
        source = MotFieldAcquisitionResult(
            artifact.materialize_snapshot(),
            artifact.provenance,
        )
        result = analyze_mot_scan(request, program, source)
        if source.snapshot.ref != result.source_ref:
            raise RuntimeError("MOT result belongs to another source scan")
        execution_context.publish_final(
            {
                _FIELD_OUTPUT.name: FinalDatasetOutput(
                    _FIELD_OUTPUT,
                    materialize_mot_field_snapshot(result, output_schema),
                ),
                _SCAN_OUTPUT.name: FinalDatasetOutput(
                    _SCAN_OUTPUT,
                    source.snapshot,
                ),
            }
        )
        return result

    return execute


LOGIC_NODE = LogicNodeDescriptor(
    api_name="mot_field",
    definition=LogicNodeDefinition(
        DefinitionKey(
            "zlc_neutral_atom.logic_nodes.mot_field",
            "optimize-mot-field",
        ),
        "Optimize MOT field",
        "task",
    ),
    description=(
        "Sweep da_x/da_y/da_z in one autonomous hardware scan, measure MOT "
        "fluorescence, and report the refined optimum"
    ),
    authoring_schema=_AUTHORING_SCHEMA,
    input_specs=(),
    outputs=(
        DatasetOutputSpec(
            _GRID_OUTPUT,
            "MOT intensity grid",
            "Counts",
            "Provisional Bx/By/Bz intensity while the scan runs",
        ),
        DatasetOutputSpec(
            _FIELD_OUTPUT,
            "MOT field",
            "Counts",
            "FINAL optimum and complete three-dimensional intensity grid",
        ),
        DatasetOutputSpec(
            _SCAN_OUTPUT,
            "Scan",
            "Signal",
            "Exact raw Camera source scan",
        ),
    ),
    build_request=_build_request,
    bind_execute=_bind_execute,
    device_requirements=(
        ("camera_instance_id", (CAPABILITY_MOT_FIELD_CAPTURE,)),
        ("sequencer_instance_id", (CAPABILITY_PULSE_EXECUTE,)),
    ),
    task_previews=(
        TaskPreview("grid", _GRID_PLOT),
        TaskPreview("mot_field", _GRID_PLOT),
    ),
)


__all__ = ["LOGIC_NODE"]
