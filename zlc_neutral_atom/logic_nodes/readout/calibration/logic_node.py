"""The sole discovered declaration and operation for Calibration."""

from __future__ import annotations

from pathlib import Path

from zlc_data import BlockId
from zlc_neutral_atom.capture.application import bind_finite_capture_request
from zlc_neutral_atom.capture.artifact import compile_capture_artifact_pipeline
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_CAPTURE,
    CAPABILITY_PULSE_EXECUTE,
)
from zlc_neutral_atom.logic_node import (
    ArtifactOutputSpec,
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
    TaskPreview,
    UiContribution,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeExecutionContext
from zlc_neutral_atom.runtime.preview import ExactDatasetPreviewSpec
from zlc_plot.kinds import PlotKind
from zlc_pulse import load_pulse_document
from zlc_storage.paths import resolve_under

from .analysis import CalibrationComputation
from .artifact import (
    CommittedCalibration,
    compile_calibration_analysis_plan,
    load_calibration_artifact,
)
from .installation import build_sitemap_acquisition_profile
from .outputs import (
    CALIBRATION_ARTIFACT_OUTPUT_DECLARATION,
    CALIBRATION_CAPTURE_PREVIEW_DECLARATION,
    CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
    calibration_final_outputs,
)
from .preview import CalibrationCapturePreview
from .reference import CalibrationArtifactRef
from .sitemap import SitemapCalibrationRequest, build_sitemap_calibration_request
from .task import (
    CalibrationTaskRequest,
    build_calibration_task_request,
    calibration_task_authoring_schema,
    write_calibration_post_final_exports,
)


_OUTPUT_LABELS = {
    "capture_preview": ("capture preview", "Raw camera frame"),
    "site_map": ("site map", "Counts"),
    "fidelity_site": ("site fidelity", "Readout fidelity"),
    "fidelity_threshold": ("site threshold", "Readout threshold"),
    "fidelity_centers": ("site centres", "Site centre"),
    "readout_samples": ("readout samples", "Readout signal"),
    "aggregate_fidelity": ("aggregate fidelity", "Aggregate fidelity"),
    "global_fidelity": ("global fidelity", "Global fidelity"),
}
_CALIBRATION_ARTIFACT_CONTRACT = (
    CALIBRATION_ARTIFACT_OUTPUT_DECLARATION.contract_id
)
_REPORT_UI = UiContribution(
    "report",
    "zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_window",
    "CalibrationReportWindow",
)
_CALIBRATION_DEFINITION = LogicNodeDefinition(
    DefinitionKey(
        "zlc_neutral_atom.logic_nodes.readout.calibration",
        "calibrate-readout",
    ),
    "Calibrate readout",
    "task",
)


def _readout_binding_for(
    request: CalibrationTaskRequest,
    context: LogicNodeApplicationContext,
):
    matches = tuple(
        value
        for value in context.readout_bindings
        if value.camera_instance_id == request.camera_instance_id
        and value.sequencer_instance_id == request.sequencer_instance_id
    )
    if len(matches) != 1:
        raise ValueError(
            "Calibration requires one installed readout binding matching "
            "the selected camera and sequencer instances"
        )
    return matches[0]


def _sequence_for(
    request: CalibrationTaskRequest,
    context: LogicNodeApplicationContext,
) -> tuple[SitemapCalibrationRequest, object, object, object]:
    camera = context.device("camera_instance_id")
    sequencer = context.device("sequencer_instance_id")
    catalog = context.device_catalog
    camera_ref = catalog.require(request.camera_instance_id).ref
    sequencer_ref = catalog.require(request.sequencer_instance_id).ref
    profile = build_sitemap_acquisition_profile(
        _readout_binding_for(request, context),
        grid_shape_yx=request.grid_shape_yx,
        camera_ref=camera_ref,
        sequencer_ref=sequencer_ref,
        camera_port=camera,
        pulse_port=sequencer,
    )
    pulse = load_pulse_document(resolve_under(context.pulses_root, request.pulse))
    sequence = build_sitemap_calibration_request(
        profile,
        camera_ref=camera_ref,
        sequencer_ref=sequencer_ref,
        repeat_groups=request.threshold_frames,
        pulse_document=pulse,
        reference_exposure_s=request.reference_exposure_s,
        readout_exposure_s=request.readout_exposure_s,
        threshold_method=request.threshold_method,
        roi_radius=request.roi_radius,
    )
    return sequence, camera, sequencer, profile.readout_binding


def _capture_plan(
    sequence: SitemapCalibrationRequest,
    *,
    camera,
    sequencer,
    project_root: Path,
    exact_preview=None,
):
    binding = bind_finite_capture_request(
        sequence.capture_request,
        pulse_port=sequencer,
        camera_port=camera,
    )
    pipeline = MinimalPipelineSpec(
        "Calibration capture",
        binding.capture,
        BlockId("calibration-capture"),
    )
    triggered = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    return compile_capture_artifact_pipeline(
        triggered,
        project_root,
        exact_preview=exact_preview,
    )


def _capture_source_schema(
    sequence: SitemapCalibrationRequest,
    *,
    camera,
    sequencer,
):
    """Resolve the capture schema without opening hardware.

    The preview projection needs the capability-owned frame ``ValueSchema``;
    binding it here keeps the preview contract identical to the actual capture
    plan while leaving all device I/O inside the RunPlan.
    """

    binding = bind_finite_capture_request(
        sequence.capture_request,
        pulse_port=sequencer,
        camera_port=camera,
    )
    return binding.capture.capture_contract.dataset_schema


def _bind_execute(
    request: object,
    context: LogicNodeApplicationContext,
):
    if not isinstance(request, CalibrationTaskRequest):
        raise TypeError("Calibration request must be CalibrationTaskRequest")
    sequence, camera, sequencer, readout_binding = _sequence_for(request, context)
    project_root = context.project_root
    source_schema = _capture_source_schema(
        sequence,
        camera=camera,
        sequencer=sequencer,
    )

    def execute(execution: LogicNodeExecutionContext):
        preview = execution.open_exact_dataset(
            ExactDatasetPreviewSpec(source_schema.fingerprint),
            projection=CalibrationCapturePreview(
                source_schema,
                CALIBRATION_CAPTURE_PREVIEW_DECLARATION,
            ),
        )
        capture_plan = _capture_plan(
            sequence,
            camera=camera,
            sequencer=sequencer,
            project_root=project_root,
            exact_preview=preview,
        )
        source = execution.start_and_wait(
            lambda: context.start_run(
                capture_plan.with_lifecycle(execution, preemptible=False)
            )
        )
        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("Calibration Capture Run returned another result type")
        analysis_plan = compile_calibration_analysis_plan(
            source,
            project_root,
            sequence.analysis,
            expected_readout_binding=readout_binding,
            timeout_seconds=300.0,
        )
        committed = execution.start_and_wait(
            lambda: context.start_run(
                analysis_plan.with_lifecycle(execution, preemptible=False)
            )
        )
        if not isinstance(committed, CommittedCalibration):
            raise TypeError("Calibration Analysis Run returned another result type")
        computation = CalibrationComputation(
            committed.result.artifact,
            committed.result.report,
        )
        outputs = calibration_final_outputs(computation, committed.reference)
        execution.publish_final(outputs)
        context.remember_artifact(
            _CALIBRATION_ARTIFACT_CONTRACT,
            committed.reference,
        )
        from .ui.plot_report import export_calibration_plot_pages

        write_calibration_post_final_exports(
            committed.result,
            committed.reference,
            project_root=project_root,
            export_plots=export_calibration_plot_pages,
            save_frames=request.save_frames,
            warn=execution.warn,
        )
        return committed.reference

    return execute


def _load(
    context: LogicNodeApplicationContext,
    reference: CalibrationArtifactRef | None = None,
):
    selected = reference
    if selected is None:
        selected = context.default_artifact(_CALIBRATION_ARTIFACT_CONTRACT)
    if not isinstance(selected, CalibrationArtifactRef):
        raise ValueError("no Calibration artifact is selected")
    return load_calibration_artifact(
        context.project_root,
        selected,
    )


def _sitemap(
    context: LogicNodeApplicationContext,
    reference: CalibrationArtifactRef | None = None,
):
    return _load(context, reference).artifact.site_map


def _report(
    context: LogicNodeApplicationContext,
    reference: CalibrationArtifactRef | None = None,
):
    resolved = _load(context, reference)
    root = resolve_under(
        context.project_root,
        resolved.reference.record_path,
    ).parent / "report"
    return context.open_ui(_REPORT_UI, root, resolved.reference)


LOGIC_NODE = LogicNodeDescriptor(
    api_name="calibration",
    definition=_CALIBRATION_DEFINITION,
    description="Acquire reference/readout frames and commit a Calibration",
    authoring_schema=calibration_task_authoring_schema(),
    input_specs=(),
    outputs=(
        DatasetOutputSpec(
            CALIBRATION_CAPTURE_PREVIEW_DECLARATION,
            _OUTPUT_LABELS[CALIBRATION_CAPTURE_PREVIEW_DECLARATION.name][0],
            _OUTPUT_LABELS[CALIBRATION_CAPTURE_PREVIEW_DECLARATION.name][1],
        ),
        *tuple(
            DatasetOutputSpec(
                declaration,
                _OUTPUT_LABELS[declaration.name][0],
                _OUTPUT_LABELS[declaration.name][1],
            )
            for declaration in CALIBRATION_FINAL_OUTPUT_DECLARATIONS
        ),
    ),
    artifact_outputs=(
        ArtifactOutputSpec(
            CALIBRATION_ARTIFACT_OUTPUT_DECLARATION,
            "calibration",
            "FINAL runtime readout calibration",
        ),
    ),
    build_request=build_calibration_task_request,
    bind_execute=_bind_execute,
    device_requirements=(
        ("camera_instance_id", (CAPABILITY_CAMERA_CAPTURE,)),
        ("sequencer_instance_id", (CAPABILITY_PULSE_EXECUTE,)),
    ),
    task_previews=(TaskPreview("capture_preview", PlotKind.IMAGE),),
    ui_contributions=(_REPORT_UI,),
    operations={"load": _load, "sitemap": _sitemap, "report": _report},
)


__all__ = ["LOGIC_NODE"]
