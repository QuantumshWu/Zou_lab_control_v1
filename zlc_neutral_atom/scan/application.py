"""One-Run application boundary for canonical pulse-scan artifacts."""

from __future__ import annotations

from dataclasses import replace
import threading
from typing import Callable

from zlc_data import (
    REPEAT,
    AxisId,
    AxisSpec,
    BlockId,
    DataTransformSpec,
    DatasetSchema,
    OwnedSnapshot,
    Selection,
    commit_transform,
    materialize_transformed_snapshot,
)
from zlc_neutral_atom.bootstrap._triggered_capture import (
    ApiSlotSegmentedCameraBinding,
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_api_slot_segmented_camera_acquisition,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.capture_application import (
    PlanDescriptor,
    bind_finite_capture_spec,
)
from zlc_neutral_atom.occupancy_application import bind_occupancy_pipeline
from zlc_neutral_atom.readout.calibration import ResolvedCalibration
from zlc_neutral_atom.readout.occupancy import resolve_occupancy_stream_schema
from zlc_neutral_atom.readout.occupancy_pipeline import (
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    _occupancy_preview_spec,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.dataset import (
    DatasetPreviewSnapshot,
    DatasetSealProvenance,
)
from zlc_neutral_atom.runtime.pipeline import (
    ExactDatasetPreviewSpec,
    ExactDatasetPreviewPort,
    MinimalPipelineSpec,
    PipelineResult,
    _notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunHandle,
    RunPlan,
)
from zlc_neutral_atom.timing.capture import (
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_neutral_atom.timing.occupancy import (
    TriggeredOccupancyPipelineResult,
    TriggeredOccupancySpec,
    compile_triggered_occupancy_pipeline,
)
from zlc_neutral_atom.timing.segmented import (
    ApiSlotSegmentedResult,
    ApiSlotSegmentedSpec,
    compile_api_slot_segmented_occupancy_pipeline,
    compile_api_slot_segmented_pipeline,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    bind_pulse_document_target,
    expand_autonomous_scan_repeats,
    require_autonomous_scan_resident_capacity,
)
from zlc_storage import RepositoryRootLeaseBorrow, canonical_digest

from .contracts import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseScanProgram,
    ScanOutputContract,
    ScanPointTable,
    bind_scan_output_contract,
)
from .lineage import (
    ApiSegmentEvidence,
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    camera_run_evidence_from_pipeline,
)
from .reference import ScanArtifactRef
from .final_output import scan_final_outputs
from .repository import (
    ScanRepository,
    _PreparedScanDataset,
    _SCAN_APPLICATION_TOKEN,
    _StagedScanLineage,
    _scan_output_dataset_ref,
)
from .source_binding import OccupancyScanRequest, ScanRequest


_ResultAdapter = Callable[
    [object],
    tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseScanExecution],
]


_SCAN_REPEAT_AXIS_ID = AxisId("scan.repeat")
_SCAN_READOUT_EVENT_AXIS_ID = AxisId("scan.readout_event")
_ScanRequest = ScanRequest | OccupancyScanRequest
_ScanCameraBinding = TriggeredCameraBinding | ApiSlotSegmentedCameraBinding


class PreparedExactScan:
    """One-shot exact scan command exposing one typed provisional dataset seam."""

    __slots__ = (
        "_descriptor",
        "_lock",
        "_output_contract",
        "_program",
        "_repository",
        "_source_schema",
        "_start",
        "_started",
    )

    def __init__(
        self,
        *,
        program: PulseScanProgram,
        source_schema: DatasetSchema,
        output_contract: ScanOutputContract,
        descriptor: PlanDescriptor,
        repository: ScanRepository,
        start: Callable[[ExactDatasetPreviewPort | None], RunHandle],
    ) -> None:
        if not isinstance(
            program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a current pulse-scan program")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(descriptor, PlanDescriptor):
            raise TypeError("descriptor must be PlanDescriptor")
        if type(repository) is not ScanRepository:
            raise TypeError("repository must be ScanRepository")
        if not callable(start):
            raise TypeError("start must be callable")
        _require_output_binding(
            program=program,
            source_schema=source_schema,
            output_contract=output_contract,
        )
        self._program = program
        self._source_schema = source_schema
        self._output_contract = output_contract
        self._descriptor = descriptor
        self._repository = repository
        self._start = start
        self._lock = threading.Lock()
        self._started = False

    @property
    def source_schema(self) -> DatasetSchema:
        return self._source_schema

    @property
    def output_contract(self) -> ScanOutputContract:
        return self._output_contract

    @property
    def descriptor(self) -> PlanDescriptor:
        return self._descriptor

    @property
    def preview_spec(self) -> ExactDatasetPreviewSpec:
        """The sole exact-preview admission contract for this prepared scan."""

        return ExactDatasetPreviewSpec(self._source_schema.fingerprint)

    def materialize_provisional_output(
        self,
        source: DatasetPreviewSnapshot,
    ) -> OwnedSnapshot:
        """Apply this scan's authoritative output contract to one preview.

        A Workbench may render the returned immutable Dataset, but it never
        manufactures the scan output identity or replays the physical
        transform.  Provisional and FINAL publication therefore share the
        same program/contract owner and differ only in commit status.
        """

        if not isinstance(source, DatasetPreviewSnapshot):
            raise TypeError("source must be DatasetPreviewSnapshot")
        if source.snapshot.block.schema != self._source_schema:
            raise ValueError(
                "provisional scan source differs from the prepared source schema"
            )
        return _materialize_scan_output(
            self._program,
            source.snapshot,
            self._output_contract,
        )

    def start(
        self,
        preview: ExactDatasetPreviewPort | None = None,
    ) -> RunHandle:
        """Start once, optionally attaching one display port."""

        try:
            self._claim_start()
            return self._start(preview)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise

    def final_dataset_outputs(self, reference: ScanArtifactRef):
        """Materialize the committed scan through this prepared command."""

        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("scan FINAL result must be ScanArtifactRef")
        materialized = self._repository.materialize(reference)
        if materialized.program_fingerprint != self._program.fingerprint:
            raise ValueError("scan FINAL result belongs to another pulse program")
        if materialized.output_contract != self._output_contract:
            raise ValueError("scan FINAL result belongs to another output contract")
        return scan_final_outputs(materialized)

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedExactScan is one-shot")
            self._started = True


def _bind_scan_camera(
    request: _ScanRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> tuple[PulseScanProgram, ScanPointTable, _ScanCameraBinding]:
    """Bind one request to the exact target/camera generation once.

    This is the physical scan composition owner: a facade may resolve which
    installed ports to pass, but it cannot recreate repeat/readout axes,
    target-bound pulse documents, camera cardinality, or point layout.
    """

    if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
        raise TypeError("request must be a current scan request")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")

    if isinstance(request.program, AutonomousScanSlotProgram):
        program = AutonomousScanSlotProgram(
            bind_pulse_document_target(
                request.program.document,
                pulse_port.capability.target,
            ),
            request.program.api_values,
        )
        logical_document = program.execution_document
        require_autonomous_scan_resident_capacity(
            logical_document,
            pulse_port.capability.resident_scan_point_capacity,
        )
        execution_document = expand_autonomous_scan_repeats(logical_document)
        point_table = program.point_table
        repeat_count = program.repeat_count
        repeat_axis = AxisSpec(
            _SCAN_REPEAT_AXIS_ID,
            "repeat",
            REPEAT,
            repeat_count,
            tuple(range(repeat_count)),
        )
        binding = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=execution_document,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            trigger_channel=request.trigger_channel,
            layout=TriggeredCameraLayout(
                repeat_axis=repeat_axis,
                readout_event_axis_id=_SCAN_READOUT_EVENT_AXIS_ID,
                readout_events_per_repeat=1,
                scan_axes=point_table.point_axes,
                scan_point_layout=point_table.point_layout,
            ),
        )
        if (
            binding.compiled_artifact.source_document_digest
            != execution_document.fingerprint
        ):
            raise RuntimeError(
                "compiled scan pulse differs from the repeat-major execution document"
            )
        return program, point_table, binding

    if isinstance(request.program, ApiSlotSegmentedProgram):
        binding = bind_api_slot_segmented_camera_acquisition(
            pulse_port,
            camera_port,
            program=request.program,
            trigger_channel=request.trigger_channel,
            repeat_axis_id=_SCAN_REPEAT_AXIS_ID,
            readout_event_axis_id=_SCAN_READOUT_EVENT_AXIS_ID,
        )
        return binding.program, binding.point_table, binding

    raise TypeError("request has an unknown pulse-scan program")


def _compiled_scan_artifacts_digest(
    binding: _ScanCameraBinding,
) -> str:
    if isinstance(binding, TriggeredCameraBinding):
        return binding.compiled_artifact.fingerprint
    if isinstance(binding, ApiSlotSegmentedCameraBinding):
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.scan.compiled-lineage",
                "artifacts": tuple(
                    artifact.fingerprint
                    for artifact in binding.compiled_artifacts
                ),
            }
        )
    raise TypeError("binding must be a scan camera binding")


def _bind_scan_output(
    source_schema: DatasetSchema,
    point_table: ScanPointTable,
    requested: DataTransformSpec | None,
) -> ScanOutputContract:
    requested_operations = () if requested is None else requested.operations
    committed = commit_transform(
        source_schema,
        DataTransformSpec(
            (
                Selection.index(_SCAN_READOUT_EVENT_AXIS_ID, 0),
                *requested_operations,
            )
        ),
    )
    return bind_scan_output_contract(source_schema, point_table, committed)


def _scan_descriptor(
    *,
    name: str,
    request: _ScanRequest,
    binding: _ScanCameraBinding,
    output_schema: DatasetSchema,
    compiled_digest: str,
) -> PlanDescriptor:
    execution_form = (
        PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
        if isinstance(binding, TriggeredCameraBinding)
        else PulseExecutionForm.STATIC_ONCE
    )
    return PlanDescriptor(
        name,
        request.camera_ref.role,
        request.sequencer_ref.role,
        execution_form,
        binding.trigger_channel,
        binding.expected_frames,
        output_schema,
        compiled_digest,
        (
            str(binding.pulse_port.resource_claim.key),
            str(binding.measurement.capture_port.resource_claim.key),
        ),
    )


def prepare_exact_scan(
    request: _ScanRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    repository: ScanRepository,
    start_run: Callable[[RunPlan], RunHandle],
    calibration: ResolvedCalibration | None = None,
) -> PreparedExactScan:
    """Prepare one direct-Camera or Camera→Occupancy exact scan.

    Installation code contributes only already-resolved ports, repository and
    start capability.  This application owner freezes all physical binding,
    dataset identity, authoritative transform, processor pipeline and plan
    construction, so notebook and Workbench cannot grow separate scan rules.
    """

    if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
        raise TypeError("request must be ScanRequest or OccupancyScanRequest")
    if type(repository) is not ScanRepository:
        raise TypeError("repository must be ScanRepository")
    if not callable(start_run):
        raise TypeError("start_run must be callable")
    if isinstance(request, ScanRequest):
        if calibration is not None:
            raise ValueError("a direct Camera scan does not accept calibration")
    elif type(calibration) is not ResolvedCalibration:
        raise TypeError("an Occupancy scan requires an admitted calibration")

    program, point_table, binding = _bind_scan_camera(
        request,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )
    compiled_digest = _compiled_scan_artifacts_digest(binding)
    camera_schema = binding.measurement.capture_contract.dataset_schema

    if isinstance(request, ScanRequest):
        source_schema = camera_schema
        output_contract = _bind_scan_output(
            source_schema,
            point_table,
            request.output_transform_spec,
        )
        if isinstance(binding, TriggeredCameraBinding):
            scan_spec, descriptor = bind_finite_capture_spec(
                binding=binding,
                block_id=BlockId(
                    f"scan-camera-{binding.compiled_artifact.fingerprint[:20]}"
                ),
                camera_ref=request.camera_ref,
                sequencer_ref=request.sequencer_ref,
                execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
                name_prefix="Direct scan",
            )
            descriptor = replace(
                descriptor,
                output_schema=output_contract.output_dataset_schema,
            )

            def start(preview):
                return start_run(
                    compile_direct_scan_artifact_plan(
                        scan_spec,
                        repository,
                        program=program,
                        output_contract=output_contract,
                        preview=preview,
                    )
                )

        else:
            capture = MinimalPipelineSpec(
                f"API segmented scan {program.document.name}",
                binding.measurement,
                BlockId(f"api-scan-camera-{compiled_digest[:20]}"),
            )
            scan_spec = ApiSlotSegmentedSpec(
                capture,
                binding.pulse_port,
                binding.point_descriptors,
                program.repeat_count,
            )
            descriptor = _scan_descriptor(
                name=capture.name,
                request=request,
                binding=binding,
                output_schema=output_contract.output_dataset_schema,
                compiled_digest=compiled_digest,
            )

            def start(preview):
                if preview is not None:
                    raise ValueError(
                        "API segmented direct scan is FINAL-only; preview is unsupported"
                    )
                return start_run(
                    compile_api_direct_scan_artifact_plan(
                        scan_spec,
                        repository,
                        program=program,
                        output_contract=output_contract,
                    )
                )

    else:
        assert calibration is not None
        occupancy, counts_schema, occupied_schema = bind_occupancy_pipeline(
            binding.measurement,
            calibration,
            model_kind=request.model_kind,
            timing_identity={
                "owner": "pulse-scan",
                "program": program.fingerprint,
                "compiled_pulse_lineage": compiled_digest,
            },
            name=f"Occupancy scan {program.document.name}",
        )
        source_schema = (
            counts_schema
            if request.output_name == "counts"
            else occupied_schema
        )
        output_contract = _bind_scan_output(
            source_schema,
            point_table,
            request.output_transform_spec,
        )
        if isinstance(binding, TriggeredCameraBinding):
            scan_spec = TriggeredOccupancySpec(
                occupancy,
                binding.pulse_port,
                binding.pulse_request,
                binding.trigger_channel,
                binding.cell_plan,
            )
        else:
            scan_spec = ApiSlotSegmentedSpec(
                occupancy,
                binding.pulse_port,
                binding.point_descriptors,
                program.repeat_count,
            )
        descriptor = _scan_descriptor(
            name=occupancy.name,
            request=request,
            binding=binding,
            output_schema=output_contract.output_dataset_schema,
            compiled_digest=compiled_digest,
        )

        def start(preview):
            if preview is not None and request.output_name == "occupied":
                raise ValueError(
                    "occupied Pulse scan output is FINAL-only; the exact "
                    "progressive publisher owns counts"
                )
            if isinstance(scan_spec, TriggeredOccupancySpec):
                plan = compile_occupancy_scan_artifact_plan(
                    scan_spec,
                    repository,
                    program=program,
                    output_contract=output_contract,
                    source_output_name=request.output_name,
                    preview=preview,
                )
            else:
                if preview is not None:
                    raise ValueError(
                        "API segmented occupancy is FINAL-only; preview is unsupported"
                    )
                plan = compile_api_occupancy_scan_artifact_plan(
                    scan_spec,
                    repository,
                    program=program,
                    output_contract=output_contract,
                    source_output_name=request.output_name,
                )
            return start_run(plan)

    return PreparedExactScan(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
        descriptor=descriptor,
        repository=repository,
        start=start,
    )


def _require_output_binding(
    *,
    program: PulseScanProgram,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
) -> None:
    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a current pulse-scan program")
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    point_table = program.point_table
    rebound = bind_scan_output_contract(
        source_schema,
        point_table,
        output_contract.committed_transform,
    )
    if rebound != output_contract:
        raise ValueError("ScanOutputContract differs from its exact source schema")


def _materialize_scan_output(
    program: PulseScanProgram,
    source: OwnedSnapshot,
    output_contract: ScanOutputContract,
) -> OwnedSnapshot:
    """Single physical projection used by provisional and FINAL scan paths."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be OwnedSnapshot")
    _require_output_binding(
        program=program,
        source_schema=source.block.schema,
        output_contract=output_contract,
    )
    output_ref = _scan_output_dataset_ref(
        program,
        source.ref,
        output_contract,
    )
    return materialize_transformed_snapshot(
        source,
        output_contract.committed_transform,
        output_ref=output_ref,
        output_schema=output_contract.output_dataset_schema,
    )


def _prepare_dataset(
    context: PostSafetyContext,
    *,
    run_id: str,
    execution: PulseScanExecution,
    source: OwnedSnapshot,
    output_contract: ScanOutputContract,
    provenance,
    staged_lineage: _StagedScanLineage,
) -> _PreparedScanDataset:
    context.checkpoint()
    output = _materialize_scan_output(
        execution.program,
        source,
        output_contract,
    )
    context.checkpoint()
    return _PreparedScanDataset(
        _SCAN_APPLICATION_TOKEN,
        run_id=run_id,
        execution=execution,
        source_snapshot=source,
        output_contract=output_contract,
        output_snapshot=output,
        provenance=provenance,
        staged_lineage=staged_lineage,
    )


def _compile_scan_artifact_plan(
    base: RunPlan,
    repository: ScanRepository,
    *,
    program: PulseScanProgram,
    output_contract: ScanOutputContract,
    compiled_pulses: tuple[CompiledPulseArtifact, ...],
    adapt: _ResultAdapter,
) -> RunPlan:
    """Keep the durable sink admitted from preflight through FINAL commit."""

    if not isinstance(base, RunPlan):
        raise TypeError("base must be RunPlan")
    if type(repository) is not ScanRepository:
        raise TypeError("repository must be ScanRepository")
    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a current pulse-scan program")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    compiled_pulses = tuple(compiled_pulses)
    if not compiled_pulses or any(
        not isinstance(item, CompiledPulseArtifact) for item in compiled_pulses
    ):
        raise TypeError("compiled_pulses must contain CompiledPulseArtifact values")
    if not callable(adapt):
        raise TypeError("adapt must be callable")
    repository._require_active()

    base_preflight = base.preflight
    base_execute = base.execute
    base_cleanup = base.cleanup
    base_finalize = base.finalize
    base_dispose_unfinalized = base.dispose_unfinalized

    def preflight(
        context: RunContext,
    ) -> tuple[object, RepositoryRootLeaseBorrow, _StagedScanLineage]:
        borrow = repository._root_lease.borrow()
        try:
            borrow.require_active()
            staged = repository._stage_static_lineage(
                program,
                compiled_pulses,
            )
            return base_preflight(context), borrow, staged
        except BaseException:
            borrow.close()
            raise

    def execute(
        context: RunContext,
        prepared: tuple[object, RepositoryRootLeaseBorrow, _StagedScanLineage],
    ) -> tuple[object, RepositoryRootLeaseBorrow, _StagedScanLineage]:
        base_prepared, borrow, staged = prepared
        borrow.require_active()
        return base_execute(context, base_prepared), borrow, staged

    def cleanup(
        context: RunContext,
        prepared: tuple[
            object,
            RepositoryRootLeaseBorrow,
            _StagedScanLineage,
        ]
        | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        base_prepared = None if prepared is None else prepared[0]
        borrow = None if prepared is None else prepared[1]
        try:
            report = base_cleanup(context, base_prepared, primary)
        except BaseException:
            if borrow is not None:
                borrow.close()
            raise
        if borrow is not None and (primary is not None or report.errors):
            borrow.close()
        return report

    def finalize(
        context: PostSafetyContext,
        result: tuple[object, RepositoryRootLeaseBorrow, _StagedScanLineage],
    ) -> ScanArtifactRef:
        base_result, borrow, staged_lineage = result
        result = None  # type: ignore[assignment]
        try:
            borrow.require_active()
            finalized = base_finalize(context, base_result)
            adapted = adapt(finalized)
            # Release the opaque joint result before transforming.  In the
            # occupancy path this drops the unselected sibling and event
            # metadata; only the selected authoritative output remains.
            # Calibration arrays retained by the frozen plan remain owned by
            # that plan.
            base_result = None
            finalized = None
            run_id, source, provenance, execution = adapted
            prepared = _prepare_dataset(
                context,
                run_id=run_id,
                execution=execution,
                source=source,
                output_contract=output_contract,
                provenance=provenance,
                staged_lineage=staged_lineage,
            )
            operation = repository.final_commit(context, prepared)
            return context.commit_final(operation)
        finally:
            borrow.close()

    def dispose_unfinalized(
        result: tuple[object, RepositoryRootLeaseBorrow, _StagedScanLineage],
    ) -> None:
        base_result, borrow, _staged_lineage = result
        error: BaseException | None = None
        if base_dispose_unfinalized is not None:
            try:
                base_dispose_unfinalized(base_result)
            except BaseException as dispose_error:
                error = dispose_error
        try:
            borrow.close()
        except BaseException as close_error:
            if error is None:
                error = close_error
            else:
                error.add_note(
                    "scan repository borrow close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        if error is not None:
            raise error

    return RunPlan(
        name=base.name,
        resource_claims=base.resource_claims,
        bound_devices=base.bound_devices,
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=base.interrupt_operations,
        timeout_seconds=base.timeout_seconds,
        requires_final_commit=True,
        dispose_unfinalized=dispose_unfinalized,
    )


def compile_direct_scan_artifact_plan(
    spec: TriggeredCaptureSpec,
    repository: ScanRepository,
    *,
    program: AutonomousScanSlotProgram,
    output_contract: ScanOutputContract,
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile direct camera y with an optional exact provisional view."""

    try:
        return _compile_direct_scan_artifact_plan(
            spec,
            repository,
            program=program,
            output_contract=output_contract,
            preview=preview,
        )
    except BaseException as error:
        _notify_preview_failure(preview, error)
        raise


def _compile_direct_scan_artifact_plan(
    spec: TriggeredCaptureSpec,
    repository: ScanRepository,
    *,
    program: AutonomousScanSlotProgram,
    output_contract: ScanOutputContract,
    preview: ExactDatasetPreviewPort | None,
) -> RunPlan:
    """Validated direct-scan compiler implementation."""

    if not isinstance(spec, TriggeredCaptureSpec):
        raise TypeError("spec must be TriggeredCaptureSpec")
    source_schema = spec.capture.measurement.capture_contract.dataset_schema
    _require_output_binding(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
    )
    def adapt(
        value: object,
    ) -> tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseScanExecution]:
        if type(value) is not TriggeredPipelineResult:
            raise TypeError("direct scan base plan returned another result")
        pipeline = value.capture
        return (
            pipeline.run_id,
            pipeline.dataset.snapshot,
            pipeline.dataset.provenance,
            AutonomousScanExecution(
                program,
                value.lineage.evidence(),
                camera_run_evidence_from_pipeline(pipeline),
            ),
        )

    base = compile_triggered_pipeline(spec, exact_preview=preview)
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        compiled_pulses=(spec.pulse_request.artifact,),
        adapt=adapt,
    )


def compile_occupancy_scan_artifact_plan(
    spec: TriggeredOccupancySpec,
    repository: ScanRepository,
    *,
    program: AutonomousScanSlotProgram,
    output_contract: ScanOutputContract,
    source_output_name: str = "counts",
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile one occupancy scan and terminalize any rejected preview port."""

    try:
        return _compile_occupancy_scan_artifact_plan(
            spec,
            repository,
            program=program,
            output_contract=output_contract,
            source_output_name=source_output_name,
            preview=preview,
        )
    except BaseException as error:
        _notify_preview_failure(preview, error)
        raise


def _compile_occupancy_scan_artifact_plan(
    spec: TriggeredOccupancySpec,
    repository: ScanRepository,
    *,
    program: AutonomousScanSlotProgram,
    output_contract: ScanOutputContract,
    source_output_name: str,
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile one selected occupancy output into a canonical FINAL scan Run."""

    if not isinstance(spec, TriggeredOccupancySpec):
        raise TypeError("spec must be TriggeredOccupancySpec")
    camera_schema = spec.occupancy.measurement.capture_contract.dataset_schema
    resolved = resolve_occupancy_stream_schema(
        spec.occupancy.processor,
        camera_schema,
    )
    if source_output_name not in ("counts", "occupied"):
        raise ValueError("source_output_name must be 'counts' or 'occupied'")
    source_schema = (
        resolved.counts_schema
        if source_output_name == "counts"
        else resolved.occupied_schema
    )
    preview_spec = _occupancy_preview_spec(spec.occupancy, preview)
    _require_output_binding(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
    )

    def adapt(
        value: object,
    ) -> tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseScanExecution]:
        if type(value) is not TriggeredOccupancyPipelineResult:
            raise TypeError("occupancy scan base plan returned another result")
        pipeline = value.occupancy.pipeline
        return (
            pipeline.run_id,
            (
                value.occupancy.dataset.counts
                if source_output_name == "counts"
                else value.occupancy.dataset.occupied
            ),
            pipeline.dataset.provenance,
            AutonomousScanExecution(
                program,
                value.lineage.evidence(),
                camera_run_evidence_from_pipeline(pipeline),
            ),
        )

    base = compile_triggered_occupancy_pipeline(
        spec,
        preview=preview,
        _admitted_preview_spec=preview_spec,
    )
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        compiled_pulses=(spec.pulse_request.artifact,),
        adapt=adapt,
    )


def _segmented_compiled_pulses(
    program: ApiSlotSegmentedProgram,
    point_descriptors,
) -> tuple[CompiledPulseArtifact, ...]:
    point_count = program.point_count
    values = tuple(point_descriptors)
    if len(values) != point_count or tuple(
        value.point_ordinal for value in values
    ) != tuple(range(point_count)):
        raise ValueError("API point descriptors do not cover P in row order")
    return tuple(value.pulse_request.artifact for value in values)


def _api_execution(
    program: ApiSlotSegmentedProgram,
    segments,
    pipeline: PipelineResult,
) -> ApiSegmentedScanExecution:
    return ApiSegmentedScanExecution(
        program,
        tuple(
            ApiSegmentEvidence(
                value.address.repeat_index,
                value.address.point_storage_index,
                value.evidence,
            )
            for value in segments
        ),
        camera_run_evidence_from_pipeline(pipeline),
    )


def compile_api_direct_scan_artifact_plan(
    spec: ApiSlotSegmentedSpec,
    repository: ScanRepository,
    *,
    program: ApiSlotSegmentedProgram,
    output_contract: ScanOutputContract,
) -> RunPlan:
    """Commit direct camera y from the accepted API segmented exception."""

    if not isinstance(spec, ApiSlotSegmentedSpec) or not isinstance(
        spec.pipeline,
        MinimalPipelineSpec,
    ):
        raise TypeError("spec must contain a direct capture pipeline")
    if spec.repeat_count != program.repeat_count:
        raise ValueError("API segmented spec repeat count differs from its program")
    source_schema = spec.pipeline.measurement.capture_contract.dataset_schema
    compiled_pulses = _segmented_compiled_pulses(
        program,
        spec.point_descriptors,
    )
    _require_output_binding(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
    )
    def adapt(value: object):
        if type(value) is not ApiSlotSegmentedResult:
            raise TypeError("segmented direct scan returned another result")
        pipeline = value.payload
        if not isinstance(pipeline, PipelineResult):
            raise TypeError("segmented direct scan returned occupancy data")
        return (
            pipeline.run_id,
            pipeline.dataset.snapshot,
            pipeline.dataset.provenance,
            _api_execution(program, value.segments, pipeline),
        )

    base = compile_api_slot_segmented_pipeline(spec)
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        compiled_pulses=compiled_pulses,
        adapt=adapt,
    )


def compile_api_occupancy_scan_artifact_plan(
    spec: ApiSlotSegmentedSpec,
    repository: ScanRepository,
    *,
    program: ApiSlotSegmentedProgram,
    output_contract: ScanOutputContract,
    source_output_name: str = "counts",
) -> RunPlan:
    """Commit one FINAL occupancy output from finite API segments."""

    if not isinstance(spec, ApiSlotSegmentedSpec) or not isinstance(
        spec.pipeline,
        OccupancyPipelineSpec,
    ):
        raise TypeError("spec must contain an occupancy pipeline")
    if spec.repeat_count != program.repeat_count:
        raise ValueError("API segmented spec repeat count differs from its program")
    occupancy_spec = spec.pipeline
    camera_schema = occupancy_spec.measurement.capture_contract.dataset_schema
    resolved_source = resolve_occupancy_stream_schema(
        occupancy_spec.processor,
        camera_schema,
    )
    if source_output_name not in ("counts", "occupied"):
        raise ValueError("source_output_name must be 'counts' or 'occupied'")
    source_schema = (
        resolved_source.counts_schema
        if source_output_name == "counts"
        else resolved_source.occupied_schema
    )
    compiled_pulses = _segmented_compiled_pulses(
        program,
        spec.point_descriptors,
    )
    _require_output_binding(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
    )
    def adapt(value: object):
        if type(value) is not ApiSlotSegmentedResult:
            raise TypeError("segmented occupancy scan returned another result")
        occupancy = value.payload
        if not isinstance(occupancy, OccupancyPipelineResult):
            raise TypeError("segmented occupancy scan returned direct data")
        pipeline = occupancy.pipeline
        return (
            pipeline.run_id,
            (
                occupancy.dataset.counts
                if source_output_name == "counts"
                else occupancy.dataset.occupied
            ),
            pipeline.dataset.provenance,
            _api_execution(program, value.segments, pipeline),
        )

    base = compile_api_slot_segmented_occupancy_pipeline(spec)
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        compiled_pulses=compiled_pulses,
        adapt=adapt,
    )


__all__ = [
    "compile_api_direct_scan_artifact_plan",
    "compile_api_occupancy_scan_artifact_plan",
    "compile_direct_scan_artifact_plan",
    "compile_occupancy_scan_artifact_plan",
    "PreparedExactScan",
    "prepare_exact_scan",
]
