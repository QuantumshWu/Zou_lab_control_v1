"""One-Run application boundary for canonical pulse-scan artifacts."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import (
    DatasetSchema,
    OwnedSnapshot,
    materialize_transformed_snapshot,
    transformed_snapshot_peak_nbytes,
)
from zlc_neutral_atom.readout.calibration import calibration_retained_array_nbytes
from zlc_neutral_atom.readout.occupancy import resolve_occupancy_stream_schema
from zlc_neutral_atom.readout.occupancy_pipeline import (
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    _occupancy_preview_spec,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import DatasetSealProvenance
from zlc_neutral_atom.runtime.pipeline import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
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
from zlc_neutral_atom.timing.occupancy import (
    TriggeredOccupancyPipelineResult,
    TriggeredOccupancySpec,
    compile_triggered_occupancy_pipeline,
)
from zlc_neutral_atom.timing.segmented import (
    ApiSlotSegmentedResult,
    ApiSlotSegmentedSpec,
    api_slot_segmented_control_retained_upper_bound_bytes,
    compile_api_slot_segmented_occupancy_pipeline,
    compile_api_slot_segmented_pipeline,
)
from zlc_pulse import CompiledPulseArtifact
from zlc_storage import (
    RepositoryRootLeaseBorrow,
    nonnegative_integer,
    positive_integer,
)

from .contracts import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseScanProgram,
    ScanOutputContract,
    bind_scan_output_contract,
)
from .lineage import (
    ApiSegmentEvidence,
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    api_segmented_metadata_static_shape_to_tree,
    camera_run_evidence_from_pipeline,
)
from .reference import ScanArtifactRef
from .repository import (
    ScanRepository,
    _ApiFinalMetadataAdmission,
    _PreparedScanDataset,
    _SCAN_APPLICATION_TOKEN,
    _StaticScanLineageAdmission,
    _StagedScanLineage,
    _scan_output_dataset_ref,
)


_ResultAdapter = Callable[
    [object],
    tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseScanExecution],
]


class PreparedOccupancyScan:
    """One-shot command exposing only the typed progressive preview seam."""

    __slots__ = (
        "_lock",
        "_output_contract",
        "_source_schema",
        "_start",
        "_started",
    )

    def __init__(
        self,
        *,
        source_schema: DatasetSchema,
        output_contract: ScanOutputContract,
        start: Callable[[ExactDatasetPreviewPort | None], RunHandle],
    ) -> None:
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not callable(start):
            raise TypeError("start must be callable")
        self._source_schema = source_schema
        self._output_contract = output_contract
        self._start = start
        self._lock = threading.Lock()
        self._started = False

    @property
    def source_schema(self) -> DatasetSchema:
        return self._source_schema

    @property
    def output_contract(self) -> ScanOutputContract:
        return self._output_contract

    def start(
        self,
        preview: ExactDatasetPreviewPort | None = None,
    ) -> RunHandle:
        """Start once, optionally attaching one already-budgeted display port."""

        try:
            self._claim_start()
            return self._start(preview)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedOccupancyScan is one-shot")
            self._started = True


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


def _prepare_dataset(
    context: PostSafetyContext,
    *,
    run_id: str,
    execution: PulseScanExecution,
    source: OwnedSnapshot,
    output_contract: ScanOutputContract,
    provenance,
    staged_lineage: _StagedScanLineage,
    api_metadata_admission: _ApiFinalMetadataAdmission | None,
    memory_limit_bytes: int,
) -> _PreparedScanDataset:
    context.checkpoint()
    output_ref = _scan_output_dataset_ref(
        execution.program,
        source.ref,
        output_contract,
    )
    output = materialize_transformed_snapshot(
        source,
        output_contract.committed_transform,
        output_ref=output_ref,
        output_schema=output_contract.output_dataset_schema,
        memory_limit_bytes=memory_limit_bytes,
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
        api_metadata_admission=api_metadata_admission,
        memory_limit_bytes=memory_limit_bytes,
    )


def _admit_final_data_limit(
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    *,
    memory_limit_bytes: int,
    retained_overhead_bytes: int,
) -> int:
    """Reject the final scan working set before compiling a hardware plan."""

    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    overhead = nonnegative_integer(
        retained_overhead_bytes,
        "retained_overhead_bytes",
    )
    if overhead >= limit:
        raise MemoryError(
            f"scan final retained overhead {overhead} exceeds limit {limit}"
        )
    final_data_limit = limit - overhead
    final_data_peak = transformed_snapshot_peak_nbytes(
        source_schema,
        output_contract.committed_transform,
    )
    if final_data_peak > final_data_limit:
        raise MemoryError(
            "scan final data-plane peak "
            f"{final_data_peak + overhead} exceeds limit {limit}"
        )
    return final_data_limit


def _compile_scan_artifact_plan(
    base: RunPlan,
    repository: ScanRepository,
    *,
    program: PulseScanProgram,
    output_contract: ScanOutputContract,
    final_data_limit_bytes: int,
    static_lineage: _StaticScanLineageAdmission,
    api_metadata_admission: _ApiFinalMetadataAdmission | None,
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
    if not isinstance(static_lineage, _StaticScanLineageAdmission):
        raise TypeError("static_lineage must be admitted scan lineage")
    if isinstance(program, ApiSlotSegmentedProgram):
        if not isinstance(api_metadata_admission, _ApiFinalMetadataAdmission):
            raise TypeError("API scan requires FINAL metadata admission")
    elif api_metadata_admission is not None:
        raise ValueError("autonomous scan cannot carry API metadata admission")
    compiled_pulses = tuple(compiled_pulses)
    if not compiled_pulses or any(
        not isinstance(item, CompiledPulseArtifact) for item in compiled_pulses
    ):
        raise TypeError("compiled_pulses must contain CompiledPulseArtifact values")
    final_data_limit = positive_integer(
        final_data_limit_bytes,
        "final_data_limit_bytes",
    )
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
                static_lineage,
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
        if borrow is not None and (
            primary is not None or report.errors or report.decisions
        ):
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
            # occupancy path this drops the occupied sibling and event metadata;
            # only the authoritative counts source remains.  Calibration arrays
            # still retained by the frozen plan were deducted from the admitted
            # ``final_data_limit`` before the base hardware plan was compiled.
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
                api_metadata_admission=api_metadata_admission,
                memory_limit_bytes=final_data_limit,
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
    memory_limit_bytes: int,
) -> RunPlan:
    """Compile direct camera y into one canonical FINAL scan Run."""

    if not isinstance(spec, TriggeredCaptureSpec):
        raise TypeError("spec must be TriggeredCaptureSpec")
    source_schema = spec.capture.measurement.capture_contract.dataset_schema
    _require_output_binding(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
    )
    _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=0,
    )
    static_lineage = repository._admit_static_lineage(
        program,
        (spec.pulse_request.artifact,),
        memory_limit_bytes=memory_limit_bytes,
    )
    final_data_limit = _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=static_lineage.retained_upper_bound_bytes,
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

    base = compile_triggered_pipeline(
        spec,
        _retained_overhead_bytes=static_lineage.retained_upper_bound_bytes,
    )
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        final_data_limit_bytes=final_data_limit,
        static_lineage=static_lineage,
        api_metadata_admission=None,
        compiled_pulses=(spec.pulse_request.artifact,),
        adapt=adapt,
    )


def compile_occupancy_scan_artifact_plan(
    spec: TriggeredOccupancySpec,
    repository: ScanRepository,
    *,
    program: AutonomousScanSlotProgram,
    output_contract: ScanOutputContract,
    memory_limit_bytes: int,
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile one occupancy scan and terminalize any rejected preview port."""

    try:
        return _compile_occupancy_scan_artifact_plan(
            spec,
            repository,
            program=program,
            output_contract=output_contract,
            memory_limit_bytes=memory_limit_bytes,
            preview=preview,
        )
    except BaseException as error:
        _notify_preview_failure(preview, error)
        raise


def _admit_optional_preview_data_limit(
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    *,
    memory_limit_bytes: int,
    retained_base_bytes: int,
    preview: ExactDatasetPreviewPort | None,
    preview_spec: ExactDatasetPreviewSpec | None,
) -> tuple[
    int,
    ExactDatasetPreviewPort | None,
    ExactDatasetPreviewSpec | None,
]:
    """Admit science data first, then retain a preview only when it also fits.

    Type/schema/terminal errors are validated before this helper and remain
    hard failures.  Only the exact delta introduced by an otherwise valid
    preview may degrade to FINAL-only, and the rejected port receives that
    capacity reason from the neutral owner of the memory formula.
    """

    if (preview is None) != (preview_spec is None):
        raise ValueError("preview and preview_spec must be present together")
    preview_bytes = (
        0 if preview_spec is None else preview_spec.downstream_peak_bytes
    )
    try:
        admitted = _admit_final_data_limit(
            source_schema,
            output_contract,
            memory_limit_bytes=memory_limit_bytes,
            retained_overhead_bytes=retained_base_bytes + preview_bytes,
        )
    except MemoryError as preview_error:
        if preview is None:
            raise
        baseline = _admit_final_data_limit(
            source_schema,
            output_contract,
            memory_limit_bytes=memory_limit_bytes,
            retained_overhead_bytes=retained_base_bytes,
        )
        _notify_preview_failure(preview, preview_error)
        if preview.terminal is not True:
            raise RuntimeError(
                "capacity-rejected preview did not reach a terminal state"
            ) from preview_error
        return baseline, None, None
    return admitted, preview, preview_spec


def _compile_occupancy_scan_artifact_plan(
    spec: TriggeredOccupancySpec,
    repository: ScanRepository,
    *,
    program: AutonomousScanSlotProgram,
    output_contract: ScanOutputContract,
    memory_limit_bytes: int,
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile camera→occupancy counts y into one canonical FINAL scan Run."""

    if not isinstance(spec, TriggeredOccupancySpec):
        raise TypeError("spec must be TriggeredOccupancySpec")
    camera_schema = spec.occupancy.measurement.capture_contract.dataset_schema
    resolved = resolve_occupancy_stream_schema(
        spec.occupancy.processor,
        camera_schema,
    )
    preview_spec = _occupancy_preview_spec(spec.occupancy, preview)
    _require_output_binding(
        program=program,
        source_schema=resolved.counts_schema,
        output_contract=output_contract,
    )
    static_lineage = repository._admit_static_lineage(
        program,
        (spec.pulse_request.artifact,),
        memory_limit_bytes=memory_limit_bytes,
    )
    final_data_limit, preview, preview_spec = _admit_optional_preview_data_limit(
        resolved.counts_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_base_bytes=(
            static_lineage.retained_upper_bound_bytes
            + calibration_retained_array_nbytes(
                spec.occupancy.processor.calibration.artifact
            )
        ),
        preview=preview,
        preview_spec=preview_spec,
    )

    def adapt(
        value: object,
    ) -> tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseScanExecution]:
        if type(value) is not TriggeredOccupancyPipelineResult:
            raise TypeError("occupancy scan base plan returned another result")
        pipeline = value.occupancy.pipeline
        return (
            pipeline.run_id,
            value.occupancy.dataset.counts,
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
        _retained_overhead_bytes=static_lineage.retained_upper_bound_bytes,
    )
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        final_data_limit_bytes=final_data_limit,
        static_lineage=static_lineage,
        api_metadata_admission=None,
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


def _api_execution_static_shape(
    spec: ApiSlotSegmentedSpec,
    program: ApiSlotSegmentedProgram,
) -> dict[str, object]:
    measurement = spec.pipeline.measurement
    contract = measurement.capture_contract
    camera_schema = contract.dataset_schema
    if isinstance(spec.pipeline, MinimalPipelineSpec):
        result_stream_id = contract.stream_id
        result_source_id = contract.source_id
        derivation_root_stream_id = None
    else:
        result_stream_id = spec.pipeline.processor.output_stream_id
        result_source_id = spec.pipeline.processor.output_source_id
        derivation_root_stream_id = contract.stream_id
    return api_segmented_metadata_static_shape_to_tree(
        program,
        tuple(value.pulse_binding for value in spec.point_descriptors),
        camera_source_stream_id=contract.stream_id,
        result_stream_id=result_stream_id,
        result_source_id=result_source_id,
        derivation_root_stream_id=derivation_root_stream_id,
        camera_provenance=contract.camera_provenance,
        camera_capability=contract.capability.camera_capability_evidence,
        camera_arm_spec=measurement.capture_spec,
        camera_source_value_schema=camera_schema.cell_schema,
        camera_source_schema_fingerprint=camera_schema.fingerprint,
    )


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
    memory_limit_bytes: int,
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
    control_retained = api_slot_segmented_control_retained_upper_bound_bytes(
        program.point_count,
        program.repeat_count,
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
    _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=control_retained,
    )
    static_lineage = repository._admit_static_lineage(
        program,
        compiled_pulses,
        memory_limit_bytes=memory_limit_bytes,
    )
    api_metadata_admission = repository._admit_api_final_metadata(
        program,
        static_lineage,
        spec.pipeline.block_id,
        source_schema,
        output_contract,
        _api_execution_static_shape(spec, program),
    )
    final_data_limit = _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=(
            static_lineage.retained_upper_bound_bytes + control_retained
        ),
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

    base = compile_api_slot_segmented_pipeline(
        spec,
        _retained_overhead_bytes=(
            static_lineage.retained_upper_bound_bytes + control_retained
        ),
    )
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        final_data_limit_bytes=final_data_limit,
        static_lineage=static_lineage,
        api_metadata_admission=api_metadata_admission,
        compiled_pulses=compiled_pulses,
        adapt=adapt,
    )


def compile_api_occupancy_scan_artifact_plan(
    spec: ApiSlotSegmentedSpec,
    repository: ScanRepository,
    *,
    program: ApiSlotSegmentedProgram,
    output_contract: ScanOutputContract,
    memory_limit_bytes: int,
) -> RunPlan:
    """Commit FINAL-only occupancy counts from finite API segments."""

    if not isinstance(spec, ApiSlotSegmentedSpec) or not isinstance(
        spec.pipeline,
        OccupancyPipelineSpec,
    ):
        raise TypeError("spec must contain an occupancy pipeline")
    if spec.repeat_count != program.repeat_count:
        raise ValueError("API segmented spec repeat count differs from its program")
    occupancy_spec = spec.pipeline
    camera_schema = occupancy_spec.measurement.capture_contract.dataset_schema
    source_schema = resolve_occupancy_stream_schema(
        occupancy_spec.processor,
        camera_schema,
    ).counts_schema
    control_retained = api_slot_segmented_control_retained_upper_bound_bytes(
        program.point_count,
        program.repeat_count,
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
    _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=control_retained,
    )
    static_lineage = repository._admit_static_lineage(
        program,
        compiled_pulses,
        memory_limit_bytes=memory_limit_bytes,
    )
    api_metadata_admission = repository._admit_api_final_metadata(
        program,
        static_lineage,
        occupancy_spec.counts_block_id,
        source_schema,
        output_contract,
        _api_execution_static_shape(spec, program),
    )
    final_data_limit = _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=(
            static_lineage.retained_upper_bound_bytes
            + control_retained
            + calibration_retained_array_nbytes(
                occupancy_spec.processor.calibration.artifact
            )
        ),
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
            occupancy.dataset.counts,
            pipeline.dataset.provenance,
            _api_execution(program, value.segments, pipeline),
        )

    base = compile_api_slot_segmented_occupancy_pipeline(
        spec,
        _retained_overhead_bytes=(
            static_lineage.retained_upper_bound_bytes + control_retained
        ),
    )
    return _compile_scan_artifact_plan(
        base,
        repository,
        program=program,
        output_contract=output_contract,
        final_data_limit_bytes=final_data_limit,
        static_lineage=static_lineage,
        api_metadata_admission=api_metadata_admission,
        compiled_pulses=compiled_pulses,
        adapt=adapt,
    )


__all__ = [
    "compile_api_direct_scan_artifact_plan",
    "compile_api_occupancy_scan_artifact_plan",
    "compile_direct_scan_artifact_plan",
    "compile_occupancy_scan_artifact_plan",
    "PreparedOccupancyScan",
]
