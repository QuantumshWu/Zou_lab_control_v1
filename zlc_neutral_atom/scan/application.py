"""One-Run application boundary for canonical pulse-scan artifacts."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import (
    DatasetSchema,
    OwnedSnapshot,
    materialize_transformed_snapshot,
)
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
    compile_api_slot_segmented_occupancy_pipeline,
    compile_api_slot_segmented_pipeline,
)
from zlc_pulse import CompiledPulseArtifact
from zlc_storage import RepositoryRootLeaseBorrow

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
    camera_run_evidence_from_pipeline,
)
from .reference import ScanArtifactRef
from .repository import (
    ScanRepository,
    _PreparedScanDataset,
    _SCAN_APPLICATION_TOKEN,
    _StagedScanLineage,
    _scan_output_dataset_ref,
)


_ResultAdapter = Callable[
    [object],
    tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseScanExecution],
]


class PreparedExactScan:
    """One-shot exact scan command exposing one typed provisional dataset seam."""

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
        """Start once, optionally attaching one display port."""

        try:
            self._claim_start()
            return self._start(preview)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedExactScan is one-shot")
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
]
