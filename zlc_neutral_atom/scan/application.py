"""One-Run application boundary for canonical autonomous scan artifacts."""

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
from zlc_neutral_atom.readout.occupancy_pipeline import _occupancy_preview_spec
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import DatasetSealProvenance
from zlc_neutral_atom.runtime.pipeline import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
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
from zlc_neutral_atom.timing.lineage import PulseCaptureEvidence
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    expand_autonomous_scan_repeats,
)
from zlc_storage import (
    RepositoryRootLeaseBorrow,
    nonnegative_integer,
    positive_integer,
)

from .contracts import ScanOutputContract, ScanPointTable, bind_scan_output_contract
from .reference import ScanArtifactRef
from .repository import (
    ScanRepository,
    _PreparedScanDataset,
    _SCAN_APPLICATION_TOKEN,
    _StaticScanLineageAdmission,
    _StagedScanLineage,
    _scan_output_dataset_ref,
)


_ResultAdapter = Callable[
    [object],
    tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseCaptureEvidence],
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


def _require_compile_binding(
    *,
    document: PulseDocument,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    execution_form: PulseExecutionForm,
    compiled_source_document_digest: str,
) -> None:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    if execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        raise ValueError("formal SCAN_SLOT requires AUTONOMOUS_SCAN_ONCE")
    expanded = expand_autonomous_scan_repeats(document)
    if compiled_source_document_digest != expanded.fingerprint:
        raise ValueError("compiled pulse differs from the frozen repeat-major scan")
    point_table = ScanPointTable.from_pulse_document(document)
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
    document: PulseDocument,
    source: OwnedSnapshot,
    output_contract: ScanOutputContract,
    provenance,
    pulse_evidence,
    staged_lineage: _StagedScanLineage,
    memory_limit_bytes: int,
) -> _PreparedScanDataset:
    context.checkpoint()
    output_ref = _scan_output_dataset_ref(document, source.ref, output_contract)
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
        pulse_document=document,
        source_snapshot=source,
        output_contract=output_contract,
        output_snapshot=output,
        provenance=provenance,
        pulse_evidence=pulse_evidence,
        staged_lineage=staged_lineage,
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
    document: PulseDocument,
    output_contract: ScanOutputContract,
    final_data_limit_bytes: int,
    static_lineage: _StaticScanLineageAdmission,
    compiled_pulse: CompiledPulseArtifact,
    adapt: _ResultAdapter,
) -> RunPlan:
    """Keep the durable sink admitted from preflight through FINAL commit."""

    if not isinstance(base, RunPlan):
        raise TypeError("base must be RunPlan")
    if type(repository) is not ScanRepository:
        raise TypeError("repository must be ScanRepository")
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    if not isinstance(static_lineage, _StaticScanLineageAdmission):
        raise TypeError("static_lineage must be admitted scan lineage")
    if not isinstance(compiled_pulse, CompiledPulseArtifact):
        raise TypeError("compiled_pulse must be CompiledPulseArtifact")
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
                document,
                compiled_pulse,
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
            run_id, source, provenance, pulse_evidence = adapted
            prepared = _prepare_dataset(
                context,
                run_id=run_id,
                document=document,
                source=source,
                output_contract=output_contract,
                provenance=provenance,
                pulse_evidence=pulse_evidence,
                staged_lineage=staged_lineage,
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
    document: PulseDocument,
    output_contract: ScanOutputContract,
    memory_limit_bytes: int,
) -> RunPlan:
    """Compile direct camera y into one canonical FINAL scan Run."""

    if not isinstance(spec, TriggeredCaptureSpec):
        raise TypeError("spec must be TriggeredCaptureSpec")
    source_schema = spec.capture.measurement.capture_contract.dataset_schema
    _require_compile_binding(
        document=document,
        source_schema=source_schema,
        output_contract=output_contract,
        execution_form=spec.pulse_request.artifact.execution_form,
        compiled_source_document_digest=(
            spec.pulse_request.artifact.source_document_digest
        ),
    )
    _admit_final_data_limit(
        source_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=0,
    )
    static_lineage = repository._admit_static_lineage(
        document,
        spec.pulse_request.artifact,
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
    ) -> tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseCaptureEvidence]:
        if type(value) is not TriggeredPipelineResult:
            raise TypeError("direct scan base plan returned another result")
        pipeline = value.capture
        return (
            pipeline.run_id,
            pipeline.dataset.snapshot,
            pipeline.dataset.provenance,
            value.lineage.evidence(),
        )

    base = compile_triggered_pipeline(
        spec,
        _retained_overhead_bytes=static_lineage.retained_upper_bound_bytes,
    )
    return _compile_scan_artifact_plan(
        base,
        repository,
        document=document,
        output_contract=output_contract,
        final_data_limit_bytes=final_data_limit,
        static_lineage=static_lineage,
        compiled_pulse=spec.pulse_request.artifact,
        adapt=adapt,
    )


def compile_occupancy_scan_artifact_plan(
    spec: TriggeredOccupancySpec,
    repository: ScanRepository,
    *,
    document: PulseDocument,
    output_contract: ScanOutputContract,
    memory_limit_bytes: int,
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile one occupancy scan and terminalize any rejected preview port."""

    try:
        return _compile_occupancy_scan_artifact_plan(
            spec,
            repository,
            document=document,
            output_contract=output_contract,
            memory_limit_bytes=memory_limit_bytes,
            preview=preview,
        )
    except BaseException as error:
        _notify_preview_failure(preview, error)
        raise


def _compile_occupancy_scan_artifact_plan(
    spec: TriggeredOccupancySpec,
    repository: ScanRepository,
    *,
    document: PulseDocument,
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
    preview_bytes = (
        0 if preview_spec is None else preview_spec.downstream_peak_bytes
    )
    _require_compile_binding(
        document=document,
        source_schema=resolved.counts_schema,
        output_contract=output_contract,
        execution_form=spec.pulse_request.artifact.execution_form,
        compiled_source_document_digest=(
            spec.pulse_request.artifact.source_document_digest
        ),
    )
    _admit_final_data_limit(
        resolved.counts_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=preview_bytes,
    )
    static_lineage = repository._admit_static_lineage(
        document,
        spec.pulse_request.artifact,
        memory_limit_bytes=memory_limit_bytes,
    )
    final_data_limit = _admit_final_data_limit(
        resolved.counts_schema,
        output_contract,
        memory_limit_bytes=memory_limit_bytes,
        retained_overhead_bytes=(
            static_lineage.retained_upper_bound_bytes
            + calibration_retained_array_nbytes(
                spec.occupancy.processor.calibration.artifact
            )
            + preview_bytes
        ),
    )

    def adapt(
        value: object,
    ) -> tuple[str, OwnedSnapshot, DatasetSealProvenance, PulseCaptureEvidence]:
        if type(value) is not TriggeredOccupancyPipelineResult:
            raise TypeError("occupancy scan base plan returned another result")
        pipeline = value.occupancy.pipeline
        return (
            pipeline.run_id,
            value.occupancy.dataset.counts,
            pipeline.dataset.provenance,
            value.lineage.evidence(),
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
        document=document,
        output_contract=output_contract,
        final_data_limit_bytes=final_data_limit,
        static_lineage=static_lineage,
        compiled_pulse=spec.pulse_request.artifact,
        adapt=adapt,
    )


__all__ = [
    "compile_direct_scan_artifact_plan",
    "compile_occupancy_scan_artifact_plan",
    "PreparedOccupancyScan",
]
