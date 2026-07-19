"""Persist fit results derived from committed capture or scan artifacts.

``zlc_data`` owns fitting and the result payload.  This module is the narrow
neutral-atom seam that binds that payload to either of the two durable dataset
sources that exist today.  The source set is deliberately closed: this is not
an analysis repository, source registry, workflow engine, or plugin surface.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
import time

from zlc_data import (
    BoundFit,
    FitResultBatch,
    FitCancelled,
    FitDeadlineExceeded,
    FitSpec,
    bind_fit,
    bound_fit_execution_peak_upper_bound_nbytes,
    decode_fit_result_batch,
    encode_fit_result_batch,
    fit_result_decode_additional_peak_upper_bound_nbytes,
    fit_result_encode_additional_peak_upper_bound_nbytes,
    fit_binding_additional_peak_upper_bound_nbytes,
    fit_binding_retained_upper_bound_nbytes,
    fit_result_retained_upper_bound_nbytes,
    fit_result_source_validation_additional_peak_upper_bound_nbytes,
    validate_fit_result_source_binding,
)
from zlc_storage import (
    ContentAddressedStore,
    RepositoryRootLease,
    canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    positive_integer,
)

from zlc_neutral_atom.capture_reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.fit_reference import (
    FIT_RESULT_ARTIFACT_NAMESPACE,
    FitResultArtifactRef,
)
from zlc_neutral_atom.scan.reference import (
    ScanArtifactRef,
    scan_artifact_ref_from_tree,
    scan_artifact_ref_to_tree,
)


FIT_RESULT_ARTIFACT_SCHEMA = "zlc_neutral_atom.FitResultArtifact"

_CAPTURE_SOURCE_KIND = "capture"
_SCAN_SOURCE_KIND = "scan"
_FIT_EXECUTION_TOKEN = object()
_ADMITTED_FIT_TOKEN = object()
_DEFAULT_MATERIALIZATION_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_FIT_EXECUTION_FIXED_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_RESULT_BLOB_BYTES = 64 * 1024 * 1024
_FIT_SAVE_REPOSITORY_FIXED_BYTES = (
    _FIT_EXECUTION_FIXED_BYTES + 4 * _MAX_MANIFEST_BYTES
)
_FIT_LOAD_REPOSITORY_FIXED_BYTES = (
    _FIT_EXECUTION_FIXED_BYTES + 4 * _MAX_MANIFEST_BYTES
)


def _require_fit_active(
    cancel_check: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    if cancel_check is not None and cancel_check():
        raise FitCancelled("fit was cancelled during source materialization")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired during source materialization")


def _source_ref_to_tree(
    source: CaptureArtifactRef | ScanArtifactRef,
) -> dict[str, object]:
    if isinstance(source, CaptureArtifactRef):
        return {
            "kind": _CAPTURE_SOURCE_KIND,
            "ref": capture_artifact_ref_to_tree(source),
        }
    if isinstance(source, ScanArtifactRef):
        return {
            "kind": _SCAN_SOURCE_KIND,
            "ref": scan_artifact_ref_to_tree(source),
        }
    raise TypeError("fit source must be CaptureArtifactRef or ScanArtifactRef")


def _source_ref_from_tree(
    tree: object,
) -> CaptureArtifactRef | ScanArtifactRef:
    data = exact_mapping(
        tree,
        {"kind", "ref"},
        "fit source artifact",
        discriminator=None,
    )
    kind = data["kind"]
    if kind == _CAPTURE_SOURCE_KIND:
        source = capture_artifact_ref_from_tree(data["ref"])
    elif kind == _SCAN_SOURCE_KIND:
        source = scan_artifact_ref_from_tree(data["ref"])
    else:
        raise ValueError("fit source artifact kind is not current")
    if _source_ref_to_tree(source) != data:
        raise ValueError("fit source artifact tree is typed but non-canonical")
    return source


class _ProcessLocalProof:
    __slots__ = ()

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self):
        raise TypeError(
            f"{type(self).__name__} is process-local and cannot be serialized"
        )

    def __reduce_ex__(self, _protocol: int):
        return self.__reduce__()


class FitExecution(_ProcessLocalProof):
    """Non-serializable result minted only from an admitted durable source."""

    __slots__ = ("_token", "_repository", "_source_artifact_ref", "_result")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("FitExecution is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        repository: "FitResultRepository",
        source_artifact_ref: CaptureArtifactRef | ScanArtifactRef,
        result: FitResultBatch,
    ) -> None:
        if token is not _FIT_EXECUTION_TOKEN:
            raise PermissionError(
                "FitExecution can only be minted by FitResultRepository"
            )
        if type(repository) is not FitResultRepository:
            raise TypeError("repository must be FitResultRepository")
        if not isinstance(source_artifact_ref, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError(
                "source_artifact_ref must be CaptureArtifactRef or ScanArtifactRef"
            )
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository", repository)
        object.__setattr__(self, "_source_artifact_ref", source_artifact_ref)
        object.__setattr__(self, "_result", result)

    def _require_authority(self, repository: "FitResultRepository") -> None:
        if (
            type(self) is not FitExecution
            or self._token is not _FIT_EXECUTION_TOKEN
            or self._repository is not repository
            or not isinstance(
                self._source_artifact_ref,
                (CaptureArtifactRef, ScanArtifactRef),
            )
            or not isinstance(self._result, FitResultBatch)
        ):
            raise PermissionError("FitExecution authority is invalid")

    @property
    def source_artifact_ref(self) -> CaptureArtifactRef | ScanArtifactRef:
        self._require_authority(self._repository)
        return self._source_artifact_ref

    @property
    def result(self) -> FitResultBatch:
        self._require_authority(self._repository)
        return self._result

    def save(
        self,
        *,
        operation_memory_limit_bytes: int | None = None,
    ) -> FitResultArtifactRef:
        """Publish under a bounded additional-workspace memory budget.

        Interactive hosts pass only the memory still available after their
        resident Figure/result front.  Headless notebook calls may omit the
        argument and use the repository's installation-level bounded default;
        the already-resident result is never counted as new workspace.
        """

        return self._repository._save_execution(
            self,
            operation_memory_limit_bytes=operation_memory_limit_bytes,
        )


class AdmittedFitResult(_ProcessLocalProof):
    """Content-integral result rebound to its exact committed source artifact."""

    __slots__ = ("_token", "_reference", "_source_artifact_ref", "_result")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("AdmittedFitResult is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        reference: FitResultArtifactRef,
        source_artifact_ref: CaptureArtifactRef | ScanArtifactRef,
        result: FitResultBatch,
    ) -> None:
        if token is not _ADMITTED_FIT_TOKEN:
            raise PermissionError(
                "AdmittedFitResult can only be minted by FitResultRepository.load"
            )
        if not isinstance(reference, FitResultArtifactRef):
            raise TypeError("reference must be FitResultArtifactRef")
        if not isinstance(source_artifact_ref, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError(
                "source_artifact_ref must be CaptureArtifactRef or ScanArtifactRef"
            )
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_source_artifact_ref", source_artifact_ref)
        object.__setattr__(self, "_result", result)

    def _require_authority(self) -> None:
        if (
            type(self) is not AdmittedFitResult
            or self._token is not _ADMITTED_FIT_TOKEN
            or not isinstance(self._reference, FitResultArtifactRef)
            or not isinstance(
                self._source_artifact_ref,
                (CaptureArtifactRef, ScanArtifactRef),
            )
            or not isinstance(self._result, FitResultBatch)
        ):
            raise PermissionError("AdmittedFitResult authority is invalid")

    @property
    def reference(self) -> FitResultArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def source_artifact_ref(self) -> CaptureArtifactRef | ScanArtifactRef:
        self._require_authority()
        return self._source_artifact_ref

    @property
    def result(self) -> FitResultBatch:
        self._require_authority()
        return self._result


class FitResultRepository:
    """CAS repository for Capture/Scan -> FitResult derivations."""

    __slots__ = (
        "root",
        "repository_id",
        "_root_lease",
        "_lifecycle_lock",
        "_store",
        "_store_authority",
        "materialization_memory_limit_bytes",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("FitResultRepository is final and cannot be subclassed")

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-fit",
        materialization_memory_limit_bytes: int = (
            _DEFAULT_MATERIALIZATION_MEMORY_LIMIT_BYTES
        ),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
        self.materialization_memory_limit_bytes = positive_integer(
            materialization_memory_limit_bytes,
            "materialization_memory_limit_bytes",
        )
        lease = RepositoryRootLease(self.root)
        self._root_lease = lease
        self._lifecycle_lock = threading.RLock()
        try:
            store = ContentAddressedStore(self.root / "content")
            self._store = store
            self._store_authority = store.authority()
        except BaseException:
            lease.close()
            raise

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("FitResultRepository is immutable")
        object.__setattr__(self, name, value)

    def _require_integrity(self) -> None:
        if (
            type(self) is not FitResultRepository
            or self.root != self._root_lease.root
            or self._store_authority.root != self.root / "content"
            or self._store.authority() is not self._store_authority
        ):
            raise RuntimeError("fit-result repository authority changed")
        self._root_lease.require_active()

    def close(self) -> None:
        with self._lifecycle_lock:
            self._root_lease.close()

    def _aggregate_limit(self, memory_limit_bytes: int | None) -> int:
        if memory_limit_bytes is None:
            return self.materialization_memory_limit_bytes
        return min(
            self.materialization_memory_limit_bytes,
            positive_integer(memory_limit_bytes, "memory_limit_bytes"),
        )

    def source_materialization_memory_limit_bytes(
        self,
        memory_limit_bytes: int | None = None,
    ) -> int:
        """Return the source-owner share of one aggregate Fit execution budget."""

        with self._lifecycle_lock:
            self._require_integrity()
            limit = self._aggregate_limit(memory_limit_bytes)
        source_limit = limit - _FIT_EXECUTION_FIXED_BYTES
        if source_limit <= 0:
            raise MemoryError("fit fixed state leaves no source materialization budget")
        return int(source_limit)

    def _execute_snapshot(
        self,
        source_artifact_ref: CaptureArtifactRef | ScanArtifactRef,
        snapshot,
        spec: FitSpec,
        *,
        aggregate_limit: int,
        cancel_check: Callable[[], bool] | None,
        deadline_monotonic: float | None,
        bound: BoundFit,
        binding_retained_bytes: int,
    ) -> FitExecution:
        from zlc_neutral_atom.runtime.dataset import dataset_storage_nbytes

        _require_fit_active(cancel_check, deadline_monotonic)
        source_bytes = dataset_storage_nbytes(snapshot.block.schema)
        if (
            type(bound) is not BoundFit
            or bound.spec != spec
            or bound.expected_schema != snapshot.block.schema
        ):
            raise ValueError("preflight Fit binding differs from materialized source")
        binding_retained_bytes = positive_integer(
            binding_retained_bytes,
            "binding_retained_bytes",
        )
        peak = (
            source_bytes
            + binding_retained_bytes
            + bound_fit_execution_peak_upper_bound_nbytes(bound)
            + _FIT_EXECUTION_FIXED_BYTES
        )
        if peak > aggregate_limit:
            raise MemoryError(
                f"fit execution peak {peak} exceeds aggregate memory limit "
                f"{aggregate_limit}"
            )
        result = bound.run(
            snapshot,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        retained = (
            source_bytes
            + binding_retained_bytes
            + fit_result_retained_upper_bound_nbytes(result)
            + _FIT_EXECUTION_FIXED_BYTES
        )
        if retained > aggregate_limit:
            raise MemoryError(
                f"fit retained state {retained} exceeds aggregate memory limit "
                f"{aggregate_limit}"
            )
        with self._lifecycle_lock:
            self._require_integrity()
            return FitExecution(
                _FIT_EXECUTION_TOKEN,
                repository=self,
                source_artifact_ref=source_artifact_ref,
                result=result,
            )

    def execute_capture(
        self,
        capture_repository,
        source: CaptureArtifactRef,
        spec: FitSpec,
        *,
        memory_limit_bytes: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> FitExecution:
        """Fit one exact FINAL capture under one source/solver budget."""

        from .capture import CaptureRepository

        if type(capture_repository) is not CaptureRepository:
            raise TypeError("capture_repository must be CaptureRepository")
        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        with self._lifecycle_lock:
            self._require_integrity()
            limit = self._aggregate_limit(memory_limit_bytes)
        abort_check = lambda: _require_fit_active(
            cancel_check,
            deadline_monotonic,
        )
        source_limit = limit - _FIT_EXECUTION_FIXED_BYTES
        if source_limit <= 0:
            raise MemoryError("fit fixed state leaves no capture source budget")
        abort_check()
        inspection = capture_repository.inspect_final(
            source,
            memory_limit_bytes=source_limit,
        )
        source_phase = (
            _FIT_EXECUTION_FIXED_BYTES
            + inspection.materialization_peak_upper_bound_bytes
        )
        if source_phase > limit:
            raise MemoryError(
                f"fit capture materialization peak {source_phase} exceeds "
                f"aggregate memory limit {limit}"
            )
        abort_check()
        source_schema = inspection.dataset_schema
        binding_additional = fit_binding_additional_peak_upper_bound_nbytes(
            spec,
            source_schema,
        )
        binding_phase = (
            _FIT_EXECUTION_FIXED_BYTES
            + inspection.inspection_retained_upper_bound_bytes
            + binding_additional
        )
        if binding_phase > limit:
            raise MemoryError(
                f"fit binding peak {binding_phase} exceeds aggregate memory "
                f"limit {limit}"
            )
        # Validate the request before expensive source materialization, then
        # release the inspection-derived schema/binding.  The repository will
        # decode an equal but separately owned schema with the snapshot; keeping
        # both would make a large explicit layout escape the aggregate budget.
        prebound = bind_fit(spec, source_schema)
        from zlc_neutral_atom.runtime.dataset import dataset_storage_nbytes

        execution_preflight = (
            _FIT_EXECUTION_FIXED_BYTES
            + dataset_storage_nbytes(source_schema)
            + fit_binding_retained_upper_bound_nbytes(spec, source_schema)
            + bound_fit_execution_peak_upper_bound_nbytes(prebound)
        )
        if execution_preflight > limit:
            raise MemoryError(
                f"fit execution peak {execution_preflight} exceeds aggregate "
                f"memory limit {limit}"
            )
        rebind_preflight = (
            _FIT_EXECUTION_FIXED_BYTES
            + dataset_storage_nbytes(source_schema)
            + binding_additional
        )
        if rebind_preflight > limit:
            raise MemoryError(
                f"fit snapshot rebind peak {rebind_preflight} exceeds aggregate "
                f"memory limit {limit}"
            )
        del prebound, source_schema, inspection
        abort_check()
        snapshot = capture_repository.materialize_final(
            source,
            memory_limit_bytes=source_limit,
            abort_check=abort_check,
        )
        abort_check()
        source_bytes = dataset_storage_nbytes(snapshot.block.schema)
        rebind_peak = (
            _FIT_EXECUTION_FIXED_BYTES
            + source_bytes
            + fit_binding_additional_peak_upper_bound_nbytes(
                spec,
                snapshot.block.schema,
            )
        )
        if rebind_peak > limit:
            raise MemoryError(
                f"fit snapshot rebind peak {rebind_peak} exceeds aggregate "
                f"memory limit {limit}"
            )
        bound = bind_fit(spec, snapshot.block.schema)
        binding_retained = fit_binding_retained_upper_bound_nbytes(
            spec,
            snapshot.block.schema,
        )
        return self._execute_snapshot(
            source,
            snapshot,
            spec,
            aggregate_limit=limit,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
            bound=bound,
            binding_retained_bytes=binding_retained,
        )

    def execute_scan(
        self,
        scan_repository,
        source: ScanArtifactRef,
        spec: FitSpec,
        *,
        memory_limit_bytes: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> FitExecution:
        """Fit one exact FINAL scan output without accepting a naked snapshot."""

        from zlc_neutral_atom.scan.repository import ScanRepository

        if type(scan_repository) is not ScanRepository:
            raise TypeError("scan_repository must be ScanRepository")
        if not isinstance(source, ScanArtifactRef):
            raise TypeError("source must be ScanArtifactRef")
        with self._lifecycle_lock:
            self._require_integrity()
            limit = self._aggregate_limit(memory_limit_bytes)
        abort_check = lambda: _require_fit_active(
            cancel_check,
            deadline_monotonic,
        )
        source_limit = limit - _FIT_EXECUTION_FIXED_BYTES
        if source_limit <= 0:
            raise MemoryError("fit fixed state leaves no scan source budget")
        abort_check()
        inspection = scan_repository.inspect_final(
            source,
            memory_limit_bytes=source_limit,
        )
        source_phase = (
            _FIT_EXECUTION_FIXED_BYTES
            + inspection.materialization_peak_upper_bound_bytes
        )
        if source_phase > limit:
            raise MemoryError(
                f"fit scan materialization peak {source_phase} exceeds aggregate "
                f"memory limit {limit}"
            )
        abort_check()
        source_schema = inspection.output_schema
        binding_additional = fit_binding_additional_peak_upper_bound_nbytes(
            spec,
            source_schema,
        )
        binding_phase = (
            _FIT_EXECUTION_FIXED_BYTES
            + inspection.inspection_retained_upper_bound_bytes
            + binding_additional
        )
        if binding_phase > limit:
            raise MemoryError(
                f"fit binding peak {binding_phase} exceeds aggregate memory "
                f"limit {limit}"
            )
        prebound = bind_fit(spec, source_schema)
        from zlc_neutral_atom.runtime.dataset import dataset_storage_nbytes

        execution_preflight = (
            _FIT_EXECUTION_FIXED_BYTES
            + dataset_storage_nbytes(source_schema)
            + fit_binding_retained_upper_bound_nbytes(spec, source_schema)
            + bound_fit_execution_peak_upper_bound_nbytes(prebound)
        )
        if execution_preflight > limit:
            raise MemoryError(
                f"fit execution peak {execution_preflight} exceeds aggregate "
                f"memory limit {limit}"
            )
        rebind_preflight = (
            _FIT_EXECUTION_FIXED_BYTES
            + dataset_storage_nbytes(source_schema)
            + binding_additional
        )
        if rebind_preflight > limit:
            raise MemoryError(
                f"fit snapshot rebind peak {rebind_preflight} exceeds aggregate "
                f"memory limit {limit}"
            )
        del prebound, source_schema, inspection
        abort_check()
        materialized = scan_repository.materialize(
            source,
            memory_limit_bytes=source_limit,
            abort_check=abort_check,
        )
        abort_check()
        snapshot = materialized.snapshot
        del materialized
        source_bytes = dataset_storage_nbytes(snapshot.block.schema)
        rebind_peak = (
            _FIT_EXECUTION_FIXED_BYTES
            + source_bytes
            + fit_binding_additional_peak_upper_bound_nbytes(
                spec,
                snapshot.block.schema,
            )
        )
        if rebind_peak > limit:
            raise MemoryError(
                f"fit snapshot rebind peak {rebind_peak} exceeds aggregate "
                f"memory limit {limit}"
            )
        bound = bind_fit(spec, snapshot.block.schema)
        binding_retained = fit_binding_retained_upper_bound_nbytes(
            spec,
            snapshot.block.schema,
        )
        return self._execute_snapshot(
            source,
            snapshot,
            spec,
            aggregate_limit=limit,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
            bound=bound,
            binding_retained_bytes=binding_retained,
        )

    def _save_execution(
        self,
        execution: FitExecution,
        *,
        operation_memory_limit_bytes: int | None,
    ) -> FitResultArtifactRef:
        with self._lifecycle_lock:
            self._require_integrity()
            FitExecution._require_authority(execution, self)
            limit = self._aggregate_limit(operation_memory_limit_bytes)
            required = (
                fit_result_encode_additional_peak_upper_bound_nbytes(
                    execution._result
                )
                + _FIT_SAVE_REPOSITORY_FIXED_BYTES
            )
            if required > limit:
                raise MemoryError(
                    f"fit save additional workspace {required} exceeds "
                    f"operation memory limit {limit}"
                )
            result_payload = encode_fit_result_batch(execution._result)
            if len(result_payload) > _MAX_RESULT_BLOB_BYTES:
                raise ValueError("fit result blob exceeds repository limit")
            result_ref = self._store_authority.put_blob(result_payload)
            manifest_payload = encode(
                {
                    "schema": FIT_RESULT_ARTIFACT_SCHEMA,
                    "repository_id": self.repository_id,
                    "source": _source_ref_to_tree(
                        execution._source_artifact_ref
                    ),
                    "result_blob": content_ref_to_tree(result_ref),
                }
            )
            if len(manifest_payload) > _MAX_MANIFEST_BYTES:
                raise ValueError("fit-result manifest exceeds repository limit")
            stored = self._store_authority.publish_manifest(
                FIT_RESULT_ARTIFACT_NAMESPACE,
                manifest_payload,
            )
            return FitResultArtifactRef(
                self.repository_id,
                stored.content.digest,
            )

    def load(
        self,
        reference: FitResultArtifactRef,
        *,
        capture_repository=None,
        scan_repository=None,
        memory_limit_bytes: int | None = None,
    ) -> AdmittedFitResult:
        """Admit one result after its exact source owner revalidates lineage."""

        with self._lifecycle_lock:
            self._require_integrity()
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("reference must be FitResultArtifactRef")
            if reference.repository_id != self.repository_id:
                raise ValueError("FitResultArtifactRef belongs to another repository")
            limit = self._aggregate_limit(memory_limit_bytes)
            if _FIT_LOAD_REPOSITORY_FIXED_BYTES > limit:
                raise MemoryError(
                    "fit load repository workspace exceeds aggregate memory limit"
                )
            manifest_payload = self._store_authority.read_manifest(
                FIT_RESULT_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=_MAX_MANIFEST_BYTES,
            )
            manifest = exact_mapping(
                decode(manifest_payload),
                {"schema", "repository_id", "source", "result_blob"},
                FIT_RESULT_ARTIFACT_SCHEMA,
            )
            if manifest["repository_id"] != self.repository_id:
                raise ValueError("fit-result manifest belongs to another repository")
            if encode(manifest) != manifest_payload:
                raise ValueError(
                    "fit-result manifest is not canonical current schema"
                )
            source_ref = _source_ref_from_tree(manifest["source"])
            result_ref = content_ref_from_tree(manifest["result_blob"])
            if result_ref.size > _MAX_RESULT_BLOB_BYTES:
                raise ValueError("fit result blob exceeds repository limit")
            inspection_limit = limit - _FIT_LOAD_REPOSITORY_FIXED_BYTES
            if inspection_limit <= 0:
                raise MemoryError("fit load leaves no source inspection budget")
            if isinstance(source_ref, CaptureArtifactRef):
                from .capture import CaptureRepository

                if type(capture_repository) is not CaptureRepository:
                    raise TypeError(
                        "capture_repository is required for a capture fit result"
                    )
                inspection = capture_repository.inspect_final(
                    source_ref,
                    memory_limit_bytes=inspection_limit,
                )
                source_dataset_ref = inspection.dataset_revision_ref
                source_schema = inspection.dataset_schema
                inspection_retained = (
                    inspection.inspection_retained_upper_bound_bytes
                )
                inspection_peak = (
                    inspection.inspection_decode_peak_upper_bound_bytes
                )
            else:
                from zlc_neutral_atom.scan.repository import ScanRepository

                if type(scan_repository) is not ScanRepository:
                    raise TypeError(
                        "scan_repository is required for a scan fit result"
                    )
                inspection = scan_repository.inspect_final(
                    source_ref,
                    memory_limit_bytes=inspection_limit,
                )
                source_dataset_ref = inspection.output_dataset_ref
                source_schema = inspection.output_schema
                inspection_retained = (
                    inspection.inspection_retained_upper_bound_bytes
                )
                inspection_peak = (
                    inspection.inspection_decode_peak_upper_bound_bytes
                )

            inspection_phase = (
                _FIT_LOAD_REPOSITORY_FIXED_BYTES + inspection_peak
            )
            codec_phase = (
                _FIT_LOAD_REPOSITORY_FIXED_BYTES
                + inspection_retained
                + fit_result_decode_additional_peak_upper_bound_nbytes(
                    result_ref.size
                )
            )
            aggregate_predecode = max(inspection_phase, codec_phase)
            if aggregate_predecode > limit:
                raise MemoryError(
                    f"fit load aggregate predecode peak {aggregate_predecode} "
                    f"exceeds memory limit {limit}"
                )
            result = decode_fit_result_batch(
                self._store_authority.read_blob(
                    result_ref,
                    max_bytes=_MAX_RESULT_BLOB_BYTES,
                )
            )
            result_bytes = fit_result_retained_upper_bound_nbytes(result)
            validation_peak = (
                fit_result_source_validation_additional_peak_upper_bound_nbytes(
                    result,
                    source_schema,
                )
            )
            validation_phase = (
                _FIT_LOAD_REPOSITORY_FIXED_BYTES
                + inspection_retained
                + result_bytes
                + validation_peak
            )
            if validation_phase > limit:
                raise MemoryError(
                    f"fit load source-validation peak {validation_phase} exceeds "
                    f"aggregate memory limit {limit}"
                )

            validate_fit_result_source_binding(
                result,
                source_dataset_ref,
                source_schema,
            )
            return AdmittedFitResult(
                _ADMITTED_FIT_TOKEN,
                reference=reference,
                source_artifact_ref=source_ref,
                result=result,
            )


__all__ = [
    "AdmittedFitResult",
    "FIT_RESULT_ARTIFACT_SCHEMA",
    "FitExecution",
    "FitResultRepository",
]
