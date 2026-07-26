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
    FitResultBatch,
    FitCancelled,
    FitDeadlineExceeded,
    FitSpec,
    bind_fit,
    decode_fit_result_batch,
    encode_fit_result_batch,
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
)

from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from .fit_reference import (
    FIT_RESULT_ARTIFACT_NAMESPACE,
    FitResultArtifactRef,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import (
    ScanArtifactRef,
    scan_artifact_ref_from_tree,
    scan_artifact_ref_to_tree,
)


FIT_RESULT_ARTIFACT_SCHEMA = "zlc_neutral_atom.FitResultArtifact"

_CAPTURE_SOURCE_KIND = "capture"
_SCAN_SOURCE_KIND = "scan"
_FIT_EXECUTION_TOKEN = object()
_ADMITTED_FIT_TOKEN = object()


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

    def save(self) -> FitResultArtifactRef:
        """Publish this result."""

        return self._repository._save_execution(self)


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
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("FitResultRepository is final and cannot be subclassed")

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-fit",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
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

    def _execute_snapshot(
        self,
        source_artifact_ref: CaptureArtifactRef | ScanArtifactRef,
        snapshot,
        spec: FitSpec,
        *,
        cancel_check: Callable[[], bool] | None,
        deadline_monotonic: float | None,
    ) -> FitExecution:
        _require_fit_active(cancel_check, deadline_monotonic)
        bound = bind_fit(spec, snapshot.block.schema)
        result = bound.run(
            snapshot,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
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
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> FitExecution:
        """Fit one exact FINAL capture."""

        from zlc_neutral_atom.capture.artifact import CaptureRepository

        if type(capture_repository) is not CaptureRepository:
            raise TypeError("capture_repository must be CaptureRepository")
        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        abort_check = lambda: _require_fit_active(
            cancel_check,
            deadline_monotonic,
        )
        abort_check()
        snapshot = capture_repository.materialize_final(
            source,
            abort_check=abort_check,
        )
        abort_check()
        return self._execute_snapshot(
            source,
            snapshot,
            spec,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )

    def execute_scan(
        self,
        scan_repository,
        source: ScanArtifactRef,
        spec: FitSpec,
        *,
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> FitExecution:
        """Fit one exact FINAL scan output without accepting a naked snapshot."""

        from zlc_neutral_atom.logic_nodes.pulse_scan.repository import ScanRepository

        if type(scan_repository) is not ScanRepository:
            raise TypeError("scan_repository must be ScanRepository")
        if not isinstance(source, ScanArtifactRef):
            raise TypeError("source must be ScanArtifactRef")
        abort_check = lambda: _require_fit_active(
            cancel_check,
            deadline_monotonic,
        )
        abort_check()
        materialized = scan_repository.materialize(
            source,
            abort_check=abort_check,
        )
        abort_check()
        snapshot = materialized.snapshot
        del materialized
        return self._execute_snapshot(
            source,
            snapshot,
            spec,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )

    def _save_execution(
        self,
        execution: FitExecution,
    ) -> FitResultArtifactRef:
        with self._lifecycle_lock:
            self._require_integrity()
            FitExecution._require_authority(execution, self)
            result_payload = encode_fit_result_batch(execution._result)
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
    ) -> AdmittedFitResult:
        """Admit one result after its exact source owner revalidates lineage."""

        with self._lifecycle_lock:
            self._require_integrity()
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("reference must be FitResultArtifactRef")
            if reference.repository_id != self.repository_id:
                raise ValueError("FitResultArtifactRef belongs to another repository")
            manifest_payload = self._store_authority.read_manifest(
                FIT_RESULT_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
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
            if isinstance(source_ref, CaptureArtifactRef):
                from zlc_neutral_atom.capture.artifact import CaptureRepository

                if type(capture_repository) is not CaptureRepository:
                    raise TypeError(
                        "capture_repository is required for a capture fit result"
                    )
                source_admission = capture_repository.admit(source_ref)
                source_artifact = source_admission.artifact
                source_schema = source_artifact.frame_source.schema
                source_dataset_ref = source_artifact.frame_source.ref(
                    source_artifact.provenance.generation
                )
            else:
                from zlc_neutral_atom.logic_nodes.pulse_scan.repository import (
                    ScanRepository,
                )

                if type(scan_repository) is not ScanRepository:
                    raise TypeError(
                        "scan_repository is required for a scan fit result"
                    )
                materialized = scan_repository.materialize(source_ref)
                source_dataset_ref = materialized.snapshot.ref
                source_schema = materialized.snapshot.block.schema
            result = decode_fit_result_batch(
                self._store_authority.read_blob(result_ref)
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
