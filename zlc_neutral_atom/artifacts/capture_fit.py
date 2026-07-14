"""Persist one concrete committed-capture -> fit-result derivation.

``zlc_data`` owns fitting and the persisted result identity.  This module owns
only the neutral-atom seam that resolves a committed ``CaptureArtifact`` and
binds it to those data-owned bytes.  It deliberately provides no generic
analysis repository, workflow, compatibility reader, plugin registry, or
commit journal.
"""

from __future__ import annotations

from pathlib import Path
import threading

from zlc_data import (
    FIT_RESULT_ARTIFACT_NAMESPACE,
    FitResultArtifactRef,
    FitResultBatch,
    FitSpec,
    OwnedSnapshot,
    bind_fit,
    decode_fit_result_batch,
    encode_fit_result_batch,
    validate_fit_result_binding,
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

from .capture import AdmittedCapture, CaptureRepository


CAPTURE_FIT_RESULT_ARTIFACT_SCHEMA = "zlc_neutral_atom.CaptureFitResultArtifact"

_FIT_EXECUTION_TOKEN = object()
_ADMITTED_FIT_TOKEN = object()
_DEFAULT_MATERIALIZATION_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


def _capture_snapshot(
    admission: AdmittedCapture,
    memory_limit_bytes: int,
) -> OwnedSnapshot:
    if type(admission) is not AdmittedCapture:
        raise TypeError("capture source must be an AdmittedCapture")
    capture = admission.artifact
    block = capture.frame_source.materialize(
        memory_limit_bytes=memory_limit_bytes,
    )
    return OwnedSnapshot(
        block.ref(capture.provenance.generation),
        block,
    )


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
    """Non-serializable result minted only by the capture-fit executor."""

    __slots__ = ("_token", "_repository", "_source_admission", "_result")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("FitExecution is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        repository: "CaptureFitResultRepository",
        source_admission: AdmittedCapture,
        result: FitResultBatch,
    ) -> None:
        if token is not _FIT_EXECUTION_TOKEN:
            raise PermissionError(
                "FitExecution can only be minted by "
                "CaptureFitResultRepository.execute"
            )
        if type(repository) is not CaptureFitResultRepository:
            raise TypeError("repository must be CaptureFitResultRepository")
        if type(source_admission) is not AdmittedCapture:
            raise TypeError("source_admission must be AdmittedCapture")
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository", repository)
        object.__setattr__(self, "_source_admission", source_admission)
        object.__setattr__(self, "_result", result)

    def _require_authority(
        self,
        repository: "CaptureFitResultRepository",
    ) -> None:
        if (
            type(self) is not FitExecution
            or self._token is not _FIT_EXECUTION_TOKEN
            or self._repository is not repository
        ):
            raise PermissionError("FitExecution authority is invalid")

    @property
    def source_capture_ref(self) -> CaptureArtifactRef:
        self._require_authority(self._repository)
        return self._source_admission.reference

    @property
    def result(self) -> FitResultBatch:
        self._require_authority(self._repository)
        return self._result

    def save(self) -> FitResultArtifactRef:
        return self._repository._save_execution(self)


class AdmittedCaptureFitResult(_ProcessLocalProof):
    """Content-integral, capture-bound result admitted from a trusted root.

    Admission rechecks the persisted source binding but deliberately does not
    rerun the numerical solver.  It is not a cryptographic attestation against
    an external actor that can rewrite the repository filesystem.
    """

    __slots__ = ("_token", "_reference", "_source_capture_ref", "_result")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError(
            "AdmittedCaptureFitResult is final and cannot be subclassed"
        )

    def __init__(
        self,
        token: object,
        *,
        reference: FitResultArtifactRef,
        source_capture_ref: CaptureArtifactRef,
        result: FitResultBatch,
    ) -> None:
        if token is not _ADMITTED_FIT_TOKEN:
            raise PermissionError(
                "AdmittedCaptureFitResult can only be minted by "
                "CaptureFitResultRepository.load"
            )
        if not isinstance(reference, FitResultArtifactRef):
            raise TypeError("reference must be FitResultArtifactRef")
        if not isinstance(source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_source_capture_ref", source_capture_ref)
        object.__setattr__(self, "_result", result)

    def _require_authority(self) -> None:
        if (
            type(self) is not AdmittedCaptureFitResult
            or self._token is not _ADMITTED_FIT_TOKEN
        ):
            raise PermissionError(
                "AdmittedCaptureFitResult authority is invalid"
            )

    @property
    def reference(self) -> FitResultArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def source_capture_ref(self) -> CaptureArtifactRef:
        self._require_authority()
        return self._source_capture_ref

    @property
    def result(self) -> FitResultBatch:
        self._require_authority()
        return self._result


class CaptureFitResultRepository:
    """CAS repository for one concrete Capture -> FitResult derivation."""

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
        raise TypeError(
            "CaptureFitResultRepository is final and cannot be subclassed"
        )

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-capture-fit",
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
        lease = RepositoryRootLease(
            self.root,
            owner=f"capture-fit:{self.repository_id}",
        )
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
            raise AttributeError("CaptureFitResultRepository is immutable")
        object.__setattr__(self, name, value)

    def _require_integrity(self) -> None:
        if (
            type(self) is not CaptureFitResultRepository
            or self.root != self._root_lease.root
            or self._store_authority.root != self.root / "content"
            or self._store.authority() is not self._store_authority
        ):
            raise RuntimeError("capture-fit repository authority changed")
        self._root_lease.require_active()

    def close(self) -> None:
        with self._lifecycle_lock:
            self._root_lease.close()

    def execute(
        self,
        source: AdmittedCapture,
        spec: FitSpec,
    ) -> FitExecution:
        with self._lifecycle_lock:
            self._require_integrity()
        snapshot = _capture_snapshot(
            source,
            self.materialization_memory_limit_bytes,
        )
        result = bind_fit(spec, snapshot.block.schema).run(snapshot)
        with self._lifecycle_lock:
            self._require_integrity()
            return FitExecution(
                _FIT_EXECUTION_TOKEN,
                repository=self,
                source_admission=source,
                result=result,
            )

    def _save_execution(
        self,
        execution: FitExecution,
    ) -> FitResultArtifactRef:
        with self._lifecycle_lock:
            self._require_integrity()
            FitExecution._require_authority(execution, self)
            with self._root_lease.borrow() as lease_borrow:
                lease_borrow.require_active()
                result_ref = self._store_authority.put_blob(
                    encode_fit_result_batch(execution._result)
                )
                manifest_payload = encode(
                    {
                        "schema": CAPTURE_FIT_RESULT_ARTIFACT_SCHEMA,
                        "repository_id": self.repository_id,
                        "source_capture_ref": capture_artifact_ref_to_tree(
                            execution._source_admission.reference
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
        capture_repository: CaptureRepository,
    ) -> AdmittedCaptureFitResult:
        with self._lifecycle_lock:
            self._require_integrity()
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("reference must be FitResultArtifactRef")
            if reference.repository_id != self.repository_id:
                raise ValueError(
                    "FitResultArtifactRef belongs to another repository"
                )
            if type(capture_repository) is not CaptureRepository:
                raise TypeError("capture_repository must be CaptureRepository")
            manifest_payload = self._store_authority.read_manifest(
                FIT_RESULT_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
            )
            manifest = exact_mapping(
                decode(manifest_payload),
                {
                    "schema",
                    "repository_id",
                    "source_capture_ref",
                    "result_blob",
                },
                CAPTURE_FIT_RESULT_ARTIFACT_SCHEMA,
            )
            if manifest["repository_id"] != self.repository_id:
                raise ValueError(
                    "capture-fit manifest belongs to another repository"
                )
            if encode(manifest) != manifest_payload:
                raise ValueError(
                    "capture-fit manifest is not canonical current schema"
                )
            source_ref = capture_artifact_ref_from_tree(
                manifest["source_capture_ref"]
            )
            result = decode_fit_result_batch(
                self._store_authority.read_blob(
                    content_ref_from_tree(manifest["result_blob"])
                )
            )
            source = capture_repository.admit(source_ref)
            validate_fit_result_binding(
                result,
                _capture_snapshot(
                    source,
                    self.materialization_memory_limit_bytes,
                ),
            )
            return AdmittedCaptureFitResult(
                _ADMITTED_FIT_TOKEN,
                reference=reference,
                source_capture_ref=source_ref,
                result=result,
            )


__all__ = [
    "AdmittedCaptureFitResult",
    "CAPTURE_FIT_RESULT_ARTIFACT_SCHEMA",
    "CaptureFitResultRepository",
    "FitExecution",
]
