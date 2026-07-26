"""Persist fit results derived from owner-admitted Dataset artifacts.

``zlc_data`` owns fitting and the result payload.  This module is the narrow
neutral-atom seam that binds that payload to a durable Dataset source admitted
through the Experiment's frozen artifact-owner dispatch.  This repository never
imports a concrete Logic node or interprets an owner's storage.
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

from zlc_neutral_atom.artifact_dispatch import ArtifactDispatch
from .fit_reference import (
    FIT_RESULT_ARTIFACT_NAMESPACE,
    FitResultArtifactRef,
)


FIT_RESULT_ARTIFACT_SCHEMA = "zlc_neutral_atom.FitResultArtifact"

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

    __slots__ = (
        "_token",
        "_repository",
        "_source_artifact_ref",
        "_source_reference_payload",
        "_result",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("FitExecution is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        repository: "FitResultRepository",
        source_artifact_ref: object,
        source_reference_payload: bytes,
        result: FitResultBatch,
    ) -> None:
        if token is not _FIT_EXECUTION_TOKEN:
            raise PermissionError(
                "FitExecution can only be minted by FitResultRepository"
            )
        if type(repository) is not FitResultRepository:
            raise TypeError("repository must be FitResultRepository")
        if source_artifact_ref is None:
            raise TypeError("source_artifact_ref must be a durable owner reference")
        if not isinstance(source_reference_payload, bytes):
            raise TypeError("source_reference_payload must be canonical bytes")
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository", repository)
        object.__setattr__(self, "_source_artifact_ref", source_artifact_ref)
        object.__setattr__(
            self,
            "_source_reference_payload",
            source_reference_payload,
        )
        object.__setattr__(self, "_result", result)

    def _require_authority(self, repository: "FitResultRepository") -> None:
        if (
            type(self) is not FitExecution
            or self._token is not _FIT_EXECUTION_TOKEN
            or self._repository is not repository
            or self._source_artifact_ref is None
            or not isinstance(self._source_reference_payload, bytes)
            or not isinstance(self._result, FitResultBatch)
        ):
            raise PermissionError("FitExecution authority is invalid")

    @property
    def source_artifact_ref(self) -> object:
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
        source_artifact_ref: object,
        result: FitResultBatch,
    ) -> None:
        if token is not _ADMITTED_FIT_TOKEN:
            raise PermissionError(
                "AdmittedFitResult can only be minted by FitResultRepository.load"
            )
        if not isinstance(reference, FitResultArtifactRef):
            raise TypeError("reference must be FitResultArtifactRef")
        if source_artifact_ref is None:
            raise TypeError("source_artifact_ref must be a durable owner reference")
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
            or self._source_artifact_ref is None
            or not isinstance(self._result, FitResultBatch)
        ):
            raise PermissionError("AdmittedFitResult authority is invalid")

    @property
    def reference(self) -> FitResultArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def source_artifact_ref(self) -> object:
        self._require_authority()
        return self._source_artifact_ref

    @property
    def result(self) -> FitResultBatch:
        self._require_authority()
        return self._result


class FitResultRepository:
    """CAS repository for durable Dataset artifact -> FitResult derivations."""

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
        source_artifact_ref: object,
        source_reference_payload: bytes,
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
                source_reference_payload=source_reference_payload,
                result=result,
            )

    def execute(
        self,
        artifacts: ArtifactDispatch,
        source: object,
        spec: FitSpec,
        *,
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> FitExecution:
        """Fit one exact FINAL Dataset artifact without accepting a naked snapshot."""

        if not isinstance(artifacts, ArtifactDispatch):
            raise TypeError("artifacts must be ArtifactDispatch")
        if not isinstance(spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        abort_check = lambda: _require_fit_active(
            cancel_check,
            deadline_monotonic,
        )
        abort_check()
        source_reference_payload = encode(
            artifacts.encode_dataset_reference(source)
        )
        source_projection = artifacts.project_dataset(
            source,
            materialize=True,
            abort_check=abort_check,
        )
        abort_check()
        return self._execute_snapshot(
            source,
            source_reference_payload,
            source_projection.require_owned_snapshot(),
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
                    "source": decode(execution._source_reference_payload),
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
        artifacts: ArtifactDispatch,
    ) -> AdmittedFitResult:
        """Admit one result after its exact source owner revalidates lineage."""

        with self._lifecycle_lock:
            self._require_integrity()
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("reference must be FitResultArtifactRef")
            if not isinstance(artifacts, ArtifactDispatch):
                raise TypeError("artifacts must be ArtifactDispatch")
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
            source_ref = artifacts.decode_dataset_reference(manifest["source"])
            result_ref = content_ref_from_tree(manifest["result_blob"])
            source_projection = artifacts.admit_dataset_reference(source_ref)
            result = decode_fit_result_batch(
                self._store_authority.read_blob(result_ref)
            )
            validate_fit_result_source_binding(
                result,
                source_projection.ref,
                source_projection.schema,
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
