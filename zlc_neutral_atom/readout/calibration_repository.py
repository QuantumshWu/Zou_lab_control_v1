"""Trusted final-commit repository for immutable readout calibrations.

The persistent reference names both the calibration value and the complete
derivation evidence that made it authoritative.  ``load`` is intentionally an
inspection operation.  Runtime consumers require an :class:`AdmittedCalibration`
minted only after this repository reloads the exact source CaptureArtifact and
rederives its FrameContract/source binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any
import weakref

from zlc_storage import (
    CanonicalDecodeLimits,
    ContentAddressedStore,
    ContentRef,
    ContentSizeLimitError,
    ContentStoreAuthority,
    RepositoryRootLease,
    canonical_digest,
    canonical_text as _canonical_text,
    decode,
    encode,
    sha256_digest,
    sha256_text as _sha256,
)

from zlc_neutral_atom.capture_reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitKind,
    CommitRecovery,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishVisibilityUnknown,
    PublishedManifest,
    RepositoryCommitCoordinator,
)
from zlc_neutral_atom.runtime.run import (
    CleanupReport,
    PostSafetyContext,
    RunContext,
    RunMode,
    RunPlan,
)

from .analysis import (
    CALIBRATION_ANALYSIS_ALGORITHM_ID,
    CALIBRATION_ANALYSIS_ALGORITHM_VERSION,
    CalibrationAnalysisDiagnostics,
    CalibrationAnalysisRequest,
    CalibrationAnalysisResult,
    CalibrationWorkPlan,
    _prepare_calibration_work,
    analyze_calibration,
    validate_calibration_analysis_contract,
    validate_calibration_partition_against_source,
)
from .analysis_codec import (
    decode_calibration_analysis_diagnostics,
    decode_calibration_analysis_request,
    decode_calibration_work_plan,
    encode_calibration_analysis_diagnostics,
    encode_calibration_analysis_request,
    encode_calibration_work_plan,
)
from .calibration import (
    DEFAULT_CALIBRATION_RESOURCE_POLICY,
    CalibrationArtifact,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    CalibrationResourceSummary,
    CalibrationSourceBinding,
    calibration_resource_summary,
    validate_calibration_artifact_resources,
    validate_calibration_resource_summary,
)
from .calibration_codec import (
    CALIBRATION_ARTIFACT_SCHEMA,
    decode_calibration_artifact,
    encode_calibration_artifact,
    encode_calibration_source_binding,
)
from .calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)
from .codec import encode_frame_contract

if TYPE_CHECKING:
    from zlc_neutral_atom.artifacts.capture import (
        AdmittedCapture,
        CaptureArtifact,
        CaptureRepository,
    )
else:
    # Keep runtime annotation introspection total without eagerly importing the
    # implementation module and recreating the acquisition/readout cycle.
    AdmittedCapture = Any
    CaptureArtifact = Any
    CaptureRepository = Any


CALIBRATION_MANIFEST_SCHEMA = "zlc_neutral_atom.calibration-manifest"
_CALIBRATION_DERIVATION_SCHEMA = "zlc_neutral_atom.CalibrationDerivation"
_CALIBRATION_PLAN_BINDING_SCHEMA = "zlc_neutral_atom.CalibrationPlanBinding"
_CALIBRATION_ANALYSIS_RESULT_SCHEMA = "zlc_neutral_atom.CalibrationAnalysisResult"
_ADMITTED_CALIBRATION_EVIDENCE_SCHEMA = (
    "zlc_neutral_atom.AdmittedCalibrationEvidence"
)
_ADMITTED_CALIBRATION_INTEGRITY_SCHEMA = (
    "zlc_neutral_atom.AdmittedCalibrationIntegrity"
)
_CALIBRATION_NAMESPACE = "calibration"
_EXECUTED_TOKEN = object()
_ADMISSION_TOKEN = object()
_DERIVATION_DECODE_LIMITS = CanonicalDecodeLimits(
    max_depth=12,
    max_nodes=128,
    max_container_entries=128,
    max_arrays=0,
    max_total_array_bytes=0,
)
_MANIFEST_DECODE_LIMITS = CanonicalDecodeLimits(
    max_depth=8,
    max_nodes=128,
    max_container_entries=128,
    max_arrays=0,
    max_total_array_bytes=0,
)


@dataclass(frozen=True)
class _RepositoryAuthority:
    root: Path
    repository_id: str
    root_lease: RepositoryRootLease
    resource_policy: CalibrationResourcePolicy
    resource_policy_digest: str
    store_authority: ContentStoreAuthority
    journal: PersistentCommitJournal
    coordinator: RepositoryCommitCoordinator[CalibrationArtifactRef]
    token: object


def _resource_policy_snapshot(
    policy: CalibrationResourcePolicy,
) -> CalibrationResourcePolicy:
    if not isinstance(policy, CalibrationResourcePolicy):
        raise TypeError("resource_policy must be CalibrationResourcePolicy")
    return CalibrationResourcePolicy(
        max_manifest_bytes=policy.max_manifest_bytes,
        max_artifact_blob_bytes=policy.max_artifact_blob_bytes,
        max_models=policy.max_models,
        max_sites=policy.max_sites,
        max_kernel_elements=policy.max_kernel_elements,
        max_sampled_pixels_per_model=policy.max_sampled_pixels_per_model,
        max_total_sampled_pixels_all_models=(
            policy.max_total_sampled_pixels_all_models
        ),
    )


def _resource_policy_digest(policy: CalibrationResourcePolicy) -> str:
    snapshot = _resource_policy_snapshot(policy)
    return canonical_digest(
        {
            "schema": "zlc_neutral_atom.CalibrationRepositoryResourcePolicy",
            "max_manifest_bytes": snapshot.max_manifest_bytes,
            "max_artifact_blob_bytes": snapshot.max_artifact_blob_bytes,
            "max_models": snapshot.max_models,
            "max_sites": snapshot.max_sites,
            "max_kernel_elements": snapshot.max_kernel_elements,
            "max_sampled_pixels_per_model": (
                snapshot.max_sampled_pixels_per_model
            ),
            "max_total_sampled_pixels_all_models": (
                snapshot.max_total_sampled_pixels_all_models
            ),
        }
    )


def _capture_types():
    """Import capture implementation types only after package initialization.

    ``runtime.capture`` owns FrameContract and is imported while
    ``zlc_neutral_atom.readout`` is initialized.  Eagerly importing the
    artifact implementation here would close a package-initialization cycle
    through ``acquisition``.  Runtime authority still requires these concrete
    classes; only their lookup is deferred.
    """

    from zlc_neutral_atom.artifacts.capture import (
        AdmittedCapture,
        CaptureArtifact,
        CaptureRepository,
    )

    return AdmittedCapture, CaptureArtifact, CaptureRepository


def _optional_canonical_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _canonical_text(value, field_name)


def _parameter_digest(artifact: CalibrationArtifact, name: str) -> str:
    parameters = {item.name: item.value for item in artifact.parameters}
    return _sha256(parameters.get(name), f"calibration artifact {name}")


def _source_binding_fingerprint(binding: CalibrationSourceBinding) -> str:
    if not isinstance(binding, CalibrationSourceBinding):
        raise TypeError("binding must be CalibrationSourceBinding")
    return sha256_digest(encode_calibration_source_binding(binding))


def _analysis_result_digest(
    artifact: CalibrationArtifact,
    diagnostics_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": _CALIBRATION_ANALYSIS_RESULT_SCHEMA,
            "artifact_fingerprint": artifact.fingerprint,
            "diagnostics_digest": _sha256(
                diagnostics_digest,
                "diagnostics_digest",
            ),
            "algorithm_id": artifact.algorithm_id,
            "algorithm_version": artifact.algorithm_version,
        }
    )


def _plan_binding_digest(
    source_capture_ref: CaptureArtifactRef,
    *,
    capture_repository_id: str,
    source_capture_evidence_digest: str,
    source_capture_commit_kind: CommitKind,
    source_capture_commit_id: str,
    calibration_repository_id: str,
    request_fingerprint: str,
) -> str:
    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    if not isinstance(source_capture_commit_kind, CommitKind):
        raise TypeError("source_capture_commit_kind must be CommitKind")
    return canonical_digest(
        {
            "schema": _CALIBRATION_PLAN_BINDING_SCHEMA,
            "source_capture_ref": capture_artifact_ref_to_tree(source_capture_ref),
            "capture_repository_id": _canonical_text(
                capture_repository_id,
                "capture_repository_id",
            ),
            "source_capture_evidence_digest": _sha256(
                source_capture_evidence_digest,
                "source_capture_evidence_digest",
            ),
            "source_capture_commit_kind": source_capture_commit_kind.value,
            "source_capture_commit_id": _canonical_text(
                source_capture_commit_id,
                "source_capture_commit_id",
            ),
            "calibration_repository_id": _canonical_text(
                calibration_repository_id,
                "calibration_repository_id",
            ),
            "request_fingerprint": _sha256(
                request_fingerprint,
                "request_fingerprint",
            ),
            "manifest_schema": CALIBRATION_MANIFEST_SCHEMA,
            "artifact_schema": CALIBRATION_ARTIFACT_SCHEMA,
        }
    )


def _calibration_commit_target(
    repository_id: str,
    reference: CalibrationArtifactRef,
) -> CommitTarget:
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    repository_id = _canonical_text(repository_id, "repository_id")
    if reference.repository_id != repository_id:
        raise ValueError("CalibrationArtifactRef belongs to another repository")
    return CommitTarget(
        repository_id,
        "calibration",
        CALIBRATION_MANIFEST_SCHEMA,
        reference.target_ref,
        reference.manifest_digest,
    )


def _calibration_final_commit_id(run_id: str, manifest_digest: str) -> str:
    return (
        f"calibration-final-{_canonical_text(run_id, 'run_id')}-"
        f"{_sha256(manifest_digest, 'manifest_digest')}"
    )


def _admitted_calibration_evidence_digest(
    *,
    repository_id: str,
    reference: CalibrationArtifactRef,
    artifact_fingerprint: str,
    derivation_evidence_digest: str,
    source_capture_evidence_digest: str,
    source_capture_commit_kind: CommitKind,
    source_capture_commit_id: str,
    commit_intent: CommitIntent,
) -> str:
    target = _calibration_commit_target(repository_id, reference)
    if commit_intent.kind is not CommitKind.FINAL:
        raise ValueError("calibration admission requires FINAL commit evidence")
    if commit_intent.target != target:
        raise ValueError("calibration admission commit names another target")
    if not isinstance(source_capture_commit_kind, CommitKind):
        raise TypeError("source_capture_commit_kind must be CommitKind")
    return sha256_digest(
        encode(
            {
                "schema": _ADMITTED_CALIBRATION_EVIDENCE_SCHEMA,
                "repository_id": repository_id,
                "reference": calibration_artifact_ref_to_tree(reference),
                "artifact_fingerprint": _sha256(
                    artifact_fingerprint,
                    "artifact_fingerprint",
                ),
                "derivation_evidence_digest": _sha256(
                    derivation_evidence_digest,
                    "derivation_evidence_digest",
                ),
                "source_capture_evidence_digest": _sha256(
                    source_capture_evidence_digest,
                    "source_capture_evidence_digest",
                ),
                "source_capture_commit_kind": source_capture_commit_kind.value,
                "source_capture_commit_id": _canonical_text(
                    source_capture_commit_id,
                    "source_capture_commit_id",
                ),
                "commit": {
                    "kind": commit_intent.kind.value,
                    "commit_id": commit_intent.commit_id,
                    "run_id": commit_intent.run_id,
                    "safety_bundle_id": commit_intent.safety_bundle_id,
                    "created_at": commit_intent.created_at,
                    "target": {
                        "repository_id": target.repository_id,
                        "artifact_kind": target.artifact_kind,
                        "artifact_format": target.artifact_format,
                        "target_ref": target.target_ref,
                        "expected_manifest_digest": (
                            target.expected_manifest_digest
                        ),
                    },
                },
            }
        )
    )


def _admitted_calibration_integrity_digest(
    *,
    repository_token: object,
    reference: CalibrationArtifactRef,
    artifact: CalibrationArtifact,
    commit_kind: CommitKind,
    commit_id: str,
    evidence_digest: str,
) -> str:
    """Bind every public admission fact to its process-local repository owner."""

    if repository_token is None:
        raise ValueError("AdmittedCalibration repository authority is absent")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    if commit_kind is not CommitKind.FINAL:
        raise ValueError("AdmittedCalibration requires FINAL commit evidence")
    return canonical_digest(
        {
            "schema": _ADMITTED_CALIBRATION_INTEGRITY_SCHEMA,
            "repository_authority_identity": id(repository_token),
            "reference": calibration_artifact_ref_to_tree(reference),
            "artifact_fingerprint": artifact.fingerprint,
            "commit_kind": commit_kind.value,
            "commit_id": _canonical_text(commit_id, "commit_id"),
            "evidence_digest": _sha256(evidence_digest, "evidence_digest"),
        }
    )


@dataclass(frozen=True)
class _PreparedCalibrationAnalysis:
    capture_admission: AdmittedCapture
    request: CalibrationAnalysisRequest
    work_plan: CalibrationWorkPlan
    source_binding: CalibrationSourceBinding
    frame_contract_payload: bytes
    plan_binding_digest: str
    capture_repository_id: str
    calibration_repository_id: str
    calibration_repository_token: object
    run_id: str

    def __post_init__(self) -> None:
        admitted_capture_type, capture_artifact_type, _ = _capture_types()
        if type(self.capture_admission) is not admitted_capture_type:
            raise TypeError("prepared capture must be AdmittedCapture")
        if type(self.capture_admission.artifact) is not capture_artifact_type:
            raise TypeError("prepared admitted artifact must be CaptureArtifact")
        if self.capture_admission.reference != self.capture_admission.artifact.ref:
            raise ValueError("prepared capture admission reference differs from artifact")
        if not isinstance(self.request, CalibrationAnalysisRequest):
            raise TypeError("prepared request must be CalibrationAnalysisRequest")
        if not isinstance(self.work_plan, CalibrationWorkPlan):
            raise TypeError("prepared work_plan must be CalibrationWorkPlan")
        if not isinstance(self.source_binding, CalibrationSourceBinding):
            raise TypeError("prepared source_binding must be CalibrationSourceBinding")
        if not isinstance(self.frame_contract_payload, bytes):
            raise TypeError("prepared FrameContract payload must be bytes")
        _sha256(self.plan_binding_digest, "prepared plan_binding_digest")
        _canonical_text(self.capture_repository_id, "capture_repository_id")
        _canonical_text(self.calibration_repository_id, "calibration_repository_id")
        if self.calibration_repository_token is None:
            raise ValueError("prepared calibration repository token is absent")
        _canonical_text(self.run_id, "prepared run_id")

    @property
    def capture(self) -> CaptureArtifact:
        return self.capture_admission.artifact


@dataclass(frozen=True)
class _ExecutedCalibrationAnalysis:
    _token: object
    prepared: _PreparedCalibrationAnalysis
    result: CalibrationAnalysisResult

    def __post_init__(self) -> None:
        if self._token is not _EXECUTED_TOKEN:
            raise PermissionError(
                "calibration commit candidates can only be minted by the trusted plan"
            )
        if not isinstance(self.prepared, _PreparedCalibrationAnalysis):
            raise TypeError("executed candidate requires prepared analysis")
        if not isinstance(self.result, CalibrationAnalysisResult):
            raise TypeError("executed candidate requires CalibrationAnalysisResult")


@dataclass(frozen=True)
class _CalibrationDerivation:
    request: CalibrationAnalysisRequest
    work_plan: CalibrationWorkPlan
    diagnostics: CalibrationAnalysisDiagnostics
    source_capture_ref: CaptureArtifactRef
    source_capture_run_id: str
    source_capture_safety_bundle_id: str
    source_capture_evidence_digest: str
    source_capture_commit_kind: CommitKind
    source_capture_commit_id: str
    source_binding_fingerprint: str
    artifact_fingerprint: str
    request_fingerprint: str
    work_plan_fingerprint: str
    diagnostics_digest: str
    analysis_result_digest: str
    plan_binding_digest: str
    algorithm_id: str
    algorithm_version: str
    analysis_run_id: str
    analysis_safety_bundle_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, CalibrationAnalysisRequest):
            raise TypeError("derivation request must be CalibrationAnalysisRequest")
        if not isinstance(self.work_plan, CalibrationWorkPlan):
            raise TypeError("derivation work_plan must be CalibrationWorkPlan")
        if not isinstance(self.diagnostics, CalibrationAnalysisDiagnostics):
            raise TypeError("derivation diagnostics must be CalibrationAnalysisDiagnostics")
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("derivation source_capture_ref must be CaptureArtifactRef")
        _canonical_text(self.source_capture_run_id, "source_capture_run_id")
        _canonical_text(
            self.source_capture_safety_bundle_id,
            "source_capture_safety_bundle_id",
        )
        _sha256(
            self.source_capture_evidence_digest,
            "source_capture_evidence_digest",
        )
        if not isinstance(self.source_capture_commit_kind, CommitKind):
            raise TypeError("source_capture_commit_kind must be CommitKind")
        _canonical_text(self.source_capture_commit_id, "source_capture_commit_id")
        for name in (
            "source_binding_fingerprint",
            "artifact_fingerprint",
            "request_fingerprint",
            "work_plan_fingerprint",
            "diagnostics_digest",
            "analysis_result_digest",
            "plan_binding_digest",
        ):
            _sha256(getattr(self, name), name)
        _canonical_text(self.algorithm_id, "algorithm_id")
        _canonical_text(self.algorithm_version, "algorithm_version")
        _canonical_text(self.analysis_run_id, "analysis_run_id")
        _optional_canonical_text(
            self.analysis_safety_bundle_id,
            "analysis_safety_bundle_id",
        )
        if self.request.fingerprint != self.request_fingerprint:
            raise ValueError("derivation request fingerprint differs from request")
        if self.work_plan.fingerprint != self.work_plan_fingerprint:
            raise ValueError("derivation work-plan fingerprint differs from work plan")


@dataclass(frozen=True)
class _LoadedCalibration:
    artifact: CalibrationArtifact
    derivation: _CalibrationDerivation
    evidence_digest: str
    analysis_result: CalibrationAnalysisResult


@dataclass(frozen=True)
class _AdmittedCalibrationAuthority:
    repository_token: object
    reference: CalibrationArtifactRef
    artifact: CalibrationArtifact
    commit_kind: CommitKind
    commit_id: str
    evidence_digest: str
    integrity_digest: str


_ADMITTED_CALIBRATION_AUTHORITIES = weakref.WeakKeyDictionary()
_ADMITTED_CALIBRATION_AUTHORITIES_LOCK = threading.RLock()


class AdmittedCalibration:
    """Non-serializable process-local proof of source-verified calibration admission."""

    __slots__ = (
        "_token",
        "_repository_token",
        "_reference",
        "_artifact",
        "_commit_kind",
        "_commit_id",
        "_evidence_digest",
        "_integrity_digest",
        "__weakref__",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("AdmittedCalibration is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        repository_token: object,
        reference: CalibrationArtifactRef,
        artifact: CalibrationArtifact,
        commit_kind: CommitKind,
        commit_id: str,
        evidence_digest: str,
    ) -> None:
        if token is not _ADMISSION_TOKEN:
            raise PermissionError(
                "AdmittedCalibration can only be minted by CalibrationRepository.admit"
            )
        if repository_token is None:
            raise ValueError("AdmittedCalibration repository authority is absent")
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if not isinstance(artifact, CalibrationArtifact):
            raise TypeError("artifact must be CalibrationArtifact")
        if commit_kind is not CommitKind.FINAL:
            raise ValueError("AdmittedCalibration requires FINAL commit evidence")
        _canonical_text(commit_id, "commit_id")
        _sha256(evidence_digest, "evidence_digest")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository_token", repository_token)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_artifact", artifact)
        object.__setattr__(self, "_commit_kind", commit_kind)
        object.__setattr__(self, "_commit_id", commit_id)
        object.__setattr__(self, "_evidence_digest", evidence_digest)
        integrity_digest = _admitted_calibration_integrity_digest(
            repository_token=repository_token,
            reference=reference,
            artifact=artifact,
            commit_kind=commit_kind,
            commit_id=commit_id,
            evidence_digest=evidence_digest,
        )
        object.__setattr__(self, "_integrity_digest", integrity_digest)
        with _ADMITTED_CALIBRATION_AUTHORITIES_LOCK:
            _ADMITTED_CALIBRATION_AUTHORITIES[self] = _AdmittedCalibrationAuthority(
                repository_token,
                reference,
                artifact,
                commit_kind,
                commit_id,
                evidence_digest,
                integrity_digest,
            )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("AdmittedCalibration is immutable")

    def __reduce__(self):
        raise TypeError("AdmittedCalibration is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("AdmittedCalibration is process-local and cannot be serialized")

    def _require_authority(self) -> None:
        with _ADMITTED_CALIBRATION_AUTHORITIES_LOCK:
            authority = _ADMITTED_CALIBRATION_AUTHORITIES.get(self)
        if not isinstance(authority, _AdmittedCalibrationAuthority):
            raise PermissionError("AdmittedCalibration authority is invalid")
        try:
            token = self._token
            repository_token = self._repository_token
            reference = self._reference
            artifact = self._artifact
            commit_kind = self._commit_kind
            commit_id = self._commit_id
            evidence_digest = self._evidence_digest
            integrity_digest = self._integrity_digest
        except AttributeError:
            raise PermissionError("AdmittedCalibration authority is invalid") from None
        if type(self) is not AdmittedCalibration or token is not _ADMISSION_TOKEN:
            raise PermissionError("AdmittedCalibration authority is invalid")
        if (
            repository_token is not authority.repository_token
            or reference is not authority.reference
            or artifact is not authority.artifact
            or commit_kind is not authority.commit_kind
            or commit_id != authority.commit_id
            or evidence_digest != authority.evidence_digest
            or integrity_digest != authority.integrity_digest
        ):
            raise PermissionError("AdmittedCalibration authority binding changed")
        try:
            expected = _admitted_calibration_integrity_digest(
                repository_token=repository_token,
                reference=reference,
                artifact=artifact,
                commit_kind=commit_kind,
                commit_id=commit_id,
                evidence_digest=evidence_digest,
            )
        except (TypeError, ValueError) as error:
            raise PermissionError("AdmittedCalibration authority is invalid") from error
        if integrity_digest != expected:
            raise PermissionError("AdmittedCalibration integrity binding changed")

    @property
    def reference(self) -> CalibrationArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def artifact(self) -> CalibrationArtifact:
        self._require_authority()
        return self._artifact

    @property
    def artifact_fingerprint(self) -> str:
        self._require_authority()
        return self._artifact.fingerprint

    @property
    def commit_kind(self) -> CommitKind:
        self._require_authority()
        return self._commit_kind

    @property
    def commit_id(self) -> str:
        self._require_authority()
        return self._commit_id

    @property
    def evidence_digest(self) -> str:
        self._require_authority()
        return self._evidence_digest

    @property
    def source_capture_ref(self) -> CaptureArtifactRef:
        self._require_authority()
        return self._artifact.source_binding.source_capture_ref


class CalibrationRepository:
    """Current-only calibration CAS with durable final-commit authority."""

    __slots__ = (
        "root",
        "repository_id",
        "resource_policy",
        "_root_lease",
        "_store",
        "_store_authority",
        "_journal",
        "_coordinator",
        "_authority",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CalibrationRepository is final and cannot be subclassed")

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-calibration",
        resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        object.__setattr__(
            self,
            "repository_id",
            _canonical_text(repository_id, "repository_id"),
        )
        public_policy = _resource_policy_snapshot(resource_policy)
        authority_policy = _resource_policy_snapshot(public_policy)
        object.__setattr__(self, "resource_policy", public_policy)
        root_lease = RepositoryRootLease(
            self.root,
            owner=f"calibration:{self.repository_id}",
        )
        object.__setattr__(self, "_root_lease", root_lease)
        try:
            store = ContentAddressedStore(self.root / "content")
            object.__setattr__(self, "_store", store)
            object.__setattr__(self, "_store_authority", store.authority())
            journal = PersistentCommitJournal(
                self.root / "calibration-commit.journal",
                self.repository_id,
            )
            object.__setattr__(self, "_journal", journal)
            # Startup reconciliation calls back into ``_recover`` while the
            # coordinator is being constructed, so the full immutable
            # authority snapshot is installed immediately afterwards.
            object.__setattr__(self, "_authority", None)
            coordinator: RepositoryCommitCoordinator[CalibrationArtifactRef] = (
                RepositoryCommitCoordinator(
                    journal,
                    self._recover,
                    root_lease=root_lease,
                )
            )
            object.__setattr__(self, "_coordinator", coordinator)
            object.__setattr__(
                self,
                "_authority",
                _RepositoryAuthority(
                    self.root,
                    self.repository_id,
                    root_lease,
                    authority_policy,
                    _resource_policy_digest(authority_policy),
                    self._store_authority,
                    journal,
                    coordinator,
                    object(),
                ),
            )
        except BaseException:
            root_lease.close()
            raise
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CalibrationRepository authority is immutable")
        object.__setattr__(self, _name, _value)

    def _assert_authority_integrity(self) -> _RepositoryAuthority | None:
        authority = self._authority
        if authority is None:
            # The only legal window is synchronous startup reconciliation in
            # RepositoryCommitCoordinator.__init__.
            if getattr(self, "_sealed", False):
                raise RuntimeError("calibration repository authority is absent")
            return None
        if (
            type(self) is not CalibrationRepository
            or self.root != authority.root
            or self.repository_id != authority.repository_id
            or self._root_lease is not authority.root_lease
            or self.resource_policy != authority.resource_policy
            or _resource_policy_digest(self.resource_policy)
            != authority.resource_policy_digest
            or _resource_policy_digest(authority.resource_policy)
            != authority.resource_policy_digest
            or type(self._store) is not ContentAddressedStore
            or type(self._store_authority) is not ContentStoreAuthority
            or self._store_authority is not authority.store_authority
            or self._store.authority() is not authority.store_authority
            or authority.store_authority.root != authority.root / "content"
            or self._journal is not authority.journal
            or self._coordinator is not authority.coordinator
            or self._journal.repository_id != authority.repository_id
            or self._journal.repository_root != authority.root
            or self._coordinator.repository_id != authority.repository_id
        ):
            raise RuntimeError("calibration repository durability authority changed")
        authority.root_lease.require_active()
        if authority.store_authority.root != authority.root / "content":
            raise RuntimeError("calibration content store escaped its repository root")
        return authority

    def close(self) -> None:
        """Close this owner after its prepared/in-flight commits resolve."""

        root_lease = getattr(self, "_root_lease", None)
        if root_lease is not None:
            root_lease.close()

    def __enter__(self) -> "CalibrationRepository":
        self._assert_authority_integrity()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    @property
    def startup_reconciliations(self):
        self._assert_authority_integrity()
        return self._coordinator.startup_reconciliations

    def load(self, reference: CalibrationArtifactRef) -> CalibrationArtifact:
        """Fully validate persistent evidence for inspection, without admission."""

        self._assert_authority_integrity()
        return self._load_record(reference).artifact

    def load_analysis_result(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationAnalysisResult:
        """Load the artifact and its persisted diagnostics as one inspection value.

        This does not mint runtime admission authority.  Occupancy composition
        must still call :meth:`admit`, which reloads and validates the exact
        source CaptureArtifact and FINAL commit evidence.
        """

        self._assert_authority_integrity()
        loaded = self._load_record(reference)
        return loaded.analysis_result

    def admit(
        self,
        reference: CalibrationArtifactRef,
        capture_repository: CaptureRepository,
    ) -> AdmittedCalibration:
        """Reload the exact source capture and mint one process-local authority."""

        authority = self._assert_authority_integrity()
        assert authority is not None
        admitted_capture_type, _, capture_repository_type = _capture_types()
        if type(capture_repository) is not capture_repository_type:
            raise TypeError("capture_repository must be CaptureRepository")
        loaded = self._load_record(reference)
        derivation = loaded.derivation
        target = _calibration_commit_target(authority.repository_id, reference)
        matching = tuple(
            intent
            for intent in authority.coordinator.committed_intents()
            if intent.target == target
        )
        if not matching:
            raise PermissionError(
                "CalibrationArtifact is visible but has no committed FINAL authority"
            )
        expected_commit_id = _calibration_final_commit_id(
            derivation.analysis_run_id,
            reference.manifest_digest,
        )
        for intent in matching:
            if (
                intent.kind is not CommitKind.FINAL
                or intent.commit_id != expected_commit_id
                or intent.run_id != derivation.analysis_run_id
                or intent.safety_bundle_id
                != derivation.analysis_safety_bundle_id
            ):
                raise ValueError(
                    "committed calibration intent differs from persisted evidence"
                )
        selected = min(matching, key=lambda intent: intent.commit_id)
        if capture_repository.repository_id != derivation.source_capture_ref.repository_id:
            raise ValueError("source CaptureArtifactRef belongs to another repository")
        capture_admission = capture_repository_type.admit(
            capture_repository,
            derivation.source_capture_ref,
        )
        if type(capture_admission) is not admitted_capture_type:
            raise TypeError("CaptureRepository returned another admission type")
        _validate_source_capture(
            loaded.artifact,
            derivation,
            capture_admission,
        )
        expected_plan_binding = _plan_binding_digest(
            derivation.source_capture_ref,
            capture_repository_id=capture_repository.repository_id,
            source_capture_evidence_digest=capture_admission.evidence_digest,
            source_capture_commit_kind=capture_admission.commit_kind,
            source_capture_commit_id=capture_admission.commit_id,
            calibration_repository_id=self.repository_id,
            request_fingerprint=derivation.request.fingerprint,
        )
        if expected_plan_binding != derivation.plan_binding_digest:
            raise ValueError("calibration plan binding differs from admitted repositories")
        evidence_digest = _admitted_calibration_evidence_digest(
            repository_id=authority.repository_id,
            reference=reference,
            artifact_fingerprint=loaded.artifact.fingerprint,
            derivation_evidence_digest=loaded.evidence_digest,
            source_capture_evidence_digest=capture_admission.evidence_digest,
            source_capture_commit_kind=capture_admission.commit_kind,
            source_capture_commit_id=capture_admission.commit_id,
            commit_intent=selected,
        )
        return AdmittedCalibration(
            _ADMISSION_TOKEN,
            repository_token=authority.token,
            reference=reference,
            artifact=loaded.artifact,
            commit_kind=selected.kind,
            commit_id=selected.commit_id,
            evidence_digest=evidence_digest,
        )

    def has(self, reference: CalibrationArtifactRef) -> bool:
        self._assert_authority_integrity()
        self._validate_reference(reference)
        try:
            self._read_manifest_payload(reference)
        except FileNotFoundError:
            return False
        return True

    def final_commit(
        self,
        context: PostSafetyContext,
        executed: _ExecutedCalibrationAnalysis,
    ) -> FinalCommit[CalibrationArtifactRef]:
        if not isinstance(context, PostSafetyContext):
            raise TypeError("calibration commit requires PostSafetyContext")
        if not isinstance(executed, _ExecutedCalibrationAnalysis):
            raise TypeError("calibration commit requires a trusted executed candidate")
        authority = self._assert_authority_integrity()
        assert authority is not None
        prepared = executed.prepared
        if prepared.calibration_repository_id != authority.repository_id:
            raise ValueError("calibration repository identity changed after plan binding")
        if prepared.calibration_repository_token is not authority.token:
            raise PermissionError(
                "calibration candidate belongs to another durability authority"
            )
        if prepared.run_id != context.run_id.value:
            raise ValueError("calibration candidate belongs to another Run")
        if context.safety_bundle_id is not None:
            raise RuntimeError(
                "software-only calibration analysis unexpectedly acquired a safety bundle"
            )
        # ``_ExecutedCalibrationAnalysis`` is minted only after execute has
        # replayed the complete request/result contract.  Its closed result is
        # immutable, so finalize need not pay the same O(site * evidence)
        # statistical replay a second time before encoding it.
        subject = context.authorize_commit_preparation(CommitKind.FINAL)
        reference, manifest_payload = self._stage_execution(
            executed,
            authority=authority,
            analysis_safety_bundle_id=context.safety_bundle_id,
        )
        self._assert_authority_integrity()
        confirmed_subject = context.authorize_commit_preparation(CommitKind.FINAL)
        if confirmed_subject != subject:
            raise RuntimeError("calibration commit subject changed while staging")
        target = _calibration_commit_target(authority.repository_id, reference)

        def publish() -> PublishedManifest[CalibrationArtifactRef]:
            self._assert_authority_integrity()
            try:
                stored = authority.store_authority.publish_manifest(
                    _CALIBRATION_NAMESPACE,
                    manifest_payload,
                    expected_digest=reference.manifest_digest,
                )
            except PublishVisibilityUnknown:
                raise
            except BaseException as publish_error:
                # ContentAddressedStore publishes with atomic replace and then
                # performs durability fsync/reverification.  A failure in
                # those post-visibility steps cannot be called a deterministic
                # abort.  Probe the exact target through the same bounded,
                # digest-verifying read and hand uncertainty to the Run
                # recovery protocol instead of aborting a visible artifact.
                try:
                    visible = authority.store_authority.read_manifest(
                        _CALIBRATION_NAMESPACE,
                        reference.manifest_digest,
                        max_bytes=authority.resource_policy.max_manifest_bytes,
                    )
                except FileNotFoundError:
                    raise publish_error
                except BaseException as visibility_error:
                    raise PublishVisibilityUnknown(
                        "calibration manifest visibility could not be verified"
                    ) from visibility_error
                if visible != manifest_payload:
                    raise PublishVisibilityUnknown(
                        "calibration manifest target is visible with unexpected bytes"
                    ) from publish_error
                raise PublishVisibilityUnknown(
                    "calibration manifest became visible before publication "
                    "acknowledgement completed"
                ) from publish_error
            if stored.content.digest != reference.manifest_digest:
                raise RuntimeError("published calibration manifest digest changed")
            return PublishedManifest(
                reference.target_ref,
                reference.manifest_digest,
                reference,
            )

        commit_id = _calibration_final_commit_id(
            subject.run_id,
            reference.manifest_digest,
        )
        operation = FinalCommit(
            authority.coordinator.prepare(
                CommitKind.FINAL,
                commit_id,
                subject,
                target,
                publish,
            )
        )
        try:
            context._track_prepared_commit(operation)
        except BaseException:
            operation.abandon()
            raise
        return operation

    def _stage_execution(
        self,
        executed: _ExecutedCalibrationAnalysis,
        *,
        authority: _RepositoryAuthority,
        analysis_safety_bundle_id: str | None,
    ) -> tuple[CalibrationArtifactRef, bytes]:
        current = self._assert_authority_integrity()
        if current is not authority:
            raise RuntimeError("calibration repository durability authority changed")
        prepared = executed.prepared
        result = executed.result
        artifact = result.artifact
        validate_calibration_artifact_resources(artifact, authority.resource_policy)
        resource_summary = calibration_resource_summary(artifact)
        artifact_payload = encode_calibration_artifact(artifact)
        if len(artifact_payload) > prepared.work_plan.artifact_encoding_upper_bound_bytes:
            raise RuntimeError(
                "calibration artifact encoding exceeded its frozen work-plan bound"
            )
        if len(artifact_payload) > authority.resource_policy.max_artifact_blob_bytes:
            raise CalibrationResourceExceeded(
                "calibration artifact blob exceeds repository resource policy"
            )
        request_payload = encode_calibration_analysis_request(prepared.request)
        work_plan_payload = encode_calibration_work_plan(prepared.work_plan)
        diagnostics_payload = encode_calibration_analysis_diagnostics(
            result.diagnostics,
            resource_policy=prepared.request.resource_policy,
        )
        if (
            len(diagnostics_payload)
            > prepared.work_plan.diagnostics_encoding_upper_bound_bytes
        ):
            raise RuntimeError(
                "calibration diagnostics encoding exceeded its frozen work-plan bound"
            )
        diagnostics_digest = sha256_digest(diagnostics_payload)
        derivation = _CalibrationDerivation(
            request=prepared.request,
            work_plan=prepared.work_plan,
            diagnostics=result.diagnostics,
            source_capture_ref=prepared.capture.ref,
            source_capture_run_id=prepared.capture.run_id,
            source_capture_safety_bundle_id=prepared.capture.safety_bundle_id,
            source_capture_evidence_digest=(
                prepared.capture_admission.evidence_digest
            ),
            source_capture_commit_kind=prepared.capture_admission.commit_kind,
            source_capture_commit_id=prepared.capture_admission.commit_id,
            source_binding_fingerprint=_source_binding_fingerprint(
                artifact.source_binding
            ),
            artifact_fingerprint=artifact.fingerprint,
            request_fingerprint=prepared.request.fingerprint,
            work_plan_fingerprint=prepared.work_plan.fingerprint,
            diagnostics_digest=diagnostics_digest,
            analysis_result_digest=_analysis_result_digest(
                artifact,
                diagnostics_digest,
            ),
            plan_binding_digest=prepared.plan_binding_digest,
            algorithm_id=artifact.algorithm_id,
            algorithm_version=artifact.algorithm_version,
            analysis_run_id=prepared.run_id,
            analysis_safety_bundle_id=analysis_safety_bundle_id,
        )
        derivation_payload = _derivation_payload(
            derivation,
            request_payload=request_payload,
            work_plan_payload=work_plan_payload,
            diagnostics_payload=diagnostics_payload,
        )
        if len(derivation_payload) > authority.resource_policy.max_artifact_blob_bytes:
            raise CalibrationResourceExceeded(
                "calibration derivation blob exceeds repository resource policy"
            )
        artifact_blob = ContentRef(
            sha256_digest(artifact_payload),
            len(artifact_payload),
        )
        derivation_blob = ContentRef(
            sha256_digest(derivation_payload),
            len(derivation_payload),
        )
        manifest_payload = _manifest_payload(
            repository_id=authority.repository_id,
            artifact_blob=artifact_blob,
            derivation_blob=derivation_blob,
            derivation=derivation,
            resource_summary=resource_summary,
        )
        if len(manifest_payload) > authority.resource_policy.max_manifest_bytes:
            raise CalibrationResourceExceeded(
                "calibration manifest exceeds repository resource policy"
            )
        stored_artifact = authority.store_authority.put_blob(artifact_payload)
        stored_derivation = authority.store_authority.put_blob(derivation_payload)
        if stored_artifact != artifact_blob or stored_derivation != derivation_blob:
            raise RuntimeError("calibration content address changed while staging")
        reference = CalibrationArtifactRef(
            authority.repository_id,
            sha256_digest(manifest_payload),
        )
        return reference, manifest_payload

    def _read_manifest_payload(self, reference: CalibrationArtifactRef) -> bytes:
        authority = self._assert_authority_integrity()
        store_authority = (
            self._store_authority
            if authority is None
            else authority.store_authority
        )
        policy = self.resource_policy if authority is None else authority.resource_policy
        self._validate_reference(reference)
        try:
            return store_authority.read_manifest(
                _CALIBRATION_NAMESPACE,
                reference.manifest_digest,
                max_bytes=policy.max_manifest_bytes,
            )
        except ContentSizeLimitError as exc:
            raise CalibrationResourceExceeded(
                "calibration manifest exceeds repository resource policy"
            ) from exc

    def _load_record(
        self,
        reference: CalibrationArtifactRef,
        *,
        _verified_manifest_payload: bytes | None = None,
    ) -> _LoadedCalibration:
        authority = self._assert_authority_integrity()
        store_authority = (
            self._store_authority
            if authority is None
            else authority.store_authority
        )
        policy = self.resource_policy if authority is None else authority.resource_policy
        repository_id = (
            self.repository_id if authority is None else authority.repository_id
        )
        self._validate_reference(reference)
        manifest_payload = (
            self._read_manifest_payload(reference)
            if _verified_manifest_payload is None
            else _verified_manifest_payload
        )
        if not isinstance(manifest_payload, bytes):
            raise TypeError("verified calibration manifest payload must be bytes")
        data = _manifest_from_tree(
            decode(manifest_payload, limits=_MANIFEST_DECODE_LIMITS)
        )
        if data["repository_id"] != repository_id:
            raise ValueError("CalibrationArtifact belongs to another repository")
        declared_summary = _resource_summary_from_tree(data["resource_summary"])
        validate_calibration_resource_summary(declared_summary, policy)
        artifact_blob = _content_ref_from_tree(data["artifact_blob"])
        derivation_blob = _content_ref_from_tree(data["derivation_blob"])
        for name, content in (
            ("artifact", artifact_blob),
            ("derivation", derivation_blob),
        ):
            if content.size > policy.max_artifact_blob_bytes:
                raise CalibrationResourceExceeded(
                    f"calibration {name} blob exceeds repository resource policy"
                )
        try:
            artifact_payload = store_authority.read_blob(
                artifact_blob,
                max_bytes=policy.max_artifact_blob_bytes,
            )
            derivation_payload = store_authority.read_blob(
                derivation_blob,
                max_bytes=policy.max_artifact_blob_bytes,
            )
        except ContentSizeLimitError as exc:
            raise CalibrationResourceExceeded(
                "calibration content blob exceeds repository resource policy"
            ) from exc
        artifact = decode_calibration_artifact(
            artifact_payload,
            resource_policy=policy,
        )
        (
            derivation,
            _request_payload,
            _work_plan_payload,
            diagnostics_payload,
        ) = _derivation_from_payload(derivation_payload)
        analysis_result = _validate_persistent_record(
            artifact,
            derivation,
            diagnostics_payload=diagnostics_payload,
            calibration_repository_id=repository_id,
        )
        computed_summary = calibration_resource_summary(artifact)
        if computed_summary != declared_summary:
            raise ValueError("calibration resource summary differs from artifact content")
        _validate_manifest_evidence(data, artifact, derivation, derivation_blob)
        rebuilt = _manifest_payload(
            repository_id=repository_id,
            artifact_blob=artifact_blob,
            derivation_blob=derivation_blob,
            derivation=derivation,
            resource_summary=computed_summary,
        )
        if rebuilt != manifest_payload or sha256_digest(rebuilt) != reference.manifest_digest:
            raise ValueError("CalibrationArtifact manifest is not canonical")
        return _LoadedCalibration(
            artifact,
            derivation,
            derivation_blob.digest,
            analysis_result,
        )

    def _recover(self, intent: CommitIntent) -> CommitRecovery[CalibrationArtifactRef]:
        authority = self._assert_authority_integrity()
        store_authority = (
            self._store_authority
            if authority is None
            else authority.store_authority
        )
        policy = self.resource_policy if authority is None else authority.resource_policy
        target = intent.target
        prefix = f"{_CALIBRATION_NAMESPACE}/"
        if (
            intent.kind is not CommitKind.FINAL
            or target.repository_id != self.repository_id
            or target.artifact_kind != "calibration"
            or target.artifact_format != CALIBRATION_MANIFEST_SCHEMA
            or not target.target_ref.startswith(prefix)
        ):
            raise ValueError("commit intent is not a CalibrationArtifact target")
        digest = _sha256(
            target.target_ref[len(prefix) :],
            "target manifest digest",
        )
        if digest != target.expected_manifest_digest:
            raise ValueError("calibration commit target ref and digest differ")
        expected_commit_id = _calibration_final_commit_id(intent.run_id, digest)
        if intent.commit_id != expected_commit_id:
            raise ValueError("calibration commit id differs from kind/run/target")
        reference = CalibrationArtifactRef(self.repository_id, digest)
        try:
            manifest_payload = self._read_manifest_payload(reference)
        except FileNotFoundError:
            return CommitRecovery(False)
        # Once the manifest visibility point exists, every missing/corrupt
        # referenced blob is a repository fault, never evidence that the
        # commit was absent.  Keep such an intent pending by failing closed.
        loaded = self._load_record(
            reference,
            _verified_manifest_payload=manifest_payload,
        )
        derivation = loaded.derivation
        if derivation.analysis_run_id != intent.run_id:
            raise ValueError("calibration manifest run_id differs from commit intent")
        if derivation.analysis_safety_bundle_id != intent.safety_bundle_id:
            raise ValueError(
                "calibration manifest safety bundle differs from commit intent"
            )
        # A readable target may still be the visible residue of a replace whose
        # parent-directory flush was not acknowledged.  This storage-owned
        # barrier verifies/fsyncs only the existing immutable target and never
        # creates a missing manifest.
        confirmed_payload = store_authority.confirm_manifest_durable(
            _CALIBRATION_NAMESPACE,
            digest,
            max_bytes=policy.max_manifest_bytes,
        )
        if confirmed_payload != manifest_payload:
            raise RuntimeError(
                "calibration recovery durability confirmation changed payload"
            )
        return CommitRecovery(
            True,
            PublishedManifest(reference.target_ref, digest, reference),
        )

    def _validate_reference(self, reference: CalibrationArtifactRef) -> None:
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("CalibrationArtifactRef belongs to another repository")


_DERIVATION_METADATA_FIELDS = frozenset(
    {
        "source_capture_ref",
        "source_capture_run_id",
        "source_capture_safety_bundle_id",
        "source_capture_evidence_digest",
        "source_capture_commit_kind",
        "source_capture_commit_id",
        "source_binding_fingerprint",
        "artifact_fingerprint",
        "request_fingerprint",
        "work_plan_fingerprint",
        "diagnostics_digest",
        "analysis_result_digest",
        "plan_binding_digest",
        "algorithm_id",
        "algorithm_version",
        "analysis_run_id",
        "analysis_safety_bundle_id",
    }
)


def _derivation_metadata_values(
    *,
    source_capture_ref: CaptureArtifactRef,
    source_capture_run_id: str,
    source_capture_safety_bundle_id: str,
    source_capture_evidence_digest: str,
    source_capture_commit_kind: CommitKind,
    source_capture_commit_id: str,
    source_binding_fingerprint: str,
    artifact_fingerprint: str,
    request_fingerprint: str,
    work_plan_fingerprint: str,
    diagnostics_digest: str,
    analysis_result_digest: str,
    plan_binding_digest: str,
    algorithm_id: str,
    algorithm_version: str,
    analysis_run_id: str,
    analysis_safety_bundle_id: str | None,
) -> dict[str, object]:
    return {
        "source_capture_ref": capture_artifact_ref_to_tree(source_capture_ref),
        "source_capture_run_id": source_capture_run_id,
        "source_capture_safety_bundle_id": source_capture_safety_bundle_id,
        "source_capture_evidence_digest": source_capture_evidence_digest,
        "source_capture_commit_kind": source_capture_commit_kind.value,
        "source_capture_commit_id": source_capture_commit_id,
        "source_binding_fingerprint": source_binding_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "request_fingerprint": request_fingerprint,
        "work_plan_fingerprint": work_plan_fingerprint,
        "diagnostics_digest": diagnostics_digest,
        "analysis_result_digest": analysis_result_digest,
        "plan_binding_digest": plan_binding_digest,
        "algorithm_id": algorithm_id,
        "algorithm_version": algorithm_version,
        "analysis_run_id": analysis_run_id,
        "analysis_safety_bundle_id": analysis_safety_bundle_id,
    }


def _derivation_envelope_tree(
    metadata: dict[str, object],
    *,
    request_payload: bytes,
    work_plan_payload: bytes,
    diagnostics_payload: bytes,
) -> dict[str, object]:
    if set(metadata) != _DERIVATION_METADATA_FIELDS:
        raise ValueError("derivation metadata field set is not current")
    if any(
        not isinstance(value, bytes)
        for value in (request_payload, work_plan_payload, diagnostics_payload)
    ):
        raise TypeError("derivation owner payloads must be bytes")
    return {
        "schema": _CALIBRATION_DERIVATION_SCHEMA,
        "request_payload": request_payload,
        "work_plan_payload": work_plan_payload,
        "diagnostics_payload": diagnostics_payload,
        **metadata,
    }


def _manifest_tree_from_metadata(
    *,
    repository_id: str,
    artifact_blob: ContentRef,
    derivation_blob: ContentRef,
    metadata: dict[str, object],
    resource_summary: CalibrationResourceSummary,
) -> dict[str, object]:
    if set(metadata) != _DERIVATION_METADATA_FIELDS:
        raise ValueError("manifest derivation metadata field set is not current")
    return {
        "schema": CALIBRATION_MANIFEST_SCHEMA,
        "repository_id": repository_id,
        "artifact_schema": CALIBRATION_ARTIFACT_SCHEMA,
        "artifact_blob": _content_ref_to_tree(artifact_blob),
        "derivation_blob": _content_ref_to_tree(derivation_blob),
        "artifact_fingerprint": metadata["artifact_fingerprint"],
        "source_capture_ref": metadata["source_capture_ref"],
        "source_capture_run_id": metadata["source_capture_run_id"],
        "source_capture_safety_bundle_id": metadata[
            "source_capture_safety_bundle_id"
        ],
        "source_capture_evidence_digest": metadata[
            "source_capture_evidence_digest"
        ],
        "source_capture_commit_kind": metadata["source_capture_commit_kind"],
        "source_capture_commit_id": metadata["source_capture_commit_id"],
        "source_binding_fingerprint": metadata["source_binding_fingerprint"],
        "request_fingerprint": metadata["request_fingerprint"],
        "work_plan_fingerprint": metadata["work_plan_fingerprint"],
        "diagnostics_digest": metadata["diagnostics_digest"],
        "analysis_result_digest": metadata["analysis_result_digest"],
        "plan_binding_digest": metadata["plan_binding_digest"],
        "algorithm_id": metadata["algorithm_id"],
        "algorithm_version": metadata["algorithm_version"],
        "analysis_run_id": metadata["analysis_run_id"],
        "analysis_safety_bundle_id": metadata["analysis_safety_bundle_id"],
        "evidence_digest": derivation_blob.digest,
        "resource_summary": _resource_summary_to_tree(resource_summary),
    }


def _canonical_bytes_wire_growth(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("canonical byte payload size must be an integer")
    if size < 0:
        raise ValueError("canonical byte payload size must be non-negative")
    return 4 * ((size + 2) // 3)


def _validate_prepared_repository_resources(
    prepared: _PreparedCalibrationAnalysis,
    policy: CalibrationResourcePolicy,
) -> None:
    """Reject a target repository budget before numerical analysis starts."""

    if not isinstance(prepared, _PreparedCalibrationAnalysis):
        raise TypeError("prepared must be _PreparedCalibrationAnalysis")
    if not isinstance(policy, CalibrationResourcePolicy):
        raise TypeError("policy must be CalibrationResourcePolicy")
    plan = prepared.work_plan
    request = prepared.request
    if request.site_count > policy.max_sites:
        raise CalibrationResourceExceeded(
            "planned calibration sites exceed target repository policy"
        )
    if len(request.model_kinds) > policy.max_models:
        raise CalibrationResourceExceeded(
            "planned calibration models exceed target repository policy"
        )
    if plan.planned_kernel_elements > policy.max_kernel_elements:
        raise CalibrationResourceExceeded(
            "planned calibration kernels exceed target repository policy"
        )
    if plan.maximum_model_sampled_pixels > policy.max_sampled_pixels_per_model:
        raise CalibrationResourceExceeded(
            "planned per-model sampled pixels exceed target repository policy"
        )
    if (
        plan.total_model_sampled_pixels
        > policy.max_total_sampled_pixels_all_models
    ):
        raise CalibrationResourceExceeded(
            "planned total sampled pixels exceed target repository policy"
        )
    if plan.artifact_encoding_upper_bound_bytes > policy.max_artifact_blob_bytes:
        raise CalibrationResourceExceeded(
            "planned calibration artifact encoding exceeds target repository policy"
        )

    request_payload = encode_calibration_analysis_request(request)
    work_plan_payload = encode_calibration_work_plan(plan)
    capture = prepared.capture
    admission = prepared.capture_admission
    digest = "f" * 64
    metadata = _derivation_metadata_values(
        source_capture_ref=capture.ref,
        source_capture_run_id=capture.run_id,
        source_capture_safety_bundle_id=capture.safety_bundle_id,
        source_capture_evidence_digest=admission.evidence_digest,
        source_capture_commit_kind=admission.commit_kind,
        source_capture_commit_id=admission.commit_id,
        source_binding_fingerprint=_source_binding_fingerprint(
            prepared.source_binding
        ),
        artifact_fingerprint=digest,
        request_fingerprint=request.fingerprint,
        work_plan_fingerprint=plan.fingerprint,
        diagnostics_digest=digest,
        analysis_result_digest=digest,
        plan_binding_digest=prepared.plan_binding_digest,
        algorithm_id=CALIBRATION_ANALYSIS_ALGORITHM_ID,
        algorithm_version=CALIBRATION_ANALYSIS_ALGORITHM_VERSION,
        analysis_run_id=prepared.run_id,
        # This software-only plan has no resource/hazard claims.  Finalize
        # asserts that runtime preserved the corresponding absent bundle.
        analysis_safety_bundle_id=None,
    )
    derivation_base_tree = _derivation_envelope_tree(
        metadata,
        request_payload=b"",
        work_plan_payload=b"",
        diagnostics_payload=b"",
    )
    derivation_upper_bound = len(encode(derivation_base_tree)) + sum(
        _canonical_bytes_wire_growth(size)
        for size in (
            len(request_payload),
            len(work_plan_payload),
            plan.diagnostics_encoding_upper_bound_bytes,
        )
    )
    if derivation_upper_bound > policy.max_artifact_blob_bytes:
        raise CalibrationResourceExceeded(
            "planned calibration derivation encoding exceeds target repository policy"
        )

    summary = CalibrationResourceSummary(
        request.site_count,
        len(request.model_kinds),
        plan.planned_kernel_elements,
        plan.maximum_model_sampled_pixels,
        plan.total_model_sampled_pixels,
    )
    artifact_blob = ContentRef(digest, plan.artifact_encoding_upper_bound_bytes)
    derivation_blob = ContentRef(digest, derivation_upper_bound)
    manifest_upper_tree = _manifest_tree_from_metadata(
        repository_id=prepared.calibration_repository_id,
        artifact_blob=artifact_blob,
        derivation_blob=derivation_blob,
        metadata=metadata,
        resource_summary=summary,
    )
    if len(encode(manifest_upper_tree)) > policy.max_manifest_bytes:
        raise CalibrationResourceExceeded(
            "planned calibration manifest exceeds target repository policy"
        )


def compile_calibration_artifact_plan(
    source_capture_ref: CaptureArtifactRef,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
    request: CalibrationAnalysisRequest,
) -> RunPlan:
    """Compile one source-committed offline/live calibration analysis Run."""

    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    admitted_capture_type, _, capture_repository_type = _capture_types()
    if type(capture_repository) is not capture_repository_type:
        raise TypeError("capture_repository must be CaptureRepository")
    if type(calibration_repository) is not CalibrationRepository:
        raise TypeError("calibration_repository must be CalibrationRepository")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if source_capture_ref.repository_id != capture_repository.repository_id:
        raise ValueError("source CaptureArtifactRef belongs to another repository")
    calibration_authority = calibration_repository._assert_authority_integrity()
    assert calibration_authority is not None
    frozen_ref = capture_artifact_ref_from_tree(
        capture_artifact_ref_to_tree(source_capture_ref)
    )
    frozen_request_payload = encode_calibration_analysis_request(request)
    frozen_request = decode_calibration_analysis_request(frozen_request_payload)
    capture_repository_id = capture_repository.repository_id
    calibration_repository_id = calibration_repository.repository_id
    def validate_repository_bindings() -> None:
        if capture_repository.repository_id != capture_repository_id:
            raise RuntimeError("capture repository identity changed after plan binding")
        if calibration_repository.repository_id != calibration_repository_id:
            raise RuntimeError("calibration repository identity changed after plan binding")
        current = calibration_repository._assert_authority_integrity()
        if current is None or current.token is not calibration_authority.token:
            raise RuntimeError("calibration repository durability authority changed")

    def preflight(context: RunContext) -> _PreparedCalibrationAnalysis:
        validate_repository_bindings()
        capture_admission = capture_repository_type.admit(
            capture_repository,
            frozen_ref,
        )
        if type(capture_admission) is not admitted_capture_type:
            raise TypeError("CaptureRepository returned another admission type")
        if capture_admission.reference != frozen_ref:
            raise ValueError("CaptureRepository admitted another CaptureArtifact")
        capture = capture_admission.artifact
        work_preparation = _prepare_calibration_work(capture, frozen_request)
        work_plan = work_preparation.plan
        source_binding = work_preparation.source_binding
        frame_contract = work_preparation.frame_contract
        plan_binding = _plan_binding_digest(
            frozen_ref,
            capture_repository_id=capture_repository_id,
            source_capture_evidence_digest=capture_admission.evidence_digest,
            source_capture_commit_kind=capture_admission.commit_kind,
            source_capture_commit_id=capture_admission.commit_id,
            calibration_repository_id=calibration_repository_id,
            request_fingerprint=frozen_request.fingerprint,
        )
        prepared = _PreparedCalibrationAnalysis(
            capture_admission,
            frozen_request,
            work_plan,
            source_binding,
            encode_frame_contract(frame_contract),
            plan_binding,
            capture_repository_id,
            calibration_repository_id,
            calibration_authority.token,
            context.run_id.value,
        )
        _validate_prepared_repository_resources(
            prepared,
            calibration_authority.resource_policy,
        )
        return prepared

    def execute(
        context: RunContext,
        prepared: _PreparedCalibrationAnalysis,
    ) -> _ExecutedCalibrationAnalysis:
        if not isinstance(prepared, _PreparedCalibrationAnalysis):
            raise TypeError("calibration execute requires its prepared value")
        if prepared.run_id != context.run_id.value:
            raise ValueError("prepared calibration belongs to another Run")
        validate_repository_bindings()
        context.checkpoint()
        result = analyze_calibration(prepared.capture, prepared.request)
        context.checkpoint()
        _validate_analysis_result(prepared, result)
        return _ExecutedCalibrationAnalysis(_EXECUTED_TOKEN, prepared, result)

    def cleanup(
        _context: RunContext,
        _prepared: _PreparedCalibrationAnalysis | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return CleanupReport()

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedCalibrationAnalysis,
    ) -> CalibrationArtifactRef:
        validate_repository_bindings()
        return context.commit_final(
            calibration_repository.final_commit(context, executed)
        )

    return RunPlan(
        name="calibrate committed camera capture",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(),
        hazard_claims=(),
        bound_devices=(),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        requires_final_commit=True,
    )


def _validate_analysis_result(
    prepared: _PreparedCalibrationAnalysis,
    result: CalibrationAnalysisResult,
) -> None:
    if not isinstance(result, CalibrationAnalysisResult):
        raise TypeError("calibration analysis returned another result type")
    validate_calibration_analysis_contract(
        result,
        prepared.request,
        prepared.work_plan,
        source_brackets=prepared.request.layout.brackets(
            prepared.capture.block.schema
        ),
    )
    artifact = result.artifact
    if artifact.source_binding != prepared.source_binding:
        raise ValueError("analysis source binding differs from preflight")
    if artifact.source_binding.source_capture_ref != prepared.capture.ref:
        raise ValueError("analysis result names another source CaptureArtifact")
    if encode_frame_contract(artifact.frame_contract) != prepared.frame_contract_payload:
        raise ValueError("analysis FrameContract differs from preflight")
    if _parameter_digest(artifact, "analysis-request-fingerprint") != (
        prepared.request.fingerprint
    ):
        raise ValueError("analysis request fingerprint differs from frozen request")
    if _parameter_digest(artifact, "analysis-work-plan-fingerprint") != (
        prepared.work_plan.fingerprint
    ):
        raise ValueError("analysis work-plan fingerprint differs from preflight")


def _validate_persistent_record(
    artifact: CalibrationArtifact,
    derivation: _CalibrationDerivation,
    *,
    diagnostics_payload: bytes,
    calibration_repository_id: str,
) -> CalibrationAnalysisResult:
    if not isinstance(diagnostics_payload, bytes):
        raise TypeError("diagnostics_payload must be bytes")
    result = CalibrationAnalysisResult(artifact, derivation.diagnostics)
    validate_calibration_analysis_contract(
        result,
        derivation.request,
        derivation.work_plan,
    )
    if artifact.fingerprint != derivation.artifact_fingerprint:
        raise ValueError("artifact fingerprint differs from derivation")
    if artifact.source_binding.source_capture_ref != derivation.source_capture_ref:
        raise ValueError("artifact source ref differs from derivation")
    if _source_binding_fingerprint(artifact.source_binding) != (
        derivation.source_binding_fingerprint
    ):
        raise ValueError("artifact source binding differs from derivation")
    if _parameter_digest(artifact, "analysis-request-fingerprint") != (
        derivation.request_fingerprint
    ):
        raise ValueError("artifact request fingerprint differs from derivation")
    if _parameter_digest(artifact, "analysis-work-plan-fingerprint") != (
        derivation.work_plan_fingerprint
    ):
        raise ValueError("artifact work-plan fingerprint differs from derivation")
    if sha256_digest(diagnostics_payload) != derivation.diagnostics_digest:
        raise ValueError("diagnostics digest differs from derivation")
    if _analysis_result_digest(artifact, derivation.diagnostics_digest) != (
        derivation.analysis_result_digest
    ):
        raise ValueError("analysis result digest differs from derivation")
    if (
        artifact.algorithm_id != derivation.algorithm_id
        or artifact.algorithm_version != derivation.algorithm_version
    ):
        raise ValueError("analysis algorithm differs from derivation")
    expected_plan_binding = _plan_binding_digest(
        derivation.source_capture_ref,
        capture_repository_id=derivation.source_capture_ref.repository_id,
        source_capture_evidence_digest=(
            derivation.source_capture_evidence_digest
        ),
        source_capture_commit_kind=derivation.source_capture_commit_kind,
        source_capture_commit_id=derivation.source_capture_commit_id,
        calibration_repository_id=calibration_repository_id,
        request_fingerprint=derivation.request_fingerprint,
    )
    if expected_plan_binding != derivation.plan_binding_digest:
        raise ValueError("calibration plan binding differs from persistent evidence")
    return result


def _validate_source_capture(
    artifact: CalibrationArtifact,
    derivation: _CalibrationDerivation,
    capture_admission: AdmittedCapture,
) -> None:
    admitted_capture_type, capture_artifact_type, _ = _capture_types()
    if type(capture_admission) is not admitted_capture_type:
        raise TypeError("source capture must be AdmittedCapture")
    capture = capture_admission.artifact
    if type(capture) is not capture_artifact_type:
        raise TypeError("source capture must be CaptureArtifact")
    if capture_admission.reference != derivation.source_capture_ref:
        raise ValueError("admitted source capture has another reference")
    if capture_admission.evidence_digest != derivation.source_capture_evidence_digest:
        raise ValueError("source capture admission evidence differs from derivation")
    if capture_admission.commit_kind is not derivation.source_capture_commit_kind:
        raise ValueError("source capture commit kind differs from derivation")
    if capture_admission.commit_id != derivation.source_capture_commit_id:
        raise ValueError("source capture commit id differs from derivation")
    if capture.ref != derivation.source_capture_ref:
        raise ValueError("resolved source capture has another reference")
    if capture.run_id != derivation.source_capture_run_id:
        raise ValueError("resolved source capture run_id differs from derivation")
    if capture.safety_bundle_id != derivation.source_capture_safety_bundle_id:
        raise ValueError("resolved source capture safety bundle differs from derivation")
    preparation = _prepare_calibration_work(capture, derivation.request)
    if preparation.source_binding != artifact.source_binding:
        raise ValueError("calibration source binding differs from resolved capture")
    if encode_frame_contract(preparation.frame_contract) != encode_frame_contract(
        artifact.frame_contract
    ):
        raise ValueError("calibration FrameContract differs from resolved capture")
    expected_work_plan = preparation.plan
    if (
        expected_work_plan != derivation.work_plan
        or expected_work_plan.fingerprint != derivation.work_plan_fingerprint
    ):
        raise ValueError(
            "calibration work plan differs from resolved source and request"
        )
    validate_calibration_partition_against_source(
        derivation.diagnostics,
        derivation.request,
        preparation.brackets,
    )


def _content_ref_to_tree(reference: ContentRef) -> dict[str, object]:
    return {"digest": reference.digest, "size": reference.size}


def _content_ref_from_tree(tree: Any) -> ContentRef:
    if not isinstance(tree, dict) or set(tree) != {"digest", "size"}:
        raise ValueError("calibration content reference has an unknown field set")
    return ContentRef(tree["digest"], tree["size"])


def _derivation_tree(
    derivation: _CalibrationDerivation,
    *,
    request_payload: bytes,
    work_plan_payload: bytes,
    diagnostics_payload: bytes,
) -> dict[str, object]:
    if not isinstance(derivation, _CalibrationDerivation):
        raise TypeError("derivation must be _CalibrationDerivation")
    metadata = _derivation_metadata_values(
        source_capture_ref=derivation.source_capture_ref,
        source_capture_run_id=derivation.source_capture_run_id,
        source_capture_safety_bundle_id=derivation.source_capture_safety_bundle_id,
        source_capture_evidence_digest=derivation.source_capture_evidence_digest,
        source_capture_commit_kind=derivation.source_capture_commit_kind,
        source_capture_commit_id=derivation.source_capture_commit_id,
        source_binding_fingerprint=derivation.source_binding_fingerprint,
        artifact_fingerprint=derivation.artifact_fingerprint,
        request_fingerprint=derivation.request_fingerprint,
        work_plan_fingerprint=derivation.work_plan_fingerprint,
        diagnostics_digest=derivation.diagnostics_digest,
        analysis_result_digest=derivation.analysis_result_digest,
        plan_binding_digest=derivation.plan_binding_digest,
        algorithm_id=derivation.algorithm_id,
        algorithm_version=derivation.algorithm_version,
        analysis_run_id=derivation.analysis_run_id,
        analysis_safety_bundle_id=derivation.analysis_safety_bundle_id,
    )
    return _derivation_envelope_tree(
        metadata,
        request_payload=request_payload,
        work_plan_payload=work_plan_payload,
        diagnostics_payload=diagnostics_payload,
    )


def _derivation_payload(
    derivation: _CalibrationDerivation,
    *,
    request_payload: bytes | None = None,
    work_plan_payload: bytes | None = None,
    diagnostics_payload: bytes | None = None,
) -> bytes:
    if request_payload is None:
        request_payload = encode_calibration_analysis_request(derivation.request)
    if work_plan_payload is None:
        work_plan_payload = encode_calibration_work_plan(derivation.work_plan)
    if diagnostics_payload is None:
        diagnostics_payload = encode_calibration_analysis_diagnostics(
            derivation.diagnostics,
            resource_policy=derivation.request.resource_policy,
        )
    return encode(
        _derivation_tree(
            derivation,
            request_payload=request_payload,
            work_plan_payload=work_plan_payload,
            diagnostics_payload=diagnostics_payload,
        )
    )


def _derivation_from_payload(
    payload: bytes,
) -> tuple[_CalibrationDerivation, bytes, bytes, bytes]:
    fields = {
        "schema",
        "request_payload",
        "work_plan_payload",
        "diagnostics_payload",
        "source_capture_ref",
        "source_capture_run_id",
        "source_capture_safety_bundle_id",
        "source_capture_evidence_digest",
        "source_capture_commit_kind",
        "source_capture_commit_id",
        "source_binding_fingerprint",
        "artifact_fingerprint",
        "request_fingerprint",
        "work_plan_fingerprint",
        "diagnostics_digest",
        "analysis_result_digest",
        "plan_binding_digest",
        "algorithm_id",
        "algorithm_version",
        "analysis_run_id",
        "analysis_safety_bundle_id",
    }
    tree = decode(payload, limits=_DERIVATION_DECODE_LIMITS)
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CalibrationDerivation has an unknown field set")
    if tree["schema"] != _CALIBRATION_DERIVATION_SCHEMA:
        raise ValueError("unsupported CalibrationDerivation schema")
    request_payload = tree["request_payload"]
    work_plan_payload = tree["work_plan_payload"]
    diagnostics_payload = tree["diagnostics_payload"]
    if any(
        not isinstance(value, bytes)
        for value in (request_payload, work_plan_payload, diagnostics_payload)
    ):
        raise ValueError("CalibrationDerivation owner payloads must be bytes")
    request = decode_calibration_analysis_request(request_payload)
    work_plan = decode_calibration_work_plan(work_plan_payload)
    diagnostics = decode_calibration_analysis_diagnostics(
        diagnostics_payload,
        resource_policy=request.resource_policy,
    )
    try:
        source_capture_commit_kind = CommitKind(tree["source_capture_commit_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown source capture commit kind") from exc
    derivation = _CalibrationDerivation(
        request,
        work_plan,
        diagnostics,
        capture_artifact_ref_from_tree(tree["source_capture_ref"]),
        tree["source_capture_run_id"],
        tree["source_capture_safety_bundle_id"],
        tree["source_capture_evidence_digest"],
        source_capture_commit_kind,
        tree["source_capture_commit_id"],
        tree["source_binding_fingerprint"],
        tree["artifact_fingerprint"],
        tree["request_fingerprint"],
        tree["work_plan_fingerprint"],
        tree["diagnostics_digest"],
        tree["analysis_result_digest"],
        tree["plan_binding_digest"],
        tree["algorithm_id"],
        tree["algorithm_version"],
        tree["analysis_run_id"],
        tree["analysis_safety_bundle_id"],
    )
    if _derivation_payload(
        derivation,
        request_payload=request_payload,
        work_plan_payload=work_plan_payload,
        diagnostics_payload=diagnostics_payload,
    ) != payload:
        raise ValueError("CalibrationDerivation is not canonical")
    return derivation, request_payload, work_plan_payload, diagnostics_payload


def _manifest_payload(
    *,
    repository_id: str,
    artifact_blob: ContentRef,
    derivation_blob: ContentRef,
    derivation: _CalibrationDerivation,
    resource_summary: CalibrationResourceSummary,
) -> bytes:
    metadata = _derivation_metadata_values(
        source_capture_ref=derivation.source_capture_ref,
        source_capture_run_id=derivation.source_capture_run_id,
        source_capture_safety_bundle_id=derivation.source_capture_safety_bundle_id,
        source_capture_evidence_digest=derivation.source_capture_evidence_digest,
        source_capture_commit_kind=derivation.source_capture_commit_kind,
        source_capture_commit_id=derivation.source_capture_commit_id,
        source_binding_fingerprint=derivation.source_binding_fingerprint,
        artifact_fingerprint=derivation.artifact_fingerprint,
        request_fingerprint=derivation.request_fingerprint,
        work_plan_fingerprint=derivation.work_plan_fingerprint,
        diagnostics_digest=derivation.diagnostics_digest,
        analysis_result_digest=derivation.analysis_result_digest,
        plan_binding_digest=derivation.plan_binding_digest,
        algorithm_id=derivation.algorithm_id,
        algorithm_version=derivation.algorithm_version,
        analysis_run_id=derivation.analysis_run_id,
        analysis_safety_bundle_id=derivation.analysis_safety_bundle_id,
    )
    return encode(
        _manifest_tree_from_metadata(
            repository_id=repository_id,
            artifact_blob=artifact_blob,
            derivation_blob=derivation_blob,
            metadata=metadata,
            resource_summary=resource_summary,
        )
    )


def _manifest_from_tree(tree: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "repository_id",
        "artifact_schema",
        "artifact_blob",
        "derivation_blob",
        "artifact_fingerprint",
        "source_capture_ref",
        "source_capture_run_id",
        "source_capture_safety_bundle_id",
        "source_capture_evidence_digest",
        "source_capture_commit_kind",
        "source_capture_commit_id",
        "source_binding_fingerprint",
        "request_fingerprint",
        "work_plan_fingerprint",
        "diagnostics_digest",
        "analysis_result_digest",
        "plan_binding_digest",
        "algorithm_id",
        "algorithm_version",
        "analysis_run_id",
        "analysis_safety_bundle_id",
        "evidence_digest",
        "resource_summary",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CalibrationArtifact manifest has an unknown field set")
    if tree["schema"] != CALIBRATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported CalibrationArtifact manifest schema")
    if tree["artifact_schema"] != CALIBRATION_ARTIFACT_SCHEMA:
        raise ValueError("CalibrationArtifact manifest names another artifact schema")
    _canonical_text(tree["repository_id"], "repository_id")
    _content_ref_from_tree(tree["artifact_blob"])
    _content_ref_from_tree(tree["derivation_blob"])
    capture_artifact_ref_from_tree(tree["source_capture_ref"])
    for name in (
        "artifact_fingerprint",
        "source_capture_evidence_digest",
        "source_binding_fingerprint",
        "request_fingerprint",
        "work_plan_fingerprint",
        "diagnostics_digest",
        "analysis_result_digest",
        "plan_binding_digest",
        "evidence_digest",
    ):
        _sha256(tree[name], name)
    for name in (
        "source_capture_run_id",
        "source_capture_safety_bundle_id",
        "source_capture_commit_id",
        "algorithm_id",
        "algorithm_version",
        "analysis_run_id",
    ):
        _canonical_text(tree[name], name)
    try:
        CommitKind(tree["source_capture_commit_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown source capture commit kind") from exc
    _optional_canonical_text(
        tree["analysis_safety_bundle_id"],
        "analysis_safety_bundle_id",
    )
    _resource_summary_from_tree(tree["resource_summary"])
    return tree


def _validate_manifest_evidence(
    data: dict[str, Any],
    artifact: CalibrationArtifact,
    derivation: _CalibrationDerivation,
    derivation_blob: ContentRef,
) -> None:
    expected = {
        "artifact_fingerprint": artifact.fingerprint,
        "source_capture_ref": capture_artifact_ref_to_tree(
            derivation.source_capture_ref
        ),
        "source_capture_run_id": derivation.source_capture_run_id,
        "source_capture_safety_bundle_id": derivation.source_capture_safety_bundle_id,
        "source_capture_evidence_digest": (
            derivation.source_capture_evidence_digest
        ),
        "source_capture_commit_kind": derivation.source_capture_commit_kind.value,
        "source_capture_commit_id": derivation.source_capture_commit_id,
        "source_binding_fingerprint": derivation.source_binding_fingerprint,
        "request_fingerprint": derivation.request_fingerprint,
        "work_plan_fingerprint": derivation.work_plan_fingerprint,
        "diagnostics_digest": derivation.diagnostics_digest,
        "analysis_result_digest": derivation.analysis_result_digest,
        "plan_binding_digest": derivation.plan_binding_digest,
        "algorithm_id": derivation.algorithm_id,
        "algorithm_version": derivation.algorithm_version,
        "analysis_run_id": derivation.analysis_run_id,
        "analysis_safety_bundle_id": derivation.analysis_safety_bundle_id,
        "evidence_digest": derivation_blob.digest,
    }
    for name, value in expected.items():
        if data[name] != value:
            raise ValueError(f"calibration manifest {name} differs from evidence")


def _resource_summary_to_tree(
    summary: CalibrationResourceSummary,
) -> dict[str, int]:
    if not isinstance(summary, CalibrationResourceSummary):
        raise TypeError("summary must be CalibrationResourceSummary")
    return {
        "site_count": summary.site_count,
        "model_count": summary.model_count,
        "kernel_elements": summary.kernel_elements,
        "max_sampled_pixels_per_model": summary.max_sampled_pixels_per_model,
        "total_sampled_pixels_all_models": summary.total_sampled_pixels_all_models,
    }


def _resource_summary_from_tree(tree: Any) -> CalibrationResourceSummary:
    fields = {
        "site_count",
        "model_count",
        "kernel_elements",
        "max_sampled_pixels_per_model",
        "total_sampled_pixels_all_models",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("calibration resource summary has an unknown field set")
    if any(type(tree[field]) is not int for field in fields):
        raise ValueError("calibration resource summary fields must be canonical integers")
    return CalibrationResourceSummary(
        tree["site_count"],
        tree["model_count"],
        tree["kernel_elements"],
        tree["max_sampled_pixels_per_model"],
        tree["total_sampled_pixels_all_models"],
    )


__all__ = [
    "AdmittedCalibration",
    "CALIBRATION_MANIFEST_SCHEMA",
    "CalibrationRepository",
    "compile_calibration_artifact_plan",
]
