"""Virtual-only durable authority for autonomous scan captures.

The repository is intentionally thin: camera frame bytes remain owned by the
FINAL ``CaptureArtifact``.  A scan manifest adds only its authoritative output
contract and the exact PulseDocument from which the physical scan domain is
derived; compiled lineage is already sealed into the source capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from zlc_data import (
    BlockId,
    READOUT_EVENT,
    DatasetSchema,
    apply_transform,
)
from zlc_neutral_atom.artifacts.capture import (
    AdmittedCapture,
    CaptureArtifact,
    CaptureRepository,
)
from zlc_neutral_atom.capture_reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.timing.pulse import PulseTerminalEvidenceKind
from zlc_neutral_atom.timing.lineage import PulseCaptureEvidence
from zlc_neutral_atom.runtime.commit import (
    PublishVisibilityUnknown,
    publish_manifest_with_visibility_reconciliation,
)
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    expand_autonomous_scan_repeats,
    pulse_document_from_tree,
    pulse_document_to_tree,
)
from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalDecodeLimits,
    CanonicalEncodingError,
    ContentAddressedStore,
    ContentRef,
    ContentSizeLimitError,
    RepositoryRootLease,
    canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    positive_integer,
    sha256_digest,
)

from .contracts import (
    MaterializedScanData,
    ScanOutputContract,
    ScanPointTable,
    scan_capture_block_id,
    scan_intent_digest_from_block_id,
    scan_output_contract_from_tree,
    scan_output_contract_to_tree,
)
from .reference import SCAN_ARTIFACT_NAMESPACE, ScanArtifactRef


SCAN_ARTIFACT_SCHEMA = "zlc_neutral_atom.ScanArtifact"
_SCAN_INTENT_SCHEMA = "zlc_neutral_atom.ScanIntent"
_SCAN_INTENT_NAMESPACE = "scan-intent"
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "repository_id",
        "source_capture_ref",
        "scan_intent",
    }
)
_INTENT_FIELDS = frozenset(
    {
        "schema",
        "repository_id",
        "capture_repository_id",
        "pulse_document_blob",
        "output_contract",
    }
)


class ScanResourceExceeded(RuntimeError):
    """Scan metadata exceeds an explicit repository admission budget."""


class HardwareScanAuthorityUnavailable(RuntimeError):
    """Hardware receipts cannot yet be promoted to current ScanArtifact authority."""


class ScanCommitVisibilityUnknown(PublishVisibilityUnknown):
    """A final scan manifest needs explicit visibility reconciliation."""

    def __init__(self, reference: ScanArtifactRef) -> None:
        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("reference must be ScanArtifactRef")
        self.reference = reference
        super().__init__(
            "scan manifest visibility is unknown; retry promotion with the "
            f"same raw capture or inspect {reference!r}"
        )


@dataclass(frozen=True, slots=True)
class ScanRepositoryResourcePolicy:
    """Finite limits for one metadata-only scan artifact."""

    max_manifest_bytes: int = 16 * 1024 * 1024
    max_pulse_document_blob_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for field in (
            "max_manifest_bytes",
            "max_pulse_document_blob_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")


DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY = ScanRepositoryResourcePolicy()
_SCAN_CANONICAL_LIMITS = CanonicalDecodeLimits(
    max_arrays=0,
    max_total_array_bytes=0,
)


@dataclass(frozen=True, slots=True)
class ScanArtifact:
    """An admitted virtual scan view over one FINAL raw capture."""

    ref: ScanArtifactRef
    source_capture_ref: CaptureArtifactRef
    pulse_document: PulseDocument
    output_contract: ScanOutputContract

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ScanArtifactRef):
            raise TypeError("ref must be ScanArtifactRef")
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        if not isinstance(self.output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")

def _reject_arrays(events) -> None:
    if any(isinstance(event, CanonicalArrayEvent) for event in events):
        raise ScanResourceExceeded("scan authority metadata cannot embed ndarrays")


def _encode_pulse_document(
    document: PulseDocument,
    policy: ScanRepositoryResourcePolicy,
) -> bytes:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(policy, ScanRepositoryResourcePolicy):
        raise TypeError("policy must be ScanRepositoryResourcePolicy")
    try:
        payload = encode(
            pulse_document_to_tree(document),
            limits=_SCAN_CANONICAL_LIMITS,
        )
    except CanonicalEncodingError as exc:
        raise ScanResourceExceeded(
            "PulseDocument exceeds canonical repository policy"
        ) from exc
    if len(payload) > policy.max_pulse_document_blob_bytes:
        raise ScanResourceExceeded("pulse document exceeds repository policy")
    return payload


def _decode_pulse_document(
    payload: bytes,
    policy: ScanRepositoryResourcePolicy,
) -> PulseDocument:
    if not isinstance(payload, bytes):
        raise TypeError("pulse document payload must be bytes")
    if len(payload) > policy.max_pulse_document_blob_bytes:
        raise ScanResourceExceeded("pulse document exceeds repository policy")
    document = pulse_document_from_tree(
        decode(
            payload,
            admit_structure=_reject_arrays,
            limits=_SCAN_CANONICAL_LIMITS,
        )
    )
    if _encode_pulse_document(document, policy) != payload:
        raise ValueError("PulseDocument blob is typed but non-canonical")
    return document


def _intent_payload(
    *,
    repository_id: str,
    capture_repository_id: str,
    pulse_document_blob: ContentRef,
    output_contract: ScanOutputContract,
    policy: ScanRepositoryResourcePolicy,
) -> bytes:
    try:
        payload = encode(
            {
                "schema": _SCAN_INTENT_SCHEMA,
                "repository_id": canonical_text(repository_id, "repository_id"),
                "capture_repository_id": canonical_text(
                    capture_repository_id,
                    "capture_repository_id",
                ),
                "pulse_document_blob": content_ref_to_tree(pulse_document_blob),
                "output_contract": scan_output_contract_to_tree(output_contract),
            },
            limits=_SCAN_CANONICAL_LIMITS,
        )
    except CanonicalEncodingError as exc:
        raise ScanResourceExceeded(
            "scan intent exceeds canonical repository policy"
        ) from exc
    if len(payload) > policy.max_manifest_bytes:
        raise ScanResourceExceeded("scan intent exceeds repository policy")
    return payload


def _manifest_payload(
    *,
    repository_id: str,
    source_capture_ref: CaptureArtifactRef,
    scan_intent: ContentRef,
    policy: ScanRepositoryResourcePolicy,
) -> bytes:
    try:
        payload = encode(
            {
                "schema": SCAN_ARTIFACT_SCHEMA,
                "repository_id": canonical_text(repository_id, "repository_id"),
                "source_capture_ref": capture_artifact_ref_to_tree(
                    source_capture_ref
                ),
                "scan_intent": content_ref_to_tree(scan_intent),
            },
            limits=_SCAN_CANONICAL_LIMITS,
        )
    except CanonicalEncodingError as exc:
        raise ScanResourceExceeded(
            "scan manifest exceeds canonical repository policy"
        ) from exc
    if len(payload) > policy.max_manifest_bytes:
        raise ScanResourceExceeded("scan manifest exceeds repository policy")
    return payload


def _decode_intent(
    payload: bytes,
    policy: ScanRepositoryResourcePolicy,
) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("scan intent payload must be bytes")
    if len(payload) > policy.max_manifest_bytes:
        raise ScanResourceExceeded("scan intent exceeds repository policy")
    return exact_mapping(
        decode(
            payload,
            admit_structure=_reject_arrays,
            limits=_SCAN_CANONICAL_LIMITS,
        ),
        _INTENT_FIELDS,
        _SCAN_INTENT_SCHEMA,
    )


def _decode_manifest(
    payload: bytes,
    policy: ScanRepositoryResourcePolicy,
) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("scan manifest payload must be bytes")
    if len(payload) > policy.max_manifest_bytes:
        raise ScanResourceExceeded("scan manifest exceeds repository policy")
    return exact_mapping(
        decode(
            payload,
            admit_structure=_reject_arrays,
            limits=_SCAN_CANONICAL_LIMITS,
        ),
        _MANIFEST_FIELDS,
        SCAN_ARTIFACT_SCHEMA,
    )


def _publish_manifest_exactly(
    *,
    store,
    namespace: str,
    payload: bytes,
    max_bytes: int,
    accept_visible_interrupt: bool,
) -> ContentRef:
    """Publish or classify the exact content-addressed visibility point."""

    digest = sha256_digest(payload)
    reference = ContentRef(digest, len(payload))
    try:
        stored = publish_manifest_with_visibility_reconciliation(
            store,
            namespace,
            payload,
            expected_digest=digest,
            max_bytes=max_bytes,
        )
    except PublishVisibilityUnknown as error:
        try:
            visible = store.confirm_manifest_durable(
                namespace,
                digest,
                max_bytes=max_bytes,
            )
        except FileNotFoundError:
            cause = error.__cause__
            if isinstance(cause, BaseException):
                raise cause
            raise
        except BaseException as confirmation_error:
            if (
                not accept_visible_interrupt
                and isinstance(confirmation_error, KeyboardInterrupt)
            ):
                raise
            raise PublishVisibilityUnknown(
                "manifest visibility could not be durably confirmed"
            ) from confirmation_error
        if visible != payload:
            raise PublishVisibilityUnknown(
                "visible manifest differs from its expected canonical payload"
            ) from error
        if not accept_visible_interrupt and isinstance(
            error.__cause__, KeyboardInterrupt
        ):
            raise error.__cause__
        return reference
    if stored.content != reference:
        raise RuntimeError("published manifest identity changed")
    return reference


def _require_virtual_autonomous_capture(
    capture: CaptureArtifact,
    document: PulseDocument,
) -> None:
    """Validate facts independent of the scan contract's named domain."""

    if not isinstance(capture, CaptureArtifact):
        raise TypeError("capture must be CaptureArtifact")
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    evidence = capture.pulse_evidence
    if evidence is None:
        raise ValueError("scan authority requires persisted pulse lineage")
    evidence_kind = evidence.terminal.evidence_kind
    if evidence_kind is not PulseTerminalEvidenceKind.SIMULATED:
        raise HardwareScanAuthorityUnavailable(
            "hardware terminal receipts remain typed NO-GO for ScanArtifact"
        )
    compiled = evidence.compiled_artifact
    if compiled.execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        raise ValueError("scan authority requires AUTONOMOUS_SCAN_ONCE capture")
    table = document.scan_table
    if table is None or not table.rows:
        raise ValueError("autonomous scan authority requires a frozen scan table")
    schedule = evidence.trigger_schedule
    execution_document = expand_autonomous_scan_repeats(document)
    if compiled.source_document_digest != execution_document.fingerprint:
        raise ValueError(
            "compiled capture lineage differs from the deterministic repeat-major "
            "execution document"
        )
    repeat_count = 1 if document.repeat is None else document.repeat.count
    if (
        schedule.point_count != repeat_count * len(table.rows)
        or schedule.loop_count != 1
        or not schedule.full_point_loop
    ):
        raise ValueError(
            "compiled pulse is not one complete repeat-major finite scan table"
        )
    schema = capture.frame_source.schema
    event_axes = tuple(axis for axis in schema.point_axes if axis.role == READOUT_EVENT)
    if len(event_axes) != 1 or event_axes[0].size != 1:
        raise ValueError(
            "current scan authority requires one singleton raw READOUT_EVENT axis"
        )
    if schema.repeat_axis.size != repeat_count:
        raise ValueError("source repeat axis differs from the logical PulseDocument")


class ScanRepository:
    """Current-only CAS authority for virtual autonomous scan metadata."""

    __slots__ = (
        "root",
        "repository_id",
        "resource_policy",
        "_lock",
        "_closed",
        "_root_lease",
        "_store",
        "_store_authority",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("ScanRepository is final and cannot be subclassed")

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-scan",
        resource_policy: ScanRepositoryResourcePolicy = (
            DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY
        ),
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        object.__setattr__(
            self,
            "repository_id",
            canonical_text(repository_id, "repository_id"),
        )
        if not isinstance(resource_policy, ScanRepositoryResourcePolicy):
            raise TypeError("resource_policy must be ScanRepositoryResourcePolicy")
        object.__setattr__(self, "resource_policy", resource_policy)
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_root_lease", RepositoryRootLease(self.root))
        try:
            store = ContentAddressedStore(self.root / "content")
            object.__setattr__(self, "_store", store)
            object.__setattr__(self, "_store_authority", store.authority())
        except BaseException:
            self._root_lease.close()
            raise
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ScanRepository authority is immutable")
        object.__setattr__(self, _name, _value)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("scan repository is closed")
        self._root_lease.require_active()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._root_lease.close()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> "ScanRepository":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def _validate_reference(self, reference: ScanArtifactRef) -> None:
        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("reference must be ScanArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("ScanArtifactRef belongs to another repository")

    def identify_intent(
        self,
        document: PulseDocument,
        output_contract: ScanOutputContract,
        *,
        capture_repository_id: str,
    ) -> BlockId:
        """Compute the exact bounded pre-FIRE identity without publishing it."""

        with self._lock:
            self._require_open()
            with self._root_lease.borrow() as borrow:
                borrow.require_active()
                _document_payload, _document_ref, _intent_payload_bytes, intent_ref = (
                    self._freeze_intent(
                        document,
                        output_contract,
                        capture_repository_id=capture_repository_id,
                    )
                )
                return scan_capture_block_id(intent_ref.digest)

    def prepare(
        self,
        document: PulseDocument,
        output_contract: ScanOutputContract,
        *,
        capture_repository_id: str,
    ) -> BlockId:
        """Durably publish bounded scan intent before runtime start/FIRE."""

        with self._lock:
            self._require_open()
            with self._root_lease.borrow() as borrow:
                borrow.require_active()
                document_payload, document_ref, intent_payload, intent_ref = (
                    self._freeze_intent(
                        document,
                        output_contract,
                        capture_repository_id=capture_repository_id,
                    )
                )
                if self._store_authority.put_blob(document_payload) != document_ref:
                    raise RuntimeError("published PulseDocument blob identity changed")
                published_intent = _publish_manifest_exactly(
                    store=self._store_authority,
                    namespace=_SCAN_INTENT_NAMESPACE,
                    payload=intent_payload,
                    max_bytes=self.resource_policy.max_manifest_bytes,
                    accept_visible_interrupt=False,
                )
                if published_intent != intent_ref:
                    raise RuntimeError("published scan intent identity changed")
                return scan_capture_block_id(intent_ref.digest)

    def _freeze_intent(
        self,
        document: PulseDocument,
        output_contract: ScanOutputContract,
        *,
        capture_repository_id: str,
    ) -> tuple[bytes, ContentRef, bytes, ContentRef]:
        """Build the one canonical intent used by inspect and durable prepare."""

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        capture_repository_id = canonical_text(
            capture_repository_id,
            "capture_repository_id",
        )
        document_payload = _encode_pulse_document(document, self.resource_policy)
        document_ref = self._store_authority.identify_blob(document_payload)
        intent_payload = _intent_payload(
            repository_id=self.repository_id,
            capture_repository_id=capture_repository_id,
            pulse_document_blob=document_ref,
            output_contract=output_contract,
            policy=self.resource_policy,
        )
        intent_ref = ContentRef(sha256_digest(intent_payload), len(intent_payload))
        _manifest_payload(
            repository_id=self.repository_id,
            source_capture_ref=CaptureArtifactRef(
                capture_repository_id,
                "0" * 64,
            ),
            scan_intent=intent_ref,
            policy=self.resource_policy,
        )
        return document_payload, document_ref, intent_payload, intent_ref

    def promote(self, admitted_capture: AdmittedCapture) -> ScanArtifactRef:
        """Recover promotion solely from a FINAL raw capture's durable intent."""

        if type(admitted_capture) is not AdmittedCapture:
            raise TypeError("admitted_capture must be exact AdmittedCapture authority")
        with self._lock:
            self._require_open()
            capture_ref = admitted_capture.reference
            capture = admitted_capture.artifact
            intent_digest = scan_intent_digest_from_block_id(
                capture.frame_source.block_id
            )
            (
                intent_ref,
                capture_repository_id,
                document,
                point_table,
                output_contract,
            ) = self._load_intent(
                intent_digest,
                source_schema=capture.frame_source.schema,
            )
            if capture_ref.repository_id != capture_repository_id:
                raise ValueError("raw capture belongs to another scan intent")
            _require_virtual_autonomous_capture(capture, document)
            _require_capture_matches_intent(
                point_table,
                capture,
                intent_ref.digest,
            )
            return self._publish_scan(capture_ref, intent_ref)

    def admit(
        self,
        reference: ScanArtifactRef,
        capture_repository: CaptureRepository,
    ) -> ScanArtifact:
        """Load and re-admit every dependency before returning an admitted view."""

        artifact, _source = self._admit_with_source(reference, capture_repository)
        return artifact

    def materialize(
        self,
        reference: ScanArtifactRef,
        capture_repository: CaptureRepository,
        *,
        capture_memory_limit_bytes: int,
    ) -> MaterializedScanData:
        """Apply the frozen authority transform without collapsing named axes."""

        limit = positive_integer(
            capture_memory_limit_bytes,
            "capture_memory_limit_bytes",
        )
        artifact, source = self._admit_with_source(reference, capture_repository)
        snapshot = source.materialize_snapshot(memory_limit_bytes=limit)
        return MaterializedScanData(
            apply_transform(
                snapshot,
                artifact.output_contract.committed_transform,
            ),
            artifact.output_contract,
        )

    def _admit_with_source(
        self,
        reference: ScanArtifactRef,
        capture_repository: CaptureRepository,
    ) -> tuple[ScanArtifact, AdmittedCapture]:
        if not isinstance(capture_repository, CaptureRepository):
            raise TypeError("capture_repository must be CaptureRepository")
        with self._lock:
            self._require_open()
            self._validate_reference(reference)
            with self._root_lease.borrow() as borrow:
                borrow.require_active()
                try:
                    payload = self._store_authority.read_manifest(
                        SCAN_ARTIFACT_NAMESPACE,
                        reference.manifest_digest,
                        max_bytes=self.resource_policy.max_manifest_bytes,
                    )
                except ContentSizeLimitError as exc:
                    raise ScanResourceExceeded(
                        "scan manifest exceeds repository policy"
                    ) from exc
                data = _decode_manifest(payload, self.resource_policy)
                if data["repository_id"] != self.repository_id:
                    raise ValueError("ScanArtifact belongs to another repository")
                capture_ref = capture_artifact_ref_from_tree(
                    data["source_capture_ref"]
                )
                intent_ref = content_ref_from_tree(data["scan_intent"])
                admitted_capture = capture_repository.admit(capture_ref)
                capture = admitted_capture.artifact
                (
                    resolved_intent_ref,
                    capture_repository_id,
                    document,
                    point_table,
                    output_contract,
                ) = self._load_intent(
                    intent_ref,
                    source_schema=capture.frame_source.schema,
                )
                if resolved_intent_ref != intent_ref:
                    raise RuntimeError("resolved scan intent identity changed")
                if capture_ref.repository_id != capture_repository_id:
                    raise ValueError("raw capture belongs to another scan intent")
                _require_virtual_autonomous_capture(capture, document)
                rebuilt = _manifest_payload(
                    repository_id=self.repository_id,
                    source_capture_ref=capture_ref,
                    scan_intent=intent_ref,
                    policy=self.resource_policy,
                )
                if rebuilt != payload or sha256_digest(rebuilt) != reference.manifest_digest:
                    raise ValueError("ScanArtifact manifest is not canonical")
                _require_capture_matches_intent(
                    point_table,
                    capture,
                    intent_ref.digest,
                )
                return (
                    ScanArtifact(
                        reference,
                        capture_ref,
                        document,
                        output_contract,
                    ),
                    admitted_capture,
                )

    def _load_intent(
        self,
        reference: ContentRef | str,
        *,
        source_schema: DatasetSchema,
    ) -> tuple[
        ContentRef,
        str,
        PulseDocument,
        ScanPointTable,
        ScanOutputContract,
    ]:
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if isinstance(reference, ContentRef):
            intent_ref = reference
            digest = reference.digest
            if reference.size > self.resource_policy.max_manifest_bytes:
                raise ScanResourceExceeded("scan intent exceeds repository policy")
        else:
            digest = reference
            intent_ref = None
        try:
            payload = self._store_authority.read_manifest(
                _SCAN_INTENT_NAMESPACE,
                digest,
                max_bytes=self.resource_policy.max_manifest_bytes,
            )
        except ContentSizeLimitError as exc:
            raise ScanResourceExceeded("scan intent exceeds repository policy") from exc
        resolved_ref = ContentRef(digest, len(payload))
        if intent_ref is not None and resolved_ref != intent_ref:
            raise ValueError("scan intent size differs from its ContentRef")
        data = _decode_intent(payload, self.resource_policy)
        if data["repository_id"] != self.repository_id:
            raise ValueError("scan intent belongs to another repository")
        capture_repository_id = canonical_text(
            data["capture_repository_id"],
            "capture_repository_id",
        )
        document_ref = content_ref_from_tree(data["pulse_document_blob"])
        if document_ref.size > self.resource_policy.max_pulse_document_blob_bytes:
            raise ScanResourceExceeded("pulse document exceeds repository policy")
        try:
            document_payload = self._store_authority.read_blob(
                document_ref,
                max_bytes=self.resource_policy.max_pulse_document_blob_bytes,
            )
        except ContentSizeLimitError as exc:
            raise ScanResourceExceeded("pulse document exceeds repository policy") from exc
        document = _decode_pulse_document(document_payload, self.resource_policy)
        point_table = ScanPointTable.from_pulse_document(document)
        output_contract = scan_output_contract_from_tree(
            data["output_contract"],
            input_schema=source_schema,
            scan_points=point_table,
        )
        rebuilt = _intent_payload(
            repository_id=self.repository_id,
            capture_repository_id=capture_repository_id,
            pulse_document_blob=document_ref,
            output_contract=output_contract,
            policy=self.resource_policy,
        )
        if rebuilt != payload or sha256_digest(rebuilt) != resolved_ref.digest:
            raise ValueError("scan intent is not canonical")
        return (
            resolved_ref,
            capture_repository_id,
            document,
            point_table,
            output_contract,
        )

    def _publish_scan(
        self,
        capture_ref: CaptureArtifactRef,
        intent_ref: ContentRef,
    ) -> ScanArtifactRef:
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            payload = _manifest_payload(
                repository_id=self.repository_id,
                source_capture_ref=capture_ref,
                scan_intent=intent_ref,
                policy=self.resource_policy,
            )
            reference = ScanArtifactRef(
                self.repository_id,
                sha256_digest(payload),
            )
            try:
                published = _publish_manifest_exactly(
                    store=self._store_authority,
                    namespace=SCAN_ARTIFACT_NAMESPACE,
                    payload=payload,
                    max_bytes=self.resource_policy.max_manifest_bytes,
                    accept_visible_interrupt=True,
                )
            except PublishVisibilityUnknown as error:
                raise ScanCommitVisibilityUnknown(reference) from error
            if published.digest != reference.manifest_digest:
                raise RuntimeError("published scan manifest identity changed")
            return reference


def _require_capture_matches_intent(
    point_table: ScanPointTable,
    capture: CaptureArtifact,
    intent_manifest_digest: str,
) -> None:
    """Cross-check durable scan intent against the FINAL capture lineage."""

    if not isinstance(point_table, ScanPointTable):
        raise TypeError("point_table must be ScanPointTable")
    evidence = capture.pulse_evidence
    assert evidence is not None
    if evidence.join_contract.scan_point_layout != point_table.point_layout:
        raise ValueError("pulse cell mapping differs from ScanPointTable layout")
    if capture.frame_source.block_id != scan_capture_block_id(
        intent_manifest_digest
    ):
        raise ValueError("raw Capture BlockId differs from the frozen scan intent")


__all__ = [
    "DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY",
    "HardwareScanAuthorityUnavailable",
    "SCAN_ARTIFACT_SCHEMA",
    "ScanCommitVisibilityUnknown",
    "ScanArtifact",
    "ScanRepository",
    "ScanRepositoryResourcePolicy",
    "ScanResourceExceeded",
]
