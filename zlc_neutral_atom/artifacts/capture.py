"""Current lazy raw-frame CaptureArtifact and crash-safe repository."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from zlc_data import StreamGenerationId
from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalDecodeLimits,
    ContentAddressedStore,
    ContentRef,
    ContentSizeLimitError,
    ContentStoreAuthority,
    RepositoryRootLease,
    RepositoryRootLeaseBorrow,
    canonical_text as _canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping as _exact_map,
    nonnegative_integer as _integer,
    sha256_digest,
    sha256_text as _sha256,
)

from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraFrameMetadataContract,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.camera_operator import (
    CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
)
from zlc_neutral_atom.capture_reference import (
    CAPTURE_ARTIFACT_NAMESPACE,
    CaptureArtifactRef,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.runtime.commit import (
    CheckpointCommit,
    CommitIntent,
    CommitKind,
    CommitRecovery,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
    publish_manifest_with_visibility_reconciliation,
)
from zlc_neutral_atom.readout.codec import (
    camera_capture_descriptor_from_tree,
    camera_capture_descriptor_to_tree,
    readout_binding_key_from_tree,
    readout_binding_key_to_tree,
)
from zlc_neutral_atom.runtime.capture import (
    CameraCapabilityEvidence,
    CameraCaptureProvenance,
    CaptureTerminalAck,
    FrozenCaptureSpec,
    camera_capability_evidence_from_tree,
    camera_capability_evidence_to_tree,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetSealProvenance,
    dataset_cell_permutation_digest,
    dataset_consumer_contract_digest,
)
from zlc_neutral_atom.runtime.pipeline import (
    MinimalPipelineSpec,
    PipelineResult,
    compile_pipeline,
    dataset_storage_nbytes,
    estimate_pipeline_peak_bytes,
)
from zlc_neutral_atom.runtime.run import CleanupReport, PostSafetyContext, RunContext, RunPlan
from zlc_neutral_atom.runtime.streams import (
    StreamId,
    TraceBinding,
)
from zlc_neutral_atom.timing.capture import (
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.timing.capture_plan import (
    CompiledCaptureCellPlan,
    decode_compiled_capture_cell_plan,
    encode_compiled_capture_cell_plan,
)
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding, PulseCaptureLineage
from zlc_neutral_atom.timing.pulse import (
    pulse_terminal_ack_from_tree,
    pulse_terminal_ack_to_tree,
)

from .capture_frames import (
    CaptureFrameSource,
    _FrameResourceExceeded,
    _capture_frame_source_scratch_upper_bound,
    _load_capture_frame_source,
    _stage_capture_frame_source,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)


CAPTURE_ARTIFACT_SCHEMA = "zlc_neutral_atom.CaptureArtifact"
_CAPTURE_ARTIFACT_KIND = "capture"
_ADMITTED_CAPTURE_EVIDENCE_SCHEMA = "zlc_neutral_atom.AdmittedCaptureEvidence"
_ADMITTED_CAPTURE_TOKEN = object()


class CaptureResourceExceeded(RuntimeError):
    """A capture exceeds an explicit repository admission budget."""


@dataclass(frozen=True)
class CaptureRepositoryResourcePolicy:
    """Finite limits for one lazy binary-frame capture."""

    max_cells: int = 1_000_000
    max_manifest_bytes: int = 16 * 1024 * 1024
    max_total_frame_bytes: int = 8 * 1024 * 1024 * 1024
    max_frame_chunk_blob_bytes: int = 512 * 1024 * 1024
    max_frame_index_blob_bytes: int = 512 * 1024 * 1024
    max_compiled_pulse_blob_bytes: int = 256 * 1024 * 1024
    max_cell_plan_blob_bytes: int = 512 * 1024 * 1024
    max_canonical_nodes: int = 32_000_000
    max_canonical_container_entries: int = 16_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_cells",
            "max_manifest_bytes",
            "max_total_frame_bytes",
            "max_frame_chunk_blob_bytes",
            "max_frame_index_blob_bytes",
            "max_compiled_pulse_blob_bytes",
            "max_cell_plan_blob_bytes",
            "max_canonical_nodes",
            "max_canonical_container_entries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY = CaptureRepositoryResourcePolicy()


class AdmittedCapture:
    """Process-local proof that one exact CaptureArtifact target was committed."""

    __slots__ = (
        "_token",
        "_repository_token",
        "_reference",
        "_artifact",
        "_commit_kind",
        "_commit_id",
        "_evidence_digest",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("AdmittedCapture is final and cannot be subclassed")

    def __init__(
        self,
        token: object,
        *,
        repository_token: object,
        reference: CaptureArtifactRef,
        artifact: "CaptureArtifact",
        commit_kind: CommitKind,
        commit_id: str,
        evidence_digest: str,
    ) -> None:
        if token is not _ADMITTED_CAPTURE_TOKEN:
            raise PermissionError(
                "AdmittedCapture can only be minted by CaptureRepository.admit"
            )
        if repository_token is None:
            raise ValueError("AdmittedCapture repository authority is absent")
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        if not isinstance(artifact, CaptureArtifact):
            raise TypeError("artifact must be CaptureArtifact")
        if not isinstance(commit_kind, CommitKind):
            raise TypeError("commit_kind must be CommitKind")
        _canonical_text(commit_id, "commit_id")
        _sha256(evidence_digest, "evidence_digest")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository_token", repository_token)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_artifact", artifact)
        object.__setattr__(self, "_commit_kind", commit_kind)
        object.__setattr__(self, "_commit_id", commit_id)
        object.__setattr__(self, "_evidence_digest", evidence_digest)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("AdmittedCapture is immutable")

    def __reduce__(self):
        raise TypeError("AdmittedCapture is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("AdmittedCapture is process-local and cannot be serialized")

    def _require_authority(self) -> None:
        if (
            type(self) is not AdmittedCapture
            or self._token is not _ADMITTED_CAPTURE_TOKEN
            or self._repository_token is None
        ):
            raise PermissionError("AdmittedCapture authority is invalid")

    @property
    def reference(self) -> CaptureArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def artifact(self) -> "CaptureArtifact":
        self._require_authority()
        return self._artifact

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

    def _matches_admission(self, other: object) -> bool:
        """Compare exact process-local repository and journal authority."""

        self._require_authority()
        if type(other) is not AdmittedCapture:
            return False
        other._require_authority()
        return (
            self._repository_token is other._repository_token
            and self._reference == other._reference
            and self._commit_kind is other._commit_kind
            and self._commit_id == other._commit_id
            and self._evidence_digest == other._evidence_digest
        )


@dataclass(frozen=True)
class CaptureArtifact:
    ref: CaptureArtifactRef
    frame_source: CaptureFrameSource
    coverage: DatasetCoverage
    provenance: DatasetSealProvenance
    terminal: CaptureTerminalAck
    aggregate_peak_bytes: int
    camera_provenance: CameraCaptureProvenance
    camera_capability_evidence: CameraCapabilityEvidence
    camera_arm_spec: FrozenCaptureSpec
    run_id: str
    safety_bundle_id: str
    chain_contract_digest: str
    pulse_lineage: PulseCaptureLineage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, CaptureArtifactRef):
            raise TypeError("ref must be CaptureArtifactRef")
        if not isinstance(self.frame_source, CaptureFrameSource):
            raise TypeError("frame_source must be CaptureFrameSource")
        schema = self.frame_source.schema
        metadata = self.frame_source.metadata_in_event_order
        schedule = self.frame_source.cell_schedule
        if not isinstance(self.coverage, DatasetCoverage) or not self.coverage.complete:
            raise ValueError("CaptureArtifact requires complete dataset coverage")
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if self.provenance.derivation is not None:
            raise ValueError(
                "CaptureArtifact is the raw boundary and cannot persist processor output"
            )
        if not isinstance(self.terminal, CaptureTerminalAck):
            raise TypeError("terminal must be CaptureTerminalAck")
        peak = _integer(self.aggregate_peak_bytes, "aggregate_peak_bytes")
        if peak == 0:
            raise ValueError("aggregate_peak_bytes must be positive")
        object.__setattr__(self, "aggregate_peak_bytes", peak)
        if not isinstance(self.camera_provenance, CameraCaptureProvenance):
            raise TypeError("camera_provenance must be CameraCaptureProvenance")
        self.camera_provenance.validate_schema(schema)
        if not isinstance(
            self.camera_capability_evidence,
            CameraCapabilityEvidence,
        ):
            raise TypeError(
                "camera_capability_evidence must be CameraCapabilityEvidence"
            )
        capability_evidence = self.camera_capability_evidence
        if not isinstance(self.camera_arm_spec, FrozenCaptureSpec):
            raise TypeError("camera_arm_spec must be FrozenCaptureSpec")
        arm_spec = self.camera_arm_spec
        if arm_spec.owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
            raise ValueError("camera arm spec belongs to an unknown owner")
        decoded_arm_spec = decode_camera_capture_spec(arm_spec.payload)
        if freeze_camera_capture_spec(decoded_arm_spec) != arm_spec:
            raise ValueError("camera arm spec is not the canonical owner encoding")
        if decoded_arm_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
            raise ValueError(
                "finite raw CaptureArtifact requires EXTERNAL_TRIGGERED camera mode"
            )
        if capability_evidence.fingerprint != self.terminal.capability_fingerprint:
            raise ValueError(
                "canonical camera capability evidence differs from terminal evidence"
            )
        if (
            capability_evidence.fingerprint
            != self.camera_provenance.capability_fingerprint
        ):
            raise ValueError(
                "canonical camera capability evidence differs from provenance"
            )
        if (
            capability_evidence.settings_fingerprint
            != self.terminal.settings_fingerprint
        ):
            raise ValueError(
                "camera capability settings differ from terminal evidence"
            )
        capability_evidence.physical_facts.validate_descriptor(
            self.camera_provenance.descriptor
        )
        expected_payload_contract = CameraSampleContract(schema.cell_schema)
        if (
            capability_evidence.payload_contract_fingerprint
            != expected_payload_contract.fingerprint
        ):
            raise ValueError(
                "camera capability payload contract differs from persisted DataBlock"
            )
        if (
            capability_evidence.capture_spec_owner_fingerprint
            != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT
        ):
            raise ValueError(
                "camera capability capture-spec owner is not the current camera owner"
            )
        if (
            self.camera_provenance.active_settings_fingerprint
            != self.terminal.settings_fingerprint
        ):
            raise ValueError(
                "camera readout settings differ from terminal capture evidence"
            )
        if (
            self.camera_provenance.camera_arm_spec_fingerprint
            != self.terminal.capture_spec_fingerprint
        ):
            raise ValueError(
                "camera arm spec differs from terminal capture evidence"
            )
        if arm_spec.digest != self.terminal.capture_spec_fingerprint:
            raise ValueError(
                "canonical camera arm spec differs from terminal capture evidence"
            )
        if self.camera_provenance.binding_id != self.terminal.binding_id:
            raise ValueError("camera binding_id differs from terminal evidence")
        if (
            self.camera_provenance.connection_generation
            != self.terminal.connection_generation
        ):
            raise ValueError(
                "camera connection generation differs from terminal evidence"
            )
        if (
            self.camera_provenance.capability_fingerprint
            != self.terminal.capability_fingerprint
        ):
            raise ValueError(
                "camera capability fingerprint differs from terminal evidence"
            )
        _canonical_text(self.run_id, "run_id")
        _canonical_text(self.safety_bundle_id, "safety_bundle_id")
        _sha256(self.chain_contract_digest, "chain_contract_digest")
        if self.provenance.trace_binding.run_id != self.run_id:
            raise ValueError("capture run_id differs from sealed dataset provenance")
        if (
            self.camera_provenance.binding.value
            != self.provenance.trace_binding.source_id
        ):
            raise ValueError(
                "camera binding differs from sealed dataset source lineage"
            )
        if capability_evidence.source_id != self.provenance.trace_binding.source_id:
            raise ValueError(
                "camera capability evidence differs from sealed dataset source"
            )
        if self.pulse_lineage is not None and not isinstance(
            self.pulse_lineage,
            PulseCaptureLineage,
        ):
            raise TypeError("pulse_lineage must be PulseCaptureLineage or None")
        count = len(metadata)
        physical_cells = (
            schema.repeat_axis.size
            * schema.point_layout.storage_size
        )
        if self.coverage.total_cells != physical_cells or count != physical_cells:
            raise ValueError("capture metadata cardinality differs from DataBlock cells")
        if decoded_arm_spec.expected_frames != count:
            raise ValueError(
                "camera arm expected_frames differs from persisted capture count"
            )
        if len(
            {
                decoded_arm_spec.settings_fingerprint,
                capability_evidence.settings_fingerprint,
                self.terminal.settings_fingerprint,
                self.camera_provenance.active_settings_fingerprint,
            }
        ) != 1:
            raise ValueError(
                "camera arm settings differ from capability and terminal evidence"
            )
        if self.provenance.end_sequence - self.provenance.start_sequence != count:
            raise ValueError("capture provenance interval differs from metadata cardinality")
        if (
            dataset_cell_permutation_digest(schema, schedule)
            != self.provenance.join_plan_digest
        ):
            raise ValueError("source cell schedule differs from sealed join plan")
        expected_direct_chain = dataset_consumer_contract_digest(
            schema,
            schedule,
            self.provenance.metadata_contract_fingerprint,
            CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
        )
        if self.chain_contract_digest != expected_direct_chain:
            raise ValueError(
                "CaptureArtifact requires a direct source-to-DatasetBuilder chain"
            )
        if tuple(item.source_ordinal for item in metadata) != tuple(range(count)):
            raise ValueError("capture source ordinals are not contiguous from zero")
        metadata_contract = CameraFrameMetadataContract()
        if self.provenance.metadata_contract_fingerprint != metadata_contract.fingerprint:
            raise ValueError("capture metadata contract is not the current camera contract")
        hasher = hashlib.sha256()
        hasher.update(metadata_contract.fingerprint.encode("ascii"))
        for item in metadata:
            metadata_contract.validate(item)
            metadata_digest = metadata_contract.digest(item)
            hasher.update(metadata_digest.encode("ascii"))
        if hasher.hexdigest() != self.provenance.ordered_metadata_digest:
            raise ValueError("capture metadata sequence digest differs from provenance")
        if (
            self.terminal.produced_count != count
            or self.terminal.drained_count != count
            or self.terminal.ordered_metadata_digest
            != self.provenance.ordered_metadata_digest
            or not self.terminal.source_stopped
            or not self.terminal.no_more_frames
            or not self.terminal.joined
        ):
            raise ValueError("capture terminal evidence differs from persisted dataset")
        if (
            self.pulse_lineage is not None
            and self.pulse_lineage.expected_trigger_count != count
        ):
            raise ValueError("pulse trigger count differs from persisted capture")
        if self.pulse_lineage is not None:
            capability_evidence.physical_facts.require_single_capture_trigger_channel(
                self.pulse_lineage.trigger_channel
            )
            plan = self.pulse_lineage.cell_plan
            plan.validate_dataset_schema(schema)
            if plan.cell_permutation_digest != self.provenance.join_plan_digest:
                raise ValueError("capture cell plan differs from sealed cell permutation")
            if tuple(plan.expected_cells) != schedule:
                raise ValueError(
                    "triggered capture cell plan differs from source cell schedule"
                )


def _capture_decode_limits(
    policy: CaptureRepositoryResourcePolicy,
    *,
    max_arrays: int,
    max_total_array_bytes: int,
) -> CanonicalDecodeLimits:
    return CanonicalDecodeLimits(
        max_depth=128,
        max_nodes=policy.max_canonical_nodes,
        max_container_entries=policy.max_canonical_container_entries,
        max_arrays=max_arrays,
        max_total_array_bytes=max_total_array_bytes,
    )


def _require_content_size(
    reference: ContentRef,
    maximum: int,
    field: str,
) -> None:
    if reference.size > maximum:
        raise CaptureResourceExceeded(
            f"capture {field} blob exceeds repository resource policy"
        )


class CaptureRepository:
    """Current-only raw-capture CAS with durable commit/admission authority."""

    __slots__ = (
        "root",
        "repository_id",
        "resource_policy",
        "_root_lease",
        "_store",
        "_store_authority",
        "_journal",
        "_coordinator",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CaptureRepository is final and cannot be subclassed")

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-capture",
        resource_policy: CaptureRepositoryResourcePolicy = (
            DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY
        ),
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        object.__setattr__(
            self,
            "repository_id",
            _canonical_text(repository_id, "repository_id"),
        )
        if not isinstance(resource_policy, CaptureRepositoryResourcePolicy):
            raise TypeError(
                "resource_policy must be CaptureRepositoryResourcePolicy"
            )
        owned_policy = replace(resource_policy)
        object.__setattr__(self, "resource_policy", owned_policy)
        root_lease = RepositoryRootLease(
            self.root,
            owner=f"capture:{self.repository_id}",
        )
        object.__setattr__(self, "_root_lease", root_lease)
        try:
            # The root lease is deliberately acquired before any content-store,
            # journal, or startup-reconciliation I/O.  A losing second writer
            # therefore cannot inspect or resolve another live owner's intents.
            store = ContentAddressedStore(self.root / "content")
            object.__setattr__(self, "_store", store)
            object.__setattr__(self, "_store_authority", store.authority())
            journal = PersistentCommitJournal(
                self.root / "capture-commit.journal",
                self.repository_id,
            )
            object.__setattr__(self, "_journal", journal)
            # RepositoryCommitCoordinator performs synchronous startup recovery.
            coordinator: RepositoryCommitCoordinator[CaptureArtifactRef] = (
                RepositoryCommitCoordinator(
                    journal,
                    self._recover,
                    root_lease=root_lease,
                )
            )
            object.__setattr__(self, "_coordinator", coordinator)
        except BaseException:
            root_lease.close()
            raise
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CaptureRepository authority is immutable")
        object.__setattr__(self, _name, _value)

    def _require_active(self) -> None:
        self._root_lease.require_active()

    def close(self) -> None:
        """Close only after every prepared/in-flight commit authority is resolved."""

        root_lease = getattr(self, "_root_lease", None)
        if root_lease is not None:
            root_lease.close()

    def __enter__(self) -> "CaptureRepository":
        self._require_active()
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
        self._require_active()
        return self._coordinator.startup_reconciliations

    def load(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        """Fully validate a structurally visible artifact for inspection only."""

        self._require_active()
        self._validate_ref(reference)
        try:
            manifest_payload = self._store_authority.read_manifest(
                CAPTURE_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=self.resource_policy.max_manifest_bytes,
            )
        except ContentSizeLimitError as exc:
            raise CaptureResourceExceeded(
                "capture manifest exceeds repository resource policy"
            ) from exc
        return self._load_manifest(
            reference,
            manifest_payload,
            store_authority=self._store_authority,
            policy=self.resource_policy,
            repository_id=self.repository_id,
        )

    def _load_manifest(
        self,
        reference: CaptureArtifactRef,
        manifest_payload: bytes,
        *,
        store_authority: ContentStoreAuthority,
        policy: CaptureRepositoryResourcePolicy,
        repository_id: str,
    ) -> CaptureArtifact:
        self._require_active()
        if len(manifest_payload) > policy.max_manifest_bytes:
            raise CaptureResourceExceeded(
                "capture manifest exceeds repository resource policy"
            )
        def admit_manifest_structure(events) -> None:
            for event in events:
                if isinstance(event, CanonicalArrayEvent):
                    raise CaptureResourceExceeded(
                        "capture manifest cannot embed ndarray payloads"
                    )

        tree = decode(
            manifest_payload,
            admit_structure=admit_manifest_structure,
            limits=_capture_decode_limits(
                policy,
                max_arrays=0,
                max_total_array_bytes=0,
            ),
        )
        data = _exact_map(
            tree,
            {
                "schema",
                "repository_id",
                "frame_index_blob",
                "compiled_pulse_blob",
                "cell_plan_blob",
                "coverage",
                "provenance",
                "terminal",
                "aggregate_peak_bytes",
                "camera_provenance",
                "camera_capability_evidence",
                "camera_arm_spec",
                "run_id",
                "safety_bundle_id",
                "chain_contract_digest",
                "pulse_lineage",
            },
            CAPTURE_ARTIFACT_SCHEMA,
        )
        if data["repository_id"] != repository_id:
            raise ValueError("CaptureArtifact belongs to another repository")
        frame_index_ref = content_ref_from_tree(data["frame_index_blob"])
        pulse_ref = (
            None
            if data["compiled_pulse_blob"] is None
            else content_ref_from_tree(data["compiled_pulse_blob"])
        )
        plan_ref = (
            None
            if data["cell_plan_blob"] is None
            else content_ref_from_tree(data["cell_plan_blob"])
        )
        _require_content_size(
            frame_index_ref,
            policy.max_frame_index_blob_bytes,
            "frame-index",
        )
        if pulse_ref is not None:
            _require_content_size(
                pulse_ref,
                policy.max_compiled_pulse_blob_bytes,
                "compiled-pulse",
            )
        if plan_ref is not None:
            _require_content_size(
                plan_ref,
                policy.max_cell_plan_blob_bytes,
                "cell-plan",
            )
        try:
            frame_source = _load_capture_frame_source(
                frame_index_ref,
                store_authority=store_authority,
                root_lease=self._root_lease,
                max_cells=policy.max_cells,
                max_total_frame_bytes=policy.max_total_frame_bytes,
                max_chunk_blob_bytes=policy.max_frame_chunk_blob_bytes,
                max_frame_index_blob_bytes=policy.max_frame_index_blob_bytes,
                max_canonical_nodes=policy.max_canonical_nodes,
                max_canonical_container_entries=(
                    policy.max_canonical_container_entries
                ),
            )
        except (_FrameResourceExceeded, ContentSizeLimitError) as exc:
            raise CaptureResourceExceeded(
                "capture frame storage exceeds repository resource policy"
            ) from exc
        try:
            cell_plan = (
                None
                if plan_ref is None
                else decode_compiled_capture_cell_plan(
                    store_authority.read_blob(
                        plan_ref,
                        max_bytes=policy.max_cell_plan_blob_bytes,
                    )
                )
            )
            compiled_pulse = (
                None
                if pulse_ref is None
                else decode_compiled_pulse_artifact(
                    store_authority.read_blob(
                        pulse_ref,
                        max_bytes=policy.max_compiled_pulse_blob_bytes,
                    )
                )
            )
        except ContentSizeLimitError as exc:
            raise CaptureResourceExceeded(
                "capture lineage blob exceeds repository resource policy"
            ) from exc
        if compiled_pulse is not None and pulse_ref is not None:
            if pulse_ref.digest != compiled_pulse.fingerprint:
                raise ValueError(
                    "compiled-pulse blob digest differs from artifact fingerprint"
                )
        if cell_plan is not None and plan_ref is not None:
            if plan_ref.digest != cell_plan.fingerprint:
                raise ValueError("cell-plan blob digest differs from plan fingerprint")
        artifact = CaptureArtifact(
            ref=reference,
            frame_source=frame_source,
            coverage=_coverage_from_tree(data["coverage"]),
            provenance=_provenance_from_tree(data["provenance"]),
            terminal=_terminal_from_tree(data["terminal"]),
            aggregate_peak_bytes=data["aggregate_peak_bytes"],
            camera_provenance=_camera_provenance_from_tree(
                data["camera_provenance"]
            ),
            camera_capability_evidence=camera_capability_evidence_from_tree(
                data["camera_capability_evidence"]
            ),
            camera_arm_spec=_frozen_capture_spec_from_tree(data["camera_arm_spec"]),
            run_id=data["run_id"],
            safety_bundle_id=data["safety_bundle_id"],
            chain_contract_digest=data["chain_contract_digest"],
            pulse_lineage=_pulse_lineage_from_tree(
                data["pulse_lineage"],
                compiled_pulse,
                cell_plan,
            ),
        )
        # Enforce one canonical current representation, not merely a decodable one.
        rebuilt_payload = _manifest_payload(
            artifact,
            frame_index_ref,
            pulse_ref,
            plan_ref,
        )
        if (
            sha256_digest(rebuilt_payload) != reference.manifest_digest
            or rebuilt_payload != manifest_payload
        ):
            raise ValueError("CaptureArtifact manifest is not canonical")
        return artifact

    def admit(self, reference: CaptureArtifactRef) -> AdmittedCapture:
        """Mint authority only for an exact journal-committed capture target."""

        self._require_active()
        artifact = self.load(reference)
        target = CommitTarget(
            self.repository_id,
            _CAPTURE_ARTIFACT_KIND,
            CAPTURE_ARTIFACT_SCHEMA,
            reference.target_ref,
            reference.manifest_digest,
        )
        matching = tuple(
            intent
            for intent in self._coordinator.committed_intents()
            if intent.target == target
        )
        if not matching:
            raise PermissionError(
                "CaptureArtifact is visible but has no committed journal authority"
            )
        for intent in matching:
            expected_kind = (
                "final" if intent.kind is CommitKind.FINAL else "checkpoint"
            )
            expected_commit_id = (
                f"capture-{expected_kind}-{artifact.run_id}-"
                f"{reference.manifest_digest}"
            )
            if (
                intent.run_id != artifact.run_id
                or intent.safety_bundle_id != artifact.safety_bundle_id
                or intent.commit_id != expected_commit_id
            ):
                raise ValueError(
                    "committed capture intent differs from persisted artifact evidence"
                )
        selected = min(
            matching,
            key=lambda intent: (
                0 if intent.kind is CommitKind.FINAL else 1,
                intent.commit_id,
            ),
        )
        evidence_digest = sha256_digest(
            encode(
                {
                    "schema": _ADMITTED_CAPTURE_EVIDENCE_SCHEMA,
                    "repository_id": self.repository_id,
                    "reference": capture_artifact_ref_to_tree(reference),
                    "commit": {
                        "kind": selected.kind.value,
                        "commit_id": selected.commit_id,
                        "run_id": selected.run_id,
                        "safety_bundle_id": selected.safety_bundle_id,
                        "created_at": selected.created_at,
                        "target": {
                            "repository_id": selected.target.repository_id,
                            "artifact_kind": selected.target.artifact_kind,
                            "artifact_format": selected.target.artifact_format,
                            "target_ref": selected.target.target_ref,
                            "expected_manifest_digest": (
                                selected.target.expected_manifest_digest
                            ),
                        },
                    },
                }
            )
        )
        return AdmittedCapture(
            _ADMITTED_CAPTURE_TOKEN,
            repository_token=self._root_lease,
            reference=reference,
            artifact=artifact,
            commit_kind=selected.kind,
            commit_id=selected.commit_id,
            evidence_digest=evidence_digest,
        )

    def final_commit(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
    ) -> FinalCommit[CaptureArtifactRef]:
        operation = self._commit_operation(context, result, checkpoint=False)
        assert isinstance(operation, FinalCommit)
        return operation

    def checkpoint_commit(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
    ) -> CheckpointCommit[CaptureArtifactRef]:
        """Prepare one typed raw-capture checkpoint without committing the Run.

        Checkpoints and final commits intentionally share the exact staging,
        target, publication, and recovery path.  This factory only chooses the
        runtime operation type and a collision-free intent id; the caller still
        has to submit the returned operation through its PostSafetyContext.
        """

        operation = self._commit_operation(context, result, checkpoint=True)
        assert isinstance(operation, CheckpointCommit)
        return operation

    def _commit_operation(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
        *,
        checkpoint: bool,
    ) -> FinalCommit[CaptureArtifactRef] | CheckpointCommit[CaptureArtifactRef]:
        self._require_active()
        if not isinstance(context, PostSafetyContext):
            raise TypeError("capture commit requires PostSafetyContext")
        if not isinstance(result, (PipelineResult, TriggeredPipelineResult)):
            raise TypeError("capture commit requires an exact pipeline result")
        if not isinstance(checkpoint, bool):
            raise TypeError("checkpoint selector must be bool")
        kind = CommitKind.CHECKPOINT if checkpoint else CommitKind.FINAL
        subject = context.authorize_commit_preparation(kind)
        reference, manifest_payload = self._stage_pipeline_result(result, context)
        confirmed_subject = context.authorize_commit_preparation(kind)
        if confirmed_subject != subject:
            raise RuntimeError("capture commit subject changed while staging")
        target = CommitTarget(
            self.repository_id,
            _CAPTURE_ARTIFACT_KIND,
            CAPTURE_ARTIFACT_SCHEMA,
            reference.target_ref,
            reference.manifest_digest,
        )

        def publish() -> PublishedManifest[CaptureArtifactRef]:
            self._require_active()
            stored = publish_manifest_with_visibility_reconciliation(
                self._store_authority,
                CAPTURE_ARTIFACT_NAMESPACE,
                manifest_payload,
                expected_digest=reference.manifest_digest,
                max_bytes=self.resource_policy.max_manifest_bytes,
            )
            if stored.content.digest != reference.manifest_digest:
                raise RuntimeError("published capture manifest digest changed")
            return PublishedManifest(
                reference.target_ref,
                reference.manifest_digest,
                reference,
            )

        operation_type = CheckpointCommit if checkpoint else FinalCommit
        operation_kind = "checkpoint" if checkpoint else "final"
        commit_id = (
            f"capture-{operation_kind}-{subject.run_id}-"
            f"{reference.manifest_digest}"
        )
        operation = operation_type(
            self._coordinator.prepare(
                kind,
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

    def _recover(self, intent: CommitIntent) -> CommitRecovery[CaptureArtifactRef]:
        self._require_active()
        store_authority = self._store_authority
        policy = self.resource_policy
        repository_id = self.repository_id
        target = intent.target
        if (
            target.repository_id != repository_id
            or target.artifact_kind != _CAPTURE_ARTIFACT_KIND
            or target.artifact_format != CAPTURE_ARTIFACT_SCHEMA
        ):
            raise ValueError("commit intent is not a CaptureArtifact target")
        reference = CaptureArtifactRef(
            repository_id,
            target.expected_manifest_digest,
        )
        if target.target_ref != reference.target_ref:
            raise ValueError("capture commit target ref and digest differ")
        digest = reference.manifest_digest
        operation_kind = (
            "final" if intent.kind is CommitKind.FINAL else "checkpoint"
        )
        expected_commit_id = (
            f"capture-{operation_kind}-{intent.run_id}-{digest}"
        )
        if intent.commit_id != expected_commit_id:
            raise ValueError("capture commit id differs from kind/run/target")
        try:
            manifest_payload = store_authority.read_manifest(
                CAPTURE_ARTIFACT_NAMESPACE,
                digest,
                max_bytes=policy.max_manifest_bytes,
            )
        except FileNotFoundError:
            return CommitRecovery(False)
        except ContentSizeLimitError as exc:
            raise CaptureResourceExceeded(
                "capture manifest exceeds repository resource policy"
            ) from exc
        # The manifest was observed first.  Any missing/corrupt referenced blob
        # is now a visible corrupt artifact and must fail startup closed rather
        # than be misreported as an absent/uncommitted manifest.
        artifact = self._load_manifest(
            reference,
            manifest_payload,
            store_authority=store_authority,
            policy=policy,
            repository_id=repository_id,
        )
        artifact.frame_source._verify_all_chunks()
        if artifact.run_id != intent.run_id:
            raise ValueError("capture manifest run_id differs from commit intent")
        if artifact.safety_bundle_id != intent.safety_bundle_id:
            raise ValueError(
                "capture manifest safety bundle differs from commit intent"
            )
        # A readable target may be the visible residue of a replace whose
        # parent-directory flush acknowledgement failed.  This storage-owned
        # barrier verifies/fsyncs only an existing immutable target and never
        # creates one.  Recovery cannot resolve COMMITTED until it succeeds.
        confirmed_payload = store_authority.confirm_manifest_durable(
            CAPTURE_ARTIFACT_NAMESPACE,
            digest,
            max_bytes=policy.max_manifest_bytes,
        )
        if confirmed_payload != manifest_payload:
            raise RuntimeError(
                "capture recovery durability confirmation changed payload"
            )
        return CommitRecovery(
            True,
            PublishedManifest(reference.target_ref, digest, reference),
        )

    def _stage_pipeline_result(
        self,
        result: PipelineResult | TriggeredPipelineResult,
        context: PostSafetyContext,
    ) -> tuple[CaptureArtifactRef, bytes]:
        self._require_active()
        lineage = None
        if isinstance(result, TriggeredPipelineResult):
            lineage = result.lineage
            base = result.capture
        else:
            base = result
        if base.run_id != context.run_id.value:
            raise ValueError("PostSafetyContext run_id differs from pipeline result")
        if context.safety_bundle_id is None:
            raise ValueError("CaptureArtifact requires a durable safety bundle id")
        if not base.is_direct_raw_capture:
            raise ValueError(
                "CaptureArtifact only accepts direct raw camera datasets"
            )
        if base.camera_provenance is None:
            raise ValueError(
                "CaptureArtifact requires frozen CameraCaptureDescriptor provenance"
            )
        if base.camera_capability_evidence is None:
            raise ValueError(
                "CaptureArtifact requires broker camera capability evidence"
            )
        policy = self.resource_policy
        try:
            frame_source, frame_index_ref = _stage_capture_frame_source(
                block=base.dataset.block,
                event_metadata=tuple(base.dataset.event_metadata),
                cell_schedule=base.source_cell_schedule,
                store_authority=self._store_authority,
                root_lease=self._root_lease,
                max_cells=policy.max_cells,
                max_total_frame_bytes=policy.max_total_frame_bytes,
                max_chunk_blob_bytes=policy.max_frame_chunk_blob_bytes,
                max_frame_index_blob_bytes=policy.max_frame_index_blob_bytes,
                max_canonical_nodes=policy.max_canonical_nodes,
                max_canonical_container_entries=(
                    policy.max_canonical_container_entries
                ),
            )
        except _FrameResourceExceeded as exc:
            raise CaptureResourceExceeded(
                "capture frame storage exceeds repository resource policy"
            ) from exc
        provisional = CaptureArtifact(
            ref=CaptureArtifactRef(self.repository_id, "0" * 64),
            frame_source=frame_source,
            coverage=base.dataset.coverage,
            provenance=base.dataset.provenance,
            terminal=base.capture_terminal,
            aggregate_peak_bytes=base.aggregate_peak_bytes,
            camera_provenance=base.camera_provenance,
            camera_capability_evidence=base.camera_capability_evidence,
            camera_arm_spec=base.camera_arm_spec,
            run_id=base.run_id,
            safety_bundle_id=context.safety_bundle_id,
            chain_contract_digest=base.chain_contract_digest,
            pulse_lineage=lineage,
        )
        return self._stage_manifest(provisional, frame_index_ref)

    def _stage_manifest(
        self,
        artifact: CaptureArtifact,
        frame_index_ref: ContentRef,
    ) -> tuple[CaptureArtifactRef, bytes]:
        self._require_active()
        policy = self.resource_policy
        _require_content_size(
            frame_index_ref,
            policy.max_frame_index_blob_bytes,
            "frame-index",
        )
        pulse_payload = (
            None
            if artifact.pulse_lineage is None
            else encode_compiled_pulse_artifact(
                artifact.pulse_lineage.compiled_artifact
            )
        )
        if (
            pulse_payload is not None
            and len(pulse_payload) > policy.max_compiled_pulse_blob_bytes
        ):
            raise CaptureResourceExceeded(
                "capture compiled-pulse blob exceeds repository resource policy"
            )
        pulse_ref = (
            None
            if pulse_payload is None
            else self._store_authority.put_blob(pulse_payload)
        )
        plan_payload = (
            None
            if artifact.pulse_lineage is None
            else encode_compiled_capture_cell_plan(artifact.pulse_lineage.cell_plan)
        )
        if (
            plan_payload is not None
            and len(plan_payload) > policy.max_cell_plan_blob_bytes
        ):
            raise CaptureResourceExceeded(
                "capture cell-plan blob exceeds repository resource policy"
            )
        plan_ref = (
            None
            if plan_payload is None
            else self._store_authority.put_blob(plan_payload)
        )
        manifest_payload = _manifest_payload(
            artifact,
            frame_index_ref,
            pulse_ref,
            plan_ref,
        )
        if len(manifest_payload) > policy.max_manifest_bytes:
            raise CaptureResourceExceeded(
                "capture manifest exceeds repository resource policy"
            )
        reference = CaptureArtifactRef(
            self.repository_id,
            sha256_digest(manifest_payload),
        )
        return reference, manifest_payload

    def _validate_ref(self, reference: CaptureArtifactRef) -> None:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("load requires CaptureArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("CaptureArtifactRef belongs to another repository")


def _capture_artifact_staging_peak_bytes(
    spec: MinimalPipelineSpec,
    policy: CaptureRepositoryResourcePolicy,
) -> int:
    """Bound post-safety staging inside the already admitted pipeline peak."""

    contract = spec.measurement.capture_contract
    events = contract.total_events
    memory = spec.materializer.memory
    return (
        dataset_storage_nbytes(contract.dataset_schema)
        + _capture_frame_source_scratch_upper_bound(
            contract.dataset_schema,
            events,
            policy.max_frame_chunk_blob_bytes,
        )
        + events * contract.dataset_edge.metadata_max_retained_nbytes
        + memory.fixed_runtime_overhead_bytes
        + events * memory.per_event_reference_overhead_bytes
    )


def compile_capture_artifact_pipeline(
    spec: MinimalPipelineSpec | TriggeredCaptureSpec,
    repository: CaptureRepository,
) -> RunPlan:
    """Add one post-safety CaptureArtifact commit to the exact pipeline."""

    if type(repository) is not CaptureRepository:
        raise TypeError("repository must be CaptureRepository")
    capture_spec = spec.capture if isinstance(spec, TriggeredCaptureSpec) else spec
    if not isinstance(capture_spec, MinimalPipelineSpec):
        raise TypeError("capture artifact pipeline requires MinimalPipelineSpec")
    if capture_spec.measurement.capture_contract.camera_provenance is None:
        raise ValueError(
            "CaptureArtifact requires frozen raw camera provenance"
        )
    repository._require_active()
    policy = repository.resource_policy
    pipeline_peak = estimate_pipeline_peak_bytes(capture_spec)
    staging_peak = _capture_artifact_staging_peak_bytes(
        capture_spec,
        policy,
    )
    if staging_peak > pipeline_peak:
        raise RuntimeError(
            "capture artifact staging escaped the pipeline memory admission"
        )
    base = (
        compile_triggered_pipeline(spec)
        if isinstance(spec, TriggeredCaptureSpec)
        else compile_pipeline(spec)
    )

    def preflight(
        context: RunContext,
    ) -> tuple[object, RepositoryRootLeaseBorrow]:
        # The durable sink is part of this hardware plan's admission, not a
        # post-capture convenience.  Taking the borrow before base preflight
        # makes close-vs-run linearizable: close wins and no device is touched,
        # or this run wins and close cannot invalidate its sink mid-capture.
        repository._require_active()
        borrow = repository._root_lease.borrow()
        try:
            return base.preflight(context), borrow
        except BaseException:
            borrow.close()
            raise

    def execute(
        context: RunContext,
        prepared: tuple[object, RepositoryRootLeaseBorrow],
    ) -> tuple[PipelineResult | TriggeredPipelineResult, RepositoryRootLeaseBorrow]:
        base_prepared, borrow = prepared
        borrow.require_active()
        return base.execute(context, base_prepared), borrow

    def cleanup(
        context: RunContext,
        prepared: tuple[object, RepositoryRootLeaseBorrow] | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        base_prepared = None if prepared is None else prepared[0]
        borrow = None if prepared is None else prepared[1]
        try:
            report = base.cleanup(context, base_prepared, primary)
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
        result: tuple[
            PipelineResult | TriggeredPipelineResult,
            RepositoryRootLeaseBorrow,
        ],
    ) -> CaptureArtifactRef:
        base_result, borrow = result
        try:
            borrow.require_active()
            finalized = base.finalize(context, base_result)
            if not isinstance(finalized, (PipelineResult, TriggeredPipelineResult)):
                raise TypeError("base exact pipeline changed its result contract")
            return context.commit_final(repository.final_commit(context, finalized))
        finally:
            # Visibility reconciliation owns its own commit authority; this
            # broader run admission ends when finalize has handed off or failed.
            borrow.close()

    return RunPlan(
        name=base.name,
        mode=base.mode,
        resource_claims=base.resource_claims,
        hazard_claims=base.hazard_claims,
        bound_devices=base.bound_devices,
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=base.interrupt_operations,
        timeout_seconds=base.timeout_seconds,
        requires_final_commit=True,
    )


def _manifest_payload(
    artifact: CaptureArtifact,
    frame_index_ref: ContentRef,
    compiled_pulse_ref: ContentRef | None,
    cell_plan_ref: ContentRef | None,
) -> bytes:
    absent = artifact.pulse_lineage is None
    if absent != (compiled_pulse_ref is None) or absent != (cell_plan_ref is None):
        raise ValueError("pulse lineage and compiled-plan blob presence differ")
    return encode(
        {
            "schema": CAPTURE_ARTIFACT_SCHEMA,
            "repository_id": artifact.ref.repository_id,
            "frame_index_blob": content_ref_to_tree(frame_index_ref),
            "compiled_pulse_blob": (
                None
                if compiled_pulse_ref is None
                else content_ref_to_tree(compiled_pulse_ref)
            ),
            "cell_plan_blob": (
                None
                if cell_plan_ref is None
                else content_ref_to_tree(cell_plan_ref)
            ),
            "coverage": _coverage_to_tree(artifact.coverage),
            "provenance": _provenance_to_tree(artifact.provenance),
            "terminal": _terminal_to_tree(artifact.terminal),
            "aggregate_peak_bytes": artifact.aggregate_peak_bytes,
            "camera_provenance": _camera_provenance_to_tree(
                artifact.camera_provenance
            ),
            "camera_capability_evidence": camera_capability_evidence_to_tree(
                artifact.camera_capability_evidence
            ),
            "camera_arm_spec": _frozen_capture_spec_to_tree(
                artifact.camera_arm_spec
            ),
            "run_id": artifact.run_id,
            "safety_bundle_id": artifact.safety_bundle_id,
            "chain_contract_digest": artifact.chain_contract_digest,
            "pulse_lineage": _pulse_lineage_to_tree(artifact.pulse_lineage),
        }
    )


def _camera_provenance_to_tree(
    value: CameraCaptureProvenance,
) -> dict[str, object]:
    if not isinstance(value, CameraCaptureProvenance):
        raise TypeError("value must be CameraCaptureProvenance")
    return {
        "descriptor": camera_capture_descriptor_to_tree(value.descriptor),
        "camera_arm_spec_fingerprint": value.camera_arm_spec_fingerprint,
        "binding": readout_binding_key_to_tree(value.binding),
        "active_settings_fingerprint": value.active_settings_fingerprint,
        "binding_id": value.binding_id,
        "connection_generation": value.connection_generation,
        "capability_fingerprint": value.capability_fingerprint,
    }


def _frozen_capture_spec_to_tree(value: FrozenCaptureSpec) -> dict[str, object]:
    if not isinstance(value, FrozenCaptureSpec):
        raise TypeError("value must be FrozenCaptureSpec")
    return {
        "owner_fingerprint": value.owner_fingerprint,
        "payload": value.payload,
        "digest": value.digest,
    }


def _frozen_capture_spec_from_tree(tree: object) -> FrozenCaptureSpec:
    fields = {"owner_fingerprint", "payload", "digest"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("camera arm spec has an unknown field set")
    return FrozenCaptureSpec(
        owner_fingerprint=_sha256(
            tree["owner_fingerprint"],
            "camera arm spec owner_fingerprint",
        ),
        payload=tree["payload"],
        digest=_sha256(tree["digest"], "camera arm spec digest"),
    )


def _camera_provenance_from_tree(tree: object) -> CameraCaptureProvenance:
    fields = {
        "descriptor",
        "camera_arm_spec_fingerprint",
        "binding",
        "active_settings_fingerprint",
        "binding_id",
        "connection_generation",
        "capability_fingerprint",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("camera capture provenance has an unknown field set")
    descriptor = camera_capture_descriptor_from_tree(tree["descriptor"])
    arm_spec = _sha256(
        tree["camera_arm_spec_fingerprint"],
        "camera_arm_spec_fingerprint",
    )
    if descriptor.camera_arm_spec_fingerprint != arm_spec:
        raise ValueError(
            "camera arm-spec fingerprint differs from descriptor owner field"
        )
    return CameraCaptureProvenance(
        descriptor=descriptor,
        binding=readout_binding_key_from_tree(tree["binding"]),
        active_settings_fingerprint=_sha256(
            tree["active_settings_fingerprint"],
            "active_settings_fingerprint",
        ),
        binding_id=_canonical_text(tree["binding_id"], "binding_id"),
        connection_generation=_canonical_text(
            tree["connection_generation"],
            "connection_generation",
        ),
        capability_fingerprint=_sha256(
            tree["capability_fingerprint"],
            "capability_fingerprint",
        ),
    )


def _pulse_lineage_to_tree(
    value: PulseCaptureLineage | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "compiled_artifact_fingerprint": value.compiled_artifact.fingerprint,
        "trigger_channel": value.trigger_channel,
        "terminal": pulse_terminal_ack_to_tree(value.terminal),
        "cell_plan_fingerprint": value.cell_plan.fingerprint,
    }


def _pulse_lineage_from_tree(
    tree: object,
    compiled_pulse: CompiledPulseArtifact | None,
    cell_plan: CompiledCaptureCellPlan | None,
) -> PulseCaptureLineage | None:
    if tree is None:
        if compiled_pulse is not None or cell_plan is not None:
            raise ValueError("compiled-plan blob exists without pulse lineage")
        return None
    fields = {
        "compiled_artifact_fingerprint",
        "trigger_channel",
        "terminal",
        "cell_plan_fingerprint",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseCaptureLineage has an unknown field set")
    if compiled_pulse is None or cell_plan is None:
        raise ValueError("pulse lineage omits a compiled-plan blob")
    if tree["compiled_artifact_fingerprint"] != compiled_pulse.fingerprint:
        raise ValueError("pulse lineage fingerprint differs from compiled artifact blob")
    if tree["cell_plan_fingerprint"] != cell_plan.fingerprint:
        raise ValueError("pulse lineage cell-plan fingerprint differs from blob")
    binding = PulseCaptureBinding(
        compiled_pulse,
        tree["trigger_channel"],
        cell_plan,
    )
    return PulseCaptureLineage(
        binding,
        pulse_terminal_ack_from_tree(tree["terminal"]),
    )


def _coverage_to_tree(coverage: DatasetCoverage) -> dict[str, object]:
    return {
        "written_cells": coverage.written_cells,
        "total_cells": coverage.total_cells,
        "missed_events": coverage.missed_events,
    }


def _coverage_from_tree(tree: object) -> DatasetCoverage:
    if not isinstance(tree, dict) or set(tree) != {
        "written_cells",
        "total_cells",
        "missed_events",
    }:
        raise ValueError("dataset coverage has an unknown field set")
    return DatasetCoverage(**tree)


def _provenance_to_tree(provenance: DatasetSealProvenance) -> dict[str, object]:
    return {
        "stream_id": provenance.stream_id.value,
        "generation": provenance.generation.value,
        "start_sequence": provenance.start_sequence,
        "end_sequence": provenance.end_sequence,
        "join_plan_digest": provenance.join_plan_digest,
        "ordered_event_digest": provenance.ordered_event_digest,
        "ordered_metadata_digest": provenance.ordered_metadata_digest,
        "metadata_contract_fingerprint": provenance.metadata_contract_fingerprint,
        "trace_binding": {
            "run_id": provenance.trace_binding.run_id,
            "source_id": provenance.trace_binding.source_id,
        },
        "derivation": None,
    }


def _provenance_from_tree(tree: object) -> DatasetSealProvenance:
    fields = {
        "stream_id",
        "generation",
        "start_sequence",
        "end_sequence",
        "join_plan_digest",
        "ordered_event_digest",
        "ordered_metadata_digest",
        "metadata_contract_fingerprint",
        "trace_binding",
        "derivation",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("dataset provenance has an unknown field set")
    trace = tree["trace_binding"]
    if not isinstance(trace, dict) or set(trace) != {"run_id", "source_id"}:
        raise ValueError("trace binding has an unknown field set")
    if tree["derivation"] is not None:
        raise ValueError(
            "raw CaptureArtifact provenance cannot contain processor derivation"
        )
    return DatasetSealProvenance(
        StreamId(tree["stream_id"]),
        StreamGenerationId(tree["generation"]),
        _integer(tree["start_sequence"], "start_sequence"),
        _integer(tree["end_sequence"], "end_sequence"),
        _sha256(tree["join_plan_digest"], "join_plan_digest"),
        _sha256(tree["ordered_event_digest"], "ordered_event_digest"),
        _sha256(tree["ordered_metadata_digest"], "ordered_metadata_digest"),
        _sha256(
            tree["metadata_contract_fingerprint"],
            "metadata_contract_fingerprint",
        ),
        TraceBinding(trace["run_id"], trace["source_id"]),
    )


def _terminal_to_tree(terminal: CaptureTerminalAck) -> dict[str, object]:
    return {
        "session_id": terminal.session_id,
        "binding_id": terminal.binding_id,
        "connection_generation": terminal.connection_generation,
        "produced_count": terminal.produced_count,
        "drained_count": terminal.drained_count,
        "source_stopped": terminal.source_stopped,
        "no_more_frames": terminal.no_more_frames,
        "joined": terminal.joined,
        "ordered_metadata_digest": terminal.ordered_metadata_digest,
        "settings_fingerprint": terminal.settings_fingerprint,
        "capability_fingerprint": terminal.capability_fingerprint,
        "capture_spec_fingerprint": terminal.capture_spec_fingerprint,
    }


def _terminal_from_tree(tree: object) -> CaptureTerminalAck:
    fields = {
        "session_id",
        "binding_id",
        "connection_generation",
        "produced_count",
        "drained_count",
        "source_stopped",
        "no_more_frames",
        "joined",
        "ordered_metadata_digest",
        "settings_fingerprint",
        "capability_fingerprint",
        "capture_spec_fingerprint",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("capture terminal has an unknown field set")
    return CaptureTerminalAck(**tree)


__all__ = [
    "AdmittedCapture",
    "CAPTURE_ARTIFACT_SCHEMA",
    "CaptureArtifact",
    "CaptureArtifactRef",
    "CaptureFrameSource",
    "CaptureRepositoryResourcePolicy",
    "CaptureRepository",
    "CaptureResourceExceeded",
    "DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY",
    "compile_capture_artifact_pipeline",
]
