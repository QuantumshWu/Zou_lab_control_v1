"""Current lazy raw-frame CaptureArtifact and crash-safe repository."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

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
    positive_integer,
    sha256_digest,
)
from zlc_data import OwnedSnapshot

from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraFrameMetadataContract,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.capture_reference import (
    CAPTURE_ARTIFACT_NAMESPACE,
    CaptureArtifactRef,
)
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
    publish_manifest_with_visibility_reconciliation,
)
from zlc_neutral_atom.runtime.capture import (
    CameraCapabilityEvidence,
    CameraCaptureProvenance,
    CaptureTerminalAck,
    FrozenCaptureSpec,
    camera_capability_evidence_from_tree,
    camera_capability_evidence_to_tree,
    camera_capture_provenance_from_tree,
    camera_capture_provenance_to_tree,
    capture_terminal_ack_from_tree,
    capture_terminal_ack_to_tree,
    frozen_capture_spec_from_tree,
    frozen_capture_spec_to_tree,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetSealProvenance,
    raw_dataset_seal_provenance_from_tree,
    raw_dataset_seal_provenance_to_tree,
)
from zlc_neutral_atom.runtime.pipeline import (
    CapturePreviewPort,
    MinimalPipelineSpec,
    PipelineResult,
    _notify_preview_failure,
    compile_pipeline,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_neutral_atom.timing.capture import (
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.timing.lineage import (
    PulseCaptureEvidence,
    pulse_capture_evidence_from_tree,
    pulse_capture_evidence_to_tree,
)

from .capture_frames import (
    CaptureFrameSource,
    _CaptureFrameSourceInspection,
    _FrameResourceExceeded,
    _inspect_capture_frame_source,
    _load_capture_frame_source,
    _stage_capture_frame_source,
)
from zlc_pulse import (
    MAX_COMPILED_PULSE_ARTIFACT_BYTES,
    CompiledPulseRuntimeSummary,
    compiled_pulse_runtime_summary,
    compiled_pulse_runtime_summary_from_tree,
    compiled_pulse_runtime_summary_to_tree,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)

if TYPE_CHECKING:
    from zlc_data import DatasetSchema
    from zlc_neutral_atom.readout.contracts import ReadoutBindingKey


CAPTURE_ARTIFACT_SCHEMA = "zlc_neutral_atom.CaptureArtifact"
_CAPTURE_ARTIFACT_KIND = "capture"
_ADMITTED_CAPTURE_TOKEN = object()
_CAPTURE_ADMISSION_FIXED_BYTES = 512 * 1024
_CAPTURE_MANIFEST_DECODE_MULTIPLIER = 8


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
    max_compiled_pulse_blob_bytes: int = MAX_COMPILED_PULSE_ARTIFACT_BYTES
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
            "max_canonical_nodes",
            "max_canonical_container_entries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_compiled_pulse_blob_bytes > MAX_COMPILED_PULSE_ARTIFACT_BYTES:
            raise ValueError(
                "capture compiled-pulse budget cannot exceed the pulse owner limit"
            )

DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY = CaptureRepositoryResourcePolicy()


@dataclass(frozen=True, slots=True)
class CaptureArtifactInspection:
    """FINAL request/preflight facts obtained without decoding pulse IR."""

    reference: CaptureArtifactRef
    dataset_schema: DatasetSchema
    readout_binding: ReadoutBindingKey
    event_count: int
    max_read_scratch_bytes: int
    inspection_retained_upper_bound_bytes: int
    admission_retained_upper_bound_bytes: int
    admission_decode_peak_upper_bound_bytes: int
    pulse_runtime_summary: CompiledPulseRuntimeSummary | None

    def __post_init__(self) -> None:
        from zlc_data import DatasetSchema
        from zlc_neutral_atom.readout.contracts import ReadoutBindingKey

        if not isinstance(self.reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        if not isinstance(self.dataset_schema, DatasetSchema):
            raise TypeError("dataset_schema must be DatasetSchema")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        for field in (
            "event_count",
            "max_read_scratch_bytes",
            "inspection_retained_upper_bound_bytes",
            "admission_retained_upper_bound_bytes",
            "admission_decode_peak_upper_bound_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.event_count == 0:
            raise ValueError("event_count must be positive")
        if (
            self.admission_decode_peak_upper_bound_bytes
            < self.admission_retained_upper_bound_bytes
        ):
            raise ValueError("capture decode bound is smaller than retained state")
        if (
            self.inspection_retained_upper_bound_bytes
            > self.admission_retained_upper_bound_bytes
        ):
            raise ValueError("capture inspection bound exceeds admitted state")
        if self.pulse_runtime_summary is not None and not isinstance(
            self.pulse_runtime_summary,
            CompiledPulseRuntimeSummary,
        ):
            raise TypeError(
                "pulse_runtime_summary must be CompiledPulseRuntimeSummary or None"
            )


_CAPTURE_MANIFEST_FIELDS = {
    "schema",
    "repository_id",
    "frame_index_blob",
    "compiled_pulse_blob",
    "compiled_pulse_runtime_summary",
    "provenance",
    "terminal",
    "camera_provenance",
    "camera_capability_evidence",
    "camera_arm_spec",
    "safety_bundle_id",
    "pulse_evidence",
}


def _decode_capture_manifest(
    payload: bytes,
    policy: CaptureRepositoryResourcePolicy,
) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("capture manifest payload must be bytes")
    if not isinstance(policy, CaptureRepositoryResourcePolicy):
        raise TypeError("policy must be CaptureRepositoryResourcePolicy")
    if len(payload) > policy.max_manifest_bytes:
        raise CaptureResourceExceeded(
            "capture manifest exceeds repository resource policy"
        )

    def admit_manifest_structure(events) -> None:
        if any(isinstance(event, CanonicalArrayEvent) for event in events):
            raise CaptureResourceExceeded(
                "capture manifest cannot embed ndarray payloads"
            )

    return _exact_map(
        decode(
            payload,
            admit_structure=admit_manifest_structure,
            limits=CanonicalDecodeLimits(
                max_depth=128,
                max_nodes=policy.max_canonical_nodes,
                max_container_entries=policy.max_canonical_container_entries,
                max_arrays=0,
                max_total_array_bytes=0,
            ),
        ),
        _CAPTURE_MANIFEST_FIELDS,
        CAPTURE_ARTIFACT_SCHEMA,
    )


class AdmittedCapture:
    """Process-local proof that one exact CaptureArtifact target was committed."""

    __slots__ = (
        "_token",
        "_repository_token",
        "_reference",
        "_artifact",
        "_commit_id",
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
        commit_id: str,
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
        _canonical_text(commit_id, "commit_id")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository_token", repository_token)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_artifact", artifact)
        object.__setattr__(self, "_commit_id", commit_id)

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
    def commit_id(self) -> str:
        self._require_authority()
        return self._commit_id

    def _matches_admission(self, other: object) -> bool:
        """Compare exact process-local repository and journal authority."""

        self._require_authority()
        if type(other) is not AdmittedCapture:
            return False
        other._require_authority()
        return (
            self._repository_token is other._repository_token
            and self._reference == other._reference
            and self._commit_id == other._commit_id
        )

    def materialize_snapshot(
        self,
        *,
        memory_limit_bytes: int,
    ) -> OwnedSnapshot:
        """Materialize this admitted raw capture with its exact dataset identity."""

        self._require_authority()
        block = self._artifact.frame_source.materialize(
            memory_limit_bytes=memory_limit_bytes,
        )
        return OwnedSnapshot(
            block.ref(self._artifact.provenance.generation),
            block,
        )


@dataclass(frozen=True)
class CaptureArtifact:
    ref: CaptureArtifactRef
    frame_source: CaptureFrameSource
    provenance: DatasetSealProvenance
    terminal: CaptureTerminalAck
    camera_provenance: CameraCaptureProvenance
    camera_capability_evidence: CameraCapabilityEvidence
    camera_arm_spec: FrozenCaptureSpec
    safety_bundle_id: str
    pulse_evidence: PulseCaptureEvidence | None = None

    @property
    def run_id(self) -> str:
        """The run authority already sealed into the dataset provenance."""

        return self.provenance.trace_binding.run_id

    def __post_init__(self) -> None:
        if not isinstance(self.ref, CaptureArtifactRef):
            raise TypeError("ref must be CaptureArtifactRef")
        if not isinstance(self.frame_source, CaptureFrameSource):
            raise TypeError("frame_source must be CaptureFrameSource")
        schema = self.frame_source.schema
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if self.provenance.derivation is not None:
            raise ValueError(
                "CaptureArtifact is the raw boundary and cannot persist processor output"
            )
        if not isinstance(self.terminal, CaptureTerminalAck):
            raise TypeError("terminal must be CaptureTerminalAck")
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
        if len(
            {
                capability_evidence.fingerprint,
                self.terminal.capability_fingerprint,
                self.camera_provenance.capability_fingerprint,
            }
        ) != 1:
            raise ValueError("camera capability lineage is inconsistent")
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
        if len(
            {
                arm_spec.digest,
                self.terminal.capture_spec_fingerprint,
                self.camera_provenance.camera_arm_spec_fingerprint,
            }
        ) != 1:
            raise ValueError("camera arm-spec lineage is inconsistent")
        if (
            self.camera_provenance.binding_stamp.binding_instance_id
            != self.terminal.binding_instance_id
        ):
            raise ValueError("camera binding lineage is inconsistent")
        _canonical_text(self.safety_bundle_id, "safety_bundle_id")
        if len(
            {
                self.camera_provenance.binding.value,
                capability_evidence.source_id,
                self.provenance.trace_binding.source_id,
            }
        ) != 1:
            raise ValueError("camera source lineage is inconsistent")
        if self.pulse_evidence is not None and not isinstance(
            self.pulse_evidence,
            PulseCaptureEvidence,
        ):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence or None")
        count = self.frame_source.event_count
        physical_cells = (
            schema.repeat_axis.size
            * schema.point_layout.storage_size
        )
        if count != physical_cells:
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
        if self.frame_source.join_plan_digest != self.provenance.join_plan_digest:
            raise ValueError("source cell schedule differs from sealed join plan")
        metadata_contract = CameraFrameMetadataContract()
        if self.provenance.metadata_contract_fingerprint != metadata_contract.fingerprint:
            raise ValueError("capture metadata contract is not the current camera contract")
        if (
            self.frame_source.ordered_metadata_digest
            != self.provenance.ordered_metadata_digest
        ):
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
            self.pulse_evidence is not None
            and self.pulse_evidence.expected_trigger_count != count
        ):
            raise ValueError("pulse trigger count differs from persisted capture")
        if self.pulse_evidence is not None:
            capability_evidence.physical_facts.require_single_capture_trigger_channel(
                self.pulse_evidence.trigger_channel
            )
            if (
                self.pulse_evidence.expected_cell_schedule_digest(schema)
                != self.frame_source.join_plan_digest
            ):
                raise ValueError(
                    "pulse trigger mapping differs from persisted capture schedule"
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
        root_lease = RepositoryRootLease(self.root)
        object.__setattr__(self, "_root_lease", root_lease)
        journal = None
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
            if journal is not None:
                journal.close()
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

        coordinator = getattr(self, "_coordinator", None)
        if coordinator is not None:
            coordinator.close()
            return
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

    def load(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        """Fully validate a structurally visible artifact for inspection only."""

        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
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

    def _committed_intents(
        self,
        reference: CaptureArtifactRef,
    ) -> tuple[CommitIntent, ...]:
        self._validate_ref(reference)
        target = CommitTarget(
            self.repository_id,
            _CAPTURE_ARTIFACT_KIND,
            CAPTURE_ARTIFACT_SCHEMA,
            reference.target_ref,
            reference.manifest_digest,
        )
        matching = self._coordinator.committed_for(target)
        if not matching:
            raise PermissionError(
                "CaptureArtifact is visible but has no committed journal authority"
            )
        for intent in matching:
            if intent.commit_id != (
                f"capture-final-{intent.run_id}-{reference.manifest_digest}"
            ):
                raise ValueError(
                    "committed capture identity differs from its FINAL target"
                )
        return matching

    def inspect_final(
        self,
        reference: CaptureArtifactRef,
        *,
        memory_limit_bytes: int | None = None,
    ) -> CaptureArtifactInspection:
        """Read FINAL request/resource facts without decoding compiled pulse IR."""

        memory_limit = (
            None
            if memory_limit_bytes is None
            else positive_integer(memory_limit_bytes, "memory_limit_bytes")
        )
        manifest_limit = self.resource_policy.max_manifest_bytes
        if memory_limit is not None:
            available = memory_limit - _CAPTURE_ADMISSION_FIXED_BYTES
            if available < _CAPTURE_MANIFEST_DECODE_MULTIPLIER:
                raise MemoryError(
                    "capture inspection fixed state exceeds caller memory limit"
                )
            manifest_limit = min(
                manifest_limit,
                available // _CAPTURE_MANIFEST_DECODE_MULTIPLIER,
            )
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            self._committed_intents(reference)
            try:
                payload = self._store_authority.read_manifest(
                    CAPTURE_ARTIFACT_NAMESPACE,
                    reference.manifest_digest,
                    max_bytes=manifest_limit,
                )
            except ContentSizeLimitError as exc:
                if memory_limit is not None and (
                    manifest_limit < self.resource_policy.max_manifest_bytes
                ):
                    raise MemoryError(
                        "capture manifest inspection exceeds caller memory limit"
                    ) from exc
                raise CaptureResourceExceeded(
                    "capture manifest exceeds repository resource policy"
                ) from exc
            data = _decode_capture_manifest(payload, self.resource_policy)
            if data["repository_id"] != self.repository_id:
                raise ValueError("CaptureArtifact belongs to another repository")
            frame_index_ref = content_ref_from_tree(data["frame_index_blob"])
            pulse_ref = (
                None
                if data["compiled_pulse_blob"] is None
                else content_ref_from_tree(data["compiled_pulse_blob"])
            )
            pulse_summary = (
                None
                if data["compiled_pulse_runtime_summary"] is None
                else compiled_pulse_runtime_summary_from_tree(
                    data["compiled_pulse_runtime_summary"]
                )
            )
            if (pulse_ref is None) != (pulse_summary is None):
                raise ValueError(
                    "compiled-pulse ref and runtime summary presence differ"
                )
            if pulse_ref is not None:
                _require_content_size(
                    pulse_ref,
                    self.resource_policy.max_compiled_pulse_blob_bytes,
                    "compiled-pulse",
                )
                assert pulse_summary is not None
                pulse_summary.require_encoded_size(pulse_ref.size)
            manifest_peak = (
                _CAPTURE_ADMISSION_FIXED_BYTES
                + _CAPTURE_MANIFEST_DECODE_MULTIPLIER * len(payload)
            )
            if memory_limit is not None and manifest_peak > memory_limit:
                raise MemoryError(
                    "capture manifest inspection exceeds caller memory limit"
                )
            frame_memory_limit = (
                None
                if memory_limit is None
                else memory_limit - manifest_peak
            )
            if frame_memory_limit is not None and frame_memory_limit < 1:
                raise MemoryError(
                    "capture frame inspection exceeds caller memory limit"
                )
            try:
                frame = _inspect_capture_frame_source(
                    frame_index_ref,
                    store_authority=self._store_authority,
                    max_cells=self.resource_policy.max_cells,
                    max_total_frame_bytes=self.resource_policy.max_total_frame_bytes,
                    max_chunk_blob_bytes=(
                        self.resource_policy.max_frame_chunk_blob_bytes
                    ),
                    max_frame_index_blob_bytes=(
                        self.resource_policy.max_frame_index_blob_bytes
                    ),
                    max_canonical_nodes=self.resource_policy.max_canonical_nodes,
                    max_canonical_container_entries=(
                        self.resource_policy.max_canonical_container_entries
                    ),
                    memory_limit_bytes=frame_memory_limit,
                )
            except (_FrameResourceExceeded, ContentSizeLimitError) as exc:
                raise CaptureResourceExceeded(
                    "capture frame storage exceeds repository resource policy"
                ) from exc
            provenance = camera_capture_provenance_from_tree(
                data["camera_provenance"]
            )
            pulse_retained = (
                0 if pulse_summary is None else pulse_summary.retained_upper_bound_bytes
            )
            pulse_decode = (
                0 if pulse_summary is None else pulse_summary.decode_peak_upper_bound_bytes
            )
            retained = (
                _CAPTURE_ADMISSION_FIXED_BYTES
                + frame.retained_upper_bound
                + pulse_retained
            )
            inspection_retained = (
                _CAPTURE_ADMISSION_FIXED_BYTES + frame.retained_upper_bound
            )
            decode_peak = max(
                frame.decode_peak_upper_bound + pulse_retained,
                frame.retained_upper_bound + pulse_decode,
            ) + (
                _CAPTURE_ADMISSION_FIXED_BYTES
                + _CAPTURE_MANIFEST_DECODE_MULTIPLIER * len(payload)
            )
            if memory_limit is not None and max(retained, decode_peak) > memory_limit:
                raise MemoryError(
                    "capture admission exceeds caller memory limit"
                )
            return CaptureArtifactInspection(
                reference,
                frame.dataset_schema,
                provenance.binding,
                frame.event_count,
                frame.max_read_scratch_bytes,
                inspection_retained,
                retained,
                max(retained, decode_peak),
                pulse_summary,
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
        data = _decode_capture_manifest(manifest_payload, policy)
        if data["repository_id"] != repository_id:
            raise ValueError("CaptureArtifact belongs to another repository")
        frame_index_ref = content_ref_from_tree(data["frame_index_blob"])
        pulse_ref = (
            None
            if data["compiled_pulse_blob"] is None
            else content_ref_from_tree(data["compiled_pulse_blob"])
        )
        pulse_summary = (
            None
            if data["compiled_pulse_runtime_summary"] is None
            else compiled_pulse_runtime_summary_from_tree(
                data["compiled_pulse_runtime_summary"]
            )
        )
        if (pulse_ref is None) != (pulse_summary is None):
            raise ValueError(
                "compiled-pulse ref and runtime summary presence differ"
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
            assert pulse_summary is not None
            pulse_summary.require_encoded_size(pulse_ref.size)
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
            if compiled_pulse_runtime_summary(
                compiled_pulse,
                encoded_size=pulse_ref.size,
            ) != pulse_summary:
                raise ValueError(
                    "compiled-pulse runtime summary differs from decoded lineage"
                )
        artifact = CaptureArtifact(
            ref=reference,
            frame_source=frame_source,
            provenance=raw_dataset_seal_provenance_from_tree(data["provenance"]),
            terminal=capture_terminal_ack_from_tree(data["terminal"]),
            camera_provenance=camera_capture_provenance_from_tree(
                data["camera_provenance"]
            ),
            camera_capability_evidence=camera_capability_evidence_from_tree(
                data["camera_capability_evidence"]
            ),
            camera_arm_spec=frozen_capture_spec_from_tree(data["camera_arm_spec"]),
            safety_bundle_id=data["safety_bundle_id"],
            pulse_evidence=pulse_capture_evidence_from_tree(
                data["pulse_evidence"],
                compiled_pulse,
            ),
        )
        # Enforce one canonical current representation, not merely a decodable one.
        rebuilt_payload = _manifest_payload(
            repository_id=artifact.ref.repository_id,
            frame_index_ref=frame_index_ref,
            compiled_pulse_ref=pulse_ref,
            compiled_pulse_runtime_summary=pulse_summary,
            provenance=artifact.provenance,
            terminal=artifact.terminal,
            camera_provenance=artifact.camera_provenance,
            camera_capability_evidence=artifact.camera_capability_evidence,
            camera_arm_spec=artifact.camera_arm_spec,
            safety_bundle_id=artifact.safety_bundle_id,
            pulse_evidence=artifact.pulse_evidence,
        )
        if (
            sha256_digest(rebuilt_payload) != reference.manifest_digest
            or rebuilt_payload != manifest_payload
        ):
            raise ValueError("CaptureArtifact manifest is not canonical")
        return artifact

    def admit(self, reference: CaptureArtifactRef) -> AdmittedCapture:
        """Mint authority only for an exact journal-committed capture target."""

        with self._root_lease.borrow() as admission_borrow:
            admission_borrow.require_active()
            artifact = self.load(reference)
            matching = self._committed_intents(reference)
            for intent in matching:
                expected_commit_id = (
                    f"capture-final-{artifact.run_id}-"
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
            selected = min(matching, key=lambda intent: intent.commit_id)
            return AdmittedCapture(
                _ADMITTED_CAPTURE_TOKEN,
                repository_token=self._root_lease,
                reference=reference,
                artifact=artifact,
                commit_id=selected.commit_id,
            )

    def _final_commit(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
        *,
        compiled_pulse_ref: ContentRef | None,
    ) -> FinalCommit[CaptureArtifactRef]:
        operation = self._commit_operation(
            context,
            result,
            compiled_pulse_ref=compiled_pulse_ref,
        )
        return operation

    def _commit_operation(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
        *,
        compiled_pulse_ref: ContentRef | None,
    ) -> FinalCommit[CaptureArtifactRef]:
        self._require_active()
        if not isinstance(context, PostSafetyContext):
            raise TypeError("capture commit requires PostSafetyContext")
        if not isinstance(result, (PipelineResult, TriggeredPipelineResult)):
            raise TypeError("capture commit requires an exact pipeline result")
        run_id, safety_bundle_id = context.authorize_commit_preparation()
        # Hold the root before the first CAS write.  prepare() synchronously
        # mints the long-lived commit borrow before this temporary hold exits.
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, manifest_payload = self._stage_pipeline_result(
                result,
                context,
                compiled_pulse_ref=compiled_pulse_ref,
            )
            confirmed_subject = context.authorize_commit_preparation()
            if confirmed_subject != (run_id, safety_bundle_id):
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

            commit_id = (
                f"capture-final-{run_id}-"
                f"{reference.manifest_digest}"
            )
            operation = self._coordinator.prepare(
                commit_id,
                run_id,
                safety_bundle_id,
                target,
                publish,
            )
        try:
            context._track_prepared_commit(operation)
        except BaseException:
            operation.abandon()
            raise
        return operation

    def _recover(
        self,
        intent: CommitIntent,
    ) -> PublishedManifest[CaptureArtifactRef] | None:
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
        expected_commit_id = f"capture-final-{intent.run_id}-{digest}"
        if intent.commit_id != expected_commit_id:
            raise ValueError("capture commit id differs from kind/run/target")
        try:
            manifest_payload = store_authority.read_manifest(
                CAPTURE_ARTIFACT_NAMESPACE,
                digest,
                max_bytes=policy.max_manifest_bytes,
            )
        except FileNotFoundError:
            return None
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
        # Event chunks were already read and validated while loading the source.
        artifact.frame_source._verify_all_frame_chunks()
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
        return PublishedManifest(reference.target_ref, digest, reference)

    def _stage_pipeline_result(
        self,
        result: PipelineResult | TriggeredPipelineResult,
        context: PostSafetyContext,
        *,
        compiled_pulse_ref: ContentRef | None,
    ) -> tuple[CaptureArtifactRef, bytes]:
        self._require_active()
        policy = self.resource_policy
        evidence = None
        if isinstance(result, TriggeredPipelineResult):
            base = result.capture
            evidence = result.lineage.evidence()
            if not isinstance(compiled_pulse_ref, ContentRef):
                raise TypeError("triggered capture requires a staged compiled-pulse ref")
            if compiled_pulse_ref.digest != evidence.compiled_artifact.fingerprint:
                raise ValueError("staged compiled-pulse ref differs from pulse evidence")
            _require_content_size(
                compiled_pulse_ref,
                policy.max_compiled_pulse_blob_bytes,
                "compiled-pulse",
            )
            self._store_authority.verify_blob(
                compiled_pulse_ref,
                max_bytes=policy.max_compiled_pulse_blob_bytes,
            )
        else:
            base = result
            if compiled_pulse_ref is not None:
                raise ValueError("untriggered capture cannot name a compiled-pulse ref")
        if base.run_id != context.run_id.value:
            raise ValueError("PostSafetyContext run_id differs from pipeline result")
        if context.safety_bundle_id is None:
            raise ValueError("CaptureArtifact requires a durable safety bundle id")
        if not base.is_direct_raw_capture:
            raise ValueError(
                "CaptureArtifact only accepts direct raw camera datasets"
            )
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
        _require_content_size(
            frame_index_ref,
            policy.max_frame_index_blob_bytes,
            "frame-index",
        )
        if (evidence is None) != (compiled_pulse_ref is None):
            raise ValueError("pulse evidence and staged compiled-pulse ref differ")
        if compiled_pulse_ref is not None:
            _require_content_size(
                compiled_pulse_ref,
                policy.max_compiled_pulse_blob_bytes,
                "compiled-pulse",
            )
        pulse_summary = (
            None
            if evidence is None or compiled_pulse_ref is None
            else compiled_pulse_runtime_summary(
                evidence.compiled_artifact,
                encoded_size=compiled_pulse_ref.size,
            )
        )
        manifest_payload = _manifest_payload(
            repository_id=self.repository_id,
            frame_index_ref=frame_index_ref,
            compiled_pulse_ref=compiled_pulse_ref,
            compiled_pulse_runtime_summary=pulse_summary,
            provenance=base.dataset.provenance,
            terminal=base.capture_terminal,
            camera_provenance=base.camera_provenance,
            camera_capability_evidence=base.camera_capability_evidence,
            camera_arm_spec=base.camera_arm_spec,
            safety_bundle_id=context.safety_bundle_id,
            pulse_evidence=evidence,
        )
        if len(manifest_payload) > policy.max_manifest_bytes:
            raise CaptureResourceExceeded(
                "capture manifest exceeds repository resource policy"
            )
        reference = CaptureArtifactRef(
            self.repository_id,
            sha256_digest(manifest_payload),
        )
        CaptureArtifact(
            ref=reference,
            frame_source=frame_source,
            provenance=base.dataset.provenance,
            terminal=base.capture_terminal,
            camera_provenance=base.camera_provenance,
            camera_capability_evidence=base.camera_capability_evidence,
            camera_arm_spec=base.camera_arm_spec,
            safety_bundle_id=context.safety_bundle_id,
            pulse_evidence=evidence,
        )
        return reference, manifest_payload

    def _validate_ref(self, reference: CaptureArtifactRef) -> None:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("load requires CaptureArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("CaptureArtifactRef belongs to another repository")


def _stage_compiled_pulse(
    spec: MinimalPipelineSpec | TriggeredCaptureSpec,
    repository: CaptureRepository,
) -> ContentRef | None:
    """Persist immutable pulse input before base preflight can arm or execute."""

    if not isinstance(spec, TriggeredCaptureSpec):
        return None
    pulse_payload = encode_compiled_pulse_artifact(
        spec.pulse_binding.compiled_artifact
    )
    if len(pulse_payload) > repository.resource_policy.max_compiled_pulse_blob_bytes:
        raise CaptureResourceExceeded(
            "capture compiled-pulse blob exceeds repository resource policy"
        )
    reference = repository._store_authority.put_blob(pulse_payload)
    if reference.digest != spec.pulse_binding.compiled_artifact.fingerprint:
        raise RuntimeError("compiled-pulse CAS identity differs from pulse owner")
    return reference


def compile_capture_artifact_pipeline(
    spec: MinimalPipelineSpec | TriggeredCaptureSpec,
    repository: CaptureRepository,
    *,
    preview: CapturePreviewPort | None = None,
) -> RunPlan:
    """Add one post-safety CaptureArtifact commit to the exact pipeline."""

    try:
        if type(repository) is not CaptureRepository:
            raise TypeError("repository must be CaptureRepository")
        capture_spec = spec.capture if isinstance(spec, TriggeredCaptureSpec) else spec
        if not isinstance(capture_spec, MinimalPipelineSpec):
            raise TypeError("capture artifact pipeline requires MinimalPipelineSpec")
        repository._require_active()
    except BaseException as error:
        _notify_preview_failure(preview, error)
        raise
    base = (
        compile_triggered_pipeline(spec, preview=preview)
        if isinstance(spec, TriggeredCaptureSpec)
        else compile_pipeline(spec, preview=preview)
    )
    base_name = base.name
    base_resource_claims = base.resource_claims
    base_bound_devices = base.bound_devices
    base_preflight = base.preflight
    base_execute = base.execute
    base_cleanup = base.cleanup
    base_finalize = base.finalize
    base_dispose_unfinalized = base.dispose_unfinalized
    base_interrupt_operations = base.interrupt_operations
    base_timeout_seconds = base.timeout_seconds

    def preflight(
        context: RunContext,
    ) -> tuple[object, RepositoryRootLeaseBorrow, ContentRef | None]:
        # The durable sink is part of this hardware plan's admission, not a
        # post-capture convenience.  Taking the borrow before base preflight
        # makes close-vs-run linearizable: close wins before plan-level hardware
        # admission, or this run wins and close cannot invalidate its sink
        # mid-capture.  Run binding may already have performed read-only identity
        # probes before this plan preflight begins.
        borrow = None
        try:
            repository._require_active()
            borrow = repository._root_lease.borrow()
            pulse_ref = _stage_compiled_pulse(spec, repository)
            return base_preflight(context), borrow, pulse_ref
        except BaseException as error:
            _notify_preview_failure(preview, error)
            if borrow is not None:
                try:
                    borrow.close()
                except BaseException as close_error:
                    try:
                        error.add_note(
                            "repository borrow close also failed: "
                            f"{type(close_error).__name__}: {close_error}"
                        )
                    except BaseException:
                        pass
            raise

    def execute(
        context: RunContext,
        prepared: tuple[object, RepositoryRootLeaseBorrow, ContentRef | None],
    ) -> tuple[
        PipelineResult | TriggeredPipelineResult,
        RepositoryRootLeaseBorrow,
        ContentRef | None,
    ]:
        base_prepared, borrow, pulse_ref = prepared
        borrow.require_active()
        return base_execute(context, base_prepared), borrow, pulse_ref

    def cleanup(
        context: RunContext,
        prepared: tuple[object, RepositoryRootLeaseBorrow, ContentRef | None] | None,
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
        result: tuple[
            PipelineResult | TriggeredPipelineResult,
            RepositoryRootLeaseBorrow,
            ContentRef | None,
        ],
    ) -> CaptureArtifactRef:
        base_result, borrow, pulse_ref = result
        try:
            borrow.require_active()
            finalized = base_finalize(context, base_result)
            if not isinstance(finalized, (PipelineResult, TriggeredPipelineResult)):
                raise TypeError("base exact pipeline changed its result contract")
            return context.commit_final(
                repository._final_commit(
                    context,
                    finalized,
                    compiled_pulse_ref=pulse_ref,
                )
            )
        finally:
            # Visibility reconciliation owns its own commit authority; this
            # broader run admission ends when finalize has handed off or failed.
            borrow.close()

    def dispose_unfinalized(
        result: tuple[
            PipelineResult | TriggeredPipelineResult,
            RepositoryRootLeaseBorrow,
            ContentRef | None,
        ],
    ) -> None:
        """Release the sink hold when RunController skips finalize."""

        base_result, borrow, _pulse_ref = result
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
                try:
                    error.add_note(
                        "repository borrow close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                except BaseException:
                    pass
        if error is not None:
            raise error

    return RunPlan(
        name=base_name,
        resource_claims=base_resource_claims,
        bound_devices=base_bound_devices,
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=base_interrupt_operations,
        timeout_seconds=base_timeout_seconds,
        requires_final_commit=True,
        dispose_unfinalized=dispose_unfinalized,
    )


def _manifest_payload(
    *,
    repository_id: str,
    frame_index_ref: ContentRef,
    compiled_pulse_ref: ContentRef | None,
    compiled_pulse_runtime_summary: CompiledPulseRuntimeSummary | None,
    provenance: DatasetSealProvenance,
    terminal: CaptureTerminalAck,
    camera_provenance: CameraCaptureProvenance,
    camera_capability_evidence: CameraCapabilityEvidence,
    camera_arm_spec: FrozenCaptureSpec,
    safety_bundle_id: str,
    pulse_evidence: PulseCaptureEvidence | None,
) -> bytes:
    absent = pulse_evidence is None
    if absent != (compiled_pulse_ref is None) or absent != (
        compiled_pulse_runtime_summary is None
    ):
        raise ValueError(
            "pulse evidence, compiled-pulse blob, and runtime summary presence differ"
        )
    return encode(
        {
            "schema": CAPTURE_ARTIFACT_SCHEMA,
            "repository_id": _canonical_text(repository_id, "repository_id"),
            "frame_index_blob": content_ref_to_tree(frame_index_ref),
            "compiled_pulse_blob": (
                None
                if compiled_pulse_ref is None
                else content_ref_to_tree(compiled_pulse_ref)
            ),
            "compiled_pulse_runtime_summary": (
                None
                if compiled_pulse_runtime_summary is None
                else compiled_pulse_runtime_summary_to_tree(
                    compiled_pulse_runtime_summary
                )
            ),
            "provenance": raw_dataset_seal_provenance_to_tree(provenance),
            "terminal": capture_terminal_ack_to_tree(terminal),
            "camera_provenance": camera_capture_provenance_to_tree(
                camera_provenance
            ),
            "camera_capability_evidence": camera_capability_evidence_to_tree(
                camera_capability_evidence
            ),
            "camera_arm_spec": frozen_capture_spec_to_tree(
                camera_arm_spec
            ),
            "safety_bundle_id": _canonical_text(
                safety_bundle_id,
                "safety_bundle_id",
            ),
            "pulse_evidence": pulse_capture_evidence_to_tree(
                pulse_evidence
            ),
        }
    )


__all__ = [
    "AdmittedCapture",
    "CAPTURE_ARTIFACT_SCHEMA",
    "CaptureArtifact",
    "CaptureArtifactInspection",
    "CaptureArtifactRef",
    "CaptureFrameSource",
    "CaptureRepositoryResourcePolicy",
    "CaptureRepository",
    "CaptureResourceExceeded",
    "DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY",
    "compile_capture_artifact_pipeline",
]
