"""Current lazy raw-frame CaptureArtifact and manifest-only repository."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zlc_storage import (
    CanonicalArrayEvent,
    ContentAddressedStore,
    ContentCorruptionError,
    ContentRef,
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
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource

from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraFrameMetadataContract,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
)
from .reference import (
    CAPTURE_ARTIFACT_NAMESPACE,
    CaptureArtifactRef,
)
from zlc_neutral_atom.runtime.commit import PreparedArtifactCommit
from .session import (
    CameraCaptureProvenance,
    camera_capture_provenance_from_tree,
    camera_capture_provenance_to_tree,
)
from zlc_neutral_atom.devices.camera.capture_port import (
    CaptureTerminalAck,
    capture_terminal_ack_from_tree,
    capture_terminal_ack_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCapabilityEvidence,
    FrozenCaptureSpec,
    camera_capability_evidence_from_tree,
    camera_capability_evidence_to_tree,
    frozen_capture_spec_from_tree,
    frozen_capture_spec_to_tree,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetSealProvenance,
    raw_dataset_seal_provenance_from_tree,
    raw_dataset_seal_provenance_to_tree,
)
from .pipeline import (
    CapturePreviewPort,
    MinimalPipelineSpec,
    PipelineResult,
    compile_pipeline,
)
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    notify_preview_failure,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from .triggered import (
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.timing.lineage import (
    PulseCaptureEvidence,
    pulse_capture_evidence_from_tree,
    pulse_capture_evidence_to_tree,
)

from .frames import (
    CaptureFrameSource,
    _load_capture_frame_source,
    _stage_capture_frame_source,
)
from zlc_pulse import (
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)

CAPTURE_ARTIFACT_SCHEMA = "zlc_neutral_atom.CaptureArtifact"
_ADMITTED_CAPTURE_TOKEN = object()
_CAPTURE_MANIFEST_FIELDS = {
    "schema",
    "repository_id",
    "frame_index_blob",
    "compiled_pulse_blob",
    "provenance",
    "terminal",
    "camera_provenance",
    "camera_capability_evidence",
    "camera_arm_spec",
    "pulse_evidence",
}


def _decode_capture_manifest(
    payload: bytes,
) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("capture manifest payload must be bytes")
    def admit_manifest_structure(events) -> None:
        if any(isinstance(event, CanonicalArrayEvent) for event in events):
            raise ValueError(
                "capture manifest cannot embed ndarray payloads"
            )

    return _exact_map(
        decode(
            payload,
            admit_structure=admit_manifest_structure,
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
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_repository_token", repository_token)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_artifact", artifact)

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

    def _matches_admission(self, other: object) -> bool:
        """Compare exact process-local repository and manifest authority."""

        self._require_authority()
        if type(other) is not AdmittedCapture:
            return False
        other._require_authority()
        return (
            self._repository_token is other._repository_token
            and self._reference == other._reference
        )

    def materialize_snapshot(
        self,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> OwnedSnapshot:
        """Materialize this admitted raw capture with its exact dataset identity."""

        self._require_authority()
        block = self._artifact.frame_source.materialize(
            abort_check=abort_check,
        )
        return OwnedSnapshot(
            block.ref(self._artifact.provenance.generation),
            block,
        )


def _validate_capture_metadata_contract(
    *,
    schema,
    count: int,
    provenance: DatasetSealProvenance,
    terminal: CaptureTerminalAck,
    camera_provenance: CameraCaptureProvenance,
    camera_capability_evidence: CameraCapabilityEvidence,
    camera_arm_spec: FrozenCaptureSpec,
) -> None:
    """Validate the typed cross-facts shared by inspection and full admission."""

    from zlc_data import DatasetSchema

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    count = positive_integer(count, "capture event count")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    if provenance.derivation is not None:
        raise ValueError(
            "CaptureArtifact is the raw boundary and cannot persist processor output"
        )
    if not isinstance(terminal, CaptureTerminalAck):
        raise TypeError("terminal must be CaptureTerminalAck")
    if not isinstance(camera_provenance, CameraCaptureProvenance):
        raise TypeError("camera_provenance must be CameraCaptureProvenance")
    camera_provenance.validate_schema(schema)
    if not isinstance(camera_capability_evidence, CameraCapabilityEvidence):
        raise TypeError(
            "camera_capability_evidence must be CameraCapabilityEvidence"
        )
    capability_evidence = camera_capability_evidence
    if not isinstance(camera_arm_spec, FrozenCaptureSpec):
        raise TypeError("camera_arm_spec must be FrozenCaptureSpec")
    arm_spec = camera_arm_spec
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
            terminal.capability_fingerprint,
            camera_provenance.capability_fingerprint,
        }
    ) != 1:
        raise ValueError("camera capability lineage is inconsistent")
    capability_evidence.physical_facts.validate_descriptor(
        camera_provenance.descriptor
    )
    if (
        capability_evidence.payload_contract_fingerprint
        != CameraSampleContract(schema.cell_schema).fingerprint
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
            terminal.capture_spec_fingerprint,
            camera_provenance.camera_arm_spec_fingerprint,
        }
    ) != 1:
        raise ValueError("camera arm-spec lineage is inconsistent")
    if (
        camera_provenance.binding_stamp.binding_instance_id
        != terminal.binding_instance_id
    ):
        raise ValueError("camera binding lineage is inconsistent")
    if len(
        {
            camera_provenance.binding.value,
            capability_evidence.source_id,
            provenance.trace_binding.source_id,
        }
    ) != 1:
        raise ValueError("camera source lineage is inconsistent")
    physical_cells = schema.repeat_axis.size * schema.point_table.row_count
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
            terminal.settings_fingerprint,
            camera_provenance.active_settings_fingerprint,
        }
    ) != 1:
        raise ValueError(
            "camera arm settings differ from capability and terminal evidence"
        )
    if provenance.end_sequence - provenance.start_sequence != count:
        raise ValueError("capture provenance interval differs from metadata cardinality")
    metadata_contract = CameraFrameMetadataContract()
    if provenance.metadata_contract_fingerprint != metadata_contract.fingerprint:
        raise ValueError("capture metadata contract is not the current camera contract")
    if (
        terminal.produced_count != count
        or terminal.drained_count != count
        or terminal.ordered_metadata_digest != provenance.ordered_metadata_digest
        or not terminal.source_stopped
        or not terminal.no_more_frames
        or not terminal.joined
    ):
        raise ValueError("capture terminal evidence differs from persisted dataset")


@dataclass(frozen=True)
class CaptureArtifact:
    ref: CaptureArtifactRef
    frame_source: CaptureFrameSource
    provenance: DatasetSealProvenance
    terminal: CaptureTerminalAck
    camera_provenance: CameraCaptureProvenance
    camera_capability_evidence: CameraCapabilityEvidence
    camera_arm_spec: FrozenCaptureSpec
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
        count = self.frame_source.event_count
        _validate_capture_metadata_contract(
            schema=schema,
            count=count,
            provenance=self.provenance,
            terminal=self.terminal,
            camera_provenance=self.camera_provenance,
            camera_capability_evidence=self.camera_capability_evidence,
            camera_arm_spec=self.camera_arm_spec,
        )
        if self.pulse_evidence is not None and not isinstance(
            self.pulse_evidence,
            PulseCaptureEvidence,
        ):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence or None")
        if self.frame_source.join_plan_digest != self.provenance.join_plan_digest:
            raise ValueError("source cell schedule differs from sealed join plan")
        if (
            self.frame_source.ordered_metadata_digest
            != self.provenance.ordered_metadata_digest
        ):
            raise ValueError("capture metadata sequence digest differs from provenance")
        if (
            self.pulse_evidence is not None
            and self.pulse_evidence.expected_trigger_count != count
        ):
            raise ValueError("pulse trigger count differs from persisted capture")
        if self.pulse_evidence is not None:
            self.camera_capability_evidence.physical_facts.require_single_capture_trigger_channel(
                self.pulse_evidence.trigger_channel
            )
            if (
                self.pulse_evidence.expected_cell_schedule_digest(schema)
                != self.frame_source.join_plan_digest
            ):
                raise ValueError(
                    "pulse trigger mapping differs from persisted capture schedule"
                )


class CaptureRepository:
    """Current-only raw-capture CAS whose manifest is the visibility authority."""

    __slots__ = (
        "root",
        "repository_id",
        "_root_lease",
        "_store",
        "_store_authority",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CaptureRepository is final and cannot be subclassed")

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-capture",
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        object.__setattr__(
            self,
            "repository_id",
            _canonical_text(repository_id, "repository_id"),
        )
        root_lease = RepositoryRootLease(self.root)
        object.__setattr__(self, "_root_lease", root_lease)
        try:
            # Acquire the root lease before any content-store I/O so a losing
            # second writer cannot inspect another live owner's repository.
            store = ContentAddressedStore(self.root / "content")
            object.__setattr__(self, "_store", store)
            object.__setattr__(self, "_store_authority", store.authority())
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
        """Close after every prepared/in-flight manifest operation is resolved."""

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

    def load(
        self,
        reference: CaptureArtifactRef,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> CaptureArtifact:
        """Fully validate a structurally visible artifact for inspection only."""

        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            if abort_check is not None:
                abort_check()
            self._validate_ref(reference)
            manifest_payload = self._store_authority.read_manifest(
                CAPTURE_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
            )
            return self._load_manifest(
                reference,
                manifest_payload,
                store_authority=self._store_authority,
                repository_id=self.repository_id,
                abort_check=abort_check,
            )

    def _load_manifest(
        self,
        reference: CaptureArtifactRef,
        manifest_payload: bytes,
        *,
        store_authority: ContentStoreAuthority,
        repository_id: str,
        abort_check: Callable[[], None] | None = None,
    ) -> CaptureArtifact:
        self._require_active()
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if abort_check is not None:
            abort_check()
        data = _decode_capture_manifest(manifest_payload)
        if data["repository_id"] != repository_id:
            raise ValueError("CaptureArtifact belongs to another repository")
        frame_index_ref = content_ref_from_tree(data["frame_index_blob"])
        pulse_ref = (
            None
            if data["compiled_pulse_blob"] is None
            else content_ref_from_tree(data["compiled_pulse_blob"])
        )
        frame_source = _load_capture_frame_source(
            frame_index_ref,
            store_authority=store_authority,
            root_lease=self._root_lease,
            abort_check=abort_check,
        )
        if abort_check is not None:
            abort_check()
        compiled_pulse = (
            None
            if pulse_ref is None
            else decode_compiled_pulse_artifact(store_authority.read_blob(pulse_ref))
        )
        if abort_check is not None:
            abort_check()
        if compiled_pulse is not None and pulse_ref is not None:
            if pulse_ref.digest != compiled_pulse.fingerprint:
                raise ValueError(
                    "compiled-pulse blob digest differs from artifact fingerprint"
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
            provenance=artifact.provenance,
            terminal=artifact.terminal,
            camera_provenance=artifact.camera_provenance,
            camera_capability_evidence=artifact.camera_capability_evidence,
            camera_arm_spec=artifact.camera_arm_spec,
            pulse_evidence=artifact.pulse_evidence,
        )
        if (
            sha256_digest(rebuilt_payload) != reference.manifest_digest
            or rebuilt_payload != manifest_payload
        ):
            raise ValueError("CaptureArtifact manifest is not canonical")
        artifact.frame_source._verify_all_frame_chunks()
        return artifact

    def admit(
        self,
        reference: CaptureArtifactRef,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> AdmittedCapture:
        """Mint process-local authority for one canonical visible manifest."""

        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        artifact = self.load(reference, abort_check=abort_check)
        if abort_check is not None:
            abort_check()
        return AdmittedCapture(
            _ADMITTED_CAPTURE_TOKEN,
            repository_token=self._root_lease,
            reference=reference,
            artifact=artifact,
        )

    def materialize_final(
        self,
        reference: CaptureArtifactRef,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> OwnedSnapshot:
        """Materialize one FINAL capture."""

        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if abort_check is not None:
            abort_check()
        admitted = self.admit(reference, abort_check=abort_check)
        return admitted.materialize_snapshot(
            abort_check=abort_check,
        )

    def project_dataset_source(
        self,
        reference: CaptureArtifactRef,
        *,
        materialize: bool,
        abort_check: Callable[[], None] | None = None,
    ) -> ArtifactDatasetSource:
        """Project this capture's exact Dataset without leaking storage fields."""

        if type(materialize) is not bool:
            raise TypeError("materialize must be bool")
        if materialize:
            snapshot = self.materialize_final(
                reference,
                abort_check=abort_check,
            )
            return ArtifactDatasetSource(
                snapshot.block.schema,
                snapshot.ref,
                snapshot,
            )
        admitted = self.admit(reference, abort_check=abort_check)
        artifact = admitted.artifact
        return ArtifactDatasetSource(
            artifact.frame_source.schema,
            artifact.frame_source.ref(artifact.provenance.generation),
        )

    def _final_commit(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
        *,
        compiled_pulse_ref: ContentRef | None,
    ) -> PreparedArtifactCommit[CaptureArtifactRef]:
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
    ) -> PreparedArtifactCommit[CaptureArtifactRef]:
        self._require_active()
        if not isinstance(context, PostSafetyContext):
            raise TypeError("capture commit requires PostSafetyContext")
        if not isinstance(result, (PipelineResult, TriggeredPipelineResult)):
            raise TypeError("capture commit requires an exact pipeline result")
        run_id = context.authorize_commit_preparation()
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
            if confirmed_subject != run_id:
                raise RuntimeError("capture commit subject changed while staging")

            def publish(payload: bytes) -> None:
                self._require_active()
                stored = self._store_authority.publish_manifest(
                    CAPTURE_ARTIFACT_NAMESPACE,
                    payload,
                    expected_digest=reference.manifest_digest,
                )
                if stored.content.digest != reference.manifest_digest:
                    raise RuntimeError("published capture manifest digest changed")

            def inspect(payload: bytes) -> bool | None:
                self._require_active()
                try:
                    durable_payload = self._store_authority.confirm_manifest_durable(
                        CAPTURE_ARTIFACT_NAMESPACE,
                        reference.manifest_digest,
                    )
                except FileNotFoundError:
                    return False
                except OSError:
                    return None
                if durable_payload != payload:
                    raise ContentCorruptionError(
                        "visible capture manifest differs from prepared payload"
                    )
                try:
                    artifact = self._load_manifest(
                        reference,
                        durable_payload,
                        store_authority=self._store_authority,
                        repository_id=self.repository_id,
                    )
                except FileNotFoundError as error:
                    raise ContentCorruptionError(
                        "visible capture manifest references missing content"
                    ) from error
                except OSError:
                    return None
                if artifact.run_id != run_id:
                    raise ValueError(
                        "visible capture provenance belongs to another Run"
                    )
                return True

            commit_borrow = self._root_lease.borrow()
            try:
                operation = PreparedArtifactCommit(
                    run_id=run_id,
                    result=reference,
                    manifest_payload=manifest_payload,
                    publish=publish,
                    inspect=inspect,
                    repository_borrow=commit_borrow,
                )
            except BaseException:
                commit_borrow.close()
                raise
        try:
            context.track_prepared_commit(operation)
        except BaseException:
            operation.abandon()
            raise
        return operation

    def _stage_pipeline_result(
        self,
        result: PipelineResult | TriggeredPipelineResult,
        context: PostSafetyContext,
        *,
        compiled_pulse_ref: ContentRef | None,
    ) -> tuple[CaptureArtifactRef, bytes]:
        self._require_active()
        evidence = None
        if isinstance(result, TriggeredPipelineResult):
            base = result.capture
            evidence = result.lineage.evidence()
            if not isinstance(compiled_pulse_ref, ContentRef):
                raise TypeError("triggered capture requires a staged compiled-pulse ref")
            if compiled_pulse_ref.digest != evidence.compiled_artifact.fingerprint:
                raise ValueError("staged compiled-pulse ref differs from pulse evidence")
            self._store_authority.verify_blob(compiled_pulse_ref)
        else:
            base = result
            if compiled_pulse_ref is not None:
                raise ValueError("untriggered capture cannot name a compiled-pulse ref")
        if base.run_id != context.run_id.value:
            raise ValueError("PostSafetyContext run_id differs from pipeline result")
        if not base.is_direct_raw_capture:
            raise ValueError(
                "CaptureArtifact only accepts direct raw camera datasets"
            )
        frame_source, frame_index_ref = _stage_capture_frame_source(
            block=base.dataset.block,
            event_metadata=tuple(base.dataset.event_metadata),
            cell_schedule=base.source_cell_schedule,
            store_authority=self._store_authority,
            root_lease=self._root_lease,
        )
        if (evidence is None) != (compiled_pulse_ref is None):
            raise ValueError("pulse evidence and staged compiled-pulse ref differ")
        manifest_payload = _manifest_payload(
            repository_id=self.repository_id,
            frame_index_ref=frame_index_ref,
            compiled_pulse_ref=compiled_pulse_ref,
            provenance=base.dataset.provenance,
            terminal=base.capture_terminal,
            camera_provenance=base.camera_provenance,
            camera_capability_evidence=base.camera_capability_evidence,
            camera_arm_spec=base.camera_arm_spec,
            pulse_evidence=evidence,
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
            pulse_evidence=evidence,
        )
        return reference, manifest_payload

    def _validate_ref(self, reference: CaptureArtifactRef) -> None:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("load requires CaptureArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("CaptureArtifactRef belongs to another repository")


class PendingCaptureArtifact:
    """Exact capture plus private repository admission awaiting FINAL commit.

    A domain Task may inspect ``pipeline_result`` for deterministic validation
    before FINAL.  Repository lifetime and staged pulse content remain owned by
    this capture-artifact boundary.
    """

    __slots__ = ("_pipeline_result", "_repository_borrow", "_compiled_pulse_ref")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("PendingCaptureArtifact is final and cannot be subclassed")

    def __init__(
        self,
        pipeline_result: PipelineResult | TriggeredPipelineResult,
        repository_borrow: RepositoryRootLeaseBorrow,
        compiled_pulse_ref: ContentRef | None,
    ) -> None:
        if not isinstance(pipeline_result, (PipelineResult, TriggeredPipelineResult)):
            raise TypeError("pipeline_result must be an exact capture result")
        if not isinstance(repository_borrow, RepositoryRootLeaseBorrow):
            raise TypeError("repository_borrow must be RepositoryRootLeaseBorrow")
        if compiled_pulse_ref is not None and not isinstance(compiled_pulse_ref, ContentRef):
            raise TypeError("compiled_pulse_ref must be ContentRef or None")
        object.__setattr__(self, "_pipeline_result", pipeline_result)
        object.__setattr__(self, "_repository_borrow", repository_borrow)
        object.__setattr__(self, "_compiled_pulse_ref", compiled_pulse_ref)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PendingCaptureArtifact is immutable")

    @property
    def pipeline_result(self) -> PipelineResult | TriggeredPipelineResult:
        return self._pipeline_result


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
    reference = repository._store_authority.put_blob(pulse_payload)
    if reference.digest != spec.pulse_binding.compiled_artifact.fingerprint:
        raise RuntimeError("compiled-pulse CAS identity differs from pulse owner")
    return reference


def compile_capture_artifact_pipeline(
    spec: MinimalPipelineSpec | TriggeredCaptureSpec,
    repository: CaptureRepository,
    *,
    preview: CapturePreviewPort | None = None,
    exact_preview: ExactDatasetPreviewPort | None = None,
    settle_exact_preview: bool = True,
) -> RunPlan:
    """Add one post-safety CaptureArtifact commit to the exact pipeline."""

    try:
        if type(repository) is not CaptureRepository:
            raise TypeError("repository must be CaptureRepository")
        capture_spec = spec.capture if isinstance(spec, TriggeredCaptureSpec) else spec
        if not isinstance(capture_spec, MinimalPipelineSpec):
            raise TypeError("capture artifact pipeline requires MinimalPipelineSpec")
        if type(settle_exact_preview) is not bool:
            raise TypeError("settle_exact_preview must be bool")
        if isinstance(spec, TriggeredCaptureSpec) and not settle_exact_preview:
            raise ValueError(
                "triggered capture owns its complete exact-preview cleanup boundary"
            )
        repository._require_active()
    except BaseException as error:
        notify_preview_failure(preview, error)
        notify_preview_failure(exact_preview, error)
        raise
    base = (
        compile_triggered_pipeline(
            spec,
            preview=preview,
            exact_preview=exact_preview,
        )
        if isinstance(spec, TriggeredCaptureSpec)
        else compile_pipeline(
            spec,
            preview=preview,
            exact_preview=exact_preview,
            settle_exact_preview=settle_exact_preview,
        )
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
            notify_preview_failure(preview, error)
            notify_preview_failure(exact_preview, error)
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
    ) -> PendingCaptureArtifact:
        base_prepared, borrow, pulse_ref = prepared
        borrow.require_active()
        return PendingCaptureArtifact(
            base_execute(context, base_prepared),
            borrow,
            pulse_ref,
        )

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
        if borrow is not None and (primary is not None or report.errors):
            borrow.close()
        return report

    def finalize(
        context: PostSafetyContext,
        result: PendingCaptureArtifact,
    ) -> CaptureArtifactRef:
        if not isinstance(result, PendingCaptureArtifact):
            raise TypeError("capture finalize requires PendingCaptureArtifact")
        base_result = result._pipeline_result
        borrow = result._repository_borrow
        pulse_ref = result._compiled_pulse_ref
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
        result: PendingCaptureArtifact,
    ) -> None:
        """Release the sink hold when RunController skips finalize."""

        if not isinstance(result, PendingCaptureArtifact):
            raise TypeError("capture disposal requires PendingCaptureArtifact")
        base_result = result._pipeline_result
        borrow = result._repository_borrow
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
    provenance: DatasetSealProvenance,
    terminal: CaptureTerminalAck,
    camera_provenance: CameraCaptureProvenance,
    camera_capability_evidence: CameraCapabilityEvidence,
    camera_arm_spec: FrozenCaptureSpec,
    pulse_evidence: PulseCaptureEvidence | None,
) -> bytes:
    absent = pulse_evidence is None
    if absent != (compiled_pulse_ref is None):
        raise ValueError(
            "pulse evidence and compiled-pulse blob presence differ"
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
            "pulse_evidence": pulse_capture_evidence_to_tree(
                pulse_evidence
            ),
        }
    )


__all__ = [
    "AdmittedCapture",
    "CAPTURE_ARTIFACT_SCHEMA",
    "CaptureArtifact",
    "CaptureRepository",
    "compile_capture_artifact_pipeline",
    "PendingCaptureArtifact",
]
