"""Committed-capture occupancy storage, admission, and one flat CPU RunPlan."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading

import numpy as np

from zlc_data import (
    AxisId,
    ComponentValidity,
    DataBlock,
    DatasetSchema,
    StreamGenerationId,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
)
from zlc_neutral_atom.logic_nodes.camera_capture.artifact import (
    AdmittedCapture,
    CaptureRepository,
)
from zlc_neutral_atom.logic_nodes.camera_capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
    publish_manifest_with_visibility_reconciliation,
)
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_storage import (
    ContentAddressedStore,
    ContentRef,
    ContentStoreAuthority,
    RepositoryRootLease,
    RepositoryRootLeaseBorrow,
    canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    positive_real,
    sha256_digest,
)

from zlc_neutral_atom.logic_nodes.calibration.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.calibration.reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_from_tree,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.logic_nodes.calibration.repository import CalibrationRepository
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from .processor import (
    OCCUPANCY_COUNTS_BLOCK_ID,
    OCCUPANCY_OCCUPIED_BLOCK_ID,
    OccupancyArtifact,
    ResolvedOccupancy,
    _CommittedOccupancyBinding,
    _ResolvedCommittedOccupancy,
    _RESOLVED_OCCUPANCY_TOKEN,
    _analyze_committed_occupancy_resolved,
    _occupancy_generation_for_run,
    _require_committed_occupancy_context,
    _require_occupancy_output_schemas,
    _resolve_committed_occupancy_structure,
)
from .reference import (
    OCCUPANCY_ARTIFACT_NAMESPACE,
    OccupancyArtifactRef,
)
from zlc_neutral_atom.runtime.resources import (
    acquire_repository_borrows,
    release_repository_borrows,
)


OCCUPANCY_ARTIFACT_FORMAT = "zlc_neutral_atom.logic_nodes.occupancy.storage"
OCCUPANCY_MANIFEST_FORMAT = "zlc_neutral_atom.logic_nodes.occupancy.manifest"
_OCCUPANCY_ARTIFACT_KIND = "occupancy"
_ARTIFACT_FIELDS = frozenset(
    {
        "format",
        "source_capture_ref",
        "calibration_ref",
        "readout_event_axis_id",
        "model_kind",
        "generation",
        "counts_schema",
        "occupied_schema",
        "counts_blob",
        "occupied_blob",
        "validity_blob",
    }
)
_MANIFEST_FIELDS = frozenset(
    {"format", "repository_id", "metadata_blob"}
)


@dataclass(frozen=True, slots=True)
class _StoredOccupancy:
    source_capture_ref: CaptureArtifactRef
    calibration_reference: CalibrationArtifactRef
    readout_event_axis_id: AxisId
    model_kind: ReadoutModelKind
    generation: StreamGenerationId
    counts_schema: DatasetSchema
    occupied_schema: DatasetSchema
    counts_blob: ContentRef
    occupied_blob: ContentRef
    validity_blob: ContentRef

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.calibration_reference, CalibrationArtifactRef):
            raise TypeError("calibration_reference must be CalibrationArtifactRef")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        _require_occupancy_output_schemas(
            self.counts_schema,
            self.occupied_schema,
        )
        refs = (self.counts_blob, self.occupied_blob, self.validity_blob)
        if any(not isinstance(reference, ContentRef) for reference in refs):
            raise TypeError("occupancy blobs must be ContentRef")
        elements = math.prod(self.counts_schema.physical_shape)
        if (
            self.counts_blob.size != 8 * elements
            or self.occupied_blob.size != elements
            or self.validity_blob.size != elements
        ):
            raise ValueError("occupancy blob sizes differ from the stored schemas")


def _artifact_to_tree(value: _StoredOccupancy) -> dict[str, object]:
    if not isinstance(value, _StoredOccupancy):
        raise TypeError("value must be stored occupancy metadata")
    return {
        "format": OCCUPANCY_ARTIFACT_FORMAT,
        "source_capture_ref": capture_artifact_ref_to_tree(
            value.source_capture_ref
        ),
        "calibration_ref": calibration_artifact_ref_to_tree(
            value.calibration_reference
        ),
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "model_kind": value.model_kind.value,
        "generation": value.generation.value,
        "counts_schema": dataset_schema_to_tree(value.counts_schema),
        "occupied_schema": dataset_schema_to_tree(value.occupied_schema),
        "counts_blob": content_ref_to_tree(value.counts_blob),
        "occupied_blob": content_ref_to_tree(value.occupied_blob),
        "validity_blob": content_ref_to_tree(value.validity_blob),
    }


def _artifact_from_tree(tree: object) -> _StoredOccupancy:
    data = exact_mapping(
        tree,
        _ARTIFACT_FIELDS,
        OCCUPANCY_ARTIFACT_FORMAT,
        discriminator="format",
    )
    return _StoredOccupancy(
        capture_artifact_ref_from_tree(data["source_capture_ref"]),
        calibration_artifact_ref_from_tree(data["calibration_ref"]),
        AxisId(
            canonical_text(data["readout_event_axis_id"], "readout_event_axis_id")
        ),
        ReadoutModelKind(
            canonical_text(data["model_kind"], "model_kind")
        ),
        StreamGenerationId(canonical_text(data["generation"], "generation")),
        dataset_schema_from_tree(data["counts_schema"]),
        dataset_schema_from_tree(data["occupied_schema"]),
        content_ref_from_tree(data["counts_blob"]),
        content_ref_from_tree(data["occupied_blob"]),
        content_ref_from_tree(data["validity_blob"]),
    )


def _encode_artifact(value: _StoredOccupancy) -> bytes:
    return encode(_artifact_to_tree(value))


def _decode_artifact(payload: bytes | bytearray | memoryview) -> _StoredOccupancy:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("occupancy artifact payload must be bytes-like")
    raw = bytes(payload)
    value = _artifact_from_tree(decode(raw))
    if _encode_artifact(value) != raw:
        raise ValueError("occupancy artifact is typed but non-canonical")
    return value


def _array_payload(
    value: np.ndarray,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...],
    field: str,
) -> memoryview:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(dtype)
        or array.shape != shape
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"{field} has a non-canonical dtype, shape, or layout")
    if array.dtype == np.dtype("<f8") and not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite")
    return memoryview(array).cast("B")


def _decode_array_payload(
    payload: bytes | bytearray | memoryview,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} payload must be bytes-like")
    target = np.dtype(dtype)
    expected = math.prod(shape) * target.itemsize
    raw = bytes(payload)
    if len(raw) != expected:
        raise ValueError(f"{field} payload size differs from its admitted schema")
    if target == np.dtype(bool):
        octets = np.frombuffer(raw, dtype="uint8")
        if np.any(octets > 1):
            raise ValueError(f"{field} payload is not canonical boolean")
        array = octets.view("bool")
    else:
        array = np.frombuffer(raw, dtype=target)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field} payload must be finite")
    return array.reshape(shape)


def _manifest_payload(repository_id: str, metadata_blob: ContentRef) -> bytes:
    return encode(
        {
            "format": OCCUPANCY_MANIFEST_FORMAT,
            "repository_id": repository_id,
            "metadata_blob": content_ref_to_tree(metadata_blob),
        }
    )


def _decode_manifest(payload: bytes) -> tuple[str, ContentRef]:
    if not isinstance(payload, bytes):
        raise TypeError("occupancy manifest payload must be bytes")
    tree = exact_mapping(
        decode(payload),
        _MANIFEST_FIELDS,
        OCCUPANCY_MANIFEST_FORMAT,
        discriminator="format",
    )
    repository_id = canonical_text(tree["repository_id"], "repository_id")
    metadata_blob = content_ref_from_tree(tree["metadata_blob"])
    if _manifest_payload(repository_id, metadata_blob) != payload:
        raise ValueError("occupancy manifest is not canonical current format")
    return repository_id, metadata_blob


def _target(repository_id: str, reference: OccupancyArtifactRef) -> CommitTarget:
    return CommitTarget(
        repository_id,
        _OCCUPANCY_ARTIFACT_KIND,
        OCCUPANCY_MANIFEST_FORMAT,
        reference.target_ref,
        reference.manifest_digest,
    )


def _commit_id(run_id: str, manifest_digest: str) -> str:
    return f"occupancy-final-{run_id}-{manifest_digest}"


class OccupancyRepository:
    """Content-addressed occupancy store with FINAL visibility authority."""

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-occupancy",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
        self._lock = threading.RLock()
        self._closed = False
        self._root_lease = RepositoryRootLease(self.root)
        journal = None
        try:
            self._store = ContentAddressedStore(self.root / "content")
            self._store_authority = self._store.authority()
            journal = PersistentCommitJournal(
                self.root / "occupancy-commit.journal",
                self.repository_id,
            )
            self._coordinator: RepositoryCommitCoordinator[
                OccupancyArtifactRef
            ] = RepositoryCommitCoordinator(
                journal,
                self._recover,
                root_lease=self._root_lease,
            )
        except BaseException:
            if journal is not None:
                journal.close()
            self._root_lease.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("occupancy repository is closed")
        self._root_lease.require_active()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._coordinator.close()
            self._closed = True

    def __enter__(self) -> "OccupancyRepository":
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

    def _content_authority(self) -> ContentStoreAuthority:
        with self._lock:
            self._require_open()
            return self._store_authority

    def _validate_reference(self, reference: OccupancyArtifactRef) -> None:
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("OccupancyArtifactRef belongs to another repository")

    def _require_final_commit(
        self,
        reference: OccupancyArtifactRef,
    ) -> CommitIntent:
        with self._lock:
            self._require_open()
            self._validate_reference(reference)
            target = _target(self.repository_id, reference)
            matching = self._coordinator.committed_for(target)
            if not matching:
                raise PermissionError("occupancy lacks FINAL commit authority")
            if len(matching) != 1:
                raise ValueError("occupancy has multiple FINAL authorities")
            intent = matching[0]
            if intent.commit_id != _commit_id(
                intent.run_id,
                reference.manifest_digest,
            ):
                raise ValueError("occupancy commit identity is inconsistent")
            return intent

    @staticmethod
    def _require_run_generation(
        artifact: OccupancyArtifact | _StoredOccupancy,
        intent: CommitIntent,
    ) -> None:
        if artifact.generation != _occupancy_generation_for_run(intent.run_id):
            raise ValueError("occupancy generation differs from its FINAL Run")

    def _stored(
        self,
        reference: OccupancyArtifactRef,
        *,
        manifest_payload: bytes | None = None,
    ) -> _StoredOccupancy:
        authority = self._content_authority()
        if manifest_payload is None:
            payload = authority.read_manifest(
                OCCUPANCY_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
            )
        else:
            payload = manifest_payload
        repository_id, metadata_ref = _decode_manifest(payload)
        if repository_id != self.repository_id:
            raise ValueError("occupancy manifest belongs to another repository")
        return _decode_artifact(authority.read_blob(metadata_ref))

    @staticmethod
    def _expected_blob_sizes(
        binding: _CommittedOccupancyBinding,
    ) -> tuple[int, int, int]:
        if not isinstance(binding, _CommittedOccupancyBinding):
            raise TypeError("binding must be resolved committed occupancy")
        resolved = binding.resolved_schema
        elements = math.prod(resolved.counts_schema.physical_shape)
        return elements * 8, elements, elements

    @classmethod
    def _validate_stored_dependencies(
        cls,
        stored: _StoredOccupancy,
        source: AdmittedCapture,
        calibration: ResolvedCalibration,
        binding: _CommittedOccupancyBinding,
    ) -> None:
        event_axis_id = binding.readout_event_axis_id
        resolved = binding.resolved_schema
        if stored.source_capture_ref != source.reference or (
            stored.calibration_reference != calibration.reference
        ):
            raise ValueError("occupancy metadata names different dependencies")
        if (
            stored.readout_event_axis_id != event_axis_id
            or stored.model_kind is not resolved.model_kind
        ):
            raise ValueError("occupancy metadata differs from its resolved binding")
        if stored.counts_schema != resolved.counts_schema or (
            stored.occupied_schema != resolved.occupied_schema
        ):
            raise ValueError("occupancy stored schemas differ from resolved inputs")
        if (
            stored.counts_blob.size,
            stored.occupied_blob.size,
            stored.validity_blob.size,
        ) != cls._expected_blob_sizes(binding):
            raise ValueError("occupancy blob sizes differ from the resolved schema")

    def _materialize(
        self,
        stored: _StoredOccupancy,
        source: AdmittedCapture,
        binding: _CommittedOccupancyBinding,
    ) -> OccupancyArtifact:
        resolved = binding.resolved_schema
        authority = self._content_authority()
        shape = resolved.counts_schema.physical_shape
        validity_payload = authority.read_blob(stored.validity_blob)
        validity = ComponentValidity(
            (resolved.selected_model.feature.site_axis.axis_id,),
            _decode_array_payload(
                validity_payload,
                dtype=bool,
                shape=shape,
                field="occupancy validity",
            ),
        )
        del validity_payload
        counts_payload = authority.read_blob(stored.counts_blob)
        counts = DataBlock(
            OCCUPANCY_COUNTS_BLOCK_ID,
            source.artifact.frame_source.revision,
            _decode_array_payload(
                counts_payload,
                dtype="<f8",
                shape=shape,
                field="occupancy counts",
            ),
            validity,
            resolved.counts_schema,
        )
        del counts_payload
        occupied_payload = authority.read_blob(stored.occupied_blob)
        occupied = DataBlock(
            OCCUPANCY_OCCUPIED_BLOCK_ID,
            source.artifact.frame_source.revision,
            _decode_array_payload(
                occupied_payload,
                dtype=bool,
                shape=shape,
                field="occupancy occupied",
            ),
            validity,
            resolved.occupied_schema,
        )
        return OccupancyArtifact(
            stored.source_capture_ref,
            stored.calibration_reference,
            stored.readout_event_axis_id,
            stored.model_kind,
            stored.generation,
            counts,
            occupied,
        )

    def admit(
        self,
        reference: OccupancyArtifactRef,
        capture_repository: CaptureRepository,
        calibration_repository: CalibrationRepository,
    ) -> ResolvedOccupancy:
        if type(capture_repository) is not CaptureRepository:
            raise TypeError("capture_repository must be CaptureRepository")
        if type(calibration_repository) is not CalibrationRepository:
            raise TypeError("calibration_repository must be CalibrationRepository")
        with self._root_lease.borrow() as admission_borrow:
            admission_borrow.require_active()
            with capture_repository._root_lease.borrow() as source_borrow:
                with calibration_repository._root_lease.borrow() as calibration_borrow:
                    source_borrow.require_active()
                    calibration_borrow.require_active()
                    intent = self._require_final_commit(reference)
                    stored = self._stored(reference)
                    self._require_run_generation(stored, intent)
                    calibration = calibration_repository.admit(
                        stored.calibration_reference,
                        capture_repository,
                    )
                    source = capture_repository.admit(stored.source_capture_ref)
                    binding = _resolve_committed_occupancy_structure(
                        source.artifact,
                        calibration,
                        readout_event_axis_id=stored.readout_event_axis_id,
                        model_kind=stored.model_kind,
                    )
                    self._validate_stored_dependencies(
                        stored,
                        source,
                        calibration,
                        binding,
                    )
                    resolved = _require_committed_occupancy_context(
                        source,
                        calibration,
                        binding,
                    )
                    artifact = self._materialize(
                        stored,
                        resolved.source,
                        resolved.binding,
                    )
                    return ResolvedOccupancy._from_admission(
                        _RESOLVED_OCCUPANCY_TOKEN,
                        repository_token=self._root_lease,
                        reference=reference,
                        artifact=artifact,
                        readout_binding=source.artifact.camera_provenance.binding,
                    )

    def has(self, reference: OccupancyArtifactRef) -> bool:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            try:
                self._require_final_commit(reference)
            except PermissionError:
                return False
            return self._content_authority().has_manifest(
                OCCUPANCY_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
            )

    def _stage_result(
        self,
        artifact: OccupancyArtifact,
    ) -> tuple[OccupancyArtifactRef, bytes]:
        if not isinstance(artifact, OccupancyArtifact):
            raise TypeError("artifact must be OccupancyArtifact")
        shape = artifact.counts.schema.physical_shape
        counts_payload = _array_payload(
            artifact.counts.values,
            dtype="<f8",
            shape=shape,
            field="occupancy counts",
        )
        occupied_payload = _array_payload(
            artifact.occupied.values,
            dtype=bool,
            shape=shape,
            field="occupancy occupied",
        )
        validity = artifact.counts.validity
        if not isinstance(validity, ComponentValidity):
            raise TypeError("occupancy artifact requires ComponentValidity")
        validity_payload = _array_payload(
            validity.mask,
            dtype=bool,
            shape=shape,
            field="occupancy validity",
        )
        authority = self._content_authority()
        stored = _StoredOccupancy(
            artifact.source_capture_ref,
            artifact.calibration_reference,
            artifact.readout_event_axis_id,
            artifact.model_kind,
            artifact.generation,
            artifact.counts.schema,
            artifact.occupied.schema,
            authority.identify_blob(counts_payload),
            authority.identify_blob(occupied_payload),
            authority.identify_blob(validity_payload),
        )
        metadata = _encode_artifact(stored)
        if _decode_artifact(metadata) != stored:
            raise ValueError("occupancy metadata failed its canonical round-trip")
        for payload, expected in (
            (counts_payload, stored.counts_blob),
            (occupied_payload, stored.occupied_blob),
            (validity_payload, stored.validity_blob),
        ):
            if authority.put_blob(payload) != expected:
                raise RuntimeError("occupancy blob identity changed while staging")
        metadata_blob = authority.put_blob(metadata)
        manifest = _manifest_payload(self.repository_id, metadata_blob)
        return (
            OccupancyArtifactRef(self.repository_id, sha256_digest(manifest)),
            manifest,
        )

    def _final_commit(
        self,
        context: PostSafetyContext,
        artifact: OccupancyArtifact,
        resolved: _ResolvedCommittedOccupancy,
    ) -> FinalCommit[OccupancyArtifactRef]:
        if not isinstance(context, PostSafetyContext):
            raise TypeError("occupancy commit requires PostSafetyContext")
        if not isinstance(artifact, OccupancyArtifact):
            raise TypeError("artifact must be OccupancyArtifact")
        run_id = context.authorize_commit_preparation()
        if not isinstance(resolved, _ResolvedCommittedOccupancy):
            raise TypeError("resolved must be admitted committed occupancy")
        source = resolved.source
        calibration = resolved.calibration
        event_axis_id = resolved.binding.readout_event_axis_id
        schemas = resolved.binding.resolved_schema
        source._require_authority()
        calibration._require_authority()
        if (
            artifact.source_capture_ref != source.reference
            or artifact.calibration_reference != calibration.reference
            or artifact.readout_event_axis_id != event_axis_id
            or artifact.model_kind is not schemas.model_kind
            or artifact.counts.schema != schemas.counts_schema
            or artifact.occupied.schema != schemas.occupied_schema
            or artifact.generation != _occupancy_generation_for_run(run_id)
        ):
            raise PermissionError("occupancy result differs from its admitted inputs")
        if artifact.counts.revision != source.artifact.frame_source.revision:
            raise ValueError("occupancy revision differs from the admitted source")
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, payload = self._stage_result(artifact)
            if context.authorize_commit_preparation() != run_id:
                raise RuntimeError("occupancy commit subject changed while staging")
            target = _target(self.repository_id, reference)

            def publish() -> PublishedManifest[OccupancyArtifactRef]:
                publish_manifest_with_visibility_reconciliation(
                    self._content_authority(),
                    OCCUPANCY_ARTIFACT_NAMESPACE,
                    payload,
                    expected_digest=reference.manifest_digest,
                )
                return PublishedManifest(
                    reference.target_ref,
                    reference.manifest_digest,
                    reference,
                )

            with self._lock:
                self._require_open()
                operation = self._coordinator.prepare(
                    _commit_id(run_id, reference.manifest_digest),
                    run_id,
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
    ) -> PublishedManifest[OccupancyArtifactRef] | None:
        target = intent.target
        if (
            target.repository_id != self.repository_id
            or target.artifact_kind != _OCCUPANCY_ARTIFACT_KIND
            or target.artifact_format != OCCUPANCY_MANIFEST_FORMAT
        ):
            raise ValueError("commit intent is not an occupancy target")
        reference = OccupancyArtifactRef(
            self.repository_id,
            target.expected_manifest_digest,
        )
        if target.target_ref != reference.target_ref or (
            intent.commit_id
            != _commit_id(intent.run_id, reference.manifest_digest)
        ):
            raise ValueError("occupancy commit identity differs from its target")
        authority = self._content_authority()
        try:
            payload = authority.read_manifest(
                OCCUPANCY_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
            )
        except FileNotFoundError:
            return None
        stored = self._stored(
            reference,
            manifest_payload=payload,
        )
        self._require_run_generation(stored, intent)
        for blob in (
            stored.counts_blob,
            stored.occupied_blob,
            stored.validity_blob,
        ):
            authority.verify_blob(blob)
        if authority.confirm_manifest_durable(
            OCCUPANCY_ARTIFACT_NAMESPACE,
            reference.manifest_digest,
        ) != payload:
            raise RuntimeError("recovery durability check changed occupancy manifest")
        return PublishedManifest(
            reference.target_ref,
            reference.manifest_digest,
            reference,
        )


_PreparedOccupancyAnalysis = tuple[
    _ResolvedCommittedOccupancy,
    tuple[RepositoryRootLeaseBorrow, ...],
]
_ExecutedOccupancyAnalysis = tuple[
    OccupancyArtifact,
    _ResolvedCommittedOccupancy,
    tuple[RepositoryRootLeaseBorrow, ...],
]


def compile_occupancy_artifact_plan(
    source_capture_ref: CaptureArtifactRef,
    capture_repository: CaptureRepository,
    calibration_ref: CalibrationArtifactRef,
    calibration_repository: CalibrationRepository,
    occupancy_repository: OccupancyRepository,
    *,
    expected_readout_binding: ReadoutBindingKey,
    readout_event_axis_id: AxisId,
    model_kind: ReadoutModelKind,
    timeout_seconds: float,
) -> RunPlan:
    """Compile committed raw frames to one FINAL occupancy artifact."""

    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    if type(capture_repository) is not CaptureRepository:
        raise TypeError("capture_repository must be CaptureRepository")
    if not isinstance(calibration_ref, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    if type(calibration_repository) is not CalibrationRepository:
        raise TypeError("calibration_repository must be CalibrationRepository")
    if type(occupancy_repository) is not OccupancyRepository:
        raise TypeError("occupancy_repository must be OccupancyRepository")
    if not isinstance(expected_readout_binding, ReadoutBindingKey):
        raise TypeError("expected_readout_binding must be ReadoutBindingKey")
    if not isinstance(readout_event_axis_id, AxisId):
        raise TypeError("readout_event_axis_id must be AxisId")
    if not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("model_kind must be a concrete ReadoutModelKind")
    timeout = positive_real(timeout_seconds, "timeout_seconds")

    def admit_inputs(
        context: RunContext,
    ) -> tuple[AdmittedCapture, ResolvedCalibration]:
        calibration = calibration_repository.admit(
            calibration_ref,
            capture_repository,
            checkpoint=context.checkpoint,
        )
        source = capture_repository.admit(source_capture_ref)
        if source.artifact.camera_provenance.binding != expected_readout_binding or (
            calibration.artifact.frame_contract.binding != expected_readout_binding
        ):
            raise ValueError("occupancy dependencies differ from the frozen binding")
        return source, calibration

    def admit_dependencies(
        context: RunContext,
    ) -> tuple[AdmittedCapture, ResolvedCalibration, _CommittedOccupancyBinding]:
        source, calibration = admit_inputs(context)
        binding = _resolve_committed_occupancy_structure(
            source.artifact,
            calibration,
            readout_event_axis_id=readout_event_axis_id,
            model_kind=model_kind,
        )
        return source, calibration, binding

    def _preflight_with_borrows(
        context: RunContext,
        borrows: tuple[RepositoryRootLeaseBorrow, ...],
    ) -> _PreparedOccupancyAnalysis:
        context.checkpoint()
        source, calibration, binding = admit_dependencies(context)
        resolved = _require_committed_occupancy_context(
            source,
            calibration,
            binding,
            checkpoint=context.checkpoint,
        )
        context.checkpoint()
        return resolved, borrows

    def preflight(context: RunContext) -> _PreparedOccupancyAnalysis:
        borrows = acquire_repository_borrows(
            capture_repository._root_lease,
            calibration_repository._root_lease,
            occupancy_repository._root_lease,
        )
        try:
            return _preflight_with_borrows(context, borrows)
        except BaseException as primary:
            try:
                release_repository_borrows(borrows)
            except BaseException as close_error:
                record_secondary_failure(
                    primary,
                    "repository borrow release also failed",
                    close_error,
                )
            raise

    def execute(
        context: RunContext,
        prepared: _PreparedOccupancyAnalysis,
    ) -> _ExecutedOccupancyAnalysis:
        resolved, borrows = prepared
        return (
            _analyze_committed_occupancy_resolved(
                resolved,
                run_id=context.run_id.value,
                checkpoint=context.checkpoint,
            ),
            resolved,
            borrows,
        )

    def cleanup(
        _context: RunContext,
        prepared: _PreparedOccupancyAnalysis | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is not None and primary is not None:
            _resolved, borrows = prepared
            release_repository_borrows(borrows)
        return CleanupReport()

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedOccupancyAnalysis,
    ) -> OccupancyArtifactRef:
        artifact, resolved, borrows = executed
        try:
            for borrow in borrows:
                borrow.require_active()
            operation = occupancy_repository._final_commit(
                context,
                artifact,
                resolved,
            )
            return context.commit_final(operation)
        finally:
            release_repository_borrows(borrows)

    def dispose_unfinalized(
        executed: _ExecutedOccupancyAnalysis,
    ) -> None:
        _artifact, _resolved, borrows = executed
        release_repository_borrows(borrows)

    return RunPlan(
        name="classify committed camera capture occupancy",
        resource_claims=(),
        bound_devices=(),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        timeout_seconds=timeout,
        requires_final_commit=True,
        dispose_unfinalized=dispose_unfinalized,
    )


__all__ = [
    "OCCUPANCY_ARTIFACT_FORMAT",
    "OCCUPANCY_MANIFEST_FORMAT",
    "OccupancyRepository",
    "compile_occupancy_artifact_plan",
]
