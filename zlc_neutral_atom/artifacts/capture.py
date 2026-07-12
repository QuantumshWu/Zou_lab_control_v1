"""Current CaptureArtifact schema, codec, and crash-safe repository.

CaptureArtifact is the durable boundary between acquisition and every offline
consumer.  It stores the full multidimensional DataBlock, ordered camera
metadata, exact-stream provenance, and terminal capture evidence.  It does not
store a driver, Port, RunPlan, or mutable builder alias.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from zlc_data import DataBlock, StreamGenerationId, decode_data_block, encode_data_block
from zlc_storage import (
    ContentAddressedStore,
    ContentRef,
    decode,
    encode,
    sha256_digest,
)

from zlc_neutral_atom.acquisition import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    camera_frame_metadata_from_tree,
    camera_frame_metadata_to_tree,
)
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitRecovery,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
)
from zlc_neutral_atom.runtime.capture import CaptureTerminalAck
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetSealProvenance,
)
from zlc_neutral_atom.runtime.pipeline import (
    MinimalPipelineSpec,
    PipelineResult,
    compile_pipeline,
)
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId, TraceBinding


CAPTURE_ARTIFACT_SCHEMA = "zlc_neutral_atom.CaptureArtifact/v1"
_CAPTURE_METADATA_SCHEMA = "zlc_neutral_atom.CameraFrameMetadataSequence/v1"
_CAPTURE_NAMESPACE = "capture"


def _canonical_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CaptureArtifactRef:
    repository_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _canonical_text(self.repository_id, "repository_id")
        _sha256(self.manifest_digest, "manifest_digest")

    @property
    def target_ref(self) -> str:
        return f"{_CAPTURE_NAMESPACE}/{self.manifest_digest}"


@dataclass(frozen=True)
class CaptureArtifact:
    ref: CaptureArtifactRef
    block: DataBlock
    event_metadata: tuple[CameraFrameMetadata, ...]
    coverage: DatasetCoverage
    provenance: DatasetSealProvenance
    terminal: CaptureTerminalAck
    aggregate_peak_bytes: int
    memory_profile_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, CaptureArtifactRef):
            raise TypeError("ref must be CaptureArtifactRef")
        if not isinstance(self.block, DataBlock):
            raise TypeError("block must be DataBlock")
        metadata = tuple(self.event_metadata)
        if any(not isinstance(item, CameraFrameMetadata) for item in metadata):
            raise TypeError("event_metadata must contain CameraFrameMetadata")
        object.__setattr__(self, "event_metadata", metadata)
        if not isinstance(self.coverage, DatasetCoverage) or not self.coverage.complete:
            raise ValueError("CaptureArtifact requires complete dataset coverage")
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if not isinstance(self.terminal, CaptureTerminalAck):
            raise TypeError("terminal must be CaptureTerminalAck")
        peak = _integer(self.aggregate_peak_bytes, "aggregate_peak_bytes")
        if peak == 0:
            raise ValueError("aggregate_peak_bytes must be positive")
        object.__setattr__(self, "aggregate_peak_bytes", peak)
        _sha256(self.memory_profile_fingerprint, "memory_profile_fingerprint")
        count = len(metadata)
        physical_cells = (
            self.block.schema.repeat_axis.size
            * self.block.schema.point_layout.storage_size
        )
        if self.coverage.total_cells != physical_cells or count != physical_cells:
            raise ValueError("capture metadata cardinality differs from DataBlock cells")
        if self.provenance.end_sequence - self.provenance.start_sequence != count:
            raise ValueError("capture provenance interval differs from metadata cardinality")
        if tuple(item.source_ordinal for item in metadata) != tuple(range(count)):
            raise ValueError("capture source ordinals are not contiguous from zero")
        metadata_contract = CameraFrameMetadataContract()
        if self.provenance.metadata_contract_fingerprint != metadata_contract.fingerprint:
            raise ValueError("capture metadata contract is not the current camera contract")
        hasher = hashlib.sha256()
        hasher.update(metadata_contract.fingerprint.encode("ascii"))
        for item in metadata:
            metadata_contract.validate(item)
            hasher.update(metadata_contract.digest(item).encode("ascii"))
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


class CaptureRepository:
    """One neutral-domain repository with a durable commit-intent gate."""

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-capture",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = _canonical_text(repository_id, "repository_id")
        self._store = ContentAddressedStore(self.root / "content")
        self._journal = PersistentCommitJournal(
            self.root / "capture-commit.journal",
            self.repository_id,
        )
        self._coordinator: RepositoryCommitCoordinator[CaptureArtifactRef] = (
            RepositoryCommitCoordinator(self._journal, self._recover)
        )

    @property
    def startup_reconciliations(self):
        return self._coordinator.startup_reconciliations

    def load(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        self._validate_ref(reference)
        manifest_payload = self._store.read_manifest(
            _CAPTURE_NAMESPACE,
            reference.manifest_digest,
        )
        tree = decode(manifest_payload)
        data = _exact_map(
            tree,
            {
                "schema",
                "repository_id",
                "data_block_blob",
                "metadata_blob",
                "coverage",
                "provenance",
                "terminal",
                "aggregate_peak_bytes",
                "memory_profile_fingerprint",
            },
            CAPTURE_ARTIFACT_SCHEMA,
        )
        if data["repository_id"] != self.repository_id:
            raise ValueError("CaptureArtifact belongs to another repository")
        block_ref = _content_ref_from_tree(data["data_block_blob"])
        metadata_ref = _content_ref_from_tree(data["metadata_blob"])
        block = decode_data_block(self._store.read_blob(block_ref))
        metadata_payload = self._store.read_blob(metadata_ref)
        metadata_tree = _exact_map(
            decode(metadata_payload),
            {"schema", "items"},
            _CAPTURE_METADATA_SCHEMA,
        )
        items = metadata_tree["items"]
        if not isinstance(items, list):
            raise ValueError("camera metadata items must be a list")
        artifact = CaptureArtifact(
            reference,
            block,
            tuple(camera_frame_metadata_from_tree(item) for item in items),
            _coverage_from_tree(data["coverage"]),
            _provenance_from_tree(data["provenance"]),
            _terminal_from_tree(data["terminal"]),
            data["aggregate_peak_bytes"],
            data["memory_profile_fingerprint"],
        )
        # Enforce one canonical current representation, not merely a decodable one.
        rebuilt_payload = _manifest_payload(artifact, block_ref, metadata_ref)
        if (
            sha256_digest(rebuilt_payload) != reference.manifest_digest
            or rebuilt_payload != manifest_payload
        ):
            raise ValueError("CaptureArtifact manifest is not canonical")
        return artifact

    def final_commit(
        self,
        context: PostSafetyContext,
        result: PipelineResult,
    ) -> FinalCommit[CaptureArtifactRef]:
        if not isinstance(context, PostSafetyContext):
            raise TypeError("final_commit requires PostSafetyContext")
        if not isinstance(result, PipelineResult):
            raise TypeError("final_commit requires PipelineResult")
        reference, manifest_payload = self._stage_pipeline_result(result)
        target = CommitTarget(
            self.repository_id,
            "capture",
            CAPTURE_ARTIFACT_SCHEMA,
            reference.target_ref,
            reference.manifest_digest,
        )

        def publish() -> PublishedManifest[CaptureArtifactRef]:
            stored = self._store.publish_manifest(
                _CAPTURE_NAMESPACE,
                manifest_payload,
                expected_digest=reference.manifest_digest,
            )
            if stored.content.digest != reference.manifest_digest:
                raise RuntimeError("published capture manifest digest changed")
            return PublishedManifest(
                reference.target_ref,
                reference.manifest_digest,
                reference,
            )

        commit_id = (
            f"capture-{context.run_id.value}-{reference.manifest_digest[:20]}"
        )
        return FinalCommit(
            commit_id,
            context.safety_bundle_id,
            self._coordinator.prepare(target, publish),
        )

    def _recover(self, intent: CommitIntent) -> CommitRecovery[CaptureArtifactRef]:
        target = intent.target
        prefix = f"{_CAPTURE_NAMESPACE}/"
        if (
            target.repository_id != self.repository_id
            or target.artifact_kind != "capture"
            or target.schema_version != CAPTURE_ARTIFACT_SCHEMA
            or not target.target_ref.startswith(prefix)
        ):
            raise ValueError("commit intent is not a CaptureArtifact target")
        digest = _sha256(target.target_ref[len(prefix) :], "target manifest digest")
        if digest != target.expected_manifest_digest:
            raise ValueError("capture commit target ref and digest differ")
        if not self._store.has_manifest(_CAPTURE_NAMESPACE, digest):
            return CommitRecovery(False)
        reference = CaptureArtifactRef(self.repository_id, digest)
        self.load(reference)
        return CommitRecovery(
            True,
            PublishedManifest(reference.target_ref, digest, reference),
        )

    def _stage_pipeline_result(
        self,
        result: PipelineResult,
    ) -> tuple[CaptureArtifactRef, bytes]:
        provisional = CaptureArtifact(
            CaptureArtifactRef(self.repository_id, "0" * 64),
            result.dataset.block,
            tuple(result.dataset.event_metadata),
            result.dataset.coverage,
            result.dataset.provenance,
            result.capture_terminal,
            result.aggregate_peak_bytes,
            result.memory_profile_fingerprint,
        )
        return self._stage_manifest(provisional)

    def _stage_manifest(
        self,
        artifact: CaptureArtifact,
    ) -> tuple[CaptureArtifactRef, bytes]:
        block_ref = self._store.put_blob(encode_data_block(artifact.block))
        metadata_payload = encode(
            {
                "schema": _CAPTURE_METADATA_SCHEMA,
                "items": [
                    camera_frame_metadata_to_tree(item)
                    for item in artifact.event_metadata
                ],
            }
        )
        metadata_ref = self._store.put_blob(metadata_payload)
        manifest_payload = _manifest_payload(artifact, block_ref, metadata_ref)
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


def compile_capture_artifact_pipeline(
    spec: MinimalPipelineSpec,
    repository: CaptureRepository,
) -> RunPlan:
    """Add one post-safety CaptureArtifact commit to the exact pipeline."""

    if not isinstance(repository, CaptureRepository):
        raise TypeError("repository must be CaptureRepository")
    base = compile_pipeline(spec)

    def finalize(
        context: PostSafetyContext,
        result: PipelineResult,
    ) -> CaptureArtifactRef:
        finalized = base.finalize(context, result)
        if not isinstance(finalized, PipelineResult):
            raise TypeError("base exact pipeline changed its result contract")
        return context.commit_final(repository.final_commit(context, finalized))

    return RunPlan(
        name=base.name,
        mode=base.mode,
        resource_claims=base.resource_claims,
        hazard_claims=base.hazard_claims,
        bound_devices=base.bound_devices,
        preflight=base.preflight,
        execute=base.execute,
        cleanup=base.cleanup,
        finalize=finalize,
        interrupt_operations=base.interrupt_operations,
        timeout_seconds=base.timeout_seconds,
        requires_final_commit=True,
    )


def _content_ref_to_tree(reference: ContentRef) -> dict[str, object]:
    return {"digest": reference.digest, "size": reference.size}


def _content_ref_from_tree(tree: object) -> ContentRef:
    if not isinstance(tree, dict) or set(tree) != {"digest", "size"}:
        raise ValueError("content reference has an unknown field set")
    return ContentRef(tree["digest"], tree["size"])


def _manifest_payload(
    artifact: CaptureArtifact,
    block_ref: ContentRef,
    metadata_ref: ContentRef,
) -> bytes:
    return encode(
        {
            "schema": CAPTURE_ARTIFACT_SCHEMA,
            "repository_id": artifact.ref.repository_id,
            "data_block_blob": _content_ref_to_tree(block_ref),
            "metadata_blob": _content_ref_to_tree(metadata_ref),
            "coverage": _coverage_to_tree(artifact.coverage),
            "provenance": _provenance_to_tree(artifact.provenance),
            "terminal": _terminal_to_tree(artifact.terminal),
            "aggregate_peak_bytes": artifact.aggregate_peak_bytes,
            "memory_profile_fingerprint": artifact.memory_profile_fingerprint,
        }
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
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("dataset provenance has an unknown field set")
    trace = tree["trace_binding"]
    if not isinstance(trace, dict) or set(trace) != {"run_id", "source_id"}:
        raise ValueError("trace binding has an unknown field set")
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


def _exact_map(tree: object, fields: set[str], schema: str) -> dict:
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError(f"{schema} has an unknown field set")
    if tree["schema"] != schema:
        raise ValueError(f"expected {schema}, got {tree['schema']!r}")
    return tree


__all__ = [
    "CAPTURE_ARTIFACT_SCHEMA",
    "CaptureArtifact",
    "CaptureArtifactRef",
    "CaptureRepository",
    "compile_capture_artifact_pipeline",
]
