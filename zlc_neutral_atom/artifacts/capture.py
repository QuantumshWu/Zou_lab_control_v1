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
from zlc_neutral_atom.timing import (
    CompiledCaptureCellPlan,
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
    decode_compiled_capture_cell_plan,
    encode_compiled_capture_cell_plan,
    PulseTerminalAck,
    pulse_terminal_ack_from_tree,
    pulse_terminal_ack_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)


CAPTURE_ARTIFACT_SCHEMA = "zlc_neutral_atom.CaptureArtifact/v3"
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
class PulseCaptureLineage:
    compiled_artifact: CompiledPulseArtifact
    trigger_channel: str
    terminal: PulseTerminalAck
    cell_plan: CompiledCaptureCellPlan

    def __post_init__(self) -> None:
        if not isinstance(self.compiled_artifact, CompiledPulseArtifact):
            raise TypeError("compiled_artifact must be CompiledPulseArtifact")
        _canonical_text(self.trigger_channel, "trigger_channel")
        if not isinstance(self.terminal, PulseTerminalAck):
            raise TypeError("terminal must be PulseTerminalAck")
        if not isinstance(self.cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        if not self.terminal.logical_done:
            raise ValueError("pulse lineage requires logical terminal")
        if self.terminal.artifact_digest != self.compiled_artifact.fingerprint:
            raise ValueError("pulse terminal belongs to another compiled artifact")
        if (
            self.cell_plan.compiled_pulse_artifact_digest
            != self.compiled_artifact.fingerprint
        ):
            raise ValueError("capture cell plan belongs to another compiled artifact")
        if self.cell_plan.execution_form is not self.compiled_artifact.execution_form:
            raise ValueError("capture cell plan execution form differs from lineage")
        if self.cell_plan.trigger_channel != self.trigger_channel:
            raise ValueError("capture cell plan trigger channel differs from lineage")
        counts = dict(self.terminal.completed_schedule_trigger_counts)
        if self.trigger_channel not in counts:
            raise ValueError("pulse terminal omits the capture trigger channel")
        if counts[self.trigger_channel] != self.cell_plan.total_events:
            raise ValueError("pulse terminal count differs from capture cell plan")

    @property
    def compiled_artifact_digest(self) -> str:
        return self.compiled_artifact.fingerprint

    @property
    def source_document_digest(self) -> str:
        return self.compiled_artifact.source_document_digest

    @property
    def execution_form(self) -> PulseExecutionForm:
        return self.compiled_artifact.execution_form

    @property
    def expected_trigger_count(self) -> int:
        return self.cell_plan.total_events


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
    pulse_lineage: PulseCaptureLineage | None = None

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
        if self.pulse_lineage is not None and not isinstance(
            self.pulse_lineage,
            PulseCaptureLineage,
        ):
            raise TypeError("pulse_lineage must be PulseCaptureLineage or None")
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
        if (
            self.pulse_lineage is not None
            and self.pulse_lineage.expected_trigger_count != count
        ):
            raise ValueError("pulse trigger count differs from persisted capture")
        if self.pulse_lineage is not None:
            plan = self.pulse_lineage.cell_plan
            plan.validate_against(
                self.pulse_lineage.compiled_artifact,
                self.block.schema,
            )
            if plan.cell_permutation_digest != self.provenance.join_plan_digest:
                raise ValueError("capture cell plan differs from sealed cell permutation")
            if plan.total_events != count:
                raise ValueError("capture cell plan count differs from persisted capture")


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
                "compiled_pulse_blob",
                "cell_plan_blob",
                "coverage",
                "provenance",
                "terminal",
                "aggregate_peak_bytes",
                "memory_profile_fingerprint",
                "pulse_lineage",
            },
            CAPTURE_ARTIFACT_SCHEMA,
        )
        if data["repository_id"] != self.repository_id:
            raise ValueError("CaptureArtifact belongs to another repository")
        block_ref = _content_ref_from_tree(data["data_block_blob"])
        metadata_ref = _content_ref_from_tree(data["metadata_blob"])
        pulse_ref = (
            None
            if data["compiled_pulse_blob"] is None
            else _content_ref_from_tree(data["compiled_pulse_blob"])
        )
        plan_ref = (
            None
            if data["cell_plan_blob"] is None
            else _content_ref_from_tree(data["cell_plan_blob"])
        )
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
        cell_plan = (
            None
            if plan_ref is None
            else decode_compiled_capture_cell_plan(self._store.read_blob(plan_ref))
        )
        compiled_pulse = (
            None
            if pulse_ref is None
            else decode_compiled_pulse_artifact(self._store.read_blob(pulse_ref))
        )
        if compiled_pulse is not None and pulse_ref is not None:
            if pulse_ref.digest != compiled_pulse.fingerprint:
                raise ValueError(
                    "compiled-pulse blob digest differs from artifact fingerprint"
                )
        if cell_plan is not None and plan_ref is not None:
            if plan_ref.digest != cell_plan.fingerprint:
                raise ValueError("cell-plan blob digest differs from plan fingerprint")
        artifact = CaptureArtifact(
            reference,
            block,
            tuple(camera_frame_metadata_from_tree(item) for item in items),
            _coverage_from_tree(data["coverage"]),
            _provenance_from_tree(data["provenance"]),
            _terminal_from_tree(data["terminal"]),
            data["aggregate_peak_bytes"],
            data["memory_profile_fingerprint"],
            _pulse_lineage_from_tree(
                data["pulse_lineage"],
                compiled_pulse,
                cell_plan,
            ),
        )
        # Enforce one canonical current representation, not merely a decodable one.
        rebuilt_payload = _manifest_payload(
            artifact,
            block_ref,
            metadata_ref,
            pulse_ref,
            plan_ref,
        )
        if (
            sha256_digest(rebuilt_payload) != reference.manifest_digest
            or rebuilt_payload != manifest_payload
        ):
            raise ValueError("CaptureArtifact manifest is not canonical")
        return artifact

    def final_commit(
        self,
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
    ) -> FinalCommit[CaptureArtifactRef]:
        if not isinstance(context, PostSafetyContext):
            raise TypeError("final_commit requires PostSafetyContext")
        if not isinstance(result, (PipelineResult, TriggeredPipelineResult)):
            raise TypeError("final_commit requires an exact pipeline result")
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
        result: PipelineResult | TriggeredPipelineResult,
    ) -> tuple[CaptureArtifactRef, bytes]:
        lineage = None
        if isinstance(result, TriggeredPipelineResult):
            lineage = PulseCaptureLineage(
                result.compiled_artifact,
                result.trigger_channel,
                result.pulse_terminal,
                result.cell_plan,
            )
            base = result.capture
        else:
            base = result
        provisional = CaptureArtifact(
            CaptureArtifactRef(self.repository_id, "0" * 64),
            base.dataset.block,
            tuple(base.dataset.event_metadata),
            base.dataset.coverage,
            base.dataset.provenance,
            base.capture_terminal,
            base.aggregate_peak_bytes,
            base.memory_profile_fingerprint,
            lineage,
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
        pulse_ref = (
            None
            if artifact.pulse_lineage is None
            else self._store.put_blob(
                encode_compiled_pulse_artifact(
                    artifact.pulse_lineage.compiled_artifact
                )
            )
        )
        plan_ref = (
            None
            if artifact.pulse_lineage is None
            else self._store.put_blob(
                encode_compiled_capture_cell_plan(artifact.pulse_lineage.cell_plan)
            )
        )
        manifest_payload = _manifest_payload(
            artifact,
            block_ref,
            metadata_ref,
            pulse_ref,
            plan_ref,
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


def compile_capture_artifact_pipeline(
    spec: MinimalPipelineSpec | TriggeredCaptureSpec,
    repository: CaptureRepository,
) -> RunPlan:
    """Add one post-safety CaptureArtifact commit to the exact pipeline."""

    if not isinstance(repository, CaptureRepository):
        raise TypeError("repository must be CaptureRepository")
    base = (
        compile_triggered_pipeline(spec)
        if isinstance(spec, TriggeredCaptureSpec)
        else compile_pipeline(spec)
    )

    def finalize(
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
    ) -> CaptureArtifactRef:
        finalized = base.finalize(context, result)
        if not isinstance(finalized, (PipelineResult, TriggeredPipelineResult)):
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
            "data_block_blob": _content_ref_to_tree(block_ref),
            "metadata_blob": _content_ref_to_tree(metadata_ref),
            "compiled_pulse_blob": (
                None
                if compiled_pulse_ref is None
                else _content_ref_to_tree(compiled_pulse_ref)
            ),
            "cell_plan_blob": (
                None
                if cell_plan_ref is None
                else _content_ref_to_tree(cell_plan_ref)
            ),
            "coverage": _coverage_to_tree(artifact.coverage),
            "provenance": _provenance_to_tree(artifact.provenance),
            "terminal": _terminal_to_tree(artifact.terminal),
            "aggregate_peak_bytes": artifact.aggregate_peak_bytes,
            "memory_profile_fingerprint": artifact.memory_profile_fingerprint,
            "pulse_lineage": _pulse_lineage_to_tree(artifact.pulse_lineage),
        }
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
    return PulseCaptureLineage(
        compiled_pulse,
        tree["trigger_channel"],
        pulse_terminal_ack_from_tree(tree["terminal"]),
        cell_plan,
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
    "PulseCaptureLineage",
    "compile_capture_artifact_pipeline",
]
