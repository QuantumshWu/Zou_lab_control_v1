"""Source-neutral signal and sequencer lineage for PulseScan artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from zlc_data import (
    StreamGenerationId,
)
from zlc_data.transform_codec import (
    data_transform_spec_from_tree,
    data_transform_spec_to_tree,
)
from zlc_neutral_atom.catalog import (
    definition_key_from_tree,
    definition_key_to_tree,
)
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    pulse_terminal_ack_from_tree,
    pulse_terminal_ack_to_tree,
    validate_pulse_terminal_for_artifact,
)
from zlc_neutral_atom.runtime.streams import (
    ArtifactInputRef,
    EventId,
    EventRef,
    ProcessorStageProvenance,
    StreamId,
    event_ref_to_tree,
    processor_stage_provenance_from_tree,
    processor_stage_provenance_to_tree,
)
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationEvidence,
    SignalProjectionAuthority,
    signal_association_evidence_from_tree,
    signal_association_evidence_to_tree,
    signal_projection_authority_from_tree,
    signal_projection_authority_to_tree,
)
from zlc_pulse import CompiledPulseArtifact, PulseExecutionForm
from zlc_storage import (
    canonical_digest,
    canonical_text,
    encode,
    exact_mapping,
    nonnegative_integer,
    positive_integer,
    sha256_text,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseParameterScanProgram,
)
from .source_binding import ScanSignalBinding


SCAN_EXECUTION_SCHEMA = "zlc_neutral_atom.logic_nodes.pulse_scan.execution"
SIGNAL_EVENT_SEQUENCE_SCHEMA = (
    "zlc_neutral_atom.logic_nodes.pulse_scan.causal-signal-event-sequence"
)
_AUTONOMOUS_KIND = "AUTONOMOUS_SCAN_SLOT"
_API_SEGMENTED_KIND = "API_SLOT_SEGMENTED_EXISTING"


@dataclass(frozen=True, slots=True)
class SignalEventSequence:
    """Expandable proof of the ordered external events consumed as scan ``y``.

    Selected outputs may be a filtered phase of a wider producer stream, so
    source sequence numbers need only increase; they are intentionally not
    required to be contiguous.  Exact selected refs and their aligned direct
    input refs are retained without retaining a second copy of payloads.
    ``ordered_event_digest`` remains a compact integrity value, but is always
    recomputed from the stored refs and therefore is not opaque lineage.
    """

    binding: ScanSignalBinding
    projection_authority: SignalProjectionAuthority
    stream_id: StreamId
    generation: StreamGenerationId
    first_sequence: int
    last_sequence: int
    count: int
    ordered_event_digest: str
    source_run_id: str
    source_id: str
    event_refs: tuple[EventRef, ...]
    direct_input_event_refs: tuple[tuple[EventRef, ...], ...]
    processor_stages: tuple[ProcessorStageProvenance, ...]
    associations: tuple[SignalAssociationEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ScanSignalBinding):
            raise TypeError("binding must be ScanSignalBinding")
        if not isinstance(self.projection_authority, SignalProjectionAuthority):
            raise TypeError(
                "projection_authority must be SignalProjectionAuthority"
            )
        committed = self.projection_authority.committed_transform
        committed_spec = None if committed is None else committed.spec
        if self.binding.transform != committed_spec:
            raise ValueError(
                "signal binding authoring transform differs from committed authority"
            )
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        first = nonnegative_integer(self.first_sequence, "first_sequence")
        last = nonnegative_integer(self.last_sequence, "last_sequence")
        count = positive_integer(self.count, "count")
        if last < first or count > last - first + 1:
            raise ValueError("signal event sequence bounds cannot contain its count")
        sha256_text(self.ordered_event_digest, "ordered_event_digest")
        canonical_text(self.source_run_id, "source_run_id")
        canonical_text(self.source_id, "source_id")
        references = tuple(self.event_refs)
        if len(references) != count or any(
            not isinstance(reference, EventRef) for reference in references
        ):
            raise ValueError("event_refs must contain exactly count EventRef values")
        if any(
            reference.stream_id != self.stream_id
            or reference.generation != self.generation
            for reference in references
        ):
            raise ValueError("event_refs differ from the selected stream generation")
        if references[0].sequence != first or references[-1].sequence != last:
            raise ValueError("event_refs differ from the declared sequence bounds")
        if any(
            right.sequence <= left.sequence
            for left, right in zip(references, references[1:])
        ):
            raise ValueError("event_refs must be strictly ordered")
        if _ordered_event_digest(references) != self.ordered_event_digest:
            raise ValueError("ordered_event_digest differs from event_refs")
        direct_rows = tuple(tuple(row) for row in self.direct_input_event_refs)
        if len(direct_rows) != count or any(
            any(not isinstance(reference, EventRef) for reference in row)
            for row in direct_rows
        ):
            raise ValueError(
                "direct_input_event_refs must align EventRef rows to scan cells"
            )
        stages = tuple(self.processor_stages)
        if any(not isinstance(stage, ProcessorStageProvenance) for stage in stages):
            raise TypeError(
                "processor_stages must contain ProcessorStageProvenance values"
            )
        associations = tuple(self.associations)
        if not associations or any(
            not isinstance(item, SignalAssociationEvidence)
            for item in associations
        ):
            raise ValueError(
                "associations must contain producer-owned SignalAssociationEvidence"
            )
        if sum(
            item.request.expected_event_count for item in associations
        ) != count:
            raise ValueError(
                "association event groups must cover exactly the selected events"
            )
        association_ids = tuple(
            item.request.association_id for item in associations
        )
        cause_ids = tuple(item.request.cause_id for item in associations)
        if len(set(association_ids)) != len(association_ids):
            raise ValueError("association ids must be unique within one scan")
        if len(set(cause_ids)) != len(cause_ids):
            raise ValueError("association cause ids must be unique within one scan")
        object.__setattr__(self, "first_sequence", first)
        object.__setattr__(self, "last_sequence", last)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "event_refs", references)
        object.__setattr__(self, "direct_input_event_refs", direct_rows)
        object.__setattr__(
            self,
            "processor_stages",
            tuple(
                ProcessorStageProvenance(
                    stage.processor_binding_digest,
                    stage.direct_artifact_inputs,
                )
                for stage in stages
            ),
        )
        object.__setattr__(self, "associations", associations)

    @property
    def artifact_inputs(self) -> tuple[ArtifactInputRef, ...]:
        """Unique direct artifact inputs in processor-chain order."""

        ordered: list[ArtifactInputRef] = []
        seen: set[str] = set()
        for stage in self.processor_stages:
            for reference in stage.direct_artifact_inputs:
                if reference.fingerprint not in seen:
                    seen.add(reference.fingerprint)
                    ordered.append(reference)
        return tuple(ordered)


def _ordered_event_digest(references: tuple[EventRef, ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"zlc_neutral_atom.PulseScanSignalEventRefs\x00")
    for reference in references:
        encoded = encode(event_ref_to_tree(reference))
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _event_ref_from_tree(tree: object) -> EventRef:
    data = exact_mapping(
        tree,
        {"stream_id", "generation", "sequence", "event_id", "payload_digest"},
        "PulseScan EventRef",
        discriminator=None,
    )
    value = EventRef(
        StreamId(data["stream_id"]),
        StreamGenerationId(data["generation"]),
        data["sequence"],
        EventId(data["event_id"]),
        data["payload_digest"],
    )
    if event_ref_to_tree(value) != tree:
        raise ValueError("PulseScan EventRef tree is non-canonical")
    return value


def _validate_association_for_execution(
    evidence: SignalAssociationEvidence,
    artifact: CompiledPulseArtifact,
    terminal: PulseTerminalAck,
    *,
    expected_event_count: int,
) -> None:
    """Bind persisted producer evidence back to its exact pulse execution."""

    schedules = artifact.trigger_schedules
    if len(schedules) != 1:
        raise ValueError(
            "PulseScan signal association requires one physical trigger schedule"
        )
    schedule = schedules[0]
    request = evidence.request
    expected = (
        terminal.session_id,
        artifact.fingerprint,
        expected_event_count,
        schedule.fingerprint,
        schedule.channel,
        schedule.total,
        schedule.minimum_interval_ticks,
        artifact.target_ir.clock_hz,
    )
    actual = (
        request.cause_id,
        request.cause_digest,
        request.expected_event_count,
        request.trigger_schedule_fingerprint,
        request.trigger_channel,
        request.trigger_count,
        request.minimum_trigger_interval_ticks,
        request.clock_hz,
    )
    if actual != expected:
        raise ValueError(
            "signal association is not bound to its exact pulse artifact and session"
        )
    if evidence.terminal_evidence_digest != canonical_digest(
        pulse_terminal_ack_to_tree(terminal)
    ):
        raise ValueError(
            "signal association is not bound to its exact pulse terminal"
        )


@dataclass(frozen=True, slots=True)
class AutonomousScanExecution:
    program: AutonomousScanSlotProgram
    artifact: CompiledPulseArtifact
    terminal: PulseTerminalAck
    source: SignalEventSequence

    def __post_init__(self) -> None:
        if not isinstance(self.program, AutonomousScanSlotProgram):
            raise TypeError("program must be AutonomousScanSlotProgram")
        if not isinstance(self.artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        if self.artifact.execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
            raise ValueError("autonomous scan lineage requires AUTONOMOUS_SCAN_ONCE")
        if not isinstance(self.terminal, PulseTerminalAck):
            raise TypeError("terminal must be PulseTerminalAck")
        validate_pulse_terminal_for_artifact(self.terminal, self.artifact)
        if not isinstance(self.source, SignalEventSequence):
            raise TypeError("source must be SignalEventSequence")
        expected = self.program.sweep_count * self.program.point_table.row_count
        if self.source.count != expected:
            raise ValueError("source event count differs from autonomous R by P")
        if len(self.source.associations) != 1:
            raise ValueError("autonomous scan requires one signal association")
        _validate_association_for_execution(
            self.source.associations[0],
            self.artifact,
            self.terminal,
            expected_event_count=expected,
        )


@dataclass(frozen=True, slots=True)
class ApiSegmentEvidence:
    repeat_index: int
    point_ordinal: int
    artifact: CompiledPulseArtifact
    terminal: PulseTerminalAck

    def __post_init__(self) -> None:
        repeat = nonnegative_integer(self.repeat_index, "repeat_index")
        point = nonnegative_integer(self.point_ordinal, "point_ordinal")
        if not isinstance(self.artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        if self.artifact.execution_form is not PulseExecutionForm.STATIC_ONCE:
            raise ValueError("API segment lineage requires STATIC_ONCE")
        if not isinstance(self.terminal, PulseTerminalAck):
            raise TypeError("terminal must be PulseTerminalAck")
        validate_pulse_terminal_for_artifact(self.terminal, self.artifact)
        object.__setattr__(self, "repeat_index", repeat)
        object.__setattr__(self, "point_ordinal", point)


@dataclass(frozen=True, slots=True)
class ApiSegmentedScanExecution:
    program: ApiSlotSegmentedProgram
    segments: tuple[ApiSegmentEvidence, ...]
    source: SignalEventSequence

    def __post_init__(self) -> None:
        if not isinstance(self.program, ApiSlotSegmentedProgram):
            raise TypeError("program must be ApiSlotSegmentedProgram")
        segments = tuple(self.segments)
        if len(segments) != self.program.segment_count or any(
            not isinstance(item, ApiSegmentEvidence) for item in segments
        ):
            raise ValueError("API segments must cover the complete R by P run")
        expected_cells = tuple(
            (repeat, point)
            for repeat in range(self.program.sweep_count)
            for point in range(self.program.point_count)
        )
        if tuple(
            (item.repeat_index, item.point_ordinal) for item in segments
        ) != expected_cells:
            raise ValueError("API segment order must be repeat-major and point-fast")
        if not isinstance(self.source, SignalEventSequence):
            raise TypeError("source must be SignalEventSequence")
        if self.source.count != self.program.segment_count:
            raise ValueError("source event count differs from API R by P")
        if len(self.source.associations) != len(segments):
            raise ValueError("API segments require one signal association per cell")
        for association, segment in zip(
            self.source.associations,
            segments,
            strict=True,
        ):
            _validate_association_for_execution(
                association,
                segment.artifact,
                segment.terminal,
                expected_event_count=1,
            )
        object.__setattr__(self, "segments", segments)


PulseScanExecution = AutonomousScanExecution | ApiSegmentedScanExecution


def execution_compiled_artifacts(
    execution: PulseScanExecution,
) -> tuple[CompiledPulseArtifact, ...]:
    if isinstance(execution, AutonomousScanExecution):
        return (execution.artifact,)
    if not isinstance(execution, ApiSegmentedScanExecution):
        raise TypeError("execution must be PulseScanExecution")
    by_point: list[CompiledPulseArtifact | None] = [None] * execution.program.point_count
    for item in execution.segments:
        existing = by_point[item.point_ordinal]
        if existing is None:
            by_point[item.point_ordinal] = item.artifact
        elif existing != item.artifact:
            raise ValueError("one API point used different artifacts across repeats")
    if any(item is None for item in by_point):
        raise ValueError("API execution omitted a point artifact")
    return tuple(item for item in by_point if item is not None)


def _binding_to_tree(value: ScanSignalBinding) -> dict[str, object]:
    return {
        "producer_definition": definition_key_to_tree(value.producer_definition),
        "output": {
            "name": value.output.name,
            "contract_id": value.output.contract_id,
        },
        "transform": (
            None
            if value.transform is None
            else data_transform_spec_to_tree(value.transform)
        ),
    }


def _binding_from_tree(tree: object) -> ScanSignalBinding:
    data = exact_mapping(
        tree,
        {"producer_definition", "output", "transform"},
        "PulseScan signal binding",
        discriminator=None,
    )
    output = exact_mapping(
        data["output"],
        {"name", "contract_id"},
        "PulseScan signal output",
        discriminator=None,
    )
    from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration

    return ScanSignalBinding(
        definition_key_from_tree(data["producer_definition"]),
        DatasetOutputDeclaration(output["name"], output["contract_id"]),
        (
            None
            if data["transform"] is None
            else data_transform_spec_from_tree(data["transform"])
        ),
    )


def signal_event_sequence_to_tree(value: SignalEventSequence) -> dict[str, object]:
    if not isinstance(value, SignalEventSequence):
        raise TypeError("value must be SignalEventSequence")
    return {
        "schema": SIGNAL_EVENT_SEQUENCE_SCHEMA,
        "binding": _binding_to_tree(value.binding),
        "projection_authority": signal_projection_authority_to_tree(
            value.projection_authority
        ),
        "stream_id": value.stream_id.value,
        "generation": value.generation.value,
        "first_sequence": value.first_sequence,
        "last_sequence": value.last_sequence,
        "count": value.count,
        "ordered_event_digest": value.ordered_event_digest,
        "source_run_id": value.source_run_id,
        "source_id": value.source_id,
        "event_refs": [event_ref_to_tree(item) for item in value.event_refs],
        "direct_input_event_refs": [
            [event_ref_to_tree(item) for item in row]
            for row in value.direct_input_event_refs
        ],
        "processor_stages": [
            processor_stage_provenance_to_tree(stage)
            for stage in value.processor_stages
        ],
        "associations": [
            signal_association_evidence_to_tree(item)
            for item in value.associations
        ],
    }


def signal_event_sequence_from_tree(tree: object) -> SignalEventSequence:
    data = exact_mapping(
        tree,
        {
            "schema",
            "binding",
            "projection_authority",
            "stream_id",
            "generation",
            "first_sequence",
            "last_sequence",
            "count",
            "ordered_event_digest",
            "source_run_id",
            "source_id",
            "event_refs",
            "direct_input_event_refs",
            "processor_stages",
            "associations",
        },
        SIGNAL_EVENT_SEQUENCE_SCHEMA,
    )
    event_refs = data["event_refs"]
    direct_rows = data["direct_input_event_refs"]
    stages = data["processor_stages"]
    associations = data["associations"]
    if not isinstance(event_refs, list):
        raise TypeError("SignalEventSequence event_refs must be a list")
    if not isinstance(direct_rows, list) or any(
        not isinstance(row, list) for row in direct_rows
    ):
        raise TypeError(
            "SignalEventSequence direct_input_event_refs must be nested lists"
        )
    if not isinstance(stages, list):
        raise TypeError("SignalEventSequence processor_stages must be a list")
    if not isinstance(associations, list):
        raise TypeError("SignalEventSequence associations must be a list")
    value = SignalEventSequence(
        _binding_from_tree(data["binding"]),
        signal_projection_authority_from_tree(data["projection_authority"]),
        StreamId(data["stream_id"]),
        StreamGenerationId(data["generation"]),
        data["first_sequence"],
        data["last_sequence"],
        data["count"],
        data["ordered_event_digest"],
        data["source_run_id"],
        data["source_id"],
        tuple(_event_ref_from_tree(item) for item in event_refs),
        tuple(
            tuple(_event_ref_from_tree(item) for item in row)
            for row in direct_rows
        ),
        tuple(processor_stage_provenance_from_tree(item) for item in stages),
        tuple(signal_association_evidence_from_tree(item) for item in associations),
    )
    if signal_event_sequence_to_tree(value) != tree:
        raise ValueError("SignalEventSequence tree is non-canonical")
    return value


def pulse_scan_execution_to_tree(value: PulseScanExecution) -> dict[str, object]:
    if isinstance(value, AutonomousScanExecution):
        return {
            "schema": SCAN_EXECUTION_SCHEMA,
            "kind": _AUTONOMOUS_KIND,
            "program_fingerprint": value.program.fingerprint,
            "terminal": pulse_terminal_ack_to_tree(value.terminal),
            "source": signal_event_sequence_to_tree(value.source),
        }
    if isinstance(value, ApiSegmentedScanExecution):
        artifacts = execution_compiled_artifacts(value)
        artifact_index = {item.fingerprint: index for index, item in enumerate(artifacts)}
        return {
            "schema": SCAN_EXECUTION_SCHEMA,
            "kind": _API_SEGMENTED_KIND,
            "program_fingerprint": value.program.fingerprint,
            "segments": [
                {
                    "repeat_index": item.repeat_index,
                    "point_ordinal": item.point_ordinal,
                    "artifact_index": artifact_index[item.artifact.fingerprint],
                    "terminal": pulse_terminal_ack_to_tree(item.terminal),
                }
                for item in value.segments
            ],
            "source": signal_event_sequence_to_tree(value.source),
        }
    raise TypeError("value must be PulseScanExecution")


def pulse_scan_execution_from_tree(
    tree: object,
    program: PulseParameterScanProgram,
    compiled_artifacts: tuple[CompiledPulseArtifact, ...],
) -> PulseScanExecution:
    if not isinstance(tree, dict):
        raise TypeError("PulseScan execution tree must be a mapping")
    artifacts = tuple(compiled_artifacts)
    if any(not isinstance(item, CompiledPulseArtifact) for item in artifacts):
        raise TypeError("compiled_artifacts must contain CompiledPulseArtifact")
    kind = tree.get("kind")
    if kind == _AUTONOMOUS_KIND:
        data = exact_mapping(
            tree,
            {"schema", "kind", "program_fingerprint", "terminal", "source"},
            SCAN_EXECUTION_SCHEMA,
        )
        if not isinstance(program, AutonomousScanSlotProgram) or len(artifacts) != 1:
            raise ValueError("autonomous execution has another program/artifact set")
        value: PulseScanExecution = AutonomousScanExecution(
            program,
            artifacts[0],
            pulse_terminal_ack_from_tree(data["terminal"]),
            signal_event_sequence_from_tree(data["source"]),
        )
    elif kind == _API_SEGMENTED_KIND:
        data = exact_mapping(
            tree,
            {"schema", "kind", "program_fingerprint", "segments", "source"},
            SCAN_EXECUTION_SCHEMA,
        )
        if not isinstance(program, ApiSlotSegmentedProgram):
            raise ValueError("API execution has another program kind")
        rows = data["segments"]
        if not isinstance(rows, list):
            raise TypeError("API execution segments must be a list")
        segments = []
        for row in rows:
            item = exact_mapping(
                row,
                {"repeat_index", "point_ordinal", "artifact_index", "terminal"},
                "API segment evidence",
                discriminator=None,
            )
            index = nonnegative_integer(item["artifact_index"], "artifact_index")
            if index >= len(artifacts):
                raise ValueError("API segment artifact_index is out of range")
            segments.append(
                ApiSegmentEvidence(
                    item["repeat_index"],
                    item["point_ordinal"],
                    artifacts[index],
                    pulse_terminal_ack_from_tree(item["terminal"]),
                )
            )
        value = ApiSegmentedScanExecution(
            program,
            tuple(segments),
            signal_event_sequence_from_tree(data["source"]),
        )
    else:
        raise ValueError("PulseScan execution kind is unknown")
    sha256_text(tree["program_fingerprint"], "program_fingerprint")
    if tree["program_fingerprint"] != program.fingerprint:
        raise ValueError("execution belongs to another PulseScan program")
    if pulse_scan_execution_to_tree(value) != tree:
        raise ValueError("PulseScan execution tree is non-canonical")
    return value


__all__ = [
    "ApiSegmentEvidence",
    "ApiSegmentedScanExecution",
    "AutonomousScanExecution",
    "PulseScanExecution",
    "SCAN_EXECUTION_SCHEMA",
    "SIGNAL_EVENT_SEQUENCE_SCHEMA",
    "SignalEventSequence",
    "execution_compiled_artifacts",
    "pulse_scan_execution_from_tree",
    "pulse_scan_execution_to_tree",
    "signal_event_sequence_from_tree",
    "signal_event_sequence_to_tree",
]
