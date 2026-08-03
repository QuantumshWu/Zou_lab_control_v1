"""Source-neutral signal and sequencer lineage for PulseScan artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import (
    StreamGenerationId,
)
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    pulse_terminal_ack_from_tree,
    pulse_terminal_ack_to_tree,
    validate_pulse_terminal_for_artifact,
)
from zlc_neutral_atom.runtime.streams import (
    EventRef,
    StreamId,
    event_ref_from_tree,
    event_ref_to_tree,
)
from zlc_neutral_atom.runtime.signal_source import (
    SignalProjectionAuthority,
    signal_projection_authority_from_tree,
    signal_projection_authority_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    materialize_scan_sweeps,
)
from zlc_storage import (
    exact_mapping,
    nonnegative_integer,
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
    The ordered refs themselves are the lineage; no second digest mirrors them.
    """

    binding: ScanSignalBinding
    projection_authority: SignalProjectionAuthority
    event_refs: tuple[EventRef, ...]
    direct_input_event_refs: tuple[tuple[EventRef, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ScanSignalBinding):
            raise TypeError("binding must be ScanSignalBinding")
        if not isinstance(self.projection_authority, SignalProjectionAuthority):
            raise TypeError(
                "projection_authority must be SignalProjectionAuthority"
            )
        if self.projection_authority.committed_transform is not None:
            raise ValueError(
                "PulseScan consumes the selected signal as declared; authoritative "
                "projection belongs to its upstream producer"
            )
        references = tuple(self.event_refs)
        if not references or any(
            not isinstance(reference, EventRef) for reference in references
        ):
            raise ValueError("event_refs must contain EventRef values")
        stream_id = references[0].stream_id
        generation = references[0].generation
        if any(
            reference.stream_id != stream_id
            or reference.generation != generation
            for reference in references
        ):
            raise ValueError("event_refs differ from the selected stream generation")
        if any(
            right.sequence <= left.sequence
            for left, right in zip(references, references[1:])
        ):
            raise ValueError("event_refs must be strictly ordered")
        direct_rows = tuple(tuple(row) for row in self.direct_input_event_refs)
        if len(direct_rows) != len(references) or any(
            any(not isinstance(reference, EventRef) for reference in row)
            for row in direct_rows
        ):
            raise ValueError(
                "direct_input_event_refs must align EventRef rows to scan cells"
            )
        object.__setattr__(self, "event_refs", references)
        object.__setattr__(self, "direct_input_event_refs", direct_rows)

    @property
    def stream_id(self) -> StreamId:
        return self.event_refs[0].stream_id

    @property
    def generation(self) -> StreamGenerationId:
        return self.event_refs[0].generation

    @property
    def first_sequence(self) -> int:
        return self.event_refs[0].sequence

    @property
    def last_sequence(self) -> int:
        return self.event_refs[-1].sequence

    @property
    def count(self) -> int:
        return len(self.event_refs)


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
        expected_document = materialize_scan_sweeps(
            self.program.execution_document,
            self.program.sweep_count,
        )
        if self.artifact.source_document_digest != expected_document.fingerprint:
            raise ValueError("autonomous pulse artifact differs from its scan program")
        if not isinstance(self.terminal, PulseTerminalAck):
            raise TypeError("terminal must be PulseTerminalAck")
        validate_pulse_terminal_for_artifact(self.terminal, self.artifact)
        if not isinstance(self.source, SignalEventSequence):
            raise TypeError("source must be SignalEventSequence")
        expected = self.program.sweep_count * self.program.point_table.row_count
        if self.source.count != expected:
            raise ValueError("source event count differs from autonomous R by P")


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
        documents = self.program.resolved_point_documents
        if any(
            item.artifact.source_document_digest
            != documents[item.point_ordinal].fingerprint
            for item in segments
        ):
            raise ValueError("API pulse artifact differs from its scan point program")
        if not isinstance(self.source, SignalEventSequence):
            raise TypeError("source must be SignalEventSequence")
        if self.source.count != self.program.segment_count:
            raise ValueError("source event count differs from API R by P")
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
        "signal_name": value.signal_name,
        "output_name": value.output_name,
    }


def _binding_from_tree(tree: object) -> ScanSignalBinding:
    data = exact_mapping(
        tree,
        {"signal_name", "output_name"},
        "PulseScan signal binding",
        discriminator=None,
    )
    return ScanSignalBinding(
        data["signal_name"],
        data["output_name"],
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
        "event_refs": [event_ref_to_tree(item) for item in value.event_refs],
        "direct_input_event_refs": [
            [event_ref_to_tree(item) for item in row]
            for row in value.direct_input_event_refs
        ],
    }


def signal_event_sequence_from_tree(tree: object) -> SignalEventSequence:
    data = exact_mapping(
        tree,
        {
            "schema",
            "binding",
            "projection_authority",
            "event_refs",
            "direct_input_event_refs",
        },
        SIGNAL_EVENT_SEQUENCE_SCHEMA,
    )
    event_refs = data["event_refs"]
    direct_rows = data["direct_input_event_refs"]
    if not isinstance(event_refs, list):
        raise TypeError("SignalEventSequence event_refs must be a list")
    if not isinstance(direct_rows, list) or any(
        not isinstance(row, list) for row in direct_rows
    ):
        raise TypeError(
            "SignalEventSequence direct_input_event_refs must be nested lists"
        )
    return SignalEventSequence(
        _binding_from_tree(data["binding"]),
        signal_projection_authority_from_tree(data["projection_authority"]),
        tuple(event_ref_from_tree(item) for item in event_refs),
        tuple(
            tuple(event_ref_from_tree(item) for item in row)
            for row in direct_rows
        ),
    )


def pulse_scan_execution_to_tree(value: PulseScanExecution) -> dict[str, object]:
    if isinstance(value, AutonomousScanExecution):
        return {
            "schema": SCAN_EXECUTION_SCHEMA,
            "kind": _AUTONOMOUS_KIND,
            "terminal": pulse_terminal_ack_to_tree(value.terminal),
            "source": signal_event_sequence_to_tree(value.source),
        }
    if isinstance(value, ApiSegmentedScanExecution):
        artifacts = execution_compiled_artifacts(value)
        artifact_index = {item.fingerprint: index for index, item in enumerate(artifacts)}
        return {
            "schema": SCAN_EXECUTION_SCHEMA,
            "kind": _API_SEGMENTED_KIND,
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
            {"schema", "kind", "terminal", "source"},
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
            {"schema", "kind", "segments", "source"},
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
