"""Exact pulse-execution lineage for the two supported scan programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from zlc_data import (
    READOUT_EVENT,
    DatasetSchema,
    ValueSchema,
    value_schema_from_tree,
    value_schema_to_tree,
)
from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraFrameMetadataContract,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
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
    DatasetCellAddress,
    DatasetCellSchedule,
    DatasetSealProvenance,
)
from zlc_neutral_atom.runtime.pipeline import PipelineResult
from zlc_neutral_atom.runtime.streams import (
    EventSpanRef,
    StreamId,
    event_span_ref_from_tree,
    event_span_ref_to_tree,
)
from zlc_neutral_atom.timing.lineage import (
    PulseCaptureBinding,
    PulseCaptureEvidence,
    pulse_capture_evidence_from_tree,
    pulse_capture_evidence_to_tree,
)
from zlc_neutral_atom.timing.capture_plan import (
    capture_cell_join_contract_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    expand_autonomous_scan_repeats,
)
from zlc_storage import canonical_text, exact_mapping, nonnegative_integer, sha256_text

from .contracts import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseScanProgram,
)


SCAN_EXECUTION_SCHEMA = "zlc_neutral_atom.PulseScanExecution"
CAMERA_RUN_EVIDENCE_SCHEMA = "zlc_neutral_atom.CameraRunEvidence"
_AUTONOMOUS_KIND = "AUTONOMOUS_SCAN_SLOT"
_API_SEGMENTED_KIND = "API_SLOT_SEGMENTED_EXISTING"
_API_STATIC_SHAPE_SCHEMA = "zlc_neutral_atom.ApiSegmentedMetadataStaticShape"


def api_segmented_metadata_static_shape_to_tree(
    program: ApiSlotSegmentedProgram,
    point_bindings: tuple[PulseCaptureBinding | PulseCaptureEvidence, ...],
    *,
    camera_source_stream_id: StreamId,
    result_stream_id: StreamId,
    result_source_id: str,
    derivation_root_stream_id: StreamId | None,
    camera_provenance: CameraCaptureProvenance,
    camera_capability: CameraCapabilityEvidence,
    camera_arm_spec: FrozenCaptureSpec,
    camera_source_value_schema: ValueSchema,
    camera_source_schema_fingerprint: str,
) -> dict[str, object]:
    """Project pre-FIRE execution and stream-topology metadata facts."""

    if not isinstance(program, ApiSlotSegmentedProgram):
        raise TypeError("program must be ApiSlotSegmentedProgram")
    bindings = tuple(point_bindings)
    if len(bindings) != program.point_count or any(
        not isinstance(item, (PulseCaptureBinding, PulseCaptureEvidence))
        for item in bindings
    ):
        raise TypeError("point_bindings must contain one pulse binding per point")
    if any(
        item.compiled_artifact.source_document_digest != document.fingerprint
        for item, document in zip(bindings, program.resolved_point_documents)
    ):
        raise ValueError("API execution shape bindings differ from resolved points")
    if not isinstance(camera_source_stream_id, StreamId):
        raise TypeError("camera_source_stream_id must be StreamId")
    if not isinstance(result_stream_id, StreamId):
        raise TypeError("result_stream_id must be StreamId")
    canonical_text(result_source_id, "result_source_id")
    if derivation_root_stream_id is not None and not isinstance(
        derivation_root_stream_id,
        StreamId,
    ):
        raise TypeError("derivation_root_stream_id must be StreamId or None")
    if not isinstance(camera_provenance, CameraCaptureProvenance):
        raise TypeError("camera_provenance must be CameraCaptureProvenance")
    if not isinstance(camera_capability, CameraCapabilityEvidence):
        raise TypeError("camera_capability must be CameraCapabilityEvidence")
    if not isinstance(camera_arm_spec, FrozenCaptureSpec):
        raise TypeError("camera_arm_spec must be FrozenCaptureSpec")
    if not isinstance(camera_source_value_schema, ValueSchema):
        raise TypeError("camera_source_value_schema must be ValueSchema")
    sha256_text(
        camera_source_schema_fingerprint,
        "camera_source_schema_fingerprint",
    )
    return {
        "schema": _API_STATIC_SHAPE_SCHEMA,
        "program_fingerprint": program.fingerprint,
        "stream_topology": {
            "camera_source_stream_id": camera_source_stream_id.value,
            "result_stream_id": result_stream_id.value,
            "result_source_id": result_source_id,
            "derivation_root_stream_id": (
                None
                if derivation_root_stream_id is None
                else derivation_root_stream_id.value
            ),
        },
        "camera": {
            "provenance": camera_capture_provenance_to_tree(camera_provenance),
            "capability": camera_capability_evidence_to_tree(camera_capability),
            "arm_spec": frozen_capture_spec_to_tree(camera_arm_spec),
            "source_value_schema": value_schema_to_tree(camera_source_value_schema),
            "source_schema_fingerprint": camera_source_schema_fingerprint,
        },
        "points": [
            {
                "artifact_fingerprint": item.compiled_artifact.fingerprint,
                "trigger_channel": item.trigger_channel,
                "join_contract": capture_cell_join_contract_to_tree(
                    item.join_contract
                    if isinstance(item, PulseCaptureEvidence)
                    else item.cell_plan.join_contract
                ),
            }
            for item in bindings
        ],
    }


@dataclass(frozen=True, slots=True)
class CameraRunEvidence:
    """Compact durable proof for the one exact camera arm behind a scan.

    Frames remain owned by the dataset artifact.  This value retains only the
    terminal receipt, owner-minted camera facts, canonical arm request, camera
    cell schema plus full-schema fingerprint, source event span, and the digest
    of the actual ordinal-to-cell schedule.  Point coordinates stay single-owned
    by the surrounding scan artifact and are composed back in during reload.
    """

    terminal: CaptureTerminalAck
    camera_provenance: CameraCaptureProvenance
    capability: CameraCapabilityEvidence
    arm_spec: FrozenCaptureSpec
    source_value_schema: ValueSchema
    source_schema_fingerprint: str
    source_schedule_digest: str
    source_event_span: EventSpanRef

    def __post_init__(self) -> None:
        if not isinstance(self.terminal, CaptureTerminalAck):
            raise TypeError("terminal must be CaptureTerminalAck")
        if not isinstance(self.camera_provenance, CameraCaptureProvenance):
            raise TypeError("camera_provenance must be CameraCaptureProvenance")
        if not isinstance(self.capability, CameraCapabilityEvidence):
            raise TypeError("capability must be CameraCapabilityEvidence")
        if not isinstance(self.arm_spec, FrozenCaptureSpec):
            raise TypeError("arm_spec must be FrozenCaptureSpec")
        if not isinstance(self.source_value_schema, ValueSchema):
            raise TypeError("source_value_schema must be ValueSchema")
        sha256_text(self.source_schema_fingerprint, "source_schema_fingerprint")
        sha256_text(self.source_schedule_digest, "source_schedule_digest")
        if not isinstance(self.source_event_span, EventSpanRef):
            raise TypeError("source_event_span must be EventSpanRef")

        provenance = self.camera_provenance
        capability = self.capability
        terminal = self.terminal
        capability.physical_facts.validate_descriptor(provenance.descriptor)
        expected_payload = CameraSampleContract(self.source_value_schema)
        if capability.payload_contract_fingerprint != expected_payload.fingerprint:
            raise ValueError(
                "camera payload contract differs from the source camera schema"
            )
        if self.arm_spec.owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
            raise ValueError("camera arm spec belongs to an unknown owner")
        decoded_arm = decode_camera_capture_spec(self.arm_spec)
        if freeze_camera_capture_spec(decoded_arm) != self.arm_spec:
            raise ValueError("camera arm spec is not the canonical owner encoding")
        if decoded_arm.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
            raise ValueError("exact scan camera must be externally triggered")
        if capability.exact_external_trigger_qualification_digest is None:
            raise ValueError(
                "exact scan camera lacks qualified one-frame-per-trigger evidence"
            )

        event_count = self.source_event_span.count
        if decoded_arm.expected_frames != event_count:
            raise ValueError("camera arm frame budget differs from source event span")
        if (
            terminal.produced_count != event_count
            or terminal.drained_count != event_count
            or not terminal.source_stopped
            or not terminal.no_more_frames
            or not terminal.joined
        ):
            raise ValueError("camera terminal does not prove exact stop, drain, and join")
        if len(
            {
                capability.fingerprint,
                provenance.capability_fingerprint,
                terminal.capability_fingerprint,
            }
        ) != 1:
            raise ValueError("camera capability lineage is inconsistent")
        if len(
            {
                self.arm_spec.digest,
                provenance.camera_arm_spec_fingerprint,
                terminal.capture_spec_fingerprint,
            }
        ) != 1:
            raise ValueError("camera arm-spec lineage is inconsistent")
        if len(
            {
                decoded_arm.settings_fingerprint,
                capability.settings_fingerprint,
                provenance.active_settings_fingerprint,
                terminal.settings_fingerprint,
            }
        ) != 1:
            raise ValueError("camera settings lineage is inconsistent")
        if (
            capability.capture_spec_owner_fingerprint
            != self.arm_spec.owner_fingerprint
        ):
            raise ValueError("camera capability and arm-spec owners differ")
        if (
            provenance.binding_stamp.binding_instance_id
            != terminal.binding_instance_id
        ):
            raise ValueError("camera binding lineage is inconsistent")
        if provenance.binding.value != capability.source_id:
            raise ValueError("camera source lineage is inconsistent")

    @property
    def event_count(self) -> int:
        return self.source_event_span.count

    def require_event_count(self, expected_count: int) -> None:
        expected = nonnegative_integer(expected_count, "expected_count")
        if expected != self.event_count:
            raise ValueError("camera event count differs from pulse execution")

    def validate_source_schema(self, domain_schema: DatasetSchema) -> DatasetSchema:
        """Rebuild the camera schema from the persisted scan sampling domain."""

        if not isinstance(domain_schema, DatasetSchema):
            raise TypeError("domain_schema must be DatasetSchema")
        schema = DatasetSchema(
            domain_schema.repeat_axis,
            domain_schema.point_axes,
            domain_schema.point_layout,
            self.source_value_schema,
        )
        if schema.fingerprint != self.source_schema_fingerprint:
            raise ValueError("camera source schema differs from the scan sampling domain")
        self.camera_provenance.validate_schema(schema)
        return schema

    def require_schedule(
        self,
        schedule: DatasetCellSchedule,
        source_schema: DatasetSchema,
    ) -> None:
        if not isinstance(schedule, DatasetCellSchedule):
            raise TypeError("schedule must be DatasetCellSchedule")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if source_schema.fingerprint != self.source_schema_fingerprint:
            raise ValueError("camera schedule uses another source schema")
        if len(schedule) != self.event_count:
            raise ValueError("camera event count differs from pulse execution")
        if schedule.digest_for_schema(source_schema) != self.source_schedule_digest:
            raise ValueError("camera source schedule differs from pulse execution")

    def validate_dataset_provenance(
        self,
        provenance: DatasetSealProvenance,
    ) -> None:
        """Rebind camera aggregate facts to the persisted source dataset."""

        if not isinstance(provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if provenance.end_sequence - provenance.start_sequence != self.event_count:
            raise ValueError("dataset interval differs from camera terminal count")
        derivation = provenance.derivation
        if derivation is None:
            span = self.source_event_span
            if (
                provenance.trace_binding.source_id != self.capability.source_id
                or provenance.stream_id != span.stream_id
                or provenance.generation != span.generation
                or provenance.start_sequence != span.start_sequence
                or provenance.end_sequence != span.end_sequence
                or provenance.join_plan_digest != self.source_schedule_digest
                or provenance.metadata_contract_fingerprint
                != CameraFrameMetadataContract().fingerprint
                or provenance.ordered_metadata_digest
                != self.terminal.ordered_metadata_digest
            ):
                raise ValueError(
                    "raw dataset provenance differs from camera aggregate evidence"
                )
        elif derivation.root_input_span != self.source_event_span:
            raise ValueError(
                "processed dataset derivation differs from camera source event span"
            )


def camera_run_evidence_from_pipeline(value: PipelineResult) -> CameraRunEvidence:
    """Snapshot only immutable camera facts from one validated pipeline result."""

    if not isinstance(value, PipelineResult):
        raise TypeError("value must be PipelineResult")
    schema = value.source_dataset_schema
    schedule = value.source_cell_schedule
    evidence = CameraRunEvidence(
        value.capture_terminal,
        value.camera_provenance,
        value.camera_capability_evidence,
        value.camera_arm_spec,
        schema.cell_schema,
        schema.fingerprint,
        schedule.digest_for_schema(schema),
        value.source_event_span,
    )
    evidence.validate_source_schema(schema)
    return evidence


def camera_run_evidence_to_tree(value: CameraRunEvidence) -> dict[str, object]:
    if not isinstance(value, CameraRunEvidence):
        raise TypeError("value must be CameraRunEvidence")
    return {
        "schema": CAMERA_RUN_EVIDENCE_SCHEMA,
        "terminal": capture_terminal_ack_to_tree(value.terminal),
        "camera_provenance": camera_capture_provenance_to_tree(
            value.camera_provenance
        ),
        "capability": camera_capability_evidence_to_tree(value.capability),
        "arm_spec": frozen_capture_spec_to_tree(value.arm_spec),
        "source_value_schema": value_schema_to_tree(value.source_value_schema),
        "source_schema_fingerprint": value.source_schema_fingerprint,
        "source_schedule_digest": value.source_schedule_digest,
        "source_event_span": event_span_ref_to_tree(value.source_event_span),
    }


def camera_run_evidence_from_tree(tree: object) -> CameraRunEvidence:
    data = exact_mapping(
        tree,
        {
            "schema",
            "terminal",
            "camera_provenance",
            "capability",
            "arm_spec",
            "source_value_schema",
            "source_schema_fingerprint",
            "source_schedule_digest",
            "source_event_span",
        },
        CAMERA_RUN_EVIDENCE_SCHEMA,
    )
    value = CameraRunEvidence(
        capture_terminal_ack_from_tree(data["terminal"]),
        camera_capture_provenance_from_tree(data["camera_provenance"]),
        camera_capability_evidence_from_tree(data["capability"]),
        frozen_capture_spec_from_tree(data["arm_spec"]),
        value_schema_from_tree(data["source_value_schema"]),
        data["source_schema_fingerprint"],
        data["source_schedule_digest"],
        event_span_ref_from_tree(data["source_event_span"]),
    )
    if camera_run_evidence_to_tree(value) != tree:
        raise ValueError("CameraRunEvidence tree is typed but non-canonical")
    return value


def _require_autonomous_evidence(
    program: AutonomousScanSlotProgram,
    evidence: PulseCaptureEvidence,
) -> None:
    if not isinstance(evidence, PulseCaptureEvidence):
        raise TypeError("evidence must be PulseCaptureEvidence")
    artifact = evidence.compiled_artifact
    if artifact.execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        raise ValueError("autonomous scan evidence requires AUTONOMOUS_SCAN_ONCE")
    expanded = expand_autonomous_scan_repeats(program.execution_document)
    if artifact.source_document_digest != expanded.fingerprint:
        raise ValueError("autonomous evidence differs from the execution document")
    schedule = evidence.trigger_schedule
    point_count = program.repeat_count * program.point_table.point_layout.storage_size
    if (
        schedule.point_count != point_count
        or schedule.loop_count != 1
        or not schedule.full_point_loop
        or schedule.total != point_count
    ):
        raise ValueError("autonomous evidence is not one complete repeat-major scan")
    if evidence.join_contract.scan_point_layout != program.point_table.point_layout:
        raise ValueError("autonomous evidence differs from the scan point layout")


def _require_api_segment_evidence(
    document: PulseDocument,
    evidence: PulseCaptureEvidence,
) -> None:
    if not isinstance(evidence, PulseCaptureEvidence):
        raise TypeError("evidence must be PulseCaptureEvidence")
    artifact = evidence.compiled_artifact
    if artifact.execution_form is not PulseExecutionForm.STATIC_ONCE:
        raise ValueError("API segment evidence requires STATIC_ONCE")
    if artifact.source_document_digest != document.fingerprint:
        raise ValueError("API segment artifact belongs to another resolved document")
    schedule = evidence.trigger_schedule
    if (
        schedule.point_count != 1
        or schedule.loop_count != 1
        or not schedule.full_point_loop
        or schedule.total != 1
    ):
        raise ValueError("each API segment must contain exactly one readout trigger")
    if (
        evidence.join_contract.scan_point_layout.storage_size != 1
        or evidence.join_contract.within_point_grouping != ((0, 0),)
    ):
        raise ValueError("API segment evidence must describe one local dataset cell")


@dataclass(frozen=True, slots=True)
class AutonomousScanExecution:
    """One autonomous scan program and its single completed pulse receipt."""

    program: AutonomousScanSlotProgram
    evidence: PulseCaptureEvidence
    camera: CameraRunEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.program, AutonomousScanSlotProgram):
            raise TypeError("program must be AutonomousScanSlotProgram")
        _require_autonomous_evidence(self.program, self.evidence)
        if not isinstance(self.camera, CameraRunEvidence):
            raise TypeError("camera must be CameraRunEvidence")
        self.camera.require_event_count(self.evidence.expected_trigger_count)
        self.camera.capability.physical_facts.require_single_capture_trigger_channel(
            self.evidence.trigger_channel
        )
        required_interval = (
            self.camera.capability.physical_facts.required_external_trigger_interval_seconds
        )
        minimum_interval_ticks = self.evidence.trigger_schedule.minimum_interval_ticks
        if required_interval is None:
            raise ValueError("camera evidence omits the qualified trigger interval")
        if (
            minimum_interval_ticks is not None
            and minimum_interval_ticks
            / self.evidence.compiled_artifact.target_ir.clock_hz
            < required_interval
        ):
            raise ValueError("autonomous camera trigger interval is not qualified")


@dataclass(frozen=True, slots=True)
class ApiSegmentEvidence:
    """One completed finite API segment at an exact repeat/point address."""

    repeat_index: int
    point_storage_index: int
    evidence: PulseCaptureEvidence

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repeat_index",
            nonnegative_integer(self.repeat_index, "repeat_index"),
        )
        object.__setattr__(
            self,
            "point_storage_index",
            nonnegative_integer(self.point_storage_index, "point_storage_index"),
        )
        if not isinstance(self.evidence, PulseCaptureEvidence):
            raise TypeError("evidence must be PulseCaptureEvidence")


@dataclass(frozen=True, slots=True)
class ApiSegmentedScanExecution:
    """All API segments, in exact repeat-major/point-fast completion order."""

    program: ApiSlotSegmentedProgram
    segments: tuple[ApiSegmentEvidence, ...]
    camera: CameraRunEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.program, ApiSlotSegmentedProgram):
            raise TypeError("program must be ApiSlotSegmentedProgram")
        if not isinstance(self.camera, CameraRunEvidence):
            raise TypeError("camera must be CameraRunEvidence")
        segments = tuple(self.segments)
        if any(not isinstance(item, ApiSegmentEvidence) for item in segments):
            raise TypeError("segments must contain ApiSegmentEvidence values")
        point_count = self.program.point_table.point_layout.storage_size
        resolved_points = self.program.resolved_point_documents
        expected_count = self.program.repeat_count * point_count
        if len(segments) != expected_count or any(
            (item.repeat_index, item.point_storage_index)
            != divmod(segment_index, point_count)
            for segment_index, item in enumerate(segments)
        ):
            raise ValueError("API segment evidence must be complete R-major/P-fast order")
        channels: set[str] = set()
        binding_instances: set[str] = set()
        evidence_kinds: set[object] = set()
        terminal_session_ids: set[str] = set()
        artifact_by_point: list[CompiledPulseArtifact | None] = [None] * point_count
        for item in segments:
            _require_api_segment_evidence(
                resolved_points[item.point_storage_index],
                item.evidence,
            )
            channels.add(item.evidence.trigger_channel)
            binding_instances.add(item.evidence.terminal.binding_instance_id)
            evidence_kinds.add(item.evidence.terminal.evidence_kind)
            terminal_session_ids.add(item.evidence.terminal.session_id)
            artifact = item.evidence.compiled_artifact
            previous = artifact_by_point[item.point_storage_index]
            if previous is None:
                artifact_by_point[item.point_storage_index] = artifact
            elif previous.fingerprint != artifact.fingerprint:
                raise ValueError("one API point changed compiled artifact across repeats")
        if len(channels) != 1:
            raise ValueError("all API segments must use the same trigger channel")
        if len(binding_instances) != 1:
            raise ValueError("API segments changed sequencer binding instance")
        if len(evidence_kinds) != 1:
            raise ValueError("API segments mixed hardware and simulated evidence")
        if len(terminal_session_ids) != len(segments):
            raise ValueError("API segments reused a pulse terminal receipt")
        artifacts = tuple(item for item in artifact_by_point if item is not None)
        if len(artifacts) != point_count:
            raise ValueError("API execution omits a point artifact")
        self.camera.require_event_count(expected_count)
        self.camera.capability.physical_facts.require_single_capture_trigger_channel(
            next(iter(channels))
        )
        object.__setattr__(self, "segments", segments)


def api_segmented_metadata_static_shape_from_execution(
    value: ApiSegmentedScanExecution,
    provenance: DatasetSealProvenance,
) -> dict[str, object]:
    """Rebuild the pre-FIRE projection from completed execution and provenance."""

    if not isinstance(value, ApiSegmentedScanExecution):
        raise TypeError("value must be ApiSegmentedScanExecution")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    point_count = value.program.point_count
    first_repeat = value.segments[:point_count]
    camera = value.camera
    derivation = provenance.derivation
    return api_segmented_metadata_static_shape_to_tree(
        value.program,
        tuple(item.evidence for item in first_repeat),
        camera_source_stream_id=camera.source_event_span.stream_id,
        result_stream_id=provenance.stream_id,
        result_source_id=provenance.trace_binding.source_id,
        derivation_root_stream_id=(
            None if derivation is None else derivation.root_input_span.stream_id
        ),
        camera_provenance=camera.camera_provenance,
        camera_capability=camera.capability,
        camera_arm_spec=camera.arm_spec,
        camera_source_value_schema=camera.source_value_schema,
        camera_source_schema_fingerprint=camera.source_schema_fingerprint,
    )


PulseScanExecution: TypeAlias = AutonomousScanExecution | ApiSegmentedScanExecution


def api_segmented_cell_schedule(
    program: ApiSlotSegmentedProgram,
    schema: DatasetSchema,
) -> DatasetCellSchedule:
    """Derive the accepted API exception's global R-major/P-fast schedule."""

    if not isinstance(program, ApiSlotSegmentedProgram):
        raise TypeError("program must be ApiSlotSegmentedProgram")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if schema.repeat_axis.size != program.repeat_count:
        raise ValueError("camera repeat axis differs from API scan repeats")
    event_positions = tuple(
        position
        for position, axis in enumerate(schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if len(event_positions) != 1:
        raise ValueError("API camera schema requires one READOUT_EVENT axis")
    event_position = event_positions[0]
    if schema.point_axes[event_position].size != 1:
        raise ValueError("API camera READOUT_EVENT axis must be singleton")
    scan_axes = tuple(
        axis
        for position, axis in enumerate(schema.point_axes)
        if position != event_position
    )
    if scan_axes != program.point_table.point_axes:
        raise ValueError("API camera scan axes differ from the point table")
    scan_positions = {
        axis.axis_id: position for position, axis in enumerate(scan_axes)
    }

    def cells():
        for repeat_index in range(program.repeat_count):
            for point_index in range(program.point_table.point_layout.storage_size):
                scan_multi = program.point_table.point_layout.multi_index(point_index)
                full_multi = tuple(
                    0
                    if position == event_position
                    else scan_multi[scan_positions[axis.axis_id]]
                    for position, axis in enumerate(schema.point_axes)
                )
                yield DatasetCellAddress(
                    repeat_index,
                    schema.point_layout.storage_index(full_multi),
                )

    return DatasetCellSchedule.from_cells(schema, cells())


def execution_compiled_artifacts(
    value: PulseScanExecution,
) -> tuple[CompiledPulseArtifact, ...]:
    """Return the exact point-indexed compiled artifact tuple for persistence."""

    if isinstance(value, AutonomousScanExecution):
        return (value.evidence.compiled_artifact,)
    if isinstance(value, ApiSegmentedScanExecution):
        point_count = value.program.point_table.point_layout.storage_size
        first_repeat = value.segments[:point_count]
        return tuple(item.evidence.compiled_artifact for item in first_repeat)
    raise TypeError("value must be a PulseScanExecution")


def _evidence_reference_to_tree(
    evidence: PulseCaptureEvidence,
    artifact_index: int,
) -> dict[str, object]:
    tree = pulse_capture_evidence_to_tree(evidence)
    if tree is None:
        raise RuntimeError("scan evidence unexpectedly encoded as None")
    return {
        "artifact_index": nonnegative_integer(artifact_index, "artifact_index"),
        "evidence": tree,
    }


def _evidence_reference_from_tree(
    tree: object,
    artifacts: tuple[CompiledPulseArtifact, ...],
) -> PulseCaptureEvidence:
    data = exact_mapping(
        tree,
        {"artifact_index", "evidence"},
        "scan pulse evidence reference",
        discriminator=None,
    )
    artifact_index = nonnegative_integer(data["artifact_index"], "artifact_index")
    try:
        artifact = artifacts[artifact_index]
    except IndexError as exc:
        raise ValueError("scan evidence references an absent compiled artifact") from exc
    evidence = pulse_capture_evidence_from_tree(data["evidence"], artifact)
    if evidence is None:
        raise ValueError("scan pulse evidence cannot be None")
    return evidence


def pulse_scan_execution_to_tree(value: PulseScanExecution) -> dict[str, object]:
    artifacts = execution_compiled_artifacts(value)
    if isinstance(value, AutonomousScanExecution):
        return {
            "schema": SCAN_EXECUTION_SCHEMA,
            "kind": _AUTONOMOUS_KIND,
            "program_fingerprint": value.program.fingerprint,
            "evidence": _evidence_reference_to_tree(value.evidence, 0),
            "camera": camera_run_evidence_to_tree(value.camera),
        }
    if isinstance(value, ApiSegmentedScanExecution):
        return {
            "schema": SCAN_EXECUTION_SCHEMA,
            "kind": _API_SEGMENTED_KIND,
            "program_fingerprint": value.program.fingerprint,
            "camera": camera_run_evidence_to_tree(value.camera),
            "segments": [
                {
                    "repeat_index": item.repeat_index,
                    "point_storage_index": item.point_storage_index,
                    "pulse": _evidence_reference_to_tree(
                        item.evidence,
                        item.point_storage_index,
                    ),
                }
                for item in value.segments
            ],
        }
    raise TypeError("value must be a PulseScanExecution")


def pulse_scan_execution_from_tree(
    tree: object,
    program: PulseScanProgram,
    compiled_artifacts: tuple[CompiledPulseArtifact, ...],
) -> PulseScanExecution:
    artifacts = tuple(compiled_artifacts)
    if any(not isinstance(item, CompiledPulseArtifact) for item in artifacts):
        raise TypeError("compiled_artifacts must contain CompiledPulseArtifact values")
    if not isinstance(tree, dict):
        raise TypeError("pulse scan execution tree must be a mapping")
    kind = tree.get("kind")
    if kind == _AUTONOMOUS_KIND:
        data = exact_mapping(
            tree,
            {"schema", "kind", "program_fingerprint", "evidence", "camera"},
            SCAN_EXECUTION_SCHEMA,
        )
        if not isinstance(program, AutonomousScanSlotProgram):
            raise ValueError("autonomous execution tree has another program kind")
        if len(artifacts) != 1:
            raise ValueError("autonomous execution requires one compiled artifact")
        sha256_text(data["program_fingerprint"], "program_fingerprint")
        if data["program_fingerprint"] != program.fingerprint:
            raise ValueError("execution metadata belongs to another scan program")
        value: PulseScanExecution = AutonomousScanExecution(
            program,
            _evidence_reference_from_tree(data["evidence"], artifacts),
            camera_run_evidence_from_tree(data["camera"]),
        )
    elif kind == _API_SEGMENTED_KIND:
        data = exact_mapping(
            tree,
            {"schema", "kind", "program_fingerprint", "segments", "camera"},
            SCAN_EXECUTION_SCHEMA,
        )
        if not isinstance(program, ApiSlotSegmentedProgram):
            raise ValueError("API execution tree has another program kind")
        point_count = program.point_table.point_layout.storage_size
        if len(artifacts) != point_count:
            raise ValueError("API execution requires one compiled artifact per point")
        sha256_text(data["program_fingerprint"], "program_fingerprint")
        if data["program_fingerprint"] != program.fingerprint:
            raise ValueError("execution metadata belongs to another scan program")
        segment_trees = data["segments"]
        if not isinstance(segment_trees, list):
            raise TypeError("segments must be a list")
        segments: list[ApiSegmentEvidence] = []
        for item in segment_trees:
            segment = exact_mapping(
                item,
                {"repeat_index", "point_storage_index", "pulse"},
                "API segment evidence",
                discriminator=None,
            )
            segments.append(
                ApiSegmentEvidence(
                    segment["repeat_index"],
                    segment["point_storage_index"],
                    _evidence_reference_from_tree(segment["pulse"], artifacts),
                )
            )
        value = ApiSegmentedScanExecution(
            program,
            tuple(segments),
            camera_run_evidence_from_tree(data["camera"]),
        )
    else:
        raise ValueError("pulse scan execution kind is unknown")
    if pulse_scan_execution_to_tree(value) != tree:
        raise ValueError("PulseScanExecution tree is typed but non-canonical")
    return value


__all__ = [
    "ApiSegmentEvidence",
    "ApiSegmentedScanExecution",
    "AutonomousScanExecution",
    "CAMERA_RUN_EVIDENCE_SCHEMA",
    "CameraRunEvidence",
    "PulseScanExecution",
    "SCAN_EXECUTION_SCHEMA",
    "api_segmented_metadata_static_shape_from_execution",
    "api_segmented_metadata_static_shape_to_tree",
    "camera_run_evidence_from_pipeline",
    "camera_run_evidence_from_tree",
    "camera_run_evidence_to_tree",
    "api_segmented_cell_schedule",
    "execution_compiled_artifacts",
    "pulse_scan_execution_from_tree",
    "pulse_scan_execution_to_tree",
]
