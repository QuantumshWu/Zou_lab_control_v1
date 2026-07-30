"""Direct durable output for finite raw camera captures."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.devices.camera.capture_port import (
    CaptureTerminalAck,
    capture_terminal_ack_from_tree,
    capture_terminal_ack_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraCapabilityEvidence,
    FrozenCaptureSpec,
    camera_capability_evidence_from_tree,
    camera_capability_evidence_to_tree,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
    frozen_capture_spec_from_tree,
    frozen_capture_spec_to_tree,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetSealProvenance,
    raw_dataset_seal_provenance_from_tree,
    raw_dataset_seal_provenance_to_tree,
)
from zlc_neutral_atom.runtime.preview import ExactDatasetPreviewPort, notify_preview_failure
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunPlan
from zlc_neutral_atom.timing.lineage import (
    PulseCaptureEvidence,
    pulse_capture_evidence_from_tree,
    pulse_capture_evidence_to_tree,
)
from zlc_pulse import decode_compiled_pulse_artifact, encode_compiled_pulse_artifact
from zlc_storage import canonical_text
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_text,
    durable_makedirs,
)
from zlc_storage.paths import resolve_under

from .frames import (
    CaptureFrameSource,
    _capture_frame_source_from_tree,
    _write_capture_frame_source,
)
from .pipeline import CapturePreviewPort, MinimalPipelineSpec, PipelineResult, compile_pipeline
from .reference import CaptureArtifactRef
from .session import (
    CameraCaptureProvenance,
    camera_capture_provenance_from_tree,
    camera_capture_provenance_to_tree,
)
from .triggered import TriggeredCaptureSpec, TriggeredPipelineResult, compile_triggered_pipeline


CAPTURE_ARTIFACT_SCHEMA = "zlc_neutral_atom.capture-artifact"
_CAPTURE_RECORD_NAME = "capture.json"
_COMPILED_PULSE_NAME = "pulse.bin"
_BYTES_MARKER = "$zlc-bytes"
_CAPTURE_FIELDS = {
    "schema",
    "run_id",
    "frames",
    "provenance",
    "terminal",
    "camera_provenance",
    "camera_capability_evidence",
    "camera_arm_spec",
    "compiled_pulse_file",
    "pulse_evidence",
}


def _json_tree(value: object) -> object:
    if isinstance(value, bytes):
        return {_BYTES_MARKER: base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("capture record mappings require string keys")
        return {key: _json_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    raise TypeError(f"capture record contains unsupported {type(value).__name__}")


def _typed_tree(value: object) -> object:
    if isinstance(value, list):
        return [_typed_tree(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_BYTES_MARKER}:
            encoded = value[_BYTES_MARKER]
            if not isinstance(encoded, str):
                raise TypeError("capture bytes marker must contain text")
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("capture bytes marker is invalid") from exc
        return {key: _typed_tree(item) for key, item in value.items()}
    return value


def _record_text(tree: dict[str, object]) -> str:
    return json.dumps(
        _json_tree(tree),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _read_record(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    tree = _typed_tree(raw)
    if not isinstance(tree, dict) or set(tree) != _CAPTURE_FIELDS:
        raise ValueError("CaptureArtifact record has an unknown field set")
    if tree["schema"] != CAPTURE_ARTIFACT_SCHEMA:
        raise ValueError("CaptureArtifact schema is not current")
    return tree


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
    """Check the physical facts that make this a coherent raw capture."""

    from zlc_data import DatasetSchema

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("capture event count must be positive")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    if provenance.direct_parent_span is not None:
        raise ValueError("raw CaptureArtifact cannot persist processor output")
    if not isinstance(terminal, CaptureTerminalAck):
        raise TypeError("terminal must be CaptureTerminalAck")
    if not isinstance(camera_provenance, CameraCaptureProvenance):
        raise TypeError("camera_provenance must be CameraCaptureProvenance")
    camera_provenance.validate_schema(schema)
    if not isinstance(camera_capability_evidence, CameraCapabilityEvidence):
        raise TypeError("camera_capability_evidence must be CameraCapabilityEvidence")
    if not isinstance(camera_arm_spec, FrozenCaptureSpec):
        raise TypeError("camera_arm_spec must be FrozenCaptureSpec")
    if camera_arm_spec.owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
        raise ValueError("camera arm spec belongs to an unknown owner")
    decoded_arm_spec = decode_camera_capture_spec(camera_arm_spec.payload)
    if freeze_camera_capture_spec(decoded_arm_spec) != camera_arm_spec:
        raise ValueError("camera arm spec is not the owner encoding")
    if decoded_arm_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError("finite CaptureArtifact requires external camera trigger")
    if len(
        {
            camera_capability_evidence.fingerprint,
            terminal.capability_fingerprint,
            camera_provenance.capability_fingerprint,
        }
    ) != 1:
        raise ValueError("camera capability lineage is inconsistent")
    camera_capability_evidence.physical_facts.validate_descriptor(
        camera_provenance.descriptor
    )
    if (
        camera_capability_evidence.capture_spec_owner_fingerprint
        != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT
    ):
        raise ValueError("camera capability names an unknown capture-spec owner")
    if len(
        {
            camera_arm_spec.digest,
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
    if camera_provenance.binding.value != camera_capability_evidence.source_id:
        raise ValueError("camera source lineage is inconsistent")
    physical_cells = schema.repeat_axis.size * schema.point_table.row_count
    if count != physical_cells or decoded_arm_spec.expected_frames != count:
        raise ValueError("capture cardinality differs from camera arm")
    if len(
        {
            decoded_arm_spec.settings_fingerprint,
            camera_capability_evidence.settings_fingerprint,
            terminal.settings_fingerprint,
            camera_provenance.active_settings_fingerprint,
        }
    ) != 1:
        raise ValueError("camera settings lineage is inconsistent")
    if provenance.end_sequence - provenance.start_sequence != count:
        raise ValueError("capture provenance interval differs from event count")
    if (
        terminal.produced_count != count
        or terminal.drained_count != count
        or not terminal.source_stopped
        or not terminal.no_more_frames
        or not terminal.joined
    ):
        raise ValueError("capture terminal evidence differs from persisted data")


@dataclass(frozen=True)
class CaptureArtifact:
    ref: CaptureArtifactRef
    run_id: str
    frame_source: CaptureFrameSource
    provenance: DatasetSealProvenance
    terminal: CaptureTerminalAck
    camera_provenance: CameraCaptureProvenance
    camera_capability_evidence: CameraCapabilityEvidence
    camera_arm_spec: FrozenCaptureSpec
    pulse_evidence: PulseCaptureEvidence | None = None
    _snapshot: OwnedSnapshot | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, CaptureArtifactRef):
            raise TypeError("ref must be CaptureArtifactRef")
        canonical_text(self.run_id, "run_id")
        if self.ref.record_path != f"{self.run_id}/{_CAPTURE_RECORD_NAME}":
            raise ValueError("CaptureArtifactRef run-name differs from run_id")
        if not isinstance(self.frame_source, CaptureFrameSource):
            raise TypeError("frame_source must be CaptureFrameSource")
        _validate_capture_metadata_contract(
            schema=self.frame_source.schema,
            count=self.frame_source.event_count,
            provenance=self.provenance,
            terminal=self.terminal,
            camera_provenance=self.camera_provenance,
            camera_capability_evidence=self.camera_capability_evidence,
            camera_arm_spec=self.camera_arm_spec,
        )
        if self.pulse_evidence is not None and not isinstance(
            self.pulse_evidence, PulseCaptureEvidence
        ):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence or None")
        if self.pulse_evidence is not None:
            if self.pulse_evidence.expected_trigger_count != self.frame_source.event_count:
                raise ValueError("pulse trigger count differs from captured frames")
            self.camera_capability_evidence.physical_facts.require_single_capture_trigger_channel(
                self.pulse_evidence.trigger_channel
            )
            expected = self.pulse_evidence.expected_cell_schedule(
                self.frame_source.schema
            )
            if not expected.same_order_as(self.frame_source.cell_schedule):
                raise ValueError("pulse trigger mapping differs from capture schedule")
        if self._snapshot is not None:
            if not isinstance(self._snapshot, OwnedSnapshot):
                raise TypeError("_snapshot must be OwnedSnapshot or None")
            if self._snapshot.ref != self.frame_source.ref(self.provenance.generation):
                raise ValueError("materialized snapshot identity differs from capture")

    def materialize_snapshot(
        self,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> OwnedSnapshot:
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if self._snapshot is not None:
            if abort_check is not None:
                abort_check()
            return self._snapshot
        block = self.frame_source.materialize(abort_check=abort_check)
        return OwnedSnapshot(self.frame_source.ref(self.provenance.generation), block)


def _capture_result(
    result: PipelineResult | TriggeredPipelineResult,
) -> tuple[PipelineResult, PulseCaptureEvidence | None]:
    if isinstance(result, TriggeredPipelineResult):
        return result.capture, result.lineage.evidence()
    if isinstance(result, PipelineResult):
        return result, None
    raise TypeError("capture output requires an exact pipeline result")


def write_capture_artifact(
    captures_root: Path,
    result: PipelineResult | TriggeredPipelineResult,
) -> CaptureArtifactRef:
    """Write payload files first and atomically publish ``capture.json`` last."""

    root = Path(captures_root).expanduser().resolve()
    durable_makedirs(root)
    base, pulse_evidence = _capture_result(result)
    if not base.is_direct_raw_capture:
        raise ValueError("CaptureArtifact accepts only direct raw camera data")
    reference = CaptureArtifactRef(f"{base.run_id}/{_CAPTURE_RECORD_NAME}")
    record_path = resolve_under(root, reference.record_path)
    run_dir = record_path.parent
    durable_makedirs(run_dir)
    if record_path.exists():
        raise FileExistsError(f"capture record already exists: {record_path}")
    frame_source, frame_tree = _write_capture_frame_source(
        run_dir,
        block=base.dataset.block,
        event_metadata=tuple(base.dataset.event_metadata),
        cell_schedule=base.source_cell_schedule,
    )
    compiled_pulse_file = None
    if pulse_evidence is not None:
        compiled_pulse_file = _COMPILED_PULSE_NAME
        atomic_write_bytes(
            resolve_under(run_dir, compiled_pulse_file),
            encode_compiled_pulse_artifact(pulse_evidence.compiled_artifact),
        )
    artifact = CaptureArtifact(
        ref=reference,
        run_id=base.run_id,
        frame_source=frame_source,
        provenance=base.dataset.provenance,
        terminal=base.capture_terminal,
        camera_provenance=base.camera_provenance,
        camera_capability_evidence=base.camera_capability_evidence,
        camera_arm_spec=base.camera_arm_spec,
        pulse_evidence=pulse_evidence,
    )
    record = {
        "schema": CAPTURE_ARTIFACT_SCHEMA,
        "run_id": artifact.run_id,
        "frames": frame_tree,
        "provenance": raw_dataset_seal_provenance_to_tree(artifact.provenance),
        "terminal": capture_terminal_ack_to_tree(artifact.terminal),
        "camera_provenance": camera_capture_provenance_to_tree(
            artifact.camera_provenance
        ),
        "camera_capability_evidence": camera_capability_evidence_to_tree(
            artifact.camera_capability_evidence
        ),
        "camera_arm_spec": frozen_capture_spec_to_tree(artifact.camera_arm_spec),
        "compiled_pulse_file": compiled_pulse_file,
        "pulse_evidence": pulse_capture_evidence_to_tree(artifact.pulse_evidence),
    }
    atomic_write_text(record_path, _record_text(record))
    return reference


def load_capture_artifact(
    captures_root: Path,
    ref: CaptureArtifactRef,
    *,
    materialize: bool = False,
) -> CaptureArtifact:
    """Load one visible direct-output capture beneath an explicit root."""

    if not isinstance(ref, CaptureArtifactRef):
        raise TypeError("ref must be CaptureArtifactRef")
    if type(materialize) is not bool:
        raise TypeError("materialize must be bool")
    root = Path(captures_root).expanduser().resolve()
    record_path = resolve_under(root, ref.record_path)
    tree = _read_record(record_path)
    run_id = tree["run_id"]
    if not isinstance(run_id, str):
        raise TypeError("capture run_id must be str")
    frame_source = _capture_frame_source_from_tree(record_path.parent, tree["frames"])
    pulse_file = tree["compiled_pulse_file"]
    if pulse_file is None:
        compiled_pulse = None
    else:
        if not isinstance(pulse_file, str):
            raise TypeError("compiled_pulse_file must be str or None")
        pulse_path = resolve_under(record_path.parent, pulse_file)
        compiled_pulse = decode_compiled_pulse_artifact(pulse_path.read_bytes())
    pulse_evidence = pulse_capture_evidence_from_tree(
        tree["pulse_evidence"], compiled_pulse
    )
    provenance = raw_dataset_seal_provenance_from_tree(tree["provenance"])
    snapshot = None
    if materialize:
        block = frame_source.materialize()
        snapshot = OwnedSnapshot(frame_source.ref(provenance.generation), block)
    return CaptureArtifact(
        ref=ref,
        run_id=run_id,
        frame_source=frame_source,
        provenance=provenance,
        terminal=capture_terminal_ack_from_tree(tree["terminal"]),
        camera_provenance=camera_capture_provenance_from_tree(
            tree["camera_provenance"]
        ),
        camera_capability_evidence=camera_capability_evidence_from_tree(
            tree["camera_capability_evidence"]
        ),
        camera_arm_spec=frozen_capture_spec_from_tree(tree["camera_arm_spec"]),
        pulse_evidence=pulse_evidence,
        _snapshot=snapshot,
    )


def project_capture_dataset_source(
    captures_root: Path,
    ref: CaptureArtifactRef,
    *,
    materialize: bool = False,
    abort_check: Callable[[], None] | None = None,
) -> ArtifactDatasetSource:
    """Project a capture through the capture owner's dataset contract."""

    artifact = load_capture_artifact(
        captures_root,
        ref,
        materialize=False,
    )
    if abort_check is not None:
        abort_check()
    snapshot = (
        artifact.materialize_snapshot(abort_check=abort_check)
        if materialize
        else None
    )
    source = artifact.frame_source
    return ArtifactDatasetSource(
        source.schema,
        source.ref(artifact.provenance.generation),
        snapshot,
    )


def compile_capture_artifact_pipeline(
    spec: MinimalPipelineSpec | TriggeredCaptureSpec,
    captures_root: Path,
    *,
    preview: CapturePreviewPort | None = None,
    exact_preview: ExactDatasetPreviewPort | None = None,
    settle_exact_preview: bool = True,
) -> RunPlan:
    """Attach one direct Capture writer to an exact hardware RunPlan."""

    try:
        root = Path(captures_root).expanduser().resolve()
        capture_spec = spec.capture if isinstance(spec, TriggeredCaptureSpec) else spec
        if not isinstance(capture_spec, MinimalPipelineSpec):
            raise TypeError("capture artifact pipeline requires MinimalPipelineSpec")
        if type(settle_exact_preview) is not bool:
            raise TypeError("settle_exact_preview must be bool")
        if isinstance(spec, TriggeredCaptureSpec) and not settle_exact_preview:
            raise ValueError("triggered capture owns exact-preview cleanup")
    except BaseException as error:
        notify_preview_failure(preview, error)
        notify_preview_failure(exact_preview, error)
        raise
    base = (
        compile_triggered_pipeline(spec, preview=preview, exact_preview=exact_preview)
        if isinstance(spec, TriggeredCaptureSpec)
        else compile_pipeline(
            spec,
            preview=preview,
            exact_preview=exact_preview,
            settle_exact_preview=settle_exact_preview,
        )
    )
    base_finalize = base.finalize

    def finalize(
        context: PostSafetyContext,
        result: PipelineResult | TriggeredPipelineResult,
    ) -> CaptureArtifactRef:
        finalized = base_finalize(context, result)
        if not isinstance(finalized, (PipelineResult, TriggeredPipelineResult)):
            raise TypeError("base exact pipeline changed its result contract")
        captured, _pulse = _capture_result(finalized)
        if captured.run_id != context.run_id.value:
            raise ValueError("capture result belongs to another Run")
        return write_capture_artifact(root, finalized)

    return RunPlan(
        name=base.name,
        resource_claims=base.resource_claims,
        bound_devices=base.bound_devices,
        preflight=base.preflight,
        execute=base.execute,
        cleanup=base.cleanup,
        finalize=finalize,
        interrupt_operations=base.interrupt_operations,
        timeout_seconds=base.timeout_seconds,
        dispose_unfinalized=base.dispose_unfinalized,
    )


__all__ = [
    "CAPTURE_ARTIFACT_SCHEMA",
    "CaptureArtifact",
    "compile_capture_artifact_pipeline",
    "load_capture_artifact",
    "project_capture_dataset_source",
    "write_capture_artifact",
]
