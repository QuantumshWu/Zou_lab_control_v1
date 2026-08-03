"""Direct durable output for finite raw camera captures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from zlc_data import AxisId, CoordinateFrameId, DatasetSchema, OwnedSnapshot
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.devices.camera.capture_port import (
    CaptureTerminalAck,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCapabilityEvidence,
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    CameraPhysicalFacts,
    CameraCaptureSpec,
    ReadoutBindingKey,
    camera_capture_spec_from_tree,
    camera_capture_spec_to_tree,
    camera_frame_facts_to_tree,
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
from .reference import CAPTURE_RECORD_PREFIX, CaptureArtifactRef
from .session import (
    CameraCaptureProvenance,
)
from .triggered import TriggeredCaptureSpec, TriggeredPipelineResult, compile_triggered_pipeline


CAPTURE_ARTIFACT_SCHEMA = "zlc.capture"
_CAPTURE_RECORD_NAME = "capture.json"
_COMPILED_PULSE_NAME = "pulse.bin"
_CAPTURE_FIELDS = {
    "schema",
    "run_id",
    "frames",
    "provenance",
    "terminal",
    "camera",
    "capture",
    "pulse",
}


def _validate_capture_metadata_contract(
    *,
    schema: DatasetSchema,
    count: int,
    provenance: DatasetSealProvenance,
    terminal: dict[str, object],
    camera: dict[str, object],
    capture_spec: CameraCaptureSpec,
) -> None:
    """Check the physical facts that make this a coherent raw capture."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("capture event count must be positive")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    if provenance.direct_parent_span is not None:
        raise ValueError("raw CaptureArtifact cannot persist processor output")
    canonical_text(camera["binding"], "camera binding")
    canonical_text(camera["source_id"], "camera source_id")
    physical_cells = schema.repeat_axis.size * schema.point_table.row_count
    if count != physical_cells:
        raise ValueError("capture cardinality differs from Dataset shape")
    if not isinstance(capture_spec, CameraCaptureSpec):
        raise TypeError("capture_spec must be CameraCaptureSpec")
    if capture_spec.expected_frames != count:
        raise ValueError("capture spec cardinality differs from persisted data")
    if provenance.end_sequence - provenance.start_sequence != count:
        raise ValueError("capture provenance interval differs from event count")
    if (
        terminal["produced_count"] != count
        or terminal["drained_count"] != count
        or not terminal["source_stopped"]
        or not terminal["no_more_frames"]
        or not terminal["joined"]
    ):
        raise ValueError("capture terminal evidence differs from persisted data")


@dataclass(frozen=True)
class CaptureArtifact:
    ref: CaptureArtifactRef
    run_id: str
    frame_source: CaptureFrameSource
    provenance: DatasetSealProvenance
    terminal: dict[str, object]
    camera: dict[str, object]
    capture_spec: CameraCaptureSpec
    pulse_evidence: PulseCaptureEvidence | None = None
    _snapshot: OwnedSnapshot | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, CaptureArtifactRef):
            raise TypeError("ref must be CaptureArtifactRef")
        canonical_text(self.run_id, "run_id")
        expected_path = "/".join(
            (*CAPTURE_RECORD_PREFIX, self.run_id, _CAPTURE_RECORD_NAME)
        )
        if self.ref.record_path != expected_path:
            raise ValueError("CaptureArtifactRef run-name differs from run_id")
        if not isinstance(self.frame_source, CaptureFrameSource):
            raise TypeError("frame_source must be CaptureFrameSource")
        _validate_capture_metadata_contract(
            schema=self.frame_source.schema,
            count=self.frame_source.event_count,
            provenance=self.provenance,
            terminal=self.terminal,
            camera=self.camera,
            capture_spec=self.capture_spec,
        )
        if self.pulse_evidence is not None and not isinstance(
            self.pulse_evidence, PulseCaptureEvidence
        ):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence or None")
        if self.pulse_evidence is not None:
            if self.pulse_evidence.expected_trigger_count != self.frame_source.event_count:
                raise ValueError("pulse trigger count differs from captured frames")
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

    @property
    def camera_binding(self) -> ReadoutBindingKey:
        """The logical readout binding persisted by the capture owner."""

        return ReadoutBindingKey(self.camera["binding"])

    @property
    def camera_descriptor(self) -> CameraCaptureDescriptor:
        """Rehydrate the typed camera geometry used by downstream analysis."""

        settings = tuple(
            CameraEventReadoutSetting(
                row["event_index"],
                row["exposure_seconds"],
                row["gain"],
                row["readout_mode"],
            )
            for row in self.camera["event_settings"]
        )
        return CameraCaptureDescriptor(
            camera_identity=self.camera["camera_identity"],
            sensor_identity=self.camera["sensor_identity"],
            optical_path=self.camera["optical_path"],
            sensor_shape_yx=tuple(self.camera["sensor_shape_yx"]),
            roi_origin_yx=tuple(self.camera["roi_origin_yx"]),
            roi_shape_yx=tuple(self.camera["roi_shape_yx"]),
            binning_yx=tuple(self.camera["binning_yx"]),
            spatial_y_axis_id=AxisId(self.camera["spatial_y_axis_id"]),
            spatial_x_axis_id=AxisId(self.camera["spatial_x_axis_id"]),
            coordinate_frame=CoordinateFrameId(self.camera["coordinate_frame"]),
            dtype=self.camera["dtype"],
            count_unit=self.camera["count_unit"],
            readout_event_axis_id=(
                None
                if self.camera["readout_event_axis_id"] is None
                else AxisId(self.camera["readout_event_axis_id"])
            ),
            event_settings=settings,
        )

    @property
    def camera_physical_facts(self) -> CameraPhysicalFacts:
        """Rehydrate the physical working-point facts needed by analysis."""

        settings = tuple(self.camera["event_settings"])
        if not settings:
            raise ValueError("capture camera record has no event settings")
        setting = settings[0]
        return CameraPhysicalFacts(
            camera_identity=self.camera["camera_identity"],
            sensor_identity=self.camera["sensor_identity"],
            optical_path=self.camera["optical_path"],
            sensor_shape_yx=tuple(self.camera["sensor_shape_yx"]),
            roi_origin_yx=tuple(self.camera["roi_origin_yx"]),
            roi_shape_yx=tuple(self.camera["roi_shape_yx"]),
            binning_yx=tuple(self.camera["binning_yx"]),
            spatial_y_axis_id=AxisId(self.camera["spatial_y_axis_id"]),
            spatial_x_axis_id=AxisId(self.camera["spatial_x_axis_id"]),
            coordinate_frame=CoordinateFrameId(self.camera["coordinate_frame"]),
            dtype=self.camera["dtype"],
            count_unit=self.camera["count_unit"],
            exposure_seconds=setting["exposure_seconds"],
            required_external_trigger_interval_seconds=(
                self.camera["required_external_trigger_interval_seconds"]
            ),
            external_trigger_integration_start_offset_seconds=(
                self.camera["external_trigger_integration_start_offset_seconds"]
            ),
            gain=setting["gain"],
            readout_mode=setting["readout_mode"],
        )


def _capture_result(
    result: PipelineResult | TriggeredPipelineResult,
) -> tuple[PipelineResult, PulseCaptureEvidence | None]:
    if isinstance(result, TriggeredPipelineResult):
        return result.capture, result.lineage.evidence()
    if isinstance(result, PipelineResult):
        return result, None
    raise TypeError("capture output requires an exact pipeline result")


def _terminal_to_record(value: CaptureTerminalAck) -> dict[str, object]:
    if not isinstance(value, CaptureTerminalAck):
        raise TypeError("value must be CaptureTerminalAck")
    return {
        "session_id": value.session_id,
        "produced_count": value.produced_count,
        "drained_count": value.drained_count,
        "source_stopped": value.source_stopped,
        "no_more_frames": value.no_more_frames,
        "joined": value.joined,
    }


def _terminal_from_record(tree: object) -> dict[str, object]:
    fields = {
        "session_id",
        "produced_count",
        "drained_count",
        "source_stopped",
        "no_more_frames",
        "joined",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("capture terminal record has an unknown field set")
    return dict(tree)


def _camera_to_record(
    provenance: CameraCaptureProvenance,
    evidence: CameraCapabilityEvidence,
) -> dict[str, object]:
    """Project the reload/science camera facts without digest mirrors."""

    if not isinstance(provenance, CameraCaptureProvenance):
        raise TypeError("provenance must be CameraCaptureProvenance")
    if not isinstance(evidence, CameraCapabilityEvidence):
        raise TypeError("evidence must be CameraCapabilityEvidence")
    descriptor = provenance.descriptor
    facts = evidence.physical_facts
    return {
        "adapter_type": evidence.adapter_type,
        "source_id": evidence.source_id,
        "binding": provenance.binding.value,
        **camera_frame_facts_to_tree(facts),
        "required_external_trigger_interval_seconds": (
            facts.required_external_trigger_interval_seconds
        ),
        "external_trigger_integration_start_offset_seconds": (
            facts.external_trigger_integration_start_offset_seconds
        ),
        "readout_event_axis_id": (
            None
            if descriptor.readout_event_axis_id is None
            else descriptor.readout_event_axis_id.value
        ),
        "event_settings": [
            {
                "event_index": setting.event_index,
                "exposure_seconds": setting.exposure_seconds,
                "gain": setting.gain,
                "readout_mode": setting.readout_mode,
            }
            for setting in descriptor.event_settings
        ],
    }


def _camera_from_record(tree: object) -> dict[str, object]:
    fields = {
        "adapter_type",
        "source_id",
        "binding",
        "camera_identity",
        "sensor_identity",
        "optical_path",
        "sensor_shape_yx",
        "roi_origin_yx",
        "roi_shape_yx",
        "binning_yx",
        "spatial_y_axis_id",
        "spatial_x_axis_id",
        "coordinate_frame",
        "dtype",
        "count_unit",
        "required_external_trigger_interval_seconds",
        "external_trigger_integration_start_offset_seconds",
        "readout_event_axis_id",
        "event_settings",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("capture camera record has an unknown field set")
    event_rows = tree["event_settings"]
    if not isinstance(event_rows, list) or not event_rows:
        raise ValueError("capture event_settings must be a non-empty list")
    setting_fields = {
        "event_index",
        "exposure_seconds",
        "gain",
        "readout_mode",
    }
    for row in event_rows:
        if not isinstance(row, dict) or set(row) != setting_fields:
            raise ValueError("capture event setting has an unknown field set")
    return dict(tree)


def write_capture_artifact(
    project_root: Path,
    result: PipelineResult | TriggeredPipelineResult,
) -> CaptureArtifactRef:
    """Write payload files first and atomically publish ``capture.json`` last."""

    root = Path(project_root).expanduser().resolve()
    durable_makedirs(resolve_under(root, "/".join(CAPTURE_RECORD_PREFIX)))
    base, pulse_evidence = _capture_result(result)
    if not base.is_direct_raw_capture:
        raise ValueError("CaptureArtifact accepts only direct raw camera data")
    reference = CaptureArtifactRef(
        "/".join((*CAPTURE_RECORD_PREFIX, base.run_id, _CAPTURE_RECORD_NAME))
    )
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
    camera = _camera_to_record(
        base.camera_provenance,
        base.camera_capability_evidence,
    )
    artifact = CaptureArtifact(
        ref=reference,
        run_id=base.run_id,
        frame_source=frame_source,
        provenance=base.dataset.provenance,
        terminal=_terminal_to_record(base.capture_terminal),
        camera=camera,
        capture_spec=base.camera_capture_spec,
        pulse_evidence=pulse_evidence,
    )
    record = {
        "schema": CAPTURE_ARTIFACT_SCHEMA,
        "run_id": artifact.run_id,
        "frames": frame_tree,
        "provenance": raw_dataset_seal_provenance_to_tree(artifact.provenance),
        "terminal": artifact.terminal,
        "camera": artifact.camera,
        "capture": camera_capture_spec_to_tree(artifact.capture_spec),
        "pulse": (
            None
            if artifact.pulse_evidence is None
            else {
                "compiled_pulse_file": compiled_pulse_file,
                "evidence": pulse_capture_evidence_to_tree(
                    artifact.pulse_evidence
                ),
            }
        ),
    }
    atomic_write_text(
        record_path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return reference


def load_capture_artifact(
    project_root: Path,
    ref: CaptureArtifactRef,
    *,
    materialize: bool = False,
) -> CaptureArtifact:
    """Load one visible direct-output capture beneath an explicit project root."""

    if not isinstance(ref, CaptureArtifactRef):
        raise TypeError("ref must be CaptureArtifactRef")
    if type(materialize) is not bool:
        raise TypeError("materialize must be bool")
    root = Path(project_root).expanduser().resolve()
    record_path = resolve_under(root, ref.record_path)
    with record_path.open("r", encoding="utf-8") as stream:
        tree = json.load(stream)
    if not isinstance(tree, dict) or set(tree) != _CAPTURE_FIELDS:
        raise ValueError("CaptureArtifact record has an unknown field set")
    if tree["schema"] != CAPTURE_ARTIFACT_SCHEMA:
        raise ValueError("CaptureArtifact schema is not current")
    run_id = tree["run_id"]
    if not isinstance(run_id, str):
        raise TypeError("capture run_id must be str")
    frame_source = _capture_frame_source_from_tree(record_path.parent, tree["frames"])
    pulse_record = tree["pulse"]
    if pulse_record is None:
        compiled_pulse = None
        pulse_evidence = None
    else:
        if not isinstance(pulse_record, dict) or set(pulse_record) != {
            "compiled_pulse_file",
            "evidence",
        }:
            raise ValueError("capture pulse record has an unknown field set")
        pulse_file = pulse_record["compiled_pulse_file"]
        if not isinstance(pulse_file, str):
            raise TypeError("compiled_pulse_file must be str")
        pulse_path = resolve_under(record_path.parent, pulse_file)
        compiled_pulse = decode_compiled_pulse_artifact(pulse_path.read_bytes())
        pulse_evidence = pulse_capture_evidence_from_tree(
            pulse_record["evidence"],
            compiled_pulse,
        )
    provenance = raw_dataset_seal_provenance_from_tree(tree["provenance"])
    camera = _camera_from_record(tree["camera"])
    snapshot = None
    if materialize:
        block = frame_source.materialize()
        snapshot = OwnedSnapshot(frame_source.ref(provenance.generation), block)
    return CaptureArtifact(
        ref=ref,
        run_id=run_id,
        frame_source=frame_source,
        provenance=provenance,
        terminal=_terminal_from_record(tree["terminal"]),
        camera=camera,
        capture_spec=camera_capture_spec_from_tree(tree["capture"]),
        pulse_evidence=pulse_evidence,
        _snapshot=snapshot,
    )


def project_capture_dataset_source(
    project_root: Path,
    ref: CaptureArtifactRef,
    *,
    materialize: bool = False,
    abort_check: Callable[[], None] | None = None,
) -> ArtifactDatasetSource:
    """Project a capture through the capture owner's dataset contract."""

    artifact = load_capture_artifact(
        project_root,
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
    project_root: Path,
    *,
    preview: CapturePreviewPort | None = None,
    exact_preview: ExactDatasetPreviewPort | None = None,
    settle_exact_preview: bool = True,
) -> RunPlan:
    """Attach one direct Capture writer to an exact hardware RunPlan."""

    try:
        root = Path(project_root).expanduser().resolve()
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
