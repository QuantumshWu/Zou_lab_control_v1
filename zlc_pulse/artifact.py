"""Current compiled pulse artifact joining source, TargetIR, wire, and triggers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from zlc_storage import canonical_digest, decode, encode

from .fpga import (
    PulseWireImage,
    pulse_wire_image_from_tree,
    pulse_wire_image_to_tree,
)
from .ir import TargetIR, target_ir_from_tree, target_ir_to_tree
from .schedule import (
    DigitalTriggerSchedule,
    build_digital_trigger_schedules,
    digital_trigger_schedule_from_tree,
    digital_trigger_schedule_to_tree,
)


COMPILED_PULSE_ARTIFACT_SCHEMA = "zlc_pulse.CompiledPulseArtifact/v2"


class PulseExecutionForm(str, Enum):
    STATIC_ONCE = "STATIC_ONCE"
    STATIC_REFERENCE_POINT = "STATIC_REFERENCE_POINT"
    CONTINUOUS_MONITOR = "CONTINUOUS_MONITOR"
    AUTONOMOUS_SCAN_ONCE = "AUTONOMOUS_SCAN_ONCE"


def _text(value: object, field: str) -> str:
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


@dataclass(frozen=True)
class CompiledPulseArtifact:
    source_document_digest: str
    compiler_id: str
    compiler_version: str
    execution_form: PulseExecutionForm
    target_ir: TargetIR
    wire_image: PulseWireImage
    trigger_schedules: tuple[DigitalTriggerSchedule, ...] = ()

    def __post_init__(self) -> None:
        _sha256(self.source_document_digest, "source_document_digest")
        object.__setattr__(self, "compiler_id", _text(self.compiler_id, "compiler_id"))
        object.__setattr__(self, "compiler_version", _text(self.compiler_version, "compiler_version"))
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if not isinstance(self.target_ir, TargetIR):
            raise TypeError("target_ir must be TargetIR")
        if not isinstance(self.wire_image, PulseWireImage):
            raise TypeError("wire_image must be PulseWireImage")
        ir_digest = canonical_digest(target_ir_to_tree(self.target_ir))
        if self.wire_image.source_ir_digest != ir_digest:
            raise ValueError("wire image is not bound to target_ir")
        schedules = tuple(self.trigger_schedules)
        if any(not isinstance(item, DigitalTriggerSchedule) for item in schedules):
            raise TypeError("trigger_schedules must contain DigitalTriggerSchedule values")
        if len({item.channel for item in schedules}) != len(schedules):
            raise ValueError("trigger schedule channels must be unique")
        object.__setattr__(self, "trigger_schedules", schedules)
        scan = self.target_ir.scan_enabled
        if self.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
            if not scan or self.target_ir.repeat_forever:
                raise ValueError("AUTONOMOUS_SCAN_ONCE requires finite scan TargetIR")
        elif scan:
            raise ValueError("only AUTONOMOUS_SCAN_ONCE may carry scan points")
        if self.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            if not self.target_ir.repeat_forever or schedules:
                raise ValueError("continuous monitor must be cyclic and has no finite trigger schedule")
        elif self.target_ir.repeat_forever:
            raise ValueError("finite execution form cannot carry cyclic TargetIR")
        expected_points = len(self.target_ir.scan_points) if scan else 1
        if any(item.point_count != expected_points for item in schedules):
            raise ValueError("trigger schedule point count differs from TargetIR")
        if (
            not self.target_ir.repeat_forever
            and bool(schedules)
            and not _same_physical_trigger_schedule(
                schedules,
                build_digital_trigger_schedules(
                    self.target_ir,
                    tuple(item.channel for item in schedules),
                ),
            )
        ):
            raise ValueError("trigger schedules are not the deterministic TargetIR expansion")

    @property
    def target_abi_fingerprint(self) -> str:
        return self.target_ir.target_abi_fingerprint

    @property
    def max_configured_output_delay_ticks(self) -> int:
        return max(
            (0, *self.target_ir.channel_delays, *(item.delay_ticks for item in self.target_ir.bus_delays))
        )

    @property
    def fingerprint(self) -> str:
        return canonical_digest(compiled_pulse_artifact_to_tree(self))


def _same_physical_trigger_schedule(
    actual: tuple[DigitalTriggerSchedule, ...],
    expected: tuple[DigitalTriggerSchedule, ...],
) -> bool:
    """Compare wire facts while retaining source loop provenance sidecars."""

    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected, strict=True):
        if (
            left.channel != right.channel
            or left.point_count != right.point_count
            or len(left.edges) != len(right.edges)
        ):
            return False
        for left_edge, right_edge in zip(left.edges, right.edges, strict=True):
            if (
                left_edge.channel != right_edge.channel
                or left_edge.point_index != right_edge.point_index
                or left_edge.trigger_ordinal != right_edge.trigger_ordinal
                or left_edge.point_trigger_ordinal
                != right_edge.point_trigger_ordinal
                or left_edge.tick_from_run_start != right_edge.tick_from_run_start
            ):
                return False
    return True


def compiled_pulse_artifact_to_tree(value: CompiledPulseArtifact) -> dict[str, object]:
    if not isinstance(value, CompiledPulseArtifact):
        raise TypeError("value must be CompiledPulseArtifact")
    return {
        "schema": COMPILED_PULSE_ARTIFACT_SCHEMA,
        "source_document_digest": value.source_document_digest,
        "compiler_id": value.compiler_id,
        "compiler_version": value.compiler_version,
        "execution_form": value.execution_form.value,
        "target_ir": target_ir_to_tree(value.target_ir),
        "wire_image": pulse_wire_image_to_tree(value.wire_image),
        "trigger_schedules": [
            digital_trigger_schedule_to_tree(item) for item in value.trigger_schedules
        ],
    }


def compiled_pulse_artifact_from_tree(tree: object) -> CompiledPulseArtifact:
    fields = {
        "schema",
        "source_document_digest",
        "compiler_id",
        "compiler_version",
        "execution_form",
        "target_ir",
        "wire_image",
        "trigger_schedules",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CompiledPulseArtifact has an unknown field set")
    if tree["schema"] != COMPILED_PULSE_ARTIFACT_SCHEMA:
        raise ValueError("CompiledPulseArtifact schema differs")
    if not isinstance(tree["trigger_schedules"], list):
        raise TypeError("trigger_schedules must be a list")
    return CompiledPulseArtifact(
        tree["source_document_digest"],
        tree["compiler_id"],
        tree["compiler_version"],
        PulseExecutionForm(tree["execution_form"]),
        target_ir_from_tree(tree["target_ir"]),
        pulse_wire_image_from_tree(tree["wire_image"]),
        tuple(
            digital_trigger_schedule_from_tree(item)
            for item in tree["trigger_schedules"]
        ),
    )


def encode_compiled_pulse_artifact(value: CompiledPulseArtifact) -> bytes:
    return encode(compiled_pulse_artifact_to_tree(value))


def decode_compiled_pulse_artifact(payload: bytes) -> CompiledPulseArtifact:
    value = compiled_pulse_artifact_from_tree(decode(payload))
    if encode_compiled_pulse_artifact(value) != bytes(payload):
        raise ValueError("CompiledPulseArtifact payload is not canonical")
    return value


__all__ = [
    "COMPILED_PULSE_ARTIFACT_SCHEMA",
    "CompiledPulseArtifact",
    "PulseExecutionForm",
    "compiled_pulse_artifact_from_tree",
    "compiled_pulse_artifact_to_tree",
    "decode_compiled_pulse_artifact",
    "encode_compiled_pulse_artifact",
]
